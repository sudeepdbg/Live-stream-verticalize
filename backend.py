from __future__ import annotations
import collections, json, math, re, subprocess, tempfile, threading, time
import urllib.error, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import cv2, numpy as np

DEFAULT_TARGET_W=540; DEFAULT_TARGET_H=960; MAX_UPLOAD_MB=400
WORKING_INPUT_W=1280; WORKING_INPUT_H=720; PLACEHOLDER_FPS=30.0
DEFAULT_OUTPUT_FPS=30.0; DEFAULT_VIDEO_BITRATE="3500k"
DEFAULT_MAXRATE="3500k"; DEFAULT_BUFSIZE="7000k"; PANEL_GAP=6

def ffmpeg_ok():
    try:
        subprocess.run(["ffmpeg","-version"],capture_output=True,check=True,timeout=5)
        subprocess.run(["ffprobe","-version"],capture_output=True,check=True,timeout=5)
        return True
    except Exception: return False

def safe_token(v):
    v=v or "stream"; return re.sub(r"[^A-Za-z0-9._-]+","_",v).strip("._-") or "stream"

def is_network_source(s):
    return (s or "").lower().strip().startswith(("rtmp://","rtmps://","srt://","udp://","tcp://","http://","https://"))

def _source_input_args(source,pace_input=False,loop_file=False):
    a=["-fflags","+genpts+discardcorrupt","-analyzeduration","20000000","-probesize","20000000"]
    if loop_file and not is_network_source(source): a+=["-stream_loop","-1"]
    if pace_input and not is_network_source(source): a+=["-re"]
    if is_network_source(source):
        a+=["-rw_timeout","15000000"]
        if source.lower().startswith(("http://","https://")): a+=["-reconnect","1","-reconnect_streamed","1","-reconnect_delay_max","2"]
    a+=["-i",source]; return a

def _safe_json_loads(t):
    try: p=json.loads(t); return p if isinstance(p,dict) else {}
    except Exception: return {}

def _ffprobe_json(source,timeout=30):
    cmd=["ffprobe","-v","quiet","-print_format","json","-show_streams","-show_format","-analyzeduration","20000000","-probesize","20000000",source]
    return _safe_json_loads(subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL,timeout=timeout))

def probe_source(source):
    res={"duration":0.0,"width":0,"height":0,"fps":0.0,"vcodec":"unknown"}
    try:
        data=_ffprobe_json(source,timeout=35 if is_network_source(source) else 20)
        fmt=data.get("format",{}); res["duration"]=float(fmt.get("duration",0) or 0)
        for s in data.get("streams",[]):
            if s.get("codec_type")=="video" and res["width"]==0:
                res["width"]=int(s.get("width",0) or 0); res["height"]=int(s.get("height",0) or 0)
                res["vcodec"]=str(s.get("codec_name","unknown"))
                try:
                    r=str(s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"); n,d=map(int,r.split("/")); res["fps"]=round(n/d,3) if d else 0.0
                except Exception: pass
                break
    except Exception: pass
    if (res["width"]<=0 or res["height"]<=0 or res["fps"]<=0) and not is_network_source(source):
        cap=None
        try:
            cap=cv2.VideoCapture(source)
            if cap.isOpened():
                res["width"]=res["width"] or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                res["height"]=res["height"] or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                res["fps"]=res["fps"] or float(cap.get(cv2.CAP_PROP_FPS) or 0)
                fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if not res["duration"] and fc>0 and res["fps"]>0: res["duration"]=fc/res["fps"]
        except Exception: pass
        finally:
            if cap: cap.release()
    if res["width"]<=0 or res["height"]<=0:
        if is_network_source(source): res["width"],res["height"]=WORKING_INPUT_W,WORKING_INPUT_H
    if res["fps"]<=0: res["fps"]=PLACEHOLDER_FPS
    return res

def _vertical_crop_box(sw,sh):
    if sw/max(sh,1)>=9/16: ch=sh; cw=int(round(sh*9/16))
    else: cw=sw; ch=int(round(sw*16/9))
    return max(32,cw-(cw%2)),max(32,ch-(ch%2))

def _clamp(v,lo,hi): return max(lo,min(hi,v))

def _resize_cover(img,w,h):
    if img is None or img.size==0 or w<=0 or h<=0: return np.zeros((max(1,h),max(1,w),3),dtype=np.uint8)
    ih,iw=img.shape[:2]; s=max(w/max(iw,1),h/max(ih,1))
    nw,nh=max(1,int(round(iw*s))),max(1,int(round(ih*s)))
    r=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
    x0,y0=max(0,(nw-w)//2),max(0,(nh-h)//2); return r[y0:y0+h,x0:x0+w]

class OverlayDetector:
    def __init__(self,sw,sh,top_ratio=0.18,bot_ratio=0.14,warmup=18,diff_th=11.0,row_th=0.018,hold=10):
        self.sw,self.sh=int(sw),int(sh); self.tsh=max(24,int(round(sh*top_ratio))); self.bsh=max(24,int(round(sh*bot_ratio)))
        self.warmup=max(4,warmup); self.diff_th=diff_th; self.row_th=row_th; self.hold_max=max(1,hold)
        self.ta=None; self.ba=None; self.fc=0
        self.top_overlay=None; self.bottom_overlay=None; self.th=0; self.bh_hold=0
        self.exclusion_mask=np.zeros((self.sh,self.sw),dtype=np.uint8)
    def _red(self,gp): return cv2.Canny(gp,60,180).mean(axis=1)/255.0
    def _rgr(self,bgr):
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV); return (cv2.inRange(hsv,np.array([30,30,25]),np.array([90,255,255]))>0).mean(axis=1)
    def _detect(self,cb,ab,*,top):
        cg=cv2.cvtColor(cb,cv2.COLOR_BGR2GRAY); ag=cv2.cvtColor(np.clip(ab,0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
        rd=np.abs(cg.astype(np.float32)-ag.astype(np.float32)).mean(axis=1)
        re_=self._red(cg); rg=self._rgr(cb); hsv=cv2.cvtColor(cb,cv2.COLOR_BGR2HSV); rs=hsv[:,:,1].mean(axis=1)/255.0
        act=(rd<self.diff_th)&(re_>self.row_th)&((rs>0.10)|(rg<0.35))
        if act.any(): act=np.convolve(act.astype(np.uint8),np.ones(5,dtype=np.uint8),mode='same')>1
        idx=np.where(act)[0]
        if idx.size==0: return None
        runs=[]; s=idx[0]; p=idx[0]
        for v in idx[1:]:
            if v==p+1: p=v
            else: runs.append((s,p+1)); s=v; p=v
        runs.append((s,p+1)); runs.sort(key=lambda x:x[1]-x[0],reverse=True)
        y0,y1=runs[0]; bh_=y1-y0
        if top:
            if y0>int(0.06*cb.shape[0]) or bh_<12: return None
            y0=max(0,y0-6); y1=min(cb.shape[0],y1+6)
            if (y1-y0)>int(0.16*self.sh): y1=y0+int(0.16*self.sh)
            return (y0,y1)
        if bh_<10 or bh_>int(0.075*self.sh) or y1<int(0.45*cb.shape[0]): return None
        return (max(0,y0-4),min(cb.shape[0],y1+4))
    def update(self,f):
        self.fc+=1; tp=f[:self.tsh].astype(np.float32); bp=f[self.sh-self.bsh:].astype(np.float32)
        if self.ta is None: self.ta=tp.copy(); self.ba=bp.copy(); return
        a=0.93; self.ta=a*self.ta+(1-a)*tp; self.ba=a*self.ba+(1-a)*bp
        if self.fc>=self.warmup:
            tr=self._detect(f[:self.tsh],self.ta,top=True)
            bl=self._detect(f[self.sh-self.bsh:],self.ba,top=False)
            br=(self.sh-self.bsh+bl[0],self.sh-self.bsh+bl[1]) if bl else None
            if tr: self.top_overlay=tr; self.th=self.hold_max
            elif self.th>0: self.th-=1
            else: self.top_overlay=None
            if br: self.bottom_overlay=br; self.bh_hold=self.hold_max
            elif self.bh_hold>0: self.bh_hold-=1
            else: self.bottom_overlay=None
        self.exclusion_mask[:]=0
        if self.top_overlay: self.exclusion_mask[self.top_overlay[0]:self.top_overlay[1],:]=255
        if self.bottom_overlay: self.exclusion_mask[self.bottom_overlay[0]:self.bottom_overlay[1],:]=255
    def get_play_area_bounds(self):
        ty=(self.top_overlay[1]+6) if self.top_overlay else 0
        by=(self.bottom_overlay[0]-6) if self.bottom_overlay else self.sh
        ty=min(self.sh-32,ty); by=max(32,by)
        if by<=ty+32: ty,by=0,self.sh
        return ty,by
    def extract_top_strip(self,f):
        if not self.top_overlay: return None
        s=f[self.top_overlay[0]:self.top_overlay[1]]; return s.copy() if s.size else None
    def extract_bottom_strip(self,f):
        if not self.bottom_overlay: return None
        s=f[self.bottom_overlay[0]:self.bottom_overlay[1]]; return s.copy() if s.size else None

class SceneChangeDetector:
    def __init__(self,ht=0.55,pt=45.0,cd=8):
        self.ht=ht; self.pt=pt; self.cdm=cd; self.ph=None; self.pg=None; self.cd=0
    def check(self,gray):
        if self.cd>0: self.cd-=1
        h=cv2.calcHist([gray],[0],None,[64],[0,256]); cv2.normalize(h,h); cut=False
        if self.ph is not None and self.cd<=0:
            c=float(cv2.compareHist(self.ph,h,cv2.HISTCMP_CORREL))
            pd=float(np.mean(cv2.absdiff(gray,self.pg))) if self.pg is not None else 0.0
            if (1.0-c)>self.ht and pd>self.pt: cut=True; self.cd=self.cdm
        self.ph=h; self.pg=gray.copy(); return cut

class BallTracker:
    def __init__(self,sw,sh,sport_profile="auto"):
        self.sw,self.sh=int(sw),int(sh); self.sport=(sport_profile or "auto").strip().lower()
        if self.sport not in {"basketball","cricket","soccer"}: self.sport="generic"
        self.cx,self.cy=sw/2.0,sh/2.0; self.radius=0.0; self.conf=0.0; self.vx=0.0; self.vy=0.0
        self.missing_count=0; self.max_missing=12; md=min(sw,sh)
        self.min_r=max(3,int(round(md*0.006))); self.max_r=max(self.min_r+4,int(round(md*0.038)))
        self.gate_radius=max(self.sw*0.35,80.0)
    def _fm(self,hsv):
        if self.sport not in {"soccer","cricket"}: return None
        m=cv2.inRange(hsv,np.array([30,30,30]),np.array([85,255,255])); k=np.ones((7,7),np.uint8)
        return cv2.dilate(cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2),k,iterations=3)
    def _ch(self,gray):
        bl=cv2.GaussianBlur(gray,(7,7),1.5)
        c=cv2.HoughCircles(bl,cv2.HOUGH_GRADIENT,dp=1.2,minDist=max(16,int(min(self.sw,self.sh)*0.035)),param1=110,param2=15,minRadius=self.min_r,maxRadius=self.max_r)
        return [(float(x[0]),float(x[1]),float(x[2])) for x in (c[0][:25] if c is not None else [])]
    def _cc(self,gray,fm):
        e=cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.0),50,150)
        if fm is not None: e=cv2.bitwise_and(e,fm)
        e=cv2.dilate(e,np.ones((3,3),np.uint8),iterations=1); r=[]
        for c in cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            a_,p_=cv2.contourArea(c),cv2.arcLength(c,True)
            if p_>0 and 4*math.pi*a_/(p_*p_)>=0.55:
                (cx,cy),rad=cv2.minEnclosingCircle(c)
                if self.min_r<=rad<=self.max_r*1.3: r.append((float(cx),float(cy),float(rad)))
        r.sort(key=lambda t:abs(t[2]-(self.min_r+self.max_r)/2.0)); return r[:20]
    def _cb(self,hsv,fm):
        ms=[]
        if self.sport=="basketball": ms.append(cv2.inRange(hsv,np.array([3,80,80]),np.array([22,255,255])))
        elif self.sport=="cricket": ms+=[cv2.inRange(hsv,np.array([0,100,60]),np.array([10,255,255])),cv2.inRange(hsv,np.array([165,100,60]),np.array([179,255,255])),cv2.inRange(hsv,np.array([0,0,180]),np.array([179,45,255]))]
        elif self.sport=="soccer": ms.append(cv2.inRange(hsv,np.array([0,0,170]),np.array([179,60,255])))
        else: ms.append(cv2.inRange(hsv,np.array([0,0,180]),np.array([179,55,255])))
        cb_=ms[0]
        for m in ms[1:]: cb_=cv2.bitwise_or(cb_,m)
        if fm is not None and self.sport in {"soccer","cricket"}: cb_=cv2.bitwise_and(cb_,fm)
        k=np.ones((3,3),np.uint8); cb_=cv2.dilate(cv2.morphologyEx(cb_,cv2.MORPH_OPEN,k,iterations=1),k,iterations=1); r=[]
        for c in cv2.findContours(cb_,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            a_,p_=cv2.contourArea(c),cv2.arcLength(c,True)
            if p_>0 and 4*math.pi*a_/(p_*p_)>=0.40:
                (cx,cy),rad=cv2.minEnclosingCircle(c)
                if self.min_r*0.7<=rad<=self.max_r*1.5: r.append((float(cx),float(cy),float(rad)))
        r.sort(key=lambda t:abs(t[2]-(self.min_r+self.max_r)/2.0)); return r[:15]
    def _score(self,cx,cy,rad,hsv,mm,src):
        ri=max(3,int(rad)); patch=hsv[max(0,int(cy-ri)):min(self.sh,int(cy+ri+1)),max(0,int(cx-ri)):min(self.sw,int(cx+ri+1))]
        cs=0.0
        if patch.size>0:
            mh,ms_,mv=patch.reshape(-1,3).mean(axis=0)
            if self.sport=="basketball": cs=1.0 if (5<=mh<=22 and ms_>=80 and mv>=70) else (0.6 if (3<=mh<=28 and ms_>=50 and mv>=50) else 0.0)
            elif self.sport=="cricket": cs=max(1.0 if (ms_<=45 and mv>=170) else 0.0,1.0 if ((mh<=10 or mh>=165) and ms_>=90 and mv>=50) else 0.0)
            elif self.sport=="soccer": cs=0.9 if (ms_<=55 and mv>=160) else (0.5 if (ms_<=80 and mv>=130) else 0.0)
            else: cs=0.3 if mv>=150 else 0.0
        mo=0.0
        if mm is not None:
            mr=max(5,int(rad*2)); mp=mm[max(0,int(cy-mr)):min(self.sh,int(cy+mr+1)),max(0,int(cx-mr)):min(self.sw,int(cx+mr+1))]
            if mp.size>0: mo=float(np.count_nonzero(mp))/float(mp.size)
        d=math.hypot(cx-(self.cx+self.vx),cy-(self.cy+self.vy)); pr=max(0.0,1.0-d/self.gate_radius)
        sz=max(0.0,1.0-abs(rad-(self.min_r+self.max_r)/2.0)/max((self.min_r+self.max_r)/2.0,1.0))
        sb=0.15 if src=="multi" else 0.0
        if self.sport=="cricket": return 0.25*cs+0.25*mo+0.25*pr+0.15*sz+0.10*sb
        if self.sport=="basketball": return 0.35*cs+0.20*mo+0.22*pr+0.13*sz+0.10*sb
        if self.sport=="soccer": return 0.22*cs+0.28*mo+0.28*pr+0.12*sz+0.10*sb
        return 0.20*cs+0.30*mo+0.30*pr+0.10*sz+0.10*sb
    def update(self,frame,gray,mm,excl=None):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); dg=gray.copy()
        if excl is not None: dg[excl>0]=0
        fm=self._fm(hsv)
        raw=[(cx,cy,r,"hough") for cx,cy,r in self._ch(dg)]
        raw+=[(cx,cy,r,"contour") for cx,cy,r in self._cc(dg,fm)]
        raw+=[(cx,cy,r,"color") for cx,cy,r in self._cb(hsv,fm)]
        if excl is not None:
            raw=[(cx,cy,r,s) for cx,cy,r,s in raw if 0<=int(round(cy))<self.sh and 0<=int(round(cx))<self.sw and excl[int(round(cy)),int(round(cx))]==0]
        used=[False]*len(raw); md_=max(self.max_r*2.5,20.0); clusters=[]
        for i,(cx1,cy1,r1,s1) in enumerate(raw):
            if used[i]: continue
            gc,gy,gr,ss=[cx1],[cy1],[r1],{s1}; used[i]=True
            for j in range(i+1,len(raw)):
                if not used[j] and math.hypot(cx1-raw[j][0],cy1-raw[j][1])<md_:
                    gc.append(raw[j][0]);gy.append(raw[j][1]);gr.append(raw[j][2]);ss.add(raw[j][3]);used[j]=True
            clusters.append((sum(gc)/len(gc),sum(gy)/len(gy),sum(gr)/len(gr),"multi" if len(ss)>1 else list(ss)[0]))
        best,bs=None,-1.0
        for cx,cy,r,s in clusters:
            sc=self._score(cx,cy,r,hsv,mm,s)
            if sc>bs: bs=sc; best=(cx,cy,r,sc)
        if best is None or bs<0.15:
            self.missing_count+=1; self.conf*=0.88
            if self.missing_count>self.max_missing: self.conf=0.0
            return None
        cx,cy,r,sc=best
        self.vx=0.5*self.vx+0.5*(cx-self.cx); self.vy=0.5*self.vy+0.5*(cy-self.cy)
        self.cx=0.6*self.cx+0.4*cx; self.cy=0.6*self.cy+0.4*cy
        self.radius=(0.6*self.radius+0.4*r) if self.radius>0 else r
        self.conf=min(1.0,0.7*self.conf+0.55*sc); self.missing_count=0
        return (self.cx,self.cy,self.radius,self.conf)
    def reset_position(self,cx,cy): self.cx,self.cy,self.vx,self.vy=cx,cy,0.0,0.0; self.conf*=0.3


class TrackedFace:
    __slots__=("cx","cy","w","h","missing","age")
    def __init__(self,cx,cy,w,h): self.cx,self.cy,self.w,self.h=float(cx),float(cy),float(w),float(h); self.missing=0; self.age=1
    def update(self,cx,cy,w,h,pa=0.90,sa=0.92):
        self.cx=pa*self.cx+(1-pa)*cx; self.cy=pa*self.cy+(1-pa)*cy; self.w=sa*self.w+(1-sa)*w; self.h=sa*self.h+(1-sa)*h; self.missing=0; self.age+=1
    def mark_missing(self): self.missing+=1; self.age+=1

class PanelLayoutEngine:
    def __init__(self,sw,sh,tw,th,pos_smooth=0.90,size_smooth=0.92,switch_frames=25,persist=18,trans=10,gap=PANEL_GAP,stride=2):
        self.sw,self.sh,self.tw,self.th=int(sw),int(sh),int(tw),int(th)
        self.pa,self.sa=float(pos_smooth),float(size_smooth)
        self.switch_frames,self.persist,self.trans_total=int(switch_frames),int(persist),max(1,int(trans))
        self.gap,self.stride=int(gap),max(1,int(stride))
        self.face_det=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        self.tracks=[]; self.current_layout=0; self.layout_cand=0; self.layout_hold=0
        self.trans_rem=0; self.prev_out=None; self.fidx=0
    def _detect(self,gray):
        ms=max(50,int(min(self.sw,self.sh)*0.06))
        try: faces=self.face_det.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(ms,ms))
        except Exception: faces=[]
        return [(int(x),int(y),int(w),int(h)) for (x,y,w,h) in faces]
    def _match(self,dets):
        dc=[(x+w/2.0,y+h/2.0,float(w),float(h)) for x,y,w,h in dets]
        mt,md_=set(),set(); mx=max(self.sw,self.sh)*0.25; pairs=[]
        for di,(dcx,dcy,_,_) in enumerate(dc):
            for ti,t in enumerate(self.tracks): pairs.append((math.hypot(dcx-t.cx,dcy-t.cy),di,ti))
        pairs.sort(key=lambda x:x[0])
        for d,di,ti in pairs:
            if di in md_ or ti in mt or d>mx: continue
            dcx,dcy,dw,dh=dc[di]; self.tracks[ti].update(dcx,dcy,dw,dh,self.pa,self.sa); mt.add(ti); md_.add(di)
        for ti in range(len(self.tracks)):
            if ti not in mt: self.tracks[ti].mark_missing()
        for di,(dcx,dcy,dw,dh) in enumerate(dc):
            if di not in md_: self.tracks.append(TrackedFace(dcx,dcy,dw,dh))
        self.tracks=[t for t in self.tracks if t.missing<=self.persist]
    def _active(self):
        a=[t for t in self.tracks if t.age>=3]; a.sort(key=lambda t:t.cx); return a
    def _decide(self,n):
        tgt=min(n,4) if n>=2 else 0
        if tgt==self.layout_cand: self.layout_hold+=1
        else: self.layout_cand=tgt; self.layout_hold=1
        if self.layout_hold>=self.switch_frames and self.layout_cand!=self.current_layout:
            old=self.current_layout; self.current_layout=self.layout_cand
            if old>=2: self.trans_rem=self.trans_total
        return self.current_layout
    def _crop_person(self,frame,face,cw,ch):
        bw,bh=face.w*3.2,face.h*3.8; bcx,bcy=face.cx,face.cy+face.h*0.45
        car=cw/max(ch,1); bar=bw/max(bh,1)
        if bar>car: bh=bw/max(car,0.01)
        else: bw=bh*car
        x0=max(0,int(round(bcx-bw/2))); y0=max(0,int(round(bcy-bh/2)))
        x1=min(self.sw,int(round(bcx+bw/2))); y1=min(self.sh,int(round(bcy+bh/2)))
        if x1<=x0 or y1<=y0: return np.zeros((ch,cw,3),dtype=np.uint8)
        crop=frame[y0:y1,x0:x1]
        return _resize_cover(crop,cw,ch) if crop.size else np.zeros((ch,cw,3),dtype=np.uint8)
    def _render(self,frame,faces,layout,aw,ah):
        out=np.zeros((ah,aw,3),dtype=np.uint8); g=self.gap
        if layout==2:
            ch=(ah-g)//2
            for i,f in enumerate(faces[:2]):
                c=self._crop_person(frame,f,aw,ch); yo=i*(ch+g)
                rh,rw=min(ch,c.shape[0]),min(aw,c.shape[1]); out[yo:yo+rh,:rw]=c[:rh,:rw]
            if g>0: out[ch:ch+g,:]=20
        elif layout==3:
            top_h=(ah-g)//2; bot_h=ah-g-top_h
            if faces:
                c=self._crop_person(frame,faces[0],aw,top_h); rh,rw=min(top_h,c.shape[0]),min(aw,c.shape[1]); out[:rh,:rw]=c[:rh,:rw]
            if g>0: out[top_h:top_h+g,:]=20
            cw_=(aw-g)//2
            for i,f in enumerate(faces[1:3]):
                c=self._crop_person(frame,f,cw_,bot_h); xo=i*(cw_+g)
                rh,rw=min(bot_h,c.shape[0]),min(cw_,c.shape[1]); out[top_h+g:top_h+g+rh,xo:xo+rw]=c[:rh,:rw]
            if g>0 and len(faces)>2: out[top_h+g:,cw_:cw_+g]=20
        elif layout>=4:
            cw_=(aw-g)//2; ch=(ah-g)//2; pos=[(0,0),(cw_+g,0),(0,ch+g),(cw_+g,ch+g)]
            for i,f in enumerate(faces[:4]):
                c=self._crop_person(frame,f,cw_,ch); xo,yo=pos[i]
                rh,rw=min(ch,c.shape[0]),min(cw_,c.shape[1]); out[yo:yo+rh,xo:xo+rw]=c[:rh,:rw]
            if g>0: out[:,cw_:cw_+g]=20; out[ch:ch+g,:]=20
        return out
    def process(self,frame,gray):
        self.fidx+=1
        if self.fidx%self.stride==0: self._match(self._detect(gray))
        else:
            for t in self.tracks: t.age+=1
        active=self._active(); layout=self._decide(len(active))
        if layout<2: return None
        po=self._render(frame,active,layout,self.tw,self.th)
        if self.trans_rem>0 and self.prev_out is not None:
            a=self.trans_rem/self.trans_total
            if self.prev_out.shape==po.shape: po=cv2.addWeighted(po,1.0-a,self.prev_out,a,0)
            self.trans_rem-=1
        self.prev_out=po.copy(); return po

class SmoothReframer:
    def __init__(self,sw,sh,tw,th,smooth_strength=0.975,analysis_stride=4,deadzone_ratio=0.05,max_pan_ratio=0.012,sport_profile="auto",ball_tracking=True,ball_weight=0.55,context_bias=0.20,overlay_composite=True,preserve_bottom_overlay=False,panel_mode=False):
        self.sw,self.sh,self.tw,self.th=int(sw),int(sh),int(tw),int(th)
        self.cw,self.ch=_vertical_crop_box(sw,sh); self.mx=max(0,sw-self.cw); self.my=max(0,sh-self.ch)
        self.ss=float(smooth_strength); self.astride=max(1,int(analysis_stride))
        self.dz=max(8.0,self.cw*deadzone_ratio); self.mp=max(2.0,self.cw*max_pan_ratio)
        self.bw_=float(ball_weight); self.cb_=float(context_bias)
        self.oc=bool(overlay_composite); self.pbo=bool(preserve_bottom_overlay)
        self.pm=bool(panel_mode); self.sp=(sport_profile or "auto").strip().lower()
        self.fd=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        self.sal=None
        try:
            if hasattr(cv2,"saliency"): self.sal=cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception: pass
        self.od=OverlayDetector(sw,sh); self.scd=SceneChangeDetector()
        self.bt=BallTracker(sw,sh,sport_profile=self.sp) if ball_tracking else None
        self.pe=PanelLayoutEngine(sw,sh,tw,th) if panel_mode else None
        self.scx,self.scy=sw/2.0,sh/2.0; self.tcx,self.tcy=self.scx,self.scy
        self.pg=None; self.fidx=0
    def _dm(self,gray):
        if self.pg is None: return [],None
        diff=cv2.GaussianBlur(cv2.absdiff(gray,self.pg),(9,9),0)
        _,mo=cv2.threshold(diff,18,255,cv2.THRESH_BINARY)
        k=np.ones((3,3),np.uint8); mo=cv2.dilate(cv2.morphologyEx(mo,cv2.MORPH_OPEN,k,iterations=1),None,iterations=2)
        ex=self.od.exclusion_mask
        if ex is not None: mo[ex>0]=0
        cs,_=cv2.findContours(mo,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); ma=0.006*self.sw*self.sh
        boxes=[]
        for c in sorted(cs,key=cv2.contourArea,reverse=True)[:10]:
            x,y,w,h=cv2.boundingRect(c)
            if w*h>ma: boxes.append((x,y,w,h))
        return boxes,mo
    def _compose(self,crop,ts,bs):
        th_=0
        if self.oc and ts is not None and ts.size:
            th_=int(round(self.th*0.082))
            if ts.shape[0]<0.05*self.sh: th_=int(round(self.th*0.075))
            th_=int(_clamp(th_,42,int(self.th*0.11)))
        bh_=0
        if self.oc and self.pbo and bs is not None and bs.size:
            bh_=int(_clamp(int(round(self.th*0.045)),20,int(self.th*0.07)))
        mid=max(1,self.th-th_-bh_); out=np.zeros((self.th,self.tw,3),dtype=np.uint8)
        out[th_:th_+mid,:]=_resize_cover(crop,self.tw,mid)
        if th_>0 and ts is not None and ts.size:
            out[:th_,:]=cv2.resize(ts,(self.tw,th_),interpolation=cv2.INTER_AREA)
            cv2.line(out,(0,th_-1),(self.tw-1,th_-1),(10,10,10),1)
        if bh_>0 and bs is not None and bs.size:
            out[self.th-bh_:,:]=cv2.resize(bs,(self.tw,bh_),interpolation=cv2.INTER_AREA)
            cv2.line(out,(0,self.th-bh_),(self.tw-1,self.th-bh_),(10,10,10),1)
        return out
    def process(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); self.od.update(frame)
        if self.pm and self.pe is not None:
            pr=self.pe.process(frame,gray)
            if pr is not None:
                ts=self.od.extract_top_strip(frame) if self.oc else None
                bs=self.od.extract_bottom_strip(frame) if self.oc else None
                if ts is not None or bs is not None: pr=self._compose(pr,ts,bs)
                self.pg=gray; self.fidx+=1; return pr
        ic=self.scd.check(gray)
        if ic:
            self.scx=(self.scx+self.sw/2.0)/2.0; self.scy=(self.scy+self.sh/2.0)/2.0
            if self.bt: self.bt.reset_position(self.sw/2.0,self.sh/2.0)
        if self.fidx%self.astride==0:
            cands=[]; ex=self.od.exclusion_mask; pt,pb=self.od.get_play_area_bounds()
            try:
                pg_=gray[pt:pb,:]; faces=self.fd.detectMultiScale(pg_,scaleFactor=1.15,minNeighbors=4,minSize=(32,32))
            except Exception: faces=[]
            for (x,y,w,h) in faces[:3]: cands.append((0.30,(x+w/2.0,pt+y+h/2.0)))
            mb,mm=self._dm(gray)
            if mb:
                x0,y0=min(p[0] for p in mb),min(p[1] for p in mb)
                x1,y1=max(p[0]+p[2] for p in mb),max(p[1]+p[3] for p in mb)
                cands.append((0.34,((x0+x1)/2.0,(y0+y1)/2.0)))
                bx,by,bw_,bh_=mb[0]; cands.append((0.16,(bx+bw_/2.0,by+bh_/2.0)))
            else: mm=None
            if self.sal:
                try:
                    ok,sm=self.sal.computeSaliency(frame)
                    if ok:
                        sm=(sm*255).astype("uint8")
                        if ex is not None: sm[ex>0]=0
                        _,th_=cv2.threshold(sm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        cs_,_=cv2.findContours(th_,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        if cs_:
                            c=max(cs_,key=cv2.contourArea); x,y,w,h=cv2.boundingRect(c)
                            if w*h>0.02*self.sw*self.sh: cands.append((0.08,(x+w/2.0,y+h/2.0)))
                except Exception: pass
            if self.bt:
                ball=self.bt.update(frame,gray,mm,ex)
                if ball:
                    bx,by,_,bc=ball; cands.append((self.bw_*max(0.25,bc),(bx,by)))
                    if mb:
                        mx0,my0=min(p[0] for p in mb),min(p[1] for p in mb)
                        mx1,my1=max(p[0]+p[2] for p in mb),max(p[1]+p[3] for p in mb)
                        cands.append((self.cb_,((mx0+mx1)/2.0,(my0+my1)/2.0)))
            if cands:
                ws=sum(w for w,_ in cands)
                self.tcx=sum(cx*w for w,(cx,_) in cands)/max(ws,1e-6)
                self.tcy=sum(cy*w for w,(_,cy) in cands)/max(ws,1e-6)
            else: self.tcx,self.tcy=self.sw/2.0,self.sh/2.0
            ym=max(12,int(0.02*self.sh)); self.tcy=_clamp(self.tcy,pt+ym,pb-ym)
        self.pg=gray
        dx=self.tcx-self.scx; dy=self.tcy-self.scy
        if abs(dx)<self.dz: dx=0.0
        if abs(dy)<self.dz*0.45: dy=0.0
        al=(1.0-self.ss)*(3.0 if ic else 1.0)
        self.scx+=max(-self.mp,min(self.mp,dx*al)); self.scy+=max(-(self.mp*0.45),min(self.mp*0.45,dy*al))
        pt,pb=self.od.get_play_area_bounds()
        x0=int(_clamp(round(self.scx-self.cw/2.0),0,self.mx))
        ph_=pb-pt
        if ph_>=self.ch: y0=int(_clamp(round(self.scy-self.ch/2.0),pt,pb-self.ch))
        else:
            y0=int(_clamp(round(self.scy-self.ch/2.0),0,self.my))
            if self.od.top_overlay and y0<pt: y0=min(self.my,pt)
            if self.od.bottom_overlay and y0+self.ch>pb: y0=max(0,pb-self.ch)
        crop=frame[y0:y0+self.ch,x0:x0+self.cw]
        if crop.size==0: crop=frame
        ts=self.od.extract_top_strip(frame) if self.oc else None
        bs=self.od.extract_bottom_strip(frame) if self.oc else None
        self.fidx+=1; return self._compose(crop,ts,bs)

def create_vertical_master(source_path,output_path,target_w=DEFAULT_TARGET_W,target_h=DEFAULT_TARGET_H,smooth_strength=0.975,analysis_stride=4,deadzone_ratio=0.05,max_pan_ratio=0.012,sport_profile="auto",ball_tracking=True,overlay_composite=True,preserve_bottom_overlay=False,panel_mode=False,progress_cb=None):
    cap=cv2.VideoCapture(source_path)
    if not cap.isOpened(): return False,"Could not open input source"
    sw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); sh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps=float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_OUTPUT_FPS); fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if sw<=0 or sh<=0: cap.release(); return False,"Invalid source dimensions"
    rf=SmoothReframer(sw,sh,target_w,target_h,smooth_strength=smooth_strength,analysis_stride=analysis_stride,deadzone_ratio=deadzone_ratio,max_pan_ratio=max_pan_ratio,sport_profile=sport_profile,ball_tracking=ball_tracking,overlay_composite=overlay_composite,preserve_bottom_overlay=preserve_bottom_overlay,panel_mode=panel_mode)
    wr=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"mp4v"),fps if fps>0 else DEFAULT_OUTPUT_FPS,(target_w,target_h))
    if not wr.isOpened(): cap.release(); return False,"Could not create output file"
    idx=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            wr.write(rf.process(frame)); idx+=1
            if progress_cb and fc>0 and idx%5==0: progress_cb(idx/fc,f"Creating vertical master {idx}/{fc}")
    finally: cap.release(); wr.release()
    return True,"Done"

@dataclass
class CFStreamConfig:
    account_id: str; api_token: str; customer_code: str; prefer_low_latency: bool = False

@dataclass
class LiveSession:
    uid: str; rtmps_url: str; stream_key: str; hls_url: str; dash_url: str; iframe_url: str
    ffmpeg_cmd: list[str]; proc: Optional[subprocess.Popen]; log_path: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker: Optional[threading.Thread] = None; status: str = "created"
    stats: dict = field(default_factory=dict); error: str = ""

def cfstream_config_from_inputs(account_id,api_token,customer_code,prefer_low_latency=False):
    if not account_id: raise ValueError("Cloudflare account ID is required.")
    if not api_token: raise ValueError("Cloudflare API token is required.")
    if not customer_code: raise ValueError("Cloudflare customer code is required.")
    code=customer_code.strip().replace("customer-","").replace(".cloudflarestream.com","").strip("/")
    return CFStreamConfig(account_id.strip(),api_token.strip(),code,bool(prefer_low_latency))

def _cf_api_request(cfg,method,path,payload=None):
    url=f"https://api.cloudflare.com/client/v4{path}"
    data=json.dumps(payload).encode("utf-8") if payload is not None else None
    headers={"Authorization":f"Bearer {cfg.api_token}","Content-Type":"application/json","User-Agent":"DualFlow-Vertical-Cloudflare"}
    req=urllib.request.Request(url,data=data,headers=headers,method=method)
    try:
        with urllib.request.urlopen(req,timeout=60) as resp:
            body=resp.read().decode("utf-8"); return resp.status,(_safe_json_loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body=exc.read().decode("utf-8",errors="ignore"); p=_safe_json_loads(body) if body else {}
        if not p: p={"success":False,"errors":[{"message":body}]}
        return exc.code,p

def create_live_input(cfg,name,recording_mode="automatic"):
    payload={"meta":{"name":name},"recording":{"mode":recording_mode,"timeoutSeconds":0},"preferLowLatency":bool(cfg.prefer_low_latency),"enabled":True}
    status,parsed=_cf_api_request(cfg,"POST",f"/accounts/{cfg.account_id}/stream/live_inputs",payload)
    if status not in (200,201) or not parsed.get("success"): raise RuntimeError(f"Create live input failed: {parsed}")
    return parsed["result"]

def disable_live_input(cfg,uid): _cf_api_request(cfg,"PUT",f"/accounts/{cfg.account_id}/stream/live_inputs/{uid}",{"enabled":False})

def build_public_playback_urls(cfg,uid):
    base=f"https://customer-{cfg.customer_code}.cloudflarestream.com/{uid}"
    hls=f"{base}/manifest/video.m3u8"+("?protocol=llhls" if cfg.prefer_low_latency else "")
    return hls,f"{base}/manifest/video.mpd",f"{base}/iframe?autoplay=true&muted=true&controls=true&preload=metadata"

def build_push_file_command(reframed_mp4,rtmps_url,stream_key,loop_input=True,output_fps=DEFAULT_OUTPUT_FPS):
    t=rtmps_url.rstrip("/")+"/"+stream_key; fi=max(24,min(60,int(round(output_fps or DEFAULT_OUTPUT_FPS))))
    cmd=["ffmpeg","-hide_banner","-loglevel","warning","-y"]
    if loop_input: cmd+=["-stream_loop","-1"]
    cmd+=["-re","-i",reframed_mp4,"-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-vsync","cfr","-r",str(fi),"-b:v",DEFAULT_VIDEO_BITRATE,"-maxrate",DEFAULT_MAXRATE,"-bufsize",DEFAULT_BUFSIZE,"-g",str(fi*2),"-keyint_min",str(fi*2),"-sc_threshold","0","-profile:v","high","-x264-params",f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fi*2}:min-keyint={fi*2}","-c:a","aac","-b:a","128k","-ar","48000","-ac","2","-f","flv",t]
    return cmd

def start_vod_to_live_push(cfg,reframed_mp4,asset_name,loop_input=True,output_fps=DEFAULT_OUTPUT_FPS):
    li=create_live_input(cfg,name=safe_token(Path(asset_name).stem))
    uid,ru,sk=li["uid"],li["rtmps"]["url"],li["rtmps"]["streamKey"]
    hls,dash,iframe=build_public_playback_urls(cfg,uid)
    cmd=build_push_file_command(reframed_mp4,ru,sk,loop_input,output_fps=output_fps)
    lp=tempfile.NamedTemporaryFile(delete=False,suffix=".log").name
    proc=subprocess.Popen(cmd,stdout=open(lp,"w",encoding="utf-8"),stderr=subprocess.STDOUT,text=True)
    return LiveSession(uid,ru,sk,hls,dash,iframe,cmd,proc,lp,status="streaming")

def build_realtime_rtmps_push_command(tw,th,fps,rtmps_url,stream_key):
    t=rtmps_url.rstrip("/")+"/"+stream_key; fi=max(24,min(60,int(round(fps or DEFAULT_OUTPUT_FPS))))
    return ["ffmpeg","-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{tw}x{th}","-r",str(fi),"-i","-","-f","lavfi","-i","anullsrc=r=48000:cl=stereo","-shortest","-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","veryfast","-tune","zerolatency","-pix_fmt","yuv420p","-vsync","cfr","-r",str(fi),"-b:v",DEFAULT_VIDEO_BITRATE,"-maxrate",DEFAULT_MAXRATE,"-bufsize",DEFAULT_BUFSIZE,"-g",str(fi*2),"-keyint_min",str(fi*2),"-sc_threshold","0","-profile:v","high","-x264-params",f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fi*2}:min-keyint={fi*2}","-c:a","aac","-b:a","128k","-ar","48000","-ac","2","-f","flv",t]

def _read_exact(stream,nbytes):
    chunks=[]; rem=nbytes
    while rem>0:
        d=stream.read(rem)
        if not d: break
        chunks.append(d); rem-=len(d)
    return b"".join(chunks)

def _build_ingest_command(source,fps,pace_input,loop_file):
    fi=max(24,min(60,int(round(fps or DEFAULT_OUTPUT_FPS))))
    vf=f"fps={fi},scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:force_original_aspect_ratio=decrease,pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:(ow-iw)/2:(oh-ih)/2:black"
    cmd=["ffmpeg","-hide_banner","-loglevel","warning"]+_source_input_args(source,pace_input=pace_input,loop_file=loop_file)+["-an","-vf",vf,"-pix_fmt","bgr24","-f","rawvideo","-"]
    return cmd

def _make_placeholder_frame(tw,th,text="Starting stream..."):
    f=np.zeros((th,tw,3),dtype=np.uint8); f[:]=(18,22,36)
    cv2.rectangle(f,(0,0),(tw,int(th*0.18)),(35,55,98),-1)
    cv2.rectangle(f,(0,int(th*0.82)),(tw,th),(24,34,60),-1)
    cv2.putText(f,"Vertical stream",(28,max(48,th//10)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(f,text,(28,th//2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(210,220,255),2,cv2.LINE_AA)
    return f

def _open_ingest(source,fps,pace_input,loop_file,log_path):
    cmd=_build_ingest_command(source,fps=fps,pace_input=pace_input,loop_file=loop_file)
    lf=open(log_path,"a",encoding="utf-8"); lf.write("\n=== INGEST CMD ===\n"+" ".join(cmd)+"\n"); lf.flush()
    return subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=lf,bufsize=0)

def _start_output(session):
    lf=open(session.log_path,"a",encoding="utf-8"); lf.write("\n=== PUSH CMD ===\n"+" ".join(session.ffmpeg_cmd)+"\n"); lf.flush()
    return subprocess.Popen(session.ffmpeg_cmd,stdin=subprocess.PIPE,stdout=lf,stderr=subprocess.STDOUT,bufsize=0)

def _realtime_worker(session,source,tw,th,delay_seconds,smooth_strength,analysis_stride,deadzone_ratio,max_pan_ratio,loop_file,pace_input,sport_profile,ball_tracking,overlay_composite,preserve_bottom_overlay,panel_mode):
    session.status="probing"; info=probe_source(source)
    fps=DEFAULT_OUTPUT_FPS; sw,sh=WORKING_INPUT_W,WORKING_INPUT_H; fb=sw*sh*3; df=max(1,int(round(delay_seconds*fps)))
    session.stats={"fps":round(fps,3),"delay_frames":df,"working_resolution":f"{sw}x{sh}","source_reported_resolution":f"{int(info.get('width',0))}x{int(info.get('height',0))}","sport_profile":sport_profile,"panel_mode":panel_mode}
    rf=SmoothReframer(sw,sh,tw,th,smooth_strength=smooth_strength,analysis_stride=analysis_stride,deadzone_ratio=deadzone_ratio,max_pan_ratio=max_pan_ratio,sport_profile=sport_profile,ball_tracking=ball_tracking,overlay_composite=overlay_composite,preserve_bottom_overlay=preserve_bottom_overlay,panel_mode=panel_mode)
    buf=collections.deque(maxlen=max(df+240,600)); ph=_make_placeholder_frame(tw,th); fi=1.0/fps; pf=0; ss_=0
    try: session.proc=_start_output(session)
    except Exception as exc: session.status="ffmpeg_start_failed"; session.error=str(exc); return
    session.status="priming_output"
    try:
        if session.proc.stdin:
            for _ in range(int(max(1.0,min(delay_seconds/2.0,3.0))*fps)):
                if session.stop_event.is_set(): break
                session.proc.stdin.write(ph.tobytes()); pf+=1
    except Exception as exc: session.status="ffmpeg_pipe_broken"; session.error=f"Prime failed: {exc}"; return
    ingest=None; nd=time.monotonic(); fin=0; fout=0; se=False
    try:
        session.status="connecting_source"
        ingest=_open_ingest(source,fps=fps,pace_input=pace_input,loop_file=loop_file,log_path=session.log_path)
        while not session.stop_event.is_set():
            if not se:
                raw=_read_exact(ingest.stdout,fb) if ingest and ingest.stdout else b""
                if len(raw)<fb: se=True; ss_+=1; session.error=f"Source unavailable: {source}"
                else: frame=np.frombuffer(raw,dtype=np.uint8).reshape((sh,sw,3)); buf.append(rf.process(frame)); fin+=1
            ftw=None
            if len(buf)>=df: session.status="streaming"; ftw=buf.popleft()
            elif not se: session.status="buffering"; ftw=ph; pf+=1
            elif buf: session.status="draining"; ftw=buf.popleft()
            else: session.status="source_ended"; break
            if session.proc and session.proc.stdin and ftw is not None:
                try: session.proc.stdin.write(ftw.tobytes()); fout+=1
                except Exception as exc: session.status="ffmpeg_pipe_broken"; session.error=str(exc); break
            nd+=fi; sl=nd-time.monotonic()
            if sl>0: time.sleep(sl)
            else: nd=time.monotonic()
            if fout%int(max(1.0,fps))==0:
                session.stats.update({"frames_in":fin,"frames_out":fout,"buffer_len":len(buf),"delay_seconds":round(df/max(fps,1.0),2),"placeholder_frames":pf,"source_stalls":ss_,"ball_confidence":round(rf.bt.conf,3) if rf.bt else 0.0,"overlay_top":rf.od.top_overlay is not None,"overlay_bottom":rf.od.bottom_overlay is not None,"panel_active":rf.pe.current_layout>=2 if rf.pe else False})
    except Exception as exc: session.status="worker_error"; session.error=str(exc)
    finally:
        try:
            if ingest and ingest.poll() is None:
                ingest.terminate()
                try: ingest.wait(timeout=3)
                except Exception: ingest.kill()
        except Exception: pass
        try:
            if session.proc and session.proc.stdin: session.proc.stdin.close()
        except Exception: pass
        if session.status not in {"ffmpeg_pipe_broken","worker_error","ffmpeg_start_failed","source_ended"}: session.status="stopped"

def start_realtime_delayed_vertical_push(cfg,source,asset_name,target_w=DEFAULT_TARGET_W,target_h=DEFAULT_TARGET_H,delay_seconds=20.0,smooth_strength=0.975,analysis_stride=4,deadzone_ratio=0.05,max_pan_ratio=0.012,loop_file=False,pace_input=True,sport_profile="auto",ball_tracking=True,overlay_composite=True,preserve_bottom_overlay=False,panel_mode=False):
    li=create_live_input(cfg,name=safe_token(Path(asset_name).stem))
    uid,ru,sk=li["uid"],li["rtmps"]["url"],li["rtmps"]["streamKey"]
    hls,dash,iframe=build_public_playback_urls(cfg,uid)
    fc=build_realtime_rtmps_push_command(target_w,target_h,DEFAULT_OUTPUT_FPS,ru,sk)
    lp=tempfile.NamedTemporaryFile(delete=False,suffix=".log").name
    session=LiveSession(uid,ru,sk,hls,dash,iframe,fc,None,lp)
    w=threading.Thread(target=_realtime_worker,args=(session,source,target_w,target_h,delay_seconds,smooth_strength,analysis_stride,deadzone_ratio,max_pan_ratio,loop_file,pace_input,sport_profile,ball_tracking,overlay_composite,preserve_bottom_overlay,panel_mode),daemon=True)
    session.worker=w; w.start(); return session

def stop_live_session(cfg,session):
    if not session: return
    session.stop_event.set()
    try:
        if session.worker and session.worker.is_alive(): session.worker.join(timeout=3)
    except Exception: pass
    try:
        if session.proc and session.proc.poll() is None:
            session.proc.terminate()
            try: session.proc.wait(timeout=5)
            except Exception: session.proc.kill()
    except Exception: pass
    try: disable_live_input(cfg,session.uid)
    except Exception: pass

def read_log_tail(path,max_chars=12000):
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as fp: return fp.read()[-max_chars:]
    except Exception: return ""
