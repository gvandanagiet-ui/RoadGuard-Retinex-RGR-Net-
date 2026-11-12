"""INDEPENDENT REFERENCE REIMPLEMENTATION; NOT ORIGINAL RGR-Net SOFTWARE.
Published architectural concepts only; arbitrary layer widths/Q weights MUST NOT be
presented as recovered original implementation or reviewer-valid RGR-Net results.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_c,out_c,k=3):
    return nn.Sequential(nn.Conv2d(in_c,out_c,k,padding=k//2), nn.LeakyReLU(0.1,inplace=True))


def grad(v):
    return v[:,:,:,1:]-v[:,:,:,:-1], v[:,:,1:,:]-v[:,:,:-1,:]


@torch.no_grad()
def condition_maps(image):
    rgb=image.clamp(0,1)
    y=0.299*rgb[:,0:1]+0.587*rgb[:,1:2]+0.114*rgb[:,2:3]
    yp=F.pad(y,(7,8,7,8),mode='replicate')
    exposure=F.avg_pool2d(yp,16,1)
    mx=rgb.max(1,keepdim=True).values
    mn=rgb.min(1,keepdim=True).values
    hsv_s=(mx-mn)/(mx+1e-6)
    glare=F.max_pool2d(((mx>=.85)&(hsv_s<=.35)).float(),3,1,1)
    lap=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],device=rgb.device).view(1,1,3,3)
    high=F.conv2d(F.pad(y,(1,1,1,1),mode='replicate'),lap).abs()
    window=F.pad(high,(2,2,2,2),mode='replicate').unfold(2,5,1).unfold(3,5,1)
    med=window.contiguous().flatten(-2).median(-1).values
    mad=(window-med[...,None,None]).abs().flatten(-2).median(-1).values
    norm=torch.quantile(mad.flatten(2),.95,dim=2).view(-1,1,1,1).clamp_min(1e-6)
    noise=(mad/norm).clamp(0,1)
    schx=torch.tensor([[3.,0.,-3.],[10.,0.,-10.],[3.,0.,-3.]],device=rgb.device).view(1,1,3,3)/16.
    schy=schx.transpose(2,3)
    py=F.pad(y,(1,1,1,1),mode='replicate')
    structure=(F.conv2d(py,schx).square()+F.conv2d(py,schy).square()+1e-9).sqrt()
    scale=torch.quantile(structure.flatten(2),.95,dim=2).view(-1,1,1,1).clamp_min(1e-6)
    structure=(structure/scale).clamp(0,1)
    shadow=(y<=.15).float()
    # These Q COEFFICIENTS ARE PROVISIONAL REFERENCE CHOICES; unreleased
    # paper Q coefficients remain unknown and MUST be confirmed by authors.
    reliability=(.3*(1-glare)+.3*(1-noise)+.2*structure+.2*(1-exposure)).clamp(0,1)
    return dict(E=exposure,G=glare,N=noise,S=structure,D=shadow,Q=reliability)


class ReferenceRGR(nn.Module):
    def __init__(self,width=12,dual_gate=True,use_q=True,illum_gate=True,noise_gate=True):
        super().__init__()
        self.dual_gate=dual_gate;self.use_q=use_q;self.illum_gate=illum_gate;self.noise_gate=noise_gate
        self.idn=nn.Sequential(_conv(3,width),_conv(width,width),nn.Conv2d(width,4,3,padding=1))
        self.gate_illum=nn.Conv2d(4,1,1)
        self.gate_noise=nn.Conv2d(4,1,1) if dual_gate else self.gate_illum
        self.icn=nn.Sequential(_conv(2,width),_conv(width,width),nn.Conv2d(width,1,3,padding=1))
        self.nrn=nn.Sequential(_conv(4,width),_conv(width,width),nn.Conv2d(width,3,3,padding=1))
    def forward(self,x, perturb=False):
        m=condition_maps(x)
        if not self.use_q: m['Q']=torch.ones_like(m['Q'])
        e,n,s=m['E'],m['N'],m['S']
        if perturb:
            e=(e+torch.randn_like(e)*.01).clamp(0,1)
            n=(n+torch.randn_like(n)*.01).clamp(0,1)
            s=(s+torch.randn_like(s)*.01).clamp(0,1)
        illum_desc=torch.cat((e,m['G'],m['Q'],s),dim=1)
        noise_desc=torch.cat((n,1-m['Q'],s,m['G']),dim=1)
        g_i=torch.sigmoid(self.gate_illum(illum_desc)/.1)
        g_n=torch.sigmoid(self.gate_noise(noise_desc if self.dual_gate else illum_desc)/.1)
        if not self.illum_gate:g_i=torch.ones_like(g_i)
        if not self.noise_gate:g_n=torch.ones_like(g_n)
        rl=torch.sigmoid(self.idn(x))
        refl,lum=rl[:,:3],rl[:,3:]
        lum_corr=torch.sigmoid(self.icn(torch.cat((lum,g_i),1)))
        prelim=(refl*lum_corr).clamp(0,1)
        residual=.15*torch.tanh(self.nrn(torch.cat((prelim,g_n),1)))
        out=(prelim-residual).clamp(0,1)
        return out,dict(maps=m,R=refl,L=lum,g_i=g_i,g_n=g_n,residual=residual)


def unsup_loss(input_,enhanced,aux, spatial_target=True,structure_loss=True,consistency=None):
    m=aux['maps'];y=enhanced.mean(1,keepdim=True)
    recon=(aux['R']*aux['L']-input_).abs().mean()
    target=(.55+.15*m['D']-.25*m['G']).clamp(.30,.70) if spatial_target else .55
    exposure=(F.avg_pool2d(y,8,8)-F.avg_pool2d(target if torch.is_tensor(target) else torch.full_like(y,target),8,8)).abs().mean()
    gx,gy=grad(y);sx,sy=grad(input_.mean(1,keepdim=True))
    structure=((gx-sx).abs().mean()+(gy-sy).abs().mean()) if structure_loss else y.new_tensor(0.)
    lum_gx,lum_gy=grad(aux['L']);illum_smooth=(lum_gx.abs().mean()+lum_gy.abs().mean())
    noise=aux['residual'].abs().mean()
    color=torch.stack([enhanced[:,i].mean() for i in range(3)]).std(unbiased=False)
    consist=((enhanced-consistency).abs().mean() if consistency is not None else y.new_tensor(0.))
    total=recon+.4*exposure+.03*illum_smooth+.05*noise+.01*color+.10*structure+.05*consist
    return total, {k:float(v.detach()) for k,v in dict(total=total,recon=recon,exposure=exposure,structure=structure,consistency=consist).items()}
