function fcost=HHmododeslizante2(x1,x2,x3)

warning off
format long 
%close all
%%
alfa=0.3; 
Tsin=50;
L0= 1;
T0= 3;
alpha1 = x1;
beta1 = x2;
Lambda=x3;
noise=10;
tao=3e-3/20;
k5=0.3;
amp=1;
%%
options = simset('SrcWorkspace','current');
simout=sim('s1',[],options);
%%
 y=simout.salida;
 t=simout.tout;
 N_1=simout.salida5;
 referen=simout.salida6;
 acc_control=simout.salida7;
 Gainadd=simout.salida8;
%%
 H= stepinfo(y,t,1);
 L = H.Overshoot
 Ts = H.SettlingTime
 if isnan(Ts), Ts=100; end , if isnan(L),  L=1000 ; end 
 ye=y(end-0.3*length(y):end);
 E=abs(1-sum(ye)/length(ye));
 if max(acc_control>50)
     pacc=10;
 else
     pacc=0;
 end

 fcost= alfa*abs(L-L0)/L0 +E+pacc+(1-alfa)*abs(Ts-T0)/T0;

end

