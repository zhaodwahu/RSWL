function [model] = main_function(X,Y,optmParameter)

lambda1=optmParameter.lambda1;
lambda2=optmParameter.lambda2;
lambda4=optmParameter.lambda3;
lambda3=optmParameter.lambda4;

p=optmParameter.p;

percent = optmParameter.percent;
rho=optmParameter.rho;
kernel_para=optmParameter.kernel_para;

max_iter=optmParameter.maxIter;

%% initializtion
[n,q]=size(Y);
tempa=randsample(1:n,round(n*percent));
tempb=zeros(n,1);
for i=1:n
    if find(i==tempa)
        tempb(i)=1;
    end
end

Gamma1=repmat(tempb,1,n);
Gamma2=repmat(tempb,1,q);
num_class=size(Y,2);
C=eye(num_class,num_class);
oldloss = 0;

kernel_type='rbf';
Kx = kernelmatrix(kernel_type,X',X',kernel_para); 

k=10;
for ii=1:n
    for jj=1:n
        distx(ii,jj)=Kx(ii,ii)-2*Kx(ii,jj)+Kx(jj,jj);
    end
end
[distX1, idx] = sort(distx,2);

rr = zeros(n,1);
for i = 1:n  %
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
end
r = mean(rr);

S0 =  construct_S(distx, r, n);
S0 = (S0+S0')/2;
P=S0;
D_s0 = diag(sum(S0,2));
L_s = D_s0 - S0;

W=(Kx+rho*speye(n))\(Gamma2.*Y);

initA=Kx+lambda1*speye(n);
L = chol(initA,'lower');
G = (L')\((L)\eye(n));
clear L initA

IsConverge = 0; iter = 1; 
    while (IsConverge == 0&&iter<max_iter+1)

        Kxw=Kx*W;
        %update F
        F=(speye(n)+lambda4*L_s+lambda2*Gamma1.*speye(n))\(lambda2*Gamma2.*Y*C'+Kxw*C');
       
        %update M
        M=updateM(Kxw,C);
        %update W
        W=G*(F*C+0.5*lambda3*M*C);
        T=F*C-Y;        
        for j=1:n
            dii(j,1)=p/2*1/(norm(T(j,:),'fro')^(1-p/2));
        end      

        Gamma1=repmat(dii.*tempb,1,n);
        Gamma2=repmat(dii.*tempb,1,q);

        % update C
        C=updateC(F,Kxw,Gamma2.*Y,M,lambda2,lambda3);

        % Updata matrix S
            distf = L2_distance_1(F',F');
            distxf = distf-2*r*P;
            [distXF, idx2] = sort(distxf,2);

            s = zeros(n);
            for i=1:n
                idxa2 = idx2(i,2:k+1);
                dxfi = distxf(i,idxa2);
                ad = -dxfi;
                dixf = distXF(i,2:k+2);
                dixp = P(i,idx2(i,2:k+2));
                s(i,idxa2) = EProjSimplex_new(ad/(2*r));
                rr(i) = (k*dixf(k+1)-sum(dixf(1:k)))/(2*(k*dixp(k+1)+1-sum(dixp(1:k))));
            end
            r = mean(rr);

            obj=lambda4*(sum(sum(distf.*s))+r*(sum(sum(s.*s))-2*sum(sum(P.*s))));
            
            S=sparse(s);
            S=(S+S')/2;
            D_s = diag(sum(S,2));
            L_s=D_s-S;

            % check convergence
           thrsh = 1e-5;
           totalloss=norm(Kxw-F*C,'fro')^2+lambda1*trace(Kxw*W')+lambda2*norm(Gamma2.*(F*C-Y),'fro')^2-lambda3*trace(M'*Kxw*C)+obj;
           
            fprintf('-iter - %d \t',iter);
            fprintf('-loss - %d \n',totalloss);
            if abs((oldloss - totalloss)/oldloss)<=thrsh
                IsConverge = 1;
            else
                oldloss = totalloss;
            end
            cov_val(iter) = totalLoss;

        iter = iter + 1;
    end

        model.W=W;
        model.C=C;
        model.Kx=Kx;        
        model.cov_val=cov_val;
end
function C=updateC(F,XW,Y,M,lambda3,lambda4)

        Q=F'*(XW+lambda3*Y)+0.5*lambda4*M'*XW;
        [U,~,V] = svd (Q,'econ'); 
        CT=U*V';
        C=CT';
end
function M=updateM(XW,C)
        Q=XW*C';
        [U,~,V] = svd (Q,'econ'); 
        M=U*V';
           
end
