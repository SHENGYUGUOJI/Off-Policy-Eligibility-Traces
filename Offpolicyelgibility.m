%Off policy Eligibility Traces
%Goal to estimate target policy, while following behaviour policy on
%5-state random walk
close all;
num_iters=1e3; %Change number of iterations
num_states=5; %number of states, start from exactly  the middle state
a=1/(num_states+1);
state=zeros(1,num_states+2); % includes start and end terminal states
trueval=zeros(1,num_states+2);
for i=2:size(trueval,2)-1
    trueval(1,i)=a*(i-1)*1;   %these compute the true value functions,for the case of equirandom policy to find the RMS.
end
policies=[0.15,0.2,0.3,0.4,0.5]; % vector for the behavior probability
num_trials=size(policies,2);
valstate=0.5*ones(1,num_states+2);
valstate(1)=0;
valstate(1,end)=0;
alpha=0.2; % alpha initial
error=zeros(num_trials,num_iters+1);
gamma=1; %Undiscounted task
lambda=0.8;
rew=1; %reward for terminal state
targetprob=0.5;   % policy whose value is to be estimate
qvalues=size(2,num_states); 
probmatrix=[targetprob;1-targetprob]; 

for j=1:num_trials
qvalues=0.5*ones(2,num_states+2);
qvalues(:,1)=0;
qvalues(:,end)=0;
behaviorprob=policies(j);
    
for i=1:num_iters
    curstate=floor(num_states/2)+2 ; %Start at the middle!
    eltraces=zeros(2,num_states+2);
    error(j,1)=(1/sqrt(num_states))*(norm(0.5*sum(qvalues,1)-trueval));
    
    while(1)
        
        if rand(1)>=1-behaviorprob;
            action=1;
            nextstate=curstate+1;
            eltraces=eltraces*gamma*lambda*targetprob;
        else
            action=2;  %
            nextstate=curstate-1;
            eltraces=eltraces*gamma*lambda*(1-targetprob);
        end
        %disp(curstate);
        
        eltraces(action,curstate)=1;
        
        if nextstate==num_states+2
             delta= rew-qvalues(action,curstate); 
             qvalues=qvalues +alpha*delta*eltraces;
             
            break;
            
        elseif nextstate==1
            delta= -qvalues(action,curstate); %zero reward
            qvalues=qvalues +alpha*delta*eltraces;
            break;
        else
            delta= 0 + sum(probmatrix.*qvalues(:,nextstate))-qvalues(action,curstate);
           qvalues=qvalues +alpha*delta*eltraces;
        end
        curstate=nextstate;
    end
    temp=(1/sqrt(num_states))*(norm(0.5*sum(qvalues,1)-trueval));
    error(j,i+1)=temp;
    
end
end

for k=1:size(policies,2)
plot(0:num_iters,error(k,:));
hold on;
end
xlabel('Mumber of episodes');ylabel('RMS Error');title('Off Policy Eligiblity Trace');
legend('mu=0.1','mu=0.2','mu=0.3','mu=0.4','mu=0.5');
title('Off Policy Eligibility traces: Random Walk');
hold off;


