% test EVD on DS and PS
clear
clc
close all

% control random number generator
% rng('default')

% number of images, and number of samples
N = 13;
L = 1000;

% base = (rand((N),1)-0.5)*400; % it ranges from -200 to 200
base = [-abs((rand((N-1)/2,1)-0.5)*400);1e-5;abs((rand((N-1)/2,1)-0.5)*400)];
% base = linspace(-200,200,N);
[base,sortInd] = sort(base);
base = base(:);

theta = 31.8/2/pi;  % incidence angle, actually not used
range = 704000;     % satellite distance 
lam = 0.031;        % wavelength
f = 4*pi/lam;       % constant
c = f*base/range;   % elevation phase model


rho_s = lam*range/2/(max(base)-min(base))

%% source
nS = 2; % number of sources

sStep = rho_s*4;

s = (1+(0:(nS-1)))*sStep;

% s = [-20 40 ];

stds = [2*sqrt(2) 2*sqrt(2)*2 0 0]; % std of DS, disable them by setting to 0
amps = [0 0 0 0]; % amplitude of PS, disable them by setting to 0


%% simulate L samples with different hight configuration
% set a constant coherence matrix for both the sources
ch = ones(N,N);

G = zeros(N,L);
R = zeros(N,L,nS);
S = zeros(nS,L);


for i = 1:nS % simulate the nS sources

    % simulate the true phases for each soruce 
    % g0 is a column in thr R matrix
    S(i,:) = s(i)*rand(1,L);
    
    R0 = exp(-1j*(c*S(i,:)));
    
    % the steering vector of all L samples
    R(:,:,i) = R0;
    
    % add very small noise to prevent numerical problem in EVD
    R_noise = 1e-10*exp(-1j*(rand(N,L)*2*pi)); % random noise with uniform phase 
    R0_noise = R0 + R_noise;
    
    % simulate L DS samples
    G0 = sim_DS(ch,stds(i),N,L);
    
    G0 = G0.*R0_noise;
       
    % sum all contribution
    G = G+G0;
end

%% simulate L sample covariance matrices with different hight configuration
% reuse the R matrix as the height configuration

G1 = zeros(N,L);
CV = zeros(N,N,L);
CV_theory = zeros(N,N,L);

for l = 1:L
    
    for i = 1:nS % simulate the nS sources
        
        % extract the r vector
        r0 = R(:,l,i);
        
        % add small noise 
        r_noise = 1e-10*exp(-1j*(rand(N,1)*2*pi));
        r0_noise = r0 + r_noise;
        
        % simulate L DS samples
        G0 = sim_DS(ch,stds(i),N,L);
        G0 = G0.*(r0_noise*ones(1,L));
        
        % sum all contribution
        G1 = G1+G0;
        
        % get true covariance matrix
        CV_theory(:,:,l) = CV_theory(:,:,l) + stds(i)^2*ch.*(r0*r0');
    end
    
    % estimate covariance matrix
    CV(:,:,l) = G1*G1'/L;
end

%% save the variables
save('kpca_nn_simulation.mat','N','L','base','theta','range','lam','f',...
    'c','rho_s', 'nS','sStep','s', 'stds','ch','G','R','S','G1','CV','CV_theory')


%% make network
% layers = [
% sequenceInputLayer(N)
% fullyConnectedLayer(10)
% leakyReluLayer
% fullyConnectedLayer(10)
% leakyReluLayer
% fullyConnectedLayer(N)
% ];
% 
% options = trainingOptions('sgdm', ...
%     'MaxEpochs',20,...
%     'InitialLearnRate',1e-4, ...
%     'Verbose',false, ...
%     'Plots','training-progress');


in = [real(G);imag(G)];
in = in - mean(in);
out =[real(R(:,:,1));imag(R(:,:,1))];
out = out - mean(out);

net = feedforwardnet([ 30 30 ]);
net = train(net,in,out);
y = net(in);
y = y(1:N,:)+1j.*y(N+1:end,:);

view(net)

angBias = acosd ( abs(sum(y.*conj(R(:,:,1))))./ sum(abs(y).*abs(R(:,:,1))) ) ;
