% Load cPic, received and training
load spydata.mat
load training.mat

% Encoded image
% image(cPic)

%% Estimate optimal linear MMSE filter

% Trade-off: Larger L allows for more filter taps and more accurate equalization, but reduces training sequence and thus accuracy.
L = 9;
L_fir = L+1; % Filter of L taps has actually L+1 coefficients
N = 32;

%% Cross-correlation X-Y
% Equivalent to using 'xcorr'

r_xy = zeros(1,L_fir);

% Unbiased estimator - seems to outperformed biased
for j = 1:L_fir
    for i = j:N
        r_xy(j) = r_xy(j) + training(i) * received(i-j+1);
    end
    r_xy(j) = r_xy(j)./(N-j+1);
end
r_xy;

% Biased estimator
% r_xy_b = zeros(1,L_fir);
% for j = 1:L_fir
%     for i = j:32
%         r_xy_b(j) = r_xy_b(j) + training(i) * received(i-j+1);
%     end
%     r_xy_b(j) = r_xy_b(j)./32;
% end
% r_xy_b;

%% Autocorrelation matrix of Y

% QUESTION: Why is the equalization better when we estimate acf with 32 samples
% and worse with all the samples (about 11000)? 
% ANSWER: Because we have to use the same Y samples to
% estimate Ry than to estimate r_xy, since they are then used together to
% calculate h_opt. 

r_y = zeros(1,L_fir);

% Unbiased ACF estimator
for j = 1:L_fir
    for i = j:N
        r_y(j) = r_y(j) + received(i)*received(i-j+1);
    end
    r_y(j) = r_y(j)./(N-j+1);
end
r_y;

% Biased ACF estimator
% r_y_b = zeros(1,L_fir);
% 
% for j = 1:L_fir
%     for i = j:N
%         r_y_b(j) = r_y_b(j) + received(i)*received(i-j+1);
%     end
%     r_y_b(j) = r_y_b(j)./32;
% end
% r_y_b


% Matrix
R_y = toeplitz(r_y_b);
R_y_inv = inv(R_y);


%% Optimal linear filter

h_opt = R_y \ r_xy';    % equivalent to inv(R_y) * r_xy'

%% Equalizer and Detector 

% Equilizer
eq_received = conv(received, h_opt);
eq_received = eq_received(1:length(eq_received)-length(h_opt)+1);

% Detector
key = sign(eq_received);

% MSE in training sequence
MSE_training = 1/(32-L_fir) * sum((eq_received(L_fir:32) - training(L_fir:32)).^2);
Training_errors = sum((key(L_fir:32) - training(L_fir:32)).^2);

% MSE using perfectly recovered key (BUT recovery is not perfect!!)
MSE_full_key = 1/length(key) * sum((eq_received - key).^2);
% MSE_optimal = 1 - r_xy * R_y_inv * r_xy'
MSE_optimal = 1 - (r_xy / R_y) * r_xy';

%% Random Bit Error Introduction (Assignment 3)
% total_bit_error = 1:100:1500;
% for k = 1:length(total_bit_error)
%     randomness = randperm(length(key),total_bit_error(k));
%     key(randomness) = key(randomness)*-1;
%     figure(1);
%     subplot(3,5,k);
%     dPic = decoder(key, cPic);
%     image(dPic)
%     title(["Bit Error =", num2str(total_bit_error(k))]);
% end
% sgtitle("Introduction of Random Bit Errors");
%% Normal Bit Error Introduction (Assignment 3)
% total_bit_error = 1:100:1500;
% for k = 1:length(total_bit_error)
%     key(1:total_bit_error(k)) = key(1:total_bit_error(k))*-1;
%     figure(1);
%     subplot(3,5,k);
%     dPic = decoder(key, cPic);
%     image(dPic)
%     title(["Bit Error =", num2str(total_bit_error(k))]);
% end
% sgtitle("Introduction of Normal Bit Errors");
%% Decode image
% 
dPic = decoder(key, cPic);
image(dPic)




