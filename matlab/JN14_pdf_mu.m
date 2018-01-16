function p = JN14_pdf_mu(x, mu, k, xi, nr, besseliln0_func)
% Edited version of Paul Bays' model.

if (size(x,2)==1), x = x'; end

if nargin<5 | isempty(nr), nr = 1001; end

if nargin<6,
    besseliln0_func = @fastbesseliln0;
end

persistent kluyver_density;
if ~isstruct(kluyver_density)
    kluyver_density = load('kluyver_density.mat');
end

n_min = poissinv(10^-4/2,xi);                    % range of Poisson counts to consider
n_max = poissinv(1-10^-4/2,xi);                  % ...

nn = n_min:n_max;
pn = poisspdf(nn,xi); pn = pn/sum(pn);

rr = linspace(0,n_max,nr);

pr = zeros(length(nn),nr);
for in = 1:length(nn)
    if nn(in) == 0                               % trivial cases n = 0, 1
       pr(in,1) = 1;
    elseif nn(in) == 1
       pr(in,round((nr-1)/n_max)+1) = 1;
       
    elseif nn(in)<=size(kluyver_density.mle,1)   % use Monte Carlo estimate for small n    
       prk = interp1(kluyver_density.rr*nn(in),kluyver_density.mle(nn(in),:),rr);
       pra = exp(besseliln0_func(k*rr) - nn(in) * besseliln0_func(k)) .* prk;
       pra(isnan(pra)) = 0;
       pr(in,:) = pra / sum(pra);
       
    else                                        % use Gaussian approximation for large n 
       m = nn(in)*besseli(1,k)/exp(besseliln0_func(k));
       s2 = nn(in)*(0.5 + 0.5*exp(besseliln(2,k)-besseliln0_func(k)) - exp(2*besseliln(1,k)-2*besseliln0_func(k)));
       pra = normpdf(rr,m,sqrt(s2));       
       pr(in,:) = pra / sum(pra); 
   end
end
    
vp = exp(bsxfun(@minus, k*bsxfun(@times,rr',cos(x-mu)), besseliln0_func(k*rr)') - log(2*pi));

p = pn*pr*vp;