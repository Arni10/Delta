function w = OfflineLearning(x, d, f, gradf, lr, stop)
[~, n] = size(x);
w = randn(n,size(d,2));
epoch = 0;
while true
v = x * w;
y = f(v);
e = y - d; % using square loss function
g = x' * (e .* gradf(v)); %
w = w - lr * g;
E = sum(e(:).^2);
if stop(E, epoch), break; end
epoch = epoch + 1;
end
