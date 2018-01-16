function b = dog(x, a, w)

c = sqrt(2) / exp(-0.5);
b = x .* a .* w .* c .* exp(-(w .* x) .^ 2);