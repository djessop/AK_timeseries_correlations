function y = gaussian(x, loc, scale)
y = exp(-((x - loc)/scale).^2);
end