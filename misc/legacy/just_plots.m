
d = [0.3, 0.7, 1.1, 1.5, 1.9, 2.3, 2.7];
% d = [2.2];
gamma = 0.5;
f = figure();
hold on
legends = [];
for k = 1: length(d)
    x = linspace(0, 1, 1000);
    
    y = zeros(1, 1000);
    
%     gamma = 1.2/d(k);
    for i = 1: 1000
        y(i) = 1/((1/gamma) + d(k)*(exp(-x(i))-1));
    end
    plot(x, y)
    legends = [legends "$\psi$: " + string(d(k)) + ", $\gamma$: " + string(gamma)];
end
ylabel("$\omega$", "Interpreter","latex");
xlabel("$\tau$", "Interpreter","latex");
legend(legends, 'Interpreter','latex');
% title("Weight function $\omega(\tau)$ shape $\gamma=0.5,\;\psi=2.2$", "Interpreter","latex");
title("Effect of hyperparameter on function shape $\gamma=$" + string(gamma), "Interpreter","latex");

