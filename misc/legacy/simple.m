
R = 0.8;
U = [0; 1];
p_o = [2.75; 4];
t = 0.02;
gamma = 0.2;
limp = 1.3;
d = limp/gamma;

f = figure;
f.Position = [200 130 1000 870];
hold on;
trusts = [0.08, 0.52, 0.94];
legends = ["Obtacle 1" "Safety Margin"];
plot(p_o(1), p_o(2), "o", "LineWidth", 2);
th = 0:pi/50:2*pi;
xunit = R * cos(th) + p_o(1);
yunit = R * sin(th) + p_o(2);
h = plot(xunit, yunit, "Color", "red", "LineWidth", 2);
xlim([0, 7]);
ylim([0, 7]);
for k = 1: 3
    data = [];
    p_c = [3; 0];
    P = [p_c];
    trust = trusts(k);
    trust
    alpha = 1/(1/gamma + d*(exp(-1*trust) - 1));
    alpha

    for i = 1: 750
        diff = p_c - p_o;
        H = eye(2) * 2;
        f = [0; 0];
        
        A = 2 * diff;
        A = -A';
        b = alpha * (diff' * diff - R^2) + 2 * diff' * U;
        lb = [-2; -2] - U;
        ub = [2; 2] - U;
        
        
        x = quadprog(H, f, A, b, [], [], lb, ub);
        p_c = p_c + (x + U) * t;
        P = [P p_c];
    end
%     legends = [legends "Trust: " + string(trusts(k))];
%     plot(P(1, :), P(2, :), "LineWidth", 2); 
end
legend(legends, "FontSize", 20);
title("Effect of trust in scenarios with a single stationary object");