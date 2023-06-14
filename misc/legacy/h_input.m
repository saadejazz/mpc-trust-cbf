R = 0.55;
p_c = [3; 
       0];
% p_o = [2.75 4.20 3.50;
%        4.00 4.20 5.80;];
p_o = [2.20 4.20;
       4.00 4.00;];
% u_o = [0.06 0.07 -0.03;
%        0.00 0.00 0.00];
u_o = [0.06 -0.05;
       -0.02 0.01;];
t = 0.02;
alpha = 0.6;
% trust = [0.1; 0.9; 0.5];
trust = [0.25; 0.71];

data = [];
P = [p_c];
P_e = [p_c];
p_e = p_c;

f = figure;
f.Position = [200 130 1000 870];
hold on

first = 0;

legends = ["ego vehicle" "human reference"];

% user desired speeds
U_1 = [0.05;
       0.18];
t_1 = 500;
U_2 = [-0.03;
       0.13];
t_2 = 1500;
U_3 = [0.03;
        0.08];
t_3 = 2500;

for i = 1: 2500
    
    if i <= t_1
        U = U_1;
    elseif i <= t_2
        U = U_2;
    elseif i <= t_3
        U = U_3;
    end

    H = eye(2) * 2;
    f = [0; 0];

   p_e = p_e + U * t;  
   
    min_U = 0.2;
    max_U = 0.75;
    prop_gain = 0.75;
    U = (p_e - p_c) * prop_gain;
        for j = 1: 2
            if abs(U(j)) < min_U && U(j) ~= 0 
                U(j) = (U(j)/abs(U(j))) * min_U;
            end
            if abs(U(j)) > max_U && U(j) ~= 0 
                U(j) = (U(j)/abs(U(j))) * max_U;
            end
        end   
    
    
    
    diff = p_c - p_o;
    A = - 2 * diff';
    b = alpha * trust .* (sum(diff.^2)' - R^2) + 2 * diff' * U;
    lb = [-2; -2] - U;
    ub = [2; 2] - U;
    
    
    x = quadprog(H, f, A, b, [], [], lb, ub);
    
    colors = ["black", "blue", "yellow", "cyan"];
    if mod(i, 50) == 3
        if first ~= 0
            delete(h); delete(d); delete(ex); delete(ex2);
            h = zeros(1, size(p_o, 2));
            d = zeros(1, size(p_o, 2));
        end        
 
        ex = plot(P(1, :), P(2, :), "Color", "magenta", "LineWidth", 2.5);
        ex2 = plot(P_e(1, :), P_e(2, :), "Color", "cyan", "LineWidth", 2.5);
            th = 0:pi/50:2*pi;
%         legends = ["ego vehicle"];
        for k = 1: size(p_o, 2)
            
            xunit = R * cos(th) + p_o(1, k);
            yunit = R * sin(th) + p_o(2, k);
            d(k) = plot(p_o(1, k), p_o(2, k), "o", "Color", colors(k), "LineWidth", 3, "MarkerSize", 9);

            h(k) = plot(xunit, yunit, "Color", "red", "LineWidth", 2.5);
            if first == 0
            legends = [legends "trust:" + string(trust(k)) ""];
            else
            legends = [legends "" ""];
            end
        end    
        
        

        first = 1;
        legend(legends, "FontSize", 20);
        xlim([0, 7]);
        ylim([0, 7]);
        xlabel("X (m)");
        ylabel("Y (m)");
        title("Moving obstacles with different trust and human reference");
        pause(0.05);
    end


    p_c = p_c + (x + U) * t;
    
    P = [P p_c];
    P_e = [P_e p_e];
    p_o = p_o + u_o * t;
end