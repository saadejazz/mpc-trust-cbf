f = figure;
f.Position = [200 130 1000 870];
hold on
plot(P(1, :), P(2, :), "LineWidth", 2);

th = 0:pi/50:2*pi;
legends = ["ego vehicle"];
for i = 1: size(p_o, 2)
    plot(p_o(1, i), p_o(2, i), "o", "LineWidth", 2);
    xunit = R * cos(th) + p_o(1, i);
    yunit = R * sin(th) + p_o(2, i);
    h = plot(xunit, yunit, "Color", "red", "LineWidth", 2);
    legends = [legends "trust:" + string(trust(i)) ""];
end
plot(goal(1), goal(2), "x", "LineWidth", 2, "MarkerSize", 15);
legends = [legends "Goal"];
legend(legends, "FontSize", 20);

xlim([0, 7]);   
ylim([0, 7]);
title("Multiple stationary objects with different trust")