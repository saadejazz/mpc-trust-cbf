figure();
hold on
plot(P(1, :), P(2, :));

th = 0:pi/50:2*pi;
legends = ["ego vehicle"];
for i = 1: size(p_o, 2)
    plot(p_o(1, i), p_o(2, i), "o");
    xunit = R * cos(th) + p_o(1, i);
    yunit = R * sin(th) + p_o(2, i);
    h = plot(xunit, yunit, "Color", "red");
    legends = [legends "trust:" + string(trust(i)) ""];
end
legend(legends);

xlim([0, 7]);
ylim([0, 7]);
