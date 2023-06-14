figure();
hold on
plot(P(1, :), P(2, :));
plot(p_o(1), p_o(2), "o");
th = 0:pi/50:2*pi;
xunit = R * cos(th) + p_o(1);
yunit = R * sin(th) + p_o(2);
h = plot(xunit, yunit, "Color", "red");
xlim([0, 7]);
ylim([0, 7]);
