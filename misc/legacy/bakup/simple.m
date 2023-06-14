R = 0.8;
U = [0; 1];
p_c = [3; 0];
p_o = [2.75; 4];
t = 0.02;
alpha = 0.5;

data = [];
P = [p_c];

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