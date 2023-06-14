R = 0.55;
U = [0;
     1];
p_c = [3; 
       0];
p_o = [2.75 4.20 4.00;
       4.00 4.20 6.00;];
t = 0.01;
alpha = 5;
trust = [0.1; 0.8; 0.4];

data = [];
P = [p_c];

for i = 1: 2000
    
    H = eye(2) * 2;
    f = [0; 0];
   
    diff = p_c - p_o;
    A = - 2 * diff';
    b = alpha * trust .* (sum(diff.^2)' - R^2) + 2 * diff' * U;
    lb = [-2; -2] - U;
    ub = [2; 2] - U;
    
    
    x = quadprog(H, f, A, b, [], [], lb, ub);
    p_c = p_c + (x + U) * t;
    P = [P p_c];
end