R = 0.55;
goal = [3.20;
        6.50];
p_c = [3; 
       0];
p_o = [2.75 4.20 3.50;
       4.00 4.20 5.80;];
t = 0.01;
alpha = 1;
trust = [0.1; 0.9; 0.5];

data = [];
P = [p_c];

for i = 1: 5000
    
    H = eye(2) * 2;
    f = [0; 0];

    min_U = 0.2;
    max_U = 0.5;
    prop_gain = 0.5;
    U = (goal - p_c) * prop_gain;
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
    p_c = p_c + (x + U) * t;
    P = [P p_c];
end