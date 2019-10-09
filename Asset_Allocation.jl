using Distributions,CPLEX,JuMP, MAT, LinearAlgebra, Dates, Convex, SCS

struct Problem_data
    n::Int64                #nr of assets
    S::Int64                #nr of scenarios
    r::Array{Float64, 2}    #returns
    b::Array{Float64,1}     #rhs
    mu::Array{Float64,1}    #means
    std::Array{Float64,1}   #stds
end

function generate_problem_normal(n::Int64 = 30,S::Int64 = 1000)
    #generate data for the asset allocation problem
    r = zeros(S,n);
    mu_min = 1; mu_max = 1.1; mu = collect(range(mu_min,stop=mu_max,length=n));
    std_min = 0; std_max = 0.1; std = collect(range(std_min,stop=std_max,length=n));
    b = zeros(S);
    for i=1:n
        r[:,i] = mu[i] .+ std[i]*randn(S);
    end
    return Problem_data(n,S,r,b,mu,std);
end

function socp_problem(n = 10,ϵ = 0.01)
    #get exact solution for the asset allocation problem
    data = generate_problem_normal(n,10)
    x = Variable(data.n);
    t = Variable(1);
    quant = quantile(Normal(),1-ϵ)
    problem = maximize(t, sum(x[i]*data.mu[i] for i=1:data.n) >= t + quant*norm(data.std.*x), sum(x[i] for i=1:data.n) <= 1, x >= 0 )
    solve!(problem, SCSSolver())
    optval = problem.optval
    return optval
end

function solve_instance(data)
    # solve the full formulation with CPLEX
    model = Model(with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0,CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, x[1:data.n]>=0)
    @variable(model, t)
    @constraint(model, sum(x[i] for i=1:data.n) == 1)
    @constraint(model, consts[s=1:data.S],t - sum(data.r[s,i] * x[i] for i=1:data.n) <= data.b[s])
    @objective(model, Min, -t);
    status = optimize!(model)
    obj = objective_value(model);
    x_opt = value.(x);
    #returns optimal solution and objective
    return x_opt, obj;
end

function solve_instance_with_pool(data)
    # Pooling
    model = Model(with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0,CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, x[1:data.n]>=0)
    @variable(model, -10^6<=t<=10^6)
    @constraint(model, sum(x[i] for i=1:data.n) == 1)
    @objective(model, Min, -t);
    status = optimize!(model)
    obj = objective_value(model);
    x_opt = value.(x);
    t_opt = value.(t);
    slacks = zeros(data.S)
    slacks = t_opt .- data.r*x_opt .- data.b;
    (maxval,maxind) = findmax(slacks)
    iter = 0;
    idxs = Int64[];
    while maxval > 0;
        iter = iter + 1;
        @constraint(model,t - sum(data.r[maxind,i] * x[i] for i=1:data.n) <= data.b[maxind])
        status = optimize!(model)
        x_opt = value.(x);
        t_opt = value.(t);
        slacks = t_opt .- data.r*x_opt .- data.b;
        push!(idxs,maxind);
        if isempty(idxs)
        else
            slacks[idxs] .= -1;
        end
        (maxval,maxind) = findmax(slacks);
    end
    obj = objective_value(model);
    #returns optimal solution, objective, number of iterations, and set I
    return x_opt, obj, iter, idxs;
end

function solve_instance_with_pool_no_ws(data)
    # Pooling without problem modification
    model = Model(with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0,CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, x[1:data.n]>=0)
    @variable(model, -10^6<=t<=10^6)
    @constraint(model, sum(x[i] for i=1:data.n) == 1)
    @objective(model, Min, -t);
    status = optimize!(model)
    obj = objective_value(model);
    x_opt = value.(x);
    t_opt = value.(t);
    slacks = zeros(data.S)
    slacks = t_opt .- data.r*x_opt .- data.b;
    (maxval,maxind) = findmax(slacks)
    iter = 0;
    idxs = Int64[];
    while maxval > 0;
        iter = iter + 1;
        push!(idxs,maxind);
        model = Model(with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0,CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
        @variable(model, x[1:data.n]>=0)
        @variable(model, -10^6<=t<=10^6)
        @constraint(model, sum(x[i] for i=1:data.n) == 1)
        @objective(model, Min, -t);
        @constraint(model,consts[id=1:length(idxs)],t - sum(data.r[idxs[id],i] * x[i] for i=1:data.n) <= data.b[idxs[id]])
        status = optimize!(model)
        x_opt = value.(x);
        t_opt = value.(t);
        slacks = t_opt .- data.r*x_opt .- data.b;
        if isempty(idxs)
        else
            slacks[idxs] .= -1;
        end
        (maxval,maxind) = findmax(slacks);
    end
    obj = objective_value(model);
    #returns optimal solution, objective, number of iterations, and set I
    return x_opt, obj, iter, idxs;
end

function solve_instance_with_pool_idxs(data::Problem_data,idxs_good::Union{Int64,Array{Int64,1}},idxs_bad::Union{Int64,Array{Int64,1}})
    # Pooling in P&D -
    # idxs_good are the ones in I^*
    # idxs_bad are the ones in i_r  ∪ I_p

    b=copy(data.b);
    if length(idxs_bad) == 1
        b[[idxs_bad]] .= 10^10;
    else
        b[idxs_bad] .= 10^10;
    end
    model = Model(with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0,CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, x[1:data.n] >= 0)
    @variable(model, -10^6<=t<=10^6)
    @constraint(model, sum(x[i] for i=1:data.n) == 1)
    @objective(model, Min, -t);
    l = length(idxs_good);
    if l > 0
        for s=1:l
            @constraint(model,t - sum(data.r[idxs_good[s],i] * x[i] for i=1:data.n) <= b[idxs_good[s]]);
        end
    end
    status = optimize!(model)
    obj = objective_value(model);
    x_opt = value.(x);
    t_opt = value.(t);
    slacks = zeros(data.S);
    slacks = t_opt .- data.r*x_opt .- b;
    slacks[idxs_good] .= -1;
    (maxval,maxind) = findmax(slacks)
    iter = 0;
    idxs = Int64[];
    while maxval > 0;
        iter = iter + 1;
        @constraint(model,t - sum(data.r[maxind,i] * x[i] for i=1:data.n) <= b[maxind])
        status = optimize!(model)
        x_opt = value.(x);
        t_opt = value.(t);
        slacks = t_opt .- data.r*x_opt .- b;
        push!(idxs,maxind);
        if isempty(idxs)
            slacks[idxs_good] .= -1;
        else
            slacks[idxs] .= -1;
            slacks[idxs_good] .= -1;
        end
        (maxval,maxind) = findmax(slacks);
        obj = objective_value(model);
    end
    idxs_good_new = [idxs_good;idxs];
    #returns optimal solution, objective and set I
    return x_opt, obj, idxs_good_new;
end

function solve_problem(data::Problem_data, ϵ::Float64 = 0.05)
    # P&D algorithm
    γ = 0.000001;
    b=copy(data.b);
    nr_to_neglect = Int(ceil( data.S * ϵ));
    (x_opt,obj,iter,idxs) = solve_instance_with_pool(data);
    idxs_bad = Int64[];
    idxs_good = idxs;
    slacks = zeros(data.S);
    cur_obj = Float64[];
    cur_feas = Float64[];
    cur_quant = Float64[];
    solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
    cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
    push!(cur_obj,obj);
    push!(cur_feas,cdf_val);
    push!(cur_quant,solution_quant_de);
    for i = 1:nr_to_neglect
        slacks = -obj .- data.r*x_opt .- b;
        idxs_work = Int64[];
        idxs_work= findall(slacks .>= -γ); #println(length(idxs_work))
        x_values = Dict{Int64, Array{Float64,1}}();
        objectives = zeros(length(idxs_work));
        idxs_used = Dict{Int64, Array{Int64,1}}();
        for j=1:length(idxs_work)
            if isempty(idxs_bad)
            idxs_bad_work = idxs_work[j];
            else
            idxs_bad_work = [idxs_bad;idxs_work[j]];
            end
            (x_values[j],objectives[j],idxs_used[j]) = solve_instance_with_pool_idxs(data,idxs_good,idxs_bad_work);
        end
        mv, best_ind = findmin(objectives);
        obj = objectives[best_ind];
        x_opt = x_values[best_ind];
        push!(idxs_bad,idxs_work[best_ind]);
        idxs_iter = Int64[];
        for j=1:length(idxs_used[best_ind])
            if idxs_used[best_ind][j] != idxs_work[best_ind]
                push!(idxs_iter,idxs_used[best_ind][j]);
            end
        end
        idxs_good = idxs_iter;
        b[idxs_bad] .= 10^10;
        solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
        cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
        push!(cur_obj,obj);
        push!(cur_feas,cdf_val);
        push!(cur_quant,solution_quant_de);
    end
    # returns optimal solutions, optimal objectives, removed scenarios, real VaR, probability of violation, and quantile of the current solution
    return x_opt,obj, idxs_good, idxs_bad, cur_obj, cur_feas, cur_quant;
end

function solve_problem_randomized(data::Problem_data, ϵ::Float64 = 0.05)
    # P&D algorithm with randomized scenario removal
    b=copy(data.b);
    γ = 0.000001;
    nr_to_neglect = Int(ceil( data.S * ϵ));
    (x_opt,obj,iter,idxs) = solve_instance_with_pool(data);
    idxs_bad = Int64[];
    idxs_good = idxs;
    slacks = zeros(data.S);
    cur_obj = Float64[];
    cur_feas = Float64[];
    cur_quant = Float64[];
    solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
    cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
    push!(cur_obj,obj);
    push!(cur_feas,cdf_val);
    push!(cur_quant,solution_quant_de);
    for i = 1:nr_to_neglect
        slacks = -obj .- data.r*x_opt .- b;
        idxs_work = Int64[];
        idxs_work= findall(slacks .>= -γ); #println(length(idxs_work))
        x_values = Dict{Int64, Array{Float64,1}}();
        idxs_used = Dict{Int64, Array{Int64,1}}();
        best_ind = rand(1:length(idxs_work))

        if isempty(idxs_bad)
            idxs_bad_work = idxs_work[best_ind];
        else
            idxs_bad_work = [idxs_bad;idxs_work[best_ind]];
        end
        (x_opt,obj,idxs_used) = solve_instance_with_pool_idxs(data,idxs_good,idxs_bad_work);
        push!(idxs_bad,idxs_work[best_ind]);
        idxs_iter = Int64[];
        for j=1:length(idxs_used)
            if idxs_used[j] != idxs_work[best_ind]
                push!(idxs_iter,idxs_used[j]);
            end
        end
        idxs_good = idxs_iter;
        b[idxs_bad] .= 10^10;
        solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
        cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
        push!(cur_obj,obj);
        push!(cur_feas,cdf_val);
        push!(cur_quant,solution_quant_de);
    end
    # returns optimal solutions, optimal objectives, removed scenarios, real VaR, probability of violation, and quantile of the current solution
    return x_opt,obj, idxs_good, idxs_bad, cur_obj, cur_feas, cur_quant;
end

function greedy_discard_normal(data::Problem_data, ϵ::Float64 = 0.05)
    # noPnoD
    γ = 0.000001;
    b=copy(data.b);
    nr_to_neglect = Int(ceil( data.S * ϵ));
    (x_opt,obj) = solve_instance(data);
    idxs_bad = Int64[];
    slacks = zeros(data.S);
    cur_obj = Float64[];
    cur_feas = Float64[];
    cur_quant = Float64[];
    solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
    cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
    push!(cur_obj,obj);
    push!(cur_feas,cdf_val);
    push!(cur_quant,solution_quant_de);
    for i = 1:nr_to_neglect
        slacks = -obj .- data.r*x_opt .- b;
        idxs_work = Int64[];
        idxs_work= findall(slacks .>= -γ);
        x_values = Dict{Int64, Array{Float64,1}}();
        objectives = zeros(length(idxs_work));
        idxs_used = Dict{Int64, Array{Int64,1}}();
        rows_used = Dict{Int64, Array{Int64,1}}();
        for j=1:length(idxs_work)
            if isempty(idxs_bad)
            idxs_bad_work = idxs_work[j];
            else
            idxs_bad_work = [idxs_bad;idxs_work[j]];
            end
            (x_values[j],objectives[j]) = solve_instance_idxs(data,idxs_bad_work);
        end
        mv, best_ind = findmin(objectives);
        obj = objectives[best_ind];
        x_opt = x_values[best_ind];
        push!(idxs_bad,idxs_work[best_ind]);
        idxs_iter = Int64[];
        b[idxs_bad] .= 10^10;
        solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
        cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
        push!(cur_obj,obj);
        push!(cur_feas,cdf_val);
        push!(cur_quant,solution_quant_de);
    end
    # returns optimal solutions, optimal objectives, removed scenarios, real VaR, probability of violation, and quantile of the current solution
    return x_opt,obj, idxs_bad, cur_obj, cur_feas, cur_quant;
end

function solve_instance_idxs(data,idxs_bad)
    # PnoD routine
    b=copy(data.b);
    if length(idxs_bad) == 1
        b[[idxs_bad]] .= 10^10;
    else
        b[idxs_bad] .= 10^10;
    end
    model = Model(with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0,CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, x[1:data.n]>=0)
    @variable(model, t)
    @constraint(model, sum(x[i] for i=1:data.n) == 1)
    @constraint(model, consts[s=1:data.S],t - sum(data.r[s,i] * x[i] for i=1:data.n) <= b[s])
    @objective(model, Min, -t);
    status = optimize!(model)
    obj = objective_value(model);
    x_opt = value.(x);
    #returns optimal solution and objective
    return x_opt, obj;
end

function greedy_discard_pool(data::Problem_data, ϵ::Float64 = 0.05)
    # PnoD
    b=copy(data.b);
    γ = 0.000001;
    nr_to_neglect = Int(ceil( data.S * ϵ));
    (x_opt,obj,val) = solve_instance_with_pool_idxs(data,Int64[],Int64[]);
    idxs_bad = Int64[];
    slacks = zeros(data.S);
    cur_obj = Float64[];
    cur_feas = Float64[];
    cur_quant = Float64[];
    solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
    cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
    push!(cur_obj,obj);
    push!(cur_feas,cdf_val);
    push!(cur_quant,solution_quant_de);
    for i = 1:nr_to_neglect
        slacks = -obj .- data.r*x_opt .- b;
        idxs_work = Int64[];
        idxs_work= findall(slacks .>= -γ);
        x_values = Dict{Int64, Array{Float64,1}}();
        objectives = zeros(length(idxs_work));
        idxs_used = Dict{Int64, Array{Int64,1}}();
        rows_used = Dict{Int64, Array{Int64,1}}();
        for j=1:length(idxs_work)
            if isempty(idxs_bad)
            idxs_bad_work = idxs_work[j];
            else
            idxs_bad_work = [idxs_bad;idxs_work[j]];
            end
            (x_values[j],objectives[j],val) = solve_instance_with_pool_idxs(data,Int64[],idxs_bad_work);
        end
        mv, best_ind = findmin(objectives);
        obj = objectives[best_ind];
        x_opt = x_values[best_ind];
        push!(idxs_bad,idxs_work[best_ind]);
        idxs_iter = Int64[];
        b[idxs_bad] .= 10^10;
        solution_quant_de = sum(x_opt[i]*data.mu[i] for i=1:data.n) - quantile(Normal(),1-ϵ) * norm(data.std.*x_opt)
        cdf_val = cdf(Normal(), (-obj - sum(x_opt[i]*data.mu[i] for i=1:data.n))/norm(data.std.*x_opt))
        push!(cur_obj,obj);
        push!(cur_feas,cdf_val);
        push!(cur_quant,solution_quant_de);
    end
    # returns optimal solutions, optimal objectives, removed scenarios, real VaR, probability of violation, and quantile of the current solution
    return x_opt,obj, idxs_bad, cur_obj, cur_feas, cur_quant;
end
