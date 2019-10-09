using Distributions, JuMP, CPLEX

struct Problem_data
    n::Int64    #problem dimension
    m::Int64    #nr of rows
    b::Array{Float64, 2}  #rhs
    S::Int64    #nr of scenarios
    ξ::Array{Float64, 3} #rand scenarios
    ϵ::Float64  #prob. of violation
end

function generate_data(n = 3,m = 2,b = 9,S = 100,ϵ = 0.1)
    #generate data for the nonlinear example
    ξ = randn(m,n,S); #standard normal
    bs = b*ones(m,S);
    return Problem_data(n,m,bs,S,ξ,ϵ);
end

function get_optimum(data::Problem_data)
    #get optimal value of the nonlinear example
    optval = -data.n*sqrt(data.b[1,1]/quantile(Chisq(data.n), (1-data.ϵ)^(1/data.m)))
end

function solve_instance(data::Problem_data)
    # solve the full problem
    model = Model(with_optimizer(CPLEX.Optimizer; CPX_PARAM_SCRIND = 0, CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, x[1:data.n] >= 0)
    @constraint(model, [i=1:data.m,s=1:data.S], sum(data.ξ[i,j,s]^2 * x[j]^2 for j=1:data.n) <= data.b[i,s])
    @objective(model, Min, - sum(x[j] for j=1:data.n));

    status = optimize!(model)
    obj = objective_value(model);
    x_opt = value.(x);
    return x_opt, obj;
end

function solve_instance_with_pool(data::Problem_data)
    # Pooling
    model = Model(with_optimizer(CPLEX.Optimizer; CPX_PARAM_SCRIND = 0, CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, 0<= x[1:data.n] <= 100)
    @objective(model, Min, - sum(x[j] for j=1:data.n));
    status = optimize!(model)
    obj = objective_value(model);
    x_val = value.(x);
    slacks = zeros(data.m,data.S);
    for s=1:data.S
        slacks[:,s] = (data.ξ[:,:,s].^2) * (x_val.^2) - data.b[:,s];
    end
    iter = 0;
    @constraint(model, myCons[1:1], 0 <= 1)
    empty!(myCons)
    idxs = Int64[];
    rows = Int64[];
    idx = 1;
    while maximum(slacks) > 0
        iter = iter + 1;
        cur = 0;
        for i=1:data.m
            (max_val,max_ind) = findmax(slacks[i,:])
            if max_val > cur
                idx = max_ind; cur= max_val;
            end
        end
        push!(idxs,idx);
        for row=1:data.m
            push!(myCons, @constraint(model, sum(data.ξ[row,j,idx]^2 * x[j]^2 for j=1:data.n) <= data.b[row,idx]));
        end
        status = optimize!(model)
        x_val = value.(x);
        for s=1:data.S
            slacks[:,s] = (data.ξ[:,:,s].^2) * (x_val.^2) - data.b[:,s];
        end
        for i=1:length(iter)
            slacks[:,idxs[i]] .= -1;
        end
    end
    obj = objective_value(model);
     return x_val, obj, idxs; #duals;
end


function solve_instance_with_pool_idxs(data::Problem_data,idxs_good::Union{Int64,Array{Int64,1}},idxs_bad::Union{Int64,Array{Int64,1}})
    # Pooling in P&D
    b=copy(data.b);
    b[:,idxs_bad] .= 10^6;
    model = Model(with_optimizer(CPLEX.Optimizer; CPX_PARAM_SCRIND = 0, CPX_PARAM_BAREPCOMP = 1e-8, CPX_PARAM_EPOPT = 1e-8,CPX_PARAM_EPRHS = 1e-8))
    @variable(model, 0<= x[1:data.n] <= 100)
    @objective(model, Min, - sum(x[j] for j=1:data.n));
    l = length(idxs_good);
    if l > 0
        for s=1:l
                @constraint(model, [row=1:data.m] ,sum(data.ξ[row,j,idxs_good[s]]^2 * x[j]^2 for j=1:data.n) <= b[row,idxs_good[s]])
        end
    end
    status = optimize!(model)
    obj = objective_value(model);
    x_val = value.(x);
    slacks = zeros(data.m,data.S);
    for s=1:data.S
        slacks[:,s] = (data.ξ[:,:,s].^2) * (x_val.^2) - b[:,s];
    end
    if l>0
        for s=1:l
            slacks[:,idxs_good[s]] .= -1;
        end
    end
    iter = 0;
    @constraint(model, myCons[1:1], 0 <= 1)
    empty!(myCons)
    idxs = Int64[];
    rows_new = Int64[];
    idx = 1;
    while maximum(slacks) > 0
        iter = iter + 1;
        cur = 0;
        for i=1:data.m
            (max_val,max_ind) = findmax(slacks[i,:])
            if max_val > cur
                idx = max_ind; cur= max_val;
            end
        end
        push!(idxs,idx);
        for row=1:data.m
            push!(myCons, @constraint(model,
                                sum(data.ξ[row,j,idx]^2 * x[j]^2 for j=1:data.n) <= b[row,idx]));
        end
        status = optimize!(model)
        x_val = value.(x);
        obj = objective_value(model);
        for s=1:data.S
            slacks[:,s] = (data.ξ[:,:,s].^2) * (x_val.^2) - b[:,s];
        end
        for i=1:l
            slacks[:,idxs_good[i]] .= -1;
        end
        for i=1:length(iter)
            slacks[:,idxs[i]] .= -1;
        end
    end
    idxs_good_new = [idxs_good;idxs];
    return x_val, obj, idxs_good_new;
end

function solve_problem(data::Problem_data)
    # P&D
    γ = 1e-3;
    b=copy(data.b);
    nr_to_neglect = Int(ceil( data.S * data.ϵ));
    (x_opt,obj,idxs) = solve_instance_with_pool(data);
    idxs_bad = Int64[];
    idxs_good = idxs;
    slacks = zeros(data.m,data.S);
    for i = 1:nr_to_neglect
        for s=1:data.S
            slacks[:,s] = (data.ξ[:,:,s].^2) * (x_opt.^2) - b[:,s];
        end
        idxs_work = Int64[];
        for q = 1:data.m
            idxs_work_new = findall(slacks[q,:] .>= -γ);
            idxs_work = union(idxs_work,idxs_work_new);
        end
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
        we,best_ind = findmin(objectives);
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
        b[:,idxs_bad] .= 10^6;
    end
    return x_opt,obj, idxs_good, idxs_bad;
end
