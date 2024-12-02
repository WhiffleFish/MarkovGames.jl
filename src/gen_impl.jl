POMDPs.gen(::Game, s, a, rng) = NamedTuple()

@generated function POMDPs.genout(::DDNOut{symbols}, m::Game, s, a, rng) where symbols

    # use anything available from gen(m, s, a, rng)
    expr = quote
        x = gen(m, s, a, rng)
        @assert x isa NamedTuple "gen(m::Game, ...) must return a NamedTuple; got a $(typeof(x))"
    end
    
    # add gen for any other variables
    for (var, depargs) in POMDPs.sorted_deppairs(m, symbols)
        if var in (:s, :a) # input nodes
            continue
        end

        sym = Meta.quot(var)

        varblock = quote
            if haskey(x, $sym) # should be constant at compile time
                $var = x[$sym]
            else
                $var = $(POMDPs.node_expr(Val(var), depargs))
            end
        end
        append!(expr.args, varblock.args)
    end

    # add return expression
    if symbols isa Tuple
        return_expr = :(return $(Expr(:tuple, symbols...)))
    else
        return_expr = :(return $symbols)
    end
    append!(expr.args, return_expr.args)

    return expr
end

function POMDPs.sorted_deppairs(::Type{<:POMG}, symbols)
    deps = Dict(:s => Symbol[],
                :a => Symbol[],
                :sp => [:s, :a],
                :o => [:s, :a, :sp],
                :r => [:s, :a, :sp, :o],
                :info => Symbol[]
               )
    return POMDPs.sorted_deppairs(deps, symbols)
end

function POMDPs.sorted_deppairs(::Type{<:MG}, symbols)
    deps = Dict(:s => Symbol[],
                :a => Symbol[],
                :sp => [:s, :a],
                :r => [:s, :a, :sp, :o],
                :info => Symbol[]
               )
    return POMDPs.sorted_deppairs(deps, symbols)
end
