
using Test

@testset "interface RepGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    q0 = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))

    obj = RepGradELBO(10)
    rng = StableRNG(seed)
    elbo_ref = estimate_objective(rng, obj, q0, model; n_samples=10^4)

    @testset "determinism" begin
        rng = StableRNG(seed)
        elbo = estimate_objective(rng, obj, q0, model; n_samples=10^4)
        @test elbo == elbo_ref
    end

    @testset "default_rng" begin
        elbo = estimate_objective(obj, q0, model; n_samples=10^4)
        @test elbo ≈ elbo_ref rtol = 0.1
    end
end

@testset "interface RepGradELBO STL variance reduction" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)
    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    ad_backends = [
        ADTypes.AutoForwardDiff(), ADTypes.AutoReverseDiff(), ADTypes.AutoZygote()
    ]
    if @isdefined(Tapir)
        push!(ad_backends, AutoTapir(; safe_mode=false))
    end
    if @isdefined(Enzyme)
        push!(ad_backends, AutoEnzyme())
    end

    @testset for ad in ad_backends
        q_true = MeanFieldGaussian(
            Vector{eltype(μ_true)}(μ_true), Diagonal(Vector{eltype(L_true)}(diag(L_true)))
        )
        params, re = Optimisers.destructure(q_true)
        obj = RepGradELBO(10; entropy=StickingTheLandingEntropy())
        out = DiffResults.DiffResult(zero(eltype(params)), similar(params))

        aux = (rng=rng, obj=obj, problem=model, restructure=re, q_stop=q_true, adtype=ad)
        AdvancedVI.value_and_gradient!(
            ad, AdvancedVI.estimate_repgradelbo_ad_forward, params, aux, out
        )
        grad = DiffResults.gradient(out)
        @test norm(grad) ≈ 0 atol = 1e-5
    end
end
