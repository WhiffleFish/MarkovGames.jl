@testset "distributions" begin
    d1 = SparseCat(['a', 'b'], [0.3, 0.7])
    d2 = Uniform([1,2])
    d = ProductDistribution(d1, d2)
    @test d[1] === d1
    @test d[2] === d2
    @test pdf(d, ('a', 1)) ≈ 0.3*0.5
    @test pdf(d, ('a', 2)) ≈ 0.3*0.5
    @test pdf(d, ('b', 1)) ≈ 0.7*0.5
    @test pdf(d, ('b', 2)) ≈ 0.7*0.5
end
