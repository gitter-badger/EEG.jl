#######################################
#
# Test beamformer
#
#######################################

fname = joinpath(dirname(@__FILE__), "..", "data", "test_Hz19.5-testing.bdf")

a = read_SSR(fname)

x = copy(a.data')

n = randn(size(x))

H = randn(38*28*26, 3, 6)

V, N, NAI = beamformer_lcmv(x, n, H, checks=true, progress=true)

NAI = reshape(NAI, (38, 28, 26, 1))

write_dat(joinpath(dirname(@__FILE__), "..", "data", "tmp", "SA.dat"), 1:size(NAI,1), 1:size(NAI,2), 1:size(NAI,3), NAI, 1:size(NAI,4))

println()
println("!! Beamforming test passed !!")
println()
