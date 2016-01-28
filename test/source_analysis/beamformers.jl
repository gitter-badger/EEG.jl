#######################################
#
# Test beamformer
#
#######################################

fname = joinpath(dirname(@__FILE__), "..", "data", "test_Hz19.5-testing.bdf")

a = read_SSR(fname)

x = copy(a.data')

n = randn(size(x))

H = randn(5*5*5, 3, 6)

V, N, NAI = beamformer_lcmv(x, n, H, checks=true, progress=true)


#
# Export as volume image
#

x = repmat(collect(1:5.0), 25)
y = repmat(vec(ones(5) * collect(1:5)'), 5)
z = vec(ones(5*5) * collect(1:5)')
t = [1.0]
NAI = vec(NAI)

vi = VolumeImage(NAI, "nA/cm^3", x, y, z, t, "LCMV", Dict(), "Talairach")

NAI = reshape(NAI, (5, 5, 5, 1))

write_dat(joinpath(dirname(@__FILE__), "..", "data", "tmp", "SA.dat"), 1:size(NAI,1), 1:size(NAI,2), 1:size(NAI,3), NAI, 1:size(NAI,4))

println()
println("!! Beamforming test passed !!")
println()
