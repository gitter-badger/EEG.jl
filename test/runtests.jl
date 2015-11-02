using EEG
using Base.Test
using Logging
using MAT, BDF
using SIUnits, SIUnits.ShortUnits
using FileFind

Logging.configure(level=DEBUG)

tests = AbstractString[]
function add_test(fname)
    global tests
    if endswith(fname, ".jl")
        push!(tests, fname)
    end
end
FileFind.find(".", add_test)

for t in tests
    include(t)
end
