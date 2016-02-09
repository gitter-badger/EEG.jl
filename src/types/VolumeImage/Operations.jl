import Base: +, -, /, *, mean, maximum, minimum


# +

function +(vi1::VolumeImage, vi2::VolumeImage)

    dimensions_equal(vi1, vi2)

    debug("Adding two volume images with $(size(vi1.data, 4)) time instances")

    vout = deepcopy(vi1)

    vout.data = vi1.data .+ vi2.data

    return vout
end


# -

function -(vi1::VolumeImage, vi2::VolumeImage)

    dimensions_equal(vi1, vi2)

    debug("Subtracting two volume images with $(size(vi1.data, 4)) time instances")

    vout = deepcopy(vi1)

    vout.data = vi1.data .- vi2.data

    return vout
end


# /

function /(vi1::VolumeImage, vi2::VolumeImage)

    dimensions_equal(vi1, vi2)

    debug("Dividing two volume images with $(size(vi1.data, 4)) time instances")

    vout = deepcopy(vi1)

    vout.data = vi1.data ./ vi2.data

    return vout
end

function /(vi::VolumeImage, c::Number)

    vout = deepcopy(vi)

    vout.data = vi.data ./ c

    return vout
end


# *

function *(vi::VolumeImage, c::Number)

    vout = deepcopy(vi)

    vout.data = vi.data .* c

    return vout
end


# mean

function mean(vi::VolumeImage)

    debug("Taking mean of one volume images with $(size(vi.data, 4)) time instances")

    vout = deepcopy(vi)

    vout.data = mean(vout.data, 4)

    # Store time as 0 to indicate its been averaged
    vout.t = [NaN]

    return vout
end

function mean(va::Array{VolumeImage,1})

    debug("Taking mean of $(length(va)) volume images with $(size(va[1].data, 4)) time instances")

    mean_va = deepcopy(va[1])

    for i in 2:length(va)

        mean_va = mean_va + va[i]

    end

    return mean_va / length(va)
end


# maximum

function maximum(vi::VolumeImage)

    maximum(vi.data)
end

function maximum(vis::Array{VolumeImage})

    maximum([maximum(vi) for vi in vis])
end


# minimum

function minimum(vi::VolumeImage)

    minimum(vi.data)
end

function minimum(vis::Array{VolumeImage})

    minimum([minimum(vi) for vi in vis])
end



# normalise

function normalise(vi::VolumeImage)

    debug("Normalising one volume images with $(size(vi.data, 4)) time instances")

    normalisation_constant = maximum(vi)

    vi = deepcopy(vi) / normalisation_constant

    vi.info["NormalisationConstant"] = normalisation_constant

    return vi
end

function normalise(va::Array{VolumeImage, 1})

    debug("Normalising $(length(va)) volume images with $(size(va[1].data, 4)) time instances")

    vo = deepcopy(va)
    for i in 1:length(vo)
        vo[i] = normalise(vo[i])
    end
    return vo
end


#
# Helper functions
# ----------------
#

function dimensions_equal(vi1::VolumeImage, vi2::VolumeImage; x::Bool=true, y::Bool=true, z::Bool=true, t::Bool=true, kwargs...)

    matching = true
    if x & !(vi1.x == vi2.x)
        throw(KeyError("X dimensions do not match"))
    end
    if y & !(vi1.y == vi2.y)
        throw(KeyError("Y dimensions do not match"))
    end
    if z & !(vi1.z == vi2.z)
        throw(KeyError("Z dimensions do not match"))
    end
    if t & !(vi1.t == vi2.t)
        throw(KeyError("T dimensions do not match"))
    end

    if matching
        return true
    end
end



###############################
#
# Correct midline for bilateral
#
###############################


"""
Correct midline estimates from bilateral beamformer

Region supression for bilateral beamformers supresses itself along the midline. This causes innacurate estimates
around 0 on the x axis. To compensate for this take the average of adjacent valid locations.

#### Input

* vi: Volume image
* pm: Locations plus or minus the midline in mm to correct
"""
function correct_midline(v::VolumeImage; pm::Real=0.005, units=Meter)

    Logging.info("Correcting volume image midline errors caused by bilateral region supression")
    Logging.debug("Supressing region ± $(pm * units)")

    x = AbstractFloat[xi / (1 * units) for xi in v.x]
    midline_idxs = abs(x)
    midline_idxs = midline_idxs .<= pm

    valid_idxs = falses(size(midline_idxs))
    valid_idxs[minimum(find(midline_idxs))-1] = true
    valid_idxs[maximum(find(midline_idxs))+1] = true

    new_val = mean(v.data[valid_idxs, :, :], 1)
    for i in find(midline_idxs)
        v.data[i, :, :] = new_val
    end

    return v
end
