
function project(s::ASSR, l::leadfield, idx::Int)

    #
    # This uses the source model from Van Veen et al '97
    #
    # For a given instance (t)
    #
    # measurements = headmodel * sources   =>  source = headmodel ^-1 * measurements
    #     Nx1           Nx3    *   3x1           3x1  =    3xN        *     Nx1
    #      x       =    H      *    m             m   =   H ^-1 = L   *      x
    #

    info("Projecting from ASSR to source $idx")

    # Sanity checks
    if size(s.data,2) != size(l.L,3); error("Leadfield and data do not match"); end
    if s.header["chanLabels"] != l.sensors; error("Sensors do not match"); end

    squeeze(l.L[idx, :, :],1) * s.data'
end

