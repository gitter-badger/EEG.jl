using ProgressMeter


#######################################
#
# LCMV Beamformer
#
# Localization of brain electrical activity via linearly constrained minimum variance spatial filtering
# Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A. (1997)
#
#######################################


function beamformer_lcmv(x::Array, n::Array, H::Array; progress::Bool=false, checks::Bool=false)
    # LCMV beamformer as described in Van Veen et al '97
    #
    # Input:    x = N x M matrix     = M sample measurements on N electrodes
    #           n = N x M matrix     = M sample measurements of noise on N electrodes
    #           H = L x 3 x N matrix = Forward head model.   !! Different to paper !!

    # This function sets everything up and checks data before passing to efficient actual calculation

    # Constants
    N = size(x, 1)   # Sensors
    M = size(x, 2)   # Samples
    L = size(H, 1)   # Locations

    Logging.info("LCMV beamformer using $M samples on $N sensors for $L sources")

    # Some sanity checks on the input
    if size(H, 3) != N; error("Leadfield and data dont match"); end
    if size(H, 2) != 3; error("Leadfield dimension is incorrect"); end
    if N > L; error("More channels than samples. Something switched?"); end
    if any(isnan(x)); error("Their is a nan in your input x data"); end

    # Covariance matrices sampled at 4 times more locations than sensors
    C_x = cov(x[:, round(Int, linspace(2, M-1, 4*N))]')
    Q   = cov(n[:, round(Int, linspace(2, size(n, 2)-1, 4*N))]')

    # Space to save results
    Variance  = Array(Float64, (L,1))         # Variance
    Noise     = Array(Float64, (L,1))         # Noise
    NAI       = Array(Float64, (L,1))         # Neural Activity Index

    # More checks
    if any(isnan(C_x)); error("Their is a nan in your signal data"); end
    if any(isnan(Q)); error("Their is a nan in your noise data"); end

    # Scan each location
    if progress; p = Progress(L, 1, "  Scanning... ", 50); end
    for l = 1:L
        Variance[l], Noise[l], NAI[l] = beamformer_lcmv_actual(C_x, squeeze(H[l,:,:], 1)', Q, N=N, checks=checks)
        if progress; next!(p); end
    end

    return Variance, Noise, NAI
end


function beamformer_lcmv_actual(C_x::Array, H::Array, Q::Array; N=64, checks::Bool=false)

    if checks
        if size(H, 1) != N; error("Leadfield = $(size(H, 1)) and data = $(N) dont match"); end
        if size(H, 2) != 3; error("Leadfield dimension is incorrect"); end
        if size(C_x, 1) != N; error("Covariance size is incorrect"); end
        if size(C_x, 2) != N; error("Covariance size is incorrect"); end
        if size(Q) != size(C_x); error("Covariance matrices dont match"); end
    end

    # Strength of source
    V_q = trace( inv( H' * pinv(C_x) * H ) )

    # Noise strength
    N_q = trace( inv(H' * inv(Q) * H) )

    # Neural activity index
    NAI = V_q / N_q

    return V_q, N_q, NAI
end


