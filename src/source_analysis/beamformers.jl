using ProgressMeter

"""
Linearly constrained minimum variance (LCMV) beamformer as described in Van Veen et al '97


### Literature

Localization of brain electrical activity via linearly constrained minimum variance spatial filtering  
Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A. (1997)

### Input

* x = N x M matrix     = M sample measurements on N electrodes
* n = N x M matrix     = M sample measurements of noise on N electrodes
* H = L x 3 x N matrix = Forward head model.   !! Different to paper !!

"""
function beamformer_lcmv{A <: AbstractFloat, B <: AbstractFloat, D <: AbstractFloat}(x::Array{A, 2}, n::Array{B, 2},
                         H::Array{D, 3}; progress::Bool=false, checks::Bool=false)

    x = convert(typeof(n), x)

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
    Logging.debug("Sanity checks passed")

    # Covariance matrices sampled at 4 times more locations than sensors
    C = cov(x')
    Q = cov(n')
    Logging.debug("Covariance matrices calculated and of size $(size(Q))")

    beamformer_lcmv_cpsd(C, Q, H, progress = progress, checks = checks)
end


function beamformer_lcmv_cpsd{T <: AbstractFloat}(C::Array{T, 2}, Q::Array{T, 2}, H::Array{T, 3};
    progress::Bool=true, checks::Bool=true)

    N = size(C, 1)   # Sensors
    L = size(H, 1)   # Locations

    # Space to save results
    Variance  = Array(Float64, (L, 1))         # Variance
    Noise     = Array(Float64, (L, 1))         # Noise
    NAI       = Array(Float64, (L, 1))         # Neural Activity Index
    Logging.debug("Results vairables pre allocated")

    # More checks
    if any(isnan(C)); error("Their is a nan in your signal data"); end
    if any(isnan(Q)); error("Their is a nan in your noise data"); end

    invC = pinv(C)
    invQ = pinv(Q)

    # Scan each location
    Logging.debug("Beamformer scan started")
    if progress; p = Progress(L, 1, "  Scanning... ", 50); end
    for l = 1:L
        Variance[l], Noise[l], NAI[l] = beamformer_lcmv_actual(invC, squeeze(H[l,:,:], 1)', invQ, N=N, checks=checks)
        if progress; next!(p); end
    end

    return Variance, Noise, NAI
end


function beamformer_lcmv_actual{A <: AbstractFloat}(invC::Array{A, 2}, H::Array{A, 2}, invQ::Array{A, 2}; N=64, checks::Bool=false)

    if checks
        if size(H, 1) != N; error("Leadfield = $(size(H, 1)) and data = $(N) dont match"); end
        if size(H, 2) != 3; error("Leadfield dimension is incorrect $(size(H, 2)) != 3"); end
        if size(invC, 1) != N; error("Covariance size is incorrect $(size(invC, 1)) != $N"); end
        if size(invC, 2) != N; error("Covariance size is incorrect $(size(invC, 2)) != $N"); end
        if size(invQ) != size(invC); error("Covariance matrices dont match $(size(invQ)) != $(size(invC))"); end
    end

    # Strength of source
    V_q = trace( inv(H' * invC * H ) )   # Eqn 24: trace(3x3)

    # Noise strength
    N_q = trace( inv(H' * invQ * H) )    # Eqn 26: trace(3x3)

    # Neural activity index
    NAI = V_q / N_q                      # Eqn 27

    return V_q, N_q, NAI
end


