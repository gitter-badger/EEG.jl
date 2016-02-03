using ProgressMeter

function beamformer_lcmv(s::SSR, n::SSR, l::Leadfield; foi::Real=modulationrate(s), fs::Real=samplingrate(s), kwargs...)

    if !haskey(s.processing, "epochs")
        s = extract_epochs(s)
    end
    if !haskey(n.processing, "epochs")
        n = extract_epochs(n)
    end

    beamformer_lcmv(s.processing["epochs"], n.processing["epochs"], l.L, l.x, l.y, l.z, fs, foi; kwargs...)
end

"""
Linearly constrained minimum variance (LCMV) beamformer for epoched data

LCMV beamformer returning neural activity index (NAI), source and noise variance (Van Veen et al 1997).
Coherent source supression for a source or region is also implemented (Dalal et al 2006).


### TODO

* Eigenspace projection (Sekihara et al 2001)


### Literature

Localization of brain electrical activity via linearly constrained minimum variance spatial filtering  
Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A.  
Biomedical Engineering, IEEE Transactions on, 44(9):867–880, 1997.

Reconstructing spatio-temporal activities of neural sources using an meg vector beamformer technique  
Kensuke Sekihara, Srikantan S Nagarajan, David Poeppel, Alec Marantz, and Yasushi Miyashita.  
Biomedical Engineering, IEEE Transactions on, 48(7):760–771, 2001.

Modified beamformers for coherent source region suppression  
Sarang S Dalal, Kensuke Sekihara, and Srikantan S Nagarajan.  
Biomedical Engineering, IEEE Transactions on, 53(7):1357–1363, 2006.


### Input

* x = M * T x N matrix = Signal M sample measurements over T trials on N electrodes
* n = M * T x N matrix = Noise  M sample measurements over T trials on N electrodes
* H = L x D x N matrix = Forward head model for L locations on N electrodes in D dimensions
* #_loc = Cartesian locations for headmodel
* fs = Sample rate
* foi = Frequency of interest for cross spectral density
* freq_pm = Frequency above and below `foi` to include in csd calculation (1.0)
* bilateral =  Radius of bilateral region to supress. 0 implies no supression

"""
function beamformer_lcmv{A <: AbstractFloat}(x::Array{A, 3}, n::Array{A, 3}, H::Array{A, 3},
                         x_loc::Vector{A}, y_loc::Vector{A}, z_loc::Vector{A},
                         fs::Real = 8192, foi::Real = 40.0;
                         freq_pm::Real = 1.0, bilateral::Real = 0.0, kwargs...)

    Logging.debug("Starting LCMV beamforming on epoch data")

    # Constants
    M = size(x, 1)   # Samples
    N = size(x, 3)   # Sensors
    L = size(H, 1)   # Locations
    D = size(H, 2)   # Dimensions

    # Check input
    @assert size(n, 3) == N     # Ensure inputs match
    @assert size(H, 3) == N     # Ensure inputs match
    @assert M > N               # Should have more samples than sensors
    @assert L == length(x_loc)
    @assert L == length(y_loc)
    @assert L == length(z_loc)
    @assert !any(isnan(x))
    @assert !any(isnan(n))
    @assert !any(isnan(H))

    Logging.info("LCMV epoch beamformer using $M samples on $N sensors for $L sources over $D dimensions")

    C = cross_spectral_density(x, foi - freq_pm, foi + freq_pm, fs)
    Q = cross_spectral_density(n, foi - freq_pm, foi + freq_pm, fs)

    # More (probably unnecessary) checks
    @assert size(C) == size(Q)
    @assert C != Q

    Logging.debug("Covariance matrices calculated and of size $(size(Q))")

    beamformer_lcmv(C, Q, H, x_loc, y_loc, z_loc, bilateral; kwargs...)
end


function beamformer_lcmv{A <: AbstractFloat}(C::Array{Complex{A}, 2}, Q::Array{Complex{A}, 2}, H::Array{A, 3},
                              x::Vector{A}, y::Vector{A}, z::Vector{A}, bilateral::Real; reduce_dim::Bool=false, kwargs...)

    Logging.debug("Computing LCMV beamformer from CPSD data")

    N = size(C, 1)   # Sensors
    L = size(H, 1)   # Locations

    # Space to save results
    Variance  = Array(Float64, (L, 1))         # Variance
    Noise     = Array(Float64, (L, 1))         # Noise
    NAI       = Array(Float64, (L, 1))         # Neural Activity Index

    Logging.debug("Result variables pre allocated")

    # Compute inverse outside loop
    invC = pinv(C) # + 0.01 * eye(C)
    invQ = pinv(Q) # + 0.01 * eye(C)

    prog = Progress(L, 1, "  LCMV scan... ", 50)
    Logging.debug("Beamformer scan started")

    for l = 1:L

        # Extract leadfield for location
        H_l = squeeze(H[l,:,:], 1)'

        # Coherent source suppression
        if bilateral > 0

            # Determine locations within radius of bilateral source
            to_supress = falses(size(x))
            for loc in 1:L
                if euclidean([-x[l], y[l], z[l]], [x[loc], y[loc], z[loc]]) < bilateral
                    to_supress[loc] = true
                end
            end
            Ls = H[to_supress, :, :]

            # Append bilateral sources to leadfield
            Ls = reshape(permutedims(Ls, [3, 2, 1]), N, size(Ls, 1) * 3)
            if reduce_dim
                H_l = hcat(H_l, svdfact(Ls).U[:, 1:6])
            else
                H_l = hcat(H_l, Ls)
            end
        end

        Variance[l], Noise[l], NAI[l] = beamformer_lcmv(invC, invQ, H_l)

        next!(prog)
    end

    Logging.debug("Beamformer scan completed")

    return Variance, Noise, NAI
end


function beamformer_lcmv(invC::Array{Complex{Float64}, 2}, invQ::Array{Complex{Float64}, 2}, H::Array{Float64, 2})

    V_q = trace(pinv(H' * invC * H)[1:3, 1:3])   # Strength of source     Eqn 24: trace(3x3)

    N_q = trace(pinv(H' * invQ * H)[1:3, 1:3])   # Noise strength         Eqn 26: trace(3x3)

    NAI = V_q / N_q                              # Neural activity index  Eqn 27

    return abs(V_q), abs(N_q), abs(NAI)
end


###############################
#
# Cross power spectral density
#
###############################

"""
Compute complex cross spectral density of epoch data

Cross spectral density averaged over frequencies of interest

### Input

* epochs: data shaped as epochs (samples x trials x channels)
* fmin: minimum frequency of interest
* max: maximum frequency of interest

### Implementation

Currently uses MNE python library.
Will change to Synchrony.jl when its stabilised.

"""
function cross_spectral_density{T <: AbstractFloat}(epochs::Array{T, 3}, fmin::Real, fmax::Real, fs::Real)

    @pyimport mne as mne
    @pyimport mne.time_frequency as tf

    Logging.debug("Cross power spectral density between $fmin - $fmax")

    # Convert from EEG.jl to MNE format for epochs
    epochs = permutedims(epochs, [2, 3, 1])                      # Change to trials x channels x samples
    names = AbstractString[string(i) for i = 1:size(epochs, 2)]  # Hack to put in fake names
    events = ones(Int, size(epochs, 1), 3)                       # Make all events the same to use everything

    # Run MNE processing
    i = pycall(mne.create_info, PyObject, ch_names = vec(names), ch_types = vec(repmat(["eeg"], size(epochs, 2))), sfreq = fs)
    epochs = mne.EpochsArray(epochs, i, events, 0.0, verbose = false)
    csd = tf.compute_epochs_csd(epochs, fmin = fmin, fmax = fmax, verbose = false, fsum = true)
    csd = csd[:data]
end
