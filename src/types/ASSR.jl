using DataFrames
using MAT
using SIUnits
import SIUnits

typealias FreqHz{T} SIUnits.SIQuantity{T,0,0,-1,0,0,0,0}

type ASSR
    data::Array
    triggers::Dict
    system_codes::Dict
    sample_rate::FreqHz{Number}
    modulation_frequency::FreqHz{Number}
    reference_channel::Array{String}
    file_path::String
    file_name::String
    channel_names::Array{String}
    processing::Dict
end


import Base.show
function Base.show(io::IO, a::ASSR)
    time_length = round(size(a.data,1) / a.sample_rate / 60, 2)
    println(io, "ASSR measurement of $time_length mins with $(size(a.data,2)) channels sampled at $(a.sample_rate)")
    println(io, "  Modulation frequency: $(a.modulation_frequency )")

    if haskey(a.processing, "Amplitude")
        println(io, "  Stimulation amplitude: $(a.processing["Amplitude"]) dB")
    end
    if haskey(a.processing, "Name")
        println(io, "  Participant name: $(a.processing["Name"] )")
    end
    if haskey(a.processing, "Side")
        println(io, "  Stimulation side: $(a.processing["Side"] )")
    end

end


#######################################
#
# Read ASSR
#
#######################################

function read_ASSR(fname::Union(String, IO); kwargs...)

    info("Importing file $fname")

    if isa(fname, String)
        file_path, file_name, ext = fileparts(fname)
        debug("Importing file for ASSR processing")
    else
        warn("Filetype is IO. Might be bugged")
        file_path = "IO"
        file_name = fname
        ext = "IO"
    end

    # Extract frequency from the file name
    if contains(file_name, "Hz")
        a = match(r"[-_](\d+[_.]?[\d+]?)Hz|Hz(\d+[_.]?[\d+]?)[-_]", file_name).captures
        modulation_frequency = assr_frequency(float(a[[i !== nothing for i = a]][1])) * Hertz
    else
        modulation_frequency = NaN
    end

    # Or even better if there is a mat file read it
    mat_path = string(file_path, file_name, ".mat")
    stimulation_side      = nothing  # In case it is not
    participant_name      = nothing  # defined in the mat file
    stimulation_amplitude = nothing
    if isreadable(mat_path)
        modulation_frequency, stimulation_side, participant_name, stimulation_amplitude = read_rba_mat(mat_path)
    end

    # Import raw data
    if ext == "bdf"
        data, triggers, system_codes, sample_rate, reference_channel, header = import_biosemi(fname)
    else
        warn("File type $ext is unknown")
    end

    # Create ASSR type
    a = ASSR(data, triggers, system_codes, sample_rate * Hertz, modulation_frequency, [reference_channel], file_path, file_name,
             header["chanLabels"], Dict())

    if stimulation_side != nothing
        a.processing["Side"] = stimulation_side
    end
    if participant_name != nothing
        a.processing["Name"] = participant_name
    end
    if stimulation_amplitude != nothing
        a.processing["Amplitude"] = stimulation_amplitude
    end

    # Remove status channel information
    remove_channel!(a, "Status")

    # Clean epoch index
    a.triggers = clean_triggers(a.triggers; kwargs...)

    return a
end




#######################################
#
# Read event files
#
#######################################

function read_evt(a::ASSR, fname::String; kwargs...)
    d = read_evt(fname, a.sample_rate; kwargs...)
    validate_triggers(d)
    a.triggers = d
    return a
end


#######################################
#
# Add channels
#
#######################################

function add_channel(a::ASSR, data::Array, chanLabels::ASCIIString; kwargs...)

    info("Adding channel $chanLabels")

    a.data = hcat(a.data, data)
    push!(a.channel_names, chanLabels)

    return a
end


#######################################
#
# Remove channels
#
#######################################

function remove_channel!(a::ASSR, channel_idx::Array{Int}; kwargs...)

    channel_idx = channel_idx[channel_idx .!= 0]

    info("Removing channel(s) $channel_idx")

    keep_idx = [1:size(a.data)[end]]
    for c = sort(channel_idx, rev=true)
        try
            splice!(keep_idx, c)
        end
    end

    a.data = a.data[:, keep_idx]

    a.channel_names = a.channel_names[keep_idx]
end

function remove_channel!(a::ASSR, channel_names::Array{ASCIIString}; kwargs...)
    remove_channel!(a, int([findfirst(a.channel_names, c) for c=channel_names]))
    info("Removing channel(s) $(append_strings(channel_names))"); end

function remove_channel!(a::ASSR, channel_names::Union(Array{String}); kwargs...)
    remove_channel!(a, convert(Array{ASCIIString}, channel_names)); end

function remove_channel!(a::ASSR, channel_name::Union(Int, String, ASCIIString); kwargs...)
    remove_channel!(a, [channel_name]); end

#######################################
#
# Trim channels
#
#######################################

function trim_channel(a::ASSR, stop::Int; start::Int=1, kwargs...)

    info("Trimming $(size(a.data)[end]) channels between $start and $stop")

    a.data = a.data[start:stop,:]

    to_keep = find(a.triggers["Index"] .<= stop)
    a.triggers["Index"]        = a.triggers["Index"][to_keep]
    a.triggers["Duration"]     = a.triggers["Duration"][to_keep]
    a.triggers["Code"]         = a.triggers["Code"][to_keep]

    to_keep = find(a.system_codes["Index"] .<= stop)
    a.system_codes["Index"]    = a.system_codes["Index"][to_keep]
    a.system_codes["Duration"] = a.system_codes["Duration"][to_keep]
    a.system_codes["Code"]     = a.system_codes["Code"][to_keep]

    return a
end


#######################################
#
# Merge channels
#
#######################################

function merge_channels(a::ASSR, merge_Chans::Array{ASCIIString}, new_name::String; kwargs...)

    debug("Total origin channels: $(length(a.channel_names))")

    keep_idxs = [findfirst(a.channel_names, i) for i = merge_Chans]
    keep_idxs = int(keep_idxs)

    if sum(keep_idxs .== 0) > 0
        warn("Could not merge channels as don't exist: $(append_strings(vec(merge_Chans[keep_idxs .== 0])))")
        keep_idxs = keep_idxs[keep_idxs .> 0]
    end

    info("Merging channels $(append_strings(vec(a.channel_names[keep_idxs,:])))")
    debug("Merging channels $keep_idxs")

    a = add_channel(a, mean(a.data[:,keep_idxs], 2), new_name; kwargs...)
end


#######################################
#
# Filtering
#
#######################################


function highpass_filter(a::ASSR; cutOff::Number=2, order::Int=3, tolerance::Number=0.01, kwargs...)

    a.data, f = highpass_filter(a.data, cutOff=cutOff, order=order, fs=int(a.sample_rate))

    _filter_check(f, float(a.modulation_frequency), int(a.sample_rate), tolerance)

    _append_filter(a, f)
end


function lowpass_filter(a::ASSR; cutOff::Number=150, order::Int=3, tolerance::Number=0.01, kwargs...)

    a.data, f = lowpass_filter(a.data, cutOff=cutOff, order=order, fs=int(a.sample_rate))

    _filter_check(f, float(a.modulation_frequency), int(a.sample_rate), tolerance)

    _append_filter(a, f)
end


function bandpass_filter(a::ASSR;
                         lower::Number=float(a.modulation_frequency)-1,
                         upper::Number=float(a.modulation_frequency)+1,
                         n::Int=24, rp::Number=0.0001, tolerance::Number=0.01, kwargs...)

    # Type 1 Chebychev filter
    # The default options here are optimised for modulation frequencies 4, 10, 20, 40, 80
    # TODO filter check does not work here. Why not?
    # TODO automatic minimum filter order selection

    a.data, f = bandpass_filter(a.data, lower, upper, int(a.sample_rate), n, rp)

    _filter_check(f, float(a.modulation_frequency), int(a.sample_rate), tolerance)

    _append_filter(a, f)
end


function _filter_check(f::Filter, mod_freq::Number, fs::Number, tolerance::Number)
    #
    # Ensure that the filter does not alter the modulation frequency greater than a set tolerance
    #

    mod_change = abs(freqz(f, mod_freq, fs))
    if mod_change > 1 + tolerance || mod_change < 1 - tolerance
        warn("Filtering has modified modulation frequency greater than set tolerance: $mod_change")
    end
    debug("Filter magnitude at modulation frequency: $(mod_change)")
end


function _append_filter(a::ASSR, f::Filter; name::String="filter")
    #
    # Put the filter information in the ASSR processing structure
    #

    key_name = new_processing_key(a.processing, "filter")
    merge!(a.processing, [key_name => f])

    return a
end


#######################################
#
# Change reference channels
#
#######################################

function rereference(a::ASSR, refChan::Union(String, Array{ASCIIString}); kwargs...)

    a.data = rereference(a.data, refChan, a.channel_names)

    a.reference_channel = [refChan]

    return a
end


#######################################
#
# Rejection channels
#
#######################################

function channel_rejection(a::ASSR; kwargs...)

    if haskey(a.processing, "epochs")

        data = reshape(a.processing["epochs"],
                size(a.processing["epochs"], 1) * size(a.processing["epochs"], 2), size(a.processing["epochs"],3))
    else
        data = a.data
    end

    valid = channel_rejection(data, kwargs...)

    info("Rejected $(sum(!valid)) channels $(append_strings(a.channel_names[find(!valid)]))")

    remove_channel!(a, a.channel_names[find(!valid)])

    return a
end


#######################################
#
# Add triggers for more epochs
#
#######################################


function add_triggers(a::ASSR; kwargs...)

    debug("Adding triggers to reduce ASSR. Using ASSR modulation frequency")

    add_triggers(a, float(a.modulation_frequency); kwargs...)
end


function add_triggers(a::ASSR, mod_freq::Number; kwargs...)

    debug("Adding triggers to reduce ASSR. Using $(mod_freq)Hz")

    epochIndex = DataFrame(Code = a.triggers["Code"], Index = a.triggers["Index"]);
    epochIndex[:Code] = epochIndex[:Code] - 252

    add_triggers(a, mod_freq, epochIndex; kwargs...)
end


function add_triggers(a::ASSR, mod_freq::Number, epochIndex; cycle_per_epoch::Int=1, kwargs...)

    info("Adding triggers to reduce ASSR. Reducing $(mod_freq)Hz to $cycle_per_epoch cycle(s).")

    # Existing epochs
    existing_epoch_length   = median(diff(epochIndex[:Index]))     # samples
    existing_epoch_length_s = existing_epoch_length / float(a.sample_rate)
    debug("Existing epoch length: $(existing_epoch_length_s)s")

    # New epochs
    new_epoch_length_s = cycle_per_epoch / mod_freq
    new_epochs_num     = round(existing_epoch_length_s / new_epoch_length_s) - 2
    new_epoch_times    = [1:new_epochs_num]*new_epoch_length_s
    new_epoch_indx     = [0, round(new_epoch_times * float(a.sample_rate))]
    debug("New epoch length = $new_epoch_length_s")
    debug("New # epochs     = $new_epochs_num")

    # Place new epoch indices
    debug("Was $(length(epochIndex[:Index])) indices")
    new_indx = epochIndex[:Index][1:end-1] .+ new_epoch_indx'
    new_indx = reshape(new_indx', length(new_indx), 1)[1:end-1]
    debug("Now $(length(new_indx)) indices")

    # Place in dict
    new_code = int(ones(1, length(new_indx))) .+ 252
    a.triggers = ["Index" => vec(int(new_indx)'), "Code" => vec(new_code), "Duration" => ones(length(new_code), 1)']
    #TODO Possible the trigger duration of one is not long enough

    return a
end



#######################################
#
# Extract epochs
#
#######################################

function extract_epochs(a::ASSR; kwargs...)

    merge!(a.processing, ["epochs" => extract_epochs(a.data, a.triggers)])

    return a
end


function create_sweeps(a::ASSR; epochsPerSweep::Int=32, kwargs...)

    merge!(a.processing, ["sweeps" => create_sweeps(a.processing["epochs"], epochsPerSweep = epochsPerSweep)])

    return a
end



#######################################
#
# File IO
#
#######################################

function trigger_channel(a::ASSR; kwargs...)

    create_channel(a.triggers, a.data, float(a.sample_rate))
end


function system_code_channel(a::ASSR; kwargs...)

    create_channel(a.system_codes, a.data, float(a.sample_rate))
end


function write_ASSR(a::ASSR, fname::String; kwargs...)

    info("Saving $(size(a.data)[end]) channels to $fname")

    writeBDF(fname, a.data', trigger_channel(a), system_code_channel(a), int(a.sample_rate), chanLabels=a.channel_names)

end


#######################################
#
# Statistics
#
#######################################


function ftest(a::ASSR, freq_of_interest::Number; ID::String="", kwargs... )

    # TODO: Account for multiple applied filters
    if haskey(a.processing, "filter1")
        used_filter = a.processing["filter1"]
    else
        used_filter = nothing
    end

    snrDb, phase, signal, noise, statistic =
        ftest(a.processing["sweeps"], freq_of_interest, int(a.sample_rate), used_filter = used_filter; kwargs...)

    result = DataFrame(
                        ID                  = vec(repmat([ID], length(a.channel_names), 1)),
                        Channel             = copy(a.channel_names),
                        ModulationFrequency = copy(float(a.modulation_frequency)),
                        AnalysisType        = "ftest",
                        AnalysisFrequency   = freq_of_interest,
                        SignalPower         = vec(signal),
                        SignalPhase         = vec(phase),
                        NoisePower          = vec(noise),
                        SNRdB               = vec(snrDb),
                        Statistic           = vec(statistic)
                      )

    result = add_dataframe_static_rows(result, kwargs)

    key_name = new_processing_key(a.processing, "ftest")
    merge!(a.processing, [key_name => result])

    return a
end


# If more than one frequency of interest is specified then run for all
function ftest(a::ASSR, freq_of_interest::Array; kwargs...)

    for f = freq_of_interest; a = ftest(a, f; kwargs...); end; return a
end


# If no frequency of interest is specified then use the modulation frequency
function ftest(a::ASSR; kwargs...)

    ftest(a, float(a.modulation_frequency); kwargs...)
end


# Save ftest results to file
function save_results(a::ASSR; name_extension::String="", kwargs...)

    file_name = string(a.file_name, name_extension, ".csv")

    # Rename to save space
    results = a.processing

    # Index of keys to be exported
    result_idx = find_keys_containing(results, "ftest")

    if length(result_idx) > 0

        to_save = get(results, collect(keys(results))[result_idx[1]], 0)

        if length(result_idx) > 1
            for k = result_idx[2:end]
                result_data = get(results, collect(keys(results))[k], 0)
                to_save = rbind(to_save, result_data)
            end
        end

    writetable(file_name, to_save)
    end

    info("File saved to $file_name")

    return a
end

#######################################
#
# Helper functions
#
#######################################


function assr_frequency(rounded_freq::Number; stimulation_sample_rate::Number=32000,
                        stimulation_frames_per_epoch::Number=32768)

    round(rounded_freq/(stimulation_sample_rate / stimulation_frames_per_epoch)) *
                                                                stimulation_sample_rate / stimulation_frames_per_epoch
end

function assr_frequency(rounded_freq::AbstractVector)

    [assr_frequency(f) for f = rounded_freq]
end


