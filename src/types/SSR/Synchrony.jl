#######################################
#
# Connectivity
#
#######################################

@doc md"""
Calculate phase lag index between SSR sensors.

This is a wrapper function for the SSR type.
The calculation of PLI is calculated using [Synchrony.jl](www.github.com/.....)
""" ->
function phase_lag_index(a::SSR, ChannelOrigin::Int, ChannelDestination::Int, freq_of_interest::Real;
     ID::String="", kwargs... )

    err("PLI code has not been validated. Do not use")

    info("Phase lag index on SSR channels $(a.channel_names[ChannelOrigin]) and $(a.channel_names[ChannelDestination]) for $freq_of_interest Hz")

    data = permutedims(a.processing["epochs"], [1, 3, 2])

    pli = phase_lag_index(data[:, [ChannelOrigin, ChannelDestination], :], freq_of_interest, samplingrate(a))

    result = DataFrame(
                        ID                  = ID,
                        AnalysisFrequency   = freq_of_interest,
                        ModulationFrequency = float(a.modulationfreq),
                        ChannelOrigin       = copy(a.channel_names[ChannelOrigin]),
                        ChannelDestination  = copy(a.channel_names[ChannelDestination]),
                        AnalysisType        = "phase_lag_index",
                        Strength            = pli
                      )

    result = add_dataframe_static_rows(result, kwargs)

    key_name = new_processing_key(a.processing, "pli")
    merge!(a.processing, [key_name => result])

    return a
end


# If you want multiple frequencies analyse each one in turn
function phase_lag_index(a::SSR, ChannelOrigin::Int, ChannelDestination::Int, freq_of_interest::AbstractArray; kwargs...)

    for f in freq_of_interest
        a = phase_lag_index(a, ChannelOrigin, ChannelDestination, freq_of_interest=f; kwargs...)
    end

    return a
end


# If you dont specify an analysis frequency, use modulation frequency
function phase_lag_index(a::SSR, ChannelOrigin::Int, ChannelDestination::Int;
        freq_of_interest::Union(Real, AbstractArray)=[float(a.modulationfreq)],
        ID::String="", kwargs...)
    phase_lag_index(a, ChannelOrigin, ChannelDestination, freq_of_interest, ID=ID; kwargs...)
end


# Analyse between two sensors by name
function phase_lag_index(a::SSR, ChannelOrigin::String, ChannelDestination::String; kwargs... )

    ChannelOrigin =      int(findfirst(a.channel_names, ChannelOrigin))
    ChannelDestination = int(findfirst(a.channel_names, ChannelDestination))

    debug("Converted channel names to indices $ChannelOrigin $ChannelDestination")

    a = phase_lag_index(a, ChannelOrigin, ChannelDestination, freq_of_interest, ID=ID; kwargs... )

end


# Analyse list of sensors provided by index
function phase_lag_index(a::SSR, ChannelOrigin::Array{Int}; kwargs...)

    for i = 1:length(ChannelOrigin)-1
        for j = i+1:length(ChannelOrigin)
            a = phase_lag_index(a, ChannelOrigin[i], ChannelOrigin[j]; kwargs...)
        end
    end

    return a
end


# Analyse list of sensors provided by name
function phase_lag_index(a::SSR, ChannelOrigin::Array{ASCIIString}; kwargs...)

    idxs = [int(findfirst(a.channel_names, co)) for co in ChannelOrigin]

    phase_lag_index(a, idxs; kwargs...)
end


# Analyse all sensors
function phase_lag_index(a::SSR; kwargs... )

    phase_lag_index(a, a.channel_names; kwargs...)
end



#
# Save results
#

# Save synchrony results to file
function save_synchrony_results(a::SSR; name_extension::String="-synchrony", kwargs...)

    file_name = string(a.file_name, name_extension, ".csv")

    # Rename to save space
    results = a.processing

    # Index of keys to be exported
    result_idx = find_keys_containing(results, "pli")

    debug("Found $(length(result_idx)) synchrony results")

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


