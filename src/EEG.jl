module EEG

using Logging,  # For user feedback
      Docile,   # For documentation
      Compat,   # For backward julia compatability
      SIUnits,  # Smarter units
      SIUnits.ShortUnits,
      #= Synchrony, =#
      DataFrames,
      Distances,
      ProgressMeter,
      AWS,
      AWS.S3,
      BDF,
      DSP,
      Distributions,
      Winston,
      Gadfly,
      Images,
      BDF,
      MAT

import Winston: oplot, plot

export # Helper functions
       new_processing_key,
       find_keys_containing,
       fileparts,
       add_dataframe_static_rows,
       _find_closest_number_idx,
       # File type reading and writing
       import_biosemi,
       channelNames_biosemi_1020,
       create_channel,
       create_events,
       read_avr,
       read_bsa,
       read_dat,
       read_sfp,
       read_elp,
       write_dat,
       prepare_dat,
       write_avr,
       read_evt,
       read_rba_mat,
       # Pre-processing
       epoch_rejection,
       channel_rejection,
       highpass_filter,
       lowpass_filter,
       bandpass_filter,
       compensate_for_filter,
       remove_template,
       rereference,
       clean_triggers,
       validate_triggers,
       extra_triggers,
       downsample,
       # Reshaping of data
       extract_epochs,
       average_epochs,
       create_sweeps,
       # Statistics
       ftest,
       gfp,
       # Synchrony
       phase_lag_index,
       save_synchrony_results,
       # Type - Volume Image
       VolumeImage,
       read_VolumeImage,
       plot,
       normalise,
       find_dipoles,
       # Type - SSR
       SSR,
       samplingrate,
       modulationrate,
       read_SSR,
       trim_channel,
       add_triggers,
       remove_channel!,
       keep_channel!,
       add_channel,
       assr_frequency,
       save_results,
       trigger_channel,
       write_SSR,
       merge_channels,
       bootstrap,
       hcat,
       # Source analysis
       Electrodes,
           show,
           match_sensors,
       EEG_64_10_20,
       EEG_Vanvooren_2014,
       EEG_Vanvooren_2014_Left,
       EEG_Vanvooren_2014_Right,
       Coordinate,
           SPM,
           BrainVision,
           Talairach,
       convert,
       conv_bv2tal,
       conv_spm_mni2tal,
       beamformer_lcmv,
       Dipole,
       best_dipole,
       orient_dipole,
       best_ftest_dipole,
       Leadfield,
       match_leadfield,
       find_location,
       project,
       euclidean,
       # Plotting
       plot,
       oplot,
       plot_dat,
       plot_spectrum,
       plot_timeseries,
       plot_single_channel_timeseries,
       plot_multi_channel_timeseries,
       oplot_dipoles,
       SSR_spectrogram,
       plot_filter_response,
       plot_ftest



@document(manual = ["../doc/manual.md"])


# Helper functions
include("miscellaneous/helper.jl")

# File type reading and writing
include("read_write/avr.jl")
include("read_write/bdf.jl")
include("read_write/bsa.jl")
include("read_write/dat.jl")
include("read_write/elp.jl")
include("read_write/evt.jl")
include("read_write/rba.jl")
include("read_write/sfp.jl")

# Pre-processing
include("preprocessing/data_rejection.jl")
include("preprocessing/filtering.jl")
include("preprocessing/reference.jl")
include("preprocessing/triggers.jl")

# Reshaping of data
include("reshaping/epochs.jl")
include("reshaping/sweeps.jl")

# Statistics
include("statistics/ftest.jl")
include("statistics/gfp.jl")

# Synchrony
include("synchrony/phase_lag_index.jl")

# Type - SSR
include("types/SSR/SSR.jl")
include("types/SSR/Preprocessing.jl")
include("types/SSR/ReadWrite.jl")
include("types/SSR/Reshaping.jl")
include("types/SSR/Statistics.jl")
include("types/SSR/Synchrony.jl")

# Type - Dipole
include("types/Dipole/Dipole.jl")
include("types/Dipole/Plotting.jl")

# Source analysis
include("source_analysis/beamformers.jl")
include("source_analysis/projection.jl")

# Type - Sensors
include("types/Sensors/Sensors.jl")

# Type - Coordinates
include("types/Coordinates/Coordinates.jl")
include("source_analysis/dipoles.jl")           # wrong section

# Type - Leadfield
include("types/Leadfield/Leadfield.jl")

# Type - Volume Image
include("types/VolumeImage/VolumeImage.jl")
include("types/VolumeImage/ReadWrite.jl")
include("types/VolumeImage/Plotting.jl")
include("types/VolumeImage/Dipoles.jl")
include("types/VolumeImage/Operations.jl")

# Plotting functions
include("plotting/plot.jl")
include("plotting/winston.jl")
include("types/SSR/plotting.jl")

end # module
