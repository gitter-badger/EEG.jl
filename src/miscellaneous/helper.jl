@doc doc"""
Return a new processing key with the number incremented.
It checks for existing keys and returns a string with the next key to be used.

#### Arguments

* `d`: Dictionary containing existing keys
* `key_name`: Base of the

#### Returns

* String with new key name

#### Returns

```julia
results_storage = Dict()
results_storage[new_processing_key(results_storage, "FTest")] = 4
results_storage[new_processing_key(results_storage, "FTest")] = 49

# Dict(Any, Any) with 2 entries
#   "FTest1" => 4
#   "FTest2" => 49
```
""" ->
function new_processing_key(d::Dict, key_name::String)
    key_numb = 1
    key = string(key_name, key_numb)
    while haskey(d, key)
        key_numb += 1
        key = string(key_name, key_numb)
    end
    return key
end


@doc doc"""
Find dictionary keys containing a string.

#### Arguments

* `d`: Dictionary containing existing keys
* `partial_key`: String you want to find in key names

#### Returns

* Array containg the indices of dictionary containing the partial_key

#### Returns

```julia
results_storage = Dict()
results_storage[new_processing_key(results_storage, "FTest")] = 4
results_storage[new_processing_key(results_storage, "Turtle")] = 5
results_storage[new_processing_key(results_storage, "FTest")] = 49

find_keys_containing(results_storage, "FTest")

# 2-element Array{Int64,1}:
#  1
#  3
```
""" ->
function find_keys_containing(d::Dict, partial_key::String)
    valid_keys = [beginswith(i, partial_key) for i = collect(keys(d))]
    findin(valid_keys, true)
end


@doc doc"""
Extract the path, filename and extension of a file

#### Arguments

* `fname`: String with the full path to a file

#### Output

* Three strings containing the path, file name and file extension

#### Returns

```julia
fileparts("/Users/test/subdir/test-file.bdf")

# ("/Users/test/subdir/","test-file","bdf")
```
""" ->
function fileparts(fname::String)
    if fname==""
        pathname  = ""
        filename  = ""
        extension = ""
    else
        separators = sort(unique([search(fname, '/', i) for i = 1:length(fname)]))
        pathname = fname[1:last(separators)]
        extension  = last(sort(unique([search(fname, '.', i) for i = 1:length(fname)])))
        filename = fname[last(separators)+1:extension-1]
        extension  = fname[extension+1:end]
    end

    return pathname, filename, extension
end


@doc doc"""
Find the closest number to a target in an array and return the index

#### Arguments

* `list`: Array containing numbers
* `target`: Number to find closest to in the list

#### Output

* Index of the closest number to the target

#### Returns

```julia
_find_closest_number_idx([1, 2, 2.7, 3.2, 4, 3.1, 7], 3)

# 6
```
""" ->
function _find_closest_number_idx{T <: Number}(list::Array{T, 1}, target::Number)
    diff_array = abs(list .- target)
    targetIdx  = findfirst(diff_array , minimum(diff_array))
end


#######################################
#
# DataFrame manipulation
#
#######################################

function add_dataframe_static_rows(a::DataFrame, args...)
    debug("Adding column(s)")
    for kwargs in args
        debug(kwargs)
        for k in kwargs
            name = convert(Symbol, k[1])
            code = k[2]
            expanded_code = vec(repmat([k[2]], size(a, 1), 1))
            debug("Name: $name  Code: $code")
            DataFrames.insert_single_column!(a, DataFrames.upgrade_vector(expanded_code), size(a,2)+1)
            rename!(a, convert(Symbol, string("x", size(a,2))),  name)
        end
    end
    return a
end
