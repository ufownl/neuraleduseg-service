Files are used in this module:

```
>>> __file__                                                                                             
'/usr/local/lib/python3.6/site-packages/allennlp/commands/elmo.py'
```

and are downloaded from here:

```
>>> DEFAULT_OPTIONS_FILE                                                                                                                                       
'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'                          
>>> DEFAULT_WEIGHT_FILE                                                                                                                                        
'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
```

The hdf5 file is too large to store in a Github repo, therefore I download it
in the Dockerfile before the installation. I couldn't get git-lfs to work with Github.
