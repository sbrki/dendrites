Quickstart guide
================================

Basic quickstart
------------------------

Here is the basic quickstart, in one piece.

.. literalinclude:: ../examples/basic.py

Which will produce something like:

.. code-block:: none

    [0.057752093363465082, 0.94236022733877001, 0.057788004623543256]
    [0.057752093363465082, 0.94236022733877001, 0.057788004623543256]


.. seealso::

    :ref:`api-docs-ref` to see this code broken down piece by piece, as well as some other features.


Quickstart with logging
------------------------

If you want to use **the logging module** and see detailed logs from
dendrites *(recommended)*, try this:

.. literalinclude:: ../examples/basic_with_logging.py

Which will produce somthing like:

.. code-block:: none

    DEBUG:root:[INIT] Initialization started
    DEBUG:root:[INIT] Getting network dimensions
    DEBUG:root:[INIT] Created a network with dimensions: (2, 3)
    DEBUG:root:[INIT] Creating the synapse...
    DEBUG:root:[INIT] Created the synapse with 1 elements.
    INFO:root:[INIT] Initialization Done
    INFO:root:[TRAINING] Training started
    DEBUG:root:[TRAINING] Generation = 0, error = 0.6400547667802593
    DEBUG:root:[TRAINING] Generation = 10, error = 0.6092681116887855
    DEBUG:root:[TRAINING] Generation = 20, error = 0.5802111707589749

    # (...)

    DEBUG:root:[TRAINING] Generation = 4650, error = 0.010036204401125974
    DEBUG:root:[TRAINING] Generation = 4660, error = 0.010012413606938565
    INFO:root:[TRAINING] Training done, in 4666 generations, with a final error of 0.009998191544524379
    INFO:root:[0.057715475761935281, 0.94223379271320262, 0.057687209685138088]
    INFO:root:[SAVE]Saved neural network to file <net.dat>.
    DEBUG:root:[INIT] Initialization started
    DEBUG:root:[INIT] Getting network dimensions
    DEBUG:root:[INIT] Created a network with dimensions: ()
    DEBUG:root:[INIT] Creating the synapse...
    DEBUG:root:[INIT] Created the synapse with 0 elements.
    INFO:root:[INIT] Initialization Done
    INFO:root:[LOAD]Loaded neural network from file <net.dat>
    INFO:root:[0.057715475761935281, 0.94223379271320262, 0.057687209685138088]
