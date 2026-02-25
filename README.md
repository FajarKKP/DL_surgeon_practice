# DL_surgeon_practice
A repo for me on how to operate and edit a DL model.

Lessons so far:

- Width directly increases representational capacity.
- Width is one of the best way to improve CNN performance at the price of higher computational cost.

- Downpooling is not optional but necessary. 
- Pooling / using stride is how CNN transition from local / simple feature to semantic feature.

- Learnable downsampling often produces better results and more stable representations compared to hard max selection (ex MaxAvgPool).

- Depth improves abstraction capacity if the optimization and downpoolng is handled correctly / well designed.

- Regularization reduces memorization but does not increase representational power. If architecture is weak, regularization wont fix it like magic.

- Batch size mainly affects optimization dynamics, not model capacity.

