       �K"	   6���Abrain.Event:2�\p�>K     6�.	�>6���A"��
y
input/Spectrum-inputPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
t
input/Label-inputPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
l
weights/random_normal/shapeConst*
valueB"X  �   *
dtype0*
_output_shapes
:
_
weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
weights/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
*weights/random_normal/RandomStandardNormalRandomStandardNormalweights/random_normal/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights/random_normal/mulMul*weights/random_normal/RandomStandardNormalweights/random_normal/stddev* 
_output_shapes
:
��*
T0
~
weights/random_normalAddweights/random_normal/mulweights/random_normal/mean* 
_output_shapes
:
��*
T0
�
weights/weight1
VariableV2*
shape:
��*
shared_name *
dtype0* 
_output_shapes
:
��*
	container 
�
weights/weight1/AssignAssignweights/weight1weights/random_normal*
use_locking(*
T0*"
_class
loc:@weights/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights/weight1/readIdentityweights/weight1*
T0*"
_class
loc:@weights/weight1* 
_output_shapes
:
��
n
weights/random_normal_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:
a
weights/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights/random_normal_1/RandomStandardNormalRandomStandardNormalweights/random_normal_1/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights/random_normal_1/mulMul,weights/random_normal_1/RandomStandardNormalweights/random_normal_1/stddev*
T0* 
_output_shapes
:
��
�
weights/random_normal_1Addweights/random_normal_1/mulweights/random_normal_1/mean*
T0* 
_output_shapes
:
��
�
weights/weight2
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights/weight2/AssignAssignweights/weight2weights/random_normal_1*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*"
_class
loc:@weights/weight2
�
weights/weight2/readIdentityweights/weight2*
T0*"
_class
loc:@weights/weight2* 
_output_shapes
:
��
n
weights/random_normal_2/shapeConst*
valueB"�   d   *
dtype0*
_output_shapes
:
a
weights/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights/random_normal_2/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights/random_normal_2/RandomStandardNormalRandomStandardNormalweights/random_normal_2/shape*

seed *
T0*
dtype0*
_output_shapes
:	�d*
seed2 
�
weights/random_normal_2/mulMul,weights/random_normal_2/RandomStandardNormalweights/random_normal_2/stddev*
_output_shapes
:	�d*
T0
�
weights/random_normal_2Addweights/random_normal_2/mulweights/random_normal_2/mean*
T0*
_output_shapes
:	�d
�
weights/weight3
VariableV2*
shape:	�d*
shared_name *
dtype0*
_output_shapes
:	�d*
	container 
�
weights/weight3/AssignAssignweights/weight3weights/random_normal_2*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*"
_class
loc:@weights/weight3

weights/weight3/readIdentityweights/weight3*
T0*"
_class
loc:@weights/weight3*
_output_shapes
:	�d
n
weights/random_normal_3/shapeConst*
valueB"d      *
dtype0*
_output_shapes
:
a
weights/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights/random_normal_3/RandomStandardNormalRandomStandardNormalweights/random_normal_3/shape*
dtype0*
_output_shapes

:d*
seed2 *

seed *
T0
�
weights/random_normal_3/mulMul,weights/random_normal_3/RandomStandardNormalweights/random_normal_3/stddev*
T0*
_output_shapes

:d
�
weights/random_normal_3Addweights/random_normal_3/mulweights/random_normal_3/mean*
_output_shapes

:d*
T0
�
weights/weight_out
VariableV2*
dtype0*
_output_shapes

:d*
	container *
shape
:d*
shared_name 
�
weights/weight_out/AssignAssignweights/weight_outweights/random_normal_3*
use_locking(*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d
�
weights/weight_out/readIdentityweights/weight_out*
_output_shapes

:d*
T0*%
_class
loc:@weights/weight_out
e
biases/random_normal/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
^
biases/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
`
biases/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
)biases/random_normal/RandomStandardNormalRandomStandardNormalbiases/random_normal/shape*

seed *
T0*
dtype0*
_output_shapes	
:�*
seed2 
�
biases/random_normal/mulMul)biases/random_normal/RandomStandardNormalbiases/random_normal/stddev*
T0*
_output_shapes	
:�
v
biases/random_normalAddbiases/random_normal/mulbiases/random_normal/mean*
T0*
_output_shapes	
:�
z
biases/bias1
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
biases/bias1/AssignAssignbiases/bias1biases/random_normal*
use_locking(*
T0*
_class
loc:@biases/bias1*
validate_shape(*
_output_shapes	
:�
r
biases/bias1/readIdentitybiases/bias1*
T0*
_class
loc:@biases/bias1*
_output_shapes	
:�
g
biases/random_normal_1/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
`
biases/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
biases/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+biases/random_normal_1/RandomStandardNormalRandomStandardNormalbiases/random_normal_1/shape*

seed *
T0*
dtype0*
_output_shapes	
:�*
seed2 
�
biases/random_normal_1/mulMul+biases/random_normal_1/RandomStandardNormalbiases/random_normal_1/stddev*
_output_shapes	
:�*
T0
|
biases/random_normal_1Addbiases/random_normal_1/mulbiases/random_normal_1/mean*
T0*
_output_shapes	
:�
z
biases/bias2
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
biases/bias2/AssignAssignbiases/bias2biases/random_normal_1*
use_locking(*
T0*
_class
loc:@biases/bias2*
validate_shape(*
_output_shapes	
:�
r
biases/bias2/readIdentitybiases/bias2*
_output_shapes	
:�*
T0*
_class
loc:@biases/bias2
f
biases/random_normal_2/shapeConst*
dtype0*
_output_shapes
:*
valueB:d
`
biases/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
biases/random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+biases/random_normal_2/RandomStandardNormalRandomStandardNormalbiases/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:d*
seed2 *

seed 
�
biases/random_normal_2/mulMul+biases/random_normal_2/RandomStandardNormalbiases/random_normal_2/stddev*
T0*
_output_shapes
:d
{
biases/random_normal_2Addbiases/random_normal_2/mulbiases/random_normal_2/mean*
T0*
_output_shapes
:d
x
biases/bias3
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
�
biases/bias3/AssignAssignbiases/bias3biases/random_normal_2*
use_locking(*
T0*
_class
loc:@biases/bias3*
validate_shape(*
_output_shapes
:d
q
biases/bias3/readIdentitybiases/bias3*
T0*
_class
loc:@biases/bias3*
_output_shapes
:d
f
biases/random_normal_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:
`
biases/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
biases/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases/random_normal_3/RandomStandardNormalRandomStandardNormalbiases/random_normal_3/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
�
biases/random_normal_3/mulMul+biases/random_normal_3/RandomStandardNormalbiases/random_normal_3/stddev*
T0*
_output_shapes
:
{
biases/random_normal_3Addbiases/random_normal_3/mulbiases/random_normal_3/mean*
T0*
_output_shapes
:
{
biases/bias_out
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
biases/bias_out/AssignAssignbiases/bias_outbiases/random_normal_3*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
z
biases/bias_out/readIdentitybiases/bias_out*
T0*"
_class
loc:@biases/bias_out*
_output_shapes
:
n
weights_1/random_normal/shapeConst*
valueB"X  �   *
dtype0*
_output_shapes
:
a
weights_1/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights_1/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights_1/random_normal/RandomStandardNormalRandomStandardNormalweights_1/random_normal/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights_1/random_normal/mulMul,weights_1/random_normal/RandomStandardNormalweights_1/random_normal/stddev*
T0* 
_output_shapes
:
��
�
weights_1/random_normalAddweights_1/random_normal/mulweights_1/random_normal/mean*
T0* 
_output_shapes
:
��
�
weights_1/weight1
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights_1/weight1/AssignAssignweights_1/weight1weights_1/random_normal*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
weights_1/weight1/readIdentityweights_1/weight1* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight1
p
weights_1/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   
c
weights_1/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 weights_1/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
.weights_1/random_normal_1/RandomStandardNormalRandomStandardNormalweights_1/random_normal_1/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights_1/random_normal_1/mulMul.weights_1/random_normal_1/RandomStandardNormal weights_1/random_normal_1/stddev* 
_output_shapes
:
��*
T0
�
weights_1/random_normal_1Addweights_1/random_normal_1/mulweights_1/random_normal_1/mean*
T0* 
_output_shapes
:
��
�
weights_1/weight2
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights_1/weight2/AssignAssignweights_1/weight2weights_1/random_normal_1*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight2/readIdentityweights_1/weight2*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
p
weights_1/random_normal_2/shapeConst*
valueB"�   d   *
dtype0*
_output_shapes
:
c
weights_1/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 weights_1/random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
.weights_1/random_normal_2/RandomStandardNormalRandomStandardNormalweights_1/random_normal_2/shape*

seed *
T0*
dtype0*
_output_shapes
:	�d*
seed2 
�
weights_1/random_normal_2/mulMul.weights_1/random_normal_2/RandomStandardNormal weights_1/random_normal_2/stddev*
T0*
_output_shapes
:	�d
�
weights_1/random_normal_2Addweights_1/random_normal_2/mulweights_1/random_normal_2/mean*
T0*
_output_shapes
:	�d
�
weights_1/weight3
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
�
weights_1/weight3/AssignAssignweights_1/weight3weights_1/random_normal_2*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
�
weights_1/weight3/readIdentityweights_1/weight3*
_output_shapes
:	�d*
T0*$
_class
loc:@weights_1/weight3
p
weights_1/random_normal_3/shapeConst*
valueB"d      *
dtype0*
_output_shapes
:
c
weights_1/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 weights_1/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
.weights_1/random_normal_3/RandomStandardNormalRandomStandardNormalweights_1/random_normal_3/shape*
dtype0*
_output_shapes

:d*
seed2 *

seed *
T0
�
weights_1/random_normal_3/mulMul.weights_1/random_normal_3/RandomStandardNormal weights_1/random_normal_3/stddev*
T0*
_output_shapes

:d
�
weights_1/random_normal_3Addweights_1/random_normal_3/mulweights_1/random_normal_3/mean*
T0*
_output_shapes

:d
�
weights_1/weight_out
VariableV2*
shape
:d*
shared_name *
dtype0*
_output_shapes

:d*
	container 
�
weights_1/weight_out/AssignAssignweights_1/weight_outweights_1/random_normal_3*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
�
weights_1/weight_out/readIdentityweights_1/weight_out*
T0*'
_class
loc:@weights_1/weight_out*
_output_shapes

:d
g
biases_1/random_normal/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
`
biases_1/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
biases_1/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases_1/random_normal/RandomStandardNormalRandomStandardNormalbiases_1/random_normal/shape*
dtype0*
_output_shapes	
:�*
seed2 *

seed *
T0
�
biases_1/random_normal/mulMul+biases_1/random_normal/RandomStandardNormalbiases_1/random_normal/stddev*
T0*
_output_shapes	
:�
|
biases_1/random_normalAddbiases_1/random_normal/mulbiases_1/random_normal/mean*
T0*
_output_shapes	
:�
|
biases_1/bias1
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
biases_1/bias1/AssignAssignbiases_1/bias1biases_1/random_normal*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
x
biases_1/bias1/readIdentitybiases_1/bias1*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
i
biases_1/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
b
biases_1/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
biases_1/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-biases_1/random_normal_1/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_1/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
�
biases_1/random_normal_1/mulMul-biases_1/random_normal_1/RandomStandardNormalbiases_1/random_normal_1/stddev*
_output_shapes	
:�*
T0
�
biases_1/random_normal_1Addbiases_1/random_normal_1/mulbiases_1/random_normal_1/mean*
T0*
_output_shapes	
:�
|
biases_1/bias2
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
biases_1/bias2/AssignAssignbiases_1/bias2biases_1/random_normal_1*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
x
biases_1/bias2/readIdentitybiases_1/bias2*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
h
biases_1/random_normal_2/shapeConst*
valueB:d*
dtype0*
_output_shapes
:
b
biases_1/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
biases_1/random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
-biases_1/random_normal_2/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:d*
seed2 *

seed 
�
biases_1/random_normal_2/mulMul-biases_1/random_normal_2/RandomStandardNormalbiases_1/random_normal_2/stddev*
T0*
_output_shapes
:d
�
biases_1/random_normal_2Addbiases_1/random_normal_2/mulbiases_1/random_normal_2/mean*
T0*
_output_shapes
:d
z
biases_1/bias3
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
�
biases_1/bias3/AssignAssignbiases_1/bias3biases_1/random_normal_2*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
w
biases_1/bias3/readIdentitybiases_1/bias3*
T0*!
_class
loc:@biases_1/bias3*
_output_shapes
:d
h
biases_1/random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
b
biases_1/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
biases_1/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-biases_1/random_normal_3/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_3/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
�
biases_1/random_normal_3/mulMul-biases_1/random_normal_3/RandomStandardNormalbiases_1/random_normal_3/stddev*
T0*
_output_shapes
:
�
biases_1/random_normal_3Addbiases_1/random_normal_3/mulbiases_1/random_normal_3/mean*
T0*
_output_shapes
:
}
biases_1/bias_out
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
biases_1/bias_out/AssignAssignbiases_1/bias_outbiases_1/random_normal_3*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
�
biases_1/bias_out/readIdentitybiases_1/bias_out*
T0*$
_class
loc:@biases_1/bias_out*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Spectrum-inputweights_1/weight1/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
j
layer_1/AddAddlayer_1/MatMulbiases_1/bias1/read*
T0*(
_output_shapes
:����������
T
layer_2/ReluRelulayer_1/Add*
T0*(
_output_shapes
:����������
�
layer_2/MatMulMatMullayer_2/Reluweights_1/weight2/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
j
layer_2/AddAddlayer_2/MatMulbiases_1/bias2/read*
T0*(
_output_shapes
:����������
Z
layer_3/SigmoidSigmoidlayer_2/Add*(
_output_shapes
:����������*
T0
�
layer_3/MatMulMatMullayer_3/Sigmoidweights_1/weight3/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
i
layer_3/AddAddlayer_3/MatMulbiases_1/bias3/read*
T0*'
_output_shapes
:���������d
R
result/ReluRelulayer_3/Add*
T0*'
_output_shapes
:���������d
�
result/MatMulMatMulresult/Reluweights_1/weight_out/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j

result/AddAddresult/MatMulbiases_1/bias_out/read*
T0*'
_output_shapes
:���������
[
subSub
result/Addinput/Label-input*
T0*'
_output_shapes
:���������
A
AbsAbssub*'
_output_shapes
:���������*
T0
]
sub_1Sub
result/Addinput/Label-input*
T0*'
_output_shapes
:���������
I
SquareSquaresub_1*'
_output_shapes
:���������*
T0
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
q
,learning_rate/ExponentialDecay/learning_rateConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
j
#learning_rate/ExponentialDecay/CastCastVariable/read*
_output_shapes
: *

DstT0*

SrcT0
j
'learning_rate/ExponentialDecay/Cast_1/xConst*
value
B :�'*
dtype0*
_output_shapes
: 
�
%learning_rate/ExponentialDecay/Cast_1Cast'learning_rate/ExponentialDecay/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
l
'learning_rate/ExponentialDecay/Cast_2/xConst*
valueB
 *��u?*
dtype0*
_output_shapes
: 
�
&learning_rate/ExponentialDecay/truedivRealDiv#learning_rate/ExponentialDecay/Cast%learning_rate/ExponentialDecay/Cast_1*
_output_shapes
: *
T0
v
$learning_rate/ExponentialDecay/FloorFloor&learning_rate/ExponentialDecay/truediv*
_output_shapes
: *
T0
�
"learning_rate/ExponentialDecay/PowPow'learning_rate/ExponentialDecay/Cast_2/x$learning_rate/ExponentialDecay/Floor*
_output_shapes
: *
T0
�
learning_rate/ExponentialDecayMul,learning_rate/ExponentialDecay/learning_rate"learning_rate/ExponentialDecay/Pow*
_output_shapes
: *
T0
[

loss/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
c
	loss/MeanMeanSquare
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
}
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
j
$train/gradients/loss/Mean_grad/ShapeShapeSquare*
_output_shapes
:*
T0*
out_type0
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
l
&train/gradients/loss/Mean_grad/Shape_1ShapeSquare*
_output_shapes
:*
T0*
out_type0
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
!train/gradients/Square_grad/mul/xConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
train/gradients/Square_grad/mulMul!train/gradients/Square_grad/mul/xsub_1*'
_output_shapes
:���������*
T0
�
!train/gradients/Square_grad/mul_1Mul&train/gradients/loss/Mean_grad/truedivtrain/gradients/Square_grad/mul*
T0*'
_output_shapes
:���������
j
 train/gradients/sub_1_grad/ShapeShape
result/Add*
_output_shapes
:*
T0*
out_type0
s
"train/gradients/sub_1_grad/Shape_1Shapeinput/Label-input*
T0*
out_type0*
_output_shapes
:
�
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
"train/gradients/sub_1_grad/ReshapeReshapetrain/gradients/sub_1_grad/Sum train/gradients/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
 train/gradients/sub_1_grad/Sum_1Sum!train/gradients/Square_grad/mul_12train/gradients/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
j
train/gradients/sub_1_grad/NegNeg train/gradients/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
$train/gradients/sub_1_grad/Reshape_1Reshapetrain/gradients/sub_1_grad/Neg"train/gradients/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

+train/gradients/sub_1_grad/tuple/group_depsNoOp#^train/gradients/sub_1_grad/Reshape%^train/gradients/sub_1_grad/Reshape_1
�
3train/gradients/sub_1_grad/tuple/control_dependencyIdentity"train/gradients/sub_1_grad/Reshape,^train/gradients/sub_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@train/gradients/sub_1_grad/Reshape*'
_output_shapes
:���������
�
5train/gradients/sub_1_grad/tuple/control_dependency_1Identity$train/gradients/sub_1_grad/Reshape_1,^train/gradients/sub_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@train/gradients/sub_1_grad/Reshape_1
r
%train/gradients/result/Add_grad/ShapeShaperesult/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/result/Add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/result/Add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/result/Add_grad/Shape'train/gradients/result/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/result/Add_grad/Reshape_1Reshape%train/gradients/result/Add_grad/Sum_1'train/gradients/result/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/result/Add_grad/tuple/group_depsNoOp(^train/gradients/result/Add_grad/Reshape*^train/gradients/result/Add_grad/Reshape_1
�
8train/gradients/result/Add_grad/tuple/control_dependencyIdentity'train/gradients/result/Add_grad/Reshape1^train/gradients/result/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/result/Add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/result/Add_grad/tuple/control_dependency_1Identity)train/gradients/result/Add_grad/Reshape_11^train/gradients/result/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/result/Add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/result/MatMul_grad/MatMulMatMul8train/gradients/result/Add_grad/tuple/control_dependencyweights_1/weight_out/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
+train/gradients/result/MatMul_grad/MatMul_1MatMulresult/Relu8train/gradients/result/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:d*
transpose_a(*
transpose_b( 
�
3train/gradients/result/MatMul_grad/tuple/group_depsNoOp*^train/gradients/result/MatMul_grad/MatMul,^train/gradients/result/MatMul_grad/MatMul_1
�
;train/gradients/result/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/result/MatMul_grad/MatMul4^train/gradients/result/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*<
_class2
0.loc:@train/gradients/result/MatMul_grad/MatMul
�
=train/gradients/result/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/result/MatMul_grad/MatMul_14^train/gradients/result/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/result/MatMul_grad/MatMul_1*
_output_shapes

:d
�
)train/gradients/result/Relu_grad/ReluGradReluGrad;train/gradients/result/MatMul_grad/tuple/control_dependencyresult/Relu*
T0*'
_output_shapes
:���������d
t
&train/gradients/layer_3/Add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/Add_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/Add_grad/Shape(train/gradients/layer_3/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/Add_grad/SumSum)train/gradients/result/Relu_grad/ReluGrad6train/gradients/layer_3/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/Add_grad/ReshapeReshape$train/gradients/layer_3/Add_grad/Sum&train/gradients/layer_3/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
&train/gradients/layer_3/Add_grad/Sum_1Sum)train/gradients/result/Relu_grad/ReluGrad8train/gradients/layer_3/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_3/Add_grad/Reshape_1Reshape&train/gradients/layer_3/Add_grad/Sum_1(train/gradients/layer_3/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
1train/gradients/layer_3/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/Add_grad/Reshape+^train/gradients/layer_3/Add_grad/Reshape_1
�
9train/gradients/layer_3/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/Add_grad/Reshape2^train/gradients/layer_3/Add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/Add_grad/Reshape*'
_output_shapes
:���������d
�
;train/gradients/layer_3/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/Add_grad/Reshape_12^train/gradients/layer_3/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/Add_grad/Reshape_1*
_output_shapes
:d
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/Add_grad/tuple/control_dependencyweights_1/weight3/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_3/Sigmoid9train/gradients/layer_3/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�d*
transpose_a(
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
0train/gradients/layer_3/Sigmoid_grad/SigmoidGradSigmoidGradlayer_3/Sigmoid<train/gradients/layer_3/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
t
&train/gradients/layer_2/Add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
s
(train/gradients/layer_2/Add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/Add_grad/Shape(train/gradients/layer_2/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/Add_grad/Reshape_1Reshape&train/gradients/layer_2/Add_grad/Sum_1(train/gradients/layer_2/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
1train/gradients/layer_2/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/Add_grad/Reshape+^train/gradients/layer_2/Add_grad/Reshape_1
�
9train/gradients/layer_2/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/Add_grad/Reshape2^train/gradients/layer_2/Add_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*;
_class1
/-loc:@train/gradients/layer_2/Add_grad/Reshape
�
;train/gradients/layer_2/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/Add_grad/Reshape_12^train/gradients/layer_2/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/Add_grad/Reshape_1*
_output_shapes	
:�
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/Add_grad/tuple/control_dependencyweights_1/weight2/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_2/Add_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*(
_output_shapes
:����������
t
&train/gradients/layer_1/Add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
s
(train/gradients/layer_1/Add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/Add_grad/Shape(train/gradients/layer_1/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/Add_grad/ReshapeReshape$train/gradients/layer_1/Add_grad/Sum&train/gradients/layer_1/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_1/Add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_1/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/Add_grad/Reshape_1Reshape&train/gradients/layer_1/Add_grad/Sum_1(train/gradients/layer_1/Add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
1train/gradients/layer_1/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/Add_grad/Reshape+^train/gradients/layer_1/Add_grad/Reshape_1
�
9train/gradients/layer_1/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/Add_grad/Reshape2^train/gradients/layer_1/Add_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*;
_class1
/-loc:@train/gradients/layer_1/Add_grad/Reshape
�
;train/gradients/layer_1/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/Add_grad/Reshape_12^train/gradients/layer_1/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/Add_grad/Reshape_1*
_output_shapes	
:�
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/Add_grad/tuple/control_dependencyweights_1/weight1/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Spectrum-input9train/gradients/layer_1/Add_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
train/beta1_power/initial_valueConst*
valueB
 *fff?*!
_class
loc:@biases_1/bias1*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *!
_class
loc:@biases_1/bias1*
	container 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking(
y
train/beta1_power/readIdentitytrain/beta1_power*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*!
_class
loc:@biases_1/bias1
�
train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
y
train/beta2_power/readIdentitytrain/beta2_power*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
(weights_1/weight1/Adam/Initializer/zerosConst*$
_class
loc:@weights_1/weight1*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
weights_1/weight1/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight1*
	container *
shape:
��
�
weights_1/weight1/Adam/AssignAssignweights_1/weight1/Adam(weights_1/weight1/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight1/Adam/readIdentityweights_1/weight1/Adam* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight1
�
*weights_1/weight1/Adam_1/Initializer/zerosConst*$
_class
loc:@weights_1/weight1*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
weights_1/weight1/Adam_1
VariableV2*
shared_name *$
_class
loc:@weights_1/weight1*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
weights_1/weight1/Adam_1/AssignAssignweights_1/weight1/Adam_1*weights_1/weight1/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight1/Adam_1/readIdentityweights_1/weight1/Adam_1*
T0*$
_class
loc:@weights_1/weight1* 
_output_shapes
:
��
�
(weights_1/weight2/Adam/Initializer/zerosConst*$
_class
loc:@weights_1/weight2*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
weights_1/weight2/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight2*
	container *
shape:
��
�
weights_1/weight2/Adam/AssignAssignweights_1/weight2/Adam(weights_1/weight2/Adam/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
weights_1/weight2/Adam/readIdentityweights_1/weight2/Adam*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
�
*weights_1/weight2/Adam_1/Initializer/zerosConst*
dtype0* 
_output_shapes
:
��*$
_class
loc:@weights_1/weight2*
valueB
��*    
�
weights_1/weight2/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight2*
	container *
shape:
��
�
weights_1/weight2/Adam_1/AssignAssignweights_1/weight2/Adam_1*weights_1/weight2/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
�
weights_1/weight2/Adam_1/readIdentityweights_1/weight2/Adam_1*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
�
(weights_1/weight3/Adam/Initializer/zerosConst*$
_class
loc:@weights_1/weight3*
valueB	�d*    *
dtype0*
_output_shapes
:	�d
�
weights_1/weight3/Adam
VariableV2*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name *$
_class
loc:@weights_1/weight3
�
weights_1/weight3/Adam/AssignAssignweights_1/weight3/Adam(weights_1/weight3/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
�
weights_1/weight3/Adam/readIdentityweights_1/weight3/Adam*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
�
*weights_1/weight3/Adam_1/Initializer/zerosConst*$
_class
loc:@weights_1/weight3*
valueB	�d*    *
dtype0*
_output_shapes
:	�d
�
weights_1/weight3/Adam_1
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *$
_class
loc:@weights_1/weight3*
	container *
shape:	�d
�
weights_1/weight3/Adam_1/AssignAssignweights_1/weight3/Adam_1*weights_1/weight3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
�
weights_1/weight3/Adam_1/readIdentityweights_1/weight3/Adam_1*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
�
+weights_1/weight_out/Adam/Initializer/zerosConst*'
_class
loc:@weights_1/weight_out*
valueBd*    *
dtype0*
_output_shapes

:d
�
weights_1/weight_out/Adam
VariableV2*
shared_name *'
_class
loc:@weights_1/weight_out*
	container *
shape
:d*
dtype0*
_output_shapes

:d
�
 weights_1/weight_out/Adam/AssignAssignweights_1/weight_out/Adam+weights_1/weight_out/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
weights_1/weight_out/Adam/readIdentityweights_1/weight_out/Adam*
T0*'
_class
loc:@weights_1/weight_out*
_output_shapes

:d
�
-weights_1/weight_out/Adam_1/Initializer/zerosConst*'
_class
loc:@weights_1/weight_out*
valueBd*    *
dtype0*
_output_shapes

:d
�
weights_1/weight_out/Adam_1
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *'
_class
loc:@weights_1/weight_out*
	container 
�
"weights_1/weight_out/Adam_1/AssignAssignweights_1/weight_out/Adam_1-weights_1/weight_out/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
�
 weights_1/weight_out/Adam_1/readIdentityweights_1/weight_out/Adam_1*
_output_shapes

:d*
T0*'
_class
loc:@weights_1/weight_out
�
%biases_1/bias1/Adam/Initializer/zerosConst*!
_class
loc:@biases_1/bias1*
valueB�*    *
dtype0*
_output_shapes	
:�
�
biases_1/bias1/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias1*
	container 
�
biases_1/bias1/Adam/AssignAssignbiases_1/bias1/Adam%biases_1/bias1/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias1/Adam/readIdentitybiases_1/bias1/Adam*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
�
'biases_1/bias1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias1*
valueB�*    
�
biases_1/bias1/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape:�
�
biases_1/bias1/Adam_1/AssignAssignbiases_1/bias1/Adam_1'biases_1/bias1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
�
biases_1/bias1/Adam_1/readIdentitybiases_1/bias1/Adam_1*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes	
:�
�
%biases_1/bias2/Adam/Initializer/zerosConst*!
_class
loc:@biases_1/bias2*
valueB�*    *
dtype0*
_output_shapes	
:�
�
biases_1/bias2/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias2*
	container *
shape:�
�
biases_1/bias2/Adam/AssignAssignbiases_1/bias2/Adam%biases_1/bias2/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias2/Adam/readIdentitybiases_1/bias2/Adam*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
�
'biases_1/bias2/Adam_1/Initializer/zerosConst*!
_class
loc:@biases_1/bias2*
valueB�*    *
dtype0*
_output_shapes	
:�
�
biases_1/bias2/Adam_1
VariableV2*
shared_name *!
_class
loc:@biases_1/bias2*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
biases_1/bias2/Adam_1/AssignAssignbiases_1/bias2/Adam_1'biases_1/bias2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias2
�
biases_1/bias2/Adam_1/readIdentitybiases_1/bias2/Adam_1*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
�
%biases_1/bias3/Adam/Initializer/zerosConst*!
_class
loc:@biases_1/bias3*
valueBd*    *
dtype0*
_output_shapes
:d
�
biases_1/bias3/Adam
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *!
_class
loc:@biases_1/bias3*
	container *
shape:d
�
biases_1/bias3/Adam/AssignAssignbiases_1/bias3/Adam%biases_1/bias3/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
�
biases_1/bias3/Adam/readIdentitybiases_1/bias3/Adam*
T0*!
_class
loc:@biases_1/bias3*
_output_shapes
:d
�
'biases_1/bias3/Adam_1/Initializer/zerosConst*!
_class
loc:@biases_1/bias3*
valueBd*    *
dtype0*
_output_shapes
:d
�
biases_1/bias3/Adam_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *!
_class
loc:@biases_1/bias3*
	container *
shape:d
�
biases_1/bias3/Adam_1/AssignAssignbiases_1/bias3/Adam_1'biases_1/bias3/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
�
biases_1/bias3/Adam_1/readIdentitybiases_1/bias3/Adam_1*
T0*!
_class
loc:@biases_1/bias3*
_output_shapes
:d
�
(biases_1/bias_out/Adam/Initializer/zerosConst*$
_class
loc:@biases_1/bias_out*
valueB*    *
dtype0*
_output_shapes
:
�
biases_1/bias_out/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out
�
biases_1/bias_out/Adam/AssignAssignbiases_1/bias_out/Adam(biases_1/bias_out/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
�
biases_1/bias_out/Adam/readIdentitybiases_1/bias_out/Adam*
_output_shapes
:*
T0*$
_class
loc:@biases_1/bias_out
�
*biases_1/bias_out/Adam_1/Initializer/zerosConst*$
_class
loc:@biases_1/bias_out*
valueB*    *
dtype0*
_output_shapes
:
�
biases_1/bias_out/Adam_1
VariableV2*
shared_name *$
_class
loc:@biases_1/bias_out*
	container *
shape:*
dtype0*
_output_shapes
:
�
biases_1/bias_out/Adam_1/AssignAssignbiases_1/bias_out/Adam_1*biases_1/bias_out/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
�
biases_1/bias_out/Adam_1/readIdentitybiases_1/bias_out/Adam_1*
_output_shapes
:*
T0*$
_class
loc:@biases_1/bias_out
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
-train/Adam/update_weights_1/weight1/ApplyAdam	ApplyAdamweights_1/weight1weights_1/weight1/Adamweights_1/weight1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*$
_class
loc:@weights_1/weight1
�
-train/Adam/update_weights_1/weight2/ApplyAdam	ApplyAdamweights_1/weight2weights_1/weight2/Adamweights_1/weight2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@weights_1/weight2*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
-train/Adam/update_weights_1/weight3/ApplyAdam	ApplyAdamweights_1/weight3weights_1/weight3/Adamweights_1/weight3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@weights_1/weight3*
use_nesterov( *
_output_shapes
:	�d
�
0train/Adam/update_weights_1/weight_out/ApplyAdam	ApplyAdamweights_1/weight_outweights_1/weight_out/Adamweights_1/weight_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/result/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:d*
use_locking( *
T0*'
_class
loc:@weights_1/weight_out
�
*train/Adam/update_biases_1/bias1/ApplyAdam	ApplyAdambiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias1*
use_nesterov( *
_output_shapes	
:�
�
*train/Adam/update_biases_1/bias2/ApplyAdam	ApplyAdambiases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias2*
use_nesterov( *
_output_shapes	
:�
�
*train/Adam/update_biases_1/bias3/ApplyAdam	ApplyAdambiases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias3*
use_nesterov( *
_output_shapes
:d
�
-train/Adam/update_biases_1/bias_out/ApplyAdam	ApplyAdambiases_1/bias_outbiases_1/bias_out/Adambiases_1/bias_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/result/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@biases_1/bias_out*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
�
train/Adam/updateNoOp.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
�
train/Adam/valueConst^train/Adam/update*
value	B :*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�

train/Adam	AssignAddVariabletrain/Adam/value*
T0*
_class
loc:@Variable*
_output_shapes
: *
use_locking( 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�#BVariableBbiases/bias1Bbiases/bias2Bbiases/bias3Bbiases/bias_outBbiases_1/bias1Bbiases_1/bias1/AdamBbiases_1/bias1/Adam_1Bbiases_1/bias2Bbiases_1/bias2/AdamBbiases_1/bias2/Adam_1Bbiases_1/bias3Bbiases_1/bias3/AdamBbiases_1/bias3/Adam_1Bbiases_1/bias_outBbiases_1/bias_out/AdamBbiases_1/bias_out/Adam_1Btrain/beta1_powerBtrain/beta2_powerBweights/weight1Bweights/weight2Bweights/weight3Bweights/weight_outBweights_1/weight1Bweights_1/weight1/AdamBweights_1/weight1/Adam_1Bweights_1/weight2Bweights_1/weight2/AdamBweights_1/weight2/Adam_1Bweights_1/weight3Bweights_1/weight3/AdamBweights_1/weight3/Adam_1Bweights_1/weight_outBweights_1/weight_out/AdamBweights_1/weight_out/Adam_1*
dtype0*
_output_shapes
:#
�
save/SaveV2/shape_and_slicesConst*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:#
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablebiases/bias1biases/bias2biases/bias3biases/bias_outbiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1biases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1biases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1biases_1/bias_outbiases_1/bias_out/Adambiases_1/bias_out/Adam_1train/beta1_powertrain/beta2_powerweights/weight1weights/weight2weights/weight3weights/weight_outweights_1/weight1weights_1/weight1/Adamweights_1/weight1/Adam_1weights_1/weight2weights_1/weight2/Adamweights_1/weight2/Adam_1weights_1/weight3weights_1/weight3/Adamweights_1/weight3/Adam_1weights_1/weight_outweights_1/weight_out/Adamweights_1/weight_out/Adam_1*1
dtypes'
%2#
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
l
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBVariable
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
r
save/RestoreV2_1/tensor_namesConst*!
valueBBbiases/bias1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1Assignbiases/bias1save/RestoreV2_1*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@biases/bias1
r
save/RestoreV2_2/tensor_namesConst*!
valueBBbiases/bias2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assignbiases/bias2save/RestoreV2_2*
use_locking(*
T0*
_class
loc:@biases/bias2*
validate_shape(*
_output_shapes	
:�
r
save/RestoreV2_3/tensor_namesConst*!
valueBBbiases/bias3*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignbiases/bias3save/RestoreV2_3*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@biases/bias3
u
save/RestoreV2_4/tensor_namesConst*
dtype0*
_output_shapes
:*$
valueBBbiases/bias_out
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assignbiases/bias_outsave/RestoreV2_4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@biases/bias_out
t
save/RestoreV2_5/tensor_namesConst*#
valueBBbiases_1/bias1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assignbiases_1/bias1save/RestoreV2_5*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
y
save/RestoreV2_6/tensor_namesConst*(
valueBBbiases_1/bias1/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assignbiases_1/bias1/Adamsave/RestoreV2_6*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
{
save/RestoreV2_7/tensor_namesConst**
value!BBbiases_1/bias1/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assignbiases_1/bias1/Adam_1save/RestoreV2_7*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
t
save/RestoreV2_8/tensor_namesConst*#
valueBBbiases_1/bias2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assignbiases_1/bias2save/RestoreV2_8*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
y
save/RestoreV2_9/tensor_namesConst*(
valueBBbiases_1/bias2/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9Assignbiases_1/bias2/Adamsave/RestoreV2_9*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
|
save/RestoreV2_10/tensor_namesConst**
value!BBbiases_1/bias2/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assignbiases_1/bias2/Adam_1save/RestoreV2_10*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
u
save/RestoreV2_11/tensor_namesConst*#
valueBBbiases_1/bias3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11Assignbiases_1/bias3save/RestoreV2_11*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
z
save/RestoreV2_12/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBbiases_1/bias3/Adam
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assignbiases_1/bias3/Adamsave/RestoreV2_12*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
|
save/RestoreV2_13/tensor_namesConst**
value!BBbiases_1/bias3/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assignbiases_1/bias3/Adam_1save/RestoreV2_13*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
x
save/RestoreV2_14/tensor_namesConst*&
valueBBbiases_1/bias_out*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_14Assignbiases_1/bias_outsave/RestoreV2_14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out
}
save/RestoreV2_15/tensor_namesConst*+
value"B Bbiases_1/bias_out/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15Assignbiases_1/bias_out/Adamsave/RestoreV2_15*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out

save/RestoreV2_16/tensor_namesConst*-
value$B"Bbiases_1/bias_out/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assignbiases_1/bias_out/Adam_1save/RestoreV2_16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out
x
save/RestoreV2_17/tensor_namesConst*&
valueBBtrain/beta1_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assigntrain/beta1_powersave/RestoreV2_17*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@biases_1/bias1
x
save/RestoreV2_18/tensor_namesConst*&
valueBBtrain/beta2_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assigntrain/beta2_powersave/RestoreV2_18*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
v
save/RestoreV2_19/tensor_namesConst*
dtype0*
_output_shapes
:*$
valueBBweights/weight1
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_19Assignweights/weight1save/RestoreV2_19*
use_locking(*
T0*"
_class
loc:@weights/weight1*
validate_shape(* 
_output_shapes
:
��
v
save/RestoreV2_20/tensor_namesConst*$
valueBBweights/weight2*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_20Assignweights/weight2save/RestoreV2_20*
T0*"
_class
loc:@weights/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
v
save/RestoreV2_21/tensor_namesConst*$
valueBBweights/weight3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_21Assignweights/weight3save/RestoreV2_21*
use_locking(*
T0*"
_class
loc:@weights/weight3*
validate_shape(*
_output_shapes
:	�d
y
save/RestoreV2_22/tensor_namesConst*'
valueBBweights/weight_out*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
use_locking(*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d
x
save/RestoreV2_23/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBweights_1/weight1
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_23Assignweights_1/weight1save/RestoreV2_23*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
}
save/RestoreV2_24/tensor_namesConst*+
value"B Bweights_1/weight1/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_24Assignweights_1/weight1/Adamsave/RestoreV2_24*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1

save/RestoreV2_25/tensor_namesConst*-
value$B"Bweights_1/weight1/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_25Assignweights_1/weight1/Adam_1save/RestoreV2_25*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1
x
save/RestoreV2_26/tensor_namesConst*&
valueBBweights_1/weight2*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_26Assignweights_1/weight2save/RestoreV2_26*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��
}
save/RestoreV2_27/tensor_namesConst*+
value"B Bweights_1/weight2/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_27Assignweights_1/weight2/Adamsave/RestoreV2_27*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��

save/RestoreV2_28/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bweights_1/weight2/Adam_1
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_28Assignweights_1/weight2/Adam_1save/RestoreV2_28*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
x
save/RestoreV2_29/tensor_namesConst*&
valueBBweights_1/weight3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_29Assignweights_1/weight3save/RestoreV2_29*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
}
save/RestoreV2_30/tensor_namesConst*+
value"B Bweights_1/weight3/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_30Assignweights_1/weight3/Adamsave/RestoreV2_30*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d

save/RestoreV2_31/tensor_namesConst*-
value$B"Bweights_1/weight3/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_31Assignweights_1/weight3/Adam_1save/RestoreV2_31*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
{
save/RestoreV2_32/tensor_namesConst*)
value BBweights_1/weight_out*
dtype0*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_32Assignweights_1/weight_outsave/RestoreV2_32*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
save/RestoreV2_33/tensor_namesConst*
dtype0*
_output_shapes
:*.
value%B#Bweights_1/weight_out/Adam
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assignweights_1/weight_out/Adamsave/RestoreV2_33*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
save/RestoreV2_34/tensor_namesConst*0
value'B%Bweights_1/weight_out/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_34Assignweights_1/weight_out/Adam_1save/RestoreV2_34*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34
Z
total_loss/tagsConst*
valueB B
total_loss*
dtype0*
_output_shapes
: 
X

total_lossScalarSummarytotal_loss/tags	loss/Mean*
T0*
_output_shapes
: 
X
Mean/reduction_indicesConst*
value	B : *
dtype0*
_output_shapes
: 
k
MeanMeanAbsMean/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
T
error_R/tagsConst*
dtype0*
_output_shapes
: *
valueB Berror_R
V
error_RScalarSummaryerror_R/tagsstrided_slice*
_output_shapes
: *
T0
d
learning_rate_1/tagsConst*
dtype0*
_output_shapes
: * 
valueB Blearning_rate_1
w
learning_rate_1ScalarSummarylearning_rate_1/tagslearning_rate/ExponentialDecay*
_output_shapes
: *
T0
i
Merge/MergeSummaryMergeSummary
total_losserror_Rlearning_rate_1*
N*
_output_shapes
: 
�
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"zI��Ph     6Ԇg	�O6���AJ��
��
+
Abs
x"T
y"T"
Ttype:	
2	
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
5
Pow
x"T
y"T
z"T"
Ttype:
	2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sigmoid
x"T
y"T"
Ttype:	
2
;
SigmoidGrad
x"T
y"T
z"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.3.02
b'unknown'��
y
input/Spectrum-inputPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
t
input/Label-inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
l
weights/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"X  �   
_
weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*weights/random_normal/RandomStandardNormalRandomStandardNormalweights/random_normal/shape*

seed *
T0*
dtype0* 
_output_shapes
:
��*
seed2 
�
weights/random_normal/mulMul*weights/random_normal/RandomStandardNormalweights/random_normal/stddev*
T0* 
_output_shapes
:
��
~
weights/random_normalAddweights/random_normal/mulweights/random_normal/mean*
T0* 
_output_shapes
:
��
�
weights/weight1
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights/weight1/AssignAssignweights/weight1weights/random_normal*
use_locking(*
T0*"
_class
loc:@weights/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights/weight1/readIdentityweights/weight1*
T0*"
_class
loc:@weights/weight1* 
_output_shapes
:
��
n
weights/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   
a
weights/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights/random_normal_1/RandomStandardNormalRandomStandardNormalweights/random_normal_1/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights/random_normal_1/mulMul,weights/random_normal_1/RandomStandardNormalweights/random_normal_1/stddev*
T0* 
_output_shapes
:
��
�
weights/random_normal_1Addweights/random_normal_1/mulweights/random_normal_1/mean* 
_output_shapes
:
��*
T0
�
weights/weight2
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights/weight2/AssignAssignweights/weight2weights/random_normal_1*
T0*"
_class
loc:@weights/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
weights/weight2/readIdentityweights/weight2*
T0*"
_class
loc:@weights/weight2* 
_output_shapes
:
��
n
weights/random_normal_2/shapeConst*
valueB"�   d   *
dtype0*
_output_shapes
:
a
weights/random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
c
weights/random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
,weights/random_normal_2/RandomStandardNormalRandomStandardNormalweights/random_normal_2/shape*
dtype0*
_output_shapes
:	�d*
seed2 *

seed *
T0
�
weights/random_normal_2/mulMul,weights/random_normal_2/RandomStandardNormalweights/random_normal_2/stddev*
T0*
_output_shapes
:	�d
�
weights/random_normal_2Addweights/random_normal_2/mulweights/random_normal_2/mean*
_output_shapes
:	�d*
T0
�
weights/weight3
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
�
weights/weight3/AssignAssignweights/weight3weights/random_normal_2*
use_locking(*
T0*"
_class
loc:@weights/weight3*
validate_shape(*
_output_shapes
:	�d

weights/weight3/readIdentityweights/weight3*
_output_shapes
:	�d*
T0*"
_class
loc:@weights/weight3
n
weights/random_normal_3/shapeConst*
valueB"d      *
dtype0*
_output_shapes
:
a
weights/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
,weights/random_normal_3/RandomStandardNormalRandomStandardNormalweights/random_normal_3/shape*

seed *
T0*
dtype0*
_output_shapes

:d*
seed2 
�
weights/random_normal_3/mulMul,weights/random_normal_3/RandomStandardNormalweights/random_normal_3/stddev*
T0*
_output_shapes

:d
�
weights/random_normal_3Addweights/random_normal_3/mulweights/random_normal_3/mean*
T0*
_output_shapes

:d
�
weights/weight_out
VariableV2*
shape
:d*
shared_name *
dtype0*
_output_shapes

:d*
	container 
�
weights/weight_out/AssignAssignweights/weight_outweights/random_normal_3*
use_locking(*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d
�
weights/weight_out/readIdentityweights/weight_out*
T0*%
_class
loc:@weights/weight_out*
_output_shapes

:d
e
biases/random_normal/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
^
biases/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
`
biases/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)biases/random_normal/RandomStandardNormalRandomStandardNormalbiases/random_normal/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
�
biases/random_normal/mulMul)biases/random_normal/RandomStandardNormalbiases/random_normal/stddev*
T0*
_output_shapes	
:�
v
biases/random_normalAddbiases/random_normal/mulbiases/random_normal/mean*
T0*
_output_shapes	
:�
z
biases/bias1
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
biases/bias1/AssignAssignbiases/bias1biases/random_normal*
T0*
_class
loc:@biases/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
r
biases/bias1/readIdentitybiases/bias1*
T0*
_class
loc:@biases/bias1*
_output_shapes	
:�
g
biases/random_normal_1/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
`
biases/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
b
biases/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+biases/random_normal_1/RandomStandardNormalRandomStandardNormalbiases/random_normal_1/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
�
biases/random_normal_1/mulMul+biases/random_normal_1/RandomStandardNormalbiases/random_normal_1/stddev*
T0*
_output_shapes	
:�
|
biases/random_normal_1Addbiases/random_normal_1/mulbiases/random_normal_1/mean*
T0*
_output_shapes	
:�
z
biases/bias2
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
biases/bias2/AssignAssignbiases/bias2biases/random_normal_1*
T0*
_class
loc:@biases/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
r
biases/bias2/readIdentitybiases/bias2*
T0*
_class
loc:@biases/bias2*
_output_shapes	
:�
f
biases/random_normal_2/shapeConst*
valueB:d*
dtype0*
_output_shapes
:
`
biases/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
biases/random_normal_2/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases/random_normal_2/RandomStandardNormalRandomStandardNormalbiases/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:d*
seed2 *

seed 
�
biases/random_normal_2/mulMul+biases/random_normal_2/RandomStandardNormalbiases/random_normal_2/stddev*
_output_shapes
:d*
T0
{
biases/random_normal_2Addbiases/random_normal_2/mulbiases/random_normal_2/mean*
T0*
_output_shapes
:d
x
biases/bias3
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
�
biases/bias3/AssignAssignbiases/bias3biases/random_normal_2*
use_locking(*
T0*
_class
loc:@biases/bias3*
validate_shape(*
_output_shapes
:d
q
biases/bias3/readIdentitybiases/bias3*
T0*
_class
loc:@biases/bias3*
_output_shapes
:d
f
biases/random_normal_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:
`
biases/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
biases/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases/random_normal_3/RandomStandardNormalRandomStandardNormalbiases/random_normal_3/shape*

seed *
T0*
dtype0*
_output_shapes
:*
seed2 
�
biases/random_normal_3/mulMul+biases/random_normal_3/RandomStandardNormalbiases/random_normal_3/stddev*
_output_shapes
:*
T0
{
biases/random_normal_3Addbiases/random_normal_3/mulbiases/random_normal_3/mean*
T0*
_output_shapes
:
{
biases/bias_out
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
biases/bias_out/AssignAssignbiases/bias_outbiases/random_normal_3*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
z
biases/bias_out/readIdentitybiases/bias_out*
T0*"
_class
loc:@biases/bias_out*
_output_shapes
:
n
weights_1/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"X  �   
a
weights_1/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
weights_1/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights_1/random_normal/RandomStandardNormalRandomStandardNormalweights_1/random_normal/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights_1/random_normal/mulMul,weights_1/random_normal/RandomStandardNormalweights_1/random_normal/stddev* 
_output_shapes
:
��*
T0
�
weights_1/random_normalAddweights_1/random_normal/mulweights_1/random_normal/mean*
T0* 
_output_shapes
:
��
�
weights_1/weight1
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights_1/weight1/AssignAssignweights_1/weight1weights_1/random_normal*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight1/readIdentityweights_1/weight1*
T0*$
_class
loc:@weights_1/weight1* 
_output_shapes
:
��
p
weights_1/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   
c
weights_1/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 weights_1/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
.weights_1/random_normal_1/RandomStandardNormalRandomStandardNormalweights_1/random_normal_1/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
weights_1/random_normal_1/mulMul.weights_1/random_normal_1/RandomStandardNormal weights_1/random_normal_1/stddev*
T0* 
_output_shapes
:
��
�
weights_1/random_normal_1Addweights_1/random_normal_1/mulweights_1/random_normal_1/mean*
T0* 
_output_shapes
:
��
�
weights_1/weight2
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
weights_1/weight2/AssignAssignweights_1/weight2weights_1/random_normal_1*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
weights_1/weight2/readIdentityweights_1/weight2*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
p
weights_1/random_normal_2/shapeConst*
valueB"�   d   *
dtype0*
_output_shapes
:
c
weights_1/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 weights_1/random_normal_2/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
.weights_1/random_normal_2/RandomStandardNormalRandomStandardNormalweights_1/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2 *

seed 
�
weights_1/random_normal_2/mulMul.weights_1/random_normal_2/RandomStandardNormal weights_1/random_normal_2/stddev*
T0*
_output_shapes
:	�d
�
weights_1/random_normal_2Addweights_1/random_normal_2/mulweights_1/random_normal_2/mean*
T0*
_output_shapes
:	�d
�
weights_1/weight3
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
�
weights_1/weight3/AssignAssignweights_1/weight3weights_1/random_normal_2*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
weights_1/weight3/readIdentityweights_1/weight3*
_output_shapes
:	�d*
T0*$
_class
loc:@weights_1/weight3
p
weights_1/random_normal_3/shapeConst*
valueB"d      *
dtype0*
_output_shapes
:
c
weights_1/random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
e
 weights_1/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
.weights_1/random_normal_3/RandomStandardNormalRandomStandardNormalweights_1/random_normal_3/shape*
T0*
dtype0*
_output_shapes

:d*
seed2 *

seed 
�
weights_1/random_normal_3/mulMul.weights_1/random_normal_3/RandomStandardNormal weights_1/random_normal_3/stddev*
T0*
_output_shapes

:d
�
weights_1/random_normal_3Addweights_1/random_normal_3/mulweights_1/random_normal_3/mean*
T0*
_output_shapes

:d
�
weights_1/weight_out
VariableV2*
shape
:d*
shared_name *
dtype0*
_output_shapes

:d*
	container 
�
weights_1/weight_out/AssignAssignweights_1/weight_outweights_1/random_normal_3*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
weights_1/weight_out/readIdentityweights_1/weight_out*
_output_shapes

:d*
T0*'
_class
loc:@weights_1/weight_out
g
biases_1/random_normal/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
`
biases_1/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
b
biases_1/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases_1/random_normal/RandomStandardNormalRandomStandardNormalbiases_1/random_normal/shape*
dtype0*
_output_shapes	
:�*
seed2 *

seed *
T0
�
biases_1/random_normal/mulMul+biases_1/random_normal/RandomStandardNormalbiases_1/random_normal/stddev*
_output_shapes	
:�*
T0
|
biases_1/random_normalAddbiases_1/random_normal/mulbiases_1/random_normal/mean*
T0*
_output_shapes	
:�
|
biases_1/bias1
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
biases_1/bias1/AssignAssignbiases_1/bias1biases_1/random_normal*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
x
biases_1/bias1/readIdentitybiases_1/bias1*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes	
:�
i
biases_1/random_normal_1/shapeConst*
valueB:�*
dtype0*
_output_shapes
:
b
biases_1/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
biases_1/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-biases_1/random_normal_1/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_1/shape*
dtype0*
_output_shapes	
:�*
seed2 *

seed *
T0
�
biases_1/random_normal_1/mulMul-biases_1/random_normal_1/RandomStandardNormalbiases_1/random_normal_1/stddev*
T0*
_output_shapes	
:�
�
biases_1/random_normal_1Addbiases_1/random_normal_1/mulbiases_1/random_normal_1/mean*
T0*
_output_shapes	
:�
|
biases_1/bias2
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
biases_1/bias2/AssignAssignbiases_1/bias2biases_1/random_normal_1*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
x
biases_1/bias2/readIdentitybiases_1/bias2*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
h
biases_1/random_normal_2/shapeConst*
dtype0*
_output_shapes
:*
valueB:d
b
biases_1/random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
biases_1/random_normal_2/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-biases_1/random_normal_2/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:d*
seed2 *

seed 
�
biases_1/random_normal_2/mulMul-biases_1/random_normal_2/RandomStandardNormalbiases_1/random_normal_2/stddev*
T0*
_output_shapes
:d
�
biases_1/random_normal_2Addbiases_1/random_normal_2/mulbiases_1/random_normal_2/mean*
T0*
_output_shapes
:d
z
biases_1/bias3
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
�
biases_1/bias3/AssignAssignbiases_1/bias3biases_1/random_normal_2*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
w
biases_1/bias3/readIdentitybiases_1/bias3*
T0*!
_class
loc:@biases_1/bias3*
_output_shapes
:d
h
biases_1/random_normal_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:
b
biases_1/random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
biases_1/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-biases_1/random_normal_3/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_3/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
�
biases_1/random_normal_3/mulMul-biases_1/random_normal_3/RandomStandardNormalbiases_1/random_normal_3/stddev*
_output_shapes
:*
T0
�
biases_1/random_normal_3Addbiases_1/random_normal_3/mulbiases_1/random_normal_3/mean*
T0*
_output_shapes
:
}
biases_1/bias_out
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
biases_1/bias_out/AssignAssignbiases_1/bias_outbiases_1/random_normal_3*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
�
biases_1/bias_out/readIdentitybiases_1/bias_out*
_output_shapes
:*
T0*$
_class
loc:@biases_1/bias_out
�
layer_1/MatMulMatMulinput/Spectrum-inputweights_1/weight1/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
j
layer_1/AddAddlayer_1/MatMulbiases_1/bias1/read*
T0*(
_output_shapes
:����������
T
layer_2/ReluRelulayer_1/Add*
T0*(
_output_shapes
:����������
�
layer_2/MatMulMatMullayer_2/Reluweights_1/weight2/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
j
layer_2/AddAddlayer_2/MatMulbiases_1/bias2/read*
T0*(
_output_shapes
:����������
Z
layer_3/SigmoidSigmoidlayer_2/Add*
T0*(
_output_shapes
:����������
�
layer_3/MatMulMatMullayer_3/Sigmoidweights_1/weight3/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
i
layer_3/AddAddlayer_3/MatMulbiases_1/bias3/read*
T0*'
_output_shapes
:���������d
R
result/ReluRelulayer_3/Add*'
_output_shapes
:���������d*
T0
�
result/MatMulMatMulresult/Reluweights_1/weight_out/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j

result/AddAddresult/MatMulbiases_1/bias_out/read*
T0*'
_output_shapes
:���������
[
subSub
result/Addinput/Label-input*
T0*'
_output_shapes
:���������
A
AbsAbssub*
T0*'
_output_shapes
:���������
]
sub_1Sub
result/Addinput/Label-input*
T0*'
_output_shapes
:���������
I
SquareSquaresub_1*'
_output_shapes
:���������*
T0
X
Variable/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
l
Variable
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Variable/AssignAssignVariableVariable/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
a
Variable/readIdentityVariable*
_output_shapes
: *
T0*
_class
loc:@Variable
q
,learning_rate/ExponentialDecay/learning_rateConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
j
#learning_rate/ExponentialDecay/CastCastVariable/read*

SrcT0*
_output_shapes
: *

DstT0
j
'learning_rate/ExponentialDecay/Cast_1/xConst*
dtype0*
_output_shapes
: *
value
B :�'
�
%learning_rate/ExponentialDecay/Cast_1Cast'learning_rate/ExponentialDecay/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
l
'learning_rate/ExponentialDecay/Cast_2/xConst*
valueB
 *��u?*
dtype0*
_output_shapes
: 
�
&learning_rate/ExponentialDecay/truedivRealDiv#learning_rate/ExponentialDecay/Cast%learning_rate/ExponentialDecay/Cast_1*
T0*
_output_shapes
: 
v
$learning_rate/ExponentialDecay/FloorFloor&learning_rate/ExponentialDecay/truediv*
T0*
_output_shapes
: 
�
"learning_rate/ExponentialDecay/PowPow'learning_rate/ExponentialDecay/Cast_2/x$learning_rate/ExponentialDecay/Floor*
_output_shapes
: *
T0
�
learning_rate/ExponentialDecayMul,learning_rate/ExponentialDecay/learning_rate"learning_rate/ExponentialDecay/Pow*
T0*
_output_shapes
: 
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
c
	loss/MeanMeanSquare
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
_output_shapes
: *
T0
}
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
j
$train/gradients/loss/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
l
&train/gradients/loss/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*'
_output_shapes
:���������*
T0
�
!train/gradients/Square_grad/mul/xConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
train/gradients/Square_grad/mulMul!train/gradients/Square_grad/mul/xsub_1*
T0*'
_output_shapes
:���������
�
!train/gradients/Square_grad/mul_1Mul&train/gradients/loss/Mean_grad/truedivtrain/gradients/Square_grad/mul*'
_output_shapes
:���������*
T0
j
 train/gradients/sub_1_grad/ShapeShape
result/Add*
T0*
out_type0*
_output_shapes
:
s
"train/gradients/sub_1_grad/Shape_1Shapeinput/Label-input*
T0*
out_type0*
_output_shapes
:
�
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"train/gradients/sub_1_grad/ReshapeReshapetrain/gradients/sub_1_grad/Sum train/gradients/sub_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
 train/gradients/sub_1_grad/Sum_1Sum!train/gradients/Square_grad/mul_12train/gradients/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
j
train/gradients/sub_1_grad/NegNeg train/gradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:
�
$train/gradients/sub_1_grad/Reshape_1Reshapetrain/gradients/sub_1_grad/Neg"train/gradients/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

+train/gradients/sub_1_grad/tuple/group_depsNoOp#^train/gradients/sub_1_grad/Reshape%^train/gradients/sub_1_grad/Reshape_1
�
3train/gradients/sub_1_grad/tuple/control_dependencyIdentity"train/gradients/sub_1_grad/Reshape,^train/gradients/sub_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@train/gradients/sub_1_grad/Reshape*'
_output_shapes
:���������
�
5train/gradients/sub_1_grad/tuple/control_dependency_1Identity$train/gradients/sub_1_grad/Reshape_1,^train/gradients/sub_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@train/gradients/sub_1_grad/Reshape_1*'
_output_shapes
:���������
r
%train/gradients/result/Add_grad/ShapeShaperesult/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/result/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
5train/gradients/result/Add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/result/Add_grad/Shape'train/gradients/result/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/result/Add_grad/Reshape_1Reshape%train/gradients/result/Add_grad/Sum_1'train/gradients/result/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/result/Add_grad/tuple/group_depsNoOp(^train/gradients/result/Add_grad/Reshape*^train/gradients/result/Add_grad/Reshape_1
�
8train/gradients/result/Add_grad/tuple/control_dependencyIdentity'train/gradients/result/Add_grad/Reshape1^train/gradients/result/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/result/Add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/result/Add_grad/tuple/control_dependency_1Identity)train/gradients/result/Add_grad/Reshape_11^train/gradients/result/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/result/Add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/result/MatMul_grad/MatMulMatMul8train/gradients/result/Add_grad/tuple/control_dependencyweights_1/weight_out/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
+train/gradients/result/MatMul_grad/MatMul_1MatMulresult/Relu8train/gradients/result/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d*
transpose_a(
�
3train/gradients/result/MatMul_grad/tuple/group_depsNoOp*^train/gradients/result/MatMul_grad/MatMul,^train/gradients/result/MatMul_grad/MatMul_1
�
;train/gradients/result/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/result/MatMul_grad/MatMul4^train/gradients/result/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/result/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
=train/gradients/result/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/result/MatMul_grad/MatMul_14^train/gradients/result/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/result/MatMul_grad/MatMul_1*
_output_shapes

:d
�
)train/gradients/result/Relu_grad/ReluGradReluGrad;train/gradients/result/MatMul_grad/tuple/control_dependencyresult/Relu*
T0*'
_output_shapes
:���������d
t
&train/gradients/layer_3/Add_grad/ShapeShapelayer_3/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_3/Add_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/Add_grad/Shape(train/gradients/layer_3/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_3/Add_grad/SumSum)train/gradients/result/Relu_grad/ReluGrad6train/gradients/layer_3/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_3/Add_grad/ReshapeReshape$train/gradients/layer_3/Add_grad/Sum&train/gradients/layer_3/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
&train/gradients/layer_3/Add_grad/Sum_1Sum)train/gradients/result/Relu_grad/ReluGrad8train/gradients/layer_3/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_3/Add_grad/Reshape_1Reshape&train/gradients/layer_3/Add_grad/Sum_1(train/gradients/layer_3/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
1train/gradients/layer_3/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/Add_grad/Reshape+^train/gradients/layer_3/Add_grad/Reshape_1
�
9train/gradients/layer_3/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/Add_grad/Reshape2^train/gradients/layer_3/Add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/Add_grad/Reshape*'
_output_shapes
:���������d
�
;train/gradients/layer_3/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/Add_grad/Reshape_12^train/gradients/layer_3/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/Add_grad/Reshape_1*
_output_shapes
:d
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/Add_grad/tuple/control_dependencyweights_1/weight3/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_3/Sigmoid9train/gradients/layer_3/Add_grad/tuple/control_dependency*
T0*
_output_shapes
:	�d*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
0train/gradients/layer_3/Sigmoid_grad/SigmoidGradSigmoidGradlayer_3/Sigmoid<train/gradients/layer_3/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
t
&train/gradients/layer_2/Add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
s
(train/gradients/layer_2/Add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/Add_grad/Shape(train/gradients/layer_2/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/Add_grad/Reshape_1Reshape&train/gradients/layer_2/Add_grad/Sum_1(train/gradients/layer_2/Add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
1train/gradients/layer_2/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/Add_grad/Reshape+^train/gradients/layer_2/Add_grad/Reshape_1
�
9train/gradients/layer_2/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/Add_grad/Reshape2^train/gradients/layer_2/Add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/Add_grad/Reshape*(
_output_shapes
:����������
�
;train/gradients/layer_2/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/Add_grad/Reshape_12^train/gradients/layer_2/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/Add_grad/Reshape_1*
_output_shapes	
:�
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/Add_grad/tuple/control_dependencyweights_1/weight2/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_2/Add_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_2/Relu*(
_output_shapes
:����������*
T0
t
&train/gradients/layer_1/Add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
s
(train/gradients/layer_1/Add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/Add_grad/Shape(train/gradients/layer_1/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/Add_grad/ReshapeReshape$train/gradients/layer_1/Add_grad/Sum&train/gradients/layer_1/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_1/Add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_1/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_1/Add_grad/Reshape_1Reshape&train/gradients/layer_1/Add_grad/Sum_1(train/gradients/layer_1/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
1train/gradients/layer_1/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/Add_grad/Reshape+^train/gradients/layer_1/Add_grad/Reshape_1
�
9train/gradients/layer_1/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/Add_grad/Reshape2^train/gradients/layer_1/Add_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*;
_class1
/-loc:@train/gradients/layer_1/Add_grad/Reshape
�
;train/gradients/layer_1/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/Add_grad/Reshape_12^train/gradients/layer_1/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/Add_grad/Reshape_1*
_output_shapes	
:�
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/Add_grad/tuple/control_dependencyweights_1/weight1/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Spectrum-input9train/gradients/layer_1/Add_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
train/beta1_power/initial_valueConst*
valueB
 *fff?*!
_class
loc:@biases_1/bias1*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *!
_class
loc:@biases_1/bias1
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@biases_1/bias1
y
train/beta1_power/readIdentitytrain/beta1_power*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
valueB
 *w�?*!
_class
loc:@biases_1/bias1*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
y
train/beta2_power/readIdentitytrain/beta2_power*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
(weights_1/weight1/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
��*$
_class
loc:@weights_1/weight1*
valueB
��*    
�
weights_1/weight1/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight1*
	container *
shape:
��
�
weights_1/weight1/Adam/AssignAssignweights_1/weight1/Adam(weights_1/weight1/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight1/Adam/readIdentityweights_1/weight1/Adam* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight1
�
*weights_1/weight1/Adam_1/Initializer/zerosConst*
dtype0* 
_output_shapes
:
��*$
_class
loc:@weights_1/weight1*
valueB
��*    
�
weights_1/weight1/Adam_1
VariableV2*
shared_name *$
_class
loc:@weights_1/weight1*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
weights_1/weight1/Adam_1/AssignAssignweights_1/weight1/Adam_1*weights_1/weight1/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight1/Adam_1/readIdentityweights_1/weight1/Adam_1*
T0*$
_class
loc:@weights_1/weight1* 
_output_shapes
:
��
�
(weights_1/weight2/Adam/Initializer/zerosConst*$
_class
loc:@weights_1/weight2*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
weights_1/weight2/Adam
VariableV2*
shared_name *$
_class
loc:@weights_1/weight2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
weights_1/weight2/Adam/AssignAssignweights_1/weight2/Adam(weights_1/weight2/Adam/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
weights_1/weight2/Adam/readIdentityweights_1/weight2/Adam* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight2
�
*weights_1/weight2/Adam_1/Initializer/zerosConst*$
_class
loc:@weights_1/weight2*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
weights_1/weight2/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight2*
	container *
shape:
��
�
weights_1/weight2/Adam_1/AssignAssignweights_1/weight2/Adam_1*weights_1/weight2/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
�
weights_1/weight2/Adam_1/readIdentityweights_1/weight2/Adam_1*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
�
(weights_1/weight3/Adam/Initializer/zerosConst*$
_class
loc:@weights_1/weight3*
valueB	�d*    *
dtype0*
_output_shapes
:	�d
�
weights_1/weight3/Adam
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *$
_class
loc:@weights_1/weight3*
	container *
shape:	�d
�
weights_1/weight3/Adam/AssignAssignweights_1/weight3/Adam(weights_1/weight3/Adam/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
weights_1/weight3/Adam/readIdentityweights_1/weight3/Adam*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
�
*weights_1/weight3/Adam_1/Initializer/zerosConst*$
_class
loc:@weights_1/weight3*
valueB	�d*    *
dtype0*
_output_shapes
:	�d
�
weights_1/weight3/Adam_1
VariableV2*
shared_name *$
_class
loc:@weights_1/weight3*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
weights_1/weight3/Adam_1/AssignAssignweights_1/weight3/Adam_1*weights_1/weight3/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
�
weights_1/weight3/Adam_1/readIdentityweights_1/weight3/Adam_1*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
�
+weights_1/weight_out/Adam/Initializer/zerosConst*'
_class
loc:@weights_1/weight_out*
valueBd*    *
dtype0*
_output_shapes

:d
�
weights_1/weight_out/Adam
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *'
_class
loc:@weights_1/weight_out*
	container *
shape
:d
�
 weights_1/weight_out/Adam/AssignAssignweights_1/weight_out/Adam+weights_1/weight_out/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
�
weights_1/weight_out/Adam/readIdentityweights_1/weight_out/Adam*
T0*'
_class
loc:@weights_1/weight_out*
_output_shapes

:d
�
-weights_1/weight_out/Adam_1/Initializer/zerosConst*'
_class
loc:@weights_1/weight_out*
valueBd*    *
dtype0*
_output_shapes

:d
�
weights_1/weight_out/Adam_1
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *'
_class
loc:@weights_1/weight_out
�
"weights_1/weight_out/Adam_1/AssignAssignweights_1/weight_out/Adam_1-weights_1/weight_out/Adam_1/Initializer/zeros*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
�
 weights_1/weight_out/Adam_1/readIdentityweights_1/weight_out/Adam_1*
_output_shapes

:d*
T0*'
_class
loc:@weights_1/weight_out
�
%biases_1/bias1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias1*
valueB�*    
�
biases_1/bias1/Adam
VariableV2*!
_class
loc:@biases_1/bias1*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
biases_1/bias1/Adam/AssignAssignbiases_1/bias1/Adam%biases_1/bias1/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias1/Adam/readIdentitybiases_1/bias1/Adam*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
�
'biases_1/bias1/Adam_1/Initializer/zerosConst*!
_class
loc:@biases_1/bias1*
valueB�*    *
dtype0*
_output_shapes	
:�
�
biases_1/bias1/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape:�
�
biases_1/bias1/Adam_1/AssignAssignbiases_1/bias1/Adam_1'biases_1/bias1/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias1/Adam_1/readIdentitybiases_1/bias1/Adam_1*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes	
:�
�
%biases_1/bias2/Adam/Initializer/zerosConst*!
_class
loc:@biases_1/bias2*
valueB�*    *
dtype0*
_output_shapes	
:�
�
biases_1/bias2/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias2*
	container *
shape:�
�
biases_1/bias2/Adam/AssignAssignbiases_1/bias2/Adam%biases_1/bias2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias2
�
biases_1/bias2/Adam/readIdentitybiases_1/bias2/Adam*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
�
'biases_1/bias2/Adam_1/Initializer/zerosConst*!
_class
loc:@biases_1/bias2*
valueB�*    *
dtype0*
_output_shapes	
:�
�
biases_1/bias2/Adam_1
VariableV2*!
_class
loc:@biases_1/bias2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
biases_1/bias2/Adam_1/AssignAssignbiases_1/bias2/Adam_1'biases_1/bias2/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias2/Adam_1/readIdentitybiases_1/bias2/Adam_1*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
�
%biases_1/bias3/Adam/Initializer/zerosConst*!
_class
loc:@biases_1/bias3*
valueBd*    *
dtype0*
_output_shapes
:d
�
biases_1/bias3/Adam
VariableV2*
shared_name *!
_class
loc:@biases_1/bias3*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
biases_1/bias3/Adam/AssignAssignbiases_1/bias3/Adam%biases_1/bias3/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
�
biases_1/bias3/Adam/readIdentitybiases_1/bias3/Adam*
T0*!
_class
loc:@biases_1/bias3*
_output_shapes
:d
�
'biases_1/bias3/Adam_1/Initializer/zerosConst*!
_class
loc:@biases_1/bias3*
valueBd*    *
dtype0*
_output_shapes
:d
�
biases_1/bias3/Adam_1
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name *!
_class
loc:@biases_1/bias3
�
biases_1/bias3/Adam_1/AssignAssignbiases_1/bias3/Adam_1'biases_1/bias3/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
�
biases_1/bias3/Adam_1/readIdentitybiases_1/bias3/Adam_1*
T0*!
_class
loc:@biases_1/bias3*
_output_shapes
:d
�
(biases_1/bias_out/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@biases_1/bias_out*
valueB*    
�
biases_1/bias_out/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out*
	container 
�
biases_1/bias_out/Adam/AssignAssignbiases_1/bias_out/Adam(biases_1/bias_out/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out
�
biases_1/bias_out/Adam/readIdentitybiases_1/bias_out/Adam*
T0*$
_class
loc:@biases_1/bias_out*
_output_shapes
:
�
*biases_1/bias_out/Adam_1/Initializer/zerosConst*$
_class
loc:@biases_1/bias_out*
valueB*    *
dtype0*
_output_shapes
:
�
biases_1/bias_out/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out
�
biases_1/bias_out/Adam_1/AssignAssignbiases_1/bias_out/Adam_1*biases_1/bias_out/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out
�
biases_1/bias_out/Adam_1/readIdentitybiases_1/bias_out/Adam_1*
T0*$
_class
loc:@biases_1/bias_out*
_output_shapes
:
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
-train/Adam/update_weights_1/weight1/ApplyAdam	ApplyAdamweights_1/weight1weights_1/weight1/Adamweights_1/weight1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@weights_1/weight1*
use_nesterov( * 
_output_shapes
:
��
�
-train/Adam/update_weights_1/weight2/ApplyAdam	ApplyAdamweights_1/weight2weights_1/weight2/Adamweights_1/weight2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@weights_1/weight2*
use_nesterov( * 
_output_shapes
:
��
�
-train/Adam/update_weights_1/weight3/ApplyAdam	ApplyAdamweights_1/weight3weights_1/weight3/Adamweights_1/weight3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@weights_1/weight3*
use_nesterov( *
_output_shapes
:	�d
�
0train/Adam/update_weights_1/weight_out/ApplyAdam	ApplyAdamweights_1/weight_outweights_1/weight_out/Adamweights_1/weight_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/result/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@weights_1/weight_out*
use_nesterov( *
_output_shapes

:d
�
*train/Adam/update_biases_1/bias1/ApplyAdam	ApplyAdambiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@biases_1/bias1
�
*train/Adam/update_biases_1/bias2/ApplyAdam	ApplyAdambiases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias2*
use_nesterov( *
_output_shapes	
:�
�
*train/Adam/update_biases_1/bias3/ApplyAdam	ApplyAdambiases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:d*
use_locking( *
T0*!
_class
loc:@biases_1/bias3
�
-train/Adam/update_biases_1/bias_out/ApplyAdam	ApplyAdambiases_1/bias_outbiases_1/bias_out/Adambiases_1/bias_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/result/Add_grad/tuple/control_dependency_1*
T0*$
_class
loc:@biases_1/bias_out*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
�
train/Adam/updateNoOp.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
�
train/Adam/valueConst^train/Adam/update*
value	B :*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�

train/Adam	AssignAddVariabletrain/Adam/value*
T0*
_class
loc:@Variable*
_output_shapes
: *
use_locking( 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�#BVariableBbiases/bias1Bbiases/bias2Bbiases/bias3Bbiases/bias_outBbiases_1/bias1Bbiases_1/bias1/AdamBbiases_1/bias1/Adam_1Bbiases_1/bias2Bbiases_1/bias2/AdamBbiases_1/bias2/Adam_1Bbiases_1/bias3Bbiases_1/bias3/AdamBbiases_1/bias3/Adam_1Bbiases_1/bias_outBbiases_1/bias_out/AdamBbiases_1/bias_out/Adam_1Btrain/beta1_powerBtrain/beta2_powerBweights/weight1Bweights/weight2Bweights/weight3Bweights/weight_outBweights_1/weight1Bweights_1/weight1/AdamBweights_1/weight1/Adam_1Bweights_1/weight2Bweights_1/weight2/AdamBweights_1/weight2/Adam_1Bweights_1/weight3Bweights_1/weight3/AdamBweights_1/weight3/Adam_1Bweights_1/weight_outBweights_1/weight_out/AdamBweights_1/weight_out/Adam_1*
dtype0*
_output_shapes
:#
�
save/SaveV2/shape_and_slicesConst*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:#
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablebiases/bias1biases/bias2biases/bias3biases/bias_outbiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1biases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1biases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1biases_1/bias_outbiases_1/bias_out/Adambiases_1/bias_out/Adam_1train/beta1_powertrain/beta2_powerweights/weight1weights/weight2weights/weight3weights/weight_outweights_1/weight1weights_1/weight1/Adamweights_1/weight1/Adam_1weights_1/weight2weights_1/weight2/Adamweights_1/weight2/Adam_1weights_1/weight3weights_1/weight3/Adamweights_1/weight3/Adam_1weights_1/weight_outweights_1/weight_out/Adamweights_1/weight_out/Adam_1*1
dtypes'
%2#
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
l
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
r
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBbiases/bias1
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1Assignbiases/bias1save/RestoreV2_1*
use_locking(*
T0*
_class
loc:@biases/bias1*
validate_shape(*
_output_shapes	
:�
r
save/RestoreV2_2/tensor_namesConst*!
valueBBbiases/bias2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assignbiases/bias2save/RestoreV2_2*
T0*
_class
loc:@biases/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
r
save/RestoreV2_3/tensor_namesConst*!
valueBBbiases/bias3*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignbiases/bias3save/RestoreV2_3*
use_locking(*
T0*
_class
loc:@biases/bias3*
validate_shape(*
_output_shapes
:d
u
save/RestoreV2_4/tensor_namesConst*$
valueBBbiases/bias_out*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assignbiases/bias_outsave/RestoreV2_4*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
t
save/RestoreV2_5/tensor_namesConst*#
valueBBbiases_1/bias1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assignbiases_1/bias1save/RestoreV2_5*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
y
save/RestoreV2_6/tensor_namesConst*(
valueBBbiases_1/bias1/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assignbiases_1/bias1/Adamsave/RestoreV2_6*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
{
save/RestoreV2_7/tensor_namesConst**
value!BBbiases_1/bias1/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assignbiases_1/bias1/Adam_1save/RestoreV2_7*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
t
save/RestoreV2_8/tensor_namesConst*#
valueBBbiases_1/bias2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assignbiases_1/bias2save/RestoreV2_8*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
y
save/RestoreV2_9/tensor_namesConst*(
valueBBbiases_1/bias2/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assignbiases_1/bias2/Adamsave/RestoreV2_9*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
|
save/RestoreV2_10/tensor_namesConst*
dtype0*
_output_shapes
:**
value!BBbiases_1/bias2/Adam_1
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assignbiases_1/bias2/Adam_1save/RestoreV2_10*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
u
save/RestoreV2_11/tensor_namesConst*#
valueBBbiases_1/bias3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11Assignbiases_1/bias3save/RestoreV2_11*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
z
save/RestoreV2_12/tensor_namesConst*(
valueBBbiases_1/bias3/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assignbiases_1/bias3/Adamsave/RestoreV2_12*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
|
save/RestoreV2_13/tensor_namesConst**
value!BBbiases_1/bias3/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assignbiases_1/bias3/Adam_1save/RestoreV2_13*
use_locking(*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d
x
save/RestoreV2_14/tensor_namesConst*&
valueBBbiases_1/bias_out*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_14Assignbiases_1/bias_outsave/RestoreV2_14*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
}
save/RestoreV2_15/tensor_namesConst*+
value"B Bbiases_1/bias_out/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15Assignbiases_1/bias_out/Adamsave/RestoreV2_15*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(

save/RestoreV2_16/tensor_namesConst*-
value$B"Bbiases_1/bias_out/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assignbiases_1/bias_out/Adam_1save/RestoreV2_16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out
x
save/RestoreV2_17/tensor_namesConst*&
valueBBtrain/beta1_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_17Assigntrain/beta1_powersave/RestoreV2_17*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
x
save/RestoreV2_18/tensor_namesConst*&
valueBBtrain/beta2_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_18Assigntrain/beta2_powersave/RestoreV2_18*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking(
v
save/RestoreV2_19/tensor_namesConst*$
valueBBweights/weight1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assignweights/weight1save/RestoreV2_19*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*"
_class
loc:@weights/weight1
v
save/RestoreV2_20/tensor_namesConst*$
valueBBweights/weight2*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assignweights/weight2save/RestoreV2_20*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*"
_class
loc:@weights/weight2
v
save/RestoreV2_21/tensor_namesConst*$
valueBBweights/weight3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_21Assignweights/weight3save/RestoreV2_21*
T0*"
_class
loc:@weights/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
y
save/RestoreV2_22/tensor_namesConst*'
valueBBweights/weight_out*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
use_locking(*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d
x
save/RestoreV2_23/tensor_namesConst*&
valueBBweights_1/weight1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_23Assignweights_1/weight1save/RestoreV2_23*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
}
save/RestoreV2_24/tensor_namesConst*+
value"B Bweights_1/weight1/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_24Assignweights_1/weight1/Adamsave/RestoreV2_24*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��

save/RestoreV2_25/tensor_namesConst*-
value$B"Bweights_1/weight1/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_25Assignweights_1/weight1/Adam_1save/RestoreV2_25*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
x
save/RestoreV2_26/tensor_namesConst*&
valueBBweights_1/weight2*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_26Assignweights_1/weight2save/RestoreV2_26*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��
}
save/RestoreV2_27/tensor_namesConst*+
value"B Bweights_1/weight2/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_27Assignweights_1/weight2/Adamsave/RestoreV2_27*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��

save/RestoreV2_28/tensor_namesConst*-
value$B"Bweights_1/weight2/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_28Assignweights_1/weight2/Adam_1save/RestoreV2_28*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
x
save/RestoreV2_29/tensor_namesConst*&
valueBBweights_1/weight3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_29Assignweights_1/weight3save/RestoreV2_29*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
}
save/RestoreV2_30/tensor_namesConst*+
value"B Bweights_1/weight3/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_30Assignweights_1/weight3/Adamsave/RestoreV2_30*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d

save/RestoreV2_31/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bweights_1/weight3/Adam_1
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_31Assignweights_1/weight3/Adam_1save/RestoreV2_31*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
{
save/RestoreV2_32/tensor_namesConst*
dtype0*
_output_shapes
:*)
value BBweights_1/weight_out
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_32Assignweights_1/weight_outsave/RestoreV2_32*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
save/RestoreV2_33/tensor_namesConst*.
value%B#Bweights_1/weight_out/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_33Assignweights_1/weight_out/Adamsave/RestoreV2_33*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
�
save/RestoreV2_34/tensor_namesConst*0
value'B%Bweights_1/weight_out/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assignweights_1/weight_out/Adam_1save/RestoreV2_34*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34
Z
total_loss/tagsConst*
valueB B
total_loss*
dtype0*
_output_shapes
: 
X

total_lossScalarSummarytotal_loss/tags	loss/Mean*
T0*
_output_shapes
: 
X
Mean/reduction_indicesConst*
value	B : *
dtype0*
_output_shapes
: 
k
MeanMeanAbsMean/reduction_indices*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
T
error_R/tagsConst*
valueB Berror_R*
dtype0*
_output_shapes
: 
V
error_RScalarSummaryerror_R/tagsstrided_slice*
_output_shapes
: *
T0
d
learning_rate_1/tagsConst*
dtype0*
_output_shapes
: * 
valueB Blearning_rate_1
w
learning_rate_1ScalarSummarylearning_rate_1/tagslearning_rate/ExponentialDecay*
T0*
_output_shapes
: 
i
Merge/MergeSummaryMergeSummary
total_losserror_Rlearning_rate_1*
N*
_output_shapes
: 
�
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"";
	summaries.
,
total_loss:0
	error_R:0
learning_rate_1:0"�
trainable_variables��
C
weights/weight1:0weights/weight1/Assignweights/weight1/read:0
C
weights/weight2:0weights/weight2/Assignweights/weight2/read:0
C
weights/weight3:0weights/weight3/Assignweights/weight3/read:0
L
weights/weight_out:0weights/weight_out/Assignweights/weight_out/read:0
:
biases/bias1:0biases/bias1/Assignbiases/bias1/read:0
:
biases/bias2:0biases/bias2/Assignbiases/bias2/read:0
:
biases/bias3:0biases/bias3/Assignbiases/bias3/read:0
C
biases/bias_out:0biases/bias_out/Assignbiases/bias_out/read:0
I
weights_1/weight1:0weights_1/weight1/Assignweights_1/weight1/read:0
I
weights_1/weight2:0weights_1/weight2/Assignweights_1/weight2/read:0
I
weights_1/weight3:0weights_1/weight3/Assignweights_1/weight3/read:0
R
weights_1/weight_out:0weights_1/weight_out/Assignweights_1/weight_out/read:0
@
biases_1/bias1:0biases_1/bias1/Assignbiases_1/bias1/read:0
@
biases_1/bias2:0biases_1/bias2/Assignbiases_1/bias2/read:0
@
biases_1/bias3:0biases_1/bias3/Assignbiases_1/bias3/read:0
I
biases_1/bias_out:0biases_1/bias_out/Assignbiases_1/bias_out/read:0"
train_op


train/Adam"�
	variables��
C
weights/weight1:0weights/weight1/Assignweights/weight1/read:0
C
weights/weight2:0weights/weight2/Assignweights/weight2/read:0
C
weights/weight3:0weights/weight3/Assignweights/weight3/read:0
L
weights/weight_out:0weights/weight_out/Assignweights/weight_out/read:0
:
biases/bias1:0biases/bias1/Assignbiases/bias1/read:0
:
biases/bias2:0biases/bias2/Assignbiases/bias2/read:0
:
biases/bias3:0biases/bias3/Assignbiases/bias3/read:0
C
biases/bias_out:0biases/bias_out/Assignbiases/bias_out/read:0
I
weights_1/weight1:0weights_1/weight1/Assignweights_1/weight1/read:0
I
weights_1/weight2:0weights_1/weight2/Assignweights_1/weight2/read:0
I
weights_1/weight3:0weights_1/weight3/Assignweights_1/weight3/read:0
R
weights_1/weight_out:0weights_1/weight_out/Assignweights_1/weight_out/read:0
@
biases_1/bias1:0biases_1/bias1/Assignbiases_1/bias1/read:0
@
biases_1/bias2:0biases_1/bias2/Assignbiases_1/bias2/read:0
@
biases_1/bias3:0biases_1/bias3/Assignbiases_1/bias3/read:0
I
biases_1/bias_out:0biases_1/bias_out/Assignbiases_1/bias_out/read:0
.

Variable:0Variable/AssignVariable/read:0
I
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:0
I
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:0
X
weights_1/weight1/Adam:0weights_1/weight1/Adam/Assignweights_1/weight1/Adam/read:0
^
weights_1/weight1/Adam_1:0weights_1/weight1/Adam_1/Assignweights_1/weight1/Adam_1/read:0
X
weights_1/weight2/Adam:0weights_1/weight2/Adam/Assignweights_1/weight2/Adam/read:0
^
weights_1/weight2/Adam_1:0weights_1/weight2/Adam_1/Assignweights_1/weight2/Adam_1/read:0
X
weights_1/weight3/Adam:0weights_1/weight3/Adam/Assignweights_1/weight3/Adam/read:0
^
weights_1/weight3/Adam_1:0weights_1/weight3/Adam_1/Assignweights_1/weight3/Adam_1/read:0
a
weights_1/weight_out/Adam:0 weights_1/weight_out/Adam/Assign weights_1/weight_out/Adam/read:0
g
weights_1/weight_out/Adam_1:0"weights_1/weight_out/Adam_1/Assign"weights_1/weight_out/Adam_1/read:0
O
biases_1/bias1/Adam:0biases_1/bias1/Adam/Assignbiases_1/bias1/Adam/read:0
U
biases_1/bias1/Adam_1:0biases_1/bias1/Adam_1/Assignbiases_1/bias1/Adam_1/read:0
O
biases_1/bias2/Adam:0biases_1/bias2/Adam/Assignbiases_1/bias2/Adam/read:0
U
biases_1/bias2/Adam_1:0biases_1/bias2/Adam_1/Assignbiases_1/bias2/Adam_1/read:0
O
biases_1/bias3/Adam:0biases_1/bias3/Adam/Assignbiases_1/bias3/Adam/read:0
U
biases_1/bias3/Adam_1:0biases_1/bias3/Adam_1/Assignbiases_1/bias3/Adam_1/read:0
X
biases_1/bias_out/Adam:0biases_1/bias_out/Adam/Assignbiases_1/bias_out/Adam/read:0
^
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0-�%F       r5��	��6���A*;


total_loss�n�@

error_R�S?

learning_rate_1{ht7��	H       ��H�	�n6���A*;


total_lossH�@

error_RS?

learning_rate_1{ht7/��H       ��H�	Ѿ6���A*;


total_loss���@

error_R&�I?

learning_rate_1{ht7z���H       ��H�	�6���A*;


total_loss{��@

error_R�eV?

learning_rate_1{ht7~W4H       ��H�	��6���A*;


total_lossA�@

error_Rs�M?

learning_rate_1{ht7�G̶H       ��H�	<�6���A*;


total_lossm��@

error_R��V?

learning_rate_1{ht7��i�H       ��H�	s" 6���A*;


total_loss��@

error_Rd�L?

learning_rate_1{ht7-��H       ��H�	Ք 6���A*;


total_loss���@

error_RO�J?

learning_rate_1{ht76��HH       ��H�	J� 6���A*;


total_loss�l�@

error_RF�;?

learning_rate_1{ht7k�^H       ��H�	n*!6���A	*;


total_loss�V�@

error_R$�H?

learning_rate_1{ht7��9H       ��H�	�!6���A
*;


total_loss���@

error_Rڹ>?

learning_rate_1{ht7����H       ��H�	�!6���A*;


total_loss5A

error_Rx�O?

learning_rate_1{ht7�e H       ��H�	T5"6���A*;


total_loss���@

error_Rv,X?

learning_rate_1{ht7� i�H       ��H�	��"6���A*;


total_loss��@

error_R�V?

learning_rate_1{ht7�zZH       ��H�	�#6���A*;


total_loss�m�@

error_R��>?

learning_rate_1{ht7_Ո�H       ��H�	dF#6���A*;


total_loss�B�@

error_R;�_?

learning_rate_1{ht7����H       ��H�	��#6���A*;


total_loss�A�@

error_R�R?

learning_rate_1{ht7�ݔH       ��H�	�$6���A*;


total_loss��@

error_R�kH?

learning_rate_1{ht7*��	H       ��H�	�b$6���A*;


total_losst��@

error_R��L?

learning_rate_1{ht7Ch��H       ��H�	#�$6���A*;


total_lossD�s@

error_R]??

learning_rate_1{ht7d׀FH       ��H�	R:%6���A*;


total_lossd��@

error_RzL?

learning_rate_1{ht7�x��H       ��H�	Ǎ%6���A*;


total_loss<��@

error_R��R?

learning_rate_1{ht7�/�;H       ��H�	[&6���A*;


total_loss�7�@

error_R�2?

learning_rate_1{ht7��FH       ��H�	�T&6���A*;


total_loss�@

error_R�H?

learning_rate_1{ht7\�H       ��H�	ǜ&6���A*;


total_loss�S�@

error_Re=?

learning_rate_1{ht71��aH       ��H�	<'6���A*;


total_loss:ӻ@

error_RfZK?

learning_rate_1{ht7���nH       ��H�	nY'6���A*;


total_lossu�@

error_R.�G?

learning_rate_1{ht7�7H       ��H�	Ϟ'6���A*;


total_lossz��@

error_Rq@?

learning_rate_1{ht7���H       ��H�	K(6���A*;


total_loss�@

error_R�,N?

learning_rate_1{ht7瀥H       ��H�	�[(6���A*;


total_loss�ȡ@

error_R��G?

learning_rate_1{ht7q�ؙH       ��H�	��(6���A*;


total_loss�9A

error_R�8W?

learning_rate_1{ht7(��H       ��H�	�	)6���A*;


total_loss�^�@

error_R -J?

learning_rate_1{ht7�>?�H       ��H�	�R)6���A *;


total_lossO٤@

error_R_�]?

learning_rate_1{ht7�oH       ��H�	՞)6���A!*;


total_loss.$�@

error_RJM?

learning_rate_1{ht7�}�BH       ��H�	R*6���A"*;


total_lossA�@

error_R��g?

learning_rate_1{ht7Nk��H       ��H�	�a*6���A#*;


total_loss��@

error_R��K?

learning_rate_1{ht7�N�CH       ��H�	��*6���A$*;


total_loss�
�@

error_R%L?

learning_rate_1{ht7��kVH       ��H�		+6���A%*;


total_lossl��@

error_RRqM?

learning_rate_1{ht7?ݜ�H       ��H�	so+6���A&*;


total_losse.�@

error_R,�W?

learning_rate_1{ht7��IH       ��H�	��+6���A'*;


total_loss}A

error_R�(Y?

learning_rate_1{ht7Hd��H       ��H�	�&,6���A(*;


total_loss6��@

error_RҷN?

learning_rate_1{ht70N�|H       ��H�	�},6���A)*;


total_loss	��@

error_Rt�D?

learning_rate_1{ht7��`H       ��H�	��,6���A**;


total_loss�gA

error_RM|H?

learning_rate_1{ht7���H       ��H�	v-6���A+*;


total_loss\m�@

error_Ry;?

learning_rate_1{ht7���H       ��H�	�~-6���A,*;


total_loss_P�@

error_RZ�`?

learning_rate_1{ht7i�vnH       ��H�	��-6���A-*;


total_loss|�A

error_Rv�C?

learning_rate_1{ht7�wgPH       ��H�	2.6���A.*;


total_loss�n�@

error_R�HO?

learning_rate_1{ht7?�H       ��H�	y^.6���A/*;


total_loss;��@

error_R�Y?

learning_rate_1{ht7�;B�H       ��H�	�.6���A0*;


total_loss{��@

error_R�M?

learning_rate_1{ht7�B�H       ��H�	x�.6���A1*;


total_lossV��@

error_R�M?

learning_rate_1{ht7�z��H       ��H�	�:/6���A2*;


total_lossO��@

error_RaSB?

learning_rate_1{ht7I� 'H       ��H�	�}/6���A3*;


total_loss�C7A

error_R�\?

learning_rate_1{ht7�W�H       ��H�	�/6���A4*;


total_loss=+A

error_R&I?

learning_rate_1{ht7x��H       ��H�	�06���A5*;


total_loss��@

error_R�[?

learning_rate_1{ht7��0H       ��H�	�H06���A6*;


total_loss`-�@

error_R��T?

learning_rate_1{ht7�#�H       ��H�	K�06���A7*;


total_loss�y�@

error_R�~J?

learning_rate_1{ht7K �H       ��H�	��06���A8*;


total_loss�@

error_R�kF?

learning_rate_1{ht7����H       ��H�	�16���A9*;


total_loss���@

error_RI05?

learning_rate_1{ht7�<7(H       ��H�	�Z16���A:*;


total_loss�U�@

error_RlqI?

learning_rate_1{ht7F��5H       ��H�	�16���A;*;


total_loss��@

error_R�P?

learning_rate_1{ht7GS?>H       ��H�	��16���A<*;


total_loss�_�@

error_R$Q?

learning_rate_1{ht7J��H       ��H�	�(26���A=*;


total_loss��@

error_R�J?

learning_rate_1{ht7�~�H       ��H�	�p26���A>*;


total_loss6k�@

error_R&�M?

learning_rate_1{ht7O0�H       ��H�	m�26���A?*;


total_lossɎ�@

error_R1QV?

learning_rate_1{ht7��DH       ��H�	��26���A@*;


total_loss?A

error_R��Q?

learning_rate_1{ht7I^mUH       ��H�	�G36���AA*;


total_lossZ6�@

error_R�KB?

learning_rate_1{ht7�S��H       ��H�	�36���AB*;


total_loss��@

error_RD�Q?

learning_rate_1{ht7� ]�H       ��H�	�36���AC*;


total_loss:�@

error_RV�P?

learning_rate_1{ht75�;H       ��H�	Z46���AD*;


total_loss.4�@

error_RdV?

learning_rate_1{ht7��l�H       ��H�	�a46���AE*;


total_lossPi�@

error_RJZ?

learning_rate_1{ht7���H       ��H�	{�46���AF*;


total_lossb�@

error_ROR?

learning_rate_1{ht7�rsH       ��H�	��46���AG*;


total_loss�v�@

error_R,�P?

learning_rate_1{ht7Ƙ�H       ��H�	P56���AH*;


total_loss/ҕ@

error_R��<?

learning_rate_1{ht76l�H       ��H�	��56���AI*;


total_loss� �@

error_Rva?

learning_rate_1{ht7�S��H       ��H�	1�56���AJ*;


total_loss�I�@

error_R�Q?

learning_rate_1{ht7��ϜH       ��H�	#$66���AK*;


total_loss@� A

error_R�[?

learning_rate_1{ht7���H       ��H�	�g66���AL*;


total_loss�{�@

error_Rn�M?

learning_rate_1{ht7l;H       ��H�	<�66���AM*;


total_lossV�@

error_R/�L?

learning_rate_1{ht7�DuH       ��H�	i�66���AN*;


total_lossa6�@

error_R`�L?

learning_rate_1{ht7nyH       ��H�	2/76���AO*;


total_loss���@

error_R�8W?

learning_rate_1{ht7�8�H       ��H�	�v76���AP*;


total_loss��@

error_RF:C?

learning_rate_1{ht7��JNH       ��H�	)�76���AQ*;


total_lossW.t@

error_R�D?

learning_rate_1{ht7�ܼZH       ��H�	A86���AR*;


total_lossT�@

error_RW�W?

learning_rate_1{ht7���:H       ��H�	aH86���AS*;


total_loss�7�@

error_RT�R?

learning_rate_1{ht7P_�gH       ��H�	��86���AT*;


total_loss�U�@

error_R(Tg?

learning_rate_1{ht72B��H       ��H�	��86���AU*;


total_loss,a�@

error_RܿW?

learning_rate_1{ht7ʹ��H       ��H�	�-96���AV*;


total_lossw�@

error_R�\?

learning_rate_1{ht7ɶV�H       ��H�	�v96���AW*;


total_loss&��@

error_R�^?

learning_rate_1{ht7]j�zH       ��H�	�96���AX*;


total_lossW��@

error_R�CT?

learning_rate_1{ht7Y���H       ��H�	j:6���AY*;


total_loss(��@

error_R8EI?

learning_rate_1{ht77�SHH       ��H�	ZV:6���AZ*;


total_loss���@

error_R&T?

learning_rate_1{ht7}��<H       ��H�	t�:6���A[*;


total_loss-�A

error_Rl�H?

learning_rate_1{ht7��tH       ��H�	��:6���A\*;


total_loss�@

error_R(a]?

learning_rate_1{ht7�W�H       ��H�	�);6���A]*;


total_lossaO�@

error_R�Q?

learning_rate_1{ht7[h}H       ��H�	�q;6���A^*;


total_loss e�@

error_R1�]?

learning_rate_1{ht7H�u�H       ��H�	��;6���A_*;


total_lossv3�@

error_R��L?

learning_rate_1{ht7�_[�H       ��H�	�0<6���A`*;


total_loss��@

error_R��S?

learning_rate_1{ht7�׾�H       ��H�	�{<6���Aa*;


total_loss(��@

error_R�xL?

learning_rate_1{ht7v�H       ��H�	{�<6���Ab*;


total_lossr��@

error_R�P?

learning_rate_1{ht7�ּH       ��H�	=6���Ac*;


total_losshd�@

error_R�ia?

learning_rate_1{ht7"N��H       ��H�		Y=6���Ad*;


total_loss�h�@

error_R��\?

learning_rate_1{ht7��.H       ��H�	5�=6���Ae*;


total_loss`^�@

error_R��O?

learning_rate_1{ht7���H       ��H�	��=6���Af*;


total_lossc��@

error_R(�??

learning_rate_1{ht7D�ׂH       ��H�	�/>6���Ag*;


total_loss���@

error_R�9?

learning_rate_1{ht7C���H       ��H�	s>6���Ah*;


total_loss�z�@

error_RM�C?

learning_rate_1{ht7���H       ��H�	t�>6���Ai*;


total_lossm��@

error_R�H?

learning_rate_1{ht7w�3RH       ��H�	�
?6���Aj*;


total_loss.��@

error_R�@?

learning_rate_1{ht7���H       ��H�	�T?6���Ak*;


total_lossa��@

error_Rd�6?

learning_rate_1{ht7����H       ��H�	w�?6���Al*;


total_losst��@

error_Rq�F?

learning_rate_1{ht7��1H       ��H�	L�?6���Am*;


total_lossΥ�@

error_R�\K?

learning_rate_1{ht7m�ZH       ��H�	�0@6���An*;


total_loss���@

error_R�B?

learning_rate_1{ht7k�H       ��H�	�t@6���Ao*;


total_loss�.�@

error_R��<?

learning_rate_1{ht7Ҙ��H       ��H�	ɶ@6���Ap*;


total_lossTJ�@

error_Rx�W?

learning_rate_1{ht7�$�gH       ��H�	��@6���Aq*;


total_loss/��@

error_RNd?

learning_rate_1{ht7gC�[H       ��H�	�@A6���Ar*;


total_loss�>�@

error_RW�P?

learning_rate_1{ht7�J�8H       ��H�	)�A6���As*;


total_loss"�
A

error_R}xE?

learning_rate_1{ht7���H       ��H�	��A6���At*;


total_loss͉�@

error_RN?

learning_rate_1{ht7rD"H       ��H�	�
B6���Au*;


total_loss��@

error_R!L?

learning_rate_1{ht7��YH       ��H�	�NB6���Av*;


total_loss?{@

error_R��G?

learning_rate_1{ht7Ч��H       ��H�	6�B6���Aw*;


total_lossa��@

error_R*C?

learning_rate_1{ht7hB'H       ��H�	y�B6���Ax*;


total_lossN��@

error_R?�I?

learning_rate_1{ht7�[�(H       ��H�	�C6���Ay*;


total_loss{[A

error_R�`Y?

learning_rate_1{ht7FE��H       ��H�	�fC6���Az*;


total_loss�I�@

error_R�RT?

learning_rate_1{ht7 �H       ��H�	�C6���A{*;


total_loss��@

error_R��H?

learning_rate_1{ht7�럗H       ��H�	k�C6���A|*;


total_lossڇ�@

error_R��K?

learning_rate_1{ht7Z�p)H       ��H�	76D6���A}*;


total_lossaJ�@

error_R�O?

learning_rate_1{ht7���H       ��H�	|D6���A~*;


total_loss}�@

error_R�eN?

learning_rate_1{ht7KQ�H       ��H�	$�D6���A*;


total_loss�sA

error_R�|_?

learning_rate_1{ht7���I       6%�	!E6���A�*;


total_lossL��@

error_R��>?

learning_rate_1{ht7�8]�I       6%�	�pE6���A�*;


total_loss���@

error_R��U?

learning_rate_1{ht7G>\I       6%�	:�E6���A�*;


total_loss8�@

error_RV�L?

learning_rate_1{ht78' "I       6%�	�F6���A�*;


total_loss�Ͷ@

error_Rw4F?

learning_rate_1{ht7�F0�I       6%�	oQF6���A�*;


total_loss}J�@

error_R��D?

learning_rate_1{ht7 �I       6%�	֗F6���A�*;


total_loss��@

error_Ra�S?

learning_rate_1{ht7�x�I       6%�	5�F6���A�*;


total_loss��@

error_R��J?

learning_rate_1{ht7o�}I       6%�	/G6���A�*;


total_loss���@

error_R=F?

learning_rate_1{ht7=9NQI       6%�	]cG6���A�*;


total_loss�
�@

error_Ro}F?

learning_rate_1{ht7Q)��I       6%�	�G6���A�*;


total_lossq,P@

error_R>?

learning_rate_1{ht7��`HI       6%�	�G6���A�*;


total_loss�>�@

error_R�L?

learning_rate_1{ht7���\I       6%�	~3H6���A�*;


total_loss��@

error_R�Q?

learning_rate_1{ht7�~��I       6%�	�~H6���A�*;


total_loss ��@

error_RFS?

learning_rate_1{ht7

��I       6%�	t�H6���A�*;


total_lossjE�@

error_R� P?

learning_rate_1{ht7,0:tI       6%�	�I6���A�*;


total_loss��@

error_R`�H?

learning_rate_1{ht7��ިI       6%�	_I6���A�*;


total_lossdٽ@

error_R5??

learning_rate_1{ht7���I       6%�	�I6���A�*;


total_loss�5�@

error_RZG?

learning_rate_1{ht7���I       6%�	�I6���A�*;


total_loss�@

error_RWmF?

learning_rate_1{ht7�/o�I       6%�	>7J6���A�*;


total_loss���@

error_R��K?

learning_rate_1{ht7��FI       6%�	�}J6���A�*;


total_lossC �@

error_R%�c?

learning_rate_1{ht7�\�WI       6%�	�J6���A�*;


total_loss흈@

error_R�)N?

learning_rate_1{ht77�I       6%�	K6���A�*;


total_loss�f�@

error_R�.W?

learning_rate_1{ht7�0�I       6%�	WK6���A�*;


total_loss��@

error_Rs�G?

learning_rate_1{ht7�UJ/I       6%�	��K6���A�*;


total_loss1�@

error_Ro�Y?

learning_rate_1{ht7�͛I       6%�	|L6���A�*;


total_loss �@

error_R4O>?

learning_rate_1{ht7LI       6%�	c\L6���A�*;


total_loss1�L@

error_R��F?

learning_rate_1{ht7*V��I       6%�	�L6���A�*;


total_loss�f@

error_Rh�I?

learning_rate_1{ht7���I       6%�	3�L6���A�*;


total_loss� �@

error_R'W?

learning_rate_1{ht7��I       6%�	/=M6���A�*;


total_lossW��@

error_R6�e?

learning_rate_1{ht7��NI       6%�	�M6���A�*;


total_loss�*�@

error_R�7J?

learning_rate_1{ht7�D�nI       6%�	��M6���A�*;


total_loss�{�@

error_R_P?

learning_rate_1{ht7�hq�I       6%�	9N6���A�*;


total_losssT�@

error_R�*Z?

learning_rate_1{ht7�I       6%�	<WN6���A�*;


total_loss�~�@

error_Rs�>?

learning_rate_1{ht7��I       6%�	?�N6���A�*;


total_loss�7A

error_R��H?

learning_rate_1{ht7�~��I       6%�	��N6���A�*;


total_loss@

error_R�D?

learning_rate_1{ht7��EsI       6%�	�*O6���A�*;


total_loss�Y�@

error_R@�D?

learning_rate_1{ht7��)�I       6%�	�nO6���A�*;


total_loss�mw@

error_R�[?

learning_rate_1{ht7�#��I       6%�	�O6���A�*;


total_loss6�@

error_R�m?

learning_rate_1{ht7SOcI       6%�	w�O6���A�*;


total_lossR�@

error_R�@Z?

learning_rate_1{ht7�a"�I       6%�	�CP6���A�*;


total_loss$�@

error_R�@[?

learning_rate_1{ht7��z8I       6%�	��P6���A�*;


total_lossd0�@

error_RԲP?

learning_rate_1{ht7�F�I       6%�	�P6���A�*;


total_lossMI�@

error_R�O?

learning_rate_1{ht7��nI       6%�	�Q6���A�*;


total_lossn��@

error_R*�W?

learning_rate_1{ht7̿Y�I       6%�	�dQ6���A�*;


total_loss"�@

error_Rix]?

learning_rate_1{ht7�"*I       6%�	�Q6���A�*;


total_loss���@

error_R�VK?

learning_rate_1{ht749amI       6%�	��Q6���A�*;


total_loss�@

error_R[T?

learning_rate_1{ht7I��I       6%�	[9R6���A�*;


total_lossQyA

error_R��E?

learning_rate_1{ht7ʮ�NI       6%�	&�R6���A�*;


total_lossS�@

error_R8T?

learning_rate_1{ht7`=I       6%�	��R6���A�*;


total_lossx&A

error_Rt�6?

learning_rate_1{ht7�up�I       6%�	rS6���A�*;


total_loss�TA

error_R�[I?

learning_rate_1{ht7U{I       6%�	�TS6���A�*;


total_loss	 �@

error_R��Y?

learning_rate_1{ht7]��SI       6%�	�S6���A�*;


total_loss�lZ@

error_R�;?

learning_rate_1{ht7gϾ[I       6%�	"�S6���A�*;


total_loss	q�@

error_RT�N?

learning_rate_1{ht7�,bZI       6%�	�'T6���A�*;


total_loss.��@

error_R:�P?

learning_rate_1{ht7B���I       6%�	[lT6���A�*;


total_loss��t@

error_Rڌ=?

learning_rate_1{ht7��{I       6%�	ˮT6���A�*;


total_loss�%�@

error_RE�B?

learning_rate_1{ht7%��I       6%�	�U6���A�*;


total_loss&x�@

error_R�F?

learning_rate_1{ht7(
�uI       6%�	�hU6���A�*;


total_lossד@

error_R�uA?

learning_rate_1{ht7ʝ��I       6%�	��U6���A�*;


total_loss�@

error_R��b?

learning_rate_1{ht7̡��I       6%�	�U6���A�*;


total_loss�ו@

error_R63X?

learning_rate_1{ht7��6I       6%�	A2V6���A�*;


total_loss=��@

error_R�4K?

learning_rate_1{ht7��Z�I       6%�	�sV6���A�*;


total_loss }@

error_R}]?

learning_rate_1{ht7M<�I       6%�	c�V6���A�*;


total_loss�c�@

error_R�^?

learning_rate_1{ht7�q�I       6%�	�V6���A�*;


total_loss2�@

error_R;(I?

learning_rate_1{ht7�_h�I       6%�	�CW6���A�*;


total_loss��@

error_R��M?

learning_rate_1{ht7Ej}I       6%�	F�W6���A�*;


total_loss���@

error_R�H?

learning_rate_1{ht7.��PI       6%�	�W6���A�*;


total_loss2Vq@

error_RD�F?

learning_rate_1{ht7ʴ)�I       6%�	PX6���A�*;


total_loss0�@

error_R�kZ?

learning_rate_1{ht7��$I       6%�	[X6���A�*;


total_loss��@

error_R�J?

learning_rate_1{ht7�gܞI       6%�	L�X6���A�*;


total_lossN�@

error_R��R?

learning_rate_1{ht7֒OpI       6%�	d�X6���A�*;


total_loss��@

error_Rv5>?

learning_rate_1{ht7���I       6%�	U*Y6���A�*;


total_lossM��@

error_RO{C?

learning_rate_1{ht7xp׆I       6%�	^sY6���A�*;


total_loss-�q@

error_R#�G?

learning_rate_1{ht7p7VDI       6%�	H�Y6���A�*;


total_loss�)�@

error_R�CG?

learning_rate_1{ht7�7�I       6%�	�Z6���A�*;


total_losse>�@

error_Rd^I?

learning_rate_1{ht7���PI       6%�	 gZ6���A�*;


total_lossֽ�@

error_R�jZ?

learning_rate_1{ht7k��I       6%�	ڳZ6���A�*;


total_loss��@

error_R�JQ?

learning_rate_1{ht7YP.�I       6%�	 �Z6���A�*;


total_loss{d:A

error_R�L?

learning_rate_1{ht7f��I       6%�	2A[6���A�*;


total_loss��@

error_R�Z?

learning_rate_1{ht7����I       6%�	��[6���A�*;


total_loss�@

error_R��^?

learning_rate_1{ht7I(��I       6%�	,�[6���A�*;


total_loss��@

error_R��I?

learning_rate_1{ht7p#��I       6%�		\6���A�*;


total_loss�ƫ@

error_R��I?

learning_rate_1{ht7=�'�I       6%�	�R\6���A�*;


total_loss<��@

error_Rߗ\?

learning_rate_1{ht7�IeI       6%�	ݖ\6���A�*;


total_lossh�@

error_R\4U?

learning_rate_1{ht7V��I       6%�	��\6���A�*;


total_lossSǅ@

error_Rz$C?

learning_rate_1{ht7���>I       6%�	�$]6���A�*;


total_losszL�@

error_R��G?

learning_rate_1{ht7(��yI       6%�	7k]6���A�*;


total_loss-�@

error_R��??

learning_rate_1{ht7�0hI       6%�	�]6���A�*;


total_lossڙ�@

error_RWCF?

learning_rate_1{ht7�zI       6%�	�]6���A�*;


total_loss(��@

error_R�bI?

learning_rate_1{ht7Y���I       6%�	�?^6���A�*;


total_losst��@

error_R�4P?

learning_rate_1{ht7E�+I       6%�	�^6���A�*;


total_loss
Ū@

error_R;cZ?

learning_rate_1{ht77u+�I       6%�	2�^6���A�*;


total_loss��@

error_R�QS?

learning_rate_1{ht7l�I       6%�	�_6���A�*;


total_loss�6�@

error_R��>?

learning_rate_1{ht7dB�I       6%�	�X_6���A�*;


total_lossWæ@

error_Ro?

learning_rate_1{ht7�U��I       6%�	%�_6���A�*;


total_loss~��@

error_R!�E?

learning_rate_1{ht7�{�I       6%�	f�_6���A�*;


total_losszށ@

error_RN�E?

learning_rate_1{ht7���pI       6%�	�&`6���A�*;


total_lossM�t@

error_R�=?

learning_rate_1{ht7�#΋I       6%�	�n`6���A�*;


total_lossh�}@

error_RC�R?

learning_rate_1{ht7�m�I       6%�	��`6���A�*;


total_loss�vJA

error_RmK?

learning_rate_1{ht7�`�I       6%�	��`6���A�*;


total_loss2^d@

error_R|F8?

learning_rate_1{ht7��I       6%�	�?a6���A�*;


total_loss�"�@

error_R�7B?

learning_rate_1{ht7r7DI       6%�	"�a6���A�*;


total_loss��@

error_ReT?

learning_rate_1{ht7�x�I       6%�	^�a6���A�*;


total_loss��@

error_R8>Y?

learning_rate_1{ht7�-8>I       6%�	�	b6���A�*;


total_loss��@

error_R��O?

learning_rate_1{ht7�?I       6%�	�Lb6���A�*;


total_lossϠ�@

error_RDN\?

learning_rate_1{ht7�>HkI       6%�	��b6���A�*;


total_lossh��@

error_RT�S?

learning_rate_1{ht7j��I       6%�	��b6���A�*;


total_loss���@

error_RϷR?

learning_rate_1{ht7��9:I       6%�	�c6���A�*;


total_lossR��@

error_Rx�X?

learning_rate_1{ht7��I       6%�	�cc6���A�*;


total_loss��@

error_R6(Y?

learning_rate_1{ht7?+d�I       6%�	R�c6���A�*;


total_loss�`�@

error_R��g?

learning_rate_1{ht7l
��I       6%�	��c6���A�*;


total_loss�p�@

error_R�Q?

learning_rate_1{ht7&[;�I       6%�	P;d6���A�*;


total_loss.��@

error_R�P?

learning_rate_1{ht7R3�I       6%�	�~d6���A�*;


total_loss��@

error_R�%F?

learning_rate_1{ht7��V2I       6%�	7�d6���A�*;


total_loss�(�@

error_R�xQ?

learning_rate_1{ht7Rp8�I       6%�	�)e6���A�*;


total_loss���@

error_R.�a?

learning_rate_1{ht73�9�I       6%�	�xe6���A�*;


total_loss\��@

error_R�3O?

learning_rate_1{ht7�e�I       6%�	��e6���A�*;


total_lossc��@

error_R��=?

learning_rate_1{ht7T3�I       6%�	��e6���A�*;


total_loss	�@

error_RMV?

learning_rate_1{ht7�8�3I       6%�	3Hf6���A�*;


total_loss��@

error_R�EC?

learning_rate_1{ht7��I       6%�	x�f6���A�*;


total_loss<�{@

error_R%�J?

learning_rate_1{ht7l�|I       6%�	!�f6���A�*;


total_loss}��@

error_R&�Q?

learning_rate_1{ht7JV9�I       6%�	�g6���A�*;


total_loss�͋@

error_R��D?

learning_rate_1{ht7��HI       6%�	7gg6���A�*;


total_loss���@

error_RE�Q?

learning_rate_1{ht7����I       6%�	$�g6���A�*;


total_loss:/n@

error_R��:?

learning_rate_1{ht7�C�XI       6%�	�g6���A�*;


total_lossi��@

error_R=+L?

learning_rate_1{ht7�X��I       6%�	�<h6���A�*;


total_loss]m@

error_RW5]?

learning_rate_1{ht7�j�I       6%�	a�h6���A�*;


total_loss�@

error_RqC?

learning_rate_1{ht7��I       6%�	��h6���A�*;


total_loss�s�@

error_R� N?

learning_rate_1{ht7cs�I       6%�	�i6���A�*;


total_lossN�@

error_R�??

learning_rate_1{ht7���2I       6%�	O\i6���A�*;


total_loss3��@

error_R��J?

learning_rate_1{ht7�r�`I       6%�	�i6���A�*;


total_loss��@

error_RO�a?

learning_rate_1{ht7��%MI       6%�	��i6���A�*;


total_loss��@

error_R�RB?

learning_rate_1{ht7c�HI       6%�	+j6���A�*;


total_loss88�@

error_R�K?

learning_rate_1{ht7�a�I       6%�	�pj6���A�*;


total_loss/�@

error_RcwX?

learning_rate_1{ht7���dI       6%�	��j6���A�*;


total_loss}��@

error_R�RV?

learning_rate_1{ht7ɻ��I       6%�	
�j6���A�*;


total_loss:��@

error_Rx�U?

learning_rate_1{ht7����I       6%�	�@k6���A�*;


total_lossH|�@

error_R�+J?

learning_rate_1{ht7<���I       6%�	��k6���A�*;


total_loss���@

error_RI^F?

learning_rate_1{ht7���I       6%�	��k6���A�*;


total_loss��@

error_R\eW?

learning_rate_1{ht7��I       6%�	�!l6���A�*;


total_loss���@

error_R�}J?

learning_rate_1{ht7�V�iI       6%�	"jl6���A�*;


total_loss���@

error_Rf5L?

learning_rate_1{ht7ъ(�I       6%�	�l6���A�*;


total_loss�I�@

error_R�K?

learning_rate_1{ht7fy� I       6%�	�l6���A�*;


total_loss�]@

error_R�c=?

learning_rate_1{ht7W���I       6%�	�9m6���A�*;


total_lossvDA

error_R3�G?

learning_rate_1{ht7���I       6%�	r|m6���A�*;


total_loss^~@

error_RL�R?

learning_rate_1{ht7��7BI       6%�	��m6���A�*;


total_lossښ�@

error_RI�g?

learning_rate_1{ht7)�J�I       6%�	]n6���A�*;


total_loss/1�@

error_R��H?

learning_rate_1{ht7D��iI       6%�	3Gn6���A�*;


total_lossa�@

error_R�E?

learning_rate_1{ht7���I       6%�	��n6���A�*;


total_loss�{�@

error_R��=?

learning_rate_1{ht7���I       6%�	�n6���A�*;


total_loss�(�@

error_Rz^L?

learning_rate_1{ht7�uI       6%�	o6���A�*;


total_loss�b�@

error_R�j??

learning_rate_1{ht7 ۚ�I       6%�	6Zo6���A�*;


total_loss��@

error_RqqU?

learning_rate_1{ht7E�&I       6%�	�o6���A�*;


total_lossMO�@

error_R��C?

learning_rate_1{ht7�x�I       6%�	��o6���A�*;


total_loss�h�@

error_RڈP?

learning_rate_1{ht7�2�I       6%�	�$p6���A�*;


total_loss�|@

error_R �8?

learning_rate_1{ht7p�?�I       6%�	�jp6���A�*;


total_loss��@

error_RW0T?

learning_rate_1{ht7t#�LI       6%�	Ӯp6���A�*;


total_loss�I�@

error_R�rM?

learning_rate_1{ht7��9I       6%�	�p6���A�*;


total_loss���@

error_RE�W?

learning_rate_1{ht7��I       6%�	"8q6���A�*;


total_loss��@

error_R�hn?

learning_rate_1{ht7��	AI       6%�	�yq6���A�*;


total_lossv��@

error_R��<?

learning_rate_1{ht7N���I       6%�	t�q6���A�*;


total_loss��@

error_RX�2?

learning_rate_1{ht7���mI       6%�	r6���A�*;


total_loss��@

error_R��W?

learning_rate_1{ht7K�m(I       6%�	|Qr6���A�*;


total_loss�>A

error_R�O?

learning_rate_1{ht7@J�eI       6%�	q�r6���A�*;


total_loss���@

error_R��W?

learning_rate_1{ht7��h�I       6%�	S�r6���A�*;


total_lossZ0�@

error_R��F?

learning_rate_1{ht7ԁf_I       6%�	�+s6���A�*;


total_lossJٹ@

error_R�a?

learning_rate_1{ht7�jo	I       6%�	Bps6���A�*;


total_loss6�@

error_R�%U?

learning_rate_1{ht7�{ �I       6%�	��s6���A�*;


total_loss��@

error_R�2H?

learning_rate_1{ht7�;hI       6%�	�s6���A�*;


total_lossZc�@

error_R� ??

learning_rate_1{ht7��=RI       6%�	�>t6���A�*;


total_loss6G�@

error_R��D?

learning_rate_1{ht7s�Z�I       6%�	*�t6���A�*;


total_loss��@

error_RH�L?

learning_rate_1{ht7��4eI       6%�	��t6���A�*;


total_loss��@

error_R��P?

learning_rate_1{ht7X�^I       6%�	�(u6���A�*;


total_loss-��@

error_R�S?

learning_rate_1{ht7�M�I       6%�	Uqu6���A�*;


total_loss�P@

error_R�&:?

learning_rate_1{ht7=z�I       6%�	��u6���A�*;


total_loss��A

error_R_�M?

learning_rate_1{ht7P���I       6%�	t�u6���A�*;


total_loss��{@

error_R H?

learning_rate_1{ht7A�	�I       6%�	:v6���A�*;


total_loss�3�@

error_Rv�N?

learning_rate_1{ht70��tI       6%�	 {v6���A�*;


total_loss�@

error_R�T?

learning_rate_1{ht7Q�>I       6%�	��v6���A�*;


total_loss�@

error_R,�T?

learning_rate_1{ht71#� I       6%�	�w6���A�*;


total_loss�p�@

error_R/*J?

learning_rate_1{ht7
��I       6%�	sGw6���A�*;


total_loss%=n@

error_R�!T?

learning_rate_1{ht7f��I       6%�	��w6���A�*;


total_loss(@�@

error_R��J?

learning_rate_1{ht7/��I       6%�	k�w6���A�*;


total_loss��@

error_RM�O?

learning_rate_1{ht7^I       6%�	�x6���A�*;


total_lossψ@

error_R\�R?

learning_rate_1{ht7����I       6%�	Tx6���A�*;


total_loss��@

error_R�<H?

learning_rate_1{ht7��I       6%�	H�x6���A�*;


total_loss=�@

error_R�C?

learning_rate_1{ht7�݈I       6%�	��x6���A�*;


total_loss��@

error_RaL?

learning_rate_1{ht7��8I       6%�	5.y6���A�*;


total_loss�@

error_R�K?

learning_rate_1{ht7�W+{I       6%�	�zy6���A�*;


total_loss��@

error_R�J?

learning_rate_1{ht7��F I       6%�	N�y6���A�*;


total_lossA

error_R�\?

learning_rate_1{ht7n�)$I       6%�	�
z6���A�*;


total_lossF�@

error_R��8?

learning_rate_1{ht7d[��I       6%�	Sz6���A�*;


total_lossñ�@

error_Rd�;?

learning_rate_1{ht7�q��I       6%�	��z6���A�*;


total_loss\ �@

error_R�N?

learning_rate_1{ht7�m�I       6%�	��z6���A�*;


total_loss�Ҫ@

error_R�gC?

learning_rate_1{ht7ܬ:�I       6%�	A,{6���A�*;


total_loss�A

error_R�E?

learning_rate_1{ht7���I       6%�	 v{6���A�*;


total_loss���@

error_R�O?

learning_rate_1{ht77�:I       6%�	^�{6���A�*;


total_lossm�@

error_R�J?

learning_rate_1{ht7b�I       6%�	x�{6���A�*;


total_loss�+�@

error_R��I?

learning_rate_1{ht7+IH�I       6%�	�A|6���A�*;


total_loss я@

error_RL�6?

learning_rate_1{ht7cb8I       6%�	K�|6���A�*;


total_lossM7�@

error_RcbJ?

learning_rate_1{ht7���9I       6%�	h�|6���A�*;


total_loss=��@

error_R��A?

learning_rate_1{ht7�˞�I       6%�	9}6���A�*;


total_loss��@

error_Ro�]?

learning_rate_1{ht7�6�JI       6%�	FN}6���A�*;


total_loss<�A

error_R��W?

learning_rate_1{ht7��ɊI       6%�	�}6���A�*;


total_loss�~�@

error_R�XC?

learning_rate_1{ht7��,I       6%�	��}6���A�*;


total_loss���@

error_R�I;?

learning_rate_1{ht7&ܰ.I       6%�	~6���A�*;


total_loss��p@

error_R�P?

learning_rate_1{ht7�),UI       6%�	Y~6���A�*;


total_loss�M�@

error_Ra�E?

learning_rate_1{ht7��I       6%�	8�~6���A�*;


total_lossض�@

error_Rz`G?

learning_rate_1{ht7Q��I       6%�	��~6���A�*;


total_loss�υ@

error_R�N?

learning_rate_1{ht7PԹI       6%�	�/6���A�*;


total_loss���@

error_RE�U?

learning_rate_1{ht7ާI       6%�	�w6���A�*;


total_loss��@

error_R�D?

learning_rate_1{ht7�I�{I       6%�	}�6���A�*;


total_loss�	A

error_RR�J?

learning_rate_1{ht7[�I       6%�	L�6���A�*;


total_loss�"�@

error_RvAi?

learning_rate_1{ht7�$�*I       6%�	 W�6���A�*;


total_loss�T�@

error_Rv�V?

learning_rate_1{ht7�M[�I       6%�	R��6���A�*;


total_lossc��@

error_RL5O?

learning_rate_1{ht7P4H�I       6%�	��6���A�*;


total_loss�i�@

error_RZUW?

learning_rate_1{ht7t�#I       6%�	�5�6���A�*;


total_lossf�@

error_R��??

learning_rate_1{ht7|)�VI       6%�	L��6���A�*;


total_loss��@

error_R3T?

learning_rate_1{ht7f�u�I       6%�	}΁6���A�*;


total_loss��@

error_R��Y?

learning_rate_1{ht7����I       6%�	F�6���A�*;


total_loss�:�@

error_R��T?

learning_rate_1{ht7��I       6%�	a�6���A�*;


total_loss���@

error_R=�M?

learning_rate_1{ht7�iNI       6%�	���6���A�*;


total_loss���@

error_RTD?

learning_rate_1{ht7���I       6%�	@�6���A�*;


total_loss���@

error_RT�B?

learning_rate_1{ht7�k��I       6%�	[0�6���A�*;


total_loss	�@

error_R��B?

learning_rate_1{ht7��VI       6%�	�w�6���A�*;


total_loss6r�@

error_R��I?

learning_rate_1{ht7�U,I       6%�	���6���A�*;


total_loss}ސ@

error_R��G?

learning_rate_1{ht7�Ӽ]I       6%�		
�6���A�*;


total_loss�s�@

error_R`�W?

learning_rate_1{ht7�v�I       6%�	L�6���A�*;


total_loss�@

error_R��R?

learning_rate_1{ht7E"��I       6%�	���6���A�*;


total_loss�Z�@

error_RL�6?

learning_rate_1{ht73�I       6%�	�6���A�*;


total_loss��@

error_Rf�R?

learning_rate_1{ht7��#�I       6%�	�E�6���A�*;


total_loss�A

error_RVW?

learning_rate_1{ht7�y��I       6%�	.��6���A�*;


total_loss!֒@

error_R��a?

learning_rate_1{ht71�I       6%�	�ۅ6���A�*;


total_lossqQ�@

error_R1ya?

learning_rate_1{ht7��F0I       6%�	+�6���A�*;


total_loss	_�@

error_Rj�]?

learning_rate_1{ht7I��I       6%�	f�6���A�*;


total_loss�]�@

error_Rfl<?

learning_rate_1{ht7-M��I       6%�	㬆6���A�*;


total_loss1�A

error_R�FP?

learning_rate_1{ht7#/ވI       6%�	��6���A�*;


total_loss3�@

error_R�W?

learning_rate_1{ht7���VI       6%�	U=�6���A�*;


total_loss�4�@

error_RE1:?

learning_rate_1{ht7���I       6%�	���6���A�*;


total_lossw+�@

error_R��C?

learning_rate_1{ht7�%I       6%�	�Ǉ6���A�*;


total_loss�$�@

error_RI|S?

learning_rate_1{ht7c �I       6%�	��6���A�*;


total_loss���@

error_R,�\?

learning_rate_1{ht7{>@I       6%�	uR�6���A�*;


total_lossH��@

error_R�;T?

learning_rate_1{ht7�i�I       6%�	���6���A�*;


total_loss:W�@

error_RM�=?

learning_rate_1{ht7iUI       6%�	�݈6���A�*;


total_loss��@

error_R2[?

learning_rate_1{ht7R<�BI       6%�	�!�6���A�*;


total_loss�m�@

error_RɤK?

learning_rate_1{ht7�ښI       6%�	Qe�6���A�*;


total_lossAq@

error_R8�L?

learning_rate_1{ht7�5�I       6%�	ͧ�6���A�*;


total_lossa4�@

error_R��K?

learning_rate_1{ht7��ʑI       6%�	e�6���A�*;


total_loss��o@

error_R�HH?

learning_rate_1{ht7��V�I       6%�	�,�6���A�*;


total_loss���@

error_R�[?

learning_rate_1{ht7���+I       6%�	�q�6���A�*;


total_lossj}�@

error_Rq�=?

learning_rate_1{ht7���I       6%�	Ե�6���A�*;


total_loss�}�@

error_R�;?

learning_rate_1{ht7�2-�I       6%�	���6���A�*;


total_loss�{�@

error_RT�I?

learning_rate_1{ht7V�o*I       6%�	[@�6���A�*;


total_loss�џ@

error_R�vc?

learning_rate_1{ht7��I       6%�	��6���A�*;


total_lossLVA

error_R��@?

learning_rate_1{ht7�ڸ�I       6%�	dы6���A�*;


total_loss�o�@

error_RCUK?

learning_rate_1{ht7���I       6%�	K�6���A�*;


total_losse;�@

error_R��G?

learning_rate_1{ht7���I       6%�	 i�6���A�*;


total_loss-��@

error_RC�B?

learning_rate_1{ht7�kr�I       6%�	綌6���A�*;


total_loss� �@

error_R�i>?

learning_rate_1{ht7z��6I       6%�	���6���A�*;


total_loss}��@

error_R�d?

learning_rate_1{ht7m~"�I       6%�	�B�6���A�*;


total_loss��@

error_Rs�L?

learning_rate_1{ht7�~iI       6%�	≍6���A�*;


total_loss�̛@

error_R��K?

learning_rate_1{ht7o�Q�I       6%�	F͍6���A�*;


total_loss�m�@

error_R�1P?

learning_rate_1{ht7���I       6%�	��6���A�*;


total_loss@3�@

error_R��K?

learning_rate_1{ht7��'�I       6%�	�U�6���A�*;


total_loss�.�@

error_R��X?

learning_rate_1{ht7=$@I       6%�	䛎6���A�*;


total_loss�L�@

error_R�NN?

learning_rate_1{ht7��&I       6%�	F��6���A�*;


total_loss���@

error_R\�O?

learning_rate_1{ht7�|��I       6%�	:&�6���A�*;


total_loss��@

error_R�X?

learning_rate_1{ht7X��I       6%�	�l�6���A�*;


total_loss�%�@

error_R4%P?

learning_rate_1{ht7�z�I       6%�	���6���A�*;


total_loss$��@

error_R��S?

learning_rate_1{ht7��j|I       6%�	���6���A�*;


total_loss��@

error_R}ua?

learning_rate_1{ht7�E| I       6%�	�>�6���A�*;


total_lossn��@

error_R�YX?

learning_rate_1{ht7��DI       6%�	���6���A�*;


total_loss#�@

error_RE:?

learning_rate_1{ht74�lI       6%�	GĐ6���A�*;


total_loss���@

error_R!N?

learning_rate_1{ht7O�6jI       6%�	��6���A�*;


total_loss�&�@

error_R��Q?

learning_rate_1{ht7O���I       6%�	.K�6���A�*;


total_loss}˩@

error_R�K?

learning_rate_1{ht7���I       6%�	���6���A�*;


total_loss!�@

error_R1M?

learning_rate_1{ht7�',*I       6%�	nԑ6���A�*;


total_loss�4�@

error_R��K?

learning_rate_1{ht7
��I       6%�	��6���A�*;


total_loss�6�@

error_R��N?

learning_rate_1{ht7�!��I       6%�	/_�6���A�*;


total_loss={�@

error_R�N;?

learning_rate_1{ht716��I       6%�	ߣ�6���A�*;


total_loss*ݍ@

error_R)cY?

learning_rate_1{ht7�"�I       6%�	k�6���A�*;


total_loss`�@

error_R�_?

learning_rate_1{ht7Ð�I       6%�	-�6���A�*;


total_losse�@

error_R�]?

learning_rate_1{ht7����I       6%�	tq�6���A�*;


total_loss΃�@

error_R�B?

learning_rate_1{ht7��I       6%�	㴓6���A�*;


total_loss���@

error_R�[X?

learning_rate_1{ht7<�JI       6%�	���6���A�*;


total_loss�~�@

error_R	*G?

learning_rate_1{ht7t@b/I       6%�	+B�6���A�*;


total_loss]��@

error_R\�T?

learning_rate_1{ht7o�r�I       6%�	���6���A�*;


total_loss�j�@

error_R�?G?

learning_rate_1{ht7+��rI       6%�	h̔6���A�*;


total_loss�ҡ@

error_RRH?

learning_rate_1{ht7=GJ_I       6%�	|<�6���A�*;


total_lossZ��@

error_R$�>?

learning_rate_1{ht7l�#�I       6%�	!��6���A�*;


total_loss��@

error_R��B?

learning_rate_1{ht7� 7�I       6%�	0ʕ6���A�*;


total_loss�ڹ@

error_R��I?

learning_rate_1{ht7�Ǆ�I       6%�	��6���A�*;


total_loss�
�@

error_RN-??

learning_rate_1{ht7��@I       6%�	�[�6���A�*;


total_loss���@

error_R��P?

learning_rate_1{ht7���I       6%�	���6���A�*;


total_loss]��@

error_Rfs0?

learning_rate_1{ht7}ZQbI       6%�	��6���A�*;


total_loss�ޑ@

error_RWD?

learning_rate_1{ht7���I       6%�	�4�6���A�*;


total_lossZ��@

error_R.9W?

learning_rate_1{ht7M��I       6%�	�}�6���A�*;


total_lossQ;�@

error_R\�G?

learning_rate_1{ht7P��I       6%�	��6���A�*;


total_loss|T�@

error_R��B?

learning_rate_1{ht7�,�I       6%�	k�6���A�*;


total_loss�ҕ@

error_R;0L?

learning_rate_1{ht7��vI       6%�	?H�6���A�*;


total_loss�6�@

error_R��P?

learning_rate_1{ht77�I       6%�		��6���A�*;


total_loss���@

error_R�J?

learning_rate_1{ht7��YI       6%�	�՘6���A�*;


total_loss��A

error_R�!J?

learning_rate_1{ht7͜�JI       6%�	�6���A�*;


total_loss�~�@

error_R�wQ?

learning_rate_1{ht7s1�LI       6%�	$f�6���A�*;


total_loss=�@

error_R�*=?

learning_rate_1{ht7��j6I       6%�	⮙6���A�*;


total_loss�+�@

error_R�<M?

learning_rate_1{ht7�)�nI       6%�	���6���A�*;


total_loss�y�@

error_R:�C?

learning_rate_1{ht7s9vI       6%�	5J�6���A�*;


total_loss�z�@

error_R{�W?

learning_rate_1{ht7���$I       6%�	ߎ�6���A�*;


total_loss?m�@

error_RCoK?

learning_rate_1{ht7��I       6%�	�՚6���A�*;


total_loss�@

error_R�"7?

learning_rate_1{ht7�o7I       6%�	f�6���A�*;


total_loss)	A

error_R�W?

learning_rate_1{ht7�>�I       6%�	�f�6���A�*;


total_lossE\�@

error_R�Q?

learning_rate_1{ht7��5-I       6%�	���6���A�*;


total_lossy�@

error_Rq�\?

learning_rate_1{ht7��sxI       6%�	���6���A�*;


total_loss���@

error_R�cE?

learning_rate_1{ht7�3�I       6%�	rM�6���A�*;


total_loss���@

error_RkQ?

learning_rate_1{ht7��1I       6%�	��6���A�*;


total_loss���@

error_R
i??

learning_rate_1{ht7Q�@I       6%�	�ܜ6���A�*;


total_lossQ�@

error_R�O5?

learning_rate_1{ht7��ƿI       6%�	y!�6���A�*;


total_loss�k�@

error_R$rS?

learning_rate_1{ht7VggGI       6%�	�b�6���A�*;


total_loss�,�@

error_R��d?

learning_rate_1{ht7%mX�I       6%�	G��6���A�*;


total_loss�P�@

error_RE�K?

learning_rate_1{ht7v(,�I       6%�	'�6���A�*;


total_loss{�@

error_RX�_?

learning_rate_1{ht7�'��I       6%�	g2�6���A�*;


total_loss�O�@

error_R�^E?

learning_rate_1{ht7�5I       6%�	Xz�6���A�*;


total_loss���@

error_RD]h?

learning_rate_1{ht7_4~�I       6%�	��6���A�*;


total_loss���@

error_Rl2X?

learning_rate_1{ht7R�TI       6%�	�6���A�*;


total_loss=M�@

error_R��O?

learning_rate_1{ht7J�=aI       6%�	SL�6���A�*;


total_loss��@

error_R�b?

learning_rate_1{ht7�WB�I       6%�	N��6���A�*;


total_loss��@

error_Rq4T?

learning_rate_1{ht7�RzI       6%�	|ݟ6���A�*;


total_loss��@

error_Rd�E?

learning_rate_1{ht7��sI       6%�	$�6���A�*;


total_loss4�@

error_R�K@?

learning_rate_1{ht7kg�JI       6%�	�o�6���A�*;


total_lossn��@

error_R_,U?

learning_rate_1{ht7����I       6%�	��6���A�*;


total_loss$��@

error_R�#F?

learning_rate_1{ht7��5�I       6%�	�6���A�*;


total_lossg4�@

error_R�FK?

learning_rate_1{ht7����I       6%�	�S�6���A�*;


total_lossq��@

error_R��V?

learning_rate_1{ht7�uZ�I       6%�	ٝ�6���A�*;


total_loss�sj@

error_R_GI?

learning_rate_1{ht7@9�I       6%�	d�6���A�*;


total_loss�ۯ@

error_R��I?

learning_rate_1{ht7X�7LI       6%�	�*�6���A�*;


total_loss� �@

error_R�S?

learning_rate_1{ht7ݱ��I       6%�	�r�6���A�*;


total_lossL��@

error_R�	R?

learning_rate_1{ht7���I       6%�	'��6���A�*;


total_loss�̿@

error_R�!T?

learning_rate_1{ht7_Te�I       6%�	��6���A�*;


total_loss�,�@

error_R&AL?

learning_rate_1{ht7��gI       6%�	�f�6���A�*;


total_loss�G�@

error_R�C?

learning_rate_1{ht7�~��I       6%�	���6���A�*;


total_lossDƋ@

error_R�MC?

learning_rate_1{ht71T��I       6%�	��6���A�*;


total_lossM�2A

error_R:�??

learning_rate_1{ht7�q�pI       6%�	�Q�6���A�*;


total_losst$�@

error_R_�O?

learning_rate_1{ht7�m��I       6%�	�ɤ6���A�*;


total_loss��@

error_R�A?

learning_rate_1{ht7��|&I       6%�	�9�6���A�*;


total_loss�@

error_R� D?

learning_rate_1{ht7Ia�I       6%�	���6���A�*;


total_loss��@

error_RWv\?

learning_rate_1{ht7�Q`6I       6%�	s��6���A�*;


total_loss�ߧ@

error_R�?N?

learning_rate_1{ht7%�a�I       6%�	�K�6���A�*;


total_loss ��@

error_RqK?

learning_rate_1{ht70}�I       6%�	�ʦ6���A�*;


total_loss�@

error_RkQ?

learning_rate_1{ht7OkR�I       6%�	�6���A�*;


total_loss��@

error_RH�K?

learning_rate_1{ht7����I       6%�	S\�6���A�*;


total_loss���@

error_R��3?

learning_rate_1{ht7��8�I       6%�	�ʧ6���A�*;


total_loss<��@

error_R��J?

learning_rate_1{ht7O��mI       6%�	��6���A�*;


total_loss���@

error_RH�6?

learning_rate_1{ht7�I�I       6%�	\�6���A�*;


total_loss.��@

error_R��V?

learning_rate_1{ht7ſ��I       6%�	VŨ6���A�*;


total_loss��@

error_R��E?

learning_rate_1{ht7�v6�I       6%�	i�6���A�*;


total_loss�/�@

error_RH�T?

learning_rate_1{ht7�P5I       6%�	XT�6���A�*;


total_loss�N�@

error_R�fG?

learning_rate_1{ht7R��I       6%�	磩6���A�*;


total_loss��A

error_R�ha?

learning_rate_1{ht7���I       6%�	��6���A�*;


total_loss���@

error_R��9?

learning_rate_1{ht7׻�PI       6%�	�W�6���A�*;


total_lossR2�@

error_Rl^E?

learning_rate_1{ht7�ڗI       6%�	n��6���A�*;


total_loss�O3A

error_R�N?

learning_rate_1{ht70���I       6%�	�6���A�*;


total_loss�`�@

error_R�
K?

learning_rate_1{ht7t�C�I       6%�	�X�6���A�*;


total_loss�w�@

error_R$_?

learning_rate_1{ht7ε��I       6%�	���6���A�*;


total_loss�/�@

error_R�[T?

learning_rate_1{ht7iڢ?I       6%�	q�6���A�*;


total_loss1Oa@

error_R��M?

learning_rate_1{ht7�s��I       6%�	�W�6���A�*;


total_loss�z�@

error_R�]I?

learning_rate_1{ht7���I       6%�	��6���A�*;


total_loss��@

error_R�hP?

learning_rate_1{ht7�Z�I       6%�	�	�6���A�*;


total_loss��	A

error_Rn%P?

learning_rate_1{ht7X�U�I       6%�	SU�6���A�*;


total_lossD��@

error_R�F?

learning_rate_1{ht7k-� I       6%�	=��6���A�*;


total_loss�?�@

error_R3bG?

learning_rate_1{ht7IK �I       6%�	��6���A�*;


total_loss���@

error_R
w??

learning_rate_1{ht7�Rm@I       6%�	�U�6���A�*;


total_loss���@

error_R�KM?

learning_rate_1{ht7�O5�I       6%�	���6���A�*;


total_loss��@

error_R�~N?

learning_rate_1{ht7ưtI       6%�	��6���A�*;


total_loss�ޥ@

error_Ro�H?

learning_rate_1{ht7�ɸ�I       6%�	C_�6���A�*;


total_loss�c�@

error_R��1?

learning_rate_1{ht7`���I       6%�	���6���A�*;


total_loss��@

error_R%�\?

learning_rate_1{ht7�{f�I       6%�	��6���A�*;


total_losst��@

error_RO�W?

learning_rate_1{ht7�;	I       6%�	2U�6���A�*;


total_loss���@

error_Ri�X?

learning_rate_1{ht7���I       6%�	��6���A�*;


total_lossD"�@

error_Rq<?

learning_rate_1{ht7d�F�I       6%�	n�6���A�*;


total_loss���@

error_RLT?

learning_rate_1{ht7h��I       6%�	�0�6���A�*;


total_loss��@

error_Ri�Y?

learning_rate_1{ht7�J1I       6%�	gw�6���A�*;


total_loss�v
A

error_RsD?

learning_rate_1{ht7�t��I       6%�	���6���A�*;


total_lossxԌ@

error_R�hM?

learning_rate_1{ht7��}�I       6%�	��6���A�*;


total_loss:��@

error_R��R?

learning_rate_1{ht7�^�I       6%�	�H�6���A�*;


total_lossI�@

error_R=�[?

learning_rate_1{ht7p�I       6%�	��6���A�*;


total_loss�'�@

error_R�g?

learning_rate_1{ht7Q��UI       6%�	в6���A�*;


total_loss�P�@

error_RSI?

learning_rate_1{ht7�0�I       6%�	&�6���A�*;


total_loss4#�@

error_Rz]?

learning_rate_1{ht7�sfI       6%�	�Z�6���A�*;


total_loss���@

error_R�O?

learning_rate_1{ht7�bZI       6%�	���6���A�*;


total_loss�Ѳ@

error_R��L?

learning_rate_1{ht75�}I       6%�	H�6���A�*;


total_loss)z�@

error_R��K?

learning_rate_1{ht77���I       6%�	j*�6���A�*;


total_lossH��@

error_R�G?

learning_rate_1{ht7�e��I       6%�	�k�6���A�*;


total_loss@�@

error_R�cU?

learning_rate_1{ht7�~VI       6%�	в�6���A�*;


total_loss�A

error_R�*<?

learning_rate_1{ht7Q�[VI       6%�	:��6���A�*;


total_lossO�@

error_R\:Q?

learning_rate_1{ht7��I=I       6%�	�[�6���A�*;


total_loss�N�@

error_R
F>?

learning_rate_1{ht7��}�I       6%�	���6���A�*;


total_loss�@

error_R!mX?

learning_rate_1{ht7��I       6%�	��6���A�*;


total_loss�k@

error_R�SL?

learning_rate_1{ht7X��I       6%�	S3�6���A�*;


total_loss���@

error_Rn�V?

learning_rate_1{ht7��cI       6%�	�w�6���A�*;


total_loss���@

error_R��c?

learning_rate_1{ht7J=x�I       6%�	f��6���A�*;


total_loss��A

error_R�%O?

learning_rate_1{ht7;�*I       6%�	��6���A�*;


total_loss�iA

error_R��0?

learning_rate_1{ht7���I       6%�	�A�6���A�*;


total_loss�I@

error_RP?

learning_rate_1{ht7?��I       6%�	���6���A�*;


total_loss�@

error_Rffc?

learning_rate_1{ht74B'uI       6%�	�̷6���A�*;


total_loss�N�@

error_R�}V?

learning_rate_1{ht7W�uTI       6%�	'�6���A�*;


total_lossla�@

error_R��J?

learning_rate_1{ht7��I.I       6%�	�e�6���A�*;


total_loss�n�@

error_Rz�I?

learning_rate_1{ht7�2�BI       6%�	L��6���A�*;


total_loss�v�@

error_RMe?

learning_rate_1{ht7Z�bI       6%�	���6���A�*;


total_loss2�@

error_R�%K?

learning_rate_1{ht78�9WI       6%�	�@�6���A�*;


total_loss`��@

error_R�[\?

learning_rate_1{ht7#ޗ�I       6%�	��6���A�*;


total_loss�y�@

error_R�V?

learning_rate_1{ht7�]�4I       6%�	ȹ6���A�*;


total_loss���@

error_R�L?

learning_rate_1{ht7���I       6%�	�6���A�*;


total_loss�$�@

error_R-�Z?

learning_rate_1{ht7����I       6%�	�O�6���A�*;


total_loss���@

error_R��L?

learning_rate_1{ht7�li�I       6%�	��6���A�*;


total_loss�R�@

error_R{�R?

learning_rate_1{ht7��ҝI       6%�	�ֺ6���A�*;


total_loss-��@

error_RaH?

learning_rate_1{ht7�iI       6%�	6�6���A�*;


total_lossꕚ@

error_R�jK?

learning_rate_1{ht7��c
I       6%�	�_�6���A�*;


total_loss̸�@

error_Rn�U?

learning_rate_1{ht7�\�I       6%�	̨�6���A�*;


total_loss�A

error_R-QH?

learning_rate_1{ht79��6I       6%�	��6���A�*;


total_loss�B�@

error_R�\D?

learning_rate_1{ht7�v�I       6%�	7k�6���A�*;


total_loss�u�@

error_R�&P?

learning_rate_1{ht7�$]!I       6%�	���6���A�*;


total_loss�B�@

error_R;�=?

learning_rate_1{ht7.礧I       6%�	��6���A�*;


total_lossI�@

error_R@�9?

learning_rate_1{ht72Pd�I       6%�	-H�6���A�*;


total_loss� A

error_R4�I?

learning_rate_1{ht7�HI       6%�	O��6���A�*;


total_loss���@

error_R�MI?

learning_rate_1{ht7�7�I       6%�	Q۽6���A�*;


total_lossZ�@

error_R�w8?

learning_rate_1{ht7��RkI       6%�	#�6���A�*;


total_lossF��@

error_R�!T?

learning_rate_1{ht7d��I       6%�	i�6���A�*;


total_loss���@

error_R1�[?

learning_rate_1{ht7>��I       6%�	د�6���A�*;


total_loss��@

error_R�pR?

learning_rate_1{ht7Jl��I       6%�	���6���A�*;


total_loss{r}@

error_R�M?

learning_rate_1{ht7 ��I       6%�	�C�6���A�*;


total_loss3�@

error_R$`F?

learning_rate_1{ht7_���I       6%�	x��6���A�*;


total_loss��@

error_RT2i?

learning_rate_1{ht7����I       6%�	�Ͽ6���A�*;


total_loss�Ȩ@

error_R�`?

learning_rate_1{ht7V��I       6%�	�6���A�*;


total_loss/#A

error_R�nM?

learning_rate_1{ht7��	I       6%�	=U�6���A�*;


total_lossP��@

error_R��V?

learning_rate_1{ht7[��I       6%�	��6���A�*;


total_loss
C�@

error_R�B?

learning_rate_1{ht7S�I       6%�	���6���A�*;


total_lossd�y@

error_R��A?

learning_rate_1{ht7��-$I       6%�	&�6���A�*;


total_loss��@

error_R��.?

learning_rate_1{ht7=4{I       6%�	�l�6���A�*;


total_loss&�@

error_Rv6\?

learning_rate_1{ht7n��qI       6%�	W��6���A�*;


total_loss��@

error_RV?

learning_rate_1{ht7]�|$I       6%�	��6���A�*;


total_loss=�@

error_R�K?

learning_rate_1{ht7{%1�I       6%�	�L�6���A�*;


total_loss��@

error_R�mP?

learning_rate_1{ht7���`I       6%�	,��6���A�*;


total_loss��@

error_R�P?

learning_rate_1{ht7���jI       6%�	���6���A�*;


total_lossS�@

error_R��R?

learning_rate_1{ht7�9��I       6%�	X�6���A�*;


total_lossW �@

error_R3F?

learning_rate_1{ht72D��I       6%�	�^�6���A�*;


total_lossx��@

error_R��F?

learning_rate_1{ht7��I       6%�	��6���A�*;


total_loss=?�@

error_R��O?

learning_rate_1{ht7�x_oI       6%�	~��6���A�*;


total_losse��@

error_R�2M?

learning_rate_1{ht7hf[CI       6%�	8�6���A�*;


total_loss9�@

error_R��Z?

learning_rate_1{ht7���I       6%�	��6���A�*;


total_lossZ`�@

error_RN:?

learning_rate_1{ht7txEI       6%�	P��6���A�*;


total_loss�ap@

error_R&3:?

learning_rate_1{ht7�� I       6%�	�,�6���A�*;


total_loss���@

error_R��G?

learning_rate_1{ht7�em\I       6%�	�w�6���A�*;


total_loss̘�@

error_R,U?

learning_rate_1{ht7[��I       6%�	��6���A�*;


total_loss�A

error_R��I?

learning_rate_1{ht7#�x�I       6%�	��6���A�*;


total_lossm��@

error_R�U?

learning_rate_1{ht7��v�I       6%�	�D�6���A�*;


total_loss6wA

error_RMXG?

learning_rate_1{ht7���hI       6%�	��6���A�*;


total_lossw��@

error_R�hQ?

learning_rate_1{ht7j� QI       6%�	���6���A�*;


total_loss�5�@

error_R��U?

learning_rate_1{ht7P���I       6%�	��6���A�*;


total_loss�5�@

error_R��<?

learning_rate_1{ht7׻�I       6%�	b�6���A�*;


total_loss�L�@

error_R6mM?

learning_rate_1{ht7 �tI       6%�	���6���A�*;


total_loss�<�@

error_R@�L?

learning_rate_1{ht7�[aGI       6%�	u��6���A�*;


total_loss&A�@

error_Rd�Y?

learning_rate_1{ht7�:�I       6%�		6�6���A�*;


total_loss�å@

error_R��O?

learning_rate_1{ht7Y{m�I       6%�	�{�6���A�*;


total_loss��@

error_R��X?

learning_rate_1{ht7F�#I       6%�	���6���A�*;


total_loss�)�@

error_R�9\?

learning_rate_1{ht7 �#.I       6%�	��6���A�*;


total_loss��@

error_Rd�H?

learning_rate_1{ht7�UvI       6%�	�L�6���A�*;


total_lossX̂@

error_Rn_F?

learning_rate_1{ht7�5b�I       6%�	M��6���A�*;


total_loss��@

error_R9M?

learning_rate_1{ht7=}��I       6%�	���6���A�*;


total_loss߶@

error_R�A?

learning_rate_1{ht7`:�tI       6%�	��6���A�*;


total_loss��@

error_Rhh?

learning_rate_1{ht7ux�'I       6%�	�Z�6���A�*;


total_loss��@

error_R�bC?

learning_rate_1{ht7HYԌI       6%�	���6���A�*;


total_loss/թ@

error_RM9P?

learning_rate_1{ht7)�I       6%�	���6���A�*;


total_lossޟA

error_RnY?

learning_rate_1{ht7�K�I       6%�	E&�6���A�*;


total_loss���@

error_R�U?

learning_rate_1{ht7�MBI       6%�	,l�6���A�*;


total_loss$l�@

error_RťL?

learning_rate_1{ht7]ہI       6%�	ҵ�6���A�*;


total_lossW�m@

error_RN�P?

learning_rate_1{ht7cB��I       6%�	���6���A�*;


total_loss;��@

error_R�J?

learning_rate_1{ht7�Q��I       6%�	R>�6���A�*;


total_loss�׺@

error_Rx�\?

learning_rate_1{ht7��eI       6%�	;��6���A�*;


total_loss��@

error_R��L?

learning_rate_1{ht74|I       6%�	@��6���A�*;


total_loss�ў@

error_R�?F?

learning_rate_1{ht7�{=�I       6%�	��6���A�*;


total_lossP�@

error_Rq,^?

learning_rate_1{ht7�w5eI       6%�	X�6���A�*;


total_loss4��@

error_R��E?

learning_rate_1{ht7��QI       6%�	���6���A�*;


total_loss�^�@

error_R=:>?

learning_rate_1{ht7:sz.I       6%�	���6���A�*;


total_lossԦ�@

error_R��G?

learning_rate_1{ht7�f�I       6%�	�"�6���A�*;


total_loss��@

error_R��6?

learning_rate_1{ht7��- I       6%�	[h�6���A�*;


total_lossF��@

error_R#�Z?

learning_rate_1{ht7rM��I       6%�	Ԭ�6���A�*;


total_loss7d�@

error_Rs�A?

learning_rate_1{ht7��UtI       6%�	���6���A�*;


total_loss�A

error_R�zG?

learning_rate_1{ht7�`��I       6%�	�3�6���A�*;


total_loss���@

error_R��P?

learning_rate_1{ht7�pbI       6%�	�x�6���A�*;


total_loss!A

error_Rp[?

learning_rate_1{ht7e�rWI       6%�	ͼ�6���A�*;


total_loss�3�@

error_R%>2?

learning_rate_1{ht74�I       6%�	A�6���A�*;


total_loss�%�@

error_R�R?

learning_rate_1{ht7Sn�I       6%�	(S�6���A�*;


total_loss�h�@

error_RL'O?

learning_rate_1{ht7�@LI       6%�	���6���A�*;


total_loss�R�@

error_RO[?

learning_rate_1{ht7=ԱI       6%�	��6���A�*;


total_loss�;�@

error_R�bW?

learning_rate_1{ht7���I       6%�	�.�6���A�*;


total_loss1��@

error_R�^?

learning_rate_1{ht7Gr"8I       6%�	�x�6���A�*;


total_lossIH�@

error_R
L?

learning_rate_1{ht7�4�I       6%�	y��6���A�*;


total_loss��@

error_R�J?

learning_rate_1{ht7��V�I       6%�	� �6���A�*;


total_loss�A

error_R��J?

learning_rate_1{ht7_���I       6%�	8h�6���A�*;


total_lossr�]@

error_R�LK?

learning_rate_1{ht7a��mI       6%�	���6���A�*;


total_lossУ�@

error_Rn�M?

learning_rate_1{ht7D��/I       6%�	E��6���A�*;


total_loss��@

error_R��6?

learning_rate_1{ht7�M�[I       6%�	HF�6���A�*;


total_loss��@

error_R��X?

learning_rate_1{ht7��kI       6%�	���6���A�*;


total_loss���@

error_RQU?

learning_rate_1{ht7�UiI       6%�	���6���A�*;


total_loss�R�@

error_R��Y?

learning_rate_1{ht7_�A�I       6%�	U�6���A�*;


total_loss�l�@

error_R_P?

learning_rate_1{ht7�STI       6%�	n\�6���A�*;


total_loss�@

error_R=Z?

learning_rate_1{ht7���I       6%�	3��6���A�*;


total_losss�@

error_R*HB?

learning_rate_1{ht7��I       6%�	���6���A�*;


total_loss��@

error_R>?

learning_rate_1{ht7m���I       6%�	 N�6���A�*;


total_loss2A

error_Rd2O?

learning_rate_1{ht7���dI       6%�	��6���A�*;


total_loss8�@

error_Rl?]?

learning_rate_1{ht7�k�I       6%�	���6���A�*;


total_lossF�@

error_R�	S?

learning_rate_1{ht7:�(I       6%�	��6���A�*;


total_loss��@

error_RȰF?

learning_rate_1{ht7���9I       6%�	�e�6���A�*;


total_loss҄�@

error_RE�=?

learning_rate_1{ht7a�9�I       6%�	7��6���A�*;


total_loss�Ԕ@

error_R�)P?

learning_rate_1{ht7�H��I       6%�	���6���A�*;


total_loss���@

error_R{�P?

learning_rate_1{ht7nyI       6%�	O/�6���A�*;


total_loss��A

error_R�TB?

learning_rate_1{ht7�6��I       6%�	av�6���A�*;


total_loss8�@

error_RM�^?

learning_rate_1{ht7�tI       6%�	���6���A�*;


total_lossWG�@

error_R�hN?

learning_rate_1{ht7=2�I       6%�	���6���A�*;


total_lossSp@

error_R�aW?

learning_rate_1{ht7�5�I       6%�	�@�6���A�*;


total_lossk�A

error_R�uY?

learning_rate_1{ht7]|n�I       6%�	���6���A�*;


total_loss�{@

error_RֈS?

learning_rate_1{ht7y�-2I       6%�	���6���A�*;


total_loss�
A

error_RvCJ?

learning_rate_1{ht7�+I       6%�	�6���A�*;


total_loss֐�@

error_R��1?

learning_rate_1{ht7h/�6I       6%�	a`�6���A�*;


total_loss`�@

error_RitJ?

learning_rate_1{ht7�^�&I       6%�	<��6���A�*;


total_loss[��@

error_RMAB?

learning_rate_1{ht7òW�I       6%�	���6���A�*;


total_loss[��@

error_R��L?

learning_rate_1{ht79�]I       6%�	T1�6���A�*;


total_loss͢A

error_RLI?

learning_rate_1{ht7M~��I       6%�	^x�6���A�*;


total_loss6��@

error_R�_L?

learning_rate_1{ht7����I       6%�	]��6���A�*;


total_lossXF�@

error_R��T?

learning_rate_1{ht7��d�I       6%�	�6���A�*;


total_loss��@

error_R��^?

learning_rate_1{ht7���BI       6%�	c�6���A�*;


total_loss�Y�@

error_R DO?

learning_rate_1{ht7�_�I       6%�	���6���A�*;


total_lossF��@

error_R�S?

learning_rate_1{ht7ߓR�I       6%�	��6���A�*;


total_loss<�@

error_R��]?

learning_rate_1{ht7�Y�2I       6%�	�l�6���A�*;


total_loss��@

error_R��-?

learning_rate_1{ht7��NI       6%�	���6���A�*;


total_loss�(m@

error_R��F?

learning_rate_1{ht7��Z;I       6%�	[U�6���A�*;


total_loss u}@

error_R=>?

learning_rate_1{ht7��ϩI       6%�	#��6���A�*;


total_loss4��@

error_R�p@?

learning_rate_1{ht7E��I       6%�	-&�6���A�*;


total_loss���@

error_RdxJ?

learning_rate_1{ht7�L=�I       6%�	Kk�6���A�*;


total_loss�"�@

error_R�C5?

learning_rate_1{ht7��>LI       6%�	���6���A�*;


total_loss�.�@

error_R��c?

learning_rate_1{ht7ruD�I       6%�	�5�6���A�*;


total_loss:��@

error_R_�F?

learning_rate_1{ht7{�_�I       6%�	���6���A�*;


total_loss���@

error_R72Q?

learning_rate_1{ht7�#��I       6%�	9��6���A�*;


total_lossh�@

error_R�@E?

learning_rate_1{ht7�CʩI       6%�	F�6���A�*;


total_loss��@

error_R<�S?

learning_rate_1{ht7��7I       6%�	)��6���A�*;


total_loss���@

error_R�O?

learning_rate_1{ht7��l�I       6%�	��6���A�*;


total_loss}FA

error_R�:?

learning_rate_1{ht7P��bI       6%�	�x�6���A�*;


total_loss���@

error_R\JC?

learning_rate_1{ht7z�xGI       6%�	(��6���A�*;


total_loss���@

error_R��B?

learning_rate_1{ht7iYI       6%�	"!�6���A�*;


total_loss�K�@

error_R�!L?

learning_rate_1{ht7��WI       6%�	��6���A�*;


total_loss/ �@

error_R_Ba?

learning_rate_1{ht7���I       6%�	��6���A�*;


total_lossF	�@

error_R��W?

learning_rate_1{ht7ag �I       6%�	g�6���A�*;


total_lossZ��@

error_R,�O?

learning_rate_1{ht7%meI       6%�	!r�6���A�*;


total_loss�1�@

error_R��T?

learning_rate_1{ht7n�ˣI       6%�	���6���A�*;


total_loss�{�@

error_R@aG?

learning_rate_1{ht7��I       6%�	1��6���A�*;


total_lossS��@

error_R��W?

learning_rate_1{ht7��.<I       6%�	G^�6���A�*;


total_lossAU�@

error_R�;P?

learning_rate_1{ht7�X,I       6%�	̨�6���A�*;


total_loss���@

error_R�]R?

learning_rate_1{ht7at�I       6%�	���6���A�*;


total_loss���@

error_Rf	V?

learning_rate_1{ht7���FI       6%�	�\�6���A�*;


total_losss�@

error_RɧG?

learning_rate_1{ht7?5_I       6%�	���6���A�*;


total_loss�=�@

error_R��X?

learning_rate_1{ht7�|�iI       6%�	��6���A�*;


total_lossP�@

error_Rl�@?

learning_rate_1{ht7&=^5I       6%�	=�6���A�*;


total_lossX@�@

error_R��N?

learning_rate_1{ht7~�E�I       6%�	,��6���A�*;


total_lossሇ@

error_R��:?

learning_rate_1{ht7!�u;I       6%�	\��6���A�*;


total_loss�W�@

error_R��]?

learning_rate_1{ht7��/I       6%�	��6���A�*;


total_loss\�f@

error_R,�O?

learning_rate_1{ht7��fI       6%�	N�6���A�*;


total_loss�|�@

error_R=�N?

learning_rate_1{ht7��ȿI       6%�	t��6���A�*;


total_loss)ɔ@

error_RCX>?

learning_rate_1{ht7�o��I       6%�	Z��6���A�*;


total_loss}�@

error_R�7?

learning_rate_1{ht7K�dI       6%�	z�6���A�*;


total_lossH��@

error_R��W?

learning_rate_1{ht70)w�I       6%�	e�6���A�*;


total_lossZ��@

error_R�;@?

learning_rate_1{ht7�:�I       6%�	��6���A�*;


total_loss\۽@

error_Rv�P?

learning_rate_1{ht7��I       6%�	���6���A�*;


total_loss�$�@

error_R�G?

learning_rate_1{ht7�1�I       6%�	9G�6���A�*;


total_loss�p�@

error_R�:X?

learning_rate_1{ht7�Z��I       6%�	���6���A�*;


total_loss��@

error_R��m?

learning_rate_1{ht7Uh7�I       6%�	��6���A�*;


total_loss��@

error_R�;?

learning_rate_1{ht7�g�I       6%�	�H�6���A�*;


total_loss-��@

error_RfF?

learning_rate_1{ht78��\I       6%�	���6���A�*;


total_loss��@

error_Rۿ]?

learning_rate_1{ht7�,��I       6%�	���6���A�*;


total_loss��@

error_R!�L?

learning_rate_1{ht7��I       6%�	t)�6���A�*;


total_lossLѝ@

error_R�D?

learning_rate_1{ht7wd�II       6%�	n�6���A�*;


total_loss�7A

error_R$??

learning_rate_1{ht7yE�kI       6%�	ı�6���A�*;


total_lossi�@

error_RM?

learning_rate_1{ht7�3I       6%�		��6���A�*;


total_lossx�@

error_RtA?

learning_rate_1{ht7�s9'I       6%�	�9�6���A�*;


total_lossgM�@

error_RI{K?

learning_rate_1{ht7<O@�I       6%�	�}�6���A�*;


total_loss��@

error_R��K?

learning_rate_1{ht7*I       6%�	V��6���A�*;


total_loss�:�@

error_R�kH?

learning_rate_1{ht7�"��I       6%�	��6���A�*;


total_loss*�@

error_Rq�S?

learning_rate_1{ht7��v9I       6%�	zh�6���A�*;


total_loss#��@

error_RJ\?

learning_rate_1{ht7m��I       6%�	���6���A�*;


total_loss]�@

error_R_�D?

learning_rate_1{ht7�,�I       6%�	���6���A�*;


total_loss���@

error_R��P?

learning_rate_1{ht7~�VI       6%�	E?�6���A�*;


total_lossk,A

error_R�H?

learning_rate_1{ht7=N�I       6%�	ʃ�6���A�*;


total_loss��@

error_R�=?

learning_rate_1{ht7��I       6%�	E��6���A�*;


total_loss��@

error_R!'=?

learning_rate_1{ht7 SB�I       6%�	��6���A�*;


total_lossQ��@

error_RNN?

learning_rate_1{ht7!�G�I       6%�	
Y�6���A�*;


total_lossOF�@

error_RH;N?

learning_rate_1{ht75I       6%�	���6���A�*;


total_loss�[�@

error_R�C@?

learning_rate_1{ht7��aI       6%�	���6���A�*;


total_loss�	A

error_R �B?

learning_rate_1{ht7C오I       6%�	:,�6���A�*;


total_loss���@

error_R&�K?

learning_rate_1{ht7�u�I       6%�	Qp�6���A�*;


total_loss,
�@

error_RήO?

learning_rate_1{ht7�)��I       6%�	;��6���A�*;


total_loss�@

error_RįF?

learning_rate_1{ht7+_\I       6%�	6��6���A�*;


total_loss�;�@

error_R(ON?

learning_rate_1{ht7"��'I       6%�	�9�6���A�*;


total_loss�8�@

error_R�N?

learning_rate_1{ht7Ir�I       6%�	�}�6���A�*;


total_loss��@

error_R�UK?

learning_rate_1{ht7��bI       6%�	��6���A�*;


total_loss?��@

error_R@�C?

learning_rate_1{ht7�L~&I       6%�	3�6���A�*;


total_loss<�@

error_R�CT?

learning_rate_1{ht7a3I       6%�	bH�6���A�*;


total_loss�V�@

error_R,T?

learning_rate_1{ht7�J�I       6%�	'��6���A�*;


total_loss�~�@

error_R*}K?

learning_rate_1{ht76�w�I       6%�	���6���A�*;


total_loss�q�@

error_R�N?

learning_rate_1{ht7 49I       6%�	'�6���A�*;


total_loss�A

error_R
�V?

learning_rate_1{ht7�mrI       6%�		Y�6���A�*;


total_loss;C�@

error_RMe?

learning_rate_1{ht7K8.�I       6%�	y��6���A�*;


total_lossW�A

error_RC�X?

learning_rate_1{ht7|�I       6%�	���6���A�*;


total_loss���@

error_R��V?

learning_rate_1{ht7�BB�I       6%�	�,�6���A�*;


total_loss%�~@

error_RωE?

learning_rate_1{ht76�I       6%�	�t�6���A�*;


total_loss��@

error_R��N?

learning_rate_1{ht7.�3�I       6%�	'��6���A�*;


total_loss$��@

error_R�S?

learning_rate_1{ht7�#��I       6%�	$�6���A�*;


total_loss�@

error_R97?

learning_rate_1{ht7��aI       6%�	~e�6���A�*;


total_lossmj�@

error_R��Y?

learning_rate_1{ht7um�(I       6%�	U��6���A�*;


total_loss,&�@

error_R hL?

learning_rate_1{ht7���I       6%�	'��6���A�*;


total_loss�R@

error_R�\<?

learning_rate_1{ht7���I       6%�	73�6���A�*;


total_loss���@

error_R�W?

learning_rate_1{ht7vL�I       6%�	'|�6���A�*;


total_loss_�@

error_RE�A?

learning_rate_1{ht7�6I       6%�	¿�6���A�*;


total_loss�}A

error_RH�=?

learning_rate_1{ht75��I       6%�		�6���A�*;


total_lossĲ�@

error_R��K?

learning_rate_1{ht7�x3I       6%�	�F�6���A�*;


total_loss�(�@

error_R��B?

learning_rate_1{ht7013UI       6%�	׉�6���A�*;


total_loss۵�@

error_R)�M?

learning_rate_1{ht7\�~�I       6%�	���6���A�*;


total_loss���@

error_R�T?

learning_rate_1{ht7�@�I       6%�	D�6���A�*;


total_loss�s�@

error_R�><?

learning_rate_1{ht7#��;I       6%�	�P�6���A�*;


total_loss<E�@

error_RR>J?

learning_rate_1{ht7~��*I       6%�	���6���A�*;


total_loss��@

error_R{�e?

learning_rate_1{ht7h?I       6%�	��6���A�*;


total_loss��@

error_Rڦ6?

learning_rate_1{ht7�@��I       6%�	��6���A�*;


total_lossE�@

error_R��O?

learning_rate_1{ht7�w�I       6%�	\�6���A�*;


total_loss���@

error_R�F?

learning_rate_1{ht7� �^I       6%�	<��6���A�*;


total_losse_�@

error_R��W?

learning_rate_1{ht7� �I       6%�	J��6���A�*;


total_loss4A�@

error_RVS?

learning_rate_1{ht7ܖ�I       6%�	'�6���A�*;


total_loss��@

error_R�?[?

learning_rate_1{ht7��בI       6%�	hn�6���A�*;


total_loss��@

error_Rx|_?

learning_rate_1{ht7����I       6%�	[��6���A�*;


total_loss���@

error_R�8S?

learning_rate_1{ht7�AF�I       6%�	F��6���A�*;


total_loss�g�@

error_RC�F?

learning_rate_1{ht7�[-I       6%�	JE�6���A�*;


total_losss��@

error_R <O?

learning_rate_1{ht71�{�I       6%�	}��6���A�*;


total_loss��@

error_R�L?

learning_rate_1{ht7P���I       6%�	#��6���A�*;


total_loss'�@

error_R8�M?

learning_rate_1{ht7���I       6%�	H�6���A�*;


total_loss!��@

error_R��@?

learning_rate_1{ht7�`#sI       6%�	XS�6���A�*;


total_lossq��@

error_RH�X?

learning_rate_1{ht7�}
I       6%�	���6���A�*;


total_lossL6�@

error_R6�H?

learning_rate_1{ht7�_��I       6%�	|��6���A�*;


total_lossv̆@

error_R��:?

learning_rate_1{ht7rٔI       6%�	j�6���A�*;


total_loss��@

error_RS�P?

learning_rate_1{ht7� �I       6%�	�a�6���A�*;


total_loss� �@

error_R��D?

learning_rate_1{ht7��I       6%�	���6���A�*;


total_loss&�@

error_R��B?

learning_rate_1{ht7Y �I       6%�	���6���A�*;


total_loss*�A

error_R��V?

learning_rate_1{ht7�I       6%�	<�6���A�*;


total_loss�c�@

error_Rf�Y?

learning_rate_1{ht7�K��I       6%�	�~�6���A�*;


total_loss��@

error_R�_?

learning_rate_1{ht7pə�I       6%�	���6���A�*;


total_loss�K�@

error_R6�S?

learning_rate_1{ht7A�|I       6%�	��6���A�*;


total_loss�e�@

error_R�uK?

learning_rate_1{ht7e�w�I       6%�	mI�6���A�*;


total_loss��@

error_R��I?

learning_rate_1{ht7 ��I       6%�	 ��6���A�*;


total_loss���@

error_R&�C?

learning_rate_1{ht7�NW�I       6%�	���6���A�*;


total_loss��@

error_R��1?

learning_rate_1{ht7mvӁI       6%�	� 7���A�*;


total_loss���@

error_Rxf`?

learning_rate_1{ht7!��^I       6%�	CY 7���A�*;


total_loss��@

error_R��N?

learning_rate_1{ht71���I       6%�	�� 7���A�*;


total_loss1	�@

error_Rf�V?

learning_rate_1{ht7H�d�I       6%�	3� 7���A�*;


total_loss�C�@

error_R�RT?

learning_rate_1{ht7t��I       6%�	�$7���A�*;


total_loss��@

error_R�5G?

learning_rate_1{ht7W�C�I       6%�	�k7���A�*;


total_loss"�@

error_R[I?

learning_rate_1{ht7ֵuxI       6%�	ۯ7���A�*;


total_lossl�@

error_Rs�B?

learning_rate_1{ht7H+��I       6%�	��7���A�*;


total_loss���@

error_R�S?

learning_rate_1{ht7�6u�I       6%�	/H7���A�*;


total_loss3��@

error_RW�T?

learning_rate_1{ht7�E׍I       6%�	Ԕ7���A�*;


total_loss`��@

error_R��j?

learning_rate_1{ht7�el�I       6%�	��7���A�*;


total_lossa��@

error_R��L?

learning_rate_1{ht7��0I       6%�	l!7���A�*;


total_loss�q�@

error_R]?

learning_rate_1{ht7Mc�I       6%�	:g7���A�*;


total_lossX��@

error_R2lM?

learning_rate_1{ht7� �I       6%�	�7���A�*;


total_loss�'�@

error_R�Y?

learning_rate_1{ht7��W�I       6%�	�7���A�*;


total_loss��@

error_R%;;?

learning_rate_1{ht7P�'�I       6%�	�47���A�*;


total_lossځ�@

error_RM�[?

learning_rate_1{ht7����I       6%�	�v7���A�*;


total_loss��@

error_R��I?

learning_rate_1{ht7d�m�I       6%�	��7���A�*;


total_loss7k�@

error_RڭR?

learning_rate_1{ht7f'I       6%�	�7���A�*;


total_loss���@

error_REQ?

learning_rate_1{ht77T�I       6%�	-Y7���A�*;


total_lossl��@

error_R5`?

learning_rate_1{ht7�2I       6%�	9�7���A�*;


total_lossZ_�@

error_R8tR?

learning_rate_1{ht7TǑ�I       6%�	��7���A�*;


total_lossZ�@

error_R�@Q?

learning_rate_1{ht7K�DkI       6%�	� 7���A�*;


total_loss�D�@

error_R[i>?

learning_rate_1{ht7p�;VI       6%�	�b7���A�*;


total_loss�@

error_R}�I?

learning_rate_1{ht7�5<I       6%�	Ť7���A�*;


total_loss �@

error_R�ZR?

learning_rate_1{ht7��I       6%�	Y�7���A�*;


total_loss���@

error_RfyE?

learning_rate_1{ht7�'JwI       6%�	q%7���A�*;


total_loss�U�@

error_R��T?

learning_rate_1{ht7�A7�I       6%�	1c7���A�*;


total_loss�jA

error_R�`?

learning_rate_1{ht7���I       6%�	�7���A�*;


total_loss�e�@

error_R E?

learning_rate_1{ht7���*I       6%�	|�7���A�*;


total_lossڈ@

error_R�M[?

learning_rate_1{ht7a�3�I       6%�	�$7���A�*;


total_loss=��@

error_R �J?

learning_rate_1{ht7����I       6%�	�e7���A�*;


total_loss�8�@

error_RRWR?

learning_rate_1{ht7k<CI       6%�	��7���A�*;


total_loss�*�@

error_R��>?

learning_rate_1{ht7�_I       6%�	B�7���A�*;


total_loss���@

error_R�Wb?

learning_rate_1{ht7&�I       6%�	9'	7���A�*;


total_losse�@

error_R&GY?

learning_rate_1{ht7��;�I       6%�	Yf	7���A�*;


total_losse�@

error_Rv�:?

learning_rate_1{ht7��J�I       6%�	L�	7���A�*;


total_lossou�@

error_R�}Q?

learning_rate_1{ht7��|�I       6%�	j�	7���A�*;


total_loss�_�@

error_R��]?

learning_rate_1{ht7P�r'I       6%�	t0
7���A�*;


total_loss��@

error_R ?L?

learning_rate_1{ht7m���I       6%�	Tu
7���A�*;


total_loss[��@

error_Rq�Q?

learning_rate_1{ht7��I       6%�	�
7���A�*;


total_loss�N�@

error_RaL9?

learning_rate_1{ht7;U�I       6%�	��
7���A�*;


total_loss3��@

error_R��W?

learning_rate_1{ht7���zI       6%�	B7���A�*;


total_lossO�@

error_R�O?

learning_rate_1{ht7��z�I       6%�	S�7���A�*;


total_loss<M�@

error_R�K:?

learning_rate_1{ht7V���I       6%�	��7���A�*;


total_loss  �@

error_R1:I?

learning_rate_1{ht7�T mI       6%�	�7���A�*;


total_loss��@

error_R(�]?

learning_rate_1{ht7���I       6%�	�Q7���A�*;


total_loss���@

error_R_�X?

learning_rate_1{ht7�c|XI       6%�	ԙ7���A�*;


total_loss�@

error_RZ�:?

learning_rate_1{ht77lJ�I       6%�	��7���A�*;


total_lossܜ�@

error_R}�b?

learning_rate_1{ht7��McI       6%�	�!7���A�*;


total_loss@'�@

error_R!9K?

learning_rate_1{ht7M�OI       6%�	j7���A�*;


total_lossz��@

error_R��U?

learning_rate_1{ht7~��QI       6%�	R�7���A�*;


total_loss,c@

error_R]zr?

learning_rate_1{ht7� I       6%�	��7���A�*;


total_loss�"u@

error_Rc@=?

learning_rate_1{ht7R�SI       6%�	�/7���A�*;


total_loss
��@

error_R�^T?

learning_rate_1{ht7����I       6%�	bp7���A�*;


total_loss_U�@

error_RaC?

learning_rate_1{ht7[Om�I       6%�	\�7���A�*;


total_lossʴ�@

error_R=�^?

learning_rate_1{ht7xC�(I       6%�	��7���A�*;


total_lossY�@

error_RIV?

learning_rate_1{ht7�� OI       6%�	E47���A�*;


total_loss���@

error_R�sA?

learning_rate_1{ht7f�I       6%�	u7���A�*;


total_loss�3�@

error_R�Z;?

learning_rate_1{ht7�*,I       6%�	�7���A�*;


total_loss	��@

error_R&�\?

learning_rate_1{ht7'7��I       6%�	E�7���A�*;


total_loss��@

error_R�/;?

learning_rate_1{ht7��&MI       6%�	*87���A�*;


total_lossM��@

error_R��P?

learning_rate_1{ht7�l�eI       6%�	fy7���A�*;


total_lossg�@

error_R�SL?

learning_rate_1{ht7"8��I       6%�	�7���A�*;


total_loss���@

error_R��M?

learning_rate_1{ht7�A��I       6%�	��7���A�*;


total_loss-�@

error_R$D?

learning_rate_1{ht7W2jhI       6%�	<7���A�*;


total_lossl��@

error_R$cN?

learning_rate_1{ht7�H�wI       6%�	v�7���A�*;


total_loss�>�@

error_R�%V?

learning_rate_1{ht7�n�I       6%�	�7���A�*;


total_loss���@

error_R��K?

learning_rate_1{ht7F�~I       6%�	�7���A�*;


total_loss���@

error_R�cN?

learning_rate_1{ht7�9׮I       6%�	�L7���A�*;


total_loss��@

error_R�C?

learning_rate_1{ht7���I       6%�	7�7���A�*;


total_losss"�@

error_Rq^?

learning_rate_1{ht7��I       6%�	<�7���A�*;


total_loss��@

error_R�fM?

learning_rate_1{ht71�I       6%�	B7���A�*;


total_loss�Y�@

error_Rf?

learning_rate_1{ht7Mr�I       6%�	L[7���A�*;


total_loss%$�@

error_R.�N?

learning_rate_1{ht7Ջ`�I       6%�	\�7���A�*;


total_lossq�A

error_R�)V?

learning_rate_1{ht7���mI       6%�	��7���A�*;


total_loss�Ӄ@

error_RU?

learning_rate_1{ht7&f�+I       6%�	�7���A�*;


total_loss�D�@

error_R!�F?

learning_rate_1{ht7��P�I       6%�	�X7���A�*;


total_loss���@

error_R�O?

learning_rate_1{ht7m�lI       6%�	l�7���A�*;


total_loss�w�@

error_R��C?

learning_rate_1{ht7��K�I       6%�	T�7���A�*;


total_loss[m�@

error_R}�E?

learning_rate_1{ht7�G�dI       6%�	�B7���A�*;


total_loss���@

error_R��D?

learning_rate_1{ht75d��I       6%�	��7���A�*;


total_loss��@

error_R��M?

learning_rate_1{ht7�;̾I       6%�	��7���A�*;


total_loss�̞@

error_R�M?

learning_rate_1{ht7n�K7I       6%�	{
7���A�*;


total_loss��@

error_R�??

learning_rate_1{ht7a�>�I       6%�	�J7���A�*;


total_loss&�@

error_RlEL?

learning_rate_1{ht7 ķ�I       6%�	ȍ7���A�*;


total_lossK@�@

error_R�C?

learning_rate_1{ht7�jG	I       6%�	��7���A�*;


total_loss�ë@

error_R!sX?

learning_rate_1{ht7�0%I       6%�	�7���A�*;


total_lossi��@

error_R��[?

learning_rate_1{ht7H�YI       6%�	�T7���A�*;


total_loss�"�@

error_R��[?

learning_rate_1{ht7er�I       6%�	��7���A�*;


total_loss� �@

error_R�L?

learning_rate_1{ht7*i��I       6%�	n�7���A�*;


total_lossSc�@

error_R��G?

learning_rate_1{ht7�$�I       6%�	�7���A�*;


total_lossWU�@

error_R��@?

learning_rate_1{ht7��;hI       6%�	�_7���A�*;


total_loss�X}@

error_R�#P?

learning_rate_1{ht7�*+TI       6%�	4�7���A�*;


total_loss�@

error_RF0X?

learning_rate_1{ht7.�B�I       6%�	 �7���A�*;


total_loss�C�@

error_Rz	H?

learning_rate_1{ht7h�@I       6%�	!7���A�*;


total_loss�W�@

error_R�MT?

learning_rate_1{ht7�u�I       6%�	`c7���A�*;


total_lossn�A

error_R�i?

learning_rate_1{ht7x�t"I       6%�	��7���A�*;


total_loss���@

error_R�zN?

learning_rate_1{ht7��QaI       6%�	7�7���A�*;


total_loss#״@

error_R�Wc?

learning_rate_1{ht7Z�i�I       6%�	�$7���A�*;


total_loss�S�@

error_Rx�O?

learning_rate_1{ht7��VtI       6%�	?g7���A�*;


total_loss��@

error_R�E;?

learning_rate_1{ht7���I       6%�	(�7���A�*;


total_loss;՘@

error_RWS?

learning_rate_1{ht7���I       6%�	��7���A�*;


total_lossŋ�@

error_Rq�K?

learning_rate_1{ht7�D�5I       6%�	�/7���A�*;


total_loss�^�@

error_R�1D?

learning_rate_1{ht7�~|I       6%�	�t7���A�*;


total_loss�}z@

error_R�]E?

learning_rate_1{ht7�	��I       6%�	�7���A�*;


total_loss�e�@

error_R�K?

learning_rate_1{ht7񻸙I       6%�	7���A�*;


total_loss�@

error_R�A?

learning_rate_1{ht7A�ܳI       6%�	`G7���A�*;


total_loss��@

error_R\�G?

learning_rate_1{ht7/kI       6%�	'�7���A�*;


total_loss��@

error_R�1P?

learning_rate_1{ht7��C�I       6%�	r�7���A�*;


total_lossH��@

error_R��X?

learning_rate_1{ht7:��aI       6%�	:7���A�*;


total_loss�[�@

error_R}�6?

learning_rate_1{ht7R��I       6%�	�Z7���A�*;


total_lossً@

error_RJ#I?

learning_rate_1{ht7QzEI       6%�	�7���A�*;


total_loss"�@

error_R#�G?

learning_rate_1{ht7BZ�"I       6%�	 �7���A�*;


total_loss;��@

error_R�V?

learning_rate_1{ht7��]I       6%�	�$7���A�*;


total_loss?��@

error_R[e?

learning_rate_1{ht7(ݩ�I       6%�	�f7���A�*;


total_loss�;�@

error_R�R?

learning_rate_1{ht7[PŋI       6%�	��7���A�*;


total_loss�P�@

error_R�8J?

learning_rate_1{ht7��L�I       6%�	��7���A�*;


total_loss���@

error_R�HO?

learning_rate_1{ht7���I       6%�	�.7���A�*;


total_loss$��@

error_R2pF?

learning_rate_1{ht7�JmI       6%�	�q7���A�*;


total_loss,��@

error_R�jZ?

learning_rate_1{ht7�>XI       6%�	�7���A�*;


total_loss��a@

error_R��J?

learning_rate_1{ht7��I       6%�	��7���A�*;


total_loss� �@

error_RW�W?

learning_rate_1{ht7�p�I       6%�	{; 7���A�*;


total_loss�]�@

error_R}l?

learning_rate_1{ht7���I       6%�	�| 7���A�*;


total_loss{ �@

error_R��??

learning_rate_1{ht7S`� I       6%�	Ծ 7���A�*;


total_loss���@

error_R;,T?

learning_rate_1{ht7|jI       6%�	!7���A�*;


total_loss#��@

error_R��@?

learning_rate_1{ht7��Q�I       6%�	LC!7���A�*;


total_lossCި@

error_R�V?

learning_rate_1{ht7���I       6%�	Ӆ!7���A�*;


total_loss���@

error_R4fX?

learning_rate_1{ht7E�T"I       6%�	<�!7���A�*;


total_lossx�A

error_R�'D?

learning_rate_1{ht7$��BI       6%�	�"7���A�*;


total_loss1�@

error_R��P?

learning_rate_1{ht7��p�I       6%�	�\"7���A�*;


total_loss�u�@

error_RMW?

learning_rate_1{ht7*mׅI       6%�	.�"7���A�*;


total_loss���@

error_R:kT?

learning_rate_1{ht7���I       6%�	��"7���A�*;


total_lossuޞ@

error_R��<?

learning_rate_1{ht7,�4)I       6%�	U+#7���A�*;


total_loss|��@

error_Rͅa?

learning_rate_1{ht7 ���I       6%�	�l#7���A�*;


total_loss���@

error_R;�G?

learning_rate_1{ht7�WTrI       6%�	��#7���A�*;


total_loss��@

error_RJ�H?

learning_rate_1{ht7����I       6%�	'�#7���A�*;


total_lossj}�@

error_RO
S?

learning_rate_1{ht7�I       6%�	V6$7���A�*;


total_lossJ�@

error_R6JR?

learning_rate_1{ht7���I       6%�	�}$7���A�*;


total_loss��JA

error_R�MQ?

learning_rate_1{ht7=C2�I       6%�	��$7���A�*;


total_loss�G�@

error_R��B?

learning_rate_1{ht7^6II       6%�	�%7���A�*;


total_loss�M�@

error_RתY?

learning_rate_1{ht7A�qI       6%�	;j%7���A�*;


total_loss���@

error_R#�G?

learning_rate_1{ht7��abI       6%�	ֱ%7���A�*;


total_loss�v�@

error_R(�\?

learning_rate_1{ht7����I       6%�	/�%7���A�*;


total_losso�@

error_R�9D?

learning_rate_1{ht7����I       6%�	::&7���A�*;


total_loss�A�@

error_R�^X?

learning_rate_1{ht7�-�I       6%�	�y&7���A�*;


total_loss2��@

error_R�H?

learning_rate_1{ht7��I       6%�	#�&7���A�*;


total_loss�m�@

error_R�W?

learning_rate_1{ht7�>I�I       6%�	��&7���A�*;


total_loss��@

error_R�FV?

learning_rate_1{ht7���I       6%�	�<'7���A�*;


total_loss㪥@

error_RA?

learning_rate_1{ht7��mI       6%�	�}'7���A�*;


total_loss$�@

error_R;�E?

learning_rate_1{ht7#=�I       6%�	�'7���A�*;


total_loss;!�@

error_RQB?

learning_rate_1{ht7�*>I       6%�		(7���A�*;


total_lossv�@

error_R��M?

learning_rate_1{ht7 0!�I       6%�	)O(7���A�*;


total_loss�_�@

error_R_I?

learning_rate_1{ht7��5�I       6%�	��(7���A�*;


total_loss[^�@

error_R�I?

learning_rate_1{ht7$.�>I       6%�	U�(7���A�*;


total_loss: �@

error_R�V?

learning_rate_1{ht7��hcI       6%�	Z)7���A�*;


total_lossJ��@

error_RR]?

learning_rate_1{ht78�WI       6%�	�e)7���A�*;


total_loss�Xq@

error_RJcW?

learning_rate_1{ht7���I       6%�	A�)7���A�*;


total_lossڀ�@

error_R�5b?

learning_rate_1{ht7ū�fI       6%�	�)7���A�*;


total_lossq��@

error_R��W?

learning_rate_1{ht7��	I       6%�	//*7���A�*;


total_loss���@

error_R�I?

learning_rate_1{ht7���I       6%�	�p*7���A�*;


total_loss�C�@

error_R�E?

learning_rate_1{ht731�I       6%�	�*7���A�*;


total_loss���@

error_R�7E?

learning_rate_1{ht7����I       6%�	��*7���A�*;


total_losst��@

error_R�Z?

learning_rate_1{ht7��nI       6%�	�?+7���A�*;


total_loss6A

error_R��D?

learning_rate_1{ht7fE8�I       6%�	;�+7���A�*;


total_lossV��@

error_R��R?

learning_rate_1{ht77��tI       6%�	��+7���A�*;


total_loss�gA

error_R�.Q?

learning_rate_1{ht7��I       6%�	�.,7���A�*;


total_loss���@

error_R��H?

learning_rate_1{ht7��I       6%�	�|,7���A�*;


total_loss�@

error_R
H?

learning_rate_1{ht7 TI       6%�	i�,7���A�*;


total_loss�@

error_Rc�T?

learning_rate_1{ht70ЈI       6%�	-7���A�*;


total_loss@�@

error_RH�I?

learning_rate_1{ht7�,�I       6%�	�X-7���A�*;


total_lossAq�@

error_R�oF?

learning_rate_1{ht7�P��I       6%�	f�-7���A�*;


total_losso�@

error_Rf�9?

learning_rate_1{ht77\��I       6%�	�-7���A�*;


total_lossc��@

error_R8�U?

learning_rate_1{ht7�+�pI       6%�	,(.7���A�*;


total_lossǶ@

error_RcU?

learning_rate_1{ht7�|�`I       6%�	m.7���A�*;


total_loss��@

error_RY?

learning_rate_1{ht7`@�#I       6%�	Ү.7���A�*;


total_lossZjA

error_Rm�V?

learning_rate_1{ht7�oT�I       6%�	��.7���A�*;


total_loss��@

error_RA�^?

learning_rate_1{ht7���I       6%�	�4/7���A�*;


total_loss�a�@

error_Rڧ;?

learning_rate_1{ht7�HɬI       6%�	v/7���A�*;


total_lossd��@

error_R��D?

learning_rate_1{ht7]K�I       6%�	з/7���A�*;


total_loss8�@

error_R�:N?

learning_rate_1{ht7���I       6%�	j07���A�*;


total_lossN%�@

error_R�??

learning_rate_1{ht7���I       6%�	O07���A�*;


total_loss��@

error_R�J4?

learning_rate_1{ht7�v�I       6%�	�07���A�*;


total_loss`8�@

error_R��C?

learning_rate_1{ht7�eN�I       6%�	b�07���A�*;


total_loss��@

error_R��\?

learning_rate_1{ht7~u��I       6%�	�17���A�*;


total_loss�d�@

error_R\cE?

learning_rate_1{ht7��I       6%�	TY17���A�*;


total_lossᐠ@

error_R8�I?

learning_rate_1{ht7��I       6%�	0�17���A�*;


total_lossT��@

error_R��<?

learning_rate_1{ht7)���I       6%�	��17���A�*;


total_loss�Т@

error_R7�R?

learning_rate_1{ht7 2�I       6%�	�27���A�*;


total_lossw�@

error_R �S?

learning_rate_1{ht7��2�I       6%�	�`27���A�*;


total_loss<Ȼ@

error_RL�Y?

learning_rate_1{ht7^��I       6%�	ء27���A�*;


total_loss,x�@

error_Rş[?

learning_rate_1{ht7&ul8I       6%�	��27���A�*;


total_loss=�@

error_Rl J?

learning_rate_1{ht7� ��I       6%�	2*37���A�*;


total_loss�(�@

error_R�0I?

learning_rate_1{ht7{ܕI       6%�	�q37���A�*;


total_loss��y@

error_RfEB?

learning_rate_1{ht7���I       6%�	�37���A�*;


total_lossQ��@

error_R&�R?

learning_rate_1{ht7����I       6%�	��37���A�*;


total_loss3�@

error_R�9O?

learning_rate_1{ht75�t}I       6%�	�<47���A�*;


total_lossڣ}@

error_R(O;?

learning_rate_1{ht7q���I       6%�	P�47���A�*;


total_loss6]�@

error_R{sS?

learning_rate_1{ht7xjI       6%�	��47���A�*;


total_loss�S�@

error_RJ2I?

learning_rate_1{ht7A~�I       6%�	p57���A�*;


total_loss�{�@

error_Ro�T?

learning_rate_1{ht7>�t@I       6%�	a|57���A�*;


total_loss	��@

error_R�]?

learning_rate_1{ht7�ŊTI       6%�	��57���A�*;


total_loss�o�@

error_R��F?

learning_rate_1{ht7��9�I       6%�	� 67���A�*;


total_loss�ß@

error_RERA?

learning_rate_1{ht7���I       6%�	(C67���A�*;


total_loss���@

error_RjV\?

learning_rate_1{ht7~�&:I       6%�	ă67���A�*;


total_loss��@

error_R�WK?

learning_rate_1{ht7� �|I       6%�	��67���A�*;


total_loss���@

error_RS�9?

learning_rate_1{ht7p]��I       6%�	'77���A�*;


total_loss��@

error_R(I?

learning_rate_1{ht7��I       6%�	�M77���A�*;


total_loss�A

error_R,�L?

learning_rate_1{ht7d��SI       6%�	��77���A�*;


total_loss:��@

error_R��I?

learning_rate_1{ht7L��I       6%�	��77���A�*;


total_lossM�@

error_R7�P?

learning_rate_1{ht7��V+I       6%�	i87���A�*;


total_lossS��@

error_R�eK?

learning_rate_1{ht7� 	�I       6%�	�Q87���A�*;


total_loss�6�@

error_R�A?

learning_rate_1{ht7b�y	I       6%�	"�87���A�*;


total_loss,/�@

error_R�'F?

learning_rate_1{ht7��]vI       6%�	f�87���A�*;


total_lossZ	�@

error_R hR?

learning_rate_1{ht7
^VI       6%�	�97���A�*;


total_loss-�@

error_RZ?

learning_rate_1{ht7�)\"I       6%�	h[97���A�*;


total_loss���@

error_R1�j?

learning_rate_1��j7Xb��I       6%�	Y�97���A�*;


total_loss*C�@

error_R8[?

learning_rate_1��j7
���I       6%�	��97���A�*;


total_loss�R�@

error_RѥO?

learning_rate_1��j7�XhI       6%�	d�<7���A�*;


total_loss��@

error_R�$L?

learning_rate_1��j7OBKI       6%�	(�<7���A�*;


total_loss���@

error_RtD?

learning_rate_1��j7T]Z
I       6%�	�C=7���A�*;


total_lossS��@

error_R��N?

learning_rate_1��j7�yW�I       6%�	6�=7���A�*;


total_loss���@

error_R�CS?

learning_rate_1��j7�\L�I       6%�	�=7���A�*;


total_loss]�@

error_R�>?

learning_rate_1��j7!֟�I       6%�	Q>7���A�*;


total_loss��@

error_Rc�B?

learning_rate_1��j7�j�&I       6%�	�Z>7���A�*;


total_loss>X�@

error_R�G?

learning_rate_1��j7}U��I       6%�	�>7���A�*;


total_loss�Ҩ@

error_R��c?

learning_rate_1��j7��t�I       6%�	��>7���A�*;


total_loss-�@

error_R
]D?

learning_rate_1��j7ap��I       6%�	-'?7���A�*;


total_loss#4�@

error_R��W?

learning_rate_1��j7z��I       6%�	l?7���A�*;


total_loss��@

error_R�aa?

learning_rate_1��j7��I       6%�	�?7���A�*;


total_loss��@

error_Rh�M?

learning_rate_1��j7{.k(I       6%�	K�?7���A�*;


total_loss��@

error_R��F?

learning_rate_1��j7�<[5I       6%�	b:@7���A�*;


total_loss���@

error_R;9L?

learning_rate_1��j7�Љ�I       6%�	�@7���A�*;


total_loss7�@

error_R��R?

learning_rate_1��j7��1DI       6%�	��@7���A�*;


total_loss���@

error_R�`R?

learning_rate_1��j78j�I       6%�	�A7���A�*;


total_loss�r�@

error_Rq�M?

learning_rate_1��j76��I       6%�	%UA7���A�*;


total_loss�f%A

error_R\�Z?

learning_rate_1��j7UFP�I       6%�	��A7���A�*;


total_loss��@

error_R��U?

learning_rate_1��j76��I       6%�	��A7���A�*;


total_loss_ �@

error_R�a?

learning_rate_1��j7�,I       6%�	O B7���A�*;


total_loss���@

error_R��7?

learning_rate_1��j74���I       6%�	�jB7���A�*;


total_loss
��@

error_R<G?

learning_rate_1��j7���I       6%�	U�B7���A�*;


total_lossu�@

error_R��R?

learning_rate_1��j7���I       6%�	b�B7���A�*;


total_loss��z@

error_Rn'Q?

learning_rate_1��j7@��I       6%�	�>C7���A�*;


total_loss���@

error_R��;?

learning_rate_1��j7�,c�I       6%�	�C7���A�*;


total_loss�n�@

error_R�hP?

learning_rate_1��j7��@I       6%�	��C7���A�*;


total_loss�N�@

error_R��_?

learning_rate_1��j7��jI       6%�	gD7���A�*;


total_loss�@

error_R_�R?

learning_rate_1��j7E��}I       6%�	`MD7���A�*;


total_loss��@

error_R��F?

learning_rate_1��j7e'I       6%�	��D7���A�*;


total_loss��@

error_RR_[?

learning_rate_1��j7")�*I       6%�	��D7���A�*;


total_loss�@

error_RfnQ?

learning_rate_1��j7��ۛI       6%�	� E7���A�*;


total_lossY�@

error_R.�Q?

learning_rate_1��j7\�;I       6%�	 uE7���A�*;


total_loss���@

error_R�]?

learning_rate_1��j7���I       6%�	ٶE7���A�*;


total_lossh�@

error_R�d?

learning_rate_1��j7��b�I       6%�	�E7���A�*;


total_loss8ڭ@

error_R��S?

learning_rate_1��j7X=��I       6%�	�<F7���A�*;


total_losshS�@

error_RifC?

learning_rate_1��j7C�tI       6%�	��F7���A�*;


total_loss��@

error_RTU?

learning_rate_1��j7�)�uI       6%�	�F7���A�*;


total_loss��@

error_RE�A?

learning_rate_1��j7q`��I       6%�	�G7���A�*;


total_loss�R�@

error_R-�B?

learning_rate_1��j7��vI       6%�	HG7���A�*;


total_loss�c�@

error_R��N?

learning_rate_1��j7" �I       6%�	�G7���A�*;


total_loss�g�@

error_R�O?

learning_rate_1��j7DZuII       6%�	��G7���A�*;


total_loss��@

error_R�;X?

learning_rate_1��j7^�'jI       6%�	NBH7���A�*;


total_lossC��@

error_Rs^?

learning_rate_1��j7�ƤI       6%�	��H7���A�*;


total_losso��@

error_R�H?

learning_rate_1��j7�TD�I       6%�	��H7���A�*;


total_loss2%A

error_R��Y?

learning_rate_1��j7�q�I       6%�	OI7���A�*;


total_loss ��@

error_RE�C?

learning_rate_1��j7����I       6%�	zbI7���A�*;


total_loss� �@

error_R��L?

learning_rate_1��j7�OE�I       6%�	��I7���A�*;


total_loss���@

error_R�
6?

learning_rate_1��j7C��MI       6%�	��I7���A�*;


total_loss���@

error_R�6Q?

learning_rate_1��j7|�*�I       6%�	�CJ7���A�*;


total_loss;-�@

error_R$�K?

learning_rate_1��j7�8�HI       6%�	��J7���A�*;


total_loss�ٚ@

error_RC.N?

learning_rate_1��j7i(��I       6%�	��J7���A�*;


total_loss(ʚ@

error_Rs�Q?

learning_rate_1��j7c�qI       6%�	yK7���A�*;


total_lossMn�@

error_RquJ?

learning_rate_1��j7�S�I       6%�	�QK7���A�*;


total_lossi"�@

error_R�RM?

learning_rate_1��j7��^ZI       6%�	]�K7���A�*;


total_loss$�@

error_R,�<?

learning_rate_1��j7.��I       6%�	;�K7���A�*;


total_lossj��@

error_R��I?

learning_rate_1��j7�)I       6%�	m?L7���A�*;


total_loss3f�@

error_R@M?

learning_rate_1��j7��<I       6%�	C�L7���A�*;


total_lossś�@

error_R��H?

learning_rate_1��j7@�XI       6%�	��L7���A�*;


total_loss.�h@

error_R�<?

learning_rate_1��j7�g�I       6%�	�5M7���A�*;


total_loss)�@

error_RQ�K?

learning_rate_1��j7�B��I       6%�	�M7���A�*;


total_loss��@

error_R�L?

learning_rate_1��j7�:sXI       6%�	L�M7���A�*;


total_loss�@

error_R�?L?

learning_rate_1��j7�]!I       6%�	�BN7���A�*;


total_loss}�@

error_R�9Q?

learning_rate_1��j7E{�VI       6%�	��N7���A�*;


total_loss���@

error_R �`?

learning_rate_1��j7C�.I       6%�	�O7���A�*;


total_loss�y�@

error_Rf1K?

learning_rate_1��j7|L
�I       6%�	1�O7���A�*;


total_lossܼ�@

error_RaX?

learning_rate_1��j7��I       6%�	�P7���A�*;


total_lossRZ�@

error_R&�P?

learning_rate_1��j7�WqI       6%�	 `P7���A�*;


total_loss��@

error_R�O?

learning_rate_1��j7��k,I       6%�	,�P7���A�*;


total_loss�&�@

error_R�N?

learning_rate_1��j71%��I       6%�	U)Q7���A�*;


total_loss'�@

error_R�.k?

learning_rate_1��j7_���I       6%�	D�Q7���A�*;


total_loss��@

error_R��F?

learning_rate_1��j7���}I       6%�	�R7���A�*;


total_lossf׼@

error_R�M?

learning_rate_1��j78�8I       6%�	�pR7���A�*;


total_loss�\�@

error_R$�c?

learning_rate_1��j7J4�I       6%�	H�R7���A�*;


total_loss�n�@

error_R�9U?

learning_rate_1��j7�9H�I       6%�	�S7���A�*;


total_loss�2y@

error_R��J?

learning_rate_1��j7�� I       6%�	Y�S7���A�*;


total_loss��@

error_Rf�K?

learning_rate_1��j7����I       6%�	��S7���A�*;


total_loss[|A

error_R�]L?

learning_rate_1��j7M�WI       6%�	I@T7���A�*;


total_loss(�@

error_R��E?

learning_rate_1��j7���I       6%�	��T7���A�*;


total_loss��@

error_R*G?

learning_rate_1��j7aL�OI       6%�	��T7���A�*;


total_loss��@

error_R[�S?

learning_rate_1��j7�)�I       6%�	3kU7���A�*;


total_loss��@

error_Rc�b?

learning_rate_1��j7����I       6%�	Z�U7���A�*;


total_loss�`�@

error_R��N?

learning_rate_1��j7���I       6%�	:V7���A�*;


total_lossh�@

error_R�UC?

learning_rate_1��j7�/@I       6%�	v_V7���A�*;


total_loss|i�@

error_R6/R?

learning_rate_1��j7h-:I       6%�	V�V7���A�*;


total_lossoԴ@

error_Rܴ^?

learning_rate_1��j7@-I       6%�	��V7���A�*;


total_loss��@

error_R��V?

learning_rate_1��j7Q��tI       6%�	YLW7���A�*;


total_loss��@

error_R�#U?

learning_rate_1��j7E&RI       6%�	R�W7���A�*;


total_loss�	�@

error_R��c?

learning_rate_1��j7��\I       6%�	Q�W7���A�*;


total_loss�U�@

error_R�sT?

learning_rate_1��j7*��KI       6%�	�!X7���A�*;


total_loss1Û@

error_R��Z?

learning_rate_1��j7�ЌI       6%�	fgX7���A�*;


total_loss���@

error_RQ=?

learning_rate_1��j7�HeI       6%�	��X7���A�*;


total_loss-Ƚ@

error_RO�K?

learning_rate_1��j7��Y,I       6%�	��X7���A�*;


total_loss�W@

error_R==F?

learning_rate_1��j7����I       6%�	(7Y7���A�*;


total_loss):�@

error_R�V?

learning_rate_1��j7�EI<I       6%�	s~Y7���A�*;


total_loss�Y�@

error_R��T?

learning_rate_1��j7�YI       6%�	��Y7���A�*;


total_loss�G�@

error_R1L?

learning_rate_1��j7��PI       6%�	=Z7���A�*;


total_loss>�@

error_Rn�T?

learning_rate_1��j7+�^�I       6%�	PZ7���A�*;


total_loss��@

error_R�	9?

learning_rate_1��j7��I       6%�	��Z7���A�*;


total_loss.��@

error_R]�^?

learning_rate_1��j7(j�I       6%�	_�Z7���A�*;


total_loss�R�@

error_Rv�A?

learning_rate_1��j7z�?I       6%�	D"[7���A�*;


total_loss܏@

error_R�TB?

learning_rate_1��j7����I       6%�	�e[7���A�*;


total_lossU�A

error_R �K?

learning_rate_1��j7"���I       6%�	�[7���A�*;


total_loss��]@

error_R��J?

learning_rate_1��j7����I       6%�	'�[7���A�*;


total_loss���@

error_R�w>?

learning_rate_1��j7j��EI       6%�	�5\7���A�*;


total_lossꉛ@

error_RqOP?

learning_rate_1��j7��%I       6%�	�w\7���A�*;


total_lossf�@

error_RZI?

learning_rate_1��j7f��iI       6%�	��\7���A�*;


total_loss���@

error_R�bT?

learning_rate_1��j7�^�I       6%�	]7���A�*;


total_loss���@

error_Rd�O?

learning_rate_1��j7�X,�I       6%�	EX]7���A�*;


total_loss�@

error_R�'V?

learning_rate_1��j7��`�I       6%�	�]7���A�*;


total_lossd��@

error_Ra{J?

learning_rate_1��j7�\*rI       6%�	��]7���A�*;


total_lossl	�@

error_R4B?

learning_rate_1��j7���|I       6%�	I2^7���A�*;


total_loss1��@

error_R1�K?

learning_rate_1��j7��(I       6%�	�w^7���A�*;


total_loss���@

error_R�H?

learning_rate_1��j7�,�I       6%�	߹^7���A�*;


total_loss�w�@

error_RN�i?

learning_rate_1��j7W8ztI       6%�	\ _7���A�*;


total_lossx�@

error_RPN?

learning_rate_1��j7���I       6%�	�F_7���A�*;


total_lossx��@

error_R}E?

learning_rate_1��j7��RmI       6%�	M�_7���A�*;


total_loss�@

error_Rv�N?

learning_rate_1��j7�` I       6%�	d�_7���A�*;


total_loss,�@

error_Rj�S?

learning_rate_1��j7����I       6%�	�`7���A�*;


total_loss���@

error_R�I?

learning_rate_1��j7J�>RI       6%�	�X`7���A�*;


total_loss���@

error_R��>?

learning_rate_1��j7�ƚ�I       6%�	�`7���A�*;


total_lossY;�@

error_R��X?

learning_rate_1��j7�)I       6%�	��`7���A�*;


total_loss�)�@

error_R�q8?

learning_rate_1��j7��?I       6%�	�ka7���A�*;


total_loss��@

error_Rr�`?

learning_rate_1��j7LT�HI       6%�	��a7���A�*;


total_loss�̻@

error_RoZ?

learning_rate_1��j7��WI       6%�	yb7���A�*;


total_loss���@

error_R�4H?

learning_rate_1��j70p.�I       6%�	�ib7���A�*;


total_loss;�@

error_R�}D?

learning_rate_1��j7	D��I       6%�	��b7���A�*;


total_loss���@

error_R��N?

learning_rate_1��j7���I       6%�	� c7���A�*;


total_lossF�@

error_R��D?

learning_rate_1��j7W���I       6%�	�pc7���A�*;


total_lossQU�@

error_R�M?

learning_rate_1��j7�;��I       6%�	��c7���A�*;


total_loss[@�@

error_R�fI?

learning_rate_1��j7���I       6%�	Rd7���A�*;


total_lossO�@

error_RßW?

learning_rate_1��j7�F_�I       6%�	Z�d7���A�*;


total_lossM�@

error_R!�J?

learning_rate_1��j7Q�I       6%�	��d7���A�*;


total_loss��@

error_R�A?

learning_rate_1��j7)�QI       6%�	le7���A�*;


total_loss��@

error_R GW?

learning_rate_1��j7��:I       6%�	��e7���A�*;


total_lossm��@

error_R8E?

learning_rate_1��j7���SI       6%�	f7���A�*;


total_loss��@

error_RiP?

learning_rate_1��j7��(I       6%�	��f7���A�*;


total_loss���@

error_RH�T?

learning_rate_1��j7���nI       6%�	�+g7���A�*;


total_lossȇ@

error_R�(S?

learning_rate_1��j7����I       6%�	�g7���A�*;


total_loss�W�@

error_R1lR?

learning_rate_1��j7�?I       6%�	h7���A�*;


total_lossE�~@

error_R�wI?

learning_rate_1��j7Y��	I       6%�	^h7���A�*;


total_lossH��@

error_R��:?

learning_rate_1��j7'�I       6%�	5�h7���A�*;


total_loss��@

error_R�J?

learning_rate_1��j79���I       6%�	:i7���A�*;


total_loss���@

error_R
�L?

learning_rate_1��j7�<t�I       6%�	o�i7���A�*;


total_loss���@

error_Rl�M?

learning_rate_1��j7'���I       6%�	��i7���A�*;


total_losso��@

error_R��B?

learning_rate_1��j7qD�I       6%�	,Aj7���A�*;


total_loss�|�@

error_R�C?

learning_rate_1��j7��=I       6%�	��j7���A�*;


total_loss�6�@

error_RH'J?

learning_rate_1��j7�=k;I       6%�	�$k7���A�*;


total_loss@�{@

error_R�-I?

learning_rate_1��j7<t�I       6%�	��k7���A�*;


total_loss���@

error_R)fT?

learning_rate_1��j7]A�I       6%�	�k7���A�*;


total_lossd��@

error_RRJ?

learning_rate_1��j7PL@I       6%�	jPl7���A�*;


total_loss��@

error_RM�H?

learning_rate_1��j7�%I       6%�	�l7���A�	*;


total_losso�@

error_R��K?

learning_rate_1��j7-$k�I       6%�	)�l7���A�	*;


total_loss�B�@

error_R��Q?

learning_rate_1��j7V^�WI       6%�	�um7���A�	*;


total_loss���@

error_R�*I?

learning_rate_1��j7�o�I       6%�	��m7���A�	*;


total_loss���@

error_R�P?

learning_rate_1��j7��I       6%�	�&n7���A�	*;


total_loss��@

error_Rv5O?

learning_rate_1��j7�(�wI       6%�	/�n7���A�	*;


total_loss��A

error_RH7Z?

learning_rate_1��j7�F��I       6%�	��n7���A�	*;


total_losso�A

error_R��J?

learning_rate_1��j7v�O�I       6%�	�9o7���A�	*;


total_loss�^�@

error_R�aZ?

learning_rate_1��j7��'wI       6%�	�o7���A�	*;


total_loss�\A

error_R��:?

learning_rate_1��j7���4I       6%�	�p7���A�	*;


total_loss���@

error_Rj:?

learning_rate_1��j7��hI       6%�	�Jp7���A�	*;


total_loss�C�@

error_R�	P?

learning_rate_1��j7��"I       6%�	+�p7���A�	*;


total_loss��@

error_R�Q]?

learning_rate_1��j7:��I       6%�	��p7���A�	*;


total_loss��@

error_R�E?

learning_rate_1��j7�&�I       6%�	�?q7���A�	*;


total_loss@

error_RHtM?

learning_rate_1��j7���I       6%�	��q7���A�	*;


total_lossͯ�@

error_R�\?

learning_rate_1��j7�fI       6%�	��q7���A�	*;


total_losso�U@

error_R4�A?

learning_rate_1��j7�hCI       6%�	�r7���A�	*;


total_loss�@

error_R�v\?

learning_rate_1��j7���YI       6%�	iVr7���A�	*;


total_loss��@

error_RE�\?

learning_rate_1��j7�/6I       6%�	͚r7���A�	*;


total_loss�Ӝ@

error_R��M?

learning_rate_1��j7Vr�7I       6%�	��r7���A�	*;


total_loss��@

error_R��R?

learning_rate_1��j7�!�"I       6%�	)*s7���A�	*;


total_loss�ي@

error_Rd�U?

learning_rate_1��j76� %I       6%�	�ys7���A�	*;


total_loss2�c@

error_R�X?

learning_rate_1��j71�l�I       6%�	��s7���A�	*;


total_loss��@

error_RU?

learning_rate_1��j7�d�I       6%�	�mt7���A�	*;


total_loss�m�@

error_R��W?

learning_rate_1��j7D9A�I       6%�	j�t7���A�	*;


total_loss#�@

error_Ro�C?

learning_rate_1��j7�y!�I       6%�	+yu7���A�	*;


total_loss�m�@

error_R��I?

learning_rate_1��j7�L�I       6%�	Ov7���A�	*;


total_loss)�@

error_R1�A?

learning_rate_1��j7���*I       6%�	��v7���A�	*;


total_loss��@

error_R�X?

learning_rate_1��j7��K�I       6%�	0-w7���A�	*;


total_loss���@

error_RdeI?

learning_rate_1��j7BP��I       6%�	�w7���A�	*;


total_loss���@

error_R�0<?

learning_rate_1��j7�{��I       6%�	�'x7���A�	*;


total_loss��A

error_R�cV?

learning_rate_1��j7;z
�I       6%�	�x7���A�	*;


total_loss#F�@

error_R�:U?

learning_rate_1��j77�W>I       6%�	y7���A�	*;


total_loss��@

error_R��T?

learning_rate_1��j7����I       6%�	�y7���A�	*;


total_loss�^�@

error_R��V?

learning_rate_1��j7��0I       6%�	�8z7���A�	*;


total_loss���@

error_R��R?

learning_rate_1��j7���?I       6%�	��z7���A�	*;


total_loss誖@

error_R��Y?

learning_rate_1��j7�&8�I       6%�	QQ{7���A�	*;


total_loss���@

error_R��N?

learning_rate_1��j7�r��I       6%�	:�{7���A�	*;


total_loss�A

error_R�$J?

learning_rate_1��j7a9]>I       6%�	�R|7���A�	*;


total_loss�V�@

error_R,
I?

learning_rate_1��j7磅ZI       6%�	��|7���A�	*;


total_lossC�@

error_R�=?

learning_rate_1��j7���^I       6%�	R}7���A�	*;


total_lossF�?A

error_R.rG?

learning_rate_1��j7��p�I       6%�	cz}7���A�	*;


total_loss7��@

error_R��K?

learning_rate_1��j7�Jx�I       6%�	~7���A�	*;


total_lossƻ�@

error_R��Q?

learning_rate_1��j7b�*I       6%�	�~7���A�	*;


total_lossN�@

error_R��J?

learning_rate_1��j7��QI       6%�	7
7���A�	*;


total_loss� �@

error_R�F?

learning_rate_1��j7�~�I       6%�	ڱ7���A�	*;


total_loss��o@

error_R^Y?

learning_rate_1��j7�ɶ�I       6%�	�:�7���A�	*;


total_loss���@

error_R�4G?

learning_rate_1��j7�.3�I       6%�	ϫ�7���A�	*;


total_loss� �@

error_RȈ=?

learning_rate_1��j7�I       6%�	�;�7���A�	*;


total_loss�]�@

error_R�ai?

learning_rate_1��j7�"CnI       6%�	���7���A�	*;


total_loss),�@

error_RNFQ?

learning_rate_1��j7x���I       6%�	"�7���A�	*;


total_loss���@

error_R,�@?

learning_rate_1��j7o;��I       6%�	���7���A�	*;


total_losszVA

error_R��K?

learning_rate_1��j7���I       6%�	,��7���A�	*;


total_loss�k�@

error_R�K?

learning_rate_1��j7����I       6%�	cA�7���A�	*;


total_loss�Ą@

error_R�J?

learning_rate_1��j7�íbI       6%�	گ�7���A�	*;


total_loss��@

error_R�%Q?

learning_rate_1��j7�I       6%�	�#�7���A�	*;


total_loss��@

error_R!�??

learning_rate_1��j7�ݽ%I       6%�	n��7���A�	*;


total_loss�\�@

error_R�8C?

learning_rate_1��j7���I       6%�	���7���A�	*;


total_loss�_�@

error_R!�N?

learning_rate_1��j7 �I       6%�	�V�7���A�	*;


total_loss!u�@

error_RaU?

learning_rate_1��j7�گ�I       6%�	=��7���A�	*;


total_loss���@

error_R��H?

learning_rate_1��j7��ָI       6%�	��7���A�	*;


total_loss/��@

error_R3�K?

learning_rate_1��j75��I       6%�	�A�7���A�	*;


total_loss�{�@

error_R��[?

learning_rate_1��j7�"F�I       6%�	���7���A�	*;


total_loss���@

error_R%�]?

learning_rate_1��j7u��*I       6%�	��7���A�	*;


total_lossIo�@

error_R��@?

learning_rate_1��j7���I       6%�	�7�7���A�	*;


total_loss*��@

error_R��f?

learning_rate_1��j7�<�2I       6%�	K�7���A�	*;


total_loss��@

error_R@�E?

learning_rate_1��j7i�VI       6%�	0ĉ7���A�	*;


total_lossJ�@

error_RM J?

learning_rate_1��j74��I       6%�	��7���A�	*;


total_lossd��@

error_RHZc?

learning_rate_1��j7�f��I       6%�	TP�7���A�	*;


total_loss|�A

error_R]�L?

learning_rate_1��j7��I       6%�	���7���A�	*;


total_loss/A

error_RDB?

learning_rate_1��j7m�sI       6%�	|��7���A�	*;


total_loss��@

error_R_W?

learning_rate_1��j7���I       6%�	�P�7���A�	*;


total_loss���@

error_R��O?

learning_rate_1��j73},ZI       6%�	��7���A�	*;


total_lossl��@

error_R��a?

learning_rate_1��j7�zG�I       6%�	!&�7���A�	*;


total_loss>š@

error_R��C?

learning_rate_1��j7;~t�I       6%�	��7���A�	*;


total_lossMZ�@

error_RHR?

learning_rate_1��j7�~�I       6%�	��7���A�	*;


total_loss��@

error_R�C?

learning_rate_1��j7�)�I       6%�	Ya�7���A�	*;


total_lossZ�@

error_R�8P?

learning_rate_1��j7���I       6%�	���7���A�	*;


total_loss���@

error_R;_?

learning_rate_1��j7����I       6%�	��7���A�	*;


total_loss��@

error_R�PI?

learning_rate_1��j7mX�tI       6%�	�s�7���A�	*;


total_loss�[�@

error_R<`F?

learning_rate_1��j7٨\�I       6%�	�7���A�	*;


total_lossW�@

error_RN�V?

learning_rate_1��j7%�g�I       6%�	�m�7���A�	*;


total_loss�h�@

error_R�X?

learning_rate_1��j7�)@ I       6%�	{ڏ7���A�	*;


total_loss1�@

error_R;bD?

learning_rate_1��j7�Y�tI       6%�	�0�7���A�	*;


total_loss"ו@

error_R�PA?

learning_rate_1��j7���AI       6%�	D��7���A�	*;


total_loss܇�@

error_R�[?

learning_rate_1��j7��z�I       6%�	�ڐ7���A�	*;


total_loss1�s@

error_Rx�H?

learning_rate_1��j7:��I       6%�	2�7���A�	*;


total_loss&Ϻ@

error_R�KC?

learning_rate_1��j7���I       6%�	���7���A�	*;


total_loss�}@

error_R�gX?

learning_rate_1��j7��vI       6%�	n�7���A�	*;


total_lossM��@

error_R�)K?

learning_rate_1��j7�7SiI       6%�	�+�7���A�	*;


total_lossOd�@

error_R��V?

learning_rate_1��j7W?�dI       6%�	�t�7���A�	*;


total_loss���@

error_RT�K?

learning_rate_1��j7��I       6%�	���7���A�	*;


total_loss#9�@

error_R*�:?

learning_rate_1��j7q@"I       6%�	�7���A�	*;


total_loss���@

error_R�9I?

learning_rate_1��j7��A*I       6%�	;��7���A�	*;


total_loss߳�@

error_R� B?

learning_rate_1��j7t�X�I       6%�	�Г7���A�	*;


total_lossѨ@

error_Rm�;?

learning_rate_1��j7���HI       6%�	5�7���A�	*;


total_loss�k�@

error_R��R?

learning_rate_1��j7�vc�I       6%�	Z`�7���A�	*;


total_loss�Ѷ@

error_RO�b?

learning_rate_1��j7���I       6%�	���7���A�	*;


total_loss=,A

error_R(�_?

learning_rate_1��j7pÙ�I       6%�	
�7���A�	*;


total_loss�_�@

error_RH�>?

learning_rate_1��j7���#I       6%�	(v�7���A�	*;


total_loss�{�@

error_R�GD?

learning_rate_1��j7��0�I       6%�	mĕ7���A�	*;


total_loss	��@

error_R��??

learning_rate_1��j7��VI       6%�	`�7���A�	*;


total_loss�D�@

error_R�+4?

learning_rate_1��j7�_I       6%�	�e�7���A�	*;


total_loss浇@

error_R�)T?

learning_rate_1��j7ӝ	I       6%�	
��7���A�	*;


total_loss�B�@

error_R�[?

learning_rate_1��j7�\�3I       6%�	��7���A�	*;


total_loss*T�@

error_R��L?

learning_rate_1��j7��qI       6%�	�6�7���A�	*;


total_loss���@

error_R��Q?

learning_rate_1��j7�@d�I       6%�	�}�7���A�	*;


total_lossI��@

error_R�M?

learning_rate_1��j7AP�6I       6%�	�ė7���A�	*;


total_lossC�@

error_RT�I?

learning_rate_1��j7��I       6%�	��7���A�	*;


total_lossO8�@

error_R�Q?

learning_rate_1��j7@�ƀI       6%�	�c�7���A�	*;


total_lossan�@

error_Rj�N?

learning_rate_1��j7�>�[I       6%�	g��7���A�	*;


total_lossw�u@

error_Ri�B?

learning_rate_1��j77�jI       6%�	(�7���A�	*;


total_loss�F�@

error_R`Q?

learning_rate_1��j7�_��I       6%�	�7�7���A�	*;


total_loss��@

error_R,�J?

learning_rate_1��j7�g�%I       6%�	�|�7���A�	*;


total_lossj�@

error_R�B?

learning_rate_1��j7���I       6%�	(��7���A�	*;


total_loss4T�@

error_R{�R?

learning_rate_1��j7,\�I       6%�	��7���A�	*;


total_lossxP�@

error_R�%N?

learning_rate_1��j7(Hr�I       6%�	L�7���A�	*;


total_lossIE�@

error_R}�N?

learning_rate_1��j7*g��I       6%�	���7���A�	*;


total_loss� A

error_R�?P?

learning_rate_1��j7�>��I       6%�	 ۚ7���A�	*;


total_lossH�@

error_Rj C?

learning_rate_1��j7��XqI       6%�	�3�7���A�	*;


total_lossd��@

error_RJ�N?

learning_rate_1��j7�y�@I       6%�	���7���A�	*;


total_lossh0�@

error_R\R?

learning_rate_1��j7:cPI       6%�	ћ7���A�	*;


total_loss�ۺ@

error_R%W?

learning_rate_1��j7��(�I       6%�	��7���A�	*;


total_loss�@

error_R��A?

learning_rate_1��j7�T��I       6%�	,v�7���A�	*;


total_loss���@

error_R��:?

learning_rate_1��j7���6I       6%�	��7���A�	*;


total_loss�e�@

error_R�Z?

learning_rate_1��j7{^�xI       6%�	x�7���A�	*;


total_loss�p�@

error_R��O?

learning_rate_1��j7�旷I       6%�	�l�7���A�	*;


total_loss��@

error_R�bZ?

learning_rate_1��j7����I       6%�	���7���A�	*;


total_loss�}�@

error_R��@?

learning_rate_1��j7L���I       6%�	c��7���A�
*;


total_loss�y�@

error_R
�O?

learning_rate_1��j7(ů�I       6%�	wD�7���A�
*;


total_lossEz�@

error_R�=?

learning_rate_1��j7���*I       6%�	�7���A�
*;


total_loss��@

error_R��e?

learning_rate_1��j7.f�I       6%�	qҞ7���A�
*;


total_loss���@

error_R,�T?

learning_rate_1��j7���,I       6%�	�&�7���A�
*;


total_lossI��@

error_RMGR?

learning_rate_1��j7F��I       6%�	�n�7���A�
*;


total_loss%�@

error_R��V?

learning_rate_1��j7�Q��I       6%�	ٴ�7���A�
*;


total_loss��@

error_Rj�M?

learning_rate_1��j74R�9I       6%�	���7���A�
*;


total_loss���@

error_RWVR?

learning_rate_1��j7YRI       6%�	W=�7���A�
*;


total_lossZ��@

error_R��i?

learning_rate_1��j7�jI       6%�	l��7���A�
*;


total_loss�&�@

error_R�gB?

learning_rate_1��j7��I       6%�	2ʠ7���A�
*;


total_loss�(�@

error_R�@?

learning_rate_1��j78i%I       6%�	U�7���A�
*;


total_lossز�@

error_R��2?

learning_rate_1��j7��KI       6%�	�Z�7���A�
*;


total_loss*i@

error_R�!L?

learning_rate_1��j7	RI       6%�	R��7���A�
*;


total_loss�0�@

error_R�^U?

learning_rate_1��j7���qI       6%�	���7���A�
*;


total_lossM�@

error_R)�O?

learning_rate_1��j78/I       6%�	�>�7���A�
*;


total_loss�9�@

error_ROl9?

learning_rate_1��j7�q7�I       6%�	G��7���A�
*;


total_loss�"�@

error_RTV?

learning_rate_1��j7�JѦI       6%�	�Ţ7���A�
*;


total_loss	��@

error_R��Z?

learning_rate_1��j7�	�I       6%�	�	�7���A�
*;


total_loss���@

error_R�P?

learning_rate_1��j7(��QI       6%�	XM�7���A�
*;


total_loss��@

error_Rl�U?

learning_rate_1��j7�if�I       6%�	���7���A�
*;


total_loss�M�@

error_RȞS?

learning_rate_1��j7{��I       6%�	o֣7���A�
*;


total_loss�n�@

error_R�GK?

learning_rate_1��j7�Q�.I       6%�	~�7���A�
*;


total_loss*%�@

error_R��N?

learning_rate_1��j7=a|I       6%�	�c�7���A�
*;


total_loss�R�@

error_Rf�V?

learning_rate_1��j7
X4I       6%�	ۺ�7���A�
*;


total_loss /�@

error_RÐ@?

learning_rate_1��j7OtXvI       6%�	��7���A�
*;


total_loss�ӽ@

error_R�5?

learning_rate_1��j7�W�hI       6%�	0g�7���A�
*;


total_loss���@

error_RAD?

learning_rate_1��j7z���I       6%�	g��7���A�
*;


total_loss���@

error_R�Q?

learning_rate_1��j7�w�I       6%�	U��7���A�
*;


total_lossdǡ@

error_R��Z?

learning_rate_1��j7[E(I       6%�	�8�7���A�
*;


total_loss|�@

error_R�G?

learning_rate_1��j7�n��I       6%�	�{�7���A�
*;


total_loss��@

error_R1�Y?

learning_rate_1��j7��OI       6%�	J¦7���A�
*;


total_loss��@

error_RWO?

learning_rate_1��j7�'�I       6%�	h�7���A�
*;


total_loss�s&A

error_Rs�J?

learning_rate_1��j7�ƱVI       6%�	�N�7���A�
*;


total_loss1�A

error_R��Q?

learning_rate_1��j7 "I       6%�	7��7���A�
*;


total_loss��@

error_R�J?

learning_rate_1��j7���0I       6%�	Vܧ7���A�
*;


total_loss�e�@

error_R)�Y?

learning_rate_1��j7���I       6%�	�"�7���A�
*;


total_loss��@

error_R�=?

learning_rate_1��j7��r)I       6%�	g�7���A�
*;


total_loss��@

error_RC@>?

learning_rate_1��j7�×I       6%�	q��7���A�
*;


total_loss���@

error_R�LV?

learning_rate_1��j7�1�PI       6%�	��7���A�
*;


total_loss7s�@

error_R�JK?

learning_rate_1��j7�y@I       6%�	:<�7���A�
*;


total_loss�*�@

error_R��l?

learning_rate_1��j7��ԠI       6%�	@��7���A�
*;


total_loss�`�@

error_R�WH?

learning_rate_1��j7b�JI       6%�	�ĩ7���A�
*;


total_loss�N�@

error_RۮG?

learning_rate_1��j7��I       6%�	��7���A�
*;


total_loss��@

error_R�G?

learning_rate_1��j7K��I       6%�	�L�7���A�
*;


total_loss�0�@

error_R��8?

learning_rate_1��j7���I       6%�	��7���A�
*;


total_loss$�@

error_Ri7P?

learning_rate_1��j7.n�I       6%�	�ժ7���A�
*;


total_lossX��@

error_R��??

learning_rate_1��j7��I       6%�	�7���A�
*;


total_loss���@

error_R�M?

learning_rate_1��j7ߧ�LI       6%�	7`�7���A�
*;


total_loss��@

error_RɅV?

learning_rate_1��j7�ů�I       6%�	{��7���A�
*;


total_loss�
�@

error_Rx�Y?

learning_rate_1��j7��}I       6%�	��7���A�
*;


total_loss�Ԗ@

error_Ro	L?

learning_rate_1��j7]+kI       6%�	]*�7���A�
*;


total_loss�"r@

error_RF^?

learning_rate_1��j7�`uI       6%�	in�7���A�
*;


total_loss���@

error_R��C?

learning_rate_1��j7��+�I       6%�	ï�7���A�
*;


total_loss4��@

error_R|�L?

learning_rate_1��j7��1I       6%�	9��7���A�
*;


total_lossFm�@

error_Rf�S?

learning_rate_1��j7Q?ޭI       6%�	:J�7���A�
*;


total_loss���@

error_R�?N?

learning_rate_1��j7�p(�I       6%�	7���A�
*;


total_loss{n@

error_Rt�G?

learning_rate_1��j7v�ܣI       6%�	�ϭ7���A�
*;


total_loss�*�@

error_R2�L?

learning_rate_1��j72vI       6%�	B�7���A�
*;


total_loss�*|@

error_R�M?

learning_rate_1��j7�/-�I       6%�	U�7���A�
*;


total_loss�Z�@

error_R�kT?

learning_rate_1��j7L��I       6%�	c��7���A�
*;


total_lossힽ@

error_R��M?

learning_rate_1��j7ͨ�I       6%�	�׮7���A�
*;


total_loss�۠@

error_R�D?

learning_rate_1��j7���I       6%�	.�7���A�
*;


total_loss��@

error_R�'S?

learning_rate_1��j7�lx�I       6%�	�\�7���A�
*;


total_loss�I�@

error_R��T?

learning_rate_1��j7����I       6%�	z��7���A�
*;


total_loss���@

error_R��G?

learning_rate_1��j7���I       6%�	[�7���A�
*;


total_loss���@

error_RçO?

learning_rate_1��j7#�TI       6%�	3"�7���A�
*;


total_loss��@

error_R��W?

learning_rate_1��j7 2I       6%�	gd�7���A�
*;


total_losspX�@

error_RD?

learning_rate_1��j7\��I       6%�	ۦ�7���A�
*;


total_loss���@

error_R,�Y?

learning_rate_1��j7I��qI       6%�	��7���A�
*;


total_loss�A

error_RJX?

learning_rate_1��j7B�&�I       6%�	�)�7���A�
*;


total_loss���@

error_RlU?

learning_rate_1��j7e�{�I       6%�	5l�7���A�
*;


total_loss���@

error_RL�U?

learning_rate_1��j7���I       6%�	լ�7���A�
*;


total_loss���@

error_R�bM?

learning_rate_1��j71�ҬI       6%�	D�7���A�
*;


total_loss�L�@

error_R|�J?

learning_rate_1��j7	D�I       6%�	�1�7���A�
*;


total_loss-t�@

error_R`oS?

learning_rate_1��j7!pI       6%�	�q�7���A�
*;


total_loss쪪@

error_R3hI?

learning_rate_1��j7�DO/I       6%�	�7���A�
*;


total_loss�U�@

error_R2WI?

learning_rate_1��j717;�I       6%�	��7���A�
*;


total_loss�A

error_R�'W?

learning_rate_1��j7RB�,I       6%�	�Q�7���A�
*;


total_loss��r@

error_R��??

learning_rate_1��j7$���I       6%�	���7���A�
*;


total_loss�ד@

error_R�F?

learning_rate_1��j7���'I       6%�	Q�7���A�
*;


total_lossHc�@

error_R��P?

learning_rate_1��j7WxI       6%�	;)�7���A�
*;


total_loss��@

error_R�xM?

learning_rate_1��j7��еI       6%�	�k�7���A�
*;


total_loss�&.A

error_R�T?

learning_rate_1��j7nC��I       6%�	6��7���A�
*;


total_lossj��@

error_R�K?

learning_rate_1��j7$�CI       6%�	�7���A�
*;


total_loss�ę@

error_RZ�P?

learning_rate_1��j7�lïI       6%�	_L�7���A�
*;


total_loss/ݻ@

error_R��I?

learning_rate_1��j7
�aI       6%�	A��7���A�
*;


total_loss�)�@

error_Ri�E?

learning_rate_1��j7Q^�I       6%�	X�7���A�
*;


total_loss��@

error_R�}[?

learning_rate_1��j7��s�I       6%�	� �7���A�
*;


total_loss�,�@

error_R�Ia?

learning_rate_1��j7_��I       6%�	�d�7���A�
*;


total_loss.��@

error_R��Q?

learning_rate_1��j7G���I       6%�	��7���A�
*;


total_loss&g�@

error_R.�N?

learning_rate_1��j7��͆I       6%�	L�7���A�
*;


total_loss�@

error_R�>X?

learning_rate_1��j7�=JI       6%�	Q,�7���A�
*;


total_loss�ؔ@

error_RWHA?

learning_rate_1��j7�u�SI       6%�	cm�7���A�
*;


total_loss!X�@

error_Rl	6?

learning_rate_1��j7���I       6%�	���7���A�
*;


total_loss�u1A

error_R�4V?

learning_rate_1��j7�DI       6%�	��7���A�
*;


total_lossZ�@

error_RO�A?

learning_rate_1��j7t��I       6%�	F?�7���A�
*;


total_loss�0�@

error_Rt�[?

learning_rate_1��j7�I       6%�	���7���A�
*;


total_loss���@

error_RhqF?

learning_rate_1��j7N���I       6%�	�̸7���A�
*;


total_loss�^�@

error_R�=G?

learning_rate_1��j7@�}I       6%�	E�7���A�
*;


total_lossk� A

error_R�R?

learning_rate_1��j7�ܝ
I       6%�	B\�7���A�
*;


total_loss��@

error_R�
[?

learning_rate_1��j7w�L�I       6%�	���7���A�
*;


total_lossI�@

error_R�T?

learning_rate_1��j7����I       6%�	>߹7���A�
*;


total_lossV��@

error_R��P?

learning_rate_1��j7��nXI       6%�	;+�7���A�
*;


total_lossq �@

error_R��X?

learning_rate_1��j7����I       6%�	A��7���A�
*;


total_loss`�{@

error_R;�K?

learning_rate_1��j7Q��I       6%�	�ɺ7���A�
*;


total_loss��@

error_R�N?

learning_rate_1��j7
���I       6%�	��7���A�
*;


total_loss�2�@

error_R\�;?

learning_rate_1��j7���I       6%�	�O�7���A�
*;


total_lossA�@

error_RHiR?

learning_rate_1��j7�wDI       6%�	ڕ�7���A�
*;


total_loss��@

error_RܖD?

learning_rate_1��j7�LtI       6%�	Uݻ7���A�
*;


total_loss�w�@

error_R��L?

learning_rate_1��j7Pe�EI       6%�	Z$�7���A�
*;


total_lossII�@

error_R��N?

learning_rate_1��j7uU�I       6%�	�h�7���A�
*;


total_loss�1�@

error_R�8Y?

learning_rate_1��j7��>�I       6%�	J��7���A�
*;


total_lossA˪@

error_Rt$W?

learning_rate_1��j7��8�I       6%�	�7���A�
*;


total_loss���@

error_R�O?

learning_rate_1��j7)�.I       6%�	.�7���A�
*;


total_lossI~�@

error_R�\G?

learning_rate_1��j7��FI       6%�	;p�7���A�
*;


total_loss��-A

error_R�?Z?

learning_rate_1��j7�<�I       6%�	���7���A�
*;


total_loss1��@

error_RF?

learning_rate_1��j7���EI       6%�	��7���A�
*;


total_loss4�@

error_RjQb?

learning_rate_1��j7��!I       6%�	�>�7���A�
*;


total_loss���@

error_RZ�M?

learning_rate_1��j7<_�NI       6%�	f��7���A�
*;


total_loss���@

error_R\'D?

learning_rate_1��j7�|,�I       6%�	�Ǿ7���A�
*;


total_lossCv�@

error_RReV?

learning_rate_1��j7�;�?I       6%�	�
�7���A�
*;


total_loss��d@

error_R�8O?

learning_rate_1��j7�E��I       6%�	�P�7���A�
*;


total_loss&"�@

error_RQ�7?

learning_rate_1��j76�pI       6%�	�7���A�
*;


total_loss�Ǯ@

error_R8pW?

learning_rate_1��j7݆�+I       6%�	uտ7���A�
*;


total_loss���@

error_R��T?

learning_rate_1��j7�ڥ�I       6%�	��7���A�
*;


total_lossA

error_R�ST?

learning_rate_1��j7o5�PI       6%�	\�7���A�
*;


total_loss䲈@

error_R�S?

learning_rate_1��j7Bc�8I       6%�	��7���A�
*;


total_loss���@

error_R*I?

learning_rate_1��j7r��I       6%�	b��7���A�*;


total_lossګ�@

error_R��N?

learning_rate_1��j7����I       6%�	�$�7���A�*;


total_loss]��@

error_R�f?

learning_rate_1��j7��tdI       6%�	|g�7���A�*;


total_lossf	A

error_R!�C?

learning_rate_1��j7i磻I       6%�	0��7���A�*;


total_loss�ܪ@

error_R��S?

learning_rate_1��j7���I       6%�		��7���A�*;


total_loss��c@

error_R=??

learning_rate_1��j7�ـ�I       6%�	�)�7���A�*;


total_loss�tHA

error_R�K?

learning_rate_1��j75� �I       6%�	l�7���A�*;


total_lossW'�@

error_R=�d?

learning_rate_1��j7s��I       6%�	��7���A�*;


total_loss�\�@

error_R/I?

learning_rate_1��j7��(uI       6%�	���7���A�*;


total_loss���@

error_RȟM?

learning_rate_1��j7~R��I       6%�	�+�7���A�*;


total_lossRV�@

error_R��O?

learning_rate_1��j7>A�I       6%�	�l�7���A�*;


total_loss���@

error_R�A?

learning_rate_1��j7𶣙I       6%�	��7���A�*;


total_loss��@

error_R��Q?

learning_rate_1��j7'sfI       6%�	j��7���A�*;


total_loss̴�@

error_R�HG?

learning_rate_1��j7D�TMI       6%�	�2�7���A�*;


total_loss��A

error_R��H?

learning_rate_1��j7s|�I       6%�	Sw�7���A�*;


total_loss]��@

error_R��M?

learning_rate_1��j7�^�1I       6%�	���7���A�*;


total_lossZ��@

error_R�R?

learning_rate_1��j7P���I       6%�	�7���A�*;


total_loss)W�@

error_R��S?

learning_rate_1��j7��yI       6%�	�c�7���A�*;


total_loss�ш@

error_R�eW?

learning_rate_1��j7�uC�I       6%�	P��7���A�*;


total_lossO��@

error_R�
S?

learning_rate_1��j7$�<�I       6%�	���7���A�*;


total_loss ��@

error_R��0?

learning_rate_1��j7����I       6%�	�2�7���A�*;


total_loss���@

error_RȅV?

learning_rate_1��j7@�I�I       6%�	�s�7���A�*;


total_loss�@

error_R@I?

learning_rate_1��j7�*\I       6%�	��7���A�*;


total_loss��@

error_Rz�G?

learning_rate_1��j7�}�I       6%�	���7���A�*;


total_loss��A

error_RX�:?

learning_rate_1��j72��I       6%�	�=�7���A�*;


total_loss褌@

error_Rda]?

learning_rate_1��j7�e�cI       6%�	��7���A�*;


total_lossT��@

error_Rota?

learning_rate_1��j7��rI       6%�	=��7���A�*;


total_lossF��@

error_R�9>?

learning_rate_1��j7&�I       6%�	��7���A�*;


total_loss`��@

error_RhM?

learning_rate_1��j7ܵ�*I       6%�	�O�7���A�*;


total_loss�A

error_R�pW?

learning_rate_1��j7&4=I       6%�	���7���A�*;


total_loss���@

error_R��E?

learning_rate_1��j7�nI       6%�	S��7���A�*;


total_loss�=�@

error_RjK?

learning_rate_1��j7KR��I       6%�	��7���A�*;


total_lossVٖ@

error_R�0M?

learning_rate_1��j7�M�I       6%�	rY�7���A�*;


total_lossj��@

error_R.�??

learning_rate_1��j7�xo�I       6%�	N��7���A�*;


total_lossBL�@

error_R� 9?

learning_rate_1��j7d:�EI       6%�	^��7���A�*;


total_loss۸�@

error_R�)R?

learning_rate_1��j7Y�TI       6%�	4�7���A�*;


total_loss���@

error_R.YO?

learning_rate_1��j7ܱ$�I       6%�	 b�7���A�*;


total_lossb-A

error_R��R?

learning_rate_1��j7j��6I       6%�	G��7���A�*;


total_loss�n�@

error_R�.@?

learning_rate_1��j7d�I       6%�	V��7���A�*;


total_loss&�@

error_R?O?

learning_rate_1��j7�r�I       6%�	6�7���A�*;


total_loss�@

error_RqIU?

learning_rate_1��j7@�MEI       6%�	{�7���A�*;


total_loss��@

error_Rz@@?

learning_rate_1��j7���\I       6%�	���7���A�*;


total_loss�7�@

error_Rv�b?

learning_rate_1��j7�^=I       6%�	�	�7���A�*;


total_loss���@

error_R
�U?

learning_rate_1��j7-��I       6%�	�M�7���A�*;


total_lossO��@

error_R�a?

learning_rate_1��j7��tI       6%�	f��7���A�*;


total_loss��@

error_R͘K?

learning_rate_1��j7�{I       6%�	���7���A�*;


total_lossw��@

error_R�xM?

learning_rate_1��j7!ݻ�I       6%�	v�7���A�*;


total_loss���@

error_R�\[?

learning_rate_1��j7+��PI       6%�	�r�7���A�*;


total_loss���@

error_R�7M?

learning_rate_1��j7׵E�I       6%�	��7���A�*;


total_loss���@

error_R�R?

learning_rate_1��j7�U�I       6%�	h��7���A�*;


total_lossf �@

error_R;�Q?

learning_rate_1��j7�2��I       6%�	�>�7���A�*;


total_loss���@

error_R�]c?

learning_rate_1��j7���I       6%�	���7���A�*;


total_loss.�@

error_R��Y?

learning_rate_1��j7�SMXI       6%�	���7���A�*;


total_lossA��@

error_R�sa?

learning_rate_1��j7
�/�I       6%�	��7���A�*;


total_loss�#�@

error_Ro*I?

learning_rate_1��j7�N�I       6%�	�^�7���A�*;


total_loss]A

error_R�DV?

learning_rate_1��j7��{�I       6%�	��7���A�*;


total_loss1��@

error_Rn�F?

learning_rate_1��j7��^I       6%�	= �7���A�*;


total_loss���@

error_R�V?

learning_rate_1��j7,&��I       6%�	�E�7���A�*;


total_loss���@

error_R&eX?

learning_rate_1��j7��tI       6%�	3��7���A�*;


total_loss·�@

error_RD�??

learning_rate_1��j7j�BI       6%�	|��7���A�*;


total_loss��@

error_RnFN?

learning_rate_1��j7]�q�I       6%�	�#�7���A�*;


total_loss���@

error_RuG?

learning_rate_1��j7ΈyI       6%�	�i�7���A�*;


total_loss���@

error_R�zE?

learning_rate_1��j7���KI       6%�	���7���A�*;


total_lossI�@

error_R<�K?

learning_rate_1��j7/;I       6%�	���7���A�*;


total_lossם�@

error_R$�C?

learning_rate_1��j7��rUI       6%�	8�7���A�*;


total_loss;��@

error_R��e?

learning_rate_1��j7��AI       6%�	|�7���A�*;


total_lossؠ�@

error_RfGW?

learning_rate_1��j7�;I       6%�	���7���A�*;


total_loss�I�@

error_RӵY?

learning_rate_1��j7?{�I       6%�	���7���A�*;


total_loss�b�@

error_Rzid?

learning_rate_1��j7<I       6%�	�B�7���A�*;


total_lossx\�@

error_R SG?

learning_rate_1��j7�4^I       6%�	���7���A�*;


total_loss���@

error_R��P?

learning_rate_1��j7@���I       6%�	H��7���A�*;


total_loss���@

error_R@KJ?

learning_rate_1��j7�qi�I       6%�	)
�7���A�*;


total_loss�v�@

error_R��@?

learning_rate_1��j7�b�(I       6%�	O�7���A�*;


total_loss㏾@

error_R�T?

learning_rate_1��j7�r��I       6%�	M��7���A�*;


total_loss?t�@

error_R�xE?

learning_rate_1��j7�C�6I       6%�	���7���A�*;


total_loss�2�@

error_R�O?

learning_rate_1��j7��0]I       6%�	�0�7���A�*;


total_loss��@

error_R��+?

learning_rate_1��j7��;WI       6%�	���7���A�*;


total_loss�w@

error_R��W?

learning_rate_1��j7��II       6%�	���7���A�*;


total_lossX�^@

error_R��K?

learning_rate_1��j7W�L�I       6%�	��7���A�*;


total_loss1�@

error_R�,U?

learning_rate_1��j7��BGI       6%�	�Y�7���A�*;


total_loss3ӓ@

error_Ro`?

learning_rate_1��j7��MI       6%�	O��7���A�*;


total_loss۞@

error_R��<?

learning_rate_1��j7�d�I       6%�	���7���A�*;


total_lossS��@

error_R�UP?

learning_rate_1��j7ކX�I       6%�	e@�7���A�*;


total_loss�;�@

error_R3�\?

learning_rate_1��j7k>�iI       6%�	X��7���A�*;


total_loss���@

error_R�+J?

learning_rate_1��j7-�I       6%�	X��7���A�*;


total_lossӗ�@

error_R/�O?

learning_rate_1��j7�A��I       6%�	 �7���A�*;


total_loss��@

error_R�H?

learning_rate_1��j7�8��I       6%�	b\�7���A�*;


total_loss��@

error_RH#T?

learning_rate_1��j7F��:I       6%�	���7���A�*;


total_losss�x@

error_R�W?

learning_rate_1��j7#�bI       6%�	���7���A�*;


total_loss�׃@

error_R�U?

learning_rate_1��j7h=dI       6%�	�6�7���A�*;


total_loss��@

error_Ro�V?

learning_rate_1��j7��^I       6%�	{�7���A�*;


total_loss��@

error_RJ�>?

learning_rate_1��j7 U2GI       6%�	���7���A�*;


total_loss�<�@

error_RHHL?

learning_rate_1��j7�%�I       6%�	��7���A�*;


total_loss.0�@

error_R�oS?

learning_rate_1��j7#�AvI       6%�	�B�7���A�*;


total_lossf�@

error_R
�c?

learning_rate_1��j7*/MI       6%�	��7���A�*;


total_loss�N�@

error_RhV?

learning_rate_1��j7����I       6%�	���7���A�*;


total_loss;��@

error_R �H?

learning_rate_1��j7��n�I       6%�	��7���A�*;


total_loss���@

error_R�X?

learning_rate_1��j7IڙI       6%�	GW�7���A�*;


total_loss4�A

error_R!RH?

learning_rate_1��j7�QI       6%�	l��7���A�*;


total_loss8��@

error_R4=_?

learning_rate_1��j7;��-I       6%�	���7���A�*;


total_loss�I�@

error_RF@?

learning_rate_1��j7���I       6%�	J*�7���A�*;


total_loss�/A

error_R�NA?

learning_rate_1��j7,��I       6%�	wp�7���A�*;


total_lossZ��@

error_R�\?

learning_rate_1��j7;r��I       6%�	��7���A�*;


total_loss��	A

error_R��C?

learning_rate_1��j7�=oI       6%�	*��7���A�*;


total_loss�	�@

error_RC%??

learning_rate_1��j7�՟�I       6%�	�:�7���A�*;


total_loss�v�@

error_R�rb?

learning_rate_1��j7���I       6%�	H}�7���A�*;


total_loss�e�@

error_R�hS?

learning_rate_1��j7֮!-I       6%�	z��7���A�*;


total_loss��@

error_R=A?

learning_rate_1��j7G�ЮI       6%�	d��7���A�*;


total_loss'`�@

error_Re�J?

learning_rate_1��j7�բ�I       6%�	A�7���A�*;


total_loss��@

error_RҊO?

learning_rate_1��j7!�zhI       6%�	)��7���A�*;


total_lossOƞ@

error_R�dL?

learning_rate_1��j7̈q]I       6%�	���7���A�*;


total_loss�d�@

error_R1�Q?

learning_rate_1��j7]�?�I       6%�	��7���A�*;


total_loss��@

error_R�:P?

learning_rate_1��j7t77I       6%�	��7���A�*;


total_lossuA

error_R�\?

learning_rate_1��j7�ˍ�I       6%�	T��7���A�*;


total_loss/��@

error_R�\?

learning_rate_1��j77�_I       6%�	��7���A�*;


total_lossU��@

error_RH?

learning_rate_1��j7g��)I       6%�	G[�7���A�*;


total_loss���@

error_R�eH?

learning_rate_1��j7�F�7I       6%�	v��7���A�*;


total_loss�]�@

error_R��B?

learning_rate_1��j7��
�I       6%�	��7���A�*;


total_loss��@

error_R /Y?

learning_rate_1��j70�|�I       6%�	�$�7���A�*;


total_loss�@

error_R
B?

learning_rate_1��j7J��I       6%�	�h�7���A�*;


total_loss�
�@

error_ROGA?

learning_rate_1��j7?��YI       6%�	l��7���A�*;


total_loss��@

error_R;T?

learning_rate_1��j7�o;�I       6%�	���7���A�*;


total_loss���@

error_R�=L?

learning_rate_1��j7���I       6%�	.�7���A�*;


total_loss]�@

error_Rܦ]?

learning_rate_1��j7K0�I       6%�	p�7���A�*;


total_loss-��@

error_R�{c?

learning_rate_1��j7�^�%I       6%�	���7���A�*;


total_loss�K�@

error_Rʦ_?

learning_rate_1��j7����I       6%�	���7���A�*;


total_loss�{�@

error_RW�D?

learning_rate_1��j7���mI       6%�	�3�7���A�*;


total_lossO·@

error_RO�S?

learning_rate_1��j7��I       6%�	b}�7���A�*;


total_loss��@

error_R�9S?

learning_rate_1��j7��D�I       6%�	0��7���A�*;


total_loss�� A

error_RFFA?

learning_rate_1��j7y,�vI       6%�	
�7���A�*;


total_loss6�@

error_R4�T?

learning_rate_1��j7>>]rI       6%�	L�7���A�*;


total_loss�'�@

error_R�/Q?

learning_rate_1��j7.�o�I       6%�	'��7���A�*;


total_lossw��@

error_RM@?

learning_rate_1��j7n3��I       6%�	n��7���A�*;


total_loss(�@

error_R?�_?

learning_rate_1��j7����I       6%�	�%�7���A�*;


total_loss�P�@

error_Ro�J?

learning_rate_1��j7�ooI       6%�	�~�7���A�*;


total_loss�@

error_R�WC?

learning_rate_1��j7%�1�I       6%�	`��7���A�*;


total_loss���@

error_R}�H?

learning_rate_1��j7r)�QI       6%�	�7���A�*;


total_losso��@

error_Rz|N?

learning_rate_1��j7���I       6%�	�B�7���A�*;


total_loss�&�@

error_R�>V?

learning_rate_1��j7��)I       6%�	d��7���A�*;


total_loss~}@

error_R�(:?

learning_rate_1��j7�R��I       6%�	���7���A�*;


total_loss��@

error_R\�C?

learning_rate_1��j7v�_%I       6%�	��7���A�*;


total_loss1�@

error_Rh�N?

learning_rate_1��j7�.I       6%�	vL�7���A�*;


total_loss�Ƞ@

error_R�R?

learning_rate_1��j7�9[�I       6%�	��7���A�*;


total_loss�|@

error_RmbR?

learning_rate_1��j7Ւ/:I       6%�	���7���A�*;


total_loss�X�@

error_R+X?

learning_rate_1��j7j�dvI       6%�	�7���A�*;


total_loss��@

error_R-�K?

learning_rate_1��j7�X�dI       6%�	�O�7���A�*;


total_loss�wA

error_R��[?

learning_rate_1��j73�{I       6%�	���7���A�*;


total_loss���@

error_R8�Q?

learning_rate_1��j7�j�MI       6%�	5��7���A�*;


total_loss�e�@

error_R�`?

learning_rate_1��j7��>I       6%�	��7���A�*;


total_lossZ�@

error_RE?

learning_rate_1��j7��M�I       6%�	
X�7���A�*;


total_loss*��@

error_R6V?

learning_rate_1��j7+�խI       6%�	u��7���A�*;


total_lossh��@

error_RC?

learning_rate_1��j7��<I       6%�	��7���A�*;


total_lossE��@

error_R1�@?

learning_rate_1��j7 oI       6%�	X'�7���A�*;


total_loss��@

error_Rb_?

learning_rate_1��j7P�3�I       6%�	i�7���A�*;


total_loss���@

error_Rm�c?

learning_rate_1��j7���[I       6%�	K��7���A�*;


total_loss��@

error_RmT?

learning_rate_1��j7�"VI       6%�	���7���A�*;


total_loss\��@

error_RŚW?

learning_rate_1��j7��e�I       6%�	�4�7���A�*;


total_loss���@

error_RQa<?

learning_rate_1��j7�s�I       6%�	0u�7���A�*;


total_loss[�@

error_RŚV?

learning_rate_1��j7��lI       6%�	���7���A�*;


total_loss�!�@

error_RWBH?

learning_rate_1��j7<s��I       6%�	���7���A�*;


total_lossVO�@

error_R]^H?

learning_rate_1��j7s�I       6%�	�=�7���A�*;


total_loss6"�@

error_R�]?

learning_rate_1��j7��|I       6%�	��7���A�*;


total_loss��@

error_R�_?

learning_rate_1��j7��I       6%�	b��7���A�*;


total_loss!z�@

error_RZ�@?

learning_rate_1��j7��ZKI       6%�	]�7���A�*;


total_lossF |@

error_R��O?

learning_rate_1��j7���I       6%�	�H�7���A�*;


total_lossZ��@

error_R��J?

learning_rate_1��j7A��I       6%�	2��7���A�*;


total_lossSr�@

error_R�I?

learning_rate_1��j7~�
�I       6%�	��7���A�*;


total_loss��n@

error_R�ZM?

learning_rate_1��j7��>�I       6%�	P�7���A�*;


total_loss�#�@

error_RM�S?

learning_rate_1��j7@��{I       6%�	�S�7���A�*;


total_loss�}�@

error_R��]?

learning_rate_1��j7OP�I       6%�	���7���A�*;


total_loss�v�@

error_RJ/?

learning_rate_1��j7�.t�I       6%�	���7���A�*;


total_lossEx�@

error_R�<Z?

learning_rate_1��j7��v�I       6%�	��7���A�*;


total_loss�ɠ@

error_R=�E?

learning_rate_1��j7R��I       6%�	
_�7���A�*;


total_loss-��@

error_Rl6S?

learning_rate_1��j7'�zI       6%�	���7���A�*;


total_loss���@

error_Rq8L?

learning_rate_1��j7��F;I       6%�	g��7���A�*;


total_loss.��@

error_R}Fc?

learning_rate_1��j7j�@}I       6%�	�'�7���A�*;


total_lossX��@

error_R��L?

learning_rate_1��j7��Z�I       6%�	=o�7���A�*;


total_loss/>�@

error_R%6?

learning_rate_1��j7@�:I       6%�	W��7���A�*;


total_loss��{@

error_R��T?

learning_rate_1��j7H�lI       6%�	)��7���A�*;


total_loss�-�@

error_RW�>?

learning_rate_1��j7u
?I       6%�	?O�7���A�*;


total_loss���@

error_R�I?

learning_rate_1��j7ӨؠI       6%�	ђ�7���A�*;


total_loss��@

error_R�~`?

learning_rate_1��j7o���I       6%�	L��7���A�*;


total_lossP�@

error_R�8^?

learning_rate_1��j7�G@I       6%�	�.�7���A�*;


total_loss݌�@

error_R��Q?

learning_rate_1��j7�X�I       6%�	o{�7���A�*;


total_loss_��@

error_R�R?

learning_rate_1��j7��7I       6%�	o��7���A�*;


total_lossL��@

error_RE�>?

learning_rate_1��j7	4
LI       6%�	 �7���A�*;


total_loss�G�@

error_RO?

learning_rate_1��j7��I       6%�	$R�7���A�*;


total_loss�ئ@

error_R�S?

learning_rate_1��j7e��I       6%�	���7���A�*;


total_loss���@

error_R&�M?

learning_rate_1��j7�h�:I       6%�	t��7���A�*;


total_loss��@

error_REE?

learning_rate_1��j7���I       6%�	� �7���A�*;


total_loss��@

error_R	�Y?

learning_rate_1��j7t��I       6%�	�b�7���A�*;


total_loss�Z�@

error_Rx�T?

learning_rate_1��j7�wI       6%�	���7���A�*;


total_loss��@

error_R��T?

learning_rate_1��j7ts��I       6%�	���7���A�*;


total_lossU�A

error_R=F?

learning_rate_1��j7����I       6%�	�;�7���A�*;


total_loss�÷@

error_R�:R?

learning_rate_1��j7݊�;I       6%�	J��7���A�*;


total_loss=��@

error_R��Y?

learning_rate_1��j7X�yI       6%�	���7���A�*;


total_loss1��@

error_R��c?

learning_rate_1��j79k!�I       6%�	�)�7���A�*;


total_loss{z�@

error_Rv`?

learning_rate_1��j7a��I       6%�	�l�7���A�*;


total_loss
F�@

error_R�DM?

learning_rate_1��j7N_9#I       6%�	���7���A�*;


total_loss��@

error_R��[?

learning_rate_1��j7�X��I       6%�	p��7���A�*;


total_loss�"�@

error_R��H?

learning_rate_1��j7���ZI       6%�	,2�7���A�*;


total_lossۙ@

error_RI�^?

learning_rate_1��j7*C �I       6%�	Dw�7���A�*;


total_loss�Z�@

error_R�l?

learning_rate_1��j7�&sI       6%�	���7���A�*;


total_loss�A

error_R15D?

learning_rate_1��j7i���I       6%�	���7���A�*;


total_loss��@

error_RE�G?

learning_rate_1��j7�$I       6%�	�B�7���A�*;


total_lossDn�@

error_Rx�L?

learning_rate_1��j7���%I       6%�	���7���A�*;


total_losstw�@

error_R��B?

learning_rate_1��j7b4I       6%�	���7���A�*;


total_lossJ�@

error_R�V?

learning_rate_1��j7�uS�I       6%�	1�7���A�*;


total_loss��@

error_R�*K?

learning_rate_1��j7�LpI       6%�	mR�7���A�*;


total_loss�?A

error_R`?

learning_rate_1��j7���I       6%�	��7���A�*;


total_lossZU�@

error_Rh P?

learning_rate_1��j7J"�BI       6%�	n��7���A�*;


total_loss;m�@

error_R��e?

learning_rate_1��j7�f�<I       6%�	�7���A�*;


total_lossp�@

error_RxUW?

learning_rate_1��j7*ھ[I       6%�	<[�7���A�*;


total_loss��@

error_R�oZ?

learning_rate_1��j7�`kI       6%�	���7���A�*;


total_lossv��@

error_R�\[?

learning_rate_1��j7c;��I       6%�	`��7���A�*;


total_lossj��@

error_RU?

learning_rate_1��j7M �VI       6%�	C"�7���A�*;


total_loss��@

error_R��N?

learning_rate_1��j7�o	`I       6%�	�e�7���A�*;


total_lossIl�@

error_R��N?

learning_rate_1��j7;�&�I       6%�	D��7���A�*;


total_lossp<�@

error_R�tI?

learning_rate_1��j7A�_QI       6%�	L��7���A�*;


total_loss���@

error_Rs.R?

learning_rate_1��j7kI       6%�	*�7���A�*;


total_loss��@

error_R}�M?

learning_rate_1��j7~O�MI       6%�	Sk�7���A�*;


total_loss���@

error_R�J?

learning_rate_1��j75���I       6%�	i��7���A�*;


total_loss���@

error_R�WF?

learning_rate_1��j7$q��I       6%�	���7���A�*;


total_loss��@

error_R��B?

learning_rate_1��j7~!�I       6%�	/*�7���A�*;


total_loss �@

error_R8M?

learning_rate_1��j7���I       6%�	}j�7���A�*;


total_loss⿛@

error_RE�X?

learning_rate_1��j7߸"�I       6%�	8��7���A�*;


total_loss���@

error_Rl�2?

learning_rate_1��j7���I       6%�	���7���A�*;


total_loss�i@

error_R[�5?

learning_rate_1��j7��I       6%�	/�7���A�*;


total_loss�?�@

error_R�.M?

learning_rate_1��j7E��rI       6%�	sp�7���A�*;


total_lossO��@

error_R�A?

learning_rate_1��j7�LQ"I       6%�	��7���A�*;


total_loss���@

error_R�\?

learning_rate_1��j7v疜I       6%�	��7���A�*;


total_lossdڈ@

error_R�AA?

learning_rate_1��j7�K0nI       6%�	�6�7���A�*;


total_loss{��@

error_R��G?

learning_rate_1��j7���LI       6%�	�x�7���A�*;


total_lossl��@

error_Ra�_?

learning_rate_1��j7��lI       6%�	P��7���A�*;


total_loss��@

error_R�^Y?

learning_rate_1��j7�l�I       6%�	�  8���A�*;


total_loss삭@

error_Rl�T?

learning_rate_1��j7�Q�I       6%�	`E 8���A�*;


total_loss��@

error_Rl?C?

learning_rate_1��j7��`I       6%�	N� 8���A�*;


total_loss]�@

error_R�W?

learning_rate_1��j7��5I       6%�	�� 8���A�*;


total_loss�c�@

error_R� <?

learning_rate_1��j7�,>�I       6%�	?8���A�*;


total_loss��@

error_R��E?

learning_rate_1��j7�8I       6%�	U8���A�*;


total_loss֤y@

error_R��V?

learning_rate_1��j7ĵ�I       6%�	��8���A�*;


total_loss��@

error_R�vT?

learning_rate_1��j7���I       6%�	!�8���A�*;


total_loss���@

error_R]�J?

learning_rate_1��j78��I       6%�	;&8���A�*;


total_loss5�@

error_R�vL?

learning_rate_1��j7
���I       6%�	�m8���A�*;


total_lossEV�@

error_RO�G?

learning_rate_1��j7d��MI       6%�	P�8���A�*;


total_lossru@

error_R�$O?

learning_rate_1��j7�E�gI       6%�	��8���A�*;


total_loss։�@

error_R?_?

learning_rate_1��j7z��I       6%�	L;8���A�*;


total_loss��@

error_R��[?

learning_rate_1��j7P��!I       6%�	e�8���A�*;


total_loss�@�@

error_R��K?

learning_rate_1��j7i��I       6%�	��8���A�*;


total_loss��@

error_R�-S?

learning_rate_1��j7���I       6%�	�8���A�*;


total_lossɁ�@

error_R�-L?

learning_rate_1��j7��viI       6%�	�I8���A�*;


total_lossl�@

error_R�oN?

learning_rate_1��j7>�i_I       6%�	�8���A�*;


total_loss��@

error_R�gK?

learning_rate_1��j78n�I       6%�	:�8���A�*;


total_loss㈮@

error_R/�Y?

learning_rate_1��j7���iI       6%�	�8���A�*;


total_loss��@

error_R<S?

learning_rate_1��j7��\�I       6%�	�~8���A�*;


total_lossE��@

error_Rx�X?

learning_rate_1��j7z+�I       6%�	��8���A�*;


total_loss�ݦ@

error_R�S?

learning_rate_1��j7s�4I       6%�	q8���A�*;


total_lossw��@

error_R�iY?

learning_rate_1��j7�OI       6%�	�R8���A�*;


total_loss_��@

error_R�qF?

learning_rate_1��j7f��<I       6%�	��8���A�*;


total_loss=m�@

error_RO3;?

learning_rate_1��j7y)�I       6%�	��8���A�*;


total_loss��z@

error_R��N?

learning_rate_1��j7/�AXI       6%�	�38���A�*;


total_lossI��@

error_R&2E?

learning_rate_1��j7��.EI       6%�	{8���A�*;


total_loss�Ox@

error_R#+K?

learning_rate_1��j7�|BI       6%�	�8���A�*;


total_lossŅ�@

error_R��E?

learning_rate_1��j7)(�YI       6%�	38���A�*;


total_loss��@

error_R��`?

learning_rate_1��j7z�I       6%�	�F8���A�*;


total_lossSl�@

error_R�f?

learning_rate_1��j7��ssI       6%�	\�8���A�*;


total_loss�@

error_R�EW?

learning_rate_1��j7��|I       6%�	��8���A�*;


total_loss��@

error_R��??

learning_rate_1��j7��PI       6%�	�	8���A�*;


total_loss��@

error_R�3?

learning_rate_1��j7��'I       6%�	�O	8���A�*;


total_loss��@

error_R�?K?

learning_rate_1��j7X��0I       6%�	��	8���A�*;


total_loss��@

error_R�Q?

learning_rate_1��j7uo��I       6%�	�	8���A�*;


total_loss���@

error_R�`?

learning_rate_1��j7���I       6%�	�
8���A�*;


total_loss���@

error_R�J?

learning_rate_1��j7e���I       6%�	$^
8���A�*;


total_losss?�@

error_R6f?

learning_rate_1��j7�c��I       6%�	��
8���A�*;


total_lossO��@

error_R(�`?

learning_rate_1��j7.�ֲI       6%�	��
8���A�*;


total_loss��@

error_RͮG?

learning_rate_1��j7ց,=I       6%�	�&8���A�*;


total_loss?=�@

error_R�C?

learning_rate_1��j7%juI       6%�	�l8���A�*;


total_loss��@

error_R�TE?

learning_rate_1��j7,�+I       6%�	K�8���A�*;


total_lossʒ�@

error_R�F?

learning_rate_1��j7�Ľ�I       6%�	n�8���A�*;


total_loss���@

error_R�Q?

learning_rate_1��j7�g9I       6%�	&98���A�*;


total_loss�@

error_R�)D?

learning_rate_1��j72�2I       6%�	�x8���A�*;


total_loss��@

error_R��]?

learning_rate_1��j7�Ё	I       6%�	�8���A�*;


total_loss�=�@

error_R��G?

learning_rate_1��j7� �I       6%�	A 8���A�*;


total_loss��@

error_R,F?

learning_rate_1��j7�vxI       6%�	�D8���A�*;


total_loss �@

error_R�RW?

learning_rate_1��j7K(�I       6%�	��8���A�*;


total_lossL7�@

error_R��N?

learning_rate_1��j7Ź�[I       6%�	��8���A�*;


total_lossC_�@

error_RќL?

learning_rate_1��j7V��I       6%�	�8���A�*;


total_loss�ٝ@

error_R��F?

learning_rate_1��j7N�I       6%�	eY8���A�*;


total_loss`J�@

error_R$NR?

learning_rate_1��j75�ÆI       6%�	��8���A�*;


total_loss��@

error_Rv�M?

learning_rate_1��j7�p�QI       6%�	��8���A�*;


total_loss�D�@

error_R�P?

learning_rate_1��j7�#j�I       6%�	}%8���A�*;


total_loss�ͥ@

error_R��@?

learning_rate_1��j7�xk8I       6%�	�g8���A�*;


total_loss`F�@

error_R��<?

learning_rate_1��j7~�{I       6%�	W�8���A�*;


total_lossִz@

error_RX�Q?

learning_rate_1��j7�VMOI       6%�	D�8���A�*;


total_loss�>�@

error_R&@F?

learning_rate_1��j7=��I       6%�	�88���A�*;


total_lossC��@

error_R_IS?

learning_rate_1��j7�fl�I       6%�	�}8���A�*;


total_loss�ӡ@

error_RS�W?

learning_rate_1��j7�f�I       6%�	<�8���A�*;


total_loss�6�@

error_ROKA?

learning_rate_1��j7{.6I       6%�	�8���A�*;


total_loss):�@

error_R��R?

learning_rate_1��j7�\KI       6%�	L8���A�*;


total_loss���@

error_R�=S?

learning_rate_1��j7Tz�I       6%�	c�8���A�*;


total_loss;�@

error_R��\?

learning_rate_1��j7 L��I       6%�	-�8���A�*;


total_lossH��@

error_RQ�`?

learning_rate_1��j7��I       6%�	"8���A�*;


total_loss���@

error_R4@??

learning_rate_1��j7���I       6%�	t\8���A�*;


total_loss�@

error_Rx�R?

learning_rate_1��j7�S�I       6%�	��8���A�*;


total_loss�ɺ@

error_R��N?

learning_rate_1��j7��eMI       6%�	��8���A�*;


total_loss�c�@

error_R�S?

learning_rate_1��j7����I       6%�	�*8���A�*;


total_lossJ��@

error_R�\Q?

learning_rate_1��j7B��I       6%�	�m8���A�*;


total_loss?V�@

error_R8L?

learning_rate_1��j7�u�I       6%�	��8���A�*;


total_loss�ڱ@

error_RX?

learning_rate_1��j7�Ri�I       6%�	��8���A�*;


total_losse��@

error_R��Y?

learning_rate_1��j7u��I       6%�	�?8���A�*;


total_loss4�@

error_R��>?

learning_rate_1��j7M�I       6%�	��8���A�*;


total_loss
�@

error_R#O?

learning_rate_1��j7*h�aI       6%�	��8���A�*;


total_loss���@

error_R�_?

learning_rate_1��j7����I       6%�	�8���A�*;


total_lossmoz@

error_R�_W?

learning_rate_1��j7PtoGI       6%�	|f8���A�*;


total_lossm�A

error_R�Z?

learning_rate_1��j7���I       6%�	�8���A�*;


total_loss��@

error_Rq�`?

learning_rate_1��j7�:#I       6%�	��8���A�*;


total_loss�a�@

error_R�N?

learning_rate_1��j7�Jq�I       6%�	�38���A�*;


total_loss#��@

error_R�X?

learning_rate_1��j7͛�@I       6%�	�t8���A�*;


total_lossl��@

error_R�@?

learning_rate_1��j7/�lI       6%�	��8���A�*;


total_loss�A

error_R�H?

learning_rate_1��j7���I       6%�	��8���A�*;


total_loss�O�@

error_RawK?

learning_rate_1��j7���I       6%�	�<8���A�*;


total_lossx��@

error_R,�6?

learning_rate_1��j7چq�I       6%�	R}8���A�*;


total_loss���@

error_R�D_?

learning_rate_1��j7��r�I       6%�	�8���A�*;


total_loss��@

error_R�9?

learning_rate_1��j7��I       6%�	,8���A�*;


total_loss���@

error_R� \?

learning_rate_1��j7�yI       6%�	�B8���A�*;


total_loss��@

error_R�aU?

learning_rate_1��j7���)I       6%�	�8���A�*;


total_lossFς@

error_R�EY?

learning_rate_1��j7&v�I       6%�	�8���A�*;


total_loss��A

error_R}P9?

learning_rate_1��j7ȥ)-I       6%�	�8���A�*;


total_loss�P�@

error_RRJV?

learning_rate_1��j7!�W<I       6%�	V\8���A�*;


total_loss�̬@

error_R1Y?

learning_rate_1��j7j���I       6%�	H�8���A�*;


total_loss�Ů@

error_R�O?

learning_rate_1��j7zKqI       6%�	}�8���A�*;


total_lossu,�@

error_R��H?

learning_rate_1��j7�i�8I       6%�	�"8���A�*;


total_loss�J�@

error_RD�B?

learning_rate_1��j7�^�I       6%�	�e8���A�*;


total_loss�+�@

error_R!EV?

learning_rate_1��j7�aS�I       6%�	q�8���A�*;


total_loss���@

error_R��E?

learning_rate_1��j7��YI       6%�	��8���A�*;


total_loss��@

error_R�U?

learning_rate_1��j7�/��I       6%�	�+8���A�*;


total_loss���@

error_R�\?

learning_rate_1��j7W}@FI       6%�	^m8���A�*;


total_loss�4�@

error_R�E?

learning_rate_1��j7�K��I       6%�	/�8���A�*;


total_lossl��@

error_R�!j?

learning_rate_1��j7R�'QI       6%�	�8���A�*;


total_loss���@

error_R�T?

learning_rate_1��j7�Z�)I       6%�	�28���A�*;


total_loss���@

error_R=�C?

learning_rate_1��j7�LfI       6%�	x8���A�*;


total_lossuЍ@

error_R�<?

learning_rate_1��j7�{l�I       6%�	�8���A�*;


total_loss�b�@

error_R�1B?

learning_rate_1��j7�i�iI       6%�	�8���A�*;


total_loss��@

error_R�H?

learning_rate_1��j7"亵I       6%�	P8���A�*;


total_lossq��@

error_R�jI?

learning_rate_1��j7�м�I       6%�	D�8���A�*;


total_loss	�@

error_R2/I?

learning_rate_1��j7�FI       6%�	�8���A�*;


total_lossz�@

error_RM\R?

learning_rate_1��j7p�!I       6%�	�8���A�*;


total_loss�W�@

error_R{�Q?

learning_rate_1��j7�k�I       6%�	�c8���A�*;


total_loss��@

error_R�<S?

learning_rate_1��j77r8�I       6%�	�8���A�*;


total_loss��@

error_R��P?

learning_rate_1��j7�m4I       6%�	�8���A�*;


total_lossNo�@

error_RH�H?

learning_rate_1��j7˔:I       6%�	
08���A�*;


total_loss@��@

error_R�dQ?

learning_rate_1��j7�P��I       6%�	s8���A�*;


total_loss�׹@

error_R��K?

learning_rate_1��j7�I�I       6%�	޸8���A�*;


total_loss���@

error_R��`?

learning_rate_1��j7u,7,I       6%�	��8���A�*;


total_loss?��@

error_RH[P?

learning_rate_1��j74��I       6%�	6< 8���A�*;


total_loss���@

error_R۷K?

learning_rate_1��j7
'^I       6%�	؀ 8���A�*;


total_loss���@

error_RnH?

learning_rate_1��j7�WE�I       6%�	�� 8���A�*;


total_loss���@

error_R��I?

learning_rate_1��j7]E��I       6%�	d!8���A�*;


total_loss)��@

error_R�WH?

learning_rate_1��j7V,vaI       6%�	;]!8���A�*;


total_loss�$A

error_RJY?

learning_rate_1��j7^Z*I       6%�	��!8���A�*;


total_lossO!�@

error_Rq�M?

learning_rate_1��j7TYL5I       6%�	��!8���A�*;


total_lossH�	A

error_R��R?

learning_rate_1��j7a.��I       6%�	~,"8���A�*;


total_lossܦ�@

error_R,�N?

learning_rate_1��j7R�%�I       6%�	kp"8���A�*;


total_loss�Z�@

error_R��I?

learning_rate_1��j7�')�I       6%�	q�"8���A�*;


total_lossSeA

error_R�SY?

learning_rate_1��j78���I       6%�	��"8���A�*;


total_loss�@

error_R�`Z?

learning_rate_1��j7�oI       6%�	x:#8���A�*;


total_loss�]_@

error_R�xO?

learning_rate_1��j7pѲ�I       6%�	�}#8���A�*;


total_loss��A

error_R
6O?

learning_rate_1��j7���KI       6%�	9�#8���A�*;


total_loss&��@

error_R)�K?

learning_rate_1��j7���NI       6%�	O$8���A�*;


total_loss�	A

error_R��U?

learning_rate_1��j7��ڸI       6%�	�_$8���A�*;


total_loss: �@

error_RA�Q?

learning_rate_1��j7w�Q'I       6%�	��$8���A�*;


total_loss��@

error_R�oW?

learning_rate_1��j7�� I       6%�	��$8���A�*;


total_loss��@

error_R1R?

learning_rate_1��j7�y-qI       6%�	�=%8���A�*;


total_loss��|@

error_R�8R?

learning_rate_1��j7�]�AI       6%�	d�%8���A�*;


total_loss��@

error_RZ^?

learning_rate_1��j7�i&I       6%�	t�%8���A�*;


total_loss�Ϩ@

error_R.VX?

learning_rate_1��j7j��I       6%�	�V&8���A�*;


total_loss��@

error_R(�M?

learning_rate_1��j7L��7I       6%�	��&8���A�*;


total_lossVq�@

error_REkG?

learning_rate_1��j7(��
I       6%�	��&8���A�*;


total_loss�·@

error_R(�K?

learning_rate_1��j7K��3I       6%�	{8'8���A�*;


total_loss>�@

error_RC�Z?

learning_rate_1��j7m~.I       6%�	{'8���A�*;


total_loss׀�@

error_R��V?

learning_rate_1��j7�I       6%�	��'8���A�*;


total_lossϢ@

error_R�UF?

learning_rate_1��j7�/3aI       6%�	M(8���A�*;


total_loss�d�@

error_RϻN?

learning_rate_1��j7(6�I       6%�	6I(8���A�*;


total_loss4�@

error_R�Q?

learning_rate_1��j7���I       6%�	�(8���A�*;


total_loss	N�@

error_R,�O?

learning_rate_1��j7�[ʿI       6%�	��(8���A�*;


total_loss$S�@

error_R��L?

learning_rate_1��j7E��?I       6%�	7)8���A�*;


total_loss]�@

error_R2�F?

learning_rate_1��j7����I       6%�	
W)8���A�*;


total_loss���@

error_Rf+V?

learning_rate_1��j7�O+�I       6%�	�)8���A�*;


total_lossvA�@

error_R�E?

learning_rate_1��j7�6F�I       6%�	��)8���A�*;


total_loss>A

error_R8O?

learning_rate_1��j7���I       6%�	
(*8���A�*;


total_loss���@

error_R��>?

learning_rate_1��j7��~�I       6%�	)m*8���A�*;


total_loss�p�@

error_R��c?

learning_rate_1��j7���tI       6%�	��*8���A�*;


total_loss��@

error_R�F?

learning_rate_1��j7�&��I       6%�	��*8���A�*;


total_loss��@

error_RXG?

learning_rate_1��j7�:�I       6%�	J<+8���A�*;


total_lossZ<�@

error_Rvu[?

learning_rate_1��j7�(��I       6%�	�+8���A�*;


total_loss���@

error_R&gY?

learning_rate_1��j7���I       6%�	��+8���A�*;


total_loss��@

error_R��X?

learning_rate_1��j7i�I       6%�	�,8���A�*;


total_loss{e�@

error_R6E?

learning_rate_1��j7�X1�I       6%�	�J,8���A�*;


total_loss�dX@

error_R��:?

learning_rate_1��j7��-�I       6%�	~�,8���A�*;


total_loss��@

error_R��Z?

learning_rate_1��j7w#I       6%�	��,8���A�*;


total_loss�61A

error_R�rB?

learning_rate_1��j7���I       6%�	x-8���A�*;


total_loss��@

error_R��I?

learning_rate_1��j7{��I       6%�	h-8���A�*;


total_loss�[�@

error_Rl_L?

learning_rate_1��j7��5I       6%�	Z�-8���A�*;


total_lossdL�@

error_R�W?

learning_rate_1��j7��ؠI       6%�	2�-8���A�*;


total_loss6�@

error_R@�a?

learning_rate_1��j7*��I       6%�	�3.8���A�*;


total_lossE`�@

error_R�{;?

learning_rate_1��j7|�I       6%�	�v.8���A�*;


total_losso�@

error_R|�\?

learning_rate_1��j7��6I       6%�	��.8���A�*;


total_lossf&�@

error_R��R?

learning_rate_1��j7�R�I       6%�	��.8���A�*;


total_loss�}�@

error_R �O?

learning_rate_1��j7�B�I       6%�	%;/8���A�*;


total_loss�0�@

error_R��Q?

learning_rate_1��j7ӻ�I       6%�	�/8���A�*;


total_lossa{�@

error_R�B?

learning_rate_1��j7�2�kI       6%�	��/8���A�*;


total_lossU��@

error_RjUT?

learning_rate_1��j7E�p�I       6%�	A08���A�*;


total_loss8��@

error_R��I?

learning_rate_1��j7��{I       6%�	;O08���A�*;


total_loss�L�@

error_R��E?

learning_rate_1��j7��e6I       6%�	ǐ08���A�*;


total_lossNb�@

error_R��J?

learning_rate_1��j7��)kI       6%�	K�08���A�*;


total_loss�;�@

error_Rv�V?

learning_rate_1��j7�I       6%�	�18���A�*;


total_loss�t�@

error_RCpJ?

learning_rate_1��j78� �I       6%�		S18���A�*;


total_loss���@

error_RX??

learning_rate_1��j7f�I       6%�	o�18���A�*;


total_lossI�	A

error_RNW?

learning_rate_1��j7�⇾I       6%�	�18���A�*;


total_lossS��@

error_R)�T?

learning_rate_1��j7��j�I       6%�	A28���A�*;


total_loss���@

error_RV�S?

learning_rate_1��j7�װyI       6%�	iZ28���A�*;


total_loss�9�@

error_RnY?

learning_rate_1��j7�U@�I       6%�	͛28���A�*;


total_loss��@

error_RڸQ?

learning_rate_1��j7M׫_I       6%�	��28���A�*;


total_loss��A

error_R�QL?

learning_rate_1��j7QlMcI       6%�	*"38���A�*;


total_loss�'A

error_R��B?

learning_rate_1��j7�l I       6%�	<e38���A�*;


total_loss$u�@

error_R)�M?

learning_rate_1��j7���I       6%�	G�38���A�*;


total_loss�
�@

error_R�O?

learning_rate_1��j7�D�:I       6%�	��38���A�*;


total_loss�x�@

error_R��^?

learning_rate_1��j7���fI       6%�	�048���A�*;


total_loss�wo@

error_R�BT?

learning_rate_1��j7�~lI       6%�	fu48���A�*;


total_loss}V�@

error_R�&R?

learning_rate_1��j74�)�I       6%�	�48���A�*;


total_lossq�@

error_R��M?

learning_rate_1��j7����I       6%�	�48���A�*;


total_loss���@

error_R�JC?

learning_rate_1��j7A��I       6%�	�E58���A�*;


total_lossAƞ@

error_R��A?

learning_rate_1��j7\TrI       6%�	A�58���A�*;


total_lossA��@

error_R�HY?

learning_rate_1��j7.��I       6%�	z�58���A�*;


total_lossAt�@

error_R�'I?

learning_rate_1��j7���YI       6%�	�$68���A�*;


total_lossvH�@

error_R�X?

learning_rate_1��j7��JzI       6%�	�i68���A�*;


total_lossEL�@

error_R7N?

learning_rate_1��j7:FcI       6%�	�68���A�*;


total_loss�^�@

error_R��T?

learning_rate_1��j7�ce�I       6%�	H�68���A�*;


total_lossn%�@

error_R)�L?

learning_rate_1��j7G�c�I       6%�	�A78���A�*;


total_loss�u�@

error_RJ==?

learning_rate_1��j7 �I       6%�	ׁ78���A�*;


total_lossƎ�@

error_R�}A?

learning_rate_1��j7fS�@I       6%�	��78���A�*;


total_loss�O�@

error_RXo=?

learning_rate_1��j7N��I       6%�	"88���A�*;


total_lossD,�@

error_R��Z?

learning_rate_1��j7ITu�I       6%�	�S88���A�*;


total_loss 3�@

error_R�J?

learning_rate_1��j7W4�
I       6%�	��88���A�*;


total_loss3t�@

error_R�@T?

learning_rate_1��j7 n�I       6%�	��88���A�*;


total_loss*f�@

error_R��R?

learning_rate_1��j7�bI       6%�	�98���A�*;


total_loss[�@

error_R,�Y?

learning_rate_1��j7^I       6%�	v\98���A�*;


total_lossm�@

error_RߙE?

learning_rate_1��j7�v6\I       6%�	�98���A�*;


total_lossx��@

error_R��b?

learning_rate_1��j7u�}I       6%�	��98���A�*;


total_lossV1A

error_R�I?

learning_rate_1��j7�ȂFI       6%�	�):8���A�*;


total_loss��@

error_R�n]?

learning_rate_1��j7�s�	I       6%�	>r:8���A�*;


total_lossM�@

error_R}8N?

learning_rate_1��j7���I       6%�	��:8���A�*;


total_loss9�@

error_R,:E?

learning_rate_1��j7���HI       6%�	� ;8���A�*;


total_loss�){@

error_R�"L?

learning_rate_1��j7MI�I       6%�	�G;8���A�*;


total_loss$�@

error_Rs�T?

learning_rate_1��j7�N��I       6%�	��;8���A�*;


total_lossZv�@

error_R�XS?

learning_rate_1��j7%gL1I       6%�	p�;8���A�*;


total_loss�g�@

error_R�|L?

learning_rate_1��j7<dq�I       6%�	�<8���A�*;


total_loss�CA

error_RNeQ?

learning_rate_1��j7u_zI       6%�	P<8���A�*;


total_loss2��@

error_R�uH?

learning_rate_1��j7J̫I       6%�	p�<8���A�*;


total_lossi��@

error_R1qL?

learning_rate_1��j7��d3I       6%�	��<8���A�*;


total_loss�X�@

error_R
�;?

learning_rate_1��j7�W��I       6%�	�=8���A�*;


total_loss�f�@

error_R�G?

learning_rate_1��j7���I       6%�	�V=8���A�*;


total_loss�H�@

error_R4�`?

learning_rate_1��j7���I       6%�	��=8���A�*;


total_loss`l�@

error_RQ ]?

learning_rate_1��j7n�tI       6%�	C�=8���A�*;


total_loss&�@

error_RmM?

learning_rate_1��j7	J�dI       6%�	>8���A�*;


total_loss���@

error_RʙO?

learning_rate_1��j7q<BSI       6%�	3`>8���A�*;


total_loss8�@

error_R��E?

learning_rate_1��j78`�I       6%�	y�>8���A�*;


total_loss���@

error_RwS?

learning_rate_1��j7ޟ�I       6%�	�>8���A�*;


total_loss��@

error_R��8?

learning_rate_1��j7�OYI       6%�	)?8���A�*;


total_lossF�@

error_R��C?

learning_rate_1��j7F~��I       6%�	$h?8���A�*;


total_loss���@

error_R$�W?

learning_rate_1��j7\VܳI       6%�	̫?8���A�*;


total_loss���@

error_Rx�;?

learning_rate_1��j7JwdZI       6%�	��?8���A�*;


total_loss㲡@

error_R�I?

learning_rate_1��j7pL9�I       6%�	k-@8���A�*;


total_loss֙�@

error_RJ?

learning_rate_1��j7u3b�I       6%�	0m@8���A�*;


total_loss���@

error_R4rc?

learning_rate_1��j7�G`I       6%�	b�@8���A�*;


total_loss�	�@

error_R�F?

learning_rate_1��j7i�9>I       6%�	��@8���A�*;


total_lossS��@

error_R<Dc?

learning_rate_1��j7��G�I       6%�	�4A8���A�*;


total_loss�ώ@

error_R�zQ?

learning_rate_1��j7FL�I       6%�	yA8���A�*;


total_loss���@

error_R��A?

learning_rate_1��j7��I       6%�	��A8���A�*;


total_loss	\�@

error_R)�T?

learning_rate_1��j7m�N�I       6%�	CB8���A�*;


total_loss��d@

error_Ro�7?

learning_rate_1��j7���hI       6%�	�HB8���A�*;


total_loss,�S@

error_Rd�??

learning_rate_1��j7�p�sI       6%�	��B8���A�*;


total_lossC�@

error_R��a?

learning_rate_1��j7�[6I       6%�	�B8���A�*;


total_loss���@

error_R�K?

learning_rate_1��j7�?�I       6%�	�C8���A�*;


total_loss]�@

error_Rn�\?

learning_rate_1��j7���qI       6%�	0TC8���A�*;


total_loss$�@

error_R�hb?

learning_rate_1��j7��jI       6%�	G�C8���A�*;


total_lossw��@

error_R)�T?

learning_rate_1��j74*%�I       6%�	��C8���A�*;


total_loss >�@

error_R��G?

learning_rate_1��j76�'I       6%�	G)D8���A�*;


total_loss-�@

error_RFi]?

learning_rate_1��j7}[�=I       6%�	�kD8���A�*;


total_lossm͝@

error_R��F?

learning_rate_1��j7���HI       6%�	�D8���A�*;


total_loss��h@

error_R��L?

learning_rate_1��j7�2rI       6%�	�D8���A�*;


total_loss�V�@

error_R�T?

learning_rate_1��j7�[6	I       6%�	\TE8���A�*;


total_loss:�A

error_RlY?

learning_rate_1��j7�GѪI       6%�	��E8���A�*;


total_losslh�@

error_R{�C?

learning_rate_1��j7T�I       6%�	.F8���A�*;


total_lossY� A

error_R*�A?

learning_rate_1��j7�2�
I       6%�	�YF8���A�*;


total_lossO �@

error_R��b?

learning_rate_1��j7����I       6%�	àF8���A�*;


total_loss���@

error_R��Q?

learning_rate_1��j7%��I       6%�	�F8���A�*;


total_loss=�A

error_Ri]?

learning_rate_1��j7�ZOI       6%�	i#G8���A�*;


total_loss[��@

error_R��D?

learning_rate_1��j7���I       6%�	DfG8���A�*;


total_loss�b�@

error_R�??

learning_rate_1��j7N��I       6%�	0�G8���A�*;


total_lossvW�@

error_R-!\?

learning_rate_1��j7ޞ�I       6%�	��G8���A�*;


total_lossx�l@

error_R�D?

learning_rate_1��j7@��SI       6%�	�0H8���A�*;


total_loss��@

error_Rr;?

learning_rate_1��j7XK�I       6%�	NzH8���A�*;


total_losswi�@

error_R4R?

learning_rate_1��j7Y��I       6%�	�H8���A�*;


total_lossZ9�@

error_R�]E?

learning_rate_1��j7�Ow�I       6%�	�I8���A�*;


total_loss��@

error_R��N?

learning_rate_1��j75��I       6%�	�JI8���A�*;


total_loss��@

error_RֵB?

learning_rate_1��j7U60�I       6%�	��I8���A�*;


total_lossL��@

error_RV�i?

learning_rate_1��j7bK�pI       6%�	��I8���A�*;


total_loss2��@

error_R%W?

learning_rate_1��j7%A2]I       6%�	�J8���A�*;


total_loss�ǳ@

error_R\�N?

learning_rate_1��j7����I       6%�	o_J8���A�*;


total_lossM/�@

error_R�a?

learning_rate_1��j7��XI       6%�	�J8���A�*;


total_loss4s�@

error_R}�J?

learning_rate_1��j7�e�I       6%�	��J8���A�*;


total_loss�'�@

error_Rn O?

learning_rate_1��j7�q0`I       6%�	}&K8���A�*;


total_loss��@

error_R�T?

learning_rate_1��j7��I       6%�	�fK8���A�*;


total_loss2¬@

error_R�??

learning_rate_1��j7Qߙ�I       6%�	�K8���A�*;


total_lossYA

error_R�3X?

learning_rate_1��j7J}vI       6%�	T�K8���A�*;


total_lossɊA

error_RH�M?

learning_rate_1��j7��_�I       6%�	,L8���A�*;


total_loss/^r@

error_R.K?

learning_rate_1��j7��cI       6%�	�lL8���A�*;


total_loss���@

error_R@t\?

learning_rate_1��j7'�5I       6%�	?�L8���A�*;


total_loss/��@

error_R�Z?

learning_rate_1��j7����I       6%�	��L8���A�*;


total_loss���@

error_RQ?

learning_rate_1��j7}&�I       6%�	,M8���A�*;


total_loss�Я@

error_R�8K?

learning_rate_1��j7n�5uI       6%�	GtM8���A�*;


total_lossT�@

error_R.&c?

learning_rate_1��j7�^I       6%�	�M8���A�*;


total_loss�]@

error_RΦO?

learning_rate_1��j7�E��I       6%�	{�M8���A�*;


total_loss���@

error_R:�E?

learning_rate_1��j7(B�;I       6%�	=N8���A�*;


total_lossσ�@

error_R�8H?

learning_rate_1��j7dc΋I       6%�	�N8���A�*;


total_loss�#A

error_R�W?

learning_rate_1��j7�i��I       6%�	7�N8���A�*;


total_loss*��@

error_R�{=?

learning_rate_1��j7��F�I       6%�	'	O8���A�*;


total_loss��@

error_R��V?

learning_rate_1��j7x8�JI       6%�	tKO8���A�*;


total_lossz�@

error_R]pM?

learning_rate_1��j7��A�I       6%�	��O8���A�*;


total_loss���@

error_R4;C?

learning_rate_1��j7G�=�I       6%�	��O8���A�*;


total_loss�R�@

error_R�$A?

learning_rate_1��j7=�7I       6%�	P8���A�*;


total_loss��@

error_R;�Q?

learning_rate_1��j7D��I       6%�	�SP8���A�*;


total_loss���@

error_RT >?

learning_rate_1��j7�vY�I       6%�	�P8���A�*;


total_loss���@

error_R?S?

learning_rate_1��j7X��I       6%�	��P8���A�*;


total_loss I�@

error_R;�K?

learning_rate_1��j7T�'dI       6%�	Q8���A�*;


total_lossJ�@

error_R� U?

learning_rate_1��j7�I       6%�	�\Q8���A�*;


total_loss�T�@

error_R�E?

learning_rate_1��j7����I       6%�	d�Q8���A�*;


total_lossWr�@

error_Rot]?

learning_rate_1��j7��m�I       6%�	�Q8���A�*;


total_lossZә@

error_R�L?

learning_rate_1��j7w�I       6%�	�"R8���A�*;


total_loss8��@

error_R��;?

learning_rate_1��j7 ��KI       6%�	�eR8���A�*;


total_loss�"�@

error_R�-d?

learning_rate_1��j7v��|I       6%�	�R8���A�*;


total_loss/��@

error_R�"\?

learning_rate_1��j7��FI       6%�	1�R8���A�*;


total_lossY�@

error_RȸO?

learning_rate_1��j7%�|�I       6%�	�1S8���A�*;


total_loss�_�@

error_RX�T?

learning_rate_1��j7�xU�I       6%�	\zS8���A�*;


total_loss;��@

error_R@�5?

learning_rate_1��j7�h�$I       6%�	��S8���A�*;


total_loss+܋@

error_R�N?

learning_rate_1��j7�{?�I       6%�	3T8���A�*;


total_loss;u�@

error_R2�B?

learning_rate_1��j7�
�I       6%�	�BT8���A�*;


total_loss-�@

error_R�JG?

learning_rate_1��j7�ȅ	I       6%�	�T8���A�*;


total_lossT�@

error_R�k6?

learning_rate_1��j7�xj)I       6%�	�T8���A�*;


total_loss�!�@

error_Rf�R?

learning_rate_1��j7�j�I       6%�	U8���A�*;


total_loss�ʉ@

error_Rʓ@?

learning_rate_1��j7�^8I       6%�	�kU8���A�*;


total_loss.�@

error_R��V?

learning_rate_1��j7'��I       6%�	g�U8���A�*;


total_loss�׹@

error_R{^?

learning_rate_1��j7�{�I       6%�	��U8���A�*;


total_loss_ǽ@

error_R1^?

learning_rate_1��j7��7I       6%�	�9V8���A�*;


total_loss`
�@

error_R�S?

learning_rate_1��j7<���I       6%�	�{V8���A�*;


total_loss�v�@

error_RE/J?

learning_rate_1��j7~�X&I       6%�	��V8���A�*;


total_loss)��@

error_R�,N?

learning_rate_1��j7�^��I       6%�	[�V8���A�*;


total_loss��@

error_R{M?

learning_rate_1��j7�
�I       6%�	@W8���A�*;


total_loss�_�@

error_R#�P?

learning_rate_1��j7��I       6%�	+�W8���A�*;


total_loss��@

error_Rf�M?

learning_rate_1��j7����I       6%�	��W8���A�*;


total_loss��@

error_RdN?

learning_rate_1��j7ja�CI       6%�	wX8���A�*;


total_loss���@

error_RK?

learning_rate_1��j7�L�I       6%�	)HX8���A�*;


total_lossƄA

error_R �S?

learning_rate_1��j7y�I       6%�	�X8���A�*;


total_loss�n�@

error_RW|X?

learning_rate_1��j7��nI       6%�	@�X8���A�*;


total_loss
<�@

error_R�X?

learning_rate_1��j7�M�kI       6%�	�Y8���A�*;


total_loss��@

error_R �W?

learning_rate_1��j7�9��I       6%�	/WY8���A�*;


total_loss���@

error_R4�S?

learning_rate_1��j7���I       6%�	՝Y8���A�*;


total_lossR��@

error_R�Y?

learning_rate_1��j7�ΐ�I       6%�	�Y8���A�*;


total_loss}��@

error_R�U?

learning_rate_1��j7~Yq�I       6%�	HZ8���A�*;


total_loss�ʧ@

error_RWCV?

learning_rate_1��j7[�"�I       6%�	�^Z8���A�*;


total_loss*T�@

error_R�	O?

learning_rate_1��j7��z�I       6%�	�Z8���A�*;


total_loss�@�@

error_R��H?

learning_rate_1��j7���I       6%�	X�Z8���A�*;


total_lossD`z@

error_R��A?

learning_rate_1��j7|Ъ&I       6%�	B$[8���A�*;


total_loss�`�@

error_RS?

learning_rate_1��j7,)��I       6%�	Od[8���A�*;


total_loss��@

error_R[?

learning_rate_1��j7%��I       6%�	��[8���A�*;


total_losst�@

error_R�]?

learning_rate_1��j7���I       6%�	�[8���A�*;


total_loss��@

error_R��L?

learning_rate_1��j7�8!jI       6%�	�+\8���A�*;


total_losso� A

error_R}�Q?

learning_rate_1��j70�3I       6%�	m\8���A�*;


total_lossܼ�@

error_R�T=?

learning_rate_1��j7>���I       6%�	�\8���A�*;


total_loss��@

error_R)A?

learning_rate_1��j7�/��I       6%�	>�\8���A�*;


total_loss��@

error_R8!<?

learning_rate_1��j7߄Y�I       6%�	�0]8���A�*;


total_loss8�^@

error_R�Z?

learning_rate_1��j7}UV�I       6%�	xt]8���A�*;


total_loss���@

error_RlI?

learning_rate_1��j7��אI       6%�	;�]8���A�*;


total_loss��A

error_R�JP?

learning_rate_1��j7���:I       6%�	O�]8���A�*;


total_loss�5�@

error_R\;?

learning_rate_1��j7y* �I       6%�	�A^8���A�*;


total_lossi-w@

error_RZ�V?

learning_rate_1��j7� g�I       6%�	��^8���A�*;


total_loss:�w@

error_R��E?

learning_rate_1��j7'd�I       6%�	��^8���A�*;


total_lossJ��@

error_RvPZ?

learning_rate_1��j7{�)�I       6%�	�_8���A�*;


total_loss2��@

error_RZn`?

learning_rate_1��j7WѳI       6%�	l]_8���A�*;


total_loss*R�@

error_R�J?

learning_rate_1��j7�}�II       6%�	��_8���A�*;


total_loss�J�@

error_RaqO?

learning_rate_1��j7�ø�I       6%�	��_8���A�*;


total_loss���@

error_R��F?

learning_rate_1��j7��o�I       6%�	K.`8���A�*;


total_lossE��@

error_R:�]?

learning_rate_1��j7E��I       6%�	!c8���A�*;


total_loss1��@

error_R�L?

learning_rate_1��j7��dRI       6%�	�gc8���A�*;


total_loss�d�@

error_RA&??

learning_rate_1��j7dMܿI       6%�	F�c8���A�*;


total_loss��@

error_R�4T?

learning_rate_1��j7m.nI       6%�	��c8���A�*;


total_loss��A

error_R:kK?

learning_rate_1��j7V�y�I       6%�	5Ad8���A�*;


total_loss	��@

error_RA�K?

learning_rate_1��j7�d�I       6%�	��d8���A�*;


total_loss�H�@

error_R�wF?

learning_rate_1��j7��I       6%�	��d8���A�*;


total_losst�@

error_R�B?

learning_rate_1��j7�N'�I       6%�	�e8���A�*;


total_lossh�{@

error_R��=?

learning_rate_1��j7�	��I       6%�	��e8���A�*;


total_loss67�@

error_R�8?

learning_rate_1��j75��I       6%�	\�e8���A�*;


total_loss|Ǽ@

error_R��F?

learning_rate_1��j7p��I       6%�	�(f8���A�*;


total_loss�у@

error_R�>W?

learning_rate_1��j7!+0I       6%�	�f8���A�*;


total_loss�ѧ@

error_R��Q?

learning_rate_1��j7���I       6%�	
�f8���A�*;


total_loss�7�@

error_R�@T?

learning_rate_1��j7]��I       6%�	Ag8���A�*;


total_loss���@

error_R��9?

learning_rate_1��j7�bI       6%�	�Vg8���A�*;


total_losstG�@

error_R]R?

learning_rate_1��j7�%��I       6%�	՘g8���A�*;


total_loss%�@

error_R�KX?

learning_rate_1��j7�[K�I       6%�	8�g8���A�*;


total_loss*��@

error_Rr�M?

learning_rate_1��j7Y���I       6%�	�h8���A�*;


total_loss`k�@

error_RRjJ?

learning_rate_1��j7+�4�I       6%�	w`h8���A�*;


total_loss�
�@

error_R�mR?

learning_rate_1��j7��I       6%�	�h8���A�*;


total_loss/��@

error_R�1j?

learning_rate_1��j7���6I       6%�	��h8���A�*;


total_loss�-�@

error_R@�P?

learning_rate_1��j7���I       6%�	6/i8���A�*;


total_losswN�@

error_Rmb?

learning_rate_1��j7���I       6%�	�pi8���A�*;


total_lossL��@

error_R�`T?

learning_rate_1��j7��<�I       6%�	��i8���A�*;


total_lossi&�@

error_R��T?

learning_rate_1��j7�r��I       6%�	{�i8���A�*;


total_loss���@

error_R�$L?

learning_rate_1��j7e�-BI       6%�	*Hj8���A�*;


total_lossW�@

error_RvX?

learning_rate_1��j7�jH�I       6%�	ևj8���A�*;


total_loss��@

error_R�U?

learning_rate_1��j7��I       6%�	��j8���A�*;


total_loss�,�@

error_RTG?

learning_rate_1��j7���I       6%�	Lk8���A�*;


total_loss��A

error_RT�L?

learning_rate_1��j7���I       6%�	aNk8���A�*;


total_loss�6�@

error_R��P?

learning_rate_1��j7r㢌I       6%�	�k8���A�*;


total_loss
0�@

error_R�4R?

learning_rate_1��j732��I       6%�	��k8���A�*;


total_loss�k�@

error_R�S[?

learning_rate_1��j7#��WI       6%�	�l8���A�*;


total_loss\Sr@

error_R?H?

learning_rate_1��j7���MI       6%�	bQl8���A�*;


total_loss?�@

error_R��V?

learning_rate_1��j7DVCmI       6%�	�l8���A�*;


total_lossXg�@

error_Rt'\?

learning_rate_1��j7(�|I       6%�	�l8���A�*;


total_loss0͆@

error_RA�W?

learning_rate_1��j71X�=I       6%�	�m8���A�*;


total_loss�s@

error_R��9?

learning_rate_1��j7����I       6%�	~bm8���A�*;


total_loss���@

error_RSQX?

learning_rate_1��j72�H�I       6%�	v�m8���A�*;


total_loss4S�@

error_Rs�8?

learning_rate_1��j7836I       6%�	
n8���A�*;


total_loss�n�@

error_RmQV?

learning_rate_1��j7��	I       6%�	bn8���A�*;


total_loss���@

error_RgR?

learning_rate_1��j7.�	�I       6%�	6�n8���A�*;


total_loss���@

error_R:q\?

learning_rate_1��j7���I       6%�	��n8���A�*;


total_loss�_A

error_RZ�H?

learning_rate_1��j7h���I       6%�	�Jo8���A�*;


total_loss��@

error_R�U?

learning_rate_1��j7�jW8I       6%�	��o8���A�*;


total_lossv^m@

error_R@}V?

learning_rate_1��j7N�:ZI       6%�	��o8���A�*;


total_loss��@

error_RF�E?

learning_rate_1��j7
w=I       6%�	�p8���A�*;


total_lossҸ�@

error_Rr>W?

learning_rate_1��j7�~��I       6%�	2Xp8���A�*;


total_lossҒ�@

error_R�P?

learning_rate_1��j7��lI       6%�	��p8���A�*;


total_loss�|�@

error_R�fM?

learning_rate_1��j7ƭ��I       6%�	\q8���A�*;


total_lossy�@

error_R �J?

learning_rate_1��j7}{��I       6%�	�^q8���A�*;


total_loss��@

error_R�k5?

learning_rate_1��j7@x�nI       6%�	��q8���A�*;


total_loss�^�@

error_R.�[?

learning_rate_1��j76��yI       6%�	��q8���A�*;


total_loss3P�@

error_R�!R?

learning_rate_1��j7�?�I       6%�	�7r8���A�*;


total_loss�¦@

error_R�=?

learning_rate_1��j7hR�$I       6%�	�{r8���A�*;


total_lossɝ�@

error_R׎A?

learning_rate_1��j7m��I       6%�	]�r8���A�*;


total_loss��@

error_R�
J?

learning_rate_1��j7Ο�9I       6%�	�s8���A�*;


total_loss��@

error_R{�c?

learning_rate_1��j7#oI       6%�	as8���A�*;


total_loss�Ԙ@

error_Ra�C?

learning_rate_1��j7�S"�I       6%�	��s8���A�*;


total_loss��@

error_R��R?

learning_rate_1��j7��4I       6%�	��s8���A�*;


total_lossݕ�@

error_ReBF?

learning_rate_1��j7�*��I       6%�	?=t8���A�*;


total_lossT��@

error_R�UX?

learning_rate_1��j7�#�I       6%�	�t8���A�*;


total_loss��@

error_R_?

learning_rate_1��j7�n�I       6%�	��t8���A�*;


total_loss��@

error_R�q<?

learning_rate_1��j7�U�(I       6%�	�u8���A�*;


total_loss���@

error_R�vO?

learning_rate_1��j7��$I       6%�	�ou8���A�*;


total_loss�l�@

error_R�R?

learning_rate_1��j7�@'�I       6%�	�u8���A�*;


total_loss�՗@

error_R��m?

learning_rate_1��j7��2�I       6%�	��u8���A�*;


total_lossџ�@

error_R��L?

learning_rate_1��j7ؗSI       6%�	bDv8���A�*;


total_loss�c�@

error_R�xi?

learning_rate_1��j7��jI       6%�	,�v8���A�*;


total_loss�c�@

error_RԊC?

learning_rate_1��j7�O[I       6%�	�v8���A�*;


total_losse��@

error_R�F?

learning_rate_1��j7��K�I       6%�	�w8���A�*;


total_lossA��@

error_RVLM?

learning_rate_1��j7éz�I       6%�	zTw8���A�*;


total_lossJ��@

error_R� `?

learning_rate_1��j7Yf��I       6%�	��w8���A�*;


total_loss�A

error_R�yD?

learning_rate_1��j7Ԏ�6I       6%�	��w8���A�*;


total_loss��@

error_R�M?

learning_rate_1��j7�5��I       6%�	J*x8���A�*;


total_loss�r�@

error_RTD?

learning_rate_1��j7�%�I       6%�	�ox8���A�*;


total_loss3��@

error_RZ?

learning_rate_1��j7�ھI       6%�	��x8���A�*;


total_lossqA�@

error_R�U?

learning_rate_1��j7F��I       6%�	y8���A�*;


total_loss\:�@

error_R�GA?

learning_rate_1��j7ږAI       6%�	�Xy8���A�*;


total_loss���@

error_R]R?

learning_rate_1��j7j���I       6%�	��y8���A�*;


total_loss�0�@

error_R��>?

learning_rate_1��j74�I       6%�	i�y8���A�*;


total_loss��@

error_RW�O?

learning_rate_1��j7Y{WNI       6%�	G#z8���A�*;


total_loss�8�@

error_RQ\9?

learning_rate_1��j7|��I       6%�	@dz8���A�*;


total_loss��@

error_R,�V?

learning_rate_1��j7���I       6%�	\�z8���A�*;


total_loss���@

error_R�jA?

learning_rate_1��j7��zI       6%�	R�z8���A�*;


total_loss�1�@

error_RܚV?

learning_rate_1��j7��{I       6%�	�A{8���A�*;


total_loss@;�@

error_R�V?

learning_rate_1��j7<t��I       6%�	�{8���A�*;


total_loss�{@

error_RW�@?

learning_rate_1��j7$M��I       6%�	��{8���A�*;


total_loss�Me@

error_R��F?

learning_rate_1��j7C���I       6%�	�|8���A�*;


total_lossȯA

error_R�R?

learning_rate_1��j7(�]qI       6%�	�M|8���A�*;


total_lossĘ�@

error_R�7Z?

learning_rate_1��j7����I       6%�	��|8���A�*;


total_lossQO�@

error_R�{C?

learning_rate_1��j7�=�I       6%�	a�|8���A�*;


total_loss�d�@

error_R4E=?

learning_rate_1��j7��I       6%�	k}8���A�*;


total_loss�d�@

error_RnE?

learning_rate_1��j7�� I       6%�	�X}8���A�*;


total_loss���@

error_R�cN?

learning_rate_1��j7�F�I       6%�	��}8���A�*;


total_lossZ@�@

error_RzL?

learning_rate_1��j7�`I       6%�	h�}8���A�*;


total_lossR�@

error_R@�>?

learning_rate_1��j7kw�I       6%�	'"~8���A�*;


total_lossmס@

error_Rl4?

learning_rate_1��j7�?�uI       6%�	�`~8���A�*;


total_loss-!�@

error_RR@[?

learning_rate_1��j7�N�[I       6%�	P�~8���A�*;


total_loss�Q�@

error_RMF?

learning_rate_1��j77�m�I       6%�	��~8���A�*;


total_lossV�}@

error_R��=?

learning_rate_1��j7�hI       6%�	%8���A�*;


total_loss�G�@

error_RϛD?

learning_rate_1��j7�{��I       6%�	Ue8���A�*;


total_loss�q}@

error_R�]N?

learning_rate_1��j7Rϥ�I       6%�	��8���A�*;


total_loss�=�@

error_R�U?

learning_rate_1��j7ɥW}I       6%�	��8���A�*;


total_loss�*�@

error_R!.D?

learning_rate_1��j7O܆I       6%�	#*�8���A�*;


total_loss�7�@

error_R�B?

learning_rate_1��j7��I       6%�	l�8���A�*;


total_loss^�@

error_R�mG?

learning_rate_1��j7����I       6%�	���8���A�*;


total_loss��@

error_R��_?

learning_rate_1��j7��5"I       6%�	T�8���A�*;


total_loss�Y�@

error_R�N?

learning_rate_1��j7J+S#I       6%�	.�8���A�*;


total_loss8�l@

error_R��O?

learning_rate_1��j7�A��I       6%�	Vo�8���A�*;


total_lossl��@

error_R�PS?

learning_rate_1��j7
��I       6%�	��8���A�*;


total_lossa��@

error_R�mL?

learning_rate_1��j7��ϥI       6%�	��8���A�*;


total_lossB4�@

error_R�YV?

learning_rate_1��j78X�I       6%�	7�8���A�*;


total_loss���@

error_RʖO?

learning_rate_1��j7�RJI       6%�	�x�8���A�*;


total_loss"-�@

error_Re�S?

learning_rate_1��j78+�I       6%�	���8���A�*;


total_lossf��@

error_RFT\?

learning_rate_1��j7E�<�I       6%�	y��8���A�*;


total_loss-$�@

error_RCuR?

learning_rate_1��j7~��}I       6%�	/H�8���A�*;


total_loss�Y�@

error_RT�]?

learning_rate_1��j7i)sWI       6%�	;��8���A�*;


total_loss
��@

error_ROO?

learning_rate_1��j7�o��I       6%�	�̃8���A�*;


total_lossa�@

error_R�SS?

learning_rate_1��j7DJ٘I       6%�	J�8���A�*;


total_lossl��@

error_R��I?

learning_rate_1��j7D�I       6%�	�V�8���A�*;


total_loss$��@

error_R!�;?

learning_rate_1��j73�I       6%�	��8���A�*;


total_lossO��@

error_R��M?

learning_rate_1��j7-T�I       6%�	�ׄ8���A�*;


total_loss���@

error_RJ3Z?

learning_rate_1��j7�N�^I       6%�	�#�8���A�*;


total_lossԷ�@

error_RC�7?

learning_rate_1��j7�m!I       6%�	�8���A�*;


total_loss���@

error_RŮK?

learning_rate_1��j7Ӏ��I       6%�	��8���A�*;


total_loss�i@

error_R�0,?

learning_rate_1��j7����I       6%�		'�8���A�*;


total_loss���@

error_R��U?

learning_rate_1��j7��bDI       6%�	�j�8���A�*;


total_loss�b�@

error_R��\?

learning_rate_1��j7?�I       6%�	1��8���A�*;


total_loss)�@

error_RZJh?

learning_rate_1��j7/ݢI       6%�	2�8���A�*;


total_loss_q�@

error_RF-\?

learning_rate_1��j7i�KI       6%�	�2�8���A�*;


total_lossg�@

error_R��S?

learning_rate_1��j7�]��I       6%�	�v�8���A�*;


total_loss���@

error_R�kJ?

learning_rate_1��j7A�:I       6%�	7��8���A�*;


total_lossS��@

error_R��V?

learning_rate_1��j7k<\I       6%�	� �8���A�*;


total_loss���@

error_R��D?

learning_rate_1��j7W�H�I       6%�	�B�8���A�*;


total_loss��@

error_Rh�e?

learning_rate_1��j7ćL�I       6%�	���8���A�*;


total_loss���@

error_R��L?

learning_rate_1��j7M\��I       6%�	�Ĉ8���A�*;


total_lossA�@

error_R]B[?

learning_rate_1��j7oWI       6%�	b	�8���A�*;


total_loss+A

error_R�B?

learning_rate_1��j7S��I       6%�	JQ�8���A�*;


total_loss킸@

error_R!SU?

learning_rate_1��j7�?��I       6%�	���8���A�*;


total_loss�*�@

error_R3�V?

learning_rate_1��j71A�DI       6%�	ۉ8���A�*;


total_loss!��@

error_R�E?

learning_rate_1��j7�׺#I       6%�	m�8���A�*;


total_loss�'�@

error_R
�I?

learning_rate_1��j7|I       6%�	!c�8���A�*;


total_loss?+A

error_RT�S?

learning_rate_1��j7�~I       6%�	���8���A�*;


total_loss��A

error_R#PY?

learning_rate_1��j7*��CI       6%�	��8���A�*;


total_loss��@

error_R�G?

learning_rate_1��j7 �|I       6%�	~)�8���A�*;


total_loss�ɉ@

error_RL7I?

learning_rate_1��j7�u*�I       6%�	�o�8���A�*;


total_loss���@

error_R[P?

learning_rate_1��j7�H+�I       6%�	x��8���A�*;


total_loss��@

error_R �F?

learning_rate_1��j7~$��I       6%�	��8���A�*;


total_loss���@

error_R*5O?

learning_rate_1��j70�pI       6%�	�5�8���A�*;


total_loss�PA

error_R�1I?

learning_rate_1��j7�ǘ�I       6%�	�w�8���A�*;


total_loss�2�@

error_R3�Y?

learning_rate_1��j7k�˱I       6%�	漌8���A�*;


total_lossB��@

error_R}�N?

learning_rate_1��j7 ��I       6%�	e��8���A�*;


total_loss��@

error_R�U?

learning_rate_1��j7��D�I       6%�	�C�8���A�*;


total_loss�:o@

error_R;4Y?

learning_rate_1��j7&��wI       6%�	���8���A�*;


total_loss�ǒ@

error_R�I?

learning_rate_1��j7���nI       6%�	Nʍ8���A�*;


total_loss���@

error_Rm^?

learning_rate_1��j7*�9.I       6%�	�
�8���A�*;


total_loss��@

error_R��R?

learning_rate_1��j7�w��I       6%�	�K�8���A�*;


total_loss[�@

error_R kA?

learning_rate_1��j7�!VI       6%�	c��8���A�*;


total_lossDw�@

error_Rl�R?

learning_rate_1��j7�t>�I       6%�	�̎8���A�*;


total_loss�,�@

error_R�PN?

learning_rate_1��j7#,I       6%�	U�8���A�*;


total_lossAR�@

error_R��N?

learning_rate_1��j7�x>�I       6%�	�O�8���A�*;


total_loss�ڜ@

error_RQJ?

learning_rate_1��j71\��I       6%�	Ɛ�8���A�*;


total_loss�(�@

error_R��R?

learning_rate_1��j7(_9%I       6%�	�Џ8���A�*;


total_loss9�@

error_R]�R?

learning_rate_1��j7.�P�I       6%�	g�8���A�*;


total_loss�(�@

error_R(�P?

learning_rate_1��j7>�JI       6%�	cS�8���A�*;


total_loss�v@

error_R9P?

learning_rate_1��j7H76hI       6%�	^��8���A�*;


total_loss��@

error_RT5M?

learning_rate_1��j7��zI       6%�	�Ԑ8���A�*;


total_loss���@

error_RRO?

learning_rate_1��j7�y�LI       6%�	��8���A�*;


total_loss[:�@

error_RW?

learning_rate_1��j7]��tI       6%�	�8���A�*;


total_loss�|�@

error_R�;?

learning_rate_1��j7�^�I       6%�	�ב8���A�*;


total_loss���@

error_Rd�T?

learning_rate_1��j7V�PLI       6%�	'�8���A�*;


total_loss�8�@

error_R�CX?

learning_rate_1��j7_q�NI       6%�	Zz�8���A�*;


total_loss��@

error_R��B?

learning_rate_1��j7��"I       6%�	WΒ8���A�*;


total_loss�?�@

error_R�1>?

learning_rate_1��j7e;��I       6%�	)"�8���A�*;


total_loss�*�@

error_R�]?

learning_rate_1��j7�@C�I       6%�	���8���A�*;


total_loss{_�@

error_R��F?

learning_rate_1��j7,�	I       6%�	���8���A�*;


total_loss���@

error_Rx�S?

learning_rate_1��j7޵	lI       6%�	�O�8���A�*;


total_loss�I�@

error_R;?

learning_rate_1��j7�=fI       6%�	F��8���A�*;


total_lossV�@

error_R��J?

learning_rate_1��j7���mI       6%�	%��8���A�*;


total_loss8��@

error_R�BQ?

learning_rate_1��j7��\I       6%�	�R�8���A�*;


total_loss�1�@

error_R��Q?

learning_rate_1��j7a=�eI       6%�	Ҿ�8���A�*;


total_loss�!�@

error_R��D?

learning_rate_1��j7���I       6%�	",�8���A�*;


total_loss�h�@

error_R7�U?

learning_rate_1��j7��B�I       6%�	t��8���A�*;


total_loss�P�@

error_RN?

learning_rate_1��j7	�_I       6%�	z��8���A�*;


total_loss���@

error_R_�S?

learning_rate_1��j7r���I       6%�	�8���A�*;


total_loss�M�@

error_R�4?

learning_rate_1��j7=�Z�I       6%�	o�8���A�*;


total_loss$c�@

error_R*[?

learning_rate_1��j7��)I       6%�	@6�8���A�*;


total_loss���@

error_Rq�;?

learning_rate_1��j7i��VI       6%�	q��8���A�*;


total_loss��@

error_R�.C?

learning_rate_1��j7��G�I       6%�	���8���A�*;


total_loss���@

error_RX?

learning_rate_1��j7	<�I       6%�	^D�8���A�*;


total_loss�@

error_R�U?

learning_rate_1��j7�@��I       6%�	ő�8���A�*;


total_loss��@

error_R��P?

learning_rate_1��j7'+I       6%�	���8���A�*;


total_loss���@

error_R�iQ?

learning_rate_1��j7��%BI       6%�	@�8���A�*;


total_loss܈�@

error_R�Z?

learning_rate_1��j7�+�I       6%�	F��8���A�*;


total_loss;x�@

error_R�hK?

learning_rate_1��j7��O�I       6%�	+�8���A�*;


total_lossa��@

error_R�uL?

learning_rate_1��j7�f�5I       6%�	j5�8���A�*;


total_lossv��@

error_R�'R?

learning_rate_1��j7b�HI       6%�	�|�8���A�*;


total_lossɬ�@

error_R�CV?

learning_rate_1��j7q7aI       6%�	{ԛ8���A�*;


total_loss�ܥ@

error_R3U]?

learning_rate_1��j7�j��I       6%�	F#�8���A�*;


total_loss�Һ@

error_RQ$G?

learning_rate_1��j7��rzI       6%�	�e�8���A�*;


total_loss-��@

error_RnN?

learning_rate_1��j7O4}mI       6%�	笜8���A�*;


total_loss�tb@

error_R��C?

learning_rate_1��j7��q�I       6%�	��8���A�*;


total_lossE��@

error_R�1Q?

learning_rate_1��j7t{o�I       6%�	�=�8���A�*;


total_loss ��@

error_RHWd?

learning_rate_1��j7ɴ��I       6%�	���8���A�*;


total_loss���@

error_Re�T?

learning_rate_1��j7�N��I       6%�	\ӝ8���A�*;


total_loss�@

error_RC�C?

learning_rate_1��j7t��I       6%�	�+�8���A�*;


total_loss���@

error_R�Q?

learning_rate_1��j7x9^I       6%�	K��8���A�*;


total_loss#i�@

error_R6tD?

learning_rate_1��j7���I       6%�	��8���A�*;


total_lossf9�@

error_R�2G?

learning_rate_1��j7I��"I       6%�	�:�8���A�*;


total_loss���@

error_R�L?

learning_rate_1��j7��QI       6%�	��8���A�*;


total_loss}<�@

error_R2b?

learning_rate_1��j7
'��I       6%�	��8���A�*;


total_loss�1�@

error_R�\?

learning_rate_1��j7X��I       6%�	�J�8���A�*;


total_loss�|�@

error_R�D?

learning_rate_1��j7rQ��I       6%�	���8���A�*;


total_lossl��@

error_R:NW?

learning_rate_1��j7��e�I       6%�	?�8���A�*;


total_lossn�@

error_R,W?

learning_rate_1��j7���I       6%�	�7�8���A�*;


total_loss�k�@

error_R�0\?

learning_rate_1��j7�f�I       6%�	�|�8���A�*;


total_loss6{�@

error_R��P?

learning_rate_1��j7����I       6%�	��8���A�*;


total_loss���@

error_R��M?

learning_rate_1��j7��JgI       6%�	��8���A�*;


total_lossZt�@

error_R��M?

learning_rate_1��j7\�BI       6%�	rJ�8���A�*;


total_loss���@

error_Rq
X?

learning_rate_1��j7�)��I       6%�	}��8���A�*;


total_loss��@

error_R�J?

learning_rate_1��j7�*�VI       6%�	7֢8���A�*;


total_loss��@

error_R�rX?

learning_rate_1��j7��BI       6%�	��8���A�*;


total_loss��@

error_R��D?

learning_rate_1��j7�B}�I       6%�	x�8���A�*;


total_loss�Ύ@

error_R3�I?

learning_rate_1��j7���/I       6%�	 ˣ8���A�*;


total_loss�l�@

error_RT�O?

learning_rate_1��j7�S�I       6%�	a�8���A�*;


total_loss���@

error_Ra�I?

learning_rate_1��j7.�G�I       6%�	Zb�8���A�*;


total_lossz��@

error_R!a?

learning_rate_1��j7���I       6%�	`��8���A�*;


total_lossMO�@

error_R:�M?

learning_rate_1��j75���I       6%�	8�8���A�*;


total_loss#ef@

error_RQ�D?

learning_rate_1��j7:�I       6%�	���8���A�*;


total_loss��@

error_R&O?

learning_rate_1��j7����I       6%�	�ܥ8���A�*;


total_loss�@

error_R\�H?

learning_rate_1��j7���>I       6%�	�&�8���A�*;


total_loss�>�@

error_R�`?

learning_rate_1��j7e�bI       6%�	
s�8���A�*;


total_loss�ވ@

error_R�=A?

learning_rate_1��j7Lg��I       6%�	I��8���A�*;


total_loss{�@

error_R��=?

learning_rate_1��j7�?<I       6%�	j��8���A�*;


total_loss���@

error_RwM?

learning_rate_1��j7��o�I       6%�	�@�8���A�*;


total_lossS�@

error_R3EC?

learning_rate_1��j75�(!I       6%�	̓�8���A�*;


total_loss�|�@

error_Rݝ[?

learning_rate_1��j7�{�I       6%�	ȧ8���A�*;


total_loss���@

error_R�/K?

learning_rate_1��j7#��I       6%�	T�8���A�*;


total_loss�A�@

error_Rn�F?

learning_rate_1��j7��
�I       6%�	�T�8���A�*;


total_loss�9�@

error_R�1V?

learning_rate_1��j7�!܊I       6%�	@��8���A�*;


total_loss�x@

error_R�>?

learning_rate_1��j7h�tI       6%�	�ި8���A�*;


total_loss��@

error_R��U?

learning_rate_1��j7(ďI       6%�	�$�8���A�*;


total_loss��L@

error_R��@?

learning_rate_1��j7���I       6%�	�k�8���A�*;


total_loss�ej@

error_R��Y?

learning_rate_1��j7��I       6%�	_��8���A�*;


total_losst��@

error_R�E?

learning_rate_1��j7���I       6%�	���8���A�*;


total_loss���@

error_R�8?

learning_rate_1��j76�S�I       6%�	dD�8���A�*;


total_loss\r@

error_R�]Y?

learning_rate_1��j7�I6�I       6%�	���8���A�*;


total_loss���@

error_RĖU?

learning_rate_1��j7���{I       6%�	��8���A�*;


total_loss�@

error_R<�H?

learning_rate_1��j7G��I       6%�	O>�8���A�*;


total_lossz��@

error_Rl�I?

learning_rate_1��j7X� I       6%�	D��8���A�*;


total_loss�)�@

error_R�_?

learning_rate_1��j7����I       6%�	�ͫ8���A�*;


total_loss1)�@

error_Rz�R?

learning_rate_1��j7['��I       6%�	�8���A�*;


total_loss ��@

error_R��O?

learning_rate_1��j7��d}I       6%�	���8���A�*;


total_loss�_�@

error_R.D?

learning_rate_1��j7i�IUI       6%�	@ɬ8���A�*;


total_loss��@

error_R^V?

learning_rate_1��j7���I       6%�	��8���A�*;


total_lossCx�@

error_RHbD?

learning_rate_1��j7��#I       6%�	^�8���A�*;


total_loss�e�@

error_R@Ic?

learning_rate_1��j7�$NYI       6%�	稭8���A�*;


total_loss8'�@

error_R�X?

learning_rate_1��j7}!��I       6%�	\�8���A�*;


total_lossTʵ@

error_R�tg?

learning_rate_1��j7b�I       6%�		<�8���A�*;


total_lossS%�@

error_RH�U?

learning_rate_1��j7
+i|I       6%�	���8���A�*;


total_loss�~�@

error_R��j?

learning_rate_1��j7����I       6%�	-Ү8���A�*;


total_lossQ��@

error_R��J?

learning_rate_1��j7�fZI       6%�	�8���A�*;


total_loss���@

error_R�Z?

learning_rate_1��j7��eI       6%�	�V�8���A�*;


total_loss�E�@

error_R��F?

learning_rate_1��j7ܲ��I       6%�	:��8���A�*;


total_loss�M�@

error_R�G?

learning_rate_1��j7�vy�I       6%�	y߯8���A�*;


total_loss6��@

error_R��b?

learning_rate_1��j7FDI       6%�	|6�8���A�*;


total_loss\h�@

error_R.�Y?

learning_rate_1��j7.n�<I       6%�	G��8���A�*;


total_lossH��@

error_R� C?

learning_rate_1��j7�:�gI       6%�	�ư8���A�*;


total_loss���@

error_R�^?

learning_rate_1��j74�6I       6%�	��8���A�*;


total_loss�5�@

error_RckI?

learning_rate_1��j7�K��I       6%�	
h�8���A�*;


total_loss�@

error_RZ�b?

learning_rate_1��j7�Wg�I       6%�	C��8���A�*;


total_loss��@

error_R�DH?

learning_rate_1��j7D�EI       6%�	��8���A�*;


total_loss�A�@

error_R��<?

learning_rate_1��j7�޿I       6%�	�2�8���A�*;


total_loss��	A

error_R�k?

learning_rate_1��j7B���I       6%�	+w�8���A�*;


total_loss�)�@

error_RzM?

learning_rate_1��j7�kىI       6%�	w��8���A�*;


total_loss��@

error_R(B?

learning_rate_1��j7��x$I       6%�	���8���A�*;


total_loss���@

error_R)�8?

learning_rate_1��j7��I       6%�	�?�8���A�*;


total_loss�ğ@

error_RZ�3?

learning_rate_1��j7Tkd�I       6%�	'��8���A�*;


total_loss,q@

error_R�_]?

learning_rate_1��j7�
��I       6%�	6Ƴ8���A�*;


total_lossR�@

error_R��J?

learning_rate_1��j7M�T�I       6%�	l�8���A�*;


total_loss�PA

error_R݀G?

learning_rate_1��j7���I       6%�	QL�8���A�*;


total_losss�k@

error_Rt[?

learning_rate_1��j7�~�RI       6%�	���8���A�*;


total_loss�~@

error_R��P?

learning_rate_1��j7�b��I       6%�	�մ8���A�*;


total_loss`��@

error_R�=?

learning_rate_1��j7=�sI       6%�	p�8���A�*;


total_lossL��@

error_Rf;?

learning_rate_1��j7D��ZI       6%�	���8���A�*;


total_loss���@

error_R�$W?

learning_rate_1��j7�W��I       6%�	�̵8���A�*;


total_loss$@A

error_R�I[?

learning_rate_1��j7S�0�I       6%�	��8���A�*;


total_loss`��@

error_R�OV?

learning_rate_1��j77'��I       6%�	Q�8���A�*;


total_lossq�@

error_R��E?

learning_rate_1��j7eM�I       6%�	$��8���A�*;


total_lossRĬ@

error_R��W?

learning_rate_1��j7���YI       6%�	]ֶ8���A�*;


total_loss{�A

error_R��a?

learning_rate_1��j7L,�I       6%�	��8���A�*;


total_loss���@

error_R�WO?

learning_rate_1��j7�֔�I       6%�	�Y�8���A�*;


total_loss�۔@

error_ROY?

learning_rate_1��j7P�I       6%�	���8���A�*;


total_loss�n�@

error_R@=C?

learning_rate_1��j7����I       6%�	�ݷ8���A�*;


total_loss�&�@

error_R�B?

learning_rate_1��j7#n��I       6%�	@ �8���A�*;


total_loss��@

error_Ri$e?

learning_rate_1��j7��*�I       6%�	 c�8���A�*;


total_loss3��@

error_RMf?

learning_rate_1��j7ޕ�&I       6%�	_��8���A�*;


total_loss���@

error_R�O?

learning_rate_1��j7�:Y�I       6%�	*�8���A�*;


total_loss�4�@

error_R�f^?

learning_rate_1��j7{rR�I       6%�	8*�8���A�*;


total_loss��@

error_R�R?

learning_rate_1��j7�"��I       6%�	j�8���A�*;


total_loss�@

error_R@gA?

learning_rate_1��j7�rI       6%�	���8���A�*;


total_lossS��@

error_R��T?

learning_rate_1��j7��S{I       6%�	>��8���A�*;


total_loss�c�@

error_R�XF?

learning_rate_1��j7�|��I       6%�	E-�8���A�*;


total_loss��A

error_R�W?

learning_rate_1��j7C]�bI       6%�	�o�8���A�*;


total_lossaՙ@

error_R��O?

learning_rate_1��j7�sI       6%�	/��8���A�*;


total_loss���@

error_RmM?

learning_rate_1��j7��A�I       6%�	A��8���A�*;


total_losshf@

error_R V=?

learning_rate_1��j7R�lI       6%�	^6�8���A�*;


total_lossȿA

error_RhY?

learning_rate_1��j7Ij�I       6%�	�v�8���A�*;


total_loss��y@

error_R�2E?

learning_rate_1��j7�)0I       6%�	���8���A�*;


total_loss>߈@

error_RH�J?

learning_rate_1��j7xc�NI       6%�	��8���A�*;


total_lossœ�@

error_RśQ?

learning_rate_1��j7�t��I       6%�	�@�8���A�*;


total_loss��@

error_R;R[?

learning_rate_1��j7��KI       6%�	���8���A�*;


total_loss��A

error_R$eS?

learning_rate_1��j7��'I       6%�	3ļ8���A�*;


total_loss���@

error_R�c?

learning_rate_1��j7�%��I       6%�	G�8���A�*;


total_loss��@

error_R��N?

learning_rate_1��j7��\5I       6%�	�G�8���A�*;


total_lossW�x@

error_R#�E?

learning_rate_1��j7"@1I       6%�	���8���A�*;


total_lossl~�@

error_R�K?

learning_rate_1��j7����I       6%�	�ѽ8���A�*;


total_lossS��@

error_R3�T?

learning_rate_1��j7��mI       6%�	�0�8���A�*;


total_loss�:�@

error_R�)W?

learning_rate_1��j7o�8I       6%�	�t�8���A�*;


total_loss䈊@

error_R!�K?

learning_rate_1��j71���I       6%�	l��8���A�*;


total_lossC�@

error_Rh.N?

learning_rate_1��j7�e�I       6%�	��8���A�*;


total_loss?��@

error_RlO5?

learning_rate_1��j7��I       6%�	E=�8���A�*;


total_lossz�@

error_RRN;?

learning_rate_1��j7�:@XI       6%�	?�8���A�*;


total_loss��|@

error_Ra#A?

learning_rate_1��j7`�	�I       6%�	�¿8���A�*;


total_lossd��@

error_R�cP?

learning_rate_1��j7Vt�zI       6%�	 �8���A�*;


total_loss�@

error_Rn�b?

learning_rate_1��j7b{��I       6%�	9H�8���A�*;


total_loss#q�@

error_R1�@?

learning_rate_1��j7'�7�I       6%�	:��8���A�*;


total_lossn�@

error_R��J?

learning_rate_1��j7�3%�I       6%�	��8���A�*;


total_loss�8�@

error_RͲV?

learning_rate_1��j7~���I       6%�	�8���A�*;


total_lossx�@

error_R�n6?

learning_rate_1��j7�VI�I       6%�	�Q�8���A�*;


total_loss_@

error_R;XJ?

learning_rate_1��j7�~�jI       6%�	��8���A�*;


total_loss*�@

error_R�7?

learning_rate_1��j7MEA/I       6%�	���8���A�*;


total_loss���@

error_R��X?

learning_rate_1��j7[s�	I       6%�	�&�8���A�*;


total_loss�I�@

error_Rf�O?

learning_rate_1��j7���I       6%�	rg�8���A�*;


total_loss���@

error_R�<?

learning_rate_1��j7��/.I       6%�	K��8���A�*;


total_loss�x�@

error_R$�A?

learning_rate_1��j7�ǵwI       6%�	���8���A�*;


total_loss��@

error_R��O?

learning_rate_1��j7ܮ��I       6%�	�(�8���A�*;


total_loss�F�@

error_R��K?

learning_rate_1��j7��ZWI       6%�	�k�8���A�*;


total_loss(�@

error_RW�M?

learning_rate_1��j7TZ*I       6%�	Ҭ�8���A�*;


total_lossQ~�@

error_RģF?

learning_rate_1��j7LtڼI       6%�	��8���A�*;


total_loss,�e@

error_R�rC?

learning_rate_1��j7�+p|I       6%�	�=�8���A�*;


total_loss[7�@

error_R̮B?

learning_rate_1��j7�~�I       6%�	N��8���A�*;


total_loss�|�@

error_R� F?

learning_rate_1��j7O I       6%�	���8���A�*;


total_loss
A

error_RQHC?

learning_rate_1��j7�В�I       6%�	9�8���A�*;


total_loss��@

error_R;�C?

learning_rate_1��j7Z�QjI       6%�	���8���A�*;


total_loss���@

error_R��O?

learning_rate_1��j7���I       6%�	%��8���A�*;


total_loss���@

error_R��G?

learning_rate_1��j7�G�,I       6%�	v2�8���A�*;


total_loss��@

error_R��N?

learning_rate_1��j7,$c+I       6%�	y�8���A�*;


total_losss��@

error_R�iU?

learning_rate_1��j7}�Z�I       6%�	P��8���A�*;


total_loss-�@

error_R��h?

learning_rate_1��j7��2I       6%�	8�8���A�*;


total_loss�_�@

error_Rm�L?

learning_rate_1��j7?�7:I       6%�	3Y�8���A�*;


total_loss��@

error_RLdN?

learning_rate_1��j7�V��I       6%�	ћ�8���A�*;


total_loss���@

error_R�^Q?

learning_rate_1��j7e���I       6%�	���8���A�*;


total_loss�m�@

error_R�?E?

learning_rate_1��j7��I       6%�	��8���A�*;


total_loss�/�@

error_R�]U?

learning_rate_1��j7���I       6%�	�d�8���A�*;


total_loss�@

error_R��W?

learning_rate_1��j7�I       6%�	��8���A�*;


total_loss��@

error_R.[?

learning_rate_1��j7r�n%I       6%�	c��8���A�*;


total_loss�$�@

error_R{M4?

learning_rate_1��j7�:�I       6%�	>>�8���A�*;


total_loss�@

error_R�GL?

learning_rate_1��j7�%E�I       6%�	���8���A�*;


total_loss�ǹ@

error_R�AU?

learning_rate_1��j7c�7I       6%�	p��8���A�*;


total_loss�d@

error_R��P?

learning_rate_1��j7�;��I       6%�	��8���A�*;


total_loss�@

error_R)�>?

learning_rate_1��j7���kI       6%�	JK�8���A�*;


total_lossu��@

error_R�\5?

learning_rate_1��j7<`�pI       6%�	���8���A�*;


total_loss���@

error_R7RI?

learning_rate_1��j7$�i�I       6%�	8��8���A�*;


total_loss�L�@

error_Rl7H?

learning_rate_1��j7v��I       6%�	-4�8���A�*;


total_loss��@

error_R.&S?

learning_rate_1��j7�'s�I       6%�	�~�8���A�*;


total_lossS��@

error_R�F?

learning_rate_1��j7����I       6%�	��8���A�*;


total_loss�6d@

error_R��B?

learning_rate_1��j7҄�!I       6%�	��8���A�*;


total_loss!q�@

error_RR�Q?

learning_rate_1��j7*���I       6%�	_�8���A�*;


total_loss*��@

error_R�bM?

learning_rate_1��j7�NI       6%�	���8���A�*;


total_loss�@

error_R�PL?

learning_rate_1��j7_�εI       6%�	��8���A�*;


total_loss��A

error_R��Z?

learning_rate_1��j7�r�I       6%�	�H�8���A�*;


total_loss���@

error_R�Z?

learning_rate_1��j7R
��I       6%�	���8���A�*;


total_lossښ�@

error_Ra[>?

learning_rate_1��j7/�եI       6%�	���8���A�*;


total_loss_K@

error_R��K?

learning_rate_1��j72��I       6%�	��8���A�*;


total_loss�Ӧ@

error_R2R?

learning_rate_1��j7�C}I       6%�	U_�8���A�*;


total_loss�
�@

error_R�59?

learning_rate_1��j7��v�I       6%�	C��8���A�*;


total_loss���@

error_R��_?

learning_rate_1��j7�dSI       6%�	���8���A�*;


total_lossL��@

error_R_�I?

learning_rate_1��j7��<�I       6%�	�?�8���A�*;


total_loss��@

error_R�oG?

learning_rate_1��j7�p<uI       6%�	���8���A�*;


total_lossdWA

error_RʋH?

learning_rate_1��j7ؙEI       6%�	���8���A�*;


total_loss�u�@

error_R��B?

learning_rate_1��j7���WI       6%�	��8���A�*;


total_loss�L�@

error_R��J?

learning_rate_1��j7����I       6%�	�p�8���A�*;


total_loss�uY@

error_R�:?

learning_rate_1��j79ڳI       6%�	V��8���A�*;


total_lossj�@

error_R��8?

learning_rate_1��j7m H)I       6%�	��8���A�*;


total_loss���@

error_R2d?

learning_rate_1��j7Ei%�I       6%�	�K�8���A�*;


total_lossv��@

error_R�L?

learning_rate_1��j7B�
�I       6%�	-��8���A�*;


total_loss���@

error_RE�H?

learning_rate_1��j7ɰ	 I       6%�	���8���A�*;


total_loss��@

error_R�S?

learning_rate_1��j7~�I       6%�	��8���A�*;


total_loss�<�@

error_R3XY?

learning_rate_1��j7c�I       6%�	�]�8���A�*;


total_loss2ْ@

error_RR�:?

learning_rate_1��j7�(�I       6%�	A��8���A�*;


total_loss�E�@

error_R��j?

learning_rate_1��j7.���I       6%�	3��8���A�*;


total_lossɤ�@

error_R��^?

learning_rate_1��j7���#I       6%�	�#�8���A�*;


total_loss�A

error_R�vJ?

learning_rate_1��j70�?I       6%�	�d�8���A�*;


total_loss� @

error_R�O?

learning_rate_1��j7d�2I       6%�	t��8���A�*;


total_loss�0�@

error_R<�T?

learning_rate_1��j7.��I       6%�	0��8���A�*;


total_loss��RA

error_R��I?

learning_rate_1��j7��0�I       6%�	�,�8���A�*;


total_loss�_T@

error_Rf�:?

learning_rate_1��j7Z� _I       6%�	r�8���A�*;


total_lossʵ�@

error_R�.L?

learning_rate_1��j7�!0GI       6%�	s��8���A�*;


total_lossr��@

error_R7�B?

learning_rate_1��j7�&�I       6%�	b��8���A�*;


total_loss C�@

error_Rj�H?

learning_rate_1��j7:�Q�I       6%�	�Q�8���A�*;


total_loss��@

error_R��V?

learning_rate_1��j7����I       6%�	h��8���A�*;


total_loss�"�@

error_R��(?

learning_rate_1��j7�
�I       6%�	��8���A�*;


total_loss"ǂ@

error_R3�I?

learning_rate_1��j7�Ӆ�I       6%�	u0�8���A�*;


total_loss��@

error_R��=?

learning_rate_1��j7�ZhI       6%�	Mr�8���A�*;


total_loss�A�@

error_R�;V?

learning_rate_1��j7U�zI       6%�	��8���A�*;


total_loss�D�@

error_R�T?

learning_rate_1��j7�5�I       6%�	H��8���A�*;


total_loss-Q�@

error_R� B?

learning_rate_1��j7(���I       6%�	W:�8���A�*;


total_loss��k@

error_RV�E?

learning_rate_1��j7/�S�I       6%�	.{�8���A�*;


total_lossi@�@

error_RT�`?

learning_rate_1��j7,�;�I       6%�	Խ�8���A�*;


total_lossܗ_@

error_R?vW?

learning_rate_1��j7�y��I       6%�	) �8���A�*;


total_lossT��@

error_R�(S?

learning_rate_1��j72GI       6%�	b@�8���A�*;


total_loss e�@

error_RM�H?

learning_rate_1��j7���I       6%�	׀�8���A�*;


total_loss�/�@

error_Ri�Q?

learning_rate_1��j7i1��I       6%�	���8���A�*;


total_loss�O�@

error_R�N?

learning_rate_1��j7-2m�I       6%�	>�8���A�*;


total_loss2I�@

error_R|"A?

learning_rate_1��j7 �(I       6%�	0K�8���A�*;


total_loss���@

error_R��P?

learning_rate_1��j7�La!I       6%�	��8���A�*;


total_loss�b�@

error_R��O?

learning_rate_1��j7�K��I       6%�	���8���A�*;


total_loss��@

error_R�=Q?

learning_rate_1��j7�ǬII       6%�	��8���A�*;


total_loss���@

error_R�>?

learning_rate_1��j7��I       6%�	3P�8���A�*;


total_loss�ڹ@

error_R?F?

learning_rate_1��j7��AiI       6%�	ɖ�8���A�*;


total_lossT�@

error_R0X?

learning_rate_1��j7��u�I       6%�	M��8���A�*;


total_lossmG�@

error_R`\K?

learning_rate_1��j7�Z5I       6%�	;�8���A�*;


total_loss�ڮ@

error_R{~Y?

learning_rate_1��j7[��%I       6%�	e�8���A�*;


total_lossC��@

error_R�T?

learning_rate_1��j7�X�I       6%�	��8���A�*;


total_loss���@

error_R!�I?

learning_rate_1��j7�z%@I       6%�	���8���A�*;


total_losssͪ@

error_R�`R?

learning_rate_1��j7�4�I       6%�	�,�8���A�*;


total_loss�}@

error_R�O?

learning_rate_1��j7��'�I       6%�	~n�8���A�*;


total_loss���@

error_R(�c?

learning_rate_1��j7h�MI       6%�	l��8���A�*;


total_loss�U�@

error_R��B?

learning_rate_1��j7qI��I       6%�	=��8���A�*;


total_loss���@

error_R<JV?

learning_rate_1��j7�25�I       6%�	�0�8���A�*;


total_loss�>�@

error_R��`?

learning_rate_1��j7x�cI       6%�	lt�8���A�*;


total_lossF%x@

error_RT�O?

learning_rate_1��j7�[SI       6%�	_��8���A�*;


total_loss��@

error_R�p\?

learning_rate_1��j7��f�I       6%�	���8���A�*;


total_loss�_�@

error_R<�>?

learning_rate_1��j7e%��I       6%�	2?�8���A�*;


total_loss�+�@

error_RvS?

learning_rate_1��j7�v�I       6%�	 ��8���A�*;


total_loss$d�@

error_Rs�B?

learning_rate_1��j7��<I       6%�	��8���A�*;


total_loss_/ A

error_R��F?

learning_rate_1��j7]n�nI       6%�	��8���A�*;


total_loss��~@

error_Rc�W?

learning_rate_1��j7T��I       6%�	�S�8���A�*;


total_lossv�@

error_R��C?

learning_rate_1��j7����I       6%�	���8���A�*;


total_lossv^�@

error_R3�F?

learning_rate_1��j7��KI       6%�	o��8���A�*;


total_loss�@

error_RiGG?

learning_rate_1��j7����I       6%�	��8���A�*;


total_lossm܊@

error_ROJ?

learning_rate_1��j7t�AI       6%�	
f�8���A�*;


total_lossܨ�@

error_R��L?

learning_rate_1��j7&	�_I       6%�	ɲ�8���A�*;


total_loss�`�@

error_R�C?

learning_rate_1��j7t?I       6%�	�	�8���A�*;


total_loss�`�@

error_R$�\?

learning_rate_1��j7�B��I       6%�	P�8���A�*;


total_loss���@

error_R�3C?

learning_rate_1��j7;��I       6%�	]��8���A�*;


total_lossֽ�@

error_R��??

learning_rate_1��j7�Y��I       6%�	��8���A�*;


total_loss���@

error_RJ�P?

learning_rate_1��j7���1I       6%�	A+�8���A�*;


total_loss���@

error_R:�A?

learning_rate_1��j7&�-�I       6%�	2l�8���A�*;


total_loss���@

error_R�'O?

learning_rate_1��j7I��dI       6%�	���8���A�*;


total_loss 5�@

error_R`�A?

learning_rate_1��j7&^�XI       6%�	l��8���A�*;


total_loss�O�@

error_RĞQ?

learning_rate_1��j7�/�\I       6%�	�1�8���A�*;


total_lossR4�@

error_R�%_?

learning_rate_1��j7���I       6%�	;x�8���A�*;


total_loss��@

error_RvMF?

learning_rate_1��j7��/I       6%�	`��8���A�*;


total_loss��A

error_R�AD?

learning_rate_1��j7����I       6%�	7�8���A�*;


total_loss�Ҵ@

error_RçK?

learning_rate_1��j7G Q�I       6%�	#D�8���A�*;


total_loss��@

error_R��2?

learning_rate_1��j7jt�I       6%�	-��8���A�*;


total_losssO�@

error_R8L?

learning_rate_1��j7�v9�I       6%�	r��8���A�*;


total_lossT�@

error_R�]Z?

learning_rate_1��j7�R�:I       6%�		�8���A�*;


total_lossC �@

error_R�kW?

learning_rate_1��j7&���I       6%�	�T�8���A�*;


total_losskN�@

error_R;M?

learning_rate_1��j7d*�I       6%�	��8���A�*;


total_loss�+A

error_R8O?

learning_rate_1��j7�~�
I       6%�	g��8���A�*;


total_lossӀ�@

error_R�{]?

learning_rate_1��j7�Of~I       6%�	";�8���A�*;


total_loss���@

error_R��b?

learning_rate_1��j7Vmv^I       6%�	�z�8���A�*;


total_loss&��@

error_R}zZ?

learning_rate_1��j7:�?I       6%�	$��8���A�*;


total_loss}w�@

error_R�ed?

learning_rate_1��j7a?�I       6%�	���8���A�*;


total_loss]�@

error_R �M?

learning_rate_1��j7���0I       6%�	�@�8���A�*;


total_lossS,A

error_R�mE?

learning_rate_1��j7�sV|I       6%�	���8���A�*;


total_lossE�&A

error_R��5?

learning_rate_1��j7���9I       6%�	��8���A�*;


total_lossM�@

error_R��e?

learning_rate_1��j7��HI       6%�	��8���A�*;


total_lossVxP@

error_R�N??

learning_rate_1��j7���9I       6%�	F�8���A�*;


total_loss���@

error_RR�S?

learning_rate_1��j7�'=�I       6%�	ʈ�8���A�*;


total_lossc�@

error_RcK?

learning_rate_1��j7r��I       6%�	���8���A�*;


total_loss��@

error_R��U?

learning_rate_1��j7d�3*I       6%�	:�8���A�*;


total_loss��@

error_R >K?

learning_rate_1��j7Cv�I       6%�	M�8���A�*;


total_loss�i�@

error_RcbP?

learning_rate_1��j7l!2I       6%�	@��8���A�*;


total_lossR-A

error_R��Y?

learning_rate_1��j7��X/I       6%�	���8���A�*;


total_lossҨ�@

error_Rl�C?

learning_rate_1��j7;62&I       6%�	��8���A�*;


total_loss}ƶ@

error_R��G?

learning_rate_1��j7�2 �I       6%�	"[�8���A�*;


total_loss�H�@

error_Rs�X?

learning_rate_1��j7q�,RI       6%�	���8���A�*;


total_lossi֊@

error_RJr=?

learning_rate_1��j7�NQI       6%�	?��8���A�*;


total_loss3+A

error_R��[?

learning_rate_1��j7���I       6%�	�&�8���A�*;


total_loss�z�@

error_RlmM?

learning_rate_1��j7|�YI       6%�	ik�8���A�*;


total_loss� �@

error_RO�Q?

learning_rate_1��j7f��gI       6%�	���8���A�*;


total_loss���@

error_R�E9?

learning_rate_1��j7�v��I       6%�	���8���A�*;


total_loss��A

error_RM�;?

learning_rate_1��j7z�I       6%�	�=�8���A�*;


total_loss�A�@

error_R��A?

learning_rate_1��j7�	�#I       6%�	��8���A�*;


total_loss*h�@

error_R�E?

learning_rate_1��j7���I       6%�	��8���A�*;


total_loss���@

error_R�I?

learning_rate_1��j7S\��I       6%�	L!�8���A�*;


total_lossIJ�@

error_R)�Q?

learning_rate_1��j7���bI       6%�	�i�8���A�*;


total_loss�@

error_R��B?

learning_rate_1��j7�I       6%�	��8���A�*;


total_lossJ��@

error_R_)C?

learning_rate_1��j7�B^�I       6%�	���8���A�*;


total_loss{&�@

error_R��_?

learning_rate_1��j7�G4I       6%�	C5�8���A�*;


total_lossE+�@

error_R$�H?

learning_rate_1��j7I~kOI       6%�	4u�8���A�*;


total_loss��@

error_R-�O?

learning_rate_1��j7:�p�I       6%�	��8���A�*;


total_lossڭA

error_R�MN?

learning_rate_1��j7��cI       6%�	r��8���A�*;


total_loss�<�@

error_R�aJ?

learning_rate_1��j7j�e�I       6%�	k>�8���A�*;


total_loss���@

error_RE�T?

learning_rate_1��j7v��I       6%�	��8���A�*;


total_loss�W�@

error_R%�A?

learning_rate_1��j7�zI       6%�	F��8���A�*;


total_loss ��@

error_R�bd?

learning_rate_1��j7�#I       6%�	��8���A�*;


total_loss�7�@

error_R �X?

learning_rate_1��j7r�I       6%�	�R�8���A�*;


total_loss�s�@

error_R�@U?

learning_rate_1��j7�%xI       6%�	w��8���A�*;


total_lossc��@

error_R��@?

learning_rate_1��j7��#�I       6%�	��8���A�*;


total_loss1m�@

error_R�F?

learning_rate_1��j7�m��I       6%�	�8���A�*;


total_loss�7�@

error_R�Jd?

learning_rate_1��j72���I       6%�	�a�8���A�*;


total_loss��@

error_R��E?

learning_rate_1��j7��UI       6%�	%��8���A�*;


total_loss�i�@

error_R�^L?

learning_rate_1��j7�1RI       6%�	���8���A�*;


total_loss�Tm@

error_Rqab?

learning_rate_1��j7�O�YI       6%�	�+�8���A�*;


total_loss�Q�@

error_R�Q?

learning_rate_1��j7>�I       6%�	7l�8���A�*;


total_loss��@

error_R�oS?

learning_rate_1��j7�VI       6%�	��8���A�*;


total_loss*گ@

error_RZ�H?

learning_rate_1��j7i�9(I       6%�	��8���A�*;


total_lossL9�@

error_R{�Z?

learning_rate_1��j7��I       6%�	�7�8���A�*;


total_loss
E�@

error_R�,P?

learning_rate_1��j7�xI       6%�	~�8���A�*;


total_loss��A

error_R}�R?

learning_rate_1��j7gj�zI       6%�	��8���A�*;


total_lossD��@

error_R�~Z?

learning_rate_1��j7���I       6%�	��8���A�*;


total_loss�+�@

error_R�N?

learning_rate_1��j7�UKMI       6%�	?�8���A�*;


total_loss �@

error_R��V?

learning_rate_1��j7YeL�I       6%�	��8���A�*;


total_lossWj�@

error_R!�Z?

learning_rate_1��j7D���I       6%�	(��8���A�*;


total_loss)��@

error_R��K?

learning_rate_1��j7k�-�I       6%�	��8���A�*;


total_loss���@

error_R��=?

learning_rate_1��j7#ꓳI       6%�	�d�8���A�*;


total_loss`/�@

error_R3�D?

learning_rate_1��j7�i�qI       6%�	���8���A�*;


total_loss�mp@

error_R�5V?

learning_rate_1��j7�ڋ�I       6%�	��8���A�*;


total_loss���@

error_RR�P?

learning_rate_1��j7e6�I       6%�	�H�8���A�*;


total_loss��@

error_R�;<?

learning_rate_1��j7�8xI       6%�	��8���A�*;


total_loss ��@

error_R�v]?

learning_rate_1��j7r_�I       6%�	���8���A�*;


total_loss��@

error_R�K?

learning_rate_1��j7B"�I       6%�	��8���A�*;


total_lossr&�@

error_R�V?

learning_rate_1��j7���I       6%�	�X�8���A�*;


total_loss%ק@

error_R��`?

learning_rate_1��j7���I       6%�	ڛ�8���A�*;


total_loss��@

error_R��P?

learning_rate_1��j7!��ZI       6%�	��8���A�*;


total_lossft�@

error_Rs�Q?

learning_rate_1��j7��5LI       6%�	�.�8���A�*;


total_loss%��@

error_R��E?

learning_rate_1��j7DuߕI       6%�	6q�8���A�*;


total_loss�Z�@

error_R$@T?

learning_rate_1��j7�e"�I       6%�	[��8���A�*;


total_loss�D�@

error_R1KI?

learning_rate_1��j7�_��I       6%�	v��8���A�*;


total_loss�N�@

error_RNc?

learning_rate_1��j7���I       6%�	�=�8���A�*;


total_loss ��@

error_R]�I?

learning_rate_1��j7zdI       6%�	 ��8���A�*;


total_loss���@

error_R��H?

learning_rate_1��j7��{I       6%�	i��8���A�*;


total_loss�<�@

error_Re�P?

learning_rate_1��j7��I       6%�	|�8���A�*;


total_loss���@

error_R�-H?

learning_rate_1��j7�҄I       6%�	�S�8���A�*;


total_loss�@

error_R��U?

learning_rate_1��j7$�PI       6%�	���8���A�*;


total_loss@�@

error_R�V?

learning_rate_1��j7~gOoI       6%�	��8���A�*;


total_loss�^�@

error_R�Z?

learning_rate_1��j7ۍH(I       6%�	�8���A�*;


total_loss�z�@

error_R��B?

learning_rate_1��j7��jUI       6%�	8Z�8���A�*;


total_loss���@

error_Rf<=?

learning_rate_1��j7��I       6%�	b��8���A�*;


total_loss|A

error_RqT?

learning_rate_1��j7I�I       6%�	���8���A�*;


total_lossE�@

error_R)(\?

learning_rate_1��j7m�1I       6%�	d'�8���A�*;


total_loss��@

error_R�~E?

learning_rate_1��j7*���I       6%�	5j�8���A�*;


total_loss���@

error_RUX?

learning_rate_1��j7w��|I       6%�	K��8���A�*;


total_loss\�@

error_R;�I?

learning_rate_1��j7���7I       6%�	���8���A�*;


total_loss��@

error_RE$E?

learning_rate_1��j72%�SI       6%�	�5�8���A�*;


total_lossrP�@

error_R��S?

learning_rate_1��j7 \ݮI       6%�	sv�8���A�*;


total_loss�ե@

error_R*W?

learning_rate_1��j7�zoI       6%�	޴�8���A�*;


total_loss{K�@

error_R�W?

learning_rate_1��j7�`��I       6%�	���8���A�*;


total_loss�;�@

error_R�YP?

learning_rate_1��j7��I       6%�	2�8���A�*;


total_lossD]�@

error_R�hN?

learning_rate_1��j7�F:ZI       6%�	Br�8���A�*;


total_lossvٕ@

error_Rqkh?

learning_rate_1��j7�<I       6%�	��8���A�*;


total_loss��@

error_R\xa?

learning_rate_1��j7;B�I       6%�	���8���A�*;


total_lossAt�@

error_R{�H?

learning_rate_1��j7�5�I       6%�	y8�8���A�*;


total_loss�o�@

error_R��H?

learning_rate_1��j7����I       6%�	cz�8���A�*;


total_loss�k�@

error_R�K?

learning_rate_1��j7�R��I       6%�	@��8���A�*;


total_losseI�@

error_R��L?

learning_rate_1��j7�K�I       6%�	x��8���A�*;


total_loss�@

error_R�F?

learning_rate_1��j7~sz�I       6%�	[? 9���A�*;


total_lossA�`@

error_RW�B?

learning_rate_1��j7?�qLI       6%�	y� 9���A�*;


total_loss��v@

error_RTXC?

learning_rate_1��j7�3�,I       6%�	�� 9���A�*;


total_lossBA

error_R�ZT?

learning_rate_1��j7wӱ&I       6%�	�9���A�*;


total_loss���@

error_R�<?

learning_rate_1��j7Z���I       6%�	S@9���A�*;


total_loss]T�@

error_R�Y?

learning_rate_1��j7�F:�I       6%�	(�9���A�*;


total_loss�B�@

error_Rn�T?

learning_rate_1��j7��X>I       6%�	��9���A�*;


total_loss;O�@

error_R�)_?

learning_rate_1��j7��I       6%�	M'9���A�*;


total_loss3~�@

error_R��U?

learning_rate_1��j7=EI       6%�	>i9���A�*;


total_lossHf�@

error_R�;K?

learning_rate_1��j7r�̷I       6%�	!�9���A�*;


total_loss�@

error_R��J?

learning_rate_1��j7K��I       6%�	��9���A�*;


total_loss��@

error_R][J?

learning_rate_1��j7��XaI       6%�	r*9���A�*;


total_loss�E�@

error_R�ka?

learning_rate_1��j7o�[JI       6%�	�o9���A�*;


total_loss@/�@

error_R��V?

learning_rate_1��j7�@nJI       6%�	Y�9���A�*;


total_lossA�@

error_R��S?

learning_rate_1��j7���9I       6%�	��9���A�*;


total_loss��@

error_RJ�@?

learning_rate_1��j7S'(MI       6%�	�89���A�*;


total_loss�J�@

error_R$K?

learning_rate_1��j7��EI       6%�	){9���A�*;


total_loss�ި@

error_R�M?

learning_rate_1��j7lw`I       6%�	6�9���A�*;


total_loss���@

error_Rm�N?

learning_rate_1��j7k��I       6%�	)�9���A�*;


total_losss�@

error_R�c?

learning_rate_1��j7!:6I       6%�	�P9���A�*;


total_lossN�A

error_Rs�T?

learning_rate_1��j7�=1�I       6%�	��9���A�*;


total_lossIw�@

error_RRN_?

learning_rate_1��j7�_"I       6%�	��9���A�*;


total_lossV�@

error_R�"J?

learning_rate_1��j7�*�I       6%�	�/9���A�*;


total_lossev�@

error_R�Oa?

learning_rate_1��j7.�`I       6%�	�o9���A�*;


total_lossh�@

error_Rc'R?

learning_rate_1��j7��ЧI       6%�	��9���A�*;


total_loss�Z@

error_R��D?

learning_rate_1��j7����I       6%�	�.9���A�*;


total_loss���@

error_Rf/I?

learning_rate_1��j7�M��I       6%�	�n9���A�*;


total_loss�+�@

error_R#ZF?

learning_rate_1��j7hK�I       6%�	.�9���A�*;


total_loss!%�@

error_R7Q?

learning_rate_1��j7��fII       6%�	E�9���A�*;


total_loss1t�@

error_RvF?

learning_rate_1��j7�AX^I       6%�	�99���A�*;


total_loss�Ř@

error_R�6B?

learning_rate_1��j7yD'�I       6%�	9~9���A�*;


total_loss3e�@

error_RQU?

learning_rate_1��j7�!�I       6%�	&�9���A�*;


total_lossY�@

error_R�`D?

learning_rate_1��j7�~9�I       6%�	��9���A�*;


total_loss�&�@

error_R��W?

learning_rate_1��j7Kc3�I       6%�	@	9���A�*;


total_lossG��@

error_R��Q?

learning_rate_1��j7&�YI       6%�	��	9���A�*;


total_loss|�@

error_R}�??

learning_rate_1��j7�d�I       6%�	g�	9���A�*;


total_loss���@

error_R͖R?

learning_rate_1��j7lU��I       6%�	V 
9���A�*;


total_loss�@

error_RQ�a?

learning_rate_1��j7�-/�I       6%�	nA
9���A�*;


total_loss�*�@

error_R\�S?

learning_rate_1��j7ߥ��I       6%�	)�
9���A�*;


total_loss6�@

error_R{RX?

learning_rate_1��j7��p�I       6%�	�
9���A�*;


total_loss�o�@

error_R;�X?

learning_rate_1��j7��9I       6%�	�9���A�*;


total_loss�z�@

error_R��c?

learning_rate_1��j7�HX{I       6%�	I9���A�*;


total_lossh�@

error_R.�J?

learning_rate_1��j7�o�I       6%�	n�9���A�*;


total_loss8��@

error_R\
M?

learning_rate_1��j7L쁊I       6%�	�9���A�*;


total_loss�Ē@

error_R�tB?

learning_rate_1��j7q�oI       6%�	�9���A�*;


total_lossy@

error_RW�W?

learning_rate_1��j7�Nf�I       6%�	�[9���A�*;


total_loss貇@

error_R�HH?

learning_rate_1��j7��w�I       6%�	��9���A�*;


total_loss
|�@

error_R��??

learning_rate_1��j7}�CI       6%�	��9���A�*;


total_loss�{�@

error_R��;?

learning_rate_1��j7��~�I       6%�	9���A�*;


total_loss1!�@

error_R��\?

learning_rate_1��j7�,smI       6%�	�_9���A�*;


total_loss�O�@

error_RQ�=?

learning_rate_1��j7����I       6%�	S�9���A�*;


total_loss���@

error_R`�J?

learning_rate_1��j7�?�I       6%�	n�9���A�*;


total_loss��@

error_R�B?

learning_rate_1��j7��I       6%�	#%9���A�*;


total_loss}��@

error_R�%S?

learning_rate_1��j7ef��I       6%�	Qf9���A�*;


total_loss�ض@

error_Ro�6?

learning_rate_1��j7x(��I       6%�	g�9���A�*;


total_loss�+�@

error_R�G?

learning_rate_1��j7j��xI       6%�	N�9���A�*;


total_loss-B�@

error_R&|U?

learning_rate_1��j7�+&I       6%�	�,9���A�*;


total_loss�Ա@

error_R�J?

learning_rate_1��j7�+�9I       6%�	�r9���A�*;


total_loss�y�@

error_RCyG?

learning_rate_1��j7!�grI       6%�	�9���A�*;


total_loss�ޡ@

error_RZ�U?

learning_rate_1��j78���I       6%�	��9���A�*;


total_loss��@

error_R�8B?

learning_rate_1��j74B�.I       6%�	]<9���A�*;


total_loss�K�@

error_R=mK?

learning_rate_1��j7��#�I       6%�	G}9���A�*;


total_loss�и@

error_R�Jg?

learning_rate_1��j79�mI       6%�	��9���A�*;


total_loss���@

error_R,y8?

learning_rate_1��j7j�Q�I       6%�	h9���A�*;


total_loss	��@

error_R��S?

learning_rate_1��j70�qQI       6%�	�J9���A�*;


total_loss
�A

error_RJ�Y?

learning_rate_1��j7/H/I       6%�	Ґ9���A�*;


total_loss[8�@

error_R��C?

learning_rate_1��j7AI       6%�	��9���A�*;


total_lossV�@

error_R��L?

learning_rate_1��j7`ڬwI       6%�	'9���A�*;


total_lossܡ�@

error_R��U?

learning_rate_1��j7/G�I       6%�	�^9���A�*;


total_lossW��@

error_R\(I?

learning_rate_1��j7JmI       6%�	V�9���A�*;


total_loss��@

error_R!!M?

learning_rate_1��j7y�LFI       6%�	��9���A�*;


total_lossX��@

error_R�ED?

learning_rate_1��j7�kM+I       6%�	�$9���A�*;


total_loss�sA

error_R�YB?

learning_rate_1��j7�C�$I       6%�	�e9���A�*;


total_loss�<�@

error_RO�F?

learning_rate_1��j7�A��I       6%�	��9���A�*;


total_loss�!�@

error_R�vR?

learning_rate_1��j7�V�I       6%�	�9���A�*;


total_loss�ء@

error_R*Y?

learning_rate_1��j7I�r�I       6%�	:9���A�*;


total_loss���@

error_R6�Q?

learning_rate_1��j7z�ڌI       6%�	�z9���A�*;


total_loss���@

error_R��N?

learning_rate_1��j7n.��I       6%�	s�9���A�*;


total_loss��@

error_Rw�K?

learning_rate_1��j7�b�I       6%�	��9���A�*;


total_loss��@

error_R��U?

learning_rate_1��j7o0�UI       6%�	cO9���A�*;


total_loss�z�@

error_Rm�N?

learning_rate_1��j7�"�hI       6%�	�9���A�*;


total_loss�X�@

error_R#"`?

learning_rate_1��j7�[�%I       6%�	��9���A�*;


total_loss���@

error_RzbY?

learning_rate_1��j7���KI       6%�	�89���A�*;


total_loss�f@

error_R�^Z?

learning_rate_1��j7��]�I       6%�	|9���A�*;


total_lossd��@

error_R��L?

learning_rate_1��j7ȲR�I       6%�	��9���A�*;


total_loss���@

error_Rұ[?

learning_rate_1��j7m��I       6%�	��9���A�*;


total_loss�Y�@

error_R��:?

learning_rate_1��j78s|I       6%�	?9���A�*;


total_loss6�A

error_R�fB?

learning_rate_1��j7�|5�I       6%�	��9���A�*;


total_loss:�@

error_R_^E?

learning_rate_1��j7�W��I       6%�	��9���A�*;


total_loss�n�@

error_R��T?

learning_rate_1��j7�(��I       6%�	�	9���A�*;


total_loss�÷@

error_RO�T?

learning_rate_1��j7ʡk~I       6%�	wJ9���A�*;


total_loss��@

error_RAzC?

learning_rate_1��j7'ֱ]I       6%�	-�9���A�*;


total_loss��@

error_Rj�K?

learning_rate_1��j7��:8I       6%�	)�9���A�*;


total_loss<� A

error_R�G?

learning_rate_1��j7'v��I       6%�	�9���A�*;


total_loss�g~@

error_RE�O?

learning_rate_1��j78,�I       6%�	�V9���A�*;


total_loss���@

error_R�J?

learning_rate_1��j7S�I       6%�	��9���A�*;


total_loss!�@

error_R��>?

learning_rate_1��j7�C$�I       6%�	Y�9���A�*;


total_loss��@

error_Rc>M?

learning_rate_1��j71�8RI       6%�	!9���A�*;


total_loss�B�@

error_R�+N?

learning_rate_1��j7��=I       6%�	=b9���A�*;


total_loss�}�@

error_R�H?

learning_rate_1��j7����I       6%�	��9���A�*;


total_loss��@

error_R="J?

learning_rate_1��j7�7,�I       6%�	o�9���A�*;


total_loss�&�@

error_R�F?

learning_rate_1��j7� I       6%�	1 9���A�*;


total_loss
�@

error_Rx�=?

learning_rate_1��j7� ��I       6%�	�^9���A�*;


total_loss{ܙ@

error_R��V?

learning_rate_1��j7��I       6%�	��9���A�*;


total_loss���@

error_R�WN?

learning_rate_1��j7��R�I       6%�	��9���A�*;


total_loss8�@

error_ROW?

learning_rate_1��j7��8�I       6%�	/ 9���A�*;


total_loss���@

error_R@�<?

learning_rate_1��j7bKQrI       6%�	�a9���A�*;


total_loss`;�@

error_RҌV?

learning_rate_1��j7���ZI       6%�	��9���A�*;


total_loss�@

error_R�EK?

learning_rate_1��j7	}@�I       6%�	��9���A�*;


total_loss��@

error_R�PR?

learning_rate_1��j7N/�I       6%�	.+9���A�*;


total_lossA)�@

error_R
hW?

learning_rate_1��j7<p՟I       6%�	Ck9���A�*;


total_loss��@

error_R`M\?

learning_rate_1��j7{�T�I       6%�	!�9���A�*;


total_lossv�A

error_RM�A?

learning_rate_1��j7,%�I       6%�	��9���A�*;


total_loss���@

error_R�G?

learning_rate_1��j7\���I       6%�	n09���A�*;


total_lossXr�@

error_R,%T?

learning_rate_1��j7�s��I       6%�	�r9���A�*;


total_lossn�@

error_R_�;?

learning_rate_1��j7f[�8I       6%�	~�9���A�*;


total_lossc�@

error_R�2H?

learning_rate_1��j7���5I       6%�	~�9���A�*;


total_loss��`@

error_R{�C?

learning_rate_1��j7� {I       6%�	�D9���A�*;


total_loss�$�@

error_R��Q?

learning_rate_1��j7�4S�I       6%�	�9���A�*;


total_loss~(�@

error_RԈL?

learning_rate_1��j7�嚊I       6%�	7�9���A�*;


total_loss���@

error_RJ(W?

learning_rate_1��j7�5��I       6%�	� 9���A�*;


total_loss2ξ@

error_R\kD?

learning_rate_1��j7�I       6%�	|^ 9���A�*;


total_loss��@

error_R�I?

learning_rate_1��j7#unnI       6%�	O� 9���A�*;


total_lossA�@

error_RbY?

learning_rate_1��j7r;��I       6%�	E� 9���A�*;


total_loss�v�@

error_R
�H?

learning_rate_1��j7;jy�I       6%�	q&!9���A�*;


total_loss�`	A

error_R&wW?

learning_rate_1��j7�I�I       6%�	�j!9���A�*;


total_loss҅�@

error_R��T?

learning_rate_1��j7lE��I       6%�	�!9���A�*;


total_loss�s�@

error_RE�G?

learning_rate_1��j7H�2I       6%�	��!9���A�*;


total_losshe�@

error_Rx�J?

learning_rate_1��j7hO`aI       6%�	p6"9���A�*;


total_loss��@

error_R1dZ?

learning_rate_1��j7>�aI       6%�	oy"9���A�*;


total_loss�G�@

error_Rh�T?

learning_rate_1��j7a�:I       6%�	�"9���A�*;


total_loss�#�@

error_R��X?

learning_rate_1��j7�I6�I       6%�	�"9���A�*;


total_loss�q�@

error_R;1@?

learning_rate_1��j7.`�KI       6%�	p@#9���A�*;


total_loss3�A

error_R��K?

learning_rate_1��j7��[`I       6%�	�#9���A�*;


total_loss�j�@

error_Ra�f?

learning_rate_1��j7P�vI       6%�	��#9���A�*;


total_loss�ø@

error_R�A?

learning_rate_1��j7��z7I       6%�	s$9���A�*;


total_loss��@

error_R��@?

learning_rate_1��j7�V�I       6%�	D$9���A�*;


total_loss�7�@

error_R�S?

learning_rate_1��j7��0II       6%�	8�$9���A�*;


total_lossbm�@

error_R��N?

learning_rate_1��j7���I       6%�	��$9���A�*;


total_loss���@

error_R��Z?

learning_rate_1��j7=�2�I       6%�	%9���A�*;


total_lossC��@

error_R�S?

learning_rate_1��j7R�D	I       6%�	�a%9���A�*;


total_loss;�@

error_R_M?

learning_rate_1��j7�$��I       6%�	��%9���A�*;


total_loss殤@

error_R��R?

learning_rate_1��j7WCI       6%�	&9���A�*;


total_loss�.�@

error_R1Z?

learning_rate_1��j7��{I       6%�	�Z&9���A�*;


total_loss1��@

error_R��=?

learning_rate_1��j7�cW�I       6%�	l�&9���A�*;


total_loss��@

error_RC?

learning_rate_1��j7� a�I       6%�	��&9���A�*;


total_lossV�@

error_R�9X?

learning_rate_1��j7�=T�I       6%�	.!'9���A�*;


total_loss���@

error_R%�L?

learning_rate_1��j7UftI       6%�	?f'9���A�*;


total_loss���@

error_RM�J?

learning_rate_1��j7\�"�I       6%�	��'9���A�*;


total_loss[I�@

error_R�3T?

learning_rate_1��j7�,qVI       6%�	7�'9���A�*;


total_loss�(�@

error_R|�P?

learning_rate_1��j7kq�I       6%�	�3(9���A�*;


total_lossJ�@

error_R�P?

learning_rate_1��j7�_j I       6%�	R{(9���A�*;


total_loss3l�@

error_RR�??

learning_rate_1��j70�q�I       6%�	��(9���A�*;


total_loss���@

error_R{*J?

learning_rate_1��j7��5I       6%�	�)9���A�*;


total_losso?	A

error_R(H?

learning_rate_1��j7M��CI       6%�	H)9���A�*;


total_loss�r�@

error_R(zI?

learning_rate_1��j7��c�I       6%�	s�)9���A�*;


total_loss��@

error_R{�S?

learning_rate_1��j7�5XCI       6%�	D�)9���A�*;


total_loss�ə@

error_R�F?

learning_rate_1��j7���I       6%�	v	*9���A�*;


total_loss?��@

error_R)�J?

learning_rate_1��j7{��{I       6%�	�J*9���A�*;


total_loss\�@

error_R�$O?

learning_rate_1��j7nz��I       6%�	��*9���A�*;


total_loss�)�@

error_R#�Y?

learning_rate_1��j7��#�I       6%�	��*9���A�*;


total_loss��@

error_R�bO?

learning_rate_1��j7D�HSI       6%�	�+9���A�*;


total_lossjr�@

error_RM�U?

learning_rate_1��j7�A�hI       6%�	�a+9���A�*;


total_loss]Q�@

error_RlV?

learning_rate_1��j7-�ǂI       6%�	�+9���A�*;


total_loss��@

error_R3&@?

learning_rate_1��j7y��HI       6%�	��+9���A�*;


total_loss�-�@

error_R��L?

learning_rate_1��j7�R1I       6%�	37,9���A�*;


total_loss v�@

error_R�'M?

learning_rate_1��j75�cyI       6%�	I,9���A�*;


total_loss�$�@

error_R�6F?

learning_rate_1��j71�}I       6%�	�,9���A�*;


total_lossv>�@

error_R�rM?

learning_rate_1��j7�
�I       6%�	 -9���A�*;


total_lossf,�@

error_R�!>?

learning_rate_1��j7I���I       6%�	N-9���A�*;


total_loss��A

error_R�2?

learning_rate_1��j7��I       6%�	��-9���A�*;


total_loss]�@

error_R@�Y?

learning_rate_1��j7G���I       6%�	�-9���A�*;


total_lossݶ�@

error_R\�I?

learning_rate_1��j7�>�I       6%�	Y.9���A�*;


total_loss4��@

error_R�P?

learning_rate_1��j7�	͟I       6%�	�W.9���A�*;


total_loss#�@

error_R�OR?

learning_rate_1��j7,9PvI       6%�	8�.9���A�*;


total_loss�3�@

error_R�kK?

learning_rate_1��j7K�z�I       6%�	i�.9���A�*;


total_loss��@

error_Rx~P?

learning_rate_1��j7x���I       6%�	Y+/9���A�*;


total_loss�ɕ@

error_R<�T?

learning_rate_1��j7p~�[I       6%�	�w/9���A�*;


total_loss�;�@

error_RfF?

learning_rate_1��j7�i��I       6%�	�/9���A�*;


total_loss�\�@

error_RS2C?

learning_rate_1��j7�ұ)I       6%�	�09���A�*;


total_loss� A

error_RIvN?

learning_rate_1��j7p8�AI       6%�	�M09���A�*;


total_loss4F�@

error_R�pW?

learning_rate_1��j7��iI       6%�	�09���A�*;


total_loss�j�@

error_R�S?

learning_rate_1��j7K9�SI       6%�	��09���A�*;


total_loss��@

error_R�:T?

learning_rate_1��j7�V^�I       6%�	Q19���A�*;


total_lossS��@

error_R�H?

learning_rate_1��j7�{�I       6%�	�W19���A�*;


total_loss�Ֆ@

error_R;cM?

learning_rate_1��j7�ukI       6%�	�19���A�*;


total_loss�@�@

error_R
�O?

learning_rate_1��j7�Z��I       6%�	��19���A�*;


total_loss ��@

error_R�A@?

learning_rate_1��j7�I       6%�	>29���A�*;


total_loss���@

error_RܯT?

learning_rate_1��j7�C�I       6%�	�b29���A�*;


total_loss�k�@

error_R�\?

learning_rate_1��j7.Me�I       6%�	��29���A�*;


total_loss�!�@

error_R`�C?

learning_rate_1��j7�� pI       6%�	��29���A�*;


total_loss��@

error_R�=?

learning_rate_1��j7�C�I       6%�	�*39���A�*;


total_loss�d�@

error_R4�d?

learning_rate_1��j7�+mI       6%�	So39���A�*;


total_loss�ߥ@

error_Ra!Q?

learning_rate_1��j7��I       6%�	�39���A�*;


total_loss�A

error_Rs�X?

learning_rate_1��j74�U�I       6%�	��39���A�*;


total_loss��@

error_R�W?

learning_rate_1��j7��I       6%�	N949���A�*;


total_loss<3�@

error_RJ�P?

learning_rate_1��j7��sI       6%�	_y49���A�*;


total_loss��q@

error_R��I?

learning_rate_1��j7�4�I       6%�	_�49���A�*;


total_loss��@

error_R�E?

learning_rate_1��j7�\��I       6%�	u�49���A�*;


total_lossJ�@

error_RjOQ?

learning_rate_1��j7mv��I       6%�	]F59���A�*;


total_loss��A

error_RE�a?

learning_rate_1��j7�?'I       6%�	q�59���A�*;


total_lossа�@

error_RԔ:?

learning_rate_1��j7�.�RI       6%�	&�59���A�*;


total_loss,Y�@

error_R�tJ?

learning_rate_1��j7�?,I       6%�	!(69���A�*;


total_loss�a@

error_R�U?

learning_rate_1��j7�L�TI       6%�	Ci69���A�*;


total_lossm�@

error_R��T?

learning_rate_1��j7��g�I       6%�	��69���A�*;


total_loss���@

error_Rf�>?

learning_rate_1��j7Ө�=I       6%�	��69���A�*;


total_loss�@

error_R�-Q?

learning_rate_1��j7dS�BI       6%�	y,79���A�*;


total_lossq�@

error_R�
P?

learning_rate_1��j70��I       6%�	�l79���A�*;


total_loss,��@

error_R�6K?

learning_rate_1��j7�^ I       6%�	/�79���A�*;


total_loss�J�@

error_R �L?

learning_rate_1��j7����I       6%�	��79���A�*;


total_loss[��@

error_RdwF?

learning_rate_1��j7��V*I       6%�	�>89���A�*;


total_loss��@

error_R�U?

learning_rate_1��j7��eI       6%�	ց89���A�*;


total_lossvr�@

error_RFJ?

learning_rate_1��j7rBG�I       6%�	p�89���A�*;


total_lossF��@

error_Rn�M?

learning_rate_1��j7�Ã�I       6%�	99���A�*;


total_loss���@

error_R�^?

learning_rate_1��j71\�#I       6%�	�K99���A�*;


total_loss[��@

error_RfIW?

learning_rate_1��j7n�}�I       6%�	��99���A�*;


total_lossܠ�@

error_R]S?

learning_rate_1��j7�6��I       6%�	��99���A�*;


total_loss�i�@

error_RMjN?

learning_rate_1��j7��f�I       6%�	I3:9���A�*;


total_loss��@

error_Rz�Q?

learning_rate_1��j7��͂I       6%�	�x:9���A�*;


total_loss�g@

error_R�B?

learning_rate_1��j7\��I       6%�	��:9���A�*;


total_lossJ�@

error_R��V?

learning_rate_1��j7ѐ�I       6%�	�;9���A�*;


total_loss�ٻ@

error_R�I?

learning_rate_1��j7!v�mI       6%�	fI;9���A�*;


total_lossyA

error_R�3M?

learning_rate_1��j7T���I       6%�	S�;9���A�*;


total_loss�9�@

error_RC�G?

learning_rate_1��j7
{�pI       6%�	��;9���A�*;


total_lossDA�@

error_R��Y?

learning_rate_1��j7�<nI       6%�	S<9���A�*;


total_loss��@

error_RJ�??

learning_rate_1��j7��I       6%�	�O<9���A�*;


total_lossM_A

error_R�P?

learning_rate_1��j7>���I       6%�	�<9���A�*;


total_loss�ʾ@

error_R�{Y?

learning_rate_1��j7���I       6%�	��<9���A�*;


total_loss�׵@

error_R�^T?

learning_rate_1��j7�XmQI       6%�	�=9���A�*;


total_loss�s�@

error_R!_X?

learning_rate_1��j7���lI       6%�	`=9���A�*;


total_loss��@

error_R�nQ?

learning_rate_1��j7�
�EI       6%�	��=9���A�*;


total_lossM��@

error_R��B?

learning_rate_1��j7�z�RI       6%�	k�=9���A�*;


total_loss#|A

error_R7�B?

learning_rate_1��j7�ԏI       6%�	'>9���A�*;


total_loss�;�@

error_R�<C?

learning_rate_1��j7E���I       6%�	�e>9���A�*;


total_loss�?�@

error_R�BO?

learning_rate_1��j7iQ�I       6%�	��>9���A�*;


total_loss-��@

error_RL�P?

learning_rate_1��j7�LI       6%�	��>9���A�*;


total_loss���@

error_R�V?

learning_rate_1��j7.��I       6%�	�!?9���A�*;


total_lossH�A

error_R��V?

learning_rate_1��j7��}�I       6%�	*e?9���A�*;


total_loss���@

error_R�5E?

learning_rate_1��j7����I       6%�	H�?9���A�*;


total_lossQ4A

error_R�sS?

learning_rate_1��j7C��I       6%�	��?9���A�*;


total_loss���@

error_R�.?

learning_rate_1��j7��)�I       6%�	�$@9���A�*;


total_loss���@

error_R�q=?

learning_rate_1��j7�f�I       6%�	�e@9���A�*;


total_loss얬@

error_RT�R?

learning_rate_1��j7�9a�I       6%�	�@9���A�*;


total_loss��@

error_R��G?

learning_rate_1��j7cHoI       6%�	?�@9���A�*;


total_loss��@

error_R�U?

learning_rate_1��j7�C[I       6%�	*A9���A�*;


total_lossv�@

error_R�I?

learning_rate_1��j7�Z �I       6%�	mA9���A�*;


total_loss:�@

error_R��G?

learning_rate_1��j7_�I       6%�	�A9���A�*;


total_loss�t�@

error_RJ�T?

learning_rate_1��j7NJL�I       6%�	��A9���A�*;


total_lossX)�@

error_R�X?

learning_rate_1��j7Q�O�I       6%�	K/B9���A�*;


total_loss���@

error_R�L?

learning_rate_1��j7�.^nI       6%�	�pB9���A�*;


total_lossO��@

error_R�/O?

learning_rate_1��j7&9�I       6%�	=�B9���A�*;


total_lossӵ�@

error_R�>?

learning_rate_1��j7�T�1I       6%�	&�B9���A�*;


total_loss���@

error_RR�R?

learning_rate_1��j7�I       6%�	=6C9���A�*;


total_loss�*�@

error_R{?<?

learning_rate_1��j7�S?	I       6%�	>vC9���A�*;


total_lossr�@

error_R�'E?

learning_rate_1��j7X��LI       6%�	��C9���A�*;


total_loss;�@

error_Rx�A?

learning_rate_1��j7���I       6%�	�C9���A�*;


total_loss߳�@

error_R��G?

learning_rate_1��j7��d�I       6%�	�<D9���A�*;


total_loss�;�@

error_R��B?

learning_rate_1��j7�v'�I       6%�	��D9���A�*;


total_lossÁ�@

error_Rw\V?

learning_rate_1��j7�-�I       6%�	P�D9���A�*;


total_loss&`�@

error_RECD?

learning_rate_1��j7�0��I       6%�	NE9���A�*;


total_loss�ܽ@

error_R�8R?

learning_rate_1��j7(��I       6%�	ME9���A�*;


total_lossɌ�@

error_R�<L?

learning_rate_1��j7�w�yI       6%�	��E9���A�*;


total_lossV�A

error_Rr�F?

learning_rate_1��j7F��!I       6%�	6�E9���A�*;


total_loss;4�@

error_R��U?

learning_rate_1��j7�1çI       6%�	';F9���A�*;


total_loss�>�@

error_R\6?

learning_rate_1��j7%॰I       6%�	\|F9���A�*;


total_loss*3�@

error_R[3I?

learning_rate_1��j7>�`QI       6%�	��F9���A�*;


total_loss�o@

error_RB?

learning_rate_1��j7�W�I       6%�	�G9���A�*;


total_loss�י@

error_RzPB?

learning_rate_1��j7ޙ֐I       6%�	�GG9���A�*;


total_lossfʋ@

error_R��W?

learning_rate_1��j7y�_I       6%�	��G9���A�*;


total_loss���@

error_R�NQ?

learning_rate_1��j7�e��I       6%�	��G9���A�*;


total_loss1��@

error_RMw@?

learning_rate_1��j7(��.I       6%�	�H9���A�*;


total_loss��@

error_R��B?

learning_rate_1��j7F�Q�I       6%�	�SH9���A�*;


total_loss�S�@

error_R,�R?

learning_rate_1��j7�&��I       6%�	�H9���A�*;


total_loss�P�@

error_R�D?

learning_rate_1��j7eg�I       6%�	m�H9���A�*;


total_loss���@

error_R�,5?

learning_rate_1��j7���TI       6%�	�I9���A�*;


total_loss=�@

error_RŵF?

learning_rate_1��j7a��EI       6%�	_`I9���A�*;


total_loss�ʭ@

error_RѰ=?

learning_rate_1��j7/���I       6%�	��I9���A�*;


total_loss��@

error_R�1H?

learning_rate_1��j78"��I       6%�	��I9���A�*;


total_lossv��@

error_R��Y?

learning_rate_1��j7r[I       6%�	�J9���A�*;


total_loss{�@

error_R��M?

learning_rate_1��j7<�oRI       6%�	�cJ9���A�*;


total_loss�;�@

error_R�I?

learning_rate_1��j7D�<�I       6%�	��J9���A�*;


total_loss��@

error_R�;A?

learning_rate_1��j7��>�I       6%�	[�J9���A�*;


total_loss��#A

error_R�?K?

learning_rate_1��j7����I       6%�	�$K9���A�*;


total_loss���@

error_R��L?

learning_rate_1��j7�ڢI       6%�	-dK9���A�*;


total_loss�1�@

error_R{�Z?

learning_rate_1��j7���I       6%�	5�K9���A�*;


total_loss�;�@

error_RJ�S?

learning_rate_1��j7��p�I       6%�	H�K9���A�*;


total_loss���@

error_RI?

learning_rate_1��j7�O�I       6%�	�:L9���A�*;


total_loss�T�@

error_Ri�P?

learning_rate_1��j7:�:�I       6%�	�L9���A�*;


total_lossV��@

error_R�P?

learning_rate_1��j7/��I       6%�	m�L9���A�*;


total_loss��@

error_R.Z4?

learning_rate_1��j7�`KI       6%�	pM9���A�*;


total_loss�A

error_R 6W?

learning_rate_1��j7}x�vI       6%�	KUM9���A�*;


total_loss���@

error_R�L?

learning_rate_1��j7�2gI       6%�	�M9���A�*;


total_lossŎ�@

error_R�O?

learning_rate_1��j7VI       6%�	��M9���A�*;


total_loss>�@

error_RxvT?

learning_rate_1��j7&��I       6%�	�N9���A�*;


total_lossf�@

error_R�f5?

learning_rate_1��j7j"&5I       6%�	m\N9���A�*;


total_loss�<�@

error_R2;h?

learning_rate_1��j7���I       6%�	^�N9���A�*;


total_loss�s@

error_R�F?

learning_rate_1��j7����I       6%�	��N9���A�*;


total_loss�@

error_R2%J?

learning_rate_1��j7��k\I       6%�	�*O9���A�*;


total_loss�Z�@

error_RU?

learning_rate_1��j7_aSuI       6%�	doO9���A�*;


total_loss�,�@

error_Rz�U?

learning_rate_1��j7���%I       6%�	��O9���A�*;


total_lossѯ@

error_R��N?

learning_rate_1��j7�v�I       6%�	L�O9���A�*;


total_loss���@

error_R�UA?

learning_rate_1��j7p��I       6%�	�AP9���A�*;


total_loss4��@

error_R�#J?

learning_rate_1��j7"yyI       6%�	�P9���A�*;


total_loss�=�@

error_R�<?

learning_rate_1��j7���I       6%�	��P9���A�*;


total_lossc�@

error_Rt�??

learning_rate_1��j7�P�%I       6%�	�Q9���A�*;


total_loss���@

error_R?�<?

learning_rate_1��j7*�$I       6%�	�FQ9���A�*;


total_loss���@

error_R��H?

learning_rate_1��j7t�IEI       6%�	q�Q9���A�*;


total_loss��@

error_R�PO?

learning_rate_1��j7��axI       6%�	�Q9���A�*;


total_loss�˿@

error_RF�T?

learning_rate_1��j7�%��I       6%�	�
R9���A�*;


total_loss���@

error_R��P?

learning_rate_1��j7ʴ'�I       6%�	bKR9���A�*;


total_loss*E�@

error_R��k?

learning_rate_1��j7�&�I       6%�	=�R9���A�*;


total_lossp�@

error_R��I?

learning_rate_1��j7g��I       6%�	#�R9���A�*;


total_loss�P�@

error_R��N?

learning_rate_1��j7� �I       6%�	 4S9���A�*;


total_loss��@

error_RϗI?

learning_rate_1��j7�ONI       6%�	wS9���A�*;


total_loss$�@

error_R��Y?

learning_rate_1��j7e�S?I       6%�	t�S9���A�*;


total_loss3sA

error_RJ?

learning_rate_1��j7��Y�I       6%�	x�S9���A�*;


total_loss���@

error_ReS?

learning_rate_1��j7YJ"I       6%�	.5T9���A�*;


total_loss�e�@

error_Rd�B?

learning_rate_1��j7'e��I       6%�	�vT9���A�*;


total_loss��@

error_REjT?

learning_rate_1��j7?1�I       6%�	��T9���A�*;


total_loss�@

error_RDCB?

learning_rate_1��j7sMT�I       6%�	��T9���A�*;


total_loss���@

error_R��F?

learning_rate_1��j7s�!�I       6%�	�@U9���A�*;


total_loss��@

error_R�98?

learning_rate_1��j7|z&�I       6%�	��U9���A�*;


total_loss�DA

error_RںI?

learning_rate_1��j7��=�I       6%�	��U9���A�*;


total_loss�5�@

error_R�%U?

learning_rate_1��j7%L;'I       6%�	�.V9���A�*;


total_loss��@

error_R.*T?

learning_rate_1��j7U�N~I       6%�	�rV9���A�*;


total_lossF�@

error_Ri�\?

learning_rate_1��j72���I       6%�	��V9���A�*;


total_loss�@

error_RM_?

learning_rate_1��j7_�BJI       6%�	/�V9���A�*;


total_loss��@

error_R��V?

learning_rate_1��j7��&�I       6%�	7W9���A�*;


total_lossqbx@

error_R$�V?

learning_rate_1��j7v�I       6%�	�xW9���A�*;


total_loss�G�@

error_R\
D?

learning_rate_1��j7g-I       6%�	ٿW9���A�*;


total_loss���@

error_R͓Z?

learning_rate_1��j7Y޺�I       6%�	�X9���A�*;


total_lossMA

error_RM�F?

learning_rate_1��j7��(8I       6%�	�BX9���A�*;


total_loss<��@

error_R�CV?

learning_rate_1��j7�p�I       6%�	/�X9���A�*;


total_loss\��@

error_RWI?

learning_rate_1��j7��I       6%�	A�X9���A�*;


total_loss��@

error_R4G?

learning_rate_1��j7�ܱ�I       6%�	�Y9���A�*;


total_loss��@

error_R�q;?

learning_rate_1��j7����I       6%�	%uY9���A�*;


total_loss>��@

error_RnC\?

learning_rate_1��j7/�5I       6%�	��Y9���A�*;


total_loss؊�@

error_Ra�K?

learning_rate_1��j7IÜI       6%�	�Z9���A�*;


total_loss��x@

error_R�<Q?

learning_rate_1��j7�M8�I       6%�	NFZ9���A�*;


total_loss'^@

error_R
�5?

learning_rate_1��j73G�I       6%�	�Z9���A�*;


total_loss�ê@

error_RaW?

learning_rate_1��j7(��I       6%�	��Z9���A�*;


total_loss8ޠ@

error_Rn	L?

learning_rate_1��j7%��I       6%�	�	[9���A�*;


total_lossn��@

error_R(�4?

learning_rate_1��j7�N�I       6%�	�K[9���A�*;


total_lossh��@

error_R�N?

learning_rate_1��j7�zI       6%�	2�[9���A�*;


total_loss3.
A

error_R��M?

learning_rate_1��j7��%I       6%�	!�[9���A�*;


total_loss<��@

error_R)�<?

learning_rate_1��j7OU��I       6%�	L\9���A�*;


total_loss��@

error_R��J?

learning_rate_1��j7G���I       6%�	SZ\9���A�*;


total_loss_�@

error_Rq=?

learning_rate_1��j7��O�I       6%�	�\9���A�*;


total_loss�͸@

error_RS6?

learning_rate_1��j7�u	I       6%�	��\9���A�*;


total_loss��@

error_R�Q?

learning_rate_1��j7��I       6%�	&]9���A�*;


total_loss3��@

error_R�A?

learning_rate_1��j7��3I       6%�	�f]9���A�*;


total_lossO��@

error_RÞF?

learning_rate_1��j7�oI       6%�	T�]9���A�*;


total_loss�7@

error_R��F?

learning_rate_1��j7���I       6%�	D�]9���A�*;


total_loss��@

error_R�[O?

learning_rate_1��j7���PI       6%�	�&^9���A�*;


total_loss�۫@

error_R��C?

learning_rate_1��j72��I       6%�	=h^9���A�*;


total_lossVYA

error_R2V?

learning_rate_1��j7�W��I       6%�	ߩ^9���A�*;


total_loss�|�@

error_R�dT?

learning_rate_1��j7�|tI       6%�	?�^9���A�*;


total_loss�7�@

error_R�=?

learning_rate_1��j7u/|�I       6%�	�,_9���A�*;


total_lossƦ@

error_R&BY?

learning_rate_1��j7m�+�I       6%�	Lo_9���A�*;


total_lossE��@

error_R�C?

learning_rate_1��j7��� I       6%�	9�_9���A�*;


total_loss��@

error_R8N?

learning_rate_1��j7ߋyhI       6%�	��_9���A�*;


total_loss�2�@

error_Ra�J?

learning_rate_1��j7a���I       6%�	�7`9���A�*;


total_lossf�@

error_R�;P?

learning_rate_1��j7/\�I       6%�	x`9���A�*;


total_loss���@

error_R�=?

learning_rate_1��j7U]=�I       6%�	��`9���A�*;


total_loss�J�@

error_R�FZ?

learning_rate_1��j7?��I       6%�	� a9���A�*;


total_loss[J�@

error_R�fc?

learning_rate_1��j7����I       6%�	UEa9���A�*;


total_loss��@

error_R�\?

learning_rate_1��j7m�AI       6%�	x�a9���A�*;


total_loss�ե@

error_Rsm6?

learning_rate_1��j7��ߋI       6%�	m�a9���A�*;


total_loss8z�@

error_R$�M?

learning_rate_1��j7!��I       6%�	#	b9���A�*;


total_loss%�@

error_R�RN?

learning_rate_1��j7�4�&I       6%�	�Gb9���A�*;


total_loss汭@

error_RҮQ?

learning_rate_1��j7����I       6%�	��b9���A�*;


total_lossQ��@

error_R4�V?

learning_rate_1��j7�$��I       6%�	"�b9���A�*;


total_lossN�@

error_RX�Z?

learning_rate_1��j7�9�/I       6%�	zc9���A�*;


total_loss���@

error_R�X?

learning_rate_1��j7=�0�I       6%�	OQc9���A�*;


total_loss�\�@

error_R��.?

learning_rate_1��j7��_I       6%�	'�c9���A�*;


total_loss_#�@

error_R
)E?

learning_rate_1��j7"���I       6%�	�c9���A�*;


total_loss��@

error_R��[?

learning_rate_1��j7�r�I       6%�	d9���A�*;


total_loss���@

error_R!7\?

learning_rate_1��j7����I       6%�	�[d9���A�*;


total_loss��@

error_R��c?

learning_rate_1��j7A�F�I       6%�	��d9���A�*;


total_lossX�@

error_R!&S?

learning_rate_1��j7cK�I       6%�	��d9���A�*;


total_loss͋�@

error_R.VG?

learning_rate_1��j7$i�I       6%�	- e9���A�*;


total_loss4��@

error_RFJW?

learning_rate_1��j7m��cI       6%�	�we9���A�*;


total_loss�$�@

error_RH^O?

learning_rate_1��j7`-ĦI       6%�	[�e9���A�*;


total_lossjˏ@

error_R�P?

learning_rate_1��j7ɏG,I       6%�	hf9���A�*;


total_loss٬�@

error_R|D?

learning_rate_1��j7%5�%I       6%�	Jf9���A�*;


total_loss鞾@

error_RH�I?

learning_rate_1��j7�:��I       6%�	��f9���A�*;


total_loss}��@

error_R<t<?

learning_rate_1��j73M�I       6%�	g�f9���A�*;


total_loss��@

error_R�$P?

learning_rate_1��j7,!�tI       6%�	d	g9���A�*;


total_lossO��@

error_RvdJ?

learning_rate_1��j71��I       6%�	�Kg9���A�*;


total_loss2�k@

error_R�:A?

learning_rate_1��j7�dI       6%�	y�g9���A�*;


total_loss��@

error_R�d?

learning_rate_1��j7^V� I       6%�	��g9���A�*;


total_loss�s�@

error_R�C?

learning_rate_1��j7u�&I       6%�	Lh9���A�*;


total_loss$��@

error_R��J?

learning_rate_1��j7:��	I       6%�	L\h9���A�*;


total_lossݙ�@

error_R�6?

learning_rate_1��j7F��9I       6%�	D�h9���A�*;


total_loss�D�@

error_R�_?

learning_rate_1��j7�ZfI       6%�	r�h9���A�*;


total_loss�@�@

error_R��Z?

learning_rate_1��j7��/�I       6%�	3(i9���A�*;


total_lossq�@

error_Rs�K?

learning_rate_1��j7��.�I       6%�	�ji9���A�*;


total_lossf<�@

error_RN�@?

learning_rate_1��j7PheI       6%�	:�i9���A�*;


total_loss�x�@

error_R�2U?

learning_rate_1��j7�1��I       6%�	��i9���A�*;


total_loss*�@

error_R��E?

learning_rate_1��j7?���I       6%�	3j9���A�*;


total_loss6��@

error_R��S?

learning_rate_1��j7,I       6%�	3xj9���A�*;


total_loss|y
A

error_RE�j?

learning_rate_1��j7�}�$I       6%�	��j9���A�*;


total_lossD��@

error_R2�T?

learning_rate_1��j7ҥ: I       6%�	��j9���A�*;


total_loss�z�@

error_R'Q?

learning_rate_1��j7�i��I       6%�	�Ak9���A�*;


total_loss��@

error_R�T?

learning_rate_1��j7�>I       6%�	��k9���A�*;


total_loss*�F@

error_R��@?

learning_rate_1��j7FSf�I       6%�	
�k9���A�*;


total_lossm��@

error_RÉV?

learning_rate_1��j7o8s�I       6%�	�l9���A�*;


total_loss���@

error_R��K?

learning_rate_1��j7��6vI       6%�	�Fl9���A�*;


total_loss���@

error_RU[?

learning_rate_1��j7F��I       6%�	X�l9���A�*;


total_loss�G�@

error_R$_?

learning_rate_1��j7�h2�I       6%�	��l9���A�*;


total_loss�o�@

error_R�XM?

learning_rate_1��j7!��-I       6%�	�m9���A�*;


total_loss7�@

error_R��R?

learning_rate_1��j7W�I       6%�	q]m9���A�*;


total_lossV=�@

error_R��E?

learning_rate_1��j7w<�I       6%�	��m9���A�*;


total_loss�A�@

error_Rl�R?

learning_rate_1��j7��_I       6%�	��m9���A�*;


total_loss�ԯ@

error_R�rl?

learning_rate_1��j7H��$I       6%�	�/n9���A�*;


total_loss���@

error_Rx�L?

learning_rate_1��j7;'�I       6%�	-rn9���A�*;


total_lossd�@

error_RR�5?

learning_rate_1��j7"��
I       6%�	��n9���A�*;


total_loss~�A

error_Rl�S?

learning_rate_1��j7���6I       6%�	��n9���A�*;


total_loss���@

error_Rf<C?

learning_rate_1��j75�I       6%�	�Go9���A�*;


total_loss8Ж@

error_R�$f?

learning_rate_1��j7�5{I       6%�	A�o9���A�*;


total_lossiL�@

error_R�O?

learning_rate_1��j7�I       6%�	F�o9���A�*;


total_loss��m@

error_R4L?

learning_rate_1��j7RuG#I       6%�	zp9���A�*;


total_lossN1�@

error_R$9V?

learning_rate_1��j7�p`qI       6%�	�Qp9���A�*;


total_loss�B�@

error_R�S?

learning_rate_1��j7�4!LI       6%�	��p9���A�*;


total_loss3R�@

error_R�YL?

learning_rate_1��j7ԭ�tI       6%�	$�p9���A�*;


total_loss���@

error_R(bZ?

learning_rate_1��j7���I       6%�	�q9���A�*;


total_loss�+�@

error_R��O?

learning_rate_1��j7ݏ�"I       6%�	�Wq9���A�*;


total_loss3��@

error_R�]?

learning_rate_1��j7�Ro,I       6%�	.�q9���A�*;


total_loss��A

error_R�$I?

learning_rate_1��j7�_G+I       6%�	��q9���A�*;


total_loss<��@

error_R��I?

learning_rate_1��j7��I       6%�	:r9���A�*;


total_lossD6{@

error_R AK?

learning_rate_1��j77RšI       6%�	�Yr9���A�*;


total_loss�g�@

error_R��_?

learning_rate_1��j7���I       6%�	��r9���A�*;


total_lossH�@

error_R@SH?

learning_rate_1��j7��I       6%�	��r9���A�*;


total_loss6�@

error_R�7L?

learning_rate_1��j7$�[2I       6%�	V"s9���A�*;


total_losse�@

error_R�TU?

learning_rate_1��j7��\�I       6%�	fbs9���A�*;


total_loss���@

error_R�A?

learning_rate_1��j7+eV�I       6%�	��s9���A�*;


total_lossqD�@

error_Ra�5?

learning_rate_1��j7J���I       6%�	��s9���A�*;


total_loss��@

error_R��C?

learning_rate_1��j7s0�I       6%�	�%t9���A�*;


total_lossT��@

error_RɳL?

learning_rate_1��j7/��I       6%�	v>w9���A�*;


total_loss=k�@

error_R��@?

learning_rate_1��j7pI~;I       6%�	��w9���A�*;


total_loss�~y@

error_R�bG?

learning_rate_1��j7G��8I       6%�	�w9���A�*;


total_loss�P�@

error_R-�G?

learning_rate_1��j7JC�YI       6%�	�x9���A�*;


total_lossg1�@

error_RHO?

learning_rate_1��j7C��I       6%�	Ux9���A�*;


total_lossB�@

error_R��@?

learning_rate_1��j7�ŭdI       6%�	�x9���A�*;


total_lossʦT@

error_R�	;?

learning_rate_1��j7C�I       6%�	 �x9���A�*;


total_loss��@

error_R�c?

learning_rate_1��j7���I       6%�	�$y9���A�*;


total_lossR��@

error_R,I?

learning_rate_1��j7�%�yI       6%�	�ky9���A�*;


total_loss��@

error_R�D?

learning_rate_1��j7v��NI       6%�	¬y9���A�*;


total_loss�A

error_R� G?

learning_rate_1��j7pr��I       6%�	x�y9���A�*;


total_loss-�A

error_R�	W?

learning_rate_1��j7���-I       6%�	�6z9���A�*;


total_lossƚ�@

error_R3H?

learning_rate_1��j7X��I       6%�	xz9���A�*;


total_loss�@

error_R��O?

learning_rate_1��j7�9]I       6%�	��z9���A�*;


total_loss�l�@

error_Ri[?

learning_rate_1��j7p��I       6%�	i{9���A�*;


total_loss���@

error_R�[?

learning_rate_1��j78d�;I       6%�	6H{9���A�*;


total_loss\e�@

error_RsBW?

learning_rate_1��j7¼��I       6%�	)�{9���A�*;


total_loss!ǎ@

error_R �I?

learning_rate_1��j7E�
I       6%�	��{9���A�*;


total_loss��@

error_R6ce?

learning_rate_1��j7W婊I       6%�	Q|9���A�*;


total_loss���@

error_RZD?

learning_rate_1��j7�jI       6%�	�_|9���A�*;


total_lossCj�@

error_R��J?

learning_rate_1��j7�gI       6%�	��|9���A�*;


total_lossRg�@

error_R{�U?

learning_rate_1��j7�V�I       6%�	��|9���A�*;


total_loss"�@

error_R;dX?

learning_rate_1��j7��I       6%�	� }9���A�*;


total_lossέ�@

error_RQI?

learning_rate_1��j7 ��I       6%�	�_}9���A�*;


total_lossZ5H@

error_R1Dj?

learning_rate_1��j7̃N�I       6%�	0�}9���A�*;


total_loss� �@

error_R��H?

learning_rate_1��j7�c�yI       6%�	��}9���A�*;


total_lossEh�@

error_R�PN?

learning_rate_1��j7�ש/I       6%�	h#~9���A�*;


total_loss���@

error_RW,O?

learning_rate_1��j7_Y{�I       6%�	�d~9���A�*;


total_loss1ͧ@

error_R��M?

learning_rate_1��j7��nI       6%�	B�~9���A�*;


total_loss���@

error_RҤC?

learning_rate_1��j7�9�I       6%�	-�~9���A�*;


total_loss���@

error_R�"\?

learning_rate_1��j7�Z��I       6%�	F%9���A�*;


total_loss;�@

error_R�lH?

learning_rate_1��j7"a�I       6%�	�d9���A�*;


total_lossd��@

error_R��E?

learning_rate_1��j7jR�I       6%�	��9���A�*;


total_loss��@

error_Ri`U?

learning_rate_1��j7��haI       6%�	[�9���A�*;


total_loss.��@

error_R1�:?

learning_rate_1��j7����I       6%�	�(�9���A�*;


total_lossO͔@

error_R�<Z?

learning_rate_1��j7b��I       6%�	�i�9���A�*;


total_loss���@

error_R��L?

learning_rate_1��j7zI       6%�	ڨ�9���A�*;


total_loss��@

error_R�rf?

learning_rate_1��j7'��I       6%�	5�9���A�*;


total_lossNU�@

error_RF�N?

learning_rate_1��j7#?��I       6%�	a,�9���A�*;


total_lossCЭ@

error_R ;?

learning_rate_1��j7�I       6%�	nt�9���A�*;


total_loss�'�@

error_R_H?

learning_rate_1��j7+dGSI       6%�	&��9���A�*;


total_loss8�@

error_R��X?

learning_rate_1��j7���I       6%�	���9���A�*;


total_loss�Ž@

error_R�KM?

learning_rate_1��j7`��I       6%�	�C�9���A�*;


total_lossM��@

error_R�*9?

learning_rate_1��j7�zi�I       6%�	��9���A�*;


total_losst	~@

error_R	�^?

learning_rate_1��j7�KMI       6%�	�Ђ9���A�*;


total_loss焀@

error_R�}T?

learning_rate_1��j7E��I       6%�	��9���A�*;


total_loss�Հ@

error_R�{Y?

learning_rate_1��j7 %�JI       6%�	�Z�9���A�*;


total_lossi�@

error_R�M?

learning_rate_1��j7r��I       6%�	���9���A�*;


total_loss1��@

error_R��]?

learning_rate_1��j7.,a#I       6%�	G�9���A�*;


total_loss�!�@

error_RP?

learning_rate_1��j7����I       6%�	�N�9���A�*;


total_loss�׾@

error_R�2Y?

learning_rate_1��j7��`I       6%�	���9���A�*;


total_loss� A

error_R�_B?

learning_rate_1��j7�q�I       6%�	�9���A�*;


total_loss�A

error_R�'[?

learning_rate_1��j7�[k�I       6%�	PU�9���A�*;


total_loss�Ҧ@

error_R_�d?

learning_rate_1��j7���I       6%�	2��9���A�*;


total_loss4��@

error_RlhT?

learning_rate_1��j7�*��I       6%�	Y��9���A�*;


total_loss�S�@

error_R�9N?

learning_rate_1��j7�֣�I       6%�	oA�9���A�*;


total_loss$ŵ@

error_R�eU?

learning_rate_1��j7�>�I       6%�	#��9���A�*;


total_lossFe�@

error_R=PX?

learning_rate_1��j7���QI       6%�	pņ9���A�*;


total_loss �@

error_R��G?

learning_rate_1��j7Z�9�I       6%�	�9���A�*;


total_losss�@

error_RM_?

learning_rate_1��j7���I       6%�	"a�9���A�*;


total_loss�gW@

error_RX0=?

learning_rate_1��j7K���I       6%�	9���A�*;


total_loss���@

error_Ri	F?

learning_rate_1��j7���I       6%�	��9���A�*;


total_lossĠ�@

error_R�%R?

learning_rate_1��j7{m�I       6%�	�K�9���A�*;


total_loss9�@

error_R!�P?

learning_rate_1��j7���^I       6%�	r��9���A�*;


total_loss���@

error_R�;Y?

learning_rate_1��j7k�PI       6%�	�؈9���A�*;


total_loss쉾@

error_RJ�b?

learning_rate_1��j7G�I       6%�	�A�9���A�*;


total_loss]��@

error_R��L?

learning_rate_1��j7�A��I       6%�	���9���A�*;


total_loss3��@

error_R�.?

learning_rate_1��j7���nI       6%�	rӉ9���A�*;


total_loss	:�@

error_RJ�X?

learning_rate_1��j7�$CI       6%�	; �9���A�*;


total_loss��@

error_R�"Y?

learning_rate_1��j7B=MI       6%�	Jt�9���A�*;


total_loss��@

error_R��J?

learning_rate_1��j7���cI       6%�	%��9���A�*;


total_loss�>�@

error_RS�N?

learning_rate_1��j7D�bFI       6%�	�
�9���A�*;


total_loss�7�@

error_R�MM?

learning_rate_1��j7�υ�I       6%�	�a�9���A�*;


total_loss���@

error_R�8R?

learning_rate_1��j7�&�I       6%�	���9���A�*;


total_loss��@

error_R�B?

learning_rate_1��j7����I       6%�	��9���A�*;


total_loss4�A

error_R6�M?

learning_rate_1��j7�鸸I       6%�	�`�9���A�*;


total_loss)ӣ@

error_R��P?

learning_rate_1��j7�N;I       6%�	ս�9���A�*;


total_loss�l�@

error_R�A?

learning_rate_1��j7�Ϗ�I       6%�	�9���A�*;


total_loss��@

error_RK?

learning_rate_1��j7�)��I       6%�	oD�9���A�*;


total_loss�0 A

error_R.#P?

learning_rate_1��j7Q3�|I       6%�	��9���A�*;


total_loss`��@

error_R��K?

learning_rate_1��j7�sK�I       6%�	�̍9���A�*;


total_lossղ@

error_R�T?

learning_rate_1��j7��6�I       6%�	`�9���A�*;


total_lossbQ
A

error_R�AQ?

learning_rate_1��j7� �I       6%�	.P�9���A�*;


total_loss5��@

error_R�S?

learning_rate_1��j7u�zI       6%�	擎9���A�*;


total_loss���@

error_RA�K?

learning_rate_1��j7]�I       6%�	M֎9���A�*;


total_loss3��@

error_R�%>?

learning_rate_1��j7ЛI       6%�	5�9���A�*;


total_lossU��@

error_R��S?

learning_rate_1��j7�ϐI       6%�	�`�9���A�*;


total_loss�6�@

error_R�M?

learning_rate_1��j7�2>>I       6%�	D��9���A�*;


total_lossHƮ@

error_R@1V?

learning_rate_1��j7~R�I       6%�	��9���A�*;


total_loss�@

error_R�??

learning_rate_1��j7�+�I       6%�	�d�9���A�*;


total_losssù@

error_R��j?

learning_rate_1��j7��KI       6%�	e��9���A�*;


total_loss�w�@

error_R,6?

learning_rate_1��j7p AAI       6%�	��9���A�*;


total_loss~�@

error_RτE?

learning_rate_1��j7aI       6%�	]E�9���A�*;


total_loss��A

error_R��L?

learning_rate_1��j7p�zI       6%�	���9���A�*;


total_losse�@

error_R�M?

learning_rate_1��j7�3��I       6%�	�̑9���A�*;


total_loss�=�@

error_R��f?

learning_rate_1��j7��h;I       6%�	��9���A�*;


total_loss��;A

error_R�#G?

learning_rate_1��j7��I       6%�	tS�9���A�*;


total_loss��@

error_R� C?

learning_rate_1��j7kBF(I       6%�	0��9���A�*;


total_loss2��@

error_Rt�P?

learning_rate_1��j75�d5I       6%�	�Ւ9���A�*;


total_loss*�@

error_R#�J?

learning_rate_1��j7�&�NI       6%�	F�9���A�*;


total_lossO$�@

error_R�9L?

learning_rate_1��j7��b�I       6%�	�^�9���A�*;


total_lossś�@

error_R�LJ?

learning_rate_1��j7�r�RI       6%�	���9���A�*;


total_loss�8�@

error_ROgW?

learning_rate_1��j7���cI       6%�	��9���A�*;


total_lossL�A

error_R؇\?

learning_rate_1��j7�T�I       6%�	�-�9���A�*;


total_loss|'�@

error_R3�Z?

learning_rate_1��j7��I       6%�	p�9���A�*;


total_loss���@

error_RE�;?

learning_rate_1��j7s	WmI       6%�	���9���A�*;


total_loss���@

error_R�1P?

learning_rate_1��j7�H�I       6%�	
��9���A�*;


total_loss�@

error_Rn;?

learning_rate_1��j7�~ �I       6%�	d7�9���A�*;


total_loss`	�@

error_R�_D?

learning_rate_1��j7�5I       6%�	l��9���A�*;


total_loss���@

error_RM�Z?

learning_rate_1��j7*A�I       6%�	��9���A�*;


total_loss���@

error_R�xG?

learning_rate_1��j78�iHI       6%�	>+�9���A�*;


total_loss�wA

error_R�L?

learning_rate_1��j7��WI       6%�	�r�9���A�*;


total_loss�ڈ@

error_R�tT?

learning_rate_1��j7ДW�I       6%�	���9���A�*;


total_lossL|�@

error_R
)e?

learning_rate_1��j78q?QI       6%�	+��9���A�*;


total_lossM&�@

error_R,�N?

learning_rate_1��j7+�!I       6%�	�9�9���A�*;


total_loss6��@

error_R��L?

learning_rate_1��j7��I       6%�	G{�9���A�*;


total_loss�۬@

error_R�Z?

learning_rate_1��j7NY5�I       6%�	^��9���A�*;


total_loss�L�@

error_R�>_?

learning_rate_1��j7@j5I       6%�	� �9���A�*;


total_loss��@

error_RV?

learning_rate_1��j7c�ƈI       6%�	�@�9���A�*;


total_lossop�@

error_R�}K?

learning_rate_1��j7B��lI       6%�	���9���A�*;


total_loss�<�@

error_R#J?

learning_rate_1��j7�Z&I       6%�	Z9���A�*;


total_loss��A

error_Rw�t?

learning_rate_1��j7�PI       6%�	3�9���A�*;


total_losse l@

error_R��V?

learning_rate_1��j7C� �I       6%�	KG�9���A�*;


total_lossF<�@

error_R�%D?

learning_rate_1��j7�G�dI       6%�	`��9���A�*;


total_loss���@

error_R<?

learning_rate_1��j75{��I       6%�	�љ9���A�*;


total_loss*�@

error_Rn�L?

learning_rate_1��j7�~�I       6%�	��9���A�*;


total_loss}.�@

error_RcC?

learning_rate_1��j7z��I       6%�	�T�9���A�*;


total_lossrԮ@

error_RcFZ?

learning_rate_1��j7�5C�I       6%�	��9���A�*;


total_loss=�@

error_Ra�j?

learning_rate_1��j7�v�I       6%�	�֚9���A�*;


total_lossSg�@

error_R;o=?

learning_rate_1��j7��\I       6%�	��9���A�*;


total_loss}}�@

error_R��V?

learning_rate_1��j7E@T�I       6%�	�\�9���A�*;


total_loss�[�@

error_Rt�\?

learning_rate_1��j74�@I       6%�	)��9���A�*;


total_loss�A

error_R�EP?

learning_rate_1��j7�G�eI       6%�	B�9���A�*;


total_lossT}�@

error_RאX?

learning_rate_1��j7�N%dI       6%�	%"�9���A�*;


total_loss��A

error_R4~H?

learning_rate_1��j7q��I       6%�	f�9���A�*;


total_loss�s@

error_R<R?

learning_rate_1��j7��wI       6%�	��9���A�*;


total_loss�B�@

error_R�C?

learning_rate_1��j7/'Z�I       6%�	B�9���A�*;


total_loss���@

error_R�oP?

learning_rate_1��j7#�I       6%�	N�9���A�*;


total_loss�O@

error_R6d;?

learning_rate_1��j7"@$�I       6%�	S��9���A�*;


total_loss�N�@

error_R�D?

learning_rate_1��j7�6�HI       6%�	6֝9���A�*;


total_loss!T�@

error_RS\?

learning_rate_1��j7�k�I       6%�	s�9���A�*;


total_loss�+�@

error_R�{F?

learning_rate_1��j7]�k�I       6%�	�d�9���A�*;


total_loss�c�@

error_R�N?

learning_rate_1��j7����I       6%�	���9���A�*;


total_lossd��@

error_R��N?

learning_rate_1��j7�t��I       6%�	�9���A�*;


total_loss���@

error_RJ1U?

learning_rate_1��j7u��ZI       6%�	�-�9���A�*;


total_loss}Y�@

error_R�S?

learning_rate_1��j7�ʟ�I       6%�	au�9���A�*;


total_loss%��@

error_R_�P?

learning_rate_1��j7f�YI       6%�	ṟ9���A�*;


total_lossC��@

error_R�E?

learning_rate_1��j7:* �I       6%�	���9���A�*;


total_loss��@

error_RaMP?

learning_rate_1��j7}R�ZI       6%�	<�9���A�*;


total_loss;��@

error_RڤM?

learning_rate_1��j7F9[)I       6%�	�~�9���A�*;


total_loss�A

error_R�C?

learning_rate_1��j7�RH�I       6%�	Ġ9���A�*;


total_loss}6�@

error_R��O?

learning_rate_1��j7��T�I       6%�	��9���A�*;


total_lossa�|@

error_Rc@B?

learning_rate_1��j7�@I       6%�	E�9���A�*;


total_loss���@

error_R__D?

learning_rate_1��j7��HI       6%�	փ�9���A�*;


total_lossһ@

error_R�GW?

learning_rate_1��j7���I       6%�	�ġ9���A�*;


total_loss��@

error_R�M?

learning_rate_1��j7�Л�I       6%�	��9���A�*;


total_loss���@

error_RQ�`?

learning_rate_1��j7��e�I       6%�	DV�9���A�*;


total_losso��@

error_R�2P?

learning_rate_1��j7(�֋I       6%�	���9���A�*;


total_loss�x�@

error_RŔW?

learning_rate_1��j7�"��I       6%�	�ڢ9���A�*;


total_loss�U�@

error_R_?

learning_rate_1��j7B��9I       6%�	��9���A�*;


total_loss*��@

error_R&�J?

learning_rate_1��j7�G��I       6%�	]�9���A�*;


total_loss�O�@

error_RϻQ?

learning_rate_1��j7�II       6%�	5��9���A�*;


total_loss?�@

error_R6�@?

learning_rate_1��j7�7II       6%�	��9���A�*;


total_loss��@

error_Rf�[?

learning_rate_1��j7�e��I       6%�	4%�9���A�*;


total_loss�M�@

error_R�OJ?

learning_rate_1��j7j��MI       6%�	f�9���A�*;


total_loss4R�@

error_R�[?

learning_rate_1��j7ZmI       6%�	x��9���A�*;


total_loss ��@

error_R�MH?

learning_rate_1��j7�6m�I       6%�	��9���A�*;


total_loss��@

error_RmE?

learning_rate_1��j7���9I       6%�	5o�9���A�*;


total_loss7��@

error_R��a?

learning_rate_1��j7B�5�I       6%�	`ɥ9���A�*;


total_loss��@

error_R��D?

learning_rate_1��j7`8�I       6%�	
�9���A�*;


total_lossڜ�@

error_R#�J?

learning_rate_1��j7jʟ�I       6%�	�Q�9���A�*;


total_lossۙ�@

error_R�M?

learning_rate_1��j7��.�I       6%�	i��9���A�*;


total_loss�ɻ@

error_R��O?

learning_rate_1��j7�r%VI       6%�	�ۦ9���A�*;


total_loss�3�@

error_R��H?

learning_rate_1��j7���I       6%�	�!�9���A�*;


total_loss���@

error_R��7?

learning_rate_1��j7vM<I       6%�	�b�9���A�*;


total_loss	Ü@

error_R�iV?

learning_rate_1��j7�|+gI       6%�	���9���A�*;


total_losst�w@

error_R�Q?

learning_rate_1��j7`���I       6%�	:�9���A�*;


total_loss{3�@

error_R��R?

learning_rate_1��j7�/UI       6%�	�5�9���A�*;


total_losslh�@

error_R�_>?

learning_rate_1��j7u.�I       6%�	�{�9���A�*;


total_loss��@

error_RO>R?

learning_rate_1��j7MZ7I       6%�	T�9���A�*;


total_losscX�@

error_R�jQ?

learning_rate_1��j7�h��I       6%�	�>�9���A�*;


total_lossV�@

error_R�C<?

learning_rate_1��j7���mI       6%�	x��9���A�*;


total_loss�d�@

error_R#T?

learning_rate_1��j7\���I       6%�	�Ʃ9���A�*;


total_loss�m�@

error_R:uU?

learning_rate_1��j7'}�I       6%�	}
�9���A�*;


total_loss���@

error_R�A?

learning_rate_1��j7�&�I       6%�	�Q�9���A�*;


total_loss"J�@

error_R�[?

learning_rate_1��j7(޻ I       6%�	:��9���A�*;


total_loss!y�@

error_R��S?

learning_rate_1��j7�<LI       6%�	��9���A�*;


total_loss.-�@

error_R��O?

learning_rate_1��j7�4I       6%�	�*�9���A�*;


total_loss���@

error_R��S?

learning_rate_1��j7`�I       6%�	sm�9���A�*;


total_loss���@

error_R�|C?

learning_rate_1��j7fØI       6%�	ӱ�9���A�*;


total_loss�%�@

error_R��b?

learning_rate_1��j7��n�I       6%�	���9���A�*;


total_loss��@

error_R�N?

learning_rate_1��j7��I       6%�	><�9���A�*;


total_loss.	�@

error_R��B?

learning_rate_1��j7��a�I       6%�	�9���A�*;


total_loss�e�@

error_R��P?

learning_rate_1��j78���I       6%�	�¬9���A�*;


total_loss{��@

error_R]F?

learning_rate_1��j7l���I       6%�	M�9���A�*;


total_loss��@

error_R�O?

learning_rate_1��j7�SG4I       6%�	fH�9���A�*;


total_loss��@

error_RlT\?

learning_rate_1��j7�1,�I       6%�	���9���A�*;


total_lossϟ�@

error_Re�B?

learning_rate_1��j7�3�YI       6%�	Y˭9���A�*;


total_loss���@

error_RO�\?

learning_rate_1��j7i�I       6%�	0�9���A�*;


total_loss���@

error_RRk\?

learning_rate_1��j7?��5I       6%�	�R�9���A�*;


total_loss���@

error_R��S?

learning_rate_1��j7Э'4I       6%�	!��9���A�*;


total_loss�E�@

error_RC�M?

learning_rate_1��j7;�I       6%�	�ۮ9���A�*;


total_loss掜@

error_R%[?

learning_rate_1��j7D� UI       6%�	�!�9���A�*;


total_loss<ά@

error_R�P?

learning_rate_1��j7a_�I       6%�	�d�9���A�*;


total_loss��@

error_Rx�J?

learning_rate_1��j7�L7�I       6%�	u��9���A�*;


total_lossR��@

error_R,HK?

learning_rate_1��j7�j29I       6%�	�9���A�*;


total_loss�:�@

error_R�@a?

learning_rate_1��j7yn�I       6%�	�,�9���A�*;


total_loss��@

error_R
�Y?

learning_rate_1��j7�\I       6%�	zp�9���A�*;


total_loss�*�@

error_R�l?

learning_rate_1��j7V|�I       6%�	ϰ�9���A�*;


total_loss��@

error_R!p?

learning_rate_1��j7�*�%I       6%�	U��9���A�*;


total_lossd��@

error_Rn�[?

learning_rate_1��j7���8I       6%�	�8�9���A�*;


total_lossd��@

error_REmC?

learning_rate_1��j7�k�I       6%�	j�9���A�*;


total_loss`|�@

error_R�W?

learning_rate_1��j7o�3�I       6%�	�ñ9���A�*;


total_loss&��@

error_R{n[?

learning_rate_1��j78C?MI       6%�	R�9���A�*;


total_loss$��@

error_R8:?

learning_rate_1��j7�k�I       6%�	�K�9���A�*;


total_lossG�@

error_R��T?

learning_rate_1��j7
�I       6%�	`��9���A�*;


total_loss̿@

error_R��A?

learning_rate_1��j7���I       6%�	@ڲ9���A�*;


total_loss�0�@

error_R�B?

learning_rate_1��j7��I       6%�	��9���A�*;


total_loss�Ǽ@

error_R1;T?

learning_rate_1��j7���I       6%�	j�9���A�*;


total_loss@

error_R�R?

learning_rate_1��j7�s��I       6%�	��9���A�*;


total_loss�N�@

error_R�PN?

learning_rate_1��j7�HZI       6%�	��9���A�*;


total_loss��A

error_R�ls?

learning_rate_1��j7���<I       6%�	y8�9���A�*;


total_lossӵ@

error_R!�Q?

learning_rate_1��j7��6I       6%�	���9���A�*;


total_loss;'�@

error_Rs�Z?

learning_rate_1��j7��D<I       6%�	�̴9���A�*;


total_loss��@

error_R�f?

learning_rate_1��j7f�}LI       6%�	��9���A�*;


total_loss)�@

error_RJEX?

learning_rate_1��j7˲h�I       6%�	�i�9���A�*;


total_lossҘ�@

error_R�K?

learning_rate_1��j7�r��I       6%�	G��9���A�*;


total_loss��@

error_R�@E?

learning_rate_1��j7�B	�I       6%�	1�9���A�*;


total_loss�5�@

error_R�,E?

learning_rate_1��j7 �I       6%�	YM�9���A�*;


total_loss(%�@

error_RO�;?

learning_rate_1��j7{�5�I       6%�	/��9���A�*;


total_lossDl�@

error_R�ZS?

learning_rate_1��j7(��I       6%�	0Զ9���A�*;


total_losst�@

error_R��@?

learning_rate_1��j7&qpI       6%�	��9���A�*;


total_lossקy@

error_R�J?

learning_rate_1��j7�iI       6%�	�_�9���A�*;


total_loss
��@

error_R�G?

learning_rate_1��j7�g�I       6%�	���9���A�*;


total_loss.|�@

error_RH�I?

learning_rate_1��j7���(I       6%�	���9���A�*;


total_loss7��@

error_Rxd2?

learning_rate_1��j7E�ևI       6%�	�2�9���A�*;


total_lossi{�@

error_R��R?

learning_rate_1��j7
���I       6%�	�y�9���A�*;


total_loss���@

error_R�K?

learning_rate_1��j7�LI       6%�	Ľ�9���A�*;


total_loss���@

error_RfO;?

learning_rate_1��j7��I       6%�	���9���A�*;


total_loss�T�@

error_R�E?

learning_rate_1��j7o��I       6%�	uD�9���A�*;


total_loss�7�@

error_R&I?

learning_rate_1��j7KTI       6%�	툹9���A�*;


total_loss��@

error_R�	H?

learning_rate_1��j7�� I       6%�	�ʹ9���A�*;


total_loss���@

error_R��J?

learning_rate_1��j7� 8I       6%�	��9���A�*;


total_loss�Х@

error_RRE5?

learning_rate_1��j7Ф�I       6%�	�Q�9���A�*;


total_loss�"�@

error_Ra�V?

learning_rate_1��j7�W]|I       6%�	f��9���A�*;


total_lossַ�@

error_R�=D?

learning_rate_1��j7N�I       6%�	�ܺ9���A�*;


total_loss.!�@

error_R��B?

learning_rate_1��j7��H?I       6%�	��9���A�*;


total_loss���@

error_R.MJ?

learning_rate_1��j7���9I       6%�	\_�9���A�*;


total_loss�!�@

error_R�sT?

learning_rate_1��j7����I       6%�	���9���A�*;


total_loss�M�@

error_R�pN?

learning_rate_1��j7KJ��I       6%�	|�9���A�*;


total_lossL(�@

error_R�1V?

learning_rate_1��j7x��I       6%�	�"�9���A�*;


total_lossA�l@

error_R��F?

learning_rate_1��j7�&�I       6%�	Ig�9���A�*;


total_loss�á@

error_Ra'G?

learning_rate_1��j7{�#�I       6%�	���9���A�*;


total_lossS��@

error_RHvQ?

learning_rate_1��j7M�_/I       6%�	�9���A�*;


total_loss���@

error_R��S?

learning_rate_1��j7`�z�I       6%�	�0�9���A�*;


total_loss�J�@

error_Rc)D?

learning_rate_1��j7D=�I       6%�	Es�9���A�*;


total_lossX�@

error_R�.E?

learning_rate_1��j7�ځxI       6%�	���9���A�*;


total_loss�E�@

error_R��V?

learning_rate_1��j70��I       6%�	���9���A�*;


total_loss/b�@

error_RͦN?

learning_rate_1��j7�K�I       6%�	�A�9���A�*;


total_lossz A

error_R�E?

learning_rate_1��j7��tI       6%�	d��9���A�*;


total_loss���@

error_R�O?

learning_rate_1��j7�ogI       6%�	Ծ9���A�*;


total_loss�@

error_R�_<?

learning_rate_1��j7�j��I       6%�	%�9���A�*;


total_loss�j�@

error_R�UW?

learning_rate_1��j7�ރ�I       6%�	�c�9���A�*;


total_lossXLA

error_R�j\?

learning_rate_1��j7�'~I       6%�	 ��9���A�*;


total_loss���@

error_R�<U?

learning_rate_1��j7)+�}I       6%�	j�9���A�*;


total_loss�,�@

error_R�,A?

learning_rate_1��j7�!�/I       6%�	�1�9���A�*;


total_loss�
�@

error_R&4H?

learning_rate_1��j7)Q�NI       6%�	t�9���A�*;


total_lossv��@

error_R�\Q?

learning_rate_1��j7���I       6%�	���9���A�*;


total_loss�"�@

error_R��B?

learning_rate_1��j7��u�I       6%�	���9���A�*;


total_loss�K�@

error_R�G8?

learning_rate_1��j7���I       6%�	�A�9���A�*;


total_losse=�@

error_R_D?

learning_rate_1��j7�QI       6%�	·�9���A�*;


total_lossNC�@

error_R�B?

learning_rate_1��j7�>��I       6%�	z��9���A�*;


total_loss�x�@

error_R�VN?

learning_rate_1��j7�k�RI       6%�	9�9���A�*;


total_loss���@

error_R��E?

learning_rate_1��j7�k#�I       6%�	�Y�9���A�*;


total_loss'�@

error_R�^X?

learning_rate_1��j7���	I       6%�	Μ�9���A�*;


total_loss���@

error_R�O?

learning_rate_1��j7vI�I       6%�	���9���A�*;


total_loss�ܺ@

error_RZ�;?

learning_rate_1��j7����I       6%�	�$�9���A�*;


total_loss(z@

error_R�(U?

learning_rate_1��j7*�g�I       6%�	ng�9���A�*;


total_lossF7�@

error_R2�U?

learning_rate_1��j7�NI       6%�	{��9���A�*;


total_loss���@

error_Rl??

learning_rate_1��j7� �jI       6%�	���9���A�*;


total_loss��@

error_R1U?

learning_rate_1��j7��1I       6%�	l8�9���A�*;


total_loss���@

error_RjDN?

learning_rate_1��j7IYI       6%�	҆�9���A�*;


total_loss�Q�@

error_R��N?

learning_rate_1��j7/��.I       6%�	}��9���A�*;


total_loss�q�@

error_R8�1?

learning_rate_1��j7ȓ7�I       6%�	��9���A�*;


total_lossZ:�@

error_R��J?

learning_rate_1��j7i��I       6%�	9S�9���A�*;


total_loss1��@

error_R�^?

learning_rate_1��j7�"	I       6%�	]��9���A�*;


total_loss�m�@

error_R�[K?

learning_rate_1��j7��m(I       6%�	f��9���A�*;


total_loss濨@

error_RrV?

learning_rate_1��j7V���I       6%�	9�9���A�*;


total_loss�̴@

error_R1;?

learning_rate_1��j7�0��I       6%�	��9���A�*;


total_loss�iw@

error_R&�S?

learning_rate_1��j7\O\I       6%�	���9���A�*;


total_loss2B�@

error_R*�R?

learning_rate_1��j7W��I       6%�	��9���A�*;


total_loss��@

error_R�"H?

learning_rate_1��j7q[2xI       6%�	5M�9���A�*;


total_loss���@

error_R��e?

learning_rate_1��j7�ߕI       6%�	���9���A�*;


total_lossA�!A

error_R��S?

learning_rate_1��j7����I       6%�	���9���A�*;


total_loss#��@

error_R��W?

learning_rate_1��j7j\�+I       6%�	��9���A�*;


total_loss闄@

error_R@?R?

learning_rate_1��j7s{	�I       6%�	�`�9���A�*;


total_loss�Λ@

error_R|�H?

learning_rate_1��j7���I       6%�	���9���A�*;


total_loss�y�@

error_R!qB?

learning_rate_1��j7ٸ�kI       6%�	b��9���A�*;


total_lossZ�?A

error_RZIG?

learning_rate_1��j7��	�I       6%�	_0�9���A�*;


total_loss�)�@

error_Rf�T?

learning_rate_1��j7�XI       6%�	�w�9���A�*;


total_lossxP�@

error_R.�K?

learning_rate_1��j7�J3I       6%�	Z��9���A�*;


total_lossR�@

error_R��H?

learning_rate_1��j7X>٭I       6%�	��9���A�*;


total_loss1�@

error_R��Q?

learning_rate_1��j7��I       6%�	�@�9���A�*;


total_loss�&�@

error_R�G?

learning_rate_1��j7��I       6%�	i��9���A�*;


total_loss���@

error_RSZ]?

learning_rate_1��j7Z�;KI       6%�	���9���A�*;


total_loss��G@

error_R��O?

learning_rate_1��j7��0�I       6%�	��9���A�*;


total_lossH��@

error_R�pL?

learning_rate_1��j76G4I       6%�	�M�9���A�*;


total_loss���@

error_R<\?

learning_rate_1��j7+s_I       6%�	`��9���A�*;


total_lossh��@

error_Rx�Y?

learning_rate_1��j7<���I       6%�	G��9���A�*;


total_loss�8�@

error_R�A?

learning_rate_1��j7���QI       6%�	W�9���A�*;


total_lossԟ�@

error_R]�P?

learning_rate_1��j7�v I       6%�	W�9���A�*;


total_lossoG�@

error_R&�H?

learning_rate_1��j7���#I       6%�	7��9���A�*;


total_loss�[�@

error_R%�O?

learning_rate_1��j7B�Z�I       6%�	|��9���A�*;


total_loss�z�@

error_RH:?

learning_rate_1��j7+�NI       6%�	u�9���A�*;


total_loss.h�@

error_R�X?

learning_rate_1��j7����I       6%�	]�9���A�*;


total_lossA_�@

error_Rҿ/?

learning_rate_1��j7L��'I       6%�	���9���A�*;


total_loss�CQ@

error_R�;?

learning_rate_1��j7�T��I       6%�	���9���A�*;


total_loss�,�@

error_R!AA?

learning_rate_1��j7d0�I       6%�	I&�9���A�*;


total_lossj$�@

error_Rm�R?

learning_rate_1��j7�0UrI       6%�	�j�9���A�*;


total_loss#��@

error_R*[F?

learning_rate_1��j7#�rPI       6%�	6��9���A�*;


total_loss�_�@

error_RlW?

learning_rate_1��j7c/��I       6%�	x��9���A�*;


total_loss�)�@

error_R�<J?

learning_rate_1��j7(���I       6%�	�6�9���A�*;


total_loss���@

error_RjB?

learning_rate_1��j7�I~
I       6%�	�{�9���A�*;


total_loss:4�@

error_R=G\?

learning_rate_1��j7�)��I       6%�	��9���A�*;


total_loss�Ɉ@

error_R} J?

learning_rate_1��j7�T�I       6%�	��9���A�*;


total_loss���@

error_R�@?

learning_rate_1��j7	5�I       6%�	�O�9���A�*;


total_loss;��@

error_R[�J?

learning_rate_1��j7 ,�I       6%�	���9���A�*;


total_loss�N A

error_R,C?

learning_rate_1��j7�6�YI       6%�	%��9���A�*;


total_loss#��@

error_Re�W?

learning_rate_1��j7�IU3I       6%�	��9���A�*;


total_lossQ�@

error_R��Z?

learning_rate_1��j7:�ӟI       6%�	�U�9���A�*;


total_loss#�@

error_REzU?

learning_rate_1��j7�Ј�I       6%�	��9���A�*;


total_lossfI�@

error_R�JP?

learning_rate_1��j7˃�I       6%�	���9���A�*;


total_loss���@

error_R�B?

learning_rate_1��j7^JC�I       6%�		�9���A�*;


total_loss���@

error_R��W?

learning_rate_1��j7!3�	I       6%�	a_�9���A�*;


total_lossLھ@

error_RKA?

learning_rate_1��j7#���I       6%�	��9���A�*;


total_lossd$A

error_R�c?

learning_rate_1��j7�3��I       6%�	���9���A�*;


total_lossR��@

error_R�}R?

learning_rate_1��j7��I       6%�	h(�9���A�*;


total_loss���@

error_R!D?

learning_rate_1��j7���I       6%�	�h�9���A�*;


total_lossJ��@

error_R6�O?

learning_rate_1��j7%j�kI       6%�	e��9���A�*;


total_loss�Қ@

error_R�\H?

learning_rate_1��j73��I       6%�	{��9���A�*;


total_loss2��@

error_R@'O?

learning_rate_1��j7���mI       6%�	�,�9���A�*;


total_loss��@

error_R� V?

learning_rate_1��j7���JI       6%�	-q�9���A�*;


total_losss�@

error_R܁J?

learning_rate_1��j7,I       6%�	A��9���A�*;


total_loss�A

error_R6D?

learning_rate_1��j7\�+�I       6%�	-��9���A�*;


total_lossC�@

error_R��K?

learning_rate_1��j7_S3I       6%�	3�9���A�*;


total_loss%ű@

error_R6\?

learning_rate_1��j7��UI       6%�	f��9���A�*;


total_loss��@

error_R�TV?

learning_rate_1��j7��[\I       6%�	���9���A�*;


total_loss��@

error_R2aT?

learning_rate_1��j7�>dI       6%�	�#�9���A�*;


total_losst��@

error_R-�B?

learning_rate_1��j7��KI       6%�	�l�9���A�*;


total_lossɎ�@

error_RE�P?

learning_rate_1��j7g]�I       6%�	���9���A�*;


total_loss\��@

error_R�3U?

learning_rate_1��j7 #�lI       6%�	q��9���A�*;


total_loss���@

error_R��8?

learning_rate_1��j7�Q2eI       6%�	G5�9���A�*;


total_lossR��@

error_RvVD?

learning_rate_1��j7���I       6%�	_x�9���A�*;


total_loss7k�@

error_R�LN?

learning_rate_1��j7�8�VI       6%�	���9���A�*;


total_loss@��@

error_R#�E?

learning_rate_1��j7�B�%I       6%�	Y �9���A�*;


total_lossF.�@

error_R�xL?

learning_rate_1��j7���I       6%�	�F�9���A�*;


total_loss��@

error_R��9?

learning_rate_1��j7Q��I       6%�	J��9���A�*;


total_loss�M�@

error_RéA?

learning_rate_1��j7�$$�I       6%�	���9���A�*;


total_lossL%�@

error_RF)M?

learning_rate_1��j7�\^I       6%�	�9���A�*;


total_lossA�@

error_R�|Q?

learning_rate_1��j7]�ՕI       6%�	FX�9���A�*;


total_losszo�@

error_Rş??

learning_rate_1��j7�s�>I       6%�	���9���A�*;


total_lossv�@

error_R��M?

learning_rate_1��j7t��6I       6%�	g��9���A�*;


total_loss��@

error_REDH?

learning_rate_1��j7݄F�I       6%�	�%�9���A�*;


total_lossa�@

error_R�?X?

learning_rate_1��j7$���I       6%�	nh�9���A�*;


total_loss�K�@

error_Rr4O?

learning_rate_1��j7ƤQI       6%�	D��9���A�*;


total_loss���@

error_R�T?

learning_rate_1��j7��c�I       6%�		��9���A�*;


total_loss���@

error_R�z\?

learning_rate_1��j7qQ�I       6%�	�1�9���A�*;


total_loss[�y@

error_R�I?

learning_rate_1��j7�Z��I       6%�	o|�9���A�*;


total_loss��@

error_R��N?

learning_rate_1��j7�^ՀI       6%�	k��9���A�*;


total_loss;%�@

error_R_�N?

learning_rate_1��j7��=�I       6%�	��9���A�*;


total_lossac�@

error_R
<Y?

learning_rate_1��j7��I       6%�	�I�9���A�*;


total_loss�@

error_R�K?

learning_rate_1��j7�g�I       6%�	3��9���A�*;


total_loss�;�@

error_R��J?

learning_rate_1��j7�%o�I       6%�	f��9���A�*;


total_loss��l@

error_R��>?

learning_rate_1��j7��I       6%�	��9���A�*;


total_lossM�@

error_R�dA?

learning_rate_1��j7����I       6%�	�O�9���A�*;


total_loss�Е@

error_R�cS?

learning_rate_1��j7���I       6%�	U��9���A�*;


total_loss`n�@

error_R/_B?

learning_rate_1��j7�dҩI       6%�	o��9���A�*;


total_loss]��@

error_R͊O?

learning_rate_1��j7��n�I       6%�	��9���A�*;


total_loss�g�@

error_R`CW?

learning_rate_1��j7F���I       6%�	gU�9���A�*;


total_loss��@

error_R��9?

learning_rate_1��j7e*ySI       6%�	��9���A�*;


total_loss�+�@

error_Ra;M?

learning_rate_1��j7O�*I       6%�	m��9���A�*;


total_loss<܊@

error_R
"F?

learning_rate_1��j7Q�I       6%�	��9���A�*;


total_loss"-�@

error_R} T?

learning_rate_1��j7���I       6%�	�]�9���A�*;


total_loss���@

error_R�sE?

learning_rate_1��j7e5�I       6%�	^��9���A�*;


total_lossܭ�@

error_R,�`?

learning_rate_1��j7�wŹI       6%�	l��9���A�*;


total_loss���@

error_R1GM?

learning_rate_1��j7AIJI       6%�	[0�9���A�*;


total_loss���@

error_R�nZ?

learning_rate_1��j7}?7I       6%�	�v�9���A�*;


total_lossf�@

error_R�eI?

learning_rate_1��j7ڨ�.I       6%�	��9���A�*;


total_lossX��@

error_Rs[?

learning_rate_1��j7H��I       6%�	��9���A�*;


total_loss���@

error_R�,S?

learning_rate_1��j7S��!I       6%�	�G�9���A�*;


total_loss���@

error_R��M?

learning_rate_1��j7'�I       6%�	��9���A�*;


total_loss��@

error_RʜD?

learning_rate_1��j7ꄲI       6%�	+��9���A�*;


total_loss]j�@

error_R�T?

learning_rate_1��j7�%�I       6%�	X�9���A�*;


total_lossN�q@

error_R%GG?

learning_rate_1��j7�VvI       6%�	?S�9���A�*;


total_loss<M�@

error_R��]?

learning_rate_1��j73�&I       6%�	���9���A�*;


total_lossv�@

error_R��T?

learning_rate_1��j7A�d�I       6%�	3��9���A�*;


total_loss�)x@

error_R,�B?

learning_rate_1��j79>��I       6%�	�&�9���A�*;


total_lossh��@

error_R�@G?

learning_rate_1��j7�οI       6%�	#m�9���A�*;


total_loss\��@

error_R8Q?

learning_rate_1��j7g.V-I       6%�	��9���A�*;


total_loss�,�@

error_R��U?

learning_rate_1��j7<W�MI       6%�	\��9���A�*;


total_loss3o�@

error_R�g?

learning_rate_1��j7ױ|�I       6%�	F�9���A�*;


total_loss�+�@

error_R�<A?

learning_rate_1��j7���'I       6%�	0��9���A�*;


total_loss�@

error_R?R?

learning_rate_1��j7�.�I       6%�	f��9���A�*;


total_lossS��@

error_R	�^?

learning_rate_1��j7�B�CI       6%�	��9���A�*;


total_lossi��@

error_R��B?

learning_rate_1��j71���I       6%�	�^�9���A�*;


total_loss���@

error_RC�D?

learning_rate_1��j7K�CMI       6%�	q��9���A�*;


total_loss���@

error_RR]V?

learning_rate_1��j7��c�I       6%�	�
�9���A�*;


total_loss�:A

error_R��P?

learning_rate_1��j7�0�I       6%�	�N�9���A�*;


total_loss��@

error_R.`U?

learning_rate_1��j7�6�MI       6%�	��9���A�*;


total_loss�A

error_Rf~B?

learning_rate_1��j7awI       6%�	���9���A�*;


total_loss��@

error_RF�K?

learning_rate_1��j7IE¦I       6%�	� �9���A�*;


total_lossg�@

error_R�U?

learning_rate_1��j7�q�I       6%�	cf�9���A�*;


total_lossC#�@

error_R6�I?

learning_rate_1��j7�b�=I       6%�	ë�9���A�*;


total_loss�(�@

error_R|�E?

learning_rate_1��j7��6jI       6%�	m��9���A�*;


total_loss�*�@

error_R�"X?

learning_rate_1��j7��V[I       6%�	�0�9���A�*;


total_lossCҖ@

error_RPM?

learning_rate_1��j7����I       6%�	q�9���A�*;


total_lossV�@

error_RLM?

learning_rate_1��j7!���I       6%�	"��9���A�*;


total_loss�{�@

error_Rm�_?

learning_rate_1��j7Q�kkI       6%�	���9���A�*;


total_loss�d�@

error_R�[L?

learning_rate_1��j7��Q�I       6%�	sB�9���A�*;


total_loss�[�@

error_R#�B?

learning_rate_1��j7���I       6%�	9��9���A�*;


total_lossxZ�@

error_R�{T?

learning_rate_1��j7=�غI       6%�	���9���A�*;


total_lossj3A

error_R�T?

learning_rate_1��j7U��I       6%�	O�9���A�*;


total_lossa�@

error_R��E?

learning_rate_1��j7SuI       6%�	�J�9���A�*;


total_loss��@

error_R�$U?

learning_rate_1��j7�2I       6%�	 ��9���A�*;


total_loss<ի@

error_RnO?

learning_rate_1��j7A�^sI       6%�	d��9���A�*;


total_loss`f�@

error_R��O?

learning_rate_1��j7tV��I       6%�	=�9���A�*;


total_lossV�@

error_R�S>?

learning_rate_1��j7�Oy+I       6%�	�W�9���A�*;


total_loss���@

error_R��8?

learning_rate_1��j7�-�I       6%�	:��9���A�*;


total_loss|i�@

error_R#<J?

learning_rate_1��j7�LPrI       6%�	E��9���A�*;


total_loss.ZA

error_R�AN?

learning_rate_1��j7w1A�I       6%�	U!�9���A�*;


total_loss�@�@

error_R��T?

learning_rate_1��j7�*H�I       6%�	�b�9���A�*;


total_loss��@

error_R�DS?

learning_rate_1��j7��r�I       6%�	��9���A�*;


total_loss[�@

error_R��M?

learning_rate_1��j7��	vI       6%�	4��9���A�*;


total_loss��@

error_R�?J?

learning_rate_1��j7��I       6%�	�%�9���A�*;


total_lossͿ�@

error_RJ�O?

learning_rate_1��j7���xI       6%�	~k�9���A�*;


total_loss_��@

error_R�R?

learning_rate_1��j7��I       6%�	a��9���A�*;


total_lossd�@

error_Rl�A?

learning_rate_1��j7�`aI       6%�	p��9���A�*;


total_loss��@

error_R�4?

learning_rate_1��j7��t4I       6%�	TR�9���A�*;


total_lossj�@

error_R�,H?

learning_rate_1��j7NBI       6%�	-��9���A�*;


total_loss)�{@

error_R�U?

learning_rate_1��j7o��I       6%�	a��9���A�*;


total_loss��A

error_R��N?

learning_rate_1��j7���uI       6%�	�(�9���A�*;


total_lossT��@

error_RMbL?

learning_rate_1��j7˜d�I       6%�	7n�9���A�*;


total_loss��@

error_R�d^?

learning_rate_1��j7���I       6%�	��9���A�*;


total_lossP�@

error_R�(I?

learning_rate_1��j7��;EI       6%�	���9���A�*;


total_lossԤ�@

error_RarV?

learning_rate_1��j7��iVI       6%�	�?�9���A�*;


total_loss,��@

error_RE?

learning_rate_1��j7�W�I       6%�	A��9���A�*;


total_lossSP�@

error_Ra�M?

learning_rate_1��j7ω?I       6%�	1��9���A�*;


total_lossݭ�@

error_RX6=?

learning_rate_1��j7��!9I       6%�	�9���A�*;


total_loss���@

error_R�U?

learning_rate_1��j7$��LI       6%�	Y�9���A�*;


total_loss��`@

error_R��Y?

learning_rate_1��j7l�zeI       6%�	2��9���A�*;


total_loss�A

error_R��N?

learning_rate_1��j7kI       6%�	#��9���A�*;


total_loss��@

error_R�3?

learning_rate_1��j7c���I       6%�	+�9���A�*;


total_loss�k�@

error_R�3b?

learning_rate_1��j7=��I       6%�	4p�9���A�*;


total_loss�'�@

error_R�4D?

learning_rate_1��j7����I       6%�	��9���A�*;


total_loss�S�@

error_R)�[?

learning_rate_1��j7�D�I       6%�	c��9���A�*;


total_loss=PH@

error_R�M?

learning_rate_1��j7k%�I       6%�	�C�9���A�*;


total_lossEϩ@

error_R��R?

learning_rate_1��j7M>��I       6%�	n��9���A�*;


total_loss��@

error_RM?

learning_rate_1��j7�O��I       6%�	��9���A�*;


total_loss�H�@

error_R��6?

learning_rate_1��j7F�hI       6%�	��9���A�*;


total_loss㒍@

error_R1C?

learning_rate_1��j7֕YLI       6%�	{R�9���A�*;


total_lossz(r@

error_Rv�H?

learning_rate_1��j7�7lI       6%�	���9���A�*;


total_loss�إ@

error_R�jI?

learning_rate_1��j7���I       6%�	n��9���A�*;


total_loss�a�@

error_R�LL?

learning_rate_1��j7V�7NI       6%�	J"�9���A�*;


total_loss��@

error_R��F?

learning_rate_1��j7���,I       6%�	ap�9���A�*;


total_lossJ�@

error_RL�L?

learning_rate_1��j7��I       6%�	���9���A�*;


total_loss��|@

error_R��P?

learning_rate_1��j7NgFbI       6%�	��9���A�*;


total_loss#�|@

error_R߹9?

learning_rate_1��j7t�Z<I       6%�	FL�9���A�*;


total_loss���@

error_RA7X?

learning_rate_1��j7"YK�I       6%�	f��9���A�*;


total_loss7Z�@

error_RqP?

learning_rate_1��j7F`N�I       6%�	���9���A�*;


total_loss��A

error_R1�B?

learning_rate_1��j7�2Q�I       6%�	��9���A�*;


total_loss��@

error_R_@^?

learning_rate_1��j7�N�I       6%�	LN�9���A�*;


total_loss���@

error_R��K?

learning_rate_1��j73Y��I       6%�	��9���A�*;


total_losst0�@

error_R��K?

learning_rate_1��j7_�߸I       6%�	���9���A�*;


total_lossz,�@

error_RJW?

learning_rate_1��j7Qnz3I       6%�	<�9���A�*;


total_loss�&�@

error_R�}U?

learning_rate_1��j7j:��I       6%�	/��9���A�*;


total_loss,G�@

error_R�P?

learning_rate_1��j7+�`kI       6%�	���9���A�*;


total_loss*�h@

error_R!�6?

learning_rate_1��j7���I       6%�	}'�9���A�*;


total_lossM��@

error_Rt�^?

learning_rate_1��j7?T�I       6%�	G��9���A�*;


total_loss�@

error_R��T?

learning_rate_1��j7��܃I       6%�	5��9���A�*;


total_lossV6A

error_R�W?

learning_rate_1��j7���%I       6%�	�d�9���A�*;


total_loss�F�@

error_R�>?

learning_rate_1��j79��]I       6%�	��9���A�*;


total_loss4��@

error_R<�U?

learning_rate_1��j7(�<�I       6%�	)�9���A�*;


total_loss8!�@

error_R��H?

learning_rate_1��j7q�I       6%�	�w�9���A�*;


total_loss�o�@

error_R��<?

learning_rate_1��j7�j�I       6%�	.��9���A�*;


total_loss1r�@

error_RC�T?

learning_rate_1��j7��bdI       6%�	��9���A�*;


total_loss=�@

error_R��X?

learning_rate_1��j7Wҙ�I       6%�	;d�9���A�*;


total_loss�ƨ@

error_R@�2?

learning_rate_1��j7M��}I       6%�	��9���A�*;


total_loss��@

error_Rs`?

learning_rate_1��j7��uJI       6%�	�+�9���A�*;


total_lossOѽ@

error_R��S?

learning_rate_1��j7�#�I       6%�	_��9���A�*;


total_loss�j�@

error_R�9E?

learning_rate_1��j7���I       6%�	���9���A�*;


total_lossa�g@

error_R�6?

learning_rate_1��j7*��jI       6%�	�6�9���A�*;


total_loss\*�@

error_R�AR?

learning_rate_1��j79��(I       6%�	ʓ�9���A�*;


total_lossʄ�@

error_R�S?

learning_rate_1��j7�6�I       6%�	r��9���A�*;


total_loss�}�@

error_R$�V?

learning_rate_1��j7?�I�I       6%�	�1�9���A�*;


total_lossQ��@

error_Rl!`?

learning_rate_1��j7��vI       6%�	���9���A�*;


total_lossf��@

error_R�D?

learning_rate_1��j7�E�JI       6%�	���9���A�*;


total_loss<�@

error_R��\?

learning_rate_1��j7]��?I       6%�	�/ :���A�*;


total_loss���@

error_RwO?

learning_rate_1��j7��JI       6%�	� :���A�*;


total_lossI��@

error_R�G?

learning_rate_1��j7T0�)I       6%�	�� :���A�*;


total_loss�@A

error_R;tS?

learning_rate_1��j7ն�I       6%�	�,:���A�*;


total_loss��@

error_R�:A?

learning_rate_1��j7�Dx{I       6%�	�}:���A�*;


total_loss|��@

error_R�`?

learning_rate_1��j7���I       6%�	=�:���A�*;


total_loss.��@

error_R��G?

learning_rate_1��j7���nI       6%�	c,:���A�*;


total_lossɉ�@

error_R�R?

learning_rate_1��j7F�I       6%�	r:���A�*;


total_loss���@

error_R��a?

learning_rate_1��j74�J=I       6%�	:�:���A�*;


total_loss*Y�@

error_R�7O?

learning_rate_1��j7:I       6%�	��:���A�*;


total_loss��@

error_R�F@?

learning_rate_1��j7�jk�I       6%�	�?:���A�*;


total_lossx�A

error_R�eM?

learning_rate_1��j7@W�I       6%�	��:���A�*;


total_loss�f�@

error_R�|D?

learning_rate_1��j7�#_`I       6%�	��:���A�*;


total_loss0�@

error_RO�J?

learning_rate_1��j7A�h~I       6%�	O
:���A�*;


total_loss���@

error_R�B?

learning_rate_1��j7����I       6%�	M:���A�*;


total_lossC��@

error_R� P?

learning_rate_1��j7��LI       6%�	ʑ:���A�*;


total_loss�h�@

error_RA�<?

learning_rate_1��j7ʻ�HI       6%�		�:���A�*;


total_loss��@

error_R�F?

learning_rate_1��j7L��/I       6%�	f:���A�*;


total_loss�@

error_R�'A?

learning_rate_1��j7���I       6%�	}k:���A�*;


total_loss���@

error_R��P?

learning_rate_1��j7���I       6%�	��:���A�*;


total_loss���@

error_R�xF?

learning_rate_1��j7M)r�I       6%�	�:���A�*;


total_loss2��@

error_RJ\F?

learning_rate_1��j7�Y�ZI       6%�	n:���A�*;


total_loss���@

error_R߉B?

learning_rate_1��j7m��!I       6%�	@�:���A�*;


total_loss�=A

error_R�dP?

learning_rate_1��j7���I       6%�	4�:���A�*;


total_loss�	A

error_R�J?

learning_rate_1��j7+S�I       6%�	�@:���A�*;


total_lossN5�@

error_RnSC?

learning_rate_1��j7�z�I       6%�	��:���A�*;


total_loss#A

error_R��??

learning_rate_1��j7���I       6%�	��:���A�*;


total_loss�E�@

error_R�(P?

learning_rate_1��j7	��%I       6%�	2:���A�*;


total_loss���@

error_Rle\?

learning_rate_1��j7i�{I       6%�	_:���A�*;


total_lossJ��@

error_R}�C?

learning_rate_1��j7�+�NI       6%�	n�:���A�*;


total_loss�ެ@

error_Ri�M?

learning_rate_1��j77�/wI       6%�	.�:���A�*;


total_loss���@

error_Ri[N?

learning_rate_1��j7�':eI       6%�	�4	:���A�*;


total_loss� �@

error_R�,B?

learning_rate_1��j7T��\I       6%�	��	:���A�*;


total_lossJ=�@

error_R1�V?

learning_rate_1��j7	wI       6%�	��	:���A�*;


total_loss���@

error_RW�T?

learning_rate_1��j7`���I       6%�	�
:���A�*;


total_lossD��@

error_R�%d?

learning_rate_1��j7�m�$I       6%�	&R
:���A�*;


total_loss�jA

error_R��F?

learning_rate_1��j7�pp�I       6%�	��
:���A�*;


total_loss;��@

error_R��^?

learning_rate_1��j7�r�I       6%�	��
:���A�*;


total_loss�n�@

error_R;�Q?

learning_rate_1��j7p�9I       6%�	%:���A�*;


total_loss���@

error_R#@Q?

learning_rate_1��j7���I       6%�	ai:���A�*;


total_lossA�A

error_RʩW?

learning_rate_1��j74��I       6%�	��:���A�*;


total_loss���@

error_R$Q?

learning_rate_1��j72�$I       6%�	�:���A�*;


total_loss�&�@

error_R�U?

learning_rate_1��j7��VRI       6%�	f:���A�*;


total_loss4��@

error_R�+Q?

learning_rate_1��j7ߨ�UI       6%�	��:���A�*;


total_loss_��@

error_Rx6Z?

learning_rate_1��j7�x��I       6%�	x':���A�*;


total_lossSz�@

error_R�4S?

learning_rate_1��j76>�I       6%�	Cz:���A�*;


total_loss��A

error_R]=?

learning_rate_1��j7�<tI       6%�	c�:���A�*;


total_loss�
�@

error_R��K?

learning_rate_1��j7�6�&I       6%�	�#:���A�*;


total_lossR��@

error_R�xO?

learning_rate_1��j7ڽ;4I       6%�	bv:���A�*;


total_lossꘛ@

error_R�9J?

learning_rate_1��j7����I       6%�	<�:���A�*;


total_lossJ�@

error_R�Y?

learning_rate_1��j7�ԹI       6%�	O7:���A�*;


total_lossu�@

error_R�e:?

learning_rate_1��j7۰p�I       6%�	��:���A�*;


total_lossd��@

error_R�9]?

learning_rate_1��j7���I       6%�	�:���A�*;


total_loss\e�@

error_R=�_?

learning_rate_1��j7��d�I       6%�	�V:���A�*;


total_loss��@

error_R��J?

learning_rate_1��j7y���I       6%�	��:���A�*;


total_loss�>�@

error_R.nd?

learning_rate_1��j7�YI       6%�	J�:���A�*;


total_loss$_�@

error_R��S?

learning_rate_1��j7O8��I       6%�	�N:���A�*;


total_loss�_u@

error_R�CO?

learning_rate_1��j7p@t�I       6%�	��:���A�*;


total_loss���@

error_R�K?

learning_rate_1��j7 �ʐI       6%�	�F:���A�*;


total_loss��@

error_R6�W?

learning_rate_1��j76�'�I       6%�	J�:���A�*;


total_loss�ס@

error_RMJC?

learning_rate_1��j7T��rI       6%�	�":���A�*;


total_loss*�@

error_RO�H?

learning_rate_1��j7eT;HI       6%�	�q:���A�*;


total_loss�?�@

error_R�-M?

learning_rate_1��j7tH�:I       6%�	7�:���A�*;


total_loss�=�@

error_R�p\?

learning_rate_1��j7x���I       6%�	�:���A�*;


total_loss�Z A

error_R�I?

learning_rate_1��j7fu�$I       6%�	��:���A�*;


total_lossޝ@

error_R�G?

learning_rate_1��j7�L�I       6%�	��:���A�*;


total_lossF2�@

error_R��E?

learning_rate_1��j7X{��I       6%�	�>:���A�*;


total_loss��u@

error_R�P?

learning_rate_1��j7�L�I       6%�	��:���A�*;


total_lossȼ�@

error_R�nT?

learning_rate_1��j7���$I       6%�	�q:���A�*;


total_lossx��@

error_Rn�Z?

learning_rate_1��j7�ɽ_I       6%�	"�:���A�*;


total_loss�c�@

error_R{�e?

learning_rate_1��j7T�I       6%�	H!:���A�*;


total_loss_e�@

error_R!�J?

learning_rate_1��j7�l�I       6%�	9y:���A�*;


total_loss�8�@

error_R�}A?

learning_rate_1��j7����I       6%�	�:���A�*;


total_loss��@

error_R�Wc?

learning_rate_1��j7�딓I       6%�	$^:���A�*;


total_loss��@

error_R��T?

learning_rate_1��j72q?&I       6%�	~�:���A�*;


total_loss���@

error_R�\?

learning_rate_1��j7@��ZI       6%�	2 :���A�*;


total_loss^�@

error_Rá]?

learning_rate_1��j7LI       6%�	�:���A�*;


total_loss�
�@

error_R��L?

learning_rate_1��j7�u@I       6%�	Z�:���A�*;


total_loss ţ@

error_R�B?

learning_rate_1��j76���I       6%�	�(:���A�*;


total_loss�^.A

error_RdW?

learning_rate_1��j7=[�I       6%�	@p:���A�*;


total_lossr��@

error_R$
F?

learning_rate_1��j7�#B�I       6%�	s�:���A�*;


total_loss�Q�@

error_R:�O?

learning_rate_1��j7� �BI       6%�	_ :���A�*;


total_lossE��@

error_R�x]?

learning_rate_1��j7���I       6%�	�F:���A�*;


total_loss�$j@

error_RqR?

learning_rate_1��j7hJf�I       6%�	Ŏ:���A�*;


total_lossR�@

error_R�KH?

learning_rate_1��j7�VI       6%�	��:���A�*;


total_loss�.�@

error_R�A?

learning_rate_1��j7|(I       6%�	�:���A�*;


total_loss��A

error_R�V?

learning_rate_1��j7��I       6%�	Se:���A�*;


total_loss,��@

error_R#`?

learning_rate_1��j7xFq�I       6%�	��:���A�*;


total_loss���@

error_RԫK?

learning_rate_1��j77мI       6%�	��:���A�*;


total_loss-��@

error_RT�M?

learning_rate_1��j7;L�I       6%�	�8:���A�*;


total_loss;��@

error_RX�W?

learning_rate_1��j7Qr��I       6%�	��:���A�*;


total_loss�?@

error_R:�D?

learning_rate_1��j7X_� I       6%�	��:���A�*;


total_lossػ@

error_RԚJ?

learning_rate_1��j7�g�I       6%�	�:���A�*;


total_loss��@

error_R�K?

learning_rate_1��j7����I       6%�	�`:���A�*;


total_loss�W�@

error_R��H?

learning_rate_1��j7��-_I       6%�	L�:���A�*;


total_lossy��@

error_R/�J?

learning_rate_1��j7��@!I       6%�	�:���A�*;


total_loss�-�@

error_RfYG?

learning_rate_1��j7pC�I       6%�	��:���A�*;


total_loss�h�@

error_R��]?

learning_rate_1��j7�}&I       6%�	UW :���A�*;


total_loss�W�@

error_R�$O?

learning_rate_1��j7���I       6%�	� :���A�*;


total_loss�{�@

error_R�pN?

learning_rate_1��j7P��I       6%�	N!:���A�*;


total_loss���@

error_R{R?

learning_rate_1��j7N8�I       6%�	!�!:���A�*;


total_loss���@

error_R�HT?

learning_rate_1��j7���I       6%�	/�":���A�*;


total_loss��@

error_R�<E?

learning_rate_1��j7)�I       6%�	{#:���A�*;


total_loss���@

error_R6�D?

learning_rate_1��j7�J��I       6%�	��#:���A�*;


total_loss_�@

error_R͕O?

learning_rate_1��j7cQj�I       6%�	\�$:���A�*;


total_loss@\(A

error_R�7[?

learning_rate_1��j7�5�XI       6%�	%:���A�*;


total_loss^ƣ@

error_R=�D?

learning_rate_1��j7=U�I       6%�	r�&:���A�*;


total_lossjõ@

error_R��A?

learning_rate_1��j7e���I       6%�	�':���A�*;


total_loss���@

error_R%�I?

learning_rate_1��j7
�;~I       6%�	p':���A�*;


total_loss�a$A

error_R8L?

learning_rate_1��j7˽��I       6%�	�@(:���A�*;


total_loss�A

error_R��i?

learning_rate_1��j7"��I       6%�	�(:���A�*;


total_lossa��@

error_R{wI?

learning_rate_1��j7�ۯQI       6%�	u[):���A�*;


total_loss\�@

error_R�/a?

learning_rate_1��j7��l�I       6%�	`�):���A�*;


total_loss��@

error_R�R?

learning_rate_1��j7it�I       6%�	�X*:���A�*;


total_loss���@

error_R�AK?

learning_rate_1��j7�u+I       6%�	h�*:���A�*;


total_loss�@

error_R�kZ?

learning_rate_1��j7��LAI       6%�	]a+:���A�*;


total_loss/p�@

error_R?�X?

learning_rate_1��j7^`��I       6%�	(�+:���A�*;


total_loss�Ǻ@

error_R_�R?

learning_rate_1��j7f-I       6%�	m,:���A�*;


total_loss�p�@

error_R�J?

learning_rate_1��j7m�K�I       6%�	�y-:���A�*;


total_loss�.�@

error_RQK?

learning_rate_1��j7��I       6%�	��-:���A�*;


total_loss`��@

error_R6�G?

learning_rate_1��j7�I�I       6%�	%�.:���A�*;


total_loss�,�@

error_R��K?

learning_rate_1��j7X���I       6%�	�&/:���A�*;


total_loss���@

error_Rz�D?

learning_rate_1��j7�J&I       6%�	��/:���A�*;


total_loss�|�@

error_R4�F?

learning_rate_1��j7%e�I       6%�	�/:���A�*;


total_loss.�@

error_R��9?

learning_rate_1��j7�ީI       6%�	`0:���A�*;


total_lossW&�@

error_R��]?

learning_rate_1��j7��gI       6%�	�1:���A�*;


total_loss�@

error_R\R?

learning_rate_1��j7�w��I       6%�	/u1:���A�*;


total_loss��@

error_R�\?

learning_rate_1��j7p[I       6%�	��1:���A�*;


total_loss�Xt@

error_R��]?

learning_rate_1��j7`�I       6%�	�G2:���A�*;


total_losse�@

error_R��B?

learning_rate_1��j7�X��I       6%�	�2:���A�*;


total_loss�L�@

error_R��C?

learning_rate_1��j7߂ѳI       6%�	Ȟ3:���A�*;


total_loss�{�@

error_R�v@?

learning_rate_1��j7}�CI       6%�	�4:���A�*;


total_loss���@

error_R�9E?

learning_rate_1��j7�G? I       6%�	|R4:���A�*;


total_loss�x�@

error_R�R?

learning_rate_1��j7��I       6%�	ӟ4:���A�*;


total_loss���@

error_R��K?

learning_rate_1��j7�cv�I       6%�	��4:���A�*;


total_lossA|�@

error_R6`[?

learning_rate_1��j7#�8I       6%�	�55:���A�*;


total_loss���@

error_R�K?

learning_rate_1��j7U�*I       6%�	��5:���A�*;


total_loss�@

error_R�aS?

learning_rate_1��j7�6II       6%�	�5:���A�*;


total_loss*�A

error_R��X?

learning_rate_1��j7��J�I       6%�	A66:���A�*;


total_lossþ�@

error_RSGK?

learning_rate_1��j7���I       6%�	�~6:���A�*;


total_losst�@

error_RUR?

learning_rate_1��j7��88I       6%�	��6:���A�*;


total_loss6]�@

error_R�vK?

learning_rate_1��j7��I       6%�	�7:���A�*;


total_loss�^�@

error_RfoK?

learning_rate_1��j7�}��I       6%�	�^7:���A�*;


total_loss#��@

error_R`�E?

learning_rate_1��j7WV�iI       6%�	>�7:���A�*;


total_loss���@

error_RJ�A?

learning_rate_1��j7���hI       6%�	D�7:���A�*;


total_lossFw�@

error_R�F?

learning_rate_1��j7�2I       6%�	�>8:���A�*;


total_lossq0�@

error_R-O?

learning_rate_1��j7�e��I       6%�	��8:���A�*;


total_loss)MA@

error_Rs�M?

learning_rate_1��j7]��KI       6%�	�8:���A�*;


total_loss?��@

error_R��>?

learning_rate_1��j7nJ-0I       6%�	b9:���A�*;


total_loss���@

error_Ro�D?

learning_rate_1��j7��CmI       6%�	h9:���A�*;


total_loss�r�@

error_RjT?

learning_rate_1��j7WV��I       6%�	��9:���A�*;


total_loss�F�@

error_R�I?

learning_rate_1��j7y;�kI       6%�	[::���A�*;


total_lossI�@

error_R��c?

learning_rate_1��j7G��I       6%�	�R::���A�*;


total_loss2ǳ@

error_R׍T?

learning_rate_1��j7��I       6%�	h�::���A�*;


total_loss��@

error_R��D?

learning_rate_1��j7ڲI       6%�	i�::���A�*;


total_loss��@

error_R��P?

learning_rate_1��j7��{�I       6%�	?;:���A�*;


total_loss��@

error_R�B5?

learning_rate_1��j7�М�I       6%�	�;:���A�*;


total_loss���@

error_R`RH?

learning_rate_1��j7r�cI       6%�	o�;:���A�*;


total_loss���@

error_R��\?

learning_rate_1��j7�g[&I       6%�	�)<:���A�*;


total_loss��@

error_R}<?

learning_rate_1��j74��?I       6%�	˒<:���A�*;


total_loss��c@

error_R3�P?

learning_rate_1��j7W9ѩI       6%�	�=:���A�*;


total_lossdǙ@

error_R��F?

learning_rate_1��j7�I       6%�	��=:���A�*;


total_loss�tc@

error_R��N?

learning_rate_1��j7��P�I       6%�	>6>:���A�*;


total_loss�A

error_Rv�N?

learning_rate_1��j7��4I       6%�	J�>:���A�*;


total_loss��@

error_R��X?

learning_rate_1��j7P��I       6%�	`?:���A�*;


total_loss��@

error_R��B?

learning_rate_1��j7�w�I       6%�	��?:���A�*;


total_loss*��@

error_R��I?

learning_rate_1��j7�dQ%I       6%�	a8@:���A�*;


total_loss���@

error_Rs�C?

learning_rate_1��j7R��I       6%�	r�@:���A�*;


total_loss�!�@

error_RxAT?

learning_rate_1��j7NV��I       6%�	�5A:���A�*;


total_lossN�@

error_RԛD?

learning_rate_1��j7�_�]I       6%�	h�A:���A�*;


total_loss���@

error_RW�\?

learning_rate_1��j74^nMI       6%�	�A:���A�*;


total_lossW�@

error_Rj�M?

learning_rate_1��j7�*�I       6%�	�nB:���A�*;


total_loss���@

error_R)~Z?

learning_rate_1��j7ŵ�I       6%�	��B:���A�*;


total_loss��	A

error_R��G?

learning_rate_1��j7�jmI       6%�		LC:���A�*;


total_lossE�@

error_RE�G?

learning_rate_1��j7K�c�I       6%�	��C:���A�*;


total_loss�L�@

error_R_�@?

learning_rate_1��j7D�6FI       6%�	�%D:���A�*;


total_loss��@

error_R�x;?

learning_rate_1��j7C,��I       6%�	+zD:���A�*;


total_loss�B�@

error_R��;?

learning_rate_1��j7��RII       6%�	V�D:���A�*;


total_loss��{@

error_R Qi?

learning_rate_1��j7-��/I       6%�	�E:���A�*;


total_loss�z�@

error_R��C?

learning_rate_1��j7��PkI       6%�	|E:���A�*;


total_lossm�@

error_R�yO?

learning_rate_1��j7�"�cI       6%�	_�E:���A�*;


total_loss���@

error_RM?

learning_rate_1��j7?a��I       6%�	-:F:���A�*;


total_loss�6�@

error_R�i?

learning_rate_1��j7��I       6%�	؈F:���A�*;


total_loss�~w@

error_R8O?

learning_rate_1��j7��
I       6%�	��F:���A�*;


total_lossxސ@

error_R,L?

learning_rate_1��j7i�6I       6%�	�4G:���A�*;


total_lossz9�@

error_RܛP?

learning_rate_1��j7��3I       6%�	�G:���A�*;


total_loss.��@

error_R�NZ?

learning_rate_1��j7��~I       6%�	J�G:���A�*;


total_losslv�@

error_R�\?

learning_rate_1��j7K��I       6%�	("H:���A�*;


total_loss��@

error_R�G?

learning_rate_1��j7l�I       6%�	znH:���A�*;


total_loss���@

error_RU?

learning_rate_1��j7����I       6%�	��H:���A�*;


total_losso`A

error_R�F?

learning_rate_1��j7�V�I       6%�	�I:���A�*;


total_lossA

error_RC�I?

learning_rate_1��j7�0��I       6%�	RmI:���A�*;


total_loss�4A

error_R0W?

learning_rate_1��j7�rT}I       6%�	��I:���A�*;


total_loss�H�@

error_R�L?

learning_rate_1��j7=�I       6%�	T	J:���A�*;


total_loss�z�@

error_R�yH?

learning_rate_1��j7?�KI       6%�	-^J:���A�*;


total_loss���@

error_R��C?

learning_rate_1��j7���I       6%�	c�J:���A�*;


total_lossZ��@

error_RCR[?

learning_rate_1��j7�:��I       6%�	W�J:���A�*;


total_loss��@

error_RM[?

learning_rate_1��j7�YfI       6%�	OK:���A�*;


total_loss�i�@

error_R�9P?

learning_rate_1��j7ݔ҃I       6%�	H�K:���A�*;


total_lossWs�@

error_R��W?

learning_rate_1��j7(c�I       6%�	��K:���A�*;


total_loss�ª@

error_RF=L?

learning_rate_1��j7�^TI       6%�	�AL:���A�*;


total_loss�C�@

error_R��D?

learning_rate_1��j7�v/�I       6%�	6�L:���A�*;


total_loss{~�@

error_R��F?

learning_rate_1��j7�W�I       6%�	��L:���A�*;


total_loss���@

error_R��H?

learning_rate_1��j7d��I       6%�	M:���A�*;


total_loss�ۋ@

error_R:?

learning_rate_1��j7�I       6%�	G]M:���A�*;


total_loss�@

error_R�?J?

learning_rate_1��j7�Oz^I       6%�	��M:���A�*;


total_loss@H A

error_R��S?

learning_rate_1��j7�:w�I       6%�	+�M:���A�*;


total_loss���@

error_Rjuj?

learning_rate_1��j7iB��I       6%�	�,N:���A�*;


total_loss��@

error_R5^?

learning_rate_1��j7��I       6%�	�wN:���A�*;


total_loss1��@

error_R6�<?

learning_rate_1��j7� XI       6%�	 �N:���A�*;


total_loss���@

error_R�9J?

learning_rate_1��j7;��I       6%�	OO:���A�*;


total_loss�߰@

error_Rs�E?

learning_rate_1��j7�D�I       6%�	CYO:���A�*;


total_loss�k�@

error_R�cT?

learning_rate_1��j7n��FI       6%�	e�O:���A�*;


total_loss{��@

error_R6�[?

learning_rate_1��j7���I       6%�	��O:���A�*;


total_loss���@

error_R.�N?

learning_rate_1��j7��#SI       6%�	".P:���A�*;


total_lossWH�@

error_R}lW?

learning_rate_1��j7Ӎ�I       6%�	�{P:���A�*;


total_loss���@

error_R�A?

learning_rate_1��j74�]I       6%�	��P:���A�*;


total_lossoCg@

error_RX07?

learning_rate_1��j7ܴ%�I       6%�	I#Q:���A�*;


total_loss���@

error_R�m_?

learning_rate_1��j7��3I       6%�	�jQ:���A�*;


total_loss�@

error_R��U?

learning_rate_1��j7�<Y1I       6%�	:�Q:���A�*;


total_loss���@

error_R��V?

learning_rate_1��j7խI�I       6%�	��Q:���A�*;


total_loss��@

error_R��A?

learning_rate_1��j7V��[I       6%�	,:R:���A�*;


total_loss���@

error_R��W?

learning_rate_1��j7�X>�I       6%�	bR:���A�*;


total_loss�o@

error_R�B?

learning_rate_1��j7O��AI       6%�	��R:���A�*;


total_loss �@

error_R)F?

learning_rate_1��j7���jI       6%�	HS:���A�*;


total_loss��@

error_R�	W?

learning_rate_1��j7J�I       6%�	�PS:���A�*;


total_loss3f�@

error_R�l@?

learning_rate_1��j7 Y��I       6%�	��S:���A�*;


total_loss(�A

error_R=�Q?

learning_rate_1��j7��I       6%�	��S:���A�*;


total_loss��
A

error_R�O?

learning_rate_1��j7�7BI       6%�	!T:���A�*;


total_loss+�@

error_R�N?

learning_rate_1��j7`>0I       6%�	zgT:���A�*;


total_loss���@

error_Ra�C?

learning_rate_1��j7Fz(�I       6%�	��T:���A�*;


total_loss��@

error_RD?

learning_rate_1��j7Q�԰I       6%�	��T:���A�*;


total_lossc��@

error_R�AO?

learning_rate_1��j7js��I       6%�	�;U:���A�*;


total_loss�=�@

error_RHX?

learning_rate_1��j7u�I       6%�	��U:���A�*;


total_losss�@

error_R��H?

learning_rate_1��j7�ALI       6%�	(�U:���A�*;


total_loss=\�@

error_R��J?

learning_rate_1��j7i ]I       6%�	�8V:���A�*;


total_loss�$�@

error_R�{]?

learning_rate_1��j7Q~_I       6%�	x}V:���A�*;


total_lossT��@

error_R3�^?

learning_rate_1��j7��I       6%�	r�V:���A�*;


total_loss�@

error_R�JO?

learning_rate_1��j7�Q#I       6%�	�W:���A�*;


total_lossܶ@

error_R��J?

learning_rate_1��j7�ד+I       6%�	&HW:���A�*;


total_loss��@

error_RſB?

learning_rate_1��j7Y�I       6%�	��W:���A�*;


total_loss���@

error_R��W?

learning_rate_1��j7K߯�I       6%�	6�W:���A�*;


total_lossj��@

error_R��T?

learning_rate_1��j7�A��I       6%�	�X:���A�*;


total_lossX��@

error_R�`?

learning_rate_1��j7з�I       6%�	�^X:���A�*;


total_loss�R�@

error_RcP?

learning_rate_1��j7�^qI       6%�	 �X:���A�*;


total_loss��@

error_R�N?

learning_rate_1��j7qXv�I       6%�	U�X:���A�*;


total_loss�
�@

error_R�oH?

learning_rate_1��j7��F�I       6%�	1Y:���A�*;


total_loss�B�@

error_R��E?

learning_rate_1��j7Og�I       6%�	xY:���A�*;


total_loss	ƹ@

error_R\�X?

learning_rate_1��j7�-s�I       6%�	��Y:���A�*;


total_lossQN�@

error_R��P?

learning_rate_1��j7���<I       6%�	�Z:���A�*;


total_loss���@

error_R��R?

learning_rate_1��j7? ��I       6%�	�XZ:���A�*;


total_lossl�@

error_R*�O?

learning_rate_1��j7 $�I       6%�	��Z:���A�*;


total_lossf��@

error_RZ!??

learning_rate_1��j7T1�9I       6%�	<�Z:���A�*;


total_lossH��@

error_R��@?

learning_rate_1��j7��u�I       6%�	<*[:���A�*;


total_loss��c@

error_R��[?

learning_rate_1��j7-|/�I       6%�	�q[:���A�*;


total_loss��@

error_R�*R?

learning_rate_1��j7;={II       6%�	��[:���A�*;


total_loss})�@

error_R��Z?

learning_rate_1��j7%��I       6%�	� \:���A�*;


total_loss��@

error_R�O?

learning_rate_1��j7�^3I       6%�	�F\:���A�*;


total_loss�v�@

error_R&b?

learning_rate_1��j7(��3I       6%�	/�\:���A�*;


total_loss��@

error_RT�R?

learning_rate_1��j7���I       6%�	��\:���A�*;


total_loss�B�@

error_R�Q?

learning_rate_1��j7�fGI       6%�	� ]:���A�*;


total_lossv�@

error_R��W?

learning_rate_1��j7���4I       6%�	i]:���A�*;


total_lossrVA

error_R�F?

learning_rate_1��j7�U��I       6%�	̯]:���A�*;


total_loss�;�@

error_R6YX?

learning_rate_1��j7
ʅ�I       6%�	��]:���A�*;


total_loss�Q�@

error_R=�L?

learning_rate_1��j7dk��I       6%�	�<^:���A�*;


total_loss�_�@

error_Rs�T?

learning_rate_1��j7<x�I       6%�	r�^:���A�*;


total_lossR��@

error_R��\?

learning_rate_1��j7�+��I       6%�	��^:���A�*;


total_loss&��@

error_R��S?

learning_rate_1��j7{ʋ7I       6%�	;	_:���A�*;


total_loss���@

error_RR�G?

learning_rate_1��j7��I       6%�	qN_:���A�*;


total_loss���@

error_RVV?

learning_rate_1��j7��I       6%�	��_:���A�*;


total_lossɂA

error_R*�q?

learning_rate_1��j7¾RI       6%�	O�_:���A�*;


total_lossWL�@

error_R�eS?

learning_rate_1��j7ٽ�|I       6%�	v`:���A�*;


total_loss���@

error_R/[?

learning_rate_1��j7�K�dI       6%�	X_`:���A�*;


total_loss4.�@

error_R:�>?

learning_rate_1��j7�ՋI       6%�	�`:���A�*;


total_loss,��@

error_R�6M?

learning_rate_1��j7g�gI       6%�	��`:���A�*;


total_lossZzA

error_R%�R?

learning_rate_1��j7��I       6%�	�-a:���A�*;


total_loss�A

error_Rt�E?

learning_rate_1��j7�J#�I       6%�	qa:���A�*;


total_loss�J�@

error_R>=?

learning_rate_1��j7�s��I       6%�	&�a:���A�*;


total_loss��@

error_R�N?

learning_rate_1��j7����I       6%�	P�a:���A�*;


total_loss f�@

error_RL�J?

learning_rate_1��j7���I       6%�	�>b:���A�*;


total_loss��k@

error_R$|E?

learning_rate_1��j7�D�
I       6%�	��b:���A�*;


total_lossH��@

error_R&&E?

learning_rate_1��j7�ӡ8I       6%�	K�b:���A�*;


total_loss���@

error_R�L?

learning_rate_1��j7[��XI       6%�	�c:���A�*;


total_loss���@

error_R�b2?

learning_rate_1��j7��
�I       6%�	�Mc:���A�*;


total_lossX&�@

error_R�5?

learning_rate_1��j7�tWI       6%�	2�c:���A�*;


total_loss���@

error_RҍX?

learning_rate_1��j7��S�I       6%�	Y�c:���A�*;


total_loss�k�@

error_R�QQ?

learning_rate_1��j7�zI       6%�	2d:���A�*;


total_losscc�@

error_R%E?

learning_rate_1��j7@��I       6%�	bUd:���A�*;


total_loss�2�@

error_R[8`?

learning_rate_1��j7�#:�I       6%�	�d:���A�*;


total_lossF��@

error_R�rR?

learning_rate_1��j7jJ
I       6%�	��d:���A�*;


total_lossz��@

error_R\�@?

learning_rate_1��j7y_CI       6%�	@e:���A�*;


total_loss�c�@

error_RW�H?

learning_rate_1��j7&kx�I       6%�	�We:���A�*;


total_losssjd@

error_R��B?

learning_rate_1��j7WN��I       6%�	�e:���A�*;


total_loss$��@

error_RxE`?

learning_rate_1��j7�I       6%�	�f:���A�*;


total_loss�@

error_R�fX?

learning_rate_1��j7 ��I       6%�	�df:���A�*;


total_loss8��@

error_R�W?

learning_rate_1��j7�ծ�I       6%�	ѩf:���A�*;


total_losso�@

error_R��D?

learning_rate_1��j7z.TI       6%�	��f:���A�*;


total_lossԺ�@

error_RsV?

learning_rate_1��j7��I       6%�	�=g:���A�*;


total_loss���@

error_R�ve?

learning_rate_1��j7�Ǎ�I       6%�	��g:���A�*;


total_loss�k�@

error_R�P?

learning_rate_1��j7��ARI       6%�	.�g:���A�*;


total_lossR+�@

error_R�kM?

learning_rate_1��j7�&OI       6%�	]h:���A�*;


total_lossh��@

error_R#�H?

learning_rate_1��j7�n~�I       6%�	Sh:���A�*;


total_loss.��@

error_Ri=7?

learning_rate_1��j7��kI       6%�	ښh:���A�*;


total_lossV��@

error_RZ�A?

learning_rate_1��j7�0I       6%�	�h:���A�*;


total_lossjM�@

error_RiJS?

learning_rate_1��j7��xI       6%�	$"i:���A�*;


total_loss���@

error_R�wB?

learning_rate_1��j7���I       6%�	1hi:���A�*;


total_loss&	�@

error_RCO?

learning_rate_1��j7�sHI       6%�	e�i:���A�*;


total_loss�l�@

error_R�-T?

learning_rate_1��j79�I       6%�	��i:���A�*;


total_loss���@

error_R�UN?

learning_rate_1��j7�}��I       6%�	h:j:���A�*;


total_lossh��@

error_R8AF?

learning_rate_1��j7�2�I       6%�	V~j:���A�*;


total_losst��@

error_RJX?

learning_rate_1��j7�y�}I       6%�	��j:���A�*;


total_loss�n�@

error_R�B?

learning_rate_1��j7�]��I       6%�	�k:���A�*;


total_loss�2�@

error_R��Q?

learning_rate_1��j7��II       6%�	Ck:���A�*;


total_loss���@

error_RtUU?

learning_rate_1��j7̩@�I       6%�	�k:���A�*;


total_lossf�@

error_R�d]?

learning_rate_1��j7-p|�I       6%�	��k:���A�*;


total_loss-;�@

error_R�RU?

learning_rate_1��j7(H�CI       6%�	al:���A�*;


total_loss���@

error_R1|E?

learning_rate_1��j7�3AWI       6%�	Il:���A�*;


total_loss��@

error_RܝM?

learning_rate_1��j7g匄I       6%�	��l:���A�*;


total_loss��@

error_R)�B?

learning_rate_1��j7IĸGI       6%�	�l:���A�*;


total_lossv�v@

error_R|�]?

learning_rate_1��j7�}pI       6%�	�+m:���A�*;


total_loss!_�@

error_R�te?

learning_rate_1��j7���I       6%�	vom:���A�*;


total_loss2A

error_R�U?

learning_rate_1��j7���I       6%�	S�m:���A�*;


total_lossA��@

error_R%NL?

learning_rate_1��j7����I       6%�	��m:���A�*;


total_loss��@

error_R��/?

learning_rate_1��j7� I       6%�	:4n:���A�*;


total_loss9�@

error_R�T?

learning_rate_1��j7jB��I       6%�	�tn:���A�*;


total_loss��@

error_R�\N?

learning_rate_1��j7[�I       6%�	��n:���A�*;


total_loss��@

error_R6�P?

learning_rate_1��j7��I       6%�	;�n:���A�*;


total_loss��@

error_R%�i?

learning_rate_1��j77���I       6%�	�9o:���A�*;


total_lossKډ@

error_R��P?

learning_rate_1��j7%-ۜI       6%�	|o:���A�*;


total_loss���@

error_R1 I?

learning_rate_1��j7����I       6%�	�o:���A�*;


total_loss�A�@

error_R��J?

learning_rate_1��j7�MMI       6%�	��o:���A�*;


total_lossH�(A

error_Rf!R?

learning_rate_1��j75g��I       6%�	u;p:���A�*;


total_loss� A

error_RT)V?

learning_rate_1��j7�a�!I       6%�	n~p:���A�*;


total_loss��@

error_R��H?

learning_rate_1��j7H7$�I       6%�	g�p:���A�*;


total_loss= �@

error_Rf�=?

learning_rate_1��j7�8W�I       6%�	�q:���A�*;


total_lossn6�@

error_R=-H?

learning_rate_1��j70 x�I       6%�	oCq:���A�*;


total_lossm�@

error_R*�Q?

learning_rate_1��j70��[I       6%�	��q:���A�*;


total_loss=M�@

error_RA?

learning_rate_1��j7sg�I       6%�	`�q:���A�*;


total_lossQÞ@

error_Rw9H?

learning_rate_1��j7�W��I       6%�	:r:���A�*;


total_lossdYA

error_R�8\?

learning_rate_1��j7h���I       6%�	<Fr:���A�*;


total_loss��@

error_RAbI?

learning_rate_1��j7���I       6%�	ņr:���A�*;


total_loss�B�@

error_R��^?

learning_rate_1��j7m¡I       6%�	[�r:���A�*;


total_loss�@

error_R�q^?

learning_rate_1��j7o	�I       6%�	A
s:���A�*;


total_losso>A

error_R́H?

learning_rate_1��j7����I       6%�	�Is:���A�*;


total_lossì�@

error_R]S?

learning_rate_1��j7�S�I       6%�	'�s:���A�*;


total_loss1?�@

error_Rqgc?

learning_rate_1��j7r��I       6%�	��s:���A�*;


total_lossl��@

error_R�K?

learning_rate_1��j7r'�I       6%�		t:���A�*;


total_loss/�@

error_RnmN?

learning_rate_1��j7vl|I       6%�	�Nt:���A�*;


total_loss�ϙ@

error_Rz2]?

learning_rate_1��j7�b�UI       6%�	=�t:���A�*;


total_loss��@

error_R�ZB?

learning_rate_1��j7?p&�I       6%�	p�t:���A�*;


total_lossz��@

error_R��J?

learning_rate_1��j7�n!�I       6%�	�(u:���A�*;


total_loss}]`@

error_RB?

learning_rate_1��j7�T��I       6%�	L}u:���A�*;


total_lossE�@

error_R�c]?

learning_rate_1��j7�k��I       6%�	��u:���A�*;


total_loss&��@

error_R��E?

learning_rate_1��j7q�OI       6%�	Kv:���A�*;


total_loss4��@

error_Ra,:?

learning_rate_1��j7�kFI       6%�	�^v:���A�*;


total_lossl��@

error_R�D?

learning_rate_1��j7 ?ZI       6%�	��v:���A�*;


total_loss1N�@

error_R&�Q?

learning_rate_1��j7��BI       6%�	#�v:���A�*;


total_loss{�@

error_R�R?

learning_rate_1��j7k��
I       6%�	�#w:���A�*;


total_loss��|@

error_Rj�C?

learning_rate_1��j7����I       6%�	�dw:���A�*;


total_loss�
�@

error_Rv�E?

learning_rate_1��j7%�{�I       6%�	�w:���A�*;


total_loss�2�@

error_R�G?

learning_rate_1��j7��WI       6%�	��w:���A�*;


total_lossyA�@

error_RڋL?

learning_rate_1��j7�7.�I       6%�	�2x:���A�*;


total_loss�W�@

error_RtlN?

learning_rate_1��j7L\%�I       6%�	bux:���A�*;


total_lossй�@

error_Rzr:?

learning_rate_1��j7.���I       6%�	��x:���A�*;


total_loss#x�@

error_R�cW?

learning_rate_1��j7%���I       6%�	��x:���A�*;


total_loss�c�@

error_R��L?

learning_rate_1��j7B5I       6%�	@y:���A�*;


total_loss�W�@

error_R�G?

learning_rate_1��j7؟�I       6%�	ƃy:���A�*;


total_lossvǯ@

error_R�DU?

learning_rate_1��j7�d{I       6%�	��y:���A�*;


total_loss�	A

error_R$�D?

learning_rate_1��j7�"tI       6%�	Mz:���A�*;


total_loss ��@

error_R��@?

learning_rate_1��j7�#�I       6%�	�Qz:���A�*;


total_loss��@

error_R0:?

learning_rate_1��j7���CI       6%�	�z:���A�*;


total_lossX��@

error_R�4M?

learning_rate_1��j7�٬�I       6%�	F�z:���A�*;


total_loss�s�@

error_R��B?

learning_rate_1��j7��xI       6%�	� {:���A�*;


total_loss��A

error_R`B?

learning_rate_1��j7�B�I       6%�	�h{:���A�*;


total_loss�p�@

error_R�R?

learning_rate_1��j7�B�I       6%�	y�{:���A�*;


total_loss|߬@

error_R%FE?

learning_rate_1��j78�I       6%�	A�{:���A�*;


total_loss���@

error_R|�W?

learning_rate_1��j7�`��I       6%�	!2|:���A�*;


total_loss��@

error_RR�I?

learning_rate_1��j7"��JI       6%�	bx|:���A�*;


total_loss��@

error_RҊR?

learning_rate_1��j7��[�I       6%�	z�|:���A�*;


total_loss�2�@

error_Rs?N?

learning_rate_1��j7I���I       6%�	�/}:���A�*;


total_lossn��@

error_RJnd?

learning_rate_1��j7K��I       6%�	�t}:���A�*;


total_lossL��@

error_R��I?

learning_rate_1��j7߯��I       6%�	��}:���A�*;


total_loss0�@

error_R�0C?

learning_rate_1��j7��ZI       6%�	Q.~:���A�*;


total_lossV��@

error_Rw�M?

learning_rate_1��j78�nxI       6%�	�x~:���A�*;


total_lossm6�@

error_R�DG?

learning_rate_1��j7��k�I       6%�	��~:���A�*;


total_loss���@

error_R �J?

learning_rate_1��j7h��	I       6%�	-:���A�*;


total_loss��@

error_R��H?

learning_rate_1��j7D}�qI       6%�	0v:���A�*;


total_lossj{�@

error_R��[?

learning_rate_1��j7DI       6%�	��:���A�*;


total_loss_�@

error_Rs`\?

learning_rate_1��j7����I       6%�	
�:���A�*;


total_loss�U�@

error_R�@6?

learning_rate_1��j7@eG�I       6%�	yq�:���A�*;


total_loss�P�@

error_RO?

learning_rate_1��j7�R@kI       6%�	M��:���A�*;


total_losslmw@

error_RL�X?

learning_rate_1��j7���cI       6%�	���:���A�*;


total_lossU��@

error_R�|H?

learning_rate_1��j7F��I       6%�	�g�:���A�*;


total_lossn��@

error_R�D?

learning_rate_1��j7z)I       6%�	���:���A�*;


total_loss@+�@

error_R[�^?

learning_rate_1��j76��qI       6%�	��:���A�*;


total_loss���@

error_RʀK?

learning_rate_1��j7_�I       6%�	�a�:���A�*;


total_losswɕ@

error_RDL?

learning_rate_1��j7a�I       6%�	S��:���A�*;


total_lossO�@

error_R�|??

learning_rate_1��j7���NI       6%�	���:���A�*;


total_loss]�@

error_Ro%`?

learning_rate_1��j7I~�I       6%�	vB�:���A�*;


total_loss��@

error_Rv�F?

learning_rate_1��j7�F�cI       6%�	���:���A�*;


total_lossv��@

error_R�N?

learning_rate_1��j7l߁uI       6%�	!��:���A�*;


total_lossS6�@

error_R]�f?

learning_rate_1��j7ś�qI       6%�	�?�:���A�*;


total_losse��@

error_RlN?

learning_rate_1��j7���I       6%�	���:���A�*;


total_lossv�@

error_R��@?

learning_rate_1��j7�wkI       6%�	_�:���A�*;


total_loss��@

error_R�]H?

learning_rate_1��j7���I       6%�	�:�:���A�*;


total_loss��@

error_R�{I?

learning_rate_1��j7���I       6%�	�˅:���A�*;


total_loss��@

error_R&O?

learning_rate_1��j7[���I       6%�	��:���A�*;


total_loss���@

error_RsQ?

learning_rate_1��j7�S��I       6%�	 |�:���A�*;


total_loss`��@

error_R��P?

learning_rate_1��j7���I       6%�	[Ԇ:���A�*;


total_lossg�@

error_R�J\?

learning_rate_1��j7G��)I       6%�	-�:���A�*;


total_loss]��@

error_R�;?

learning_rate_1��j75G��I       6%�	�^�:���A�*;


total_lossJ4�@

error_R�?O?

learning_rate_1��j7�I       6%�	·:���A�*;


total_loss%Ν@

error_R�U?

learning_rate_1��j7s�I       6%�	��:���A�*;


total_lossV��@

error_R��E?

learning_rate_1��j7��ַI       6%�	�T�:���A�*;


total_loss�ф@

error_R�BW?

learning_rate_1��j7QU=�I       6%�	
��:���A�*;


total_loss ��@

error_R�E?

learning_rate_1��j7���,I       6%�	r�:���A�*;


total_loss�׆@

error_RڸE?

learning_rate_1��j7�N�I       6%�	�G�:���A�*;


total_loss�1�@

error_R�.N?

learning_rate_1��j7���5I       6%�	��:���A�*;


total_loss@g�@

error_R�eU?

learning_rate_1��j7��I       6%�	��:���A�*;


total_loss�)�@

error_R��??

learning_rate_1��j7LPU�I       6%�	�;�:���A�*;


total_loss�}@

error_R�fN?

learning_rate_1��j7[�2�I       6%�	"��:���A�*;


total_lossI� A

error_R\�C?

learning_rate_1��j7!+��I       6%�	�Ǌ:���A�*;


total_loss���@

error_R��T?

learning_rate_1��j7h���I       6%�	0�:���A�*;


total_loss��@

error_R=�R?

learning_rate_1��j7E���I       6%�	)��:���A�*;


total_loss��@

error_R �@?

learning_rate_1��j73Wn�I       6%�	�ɋ:���A�*;


total_loss�s@

error_RXH?

learning_rate_1��j7�ʻI       6%�	��:���A�*;


total_loss$��@

error_R��^?

learning_rate_1��j7��)YI       6%�	ԏ�:���A�*;


total_loss�n�@

error_R��K?

learning_rate_1��j7�ƴ�I       6%�	9Ԍ:���A�*;


total_loss�@

error_RS�=?

learning_rate_1��j7���'I       6%�	��:���A�*;


total_lossl�@

error_R�YB?

learning_rate_1��j7ġu�I       6%�	���:���A�*;


total_loss�Ay@

error_R`R?

learning_rate_1��j7���I       6%�	ԍ:���A�*;


total_loss<��@

error_R��L?

learning_rate_1��j7# I       6%�	�)�:���A�*;


total_loss@

error_R�C?

learning_rate_1��j7��%I       6%�	G��:���A�*;


total_loss	r�@

error_R;�K?

learning_rate_1��j7IH�SI       6%�	���:���A�*;


total_loss-U@

error_R�jF?

learning_rate_1��j7`��I       6%�	�:�:���A�*;


total_loss�r�@

error_Ra�O?

learning_rate_1��j7��>�I       6%�	Ϫ�:���A�*;


total_loss	�@

error_RR�??

learning_rate_1��j7��iI       6%�	��:���A�*;


total_loss�j�@

error_R��G?

learning_rate_1��j7�I�I       6%�	�<�:���A�*;


total_losse�@

error_R�Ub?

learning_rate_1��j7<��I       6%�	��:���A�*;


total_loss4$�@

error_RfT[?

learning_rate_1��j77?$I       6%�	��:���A�*;


total_loss!5�@

error_R��C?

learning_rate_1��j7y�kI       6%�	66�:���A�*;


total_lossT_�@

error_R�U?

learning_rate_1��j7�X�gI       6%�	*��:���A�*;


total_loss���@

error_R��T?

learning_rate_1��j7���5I       6%�	���:���A�*;


total_lossH�@

error_R&�Q?

learning_rate_1��j7Z�{�I       6%�	b8�:���A�*;


total_lossn�{@

error_R��I?

learning_rate_1��j7����I       6%�	�z�:���A�*;


total_loss89�@

error_R׍R?

learning_rate_1��j7O�`�I       6%�	��:���A�*;


total_lossz5�@

error_R1�Q?

learning_rate_1��j7B�	@I       6%�	�2�:���A�*;


total_loss���@

error_R�L?

learning_rate_1��j7B<��I       6%�	�v�:���A�*;


total_lossz�@

error_R�fE?

learning_rate_1��j7�F��I       6%�	��:���A�*;


total_loss)s�@

error_RK?

learning_rate_1��j7
"�RI       6%�	�1�:���A�*;


total_loss�5_@

error_R�A?

learning_rate_1��j7%�I�I       6%�	<z�:���A�*;


total_lossRx�@

error_RJ'X?

learning_rate_1��j7x���I       6%�	Ӕ:���A�*;


total_losseb�@

error_Ri�T?

learning_rate_1��j76&u#I       6%�	8�:���A�*;


total_lossE8�@

error_RH�R?

learning_rate_1��j7N+�nI       6%�	
��:���A�*;


total_loss|,�@

error_RrN?

learning_rate_1��j7(�I       6%�	b�:���A�*;


total_loss��@

error_RwmU?

learning_rate_1��j7y�ԸI       6%�	�n�:���A�*;


total_loss�Zk@

error_R �L?

learning_rate_1��j7o7SBI       6%�	 ��:���A�*;


total_loss��A

error_RaS?

learning_rate_1��j7iZ�`I       6%�	� �:���A�*;


total_lossWD�@

error_R�@E?

learning_rate_1��j7��8I       6%�	rh�:���A�*;


total_loss�D�@

error_R<R?

learning_rate_1��j7�rg�I       6%�	櫗:���A�*;


total_loss�(�@

error_RO?

learning_rate_1��j7(��I       6%�	��:���A�*;


total_lossI�@

error_R�.D?

learning_rate_1��j7��2I       6%�	�d�:���A�*;


total_lossj�@

error_R�sB?

learning_rate_1��j7�I       6%�	Ŧ�:���A�*;


total_loss���@

error_RQK?

learning_rate_1��j7@���I       6%�	�:���A�*;


total_lossh�@

error_R�;S?

learning_rate_1��j7�)��I       6%�		b�:���A�*;


total_loss���@

error_R��C?

learning_rate_1��j7��H�I       6%�	٣�:���A�*;


total_lossR��@

error_R*�U?

learning_rate_1��j7�y��I       6%�	}�:���A�*;


total_lossB�@

error_RDHe?

learning_rate_1��j7��)I       6%�	(�:���A�*;


total_loss���@

error_RԢU?

learning_rate_1��j7W���I       6%�	�r�:���A�*;


total_lossCA

error_R%I?

learning_rate_1��j7�_\I       6%�	,ޚ:���A�*;


total_loss���@

error_R�A?

learning_rate_1��j7�_�I       6%�	Z&�:���A�*;


total_loss�*�@

error_R��R?

learning_rate_1��j7]�D�I       6%�	&r�:���A�*;


total_loss��@

error_R��Q?

learning_rate_1��j7�wW�I       6%�	�ڛ:���A�*;


total_loss���@

error_RV?

learning_rate_1��j7��:I       6%�	h&�:���A�*;


total_lossf6�@

error_R�L?

learning_rate_1��j7�r��I       6%�	p�:���A�*;


total_loss�)�@

error_R��G?

learning_rate_1��j73�	�I       6%�	
�:���A�*;


total_loss|$�@

error_Rc
>?

learning_rate_1��j78�hSI       6%�	w2�:���A�*;


total_loss��@

error_R�I?

learning_rate_1��j7�I�I       6%�	�x�:���A�*;


total_loss�r�@

error_R
o@?

learning_rate_1��j7�8,WI       6%�	�ߝ:���A�*;


total_loss?A

error_R�sT?

learning_rate_1��j7�L�HI       6%�	e5�:���A�*;


total_loss-s�@

error_R�^V?

learning_rate_1��j7�Eh�I       6%�	�x�:���A�*;


total_loss�ҡ@

error_R�S?

learning_rate_1��j7e�BI       6%�	��:���A�*;


total_loss)$�@

error_R�M?

learning_rate_1��j7R/8vI       6%�	��:���A�*;


total_loss��@

error_R�!R?

learning_rate_1��j7��I       6%�	?K�:���A�*;


total_loss�iu@

error_Rΐ]?

learning_rate_1��j7wy��I       6%�	���:���A�*;


total_loss�mA

error_RC�L?

learning_rate_1��j7���)I       6%�	�ޟ:���A�*;


total_loss���@

error_R��7?

learning_rate_1��j7���<I       6%�	�"�:���A�*;


total_loss���@

error_R��@?

learning_rate_1��j7@�k�I       6%�	9i�:���A�*;


total_loss���@

error_RJ W?

learning_rate_1��j7��gI       6%�	f��:���A�*;


total_loss���@

error_R�oA?

learning_rate_1��j7b��I       6%�	���:���A�*;


total_lossd�d@

error_Rrq?

learning_rate_1��j7� (�I       6%�	F=�:���A�*;


total_loss���@

error_R�Lf?

learning_rate_1��j7�G�I       6%�	���:���A�*;


total_loss���@

error_R�}b?

learning_rate_1��j7(*4�I       6%�	á:���A�*;


total_loss� �@

error_Rx�H?

learning_rate_1��j7W���I       6%�	��:���A�*;


total_losstо@

error_R-A?

learning_rate_1��j7ׄ��I       6%�	UM�:���A�*;


total_loss�FA

error_R��W?

learning_rate_1��j7Ӷ��I       6%�	2��:���A�*;


total_loss聮@

error_R��_?

learning_rate_1��j7Ǝ��I       6%�	\Ԣ:���A�*;


total_loss��@

error_R�hH?

learning_rate_1��j7/f��I       6%�	��:���A�*;


total_loss4�@

error_RCoP?

learning_rate_1��j7���lI       6%�	�\�:���A�*;


total_loss�ѭ@

error_Rxk]?

learning_rate_1��j7�?XI       6%�	a��:���A�*;


total_loss��@

error_Ri.R?

learning_rate_1��j7S:ƑI       6%�	n�:���A�*;


total_loss��@

error_R �R?

learning_rate_1��j7<(dI       6%�	�/�:���A�*;


total_loss�U�@

error_Rm�K?

learning_rate_1��j7����I       6%�	�r�:���A�*;


total_loss*ϕ@

error_Rע`?

learning_rate_1��j7�5�nI       6%�	���:���A�*;


total_loss��@

error_RsKN?

learning_rate_1��j7��3I       6%�	��:���A�*;


total_loss�@�@

error_R�p@?

learning_rate_1��j7���I       6%�	�@�:���A�*;


total_loss�l�@

error_R�_P?

learning_rate_1��j7����I       6%�	:���A�*;


total_loss_^�@

error_Rq�Y?

learning_rate_1��j7��d�I       6%�	�ը:���A�*;


total_loss;��@

error_R.XJ?

learning_rate_1��j7\A�`I       6%�	��:���A�*;


total_lossx��@

error_Rs�>?

learning_rate_1��j7����I       6%�	�]�:���A�*;


total_loss-��@

error_R�|R?

learning_rate_1��j7���I       6%�	4��:���A�*;


total_lossz��@

error_R� i?

learning_rate_1��j7۽JI       6%�	�:���A�*;


total_losso"�@

error_R
�G?

learning_rate_1��j7w,��I       6%�	�)�:���A�*;


total_loss�ݟ@

error_RqoQ?

learning_rate_1��j7h1�$I       6%�	�m�:���A�*;


total_loss;�@

error_R3�R?

learning_rate_1��j7�\0.I       6%�	0��:���A�*;


total_lossX��@

error_Rj�J?

learning_rate_1��j7� ;�I       6%�	.��:���A�*;


total_loss�{�@

error_RPQ?

learning_rate_1��j7�ź�I       6%�	_9�:���A�*;


total_loss�^A

error_R�V??

learning_rate_1��j7:�YI       6%�	�z�:���A�*;


total_loss�@

error_R��T?

learning_rate_1��j7�Į�I       6%�	ܻ�:���A�*;


total_lossÞ�@

error_R��O?

learning_rate_1��j7cj�@I       6%�	;�:���A�*;


total_loss j�@

error_R�&P?

learning_rate_1��j7�ԫII       6%�	>I�:���A�*;


total_loss;��@

error_R�%\?

learning_rate_1��j7����I       6%�	~��:���A�*;


total_loss�[�@

error_R!�W?

learning_rate_1��j7bեI       6%�	�׬:���A�*;


total_loss���@

error_R�`W?

learning_rate_1��j7z��XI       6%�	��:���A�*;


total_lossh(r@

error_R�L?

learning_rate_1��j73���I       6%�	rg�:���A�*;


total_loss��@

error_R{V:?

learning_rate_1��j7���I       6%�	���:���A�*;


total_loss�J�@

error_RHc?

learning_rate_1��j7��TcI       6%�	���:���A�*;


total_loss�ɠ@

error_R
P?

learning_rate_1��j7z7j�I       6%�	@9�:���A�*;


total_loss2d�@

error_R �H?

learning_rate_1��j7xmp	I       6%�	g{�:���A�*;


total_lossDc�@

error_Rf�T?

learning_rate_1��j7kp��I       6%�	_��:���A�*;


total_loss���@

error_R�wZ?

learning_rate_1��j7�>��I       6%�	� �:���A�*;


total_loss�&}@

error_Rv�O?

learning_rate_1��j7a�I       6%�	�B�:���A�*;


total_lossA

error_R�W?

learning_rate_1��j7�wI       6%�	���:���A�*;


total_loss�J�@

error_Rv"_?

learning_rate_1��j7��&I       6%�	�ɯ:���A�*;


total_lossɡ�@

error_RWO?

learning_rate_1��j7�KBI       6%�	b�:���A�*;


total_lossd<�@

error_R��`?

learning_rate_1��j7���I       6%�	�T�:���A�*;


total_loss�4A

error_RwW[?

learning_rate_1��j7y���I       6%�	A��:���A�*;


total_loss���@

error_R�@?

learning_rate_1��j7���I       6%�	uܰ:���A�*;


total_loss��@

error_R{�B?

learning_rate_1��j7F��iI       6%�		!�:���A�*;


total_loss��@

error_R �O?

learning_rate_1��j7�^�oI       6%�	�g�:���A�*;


total_loss��@

error_R$x>?

learning_rate_1��j7�;��I       6%�	孱:���A�*;


total_loss���@

error_R��[?

learning_rate_1��j7���I       6%�	��:���A�*;


total_loss��@

error_RȠI?

learning_rate_1��j7�e�I       6%�	�7�:���A�*;


total_loss���@

error_R�K?

learning_rate_1��j7��4rI       6%�	偲:���A�*;


total_lossؿ�@

error_R��=?

learning_rate_1��j7�	)<I       6%�	��:���A�*;


total_lossL��@

error_Rc�G?

learning_rate_1��j7��^�I       6%�	�D�:���A�*;


total_loss�@

error_RO`H?

learning_rate_1��j7�ӏ�I       6%�	��:���A�*;


total_loss��w@

error_R��N?

learning_rate_1��j7����I       6%�	��:���A�*;


total_loss�	�@

error_R�I?

learning_rate_1��j7�.�FI       6%�	�Z�:���A�*;


total_loss��@

error_R�]?

learning_rate_1��j7W��YI       6%�	E��:���A�*;


total_lossEǦ@

error_RͻA?

learning_rate_1��j7�âdI       6%�	��:���A�*;


total_lossj��@

error_R@�8?

learning_rate_1��j7�lΓI       6%�	z3�:���A�*;


total_loss<��@

error_RTWA?

learning_rate_1��j7�X�kI       6%�	��:���A�*;


total_loss�%u@

error_R�N?

learning_rate_1��j7�QI       6%�	C�:���A�*;


total_loss/@�@

error_R�[?

learning_rate_1��j7��I       6%�	�\�:���A�*;


total_loss�*�@

error_R
GI?

learning_rate_1��j7l���I       6%�	���:���A�*;


total_loss�u�@

error_R�L?

learning_rate_1��j7Q��nI       6%�	���:���A�*;


total_lossAs@

error_R��e?

learning_rate_1��j7�y)�I       6%�	�{�:���A�*;


total_loss,�@

error_R�uc?

learning_rate_1��j7@H-qI       6%�	�Ʒ:���A�*;


total_lossċ�@

error_R_2K?

learning_rate_1��j7Et��I       6%�	N�:���A�*;


total_lossm�@

error_Ri�9?

learning_rate_1��j7��y#I       6%�	x�:���A�*;


total_loss�|@

error_R� M?

learning_rate_1��j7%�>I       6%�	Oϸ:���A�*;


total_loss}|A

error_R1�Q?

learning_rate_1��j7h���I       6%�	�:���A�*;


total_loss
P*A

error_R,�Q?

learning_rate_1��j7g���I       6%�	t��:���A�*;


total_loss���@

error_R�I?

learning_rate_1��j7
 ��I       6%�	ι:���A�*;


total_loss�n@

error_R�|X?

learning_rate_1��j7���JI       6%�	2�:���A�*;


total_loss�-�@

error_R�;;?

learning_rate_1��j7�$�eI       6%�	L��:���A�*;


total_loss��@

error_RZ{N?

learning_rate_1��j7͞��I       6%�	�ں:���A�*;


total_loss���@

error_R�4c?

learning_rate_1��j7c�4qI       6%�	�#�:���A�*;


total_loss<ڽ@

error_R@RC?

learning_rate_1��j7Jٜ3I       6%�	���:���A�*;


total_loss���@

error_R_�R?

learning_rate_1��j7+�I       6%�	��:���A�*;


total_loss��@

error_R��A?

learning_rate_1��j7��X�I       6%�	�-�:���A�*;


total_loss���@

error_R�W?

learning_rate_1��j7A���I       6%�	���:���A�*;


total_loss��@

error_R�/R?

learning_rate_1��j7���gI       6%�	��:���A�*;


total_loss���@

error_R84Q?

learning_rate_1��j7`#0I       6%�	�2�:���A�*;


total_lossZŅ@

error_R�sH?

learning_rate_1��j7�ůI       6%�	y��:���A�*;


total_losso�9A

error_RH^F?

learning_rate_1��j7�ʡI       6%�	�:���A�*;


total_lossvڨ@

error_RA}O?

learning_rate_1��j7��&I       6%�	�2�:���A�*;


total_loss�%�@

error_RRDH?

learning_rate_1��j7*�T
I       6%�	�{�:���A�*;


total_loss���@

error_R��P?

learning_rate_1��j7�ڞI       6%�	k�:���A�*;


total_loss[�@

error_R�BD?

learning_rate_1��j7̦ I       6%�	m2�:���A�*;


total_loss�1�@

error_R�&S?

learning_rate_1��j7�:��I       6%�	W}�:���A�*;


total_loss*<�@

error_R�7?

learning_rate_1��j7����I       6%�	���:���A�*;


total_lossR�@

error_Rs�[?

learning_rate_1��j7t;�UI       6%�	�F�:���A�*;


total_loss�¢@

error_Rj�V?

learning_rate_1��j7�f2I       6%�	7��:���A�*;


total_loss ��@

error_R��X?

learning_rate_1��j7֗��I       6%�	��:���A�*;


total_loss6�@

error_R��T?

learning_rate_1��j7zW�hI       6%�	tQ�:���A�*;


total_loss�&�@

error_R�H?

learning_rate_1��j7���I       6%�	ӗ�:���A�*;


total_loss��@

error_RT>E?

learning_rate_1��j7�n9�I       6%�	���:���A�*;


total_lossӅ�@

error_R�Q?

learning_rate_1��j7UW16I       6%�	NG�:���A�*;


total_loss�@

error_R74L?

learning_rate_1��j7���I       6%�	E��:���A�*;


total_lossA

error_RvC?

learning_rate_1��j7q���I       6%�	���:���A�*;


total_lossa�@

error_R�-N?

learning_rate_1��j7���I       6%�	�:���A�*;


total_loss�\�@

error_R�nS?

learning_rate_1��j7'��I       6%�	�`�:���A�*;


total_loss=�@

error_Rd7_?

learning_rate_1��j7%��lI       6%�	j��:���A�*;


total_lossDN�@

error_Rq Z?

learning_rate_1��j7)�uI       6%�	��:���A�*;


total_losseŲ@

error_R�DZ?

learning_rate_1��j7��WKI       6%�	�2�:���A�*;


total_loss'	A

error_Rv�R?

learning_rate_1��j7�C{I       6%�	�y�:���A�*;


total_loss��@

error_R��??

learning_rate_1��j7\���I       6%�		��:���A�*;


total_loss�3�@

error_R�IE?

learning_rate_1��j7�@�6I       6%�	��:���A� *;


total_lossa�@

error_R�@V?

learning_rate_1��j7�x|TI       6%�	�K�:���A� *;


total_loss�R�@

error_Rl?

learning_rate_1��j7�/1GI       6%�	��:���A� *;


total_loss;�5A

error_R�`N?

learning_rate_1��j7|+�I       6%�	���:���A� *;


total_loss���@

error_R�U?

learning_rate_1��j7��I       6%�	'A�:���A� *;


total_loss�@�@

error_R��S?

learning_rate_1��j7�j_KI       6%�	���:���A� *;


total_lossj5�@

error_RidC?

learning_rate_1��j7	�rOI       6%�	���:���A� *;


total_lossJw�@

error_R�zY?

learning_rate_1��j7G4CI       6%�	��:���A� *;


total_loss]�@

error_R�^T?

learning_rate_1��j7��x�I       6%�	�V�:���A� *;


total_lossC��@

error_R
S?

learning_rate_1��j7��iI       6%�	���:���A� *;


total_loss%�A

error_Rq:`?

learning_rate_1��j7|��I       6%�	���:���A� *;


total_loss�e�@

error_RoX?

learning_rate_1��j7��+PI       6%�	A�:���A� *;


total_loss��@

error_Rj�T?

learning_rate_1��j7J@�I       6%�	�b�:���A� *;


total_loss���@

error_R�S?

learning_rate_1��j7v�݄I       6%�	���:���A� *;


total_loss�ۮ@

error_R�5R?

learning_rate_1��j7<^�I       6%�	C��:���A� *;


total_loss�#�@

error_R]VV?

learning_rate_1��j7J�EfI       6%�	 /�:���A� *;


total_loss�`�@

error_R�H?

learning_rate_1��j7WȦ I       6%�	�w�:���A� *;


total_loss}��@

error_R��I?

learning_rate_1��j7��tII       6%�	���:���A� *;


total_loss��{@

error_Rv=E?

learning_rate_1��j7���I       6%�	h��:���A� *;


total_loss�4�@

error_R�J?

learning_rate_1��j7�L	I       6%�	B�:���A� *;


total_loss�T�@

error_RC�N?

learning_rate_1��j7��m:I       6%�	��:���A� *;


total_lossX��@

error_R&�M?

learning_rate_1��j7R{B�I       6%�	���:���A� *;


total_loss� �@

error_R63N?

learning_rate_1��j72��6I       6%�	��:���A� *;


total_loss8#�@

error_RX?

learning_rate_1��j7g�/$I       6%�	JW�:���A� *;


total_loss�)A

error_RȼZ?

learning_rate_1��j7���GI       6%�	`��:���A� *;


total_loss�r�@

error_R$=?

learning_rate_1��j7D�I       6%�	N��:���A� *;


total_lossyy�@

error_Rq�1?

learning_rate_1��j7�輶I       6%�	�/�:���A� *;


total_loss���@

error_RӍW?

learning_rate_1��j7�&I       6%�	�r�:���A� *;


total_loss��@

error_RZ�Q?

learning_rate_1��j7GAKI       6%�	��:���A� *;


total_losso°@

error_R@�E?

learning_rate_1��j7�� I       6%�	���:���A� *;


total_lossO��@

error_R�ZB?

learning_rate_1��j7|�I       6%�	�C�:���A� *;


total_loss
�A

error_R}�C?

learning_rate_1��j75���I       6%�	}��:���A� *;


total_loss<�@

error_R}�a?

learning_rate_1��j7��cI       6%�	��:���A� *;


total_loss�ǭ@

error_Rs}b?

learning_rate_1��j7+��I       6%�	��:���A� *;


total_loss�ݩ@

error_R�/H?

learning_rate_1��j7^k9�I       6%�	�S�:���A� *;


total_loss���@

error_R�A?

learning_rate_1��j7���I       6%�	v��:���A� *;


total_loss�s�@

error_R��J?

learning_rate_1��j7��I       6%�	v��:���A� *;


total_loss���@

error_R$�V?

learning_rate_1��j7���I       6%�	$�:���A� *;


total_lossQ�@

error_R�T?

learning_rate_1��j7�nI�I       6%�	g�:���A� *;


total_lossj6�@

error_R/�T?

learning_rate_1��j7�t`yI       6%�	���:���A� *;


total_loss</�@

error_Rq�c?

learning_rate_1��j7N�I       6%�	���:���A� *;


total_loss�-�@

error_R�]V?

learning_rate_1��j7J��I       6%�	�.�:���A� *;


total_loss���@

error_R% f?

learning_rate_1��j7g��BI       6%�	�q�:���A� *;


total_loss�r@

error_RX?

learning_rate_1��j7X�ܫI       6%�	Ӵ�:���A� *;


total_loss���@

error_R�K?

learning_rate_1��j7seqiI       6%�	q��:���A� *;


total_loss�}�@

error_R;�C?

learning_rate_1��j7T���I       6%�	~:�:���A� *;


total_loss&	�@

error_R��E?

learning_rate_1��j7e�oQI       6%�	�~�:���A� *;


total_loss��@

error_R��Q?

learning_rate_1��j7�m�?I       6%�	���:���A� *;


total_loss���@

error_RR_O?

learning_rate_1��j73��sI       6%�	5�:���A� *;


total_loss��@

error_Rl*@?

learning_rate_1��j7��i�I       6%�	5V�:���A� *;


total_loss�#�@

error_R&�@?

learning_rate_1��j7��I       6%�	G��:���A� *;


total_loss�k@

error_R��S?

learning_rate_1��j7
���I       6%�	���:���A� *;


total_lossQh�@

error_R\J?

learning_rate_1��j7��tI       6%�	m*�:���A� *;


total_loss��@

error_R�Q?

learning_rate_1��j7p�j�I       6%�	at�:���A� *;


total_loss���@

error_RiYI?

learning_rate_1��j7��N�I       6%�	D��:���A� *;


total_lossR?�@

error_R��G?

learning_rate_1��j7پ��I       6%�		��:���A� *;


total_loss���@

error_R8�I?

learning_rate_1��j7�{�JI       6%�	yA�:���A� *;


total_loss�@

error_R��H?

learning_rate_1��j7��TBI       6%�	a��:���A� *;


total_loss�U�@

error_R�;I?

learning_rate_1��j7���LI       6%�	��:���A� *;


total_lossWx�@

error_R�N?

learning_rate_1��j78	�>I       6%�	��:���A� *;


total_loss�g�@

error_Re�S?

learning_rate_1��j7���fI       6%�	nS�:���A� *;


total_loss�0�@

error_R��M?

learning_rate_1��j7���VI       6%�	���:���A� *;


total_loss_��@

error_R.<^?

learning_rate_1��j7�gr�I       6%�	�"�:���A� *;


total_loss}��@

error_RIfI?

learning_rate_1��j7I���I       6%�	�l�:���A� *;


total_loss(��@

error_Rc�D?

learning_rate_1��j7%jI       6%�	1��:���A� *;


total_loss�o@

error_R�|]?

learning_rate_1��j7)1�;I       6%�	y��:���A� *;


total_loss���@

error_R �F?

learning_rate_1��j7���I       6%�	�<�:���A� *;


total_lossA�@

error_RZ#O?

learning_rate_1��j7��zI       6%�	<��:���A� *;


total_lossƛ@

error_R��P?

learning_rate_1��j7��fI       6%�	���:���A� *;


total_loss ��@

error_R7D?

learning_rate_1��j7xF�*I       6%�	��:���A� *;


total_loss3��@

error_R�N?

learning_rate_1��j7���I       6%�	FQ�:���A� *;


total_lossa��@

error_R�L?

learning_rate_1��j7@CۈI       6%�	!��:���A� *;


total_loss�̻@

error_R�$]?

learning_rate_1��j7��7I       6%�	T��:���A� *;


total_loss��@

error_R<�M?

learning_rate_1��j7U28�I       6%�	} �:���A� *;


total_lossi��@

error_Rd�D?

learning_rate_1��j7ϯI       6%�	�e�:���A� *;


total_lossl��@

error_R��Q?

learning_rate_1��j7k�+�I       6%�	Ы�:���A� *;


total_losse�@

error_R }E?

learning_rate_1��j7���6I       6%�	���:���A� *;


total_loss��@

error_R
�O?

learning_rate_1��j7�߁I       6%�	~2�:���A� *;


total_loss��@

error_R:i?

learning_rate_1��j7wΫI       6%�	�{�:���A� *;


total_loss���@

error_R��7?

learning_rate_1��j7(9��I       6%�	��:���A� *;


total_loss\��@

error_Rv6=?

learning_rate_1��j7���I       6%�	��:���A� *;


total_loss1A�@

error_R*�W?

learning_rate_1��j7��sI       6%�	@N�:���A� *;


total_loss؈�@

error_R�A?

learning_rate_1��j7�0�I       6%�	~��:���A� *;


total_loss�;�@

error_R��Q?

learning_rate_1��j7��I       6%�	���:���A� *;


total_loss�u�@

error_R/S?

learning_rate_1��j7��zKI       6%�	.�:���A� *;


total_loss���@

error_RZ�>?

learning_rate_1��j7�y�gI       6%�	�]�:���A� *;


total_lossCVo@

error_R��G?

learning_rate_1��j7� �I       6%�	��:���A� *;


total_loss��@

error_Rq�e?

learning_rate_1��j7ᐊ�I       6%�	x��:���A� *;


total_loss]C�@

error_R.<?

learning_rate_1��j7�3�'I       6%�	�0�:���A� *;


total_lossҪ@

error_R�
P?

learning_rate_1��j7��I       6%�	�}�:���A� *;


total_loss�Ũ@

error_R3KS?

learning_rate_1��j7�C*_I       6%�	Y��:���A� *;


total_loss��A

error_R�eD?

learning_rate_1��j7οgI       6%�	O�:���A� *;


total_loss,\�@

error_RLd?

learning_rate_1��j7�K��I       6%�	�a�:���A� *;


total_loss���@

error_R�6Y?

learning_rate_1��j7��lI       6%�	���:���A� *;


total_loss���@

error_R�QX?

learning_rate_1��j7:<I       6%�	���:���A� *;


total_loss�G�@

error_R��[?

learning_rate_1��j77 %yI       6%�	76�:���A� *;


total_loss�ň@

error_R��3?

learning_rate_1��j7���I       6%�	�w�:���A� *;


total_lossi5�@

error_R��K?

learning_rate_1��j7W���I       6%�	��:���A� *;


total_lossͮ�@

error_RY?

learning_rate_1��j7�x��I       6%�	��:���A� *;


total_loss��@

error_R��O?

learning_rate_1��j7n�R�I       6%�	�B�:���A� *;


total_loss�x�@

error_R��S?

learning_rate_1��j7�6
dI       6%�	��:���A� *;


total_lossr�@

error_R��H?

learning_rate_1��j7��MLI       6%�	���:���A� *;


total_loss��@

error_R�7?

learning_rate_1��j7� I       6%�	hK�:���A� *;


total_loss�y�@

error_RT\^?

learning_rate_1��j7h�#�I       6%�	Տ�:���A� *;


total_losscA

error_R��^?

learning_rate_1��j7%V��I       6%�	|��:���A� *;


total_loss�	�@

error_R�M?

learning_rate_1��j7d�WI       6%�	��:���A� *;


total_loss�D�@

error_RTh??

learning_rate_1��j7fɟ�I       6%�	a�:���A� *;


total_loss>��@

error_R�YM?

learning_rate_1��j7��I       6%�	`��:���A� *;


total_lossH��@

error_R��W?

learning_rate_1��j7�^c@I       6%�	���:���A� *;


total_lossi��@

error_R{�I?

learning_rate_1��j7W�\�I       6%�	o/�:���A� *;


total_losso*�@

error_R�]8?

learning_rate_1��j7iMs�I       6%�	�v�:���A� *;


total_loss�)�@

error_R��L?

learning_rate_1��j7%�-3I       6%�	Ծ�:���A� *;


total_loss'�@

error_R��M?

learning_rate_1��j7V�b�I       6%�	��:���A� *;


total_loss
��@

error_R�M]?

learning_rate_1��j7�8�I       6%�	�H�:���A� *;


total_loss���@

error_R��I?

learning_rate_1��j7F�I       6%�	H��:���A� *;


total_loss4"�@

error_R�+U?

learning_rate_1��j7�m�I       6%�	���:���A� *;


total_loss���@

error_R��H?

learning_rate_1��j7��gI       6%�	'(�:���A� *;


total_loss��@

error_R|�R?

learning_rate_1��j7NN�%I       6%�	�j�:���A� *;


total_loss2d�@

error_R�4U?

learning_rate_1��j7��_zI       6%�	T��:���A� *;


total_lossM@A

error_R�5X?

learning_rate_1��j7"7�I       6%�	)�:���A� *;


total_loss���@

error_R�L?

learning_rate_1��j7�)9�I       6%�	#c�:���A� *;


total_loss4K[@

error_R��H?

learning_rate_1��j7�E�I       6%�	N��:���A� *;


total_loss�C|@

error_R�vD?

learning_rate_1��j70�.�I       6%�	���:���A� *;


total_loss���@

error_Rfoa?

learning_rate_1��j7��hjI       6%�	�9�:���A� *;


total_losss��@

error_R�U?

learning_rate_1��j7�: BI       6%�	���:���A� *;


total_loss���@

error_R�J?

learning_rate_1��j7V���I       6%�	���:���A� *;


total_loss���@

error_R@9Y?

learning_rate_1��j7�V|�I       6%�	��:���A� *;


total_loss��A

error_R�!T?

learning_rate_1��j7Nͮ6I       6%�	�a�:���A� *;


total_lossܺ�@

error_R7E?

learning_rate_1��j7��HI       6%�	)��:���A�!*;


total_loss	��@

error_R�P?

learning_rate_1��j7є�I       6%�	|��:���A�!*;


total_loss���@

error_Rv2D?

learning_rate_1��j74��I       6%�	i2�:���A�!*;


total_loss	�z@

error_RlS?

learning_rate_1��j7E$��I       6%�	�u�:���A�!*;


total_loss�@

error_R��W?

learning_rate_1��j7j��.I       6%�	���:���A�!*;


total_lossAK�@

error_R.�J?

learning_rate_1��j7�E�&I       6%�	��:���A�!*;


total_loss��@

error_R�n=?

learning_rate_1��j7��XI       6%�	�D�:���A�!*;


total_loss��@

error_R3EC?

learning_rate_1��j7OaS�I       6%�	��:���A�!*;


total_loss��@

error_R4VG?

learning_rate_1��j7�&I       6%�	���:���A�!*;


total_loss���@

error_R��G?

learning_rate_1��j7W�,�I       6%�	 �:���A�!*;


total_loss>�@

error_R��P?

learning_rate_1��j7�@I       6%�	�^�:���A�!*;


total_lossEG�@

error_R�?N?

learning_rate_1��j7c���I       6%�	��:���A�!*;


total_loss�%�@

error_R�A?

learning_rate_1��j7�5�MI       6%�	���:���A�!*;


total_loss\�@

error_RO)J?

learning_rate_1��j79�I       6%�	P0�:���A�!*;


total_loss��@

error_R1�H?

learning_rate_1��j7���I       6%�	�{�:���A�!*;


total_loss�A

error_R]�m?

learning_rate_1��j7$}�I       6%�	M��:���A�!*;


total_loss�@

error_R��_?

learning_rate_1��j7M�~I       6%�	>�:���A�!*;


total_lossܗ�@

error_R��K?

learning_rate_1��j7��U?I       6%�	�_�:���A�!*;


total_loss6HA

error_Rs�E?

learning_rate_1��j7�Ë�I       6%�	,��:���A�!*;


total_lossJ��@

error_R��V?

learning_rate_1��j70�^�I       6%�	���:���A�!*;


total_loss�_�@

error_Rx�M?

learning_rate_1��j7�V�8I       6%�	~3�:���A�!*;


total_loss�1!A

error_R8nd?

learning_rate_1��j7�{ȪI       6%�	ʀ�:���A�!*;


total_loss�8�@

error_R;�J?

learning_rate_1��j7x�I       6%�	���:���A�!*;


total_loss{`�@

error_ROKE?

learning_rate_1��j7%n�I       6%�	�:���A�!*;


total_loss]�@

error_R�Y?

learning_rate_1��j7���I       6%�	�P�:���A�!*;


total_loss��y@

error_R6?

learning_rate_1��j7�:DvI       6%�	��:���A�!*;


total_lossd��@

error_RڂG?

learning_rate_1��j7[>I�I       6%�	���:���A�!*;


total_lossN��@

error_R�d?

learning_rate_1��j7��I       6%�	��:���A�!*;


total_lossy��@

error_R�L?

learning_rate_1��j7�N�I       6%�	0]�:���A�!*;


total_loss��@

error_R��_?

learning_rate_1��j7P�)I       6%�	��:���A�!*;


total_loss��@

error_R�'M?

learning_rate_1��j7�@n�I       6%�	-��:���A�!*;


total_lossw@�@

error_RNT?

learning_rate_1��j7 �vI       6%�	 ,�:���A�!*;


total_loss{�@

error_Ri�X?

learning_rate_1��j7��d�I       6%�	�t�:���A�!*;


total_loss�-�@

error_R�O?

learning_rate_1��j7�s�zI       6%�	F��:���A�!*;


total_loss��@

error_R�H?

learning_rate_1��j7����I       6%�	�:���A�!*;


total_loss.��@

error_R��S?

learning_rate_1��j7�a�rI       6%�	6L�:���A�!*;


total_lossvI�@

error_R?6g?

learning_rate_1��j7?-�sI       6%�	ӏ�:���A�!*;


total_loss���@

error_R��B?

learning_rate_1��j7P���I       6%�	���:���A�!*;


total_loss�*�@

error_R=t?

learning_rate_1��j7��K�I       6%�	��:���A�!*;


total_lossƸ�@

error_R��H?

learning_rate_1��j7S5��I       6%�	�f�:���A�!*;


total_loss��@

error_RN�8?

learning_rate_1��j7�VcI       6%�	o��:���A�!*;


total_lossJA A

error_R}�5?

learning_rate_1��j7�n��I       6%�	���:���A�!*;


total_loss�Qy@

error_R��J?

learning_rate_1��j7�}	�I       6%�	�2�:���A�!*;


total_loss8�@

error_R�R?

learning_rate_1��j7����I       6%�	�x�:���A�!*;


total_loss���@

error_R�94?

learning_rate_1��j7���I       6%�	���:���A�!*;


total_loss�A

error_Rc�Q?

learning_rate_1��j7i�V�I       6%�	N �:���A�!*;


total_lossc�@

error_R�"D?

learning_rate_1��j7S ��I       6%�	�C�:���A�!*;


total_loss��@

error_R��Q?

learning_rate_1��j72��I       6%�	���:���A�!*;


total_loss�S�@

error_RÊT?

learning_rate_1��j7�2��I       6%�	��:���A�!*;


total_loss*�@

error_R�JQ?

learning_rate_1��j7F�Z6I       6%�	B�:���A�!*;


total_loss�e�@

error_R��J?

learning_rate_1��j7�s�I       6%�	���:���A�!*;


total_lossS��@

error_RV?

learning_rate_1��j7��Y)I       6%�	#��:���A�!*;


total_loss�$�@

error_RsfL?

learning_rate_1��j7�ݨ�I       6%�	��:���A�!*;


total_loss�{�@

error_Rf�H?

learning_rate_1��j7`�_�I       6%�	fT�:���A�!*;


total_loss��@

error_R�8M?

learning_rate_1��j7PjI       6%�	��:���A�!*;


total_loss!H�@

error_R��Q?

learning_rate_1��j7`Rk�I       6%�	 ��:���A�!*;


total_loss�y�@

error_R�h_?

learning_rate_1��j7��X0I       6%�	�"�:���A�!*;


total_loss�.�@

error_RC�L?

learning_rate_1��j7�p/I       6%�	zi�:���A�!*;


total_loss��@

error_R� Y?

learning_rate_1��j7)a,I       6%�	���:���A�!*;


total_loss��@

error_R�kQ?

learning_rate_1��j7c��I       6%�	e��:���A�!*;


total_lossO��@

error_R��M?

learning_rate_1��j7�*|.I       6%�	�<�:���A�!*;


total_lossR��@

error_RAb`?

learning_rate_1��j7K�&>I       6%�	���:���A�!*;


total_lossx��@

error_RmH?

learning_rate_1��j7ѐ-_I       6%�	9��:���A�!*;


total_loss;�Y@

error_R��Q?

learning_rate_1��j7*�VI       6%�	��:���A�!*;


total_lossh �@

error_R_S?

learning_rate_1��j7r˫�I       6%�	?W�:���A�!*;


total_loss;��@

error_R��8?

learning_rate_1��j7}�p�I       6%�	���:���A�!*;


total_loss��@

error_R��;?

learning_rate_1��j7��dI       6%�	$��:���A�!*;


total_lossnp�@

error_R��Q?

learning_rate_1��j7� �VI       6%�	T+�:���A�!*;


total_loss�U�@

error_RhK?

learning_rate_1��j7�HwI       6%�	kq�:���A�!*;


total_loss:��@

error_R; L?

learning_rate_1��j75譢I       6%�	���:���A�!*;


total_lossƳ�@

error_RI8B?

learning_rate_1��j7S8&I       6%�	F��:���A�!*;


total_loss��@

error_R�fK?

learning_rate_1��j7g|�VI       6%�	w>�:���A�!*;


total_loss���@

error_RXW?

learning_rate_1��j7����I       6%�	:��:���A�!*;


total_loss���@

error_Rz�^?

learning_rate_1��j7{��II       6%�	���:���A�!*;


total_loss�a�@

error_R��Y?

learning_rate_1��j7��!�I       6%�	q*�:���A�!*;


total_loss}F�@

error_R��\?

learning_rate_1��j7�Gb�I       6%�	�t�:���A�!*;


total_lossn��@

error_R�xS?

learning_rate_1��j7�:�I       6%�	7��:���A�!*;


total_loss��@

error_R�5L?

learning_rate_1��j7/�[�I       6%�	��:���A�!*;


total_loss�Vs@

error_R�8M?

learning_rate_1��j7~�&I       6%�	�Y�:���A�!*;


total_loss�;�@

error_R�*F?

learning_rate_1��j7iN	7I       6%�	a��:���A�!*;


total_loss@M�@

error_R��X?

learning_rate_1��j7�1��I       6%�	#��:���A�!*;


total_loss���@

error_R[TB?

learning_rate_1��j7�eOI       6%�	h+�:���A�!*;


total_loss|
�@

error_R�E?

learning_rate_1��j7îָI       6%�	Jp�:���A�!*;


total_loss#��@

error_R	�S?

learning_rate_1��j73��.I       6%�	���:���A�!*;


total_loss�J�@

error_Rl�N?

learning_rate_1��j7&e�I       6%�	K��:���A�!*;


total_lossyMA

error_R*�M?

learning_rate_1��j7�I       6%�	{= ;���A�!*;


total_loss��A

error_R��@?

learning_rate_1��j7W��I       6%�	�� ;���A�!*;


total_lossl��@

error_R�R?

learning_rate_1��j79�z�I       6%�	�� ;���A�!*;


total_lossRV�@

error_R��V?

learning_rate_1��j7!Ц�I       6%�	";���A�!*;


total_lossӦ�@

error_R�bK?

learning_rate_1��j7q��I       6%�	�P;���A�!*;


total_lossܲ�@

error_R��O?

learning_rate_1��j7 ��;I       6%�	��;���A�!*;


total_loss��@

error_R�V?

learning_rate_1��j77��I       6%�	��;���A�!*;


total_loss,c�@

error_R�XH?

learning_rate_1��j7�A6I       6%�	;���A�!*;


total_loss�O�@

error_R��c?

learning_rate_1��j7v�}�I       6%�	�d;���A�!*;


total_loss�@

error_R!�C?

learning_rate_1��j7�I�I       6%�	7�;���A�!*;


total_lossX�@

error_R-�f?

learning_rate_1��j7�F��I       6%�	��;���A�!*;


total_loss�v�@

error_Ra�V?

learning_rate_1��j7�$�I       6%�	n/;���A�!*;


total_loss�E^@

error_R��K?

learning_rate_1��j7�v9I       6%�	@q;���A�!*;


total_loss���@

error_R�I?

learning_rate_1��j7ޱI       6%�	A�;���A�!*;


total_loss��@

error_R�_Q?

learning_rate_1��j7���I       6%�	$�;���A�!*;


total_lossNǱ@

error_R]�P?

learning_rate_1��j75�ȿI       6%�	?;���A�!*;


total_loss���@

error_R�^?

learning_rate_1��j7��/I       6%�	��;���A�!*;


total_losssm�@

error_R��J?

learning_rate_1��j7?� yI       6%�	#�;���A�!*;


total_loss���@

error_Rv 4?

learning_rate_1��j7��XI       6%�	$;���A�!*;


total_loss<	�@

error_R��O?

learning_rate_1��j70��I       6%�	MY;���A�!*;


total_lossv��@

error_R�VQ?

learning_rate_1��j7��.I       6%�	a�;���A�!*;


total_lossai�@

error_RT�T?

learning_rate_1��j7�,"I       6%�	 �;���A�!*;


total_loss�C�@

error_R�/6?

learning_rate_1��j7��I       6%�	�8;���A�!*;


total_loss�W�@

error_R��^?

learning_rate_1��j7�Ҁ�I       6%�	��;���A�!*;


total_lossd-�@

error_R�9H?

learning_rate_1��j7V�hVI       6%�	r�;���A�!*;


total_loss��@

error_R��Z?

learning_rate_1��j73�ؘI       6%�	%;���A�!*;


total_lossN�A

error_RN�K?

learning_rate_1��j7��I       6%�	�V;���A�!*;


total_lossT�@

error_Rn�V?

learning_rate_1��j7b�I       6%�	[�;���A�!*;


total_loss���@

error_R�qL?

learning_rate_1��j7����I       6%�	��;���A�!*;


total_loss��@

error_R��??

learning_rate_1��j7A�e�I       6%�	-';���A�!*;


total_loss��@

error_Ra�Q?

learning_rate_1��j7"BI       6%�	�l;���A�!*;


total_lossn��@

error_R�wO?

learning_rate_1��j7�3�I       6%�	ͯ;���A�!*;


total_lossv-�@

error_Rr_G?

learning_rate_1��j7j�1�I       6%�	��;���A�!*;


total_loss\o]@

error_RJ�D?

learning_rate_1��j7/�YI       6%�	_7	;���A�!*;


total_loss�U�@

error_R��G?

learning_rate_1��j7X!�I       6%�	Az	;���A�!*;


total_loss�{@

error_R��=?

learning_rate_1��j7p�@aI       6%�	:�	;���A�!*;


total_loss|��@

error_R#�M?

learning_rate_1��j7�a�dI       6%�	�
;���A�!*;


total_loss���@

error_R�@a?

learning_rate_1��j7�XGI       6%�	YF
;���A�!*;


total_loss�°@

error_R�9L?

learning_rate_1��j7�b/2I       6%�	ԇ
;���A�!*;


total_loss�θ@

error_R��F?

learning_rate_1��j7S&I       6%�	��
;���A�!*;


total_losseqA

error_RE?

learning_rate_1��j7���I       6%�	�;���A�!*;


total_loss8e�@

error_Rs�E?

learning_rate_1��j7e0�I       6%�	O];���A�!*;


total_loss�2EA

error_R��I?

learning_rate_1��j7��I       6%�	(�;���A�!*;


total_loss���@

error_R�yB?

learning_rate_1��j7�ʶ�I       6%�	��;���A�"*;


total_lossQv�@

error_RH�T?

learning_rate_1��j7E7+�I       6%�	�&;���A�"*;


total_loss�� A

error_R�Q?

learning_rate_1��j7���I       6%�	Cm;���A�"*;


total_loss�ā@

error_R��U?

learning_rate_1��j7�8��I       6%�	��;���A�"*;


total_loss��@

error_RH�G?

learning_rate_1��j7)L��I       6%�	��;���A�"*;


total_loss:RA

error_R<8?

learning_rate_1��j7�BI       6%�	';;���A�"*;


total_loss7��@

error_R7e?

learning_rate_1��j7�$#I       6%�	��;���A�"*;


total_loss(�@

error_RsN?

learning_rate_1��j7���I       6%�	��;���A�"*;


total_loss3_�@

error_R�BM?

learning_rate_1��j7���1I       6%�	v;���A�"*;


total_loss���@

error_R:�d?

learning_rate_1��j7�U�
I       6%�	�U;���A�"*;


total_lossK�@

error_R�R?

learning_rate_1��j7bo@I       6%�	מ;���A�"*;


total_loss;�@

error_R�P?

learning_rate_1��j7�9KI       6%�	��;���A�"*;


total_loss=��@

error_RE�B?

learning_rate_1��j7%l�fI       6%�	Z.;���A�"*;


total_lossc��@

error_R�lN?

learning_rate_1��j7T�I       6%�	�v;���A�"*;


total_loss�!�@

error_R��@?

learning_rate_1��j7��TuI       6%�	W�;���A�"*;


total_loss�RA

error_R8�M?

learning_rate_1��j7֭�I       6%�	�;���A�"*;


total_loss �@

error_Rl^;?

learning_rate_1��j7�d'�I       6%�	�G;���A�"*;


total_loss�ؔ@

error_RRaJ?

learning_rate_1��j7��I       6%�	��;���A�"*;


total_loss]m�@

error_R�3Q?

learning_rate_1��j7���I       6%�	��;���A�"*;


total_loss��@

error_RRG?

learning_rate_1��j7�z$I       6%�	�;���A�"*;


total_lossԂ�@

error_Rv�D?

learning_rate_1��j7��hnI       6%�	�`;���A�"*;


total_losso3�@

error_R|N`?

learning_rate_1��j7#\(�I       6%�	j�;���A�"*;


total_loss���@

error_R�;K?

learning_rate_1��j7մ�I       6%�	��;���A�"*;


total_loss���@

error_R;Q?

learning_rate_1��j7�lI       6%�	4;���A�"*;


total_loss���@

error_RI�K?

learning_rate_1��j7=1!I       6%�	y;���A�"*;


total_loss�$�@

error_R�,C?

learning_rate_1��j7��	XI       6%�	X�;���A�"*;


total_lossA��@

error_RN>P?

learning_rate_1��j7A���I       6%�	�;���A�"*;


total_loss5�@

error_R��L?

learning_rate_1��j7�cL@I       6%�	[_;���A�"*;


total_loss�E�@

error_RÐ:?

learning_rate_1��j7a)�?I       6%�	K�;���A�"*;


total_loss�G�@

error_R� I?

learning_rate_1��j7Fʂ�I       6%�	?�;���A�"*;


total_lossD��@

error_Rl�B?

learning_rate_1��j7��I       6%�	�*;���A�"*;


total_loss{&�@

error_R K?

learning_rate_1��j7u
�I       6%�	uo;���A�"*;


total_loss�d@

error_R��E?

learning_rate_1��j7��I       6%�	g�;���A�"*;


total_loss���@

error_R�fK?

learning_rate_1��j7+j�I       6%�	�;���A�"*;


total_loss��@

error_R�3D?

learning_rate_1��j7_%!�I       6%�	�O;���A�"*;


total_loss���@

error_ReT?

learning_rate_1��j7=��I       6%�	c�;���A�"*;


total_loss�ü@

error_R��G?

learning_rate_1��j7��J�I       6%�	��;���A�"*;


total_losszC�@

error_R K?

learning_rate_1��j7�P�
I       6%�	�L;���A�"*;


total_loss���@

error_RE�I?

learning_rate_1��j7΢�I       6%�	D�;���A�"*;


total_loss0�@

error_R�OJ?

learning_rate_1��j7�X�I       6%�	
�;���A�"*;


total_loss��@

error_R�@?

learning_rate_1��j7茒�I       6%�	>;���A�"*;


total_loss���@

error_R��W?

learning_rate_1��j7����I       6%�	e�;���A�"*;


total_lossD4�@

error_R��Q?

learning_rate_1��j7�X�YI       6%�	
�;���A�"*;


total_lossl3�@

error_RJ4Y?

learning_rate_1��j7�Ee%I       6%�	;���A�"*;


total_loss�1�@

error_R�T?

learning_rate_1��j7�w�I       6%�	�Z;���A�"*;


total_loss��@

error_R�E?

learning_rate_1��j73��mI       6%�	��;���A�"*;


total_loss{X�@

error_Rl�Y?

learning_rate_1��j7��Q�I       6%�	��;���A�"*;


total_loss7A

error_R�Y?

learning_rate_1��j7o5�I       6%�	�4;���A�"*;


total_loss6�@

error_R6mJ?

learning_rate_1��j7��?�I       6%�	�w;���A�"*;


total_loss܉�@

error_RxD?

learning_rate_1��j7Y[ƾI       6%�	$�;���A�"*;


total_lossF��@

error_R[�S?

learning_rate_1��j7�X�xI       6%�	� ;���A�"*;


total_loss{�@

error_RrL?

learning_rate_1��j7pb}I       6%�	�G;���A�"*;


total_loss��@

error_R<�Q?

learning_rate_1��j7
���I       6%�	׌;���A�"*;


total_lossܒ�@

error_RBH?

learning_rate_1��j7�!bsI       6%�	��;���A�"*;


total_loss���@

error_R�}T?

learning_rate_1��j7ҵ�I       6%�	;���A�"*;


total_lossNd�@

error_R �B?

learning_rate_1��j7t���I       6%�	�V;���A�"*;


total_loss���@

error_RTT?

learning_rate_1��j7PA,�I       6%�	��;���A�"*;


total_loss�9�@

error_Rc�J?

learning_rate_1��j7����I       6%�	�;���A�"*;


total_losso0�@

error_R-�;?

learning_rate_1��j7M���I       6%�	-";���A�"*;


total_loss;A

error_R��T?

learning_rate_1��j7Uw3I       6%�	�g;���A�"*;


total_loss�h�@

error_R|�n?

learning_rate_1��j7��8I       6%�	ͱ;���A�"*;


total_loss�Zw@

error_R��F?

learning_rate_1��j7�$�I       6%�	�;���A�"*;


total_loss��@

error_RʛS?

learning_rate_1��j7W3�)I       6%�	�8;���A�"*;


total_loss5�@

error_RC�L?

learning_rate_1��j7NY�uI       6%�	�y;���A�"*;


total_lossK��@

error_R}#^?

learning_rate_1��j7�f_�I       6%�	��;���A�"*;


total_loss��@

error_R��:?

learning_rate_1��j7?��I       6%�	-;���A�"*;


total_lossݾ�@

error_Rd�g?

learning_rate_1��j7q���I       6%�	�\;���A�"*;


total_loss���@

error_RN*Z?

learning_rate_1��j7����I       6%�	Φ;���A�"*;


total_loss)�@

error_R�K?

learning_rate_1��j7a�&�I       6%�	{�;���A�"*;


total_loss:�e@

error_R�CA?

learning_rate_1��j7��
�I       6%�	S<;���A�"*;


total_loss��A

error_R�Q?

learning_rate_1��j7�ӻ�I       6%�	i�;���A�"*;


total_lossQ��@

error_R�.Y?

learning_rate_1��j7����I       6%�	��;���A�"*;


total_loss��@

error_R��P?

learning_rate_1��j7�M�yI       6%�	� ;���A�"*;


total_loss2�@

error_R�/F?

learning_rate_1��j7�
�I       6%�	�X ;���A�"*;


total_loss_��@

error_RJ%I?

learning_rate_1��j7��KII       6%�	:� ;���A�"*;


total_loss��@

error_R�a?

learning_rate_1��j7d"_�I       6%�	�� ;���A�"*;


total_loss=�A

error_R��T?

learning_rate_1��j7x)��I       6%�	;,!;���A�"*;


total_lossc��@

error_R�r_?

learning_rate_1��j7X�cI       6%�	Ft!;���A�"*;


total_lossF�	A

error_RɴU?

learning_rate_1��j7�\V�I       6%�	%�!;���A�"*;


total_loss䄈@

error_RL�@?

learning_rate_1��j7"3�I       6%�	��!;���A�"*;


total_loss��@

error_Rc�W?

learning_rate_1��j7#��I       6%�	x@";���A�"*;


total_loss7��@

error_R�D?

learning_rate_1��j7��I       6%�	r�";���A�"*;


total_loss���@

error_R1U?

learning_rate_1��j7��nI       6%�	l�";���A�"*;


total_loss���@

error_Rq�=?

learning_rate_1��j7i�nI       6%�	5#;���A�"*;


total_lossxN�@

error_R��V?

learning_rate_1��j7����I       6%�	O#;���A�"*;


total_loss���@

error_R�uI?

learning_rate_1��j7h�xfI       6%�	x�#;���A�"*;


total_loss�C�@

error_R/$D?

learning_rate_1��j7C�4�I       6%�	z�#;���A�"*;


total_loss���@

error_RW�C?

learning_rate_1��j7�b#'I       6%�	M$;���A�"*;


total_loss�۵@

error_R[?

learning_rate_1��j7��<sI       6%�	xY$;���A�"*;


total_loss_�@

error_R;�M?

learning_rate_1��j7��D6I       6%�	h�$;���A�"*;


total_lossm[�@

error_R�X?

learning_rate_1��j7�_;�I       6%�	A�$;���A�"*;


total_loss
�@

error_R�YS?

learning_rate_1��j7-�I       6%�	u+%;���A�"*;


total_loss�&�@

error_Rj�L?

learning_rate_1��j7��I       6%�	t%;���A�"*;


total_loss���@

error_R	�Y?

learning_rate_1��j7?l�aI       6%�	A�%;���A�"*;


total_loss�Ă@

error_R�7G?

learning_rate_1��j7i�:_I       6%�	�
&;���A�"*;


total_loss:��@

error_R}K?

learning_rate_1��j7��-YI       6%�	Bq&;���A�"*;


total_loss���@

error_R4l?

learning_rate_1��j7j�YI       6%�	h�&;���A�"*;


total_loss���@

error_R�X?

learning_rate_1��j7�igI       6%�	l';���A�"*;


total_loss�9�@

error_R�^?

learning_rate_1��j7�^�LI       6%�	�H';���A�"*;


total_loss�w�@

error_RC�Q?

learning_rate_1��j7��ӍI       6%�	 �';���A�"*;


total_loss�w�@

error_R��U?

learning_rate_1��j7�2�I       6%�	��';���A�"*;


total_loss���@

error_R5M?

learning_rate_1��j7�̋MI       6%�	c(;���A�"*;


total_loss$�@

error_R4�Y?

learning_rate_1��j7M���I       6%�	ja(;���A�"*;


total_loss�m�@

error_RhAW?

learning_rate_1��j7T�p�I       6%�	.�(;���A�"*;


total_loss�,�@

error_R^?

learning_rate_1��j7�4�I       6%�	��(;���A�"*;


total_loss���@

error_RRHI?

learning_rate_1��j7FV�I       6%�	�,);���A�"*;


total_loss�A

error_R`p?

learning_rate_1��j7	��yI       6%�	�p);���A�"*;


total_loss%��@

error_R�AH?

learning_rate_1��j7�38�I       6%�	��);���A�"*;


total_lossM��@

error_R?�L?

learning_rate_1��j7iF.I       6%�	��);���A�"*;


total_loss�f�@

error_R��_?

learning_rate_1��j7rJRI       6%�	�;*;���A�"*;


total_loss�*�@

error_R(�H?

learning_rate_1��j7�B��I       6%�	�*;���A�"*;


total_loss9��@

error_RRN?

learning_rate_1��j7��I       6%�	m�*;���A�"*;


total_lossW�@

error_R�LD?

learning_rate_1��j7=M��I       6%�	�
+;���A�"*;


total_loss��@

error_R��K?

learning_rate_1��j7L&�I       6%�	�P+;���A�"*;


total_loss진@

error_R�$M?

learning_rate_1��j7DG��I       6%�	/�+;���A�"*;


total_lossJ��@

error_R��T?

learning_rate_1��j7�1-�I       6%�	��+;���A�"*;


total_loss���@

error_R<,:?

learning_rate_1��j7�X�QI       6%�	),;���A�"*;


total_loss:t�@

error_R�`]?

learning_rate_1��j7zY�I       6%�	�m,;���A�"*;


total_loss���@

error_R$%M?

learning_rate_1��j7���DI       6%�	�,;���A�"*;


total_loss�O�@

error_R�N?

learning_rate_1��j7(��I       6%�	��,;���A�"*;


total_loss��@

error_R�O?

learning_rate_1��j7z�9�I       6%�	H>-;���A�"*;


total_loss	�g@

error_R�;i?

learning_rate_1��j7�7I       6%�	�-;���A�"*;


total_loss��@

error_R��H?

learning_rate_1��j7����I       6%�	�-;���A�"*;


total_lossa��@

error_R1�L?

learning_rate_1��j710��I       6%�	r.;���A�"*;


total_loss���@

error_Rq�\?

learning_rate_1��j7&&ˏI       6%�	P.;���A�"*;


total_loss�s�@

error_R�'N?

learning_rate_1��j7� ��I       6%�	&�.;���A�"*;


total_loss;U�@

error_R��??

learning_rate_1��j7-Y)�I       6%�	�.;���A�"*;


total_lossA�@

error_Rj�Y?

learning_rate_1��j77�I       6%�	8*/;���A�"*;


total_loss<��@

error_R�I?

learning_rate_1��j7hԟ)I       6%�	5o/;���A�#*;


total_lossß�@

error_RR�N?

learning_rate_1��j7:�-�I       6%�	G�/;���A�#*;


total_loss&��@

error_R�3R?

learning_rate_1��j7����I       6%�	k�/;���A�#*;


total_loss�@

error_R7cN?

learning_rate_1��j7U���I       6%�	<0;���A�#*;


total_loss��@

error_RȡR?

learning_rate_1��j7Z�I       6%�	�0;���A�#*;


total_lossm��@

error_RH�O?

learning_rate_1��j7�(��I       6%�	��0;���A�#*;


total_loss��@

error_RSY?

learning_rate_1��j7|��I       6%�	�1;���A�#*;


total_loss���@

error_R}xS?

learning_rate_1��j7d���I       6%�	�K1;���A�#*;


total_loss��@

error_R߉R?

learning_rate_1��j7�rAoI       6%�	�1;���A�#*;


total_loss�h�@

error_R/I=?

learning_rate_1��j7��I       6%�	#�1;���A�#*;


total_lossW��@

error_RʜK?

learning_rate_1��j7�V��I       6%�	�2;���A�#*;


total_lossڀ�@

error_R�j?

learning_rate_1��j7.�I1I       6%�	Z_2;���A�#*;


total_lossjl�@

error_R�`Q?

learning_rate_1��j7���I       6%�	�2;���A�#*;


total_loss�/�@

error_R�L?

learning_rate_1��j7d̸EI       6%�	~�2;���A�#*;


total_loss�'�@

error_R]�]?

learning_rate_1��j7��UFI       6%�	]*3;���A�#*;


total_loss���@

error_RMDO?

learning_rate_1��j7v$�I       6%�	�n3;���A�#*;


total_loss�u�@

error_R�'\?

learning_rate_1��j7���!I       6%�	�3;���A�#*;


total_loss��@

error_R;�[?

learning_rate_1��j7�
īI       6%�	��3;���A�#*;


total_lossi��@

error_R�@?

learning_rate_1��j70��OI       6%�	64;���A�#*;


total_loss�ȏ@

error_R��O?

learning_rate_1��j7)OI       6%�	mx4;���A�#*;


total_loss#�@

error_R��H?

learning_rate_1��j7
<ßI       6%�	��4;���A�#*;


total_loss\��@

error_Rm�G?

learning_rate_1��j7��ΫI       6%�	K
5;���A�#*;


total_loss�u�@

error_R��;?

learning_rate_1��j7�=("I       6%�	(K5;���A�#*;


total_loss�Y�@

error_R�b?

learning_rate_1��j7��܍I       6%�	�5;���A�#*;


total_losseD5A

error_R_]B?

learning_rate_1��j7�ԪI       6%�	N�5;���A�#*;


total_loss@ð@

error_R��K?

learning_rate_1��j7�6��I       6%�	,!6;���A�#*;


total_loss{[�@

error_R�O?

learning_rate_1��j7R���I       6%�	�{6;���A�#*;


total_loss���@

error_R�p[?

learning_rate_1��j7���BI       6%�	W�6;���A�#*;


total_loss��A

error_R�bO?

learning_rate_1��j7ZL�I       6%�	$�6;���A�#*;


total_loss��@

error_R.�G?

learning_rate_1��j7"ޖI       6%�	cC7;���A�#*;


total_loss%U�@

error_Rd�n?

learning_rate_1��j7�WI       6%�	�7;���A�#*;


total_loss$�@

error_R��M?

learning_rate_1��j7��I       6%�	��7;���A�#*;


total_lossr8�@

error_R�3]?

learning_rate_1��j7ƾܵI       6%�	E8;���A�#*;


total_loss�3�@

error_R�
T?

learning_rate_1��j7\���I       6%�	lP8;���A�#*;


total_loss\A�@

error_RڭV?

learning_rate_1��j7����I       6%�	ە8;���A�#*;


total_loss�%�@

error_RɀN?

learning_rate_1��j7��NI       6%�	2�8;���A�#*;


total_loss̩�@

error_R�RT?

learning_rate_1��j7�:UI       6%�	9;���A�#*;


total_lossJm�@

error_Rq�K?

learning_rate_1��j7{gq�I       6%�	�^9;���A�#*;


total_loss8h�@

error_R��T?

learning_rate_1��j7,*I       6%�	�9;���A�#*;


total_loss8��@

error_R�Z?

learning_rate_1��j7�R��I       6%�	��9;���A�#*;


total_loss���@

error_R�M?

learning_rate_1��j7c�MI       6%�	^9:;���A�#*;


total_loss�-�@

error_R�-H?

learning_rate_1��j7
��I       6%�	��:;���A�#*;


total_loss�w�@

error_Rj�c?

learning_rate_1��j7�ܺpI       6%�	m�:;���A�#*;


total_lossZ�u@

error_R�=?

learning_rate_1��j7�20I       6%�	�;;���A�#*;


total_loss��@

error_R��Q?

learning_rate_1��j7\7��I       6%�		\;;���A�#*;


total_loss%*�@

error_RzM?

learning_rate_1��j7d�НI       6%�	ݟ;;���A�#*;


total_loss��@

error_RDZJ?

learning_rate_1��j7�RI       6%�	��;;���A�#*;


total_loss�y�@

error_R8�\?

learning_rate_1��j7��1KI       6%�	�'<;���A�#*;


total_lossߊ�@

error_R&�O?

learning_rate_1��j7�4�I       6%�	�l<;���A�#*;


total_loss��b@

error_Rv�K?

learning_rate_1��j7�{��I       6%�	\�<;���A�#*;


total_loss��@

error_R��E?

learning_rate_1��j7�R	~I       6%�	�=;���A�#*;


total_loss��@

error_Rm�^?

learning_rate_1��j7���I       6%�	�L=;���A�#*;


total_loss���@

error_R6�F?

learning_rate_1��j7��d�I       6%�	��=;���A�#*;


total_loss�1�@

error_R�??

learning_rate_1��j7pI�I       6%�	��=;���A�#*;


total_loss1�@

error_Ra�M?

learning_rate_1��j7�y��I       6%�	�1>;���A�#*;


total_loss�@

error_R<�J?

learning_rate_1��j7I�JI       6%�	�z>;���A�#*;


total_lossJ��@

error_Rh�V?

learning_rate_1��j7AL�cI       6%�	]�>;���A�#*;


total_losse�v@

error_R<dE?

learning_rate_1��j7jI       6%�	�?;���A�#*;


total_loss)��@

error_R��P?

learning_rate_1��j7mtݰI       6%�	L?;���A�#*;


total_lossŊ�@

error_RMr\?

learning_rate_1��j7^�3�I       6%�	��?;���A�#*;


total_loss=��@

error_R\4Z?

learning_rate_1��j7��w�I       6%�	`�?;���A�#*;


total_loss٣@

error_RM2C?

learning_rate_1��j7s�gI       6%�	@;���A�#*;


total_loss��@

error_R�@N?

learning_rate_1��j7��Q,I       6%�	re@;���A�#*;


total_loss3�@

error_RJ2V?

learning_rate_1��j7#�DI       6%�	Z�@;���A�#*;


total_lossM�@

error_RE�<?

learning_rate_1��j7�z�I       6%�	k�@;���A�#*;


total_loss(�A

error_Rt�C?

learning_rate_1��j7W��I       6%�	�;A;���A�#*;


total_lossܘ@

error_RO<?

learning_rate_1��j7���I       6%�	9�A;���A�#*;


total_loss��@

error_R$�^?

learning_rate_1��j7����I       6%�	5�A;���A�#*;


total_loss�n�@

error_R�V[?

learning_rate_1��j7�I       6%�	LB;���A�#*;


total_lossr�@

error_R��N?

learning_rate_1��j7��II       6%�	�\B;���A�#*;


total_loss��@

error_R$z]?

learning_rate_1��j7��QMI       6%�	V�B;���A�#*;


total_loss���@

error_R=�4?

learning_rate_1��j7H�6KI       6%�	��B;���A�#*;


total_lossO�@

error_RK?

learning_rate_1��j7�I       6%�	�1C;���A�#*;


total_loss�A

error_RI?

learning_rate_1��j7��`�I       6%�	�tC;���A�#*;


total_loss��A

error_R��K?

learning_rate_1��j7V|<�I       6%�	��C;���A�#*;


total_loss�B�@

error_R��K?

learning_rate_1��j7v{I       6%�	��C;���A�#*;


total_loss_=�@

error_R��[?

learning_rate_1��j7q|ÄI       6%�	�AD;���A�#*;


total_loss���@

error_R��L?

learning_rate_1��j7Q���I       6%�	ʄD;���A�#*;


total_lossʓKA

error_R��K?

learning_rate_1��j7?��;I       6%�	��D;���A�#*;


total_loss*R�@

error_R��`?

learning_rate_1��j7r!�I       6%�	eE;���A�#*;


total_lossw+�@

error_RP?

learning_rate_1��j7*CGI       6%�	NE;���A�#*;


total_loss�RA

error_R��V?

learning_rate_1��j78U��I       6%�	6�E;���A�#*;


total_loss�X�@

error_R��8?

learning_rate_1��j7����I       6%�	6�E;���A�#*;


total_loss�s�@

error_R��R?

learning_rate_1��j7�#I       6%�	�'F;���A�#*;


total_loss��@

error_R$�5?

learning_rate_1��j7(���I       6%�	zF;���A�#*;


total_loss���@

error_R��F?

learning_rate_1��j7�C��I       6%�	��F;���A�#*;


total_loss6��@

error_R�cL?

learning_rate_1��j7��I       6%�	�G;���A�#*;


total_lossa��@

error_R�[?

learning_rate_1��j76�:�I       6%�	FPG;���A�#*;


total_lossfǪ@

error_R�T?

learning_rate_1��j7 xI       6%�	��G;���A�#*;


total_loss/:�@

error_R\T?

learning_rate_1��j7J�j0I       6%�		�G;���A�#*;


total_loss�{�@

error_R��J?

learning_rate_1��j73�?>I       6%�	�%H;���A�#*;


total_lossTM�@

error_R��M?

learning_rate_1��j7�-�I       6%�	hH;���A�#*;


total_losss��@

error_R[5]?

learning_rate_1��j7�f�I       6%�	�H;���A�#*;


total_loss��@

error_Rq�N?

learning_rate_1��j7���I       6%�	��H;���A�#*;


total_loss�*�@

error_R|{R?

learning_rate_1��j7���RI       6%�	J4I;���A�#*;


total_loss}��@

error_R��B?

learning_rate_1��j7-
TkI       6%�	�{I;���A�#*;


total_loss�F�@

error_R_CK?

learning_rate_1��j74{�I       6%�	��I;���A�#*;


total_loss��@

error_R�@Y?

learning_rate_1��j7�s��I       6%�	n
J;���A�#*;


total_lossI�@

error_R�VB?

learning_rate_1��j7%䢫I       6%�	�QJ;���A�#*;


total_loss
\�@

error_R�Y[?

learning_rate_1��j7դ~�I       6%�	�J;���A�#*;


total_loss!��@

error_R eJ?

learning_rate_1��j7���EI       6%�	!�J;���A�#*;


total_lossTʙ@

error_R}L[?

learning_rate_1��j7 V1%I       6%�	�K;���A�#*;


total_loss7R�@

error_R��O?

learning_rate_1��j7�/u�I       6%�	�cK;���A�#*;


total_loss��@

error_R
�W?

learning_rate_1��j7܍�XI       6%�	ӥK;���A�#*;


total_loss���@

error_R.{U?

learning_rate_1��j7�J�I       6%�	:�K;���A�#*;


total_loss���@

error_R��V?

learning_rate_1��j7#�+MI       6%�	�+L;���A�#*;


total_loss��@

error_R.�M?

learning_rate_1��j7�%�I       6%�	7qL;���A�#*;


total_loss��@

error_R��B?

learning_rate_1��j7H��?I       6%�	N�L;���A�#*;


total_loss��@

error_R}J?

learning_rate_1��j7���\I       6%�	g�L;���A�#*;


total_loss��@

error_R��B?

learning_rate_1��j76J�I       6%�	�9M;���A�#*;


total_loss	2�@

error_R��_?

learning_rate_1��j7�_/I       6%�	�M;���A�#*;


total_loss#ʫ@

error_R�s>?

learning_rate_1��j7�OX I       6%�	��M;���A�#*;


total_loss��@

error_R�3W?

learning_rate_1��j7<��TI       6%�	�N;���A�#*;


total_lossv��@

error_R�fF?

learning_rate_1��j7���I       6%�	ON;���A�#*;


total_loss�y�@

error_R&�H?

learning_rate_1��j7�W�I       6%�	БN;���A�#*;


total_loss6�@

error_R�QO?

learning_rate_1��j7����I       6%�	��N;���A�#*;


total_loss���@

error_R
N\?

learning_rate_1��j7Q��[I       6%�	'O;���A�#*;


total_loss~ߡ@

error_R=8?

learning_rate_1��j7�l�?I       6%�	�VO;���A�#*;


total_lossr&�@

error_R��O?

learning_rate_1��j7�:`�I       6%�	��O;���A�#*;


total_loss*��@

error_RJ?

learning_rate_1��j7hT.I       6%�	��O;���A�#*;


total_loss�s�@

error_RaX?

learning_rate_1��j7�*�HI       6%�	�-P;���A�#*;


total_loss���@

error_R��O?

learning_rate_1��j7��G�I       6%�	�rP;���A�#*;


total_loss]��@

error_R��M?

learning_rate_1��j7>ZI       6%�	żP;���A�#*;


total_loss׾�@

error_R
�X?

learning_rate_1��j7ncS�I       6%�	: Q;���A�#*;


total_loss�6�@

error_R-??

learning_rate_1��j7���DI       6%�	�GQ;���A�#*;


total_loss�@

error_R�y<?

learning_rate_1��j7 ��:I       6%�	Y�Q;���A�#*;


total_loss��@

error_R��=?

learning_rate_1��j7!�\I       6%�	��Q;���A�#*;


total_loss"�@

error_R�`?

learning_rate_1��j72F~I       6%�	R;���A�#*;


total_loss���@

error_R�CP?

learning_rate_1��j7�ׯ�I       6%�	�[R;���A�$*;


total_lossz�@

error_R��W?

learning_rate_1��j7��2�I       6%�	`�R;���A�$*;


total_loss��@

error_R��L?

learning_rate_1��j7{��iI       6%�	l�R;���A�$*;


total_loss`~�@

error_R�YB?

learning_rate_1��j7"�7I       6%�	�0S;���A�$*;


total_lossi�@

error_R �5?

learning_rate_1��j7�WEI       6%�	�tS;���A�$*;


total_loss�p�@

error_R�Q?

learning_rate_1��j7G�řI       6%�	��S;���A�$*;


total_loss�8�@

error_RC�J?

learning_rate_1��j7"��#I       6%�	NT;���A�$*;


total_loss���@

error_Rj�O?

learning_rate_1��j7ax2�I       6%�	�ET;���A�$*;


total_lossHL�@

error_R�SR?

learning_rate_1��j7̓p�I       6%�	�T;���A�$*;


total_lossW˝@

error_R��U?

learning_rate_1��j7���-I       6%�	_�T;���A�$*;


total_lossf)�@

error_R��^?

learning_rate_1��j7t��mI       6%�	�U;���A�$*;


total_lossl"�@

error_R(o=?

learning_rate_1��j7Dt�I       6%�	�WU;���A�$*;


total_loss�\�@

error_RF?

learning_rate_1��j7��0fI       6%�	/�U;���A�$*;


total_loss�'�@

error_R��/?

learning_rate_1��j7.��I       6%�	��U;���A�$*;


total_loss�8�@

error_R�R?

learning_rate_1��j7��I       6%�	)?V;���A�$*;


total_loss�+�@

error_Rl5?

learning_rate_1��j7��K�I       6%�	i�V;���A�$*;


total_lossTV�@

error_R��o?

learning_rate_1��j7ݲ<�I       6%�	��V;���A�$*;


total_loss�Έ@

error_Rq�Q?

learning_rate_1��j7�z�I       6%�	�W;���A�$*;


total_loss1��@

error_R&,I?

learning_rate_1��j7�p��I       6%�	�dW;���A�$*;


total_loss醿@

error_R�{b?

learning_rate_1��j76��I       6%�	|�W;���A�$*;


total_lossWխ@

error_R��<?

learning_rate_1��j7�J��I       6%�	��W;���A�$*;


total_lossT̗@

error_R:J?

learning_rate_1��j7�;I       6%�	J0X;���A�$*;


total_loss�@

error_R1�N?

learning_rate_1��j7D��I       6%�	$vX;���A�$*;


total_loss�o�@

error_R�1]?

learning_rate_1��j7+���I       6%�	_�X;���A�$*;


total_loss�H�@

error_R͖D?

learning_rate_1��j7�nd�I       6%�	�Y;���A�$*;


total_loss��@

error_Ri�W?

learning_rate_1��j7���rI       6%�	QJY;���A�$*;


total_lossT��@

error_R� L?

learning_rate_1��j7����I       6%�	m�Y;���A�$*;


total_loss�C�@

error_R�XC?

learning_rate_1��j7Y*�VI       6%�	��Y;���A�$*;


total_loss���@

error_R��K?

learning_rate_1��j7����I       6%�	�Z;���A�$*;


total_loss�Ź@

error_R��L?

learning_rate_1��j7"�zQI       6%�	[ZZ;���A�$*;


total_loss.�@

error_RGM?

learning_rate_1��j7���nI       6%�	��Z;���A�$*;


total_loss���@

error_R�Bd?

learning_rate_1��j7-X�I       6%�	��Z;���A�$*;


total_loss#�@

error_R�>e?

learning_rate_1��j7��UI       6%�	�#[;���A�$*;


total_loss�O�@

error_R��9?

learning_rate_1��j7̂"I       6%�	sh[;���A�$*;


total_loss-�@

error_RfRE?

learning_rate_1��j7��ޅI       6%�	V�[;���A�$*;


total_loss�;�@

error_R1�f?

learning_rate_1��j7x�ReI       6%�	H�[;���A�$*;


total_loss a�@

error_R��??

learning_rate_1��j7��RqI       6%�	�4\;���A�$*;


total_loss�w�@

error_RͦR?

learning_rate_1��j7�ijPI       6%�	�\;���A�$*;


total_loss�E�@

error_R!�S?

learning_rate_1��j7m2)I       6%�	)�\;���A�$*;


total_lossj��@

error_RaW?

learning_rate_1��j7q;�ZI       6%�	>];���A�$*;


total_loss׼�@

error_RͳP?

learning_rate_1��j7�o�I       6%�	�U];���A�$*;


total_lossH�@

error_R�	L?

learning_rate_1��j7��`I       6%�	��];���A�$*;


total_loss�h�@

error_R�/8?

learning_rate_1��j7?|W�I       6%�	E�];���A�$*;


total_loss�/�@

error_R�YW?

learning_rate_1��j7r��I       6%�	�5^;���A�$*;


total_lossnC�@

error_RX�P?

learning_rate_1��j7���I       6%�	��^;���A�$*;


total_lossȡA

error_R(�Y?

learning_rate_1��j7�܌�I       6%�	x�^;���A�$*;


total_loss���@

error_R$#T?

learning_rate_1��j7:Ą�I       6%�	(_;���A�$*;


total_loss��@

error_R�R?

learning_rate_1��j7���I       6%�	�`_;���A�$*;


total_loss�@

error_R�S?

learning_rate_1��j7wLU�I       6%�	��_;���A�$*;


total_loss�~�@

error_R��N?

learning_rate_1��j7���1I       6%�	��_;���A�$*;


total_lossj�f@

error_R�lP?

learning_rate_1��j7�?�I       6%�	�8`;���A�$*;


total_loss!�@

error_Rl<K?

learning_rate_1��j79��I       6%�	�{`;���A�$*;


total_loss���@

error_R�FN?

learning_rate_1��j7�v�dI       6%�	2�`;���A�$*;


total_loss16�@

error_R�~P?

learning_rate_1��j7�<��I       6%�	�a;���A�$*;


total_lossS�@

error_R4<?

learning_rate_1��j7���I       6%�	�Da;���A�$*;


total_loss[��@

error_RJ?

learning_rate_1��j7h{ܫI       6%�	�a;���A�$*;


total_loss���@

error_R)�<?

learning_rate_1��j7�=��I       6%�	��a;���A�$*;


total_loss|@�@

error_R=k;?

learning_rate_1��j7�$��I       6%�	[b;���A�$*;


total_loss��@

error_RE|F?

learning_rate_1��j7����I       6%�	�Vb;���A�$*;


total_loss�=�@

error_R�BM?

learning_rate_1��j7�)�TI       6%�	#�b;���A�$*;


total_loss��@

error_R$�^?

learning_rate_1��j7��!�I       6%�	z�b;���A�$*;


total_loss�1�@

error_RT�L?

learning_rate_1��j7׻�-I       6%�	C8c;���A�$*;


total_loss��A

error_R�Q?

learning_rate_1��j7Lu��I       6%�	�c;���A�$*;


total_loss �A

error_RjDF?

learning_rate_1��j7���I       6%�	C�c;���A�$*;


total_loss���@

error_R��X?

learning_rate_1��j7���I       6%�	Yd;���A�$*;


total_loss��@

error_RN�V?

learning_rate_1��j7@��I       6%�		Sd;���A�$*;


total_loss�`�@

error_R�CP?

learning_rate_1��j7*)�I       6%�	N�d;���A�$*;


total_loss��A

error_RH�_?

learning_rate_1��j7X 
I       6%�	��d;���A�$*;


total_lossm��@

error_R�H]?

learning_rate_1��j7b·yI       6%�	H"e;���A�$*;


total_loss�+�@

error_R�q@?

learning_rate_1��j7���GI       6%�	�ee;���A�$*;


total_loss ��@

error_RP?

learning_rate_1��j7'X:I       6%�	3�e;���A�$*;


total_loss���@

error_R;�@?

learning_rate_1��j7Ӂi�I       6%�	m�e;���A�$*;


total_losso�@

error_R��D?

learning_rate_1��j7H1f�I       6%�	�Nf;���A�$*;


total_loss�Օ@

error_R�jM?

learning_rate_1��j7��<I       6%�	$�f;���A�$*;


total_loss���@

error_R��B?

learning_rate_1��j7�﷩I       6%�	�g;���A�$*;


total_loss!�@

error_R!(Y?

learning_rate_1��j72�I       6%�	Qg;���A�$*;


total_lossm��@

error_R�C?

learning_rate_1��j7��I       6%�	{�g;���A�$*;


total_losss��@

error_R�S?

learning_rate_1��j7K<�I       6%�	��g;���A�$*;


total_loss��v@

error_R�M?

learning_rate_1��j7���iI       6%�	�'h;���A�$*;


total_lossX�@

error_RW�W?

learning_rate_1��j7+�I       6%�	}uh;���A�$*;


total_loss���@

error_R#�C?

learning_rate_1��j7��D@I       6%�	O�h;���A�$*;


total_loss_g�@

error_R�oB?

learning_rate_1��j7�y��I       6%�	i;���A�$*;


total_loss�|�@

error_R�KD?

learning_rate_1��j7���I       6%�	�Ki;���A�$*;


total_loss��@

error_R��J?

learning_rate_1��j7W+vI       6%�	 �i;���A�$*;


total_loss��@

error_Rn�O?

learning_rate_1��j7U�I       6%�	��i;���A�$*;


total_loss��@

error_R WL?

learning_rate_1��j7��!I       6%�	�j;���A�$*;


total_lossWg�@

error_R\�h?

learning_rate_1��j7V{�I       6%�	DZj;���A�$*;


total_loss{�@

error_R��^?

learning_rate_1��j7 M.I       6%�	>�j;���A�$*;


total_loss�G�@

error_RҰD?

learning_rate_1��j7�hCgI       6%�	��j;���A�$*;


total_loss)��@

error_R][S?

learning_rate_1��j7 rI       6%�	1)k;���A�$*;


total_loss��@

error_R�JK?

learning_rate_1��j7~Q��I       6%�	&nk;���A�$*;


total_loss�L�@

error_RRQK?

learning_rate_1��j7"	��I       6%�	�k;���A�$*;


total_loss���@

error_R�F?

learning_rate_1��j7�{�XI       6%�	��k;���A�$*;


total_loss���@

error_R��H?

learning_rate_1��j7�⃪I       6%�	o=l;���A�$*;


total_lossL�@

error_R�ar?

learning_rate_1��j7|��I       6%�	�l;���A�$*;


total_loss͞@

error_R�x@?

learning_rate_1��j7�g�"I       6%�	��l;���A�$*;


total_loss(��@

error_Rq�L?

learning_rate_1��j7+OQ�I       6%�	
m;���A�$*;


total_loss9�@

error_R��B?

learning_rate_1��j7����I       6%�	�Pm;���A�$*;


total_lossT@�@

error_R$KK?

learning_rate_1��j7��vI       6%�	!�m;���A�$*;


total_lossn��@

error_R�R?

learning_rate_1��j7/�}I       6%�	��m;���A�$*;


total_loss���@

error_R;iA?

learning_rate_1��j7�j�I       6%�	�7n;���A�$*;


total_loss��@

error_R�J[?

learning_rate_1��j7��t�I       6%�	Ԏn;���A�$*;


total_lossn��@

error_R_W?

learning_rate_1��j7tW�I       6%�	��n;���A�$*;


total_loss��@

error_R��Y?

learning_rate_1��j7z"#I       6%�	k*o;���A�$*;


total_lossp�@

error_R� I?

learning_rate_1��j7@�I       6%�	>uo;���A�$*;


total_loss�a�@

error_RM�N?

learning_rate_1��j7JV�FI       6%�	��o;���A�$*;


total_loss�x�@

error_R��R?

learning_rate_1��j7�r�I       6%�	p;���A�$*;


total_loss�=�@

error_R)�D?

learning_rate_1��j7��o;I       6%�	�Ep;���A�$*;


total_loss�@

error_R�Y?

learning_rate_1��j7S�l�I       6%�	`�p;���A�$*;


total_lossO��@

error_R�kC?

learning_rate_1��j7RcpI       6%�	��p;���A�$*;


total_loss� �@

error_R<L]?

learning_rate_1��j7��9I       6%�	�q;���A�$*;


total_loss<��@

error_R{�E?

learning_rate_1��j7��9UI       6%�	8[q;���A�$*;


total_loss��@

error_R�MT?

learning_rate_1��j7���aI       6%�	�q;���A�$*;


total_lossr��@

error_R��<?

learning_rate_1��j7)��7I       6%�	H�q;���A�$*;


total_loss|��@

error_Rl&`?

learning_rate_1��j7]� �I       6%�	�'r;���A�$*;


total_loss���@

error_R.sR?

learning_rate_1��j7��i�I       6%�	Kjr;���A�$*;


total_lossS��@

error_R�>U?

learning_rate_1��j7xw|�I       6%�	Юr;���A�$*;


total_loss%��@

error_R�	A?

learning_rate_1��j7L�@I       6%�	J�r;���A�$*;


total_lossI��@

error_RD?

learning_rate_1��j7b�K�I       6%�	�5s;���A�$*;


total_losssh�@

error_R�E?

learning_rate_1��j7�;rDI       6%�	yxs;���A�$*;


total_loss�x�@

error_R�AC?

learning_rate_1��j7 ��wI       6%�	x�s;���A�$*;


total_loss�@

error_Rn�Z?

learning_rate_1��j7�R�jI       6%�	pt;���A�$*;


total_lossT�@

error_R��T?

learning_rate_1��j7b��
I       6%�	�Pt;���A�$*;


total_lossHͻ@

error_R��L?

learning_rate_1��j7K�0hI       6%�	��t;���A�$*;


total_loss|��@

error_R�P?

learning_rate_1��j7�(>�I       6%�	��t;���A�$*;


total_loss�W�@

error_R�^`?

learning_rate_1��j7�ԳI       6%�	B.u;���A�$*;


total_loss&[�@

error_R��F?

learning_rate_1��j7w��I       6%�	eyu;���A�$*;


total_loss��@

error_Rj�M?

learning_rate_1��j7���I       6%�	��u;���A�$*;


total_lossV�@

error_R�L=?

learning_rate_1��j7�H�I       6%�	v;���A�%*;


total_lossh�@

error_R8�Y?

learning_rate_1��j7�Y�I       6%�	�sv;���A�%*;


total_loss��@

error_RñS?

learning_rate_1��j7��~I       6%�	��v;���A�%*;


total_loss>�@

error_R,6E?

learning_rate_1��j7C�.I       6%�	�w;���A�%*;


total_loss�kA

error_R��C?

learning_rate_1��j7�H��I       6%�	�Iw;���A�%*;


total_loss�s�@

error_R��P?

learning_rate_1��j7����I       6%�	��w;���A�%*;


total_loss ��@

error_R�O?

learning_rate_1��j7�A{mI       6%�	:�w;���A�%*;


total_loss��@

error_Ri�_?

learning_rate_1��j7g�N�I       6%�	�x;���A�%*;


total_loss�ʩ@

error_Rs�Q?

learning_rate_1��j7�,�I       6%�	�[x;���A�%*;


total_lossc�@

error_RY?

learning_rate_1��j7���I       6%�	s�x;���A�%*;


total_lossj&�@

error_RvM?

learning_rate_1��j7��@7I       6%�	��x;���A�%*;


total_lossF��@

error_R��5?

learning_rate_1��j7<��I       6%�	�/y;���A�%*;


total_loss���@

error_R&0F?

learning_rate_1��j7*gݬI       6%�	�uy;���A�%*;


total_loss֊�@

error_Rx�.?

learning_rate_1��j7���I       6%�	c�y;���A�%*;


total_loss��@

error_R��O?

learning_rate_1��j7ZJtI       6%�	��y;���A�%*;


total_lossnׂ@

error_R��O?

learning_rate_1��j7�9$�I       6%�	�>z;���A�%*;


total_loss�@

error_R�$_?

learning_rate_1��j7�u��I       6%�	#�z;���A�%*;


total_loss��@

error_R�U?

learning_rate_1��j7�y PI       6%�	��z;���A�%*;


total_loss��A

error_R)�J?

learning_rate_1��j7_PoMI       6%�	�{;���A�%*;


total_loss!��@

error_RZC?

learning_rate_1��j7�,�I       6%�		Q{;���A�%*;


total_loss�Ą@

error_R�#O?

learning_rate_1��j7#�4vI       6%�	��{;���A�%*;


total_loss�U�@

error_R�'T?

learning_rate_1��j7[TB�I       6%�	��{;���A�%*;


total_loss>�@

error_R�CB?

learning_rate_1��j7|���I       6%�	R|;���A�%*;


total_loss$"�@

error_R��^?

learning_rate_1��j7��~�I       6%�	K`|;���A�%*;


total_loss�Œ@

error_R��7?

learning_rate_1��j7�]��I       6%�	*�|;���A�%*;


total_loss�PA

error_R�O?

learning_rate_1��j7�Y�I       6%�	`�|;���A�%*;


total_loss!��@

error_R�	S?

learning_rate_1��j7/vhI       6%�	P.};���A�%*;


total_loss�&�@

error_R��A?

learning_rate_1��j7�y�I       6%�	�r};���A�%*;


total_loss�է@

error_R��@?

learning_rate_1��j7�U@�I       6%�	i�};���A�%*;


total_loss���@

error_R�8?

learning_rate_1��j7����I       6%�	s�};���A�%*;


total_loss��@

error_R�u[?

learning_rate_1��j7���	I       6%�	�C~;���A�%*;


total_loss���@

error_Rl�U?

learning_rate_1��j7��^I       6%�	��~;���A�%*;


total_loss���@

error_R�\N?

learning_rate_1��j7G�I       6%�	��~;���A�%*;


total_loss�;�@

error_R�yg?

learning_rate_1��j7�(#I       6%�	�$;���A�%*;


total_loss��@

error_Rx�U?

learning_rate_1��j7�eI       6%�	�q;���A�%*;


total_loss@�s@

error_RAH?

learning_rate_1��j7��[I       6%�	˻;���A�%*;


total_loss�خ@

error_R�D?

learning_rate_1��j7�g�I       6%�	��;���A�%*;


total_loss�v@

error_R��I?

learning_rate_1��j7��B�I       6%�	qP�;���A�%*;


total_loss���@

error_R(�L?

learning_rate_1��j7��*"I       6%�	�;���A�%*;


total_loss��@

error_R��M?

learning_rate_1��j7�ѭI       6%�	��;���A�%*;


total_loss5�@

error_R�R?

learning_rate_1��j7ݮ�I       6%�	�+�;���A�%*;


total_loss1�A

error_R��N?

learning_rate_1��j7ZaI       6%�	Ht�;���A�%*;


total_lossN�@

error_R�V?

learning_rate_1��j73���I       6%�	{��;���A�%*;


total_loss�e�@

error_R�Q?

learning_rate_1��j7{���I       6%�	��;���A�%*;


total_lossӸ�@

error_R�B?

learning_rate_1��j7=G:�I       6%�	@�;���A�%*;


total_lossB� A

error_RdQ?

learning_rate_1��j7H�!=I       6%�	ڂ�;���A�%*;


total_loss¬�@

error_R	R?

learning_rate_1��j7\�=I       6%�	�˂;���A�%*;


total_loss��A

error_R��D?

learning_rate_1��j7��0\I       6%�	��;���A�%*;


total_lossq��@

error_R,�K?

learning_rate_1��j7o��NI       6%�	�a�;���A�%*;


total_loss��@

error_R�&Y?

learning_rate_1��j7���I       6%�	�;���A�%*;


total_loss@�@

error_R�M?

learning_rate_1��j7�S+`I       6%�	a��;���A�%*;


total_lossz��@

error_R�,R?

learning_rate_1��j7`I       6%�	�7�;���A�%*;


total_loss<��@

error_R#�T?

learning_rate_1��j7��F�I       6%�	N|�;���A�%*;


total_loss̑�@

error_R�R?

learning_rate_1��j7*sGI       6%�	��;���A�%*;


total_loss �@

error_R�O@?

learning_rate_1��j736oI       6%�	��;���A�%*;


total_loss�5�@

error_R�hP?

learning_rate_1��j7�d��I       6%�	8J�;���A�%*;


total_loss�L�@

error_Rn�[?

learning_rate_1��j7��`II       6%�	O��;���A�%*;


total_loss�2�@

error_RJ�V?

learning_rate_1��j7"�0I       6%�	"υ;���A�%*;


total_loss*�@

error_R��C?

learning_rate_1��j7"���I       6%�	3�;���A�%*;


total_lossX��@

error_R�GQ?

learning_rate_1��j7���I       6%�	�r�;���A�%*;


total_loss�c�@

error_R�T?

learning_rate_1��j7���I       6%�	[��;���A�%*;


total_lossq��@

error_R�{B?

learning_rate_1��j7���I       6%�	���;���A�%*;


total_loss;��@

error_RڪJ?

learning_rate_1��j79��#I       6%�	�B�;���A�%*;


total_loss.p�@

error_R �J?

learning_rate_1��j7"��3I       6%�	|��;���A�%*;


total_loss�@�@

error_R6�I?

learning_rate_1��j7.��EI       6%�	�͇;���A�%*;


total_loss&�@

error_R]�F?

learning_rate_1��j7�}��I       6%�	��;���A�%*;


total_lossĎ@

error_R)�B?

learning_rate_1��j7��fI       6%�	U�;���A�%*;


total_lossfD�@

error_R��`?

learning_rate_1��j7?�v�I       6%�	���;���A�%*;


total_loss�[�@

error_R� V?

learning_rate_1��j7��!zI       6%�	�܈;���A�%*;


total_loss�y�@

error_Ri+E?

learning_rate_1��j7%?#�I       6%�	b�;���A�%*;


total_losst%�@

error_RXIP?

learning_rate_1��j7���I       6%�	�a�;���A�%*;


total_loss;�@

error_R
�N?

learning_rate_1��j7���I       6%�	���;���A�%*;


total_lossD��@

error_R�eE?

learning_rate_1��j7�2 I       6%�	L�;���A�%*;


total_losse A

error_R\�V?

learning_rate_1��j7&��I       6%�	�0�;���A�%*;


total_lossݣ@

error_RT�Z?

learning_rate_1��j7���ZI       6%�	�q�;���A�%*;


total_loss3S�@

error_R��6?

learning_rate_1��j7g�`�I       6%�	���;���A�%*;


total_loss*Ǵ@

error_R��]?

learning_rate_1��j7d�k\I       6%�	��;���A�%*;


total_loss,`�@

error_R��Q?

learning_rate_1��j7��5I       6%�	vI�;���A�%*;


total_lossڜ�@

error_RT=U?

learning_rate_1��j7�ǫI       6%�	��;���A�%*;


total_loss��@

error_R��A?

learning_rate_1��j7F�lXI       6%�	�݋;���A�%*;


total_loss*��@

error_R�J?

learning_rate_1��j7n~��I       6%�	|$�;���A�%*;


total_loss�y�@

error_R	�N?

learning_rate_1��j7�PB%I       6%�	di�;���A�%*;


total_loss?�A

error_R�D5?

learning_rate_1��j7	ZI       6%�	��;���A�%*;


total_lossF�A

error_R��a?

learning_rate_1��j7	X2I       6%�	;��;���A�%*;


total_loss��@

error_R��D?

learning_rate_1��j7;|�I       6%�	+E�;���A�%*;


total_loss�8�@

error_R2a?

learning_rate_1��j7�VI       6%�	8��;���A�%*;


total_loss���@

error_R� U?

learning_rate_1��j7-�lI       6%�	��;���A�%*;


total_lossz��@

error_R�YT?

learning_rate_1��j7�C�I       6%�	{+�;���A�%*;


total_lossݒ@

error_R��D?

learning_rate_1��j7	���I       6%�	�v�;���A�%*;


total_losso��@

error_R_X?

learning_rate_1��j7���I       6%�	�ˎ;���A�%*;


total_loss��@

error_R3�H?

learning_rate_1��j7��VmI       6%�	B�;���A�%*;


total_loss@Ü@

error_Rn�K?

learning_rate_1��j7���I       6%�	�W�;���A�%*;


total_loss�S�@

error_R�U?

learning_rate_1��j7ēI       6%�	˙�;���A�%*;


total_loss�a�@

error_R L[?

learning_rate_1��j7<ke�I       6%�	9ߏ;���A�%*;


total_loss�Ϩ@

error_R�<N?

learning_rate_1��j7Iz"I       6%�	&�;���A�%*;


total_lossݭ�@

error_R�6O?

learning_rate_1��j7��I       6%�	`q�;���A�%*;


total_loss�[�@

error_R��:?

learning_rate_1��j7;��iI       6%�	��;���A�%*;


total_loss�8�@

error_Re�B?

learning_rate_1��j7U��I       6%�	���;���A�%*;


total_loss��@

error_Ro9O?

learning_rate_1��j7���I       6%�	,I�;���A�%*;


total_lossIѾ@

error_R�3?

learning_rate_1��j7%瀶I       6%�	Е�;���A�%*;


total_lossOA

error_R��P?

learning_rate_1��j7,+I       6%�	ܑ;���A�%*;


total_loss�*�@

error_R=�U?

learning_rate_1��j7��kI       6%�	f&�;���A�%*;


total_lossB�@

error_R��N?

learning_rate_1��j7�;�I       6%�	Yt�;���A�%*;


total_loss=z�@

error_R�'Y?

learning_rate_1��j7��o�I       6%�	t��;���A�%*;


total_loss\}�@

error_R�[?

learning_rate_1��j7��I       6%�	�;���A�%*;


total_loss�@

error_R�zS?

learning_rate_1��j7j9�I       6%�	�F�;���A�%*;


total_losso�
A

error_R=�J?

learning_rate_1��j7BĭI       6%�	Ɉ�;���A�%*;


total_loss���@

error_Rv\M?

learning_rate_1��j7�6��I       6%�	�͓;���A�%*;


total_loss���@

error_R3�??

learning_rate_1��j7b[}`I       6%�	�;���A�%*;


total_loss�[�@

error_R�C?

learning_rate_1��j7�#�vI       6%�	�\�;���A�%*;


total_lossv�m@

error_R15F?

learning_rate_1��j7� C�I       6%�	���;���A�%*;


total_loss�͖@

error_R6�E?

learning_rate_1��j7 �d�I       6%�	���;���A�%*;


total_loss=�@

error_R$�g?

learning_rate_1��j7�(� I       6%�	�8�;���A�%*;


total_loss��A

error_RA/Z?

learning_rate_1��j7�ܝ�I       6%�	�~�;���A�%*;


total_loss���@

error_R��??

learning_rate_1��j7�TI       6%�	(ĕ;���A�%*;


total_loss�4�@

error_RaB\?

learning_rate_1��j7?�C�I       6%�	D
�;���A�%*;


total_loss�x�@

error_R[	V?

learning_rate_1��j7R�PI       6%�	#j�;���A�%*;


total_loss��@

error_Rv�I?

learning_rate_1��j7 �(�I       6%�	䴖;���A�%*;


total_loss�;�@

error_R�4K?

learning_rate_1��j7\$"iI       6%�	���;���A�%*;


total_loss��@

error_R c?

learning_rate_1��j7
K��I       6%�	�<�;���A�%*;


total_loss��@

error_R�IN?

learning_rate_1��j7�.J�I       6%�	���;���A�%*;


total_loss�X�@

error_R6�A?

learning_rate_1��j7^^�I       6%�	�;���A�%*;


total_loss�߳@

error_Rt�L?

learning_rate_1��j7%�E�I       6%�	��;���A�%*;


total_lossW,�@

error_R:�Y?

learning_rate_1��j7���I       6%�	SM�;���A�%*;


total_loss~�@

error_R�3^?

learning_rate_1��j7p(@�I       6%�	���;���A�%*;


total_lossHD�@

error_ROU?

learning_rate_1��j7���I       6%�	Dߘ;���A�%*;


total_loss���@

error_RfP?

learning_rate_1��j7gtkYI       6%�	i%�;���A�%*;


total_loss&G�@

error_RO�^?

learning_rate_1��j7�{�I       6%�	�j�;���A�%*;


total_loss�y�@

error_RWJ?

learning_rate_1��j7�u�KI       6%�	���;���A�&*;


total_loss���@

error_R�&T?

learning_rate_1��j7��ߟI       6%�	���;���A�&*;


total_loss@�@

error_R��\?

learning_rate_1��j7|��I       6%�	:<�;���A�&*;


total_loss��@

error_R�G?

learning_rate_1��j7��^�I       6%�	B��;���A�&*;


total_loss?�@

error_R�5H?

learning_rate_1��j7����I       6%�	�Ě;���A�&*;


total_loss!�@

error_R��??

learning_rate_1��j7�+��I       6%�	��;���A�&*;


total_loss���@

error_R��@?

learning_rate_1��j7�W\ I       6%�	+J�;���A�&*;


total_lossq`�@

error_R:�F?

learning_rate_1��j7�˄I       6%�	���;���A�&*;


total_loss*L�@

error_R@GK?

learning_rate_1��j7�W�I       6%�	�͛;���A�&*;


total_lossh�@

error_R��N?

learning_rate_1��j7
گ�I       6%�	��;���A�&*;


total_loss�	A

error_R�{<?

learning_rate_1��j7Ǣ?I       6%�	�T�;���A�&*;


total_lossSBA

error_R�^\?

learning_rate_1��j7�X[RI       6%�	���;���A�&*;


total_loss3��@

error_R6�L?

learning_rate_1��j7zMZ�I       6%�	�ޜ;���A�&*;


total_lossCF�@

error_R�u^?

learning_rate_1��j7�c��I       6%�	�$�;���A�&*;


total_loss���@

error_R �V?

learning_rate_1��j7t���I       6%�	j�;���A�&*;


total_loss
��@

error_R+J?

learning_rate_1��j7E�fI       6%�	���;���A�&*;


total_loss�Ŕ@

error_R2>\?

learning_rate_1��j7_o%�I       6%�	�;���A�&*;


total_loss�-�@

error_RN>?

learning_rate_1��j7,��|I       6%�	}.�;���A�&*;


total_lossO�p@

error_RC#M?

learning_rate_1��j7SP�<I       6%�	�q�;���A�&*;


total_loss ��@

error_R��F?

learning_rate_1��j7 ��sI       6%�	0��;���A�&*;


total_lossJ7�@

error_R6WK?

learning_rate_1��j7�.�I       6%�	��;���A�&*;


total_loss\#�@

error_R��E?

learning_rate_1��j7���I       6%�	�M�;���A�&*;


total_loss�ޝ@

error_R7�E?

learning_rate_1��j7&R�;I       6%�	t��;���A�&*;


total_loss�s�@

error_Ri�>?

learning_rate_1��j7�`|�I       6%�	�;���A�&*;


total_loss<�@

error_R �K?

learning_rate_1��j7��OmI       6%�	�0�;���A�&*;


total_loss��@

error_R��C?

learning_rate_1��j7�-ÝI       6%�	�w�;���A�&*;


total_loss���@

error_R�5?

learning_rate_1��j7��ãI       6%�	1��;���A�&*;


total_lossTtA

error_R��C?

learning_rate_1��j7���I       6%�	:�;���A�&*;


total_loss�A

error_R84[?

learning_rate_1��j7`�WI       6%�	J�;���A�&*;


total_lossZ��@

error_R��j?

learning_rate_1��j7��I       6%�	Ô�;���A�&*;


total_loss(��@

error_R*T>?

learning_rate_1��j7�b[I       6%�	�ܡ;���A�&*;


total_loss���@

error_R\�<?

learning_rate_1��j7�R�>I       6%�	,*�;���A�&*;


total_loss��@

error_Rf�R?

learning_rate_1��j7��I       6%�	�t�;���A�&*;


total_lossx�@

error_R�}A?

learning_rate_1��j7]p�I       6%�	o��;���A�&*;


total_lossx�@

error_R�A?

learning_rate_1��j7k##I       6%�	� �;���A�&*;


total_loss�d�@

error_R�cD?

learning_rate_1��j7��ǝI       6%�	�H�;���A�&*;


total_loss�k�@

error_RC�]?

learning_rate_1��j7(�s`I       6%�	���;���A�&*;


total_loss'$�@

error_R��W?

learning_rate_1��j7,\u�I       6%�	yΣ;���A�&*;


total_loss��@

error_R�?K?

learning_rate_1��j7�зI       6%�	(�;���A�&*;


total_loss#�A

error_R\�N?

learning_rate_1��j7���I       6%�	�U�;���A�&*;


total_lossʹ�@

error_R�MQ?

learning_rate_1��j7���I       6%�	��;���A�&*;


total_loss�*�@

error_R�H?

learning_rate_1��j7	��I       6%�	j�;���A�&*;


total_loss��A

error_Rl�P?

learning_rate_1��j7`�T�I       6%�	(�;���A�&*;


total_loss�Я@

error_R�h?

learning_rate_1��j7�uCI       6%�	�l�;���A�&*;


total_lossF� A

error_R=dM?

learning_rate_1��j7��?�I       6%�	���;���A�&*;


total_loss)aA

error_R.;I?

learning_rate_1��j7����I       6%�	���;���A�&*;


total_loss� A

error_R�Q?

learning_rate_1��j7E�=I       6%�	�Y�;���A�&*;


total_loss��@

error_R	W?

learning_rate_1��j7�f��I       6%�	]��;���A�&*;


total_loss�/�@

error_Rz�U?

learning_rate_1��j7���I       6%�	��;���A�&*;


total_loss���@

error_R��W?

learning_rate_1��j7b;�I       6%�	T'�;���A�&*;


total_loss�y�@

error_R��W?

learning_rate_1��j7*��I       6%�	�m�;���A�&*;


total_loss�U�@

error_R!tS?

learning_rate_1��j7��nI       6%�	Ʋ�;���A�&*;


total_loss�7�@

error_Rdd??

learning_rate_1��j7S��I       6%�	���;���A�&*;


total_loss0�@

error_R�R?

learning_rate_1��j7�@vI       6%�	D@�;���A�&*;


total_lossy1A

error_R:�@?

learning_rate_1��j7��I       6%�	킨;���A�&*;


total_lossN�@

error_R��N?

learning_rate_1��j7��WI       6%�	~ͨ;���A�&*;


total_loss���@

error_RrC?

learning_rate_1��j7�HnvI       6%�	&�;���A�&*;


total_loss�љ@

error_R,�D?

learning_rate_1��j7�O�wI       6%�	�a�;���A�&*;


total_loss���@

error_R��S?

learning_rate_1��j7�w�I       6%�	֧�;���A�&*;


total_loss�q@

error_R�rE?

learning_rate_1��j7BI       6%�	?�;���A�&*;


total_lossL��@

error_R��M?

learning_rate_1��j7�zsI       6%�	�5�;���A�&*;


total_lossu%�@

error_Rd�]?

learning_rate_1��j7��Q�I       6%�	O{�;���A�&*;


total_lossM�@

error_R�R?

learning_rate_1��j7(�$�I       6%�	��;���A�&*;


total_loss��A

error_R�T?

learning_rate_1��j7��)I       6%�	��;���A�&*;


total_loss��@

error_RԦU?

learning_rate_1��j7D�$�I       6%�	}I�;���A�&*;


total_loss��@

error_R��Z?

learning_rate_1��j7���I       6%�	���;���A�&*;


total_lossO�@

error_R��T?

learning_rate_1��j7``�I       6%�	�ӫ;���A�&*;


total_loss:�@

error_RL?

learning_rate_1��j7�Ta I       6%�	��;���A�&*;


total_loss	�@

error_R�}B?

learning_rate_1��j7J�pI       6%�	tf�;���A�&*;


total_loss�q�@

error_RJ"R?

learning_rate_1��j7�%��I       6%�	��;���A�&*;


total_lossw �@

error_R8GU?

learning_rate_1��j7�'�I       6%�	��;���A�&*;


total_loss���@

error_R��[?

learning_rate_1��j7@x��I       6%�	S8�;���A�&*;


total_loss΁�@

error_R�"@?

learning_rate_1��j7�LgI       6%�	���;���A�&*;


total_lossİ�@

error_R�
^?

learning_rate_1��j71�;cI       6%�	�ѭ;���A�&*;


total_lossc&�@

error_R��>?

learning_rate_1��j7���SI       6%�	��;���A�&*;


total_loss.�@

error_R7�T?

learning_rate_1��j7��1I       6%�	3\�;���A�&*;


total_loss�_�@

error_R��P?

learning_rate_1��j7<z��I       6%�	]î;���A�&*;


total_lossIV�@

error_R,�T?

learning_rate_1��j71'LI       6%�	��;���A�&*;


total_loss�@

error_R�K?

learning_rate_1��j7�O��I       6%�	7Q�;���A�&*;


total_loss��A

error_R�@F?

learning_rate_1��j7]١�I       6%�		��;���A�&*;


total_lossm�A

error_R��G?

learning_rate_1��j7�L=I       6%�	��;���A�&*;


total_loss�<�@

error_R�+Q?

learning_rate_1��j7#:�I       6%�	+�;���A�&*;


total_loss�w�@

error_R�'A?

learning_rate_1��j7qW�I       6%�	Hq�;���A�&*;


total_loss�^�@

error_R�P?

learning_rate_1��j7͝��I       6%�	���;���A�&*;


total_loss׽s@

error_R)lL?

learning_rate_1��j7h^I       6%�	��;���A�&*;


total_loss��A

error_R��L?

learning_rate_1��j7��7I       6%�	HM�;���A�&*;


total_loss6u�@

error_R�HH?

learning_rate_1��j7f�8*I       6%�	��;���A�&*;


total_loss$�@

error_R_ML?

learning_rate_1��j7Z��I       6%�	}Ա;���A�&*;


total_loss)�@

error_R.bU?

learning_rate_1��j7��-I       6%�	K�;���A�&*;


total_loss��A

error_Ro>?

learning_rate_1��j7�,�oI       6%�	�]�;���A�&*;


total_loss���@

error_RdL@?

learning_rate_1��j7�G�7I       6%�	䡲;���A�&*;


total_loss��@

error_R$iJ?

learning_rate_1��j7t�$I       6%�	��;���A�&*;


total_loss���@

error_R�*R?

learning_rate_1��j7$��LI       6%�	�-�;���A�&*;


total_loss4��@

error_R�V?

learning_rate_1��j7�ܸI       6%�	�z�;���A�&*;


total_loss��@

error_R��??

learning_rate_1��j7�GKiI       6%�	���;���A�&*;


total_lossh�@

error_R�I?

learning_rate_1��j7&��I       6%�	�	�;���A�&*;


total_loss�*�@

error_R�T?

learning_rate_1��j7yC�I       6%�	R�;���A�&*;


total_loss�d^@

error_R�BU?

learning_rate_1��j7U+�I       6%�	=��;���A�&*;


total_loss!~@

error_Rs�S?

learning_rate_1��j7\B��I       6%�	E�;���A�&*;


total_loss���@

error_R�a?

learning_rate_1��j7���6I       6%�	i#�;���A�&*;


total_lossR�@

error_R��g?

learning_rate_1��j7/C@�I       6%�	og�;���A�&*;


total_loss�9�@

error_R,>O?

learning_rate_1��j7��@�I       6%�	q��;���A�&*;


total_loss���@

error_Rv	E?

learning_rate_1��j7�>H�I       6%�	���;���A�&*;


total_loss�v�@

error_RC�I?

learning_rate_1��j7\!�I       6%�	�D�;���A�&*;


total_loss<��@

error_R�bE?

learning_rate_1��j7�W��I       6%�	ۣ�;���A�&*;


total_loss���@

error_R�.N?

learning_rate_1��j7�,�I       6%�	��;���A�&*;


total_loss���@

error_R��S?

learning_rate_1��j7��I       6%�	�-�;���A�&*;


total_loss5M�@

error_RLI?

learning_rate_1��j7�)j�I       6%�	�o�;���A�&*;


total_lossG��@

error_R6wG?

learning_rate_1��j7|+I       6%�	۲�;���A�&*;


total_lossQ��@

error_R��[?

learning_rate_1��j7��֪I       6%�	Y��;���A�&*;


total_lossz�@

error_R�eT?

learning_rate_1��j72㹚I       6%�	�<�;���A�&*;


total_lossֽ�@

error_R�A?

learning_rate_1��j7��2�I       6%�	��;���A�&*;


total_lossMz�@

error_R�+J?

learning_rate_1��j7���FI       6%�	?ĸ;���A�&*;


total_loss��@

error_R!lK?

learning_rate_1��j7A0�I       6%�	W�;���A�&*;


total_lossnA

error_R��K?

learning_rate_1��j7���gI       6%�	L�;���A�&*;


total_loss u�@

error_RvQ?

learning_rate_1��j7�\I�I       6%�	���;���A�&*;


total_loss.U�@

error_R�;R?

learning_rate_1��j7���4I       6%�	�ӹ;���A�&*;


total_lossƤ�@

error_R�.>?

learning_rate_1��j7��W�I       6%�	��;���A�&*;


total_loss�$�@

error_R{�A?

learning_rate_1��j7N��-I       6%�	:e�;���A�&*;


total_loss���@

error_Rv�H?

learning_rate_1��j7�yxI       6%�	"��;���A�&*;


total_loss�� A

error_RM�C?

learning_rate_1��j7X��.I       6%�	��;���A�&*;


total_lossWe�@

error_R|J?

learning_rate_1��j7@�I       6%�	N9�;���A�&*;


total_lossQ�@

error_R�iR?

learning_rate_1��j7�?�I       6%�	�}�;���A�&*;


total_lossI��@

error_R�0P?

learning_rate_1��j7��bI       6%�	û;���A�&*;


total_loss��@

error_Ra�R?

learning_rate_1��j7/�_@I       6%�	�;���A�&*;


total_loss�l�@

error_RR|E?

learning_rate_1��j7M���I       6%�	vX�;���A�&*;


total_loss<�A

error_R��W?

learning_rate_1��j7��I       6%�	��;���A�&*;


total_loss���@

error_R�H?

learning_rate_1��j7}�I       6%�	��;���A�&*;


total_loss6ƚ@

error_RZS?

learning_rate_1��j7�c#�I       6%�	�6�;���A�'*;


total_loss7�@

error_R��U?

learning_rate_1��j7����I       6%�	��;���A�'*;


total_loss�P�@

error_R�>C?

learning_rate_1��j7%S*�I       6%�	ǽ;���A�'*;


total_lossF��@

error_RK?

learning_rate_1��j7w���I       6%�	~�;���A�'*;


total_loss殞@

error_RM�X?

learning_rate_1��j7o�~UI       6%�	�W�;���A�'*;


total_lossn��@

error_R7�:?

learning_rate_1��j7�ɧ�I       6%�	q��;���A�'*;


total_loss.�@

error_R�`_?

learning_rate_1��j7#*� I       6%�	��;���A�'*;


total_lossc�A

error_R�<?

learning_rate_1��j7R�OI       6%�	�+�;���A�'*;


total_loss�t@

error_R�J?

learning_rate_1��j7���I       6%�	�p�;���A�'*;


total_loss
�@

error_R��m?

learning_rate_1��j7�/plI       6%�	���;���A�'*;


total_lossq�@

error_R��N?

learning_rate_1��j7�{?I       6%�	���;���A�'*;


total_lossz�@

error_R|QM?

learning_rate_1��j73��5I       6%�	�%�;���A�'*;


total_loss��;A

error_R��R?

learning_rate_1��j7FT4�I       6%�	�l�;���A�'*;


total_lossc��@

error_R<z^?

learning_rate_1��j71#ξI       6%�	Ƶ�;���A�'*;


total_lossߧ�@

error_RȑJ?

learning_rate_1��j7˰VI       6%�	�;���A�'*;


total_lossn��@

error_RI�I?

learning_rate_1��j75��I       6%�	�G�;���A�'*;


total_loss��@

error_R�nW?

learning_rate_1��j7ET�#I       6%�	ފ�;���A�'*;


total_lossa��@

error_RS�I?

learning_rate_1��j7��JI       6%�	���;���A�'*;


total_loss�pA

error_R��G?

learning_rate_1��j7���I       6%�	+�;���A�'*;


total_loss��@

error_RƛE?

learning_rate_1��j7�A�I       6%�	�[�;���A�'*;


total_loss4.�@

error_Rd�D?

learning_rate_1��j7#�ܣI       6%�	%��;���A�'*;


total_loss�u�@

error_R�o9?

learning_rate_1��j7�A�)I       6%�	���;���A�'*;


total_loss4`�@

error_R��Z?

learning_rate_1��j7M I       6%�	�0�;���A�'*;


total_loss��@

error_R(iF?

learning_rate_1��j7�G�*I       6%�	z��;���A�'*;


total_loss��@

error_R��Y?

learning_rate_1��j7�/��I       6%�	b��;���A�'*;


total_losss��@

error_RP?

learning_rate_1��j7m\�I       6%�	��;���A�'*;


total_loss�=�@

error_R�QO?

learning_rate_1��j7EHV�I       6%�	3[�;���A�'*;


total_lossɥ�@

error_R��F?

learning_rate_1��j7Ȁ��I       6%�	��;���A�'*;


total_loss:(�@

error_R�WL?

learning_rate_1��j7H��xI       6%�	���;���A�'*;


total_lossт�@

error_R��H?

learning_rate_1��j7�vv�I       6%�	$�;���A�'*;


total_lossfi�@

error_R6)M?

learning_rate_1��j7A��UI       6%�	�e�;���A�'*;


total_loss@�@

error_R��`?

learning_rate_1��j7~���I       6%�	���;���A�'*;


total_lossg�@

error_R6�]?

learning_rate_1��j7Qs��I       6%�	���;���A�'*;


total_lossؽ�@

error_R�=P?

learning_rate_1��j7��`hI       6%�	E<�;���A�'*;


total_lossH�@

error_R�]E?

learning_rate_1��j7)�. I       6%�	���;���A�'*;


total_loss�6�@

error_R�lQ?

learning_rate_1��j7���I       6%�	���;���A�'*;


total_loss���@

error_R@R?

learning_rate_1��j7��I       6%�	�;���A�'*;


total_loss6�@

error_R|-W?

learning_rate_1��j7-�G)I       6%�	�W�;���A�'*;


total_loss�A

error_R��S?

learning_rate_1��j7�%�QI       6%�	��;���A�'*;


total_lossN��@

error_R�JN?

learning_rate_1��j7�@t�I       6%�	���;���A�'*;


total_loss��@

error_R=B?

learning_rate_1��j7϶�I       6%�	�'�;���A�'*;


total_loss���@

error_R8�E?

learning_rate_1��j7���I       6%�	zp�;���A�'*;


total_loss8
�@

error_R;t>?

learning_rate_1��j7�"�bI       6%�	��;���A�'*;


total_lossXG�@

error_R��J?

learning_rate_1��j7,h��I       6%�	7��;���A�'*;


total_loss�[�@

error_RҎS?

learning_rate_1��j7��\I       6%�	�;�;���A�'*;


total_loss��@

error_RTzZ?

learning_rate_1��j7Ɍ��I       6%�	N��;���A�'*;


total_loss���@

error_R��M?

learning_rate_1��j7S	*�I       6%�	���;���A�'*;


total_loss�Y�@

error_R�E?

learning_rate_1��j7�M�I       6%�	+�;���A�'*;


total_lossཥ@

error_RO[?

learning_rate_1��j7�I       6%�	�c�;���A�'*;


total_loss��@

error_R��T?

learning_rate_1��j7yC�gI       6%�	}��;���A�'*;


total_loss�@�@

error_Rx.D?

learning_rate_1��j7Wj6I       6%�	<�;���A�'*;


total_lossh�@

error_RS�[?

learning_rate_1��j7J���I       6%�	�R�;���A�'*;


total_loss�g�@

error_R�J?

learning_rate_1��j7� I       6%�	���;���A�'*;


total_loss�@

error_R��O?

learning_rate_1��j72_�I       6%�	7�;���A�'*;


total_lossF��@

error_R�9I?

learning_rate_1��j7*���I       6%�	Pa�;���A�'*;


total_loss�(�@

error_R��V?

learning_rate_1��j7qq�I       6%�	��;���A�'*;


total_loss[�@

error_R�]?

learning_rate_1��j7xaWI       6%�	�;���A�'*;


total_lossٱ�@

error_R�D?

learning_rate_1��j7�z`I       6%�	�h�;���A�'*;


total_lossd:�@

error_R+U?

learning_rate_1��j7�N7I       6%�	���;���A�'*;


total_lossq��@

error_Ri&M?

learning_rate_1��j7J���I       6%�	�"�;���A�'*;


total_loss�0�@

error_R�bX?

learning_rate_1��j7�ʋ+I       6%�	�i�;���A�'*;


total_loss���@

error_Rx#C?

learning_rate_1��j7+՛�I       6%�	��;���A�'*;


total_loss�ª@

error_R�D?

learning_rate_1��j7�$�I       6%�	)��;���A�'*;


total_loss��@

error_R�K?

learning_rate_1��j71��I       6%�	�:�;���A�'*;


total_loss�O�@

error_Rn�M?

learning_rate_1��j7���I       6%�	��;���A�'*;


total_loss���@

error_R�nT?

learning_rate_1��j7��)�I       6%�	���;���A�'*;


total_loss�:�@

error_R�WB?

learning_rate_1��j7y�6=I       6%�	�7�;���A�'*;


total_loss���@

error_R\�>?

learning_rate_1��j7���I       6%�	�~�;���A�'*;


total_loss��@

error_R�h?

learning_rate_1��j7Y�VkI       6%�	V��;���A�'*;


total_loss�Y�@

error_R�@?

learning_rate_1��j7����I       6%�	�;���A�'*;


total_loss���@

error_R�R?

learning_rate_1��j7�6{�I       6%�	�P�;���A�'*;


total_lossvwA

error_R�MN?

learning_rate_1��j7Y��I       6%�	Е�;���A�'*;


total_lossXqA

error_R�?L?

learning_rate_1��j7j�	�I       6%�	X��;���A�'*;


total_loss��@

error_R�~A?

learning_rate_1��j7£OI       6%�	��;���A�'*;


total_lossd�a@

error_R[�>?

learning_rate_1��j7B$��I       6%�	�f�;���A�'*;


total_lossX^�@

error_R�Y?

learning_rate_1��j7H��I       6%�	e��;���A�'*;


total_loss}�'A

error_R��??

learning_rate_1��j7��I       6%�	a��;���A�'*;


total_loss�%�@

error_R�Q?

learning_rate_1��j78��I       6%�	�H�;���A�'*;


total_loss���@

error_R��O?

learning_rate_1��j7�&�BI       6%�	۞�;���A�'*;


total_loss�Ɔ@

error_R�Y?

learning_rate_1��j7�L�I       6%�	���;���A�'*;


total_loss���@

error_R�tR?

learning_rate_1��j7jx��I       6%�	�'�;���A�'*;


total_loss�Ώ@

error_Ri'K?

learning_rate_1��j7�ۥQI       6%�	j�;���A�'*;


total_lossr_�@

error_RiP?

learning_rate_1��j7g�G�I       6%�	~��;���A�'*;


total_loss�ܸ@

error_R��H?

learning_rate_1��j7n��I       6%�	e��;���A�'*;


total_loss�E�@

error_R�qR?

learning_rate_1��j7��(�I       6%�	�G�;���A�'*;


total_lossj=�@

error_R��J?

learning_rate_1��j7�t�lI       6%�	8��;���A�'*;


total_lossO�@

error_Rd�Q?

learning_rate_1��j7aI�I       6%�	m��;���A�'*;


total_loss���@

error_R�D@?

learning_rate_1��j7)!6&I       6%�	��;���A�'*;


total_lossq|�@

error_RO|O?

learning_rate_1��j7//`I       6%�	Uc�;���A�'*;


total_loss.��@

error_R��U?

learning_rate_1��j7�k�I       6%�	ԩ�;���A�'*;


total_loss��@

error_R��K?

learning_rate_1��j7+1��I       6%�	���;���A�'*;


total_loss��@

error_R]aT?

learning_rate_1��j7��I       6%�	�0�;���A�'*;


total_lossV�@

error_Ri�E?

learning_rate_1��j7����I       6%�	!v�;���A�'*;


total_loss���@

error_RT�B?

learning_rate_1��j7%��jI       6%�	ƻ�;���A�'*;


total_loss<��@

error_R�Q?

learning_rate_1��j7-a�I       6%�	Q�;���A�'*;


total_loss$]�@

error_R��G?

learning_rate_1��j7[--lI       6%�	rV�;���A�'*;


total_lossu~@

error_R�E?

learning_rate_1��j7���I       6%�	���;���A�'*;


total_loss���@

error_Ra+Q?

learning_rate_1��j7�\�8I       6%�	���;���A�'*;


total_lossx	�@

error_RœY?

learning_rate_1��j7�H��I       6%�	�4�;���A�'*;


total_lossd��@

error_R�O?

learning_rate_1��j7����I       6%�	K��;���A�'*;


total_loss�f�@

error_R�O^?

learning_rate_1��j7q��I       6%�	���;���A�'*;


total_loss�ɡ@

error_RsS?

learning_rate_1��j7+j��I       6%�	6$�;���A�'*;


total_loss��@

error_Ri�R?

learning_rate_1��j7��v"I       6%�	�h�;���A�'*;


total_loss&��@

error_R)�P?

learning_rate_1��j7�f<I       6%�	&��;���A�'*;


total_loss蟏@

error_Ra#V?

learning_rate_1��j7�[]I       6%�	w��;���A�'*;


total_loss0��@

error_R��N?

learning_rate_1��j7�*�I       6%�	�:�;���A�'*;


total_loss0A

error_R@�I?

learning_rate_1��j7�|}�I       6%�	���;���A�'*;


total_loss��@

error_R�O5?

learning_rate_1��j7���WI       6%�	k��;���A�'*;


total_loss���@

error_R}@?

learning_rate_1��j7qP9bI       6%�	�;���A�'*;


total_lossv��@

error_R#�R?

learning_rate_1��j7��^I       6%�	�Z�;���A�'*;


total_loss�4�@

error_R��V?

learning_rate_1��j7�%�pI       6%�	��;���A�'*;


total_loss�2�@

error_R��9?

learning_rate_1��j7�|�I       6%�	���;���A�'*;


total_loss�ն@

error_R(>Q?

learning_rate_1��j7��"I       6%�	/�;���A�'*;


total_loss{��@

error_R�hE?

learning_rate_1��j7��5 I       6%�	Zu�;���A�'*;


total_lossZ��@

error_R,�F?

learning_rate_1��j7�ŻI       6%�	���;���A�'*;


total_loss�ȉ@

error_RYA?

learning_rate_1��j7��I       6%�	���;���A�'*;


total_lossB��@

error_R}\W?

learning_rate_1��j7���I       6%�	�H�;���A�'*;


total_loss|�@

error_R��X?

learning_rate_1��j7s��>I       6%�	���;���A�'*;


total_loss47�@

error_RRC?

learning_rate_1��j7��,�I       6%�	���;���A�'*;


total_loss���@

error_R}�P?

learning_rate_1��j7v��XI       6%�	X,�;���A�'*;


total_loss3��@

error_RH�Z?

learning_rate_1��j7|�nI       6%�	�y�;���A�'*;


total_lossD� A

error_R�\?

learning_rate_1��j7jߍ�I       6%�	j��;���A�'*;


total_loss!�A

error_R�AG?

learning_rate_1��j7D�+�I       6%�	��;���A�'*;


total_loss���@

error_R6SI?

learning_rate_1��j7��A�I       6%�	5F�;���A�'*;


total_loss���@

error_R�N?

learning_rate_1��j7�^]�I       6%�	Ɖ�;���A�'*;


total_loss��@

error_R̋S?

learning_rate_1��j7��I       6%�	^��;���A�'*;


total_lossO]�@

error_Rf�A?

learning_rate_1��j7>�%I       6%�	��;���A�'*;


total_loss�;�@

error_RL1B?

learning_rate_1��j7#�<I       6%�	�[�;���A�'*;


total_loss���@

error_RPU?

learning_rate_1��j7��r
I       6%�	��;���A�(*;


total_lossb�@

error_R	�J?

learning_rate_1��j7�.NI       6%�	���;���A�(*;


total_loss�:�@

error_R�R?

learning_rate_1��j7���DI       6%�	#-�;���A�(*;


total_loss�k�@

error_RH�O?

learning_rate_1��j7�I��I       6%�	1�;���A�(*;


total_loss��@

error_RZ@N?

learning_rate_1��j7;ૉI       6%�	$��;���A�(*;


total_lossZ��@

error_R�cc?

learning_rate_1��j7��I       6%�	��;���A�(*;


total_loss_v�@

error_R�F?

learning_rate_1��j7�2��I       6%�	���;���A�(*;


total_loss'�@

error_R[�D?

learning_rate_1��j7Z"�I       6%�	���;���A�(*;


total_loss�7�@

error_R�F?

learning_rate_1��j7�~UNI       6%�	i�;���A�(*;


total_loss?�@

error_R�(??

learning_rate_1��j7��-�I       6%�	{c�;���A�(*;


total_lossCԒ@

error_RQ�N?

learning_rate_1��j7<���I       6%�	���;���A�(*;


total_lossC�@

error_R8?

learning_rate_1��j76*��I       6%�	K��;���A�(*;


total_lossXA�@

error_R��I?

learning_rate_1��j7�U�I       6%�	=<�;���A�(*;


total_loss�B�@

error_R_"V?

learning_rate_1��j7j�c�I       6%�	^��;���A�(*;


total_loss�6�@

error_R��I?

learning_rate_1��j7�nv�I       6%�	���;���A�(*;


total_lossr2CA

error_R��I?

learning_rate_1��j7P C�I       6%�	l�;���A�(*;


total_loss���@

error_R8�a?

learning_rate_1��j7��AI       6%�	J�;���A�(*;


total_lossv�@

error_RZ�>?

learning_rate_1��j7�V�I       6%�	 ��;���A�(*;


total_loss��
A

error_R��@?

learning_rate_1��j7���I       6%�	���;���A�(*;


total_loss���@

error_R6�U?

learning_rate_1��j7��s^I       6%�	��;���A�(*;


total_loss��4A

error_R�&U?

learning_rate_1��j78�p�I       6%�	S`�;���A�(*;


total_loss��@

error_Rf�Q?

learning_rate_1��j7 �iI       6%�	ţ�;���A�(*;


total_loss�4�@

error_R��J?

learning_rate_1��j7� �^I       6%�	���;���A�(*;


total_loss���@

error_Rf.P?

learning_rate_1��j7�гtI       6%�	06�;���A�(*;


total_loss�D�@

error_R=U_?

learning_rate_1��j7!���I       6%�	�}�;���A�(*;


total_loss|�@

error_R��I?

learning_rate_1��j7�M6�I       6%�	���;���A�(*;


total_loss�0�@

error_R�J<?

learning_rate_1��j7G��I       6%�	<�;���A�(*;


total_loss���@

error_R3W_?

learning_rate_1��j7\�~�I       6%�	�T�;���A�(*;


total_loss,��@

error_R�:Y?

learning_rate_1��j7&TI       6%�	���;���A�(*;


total_loss?�@

error_Rj�L?

learning_rate_1��j7�^n�I       6%�	q��;���A�(*;


total_lossa��@

error_R��[?

learning_rate_1��j7=��rI       6%�	�/�;���A�(*;


total_loss��@

error_R\?

learning_rate_1��j7|I       6%�	���;���A�(*;


total_loss�{@

error_R6�F?

learning_rate_1��j7ǺA�I       6%�	w��;���A�(*;


total_lossR�@

error_R�xX?

learning_rate_1��j7��ՎI       6%�	�6�;���A�(*;


total_loss1H�@

error_R�W?

learning_rate_1��j7$��JI       6%�	j|�;���A�(*;


total_loss���@

error_RęF?

learning_rate_1��j7�q�iI       6%�	l��;���A�(*;


total_loss��@

error_R�bB?

learning_rate_1��j7�[JI       6%�	b(�;���A�(*;


total_loss�ձ@

error_R$6?

learning_rate_1��j7��"I       6%�	�j�;���A�(*;


total_loss]3�@

error_R��E?

learning_rate_1��j7�I       6%�	,��;���A�(*;


total_lossk�@

error_R�H?

learning_rate_1��j7 ��I       6%�	���;���A�(*;


total_loss	WA

error_R�a?

learning_rate_1��j7*�21I       6%�	�A�;���A�(*;


total_loss��@

error_R�Z?

learning_rate_1��j7~'G3I       6%�	��;���A�(*;


total_lossA.�@

error_R�B?

learning_rate_1��j7{9�I       6%�	,��;���A�(*;


total_lossE�@

error_R�>X?

learning_rate_1��j75��I       6%�	��;���A�(*;


total_loss���@

error_RW�B?

learning_rate_1��j7���I       6%�	VX�;���A�(*;


total_loss��@

error_Rv�S?

learning_rate_1��j7 e1~I       6%�	���;���A�(*;


total_loss��@

error_RO�A?

learning_rate_1��j7�<�vI       6%�	���;���A�(*;


total_lossfL�@

error_R�3Z?

learning_rate_1��j7�A�I       6%�	�1�;���A�(*;


total_loss�Z�@

error_RR�D?

learning_rate_1��j7�
�I       6%�	J{�;���A�(*;


total_loss�Qr@

error_RS�G?

learning_rate_1��j7֛<I       6%�	[��;���A�(*;


total_loss(ٔ@

error_R�`G?

learning_rate_1��j7�T�UI       6%�	��;���A�(*;


total_loss��@

error_R�}m?

learning_rate_1��j7��a�I       6%�	ZP�;���A�(*;


total_loss��@

error_RC�O?

learning_rate_1��j7Z�D�I       6%�	��;���A�(*;


total_loss� �@

error_R{N?

learning_rate_1��j7��K0I       6%�	}��;���A�(*;


total_loss_t�@

error_RO\I?

learning_rate_1��j7�<e�I       6%�	��;���A�(*;


total_lossXM�@

error_R��I?

learning_rate_1��j7 j�I       6%�	�b�;���A�(*;


total_losss��@

error_RxW?

learning_rate_1��j7�
m�I       6%�	0��;���A�(*;


total_loss��@

error_R�5G?

learning_rate_1��j7��
�I       6%�	��;���A�(*;


total_loss��d@

error_R��P?

learning_rate_1��j7q"HI       6%�	r5�;���A�(*;


total_losss"�@

error_R�/5?

learning_rate_1��j7����I       6%�	�z�;���A�(*;


total_loss�:�@

error_R�CN?

learning_rate_1��j7���+I       6%�	E��;���A�(*;


total_loss�՚@

error_R�*S?

learning_rate_1��j7�D\I       6%�	�
�;���A�(*;


total_loss��@

error_RS�M?

learning_rate_1��j7��VI       6%�	�r�;���A�(*;


total_lossv�@

error_R�rI?

learning_rate_1��j7�MJI       6%�	���;���A�(*;


total_loss��@

error_R��Z?

learning_rate_1��j7��_I       6%�	���;���A�(*;


total_loss,��@

error_R3SU?

learning_rate_1��j7�.��I       6%�	�B�;���A�(*;


total_loss$�@

error_R,GU?

learning_rate_1��j7_��SI       6%�	̉�;���A�(*;


total_lossh�@

error_R)�C?

learning_rate_1��j7h�k�I       6%�	���;���A�(*;


total_loss1��@

error_RZN??

learning_rate_1��j7$��I       6%�	�;���A�(*;


total_lossl��@

error_R�Pd?

learning_rate_1��j7�u [I       6%�	ob�;���A�(*;


total_loss���@

error_RTS?

learning_rate_1��j71 =�I       6%�	R��;���A�(*;


total_loss��@

error_R��J?

learning_rate_1��j7%%�I       6%�	���;���A�(*;


total_lossdC�@

error_R��U?

learning_rate_1��j7�1�[I       6%�	[C�;���A�(*;


total_loss���@

error_R&W?

learning_rate_1��j7.�Y�I       6%�	��;���A�(*;


total_loss�ɧ@

error_R��??

learning_rate_1��j7._]�I       6%�	��;���A�(*;


total_loss���@

error_RJ�G?

learning_rate_1��j78d��I       6%�	'�;���A�(*;


total_loss�A

error_R��Z?

learning_rate_1��j7Uߗ�I       6%�	�g�;���A�(*;


total_losso<�@

error_R'R?

learning_rate_1��j7�,��I       6%�	p��;���A�(*;


total_loss��@

error_R{<D?

learning_rate_1��j7�mI       6%�	"��;���A�(*;


total_loss�5�@

error_R_�@?

learning_rate_1��j7��luI       6%�	wF�;���A�(*;


total_loss�(�@

error_R�O?

learning_rate_1��j7�#�I       6%�	���;���A�(*;


total_lossM��@

error_R�P?

learning_rate_1��j7�"J�I       6%�	���;���A�(*;


total_lossmx�@

error_Rv�K?

learning_rate_1��j7+H�kI       6%�	��;���A�(*;


total_loss<�@

error_R�A?

learning_rate_1��j7&n��I       6%�	1_�;���A�(*;


total_lossw��@

error_R,_?

learning_rate_1��j7�c��I       6%�	n��;���A�(*;


total_loss[��@

error_R��??

learning_rate_1��j76���I       6%�	���;���A�(*;


total_loss.�A

error_Rl�[?

learning_rate_1��j7�!�I       6%�	>:�;���A�(*;


total_lossn��@

error_R�Z?

learning_rate_1��j7Ov1�I       6%�	���;���A�(*;


total_loss):�@

error_R�P?

learning_rate_1��j7̢��I       6%�	3��;���A�(*;


total_loss���@

error_R��L?

learning_rate_1��j71pz0I       6%�	��;���A�(*;


total_loss���@

error_R��S?

learning_rate_1��j7 �I       6%�	�X�;���A�(*;


total_lossΕ@

error_R��G?

learning_rate_1��j7V	>ZI       6%�	V��;���A�(*;


total_loss� �@

error_R;CO?

learning_rate_1��j7�~�I       6%�	K��;���A�(*;


total_lossEA

error_R!�W?

learning_rate_1��j7�� 0I       6%�	.�;���A�(*;


total_loss�
:A

error_R��K?

learning_rate_1��j7����I       6%�	�s�;���A�(*;


total_loss_�A

error_R҆_?

learning_rate_1��j7o���I       6%�	���;���A�(*;


total_lossh|�@

error_R�J?

learning_rate_1��j7
�I       6%�	��;���A�(*;


total_loss�,�@

error_R��V?

learning_rate_1��j7ؒ��I       6%�	@ <���A�(*;


total_loss�#�@

error_R6T;?

learning_rate_1��j7�خ�I       6%�	�� <���A�(*;


total_lossr��@

error_R�5?

learning_rate_1��j7=�<�I       6%�	�� <���A�(*;


total_loss���@

error_R��P?

learning_rate_1��j7{|�I       6%�	C<���A�(*;


total_loss_1�@

error_R�`?

learning_rate_1��j7��[4I       6%�	�S<���A�(*;


total_loss\�z@

error_RJ�F?

learning_rate_1��j7�-y�I       6%�	��<���A�(*;


total_loss3I�@

error_Rt\?

learning_rate_1��j7��rI       6%�	P�<���A�(*;


total_loss\��@

error_R=�E?

learning_rate_1��j7�(�BI       6%�	�"<���A�(*;


total_lossfT�@

error_RC�P?

learning_rate_1��j7�nj-I       6%�	�e<���A�(*;


total_lossi۞@

error_R��l?

learning_rate_1��j7ҁ�I       6%�	ĩ<���A�(*;


total_losse�@

error_R�gT?

learning_rate_1��j7���I       6%�	q�<���A�(*;


total_lossz*�@

error_R�R?

learning_rate_1��j7`V��I       6%�	/<���A�(*;


total_loss���@

error_R�K?

learning_rate_1��j7�4&I       6%�	Np<���A�(*;


total_loss���@

error_R�]?

learning_rate_1��j7R	�I       6%�	~�<���A�(*;


total_loss7
�@

error_R<xK?

learning_rate_1��j7���I       6%�	)�<���A�(*;


total_loss���@

error_R\QQ?

learning_rate_1��j7'�n'I       6%�	�7<���A�(*;


total_loss$��@

error_RSzB?

learning_rate_1��j7�P��I       6%�	t~<���A�(*;


total_loss'�@

error_R��_?

learning_rate_1��j7R��I       6%�	e�<���A�(*;


total_loss
D�@

error_R[3R?

learning_rate_1��j7�*��I       6%�	�<���A�(*;


total_lossP��@

error_RSH?

learning_rate_1��j7z�4dI       6%�	�a<���A�(*;


total_loss�@

error_R�X?

learning_rate_1��j7�4�ZI       6%�	b�<���A�(*;


total_loss���@

error_Rc-\?

learning_rate_1��j7y�6I       6%�	V#<���A�(*;


total_lossey�@

error_R�QL?

learning_rate_1��j7a\II       6%�	�<���A�(*;


total_loss�`�@

error_RNZk?

learning_rate_1��j7d@�I       6%�	�<���A�(*;


total_loss�Ƶ@

error_R�S?

learning_rate_1��j7����I       6%�	�O<���A�(*;


total_lossm��@

error_R:�W?

learning_rate_1��j7� eI       6%�	��<���A�(*;


total_loss8�@

error_R�V^?

learning_rate_1��j7�?��I       6%�	��<���A�(*;


total_lossd@

error_R��G?

learning_rate_1��j7���I       6%�	�<���A�(*;


total_lossn8A

error_RIa?

learning_rate_1��j7���I       6%�	�_<���A�(*;


total_loss��@

error_RQ?

learning_rate_1��j7�x�I       6%�	��<���A�(*;


total_loss���@

error_RS�C?

learning_rate_1��j7�a�nI       6%�	��<���A�(*;


total_loss�@

error_R1e?

learning_rate_1��j7��I       6%�	�;	<���A�)*;


total_loss��@

error_R�	X?

learning_rate_1��j7�#��I       6%�	�	<���A�)*;


total_loss��@

error_R��??

learning_rate_1��j7�WI       6%�	��	<���A�)*;


total_loss��@

error_R�MB?

learning_rate_1��j7и��I       6%�	

<���A�)*;


total_loss_��@

error_R�C?

learning_rate_1��j7�9��I       6%�	P
<���A�)*;


total_loss���@

error_R��W?

learning_rate_1��j7�݉I       6%�	��
<���A�)*;


total_lossX[�@

error_R��Z?

learning_rate_1��j7r3�lI       6%�	f�
<���A�)*;


total_loss���@

error_RZ?T?

learning_rate_1��j7��̓I       6%�	g<���A�)*;


total_loss��@

error_RCK?

learning_rate_1��j7i�0dI       6%�	v]<���A�)*;


total_loss���@

error_R��M?

learning_rate_1��j7�rCI       6%�	��<���A�)*;


total_loss?`�@

error_R�S?

learning_rate_1��j7�@��I       6%�	�<���A�)*;


total_loss\�@

error_Rx�S?

learning_rate_1��j7��II       6%�	�(<���A�)*;


total_loss:=�@

error_RI�R?

learning_rate_1��j7�N6I       6%�	�o<���A�)*;


total_lossu�@

error_R��e?

learning_rate_1��j7N��I       6%�	��<���A�)*;


total_loss0A

error_R	�W?

learning_rate_1��j7M�D�I       6%�	�<���A�)*;


total_loss���@

error_R�V?

learning_rate_1��j7�ݥ�I       6%�	)A<���A�)*;


total_loss�T�@

error_R�\?

learning_rate_1��j7C}�sI       6%�	��<���A�)*;


total_loss�@�@

error_RG?

learning_rate_1��j7-�!I       6%�	��<���A�)*;


total_loss��@

error_RL?

learning_rate_1��j7���I       6%�	�-<���A�)*;


total_loss���@

error_R�O?

learning_rate_1��j7;�rI       6%�	:p<���A�)*;


total_loss���@

error_R�sC?

learning_rate_1��j7淫/I       6%�	��<���A�)*;


total_loss���@

error_R}�S?

learning_rate_1��j7
�CzI       6%�	L"<���A�)*;


total_loss�,�@

error_R#�R?

learning_rate_1��j7H)I       6%�	6h<���A�)*;


total_loss%>}@

error_R�S?

learning_rate_1��j7���I       6%�	��<���A�)*;


total_loss͐A

error_R\�T?

learning_rate_1��j7f�I       6%�	��<���A�)*;


total_loss���@

error_R�GX?

learning_rate_1��j7A�pI       6%�	�8<���A�)*;


total_loss���@

error_R(�_?

learning_rate_1��j7����I       6%�	H~<���A�)*;


total_lossm��@

error_RO_?

learning_rate_1��j7l��I       6%�	]�<���A�)*;


total_loss���@

error_R�CQ?

learning_rate_1��j7�^��I       6%�	�<���A�)*;


total_lossQ�@

error_R7D>?

learning_rate_1��j7����I       6%�	�W<���A�)*;


total_loss�ج@

error_R{�E?

learning_rate_1��j7ܝ0�I       6%�	_�<���A�)*;


total_lossx]�@

error_R׻N?

learning_rate_1��j7�2�I       6%�	B�<���A�)*;


total_loss��@

error_R�IP?

learning_rate_1��j7��9I       6%�	�*<���A�)*;


total_loss�/�@

error_R�ue?

learning_rate_1��j7��I       6%�	�m<���A�)*;


total_loss�Қ@

error_Rn�d?

learning_rate_1��j7yYNSI       6%�	�<���A�)*;


total_lossv>�@

error_R��Y?

learning_rate_1��j7�&��I       6%�	F�<���A�)*;


total_lossu�@

error_R�K?

learning_rate_1��j7�e��I       6%�	?<���A�)*;


total_lossݓ�@

error_Rv+K?

learning_rate_1��j7���DI       6%�	C�<���A�)*;


total_loss%�@

error_RMT?

learning_rate_1��j7��9I       6%�	K�<���A�)*;


total_loss�f�@

error_Rd�L?

learning_rate_1��j7�f�I       6%�	�<���A�)*;


total_loss��@

error_R�M?

learning_rate_1��j7�*�I       6%�	�J<���A�)*;


total_loss@

error_R�L?

learning_rate_1��j7).?HI       6%�	��<���A�)*;


total_loss� @

error_R�JQ?

learning_rate_1��j7R���I       6%�	i�<���A�)*;


total_loss
�@

error_R
�^?

learning_rate_1��j7��v�I       6%�	�<���A�)*;


total_lossC�@

error_R��A?

learning_rate_1��j7~�?�I       6%�	#f<���A�)*;


total_loss��@

error_RȌR?

learning_rate_1��j7�t�I       6%�	٪<���A�)*;


total_lossN�@

error_Rw3P?

learning_rate_1��j7vBxI       6%�	��<���A�)*;


total_loss)ر@

error_R�rW?

learning_rate_1��j7t�I       6%�	�A<���A�)*;


total_loss*�A

error_R�W?

learning_rate_1��j7扏WI       6%�	��<���A�)*;


total_lossk��@

error_R�Z?

learning_rate_1��j7����I       6%�	L�<���A�)*;


total_lossm#�@

error_R�.Y?

learning_rate_1��j7>�8I       6%�	�0<���A�)*;


total_loss�G�@

error_Reo>?

learning_rate_1��j7�[<3I       6%�	�t<���A�)*;


total_loss�H�@

error_R�RJ?

learning_rate_1��j7��ěI       6%�	ֶ<���A�)*;


total_loss��@

error_R�X?

learning_rate_1��j7��I       6%�	��<���A�)*;


total_lossvB�@

error_R��S?

learning_rate_1��j7��1�I       6%�	S;<���A�)*;


total_loss�z�@

error_R��@?

learning_rate_1��j7x�H�I       6%�	�|<���A�)*;


total_lossw�@

error_R� \?

learning_rate_1��j7��q�I       6%�	�<���A�)*;


total_loss\�@

error_R��L?

learning_rate_1��j7L���I       6%�	�<���A�)*;


total_loss��@

error_R��]?

learning_rate_1��j7E��I       6%�	EH<���A�)*;


total_loss��@

error_R��I?

learning_rate_1��j7"5��I       6%�	e�<���A�)*;


total_loss��@

error_RҷW?

learning_rate_1��j7w�W�I       6%�	��<���A�)*;


total_loss��@

error_R�L?

learning_rate_1��j7ɳ&
I       6%�	�!<���A�)*;


total_lossa�A

error_R��W?

learning_rate_1��j7��I       6%�	�o<���A�)*;


total_loss�x�@

error_Rnt??

learning_rate_1��j7��G-I       6%�	�<���A�)*;


total_lossnY�@

error_RH�b?

learning_rate_1��j7�I       6%�	�	<���A�)*;


total_loss���@

error_R!7V?

learning_rate_1��j7���kI       6%�	�T<���A�)*;


total_loss��@

error_R/<?

learning_rate_1��j7�f��I       6%�	1�<���A�)*;


total_losse2�@

error_R��I?

learning_rate_1��j7���I       6%�	q�<���A�)*;


total_lossM�@

error_R<�S?

learning_rate_1��j7a,t�I       6%�	�'<���A�)*;


total_loss��@

error_R!tj?

learning_rate_1��j7,�yI       6%�	&k<���A�)*;


total_loss��@

error_R��U?

learning_rate_1��j7IҝI       6%�	��<���A�)*;


total_loss���@

error_RfbR?

learning_rate_1��j7{�/�I       6%�	��<���A�)*;


total_loss��A

error_R�@?

learning_rate_1��j7����I       6%�	OB<���A�)*;


total_loss�r�@

error_R|R?

learning_rate_1��j7f�m�I       6%�	Ŋ<���A�)*;


total_loss8(�@

error_R�??

learning_rate_1��j77¢xI       6%�	��<���A�)*;


total_losso��@

error_R��U?

learning_rate_1��j7OH�I       6%�	�<���A�)*;


total_lossLU�@

error_R?UI?

learning_rate_1��j7��#�I       6%�	�X<���A�)*;


total_loss�+�@

error_Ri,X?

learning_rate_1��j7��ϝI       6%�	s�<���A�)*;


total_loss�*�@

error_R�gE?

learning_rate_1��j7��ضI       6%�	��<���A�)*;


total_loss��@

error_RںC?

learning_rate_1��j7�H�I       6%�	�<���A�)*;


total_loss�~�@

error_R�RR?

learning_rate_1��j7#��I       6%�	@b<���A�)*;


total_loss�o�@

error_R�I?

learning_rate_1��j7�x��I       6%�	��<���A�)*;


total_loss@d@

error_R�)C?

learning_rate_1��j7f���I       6%�	��<���A�)*;


total_loss�-�@

error_R�hK?

learning_rate_1��j7�Ed:I       6%�	�* <���A�)*;


total_loss��@

error_R\C?

learning_rate_1��j7�	ǌI       6%�	Vm <���A�)*;


total_loss͞�@

error_RϞB?

learning_rate_1��j7=��"I       6%�	Է <���A�)*;


total_lossɝv@

error_R��N?

learning_rate_1��j7�C%vI       6%�	�!<���A�)*;


total_loss��U@

error_R�iH?

learning_rate_1��j7�y��I       6%�	�J!<���A�)*;


total_loss�-�@

error_RaNV?

learning_rate_1��j7q��xI       6%�	��!<���A�)*;


total_loss@�@

error_R��T?

learning_rate_1��j7��VI       6%�	��!<���A�)*;


total_lossh��@

error_R
�^?

learning_rate_1��j7U'[I       6%�	�"<���A�)*;


total_lossܶ@

error_R��O?

learning_rate_1��j7d�@oI       6%�	=W"<���A�)*;


total_loss�?�@

error_R[Y?

learning_rate_1��j7���[I       6%�	�"<���A�)*;


total_loss�0A

error_R dK?

learning_rate_1��j7����I       6%�	��"<���A�)*;


total_lossiڵ@

error_ReW?

learning_rate_1��j7T�f�I       6%�	�(#<���A�)*;


total_loss["�@

error_R�N?

learning_rate_1��j7}{I       6%�	p#<���A�)*;


total_lossF�@

error_RțB?

learning_rate_1��j7\ҁ	I       6%�	M�#<���A�)*;


total_loss��A

error_Rz�^?

learning_rate_1��j73#I       6%�	� $<���A�)*;


total_loss�@

error_RѴS?

learning_rate_1��j7���I       6%�	�I$<���A�)*;


total_lossl=A

error_R�:W?

learning_rate_1��j78�n"I       6%�	)�$<���A�)*;


total_loss�2�@

error_R��B?

learning_rate_1��j7���KI       6%�	3�$<���A�)*;


total_loss���@

error_R��R?

learning_rate_1��j7�j^%I       6%�	�!%<���A�)*;


total_lossdA

error_RC�J?

learning_rate_1��j7��AI       6%�	Ge%<���A�)*;


total_loss@Y�@

error_R}�F?

learning_rate_1��j7|}(<I       6%�	��%<���A�)*;


total_lossƮ�@

error_RxdP?

learning_rate_1��j7c*�[I       6%�	^�%<���A�)*;


total_loss�,�@

error_R��E?

learning_rate_1��j7�@�#I       6%�	VA&<���A�)*;


total_lossXP�@

error_RN�P?

learning_rate_1��j7��i�I       6%�	ȣ&<���A�)*;


total_loss�߯@

error_R#eP?

learning_rate_1��j7�:��I       6%�	�&<���A�)*;


total_lossd)�@

error_R��C?

learning_rate_1��j7l��I       6%�	/'<���A�)*;


total_lossT+�@

error_R��F?

learning_rate_1��j7�>I       6%�	�x'<���A�)*;


total_lossT��@

error_R��I?

learning_rate_1��j7���I       6%�	`�'<���A�)*;


total_lossY�@

error_R�AO?

learning_rate_1��j7�H�I       6%�	�(<���A�)*;


total_loss���@

error_R��V?

learning_rate_1��j7���I       6%�	zL(<���A�)*;


total_loss4��@

error_R3M?

learning_rate_1��j73y��I       6%�	e�(<���A�)*;


total_loss�q@

error_R�zO?

learning_rate_1��j7NI       6%�	n�(<���A�)*;


total_lossd��@

error_RݘS?

learning_rate_1��j7e�5I       6%�	�)<���A�)*;


total_loss���@

error_R@�R?

learning_rate_1��j7ZA�I       6%�	Y)<���A�)*;


total_loss�J�@

error_R �@?

learning_rate_1��j7��+"I       6%�	�)<���A�)*;


total_loss��@

error_Rv]?

learning_rate_1��j7��_KI       6%�	��)<���A�)*;


total_loss�	A

error_RxIE?

learning_rate_1��j71pI       6%�	�!*<���A�)*;


total_loss��@

error_R�>b?

learning_rate_1��j7�w>�I       6%�	�f*<���A�)*;


total_loss(�@

error_R�FU?

learning_rate_1��j7S��I       6%�	�*<���A�)*;


total_loss���@

error_R�:G?

learning_rate_1��j7��f�I       6%�	L�*<���A�)*;


total_lossaB�@

error_R�=?

learning_rate_1��j7І"PI       6%�	�=+<���A�)*;


total_loss���@

error_R��E?

learning_rate_1��j7g��I       6%�	��+<���A�)*;


total_loss�^�@

error_RS�H?

learning_rate_1��j7�MI�I       6%�	r�+<���A�)*;


total_loss	Q}@

error_R�Z?

learning_rate_1��j7����I       6%�	�,<���A�)*;


total_loss�%�@

error_R��J?

learning_rate_1��j7 �N:I       6%�	rR,<���A�)*;


total_lossRGz@

error_R ]L?

learning_rate_1��j7.~�<I       6%�	a�,<���A�**;


total_loss��@

error_R�C?

learning_rate_1��j7���I       6%�	��,<���A�**;


total_loss&��@

error_R�(V?

learning_rate_1��j7� jI       6%�	\!-<���A�**;


total_loss�{�@

error_Rɮ^?

learning_rate_1��j7�XhI       6%�	h-<���A�**;


total_loss�,�@

error_R��E?

learning_rate_1��j7hj��I       6%�	�-<���A�**;


total_loss�Ʋ@

error_R��P?

learning_rate_1��j7`��(I       6%�	�.<���A�**;


total_loss�6�@

error_Ra{E?

learning_rate_1��j7lI!�I       6%�	�G.<���A�**;


total_losss�@

error_R�B?

learning_rate_1��j7�T�,I       6%�	7�.<���A�**;


total_lossg*A

error_R�I?

learning_rate_1��j7�3��I       6%�	��.<���A�**;


total_lossLs�@

error_RŶH?

learning_rate_1��j7�*['I       6%�	�0/<���A�**;


total_lossO��@

error_RaL?

learning_rate_1��j7R�b�I       6%�	bz/<���A�**;


total_lossN�^@

error_R[�H?

learning_rate_1��j7�+��I       6%�	�/<���A�**;


total_loss.=�@

error_REj8?

learning_rate_1��j7¶`MI       6%�	0<���A�**;


total_loss�R�@

error_R&�@?

learning_rate_1��j7�s��I       6%�	4P0<���A�**;


total_lossF��@

error_R�S?

learning_rate_1��j7��I       6%�	3�0<���A�**;


total_loss��@

error_R��[?

learning_rate_1��j7;q�4I       6%�	q�0<���A�**;


total_loss{�H@

error_R��8?

learning_rate_1��j7�|0eI       6%�	:/1<���A�**;


total_loss]�@

error_R�K?

learning_rate_1��j7��/I       6%�	�v1<���A�**;


total_loss���@

error_R
�E?

learning_rate_1��j7(�m�I       6%�	`�1<���A�**;


total_lossF,�@

error_RE�??

learning_rate_1��j7=	�I       6%�	2<���A�**;


total_loss�^�@

error_R�y]?

learning_rate_1��j7 Q^oI       6%�	XK2<���A�**;


total_loss���@

error_R��b?

learning_rate_1��j7��kI       6%�	˓2<���A�**;


total_lossS�@

error_R�z^?

learning_rate_1��j7��I       6%�	d�2<���A�**;


total_loss)>�@

error_RD�T?

learning_rate_1��j7Y�I       6%�	4"3<���A�**;


total_loss���@

error_R��:?

learning_rate_1��j7�JU�I       6%�	�g3<���A�**;


total_loss�c�@

error_R-^?

learning_rate_1��j7���I       6%�	��3<���A�**;


total_loss�u�@

error_R��X?

learning_rate_1��j7w���I       6%�	1�3<���A�**;


total_loss���@

error_R �Q?

learning_rate_1��j7ǇyI       6%�	�74<���A�**;


total_loss�\	A

error_RÐ\?

learning_rate_1��j7�0�~I       6%�	�}4<���A�**;


total_loss� A

error_R:aP?

learning_rate_1��j7���sI       6%�	�4<���A�**;


total_loss��@

error_RȬH?

learning_rate_1��j7��I       6%�	Z5<���A�**;


total_losssg@

error_Rw(K?

learning_rate_1��j7��I       6%�	!R5<���A�**;


total_loss���@

error_R
�H?

learning_rate_1��j7�ȯ�I       6%�	�5<���A�**;


total_loss�H�@

error_R�VN?

learning_rate_1��j7qwPI       6%�	R�5<���A�**;


total_loss0A

error_R@�9?

learning_rate_1��j7����I       6%�	�06<���A�**;


total_lossǃ�@

error_RWBN?

learning_rate_1��j7�8I       6%�	��6<���A�**;


total_loss�e�@

error_R�O?

learning_rate_1��j75���I       6%�	��6<���A�**;


total_lossȆ�@

error_RO�B?

learning_rate_1��j7e�ՒI       6%�	d 7<���A�**;


total_loss�MA

error_R��H?

learning_rate_1��j7�T
�I       6%�	.m7<���A�**;


total_loss�K�@

error_Rf�E?

learning_rate_1��j7:���I       6%�	۸7<���A�**;


total_lossv �@

error_R�^E?

learning_rate_1��j7@��bI       6%�	�7<���A�**;


total_loss��@

error_R��A?

learning_rate_1��j7��7�I       6%�	w=8<���A�**;


total_lossӏ@

error_RW�F?

learning_rate_1��j7�"��I       6%�	W�8<���A�**;


total_lossDJ�@

error_R&jK?

learning_rate_1��j7)��VI       6%�	I�8<���A�**;


total_lossRC�@

error_R��b?

learning_rate_1��j76��I       6%�	=	9<���A�**;


total_loss8�A

error_R�G?

learning_rate_1��j7�@CI       6%�	�L9<���A�**;


total_lossh��@

error_R1�D?

learning_rate_1��j7�~c�I       6%�	T�9<���A�**;


total_loss�(�@

error_R��^?

learning_rate_1��j7C>K�I       6%�		�9<���A�**;


total_losss��@

error_R_n7?

learning_rate_1��j7���=I       6%�	J:<���A�**;


total_loss��@

error_R��T?

learning_rate_1��j7�2�:I       6%�	�]:<���A�**;


total_loss��@

error_R�>H?

learning_rate_1��j7H��I       6%�	��:<���A�**;


total_lossΏ�@

error_RIH9?

learning_rate_1��j7d��I       6%�	��:<���A�**;


total_lossM��@

error_R��T?

learning_rate_1��j7����I       6%�	�*;<���A�**;


total_loss��@

error_R��F?

learning_rate_1��j7[���I       6%�	�r;<���A�**;


total_loss�F�@

error_R�E?

learning_rate_1��j7-�5I       6%�	��;<���A�**;


total_loss;�@

error_R)VX?

learning_rate_1��j7t�HI       6%�	�<<���A�**;


total_loss��w@

error_Rϋ7?

learning_rate_1��j7�#7[I       6%�	�M<<���A�**;


total_lossQ.�@

error_R:E>?

learning_rate_1��j7����I       6%�	�<<���A�**;


total_lossc��@

error_R��R?

learning_rate_1��j7�DyI       6%�	��<<���A�**;


total_loss6s�@

error_Rc�K?

learning_rate_1��j7�NʉI       6%�	�=<���A�**;


total_loss�@

error_RsOL?

learning_rate_1��j7T�6�I       6%�	�a=<���A�**;


total_lossV�A

error_RdT?

learning_rate_1��j7	�X�I       6%�	�=<���A�**;


total_loss�I�@

error_R�;?

learning_rate_1��j7���/I       6%�	3�=<���A�**;


total_loss�
A

error_R�X?

learning_rate_1��j7����I       6%�	o3><���A�**;


total_loss�כ@

error_R�X?

learning_rate_1��j7!q�DI       6%�	�z><���A�**;


total_loss�Y�@

error_Ra�B?

learning_rate_1��j7�ѷ�I       6%�	��><���A�**;


total_lossoX�@

error_R� .?

learning_rate_1��j7l�I       6%�	;?<���A�**;


total_loss7��@

error_R�EJ?

learning_rate_1��j7�O0�I       6%�	BI?<���A�**;


total_loss�N�@

error_Rn�B?

learning_rate_1��j7d���I       6%�	�?<���A�**;


total_lossJ^�@

error_R�1]?

learning_rate_1��j7��z%I       6%�	��?<���A�**;


total_lossMU�@

error_R
F?

learning_rate_1��j7MeI       6%�	7@<���A�**;


total_lossס@

error_R��W?

learning_rate_1��j7E�DI       6%�	K_@<���A�**;


total_loss�ʆ@

error_R��??

learning_rate_1��j7�3��I       6%�	��@<���A�**;


total_loss�@

error_R-�O?

learning_rate_1��j7�kTOI       6%�	x�@<���A�**;


total_lossRD�@

error_Rt1G?

learning_rate_1��j7ޞu�I       6%�	40A<���A�**;


total_losss��@

error_RM�a?

learning_rate_1��j7U_i�I       6%�	]zA<���A�**;


total_loss�9�@

error_R�KV?

learning_rate_1��j7WBCGI       6%�	q�A<���A�**;


total_loss���@

error_R9H?

learning_rate_1��j7a�BI       6%�	�B<���A�**;


total_lossfϱ@

error_R.�R?

learning_rate_1��j7��+I       6%�	�ZB<���A�**;


total_loss�T~@

error_R�9?

learning_rate_1��j7� ��I       6%�	k�B<���A�**;


total_loss_A�@

error_R�R?

learning_rate_1��j7s���I       6%�	;�B<���A�**;


total_loss���@

error_R�H?

learning_rate_1��j7	���I       6%�	�/C<���A�**;


total_loss��@

error_RW�F?

learning_rate_1��j7�6#�I       6%�	�tC<���A�**;


total_lossB5�@

error_R!�K?

learning_rate_1��j7w1j2I       6%�	��C<���A�**;


total_loss�f@

error_RH�D?

learning_rate_1��j7���I       6%�	��C<���A�**;


total_lossM��@

error_R�am?

learning_rate_1��j7��gI       6%�	�@D<���A�**;


total_lossd"�@

error_R��B?

learning_rate_1��j7���I       6%�	*�D<���A�**;


total_loss�V�@

error_R��S?

learning_rate_1��j7��{!I       6%�	��D<���A�**;


total_loss�:�@

error_Rf0F?

learning_rate_1��j7K�TI       6%�	�E<���A�**;


total_loss�m�@

error_R�C?

learning_rate_1��j7���I       6%�	�JE<���A�**;


total_loss�ݔ@

error_R��@?

learning_rate_1��j7X�'�I       6%�	ؐE<���A�**;


total_loss�Z�@

error_R.�V?

learning_rate_1��j7���I       6%�	l�E<���A�**;


total_loss*�@

error_RT�8?

learning_rate_1��j7���I       6%�	� F<���A�**;


total_loss_��@

error_Rd4?

learning_rate_1��j7���I       6%�	�~F<���A�**;


total_lossM�@

error_R�E?

learning_rate_1��j77�9�I       6%�	��F<���A�**;


total_loss �@

error_Rc?

learning_rate_1��j7��RI       6%�	�	G<���A�**;


total_loss���@

error_R��N?

learning_rate_1��j7���I       6%�	CPG<���A�**;


total_loss��@

error_R��C?

learning_rate_1��j7�A�I       6%�	��G<���A�**;


total_loss���@

error_R�S?

learning_rate_1��j7���I       6%�	��G<���A�**;


total_lossͯ�@

error_RawQ?

learning_rate_1��j7�lhI       6%�	H<���A�**;


total_loss_��@

error_R��P?

learning_rate_1��j77 �I       6%�	beH<���A�**;


total_loss���@

error_R��N?

learning_rate_1��j7��#ZI       6%�	��H<���A�**;


total_lossx��@

error_R��M?

learning_rate_1��j7!��I       6%�	 �H<���A�**;


total_loss���@

error_R,9N?

learning_rate_1��j7R�ߡI       6%�	0I<���A�**;


total_loss���@

error_RaTW?

learning_rate_1��j7�ڨI       6%�	ssI<���A�**;


total_loss��@

error_R��A?

learning_rate_1��j7�p�I       6%�	�I<���A�**;


total_loss���@

error_R�;\?

learning_rate_1��j7����I       6%�	I�I<���A�**;


total_lossF��@

error_R�R?

learning_rate_1��j7w��LI       6%�	6?J<���A�**;


total_lossxJA

error_RaLG?

learning_rate_1��j7��5lI       6%�	�J<���A�**;


total_loss���@

error_Rj�Z?

learning_rate_1��j7��I       6%�	��J<���A�**;


total_lossg�@

error_R�!@?

learning_rate_1��j7�ҤI       6%�	jK<���A�**;


total_loss�څ@

error_R�lE?

learning_rate_1��j7އ��I       6%�	�RK<���A�**;


total_loss�@

error_R�6R?

learning_rate_1��j7����I       6%�	�K<���A�**;


total_loss��@

error_RmY?

learning_rate_1��j7=L�I       6%�	��K<���A�**;


total_loss�@

error_RԩD?

learning_rate_1��j7��_�I       6%�	$L<���A�**;


total_loss��x@

error_R�AX?

learning_rate_1��j7��!I       6%�	�eL<���A�**;


total_loss�@

error_R�L?

learning_rate_1��j7��h�I       6%�	��L<���A�**;


total_loss�A

error_R@S?

learning_rate_1��j7���I       6%�	!�L<���A�**;


total_loss=��@

error_R�>P?

learning_rate_1��j7�\nI       6%�	�.M<���A�**;


total_loss4�@

error_RozJ?

learning_rate_1��j7x�<NI       6%�	�vM<���A�**;


total_loss=��@

error_R}hI?

learning_rate_1��j7�4GI       6%�	��M<���A�**;


total_loss���@

error_Rq�B?

learning_rate_1��j7T�I       6%�	�N<���A�**;


total_loss���@

error_Rd�P?

learning_rate_1��j7���I       6%�	�[N<���A�**;


total_loss�l�@

error_Rf�A?

learning_rate_1��j7�U�I       6%�	��N<���A�**;


total_loss�A

error_R�Y?

learning_rate_1��j7�W�9I       6%�	-�N<���A�**;


total_loss���@

error_R�}V?

learning_rate_1��j70���I       6%�	�GO<���A�**;


total_loss2��@

error_R/�X?

learning_rate_1��j7��=I       6%�	ӏO<���A�**;


total_loss���@

error_RHrU?

learning_rate_1��j7��ܓI       6%�	�O<���A�**;


total_loss��@

error_R�xL?

learning_rate_1��j7���I       6%�	�P<���A�+*;


total_lossO��@

error_RsoB?

learning_rate_1��j7�7��I       6%�	�aP<���A�+*;


total_lossӫ@

error_R3�K?

learning_rate_1��j7���_I       6%�	[�P<���A�+*;


total_loss[�t@

error_R��L?

learning_rate_1��j7	�g�I       6%�	��P<���A�+*;


total_loss�k�@

error_R��K?

learning_rate_1��j7� ��I       6%�	�-Q<���A�+*;


total_lossج�@

error_R��>?

learning_rate_1��j7O��I       6%�	�tQ<���A�+*;


total_lossT��@

error_Ri�O?

learning_rate_1��j7E�^I       6%�	��Q<���A�+*;


total_loss?�@

error_R�b]?

learning_rate_1��j7�|�I       6%�	�R<���A�+*;


total_loss��_@

error_R<lJ?

learning_rate_1��j7*k��I       6%�	�TR<���A�+*;


total_loss-e@

error_R4�C?

learning_rate_1��j7ɛ��I       6%�	�R<���A�+*;


total_loss��@

error_R�4O?

learning_rate_1��j7^��nI       6%�	��R<���A�+*;


total_lossth@

error_RQjS?

learning_rate_1��j7��{I       6%�	<S<���A�+*;


total_loss3b�@

error_RH?

learning_rate_1��j7t/T�I       6%�	�S<���A�+*;


total_lossS9�@

error_R��P?

learning_rate_1��j7��9I       6%�	��S<���A�+*;


total_losss��@

error_R�~A?

learning_rate_1��j7�-�I       6%�	T<���A�+*;


total_loss�P�@

error_R*�W?

learning_rate_1��j7��_]I       6%�	OT<���A�+*;


total_loss�r�@

error_R�}P?

learning_rate_1��j7����I       6%�	��T<���A�+*;


total_loss3P A

error_R6K?

learning_rate_1��j7�K2I       6%�	��T<���A�+*;


total_loss���@

error_R��@?

learning_rate_1��j7 ��?I       6%�	0*U<���A�+*;


total_loss�б@

error_R�>Q?

learning_rate_1��j7q"�II       6%�	�pU<���A�+*;


total_loss���@

error_R{W?

learning_rate_1��j7�5o0I       6%�	��U<���A�+*;


total_lossв�@

error_R�`?

learning_rate_1��j7g��I       6%�	x�U<���A�+*;


total_loss��@

error_RڣJ?

learning_rate_1��j7?/_I       6%�	JV<���A�+*;


total_loss���@

error_R��G?

learning_rate_1��j7��I       6%�	�V<���A�+*;


total_loss&͗@

error_R�X?

learning_rate_1��j7�H��I       6%�	��V<���A�+*;


total_lossэ�@

error_R�E9?

learning_rate_1��j7�)I       6%�	�9W<���A�+*;


total_lossg�	A

error_R �^?

learning_rate_1��j7�s$I       6%�	{�W<���A�+*;


total_loss�8A

error_R�,I?

learning_rate_1��j7��)I       6%�	N�W<���A�+*;


total_lossᓒ@

error_R�Q?

learning_rate_1��j7��ۨI       6%�	�X<���A�+*;


total_loss�V�@

error_R��;?

learning_rate_1��j7��I       6%�	�YX<���A�+*;


total_loss���@

error_R�HI?

learning_rate_1��j7bB�I       6%�	�X<���A�+*;


total_loss�&v@

error_R�\N?

learning_rate_1��j7����I       6%�	+�X<���A�+*;


total_loss�#A

error_RF�F?

learning_rate_1��j7\�kI       6%�	�(Y<���A�+*;


total_lossAC�@

error_R�W`?

learning_rate_1��j7�QI       6%�	�tY<���A�+*;


total_lossʫ�@

error_R�bI?

learning_rate_1��j7]��I       6%�	ݽY<���A�+*;


total_lossھ�@

error_RS'H?

learning_rate_1��j7��u�I       6%�	� Z<���A�+*;


total_loss䛍@

error_R��G?

learning_rate_1��j76`�I       6%�	pEZ<���A�+*;


total_loss�v�@

error_R8AO?

learning_rate_1��j7"��I       6%�	]�Z<���A�+*;


total_loss]j~@

error_R�Q9?

learning_rate_1��j7��ݥI       6%�	��Z<���A�+*;


total_losss�;A

error_R��I?

learning_rate_1��j7Kq7bI       6%�	r[<���A�+*;


total_loss��A

error_R;�P?

learning_rate_1��j7�p�TI       6%�	�S[<���A�+*;


total_loss �@

error_R!�T?

learning_rate_1��j7��2I       6%�	��[<���A�+*;


total_loss�{�@

error_R�I?

learning_rate_1��j7�%I       6%�	��[<���A�+*;


total_loss�*�@

error_R�"Q?

learning_rate_1��j7���I       6%�	�'\<���A�+*;


total_loss�?A

error_R1!W?

learning_rate_1��j7�G?�I       6%�	3m\<���A�+*;


total_loss�u�@

error_RU?

learning_rate_1��j7[�jI       6%�	$�\<���A�+*;


total_loss���@

error_Rv�[?

learning_rate_1��j7޺�BI       6%�	f]<���A�+*;


total_loss[��@

error_RCbL?

learning_rate_1��j7�"�+I       6%�	�M]<���A�+*;


total_lossVjA

error_R)�G?

learning_rate_1��j7懜/I       6%�	�]<���A�+*;


total_lossG��@

error_R�/N?

learning_rate_1��j7_��|I       6%�	��]<���A�+*;


total_loss�ZA

error_R-Q?

learning_rate_1��j7� ��I       6%�	�^<���A�+*;


total_loss�#�@

error_R�L?

learning_rate_1��j7�E�I       6%�	�Y^<���A�+*;


total_loss}�@

error_Rd�>?

learning_rate_1��j7Fڈ�I       6%�	۝^<���A�+*;


total_lossӬ@

error_Ra�Z?

learning_rate_1��j7����I       6%�	��^<���A�+*;


total_loss�-�@

error_R��C?

learning_rate_1��j7dQ�HI       6%�	)#_<���A�+*;


total_lossQs�@

error_R;�Q?

learning_rate_1��j7�?|I       6%�	/f_<���A�+*;


total_loss	O�@

error_R��c?

learning_rate_1��j7Ӗ�QI       6%�	˧_<���A�+*;


total_lossc��@

error_R��N?

learning_rate_1��j7��ČI       6%�	A�_<���A�+*;


total_loss��@

error_R-�>?

learning_rate_1��j7����I       6%�	'/`<���A�+*;


total_loss?�@

error_RQ�V?

learning_rate_1��j7"��I       6%�	Ys`<���A�+*;


total_lossE�@

error_R�#K?

learning_rate_1��j7��dI       6%�	��`<���A�+*;


total_loss@�@

error_RW�]?

learning_rate_1��j7|���I       6%�	��`<���A�+*;


total_loss%�@

error_R�L?

learning_rate_1��j7�;SfI       6%�	�Aa<���A�+*;


total_loss�o�@

error_R`~X?

learning_rate_1��j7�YXI       6%�	{�a<���A�+*;


total_loss};�@

error_R��F?

learning_rate_1��j7nrX�I       6%�	��a<���A�+*;


total_loss3��@

error_R J?

learning_rate_1��j7ͦrI       6%�	
b<���A�+*;


total_loss[|�@

error_R|�Q?

learning_rate_1��j7��l�I       6%�	-Rb<���A�+*;


total_lossH2�@

error_R��L?

learning_rate_1��j7�s��I       6%�	0�b<���A�+*;


total_loss@��@

error_R$�3?

learning_rate_1��j7�G"�I       6%�	�b<���A�+*;


total_lossT��@

error_Rx\>?

learning_rate_1��j7T�D�I       6%�	c/c<���A�+*;


total_loss!��@

error_RE�_?

learning_rate_1��j7�i>I       6%�	;uc<���A�+*;


total_loss�`�@

error_Ra�H?

learning_rate_1��j7h I       6%�	m�c<���A�+*;


total_lossCu�@

error_R6�P?

learning_rate_1��j7)*d�I       6%�	fd<���A�+*;


total_loss���@

error_R�MT?

learning_rate_1��j7��
I       6%�	�Dd<���A�+*;


total_loss�C�@

error_R��<?

learning_rate_1��j7�&�eI       6%�	-�d<���A�+*;


total_lossA'�@

error_R��S?

learning_rate_1��j7�X�oI       6%�	��d<���A�+*;


total_loss��@

error_RI?

learning_rate_1��j7z��	I       6%�	�e<���A�+*;


total_loss���@

error_R(Fc?

learning_rate_1��j7����I       6%�	�Re<���A�+*;


total_loss�Q�@

error_R��L?

learning_rate_1��j7ze}�I       6%�	�e<���A�+*;


total_lossP�@

error_RW�N?

learning_rate_1��j7�/}&I       6%�	�e<���A�+*;


total_loss߷�@

error_R�K?

learning_rate_1��j7�{c$I       6%�	�"f<���A�+*;


total_loss.�@

error_R��L?

learning_rate_1��j7µP�I       6%�	��f<���A�+*;


total_lossش@

error_R�84?

learning_rate_1��j7�$�I       6%�	S�f<���A�+*;


total_loss���@

error_RO2[?

learning_rate_1��j7��+�I       6%�	�g<���A�+*;


total_loss���@

error_R�;?

learning_rate_1��j7�:|I       6%�	Lfg<���A�+*;


total_loss��@

error_Rq�V?

learning_rate_1��j7`m�I       6%�	�g<���A�+*;


total_loss�Ĭ@

error_Rw�R?

learning_rate_1��j7�b�:I       6%�	��g<���A�+*;


total_loss���@

error_R��P?

learning_rate_1��j7���I       6%�	g2h<���A�+*;


total_loss
�@

error_R=�H?

learning_rate_1��j7J�A�I       6%�	<yh<���A�+*;


total_loss{��@

error_R�R?

learning_rate_1��j7m���I       6%�	k�h<���A�+*;


total_loss
��@

error_R��K?

learning_rate_1��j7cF�QI       6%�	�i<���A�+*;


total_lossϵx@

error_R�H?

learning_rate_1��j7�1I       6%�	Ii<���A�+*;


total_loss扵@

error_R�qP?

learning_rate_1��j72BqI       6%�	~�i<���A�+*;


total_loss賟@

error_RjD?

learning_rate_1��j7��I       6%�	v�i<���A�+*;


total_loss�a�@

error_RZEG?

learning_rate_1��j7!C2I       6%�	8j<���A�+*;


total_loss�l@

error_R��G?

learning_rate_1��j7�0�gI       6%�	Yj<���A�+*;


total_loss�@

error_ROvS?

learning_rate_1��j7�6�-I       6%�	��j<���A�+*;


total_loss�z@

error_R� 9?

learning_rate_1��j7E�I       6%�	�j<���A�+*;


total_loss��A

error_R,�A?

learning_rate_1��j7���DI       6%�	�'k<���A�+*;


total_loss���@

error_R�DM?

learning_rate_1��j7�2�I       6%�	[ik<���A�+*;


total_loss���@

error_R�%V?

learning_rate_1��j7
�s�I       6%�	�k<���A�+*;


total_loss���@

error_R��=?

learning_rate_1��j7gCV,I       6%�	�k<���A�+*;


total_lossd A

error_R��Q?

learning_rate_1��j7ɉ��I       6%�	i8l<���A�+*;


total_lossW�Z@

error_R�hP?

learning_rate_1��j7cH�I       6%�	#~l<���A�+*;


total_loss���@

error_R��U?

learning_rate_1��j7�%�I       6%�	r�l<���A�+*;


total_loss��@

error_R�\B?

learning_rate_1��j7��;�I       6%�	tLm<���A�+*;


total_loss�g�@

error_Ra�S?

learning_rate_1��j7YX�I       6%�	h�m<���A�+*;


total_loss�A�@

error_RfBQ?

learning_rate_1��j7�nLI       6%�	��m<���A�+*;


total_loss�A

error_RZ K?

learning_rate_1��j7��>dI       6%�	U4n<���A�+*;


total_loss|��@

error_R�E?

learning_rate_1��j7�'�I       6%�	zn<���A�+*;


total_loss�@

error_R.6=?

learning_rate_1��j7+�M6I       6%�	B�n<���A�+*;


total_loss<v@

error_R�Z?

learning_rate_1��j7Ĕ�9I       6%�	�o<���A�+*;


total_loss-��@

error_R3�G?

learning_rate_1��j7�q�zI       6%�	�Xo<���A�+*;


total_loss�'�@

error_R�HU?

learning_rate_1��j7<:�I       6%�	I�o<���A�+*;


total_loss�c�@

error_R��??

learning_rate_1��j7jζ&I       6%�	i�o<���A�+*;


total_loss]�@

error_R�6L?

learning_rate_1��j7nϽI       6%�	�p<���A�+*;


total_loss�\�@

error_R�A?

learning_rate_1��j7���I       6%�		bp<���A�+*;


total_loss;Ds@

error_R�SB?

learning_rate_1��j7k��FI       6%�	_�p<���A�+*;


total_loss��@

error_R��S?

learning_rate_1��j7��%I       6%�	Z�p<���A�+*;


total_loss��@

error_RZ8Z?

learning_rate_1��j7�' �I       6%�	�3q<���A�+*;


total_losso��@

error_R,F=?

learning_rate_1��j7����I       6%�	({q<���A�+*;


total_loss�v�@

error_RŕI?

learning_rate_1��j7@.?I       6%�	��q<���A�+*;


total_loss�W�@

error_R�KR?

learning_rate_1��j7����I       6%�	�r<���A�+*;


total_loss〧@

error_R}�E?

learning_rate_1��j7QY0�I       6%�	qWr<���A�+*;


total_loss<p�@

error_R��Q?

learning_rate_1��j7��SI       6%�	�r<���A�+*;


total_loss��]@

error_REZ?

learning_rate_1��j7�p�I       6%�	��r<���A�+*;


total_loss�<�@

error_R�~V?

learning_rate_1��j7�j��I       6%�	>&s<���A�+*;


total_loss� �@

error_R��V?

learning_rate_1��j7�9XI       6%�	�gs<���A�+*;


total_loss�@

error_R��B?

learning_rate_1��j7i�>�I       6%�	�s<���A�,*;


total_loss��@

error_R��H?

learning_rate_1��j7 ��I       6%�	��s<���A�,*;


total_loss�B�@

error_R�MO?

learning_rate_1��j7f� �I       6%�	�3t<���A�,*;


total_loss�٠@

error_RN�T?

learning_rate_1��j7]�JeI       6%�	�tt<���A�,*;


total_loss���@

error_R_�N?

learning_rate_1��j7֩T�I       6%�	Q�t<���A�,*;


total_lossY�@

error_R�pI?

learning_rate_1��j7�3vI       6%�	>�t<���A�,*;


total_loss�w�@

error_R��J?

learning_rate_1��j7���RI       6%�	�Cu<���A�,*;


total_loss<A

error_R�a?

learning_rate_1��j7�s�I       6%�	m�u<���A�,*;


total_lossZwA

error_RC>F?

learning_rate_1��j7�]�I       6%�	&�u<���A�,*;


total_loss#[�@

error_R�]R?

learning_rate_1��j7�8�I       6%�	v<���A�,*;


total_lossl�A

error_RH�K?

learning_rate_1��j7!��EI       6%�	[xv<���A�,*;


total_lossV�@

error_R6�L?

learning_rate_1��j7�~'9I       6%�	y�v<���A�,*;


total_loss�{�@

error_R��K?

learning_rate_1��j7 n��I       6%�	gw<���A�,*;


total_loss��@

error_RH�S?

learning_rate_1��j7K��wI       6%�	,\w<���A�,*;


total_loss\�A

error_R��??

learning_rate_1��j7AacI       6%�	��w<���A�,*;


total_lossf}�@

error_R�xX?

learning_rate_1��j7�(m@I       6%�	��w<���A�,*;


total_loss��@

error_R��N?

learning_rate_1��j7#��mI       6%�	�6x<���A�,*;


total_loss!�@

error_R�D?

learning_rate_1��j7��^�I       6%�	Ёx<���A�,*;


total_loss���@

error_RiEE?

learning_rate_1��j7s��I       6%�	�x<���A�,*;


total_loss�Ѩ@

error_R �P?

learning_rate_1��j7�А�I       6%�	�
y<���A�,*;


total_lossvQ�@

error_R�7?

learning_rate_1��j7�1�'I       6%�	�Ny<���A�,*;


total_loss11�@

error_R�B?

learning_rate_1��j7/�I       6%�	`�y<���A�,*;


total_lossov�@

error_RҬI?

learning_rate_1��j7���I       6%�	��y<���A�,*;


total_loss���@

error_R=�E?

learning_rate_1��j7��ɨI       6%�	�z<���A�,*;


total_loss+ކ@

error_RC�W?

learning_rate_1��j7<UI       6%�	o_z<���A�,*;


total_loss�A

error_Rzi<?

learning_rate_1��j7�Ǫ<I       6%�	1�z<���A�,*;


total_loss�6�@

error_R
�N?

learning_rate_1��j7O�jAI       6%�	��z<���A�,*;


total_loss��A

error_R�-Q?

learning_rate_1��j793�[I       6%�	�-{<���A�,*;


total_loss�_�@

error_RfZB?

learning_rate_1��j7xg��I       6%�	Or{<���A�,*;


total_loss�I@

error_R
2R?

learning_rate_1��j7��.I       6%�	r�{<���A�,*;


total_loss���@

error_R��@?

learning_rate_1��j7�fMI       6%�	�|<���A�,*;


total_loss�:A

error_RnK?

learning_rate_1��j7OOHI       6%�	�E|<���A�,*;


total_lossX�@

error_R�^I?

learning_rate_1��j75)��I       6%�	f�|<���A�,*;


total_loss���@

error_R��X?

learning_rate_1��j78F�9I       6%�	q�|<���A�,*;


total_loss=��@

error_RsR?

learning_rate_1��j7OO?�I       6%�	}<���A�,*;


total_lossX_�@

error_Rf�T?

learning_rate_1��j7���I       6%�	eV}<���A�,*;


total_loss��@

error_R.�K?

learning_rate_1��j7��bI       6%�	$�}<���A�,*;


total_loss��@

error_RF`?

learning_rate_1��j7�T�!I       6%�	#�}<���A�,*;


total_loss�$�@

error_Rd�^?

learning_rate_1��j7|��4I       6%�	�(~<���A�,*;


total_loss��@

error_R�d[?

learning_rate_1��j7��T�I       6%�	�i~<���A�,*;


total_loss.�@

error_Rs�[?

learning_rate_1��j7dO�-I       6%�	V�~<���A�,*;


total_lossnm�@

error_R�H?

learning_rate_1��j7w&I       6%�	��~<���A�,*;


total_lossI��@

error_R�O?

learning_rate_1��j7rBWI       6%�	�0<���A�,*;


total_lossq��@

error_R��V?

learning_rate_1��j7�9�uI       6%�	�z<���A�,*;


total_loss�[�@

error_RK?

learning_rate_1��j7�'�|I       6%�	��<���A�,*;


total_loss�b�@

error_R��O?

learning_rate_1��j7M���I       6%�	��<���A�,*;


total_loss&� A

error_R�a?

learning_rate_1��j7&HSI       6%�	�I�<���A�,*;


total_loss�`�@

error_R��J?

learning_rate_1��j7*/LI       6%�	�<���A�,*;


total_loss��@

error_Rϼ\?

learning_rate_1��j7&KM�I       6%�	р<���A�,*;


total_lossT��@

error_R)�J?

learning_rate_1��j7%{�I       6%�	d�<���A�,*;


total_loss�B�@

error_R�SP?

learning_rate_1��j7F�P=I       6%�	�X�<���A�,*;


total_lossxݏ@

error_REE?

learning_rate_1��j7/9C�I       6%�	���<���A�,*;


total_loss��@

error_R��H?

learning_rate_1��j7�]��I       6%�	��<���A�,*;


total_loss�R@

error_R�)??

learning_rate_1��j7Ζ%SI       6%�	�$�<���A�,*;


total_loss4l@

error_R��O?

learning_rate_1��j75��I       6%�	i�<���A�,*;


total_loss�ڔ@

error_R��J?

learning_rate_1��j7��`�I       6%�	d��<���A�,*;


total_lossc^�@

error_R3"O?

learning_rate_1��j7��I       6%�	��<���A�,*;


total_loss@8�@

error_Ra%R?

learning_rate_1��j7o�+�I       6%�	�6�<���A�,*;


total_loss{�@

error_R�jP?

learning_rate_1��j7I*7I       6%�	�<���A�,*;


total_lossx �@

error_R��U?

learning_rate_1��j7��X�I       6%�	Qȃ<���A�,*;


total_loss�p�@

error_Ra�D?

learning_rate_1��j7r�WI       6%�	��<���A�,*;


total_loss<mA

error_R �O?

learning_rate_1��j7�%�II       6%�	4\�<���A�,*;


total_loss��@

error_Rn<E?

learning_rate_1��j7��-�I       6%�	���<���A�,*;


total_loss���@

error_R��R?

learning_rate_1��j7OUBBI       6%�	��<���A�,*;


total_lossO�@

error_RdaD?

learning_rate_1��j7����I       6%�	�<�<���A�,*;


total_loss?�@

error_R�Z?

learning_rate_1��j7�2�I       6%�	ۀ�<���A�,*;


total_loss��@

error_R<e?

learning_rate_1��j7�i��I       6%�	pȅ<���A�,*;


total_loss��@

error_R�(U?

learning_rate_1��j7����I       6%�	c	�<���A�,*;


total_loss��@

error_R�TJ?

learning_rate_1��j7�i�I       6%�	b_�<���A�,*;


total_loss�q}@

error_Rw�B?

learning_rate_1��j7K�ԅI       6%�	��<���A�,*;


total_loss|�@

error_R\vI?

learning_rate_1��j7��'�I       6%�	��<���A�,*;


total_lossA��@

error_RH�D?

learning_rate_1��j7(pLI       6%�	U�<���A�,*;


total_lossZ�@

error_R�M?

learning_rate_1��j74���I       6%�	���<���A�,*;


total_lossr��@

error_R��J?

learning_rate_1��j7��,I       6%�	��<���A�,*;


total_loss���@

error_RFT?

learning_rate_1��j7�ʯ�I       6%�	0&�<���A�,*;


total_loss�f�@

error_RHEP?

learning_rate_1��j7�փ2I       6%�	Eh�<���A�,*;


total_loss�"�@

error_R!�P?

learning_rate_1��j71���I       6%�	ૈ<���A�,*;


total_loss���@

error_RnLN?

learning_rate_1��j7�MI       6%�	���<���A�,*;


total_loss�Nk@

error_R�GF?

learning_rate_1��j7�x�CI       6%�	�;�<���A�,*;


total_loss���@

error_RdIL?

learning_rate_1��j7T}]'I       6%�	2�<���A�,*;


total_lossX�@

error_RGN?

learning_rate_1��j7����I       6%�	�ĉ<���A�,*;


total_loss��@

error_R1bG?

learning_rate_1��j7xlq`I       6%�	w�<���A�,*;


total_loss$��@

error_R#LM?

learning_rate_1��j7���I       6%�	pK�<���A�,*;


total_loss�&�@

error_R� <?

learning_rate_1��j7S��XI       6%�	��<���A�,*;


total_loss��@

error_R�E?

learning_rate_1��j7��I       6%�	M֊<���A�,*;


total_loss4�A

error_RM>P?

learning_rate_1��j7���I       6%�	R�<���A�,*;


total_loss�E�@

error_R<�C?

learning_rate_1��j7�$c�I       6%�	Tc�<���A�,*;


total_loss�I�@

error_R1eG?

learning_rate_1��j7kf'HI       6%�	���<���A�,*;


total_loss���@

error_RR�F?

learning_rate_1��j7E��bI       6%�	��<���A�,*;


total_lossC�@

error_R�}S?

learning_rate_1��j7�?)I       6%�	�.�<���A�,*;


total_lossI�@

error_R)e@?

learning_rate_1��j7���;I       6%�	3u�<���A�,*;


total_loss�j�@

error_Rq?>?

learning_rate_1��j7ȱ��I       6%�	&��<���A�,*;


total_lossi?�@

error_RX�I?

learning_rate_1��j7�˒yI       6%�	^��<���A�,*;


total_loss�̈́@

error_R;�N?

learning_rate_1��j7���I       6%�	�=�<���A�,*;


total_loss��@

error_R�D?

learning_rate_1��j7'���I       6%�	؆�<���A�,*;


total_lossE��@

error_R�]?

learning_rate_1��j7j�C�I       6%�	�ލ<���A�,*;


total_losso0�@

error_R��X?

learning_rate_1��j7�D�BI       6%�	�$�<���A�,*;


total_loss���@

error_R�@S?

learning_rate_1��j7&Y�WI       6%�	�i�<���A�,*;


total_lossa��@

error_R�NM?

learning_rate_1��j7f�HI       6%�	<��<���A�,*;


total_loss��@

error_Ra�@?

learning_rate_1��j7X@��I       6%�	��<���A�,*;


total_loss���@

error_R�d;?

learning_rate_1��j7��P�I       6%�	�T�<���A�,*;


total_loss��A

error_R.�S?

learning_rate_1��j7�LJ�I       6%�	[��<���A�,*;


total_loss���@

error_R�Q?

learning_rate_1��j7����I       6%�	�ޏ<���A�,*;


total_loss��@

error_R��U?

learning_rate_1��j7]e9�I       6%�	�!�<���A�,*;


total_loss��@

error_R�{c?

learning_rate_1��j7����I       6%�	Rh�<���A�,*;


total_lossr�@

error_R��;?

learning_rate_1��j7����I       6%�	̪�<���A�,*;


total_loss�Ī@

error_RA�Y?

learning_rate_1��j7�>�<I       6%�	��<���A�,*;


total_loss襈@

error_R��I?

learning_rate_1��j7%]I       6%�	�2�<���A�,*;


total_loss_c�@

error_R{fP?

learning_rate_1��j7�L�I       6%�	}w�<���A�,*;


total_loss���@

error_R]$J?

learning_rate_1��j7G׉�I       6%�	]��<���A�,*;


total_loss;�@

error_R��A?

learning_rate_1��j78�3I       6%�	X	�<���A�,*;


total_lossCe�@

error_R�@?

learning_rate_1��j7����I       6%�	�P�<���A�,*;


total_loss��@

error_R&�\?

learning_rate_1��j7�N��I       6%�	���<���A�,*;


total_loss��@

error_R�U?

learning_rate_1��j7���BI       6%�	�ڒ<���A�,*;


total_loss�n�@

error_R^D?

learning_rate_1��j7�gI       6%�	�<���A�,*;


total_lossi�@

error_R��W?

learning_rate_1��j7���I       6%�	a�<���A�,*;


total_loss��@

error_R�,9?

learning_rate_1��j7�qyI       6%�	��<���A�,*;


total_loss��@

error_RC�N?

learning_rate_1��j7����I       6%�	��<���A�,*;


total_loss�R�@

error_Rx�T?

learning_rate_1��j7�bq�I       6%�	�)�<���A�,*;


total_loss΢�@

error_Rd~K?

learning_rate_1��j7�RG�I       6%�	|m�<���A�,*;


total_loss���@

error_R=�J?

learning_rate_1��j7���FI       6%�	ݲ�<���A�,*;


total_lossc$�@

error_R��9?

learning_rate_1��j7�{v�I       6%�	��<���A�,*;


total_loss�e�@

error_RWF`?

learning_rate_1��j7N�hI       6%�	M\�<���A�,*;


total_loss-��@

error_Rj~U?

learning_rate_1��j7Xc�zI       6%�	?��<���A�,*;


total_loss�9�@

error_R@�9?

learning_rate_1��j7!�ΑI       6%�	��<���A�,*;


total_loss_�@

error_RH�C?

learning_rate_1��j7 ���I       6%�	v;�<���A�,*;


total_loss��@

error_R��]?

learning_rate_1��j7GX)I       6%�	���<���A�,*;


total_lossQ�@

error_R�s=?

learning_rate_1��j7z�>	I       6%�	 �<���A�,*;


total_loss�T�@

error_R�pD?

learning_rate_1��j7.T=�I       6%�	�2�<���A�-*;


total_loss$Ţ@

error_R=�F?

learning_rate_1��j7�ҕ�I       6%�	�y�<���A�-*;


total_lossߋ�@

error_Rr�K?

learning_rate_1��j7�,��I       6%�	׼�<���A�-*;


total_loss�Ư@

error_R�UZ?

learning_rate_1��j7#�;�I       6%�	�<���A�-*;


total_lossQ2�@

error_R͂:?

learning_rate_1��j7�qFI       6%�	2G�<���A�-*;


total_lossR��@

error_R�I?

learning_rate_1��j7�,�_I       6%�	<���A�-*;


total_lossl�p@

error_R�R[?

learning_rate_1��j7Δ04I       6%�	Gʘ<���A�-*;


total_loss֯�@

error_R��P?

learning_rate_1��j7窈�I       6%�	��<���A�-*;


total_lossJ�@

error_R8�I?

learning_rate_1��j7�*4I       6%�	�Q�<���A�-*;


total_lossp�A

error_R{+V?

learning_rate_1��j7Q��(I       6%�	���<���A�-*;


total_lossxM�@

error_R��O?

learning_rate_1��j7d|�DI       6%�	�ڙ<���A�-*;


total_loss�ߔ@

error_R�8?

learning_rate_1��j7X)a@I       6%�	��<���A�-*;


total_lossM��@

error_R�*Z?

learning_rate_1��j7@�q(I       6%�	�a�<���A�-*;


total_loss�j�@

error_R�w@?

learning_rate_1��j7{1?�I       6%�	F��<���A�-*;


total_losso�@

error_R�!R?

learning_rate_1��j7�I       6%�	 �<���A�-*;


total_loss7հ@

error_Rz1J?

learning_rate_1��j7�(�I       6%�	�2�<���A�-*;


total_loss��@

error_R�W?

learning_rate_1��j7�QT�I       6%�	�x�<���A�-*;


total_loss��@

error_R�8?

learning_rate_1��j7�hc�I       6%�	��<���A�-*;


total_loss:�@

error_R�dS?

learning_rate_1��j7���BI       6%�	V��<���A�-*;


total_lossi=�@

error_R�jK?

learning_rate_1��j7F>P�I       6%�	�>�<���A�-*;


total_loss4�@

error_R�N?

learning_rate_1��j7�$�I       6%�	���<���A�-*;


total_loss�@

error_R}�U?

learning_rate_1��j7���I       6%�	�ќ<���A�-*;


total_lossqE�@

error_R�2\?

learning_rate_1��j7.���I       6%�	��<���A�-*;


total_lossn:�@

error_R ZR?

learning_rate_1��j7��/I       6%�	�`�<���A�-*;


total_loss�Q�@

error_R��N?

learning_rate_1��j7�v�I       6%�	��<���A�-*;


total_loss�r�@

error_R�)I?

learning_rate_1��j7fIx�I       6%�	��<���A�-*;


total_lossIG�@

error_Rt�<?

learning_rate_1��j7�N��I       6%�	]0�<���A�-*;


total_loss���@

error_RHL_?

learning_rate_1��j7N$XI       6%�	x�<���A�-*;


total_loss寭@

error_R V?

learning_rate_1��j7Hn<]I       6%�	���<���A�-*;


total_loss�2�@

error_R%K?

learning_rate_1��j7�)�KI       6%�	y�<���A�-*;


total_loss��@

error_R��J?

learning_rate_1��j7�@��I       6%�	�O�<���A�-*;


total_loss��@

error_RA�K?

learning_rate_1��j7$�XI       6%�	��<���A�-*;


total_lossv�@

error_R�}U?

learning_rate_1��j7��Z�I       6%�	l�<���A�-*;


total_loss̓A

error_R=U?

learning_rate_1��j7-&P�I       6%�	E.�<���A�-*;


total_loss!��@

error_R��D?

learning_rate_1��j7����I       6%�	�r�<���A�-*;


total_loss��@

error_R}$O?

learning_rate_1��j7+�}�I       6%�	趠<���A�-*;


total_loss��@

error_RC�T?

learning_rate_1��j7j+�.I       6%�	���<���A�-*;


total_loss��@

error_RoJ?

learning_rate_1��j76��!I       6%�	?�<���A�-*;


total_losshϤ@

error_RRpH?

learning_rate_1��j7��YI       6%�	���<���A�-*;


total_loss�=�@

error_RJ�W?

learning_rate_1��j7��3I       6%�	�ʡ<���A�-*;


total_lossC	�@

error_Rm�c?

learning_rate_1��j7�t�BI       6%�	�<���A�-*;


total_loss�ı@

error_R�N?

learning_rate_1��j7A	�sI       6%�	�Y�<���A�-*;


total_loss���@

error_R͞J?

learning_rate_1��j7��_4I       6%�	٠�<���A�-*;


total_loss%��@

error_R߾R?

learning_rate_1��j7I(�NI       6%�	w�<���A�-*;


total_loss��@

error_REiE?

learning_rate_1��j7S�LI       6%�	�+�<���A�-*;


total_loss�R�@

error_R�U?

learning_rate_1��j7�
j�I       6%�	�n�<���A�-*;


total_loss�@

error_RluM?

learning_rate_1��j7��&�I       6%�	���<���A�-*;


total_loss}��@

error_R;OP?

learning_rate_1��j7���HI       6%�	���<���A�-*;


total_loss�?�@

error_RTg>?

learning_rate_1��j7�XGWI       6%�	!@�<���A�-*;


total_loss[i�@

error_R,;??

learning_rate_1��j7{qftI       6%�	Ʌ�<���A�-*;


total_loss���@

error_R��;?

learning_rate_1��j7䤬�I       6%�	�̤<���A�-*;


total_loss:�@

error_RnN?

learning_rate_1��j7��I       6%�	��<���A�-*;


total_loss�z�@

error_Rl�J?

learning_rate_1��j7
��I       6%�	-\�<���A�-*;


total_loss�:�@

error_R3�K?

learning_rate_1��j7'�!I       6%�	��<���A�-*;


total_loss�Ȳ@

error_R�VB?

learning_rate_1��j7�Eq�I       6%�	q�<���A�-*;


total_loss\`�@

error_R��O?

learning_rate_1��j7�y��I       6%�	Z>�<���A�-*;


total_loss��@

error_R��P?

learning_rate_1��j7��WNI       6%�	.��<���A�-*;


total_lossfax@

error_R&U?

learning_rate_1��j7�M>qI       6%�	r�<���A�-*;


total_loss���@

error_Rs3d?

learning_rate_1��j7�\�I       6%�	,�<���A�-*;


total_loss��@

error_R�N?

learning_rate_1��j7��#�I       6%�	Tn�<���A�-*;


total_loss*iA

error_R�4>?

learning_rate_1��j7CŜ�I       6%�	ݱ�<���A�-*;


total_lossl �@

error_R]�[?

learning_rate_1��j7��uI       6%�	���<���A�-*;


total_loss�)�@

error_R�J7?

learning_rate_1��j7���(I       6%�	�<�<���A�-*;


total_loss�@

error_R��g?

learning_rate_1��j7����I       6%�	���<���A�-*;


total_loss��@

error_Rl{]?

learning_rate_1��j7�'15I       6%�	�ͨ<���A�-*;


total_loss�X�@

error_R�OI?

learning_rate_1��j7Cǂ|I       6%�	j�<���A�-*;


total_lossFd�@

error_RHG[?

learning_rate_1��j72��|I       6%�	OY�<���A�-*;


total_lossϠ�@

error_R�%J?

learning_rate_1��j7�D*8I       6%�	ݘ�<���A�-*;


total_loss�f�@

error_R��Z?

learning_rate_1��j7Ԁj�I       6%�	%۩<���A�-*;


total_loss:��@

error_R�zH?

learning_rate_1��j7�ĠZI       6%�	z�<���A�-*;


total_loss�J�@

error_R.:E?

learning_rate_1��j7�-
jI       6%�	!h�<���A�-*;


total_loss�c�@

error_R�G?

learning_rate_1��j7��0�I       6%�	��<���A�-*;


total_loss�_�@

error_R��V?

learning_rate_1��j7�Ѳ�I       6%�	���<���A�-*;


total_loss&��@

error_R��\?

learning_rate_1��j7h9�I       6%�	�8�<���A�-*;


total_loss��@

error_R�P?

learning_rate_1��j7"�cQI       6%�	��<���A�-*;


total_lossH��@

error_RL?L?

learning_rate_1��j7����I       6%�	�ë<���A�-*;


total_lossm��@

error_R�VC?

learning_rate_1��j7❷I       6%�	 �<���A�-*;


total_lossj�@

error_R��N?

learning_rate_1��j7 .��I       6%�	�L�<���A�-*;


total_loss�3�@

error_RE�N?

learning_rate_1��j7v�*�I       6%�	H��<���A�-*;


total_losscc	A

error_R-�M?

learning_rate_1��j7~���I       6%�	�Ҭ<���A�-*;


total_loss�o�@

error_RH�;?

learning_rate_1��j7A�YI       6%�	-�<���A�-*;


total_lossmX�@

error_Rq[D?

learning_rate_1��j7\���I       6%�	iW�<���A�-*;


total_loss�~�@

error_R MT?

learning_rate_1��j7�o�RI       6%�	ƛ�<���A�-*;


total_loss`ץ@

error_R��U?

learning_rate_1��j7�eI�I       6%�	5�<���A�-*;


total_lossAe@

error_R!r?

learning_rate_1��j7,�͉I       6%�	�:�<���A�-*;


total_loss��@

error_R�ZO?

learning_rate_1��j7�	I       6%�	*��<���A�-*;


total_loss��r@

error_R}#Q?

learning_rate_1��j7��M_I       6%�	?̮<���A�-*;


total_loss��@

error_R��E?

learning_rate_1��j7HTY�I       6%�	��<���A�-*;


total_loss�8�@

error_R�L?

learning_rate_1��j7��[I       6%�	Z�<���A�-*;


total_loss�r@

error_R.�R?

learning_rate_1��j7c�2�I       6%�	���<���A�-*;


total_loss�Ҩ@

error_R��i?

learning_rate_1��j7�jI       6%�	�<���A�-*;


total_loss�y�@

error_R�H?

learning_rate_1��j7��I       6%�	I#�<���A�-*;


total_lossLj�@

error_R��Z?

learning_rate_1��j7[���I       6%�	�j�<���A�-*;


total_lossB�@

error_R�cQ?

learning_rate_1��j7����I       6%�	��<���A�-*;


total_loss�ڂ@

error_R
�G?

learning_rate_1��j7^ҫ�I       6%�	���<���A�-*;


total_loss 5�@

error_R�'A?

learning_rate_1��j7u�zI       6%�	,E�<���A�-*;


total_loss
�L@

error_Ro^M?

learning_rate_1��j7�a_>I       6%�	N��<���A�-*;


total_loss�j�@

error_R�`?

learning_rate_1��j7���;I       6%�	�ձ<���A�-*;


total_loss|��@

error_R�F?

learning_rate_1��j7�Q�VI       6%�	 �<���A�-*;


total_lossh,�@

error_R��^?

learning_rate_1��j7VwDKI       6%�	h�<���A�-*;


total_loss*:�@

error_R3�V?

learning_rate_1��j7h�tI       6%�	Q��<���A�-*;


total_loss�ʎ@

error_R	�I?

learning_rate_1��j7�#39I       6%�	O��<���A�-*;


total_lossOro@

error_R
�O?

learning_rate_1��j7t��]I       6%�	TA�<���A�-*;


total_loss/�@

error_R}H?

learning_rate_1��j7���I       6%�	톳<���A�-*;


total_loss�٫@

error_RH�W?

learning_rate_1��j7?A��I       6%�	Gɳ<���A�-*;


total_loss�{�@

error_R��V?

learning_rate_1��j7�-�OI       6%�	�<���A�-*;


total_loss��@

error_RZUB?

learning_rate_1��j7iAWI       6%�	�Z�<���A�-*;


total_lossW4�@

error_Rq6_?

learning_rate_1��j7\�ݺI       6%�	���<���A�-*;


total_loss�@

error_RM*E?

learning_rate_1��j7+�p*I       6%�	��<���A�-*;


total_lossCb�@

error_R��\?

learning_rate_1��j7�1�PI       6%�	�1�<���A�-*;


total_lossx%}@

error_R�X?

learning_rate_1��j7��VI       6%�	�z�<���A�-*;


total_loss8n�@

error_R��Q?

learning_rate_1��j7h���I       6%�	2õ<���A�-*;


total_loss��A

error_Rڙ;?

learning_rate_1��j7��YeI       6%�	F	�<���A�-*;


total_loss;J�@

error_R*�X?

learning_rate_1��j7��(%I       6%�	&^�<���A�-*;


total_lossf$�@

error_R5?

learning_rate_1��j7���I       6%�	���<���A�-*;


total_loss�R�@

error_R�?T?

learning_rate_1��j7]�;I       6%�	!��<���A�-*;


total_loss{3�@

error_R�|:?

learning_rate_1��j7���I       6%�	?�<���A�-*;


total_loss���@

error_RԴN?

learning_rate_1��j7��i�I       6%�	<���A�-*;


total_loss���@

error_R1�C?

learning_rate_1��j7��wI       6%�	�ŷ<���A�-*;


total_loss��@

error_Rf�Q?

learning_rate_1��j7ٿ�SI       6%�	k
�<���A�-*;


total_lossR6p@

error_R��K?

learning_rate_1��j7C��/I       6%�	�N�<���A�-*;


total_loss��a@

error_R�V?

learning_rate_1��j7�O�`I       6%�	<���A�-*;


total_loss1	�@

error_R��O?

learning_rate_1��j7t�;I       6%�	�ܸ<���A�-*;


total_loss��A

error_R��e?

learning_rate_1��j7���I       6%�	�"�<���A�-*;


total_loss���@

error_R�gJ?

learning_rate_1��j7w37I       6%�	�l�<���A�-*;


total_lossS�@

error_R�dP?

learning_rate_1��j7���;I       6%�	���<���A�-*;


total_loss�/�@

error_R�z??

learning_rate_1��j7��HI       6%�	��<���A�-*;


total_loss��@

error_R@�G?

learning_rate_1��j7�IT�I       6%�	A<�<���A�-*;


total_loss�@

error_Rݕ\?

learning_rate_1��j7$�_�I       6%�	Y��<���A�.*;


total_loss���@

error_R�<?

learning_rate_1��j7Uo�I       6%�	ƺ<���A�.*;


total_loss�u�@

error_RJ?

learning_rate_1��j7t<6�I       6%�	_	�<���A�.*;


total_loss8��@

error_R��J?

learning_rate_1��j7~�!I       6%�	�M�<���A�.*;


total_loss�;�@

error_R�OH?

learning_rate_1��j7�A��I       6%�	���<���A�.*;


total_loss���@

error_R��N?

learning_rate_1��j7k7U�I       6%�	zڻ<���A�.*;


total_loss���@

error_Rw`?

learning_rate_1��j7&�tI       6%�	T$�<���A�.*;


total_loss�3�@

error_R��R?

learning_rate_1��j7Z�=I       6%�	zo�<���A�.*;


total_loss��@

error_R8MH?

learning_rate_1��j7����I       6%�	n��<���A�.*;


total_loss���@

error_R��G?

learning_rate_1��j7\�"I       6%�	��<���A�.*;


total_loss�*�@

error_Ro�A?

learning_rate_1��j7��/I       6%�	UW�<���A�.*;


total_loss���@

error_Rm�T?

learning_rate_1��j7��l�I       6%�	���<���A�.*;


total_loss���@

error_R&�Z?

learning_rate_1��j7Z�I       6%�	߽<���A�.*;


total_loss�y�@

error_R�U?

learning_rate_1��j7�u�I       6%�	[&�<���A�.*;


total_loss���@

error_R��>?

learning_rate_1��j7U!�6I       6%�	jj�<���A�.*;


total_loss���@

error_R�fP?

learning_rate_1��j7��;�I       6%�	߯�<���A�.*;


total_loss;��@

error_R�a?

learning_rate_1��j7 3�I       6%�	���<���A�.*;


total_loss	��@

error_R Q`?

learning_rate_1��j7����I       6%�	V=�<���A�.*;


total_loss�@

error_R@x??

learning_rate_1��j7pCw�I       6%�	���<���A�.*;


total_loss��@

error_R$�Y?

learning_rate_1��j7)��~I       6%�	�ɿ<���A�.*;


total_loss���@

error_R��\?

learning_rate_1��j7F���I       6%�	��<���A�.*;


total_losse�@

error_R �N?

learning_rate_1��j7k5b I       6%�	aU�<���A�.*;


total_loss�A

error_R_�H?

learning_rate_1��j7JEI       6%�	���<���A�.*;


total_lossa��@

error_R��W?

learning_rate_1��j7r��I       6%�	b��<���A�.*;


total_loss��A

error_R��R?

learning_rate_1��j7�y�uI       6%�	� �<���A�.*;


total_loss��@

error_R	zB?

learning_rate_1��j7��PI       6%�	jc�<���A�.*;


total_loss�m�@

error_R$;J?

learning_rate_1��j7Mذ�I       6%�	y��<���A�.*;


total_loss��jA

error_R��E?

learning_rate_1��j7����I       6%�	.��<���A�.*;


total_loss:y�@

error_R��<?

learning_rate_1��j7�h�I       6%�	1�<���A�.*;


total_loss�,�@

error_RxA?

learning_rate_1��j7ة<I       6%�	Ew�<���A�.*;


total_loss�W�@

error_R�Q?

learning_rate_1��j7)�Q�I       6%�	l��<���A�.*;


total_loss�ѓ@

error_R�R?

learning_rate_1��j7�+��I       6%�	f�<���A�.*;


total_loss!ɍ@

error_RZ9C?

learning_rate_1��j7�z��I       6%�	VD�<���A�.*;


total_lossϙ@

error_R�fV?

learning_rate_1��j7����I       6%�	���<���A�.*;


total_lossǛ@

error_R�OP?

learning_rate_1��j7�*�I       6%�	���<���A�.*;


total_loss=O�@

error_R4d?

learning_rate_1��j7�z1I       6%�	q
�<���A�.*;


total_loss}��@

error_RXOM?

learning_rate_1��j7�W�	I       6%�	�N�<���A�.*;


total_loss���@

error_Rm$J?

learning_rate_1��j7+�
I       6%�	&��<���A�.*;


total_loss���@

error_R_�K?

learning_rate_1��j7��+I       6%�	���<���A�.*;


total_loss�d�@

error_R!�I?

learning_rate_1��j7�y��I       6%�	�!�<���A�.*;


total_lossZ��@

error_R�]O?

learning_rate_1��j7����I       6%�	Ff�<���A�.*;


total_loss)�@

error_RaX?

learning_rate_1��j7*q�I       6%�	���<���A�.*;


total_loss�E�@

error_R�B?

learning_rate_1��j7���I       6%�	1��<���A�.*;


total_lossTx�@

error_R��;?

learning_rate_1��j7~̙�I       6%�	�/�<���A�.*;


total_loss�͕@

error_R��D?

learning_rate_1��j7�� TI       6%�	~��<���A�.*;


total_loss���@

error_R�*d?

learning_rate_1��j74�QI       6%�	��<���A�.*;


total_lossK2A

error_RaM?

learning_rate_1��j7*G�\I       6%�	:)�<���A�.*;


total_loss ��@

error_R��B?

learning_rate_1��j7��-I       6%�	yn�<���A�.*;


total_lossX��@

error_RO�[?

learning_rate_1��j7�	ƩI       6%�	j��<���A�.*;


total_loss>A

error_R�CN?

learning_rate_1��j7W:I       6%�	���<���A�.*;


total_loss0��@

error_R|:?

learning_rate_1��j7�J�I       6%�	�D�<���A�.*;


total_loss�c�@

error_R�I?

learning_rate_1��j7ʏ�CI       6%�	;��<���A�.*;


total_loss��@

error_RTX?

learning_rate_1��j7E���I       6%�	P��<���A�.*;


total_loss1A�@

error_R�)O?

learning_rate_1��j7t�xI       6%�	l
�<���A�.*;


total_loss�fA

error_R@�]?

learning_rate_1��j7DP�I       6%�	�M�<���A�.*;


total_loss仝@

error_R}m8?

learning_rate_1��j7�Qr�I       6%�	W��<���A�.*;


total_loss(��@

error_R1�>?

learning_rate_1��j7P�gI       6%�	���<���A�.*;


total_loss��@

error_R�J>?

learning_rate_1��j7K���I       6%�	G-�<���A�.*;


total_losso��@

error_R�P?

learning_rate_1��j7la_�I       6%�	X{�<���A�.*;


total_loss80�@

error_R�4Y?

learning_rate_1��j7�%�I       6%�	���<���A�.*;


total_loss�j�@

error_R�zQ?

learning_rate_1��j7�V�kI       6%�	M�<���A�.*;


total_loss�Q�@

error_RJ2S?

learning_rate_1��j7ﴦ	I       6%�	�H�<���A�.*;


total_loss�"�@

error_R l@?

learning_rate_1��j7�v�zI       6%�	��<���A�.*;


total_loss�y�@

error_R�J?

learning_rate_1��j7(!q�I       6%�	���<���A�.*;


total_loss��@

error_R}�Z?

learning_rate_1��j7���8I       6%�	2�<���A�.*;


total_loss���@

error_R�_?

learning_rate_1��j7P���I       6%�	�]�<���A�.*;


total_loss���@

error_R	7?

learning_rate_1��j7�;�I       6%�	���<���A�.*;


total_loss�ӕ@

error_R;}C?

learning_rate_1��j7\�%�I       6%�	���<���A�.*;


total_loss���@

error_R�F?

learning_rate_1��j7�'�fI       6%�	&5�<���A�.*;


total_loss��@

error_R2XQ?

learning_rate_1��j7b��I       6%�	e��<���A�.*;


total_loss.��@

error_R��G?

learning_rate_1��j7l=g�I       6%�	���<���A�.*;


total_loss@��@

error_R1�E?

learning_rate_1��j7L] 4I       6%�	�"�<���A�.*;


total_loss�g�@

error_R��>?

learning_rate_1��j7޲��I       6%�	Ij�<���A�.*;


total_loss'��@

error_R)�T?

learning_rate_1��j7�i�I       6%�	I��<���A�.*;


total_loss<��@

error_R��Q?

learning_rate_1��j7���I       6%�	��<���A�.*;


total_loss���@

error_RdS?

learning_rate_1��j7C��I       6%�	�^�<���A�.*;


total_loss,�z@

error_R	xE?

learning_rate_1��j7�v�I       6%�	!��<���A�.*;


total_loss=FA

error_Rd�Z?

learning_rate_1��j7�2�I       6%�	@��<���A�.*;


total_loss� �@

error_R�\?

learning_rate_1��j7D�AI       6%�	,1�<���A�.*;


total_loss���@

error_RC�R?

learning_rate_1��j76\ �I       6%�	zu�<���A�.*;


total_loss��@

error_R�wL?

learning_rate_1��j7��+I       6%�	���<���A�.*;


total_losseO�@

error_R��X?

learning_rate_1��j7� �#I       6%�	N��<���A�.*;


total_lossD��@

error_R�OQ?

learning_rate_1��j7�&w�I       6%�	C�<���A�.*;


total_lossdh�@

error_R�a=?

learning_rate_1��j7y��@I       6%�	���<���A�.*;


total_loss�I�@

error_RQ�O?

learning_rate_1��j7
�l�I       6%�	���<���A�.*;


total_loss��@

error_R�S?

learning_rate_1��j7*]�:I       6%�	�<���A�.*;


total_loss��@

error_R:sQ?

learning_rate_1��j7����I       6%�	�Z�<���A�.*;


total_loss�%A

error_R;]?

learning_rate_1��j7W�3I       6%�	Z��<���A�.*;


total_lossF��@

error_RdT?

learning_rate_1��j7״��I       6%�	���<���A�.*;


total_loss\�@

error_Rv/U?

learning_rate_1��j7�V��I       6%�	�*�<���A�.*;


total_lossxR�@

error_R�>?

learning_rate_1��j7Q�:gI       6%�	�m�<���A�.*;


total_loss`��@

error_R;T?

learning_rate_1��j7�$[�I       6%�	���<���A�.*;


total_loss��@

error_R�K?

learning_rate_1��j7d��I       6%�	���<���A�.*;


total_loss�4s@

error_R�t8?

learning_rate_1��j7p�?)I       6%�	Q?�<���A�.*;


total_loss���@

error_R/Z?

learning_rate_1��j7h��I       6%�	p��<���A�.*;


total_loss�%�@

error_R��J?

learning_rate_1��j7��F�I       6%�	���<���A�.*;


total_loss��@

error_R��W?

learning_rate_1��j7¡�I       6%�	I�<���A�.*;


total_loss��@

error_R!�Z?

learning_rate_1��j7r�N�I       6%�	7Q�<���A�.*;


total_loss���@

error_R!�M?

learning_rate_1��j7dقI       6%�	L��<���A�.*;


total_lossi�@

error_R�gQ?

learning_rate_1��j7㼟^I       6%�	���<���A�.*;


total_loss��@

error_RxK?

learning_rate_1��j7����I       6%�	)�<���A�.*;


total_loss$�@

error_R_�f?

learning_rate_1��j7�m��I       6%�	��<���A�.*;


total_losso�@

error_R�JZ?

learning_rate_1��j7s]\I       6%�	Q��<���A�.*;


total_loss3p@

error_R1�K?

learning_rate_1��j7O�M�I       6%�	��<���A�.*;


total_loss#j�@

error_R��X?

learning_rate_1��j7.޴I       6%�	)_�<���A�.*;


total_lossn�@

error_R�D?

learning_rate_1��j7��I       6%�	��<���A�.*;


total_lossζ�@

error_R�\Q?

learning_rate_1��j7^�c.I       6%�	B��<���A�.*;


total_loss�5�@

error_R�\?

learning_rate_1��j7%��_I       6%�	4�<���A�.*;


total_loss�Y@

error_R�yQ?

learning_rate_1��j7o���I       6%�	L}�<���A�.*;


total_loss�p�@

error_R�Q?

learning_rate_1��j7A��I       6%�	I��<���A�.*;


total_loss�_�@

error_Rv�F?

learning_rate_1��j7�Y��I       6%�	��<���A�.*;


total_lossaF�@

error_R�eQ?

learning_rate_1?a7 ��I       6%�	�M�<���A�.*;


total_loss��A

error_R	Z?

learning_rate_1?a7��h�I       6%�	9��<���A�.*;


total_loss<�@

error_R��I?

learning_rate_1?a7�%$I       6%�	A��<���A�.*;


total_lossSe@

error_R��7?

learning_rate_1?a7�/�I       6%�	D�<���A�.*;


total_loss�w@

error_R	�L?

learning_rate_1?a7wNP�I       6%�	*b�<���A�.*;


total_loss���@

error_R4�J?

learning_rate_1?a7���I       6%�	ڮ�<���A�.*;


total_loss�N�@

error_RHN?

learning_rate_1?a7H�W>I       6%�	���<���A�.*;


total_lossŸ@

error_R�nM?

learning_rate_1?a7s�UpI       6%�	tC�<���A�.*;


total_loss�@�@

error_R}�5?

learning_rate_1?a7C�XI       6%�	��<���A�.*;


total_loss���@

error_R�B\?

learning_rate_1?a7Z#�I       6%�	���<���A�.*;


total_loss�@A

error_R��T?

learning_rate_1?a7wt��I       6%�	P&�<���A�.*;


total_loss���@

error_RMvV?

learning_rate_1?a7vI       6%�	^h�<���A�.*;


total_loss��@

error_R&K?

learning_rate_1?a7����I       6%�	��<���A�.*;


total_loss���@

error_R��K?

learning_rate_1?a7����I       6%�	���<���A�.*;


total_loss���@

error_R;�R?

learning_rate_1?a7(�I       6%�	%3�<���A�.*;


total_loss�!�@

error_R�Z?

learning_rate_1?a7���I       6%�	�w�<���A�.*;


total_lossTc�@

error_R��B?

learning_rate_1?a7�z�I       6%�	���<���A�.*;


total_loss���@

error_R�K?

learning_rate_1?a7O��I       6%�	���<���A�/*;


total_loss� A

error_RJ-Z?

learning_rate_1?a7��DJI       6%�	�=�<���A�/*;


total_loss�%�@

error_RR�@?

learning_rate_1?a7g-��I       6%�	h��<���A�/*;


total_loss��@

error_R\vH?

learning_rate_1?a7�+��I       6%�	���<���A�/*;


total_loss~�@

error_R�nH?

learning_rate_1?a7j�I}I       6%�	��<���A�/*;


total_loss��A

error_R�G?

learning_rate_1?a7�[��I       6%�	�I�<���A�/*;


total_loss�m�@

error_R��]?

learning_rate_1?a7��]I       6%�	X��<���A�/*;


total_loss��@

error_RV�c?

learning_rate_1?a7	fI       6%�	$��<���A�/*;


total_loss㖚@

error_Rڷ^?

learning_rate_1?a79��|I       6%�	��<���A�/*;


total_lossl�@

error_RJkR?

learning_rate_1?a7�2i�I       6%�	�f�<���A�/*;


total_loss���@

error_R��J?

learning_rate_1?a7��>I       6%�	���<���A�/*;


total_lossԓ@

error_Rr�M?

learning_rate_1?a7	�إI       6%�	U��<���A�/*;


total_lossZ��@

error_Ra�I?

learning_rate_1?a7����I       6%�	I<�<���A�/*;


total_lossxM�@

error_RW+E?

learning_rate_1?a7��p�I       6%�	L�<���A�/*;


total_loss��@

error_R$�D?

learning_rate_1?a7|\�hI       6%�	8��<���A�/*;


total_loss(�@

error_RH4F?

learning_rate_1?a7�pYI       6%�	k�<���A�/*;


total_loss#��@

error_R/�Q?

learning_rate_1?a7&�%WI       6%�	M�<���A�/*;


total_loss�I�@

error_RCE>?

learning_rate_1?a7f�I       6%�	ϔ�<���A�/*;


total_loss��@

error_R6H?

learning_rate_1?a7����I       6%�	���<���A�/*;


total_loss�@

error_R�tN?

learning_rate_1?a7���I       6%�	'%�<���A�/*;


total_loss(��@

error_RC�M?

learning_rate_1?a7���I       6%�	��<���A�/*;


total_lossƄA

error_R/�T?

learning_rate_1?a7)��I       6%�	{��<���A�/*;


total_loss���@

error_R�*I?

learning_rate_1?a7M���I       6%�	 %�<���A�/*;


total_loss�(A

error_R�N?

learning_rate_1?a7��II       6%�	�q�<���A�/*;


total_loss�@

error_R[R?

learning_rate_1?a7h��kI       6%�	;��<���A�/*;


total_loss���@

error_R�K?

learning_rate_1?a7ih��I       6%�	1�<���A�/*;


total_loss���@

error_R��D?

learning_rate_1?a7���I       6%�	V�<���A�/*;


total_lossrͨ@

error_R��V?

learning_rate_1?a7}b��I       6%�	���<���A�/*;


total_loss�4A

error_R�L?

learning_rate_1?a73��I       6%�	1�<���A�/*;


total_loss p�@

error_R�=?

learning_rate_1?a78�I       6%�	WZ�<���A�/*;


total_loss�k�@

error_R��d?

learning_rate_1?a7ԱS�I       6%�	��<���A�/*;


total_loss�
�@

error_Rf�T?

learning_rate_1?a7�\$VI       6%�	#
�<���A�/*;


total_lossW�@

error_R�U?

learning_rate_1?a7d7CLI       6%�	eR�<���A�/*;


total_loss"ς@

error_R��R?

learning_rate_1?a7?i�I       6%�	d��<���A�/*;


total_lossE��@

error_R�P?

learning_rate_1?a7�
�I       6%�	��<���A�/*;


total_losslr�@

error_R�9:?

learning_rate_1?a7F��I       6%�	O�<���A�/*;


total_lossE'�@

error_R��F?

learning_rate_1?a7{��I       6%�	���<���A�/*;


total_loss�7�@

error_R�E?

learning_rate_1?a7��I       6%�	E��<���A�/*;


total_loss�˶@

error_R��W?

learning_rate_1?a7c���I       6%�	�(�<���A�/*;


total_loss!��@

error_RW?

learning_rate_1?a7��I       6%�	;s�<���A�/*;


total_loss_��@

error_R��H?

learning_rate_1?a7��!zI       6%�	��<���A�/*;


total_loss鈵@

error_R�DU?

learning_rate_1?a7hҐnI       6%�	*�<���A�/*;


total_loss�p�@

error_RT�K?

learning_rate_1?a7�~�
I       6%�	5q�<���A�/*;


total_loss8 �@

error_Ra�>?

learning_rate_1?a7G9��I       6%�	1��<���A�/*;


total_lossF��@

error_RSU?

learning_rate_1?a7"tO�I       6%�	L(�<���A�/*;


total_loss`��@

error_R�M?

learning_rate_1?a7x*�I       6%�	�n�<���A�/*;


total_loss�w�@

error_RT�7?

learning_rate_1?a7��'I       6%�	7��<���A�/*;


total_lossӯ�@

error_R��P?

learning_rate_1?a7�Rk�I       6%�	���<���A�/*;


total_lossmj�@

error_RAVH?

learning_rate_1?a7
C �I       6%�	9�<���A�/*;


total_loss�C�@

error_RhH=?

learning_rate_1?a7�g̗I       6%�	C��<���A�/*;


total_loss�[�@

error_R�M?

learning_rate_1?a7P�I       6%�	���<���A�/*;


total_lossnM�@

error_RJ[?

learning_rate_1?a7p�nI       6%�	C
�<���A�/*;


total_loss���@

error_R��H?

learning_rate_1?a7]�I       6%�	O�<���A�/*;


total_loss�_�@

error_R�nU?

learning_rate_1?a7*�I       6%�	���<���A�/*;


total_lossS�@

error_R_49?

learning_rate_1?a7+~��I       6%�	���<���A�/*;


total_losscn�@

error_R�U?

learning_rate_1?a7�JI       6%�	�<���A�/*;


total_lossM�@

error_R��4?

learning_rate_1?a7�� �I       6%�	�h�<���A�/*;


total_loss�{�@

error_R��I?

learning_rate_1?a7����I       6%�	��<���A�/*;


total_loss��@

error_R
~M?

learning_rate_1?a77�dI       6%�	���<���A�/*;


total_loss��A

error_R�7D?

learning_rate_1?a7HJ߱I       6%�	�7�<���A�/*;


total_loss���@

error_R�L?

learning_rate_1?a7"�RXI       6%�	�|�<���A�/*;


total_lossei�@

error_R��Z?

learning_rate_1?a7��I       6%�	���<���A�/*;


total_lossa��@

error_R��S?

learning_rate_1?a7��0�I       6%�	�
�<���A�/*;


total_loss�?�@

error_R��I?

learning_rate_1?a7��
�I       6%�	�S�<���A�/*;


total_loss���@

error_RT�T?

learning_rate_1?a7��ҒI       6%�	���<���A�/*;


total_loss/�@

error_R]a=?

learning_rate_1?a7�I�I       6%�	���<���A�/*;


total_lossM`�@

error_R�6A?

learning_rate_1?a7����I       6%�	�3�<���A�/*;


total_loss$L�@

error_R.�M?

learning_rate_1?a7)II       6%�	�y�<���A�/*;


total_loss9�@

error_R�G?

learning_rate_1?a7��I       6%�	��<���A�/*;


total_loss�	�@

error_R��F?

learning_rate_1?a7q[��I       6%�	��<���A�/*;


total_loss6��@

error_R=�T?

learning_rate_1?a7eE�AI       6%�	�N�<���A�/*;


total_loss��|@

error_R
%L?

learning_rate_1?a7��I       6%�	��<���A�/*;


total_loss�A

error_RR�M?

learning_rate_1?a7�8ȜI       6%�	���<���A�/*;


total_lossH��@

error_R�:J?

learning_rate_1?a7<�
I       6%�	g�<���A�/*;


total_lossW�@

error_R\�K?

learning_rate_1?a7LB�I       6%�	�t�<���A�/*;


total_lossG�@

error_R�LP?

learning_rate_1?a7^��I       6%�	���<���A�/*;


total_loss6L�@

error_RZ�K?

learning_rate_1?a7My�I       6%�	
�<���A�/*;


total_loss�	�@

error_R��i?

learning_rate_1?a7���zI       6%�	LN�<���A�/*;


total_loss	˰@

error_R�I?

learning_rate_1?a7(0�I       6%�		��<���A�/*;


total_loss7$�@

error_RA�]?

learning_rate_1?a7���I       6%�	[��<���A�/*;


total_lossJ��@

error_R��O?

learning_rate_1?a7Z@I       6%�	�<���A�/*;


total_lossIDA

error_R�R?

learning_rate_1?a7���I       6%�	Z�<���A�/*;


total_loss�q�@

error_R4�W?

learning_rate_1?a7�7>�I       6%�	H��<���A�/*;


total_loss�o�@

error_R��F?

learning_rate_1?a7���I       6%�	��<���A�/*;


total_lossb�@

error_R��??

learning_rate_1?a7���I       6%�	4#�<���A�/*;


total_loss���@

error_R�T?

learning_rate_1?a7$��I       6%�	�l�<���A�/*;


total_loss�k�@

error_RxkK?

learning_rate_1?a7�κyI       6%�	9��<���A�/*;


total_loss��p@

error_R�L?

learning_rate_1?a7s?I       6%�	� �<���A�/*;


total_loss<��@

error_R�E?

learning_rate_1?a7��wI       6%�	�D�<���A�/*;


total_loss!~w@

error_R7�D?

learning_rate_1?a7X���I       6%�	3��<���A�/*;


total_loss�Y�@

error_R��]?

learning_rate_1?a7��3�I       6%�	j��<���A�/*;


total_loss�{�@

error_R6rG?

learning_rate_1?a7&�GuI       6%�	y�<���A�/*;


total_lossË�@

error_Rf�D?

learning_rate_1?a7�>I       6%�	JS�<���A�/*;


total_loss��@

error_R$b?

learning_rate_1?a7�a�2I       6%�	ϗ�<���A�/*;


total_lossH>�@

error_R6Y?

learning_rate_1?a7?-��I       6%�	+��<���A�/*;


total_loss�|�@

error_R��F?

learning_rate_1?a7HK�I       6%�	��<���A�/*;


total_lossTA�@

error_RݷD?

learning_rate_1?a7�.��I       6%�	a�<���A�/*;


total_loss��@

error_RȾO?

learning_rate_1?a7�3��I       6%�	m��<���A�/*;


total_loss�*A

error_R:�Z?

learning_rate_1?a7���I       6%�	O��<���A�/*;


total_loss��@

error_RE(X?

learning_rate_1?a7
�I       6%�	�8�<���A�/*;


total_loss�z�@

error_R�[L?

learning_rate_1?a7�XhI       6%�	�}�<���A�/*;


total_lossۼ�@

error_R.�8?

learning_rate_1?a7$���I       6%�	|��<���A�/*;


total_loss�e�@

error_R�O?

learning_rate_1?a7}���I       6%�	?�<���A�/*;


total_loss}@

error_RV]?

learning_rate_1?a7/��nI       6%�	�S�<���A�/*;


total_lossQK�@

error_R�D?

learning_rate_1?a7�-��I       6%�	��<���A�/*;


total_loss�7x@

error_R��A?

learning_rate_1?a7��I       6%�	
��<���A�/*;


total_loss�<�@

error_R�M?

learning_rate_1?a7��\I       6%�	T)�<���A�/*;


total_loss�_�@

error_R�GM?

learning_rate_1?a7��$I       6%�	�p�<���A�/*;


total_loss�4�@

error_R
@N?

learning_rate_1?a7��I       6%�	���<���A�/*;


total_loss�j�@

error_R8�G?

learning_rate_1?a7��P�I       6%�	��<���A�/*;


total_loss���@

error_R�*D?

learning_rate_1?a7�'��I       6%�	Q< =���A�/*;


total_loss ni@

error_RZ�X?

learning_rate_1?a7��sI       6%�	�� =���A�/*;


total_lossIэ@

error_RC#T?

learning_rate_1?a7C�y�I       6%�	V� =���A�/*;


total_loss}��@

error_R �D?

learning_rate_1?a7,p_I       6%�	=���A�/*;


total_loss �@

error_REbP?

learning_rate_1?a7�_:I       6%�	�Y=���A�/*;


total_lossM�<A

error_RVBG?

learning_rate_1?a7�{��I       6%�	Ӟ=���A�/*;


total_loss#��@

error_R��P?

learning_rate_1?a7L�+I       6%�	�=���A�/*;


total_loss]q�@

error_Ra(J?

learning_rate_1?a7��%I       6%�	�'=���A�/*;


total_lossj�@

error_R��\?

learning_rate_1?a7-R�tI       6%�	�h=���A�/*;


total_lossi�@

error_R)e?

learning_rate_1?a7a��}I       6%�	�=���A�/*;


total_lossk�@

error_R�=?

learning_rate_1?a7ç��I       6%�	��=���A�/*;


total_loss�+]@

error_R��F?

learning_rate_1?a7v�I       6%�	4=���A�/*;


total_loss�	�@

error_R�tL?

learning_rate_1?a7���WI       6%�	Wu=���A�/*;


total_loss�?�@

error_R3pI?

learning_rate_1?a7c%�AI       6%�	d�=���A�/*;


total_lossz@

error_R��\?

learning_rate_1?a7I���I       6%�	
=���A�/*;


total_loss=�@

error_R�<?

learning_rate_1?a7S L`I       6%�	�O=���A�/*;


total_loss:� A

error_R��;?

learning_rate_1?a77I       6%�	2�=���A�/*;


total_lossڹ�@

error_RTTF?

learning_rate_1?a7FjݝI       6%�	t�=���A�/*;


total_loss�V�@

error_Ri�E?

learning_rate_1?a7�Y�<I       6%�	q=���A�0*;


total_loss(�@

error_R�Y?

learning_rate_1?a7jG��I       6%�	Ee=���A�0*;


total_lossR9�@

error_R8�L?

learning_rate_1?a7nZ~	I       6%�	ۦ=���A�0*;


total_loss�@
A

error_R�QS?

learning_rate_1?a7Hi��I       6%�	\�=���A�0*;


total_loss��@

error_R Q?

learning_rate_1?a73� �I       6%�	/3=���A�0*;


total_lossoY�@

error_R��5?

learning_rate_1?a7GE�I       6%�	��=���A�0*;


total_loss�^�@

error_R_�_?

learning_rate_1?a7�~7}I       6%�	}�=���A�0*;


total_lossdz�@

error_R�q5?

learning_rate_1?a7jo��I       6%�	�.=���A�0*;


total_loss�D�@

error_RQ�??

learning_rate_1?a7<�DI       6%�	�r=���A�0*;


total_loss�@�@

error_RΓN?

learning_rate_1?a7�mܪI       6%�	1�=���A�0*;


total_loss� �@

error_R��P?

learning_rate_1?a7�x��I       6%�	I�=���A�0*;


total_loss:��@

error_RC�N?

learning_rate_1?a7c)t�I       6%�	=>=���A�0*;


total_loss���@

error_R�?V?

learning_rate_1?a7}O��I       6%�	[�=���A�0*;


total_lossr�@

error_R��N?

learning_rate_1?a7"��I       6%�	w�=���A�0*;


total_loss:��@

error_R��T?

learning_rate_1?a7)�I       6%�	 	=���A�0*;


total_lossHn�@

error_R�lJ?

learning_rate_1?a7��Z@I       6%�	/K	=���A�0*;


total_lossӢ�@

error_RsV?

learning_rate_1?a7��=�I       6%�	l�	=���A�0*;


total_loss`g�@

error_R��O?

learning_rate_1?a7m&I       6%�	�	=���A�0*;


total_loss��@

error_RφV?

learning_rate_1?a7 @�(I       6%�	�
=���A�0*;


total_loss�	�@

error_Rn]?

learning_rate_1?a7�)I       6%�	�b
=���A�0*;


total_loss��@

error_R\�A?

learning_rate_1?a7r��aI       6%�	N�
=���A�0*;


total_loss�7�@

error_R}4P?

learning_rate_1?a7�K��I       6%�	��
=���A�0*;


total_loss��@

error_R��D?

learning_rate_1?a71�G]I       6%�	�*=���A�0*;


total_lossMz�@

error_R�tM?

learning_rate_1?a7;~�GI       6%�	+o=���A�0*;


total_loss��@

error_R:�W?

learning_rate_1?a7	K?I       6%�	7�=���A�0*;


total_loss�"�@

error_R(]W?

learning_rate_1?a7u��I       6%�	h�=���A�0*;


total_loss��@

error_RI?

learning_rate_1?a7Q��[I       6%�	9=���A�0*;


total_loss��@

error_R#�L?

learning_rate_1?a7�Q`�I       6%�	9�=���A�0*;


total_loss$ �@

error_RnY?

learning_rate_1?a7i��I       6%�	j�=���A�0*;


total_loss�`@

error_R��7?

learning_rate_1?a7�}�YI       6%�	�=���A�0*;


total_lossr��@

error_RԼJ?

learning_rate_1?a7D���I       6%�	Fd=���A�0*;


total_lossȹ�@

error_R�e=?

learning_rate_1?a7�m�I       6%�	��=���A�0*;


total_loss4��@

error_Rd]U?

learning_rate_1?a7��P�I       6%�	D=���A�0*;


total_loss3��@

error_RJ�K?

learning_rate_1?a7�,0�I       6%�	V=���A�0*;


total_loss�mA

error_RN8V?

learning_rate_1?a7kRאI       6%�	�=���A�0*;


total_loss�a�@

error_RR�K?

learning_rate_1?a7�HI�I       6%�	�=���A�0*;


total_loss�*�@

error_R<KK?

learning_rate_1?a7���I       6%�	P=���A�0*;


total_loss���@

error_R�_S?

learning_rate_1?a7Kt"�I       6%�	�=���A�0*;


total_loss7�@

error_R�T?

learning_rate_1?a7J�y1I       6%�	��=���A�0*;


total_loss��@

error_RZ�@?

learning_rate_1?a7�b�rI       6%�	"=���A�0*;


total_loss	�@

error_R��G?

learning_rate_1?a7f+kI       6%�	�i=���A�0*;


total_lossq��@

error_R�pV?

learning_rate_1?a7�+��I       6%�	Ǯ=���A�0*;


total_loss���@

error_RAnW?

learning_rate_1?a7���I       6%�	��=���A�0*;


total_loss��	A

error_R�TQ?

learning_rate_1?a7����I       6%�	O5=���A�0*;


total_loss��@

error_R�H?

learning_rate_1?a7P�+I       6%�	Ly=���A�0*;


total_loss��@

error_R�lU?

learning_rate_1?a7x��I       6%�	��=���A�0*;


total_loss��@

error_R�^?

learning_rate_1?a7{�\I       6%�	|=���A�0*;


total_lossl��@

error_R�(D?

learning_rate_1?a78�9I       6%�	�F=���A�0*;


total_loss�i�@

error_R��G?

learning_rate_1?a7e!I       6%�	z�=���A�0*;


total_loss�i�@

error_RxdK?

learning_rate_1?a7�܈�I       6%�	n�=���A�0*;


total_loss���@

error_R��C?

learning_rate_1?a7р*I       6%�	�=���A�0*;


total_lossf�@

error_RO]?

learning_rate_1?a7�X2I       6%�	�Y=���A�0*;


total_loss
��@

error_R�V?

learning_rate_1?a7����I       6%�	��=���A�0*;


total_loss���@

error_R=hC?

learning_rate_1?a7W�UI       6%�	T�=���A�0*;


total_loss�Ј@

error_R=�_?

learning_rate_1?a77�W�I       6%�	#=���A�0*;


total_loss�%f@

error_Rz�@?

learning_rate_1?a7е�eI       6%�	�f=���A�0*;


total_loss���@

error_R�B9?

learning_rate_1?a7��%I       6%�	۩=���A�0*;


total_loss�Đ@

error_R.�L?

learning_rate_1?a7���I       6%�	�=���A�0*;


total_loss�_ A

error_R�[F?

learning_rate_1?a7o��I       6%�	�.=���A�0*;


total_loss�}@

error_RBP?

learning_rate_1?a7mnϚI       6%�	,p=���A�0*;


total_loss�+�@

error_R��M?

learning_rate_1?a7���I       6%�	;�=���A�0*;


total_lossl�@

error_R}T\?

learning_rate_1?a7��\I       6%�	��=���A�0*;


total_loss#Z�@

error_R��>?

learning_rate_1?a7ͻAI       6%�	48=���A�0*;


total_lossȏ�@

error_R�zI?

learning_rate_1?a7PSI       6%�	1�=���A�0*;


total_lossE�@

error_RcK?

learning_rate_1?a7�GM�I       6%�	��=���A�0*;


total_loss���@

error_R�SO?

learning_rate_1?a7lA�I       6%�	#=���A�0*;


total_loss.P�@

error_R�`Z?

learning_rate_1?a7pI       6%�	�l=���A�0*;


total_loss���@

error_R��Q?

learning_rate_1?a7:]�MI       6%�	��=���A�0*;


total_loss�-�@

error_R� J?

learning_rate_1?a7�/��I       6%�	I�=���A�0*;


total_loss(?|@

error_R��H?

learning_rate_1?a7U�I�I       6%�	`B=���A�0*;


total_loss�)�@

error_RneO?

learning_rate_1?a7a`��I       6%�	N�=���A�0*;


total_loss�A�@

error_R�?5?

learning_rate_1?a7���I       6%�	 �=���A�0*;


total_loss�I�@

error_RqP?

learning_rate_1?a7� a�I       6%�	=���A�0*;


total_lossO��@

error_R��S?

learning_rate_1?a7즆TI       6%�	�M=���A�0*;


total_losssb�@

error_R�<:?

learning_rate_1?a7�z��I       6%�	Γ=���A�0*;


total_losst�@

error_RJ1I?

learning_rate_1?a7����I       6%�	s�=���A�0*;


total_lossLMg@

error_R%bZ?

learning_rate_1?a7lգ�I       6%�	-=���A�0*;


total_loss���@

error_R/_?

learning_rate_1?a72��I       6%�	_=���A�0*;


total_loss�m@

error_R�WV?

learning_rate_1?a7	���I       6%�	�=���A�0*;


total_loss���@

error_R{
H?

learning_rate_1?a7��dI       6%�	Y�=���A�0*;


total_lossm �@

error_R��G?

learning_rate_1?a7nc��I       6%�	#,=���A�0*;


total_loss��t@

error_R�;I?

learning_rate_1?a7˸p�I       6%�	�p=���A�0*;


total_loss�ڸ@

error_R��N?

learning_rate_1?a7����I       6%�	j�=���A�0*;


total_loss_��@

error_RR�[?

learning_rate_1?a7}�C	I       6%�	��=���A�0*;


total_loss���@

error_R�/P?

learning_rate_1?a7 	�BI       6%�	�C=���A�0*;


total_lossN$�@

error_R��J?

learning_rate_1?a7�Ϊ�I       6%�	�=���A�0*;


total_lossv[@

error_R�7Q?

learning_rate_1?a7$�*I       6%�	(�=���A�0*;


total_loss�ۦ@

error_RJC?

learning_rate_1?a7-���I       6%�	#=���A�0*;


total_lossZ+�@

error_RMyY?

learning_rate_1?a7��	-I       6%�	Y=���A�0*;


total_lossO.�@

error_RϽ^?

learning_rate_1?a7��I       6%�	0�=���A�0*;


total_loss�@

error_R,�K?

learning_rate_1?a7�V�I       6%�	n�=���A�0*;


total_lossw�@

error_R�@?

learning_rate_1?a7�I       6%�	�%=���A�0*;


total_loss]��@

error_R$�T?

learning_rate_1?a7"�I       6%�	�l=���A�0*;


total_lossg��@

error_R��M?

learning_rate_1?a7�GN�I       6%�	��=���A�0*;


total_lossFf�@

error_R�S?

learning_rate_1?a7B�[I       6%�	+�=���A�0*;


total_loss�_�@

error_R&�L?

learning_rate_1?a7��܁I       6%�	m>=���A�0*;


total_loss���@

error_RQ�Y?

learning_rate_1?a7jo�I       6%�	��=���A�0*;


total_loss�@�@

error_R�WT?

learning_rate_1?a7��I       6%�	��=���A�0*;


total_loss�Z@

error_Rl�P?

learning_rate_1?a7��jI       6%�	a =���A�0*;


total_lossZ�@

error_R�XZ?

learning_rate_1?a7Y�I�I       6%�	S =���A�0*;


total_loss7f�@

error_RsB?

learning_rate_1?a7e��I       6%�	X� =���A�0*;


total_lossa'�@

error_RׁQ?

learning_rate_1?a7���RI       6%�	�� =���A�0*;


total_loss�&�@

error_R��S?

learning_rate_1?a7���I       6%�	s!=���A�0*;


total_lossSo�@

error_R��N?

learning_rate_1?a7/*C�I       6%�	�h!=���A�0*;


total_lossZ�@

error_R�G?

learning_rate_1?a7��I       6%�	��!=���A�0*;


total_loss��@

error_R�[?

learning_rate_1?a7C��=I       6%�	��!=���A�0*;


total_loss�I|@

error_R��\?

learning_rate_1?a7L=�I       6%�	D"=���A�0*;


total_loss&S�@

error_R��7?

learning_rate_1?a7Q=�I       6%�	�"=���A�0*;


total_lossE��@

error_Rr�Y?

learning_rate_1?a7��+$I       6%�	��"=���A�0*;


total_loss��@

error_R O^?

learning_rate_1?a7��I       6%�	+#=���A�0*;


total_loss���@

error_R�rS?

learning_rate_1?a7# ��I       6%�	!b#=���A�0*;


total_loss�6�@

error_R�;[?

learning_rate_1?a7����I       6%�	��#=���A�0*;


total_loss��@

error_R؄Q?

learning_rate_1?a7=�I       6%�	��#=���A�0*;


total_loss"�@

error_Ri�K?

learning_rate_1?a7�H�I       6%�	�8$=���A�0*;


total_loss��@

error_R�V?

learning_rate_1?a7�Kz�I       6%�	�~$=���A�0*;


total_loss*��@

error_R.�>?

learning_rate_1?a7\�'I       6%�	�$=���A�0*;


total_loss=.�@

error_R��G?

learning_rate_1?a7K��BI       6%�	�%=���A�0*;


total_loss��@

error_R;2@?

learning_rate_1?a7�g��I       6%�	 I%=���A�0*;


total_loss-]�@

error_R܉I?

learning_rate_1?a7�@�I       6%�	�%=���A�0*;


total_loss�I�@

error_R�c??

learning_rate_1?a7����I       6%�	W�%=���A�0*;


total_loss͟�@

error_R�a?

learning_rate_1?a7�偫I       6%�	5&=���A�0*;


total_lossRbA

error_R�Q?

learning_rate_1?a7>��I       6%�	�`&=���A�0*;


total_loss�P�@

error_R%6?

learning_rate_1?a7� RI       6%�	�&=���A�0*;


total_loss��@

error_R��I?

learning_rate_1?a7��5@I       6%�	I'=���A�0*;


total_lossN��@

error_R�7B?

learning_rate_1?a7k�W#I       6%�	EQ'=���A�0*;


total_loss�`�@

error_R��N?

learning_rate_1?a7�1I       6%�	y�'=���A�0*;


total_loss�A

error_R�CS?

learning_rate_1?a7״��I       6%�	l�'=���A�0*;


total_loss3��@

error_R��S?

learning_rate_1?a7>�I       6%�	�,(=���A�0*;


total_loss9�@

error_R{K[?

learning_rate_1?a7�l��I       6%�	�n(=���A�1*;


total_loss�@

error_RO�;?

learning_rate_1?a7�_IeI       6%�	s�(=���A�1*;


total_loss伇@

error_R�R?

learning_rate_1?a7���rI       6%�	.)=���A�1*;


total_loss�6�@

error_RLwR?

learning_rate_1?a7��I       6%�	,F)=���A�1*;


total_loss��@

error_R�|O?

learning_rate_1?a7�A59I       6%�	}�)=���A�1*;


total_loss��@

error_R�Vb?

learning_rate_1?a7#�cI       6%�	��)=���A�1*;


total_loss:��@

error_R_J?

learning_rate_1?a7�MbI       6%�	{*=���A�1*;


total_loss��@

error_R�,V?

learning_rate_1?a7�YII       6%�	�V*=���A�1*;


total_lossئ�@

error_Rt"=?

learning_rate_1?a7�D��I       6%�	ԗ*=���A�1*;


total_loss���@

error_R��E?

learning_rate_1?a7]�	I       6%�	2�*=���A�1*;


total_loss<w�@

error_R�	T?

learning_rate_1?a7�v�I       6%�	
+=���A�1*;


total_lossJ��@

error_RE�P?

learning_rate_1?a7�2PI       6%�	�_+=���A�1*;


total_loss�k�@

error_R�L?

learning_rate_1?a7�UI       6%�	`�+=���A�1*;


total_loss�+�@

error_R�fd?

learning_rate_1?a7�6��I       6%�	C�+=���A�1*;


total_loss7��@

error_R8�H?

learning_rate_1?a7���I       6%�	/*,=���A�1*;


total_lossَ@

error_RC0Q?

learning_rate_1?a7J�^GI       6%�	q,=���A�1*;


total_loss�,�@

error_R@}n?

learning_rate_1?a7�b��I       6%�	0�,=���A�1*;


total_loss���@

error_R6�`?

learning_rate_1?a7�
�YI       6%�	��,=���A�1*;


total_loss��@

error_R�0J?

learning_rate_1?a7-�E�I       6%�	I:-=���A�1*;


total_loss�_�@

error_R�A?

learning_rate_1?a7���I       6%�	��-=���A�1*;


total_loss�Tl@

error_R�]I?

learning_rate_1?a76-�I       6%�	��-=���A�1*;


total_loss�
�@

error_R�?5?

learning_rate_1?a7�L�I       6%�	�.=���A�1*;


total_lossX%�@

error_R�wK?

learning_rate_1?a7P:13I       6%�	^.=���A�1*;


total_loss-��@

error_R��M?

learning_rate_1?a7�_��I       6%�	�.=���A�1*;


total_loss�9�@

error_R�\M?

learning_rate_1?a7�v�I       6%�	/=���A�1*;


total_lossX��@

error_RX_?

learning_rate_1?a7n,V�I       6%�	�V/=���A�1*;


total_lossv7u@

error_RWo@?

learning_rate_1?a7��=�I       6%�	Ɲ/=���A�1*;


total_loss�߫@

error_R�M?

learning_rate_1?a7���I       6%�	��/=���A�1*;


total_loss��A

error_RZ�D?

learning_rate_1?a7����I       6%�	�10=���A�1*;


total_loss���@

error_R�K?

learning_rate_1?a7�v��I       6%�	z{0=���A�1*;


total_loss��@

error_R�T?

learning_rate_1?a7U��I       6%�	
�0=���A�1*;


total_loss��@

error_RvU?

learning_rate_1?a7��7I       6%�	�1=���A�1*;


total_loss�k�@

error_R3�B?

learning_rate_1?a7 r^�I       6%�	>X1=���A�1*;


total_loss���@

error_R��T?

learning_rate_1?a7��nI       6%�	՜1=���A�1*;


total_lossh/�@

error_R[7V?

learning_rate_1?a7���5I       6%�	��1=���A�1*;


total_lossi�@

error_R�Y?

learning_rate_1?a7�?��I       6%�	"$2=���A�1*;


total_loss3 �@

error_R�L?

learning_rate_1?a7��?I       6%�	�h2=���A�1*;


total_loss���@

error_R6�O?

learning_rate_1?a7Ҝ�kI       6%�	�2=���A�1*;


total_loss�>A

error_R6�I?

learning_rate_1?a7� I       6%�	��2=���A�1*;


total_loss�Ͼ@

error_R=�B?

learning_rate_1?a7jﴗI       6%�	�B3=���A�1*;


total_loss=��@

error_R,W?

learning_rate_1?a7�&aI       6%�	�3=���A�1*;


total_lossQO�@

error_R�3??

learning_rate_1?a7%^�I       6%�	>�3=���A�1*;


total_lossj �@

error_R)I?

learning_rate_1?a7V'�I       6%�	�4=���A�1*;


total_loss�xA

error_Rl�K?

learning_rate_1?a7$귆I       6%�	q^4=���A�1*;


total_loss���@

error_RvU?

learning_rate_1?a7�V~vI       6%�	Ŧ4=���A�1*;


total_lossp6�@

error_RJCE?

learning_rate_1?a7J���I       6%�	��4=���A�1*;


total_loss��@

error_RE�N?

learning_rate_1?a7� P�I       6%�	x*5=���A�1*;


total_loss���@

error_R��S?

learning_rate_1?a7�.I       6%�	sl5=���A�1*;


total_loss�B�@

error_R��B?

learning_rate_1?a7Ȱ`NI       6%�	��5=���A�1*;


total_lossxA�@

error_RO>?

learning_rate_1?a7�6�ZI       6%�	��5=���A�1*;


total_loss��@

error_RL?

learning_rate_1?a7�'<�I       6%�	g46=���A�1*;


total_lossfj�@

error_Ri�C?

learning_rate_1?a7�iuI       6%�	`�6=���A�1*;


total_lossV��@

error_R O?

learning_rate_1?a7̤�sI       6%�	��6=���A�1*;


total_loss�)A

error_R;R?

learning_rate_1?a7�A�I       6%�	�!7=���A�1*;


total_loss��A

error_RC�g?

learning_rate_1?a7���I       6%�	rm7=���A�1*;


total_loss	�@

error_R�#L?

learning_rate_1?a7h�hI       6%�	q�7=���A�1*;


total_loss�d~@

error_R�"X?

learning_rate_1?a7Cɼ�I       6%�	��7=���A�1*;


total_loss7w�@

error_R�YF?

learning_rate_1?a7A��I       6%�	;78=���A�1*;


total_lossE9�@

error_R#oV?

learning_rate_1?a7o'�NI       6%�	Ɂ8=���A�1*;


total_loss��@

error_RL�A?

learning_rate_1?a7ѻDTI       6%�	��8=���A�1*;


total_loss�E�@

error_R��B?

learning_rate_1?a7$~��I       6%�	
9=���A�1*;


total_loss�[z@

error_R��N?

learning_rate_1?a7�Z<II       6%�	�J9=���A�1*;


total_loss�E�@

error_R�bB?

learning_rate_1?a7-��I       6%�	�9=���A�1*;


total_loss��@

error_R�P?

learning_rate_1?a7���'I       6%�	;�9=���A�1*;


total_loss���@

error_R[?

learning_rate_1?a7���MI       6%�	 :=���A�1*;


total_lossZ¯@

error_Ra�L?

learning_rate_1?a7��EI       6%�	�]:=���A�1*;


total_loss���@

error_R�,9?

learning_rate_1?a7{��I       6%�	�:=���A�1*;


total_loss�*�@

error_Ra�<?

learning_rate_1?a7���GI       6%�	��:=���A�1*;


total_loss��p@

error_R{�O?

learning_rate_1?a7jv�1I       6%�	-8;=���A�1*;


total_lossoߠ@

error_R�bZ?

learning_rate_1?a7$]|I       6%�	�};=���A�1*;


total_loss���@

error_R�	5?

learning_rate_1?a7��^�I       6%�	��;=���A�1*;


total_loss��@

error_R;�J?

learning_rate_1?a7V`SEI       6%�	<=���A�1*;


total_loss��	A

error_R3V?

learning_rate_1?a7�X�JI       6%�	CS<=���A�1*;


total_loss�m�@

error_R�M?

learning_rate_1?a7ezh�I       6%�	W�<=���A�1*;


total_loss��@

error_R�	Q?

learning_rate_1?a7�ǵ�I       6%�	8�<=���A�1*;


total_loss��@

error_Rl�J?

learning_rate_1?a7�	pI       6%�	�&==���A�1*;


total_loss�Ơ@

error_R�@?

learning_rate_1?a7�iHI       6%�	l==���A�1*;


total_loss�<�@

error_R��`?

learning_rate_1?a7,�.�I       6%�	��==���A�1*;


total_loss���@

error_Rv�D?

learning_rate_1?a7>N~I       6%�	��==���A�1*;


total_loss���@

error_R��K?

learning_rate_1?a7�I       6%�	�7>=���A�1*;


total_loss�3�@

error_R�^?

learning_rate_1?a7)�Q,I       6%�	P{>=���A�1*;


total_loss�.�@

error_R��L?

learning_rate_1?a7J�0KI       6%�	t�>=���A�1*;


total_loss�%�@

error_RؾI?

learning_rate_1?a7,Qz�I       6%�	%?=���A�1*;


total_losss0�@

error_R�;T?

learning_rate_1?a7�I       6%�	<O?=���A�1*;


total_loss��@

error_R��X?

learning_rate_1?a7��I       6%�	��?=���A�1*;


total_lossi˦@

error_R�aT?

learning_rate_1?a7�n
I       6%�	�?=���A�1*;


total_loss.��@

error_R��H?

learning_rate_1?a7�2(I       6%�	_@=���A�1*;


total_lossd]A

error_R��M?

learning_rate_1?a7N}y�I       6%�	,f@=���A�1*;


total_loss�}�@

error_R�G?

learning_rate_1?a7���I       6%�	ק@=���A�1*;


total_loss�j�@

error_R�H?

learning_rate_1?a7�dHCI       6%�	��@=���A�1*;


total_loss?/�@

error_R�O?

learning_rate_1?a7:��I       6%�	�(A=���A�1*;


total_lossؾ A

error_R�4R?

learning_rate_1?a7{�QI       6%�	lA=���A�1*;


total_lossD!�@

error_Rh;Q?

learning_rate_1?a7���I       6%�	e�A=���A�1*;


total_loss(��@

error_R�0F?

learning_rate_1?a7�Wu:I       6%�	�A=���A�1*;


total_loss���@

error_R�I?

learning_rate_1?a7�9��I       6%�	T?B=���A�1*;


total_loss��@

error_Rf<?

learning_rate_1?a7�'I       6%�	�B=���A�1*;


total_loss��@

error_R��N?

learning_rate_1?a7��I       6%�	��B=���A�1*;


total_lossq��@

error_R�J?

learning_rate_1?a7�2n$I       6%�	�C=���A�1*;


total_loss���@

error_R��C?

learning_rate_1?a7��RI       6%�	�UC=���A�1*;


total_loss���@

error_R�eM?

learning_rate_1?a7x�PI       6%�	�C=���A�1*;


total_loss�A

error_R��:?

learning_rate_1?a7^�I       6%�	��C=���A�1*;


total_loss#9�@

error_R��U?

learning_rate_1?a7�S)hI       6%�	�D=���A�1*;


total_loss��@

error_Ri�d?

learning_rate_1?a7��6KI       6%�	�ZD=���A�1*;


total_loss�6�@

error_R�>T?

learning_rate_1?a7�9I       6%�	�D=���A�1*;


total_loss���@

error_R�p@?

learning_rate_1?a7��yI       6%�	
�D=���A�1*;


total_loss�T�@

error_R�uV?

learning_rate_1?a7��0TI       6%�	�.E=���A�1*;


total_loss1	�@

error_RR�K?

learning_rate_1?a7	��I       6%�	.zE=���A�1*;


total_loss�:�@

error_R�ZU?

learning_rate_1?a7�L�I       6%�	E�E=���A�1*;


total_loss`�@

error_RW�V?

learning_rate_1?a7h6�I       6%�	� F=���A�1*;


total_loss�ż@

error_R:N?

learning_rate_1?a7�;�"I       6%�	gMF=���A�1*;


total_loss*��@

error_R�pM?

learning_rate_1?a73�ԜI       6%�	8�F=���A�1*;


total_loss���@

error_R��I?

learning_rate_1?a7/X�jI       6%�	)�F=���A�1*;


total_losseն@

error_R!T?

learning_rate_1?a7`,"yI       6%�	�EG=���A�1*;


total_loss-�@

error_R��U?

learning_rate_1?a7
�T9I       6%�	��G=���A�1*;


total_loss�@

error_R�:?

learning_rate_1?a7ߜ̝I       6%�	a�G=���A�1*;


total_lossX6�@

error_R�^?

learning_rate_1?a7��kI       6%�	zH=���A�1*;


total_loss>�@

error_RnhO?

learning_rate_1?a7����I       6%�	xUH=���A�1*;


total_loss!`�@

error_R �R?

learning_rate_1?a7�G��I       6%�	~�H=���A�1*;


total_loss�0�@

error_R��I?

learning_rate_1?a7��۔I       6%�	��H=���A�1*;


total_loss/y�@

error_RW6G?

learning_rate_1?a7l[�9I       6%�	�#I=���A�1*;


total_loss���@

error_RT�T?

learning_rate_1?a78?�7I       6%�	�fI=���A�1*;


total_loss���@

error_R�W[?

learning_rate_1?a7�˶�I       6%�	Q�I=���A�1*;


total_loss�r�@

error_Rzr=?

learning_rate_1?a7�RI       6%�	��I=���A�1*;


total_lossZ�o@

error_R3�K?

learning_rate_1?a7&R5\I       6%�	80J=���A�1*;


total_loss{�@

error_R�D?

learning_rate_1?a7f+@]I       6%�	�qJ=���A�1*;


total_losss�@

error_R��P?

learning_rate_1?a7�&�7I       6%�	x�J=���A�1*;


total_lossl�@

error_R8@Y?

learning_rate_1?a7Y�SFI       6%�	~�J=���A�1*;


total_lossXM�@

error_R��H?

learning_rate_1?a7Oр�I       6%�	�9K=���A�1*;


total_lossJ��@

error_Rz�J?

learning_rate_1?a7���I       6%�	�}K=���A�2*;


total_loss�2�@

error_R��Z?

learning_rate_1?a7x��jI       6%�	�K=���A�2*;


total_loss ��@

error_R�N?

learning_rate_1?a7��I       6%�	PL=���A�2*;


total_loss�@

error_R�@W?

learning_rate_1?a7�m�HI       6%�	-DL=���A�2*;


total_lossq�@

error_R6�R?

learning_rate_1?a7 �3�I       6%�	_�L=���A�2*;


total_loss��@

error_RnU?

learning_rate_1?a7�nTI       6%�	��L=���A�2*;


total_loss̰�@

error_R�C?

learning_rate_1?a7�6*>I       6%�	RM=���A�2*;


total_loss$��@

error_RiT?

learning_rate_1?a7�ó�I       6%�	S`M=���A�2*;


total_loss8�@

error_R�O?

learning_rate_1?a7�(�I       6%�	��M=���A�2*;


total_lossdn�@

error_R�U?

learning_rate_1?a7 ��I       6%�	��M=���A�2*;


total_loss�L|@

error_R��L?

learning_rate_1?a7#<�SI       6%�	 CN=���A�2*;


total_loss�A�@

error_R�CM?

learning_rate_1?a7x.�;I       6%�	�N=���A�2*;


total_loss|��@

error_RֵS?

learning_rate_1?a7#�	:I       6%�	j�N=���A�2*;


total_loss���@

error_Rj�E?

learning_rate_1?a7ܬɎI       6%�	JO=���A�2*;


total_loss��@

error_R�^V?

learning_rate_1?a7�`I       6%�	iO=���A�2*;


total_lossܓ@

error_RAQ?

learning_rate_1?a7��I       6%�	q�O=���A�2*;


total_loss8�@

error_R: ??

learning_rate_1?a7�{"4I       6%�	.�O=���A�2*;


total_loss�ɬ@

error_R�EG?

learning_rate_1?a7�c�I       6%�	�4P=���A�2*;


total_lossR��@

error_RQZ?

learning_rate_1?a7���I       6%�	�xP=���A�2*;


total_loss{��@

error_R�XD?

learning_rate_1?a7���I       6%�	��P=���A�2*;


total_lossh�@

error_R�xE?

learning_rate_1?a7�`�I       6%�	��P=���A�2*;


total_loss��@

error_Ri�I?

learning_rate_1?a7T%}I       6%�	+BQ=���A�2*;


total_loss!��@

error_R�G?

learning_rate_1?a7�$-zI       6%�	o�Q=���A�2*;


total_loss��@

error_R��G?

learning_rate_1?a7?-��I       6%�	��Q=���A�2*;


total_loss���@

error_R@4U?

learning_rate_1?a7BS�+I       6%�	�R=���A�2*;


total_loss�y�@

error_R�>?

learning_rate_1?a7$�x�I       6%�	�ZR=���A�2*;


total_loss/�A

error_R}�L?

learning_rate_1?a7z��I       6%�	s�R=���A�2*;


total_loss���@

error_R�N?

learning_rate_1?a7j���I       6%�	��R=���A�2*;


total_loss� A

error_RQ�I?

learning_rate_1?a76��I       6%�	�.S=���A�2*;


total_loss��@

error_R��N?

learning_rate_1?a7}�<I       6%�	�uS=���A�2*;


total_loss璏@

error_RJ�Z?

learning_rate_1?a7�<�I       6%�	�S=���A�2*;


total_lossm޿@

error_RvZ?

learning_rate_1?a7+M��I       6%�	�T=���A�2*;


total_loss�T�@

error_R��Z?

learning_rate_1?a7�PʬI       6%�	�HT=���A�2*;


total_loss���@

error_R7TR?

learning_rate_1?a7�Yr~I       6%�	ϑT=���A�2*;


total_lossJ@�@

error_R�aB?

learning_rate_1?a7��v�I       6%�	��T=���A�2*;


total_lossW��@

error_Re�R?

learning_rate_1?a7��qwI       6%�	0U=���A�2*;


total_loss��@

error_RwR?

learning_rate_1?a7�	�I       6%�	�ZU=���A�2*;


total_loss�}�@

error_RO?

learning_rate_1?a7��XfI       6%�	��U=���A�2*;


total_lossf��@

error_R)e?

learning_rate_1?a728�I       6%�	_�U=���A�2*;


total_lossT��@

error_R�OV?

learning_rate_1?a7�,I       6%�	s!V=���A�2*;


total_loss�$�@

error_R�U?

learning_rate_1?a7||�cI       6%�	�sV=���A�2*;


total_lossFe�@

error_Ra>N?

learning_rate_1?a7�~�I       6%�	�V=���A�2*;


total_lossdݽ@

error_R;�S?

learning_rate_1?a7%*�I       6%�	�W=���A�2*;


total_loss�i�@

error_R�7>?

learning_rate_1?a7n�a�I       6%�	4QW=���A�2*;


total_lossA

error_R�Z?

learning_rate_1?a74+��I       6%�	l�W=���A�2*;


total_loss��{@

error_R�S?

learning_rate_1?a7 ]�I       6%�	�W=���A�2*;


total_loss;��@

error_Ri=Z?

learning_rate_1?a7y�7?I       6%�	K"X=���A�2*;


total_lossox�@

error_Rn�N?

learning_rate_1?a7���-I       6%�	�gX=���A�2*;


total_loss���@

error_R�N?

learning_rate_1?a7&m�I       6%�	ɫX=���A�2*;


total_loss���@

error_RÎI?

learning_rate_1?a7��I       6%�	/�X=���A�2*;


total_lossC�@

error_R{.A?

learning_rate_1?a7��e�I       6%�	G2Y=���A�2*;


total_loss�@

error_R7J?

learning_rate_1?a7�xKI       6%�	�wY=���A�2*;


total_loss<G�@

error_RE6C?

learning_rate_1?a7��)I       6%�	��Y=���A�2*;


total_lossi� A

error_R�{O?

learning_rate_1?a7���I       6%�	��Y=���A�2*;


total_loss��@

error_Rc�;?

learning_rate_1?a7��I       6%�	BAZ=���A�2*;


total_loss-�@

error_R#/M?

learning_rate_1?a7���I       6%�	4�Z=���A�2*;


total_loss1D�@

error_R�m_?

learning_rate_1?a7�0�I       6%�	�Z=���A�2*;


total_lossI;�@

error_Ra�T?

learning_rate_1?a7�`��I       6%�	�[=���A�2*;


total_loss���@

error_R��Q?

learning_rate_1?a7��uI       6%�	L[[=���A�2*;


total_loss�o�@

error_R�]?

learning_rate_1?a74cڣI       6%�	��[=���A�2*;


total_loss���@

error_RTJb?

learning_rate_1?a7��I       6%�	�[=���A�2*;


total_lossT��@

error_R�KU?

learning_rate_1?a7��H6I       6%�	�&\=���A�2*;


total_lossN��@

error_R�P?

learning_rate_1?a7�eێI       6%�	�l\=���A�2*;


total_loss��@

error_R��Q?

learning_rate_1?a7�ʓ�I       6%�	��\=���A�2*;


total_loss�e�@

error_RZ-A?

learning_rate_1?a7\+I       6%�	H]=���A�2*;


total_loss&4�@

error_R{�R?

learning_rate_1?a7���;I       6%�	2S]=���A�2*;


total_lossv[�@

error_Ra�]?

learning_rate_1?a7�m(?I       6%�	��]=���A�2*;


total_loss;�@

error_R��E?

learning_rate_1?a7�q,�I       6%�	��]=���A�2*;


total_losse �@

error_R3�V?

learning_rate_1?a7q�cI       6%�	�'^=���A�2*;


total_loss�ϴ@

error_R��[?

learning_rate_1?a7��I       6%�	�p^=���A�2*;


total_loss��@

error_R��W?

learning_rate_1?a7�yI       6%�	�^=���A�2*;


total_loss/�i@

error_Ri�7?

learning_rate_1?a7�"�I       6%�	f�^=���A�2*;


total_loss)��@

error_R��I?

learning_rate_1?a7\z��I       6%�	�=_=���A�2*;


total_loss��y@

error_RzG?

learning_rate_1?a7����I       6%�	��_=���A�2*;


total_lossꝠ@

error_RM�V?

learning_rate_1?a7�ТI       6%�	�_=���A�2*;


total_loss�͉@

error_RG?

learning_rate_1?a7mn<3I       6%�	8	`=���A�2*;


total_loss;(�@

error_R_FH?

learning_rate_1?a7Z�I       6%�	FN`=���A�2*;


total_loss��A

error_R�[?

learning_rate_1?a7��W�I       6%�	��`=���A�2*;


total_loss�@

error_R4`V?

learning_rate_1?a7�9��I       6%�	`�`=���A�2*;


total_loss�2�@

error_R�~I?

learning_rate_1?a7Cpu
I       6%�	~.a=���A�2*;


total_loss.�@

error_R�P?

learning_rate_1?a7Ӵ�cI       6%�	9ta=���A�2*;


total_lossQ�s@

error_R6S]?

learning_rate_1?a7n��I       6%�	�a=���A�2*;


total_loss��@

error_R��d?

learning_rate_1?a7��e�I       6%�	$b=���A�2*;


total_loss�$e@

error_R	�H?

learning_rate_1?a7#vQgI       6%�	�Pb=���A�2*;


total_loss=�@

error_R$�_?

learning_rate_1?a7.5^I       6%�	�b=���A�2*;


total_loss)��@

error_R`�P?

learning_rate_1?a7��$]I       6%�	��b=���A�2*;


total_lossy�@

error_R�S?

learning_rate_1?a7<�+I       6%�	�c=���A�2*;


total_loss�H�@

error_RZ05?

learning_rate_1?a7X�I       6%�	:cc=���A�2*;


total_loss�	�@

error_RF|M?

learning_rate_1?a7D�r�I       6%�	�c=���A�2*;


total_loss6�A

error_RjHV?

learning_rate_1?a7*�D�I       6%�	@�c=���A�2*;


total_loss�ӊ@

error_RJU?

learning_rate_1?a7@��I       6%�	�6d=���A�2*;


total_loss���@

error_R�S?

learning_rate_1?a7�ª�I       6%�	zd=���A�2*;


total_loss���@

error_R�G[?

learning_rate_1?a7���4I       6%�	B�d=���A�2*;


total_lossW�2A

error_R��8?

learning_rate_1?a7J�d�I       6%�	^e=���A�2*;


total_loss]�@

error_R�-c?

learning_rate_1?a76�!�I       6%�	�Ce=���A�2*;


total_loss�͔@

error_R��K?

learning_rate_1?a7TJ��I       6%�	��e=���A�2*;


total_loss�ū@

error_R�QO?

learning_rate_1?a7�p�I       6%�	��e=���A�2*;


total_lossA�@

error_R��F?

learning_rate_1?a7�8�@I       6%�	�f=���A�2*;


total_loss!��@

error_R�N?

learning_rate_1?a7��]�I       6%�	8qf=���A�2*;


total_loss�A

error_R�.?

learning_rate_1?a7�҂vI       6%�	��f=���A�2*;


total_lossc�h@

error_Rq^L?

learning_rate_1?a7|��I       6%�	�g=���A�2*;


total_lossOl�@

error_R�=I?

learning_rate_1?a7E�"�I       6%�	ag=���A�2*;


total_loss2��@

error_R�7T?

learning_rate_1?a7,�]�I       6%�	Ȭg=���A�2*;


total_loss�@

error_R��W?

learning_rate_1?a7�Q��I       6%�	
�g=���A�2*;


total_loss�~�@

error_R1�O?

learning_rate_1?a7~=�I       6%�	%<h=���A�2*;


total_loss��@

error_R�K?

learning_rate_1?a7X�̎I       6%�	
�h=���A�2*;


total_loss���@

error_R��B?

learning_rate_1?a7�-kcI       6%�	�h=���A�2*;


total_lossT�@

error_R�QH?

learning_rate_1?a74+�DI       6%�	i=���A�2*;


total_loss���@

error_Rd�F?

learning_rate_1?a7Z2CI       6%�	8Si=���A�2*;


total_lossӑ�@

error_RFX?

learning_rate_1?a7�*RI       6%�	��i=���A�2*;


total_lossR�@

error_R8�Z?

learning_rate_1?a7�ERYI       6%�	�i=���A�2*;


total_loss?��@

error_Rj�>?

learning_rate_1?a7��8�I       6%�	R,j=���A�2*;


total_loss�W!A

error_R\tP?

learning_rate_1?a7\��#I       6%�	�wj=���A�2*;


total_loss�/�@

error_Rn�Z?

learning_rate_1?a75��I       6%�	C�j=���A�2*;


total_lossR6�@

error_R��>?

learning_rate_1?a7��tI       6%�	�k=���A�2*;


total_lossW�@

error_R3�F?

learning_rate_1?a7u'$I       6%�	�Hk=���A�2*;


total_lossD��@

error_R�!N?

learning_rate_1?a7ms ]I       6%�	��k=���A�2*;


total_loss���@

error_R��V?

learning_rate_1?a7��|I       6%�	��k=���A�2*;


total_loss�u�@

error_R_�k?

learning_rate_1?a7k_�I       6%�	�l=���A�2*;


total_lossN�A

error_Rd(R?

learning_rate_1?a7(�Q�I       6%�	�^l=���A�2*;


total_loss1��@

error_R�UD?

learning_rate_1?a7�#4�I       6%�	��l=���A�2*;


total_loss��{@

error_R�cG?

learning_rate_1?a7��j�I       6%�	��l=���A�2*;


total_loss@S�@

error_R�ZX?

learning_rate_1?a7'��I       6%�	4)m=���A�2*;


total_lossh�@

error_RsRR?

learning_rate_1?a7����I       6%�	rlm=���A�2*;


total_loss��@

error_R�c?

learning_rate_1?a7 �I       6%�	��m=���A�2*;


total_loss.�@

error_R��M?

learning_rate_1?a7$��I       6%�	d
n=���A�2*;


total_lossl�@

error_R&L?

learning_rate_1?a7��7�I       6%�	�Qn=���A�2*;


total_loss�@

error_R��P?

learning_rate_1?a7�6��I       6%�	}�n=���A�2*;


total_loss�A

error_R��E?

learning_rate_1?a7�QZ>I       6%�	��n=���A�3*;


total_loss�c�@

error_R�JN?

learning_rate_1?a7�}�bI       6%�	j0o=���A�3*;


total_loss
3�@

error_R}�N?

learning_rate_1?a7i���I       6%�	�vo=���A�3*;


total_loss�<�@

error_RvMA?

learning_rate_1?a7�Z^�I       6%�	�o=���A�3*;


total_loss���@

error_R�]G?

learning_rate_1?a7^�ٻI       6%�	Pp=���A�3*;


total_loss}x�@

error_Rd�L?

learning_rate_1?a7E�I       6%�	0Gp=���A�3*;


total_losszc�@

error_R��D?

learning_rate_1?a7�x��I       6%�	:�p=���A�3*;


total_loss�@

error_R�zR?

learning_rate_1?a7 I       6%�	��p=���A�3*;


total_lossA��@

error_R��K?

learning_rate_1?a7�ڈI       6%�	q=���A�3*;


total_loss/z�@

error_R{�L?

learning_rate_1?a7u�+I       6%�	[q=���A�3*;


total_loss�KA

error_R�L?

learning_rate_1?a7^�OI       6%�	��q=���A�3*;


total_loss�!�@

error_R�eI?

learning_rate_1?a7�uQEI       6%�	<�q=���A�3*;


total_loss�h�@

error_R1(R?

learning_rate_1?a7Mdt�I       6%�	@6r=���A�3*;


total_lossʐ@

error_R��A?

learning_rate_1?a7i���I       6%�	�}r=���A�3*;


total_lossr��@

error_RM_Y?

learning_rate_1?a7���I       6%�	j�r=���A�3*;


total_loss��@

error_R`?

learning_rate_1?a7k�\�I       6%�	)s=���A�3*;


total_lossz��@

error_R|<?

learning_rate_1?a7�o��I       6%�	ZPs=���A�3*;


total_loss��@

error_RwX?

learning_rate_1?a7�	��I       6%�	��s=���A�3*;


total_loss\I�@

error_RXS?

learning_rate_1?a7i�%I       6%�	��s=���A�3*;


total_loss�ܭ@

error_R=�C?

learning_rate_1?a7P�aI       6%�	� t=���A�3*;


total_loss��@

error_RJ�[?

learning_rate_1?a7��+I       6%�	�et=���A�3*;


total_lossC0�@

error_R�KU?

learning_rate_1?a7��I       6%�	"�t=���A�3*;


total_loss�W�@

error_R��Z?

learning_rate_1?a7{�D�I       6%�	�t=���A�3*;


total_loss
�@

error_R)�[?

learning_rate_1?a7<ЧNI       6%�	a+u=���A�3*;


total_loss�R�@

error_Rt(\?

learning_rate_1?a7��I       6%�	Pmu=���A�3*;


total_loss�Ү@

error_R:�P?

learning_rate_1?a7���8I       6%�	e�u=���A�3*;


total_loss�2�@

error_R}�N?

learning_rate_1?a7~]�I       6%�	�u=���A�3*;


total_loss(��@

error_R�CJ?

learning_rate_1?a7�$PI       6%�	=5v=���A�3*;


total_loss��m@

error_R��8?

learning_rate_1?a7�?��I       6%�	3�v=���A�3*;


total_lossA

error_R�EP?

learning_rate_1?a7T;UI       6%�	>�v=���A�3*;


total_lossĿ�@

error_R�!U?

learning_rate_1?a7@Y�mI       6%�	?$w=���A�3*;


total_loss�ʩ@

error_RtkL?

learning_rate_1?a7�Y�oI       6%�	lw=���A�3*;


total_loss�>�@

error_RcmK?

learning_rate_1?a7P�rJI       6%�	��w=���A�3*;


total_lossi�@

error_R�Z?

learning_rate_1?a7u�s�I       6%�	��w=���A�3*;


total_loss)��@

error_R�U]?

learning_rate_1?a7�xs(I       6%�	;x=���A�3*;


total_loss��@

error_R4F?

learning_rate_1?a7 ��`I       6%�	T�x=���A�3*;


total_loss���@

error_R�D?

learning_rate_1?a7\@|�I       6%�	t�x=���A�3*;


total_lossF��@

error_R��U?

learning_rate_1?a7��#$I       6%�	)y=���A�3*;


total_loss�Ǒ@

error_RhqJ?

learning_rate_1?a73�$�I       6%�	UOy=���A�3*;


total_lossW�@

error_R�??

learning_rate_1?a7�9��I       6%�	��y=���A�3*;


total_loss��@

error_R�=D?

learning_rate_1?a70���I       6%�	��y=���A�3*;


total_loss�֤@

error_R��S?

learning_rate_1?a7p$
XI       6%�	�z=���A�3*;


total_loss���@

error_R?E?

learning_rate_1?a7%C�/I       6%�	 \z=���A�3*;


total_loss�	�@

error_Rvf??

learning_rate_1?a7�>zI       6%�	Şz=���A�3*;


total_loss?�@

error_R�P?

learning_rate_1?a7�y*I       6%�	��z=���A�3*;


total_losscI�@

error_RmM?

learning_rate_1?a7!Sa_I       6%�	�#{=���A�3*;


total_loss�a�@

error_R�FL?

learning_rate_1?a7n_��I       6%�	kh{=���A�3*;


total_loss{2�@

error_R�?^?

learning_rate_1?a7�;KwI       6%�	��{=���A�3*;


total_lossl��@

error_Rq9V?

learning_rate_1?a7X�(I       6%�	g�{=���A�3*;


total_loss_��@

error_RlzP?

learning_rate_1?a7�ut,I       6%�	-2|=���A�3*;


total_lossxf�@

error_R�W?

learning_rate_1?a7N��"I       6%�	�v|=���A�3*;


total_loss�#�@

error_R��/?

learning_rate_1?a7�:�I       6%�	$�|=���A�3*;


total_loss��:A

error_R3�N?

learning_rate_1?a7���TI       6%�	'}=���A�3*;


total_lossC��@

error_RZU?

learning_rate_1?a7�ڵ�I       6%�	@F}=���A�3*;


total_loss<��@

error_R�$Q?

learning_rate_1?a7{�_sI       6%�	6�}=���A�3*;


total_loss,�w@

error_R3gG?

learning_rate_1?a7(_RI       6%�	Q�}=���A�3*;


total_lossAJ�@

error_R<6L?

learning_rate_1?a7~"�I       6%�	�~=���A�3*;


total_lossE"�@

error_R��Z?

learning_rate_1?a7�>�?I       6%�	}T~=���A�3*;


total_loss�`�@

error_R��??

learning_rate_1?a7����I       6%�	ۗ~=���A�3*;


total_loss�0�@

error_R��P?

learning_rate_1?a7ɲ9�I       6%�	��~=���A�3*;


total_loss
�r@

error_Ro�D?

learning_rate_1?a7)�2�I       6%�	X=���A�3*;


total_lossz��@

error_R)�D?

learning_rate_1?a7��pxI       6%�	�a=���A�3*;


total_loss�1�@

error_REJ?

learning_rate_1?a7�?2I       6%�	��=���A�3*;


total_lossi��@

error_R$�E?

learning_rate_1?a7*�OI       6%�	i�=���A�3*;


total_loss�O�@

error_R;�O?

learning_rate_1?a7�7�I       6%�	g��=���A�3*;


total_loss�c�@

error_R�UI?

learning_rate_1?a7�}B�I       6%�	���=���A�3*;


total_loss�!�@

error_R�SU?

learning_rate_1?a7cmxI       6%�	�3�=���A�3*;


total_loss�Ͱ@

error_R.�K?

learning_rate_1?a7�4��I       6%�	hu�=���A�3*;


total_loss��@

error_R_J?

learning_rate_1?a7���fI       6%�	��=���A�3*;


total_loss�F�@

error_R�%O?

learning_rate_1?a7j~oI       6%�	���=���A�3*;


total_loss{ߘ@

error_R�0e?

learning_rate_1?a7�V|xI       6%�	�@�=���A�3*;


total_loss�z@

error_RS2J?

learning_rate_1?a7�
��I       6%�	*��=���A�3*;


total_lossv<�@

error_RK?

learning_rate_1?a7!��>I       6%�	@͂=���A�3*;


total_loss*ay@

error_R�/J?

learning_rate_1?a7��yI       6%�	��=���A�3*;


total_lossw�@

error_R�>?

learning_rate_1?a7��y�I       6%�	[[�=���A�3*;


total_loss;�@

error_R�J[?

learning_rate_1?a7��wI       6%�	���=���A�3*;


total_loss���@

error_R�%Y?

learning_rate_1?a7�˸I       6%�	��=���A�3*;


total_loss:K�@

error_R�	H?

learning_rate_1?a7�0�hI       6%�	�-�=���A�3*;


total_lossL��@

error_RS�f?

learning_rate_1?a7\!��I       6%�	Ut�=���A�3*;


total_loss� �@

error_R��;?

learning_rate_1?a7��cVI       6%�	߻�=���A�3*;


total_loss�{@

error_R�_T?

learning_rate_1?a7�tLFI       6%�	��=���A�3*;


total_loss���@

error_R<�N?

learning_rate_1?a7�ѥ�I       6%�	xG�=���A�3*;


total_lossܑ�@

error_R�NO?

learning_rate_1?a7nm��I       6%�	���=���A�3*;


total_lossG�A

error_R�$Q?

learning_rate_1?a7i�U�I       6%�	�ʅ=���A�3*;


total_loss �@

error_R��Y?

learning_rate_1?a7��I       6%�	��=���A�3*;


total_loss�R�@

error_R�A?

learning_rate_1?a7��bI       6%�	DS�=���A�3*;


total_loss\�=A

error_R.�J?

learning_rate_1?a7�؋I       6%�	���=���A�3*;


total_loss!7�@

error_R=UE?

learning_rate_1?a7��=I       6%�	�=���A�3*;


total_lossAӥ@

error_RE{O?

learning_rate_1?a7
��I       6%�	>I�=���A�3*;


total_loss���@

error_R��8?

learning_rate_1?a7<���I       6%�	m��=���A�3*;


total_loss(��@

error_Re�`?

learning_rate_1?a7���FI       6%�	�և=���A�3*;


total_lossѹ�@

error_RQ�S?

learning_rate_1?a7��.�I       6%�	��=���A�3*;


total_loss�^�@

error_R��W?

learning_rate_1?a7�-UI       6%�	�e�=���A�3*;


total_loss���@

error_R�V5?

learning_rate_1?a7`���I       6%�	h��=���A�3*;


total_loss�u�@

error_R8�I?

learning_rate_1?a7��"I       6%�	�=���A�3*;


total_loss4�@

error_R�CJ?

learning_rate_1?a7����I       6%�	T7�=���A�3*;


total_loss<I�@

error_R@mO?

learning_rate_1?a7V�7`I       6%�	�=���A�3*;


total_loss��@

error_RcfJ?

learning_rate_1?a7<4�WI       6%�	�=���A�3*;


total_losseG�@

error_R�M?

learning_rate_1?a7�K	�I       6%�	n�=���A�3*;


total_loss�?�@

error_RWaC?

learning_rate_1?a7ױjmI       6%�	�K�=���A�3*;


total_loss�g�@

error_R��N?

learning_rate_1?a7�{��I       6%�	?��=���A�3*;


total_loss+�@

error_Rq�]?

learning_rate_1?a7�I       6%�	�ۊ=���A�3*;


total_loss�BA

error_Rf�D?

learning_rate_1?a7�L+�I       6%�	�"�=���A�3*;


total_loss�@�@

error_R��Q?

learning_rate_1?a7O���I       6%�	 h�=���A�3*;


total_loss(�A

error_RsXP?

learning_rate_1?a7Ϳ��I       6%�	���=���A�3*;


total_lossUE�@

error_R҃X?

learning_rate_1?a7��I       6%�	���=���A�3*;


total_lossW+�@

error_R�DL?

learning_rate_1?a7�9��I       6%�	k>�=���A�3*;


total_loss�M�@

error_RO�E?

learning_rate_1?a7�ճ3I       6%�	���=���A�3*;


total_loss<o�@

error_RDDa?

learning_rate_1?a7�e��I       6%�	uʌ=���A�3*;


total_loss�!v@

error_R�G=?

learning_rate_1?a7�h�I       6%�	#�=���A�3*;


total_lossC�@

error_RG?

learning_rate_1?a7 l`~I       6%�	VU�=���A�3*;


total_loss�3�@

error_Rō\?

learning_rate_1?a7S2¿I       6%�	n��=���A�3*;


total_lossr��@

error_R�Z?

learning_rate_1?a7�+aI       6%�	��=���A�3*;


total_loss׉�@

error_R��N?

learning_rate_1?a7��	�I       6%�	�C�=���A�3*;


total_loss�@

error_RI�I?

learning_rate_1?a7�ˢI       6%�	���=���A�3*;


total_lossư�@

error_RK?

learning_rate_1?a7I�#�I       6%�	4Ɏ=���A�3*;


total_loss͵�@

error_R�dG?

learning_rate_1?a7nGa[I       6%�	�&�=���A�3*;


total_loss���@

error_R��M?

learning_rate_1?a7�7 �I       6%�	�s�=���A�3*;


total_loss��f@

error_R�3Q?

learning_rate_1?a7�E�I       6%�	_��=���A�3*;


total_lossE��@

error_RܚT?

learning_rate_1?a7J/tI       6%�	���=���A�3*;


total_loss��@

error_R� J?

learning_rate_1?a7o��YI       6%�	,>�=���A�3*;


total_loss�\�@

error_R�l?

learning_rate_1?a7���I       6%�	H��=���A�3*;


total_lossr��@

error_RS,T?

learning_rate_1?a7���I       6%�	�ǐ=���A�3*;


total_loss���@

error_R��;?

learning_rate_1?a7���I       6%�	��=���A�3*;


total_loss��M@

error_R�"A?

learning_rate_1?a7"�XI       6%�	|R�=���A�3*;


total_loss�ޚ@

error_RؚK?

learning_rate_1?a7�]��I       6%�	ؕ�=���A�3*;


total_loss���@

error_RW�L?

learning_rate_1?a7��PvI       6%�	�ۑ=���A�3*;


total_loss�!�@

error_R��Q?

learning_rate_1?a7��I       6%�	.#�=���A�3*;


total_loss���@

error_R�_N?

learning_rate_1?a7�:3I       6%�	�n�=���A�4*;


total_lossz�k@

error_R��i?

learning_rate_1?a7"ȬI       6%�	c��=���A�4*;


total_loss?0�@

error_R��K?

learning_rate_1?a7����I       6%�	X��=���A�4*;


total_loss[��@

error_R��\?

learning_rate_1?a7���!I       6%�		@�=���A�4*;


total_lossq\�@

error_R��B?

learning_rate_1?a7�8��I       6%�	���=���A�4*;


total_loss�!�@

error_R�|l?

learning_rate_1?a7"xA�I       6%�	ԓ=���A�4*;


total_lossLe{@

error_RM�W?

learning_rate_1?a7Z��I       6%�	�=���A�4*;


total_loss.�o@

error_R�P?

learning_rate_1?a7�J��I       6%�	�\�=���A�4*;


total_loss\��@

error_RU?

learning_rate_1?a7�tI�I       6%�	g��=���A�4*;


total_lossz��@

error_RҪ_?

learning_rate_1?a7��:I       6%�	��=���A�4*;


total_lossso�@

error_R�}h?

learning_rate_1?a7��I       6%�	�&�=���A�4*;


total_loss�>�@

error_R��c?

learning_rate_1?a7c��I       6%�	Wi�=���A�4*;


total_loss� �@

error_RlM?

learning_rate_1?a7��ѳI       6%�	N��=���A�4*;


total_loss}q�@

error_R*kG?

learning_rate_1?a7����I       6%�	 �=���A�4*;


total_loss1��@

error_R{??

learning_rate_1?a7� I       6%�	�6�=���A�4*;


total_loss��@

error_R	H?

learning_rate_1?a7:��I       6%�	$��=���A�4*;


total_loss�ޭ@

error_R�BE?

learning_rate_1?a7��|I       6%�	�ږ=���A�4*;


total_loss:g�@

error_R�K?

learning_rate_1?a7�ӳ�I       6%�	�!�=���A�4*;


total_losst( A

error_RatT?

learning_rate_1?a7T�eI       6%�	1e�=���A�4*;


total_lossl��@

error_R�:J?

learning_rate_1?a7��I       6%�	���=���A�4*;


total_lossx�@

error_R\3Q?

learning_rate_1?a7n�A�I       6%�	��=���A�4*;


total_loss��@

error_R�H?

learning_rate_1?a7�sQI       6%�	�3�=���A�4*;


total_loss嬬@

error_R.U?

learning_rate_1?a7 �I       6%�	���=���A�4*;


total_loss>�@

error_R! P?

learning_rate_1?a7WȻI       6%�	kȘ=���A�4*;


total_loss�@

error_R�X?

learning_rate_1?a7�҆�I       6%�	��=���A�4*;


total_loss���@

error_R��A?

learning_rate_1?a7I'��I       6%�	�^�=���A�4*;


total_lossͤ�@

error_R��*?

learning_rate_1?a7*�۷I       6%�	���=���A�4*;


total_loss�r�@

error_R�_[?

learning_rate_1?a7^7e�I       6%�	��=���A�4*;


total_loss���@

error_Rr�S?

learning_rate_1?a7��<0I       6%�	�6�=���A�4*;


total_loss	ܱ@

error_R:?D?

learning_rate_1?a7yF
I       6%�	�{�=���A�4*;


total_loss7[�@

error_R��F?

learning_rate_1?a7��RI       6%�	�Ú=���A�4*;


total_loss��@

error_R]�a?

learning_rate_1?a7u"QI       6%�	��=���A�4*;


total_loss!�@

error_R��`?

learning_rate_1?a7� �QI       6%�	�V�=���A�4*;


total_lossh��@

error_R��J?

learning_rate_1?a7��ıI       6%�	���=���A�4*;


total_loss]3�@

error_R�zL?

learning_rate_1?a7s�+<I       6%�	��=���A�4*;


total_loss�U�@

error_Ri???

learning_rate_1?a7�=�AI       6%�	%�=���A�4*;


total_lossf��@

error_R��;?

learning_rate_1?a78'o�I       6%�	�g�=���A�4*;


total_loss6��@

error_RsOR?

learning_rate_1?a7�ޥI       6%�	:��=���A�4*;


total_lossÜ�@

error_R/O?

learning_rate_1?a7-�4�I       6%�	�=���A�4*;


total_loss��@

error_R��C?

learning_rate_1?a7�TI       6%�	@4�=���A�4*;


total_loss��A

error_R��C?

learning_rate_1?a7^<`�I       6%�	{�=���A�4*;


total_loss��@

error_R=S5?

learning_rate_1?a7r�A�I       6%�	i��=���A�4*;


total_loss%��@

error_R�X?

learning_rate_1?a7��|$I       6%�	 �=���A�4*;


total_lossC�@

error_R-V?

learning_rate_1?a7#�`I       6%�	�B�=���A�4*;


total_loss�G�@

error_R�E?

learning_rate_1?a7�c�I       6%�	懞=���A�4*;


total_losstO�@

error_R�wH?

learning_rate_1?a7K�I       6%�	=Ԟ=���A�4*;


total_loss�!�@

error_R��A?

learning_rate_1?a7`j��I       6%�	��=���A�4*;


total_loss5�@

error_R��N?

learning_rate_1?a7)�c�I       6%�		g�=���A�4*;


total_loss�@

error_RW�J?

learning_rate_1?a7�!A�I       6%�	U��=���A�4*;


total_lossN�@

error_R�OM?

learning_rate_1?a7r��I       6%�	5��=���A�4*;


total_lossO��@

error_R?�b?

learning_rate_1?a7�M��I       6%�	�:�=���A�4*;


total_loss��s@

error_R��L?

learning_rate_1?a7#��I       6%�	P��=���A�4*;


total_loss12u@

error_RZ�I?

learning_rate_1?a7���I       6%�	6ɠ=���A�4*;


total_loss�W�@

error_Rm>Y?

learning_rate_1?a7��I       6%�	��=���A�4*;


total_loss�@

error_R4�A?

learning_rate_1?a7<yI       6%�	�W�=���A�4*;


total_loss!-�@

error_RS�`?

learning_rate_1?a7�ō�I       6%�	ə�=���A�4*;


total_losss��@

error_R��I?

learning_rate_1?a7؍~�I       6%�	{ޡ=���A�4*;


total_loss1��@

error_RSvP?

learning_rate_1?a7n���I       6%�	%�=���A�4*;


total_loss���@

error_R��R?

learning_rate_1?a7��I       6%�	j�=���A�4*;


total_loss��@

error_R�Q?

learning_rate_1?a7y�_�I       6%�	���=���A�4*;


total_lossH��@

error_R��D?

learning_rate_1?a7����I       6%�	��=���A�4*;


total_loss���@

error_R�K?

learning_rate_1?a7"g:\I       6%�	�6�=���A�4*;


total_loss�ӡ@

error_R��E?

learning_rate_1?a7a�5�I       6%�	�{�=���A�4*;


total_loss/ۦ@

error_R��J?

learning_rate_1?a7��V�I       6%�	|��=���A�4*;


total_losse�@

error_R�\[?

learning_rate_1?a7���2I       6%�	"�=���A�4*;


total_lossCD�@

error_Rx�Q?

learning_rate_1?a7��I       6%�	�D�=���A�4*;


total_loss�ܑ@

error_RܵO?

learning_rate_1?a7�2I       6%�	)��=���A�4*;


total_lossJ�@

error_R��J?

learning_rate_1?a7���1I       6%�	�ˤ=���A�4*;


total_loss1��@

error_Rj	Q?

learning_rate_1?a7�BO�I       6%�	j�=���A�4*;


total_loss���@

error_RYW?

learning_rate_1?a7T�I       6%�	%]�=���A�4*;


total_loss���@

error_RT6?

learning_rate_1?a7L�`�I       6%�	I��=���A�4*;


total_lossH�@

error_R&G?

learning_rate_1?a78�I       6%�	��=���A�4*;


total_lossX��@

error_RIlT?

learning_rate_1?a7�,jzI       6%�	Q4�=���A�4*;


total_loss��@

error_R��L?

learning_rate_1?a7|�:I       6%�	���=���A�4*;


total_loss��@

error_R�@?

learning_rate_1?a7ܣ.MI       6%�	�ߦ=���A�4*;


total_loss ڨ@

error_R��R?

learning_rate_1?a7�s�CI       6%�	�"�=���A�4*;


total_loss-��@

error_R��L?

learning_rate_1?a7DTi�I       6%�	�c�=���A�4*;


total_lossR��@

error_RD�O?

learning_rate_1?a7��sI       6%�	ب�=���A�4*;


total_loss�@

error_R�x\?

learning_rate_1?a7j��kI       6%�	��=���A�4*;


total_loss�\�@

error_R�cK?

learning_rate_1?a7BI       6%�	0�=���A�4*;


total_loss��@

error_RTL?

learning_rate_1?a7SsifI       6%�	
r�=���A�4*;


total_loss���@

error_R�S?

learning_rate_1?a7B>o�I       6%�	���=���A�4*;


total_lossM'b@

error_RMcA?

learning_rate_1?a7�b�I       6%�	1��=���A�4*;


total_loss%g�@

error_R:;Q?

learning_rate_1?a7�	ΊI       6%�	�6�=���A�4*;


total_loss���@

error_Rv�I?

learning_rate_1?a7J�|!I       6%�	�w�=���A�4*;


total_loss��8A

error_R��K?

learning_rate_1?a7���I       6%�	߸�=���A�4*;


total_losst>�@

error_Rw�M?

learning_rate_1?a7��W�I       6%�	���=���A�4*;


total_loss�@

error_R��U?

learning_rate_1?a7�QI       6%�	�<�=���A�4*;


total_loss�@

error_RO�V?

learning_rate_1?a7�<9�I       6%�	�}�=���A�4*;


total_lossݪ�@

error_R.�M?

learning_rate_1?a7W7�I       6%�	���=���A�4*;


total_loss%�@

error_R,�F?

learning_rate_1?a7��]I       6%�	��=���A�4*;


total_loss(��@

error_R�X?

learning_rate_1?a7r
�I       6%�	�S�=���A�4*;


total_loss�1�@

error_R��>?

learning_rate_1?a7v���I       6%�	4��=���A�4*;


total_loss�*�@

error_R�A?

learning_rate_1?a7�,�I       6%�	>�=���A�4*;


total_loss�:�@

error_R�J?

learning_rate_1?a7�b;	I       6%�	�9�=���A�4*;


total_loss!mo@

error_R�3B?

learning_rate_1?a7�\hI       6%�	[��=���A�4*;


total_lossq��@

error_RR�R?

learning_rate_1?a7=2wI       6%�	Ĭ=���A�4*;


total_loss���@

error_R).C?

learning_rate_1?a7�6|�I       6%�	��=���A�4*;


total_losswݬ@

error_R�1E?

learning_rate_1?a7 Q}iI       6%�	N�=���A�4*;


total_losssń@

error_R��P?

learning_rate_1?a7ӨX�I       6%�	���=���A�4*;


total_loss	��@

error_R�hT?

learning_rate_1?a7f��8I       6%�	�ܭ=���A�4*;


total_loss��@

error_R�u>?

learning_rate_1?a7S<�I       6%�	=6�=���A�4*;


total_loss�@�@

error_R��B?

learning_rate_1?a7*ͧ�I       6%�	�=���A�4*;


total_loss���@

error_R�1Q?

learning_rate_1?a7�"?1I       6%�	Ӯ=���A�4*;


total_loss���@

error_RQ?

learning_rate_1?a7�ڣ�I       6%�	&�=���A�4*;


total_lossTS�@

error_R��S?

learning_rate_1?a7ܤ~%I       6%�	5��=���A�4*;


total_lossc�@

error_R@I?

learning_rate_1?a7��{�I       6%�	_ʯ=���A�4*;


total_loss8k�@

error_Rt5?

learning_rate_1?a7x�rI       6%�	H�=���A�4*;


total_lossq�@

error_R��I?

learning_rate_1?a7nKo�I       6%�	�]�=���A�4*;


total_loss7�@

error_R��S?

learning_rate_1?a7���I       6%�	���=���A�4*;


total_loss:��@

error_RaZQ?

learning_rate_1?a7^�B�I       6%�	4�=���A�4*;


total_loss[��@

error_Rx�]?

learning_rate_1?a7$�+�I       6%�	�-�=���A�4*;


total_lossJY�@

error_R��Z?

learning_rate_1?a7]�0I       6%�	�r�=���A�4*;


total_loss4N�@

error_R�=@?

learning_rate_1?a7S�rI       6%�	Ϲ�=���A�4*;


total_loss[�@

error_RxxN?

learning_rate_1?a7��aI       6%�	��=���A�4*;


total_loss�ƶ@

error_R�T?

learning_rate_1?a7t�E�I       6%�	mJ�=���A�4*;


total_lossR�@

error_R��N?

learning_rate_1?a7�$�I       6%�	��=���A�4*;


total_loss|��@

error_R��O?

learning_rate_1?a7��I       6%�	�Ӳ=���A�4*;


total_loss洯@

error_R��D?

learning_rate_1?a7o��@I       6%�	��=���A�4*;


total_loss&�@

error_R�'S?

learning_rate_1?a7TO<�I       6%�	&`�=���A�4*;


total_loss��@

error_R��W?

learning_rate_1?a7��UBI       6%�	G��=���A�4*;


total_loss� �@

error_R��E?

learning_rate_1?a7<��I       6%�	\�=���A�4*;


total_loss�b@

error_R�EH?

learning_rate_1?a7C�98I       6%�	�.�=���A�4*;


total_loss��@

error_R�S?

learning_rate_1?a7��t I       6%�	@p�=���A�4*;


total_loss�0A

error_R!-?

learning_rate_1?a7R}H�I       6%�	��=���A�4*;


total_lossCj�@

error_RlY?

learning_rate_1?a7W���I       6%�	3��=���A�4*;


total_loss���@

error_R8/U?

learning_rate_1?a7���-I       6%�	�9�=���A�4*;


total_loss�!�@

error_R܏8?

learning_rate_1?a7�<4I       6%�	M~�=���A�4*;


total_loss�Z�@

error_Rn%T?

learning_rate_1?a7��I       6%�	|µ=���A�5*;


total_loss�@

error_RfMD?

learning_rate_1?a7���qI       6%�	��=���A�5*;


total_loss1��@

error_R�K?

learning_rate_1?a7�<I       6%�	+E�=���A�5*;


total_lossL�@

error_RR^E?

learning_rate_1?a7<�DI       6%�	���=���A�5*;


total_loss�WA

error_R�cD?

learning_rate_1?a7m�aeI       6%�	|��=���A�5*;


total_loss�=�@

error_R�RD?

learning_rate_1?a7%zA:I       6%�	D=�=���A�5*;


total_loss�1�@

error_R�\?

learning_rate_1?a7��
I       6%�	�~�=���A�5*;


total_loss4�@

error_R�f?

learning_rate_1?a7�C��I       6%�	�·=���A�5*;


total_loss-��@

error_R�#O?

learning_rate_1?a7��߫I       6%�	h�=���A�5*;


total_loss�=CA

error_R~I?

learning_rate_1?a7>��I       6%�	�I�=���A�5*;


total_loss��@

error_R�M\?

learning_rate_1?a7���I       6%�	א�=���A�5*;


total_loss���@

error_R�5Z?

learning_rate_1?a7#ZEI       6%�	�ܸ=���A�5*;


total_loss(��@

error_Ri�V?

learning_rate_1?a7gğ>I       6%�	�#�=���A�5*;


total_loss�-�@

error_R!�=?

learning_rate_1?a7����I       6%�	?i�=���A�5*;


total_loss(��@

error_R�h??

learning_rate_1?a7N{�vI       6%�	c��=���A�5*;


total_lossTP�@

error_RA�8?

learning_rate_1?a7x)`jI       6%�	���=���A�5*;


total_loss� A

error_Ra?

learning_rate_1?a7˼�I       6%�	<�=���A�5*;


total_lossZX�@

error_R�"]?

learning_rate_1?a7YL�JI       6%�	��=���A�5*;


total_loss� A

error_R��C?

learning_rate_1?a7���I       6%�	SǺ=���A�5*;


total_loss�ih@

error_R C?

learning_rate_1?a7�!�zI       6%�	Q�=���A�5*;


total_loss�,�@

error_R�VD?

learning_rate_1?a7�,V�I       6%�	�W�=���A�5*;


total_loss@��@

error_Rl�n?

learning_rate_1?a7�<�I       6%�	5��=���A�5*;


total_loss o�@

error_R��V?

learning_rate_1?a7���_I       6%�		ۻ=���A�5*;


total_loss�B�@

error_R�uW?

learning_rate_1?a7����I       6%�	� �=���A�5*;


total_loss6��@

error_R� X?

learning_rate_1?a7L)-I       6%�	b�=���A�5*;


total_loss�@

error_R}dI?

learning_rate_1?a7R˗]I       6%�	���=���A�5*;


total_lossI�@

error_R�7V?

learning_rate_1?a7�* �I       6%�	9�=���A�5*;


total_loss�Ь@

error_RW6E?

learning_rate_1?a7b��yI       6%�	�/�=���A�5*;


total_loss�n�@

error_RFA?

learning_rate_1?a7��jI       6%�	�w�=���A�5*;


total_loss&�@

error_R(G?

learning_rate_1?a7�T&I       6%�	���=���A�5*;


total_lossN�@

error_R��L?

learning_rate_1?a7㐶�I       6%�	�=���A�5*;


total_loss�U�@

error_R3rJ?

learning_rate_1?a7�U`tI       6%�	�O�=���A�5*;


total_loss掾@

error_R�C?

learning_rate_1?a7�o��I       6%�	��=���A�5*;


total_loss��@

error_RR�c?

learning_rate_1?a7�|<I       6%�	l޾=���A�5*;


total_lossJ��@

error_RW�L?

learning_rate_1?a7��i�I       6%�	�%�=���A�5*;


total_loss�/�@

error_R�SQ?

learning_rate_1?a7Q8��I       6%�	�i�=���A�5*;


total_loss���@

error_R�[?

learning_rate_1?a7j���I       6%�	���=���A�5*;


total_loss�_�@

error_Ri"=?

learning_rate_1?a7�r1�I       6%�	j��=���A�5*;


total_loss���@

error_R�IQ?

learning_rate_1?a7�ψI       6%�	�8�=���A�5*;


total_loss�*�@

error_Ri�_?

learning_rate_1?a7���FI       6%�	|�=���A�5*;


total_loss�-�@

error_RCG?

learning_rate_1?a7���I       6%�	��=���A�5*;


total_loss(�r@

error_R6HG?

learning_rate_1?a7��>I       6%�	���=���A�5*;


total_loss��@

error_R�X?

learning_rate_1?a7��
I       6%�	C�=���A�5*;


total_loss�m�@

error_R��L?

learning_rate_1?a7�>~I       6%�	���=���A�5*;


total_loss�@

error_R�@X?

learning_rate_1?a7Rv�I       6%�	���=���A�5*;


total_loss){�@

error_RC�E?

learning_rate_1?a7�</I       6%�	��=���A�5*;


total_lossu֓@

error_R�8N?

learning_rate_1?a7�>MI       6%�	�T�=���A�5*;


total_loss4|�@

error_R��B?

learning_rate_1?a7c�6I       6%�	���=���A�5*;


total_loss���@

error_R<Z?

learning_rate_1?a7�遂I       6%�	���=���A�5*;


total_lossߝ_@

error_R&	U?

learning_rate_1?a7�&�I       6%�	P"�=���A�5*;


total_loss`~@

error_R}�H?

learning_rate_1?a7���I       6%�	�g�=���A�5*;


total_loss�4�@

error_RUV?

learning_rate_1?a7�y��I       6%�	<��=���A�5*;


total_loss8�@

error_R�=?

learning_rate_1?a7�v2�I       6%�	;��=���A�5*;


total_loss��@

error_R�G?

learning_rate_1?a7�߉EI       6%�	&7�=���A�5*;


total_loss�7A

error_R�_a?

learning_rate_1?a7�F,I       6%�	#{�=���A�5*;


total_loss�z�@

error_R�?U?

learning_rate_1?a7c�I       6%�	l��=���A�5*;


total_loss�#g@

error_R�C?

learning_rate_1?a7�L/I       6%�	��=���A�5*;


total_loss`��@

error_RJ�D?

learning_rate_1?a7��I       6%�	�K�=���A�5*;


total_lossJ�A

error_R��J?

learning_rate_1?a7����I       6%�	���=���A�5*;


total_loss���@

error_R�ZN?

learning_rate_1?a7��oI       6%�	Y��=���A�5*;


total_lossl��@

error_RaV?

learning_rate_1?a7�T$gI       6%�	J+�=���A�5*;


total_loss��A

error_R.(L?

learning_rate_1?a7�-��I       6%�	���=���A�5*;


total_loss�ZA

error_R�*Z?

learning_rate_1?a7�ZjiI       6%�	��=���A�5*;


total_loss�@

error_R��??

learning_rate_1?a7d���I       6%�	�%�=���A�5*;


total_loss���@

error_RW)E?

learning_rate_1?a7��I       6%�	�m�=���A�5*;


total_lossm�@

error_R�^`?

learning_rate_1?a7��͞I       6%�	���=���A�5*;


total_loss)W�@

error_R�L?

learning_rate_1?a72�^iI       6%�	���=���A�5*;


total_loss�L�@

error_R��J?

learning_rate_1?a7�\I       6%�	Z?�=���A�5*;


total_loss�v�@

error_R�R?

learning_rate_1?a7g���I       6%�	��=���A�5*;


total_lossݯ�@

error_R�oM?

learning_rate_1?a7�<ːI       6%�	Q��=���A�5*;


total_loss�:�@

error_RtC?

learning_rate_1?a7=�!�I       6%�	-	�=���A�5*;


total_loss��/A

error_RJ�@?

learning_rate_1?a7^�pI       6%�	�L�=���A�5*;


total_loss��@

error_R��B?

learning_rate_1?a7%�5;I       6%�	 ��=���A�5*;


total_loss��@

error_R��O?

learning_rate_1?a77��I       6%�	���=���A�5*;


total_loss�ҏ@

error_R2LB?

learning_rate_1?a7��I       6%�	�=���A�5*;


total_loss���@

error_RW%U?

learning_rate_1?a7�y�I       6%�	�[�=���A�5*;


total_loss���@

error_Rc�U?

learning_rate_1?a7�I       6%�	���=���A�5*;


total_loss��@

error_R��;?

learning_rate_1?a7�1fI       6%�	j��=���A�5*;


total_loss��@

error_Rq�D?

learning_rate_1?a7i�T�I       6%�	!�=���A�5*;


total_lossL��@

error_R�G?

learning_rate_1?a7�w))I       6%�	�i�=���A�5*;


total_loss	ݶ@

error_RtYR?

learning_rate_1?a7G��I       6%�	ϴ�=���A�5*;


total_loss��@

error_Rq:G?

learning_rate_1?a7��įI       6%�	E��=���A�5*;


total_loss��@

error_R
�Y?

learning_rate_1?a7^.��I       6%�	8>�=���A�5*;


total_lossC*�@

error_RuL?

learning_rate_1?a7B���I       6%�	���=���A�5*;


total_loss p�@

error_R�ND?

learning_rate_1?a7ǀv=I       6%�	���=���A�5*;


total_lossX��@

error_RS�P?

learning_rate_1?a7��G1I       6%�	u�=���A�5*;


total_loss���@

error_RԴD?

learning_rate_1?a7:��I       6%�	bS�=���A�5*;


total_loss�Q�@

error_R��Q?

learning_rate_1?a7Cu��I       6%�	"��=���A�5*;


total_loss�ӹ@

error_R�RA?

learning_rate_1?a7�l��I       6%�	(��=���A�5*;


total_loss?�}@

error_R;c@?

learning_rate_1?a7�	�I       6%�	�6�=���A�5*;


total_loss�}�@

error_R�.R?

learning_rate_1?a7q��dI       6%�	�}�=���A�5*;


total_loss�v�@

error_RW^S?

learning_rate_1?a7H\|I       6%�	Ŀ�=���A�5*;


total_loss�T�@

error_R��R?

learning_rate_1?a7��a�I       6%�	��=���A�5*;


total_loss�\�@

error_RC�;?

learning_rate_1?a7�0�I       6%�	�c�=���A�5*;


total_loss}��@

error_R��L?

learning_rate_1?a7TD�I       6%�	"��=���A�5*;


total_loss��@

error_R�X?

learning_rate_1?a7���I       6%�	,��=���A�5*;


total_loss��@

error_R��M?

learning_rate_1?a7���SI       6%�	�2�=���A�5*;


total_loss�4�@

error_R)�Z?

learning_rate_1?a7u��I       6%�	ax�=���A�5*;


total_lossY�@

error_R�F?

learning_rate_1?a7��eI       6%�	���=���A�5*;


total_loss*��@

error_R?f?

learning_rate_1?a7L	I       6%�	���=���A�5*;


total_lossJ�@

error_R��T?

learning_rate_1?a7g�d[I       6%�	�B�=���A�5*;


total_loss�K�@

error_RVB?

learning_rate_1?a7��E�I       6%�	u��=���A�5*;


total_loss=��@

error_RܭT?

learning_rate_1?a7z%6�I       6%�	^��=���A�5*;


total_loss��@

error_Rd5C?

learning_rate_1?a7���I       6%�	�=���A�5*;


total_loss]�	A

error_R�t?

learning_rate_1?a7���I       6%�	�Z�=���A�5*;


total_loss-��@

error_R|jL?

learning_rate_1?a7��7�I       6%�	���=���A�5*;


total_loss\��@

error_R�U?

learning_rate_1?a7%E/�I       6%�	���=���A�5*;


total_loss]��@

error_R`�G?

learning_rate_1?a70�ߢI       6%�	H.�=���A�5*;


total_loss��@

error_R)Q?

learning_rate_1?a7�J	I       6%�	�p�=���A�5*;


total_loss]�k@

error_R/z\?

learning_rate_1?a7�Q�kI       6%�	&��=���A�5*;


total_lossh��@

error_R?F;?

learning_rate_1?a7�{� I       6%�	���=���A�5*;


total_lossE��@

error_RF�Z?

learning_rate_1?a75h͊I       6%�	�B�=���A�5*;


total_loss�պ@

error_Rxb?

learning_rate_1?a7�.��I       6%�	��=���A�5*;


total_lossg��@

error_R��V?

learning_rate_1?a7~n�I       6%�	n��=���A�5*;


total_lossm�@

error_R��X?

learning_rate_1?a7�`�sI       6%�	��=���A�5*;


total_loss|�@

error_R�)I?

learning_rate_1?a7�eQMI       6%�	�L�=���A�5*;


total_loss��@

error_R�O?

learning_rate_1?a7�"�I       6%�	��=���A�5*;


total_loss�A�@

error_R��X?

learning_rate_1?a7��+~I       6%�	���=���A�5*;


total_lossr��@

error_R܆B?

learning_rate_1?a7�w$bI       6%�	�!�=���A�5*;


total_lossa��@

error_R6_E?

learning_rate_1?a7��(I       6%�	�t�=���A�5*;


total_loss�A

error_REVT?

learning_rate_1?a7�<�OI       6%�	���=���A�5*;


total_loss�h�@

error_Rx�O?

learning_rate_1?a7�o��I       6%�	5&�=���A�5*;


total_loss��@

error_R�VB?

learning_rate_1?a7��[�I       6%�	ek�=���A�5*;


total_lossd��@

error_R�7?

learning_rate_1?a7+��I       6%�	���=���A�5*;


total_loss���@

error_RL\?

learning_rate_1?a7�=CI       6%�	'��=���A�5*;


total_lossѰ�@

error_RTwS?

learning_rate_1?a7!	I       6%�	�C�=���A�5*;


total_loss�I�@

error_R�u9?

learning_rate_1?a7���.I       6%�	���=���A�5*;


total_loss@

error_R�UI?

learning_rate_1?a7�HF�I       6%�	��=���A�5*;


total_loss;�@

error_R}rK?

learning_rate_1?a7�_TYI       6%�	�=���A�6*;


total_loss��@

error_RC]H?

learning_rate_1?a7܏�I       6%�	�V�=���A�6*;


total_lossa	c@

error_R.�B?

learning_rate_1?a7<���I       6%�	���=���A�6*;


total_loss���@

error_R< Y?

learning_rate_1?a7�dI       6%�	[��=���A�6*;


total_loss��@

error_R�+U?

learning_rate_1?a7�9CI       6%�	��=���A�6*;


total_loss��@

error_R�#V?

learning_rate_1?a7�BmI       6%�	�b�=���A�6*;


total_loss�X�@

error_RC\I?

learning_rate_1?a7�N�VI       6%�	o��=���A�6*;


total_lossT7�@

error_R��\?

learning_rate_1?a7#�I       6%�	>��=���A�6*;


total_lossC��@

error_R	�T?

learning_rate_1?a7�+ͿI       6%�	�*�=���A�6*;


total_loss=6�@

error_R;>?

learning_rate_1?a7c�WI       6%�	8m�=���A�6*;


total_loss��@

error_Ri�X?

learning_rate_1?a7ThzI       6%�	��=���A�6*;


total_loss�t�@

error_R-�K?

learning_rate_1?a7c��I       6%�	���=���A�6*;


total_lossL�@

error_Rj�`?

learning_rate_1?a7N�~I       6%�	R5�=���A�6*;


total_loss:��@

error_R|hJ?

learning_rate_1?a7Rx�I       6%�	���=���A�6*;


total_loss�~�@

error_R��X?

learning_rate_1?a7�=�@I       6%�	���=���A�6*;


total_lossjД@

error_R��\?

learning_rate_1?a7�c�I       6%�	��=���A�6*;


total_lossO��@

error_R1�A?

learning_rate_1?a7A�5�I       6%�	�e�=���A�6*;


total_loss}��@

error_Ra#B?

learning_rate_1?a7�I       6%�	���=���A�6*;


total_lossam�@

error_R�P?

learning_rate_1?a7J4-I       6%�	S��=���A�6*;


total_loss�7�@

error_R<�@?

learning_rate_1?a7L|��I       6%�	+C�=���A�6*;


total_loss-i�@

error_Rq"E?

learning_rate_1?a7��JI       6%�	���=���A�6*;


total_loss�A�@

error_R��H?

learning_rate_1?a71LI       6%�	���=���A�6*;


total_loss�j@

error_RW�8?

learning_rate_1?a7Q
uI       6%�	�#�=���A�6*;


total_loss��q@

error_R��J?

learning_rate_1?a7�[kI       6%�	�e�=���A�6*;


total_loss9�@

error_R;[X?

learning_rate_1?a7�D��I       6%�	Ω�=���A�6*;


total_lossfQ�@

error_R��]?

learning_rate_1?a7���I       6%�	���=���A�6*;


total_loss�s�@

error_R��E?

learning_rate_1?a7��#SI       6%�	�2�=���A�6*;


total_lossN�@

error_RJdN?

learning_rate_1?a7��yI       6%�	%v�=���A�6*;


total_lossڃ�@

error_R�??

learning_rate_1?a7�6�zI       6%�	���=���A�6*;


total_loss��k@

error_R��P?

learning_rate_1?a7RI9@I       6%�	���=���A�6*;


total_loss�@

error_R�P?

learning_rate_1?a7�t��I       6%�	�;�=���A�6*;


total_loss,/A

error_Rx�K?

learning_rate_1?a7����I       6%�	~�=���A�6*;


total_loss�*�@

error_R\V?

learning_rate_1?a7>�BI       6%�	u��=���A�6*;


total_loss���@

error_RTQ?

learning_rate_1?a7o�BI       6%�	@�=���A�6*;


total_losshă@

error_R��D?

learning_rate_1?a7RLiI       6%�	�T�=���A�6*;


total_loss�,�@

error_R�R?

learning_rate_1?a7F�ˤI       6%�	С�=���A�6*;


total_loss���@

error_RL'P?

learning_rate_1?a7�w}I       6%�	���=���A�6*;


total_loss�ǭ@

error_R�S?

learning_rate_1?a7�M.OI       6%�	�7�=���A�6*;


total_loss�@

error_R��U?

learning_rate_1?a7�91�I       6%�	i~�=���A�6*;


total_loss�C�@

error_R��e?

learning_rate_1?a7M���I       6%�	���=���A�6*;


total_loss��@

error_R��6?

learning_rate_1?a7�,�6I       6%�	��=���A�6*;


total_lossw^�@

error_R�V?

learning_rate_1?a7�r�I       6%�	tN�=���A�6*;


total_loss�@

error_R�O?

learning_rate_1?a7d.��I       6%�	���=���A�6*;


total_loss6M�@

error_R!�n?

learning_rate_1?a7(�V�I       6%�	I��=���A�6*;


total_loss��|@

error_R�7?

learning_rate_1?a7dΔvI       6%�	Z�=���A�6*;


total_loss���@

error_RO�??

learning_rate_1?a7�2��I       6%�	�c�=���A�6*;


total_loss�A

error_R�ZX?

learning_rate_1?a7ԛ�I       6%�	]��=���A�6*;


total_lossC�A

error_R�'T?

learning_rate_1?a7h"�0I       6%�	J��=���A�6*;


total_loss\�@

error_R��U?

learning_rate_1?a7ֿE�I       6%�	;�=���A�6*;


total_loss#]�@

error_R��V?

learning_rate_1?a7�{��I       6%�	9��=���A�6*;


total_lossW�A

error_R�P?

learning_rate_1?a7^�Z�I       6%�	���=���A�6*;


total_loss���@

error_R��P?

learning_rate_1?a7�cFI       6%�	,�=���A�6*;


total_loss���@

error_R�+N?

learning_rate_1?a7�@)�I       6%�	�r�=���A�6*;


total_lossL��@

error_RW�V?

learning_rate_1?a7��CI       6%�	���=���A�6*;


total_loss�O�@

error_R��@?

learning_rate_1?a77���I       6%�	��=���A�6*;


total_loss�5�@

error_RW�??

learning_rate_1?a7��u-I       6%�	uE�=���A�6*;


total_loss��@

error_RS�??

learning_rate_1?a7�X�+I       6%�	���=���A�6*;


total_loss��@

error_R��`?

learning_rate_1?a7��`kI       6%�	��=���A�6*;


total_loss:�A

error_R��S?

learning_rate_1?a7�-��I       6%�	��=���A�6*;


total_loss�@

error_R(HS?

learning_rate_1?a7�^�KI       6%�	�X�=���A�6*;


total_lossi��@

error_R�wC?

learning_rate_1?a7a)2GI       6%�	���=���A�6*;


total_lossA

error_R(�Y?

learning_rate_1?a7j��I       6%�	@��=���A�6*;


total_loss���@

error_R�W?

learning_rate_1?a7�oYI       6%�	�-�=���A�6*;


total_loss`��@

error_R��P?

learning_rate_1?a7lwx�I       6%�	w�=���A�6*;


total_loss �@

error_R��\?

learning_rate_1?a7�?>I       6%�	���=���A�6*;


total_loss&��@

error_R8vT?

learning_rate_1?a7����I       6%�	��=���A�6*;


total_lossdV�@

error_R�R?

learning_rate_1?a7y=vMI       6%�	}I�=���A�6*;


total_loss}��@

error_R��0?

learning_rate_1?a7 �YI       6%�	j��=���A�6*;


total_loss)��@

error_R�K?

learning_rate_1?a7+��I       6%�	;��=���A�6*;


total_lossī�@

error_R��:?

learning_rate_1?a7���I       6%�	�=���A�6*;


total_lossE��@

error_R��[?

learning_rate_1?a7U���I       6%�	M_�=���A�6*;


total_loss�@

error_R�$Q?

learning_rate_1?a7N�KI       6%�	X��=���A�6*;


total_loss	q	A

error_R��X?

learning_rate_1?a7,�[�I       6%�	���=���A�6*;


total_loss��@

error_R�U?

learning_rate_1?a7�*�9I       6%�	�>�=���A�6*;


total_loss�o�@

error_R	�J?

learning_rate_1?a7΄�I       6%�	���=���A�6*;


total_loss���@

error_R�2[?

learning_rate_1?a7??
I       6%�	���=���A�6*;


total_loss3c@

error_R�MF?

learning_rate_1?a7��	I       6%�	r'�=���A�6*;


total_loss�j�@

error_R��L?

learning_rate_1?a7u%�I       6%�	v�=���A�6*;


total_loss��@

error_R�0>?

learning_rate_1?a7u �I       6%�	��=���A�6*;


total_loss4�@

error_R��I?

learning_rate_1?a7Z;.�I       6%�	��=���A�6*;


total_loss��@

error_R�_?

learning_rate_1?a7�I�I       6%�	$[�=���A�6*;


total_loss8B�@

error_R�KI?

learning_rate_1?a7�ҫI       6%�	���=���A�6*;


total_loss���@

error_R��Q?

learning_rate_1?a7��I       6%�	w��=���A�6*;


total_loss�Q�@

error_Rl�F?

learning_rate_1?a7;NiI       6%�		-�=���A�6*;


total_loss:t�@

error_R�G?

learning_rate_1?a7��)7I       6%�	�p�=���A�6*;


total_loss.y�@

error_R7�N?

learning_rate_1?a7~� I       6%�	���=���A�6*;


total_loss�t3A

error_RL�F?

learning_rate_1?a7%7HPI       6%�	���=���A�6*;


total_loss#�@

error_R��L?

learning_rate_1?a7ز�I       6%�	=�=���A�6*;


total_loss��@

error_R
�=?

learning_rate_1?a7찦I       6%�	U��=���A�6*;


total_loss��@

error_R�R?

learning_rate_1?a7���I       6%�	o��=���A�6*;


total_loss�P�@

error_R��J?

learning_rate_1?a7����I       6%�	�=���A�6*;


total_losst�@

error_R��b?

learning_rate_1?a7`6J�I       6%�	�a�=���A�6*;


total_loss@G�@

error_RSmN?

learning_rate_1?a7��q$I       6%�	���=���A�6*;


total_lossN�@

error_R�#c?

learning_rate_1?a7�@�2I       6%�	���=���A�6*;


total_loss(��@

error_R� P?

learning_rate_1?a7_�Y�I       6%�	b?�=���A�6*;


total_loss��@

error_R��S?

learning_rate_1?a7M�^(I       6%�	}��=���A�6*;


total_loss��@

error_R_�D?

learning_rate_1?a7�z�xI       6%�	��=���A�6*;


total_loss��@

error_R�=M?

learning_rate_1?a7}�,I       6%�	<�=���A�6*;


total_loss6��@

error_R��Z?

learning_rate_1?a7a�;I       6%�	�~�=���A�6*;


total_loss�٭@

error_R��N?

learning_rate_1?a7sl;�I       6%�	���=���A�6*;


total_loss�Г@

error_RN>?

learning_rate_1?a7��
MI       6%�	�=���A�6*;


total_loss���@

error_R�A?

learning_rate_1?a7�>�I       6%�	*O�=���A�6*;


total_lossDb�@

error_R��d?

learning_rate_1?a7'?�I       6%�	��=���A�6*;


total_lossS��@

error_RŌT?

learning_rate_1?a7��ՄI       6%�	��=���A�6*;


total_loss��@

error_R��K?

learning_rate_1?a7�+��I       6%�	#�=���A�6*;


total_lossD(�@

error_R[�R?

learning_rate_1?a7K�&�I       6%�	`f�=���A�6*;


total_loss���@

error_Ra[?

learning_rate_1?a7�qI       6%�	#��=���A�6*;


total_lossj�@

error_R�M?

learning_rate_1?a7�ҟ�I       6%�	���=���A�6*;


total_lossKmA

error_R�mF?

learning_rate_1?a7ӯ�I       6%�	W4�=���A�6*;


total_lossV�@

error_R7e?

learning_rate_1?a7x&�uI       6%�	Wx�=���A�6*;


total_loss`�@

error_RIZ?

learning_rate_1?a7}=D�I       6%�	��=���A�6*;


total_lossn�@

error_R��G?

learning_rate_1?a7���RI       6%�	��=���A�6*;


total_loss��@

error_RZ�;?

learning_rate_1?a7;o��I       6%�	"C�=���A�6*;


total_loss{�}@

error_R�\?

learning_rate_1?a7��I       6%�	Z��=���A�6*;


total_loss� �@

error_Rw�U?

learning_rate_1?a7�<2�I       6%�	���=���A�6*;


total_loss��@

error_R*`?

learning_rate_1?a7;+8I       6%�	��=���A�6*;


total_loss2�@

error_R�P?

learning_rate_1?a7��I       6%�	�R�=���A�6*;


total_loss|�@

error_R�.L?

learning_rate_1?a7Q���I       6%�	���=���A�6*;


total_loss���@

error_R�*E?

learning_rate_1?a75*��I       6%�	���=���A�6*;


total_loss��A

error_R.Z?

learning_rate_1?a7�r$I       6%�	g&�=���A�6*;


total_lossF�@

error_R�C?

learning_rate_1?a7Pk_LI       6%�	�p�=���A�6*;


total_loss�^�@

error_R�H?

learning_rate_1?a7��`eI       6%�	��=���A�6*;


total_lossi��@

error_RݪG?

learning_rate_1?a73?�I       6%�	��=���A�6*;


total_lossJ%�@

error_RO�C?

learning_rate_1?a7��II       6%�	=I�=���A�6*;


total_loss2c�@

error_RTZT?

learning_rate_1?a7=�w�I       6%�	O��=���A�6*;


total_lossOxA

error_R��S?

learning_rate_1?a7�Sl�I       6%�	p��=���A�6*;


total_loss.��@

error_RT�V?

learning_rate_1?a7�t�I       6%�	�$�=���A�6*;


total_loss���@

error_R��c?

learning_rate_1?a7���I       6%�	�w�=���A�6*;


total_loss�y�@

error_R��N?

learning_rate_1?a7��j�I       6%�	���=���A�7*;


total_loss���@

error_RڶB?

learning_rate_1?a7=a�aI       6%�	 >���A�7*;


total_loss��@

error_RC�F?

learning_rate_1?a78g`�I       6%�	�b >���A�7*;


total_lossiͫ@

error_R�&U?

learning_rate_1?a7HR^I       6%�	J� >���A�7*;


total_loss�r�@

error_Rڝ\?

learning_rate_1?a7��*I       6%�	�>���A�7*;


total_loss-�A

error_R�oS?

learning_rate_1?a7��dI       6%�	�f>���A�7*;


total_loss��@

error_R��Y?

learning_rate_1?a7�S�YI       6%�	��>���A�7*;


total_loss�'A

error_R �`?

learning_rate_1?a7�-;�I       6%�	S(>���A�7*;


total_lossm�A

error_R =I?

learning_rate_1?a7]E)I       6%�	�y>���A�7*;


total_lossHڷ@

error_R�4?

learning_rate_1?a7�Ź�I       6%�	��>���A�7*;


total_loss��@

error_R�E?

learning_rate_1?a7'!G�I       6%�	->���A�7*;


total_loss�@

error_R�J?

learning_rate_1?a7z�2�I       6%�	\u>���A�7*;


total_loss��@

error_R�??

learning_rate_1?a7m�N�I       6%�	��>���A�7*;


total_loss$�@

error_R=�\?

learning_rate_1?a7#Ҙ7I       6%�	�>���A�7*;


total_loss���@

error_Rj�T?

learning_rate_1?a7��HbI       6%�	�T>���A�7*;


total_loss�D�@

error_R1Z@?

learning_rate_1?a7Up��I       6%�	�>���A�7*;


total_loss��@

error_R�>O?

learning_rate_1?a7E���I       6%�	=	>���A�7*;


total_loss戔@

error_R;~9?

learning_rate_1?a7�O I       6%�		V>���A�7*;


total_loss�r�@

error_Rd�S?

learning_rate_1?a7��(I       6%�	i�>���A�7*;


total_loss���@

error_R)K?

learning_rate_1?a7~)\�I       6%�	5�>���A�7*;


total_loss,�U@

error_R��P?

learning_rate_1?a7����I       6%�	�&>���A�7*;


total_loss�|�@

error_Re�O?

learning_rate_1?a7LK��I       6%�	�v>���A�7*;


total_loss}�@

error_Rx�V?

learning_rate_1?a7K[�I       6%�		�>���A�7*;


total_loss���@

error_R��J?

learning_rate_1?a7��z	I       6%�	=>���A�7*;


total_loss�@

error_Rj�W?

learning_rate_1?a7v�\�I       6%�	�b>���A�7*;


total_loss ~�@

error_R�A??

learning_rate_1?a7_�I       6%�	I�>���A�7*;


total_loss���@

error_R�Z?

learning_rate_1?a7��jpI       6%�	��>���A�7*;


total_loss�x�@

error_R�\X?

learning_rate_1?a7$~��I       6%�	w1>���A�7*;


total_loss��@

error_R�X?

learning_rate_1?a7Nh�.I       6%�	�t>���A�7*;


total_loss,v�@

error_R7`?

learning_rate_1?a7=�.UI       6%�	J�>���A�7*;


total_loss���@

error_R3=?

learning_rate_1?a7���vI       6%�	c	>���A�7*;


total_lossi��@

error_R�5?

learning_rate_1?a7�8AuI       6%�	�Y	>���A�7*;


total_loss1f�@

error_R"_?

learning_rate_1?a7��	I       6%�	�	>���A�7*;


total_loss�Co@

error_R)�V?

learning_rate_1?a7�й�I       6%�	v�	>���A�7*;


total_loss���@

error_R�a?

learning_rate_1?a7f���I       6%�	�1
>���A�7*;


total_loss��@

error_RsI?

learning_rate_1?a7�,KXI       6%�	iw
>���A�7*;


total_loss��@

error_R-�I?

learning_rate_1?a7Gʘ�I       6%�	K�
>���A�7*;


total_loss���@

error_R�	H?

learning_rate_1?a7kF^I       6%�	�>���A�7*;


total_loss��@

error_R�3;?

learning_rate_1?a7q���I       6%�	�K>���A�7*;


total_loss̆�@

error_R3�u?

learning_rate_1?a7���I       6%�	Ē>���A�7*;


total_loss�s�@

error_R�*C?

learning_rate_1?a7`���I       6%�	(�>���A�7*;


total_lossv��@

error_R�GE?

learning_rate_1?a7��I       6%�	�>���A�7*;


total_loss{�A

error_R+P?

learning_rate_1?a7�AMI       6%�	;c>���A�7*;


total_losstgA

error_R�*L?

learning_rate_1?a7�n�I       6%�	��>���A�7*;


total_loss���@

error_R[U?

learning_rate_1?a7�i�I       6%�	i�>���A�7*;


total_loss���@

error_R=?

learning_rate_1?a70D�I       6%�	r<>���A�7*;


total_loss�*�@

error_R��M?

learning_rate_1?a72Ȏ�I       6%�	�~>���A�7*;


total_lossƦ�@

error_R�7f?

learning_rate_1?a7=;��I       6%�	n�>���A�7*;


total_lossT�@

error_Rk@?

learning_rate_1?a7�a*9I       6%�	>���A�7*;


total_lossQա@

error_R�;S?

learning_rate_1?a7�9�QI       6%�	�q>���A�7*;


total_lossV�@

error_R�!M?

learning_rate_1?a7=��I       6%�	b�>���A�7*;


total_loss"� A

error_R�M?

learning_rate_1?a7p�ĞI       6%�	�>���A�7*;


total_lossA@�@

error_R�D?

learning_rate_1?a7���I       6%�	�D>���A�7*;


total_loss���@

error_R��S?

learning_rate_1?a7����I       6%�	}�>���A�7*;


total_loss�:�@

error_R��L?

learning_rate_1?a7]xXI       6%�	|�>���A�7*;


total_loss3�@

error_R�@?

learning_rate_1?a7�$��I       6%�	�>���A�7*;


total_lossx��@

error_Rd�O?

learning_rate_1?a7�^�I       6%�	4[>���A�7*;


total_loss�/�@

error_R��R?

learning_rate_1?a7���mI       6%�	��>���A�7*;


total_loss���@

error_R�L?

learning_rate_1?a7�>�tI       6%�	�>���A�7*;


total_loss�W�@

error_Rl�N?

learning_rate_1?a7�'�I       6%�	;7>���A�7*;


total_loss��@

error_Rn�I?

learning_rate_1?a7;���I       6%�	�~>���A�7*;


total_loss���@

error_Rv6U?

learning_rate_1?a7A��I       6%�	��>���A�7*;


total_loss�ȸ@

error_R�yV?

learning_rate_1?a7t�7I       6%�	y
>���A�7*;


total_loss1r�@

error_Rn�P?

learning_rate_1?a7	~�I       6%�	�O>���A�7*;


total_losst<�@

error_R$N[?

learning_rate_1?a7�!��I       6%�	N�>���A�7*;


total_loss�C�@

error_Rl�L?

learning_rate_1?a7�<ŁI       6%�	��>���A�7*;


total_lossM��@

error_R_NW?

learning_rate_1?a7k|�|I       6%�	.>���A�7*;


total_loss��@

error_R��O?

learning_rate_1?a7��A�I       6%�	�b>���A�7*;


total_lossK��@

error_R�O?

learning_rate_1?a7�9��I       6%�	Ǧ>���A�7*;


total_loss���@

error_R$�^?

learning_rate_1?a7{���I       6%�	�>���A�7*;


total_lossF@�@

error_Ri�[?

learning_rate_1?a7S��I       6%�	�.>���A�7*;


total_lossN�@

error_R�%O?

learning_rate_1?a7-4�wI       6%�	�s>���A�7*;


total_loss�:�@

error_RNcL?

learning_rate_1?a7X2��I       6%�	ŵ>���A�7*;


total_loss��@

error_R$OQ?

learning_rate_1?a7��yI       6%�	��>���A�7*;


total_loss��@

error_REC?

learning_rate_1?a7@EJ�I       6%�	@>���A�7*;


total_loss}�A

error_R�U?

learning_rate_1?a7�9�I       6%�	>�>���A�7*;


total_loss�Ѵ@

error_R��Z?

learning_rate_1?a71�1�I       6%�	��>���A�7*;


total_loss�@

error_R��W?

learning_rate_1?a795UI       6%�	�>���A�7*;


total_lossf/�@

error_R�kX?

learning_rate_1?a7���I       6%�	D]>���A�7*;


total_loss�\�@

error_RhI?

learning_rate_1?a7�� [I       6%�	6�>���A�7*;


total_lossIX�@

error_R�L?

learning_rate_1?a7����I       6%�	L>���A�7*;


total_loss�<�@

error_Rá=?

learning_rate_1?a7��r3I       6%�	�V>���A�7*;


total_loss�@

error_RݾV?

learning_rate_1?a7� I       6%�	5�>���A�7*;


total_loss(u}@

error_R�RO?

learning_rate_1?a7����I       6%�	E�>���A�7*;


total_loss�^�@

error_R�j?

learning_rate_1?a7��r�I       6%�	$>���A�7*;


total_lossʹ�@

error_R��C?

learning_rate_1?a7���II       6%�	�h>���A�7*;


total_loss	�@

error_RUE?

learning_rate_1?a7���I       6%�	��>���A�7*;


total_lossR�@

error_R4H?

learning_rate_1?a7�(r0I       6%�	7�>���A�7*;


total_loss�@

error_Rx�S?

learning_rate_1?a7�ZDI       6%�	�B>���A�7*;


total_loss��@

error_Rj�Q?

learning_rate_1?a72'�iI       6%�	��>���A�7*;


total_loss�&�@

error_R!DS?

learning_rate_1?a7�=��I       6%�	��>���A�7*;


total_loss�(A

error_RR�[?

learning_rate_1?a7��R�I       6%�	5$>���A�7*;


total_loss�΅@

error_R6G?

learning_rate_1?a7N<h`I       6%�	l>���A�7*;


total_lossyA

error_R19b?

learning_rate_1?a7�I       6%�	��>���A�7*;


total_loss:ߙ@

error_R| J?

learning_rate_1?a7���I       6%�	 >���A�7*;


total_loss<��@

error_R�D?

learning_rate_1?a7f��I       6%�	�F>���A�7*;


total_loss�ܙ@

error_Rt�W?

learning_rate_1?a7�2,UI       6%�	?�>���A�7*;


total_losse��@

error_ROU?

learning_rate_1?a7��+�I       6%�	#�>���A�7*;


total_loss4�@

error_R��1?

learning_rate_1?a7��(�I       6%�	m>���A�7*;


total_loss@0�@

error_R@AO?

learning_rate_1?a7Y�B�I       6%�	;S>���A�7*;


total_lossܣ�@

error_R��J?

learning_rate_1?a7��3�I       6%�	`�>���A�7*;


total_loss_��@

error_RV}V?

learning_rate_1?a7 ���I       6%�	��>���A�7*;


total_loss�/�@

error_RM�J?

learning_rate_1?a7�S�I       6%�	G1>���A�7*;


total_loss��@

error_R4V?

learning_rate_1?a7���I       6%�	�s>���A�7*;


total_lossi�@

error_R�Gb?

learning_rate_1?a7B�F�I       6%�	k�>���A�7*;


total_loss[��@

error_RH??

learning_rate_1?a7c�~�I       6%�	�>���A�7*;


total_loss��@

error_Rd�P?

learning_rate_1?a7LQ5�I       6%�	�7>���A�7*;


total_lossi�@

error_R��j?

learning_rate_1?a7/�gI       6%�	x|>���A�7*;


total_lossfu�@

error_R��K?

learning_rate_1?a7 G�7I       6%�	��>���A�7*;


total_loss{�@

error_R3xM?

learning_rate_1?a7Ȃn�I       6%�	w	>���A�7*;


total_loss�.�@

error_R`UX?

learning_rate_1?a7:`�I       6%�	�U>���A�7*;


total_loss��@

error_Rv�L?

learning_rate_1?a7X�!I       6%�	��>���A�7*;


total_loss-��@

error_R̣]?

learning_rate_1?a7��3\I       6%�	�>���A�7*;


total_loss�-�@

error_Rd9?

learning_rate_1?a7���sI       6%�	C& >���A�7*;


total_loss|Հ@

error_Ri�T?

learning_rate_1?a7>���I       6%�	 t >���A�7*;


total_loss1��@

error_R!
K?

learning_rate_1?a70�KI       6%�	v� >���A�7*;


total_loss�n�@

error_R�Ue?

learning_rate_1?a7z�QI       6%�	G� >���A�7*;


total_loss��@

error_R��K?

learning_rate_1?a7.�jaI       6%�	�9!>���A�7*;


total_lossQ-#A

error_Rs�3?

learning_rate_1?a7��h�I       6%�	{!>���A�7*;


total_lossT��@

error_Rn�N?

learning_rate_1?a7EP�I       6%�	��!>���A�7*;


total_loss�y�@

error_R�WH?

learning_rate_1?a7� �I       6%�	F">���A�7*;


total_lossm��@

error_R��E?

learning_rate_1?a7���I       6%�	�D">���A�7*;


total_loss�Y�@

error_R��Y?

learning_rate_1?a7�I       6%�	�">���A�7*;


total_lossI�@

error_R�|R?

learning_rate_1?a7��W|I       6%�	y�">���A�7*;


total_loss��@

error_R$ ]?

learning_rate_1?a7El��I       6%�	`#>���A�7*;


total_loss�nA

error_R&FP?

learning_rate_1?a7�:I       6%�	�V#>���A�7*;


total_loss �A

error_RAnF?

learning_rate_1?a7��5I       6%�	c�#>���A�7*;


total_loss�_�@

error_R�aT?

learning_rate_1?a7���oI       6%�	��#>���A�7*;


total_loss��@

error_R�L?

learning_rate_1?a7_
��I       6%�	�&$>���A�8*;


total_loss�C�@

error_R_ U?

learning_rate_1?a7��D�I       6%�	�n$>���A�8*;


total_loss6��@

error_R,2?

learning_rate_1?a7?q�I       6%�	�$>���A�8*;


total_loss4��@

error_R�N?

learning_rate_1?a7`rI       6%�	��$>���A�8*;


total_loss[��@

error_R��)?

learning_rate_1?a7��:�I       6%�	}I%>���A�8*;


total_loss�#�@

error_Ra4H?

learning_rate_1?a7��I       6%�	��%>���A�8*;


total_lossA��@

error_R�R?

learning_rate_1?a7��N�I       6%�	��%>���A�8*;


total_loss�Y�@

error_R�=X?

learning_rate_1?a7�!�I       6%�	�"&>���A�8*;


total_loss�b�@

error_RR(?

learning_rate_1?a7����I       6%�	�l&>���A�8*;


total_losss�@

error_R�@?

learning_rate_1?a7M���I       6%�	��&>���A�8*;


total_loss:�@

error_R�V?

learning_rate_1?a7�	:I       6%�	~'>���A�8*;


total_loss?�@

error_R|�C?

learning_rate_1?a7����I       6%�	�_'>���A�8*;


total_loss.��@

error_R�sH?

learning_rate_1?a7F���I       6%�	��'>���A�8*;


total_lossD��@

error_R��[?

learning_rate_1?a7�oI       6%�	��'>���A�8*;


total_loss!T�@

error_RѸV?

learning_rate_1?a7�X]I       6%�	�4(>���A�8*;


total_loss�A

error_R\�P?

learning_rate_1?a7b�6I       6%�	8x(>���A�8*;


total_loss��.A

error_R
�U?

learning_rate_1?a7��I       6%�	ľ(>���A�8*;


total_loss V�@

error_R�!J?

learning_rate_1?a7Ƌ	jI       6%�	�)>���A�8*;


total_lossl8�@

error_RܸW?

learning_rate_1?a7�I       6%�	�F)>���A�8*;


total_lossl��@

error_R�+Q?

learning_rate_1?a7�{0LI       6%�	9�)>���A�8*;


total_lossxZ�@

error_R�0E?

learning_rate_1?a7���I       6%�	��)>���A�8*;


total_lossS��@

error_R�}\?

learning_rate_1?a7��GiI       6%�	q*>���A�8*;


total_loss�S�@

error_R}�O?

learning_rate_1?a7y�5HI       6%�	mV*>���A�8*;


total_lossd'�@

error_Rw�B?

learning_rate_1?a7���*I       6%�	��*>���A�8*;


total_loss�jd@

error_R҃T?

learning_rate_1?a7�fU�I       6%�	F�*>���A�8*;


total_lossû@

error_R�P?

learning_rate_1?a7���I       6%�	f5+>���A�8*;


total_lossL��@

error_R�dT?

learning_rate_1?a7�y�I       6%�	�x+>���A�8*;


total_lossZZ�@

error_R�4?

learning_rate_1?a7�1�I       6%�	��+>���A�8*;


total_loss���@

error_R�/C?

learning_rate_1?a7�[�I       6%�	Y,>���A�8*;


total_loss�
q@

error_R[�R?

learning_rate_1?a7���I       6%�	xN,>���A�8*;


total_loss��@

error_R�IO?

learning_rate_1?a7ʧ��I       6%�	��,>���A�8*;


total_loss���@

error_RstX?

learning_rate_1?a7�Wy�I       6%�	�,>���A�8*;


total_loss���@

error_R��>?

learning_rate_1?a7[FI       6%�	�#->���A�8*;


total_loss2ϥ@

error_RM�E?

learning_rate_1?a7%�I       6%�	j->���A�8*;


total_loss;��@

error_R�6?

learning_rate_1?a7%�|�I       6%�	$�->���A�8*;


total_loss`��@

error_R�~L?

learning_rate_1?a7?�gI       6%�	F�->���A�8*;


total_loss5]A

error_Rj�N?

learning_rate_1?a7`��I       6%�	\.>���A�8*;


total_loss���@

error_R="]?

learning_rate_1?a7�"I       6%�	-�.>���A�8*;


total_loss34h@

error_R� >?

learning_rate_1?a7,$�8I       6%�	}�.>���A�8*;


total_loss�Q�@

error_R_$^?

learning_rate_1?a7	%�mI       6%�	V0/>���A�8*;


total_lossۣ@

error_R)�`?

learning_rate_1?a7Y�X�I       6%�	�/>���A�8*;


total_loss�[�@

error_R�UE?

learning_rate_1?a7�g��I       6%�	6�/>���A�8*;


total_loss�3�@

error_RH�Z?

learning_rate_1?a7�4��I       6%�	t0>���A�8*;


total_loss��@

error_R\
<?

learning_rate_1?a7զɯI       6%�	�a0>���A�8*;


total_loss��@

error_R �[?

learning_rate_1?a7m�I       6%�	��0>���A�8*;


total_lossÛ�@

error_R�3I?

learning_rate_1?a7Qx`I       6%�	�0>���A�8*;


total_lossb	A

error_R��E?

learning_rate_1?a7`[!�I       6%�	�/1>���A�8*;


total_loss.��@

error_R�I?

learning_rate_1?a7�.I�I       6%�	yy1>���A�8*;


total_lossڠA

error_R,mK?

learning_rate_1?a7V���I       6%�	��1>���A�8*;


total_loss�K�@

error_R/j?

learning_rate_1?a7���pI       6%�	�2>���A�8*;


total_loss�XA

error_Rl�P?

learning_rate_1?a7uI�&I       6%�	\2>���A�8*;


total_loss�q�@

error_R��V?

learning_rate_1?a7h�Q"I       6%�	�2>���A�8*;


total_lossD:�@

error_R�Q?

learning_rate_1?a7VI       6%�	��2>���A�8*;


total_lossI��@

error_R}I[?

learning_rate_1?a7 ��lI       6%�	8.3>���A�8*;


total_loss f@

error_R��=?

learning_rate_1?a7�xB�I       6%�	�s3>���A�8*;


total_lossW��@

error_R%J9?

learning_rate_1?a7녇I       6%�	��3>���A�8*;


total_loss�z�@

error_R��H?

learning_rate_1?a7o�G�I       6%�	T�3>���A�8*;


total_loss��@

error_R��J?

learning_rate_1?a7Q�qI       6%�	�D4>���A�8*;


total_loss�X�@

error_RC(B?

learning_rate_1?a7��|4I       6%�	;�4>���A�8*;


total_loss��@

error_R�aM?

learning_rate_1?a7�%��I       6%�	B�4>���A�8*;


total_loss���@

error_R��E?

learning_rate_1?a7#1�PI       6%�	r5>���A�8*;


total_loss���@

error_R�"[?

learning_rate_1?a7�!��I       6%�	pV5>���A�8*;


total_loss�߳@

error_R�4N?

learning_rate_1?a7�I       6%�	�5>���A�8*;


total_lossj۹@

error_R��F?

learning_rate_1?a7$:IxI       6%�	&�5>���A�8*;


total_loss���@

error_RVhY?

learning_rate_1?a7P痌I       6%�	k%6>���A�8*;


total_loss��@

error_Rz8J?

learning_rate_1?a7�I       6%�	s6>���A�8*;


total_loss�@�@

error_R4d?

learning_rate_1?a7��I       6%�	a�6>���A�8*;


total_loss$�@

error_Rq@N?

learning_rate_1?a7�C�CI       6%�	� 7>���A�8*;


total_loss�w�@

error_RRhK?

learning_rate_1?a7e���I       6%�	ze7>���A�8*;


total_loss���@

error_R�W?

learning_rate_1?a7�X�3I       6%�	{�7>���A�8*;


total_lossl��@

error_R�J?

learning_rate_1?a7�J��I       6%�	��7>���A�8*;


total_loss��@

error_R6�G?

learning_rate_1?a7�ݡI       6%�	�58>���A�8*;


total_loss��@

error_RR[X?

learning_rate_1?a7>�`AI       6%�	Ex8>���A�8*;


total_loss��@

error_R��R?

learning_rate_1?a7>�l�I       6%�	�8>���A�8*;


total_loss���@

error_R&�P?

learning_rate_1?a7]�-�I       6%�	c9>���A�8*;


total_lossq�A

error_R�S?

learning_rate_1?a76���I       6%�	�C9>���A�8*;


total_loss,�@

error_R�iC?

learning_rate_1?a7���I       6%�	0�9>���A�8*;


total_loss�A

error_R��O?

learning_rate_1?a7R���I       6%�	��9>���A�8*;


total_loss��@

error_Rv�V?

learning_rate_1?a7�m�SI       6%�	i:>���A�8*;


total_lossυ�@

error_R�E?

learning_rate_1?a7L#��I       6%�	�Q:>���A�8*;


total_loss G�@

error_R�Z?

learning_rate_1?a7E��QI       6%�	�:>���A�8*;


total_loss�b�@

error_R��L?

learning_rate_1?a7�љ�I       6%�	��:>���A�8*;


total_lossD��@

error_R�D?

learning_rate_1?a7��I       6%�	�;>���A�8*;


total_loss�Y�@

error_Rj�Q?

learning_rate_1?a7n�P�I       6%�	|b;>���A�8*;


total_losse.�@

error_R�I?

learning_rate_1?a7g��I       6%�	��;>���A�8*;


total_lossx9�@

error_R�X?

learning_rate_1?a7G��tI       6%�	��;>���A�8*;


total_lossP�@

error_RϋT?

learning_rate_1?a7{���I       6%�	�9<>���A�8*;


total_losse�@

error_Rd6\?

learning_rate_1?a7'���I       6%�	�}<>���A�8*;


total_loss㢷@

error_RH�Q?

learning_rate_1?a7(ÌdI       6%�	��<>���A�8*;


total_loss�t�@

error_R��??

learning_rate_1?a7C�L�I       6%�	-=>���A�8*;


total_loss��|@

error_R�W?

learning_rate_1?a7c���I       6%�	?S=>���A�8*;


total_loss�h�@

error_R6BW?

learning_rate_1?a7�nOI       6%�	F�=>���A�8*;


total_lossE,A

error_R�AM?

learning_rate_1?a7'fI       6%�	��=>���A�8*;


total_loss��@

error_R��;?

learning_rate_1?a7lqI       6%�	 >>���A�8*;


total_loss�k�@

error_R�{^?

learning_rate_1?a7}�mI       6%�	�j>>���A�8*;


total_lossW=�@

error_R_iF?

learning_rate_1?a7P�I       6%�	�>>���A�8*;


total_loss�ȍ@

error_R��O?

learning_rate_1?a7���I       6%�	?�>>���A�8*;


total_loss�@

error_R�W?

learning_rate_1?a7+��I       6%�	<?>���A�8*;


total_lossہ�@

error_R�I?

learning_rate_1?a7�h7;I       6%�	�~?>���A�8*;


total_loss:Y�@

error_R=�U?

learning_rate_1?a7����I       6%�	��?>���A�8*;


total_lossnY�@

error_R��??

learning_rate_1?a7�I��I       6%�	L@>���A�8*;


total_lossa��@

error_RW?

learning_rate_1?a7���I       6%�	3O@>���A�8*;


total_loss3(�@

error_R��3?

learning_rate_1?a7�g�)I       6%�	��@>���A�8*;


total_loss���@

error_REf6?

learning_rate_1?a7�cI       6%�	2�@>���A�8*;


total_loss� �@

error_Rv[?

learning_rate_1?a7-RyI       6%�	�A>���A�8*;


total_loss��@

error_R�<U?

learning_rate_1?a7���jI       6%�	\A>���A�8*;


total_losse��@

error_R]�F?

learning_rate_1?a7_�uI       6%�	�A>���A�8*;


total_loss7�@

error_R�G?

learning_rate_1?a7�%�I       6%�	x�A>���A�8*;


total_loss8I�@

error_R�M?

learning_rate_1?a7���I       6%�	11B>���A�8*;


total_loss���@

error_R��L?

learning_rate_1?a7�5�I       6%�	uB>���A�8*;


total_loss�{�@

error_R_W?

learning_rate_1?a7�7��I       6%�	3�B>���A�8*;


total_loss���@

error_R�pE?

learning_rate_1?a7xidI       6%�	��B>���A�8*;


total_loss�q�@

error_R��N?

learning_rate_1?a7V@�I       6%�	�AC>���A�8*;


total_loss��@

error_R�L?

learning_rate_1?a7ə��I       6%�	��C>���A�8*;


total_loss���@

error_Rq�N?

learning_rate_1?a7:$I       6%�	p�C>���A�8*;


total_loss:��@

error_R��^?

learning_rate_1?a72ޏiI       6%�	hD>���A�8*;


total_loss���@

error_R@P?

learning_rate_1?a7�&��I       6%�	�ZD>���A�8*;


total_loss���@

error_R��O?

learning_rate_1?a7�N�I       6%�	��D>���A�8*;


total_loss ��@

error_RW�R?

learning_rate_1?a7�Y_I       6%�	+�D>���A�8*;


total_loss��@

error_R��U?

learning_rate_1?a7T|/tI       6%�	H2E>���A�8*;


total_loss��@

error_R��X?

learning_rate_1?a7�HfSI       6%�	�zE>���A�8*;


total_loss���@

error_R�W?

learning_rate_1?a7XT�rI       6%�	d�E>���A�8*;


total_loss9�@

error_RIT?

learning_rate_1?a7����I       6%�	��E>���A�8*;


total_lossw��@

error_R֯@?

learning_rate_1?a7��<I       6%�	5CF>���A�8*;


total_loss�A�@

error_R�)I?

learning_rate_1?a7�g
�I       6%�	2�F>���A�8*;


total_loss��@

error_R=+T?

learning_rate_1?a7p�[I       6%�	��F>���A�8*;


total_loss6s�@

error_R��K?

learning_rate_1?a72��@I       6%�	�.G>���A�8*;


total_lossC�@

error_Ro�X?

learning_rate_1?a7~���I       6%�	2sG>���A�8*;


total_lossJ2�@

error_R�OZ?

learning_rate_1?a7���I       6%�	Y�G>���A�9*;


total_loss�A

error_Rn�D?

learning_rate_1?a7/
�3I       6%�	�H>���A�9*;


total_loss!(�@

error_R�zV?

learning_rate_1?a7����I       6%�	MKH>���A�9*;


total_loss̴@

error_R�~n?

learning_rate_1?a7 Zr�I       6%�	ߍH>���A�9*;


total_loss�Ƣ@

error_R��T?

learning_rate_1?a7�O�fI       6%�	�H>���A�9*;


total_loss_��@

error_R��8?

learning_rate_1?a7_d�I       6%�	�I>���A�9*;


total_loss�zA

error_R��J?

learning_rate_1?a7�D�I       6%�	VI>���A�9*;


total_loss���@

error_R=�_?

learning_rate_1?a7��[�I       6%�	f�I>���A�9*;


total_loss�Y�@

error_R�J\?

learning_rate_1?a7g�5I       6%�	�I>���A�9*;


total_loss���@

error_R��N?

learning_rate_1?a7�W%`I       6%�	�$J>���A�9*;


total_losss�@

error_R&55?

learning_rate_1?a7����I       6%�	`gJ>���A�9*;


total_loss�ʤ@

error_R&�F?

learning_rate_1?a7t�J�I       6%�	�J>���A�9*;


total_loss�r�@

error_RR�U?

learning_rate_1?a7N�J�I       6%�	6�J>���A�9*;


total_loss�:�@

error_RԉX?

learning_rate_1?a7���I       6%�	�2K>���A�9*;


total_loss��@

error_R�0B?

learning_rate_1?a7�0{LI       6%�	)vK>���A�9*;


total_lossNA

error_R��U?

learning_rate_1?a7����I       6%�	�K>���A�9*;


total_loss�Q�@

error_RJ6H?

learning_rate_1?a7`4��I       6%�	~�K>���A�9*;


total_loss��@

error_R,fQ?

learning_rate_1?a7c�cSI       6%�	AL>���A�9*;


total_loss���@

error_RF�G?

learning_rate_1?a7*���I       6%�	�L>���A�9*;


total_loss�{�@

error_R"Z?

learning_rate_1?a7r��(I       6%�	$�L>���A�9*;


total_loss�̼@

error_ROH?

learning_rate_1?a7�p�I       6%�	�M>���A�9*;


total_loss��@

error_RQ�Y?

learning_rate_1?a7B��wI       6%�	�cM>���A�9*;


total_loss���@

error_R��<?

learning_rate_1?a7?'I       6%�	R�M>���A�9*;


total_loss��@

error_R�JW?

learning_rate_1?a7[n�wI       6%�	e�M>���A�9*;


total_lossW��@

error_R}�l?

learning_rate_1?a7u�X�I       6%�	�RN>���A�9*;


total_loss���@

error_Rx�P?

learning_rate_1?a7��
I       6%�	�N>���A�9*;


total_loss�6�@

error_R�OQ?

learning_rate_1?a7�m̊I       6%�	[�N>���A�9*;


total_lossrW�@

error_R;dG?

learning_rate_1?a7�VE�I       6%�	�:O>���A�9*;


total_loss �@

error_R�XU?

learning_rate_1?a75> I       6%�	��O>���A�9*;


total_loss
M�@

error_RO\?

learning_rate_1?a7�5C�I       6%�	E�O>���A�9*;


total_loss���@

error_R�T?

learning_rate_1?a7�N&I       6%�	�*P>���A�9*;


total_lossi,�@

error_R�8B?

learning_rate_1?a7�h5,I       6%�	{rP>���A�9*;


total_loss�y�@

error_R��K?

learning_rate_1?a7���	I       6%�	�P>���A�9*;


total_loss� �@

error_Ra�N?

learning_rate_1?a7���I       6%�	&�P>���A�9*;


total_loss��@

error_R��M?

learning_rate_1?a7���rI       6%�	CQ>���A�9*;


total_loss�<,A

error_RU?

learning_rate_1?a7�nI       6%�	]�Q>���A�9*;


total_loss�*�@

error_R�5V?

learning_rate_1?a7ǹ�vI       6%�	0�Q>���A�9*;


total_lossej�@

error_R��P?

learning_rate_1?a7_�m5I       6%�	�R>���A�9*;


total_loss�a�@

error_R)T?

learning_rate_1?a7-T>I       6%�	�aR>���A�9*;


total_loss���@

error_R��a?

learning_rate_1?a7P�I       6%�	!�R>���A�9*;


total_losscR�@

error_RX%E?

learning_rate_1?a7g�%I       6%�	��R>���A�9*;


total_loss�@

error_R%%H?

learning_rate_1?a7n�
I       6%�	:<S>���A�9*;


total_loss��@

error_R!�L?

learning_rate_1?a7b~��I       6%�	BS>���A�9*;


total_lossX�@

error_Ri;R?

learning_rate_1?a7RD�I       6%�	P�S>���A�9*;


total_loss��@

error_RѕS?

learning_rate_1?a7��q�I       6%�	�T>���A�9*;


total_lossd�@

error_Rw�F?

learning_rate_1?a7�c~'I       6%�	NT>���A�9*;


total_loss#ؼ@

error_R�W?

learning_rate_1?a7D�;�I       6%�	�T>���A�9*;


total_loss$��@

error_R�NO?

learning_rate_1?a7�N ZI       6%�	�T>���A�9*;


total_loss�1m@

error_RJ?

learning_rate_1?a7��6�I       6%�	nU>���A�9*;


total_lossZm�@

error_R�*@?

learning_rate_1?a7T��I       6%�	7\U>���A�9*;


total_loss���@

error_R�RP?

learning_rate_1?a7:��I       6%�	�U>���A�9*;


total_loss��@

error_R��F?

learning_rate_1?a7��8�I       6%�	��U>���A�9*;


total_loss]g@

error_R�I7?

learning_rate_1?a7�}I       6%�	�&V>���A�9*;


total_loss��r@

error_ReH?

learning_rate_1?a7mκ|I       6%�	�V>���A�9*;


total_loss@�"A

error_R�@K?

learning_rate_1?a7���I       6%�	U�V>���A�9*;


total_loss`�@

error_R;�R?

learning_rate_1?a7��� I       6%�	Z?W>���A�9*;


total_lossHΥ@

error_R2Y?

learning_rate_1?a7��DI       6%�	 �W>���A�9*;


total_lossj�@

error_R*�\?

learning_rate_1?a7�p�I       6%�	Y�W>���A�9*;


total_loss!�m@

error_Rq#V?

learning_rate_1?a7�7��I       6%�	�X>���A�9*;


total_loss�	�@

error_R��R?

learning_rate_1?a7�}	tI       6%�	�XX>���A�9*;


total_loss��@

error_R�PX?

learning_rate_1?a7{֩rI       6%�	ѣX>���A�9*;


total_loss�q�@

error_R̪>?

learning_rate_1?a7Xg�#I       6%�	H�X>���A�9*;


total_loss��@

error_R�DN?

learning_rate_1?a7��߮I       6%�	�3Y>���A�9*;


total_loss��@

error_Rf�Q?

learning_rate_1?a7�II       6%�	�{Y>���A�9*;


total_loss-��@

error_R*=B?

learning_rate_1?a7�n�[I       6%�	y�Y>���A�9*;


total_lossd&�@

error_R%sG?

learning_rate_1?a7�Q�I       6%�	�
Z>���A�9*;


total_loss���@

error_RɝS?

learning_rate_1?a7A��]I       6%�	~OZ>���A�9*;


total_loss@�@

error_R�;7?

learning_rate_1?a7)�ʖI       6%�	Y�Z>���A�9*;


total_loss]~�@

error_Rt S?

learning_rate_1?a7!�x�I       6%�	�Z>���A�9*;


total_loss��@

error_R߶Q?

learning_rate_1?a7jg�I       6%�	�$[>���A�9*;


total_loss���@

error_R�<?

learning_rate_1?a7����I       6%�	em[>���A�9*;


total_lossʤ�@

error_R�dP?

learning_rate_1?a7�Gf�I       6%�	;�[>���A�9*;


total_loss���@

error_R�yL?

learning_rate_1?a7���6I       6%�	�\>���A�9*;


total_loss,)i@

error_RT?

learning_rate_1?a7`wWI       6%�	�P\>���A�9*;


total_loss�j�@

error_Ro�O?

learning_rate_1?a76k�I       6%�	՜\>���A�9*;


total_loss���@

error_R�A?

learning_rate_1?a7��cI       6%�	^�\>���A�9*;


total_loss�L�@

error_R�Q?

learning_rate_1?a7���~I       6%�	*]>���A�9*;


total_loss�>�@

error_R)Q?

learning_rate_1?a7�I0eI       6%�	�l]>���A�9*;


total_losse �@

error_R�\?

learning_rate_1?a7=I       6%�	�]>���A�9*;


total_lossM��@

error_R��W?

learning_rate_1?a7�24I       6%�	��]>���A�9*;


total_loss-��@

error_R�]?

learning_rate_1?a7}�KI       6%�	XB^>���A�9*;


total_loss�~�@

error_R}TM?

learning_rate_1?a7=Y`I       6%�	��^>���A�9*;


total_lossg@

error_RR�O?

learning_rate_1?a7ֶ�I       6%�	��^>���A�9*;


total_loss.��@

error_R<�P?

learning_rate_1?a7��)�I       6%�	_>���A�9*;


total_lossLY�@

error_R�W?

learning_rate_1?a7)�akI       6%�	a_>���A�9*;


total_lossx��@

error_Ri�R?

learning_rate_1?a7/�/6I       6%�	ˠ_>���A�9*;


total_loss��@

error_R��8?

learning_rate_1?a7N���I       6%�	��_>���A�9*;


total_loss���@

error_Rwi?

learning_rate_1?a7�#�I       6%�	�)`>���A�9*;


total_loss���@

error_RJF?

learning_rate_1?a7�nm�I       6%�	wm`>���A�9*;


total_loss2A

error_R܀B?

learning_rate_1?a73nQ�I       6%�	��`>���A�9*;


total_lossԐ@

error_R��S?

learning_rate_1?a7=xII       6%�	|�`>���A�9*;


total_loss詁@

error_R�xD?

learning_rate_1?a7w:#�I       6%�	:a>���A�9*;


total_loss��@

error_R[JO?

learning_rate_1?a7�厃I       6%�	|a>���A�9*;


total_losse�@

error_R��P?

learning_rate_1?a7ً%]I       6%�	�a>���A�9*;


total_loss�ـ@

error_R3�S?

learning_rate_1?a7E��I       6%�	�b>���A�9*;


total_loss�p�@

error_R�J?

learning_rate_1?a7R�hI       6%�	�Ib>���A�9*;


total_loss�� A

error_Rs�K?

learning_rate_1?a7��l<I       6%�	w�b>���A�9*;


total_loss���@

error_R�E?

learning_rate_1?a7NG2lI       6%�	~�b>���A�9*;


total_lossZ�@

error_R�yX?

learning_rate_1?a7.�I       6%�	|c>���A�9*;


total_lossӦ@

error_R��X?

learning_rate_1?a7��i�I       6%�	4^c>���A�9*;


total_lossf�@

error_R�ib?

learning_rate_1?a7!KI       6%�	�c>���A�9*;


total_loss���@

error_R;DT?

learning_rate_1?a7⑇�I       6%�	�c>���A�9*;


total_loss��@

error_ReV?

learning_rate_1?a7�Ow]I       6%�	�0d>���A�9*;


total_loss� �@

error_R:R?

learning_rate_1?a7��nI       6%�	evd>���A�9*;


total_loss�6�@

error_R��L?

learning_rate_1?a7�2�I       6%�	!�d>���A�9*;


total_loss��@

error_R�Q?

learning_rate_1?a71(I       6%�	��d>���A�9*;


total_loss�i�@

error_Re�Y?

learning_rate_1?a7#��I       6%�	�<e>���A�9*;


total_loss���@

error_Rv�Q?

learning_rate_1?a7�?I       6%�	F�e>���A�9*;


total_lossr֮@

error_R�S[?

learning_rate_1?a7\WPI       6%�	!�e>���A�9*;


total_loss���@

error_R�9?

learning_rate_1?a7���I       6%�	'f>���A�9*;


total_loss���@

error_R�I?

learning_rate_1?a7�y�I       6%�	�Xf>���A�9*;


total_lossqu�@

error_R�P?

learning_rate_1?a7H��JI       6%�	+�f>���A�9*;


total_loss��A

error_RmX;?

learning_rate_1?a7��jI       6%�	�g>���A�9*;


total_loss��@

error_R�B?

learning_rate_1?a7X�i�I       6%�	�Pg>���A�9*;


total_loss�V�@

error_R;�V?

learning_rate_1?a7~>��I       6%�	��g>���A�9*;


total_loss���@

error_R��L?

learning_rate_1?a7j�ӨI       6%�	��g>���A�9*;


total_loss�_�@

error_R.�P?

learning_rate_1?a7H ��I       6%�	�h>���A�9*;


total_loss_�@

error_R=�F?

learning_rate_1?a7���I       6%�	�gh>���A�9*;


total_loss�τ@

error_R��N?

learning_rate_1?a7�}�6I       6%�	]�h>���A�9*;


total_loss���@

error_Rs�L?

learning_rate_1?a7:��WI       6%�	��h>���A�9*;


total_loss���@

error_Rn�I?

learning_rate_1?a7!��NI       6%�	$1i>���A�9*;


total_lossWC�@

error_R�J?

learning_rate_1?a7�ߧ�I       6%�	'si>���A�9*;


total_loss�h�@

error_Rx�U?

learning_rate_1?a76)��I       6%�	�i>���A�9*;


total_loss)��@

error_R�9?

learning_rate_1?a7�/ہI       6%�	��i>���A�9*;


total_loss�� A

error_R�I?

learning_rate_1?a7�I       6%�	�;j>���A�9*;


total_loss Ů@

error_R�HX?

learning_rate_1?a7-�	:I       6%�	��j>���A�9*;


total_loss�̮@

error_R��W?

learning_rate_1?a7�_;I       6%�	�j>���A�9*;


total_loss��@

error_RJ?

learning_rate_1?a7i
�I       6%�	bk>���A�9*;


total_lossO��@

error_R�\Q?

learning_rate_1?a7���HI       6%�	[k>���A�:*;


total_lossS�@

error_R�NG?

learning_rate_1?a7��.�I       6%�	��k>���A�:*;


total_loss�a�@

error_R҂V?

learning_rate_1?a7�Y��I       6%�	}�k>���A�:*;


total_lossG�@

error_RάS?

learning_rate_1?a7�ЯI       6%�	_&l>���A�:*;


total_loss��@

error_R� +?

learning_rate_1?a7 @�I       6%�	Kpl>���A�:*;


total_loss\�A

error_RiU?

learning_rate_1?a7�t:I       6%�	c�l>���A�:*;


total_loss�ƕ@

error_R�~V?

learning_rate_1?a7_�6nI       6%�	�l>���A�:*;


total_loss\¿@

error_RJ?

learning_rate_1?a7��R�I       6%�	�8m>���A�:*;


total_loss\�]@

error_R��F?

learning_rate_1?a79rɮI       6%�	~m>���A�:*;


total_loss{�@

error_R�L?

learning_rate_1?a7���I       6%�	��m>���A�:*;


total_loss�@

error_R&�]?

learning_rate_1?a7�f4vI       6%�	,n>���A�:*;


total_loss��A

error_RR'J?

learning_rate_1?a7@Lp�I       6%�	�kn>���A�:*;


total_loss�A�@

error_R��M?

learning_rate_1?a7OaπI       6%�	��n>���A�:*;


total_loss�ϼ@

error_R��Q?

learning_rate_1?a7�]PxI       6%�	��n>���A�:*;


total_loss�6�@

error_R1 i?

learning_rate_1?a7���LI       6%�	T=o>���A�:*;


total_loss�@

error_Rd�S?

learning_rate_1?a7c<�I       6%�	Z�o>���A�:*;


total_loss�q�@

error_RT>??

learning_rate_1?a7j��I       6%�	�o>���A�:*;


total_loss~�@

error_RE�J?

learning_rate_1?a7�'I       6%�	�"p>���A�:*;


total_loss-8A

error_R=\?

learning_rate_1?a7�P�I       6%�	�gp>���A�:*;


total_loss/u�@

error_R�T?

learning_rate_1?a7K8)I       6%�	x�p>���A�:*;


total_loss:��@

error_R\�K?

learning_rate_1?a7� &;I       6%�	u�p>���A�:*;


total_loss��z@

error_R�8=?

learning_rate_1?a7��-�I       6%�	.Bq>���A�:*;


total_lossM(�@

error_R�FN?

learning_rate_1?a7�ǓyI       6%�	u�q>���A�:*;


total_loss��@

error_R�$O?

learning_rate_1?a7!���I       6%�	@�q>���A�:*;


total_loss���@

error_R�a?

learning_rate_1?a7�8�I       6%�	�r>���A�:*;


total_loss,ʕ@

error_R��K?

learning_rate_1?a7ٰn�I       6%�	fUr>���A�:*;


total_loss�r�@

error_R�N?

learning_rate_1?a7 �#�I       6%�	V�r>���A�:*;


total_loss�ؘ@

error_R��L?

learning_rate_1?a7'7�VI       6%�	��r>���A�:*;


total_loss���@

error_R1pc?

learning_rate_1?a7�r�I       6%�	�&s>���A�:*;


total_loss4��@

error_R)�W?

learning_rate_1?a7�.BiI       6%�	4ls>���A�:*;


total_loss>�@

error_R)�H?

learning_rate_1?a7�`�I       6%�	�s>���A�:*;


total_loss�ʫ@

error_R�I?

learning_rate_1?a7��WI       6%�	��s>���A�:*;


total_loss_�@

error_R?6D?

learning_rate_1?a7n��I       6%�	I>t>���A�:*;


total_lossq��@

error_R��R?

learning_rate_1?a7�ʰzI       6%�	�~t>���A�:*;


total_loss��A

error_R��G?

learning_rate_1?a7�<��I       6%�	��t>���A�:*;


total_loss�A

error_RJ_?

learning_rate_1?a7�@a�I       6%�	Fu>���A�:*;


total_loss_	�@

error_RxKT?

learning_rate_1?a7�w+I       6%�	�Ju>���A�:*;


total_lossǷ@

error_R�I?

learning_rate_1?a7�եI       6%�	ؔu>���A�:*;


total_loss���@

error_R�R7?

learning_rate_1?a7X�3�I       6%�	 �u>���A�:*;


total_loss/�@

error_R�S?

learning_rate_1?a7�C�I       6%�	"v>���A�:*;


total_lossC
�@

error_RcR?

learning_rate_1?a7���FI       6%�	ksv>���A�:*;


total_loss�8�@

error_RI`N?

learning_rate_1?a7��QI       6%�	k�v>���A�:*;


total_loss�ɘ@

error_R��[?

learning_rate_1?a7����I       6%�	J#w>���A�:*;


total_loss�?�@

error_RK?

learning_rate_1?a7S3gXI       6%�	�gw>���A�:*;


total_loss��c@

error_R�zC?

learning_rate_1?a7��KI       6%�	ͬw>���A�:*;


total_lossV�|@

error_R��P?

learning_rate_1?a7ҍ�3I       6%�	v�w>���A�:*;


total_loss�U�@

error_R;`A?

learning_rate_1?a7�z��I       6%�	�4x>���A�:*;


total_loss��@

error_R��W?

learning_rate_1?a7qp��I       6%�	wx>���A�:*;


total_lossV2�@

error_RvF;?

learning_rate_1?a7��s+I       6%�	��x>���A�:*;


total_loss���@

error_R��S?

learning_rate_1?a7�<t�I       6%�	�x>���A�:*;


total_lossT�@

error_R��h?

learning_rate_1?a7��>WI       6%�	�Dy>���A�:*;


total_loss���@

error_R|�Z?

learning_rate_1?a7���I       6%�	-�y>���A�:*;


total_loss��w@

error_R�T?

learning_rate_1?a7Y��(I       6%�	��y>���A�:*;


total_loss7�@

error_R�yL?

learning_rate_1?a7�j��I       6%�	�z>���A�:*;


total_loss��@

error_Rx7C?

learning_rate_1?a7��^�I       6%�	)_z>���A�:*;


total_loss�T�@

error_Rl_A?

learning_rate_1?a7si�TI       6%�	u�z>���A�:*;


total_loss��@

error_RͷJ?

learning_rate_1?a7�SW�I       6%�	
�z>���A�:*;


total_lossBA

error_Rm�;?

learning_rate_1?a7�;��I       6%�	�3{>���A�:*;


total_loss��@

error_R�M?

learning_rate_1?a7���I       6%�	�u{>���A�:*;


total_loss�>m@

error_R~G?

learning_rate_1?a7�e��I       6%�	v�{>���A�:*;


total_lossŶ�@

error_RrwG?

learning_rate_1?a7�_O�I       6%�	s�{>���A�:*;


total_loss��@

error_R��B?

learning_rate_1?a7�� I       6%�	>>|>���A�:*;


total_loss�q�@

error_R��;?

learning_rate_1?a7v�I       6%�	��|>���A�:*;


total_lossȀ�@

error_RwwS?

learning_rate_1?a7�,qYI       6%�	�|>���A�:*;


total_lossf�A

error_R��U?

learning_rate_1?a7p���I       6%�	�}>���A�:*;


total_lossl�@

error_R�?R?

learning_rate_1?a7RU_�I       6%�	JS}>���A�:*;


total_loss)B�@

error_R1DB?

learning_rate_1?a7�P�wI       6%�	'�}>���A�:*;


total_lossH��@

error_R�N?

learning_rate_1?a7��jI       6%�	��}>���A�:*;


total_loss���@

error_R��??

learning_rate_1?a7*�DkI       6%�	o"~>���A�:*;


total_loss}C�@

error_RV�[?

learning_rate_1?a7���I       6%�	�l~>���A�:*;


total_lossEū@

error_R�dV?

learning_rate_1?a7�D�I       6%�	�~>���A�:*;


total_loss?a�@

error_R�O?

learning_rate_1?a7z�@�I       6%�	2�~>���A�:*;


total_lossS��@

error_R�L?

learning_rate_1?a7Y��I       6%�	A>���A�:*;


total_loss��@

error_R��V?

learning_rate_1?a7���I       6%�	�>���A�:*;


total_loss���@

error_R=�L?

learning_rate_1?a7�Z��I       6%�	l�>���A�:*;


total_loss��@

error_R$�:?

learning_rate_1?a7��I       6%�	K�>���A�:*;


total_loss
�@

error_RIU?

learning_rate_1?a7Uf��I       6%�	b�>���A�:*;


total_loss.��@

error_R;�L?

learning_rate_1?a7e���I       6%�	���>���A�:*;


total_loss��@

error_Rx(??

learning_rate_1?a7\�	I       6%�	��>���A�:*;


total_loss�@

error_R��Y?

learning_rate_1?a7E���I       6%�	�1�>���A�:*;


total_lossoM�@

error_R��R?

learning_rate_1?a7ź�+I       6%�	�s�>���A�:*;


total_loss��s@

error_R��M?

learning_rate_1?a7���I       6%�	���>���A�:*;


total_loss��@

error_Rs�N?

learning_rate_1?a76U'WI       6%�	��>���A�:*;


total_lossN�@

error_R8�U?

learning_rate_1?a7��
�I       6%�	nU�>���A�:*;


total_loss�c�@

error_R��O?

learning_rate_1?a7vTS$I       6%�	���>���A�:*;


total_loss̈́�@

error_R\W?

learning_rate_1?a7G�ehI       6%�	��>���A�:*;


total_lossh�@

error_RwuF?

learning_rate_1?a74�I       6%�	/�>���A�:*;


total_loss���@

error_R�G?

learning_rate_1?a7l�R�I       6%�	�s�>���A�:*;


total_loss[�@

error_R4�P?

learning_rate_1?a7_�\]I       6%�		��>���A�:*;


total_loss���@

error_R@�C?

learning_rate_1?a7��r�I       6%�	j��>���A�:*;


total_lossv|v@

error_R\_?

learning_rate_1?a7��srI       6%�	qB�>���A�:*;


total_loss�B�@

error_Rn�G?

learning_rate_1?a7��jI       6%�	e��>���A�:*;


total_lossd�@

error_R�L?

learning_rate_1?a7ӊϴI       6%�	�˄>���A�:*;


total_loss��A

error_R̓T?

learning_rate_1?a7Y	��I       6%�	�>���A�:*;


total_loss;�@

error_R �H?

learning_rate_1?a7c=�I       6%�	�Y�>���A�:*;


total_loss���@

error_R��R?

learning_rate_1?a7����I       6%�	���>���A�:*;


total_loss���@

error_R� R?

learning_rate_1?a7�
)I       6%�	���>���A�:*;


total_loss��@

error_RēW?

learning_rate_1?a7�)�I       6%�	")�>���A�:*;


total_loss탗@

error_R�J?

learning_rate_1?a7O[�I       6%�	8v�>���A�:*;


total_loss I�@

error_R�$J?

learning_rate_1?a7����I       6%�	��>���A�:*;


total_loss9�@

error_R�]R?

learning_rate_1?a7�$��I       6%�	B*�>���A�:*;


total_loss
��@

error_RL?

learning_rate_1?a7�F��I       6%�	�o�>���A�:*;


total_loss:|�@

error_R}d?

learning_rate_1?a7�mI       6%�	Z��>���A�:*;


total_loss�(�@

error_R��K?

learning_rate_1?a7D��I       6%�	S�>���A�:*;


total_lossȢ�@

error_R�;Z?

learning_rate_1?a7f.��I       6%�	�E�>���A�:*;


total_loss:f�@

error_R.�T?

learning_rate_1?a7���I       6%�	?��>���A�:*;


total_lossf�@

error_R}fd?

learning_rate_1?a7'��I       6%�	dΈ>���A�:*;


total_lossdy�@

error_RVO?

learning_rate_1?a7�oɷI       6%�	t�>���A�:*;


total_loss�@

error_RxU?

learning_rate_1?a7
�I       6%�	�X�>���A�:*;


total_loss2W@

error_R �M?

learning_rate_1?a7,��^I       6%�	ŝ�>���A�:*;


total_loss{��@

error_R3S?

learning_rate_1?a7�<6rI       6%�	���>���A�:*;


total_loss���@

error_R|0<?

learning_rate_1?a7�~�I       6%�	�$�>���A�:*;


total_loss�̣@

error_R�LK?

learning_rate_1?a7��v�I       6%�	�h�>���A�:*;


total_loss_p�@

error_R
�W?

learning_rate_1?a7N�}NI       6%�	Ԫ�>���A�:*;


total_loss]tA

error_R�~^?

learning_rate_1?a7m��I       6%�	��>���A�:*;


total_lossd4�@

error_R;H;?

learning_rate_1?a7���sI       6%�	�1�>���A�:*;


total_loss�,�@

error_R�P?

learning_rate_1?a7�I       6%�	/x�>���A�:*;


total_loss���@

error_RωN?

learning_rate_1?a7*�I       6%�	��>���A�:*;


total_loss�@

error_R;W@?

learning_rate_1?a7��M1I       6%�	��>���A�:*;


total_loss�̼@

error_R�d?

learning_rate_1?a7�V��I       6%�	�M�>���A�:*;


total_loss�	�@

error_R�b?

learning_rate_1?a7	���I       6%�	���>���A�:*;


total_loss88�@

error_RdA?

learning_rate_1?a7k?�yI       6%�	&݌>���A�:*;


total_loss���@

error_R�]?

learning_rate_1?a7L��[I       6%�	�(�>���A�:*;


total_lossx��@

error_R�?A?

learning_rate_1?a7�p�MI       6%�	_k�>���A�:*;


total_loss��@

error_RZ_O?

learning_rate_1?a7���nI       6%�	Z��>���A�:*;


total_loss`��@

error_R�}]?

learning_rate_1?a7��n�I       6%�	��>���A�:*;


total_loss\��@

error_Re�R?

learning_rate_1?a7���@I       6%�	�d�>���A�:*;


total_loss�X�@

error_Ri8S?

learning_rate_1?a7u)��I       6%�	a��>���A�:*;


total_loss]�@

error_R��N?

learning_rate_1?a7.;]�I       6%�	���>���A�;*;


total_loss|Ӫ@

error_Ro�P?

learning_rate_1?a7��(I       6%�	-G�>���A�;*;


total_loss�?�@

error_R�MP?

learning_rate_1?a7���9I       6%�	���>���A�;*;


total_loss`��@

error_RWN?

learning_rate_1?a7~�]�I       6%�	
�>���A�;*;


total_loss
��@

error_Rd�^?

learning_rate_1?a7��{�I       6%�	�,�>���A�;*;


total_loss��@

error_R��P?

learning_rate_1?a7,�I       6%�	(v�>���A�;*;


total_loss�Z�@

error_Rf<X?

learning_rate_1?a7͞n�I       6%�	���>���A�;*;


total_lossN��@

error_R�)>?

learning_rate_1?a7Rd�QI       6%�	��>���A�;*;


total_loss�q�@

error_Ri�X?

learning_rate_1?a7��)$I       6%�	�W�>���A�;*;


total_loss��@

error_R�K?

learning_rate_1?a7h�u(I       6%�	��>���A�;*;


total_loss���@

error_R��U?

learning_rate_1?a7���RI       6%�	Rߑ>���A�;*;


total_loss���@

error_R��N?

learning_rate_1?a7$d�I       6%�	�#�>���A�;*;


total_loss�B�@

error_R��P?

learning_rate_1?a7&9�I       6%�	f�>���A�;*;


total_loss��@

error_R
3K?

learning_rate_1?a7��uI       6%�	Y��>���A�;*;


total_lossf��@

error_R	kJ?

learning_rate_1?a7�v��I       6%�	���>���A�;*;


total_loss�3�@

error_R�:X?

learning_rate_1?a7��,I       6%�	�A�>���A�;*;


total_loss���@

error_R#�J?

learning_rate_1?a7(��%I       6%�	���>���A�;*;


total_loss��@

error_R��??

learning_rate_1?a7��I       6%�	-Փ>���A�;*;


total_loss���@

error_RX?

learning_rate_1?a7�!�I       6%�	�!�>���A�;*;


total_loss�<�@

error_R_d?

learning_rate_1?a7���I       6%�	zh�>���A�;*;


total_loss�|�@

error_RAHC?

learning_rate_1?a7�R��I       6%�	��>���A�;*;


total_loss�0�@

error_R�eX?

learning_rate_1?a7~�[PI       6%�	�>���A�;*;


total_loss_�@

error_R�O>?

learning_rate_1?a7A�_�I       6%�	w8�>���A�;*;


total_loss���@

error_R,$Z?

learning_rate_1?a7�/��I       6%�	�y�>���A�;*;


total_loss6=�@

error_R},L?

learning_rate_1?a7��+�I       6%�	ܺ�>���A�;*;


total_lossi&�@

error_R��K?

learning_rate_1?a7���I       6%�	���>���A�;*;


total_lossz��@

error_R,�L?

learning_rate_1?a7� �I       6%�	.I�>���A�;*;


total_loss.�@

error_Rvr<?

learning_rate_1?a7DX�%I       6%�	���>���A�;*;


total_loss���@

error_R8�R?

learning_rate_1?a7$G�I       6%�	C��>���A�;*;


total_loss_*�@

error_R�bK?

learning_rate_1?a7��G�I       6%�	=�>���A�;*;


total_lossG�@

error_R�BL?

learning_rate_1?a7�DaI       6%�	���>���A�;*;


total_loss��|@

error_R@�O?

learning_rate_1?a7Tc�I       6%�	�×>���A�;*;


total_loss�Л@

error_R<�S?

learning_rate_1?a7v1kbI       6%�	�	�>���A�;*;


total_loss�Ѣ@

error_R��F?

learning_rate_1?a775�XI       6%�	JN�>���A�;*;


total_loss���@

error_Rl�K?

learning_rate_1?a7�.�pI       6%�	���>���A�;*;


total_lossA:�@

error_Rl�K?

learning_rate_1?a7!ENI       6%�	�ݘ>���A�;*;


total_loss쐬@

error_R��R?

learning_rate_1?a7Gx\�I       6%�	�%�>���A�;*;


total_loss�լ@

error_Rfc?

learning_rate_1?a7��FI       6%�	)l�>���A�;*;


total_lossq��@

error_R�J?

learning_rate_1?a7�#�I       6%�	�>���A�;*;


total_loss�	�@

error_R׊[?

learning_rate_1?a7����I       6%�	���>���A�;*;


total_loss���@

error_RWZ?

learning_rate_1?a74D��I       6%�	�=�>���A�;*;


total_loss.�@

error_R_A@?

learning_rate_1?a7PEI       6%�	���>���A�;*;


total_loss�Σ@

error_R�>C?

learning_rate_1?a7W^�I       6%�	&ƚ>���A�;*;


total_lossv�!A

error_R�eG?

learning_rate_1?a7�Ζ8I       6%�	�	�>���A�;*;


total_loss�x�@

error_RT�N?

learning_rate_1?a7�
I       6%�	`L�>���A�;*;


total_lossa"�@

error_R7�V?

learning_rate_1?a7t�̲I       6%�	���>���A�;*;


total_loss(�@

error_RlHJ?

learning_rate_1?a7�TeI       6%�	�қ>���A�;*;


total_loss#��@

error_RJT?

learning_rate_1?a7i*Q�I       6%�	�>���A�;*;


total_lossj��@

error_R`�E?

learning_rate_1?a7�G�I       6%�	�\�>���A�;*;


total_loss��@

error_R�8Z?

learning_rate_1?a7c�J�I       6%�	���>���A�;*;


total_loss�pv@

error_R{R?

learning_rate_1?a7��w:I       6%�	��>���A�;*;


total_loss���@

error_Rq�Y?

learning_rate_1?a7`w,�I       6%�	z)�>���A�;*;


total_loss8�@

error_R�cC?

learning_rate_1?a7< �I       6%�	m�>���A�;*;


total_loss�;�@

error_RXOW?

learning_rate_1?a7zHI       6%�	��>���A�;*;


total_loss���@

error_RW�-?

learning_rate_1?a7;/I       6%�	���>���A�;*;


total_lossWX�@

error_R��N?

learning_rate_1?a7#��I       6%�	�:�>���A�;*;


total_loss`�@

error_R��J?

learning_rate_1?a7LR�I       6%�	��>���A�;*;


total_loss%�@

error_R��X?

learning_rate_1?a7=��I       6%�	�Ϟ>���A�;*;


total_loss���@

error_RM�G?

learning_rate_1?a7�JI       6%�	s�>���A�;*;


total_loss�T�@

error_RھI?

learning_rate_1?a7	_��I       6%�	�X�>���A�;*;


total_loss�+}@

error_R$wO?

learning_rate_1?a7R\��I       6%�	���>���A�;*;


total_loss�)�@

error_R��V?

learning_rate_1?a7N�<I       6%�	e�>���A�;*;


total_loss{˧@

error_R@>F?

learning_rate_1?a7<6�I       6%�	2�>���A�;*;


total_loss.wc@

error_R��P?

learning_rate_1?a7e'�I       6%�	�x�>���A�;*;


total_loss�b�@

error_R��=?

learning_rate_1?a7�?�I       6%�	���>���A�;*;


total_loss���@

error_R�Y?

learning_rate_1?a7*}�.I       6%�	N�>���A�;*;


total_lossG�@

error_R�:?

learning_rate_1?a7����I       6%�	IK�>���A�;*;


total_loss�@

error_R�I?

learning_rate_1?a7�µ�I       6%�	���>���A�;*;


total_loss=>�@

error_R-�L?

learning_rate_1?a75 �I       6%�	,֡>���A�;*;


total_loss<��@

error_R��\?

learning_rate_1?a7lj:yI       6%�	I�>���A�;*;


total_loss�`�@

error_R��a?

learning_rate_1?a7T���I       6%�	2[�>���A�;*;


total_loss�9�@

error_R��K?

learning_rate_1?a7��^dI       6%�	X��>���A�;*;


total_loss��@

error_R6tK?

learning_rate_1?a7
��;I       6%�	��>���A�;*;


total_loss\5�@

error_R�PA?

learning_rate_1?a7�Ij/I       6%�	�@�>���A�;*;


total_loss���@

error_R�tI?

learning_rate_1?a7!��I       6%�	���>���A�;*;


total_loss�c�@

error_R�OL?

learning_rate_1?a72��I       6%�	�ң>���A�;*;


total_loss���@

error_R��Q?

learning_rate_1?a7G��I       6%�	��>���A�;*;


total_loss���@

error_R�TX?

learning_rate_1?a7��I       6%�	�_�>���A�;*;


total_loss���@

error_R8�^?

learning_rate_1?a7!��[I       6%�	ا�>���A�;*;


total_lossM��@

error_R�eY?

learning_rate_1?a7�?�I       6%�	=�>���A�;*;


total_loss�7�@

error_R��Y?

learning_rate_1?a7e;�-I       6%�	92�>���A�;*;


total_loss���@

error_R��K?

learning_rate_1?a78���I       6%�	�x�>���A�;*;


total_lossˉ@

error_RCeI?

learning_rate_1?a7U0VI       6%�	<��>���A�;*;


total_loss�v(A

error_R�1K?

learning_rate_1?a7���HI       6%�	��>���A�;*;


total_lossQ�@

error_R6.M?

learning_rate_1?a7^bڵI       6%�	�B�>���A�;*;


total_loss)	�@

error_ROuY?

learning_rate_1?a7�ފI       6%�	���>���A�;*;


total_loss�L�@

error_Rw�B?

learning_rate_1?a7 |��I       6%�	l�>���A�;*;


total_loss�`�@

error_R��T?

learning_rate_1?a7|9�I       6%�	�1�>���A�;*;


total_loss���@

error_Rr�S?

learning_rate_1?a7�P
bI       6%�	o{�>���A�;*;


total_loss ��@

error_RQ?

learning_rate_1?a7ѣ��I       6%�	}ħ>���A�;*;


total_lossv��@

error_R!IW?

learning_rate_1?a7�戳I       6%�	F�>���A�;*;


total_loss n�@

error_Rx�S?

learning_rate_1?a7��-mI       6%�	ZU�>���A�;*;


total_loss���@

error_RېK?

learning_rate_1?a7� �I       6%�	���>���A�;*;


total_loss�]�@

error_Ri:?

learning_rate_1?a7�~�5I       6%�	��>���A�;*;


total_loss䣄@

error_Ra�\?

learning_rate_1?a7Bh�3I       6%�	�#�>���A�;*;


total_loss{o�@

error_RY?

learning_rate_1?a7c 6$I       6%�	�d�>���A�;*;


total_loss�X�@

error_R�O?

learning_rate_1?a7���I       6%�	���>���A�;*;


total_loss�A

error_R�+W?

learning_rate_1?a7SG��I       6%�	��>���A�;*;


total_loss4��@

error_R!fP?

learning_rate_1?a7��5�I       6%�	�.�>���A�;*;


total_loss��@

error_Rv�C?

learning_rate_1?a7��I       6%�	�y�>���A�;*;


total_lossD`A

error_R�[?

learning_rate_1?a7��4I       6%�	M��>���A�;*;


total_lossq![@

error_R:�5?

learning_rate_1?a7:�;�I       6%�	V�>���A�;*;


total_loss8��@

error_R�	>?

learning_rate_1?a7)[I       6%�	E�>���A�;*;


total_loss�l�@

error_Rf�X?

learning_rate_1?a7�c�I       6%�	Ɏ�>���A�;*;


total_loss�ɥ@

error_R��E?

learning_rate_1?a7D��I       6%�	֫>���A�;*;


total_loss���@

error_R�b?

learning_rate_1?a7Z���I       6%�	�'�>���A�;*;


total_loss�J�@

error_R�zb?

learning_rate_1?a7h�rI       6%�	�t�>���A�;*;


total_loss�C�@

error_R�i?

learning_rate_1?a7ۄI       6%�	R��>���A�;*;


total_loss�A

error_Rh�a?

learning_rate_1?a7j̓I       6%�	���>���A�;*;


total_loss�F�@

error_R3�m?

learning_rate_1?a7\���I       6%�	�B�>���A�;*;


total_loss���@

error_R��O?

learning_rate_1?a7�a1�I       6%�	Ȅ�>���A�;*;


total_loss\��@

error_Rq�??

learning_rate_1?a7����I       6%�	rȭ>���A�;*;


total_loss��@

error_R��N?

learning_rate_1?a7I6a2I       6%�	d�>���A�;*;


total_lossqT\@

error_Rd�>?

learning_rate_1?a7��I       6%�	xd�>���A�;*;


total_lossjG%A

error_R/�Y?

learning_rate_1?a7��I       6%�	v��>���A�;*;


total_lossEk~@

error_R�QW?

learning_rate_1?a7�\��I       6%�	��>���A�;*;


total_loss� �@

error_R҅Q?

learning_rate_1?a7Z~x I       6%�	�:�>���A�;*;


total_loss;�@

error_R�cK?

learning_rate_1?a7��"I       6%�	���>���A�;*;


total_loss��@

error_R6_G?

learning_rate_1?a7? �KI       6%�	�ۯ>���A�;*;


total_loss��@

error_R��[?

learning_rate_1?a7��oI       6%�	$�>���A�;*;


total_lossW��@

error_R�9T?

learning_rate_1?a7݉xI       6%�	�m�>���A�;*;


total_loss�N�@

error_R19X?

learning_rate_1?a7y�CI       6%�	���>���A�;*;


total_lossv��@

error_Rq�N?

learning_rate_1?a7l���I       6%�	9�>���A�;*;


total_loss`!�@

error_R�M?

learning_rate_1?a7hU�I       6%�	�X�>���A�;*;


total_losss�@

error_R8�B?

learning_rate_1?a7,�K�I       6%�	ҥ�>���A�;*;


total_lossM3�@

error_R8J?

learning_rate_1?a7p5.I       6%�	 �>���A�;*;


total_losslW�@

error_R��V?

learning_rate_1?a7��R+I       6%�	�9�>���A�;*;


total_loss}�@

error_R�MO?

learning_rate_1?a7�{I       6%�	��>���A�;*;


total_loss��@

error_R��9?

learning_rate_1?a7��jI       6%�	�Ĳ>���A�<*;


total_loss���@

error_R��R?

learning_rate_1?a7�e�I       6%�	��>���A�<*;


total_loss�}s@

error_R�M?

learning_rate_1?a7���I       6%�	]R�>���A�<*;


total_loss�M�@

error_RfJ?

learning_rate_1?a7�̎�I       6%�	㘳>���A�<*;


total_loss�W�@

error_R�$M?

learning_rate_1?a7��~I       6%�	Cܳ>���A�<*;


total_loss�ƾ@

error_R��J?

learning_rate_1?a7��5MI       6%�	��>���A�<*;


total_loss���@

error_R�a?

learning_rate_1?a7aaF�I       6%�	b�>���A�<*;


total_loss�V�@

error_R��J?

learning_rate_1?a7g��wI       6%�	���>���A�<*;


total_loss͙�@

error_R
�H?

learning_rate_1?a7�;�I       6%�	�>���A�<*;


total_loss���@

error_R�9R?

learning_rate_1?a7k���I       6%�	_3�>���A�<*;


total_loss��@

error_R-O?

learning_rate_1?a7,��~I       6%�	$x�>���A�<*;


total_loss�2�@

error_RS�O?

learning_rate_1?a7,��rI       6%�	ֻ�>���A�<*;


total_loss,ǧ@

error_R�D?

learning_rate_1?a7�qr�I       6%�	2 �>���A�<*;


total_loss-T�@

error_R�+U?

learning_rate_1?a7ttyI       6%�	D�>���A�<*;


total_loss��A

error_R4}Y?

learning_rate_1?a7�
cI       6%�	S��>���A�<*;


total_loss�2�@

error_R^<?

learning_rate_1?a7���I       6%�	w�>���A�<*;


total_lossa*�@

error_R�[\?

learning_rate_1?a7v��sI       6%�	�:�>���A�<*;


total_loss���@

error_R�Q?

learning_rate_1?a7��o#I       6%�	���>���A�<*;


total_lossT�@

error_R��6?

learning_rate_1?a7����I       6%�	�ŷ>���A�<*;


total_loss��@

error_R!�U?

learning_rate_1?a7O�=I       6%�	�>���A�<*;


total_loss���@

error_R�"B?

learning_rate_1?a7����I       6%�	gQ�>���A�<*;


total_lossPA

error_R�S?

learning_rate_1?a7��ikI       6%�	͖�>���A�<*;


total_lossn��@

error_R��H?

learning_rate_1?a7"�m7I       6%�	=ٸ>���A�<*;


total_lossV��@

error_R&�L?

learning_rate_1?a7{��NI       6%�	S�>���A�<*;


total_lossF�@

error_R[UK?

learning_rate_1?a7Br�I       6%�	�`�>���A�<*;


total_loss�Q�@

error_Ra]8?

learning_rate_1?a7�E��I       6%�	���>���A�<*;


total_lossѐ�@

error_R�<?

learning_rate_1?a7O���I       6%�	���>���A�<*;


total_loss��@

error_R�??

learning_rate_1?a7ê�I       6%�	�1�>���A�<*;


total_loss3u�@

error_R�C?

learning_rate_1?a7�=�I       6%�	<w�>���A�<*;


total_lossA1�@

error_R��J?

learning_rate_1?a7�ӗ�I       6%�	M��>���A�<*;


total_loss�@

error_R�~\?

learning_rate_1?a7Y�I       6%�	���>���A�<*;


total_loss0̉@

error_R@�P?

learning_rate_1?a7�z�I       6%�	�@�>���A�<*;


total_lossd��@

error_R�CV?

learning_rate_1?a7 D|I       6%�	���>���A�<*;


total_loss���@

error_R\�X?

learning_rate_1?a7ꈦNI       6%�	$ɻ>���A�<*;


total_loss�>�@

error_R,R?

learning_rate_1?a7�4g�I       6%�	\�>���A�<*;


total_lossME�@

error_RR�P?

learning_rate_1?a7�4~I       6%�	P�>���A�<*;


total_loss&��@

error_R�N?

learning_rate_1?a7xb��I       6%�	f��>���A�<*;


total_loss�>�@

error_R��Y?

learning_rate_1?a7i��(I       6%�	yڼ>���A�<*;


total_loss
��@

error_R��E?

learning_rate_1?a7G��I       6%�	��>���A�<*;


total_loss���@

error_R�xG?

learning_rate_1?a7���I       6%�	�b�>���A�<*;


total_loss�A

error_Rf�Q?

learning_rate_1?a7T�8I       6%�	O��>���A�<*;


total_loss��A

error_R:�M?

learning_rate_1?a7� �2I       6%�	�>���A�<*;


total_loss��@

error_R�3]?

learning_rate_1?a7�yHmI       6%�	*�>���A�<*;


total_lossq��@

error_R�N?

learning_rate_1?a7pe�=I       6%�	+l�>���A�<*;


total_loss��m@

error_R�V?

learning_rate_1?a7�&"I       6%�	���>���A�<*;


total_loss�0�@

error_R�B?

learning_rate_1?a7u�/I       6%�	
�>���A�<*;


total_loss܇�@

error_R\D??

learning_rate_1?a72�x:I       6%�	 3�>���A�<*;


total_loss�*�@

error_R��E?

learning_rate_1?a7h�:I       6%�	Kw�>���A�<*;


total_loss:6�@

error_Rn#X?

learning_rate_1?a7���I       6%�	O��>���A�<*;


total_lossۇ�@

error_RjG?

learning_rate_1?a7�K�%I       6%�	��>���A�<*;


total_loss�`�@

error_R�[?

learning_rate_1?a7��I       6%�	�F�>���A�<*;


total_loss���@

error_R�B;?

learning_rate_1?a7P�LI       6%�	��>���A�<*;


total_loss��@

error_R�W?

learning_rate_1?a7���I       6%�	���>���A�<*;


total_lossѯ�@

error_R��9?

learning_rate_1?a7Ã��I       6%�	l�>���A�<*;


total_loss��@

error_R�?V?

learning_rate_1?a7�}2I       6%�	|R�>���A�<*;


total_lossO�A

error_RA[R?

learning_rate_1?a7�9��I       6%�	5��>���A�<*;


total_losse��@

error_R`�V?

learning_rate_1?a7'I       6%�	���>���A�<*;


total_lossR]�@

error_R1.?

learning_rate_1?a7v	uUI       6%�	i�>���A�<*;


total_loss�@

error_R �G?

learning_rate_1?a73YI       6%�	}c�>���A�<*;


total_lossl$�@

error_R�:L?

learning_rate_1?a7�L"\I       6%�	^��>���A�<*;


total_loss�:�@

error_R�M?

learning_rate_1?a7����I       6%�	���>���A�<*;


total_loss\�@

error_R��X?

learning_rate_1?a7��lI       6%�	�?�>���A�<*;


total_loss�@

error_R��M?

learning_rate_1?a7#�cI       6%�	��>���A�<*;


total_loss͂�@

error_R`?

learning_rate_1?a7V���I       6%�	j��>���A�<*;


total_lossXc�@

error_R�BB?

learning_rate_1?a7oX��I       6%�	2�>���A�<*;


total_loss�2�@

error_RL�W?

learning_rate_1?a7���SI       6%�	hZ�>���A�<*;


total_loss�F�@

error_RC!K?

learning_rate_1?a7��4I       6%�	���>���A�<*;


total_loss���@

error_R�7<?

learning_rate_1?a7��Q�I       6%�	���>���A�<*;


total_loss���@

error_ReG?

learning_rate_1?a7h
\�I       6%�	3+�>���A�<*;


total_lossi��@

error_R��Z?

learning_rate_1?a7]�	bI       6%�	�m�>���A�<*;


total_lossl�@

error_R��M?

learning_rate_1?a7���|I       6%�	e��>���A�<*;


total_lossW@

error_R��F?

learning_rate_1?a7�.-I       6%�	���>���A�<*;


total_loss��}@

error_R�Pb?

learning_rate_1?a7����I       6%�	�:�>���A�<*;


total_lossQ�Y@

error_R�N>?

learning_rate_1?a7��I�I       6%�	X��>���A�<*;


total_loss��@

error_R�QL?

learning_rate_1?a7�8I       6%�	���>���A�<*;


total_loss(��@

error_R*�[?

learning_rate_1?a7��<lI       6%�	0�>���A�<*;


total_lossjb�@

error_RT�N?

learning_rate_1?a7�V\[I       6%�	;|�>���A�<*;


total_loss��@

error_R� T?

learning_rate_1?a7b&�I       6%�	���>���A�<*;


total_loss��@

error_RL'a?

learning_rate_1?a7���XI       6%�	:
�>���A�<*;


total_loss�ص@

error_R�g`?

learning_rate_1?a7�8ΞI       6%�	�M�>���A�<*;


total_loss���@

error_R&�6?

learning_rate_1?a7��(?I       6%�	I��>���A�<*;


total_loss;�z@

error_R�xW?

learning_rate_1?a7���I       6%�	+��>���A�<*;


total_loss��@

error_RX�L?

learning_rate_1?a7�,��I       6%�	��>���A�<*;


total_loss��@

error_R2B?

learning_rate_1?a7���aI       6%�	�e�>���A�<*;


total_lossEd�@

error_R��N?

learning_rate_1?a7a%A�I       6%�	ī�>���A�<*;


total_loss��@

error_R�jQ?

learning_rate_1?a7�Ap�I       6%�	��>���A�<*;


total_loss_sA

error_RֶI?

learning_rate_1?a7K�_I       6%�	5�>���A�<*;


total_loss���@

error_R��G?

learning_rate_1?a7^�WI       6%�	�v�>���A�<*;


total_lossۄ�@

error_R�??

learning_rate_1?a7n�<�I       6%�	���>���A�<*;


total_loss���@

error_R:2I?

learning_rate_1?a7�yH�I       6%�	���>���A�<*;


total_loss�V�@

error_Rt�K?

learning_rate_1?a7y�1�I       6%�	�A�>���A�<*;


total_loss��@

error_RZ�O?

learning_rate_1?a7Ym�jI       6%�	f��>���A�<*;


total_loss�/�@

error_R�#P?

learning_rate_1?a7�/ZI       6%�	^��>���A�<*;


total_loss3 �@

error_RJ�V?

learning_rate_1?a7�b�[I       6%�	"�>���A�<*;


total_loss�6�@

error_R�H?

learning_rate_1?a7�E�I       6%�	�R�>���A�<*;


total_lossRȫ@

error_RȿL?

learning_rate_1?a7�`vI       6%�	T��>���A�<*;


total_loss ��@

error_Rl�Q?

learning_rate_1?a7�z��I       6%�	��>���A�<*;


total_loss��@

error_RfkV?

learning_rate_1?a7Ϸ�I       6%�	��>���A�<*;


total_loss&&�@

error_R&�O?

learning_rate_1?a7����I       6%�	�a�>���A�<*;


total_loss
%�@

error_RD?

learning_rate_1?a7��0KI       6%�	_��>���A�<*;


total_loss?ȓ@

error_R0S?

learning_rate_1?a7��6�I       6%�	���>���A�<*;


total_losslj�@

error_Rq45?

learning_rate_1?a7��R�I       6%�	�?�>���A�<*;


total_lossq��@

error_R�E?

learning_rate_1?a7�:I       6%�	���>���A�<*;


total_loss{�@

error_RC_R?

learning_rate_1?a7-��I       6%�	���>���A�<*;


total_loss�4�@

error_R!=A?

learning_rate_1?a7��qvI       6%�	&�>���A�<*;


total_loss�l�@

error_R�[?

learning_rate_1?a7�%��I       6%�	�z�>���A�<*;


total_loss #�@

error_Rv�B?

learning_rate_1?a78�ΒI       6%�	���>���A�<*;


total_loss��@

error_RnK?

learning_rate_1?a7`l'�I       6%�	��>���A�<*;


total_loss`Ʋ@

error_RQ�P?

learning_rate_1?a7ڵ��I       6%�	�Y�>���A�<*;


total_loss�@

error_R�6N?

learning_rate_1?a7�h`I       6%�	a��>���A�<*;


total_lossԔ�@

error_RN�L?

learning_rate_1?a7�K��I       6%�	���>���A�<*;


total_lossw��@

error_R/	S?

learning_rate_1?a7��kI       6%�	�1�>���A�<*;


total_loss���@

error_RT2W?

learning_rate_1?a7��Y�I       6%�	{�>���A�<*;


total_loss�@

error_Rb?

learning_rate_1?a7�k`2I       6%�	p��>���A�<*;


total_lossV+�@

error_Ra�H?

learning_rate_1?a7�"��I       6%�	�>���A�<*;


total_loss��A

error_R�EQ?

learning_rate_1?a7<�|�I       6%�	V�>���A�<*;


total_loss���@

error_R�zP?

learning_rate_1?a7r �TI       6%�	Ӡ�>���A�<*;


total_loss�\�@

error_R3!S?

learning_rate_1?a7
k6�I       6%�	���>���A�<*;


total_loss
-�@

error_R��=?

learning_rate_1?a7�zPI       6%�	G7�>���A�<*;


total_loss���@

error_RkS?

learning_rate_1?a7��I       6%�	���>���A�<*;


total_lossvͯ@

error_R8pY?

learning_rate_1?a7�D�PI       6%�	H��>���A�<*;


total_loss6��@

error_R<�>?

learning_rate_1?a7�p�I       6%�	��>���A�<*;


total_loss2-�@

error_RRVF?

learning_rate_1?a7���I       6%�	�Y�>���A�<*;


total_loss�q�@

error_R�+M?

learning_rate_1?a7�k�I       6%�	��>���A�<*;


total_loss�@

error_R��V?

learning_rate_1?a7�vȝI       6%�	U��>���A�<*;


total_loss��y@

error_R̒L?

learning_rate_1?a7�S��I       6%�	f3�>���A�<*;


total_loss�C�@

error_R��L?

learning_rate_1?a7%l��I       6%�	�y�>���A�<*;


total_lossᱯ@

error_R�b?

learning_rate_1?a7N��dI       6%�	��>���A�<*;


total_losss�@

error_R��O?

learning_rate_1?a7ā��I       6%�	@��>���A�=*;


total_loss�_a@

error_R��N?

learning_rate_1?a7~)�>I       6%�	pB�>���A�=*;


total_loss���@

error_R�G?

learning_rate_1?a7�OZI       6%�	ދ�>���A�=*;


total_loss=��@

error_Rn�S?

learning_rate_1?a7���^I       6%�	���>���A�=*;


total_loss��@

error_RW�&?

learning_rate_1?a7�rWAI       6%�	�0�>���A�=*;


total_loss��@

error_R[�A?

learning_rate_1?a7�3�qI       6%�	�s�>���A�=*;


total_loss�L�@

error_R��J?

learning_rate_1?a7]�?�I       6%�	Z��>���A�=*;


total_loss[E�@

error_R��N?

learning_rate_1?a7�.:UI       6%�	q��>���A�=*;


total_lossa��@

error_R�\[?

learning_rate_1?a7���I       6%�	;;�>���A�=*;


total_loss(�A

error_R@�F?

learning_rate_1?a7r���I       6%�	ր�>���A�=*;


total_loss�c�@

error_Rx�H?

learning_rate_1?a7aY��I       6%�	���>���A�=*;


total_loss�}�@

error_R�_?

learning_rate_1?a7~ฯI       6%�	��>���A�=*;


total_losstۻ@

error_R�qK?

learning_rate_1?a7��MI       6%�	�I�>���A�=*;


total_loss^�@

error_RͭR?

learning_rate_1?a7e�+�I       6%�	���>���A�=*;


total_loss�ǐ@

error_R�AD?

learning_rate_1?a7m^/�I       6%�	*��>���A�=*;


total_lossg�@

error_R�
`?

learning_rate_1?a7���I       6%�	��>���A�=*;


total_losss̋@

error_R�M?

learning_rate_1?a7����I       6%�	�S�>���A�=*;


total_lossEN�@

error_R�aM?

learning_rate_1?a7V��I       6%�	���>���A�=*;


total_loss`��@

error_R��T?

learning_rate_1?a7�0�[I       6%�	r��>���A�=*;


total_lossSh�@

error_R:[?

learning_rate_1?a7|�ΤI       6%�	��>���A�=*;


total_lossh˲@

error_R%~P?

learning_rate_1?a7L�OI       6%�	r_�>���A�=*;


total_loss���@

error_R��Z?

learning_rate_1?a7��4�I       6%�	���>���A�=*;


total_loss��@

error_R`N?

learning_rate_1?a7=v:'I       6%�	2��>���A�=*;


total_loss1�@

error_RspL?

learning_rate_1?a7���I       6%�	�5�>���A�=*;


total_loss��@

error_R%�D?

learning_rate_1?a7q!l�I       6%�	2��>���A�=*;


total_loss\ʪ@

error_R��F?

learning_rate_1?a7q�I       6%�	���>���A�=*;


total_loss��@

error_R�X?

learning_rate_1?a7��}WI       6%�	��>���A�=*;


total_loss���@

error_RƪO?

learning_rate_1?a7�Sm�I       6%�	{S�>���A�=*;


total_loss�!�@

error_R(�[?

learning_rate_1?a7����I       6%�	��>���A�=*;


total_loss�?�@

error_R�9?

learning_rate_1?a7��I       6%�	���>���A�=*;


total_loss
��@

error_R@`?

learning_rate_1?a7��m�I       6%�	��>���A�=*;


total_loss��@

error_R@J?

learning_rate_1?a76 �*I       6%�	�f�>���A�=*;


total_lossq�@

error_R�!E?

learning_rate_1?a7�}1I       6%�	���>���A�=*;


total_loss��@

error_RԖA?

learning_rate_1?a7Cf^_I       6%�	C��>���A�=*;


total_loss�g�@

error_R�;?

learning_rate_1?a7��a�I       6%�	�1�>���A�=*;


total_loss*#�@

error_R&Q?

learning_rate_1?a7;�]I       6%�	'u�>���A�=*;


total_lossʆ�@

error_R��T?

learning_rate_1?a7/�I       6%�	���>���A�=*;


total_loss�f�@

error_R��^?

learning_rate_1?a7��I       6%�	���>���A�=*;


total_loss�t�@

error_R�eS?

learning_rate_1?a7�v�I       6%�	y@�>���A�=*;


total_loss*=�@

error_R�I?

learning_rate_1?a7Di]II       6%�	���>���A�=*;


total_loss7l�@

error_Rs�4?

learning_rate_1?a7����I       6%�	%��>���A�=*;


total_lossd��@

error_R?G?

learning_rate_1?a7��D�I       6%�	R�>���A�=*;


total_loss�5~@

error_R�A?

learning_rate_1?a7�'bI       6%�	\�>���A�=*;


total_loss�[�@

error_R�>d?

learning_rate_1?a7�\wvI       6%�	$��>���A�=*;


total_loss�@

error_R��E?

learning_rate_1?a7،(I       6%�	���>���A�=*;


total_loss)z�@

error_RX?

learning_rate_1?a7�PNI       6%�	2)�>���A�=*;


total_loss0�@

error_RnE?

learning_rate_1?a7�[C>I       6%�	�p�>���A�=*;


total_loss ��@

error_R��Q?

learning_rate_1?a7wؚ�I       6%�	���>���A�=*;


total_loss��@

error_R3gZ?

learning_rate_1?a7�O��I       6%�	��>���A�=*;


total_loss���@

error_RNvV?

learning_rate_1?a7E
&\I       6%�		=�>���A�=*;


total_loss���@

error_R,�T?

learning_rate_1?a76[�I       6%�	p�>���A�=*;


total_lossa�A

error_R�D?

learning_rate_1?a7E��}I       6%�	���>���A�=*;


total_losse��@

error_R<@?

learning_rate_1?a7����I       6%�	��>���A�=*;


total_loss�܍@

error_RiP?

learning_rate_1?a7-�!�I       6%�	�L�>���A�=*;


total_loss�b�@

error_R��U?

learning_rate_1?a7b�JI       6%�	���>���A�=*;


total_loss��~@

error_R��N?

learning_rate_1?a7�b�mI       6%�	U��>���A�=*;


total_loss$�w@

error_R�[?

learning_rate_1?a7��̝I       6%�	9"�>���A�=*;


total_loss��@

error_R��A?

learning_rate_1?a7j�_I       6%�	�d�>���A�=*;


total_loss*��@

error_R��:?

learning_rate_1?a7G [I       6%�	.��>���A�=*;


total_loss{��@

error_Rn%b?

learning_rate_1?a7�#i;I       6%�	���>���A�=*;


total_loss�h�@

error_R�,Y?

learning_rate_1?a7僓>I       6%�	W0�>���A�=*;


total_loss��A

error_R� K?

learning_rate_1?a7���	I       6%�	]v�>���A�=*;


total_loss��@

error_R�rB?

learning_rate_1?a7�f�zI       6%�	R��>���A�=*;


total_loss,Ց@

error_RMC?

learning_rate_1?a7�� I       6%�	��>���A�=*;


total_lossn�@

error_R��B?

learning_rate_1?a7k0I       6%�	�c�>���A�=*;


total_loss���@

error_R�2S?

learning_rate_1?a7�Q�iI       6%�	;��>���A�=*;


total_lossCD�@

error_R��J?

learning_rate_1?a7�8�I       6%�	q��>���A�=*;


total_loss���@

error_R_FR?

learning_rate_1?a7-��I       6%�	�6�>���A�=*;


total_loss���@

error_R� 8?

learning_rate_1?a7�ܘI       6%�	�>���A�=*;


total_loss�(�@

error_R��B?

learning_rate_1?a7^�\(I       6%�	r��>���A�=*;


total_loss���@

error_R�BU?

learning_rate_1?a7�}�rI       6%�		�>���A�=*;


total_loss4��@

error_R�R?

learning_rate_1?a7��YI       6%�	mP�>���A�=*;


total_lossT��@

error_RO?

learning_rate_1?a7a�I       6%�	���>���A�=*;


total_loss�@

error_Ra�N?

learning_rate_1?a7��"�I       6%�	���>���A�=*;


total_loss �@

error_R&�D?

learning_rate_1?a7`���I       6%�	�!�>���A�=*;


total_loss��@

error_Rw�C?

learning_rate_1?a7���I       6%�	�m�>���A�=*;


total_lossJ��@

error_R��Z?

learning_rate_1?a7�3k"I       6%�	ߵ�>���A�=*;


total_loss�	�@

error_R�HJ?

learning_rate_1?a7�#�I       6%�	���>���A�=*;


total_loss�n�@

error_R=�^?

learning_rate_1?a7��I       6%�	A�>���A�=*;


total_lossN&O@

error_R�>?

learning_rate_1?a7q��"I       6%�	��>���A�=*;


total_loss.;�@

error_R��??

learning_rate_1?a7!�ncI       6%�	m��>���A�=*;


total_lossFB�@

error_RF?

learning_rate_1?a7���_I       6%�	�	�>���A�=*;


total_loss���@

error_R�I?

learning_rate_1?a7���<I       6%�	W�>���A�=*;


total_loss���@

error_R�]@?

learning_rate_1?a7*���I       6%�	m��>���A�=*;


total_loss�s�@

error_R��G?

learning_rate_1?a7Ⅲ�I       6%�	���>���A�=*;


total_loss�@

error_RH�K?

learning_rate_1?a7(�_9I       6%�	�1�>���A�=*;


total_loss��@

error_R��[?

learning_rate_1?a7+���I       6%�	A~�>���A�=*;


total_loss�qA

error_R�;Q?

learning_rate_1?a7��N�I       6%�	(��>���A�=*;


total_loss�h�@

error_RdEF?

learning_rate_1?a7�b�rI       6%�	��>���A�=*;


total_loss���@

error_R��V?

learning_rate_1?a7��I       6%�	!l�>���A�=*;


total_loss{2�@

error_R�#G?

learning_rate_1?a7�8�I       6%�	ܷ�>���A�=*;


total_loss���@

error_RNgY?

learning_rate_1?a7�m�aI       6%�	J�>���A�=*;


total_loss���@

error_R�0V?

learning_rate_1?a7���I       6%�	Q�>���A�=*;


total_loss��@

error_Re�N?

learning_rate_1?a7�,v I       6%�	��>���A�=*;


total_loss���@

error_RqcW?

learning_rate_1?a7���I       6%�	��>���A�=*;


total_loss#/�@

error_R�L?

learning_rate_1?a7"���I       6%�	%7�>���A�=*;


total_loss�k�@

error_R�V?

learning_rate_1?a7V�ZI       6%�	�y�>���A�=*;


total_loss���@

error_R37N?

learning_rate_1?a7����I       6%�	l��>���A�=*;


total_loss�@

error_R��`?

learning_rate_1?a7k)I       6%�	_�>���A�=*;


total_loss��@

error_RM D?

learning_rate_1?a7��p�I       6%�	�G�>���A�=*;


total_loss�w�@

error_R�??

learning_rate_1?a7��I       6%�	���>���A�=*;


total_lossS�q@

error_R�;Q?

learning_rate_1?a71
�hI       6%�	]��>���A�=*;


total_lossS�@

error_R!�L?

learning_rate_1?a7��h�I       6%�	��>���A�=*;


total_loss�e�@

error_R��I?

learning_rate_1?a79��QI       6%�	�c�>���A�=*;


total_loss�@

error_R�^?

learning_rate_1?a7uJ�I       6%�	\��>���A�=*;


total_loss��}@

error_R�YQ?

learning_rate_1?a7��#zI       6%�	���>���A�=*;


total_loss�@

error_R�
I?

learning_rate_1?a7��0^I       6%�	�>�>���A�=*;


total_lossVנ@

error_R�MU?

learning_rate_1?a7jJ�I       6%�	܇�>���A�=*;


total_lossH�@

error_R�G?

learning_rate_1?a78���I       6%�	���>���A�=*;


total_loss�1�@

error_R{�O?

learning_rate_1?a7g!�wI       6%�	�>���A�=*;


total_loss���@

error_R��S?

learning_rate_1?a7d���I       6%�	Xk�>���A�=*;


total_loss��@

error_R�^?

learning_rate_1?a7iSQI       6%�	j��>���A�=*;


total_loss�<X@

error_R�P?

learning_rate_1?a73x�NI       6%�	���>���A�=*;


total_loss=��@

error_R��D?

learning_rate_1?a7�;I       6%�	�?�>���A�=*;


total_loss�G�@

error_R=M?

learning_rate_1?a7�h��I       6%�	���>���A�=*;


total_loss8^A

error_Rm�R?

learning_rate_1?a7���I       6%�	m��>���A�=*;


total_loss���@

error_R �G?

learning_rate_1?a7 YI       6%�	��>���A�=*;


total_losshW�@

error_R��8?

learning_rate_1?a7��G]I       6%�	�Y�>���A�=*;


total_loss���@

error_R��\?

learning_rate_1?a7�ɁI       6%�	"��>���A�=*;


total_loss��@

error_R��Z?

learning_rate_1?a7�\%I       6%�	��>���A�=*;


total_loss/w�@

error_R��P?

learning_rate_1?a7��]I       6%�	I�>���A�=*;


total_loss���@

error_R�A?

learning_rate_1?a7��zI       6%�	��>���A�=*;


total_loss�j�@

error_RV?

learning_rate_1?a7.��CI       6%�	���>���A�=*;


total_lossI�@

error_R�tZ?

learning_rate_1?a7}��I       6%�	��>���A�=*;


total_loss���@

error_R
=T?

learning_rate_1?a7�o�I       6%�	,X�>���A�=*;


total_lossH��@

error_R,sH?

learning_rate_1?a7*OӫI       6%�	���>���A�=*;


total_loss�5~@

error_Rs	]?

learning_rate_1?a7���DI       6%�	���>���A�=*;


total_loss�p�@

error_R��B?

learning_rate_1?a7R=I       6%�	�1�>���A�=*;


total_loss��@

error_R÷H?

learning_rate_1?a7�x;zI       6%�	}|�>���A�>*;


total_lossv	A

error_R�J?

learning_rate_1?a7��DI       6%�	���>���A�>*;


total_loss!Ѱ@

error_R��G?

learning_rate_1?a7qA�gI       6%�	�
�>���A�>*;


total_lossӛ�@

error_RvJ\?

learning_rate_1?a7�TQI       6%�	�P�>���A�>*;


total_lossܵ�@

error_Rvy??

learning_rate_1?a75�/mI       6%�	���>���A�>*;


total_lossClA

error_RJKN?

learning_rate_1?a7�[�I       6%�	���>���A�>*;


total_lossV��@

error_Ri�L?

learning_rate_1?a7��
~I       6%�	{$�>���A�>*;


total_loss�,�@

error_R�SL?

learning_rate_1?a7�A��I       6%�	f�>���A�>*;


total_loss�Z�@

error_R�lT?

learning_rate_1?a7T1�gI       6%�	��>���A�>*;


total_loss\��@

error_R��K?

learning_rate_1?a7,��)I       6%�	��>���A�>*;


total_loss/��@

error_RԩS?

learning_rate_1?a7G�RI       6%�	�A�>���A�>*;


total_loss��@

error_R�H?

learning_rate_1?a7��^I       6%�	Љ�>���A�>*;


total_loss�߰@

error_RҬM?

learning_rate_1?a7~�D4I       6%�	���>���A�>*;


total_loss�'LA

error_RqD^?

learning_rate_1?a7wυ�I       6%�	, �>���A�>*;


total_loss�h�@

error_R�?B?

learning_rate_1?a7��ŭI       6%�	�h�>���A�>*;


total_loss��@

error_R�%E?

learning_rate_1?a7�>t�I       6%�	��>���A�>*;


total_lossZȉ@

error_R3�@?

learning_rate_1?a7�]�XI       6%�	~��>���A�>*;


total_loss�}�@

error_R�NF?

learning_rate_1?a7O���I       6%�	�.�>���A�>*;


total_loss?2�@

error_R!�>?

learning_rate_1?a7���I       6%�	�s�>���A�>*;


total_lossɽ�@

error_R|U?

learning_rate_1?a7����I       6%�	���>���A�>*;


total_loss}��@

error_Rwjj?

learning_rate_1?a7�i�tI       6%�	���>���A�>*;


total_loss�/�@

error_RJ�J?

learning_rate_1?a7�
�I       6%�	�<�>���A�>*;


total_loss��@

error_R�MT?

learning_rate_1?a7�/�I       6%�	��>���A�>*;


total_loss��{@

error_Rs�A?

learning_rate_1?a7~��I       6%�	��>���A�>*;


total_loss��@

error_RdLF?

learning_rate_1?a7���I       6%�	k ?���A�>*;


total_loss�J�@

error_R7CA?

learning_rate_1?a7&�;I       6%�	d\ ?���A�>*;


total_lossA��@

error_R�jL?

learning_rate_1?a7D�HI       6%�	�� ?���A�>*;


total_loss��@

error_RM�S?

learning_rate_1?a7��HbI       6%�	� ?���A�>*;


total_loss#��@

error_R�	U?

learning_rate_1?a7�?��I       6%�	t+?���A�>*;


total_lossS��@

error_R�G?

learning_rate_1?a7�]��I       6%�	_v?���A�>*;


total_lossĽA

error_R��J?

learning_rate_1?a7�l��I       6%�	�?���A�>*;


total_lossEt�@

error_RJj^?

learning_rate_1?a7�I�I       6%�	��?���A�>*;


total_loss�p�@

error_R\�W?

learning_rate_1?a7�3ėI       6%�	�D?���A�>*;


total_loss,s@

error_Rd�`?

learning_rate_1?a7Ĵ7�I       6%�	!�?���A�>*;


total_loss;t�@

error_R��@?

learning_rate_1?a7{��I       6%�	9�?���A�>*;


total_lossx��@

error_R�B?

learning_rate_1?a7I       6%�	B?���A�>*;


total_loss�J�@

error_R_�h?

learning_rate_1?a7 (�I       6%�	�U?���A�>*;


total_loss.��@

error_R��M?

learning_rate_1?a7�"+�I       6%�	W�?���A�>*;


total_loss���@

error_R �9?

learning_rate_1?a7�)�SI       6%�	c�?���A�>*;


total_lossl%�@

error_R��R?

learning_rate_1?a7�}�I       6%�	�,?���A�>*;


total_loss���@

error_Rv�O?

learning_rate_1?a7ه�1I       6%�	�s?���A�>*;


total_loss�Ҩ@

error_R��E?

learning_rate_1?a7~�cnI       6%�	��?���A�>*;


total_lossզ@

error_R�VX?

learning_rate_1?a7��I       6%�	-�?���A�>*;


total_lossU�@

error_R��M?

learning_rate_1?a7��UI       6%�	??���A�>*;


total_loss�J�@

error_Rv�L?

learning_rate_1?a79���I       6%�	"�?���A�>*;


total_lossT\�@

error_R�+7?

learning_rate_1?a7V౹I       6%�	�?���A�>*;


total_loss@�@

error_R��E?

learning_rate_1?a7��iI       6%�	�?���A�>*;


total_loss���@

error_R�P[?

learning_rate_1?a7��(�I       6%�	�Y?���A�>*;


total_loss1[�@

error_R,F[?

learning_rate_1?a7/(\I       6%�	�?���A�>*;


total_loss�	�@

error_R�M?

learning_rate_1?a7��I       6%�	T?���A�>*;


total_loss���@

error_RiTL?

learning_rate_1?a7�I       6%�	�M?���A�>*;


total_loss�_�@

error_R�$M?

learning_rate_1?a7C�;fI       6%�	k�?���A�>*;


total_loss8�@

error_RDI?

learning_rate_1?a7p�I       6%�	g�?���A�>*;


total_lossmM�@

error_Rj�M?

learning_rate_1?a7�н�I       6%�	�?���A�>*;


total_loss�Bg@

error_RlFO?

learning_rate_1?a7-HZ�I       6%�	�]?���A�>*;


total_loss�3�@

error_R�=U?

learning_rate_1?a7\6�I       6%�	��?���A�>*;


total_loss���@

error_Rf�E?

learning_rate_1?a7�RތI       6%�	��?���A�>*;


total_loss��@

error_R��\?

learning_rate_1?a7��T�I       6%�	�-	?���A�>*;


total_loss�l�@

error_Rh�R?

learning_rate_1?a76�q`I       6%�	p	?���A�>*;


total_lossa��@

error_RCU?

learning_rate_1?a7��X�I       6%�	��	?���A�>*;


total_loss�zA

error_R��Y?

learning_rate_1?a7��Q0I       6%�	G�	?���A�>*;


total_loss*U�@

error_R��K?

learning_rate_1?a7���I       6%�	�8
?���A�>*;


total_loss�C�@

error_R��H?

learning_rate_1?a7�;(4I       6%�	�~
?���A�>*;


total_lossК@

error_R�??

learning_rate_1?a7�R@0I       6%�	�
?���A�>*;


total_loss}b�@

error_RqyK?

learning_rate_1?a77
�I       6%�	�?���A�>*;


total_loss0�@

error_R��_?

learning_rate_1?a7W�AI       6%�	��?���A�>*;


total_lossnq�@

error_R�M?

learning_rate_1?a7@\nfI       6%�	�>?���A�>*;


total_loss�Ք@

error_R͜S?

learning_rate_1?a7�(��I       6%�	��?���A�>*;


total_loss	�@

error_Rx]?

learning_rate_1?a7��?�I       6%�	U�?���A�>*;


total_loss���@

error_Rz�>?

learning_rate_1?a7�̢�I       6%�	O?���A�>*;


total_loss���@

error_R�>?

learning_rate_1?a7Y��I       6%�	f?���A�>*;


total_loss�k�@

error_R��>?

learning_rate_1?a7ka�I       6%�	[�?���A�>*;


total_loss�R�@

error_R_�N?

learning_rate_1?a7���I       6%�	��?���A�>*;


total_loss�0�@

error_R�uX?

learning_rate_1?a7�5��I       6%�	j7?���A�>*;


total_loss�,�@

error_R$�d?

learning_rate_1?a7EF��I       6%�	�?���A�>*;


total_lossqb�@

error_R|�W?

learning_rate_1?a7�I��I       6%�	[�?���A�>*;


total_loss۷�@

error_R��N?

learning_rate_1?a7w`�.I       6%�	_?���A�>*;


total_loss���@

error_R�mQ?

learning_rate_1?a7 �nI       6%�	�W?���A�>*;


total_loss�x�@

error_R�|G?

learning_rate_1?a7g�T�I       6%�	��?���A�>*;


total_lossl5�@

error_R�|9?

learning_rate_1?a7�Z3�I       6%�	#�?���A�>*;


total_loss5�A

error_R�K>?

learning_rate_1?a7]�#�I       6%�	�4?���A�>*;


total_lossl��@

error_R){K?

learning_rate_1?a78LTI       6%�	m�?���A�>*;


total_loss!�A

error_R� \?

learning_rate_1?a7�c�EI       6%�	��?���A�>*;


total_loss��@

error_R{~c?

learning_rate_1?a7�!��I       6%�	�?���A�>*;


total_loss㘦@

error_R0I?

learning_rate_1?a78��I       6%�	�Z?���A�>*;


total_loss\@

error_Rx#G?

learning_rate_1?a7���8I       6%�	C�?���A�>*;


total_lossd5�@

error_RdL?

learning_rate_1?a7o���I       6%�	��?���A�>*;


total_loss��@

error_R��A?

learning_rate_1?a7�<��I       6%�	*?���A�>*;


total_loss�X�@

error_R�=F?

learning_rate_1?a7�N;�I       6%�	?o?���A�>*;


total_loss&¨@

error_R��K?

learning_rate_1?a7@M �I       6%�	��?���A�>*;


total_loss��@

error_R�?G?

learning_rate_1?a7m�l�I       6%�	a?���A�>*;


total_loss��A

error_R��H?

learning_rate_1?a7 �gI       6%�	�]?���A�>*;


total_loss�N�@

error_R��L?

learning_rate_1?a7�ڏ�I       6%�	��?���A�>*;


total_losstk�@

error_R��I?

learning_rate_1?a7�	�I       6%�	��?���A�>*;


total_lossr��@

error_R�+?

learning_rate_1?a7�j�I       6%�	�)?���A�>*;


total_loss�A

error_R$Q?

learning_rate_1?a7��[I       6%�	~m?���A�>*;


total_lossr��@

error_RW�E?

learning_rate_1?a74t�>I       6%�	��?���A�>*;


total_loss�2�@

error_Rv�D?

learning_rate_1?a7D��I       6%�	�?���A�>*;


total_loss&L�@

error_R)I?

learning_rate_1?a7R`Q�I       6%�	wg?���A�>*;


total_loss�4�@

error_R�%O?

learning_rate_1?a7�T�I       6%�	��?���A�>*;


total_loss?��@

error_Rs�i?

learning_rate_1?a7&�I       6%�	�?���A�>*;


total_loss��@

error_R�OR?

learning_rate_1?a7h�_*I       6%�	�m?���A�>*;


total_lossx�)A

error_RҵS?

learning_rate_1?a7�Y�dI       6%�	}�?���A�>*;


total_loss3��@

error_R��W?

learning_rate_1?a7.]��I       6%�	�?���A�>*;


total_loss�S�@

error_R��K?

learning_rate_1?a7Ś�{I       6%�	Qx?���A�>*;


total_loss�u�@

error_R�L??

learning_rate_1?a7^�#�I       6%�	��?���A�>*;


total_loss��@

error_Rz�Z?

learning_rate_1?a7��4�I       6%�	�?���A�>*;


total_loss���@

error_R��D?

learning_rate_1?a7B��LI       6%�	cz?���A�>*;


total_loss���@

error_R��T?

learning_rate_1?a7>(��I       6%�	W�?���A�>*;


total_loss�|�@

error_RWN?

learning_rate_1?a7�GI       6%�	y?���A�>*;


total_loss���@

error_R
~D?

learning_rate_1?a76�3�I       6%�	�g?���A�>*;


total_loss�aA

error_R�TB?

learning_rate_1?a7Y?j�I       6%�	��?���A�>*;


total_loss��A

error_R�:?

learning_rate_1?a7O!-�I       6%�	x?���A�>*;


total_loss1z@

error_R[�C?

learning_rate_1?a7~HI       6%�	�l?���A�>*;


total_loss]i�@

error_R!M?

learning_rate_1?a7
�<�I       6%�	��?���A�>*;


total_loss��2A

error_R}lU?

learning_rate_1?a7C�$I       6%�	�?���A�>*;


total_lossr�s@

error_R=�I?

learning_rate_1?a7�(�wI       6%�	�\?���A�>*;


total_loss�q�@

error_R@zO?

learning_rate_1?a7��^�I       6%�	m�?���A�>*;


total_loss�@

error_RL)E?

learning_rate_1?a7�A��I       6%�	C�?���A�>*;


total_loss�^�@

error_R�+_?

learning_rate_1?a73�B�I       6%�	�% ?���A�>*;


total_loss!Ǯ@

error_R�jT?

learning_rate_1?a7�\{?I       6%�	�o ?���A�>*;


total_loss��@

error_R��X?

learning_rate_1?a7���I       6%�	c� ?���A�>*;


total_lossH9�@

error_RIQ?

learning_rate_1?a7+mG#I       6%�	�� ?���A�>*;


total_lossx�@

error_R�D?

learning_rate_1?a7�b��I       6%�	�<!?���A�>*;


total_loss&��@

error_R�KE?

learning_rate_1?a7���I       6%�	v�!?���A�>*;


total_loss8�@

error_R�^G?

learning_rate_1?a7�c��I       6%�	�!?���A�>*;


total_loss���@

error_R�DY?

learning_rate_1?a7���I       6%�	�"?���A�>*;


total_loss��@

error_R��@?

learning_rate_1?a7�LU�I       6%�	�U"?���A�>*;


total_loss4	�@

error_R��U?

learning_rate_1?a7�ƛ�I       6%�	��"?���A�?*;


total_losso�i@

error_R��G?

learning_rate_1?a7~�gI       6%�	�"?���A�?*;


total_lossZ��@

error_R��X?

learning_rate_1?a7�sdI       6%�	^#?���A�?*;


total_loss�r]@

error_R��N?

learning_rate_1?a7ic��I       6%�	�_#?���A�?*;


total_loss��@

error_R@�B?

learning_rate_1?a7B��6I       6%�	��#?���A�?*;


total_loss�@

error_R8�@?

learning_rate_1?a7�f�I       6%�	d�#?���A�?*;


total_loss�.�@

error_R 6W?

learning_rate_1?a7��NI       6%�	1$?���A�?*;


total_loss�)�@

error_R�D?

learning_rate_1?a7�GʼI       6%�	�w$?���A�?*;


total_loss��@

error_R�mU?

learning_rate_1?a7��y�I       6%�	��$?���A�?*;


total_loss~��@

error_RFsM?

learning_rate_1?a7fT��I       6%�	D %?���A�?*;


total_loss���@

error_R�E?

learning_rate_1?a7m�ǝI       6%�	TE%?���A�?*;


total_loss-Cs@

error_R��A?

learning_rate_1?a7�=�I       6%�	P�%?���A�?*;


total_lossqս@

error_R��0?

learning_rate_1?a7���I       6%�	~�%?���A�?*;


total_loss�@

error_R]1K?

learning_rate_1?a72�CUI       6%�	�&?���A�?*;


total_loss�R�@

error_R%U?

learning_rate_1?a7�y�KI       6%�	T]&?���A�?*;


total_loss	�@

error_R�8?

learning_rate_1?a7��I       6%�	��&?���A�?*;


total_loss��@

error_R��Z?

learning_rate_1?a7���I       6%�	�'?���A�?*;


total_loss� �@

error_R�/:?

learning_rate_1?a7���*I       6%�	I'?���A�?*;


total_loss�?�@

error_RM�>?

learning_rate_1?a7v]�I       6%�	o�'?���A�?*;


total_loss&��@

error_R.PC?

learning_rate_1?a7]l��I       6%�	��'?���A�?*;


total_loss:iA

error_R;�U?

learning_rate_1?a7��lI       6%�	(?���A�?*;


total_lossLœ@

error_RzS?

learning_rate_1?a7Ъ�I       6%�	@Y(?���A�?*;


total_loss�A

error_R��Q?

learning_rate_1?a7���I       6%�	��(?���A�?*;


total_loss��@

error_R�'N?

learning_rate_1?a7Z�l�I       6%�	r�(?���A�?*;


total_loss�@

error_R�;K?

learning_rate_1?a7$�	I       6%�	�$)?���A�?*;


total_loss$�{@

error_R1�K?

learning_rate_1?a7�5�I       6%�	�f)?���A�?*;


total_loss2w�@

error_R�^J?

learning_rate_1?a7�p�rI       6%�	�)?���A�?*;


total_loss��@

error_R�9U?

learning_rate_1?a7�K�I       6%�	��)?���A�?*;


total_loss���@

error_RȉO?

learning_rate_1?a7ȶ�{I       6%�	k;*?���A�?*;


total_lossP�@

error_R��M?

learning_rate_1?a7��tI       6%�	Z�*?���A�?*;


total_loss��0A

error_R��B?

learning_rate_1?a7�8΂I       6%�	�*?���A�?*;


total_loss	!�@

error_R�5?

learning_rate_1?a7�4��I       6%�	+?���A�?*;


total_loss���@

error_RM?

learning_rate_1?a7��I       6%�	�S+?���A�?*;


total_loss��@

error_RJ�j?

learning_rate_1?a7�2�8I       6%�	5�+?���A�?*;


total_loss�@

error_R\Z?

learning_rate_1?a70lڋI       6%�	��+?���A�?*;


total_loss�+�@

error_R7G?

learning_rate_1?a7d2<�I       6%�	�(,?���A�?*;


total_loss�Z�@

error_R�N=?

learning_rate_1?a7��>�I       6%�	sk,?���A�?*;


total_loss��@

error_RE[M?

learning_rate_1?a7c���I       6%�	�,?���A�?*;


total_lossRW�@

error_Rl
E?

learning_rate_1?a7�)��I       6%�	@�,?���A�?*;


total_loss=��@

error_R�T?

learning_rate_1?a7�\�I       6%�	1-?���A�?*;


total_loss�܏@

error_R��K?

learning_rate_1?a7Q�|mI       6%�	�t-?���A�?*;


total_lossq��@

error_R��\?

learning_rate_1?a70��(I       6%�	��-?���A�?*;


total_loss���@

error_R8%X?

learning_rate_1?a7�5��I       6%�	�-?���A�?*;


total_lossar@

error_R��]?

learning_rate_1?a7d>�I       6%�	sK.?���A�?*;


total_loss|�@

error_R�fM?

learning_rate_1?a7cB��I       6%�	��.?���A�?*;


total_lossj��@

error_R$�Q?

learning_rate_1?a7�b8I       6%�	�/?���A�?*;


total_loss&�@

error_R��V?

learning_rate_1?a7����I       6%�	�a/?���A�?*;


total_loss(��@

error_R�n?

learning_rate_1?a7r���I       6%�	�/?���A�?*;


total_loss�A

error_R��V?

learning_rate_1?a7q��I       6%�	�/?���A�?*;


total_loss�\�@

error_R$�Q?

learning_rate_1?a7���I       6%�	�40?���A�?*;


total_loss�@

error_RȺX?

learning_rate_1?a7A���I       6%�	z0?���A�?*;


total_loss��@

error_R�V?

learning_rate_1?a7�Ӏ�I       6%�	��0?���A�?*;


total_loss��@

error_R��Y?

learning_rate_1?a71��II       6%�	�1?���A�?*;


total_loss�/A

error_R��b?

learning_rate_1?a7{�}I       6%�	V1?���A�?*;


total_loss2+�@

error_R&U?

learning_rate_1?a7���I       6%�	y�1?���A�?*;


total_loss���@

error_R/@?

learning_rate_1?a7�UMI       6%�	�1?���A�?*;


total_loss�̤@

error_Rs�H?

learning_rate_1?a7��pnI       6%�	 "2?���A�?*;


total_loss��@

error_R�
V?

learning_rate_1?a7޹�I       6%�	�g2?���A�?*;


total_loss\��@

error_RRK_?

learning_rate_1?a7i� �I       6%�	˨2?���A�?*;


total_lossM��@

error_R��K?

learning_rate_1?a7^��xI       6%�	(�2?���A�?*;


total_loss%��@

error_R��Y?

learning_rate_1?a7�
��I       6%�	e43?���A�?*;


total_loss�AA

error_R4�P?

learning_rate_1?a7$wI       6%�	.u3?���A�?*;


total_loss�9�@

error_R&}O?

learning_rate_1?a7�aI       6%�	�3?���A�?*;


total_loss�r�@

error_R�H?

learning_rate_1?a7��"I       6%�	��3?���A�?*;


total_loss�H�@

error_R�N?

learning_rate_1?a7�q�I       6%�	�A4?���A�?*;


total_lossop�@

error_R��_?

learning_rate_1?a7x�vI       6%�	��4?���A�?*;


total_loss���@

error_R��Y?

learning_rate_1?a7 �h]I       6%�	��4?���A�?*;


total_loss�Ǫ@

error_R�L?

learning_rate_1?a7vaG�I       6%�	�5?���A�?*;


total_loss�q�@

error_R�lS?

learning_rate_1?a7�?�hI       6%�	;\5?���A�?*;


total_loss��A

error_R ,J?

learning_rate_1?a7�6�I       6%�	�5?���A�?*;


total_loss�ވ@

error_R��[?

learning_rate_1?a7y\�&I       6%�	��5?���A�?*;


total_loss=��@

error_R�Eb?

learning_rate_1?a7��c�I       6%�	�$6?���A�?*;


total_lossn�@

error_R�YJ?

learning_rate_1?a7a22I       6%�	~j6?���A�?*;


total_loss=��@

error_R��W?

learning_rate_1?a7�ڔI       6%�	^�6?���A�?*;


total_loss*��@

error_R�P?

learning_rate_1?a7L���I       6%�	&7?���A�?*;


total_lossq��@

error_R1�[?

learning_rate_1?a7F:{�I       6%�	�g7?���A�?*;


total_loss���@

error_R,�`?

learning_rate_1?a7��aI       6%�	+�7?���A�?*;


total_loss.��@

error_R�&[?

learning_rate_1?a7p�(cI       6%�	i�7?���A�?*;


total_loss�<�@

error_R�*F?

learning_rate_1?a7�
�I       6%�	k<8?���A�?*;


total_loss���@

error_R��\?

learning_rate_1?a7�O��I       6%�	�8?���A�?*;


total_loss�<�@

error_R<xo?

learning_rate_1?a7����I       6%�	��8?���A�?*;


total_loss]�@

error_R��J?

learning_rate_1?a7�;d�I       6%�	9?���A�?*;


total_loss��@

error_R�<?

learning_rate_1?a7��U�I       6%�	�Z9?���A�?*;


total_loss�	�@

error_Ri�8?

learning_rate_1?a7+�Y?I       6%�	p�9?���A�?*;


total_loss*�g@

error_R�6?

learning_rate_1?a7�6��I       6%�	��9?���A�?*;


total_loss<��@

error_R�I`?

learning_rate_1?a7�9�I       6%�	4!:?���A�?*;


total_loss��A

error_R��S?

learning_rate_1?a7D��I       6%�	�l:?���A�?*;


total_losset@

error_R�M?

learning_rate_1?a7���I       6%�	S�:?���A�?*;


total_loss]��@

error_R<�??

learning_rate_1?a7S���I       6%�	� ;?���A�?*;


total_loss�,�@

error_R)N?

learning_rate_1?a7'I       6%�	�F;?���A�?*;


total_loss�qA

error_R�U?

learning_rate_1?a7��^yI       6%�	��;?���A�?*;


total_lossC3�@

error_R�/Q?

learning_rate_1?a7
(�I       6%�	�;?���A�?*;


total_loss$A

error_R��N?

learning_rate_1?a7��6I       6%�	 <?���A�?*;


total_loss��@

error_R��=?

learning_rate_1?a7����I       6%�	\`<?���A�?*;


total_loss�p�@

error_Rf??

learning_rate_1?a7�hBGI       6%�	Ȥ<?���A�?*;


total_loss%��@

error_R&�??

learning_rate_1?a7�7p�I       6%�	��<?���A�?*;


total_loss,�@

error_R?5:?

learning_rate_1?a7��W�I       6%�	71=?���A�?*;


total_loss� �@

error_RN?

learning_rate_1?a7Q�I       6%�	�q=?���A�?*;


total_loss�¶@

error_RK?

learning_rate_1?a7��XI       6%�	�=?���A�?*;


total_lossQp�@

error_R;X?

learning_rate_1?a7 ���I       6%�	��=?���A�?*;


total_loss�0�@

error_R�S?

learning_rate_1?a73��I       6%�	mB>?���A�?*;


total_loss&I�@

error_R��E?

learning_rate_1?a7���I       6%�	��>?���A�?*;


total_lossce�@

error_R{�O?

learning_rate_1?a7zK1�I       6%�	��>?���A�?*;


total_loss���@

error_R��S?

learning_rate_1?a7�7��I       6%�	�??���A�?*;


total_loss`��@

error_R��D?

learning_rate_1?a7G1I�I       6%�	�e??���A�?*;


total_loss��@

error_R1G?

learning_rate_1?a7�_I       6%�	�??���A�?*;


total_loss�q�@

error_R��Q?

learning_rate_1?a7�"TJI       6%�	��??���A�?*;


total_loss��@

error_R$SZ?

learning_rate_1?a7v�#�I       6%�	�?@?���A�?*;


total_lossEA�@

error_R�L?

learning_rate_1?a7`��I       6%�	@�@?���A�?*;


total_loss�Ә@

error_R��Q?

learning_rate_1?a7Y�SI       6%�	��@?���A�?*;


total_loss ��@

error_R҉M?

learning_rate_1?a7��I       6%�	v	A?���A�?*;


total_loss?�@

error_RX�E?

learning_rate_1?a7W'�I       6%�	APA?���A�?*;


total_losscT�@

error_RTI?

learning_rate_1?a7}O;�I       6%�	V�A?���A�?*;


total_loss�A

error_R�-X?

learning_rate_1?a7�`k�I       6%�	��A?���A�?*;


total_loss�w@

error_RJ�=?

learning_rate_1?a7\@.�I       6%�	�B?���A�?*;


total_losse��@

error_R�i?

learning_rate_1?a7�vH�I       6%�	=aB?���A�?*;


total_loss�Q�@

error_R}�M?

learning_rate_1?a7�mu<I       6%�	(�B?���A�?*;


total_loss�@

error_Rs�E?

learning_rate_1?a7\�bI       6%�	d�B?���A�?*;


total_loss	ۥ@

error_R�@?

learning_rate_1?a7�}��I       6%�	1C?���A�?*;


total_loss{��@

error_R��Q?

learning_rate_1?a7�;�I       6%�	�sC?���A�?*;


total_loss���@

error_R�T?

learning_rate_1?a7c��I       6%�	�C?���A�?*;


total_loss=rA

error_R��W?

learning_rate_1?a7����I       6%�	�D?���A�?*;


total_loss��@

error_R�}^?

learning_rate_1?a7�0WI       6%�	END?���A�?*;


total_lossz6�@

error_R:�J?

learning_rate_1?a7:�P�I       6%�	�D?���A�?*;


total_loss���@

error_RT�B?

learning_rate_1?a7f��I       6%�	��D?���A�?*;


total_loss�)�@

error_R��[?

learning_rate_1?a7nz�BI       6%�	�'E?���A�?*;


total_lossiޞ@

error_Rn�>?

learning_rate_1?a7��xI       6%�	�jE?���A�?*;


total_loss�d�@

error_RTs@?

learning_rate_1?a7s�
�I       6%�	3�E?���A�?*;


total_loss�=�@

error_R�[S?

learning_rate_1?a7p��3I       6%�	F?���A�@*;


total_loss|�@

error_R�iI?

learning_rate_1?a7jݙ}I       6%�	KF?���A�@*;


total_loss�م@

error_R}�;?

learning_rate_1?a7렅cI       6%�	]�F?���A�@*;


total_loss���@

error_R�JB?

learning_rate_1?a7T��!I       6%�	��F?���A�@*;


total_loss�Dz@

error_R]?

learning_rate_1?a7Ylk�I       6%�	6AG?���A�@*;


total_lossL
�@

error_R�Y?

learning_rate_1?a7g-�I       6%�	��G?���A�@*;


total_loss=i�@

error_RW�L?

learning_rate_1?a7�e�I       6%�	��G?���A�@*;


total_lossĺ�@

error_R=8E?

learning_rate_1?a7d>�7I       6%�	�H?���A�@*;


total_loss ��@

error_R�TV?

learning_rate_1?a7`�yTI       6%�	]eH?���A�@*;


total_lossߦ@

error_R�H?

learning_rate_1?a7�ŹI       6%�	��H?���A�@*;


total_lossA

error_RD�P?

learning_rate_1?a7�I       6%�	:�H?���A�@*;


total_loss�@

error_R�T?

learning_rate_1?a7{�P�I       6%�	-I?���A�@*;


total_loss��@

error_R��Z?

learning_rate_1?a7X�g�I       6%�	doI?���A�@*;


total_loss%z@

error_RI	X?

learning_rate_1?a7��g7I       6%�	��I?���A�@*;


total_loss�M�@

error_Rx,S?

learning_rate_1?a72�LJI       6%�	~�I?���A�@*;


total_loss&�A

error_R�Q?

learning_rate_1?a7�UtI       6%�	�:J?���A�@*;


total_loss���@

error_R�}V?

learning_rate_1?a7��uI       6%�	��J?���A�@*;


total_loss6��@

error_R}oW?

learning_rate_1?a7�
I       6%�	�J?���A�@*;


total_loss��@

error_R8W>?

learning_rate_1?a7r�_I       6%�	K?���A�@*;


total_loss[V�@

error_R�wM?

learning_rate_1?a7qD��I       6%�	�WK?���A�@*;


total_loss�s�@

error_R��=?

learning_rate_1?a7��I�I       6%�	��K?���A�@*;


total_loss�ұ@

error_R)�J?

learning_rate_1?a7Y�e�I       6%�	8�K?���A�@*;


total_lossnE�@

error_R8�c?

learning_rate_1?a7kҲI       6%�	'#L?���A�@*;


total_loss�E�@

error_R�1P?

learning_rate_1?a7H��I       6%�	�hL?���A�@*;


total_loss�rK@

error_RI�H?

learning_rate_1?a7�>7WI       6%�	d�L?���A�@*;


total_lossdx�@

error_R�Rh?

learning_rate_1?a7���xI       6%�	R�L?���A�@*;


total_loss�(�@

error_Ra�>?

learning_rate_1?a7�c�kI       6%�	4M?���A�@*;


total_loss�� A

error_R� B?

learning_rate_1?a7�#��I       6%�	�wM?���A�@*;


total_lossAA�@

error_RW�C?

learning_rate_1?a7���I       6%�	��M?���A�@*;


total_loss*f�@

error_R�zK?

learning_rate_1?a7��|�I       6%�	��M?���A�@*;


total_loss���@

error_R�5Q?

learning_rate_1?a7�ܖKI       6%�	HN?���A�@*;


total_loss��@

error_R�.B?

learning_rate_1?a7�-��I       6%�	(�N?���A�@*;


total_loss�w�@

error_RJCE?

learning_rate_1?a7.�I       6%�	F�N?���A�@*;


total_loss�zm@

error_RiU?

learning_rate_1?a7���I       6%�	�<O?���A�@*;


total_lossH~�@

error_R��T?

learning_rate_1?a7�N�hI       6%�	z�O?���A�@*;


total_lossf��@

error_RC�U?

learning_rate_1?a7bt��I       6%�	��O?���A�@*;


total_loss=0�@

error_R��S?

learning_rate_1?a7F�4�I       6%�	�'P?���A�@*;


total_loss\#�@

error_R��b?

learning_rate_1?a7ɷP�I       6%�	nlP?���A�@*;


total_lossM�q@

error_RNH?

learning_rate_1?a7� ��I       6%�	��P?���A�@*;


total_lossQn�@

error_R "<?

learning_rate_1?a7�<��I       6%�	��P?���A�@*;


total_loss��
A

error_R=T?

learning_rate_1?a73;'.I       6%�	B6Q?���A�@*;


total_loss.��@

error_R��D?

learning_rate_1?a7}�I       6%�	�xQ?���A�@*;


total_loss,H�@

error_Rt�j?

learning_rate_1?a7Z:ԖI       6%�	��Q?���A�@*;


total_loss?��@

error_R��=?

learning_rate_1?a7w6�I       6%�	]R?���A�@*;


total_loss(�@

error_RS�P?

learning_rate_1?a7L��I       6%�	�FR?���A�@*;


total_loss��@

error_Rf�O?

learning_rate_1?a7�Ƅ%I       6%�	��R?���A�@*;


total_loss�hA

error_Rl�S?

learning_rate_1?a7���>I       6%�	V�R?���A�@*;


total_loss�ۋ@

error_R_�>?

learning_rate_1?a7w�II       6%�	�S?���A�@*;


total_loss� A

error_R�7J?

learning_rate_1?a7�:��I       6%�	8]S?���A�@*;


total_loss���@

error_R��H?

learning_rate_1?a7�r��I       6%�	��S?���A�@*;


total_loss~�@

error_R��N?

learning_rate_1?a7���MI       6%�	��S?���A�@*;


total_loss�N�@

error_R�gT?

learning_rate_1?a7u��I       6%�	_,T?���A�@*;


total_loss���@

error_R�oB?

learning_rate_1?a7pX(I       6%�	�sT?���A�@*;


total_loss}��@

error_R�Q?

learning_rate_1?a7�n)I       6%�	سT?���A�@*;


total_loss�w�@

error_R&K?

learning_rate_1?a7�t/�I       6%�	��T?���A�@*;


total_lossCF�@

error_R_<?

learning_rate_1?a7����I       6%�	�;U?���A�@*;


total_lossXm�@

error_R�W?

learning_rate_1?a7JʠsI       6%�	+�U?���A�@*;


total_loss4��@

error_Rx�Q?

learning_rate_1?a7%,�I       6%�	 �U?���A�@*;


total_loss�P�@

error_RɅT?

learning_rate_1?a7o�V�I       6%�	-V?���A�@*;


total_loss���@

error_R�II?

learning_rate_1?a7o�I       6%�	L^V?���A�@*;


total_loss�8�@

error_R.A?

learning_rate_1?a7�B�gI       6%�	�V?���A�@*;


total_lossO��@

error_R?PF?

learning_rate_1?a7��I       6%�	�W?���A�@*;


total_loss�*�@

error_R�5?

learning_rate_1?a7J�VqI       6%�		IW?���A�@*;


total_lossrL�@

error_R*=?

learning_rate_1?a7+g��I       6%�	�W?���A�@*;


total_loss]��@

error_R&a@?

learning_rate_1?a7�� �I       6%�	w�W?���A�@*;


total_loss?��@

error_R@�X?

learning_rate_1?a7^*n?I       6%�	�X?���A�@*;


total_loss��@

error_Rv@K?

learning_rate_1?a7��IFI       6%�	haX?���A�@*;


total_loss���@

error_R�K?

learning_rate_1?a7�	M�I       6%�	��X?���A�@*;


total_lossp��@

error_R}�F?

learning_rate_1?a7!**�I       6%�	��X?���A�@*;


total_loss@

error_RQ�b?

learning_rate_1?a7���I       6%�	5Y?���A�@*;


total_lossF��@

error_R��Y?

learning_rate_1?a7�#�SI       6%�	wY?���A�@*;


total_loss��@

error_RE�Y?

learning_rate_1?a7[Y3I       6%�	f�Y?���A�@*;


total_loss���@

error_R=iJ?

learning_rate_1?a7�_��I       6%�	�Z?���A�@*;


total_loss��@

error_R��M?

learning_rate_1?a7�	�I       6%�	KLZ?���A�@*;


total_loss�Z�@

error_R�a>?

learning_rate_1?a7 9�I       6%�	�Z?���A�@*;


total_lossC��@

error_RJ?

learning_rate_1?a7��*6I       6%�	<�Z?���A�@*;


total_loss�r�@

error_R�L?

learning_rate_1?a7}H�$I       6%�	�[?���A�@*;


total_loss<A

error_Rv�K?

learning_rate_1?a7����I       6%�	X[?���A�@*;


total_loss*�@

error_R�6P?

learning_rate_1?a7�-��I       6%�	Q�[?���A�@*;


total_loss�U�@

error_RJ�F?

learning_rate_1?a7����I       6%�	��[?���A�@*;


total_loss�Ⱥ@

error_R$�I?

learning_rate_1?a7����I       6%�	�!\?���A�@*;


total_loss���@

error_R�E:?

learning_rate_1?a7��̓I       6%�	Qe\?���A�@*;


total_loss���@

error_Re�D?

learning_rate_1?a7��KI       6%�	�\?���A�@*;


total_loss4��@

error_R��Z?

learning_rate_1?a7:�=I       6%�	�\?���A�@*;


total_loss���@

error_R��R?

learning_rate_1?a7�aZ�I       6%�	�4]?���A�@*;


total_loss�o�@

error_R<C?

learning_rate_1?a7~�SI       6%�	Dy]?���A�@*;


total_loss%A

error_RJ�C?

learning_rate_1?a7q��!I       6%�	5�]?���A�@*;


total_loss�d�@

error_R=Q?

learning_rate_1?a7�4'I       6%�	z�]?���A�@*;


total_loss=y�@

error_R��q?

learning_rate_1?a7nwKI       6%�	>C^?���A�@*;


total_loss��@

error_RI[?

learning_rate_1?a7�e�I       6%�	�^?���A�@*;


total_loss��A

error_R�l?

learning_rate_1?a7�B;EI       6%�	X�^?���A�@*;


total_lossOZ�@

error_R�J?

learning_rate_1?a7��I       6%�	�_?���A�@*;


total_loss���@

error_R�F?

learning_rate_1?a7����I       6%�	�Q_?���A�@*;


total_loss���@

error_R6=?

learning_rate_1?a7!l�YI       6%�	=�_?���A�@*;


total_loss��@

error_R�>_?

learning_rate_1?a7��KI       6%�	��_?���A�@*;


total_loss�Y�@

error_R�43?

learning_rate_1?a7��h[I       6%�	9.`?���A�@*;


total_lossX%�@

error_R��E?

learning_rate_1?a7Q5>I       6%�	6w`?���A�@*;


total_loss���@

error_R�2L?

learning_rate_1?a7j�I       6%�	w�`?���A�@*;


total_loss�~A

error_R\_N?

learning_rate_1?a7Kmb�I       6%�	S�`?���A�@*;


total_loss�0�@

error_R$�J?

learning_rate_1?a7�zp�I       6%�	�Aa?���A�@*;


total_loss.�.A

error_R�e@?

learning_rate_1?a7��+_I       6%�	'�a?���A�@*;


total_loss�L�@

error_R�]m?

learning_rate_1?a75��?I       6%�	X�a?���A�@*;


total_loss�;�@

error_R�b?

learning_rate_1?a7Vw��I       6%�	�b?���A�@*;


total_loss⌇@

error_R!�S?

learning_rate_1?a7z��zI       6%�	�Yb?���A�@*;


total_loss)��@

error_R3a??

learning_rate_1?a7Y;y�I       6%�	N�b?���A�@*;


total_loss)ͥ@

error_RfNH?

learning_rate_1?a73,�I       6%�	�b?���A�@*;


total_loss���@

error_R�dQ?

learning_rate_1?a7[S�I       6%�	�,c?���A�@*;


total_loss���@

error_RC�Q?

learning_rate_1?a7�JquI       6%�	wc?���A�@*;


total_lossCz�@

error_R�XZ?

learning_rate_1?a7y��II       6%�	�c?���A�@*;


total_loss6b�@

error_R�QS?

learning_rate_1?a7�;sI       6%�	�d?���A�@*;


total_loss#X�@

error_R�I?

learning_rate_1?a7��QI       6%�	=Fd?���A�@*;


total_loss��}@

error_R8�J?

learning_rate_1?a7s���I       6%�	ދd?���A�@*;


total_loss
�@

error_Rܜ\?

learning_rate_1?a7��+I       6%�	Y�d?���A�@*;


total_lossN�@

error_R$�S?

learning_rate_1?a7Ñ��I       6%�	e?���A�@*;


total_loss8��@

error_R)sS?

learning_rate_1?a7��jI       6%�	|Ue?���A�@*;


total_loss�"�@

error_R�ZI?

learning_rate_1?a7W�I       6%�	ۗe?���A�@*;


total_loss��@

error_R�rV?

learning_rate_1?a7�1�I       6%�	��e?���A�@*;


total_loss)��@

error_Rt�D?

learning_rate_1?a7��__I       6%�	>f?���A�@*;


total_loss��@

error_RH�F?

learning_rate_1?a7�s~�I       6%�	�af?���A�@*;


total_loss��@

error_R׽E?

learning_rate_1?a7)��I       6%�	��f?���A�@*;


total_loss��b@

error_RFG?

learning_rate_1?a7�~�bI       6%�	�g?���A�@*;


total_loss�:�@

error_R#�L?

learning_rate_1?a7���I       6%�	�Zg?���A�@*;


total_loss�Y�@

error_R�D?

learning_rate_1?a7%�ZI       6%�	��g?���A�@*;


total_loss؉�@

error_R�d?

learning_rate_1?a7O��I       6%�	1�g?���A�@*;


total_loss�Q�@

error_R�,?

learning_rate_1?a7��EzI       6%�	�%h?���A�@*;


total_loss4c�@

error_Rs{8?

learning_rate_1?a7)J
�I       6%�	�lh?���A�@*;


total_lossZ7A

error_R��S?

learning_rate_1?a7�>�RI       6%�	��h?���A�@*;


total_loss��@

error_R�U?

learning_rate_1?a7� 5I       6%�	��h?���A�@*;


total_lossh��@

error_RWPP?

learning_rate_1?a7=��PI       6%�	�1i?���A�A*;


total_loss<}�@

error_Ra�K?

learning_rate_1?a7"�KI       6%�	�ri?���A�A*;


total_loss�%�@

error_R��X?

learning_rate_1?a7_�e9I       6%�	��i?���A�A*;


total_loss,�@

error_R]�_?

learning_rate_1?a73�v�I       6%�	j?���A�A*;


total_lossMV�@

error_R{LQ?

learning_rate_1?a7���4I       6%�	�Fj?���A�A*;


total_loss���@

error_RT]?

learning_rate_1?a7�О�I       6%�	f�j?���A�A*;


total_loss�͔@

error_R��P?

learning_rate_1?a7�Y�I       6%�	��j?���A�A*;


total_loss���@

error_R�_?

learning_rate_1?a7�֥nI       6%�	�k?���A�A*;


total_loss=0�@

error_Rx8?

learning_rate_1?a701^I       6%�	[k?���A�A*;


total_loss�ώ@

error_R@?J?

learning_rate_1?a7���I       6%�	�k?���A�A*;


total_loss�j�@

error_R,iU?

learning_rate_1?a7V�SuI       6%�	"�k?���A�A*;


total_loss&γ@

error_RlL?

learning_rate_1?a7�l�I       6%�	12l?���A�A*;


total_loss��@

error_R��C?

learning_rate_1?a7ћhI       6%�	&yl?���A�A*;


total_lossQ��@

error_R�tV?

learning_rate_1?a7���I       6%�	��l?���A�A*;


total_loss��@

error_R>+?

learning_rate_1?a7T�yfI       6%�	�m?���A�A*;


total_lossc
�@

error_Rq�Z?

learning_rate_1?a7���wI       6%�	�Fm?���A�A*;


total_lossڅ�@

error_RZ?

learning_rate_1?a7�j��I       6%�	�m?���A�A*;


total_loss��@

error_R�MR?

learning_rate_1?a7/���I       6%�	`�m?���A�A*;


total_lossf��@

error_R��S?

learning_rate_1?a7̄��I       6%�	?n?���A�A*;


total_loss��@

error_RCH?

learning_rate_1?a78k�I       6%�	g�n?���A�A*;


total_loss3��@

error_R��^?

learning_rate_1?a7l��<I       6%�	:�n?���A�A*;


total_loss2N�@

error_R�b9?

learning_rate_1?a7FvT@I       6%�	�o?���A�A*;


total_loss,/�@

error_R�LF?

learning_rate_1?a7Z�zI       6%�	�lo?���A�A*;


total_loss��d@

error_R�U9?

learning_rate_1?a7��[&I       6%�	ָo?���A�A*;


total_lossJ�@

error_RD&:?

learning_rate_1?a7�)6I       6%�	�o?���A�A*;


total_loss��@

error_Rf�^?

learning_rate_1?a7
ȋ%I       6%�	uHp?���A�A*;


total_loss�͘@

error_R�E?

learning_rate_1?a7T��I       6%�	Ȓp?���A�A*;


total_loss_2�@

error_RO�^?

learning_rate_1?a7�XζI       6%�	r�p?���A�A*;


total_lossΏ@

error_R�][?

learning_rate_1?a7u��2I       6%�	&q?���A�A*;


total_lossF�@

error_Rj�L?

learning_rate_1?a7�:�I       6%�	qq?���A�A*;


total_lossW��@

error_RZr9?

learning_rate_1?a7x ��I       6%�	ɸq?���A�A*;


total_lossE��@

error_RJ?

learning_rate_1?a7�q3�I       6%�	�r?���A�A*;


total_loss �@

error_R�p^?

learning_rate_1?a7t��CI       6%�	�Mr?���A�A*;


total_loss�;�@

error_R�\?

learning_rate_1?a7h�P�I       6%�	ʘr?���A�A*;


total_lossU�@

error_R��I?

learning_rate_1?a7x�),I       6%�	��r?���A�A*;


total_loss$�`@

error_R�;?

learning_rate_1?a7|	g|I       6%�	�s?���A�A*;


total_loss��r@

error_R�,Z?

learning_rate_1?a7�⟺I       6%�	ks?���A�A*;


total_loss�@

error_R�Wi?

learning_rate_1?a7�|,I       6%�	��s?���A�A*;


total_loss3è@

error_RZE?

learning_rate_1?a7��I       6%�	}t?���A�A*;


total_loss��@

error_RM?[?

learning_rate_1?a7cJ�'I       6%�	�St?���A�A*;


total_loss1ݞ@

error_R��K?

learning_rate_1?a7]�-I       6%�	�t?���A�A*;


total_loss�/�@

error_RO�L?

learning_rate_1?a76lB%I       6%�	��t?���A�A*;


total_loss�`�@

error_R��P?

learning_rate_1?a7(�I       6%�	�,u?���A�A*;


total_loss��A

error_R�XA?

learning_rate_1?a7���I       6%�	�vu?���A�A*;


total_loss��@

error_R�'W?

learning_rate_1?a7�A��I       6%�	E�u?���A�A*;


total_loss��@

error_R��i?

learning_rate_1?a7`?�I       6%�	�	v?���A�A*;


total_loss}��@

error_R(�E?

learning_rate_1?a7��I       6%�	fWv?���A�A*;


total_loss8��@

error_Rm�h?

learning_rate_1?a7��a#I       6%�	+�v?���A�A*;


total_loss�ݴ@

error_R��U?

learning_rate_1?a7����I       6%�	�	w?���A�A*;


total_loss�k@

error_RҥA?

learning_rate_1?a7��rI       6%�	�Pw?���A�A*;


total_loss�F�@

error_R��T?

learning_rate_1?a7'�CI       6%�	��w?���A�A*;


total_lossD/�@

error_R��F?

learning_rate_1?a7V��I       6%�	M�w?���A�A*;


total_loss�0�@

error_R@RQ?

learning_rate_1?a7R���I       6%�	�(x?���A�A*;


total_loss��@

error_R�<G?

learning_rate_1?a7&�I       6%�	�px?���A�A*;


total_loss)��@

error_R�O?

learning_rate_1?a7�[�I       6%�	��x?���A�A*;


total_loss���@

error_R��??

learning_rate_1?a7�E�I       6%�	h�x?���A�A*;


total_loss���@

error_RR[?

learning_rate_1?a7�q��I       6%�	�Ay?���A�A*;


total_loss8^�@

error_R��Y?

learning_rate_1?a7lPI       6%�	��y?���A�A*;


total_loss�X�@

error_R�^E?

learning_rate_1?a7y�*@I       6%�	�y?���A�A*;


total_loss_�J@

error_R2=h?

learning_rate_1?a7��kI       6%�	z?���A�A*;


total_loss�L�@

error_RLK?

learning_rate_1?a7[}V;I       6%�	�Wz?���A�A*;


total_loss��@

error_R!_?

learning_rate_1?a7��I       6%�	��z?���A�A*;


total_loss@��@

error_RӡT?

learning_rate_1?a7~q�"I       6%�	[�z?���A�A*;


total_loss�G@

error_R� W?

learning_rate_1?a7h�U�I       6%�	�#{?���A�A*;


total_loss�\�@

error_R��@?

learning_rate_1?a7{���I       6%�	�g{?���A�A*;


total_loss<��@

error_R)'Q?

learning_rate_1?a7	�8�I       6%�	��{?���A�A*;


total_loss/Ou@

error_R� R?

learning_rate_1?a7q;	�I       6%�	��{?���A�A*;


total_loss\ѿ@

error_R26_?

learning_rate_1?a7�*I       6%�	�=|?���A�A*;


total_losscA�@

error_R��A?

learning_rate_1?a76U!I       6%�	��|?���A�A*;


total_loss�@

error_R�+O?

learning_rate_1?a7�ѼBI       6%�	��|?���A�A*;


total_loss��@

error_R�xH?

learning_rate_1?a7	�9lI       6%�	1}?���A�A*;


total_loss�3�@

error_Rm[P?

learning_rate_1?a7'c!RI       6%�	4o}?���A�A*;


total_loss��@

error_R��D?

learning_rate_1?a7mS}I       6%�	��}?���A�A*;


total_lossC��@

error_Rx�J?

learning_rate_1?a7I�okI       6%�	f~?���A�A*;


total_loss�r�@

error_R��Z?

learning_rate_1?a7�RrI       6%�	)O~?���A�A*;


total_lossx	�@

error_R�PD?

learning_rate_1?a7�M�yI       6%�	D�~?���A�A*;


total_loss���@

error_R G?

learning_rate_1?a7�V;�I       6%�	��~?���A�A*;


total_loss��@

error_R�d?

learning_rate_1?a7�*XI       6%�	& ?���A�A*;


total_loss_�@

error_R�kN?

learning_rate_1?a7qx�I       6%�	f?���A�A*;


total_lossT��@

error_R3[I?

learning_rate_1?a7�f+7I       6%�	˫?���A�A*;


total_loss^�@

error_R��T?

learning_rate_1?a7���I       6%�	��?���A�A*;


total_lossT��@

error_RO�R?

learning_rate_1?a72w�QI       6%�	b9�?���A�A*;


total_lossx!�@

error_R��I?

learning_rate_1?a7�ZgI       6%�	�}�?���A�A*;


total_loss���@

error_R8FF?

learning_rate_1?a7ORZ�I       6%�	iƀ?���A�A*;


total_loss<��@

error_R�wL?

learning_rate_1?a7�o@eI       6%�	��?���A�A*;


total_loss���@

error_R�Y?

learning_rate_1?a7)u��I       6%�	�L�?���A�A*;


total_lossxqA

error_RژH?

learning_rate_1?a7� �mI       6%�	��?���A�A*;


total_loss�z@

error_R�O?

learning_rate_1?a7��/I       6%�	SӁ?���A�A*;


total_lossq�@

error_R��H?

learning_rate_1?a7\jNI       6%�	@�?���A�A*;


total_loss:W A

error_R=<L?

learning_rate_1?a7qj?�I       6%�	3g�?���A�A*;


total_loss���@

error_R�K?

learning_rate_1?a70�XI       6%�	��?���A�A*;


total_loss��@

error_RJ�R?

learning_rate_1?a7~�*=I       6%�	_��?���A�A*;


total_loss!�@

error_R�~A?

learning_rate_1?a7� 9�I       6%�	tE�?���A�A*;


total_losso��@

error_R�JX?

learning_rate_1?a7X��I       6%�	z��?���A�A*;


total_loss�#�@

error_R�9L?

learning_rate_1?a7�E�,I       6%�	�΃?���A�A*;


total_loss�ߤ@

error_R.�@?

learning_rate_1?a7�EjI       6%�	u�?���A�A*;


total_loss�ʯ@

error_R|�R?

learning_rate_1?a7��Q�I       6%�	�[�?���A�A*;


total_loss��@

error_R	?X?

learning_rate_1?a7���QI       6%�	���?���A�A*;


total_lossԫm@

error_R��E?

learning_rate_1?a7X \�I       6%�	x�?���A�A*;


total_loss�ݹ@

error_R܃Q?

learning_rate_1?a7!rNnI       6%�	�%�?���A�A*;


total_losse�t@

error_R!�V?

learning_rate_1?a7-��I       6%�	Xl�?���A�A*;


total_lossx��@

error_R�-T?

learning_rate_1?a7{�DI       6%�	���?���A�A*;


total_loss���@

error_R#1B?

learning_rate_1?a7��9I       6%�	!��?���A�A*;


total_loss��@

error_R �Q?

learning_rate_1?a7�b�xI       6%�	�C�?���A�A*;


total_loss�~�@

error_R�IM?

learning_rate_1?a7�)�I       6%�	���?���A�A*;


total_loss��h@

error_RM�i?

learning_rate_1?a7�"��I       6%�	��?���A�A*;


total_loss�V�@

error_R4T?

learning_rate_1?a7D6�I       6%�	�/�?���A�A*;


total_loss ?�@

error_R�@M?

learning_rate_1?a7S��I       6%�	Iv�?���A�A*;


total_loss\��@

error_R\�Q?

learning_rate_1?a7�Q�I       6%�	��?���A�A*;


total_lossn�@

error_R�SK?

learning_rate_1?a7�X�I       6%�	M�?���A�A*;


total_loss�w�@

error_R �Q?

learning_rate_1?a7dP}I       6%�	UF�?���A�A*;


total_lossߌ@

error_R}LI?

learning_rate_1?a7�I       6%�	��?���A�A*;


total_loss��@

error_R�J?

learning_rate_1?a7&��I       6%�	�҈?���A�A*;


total_loss�ݴ@

error_R�LF?

learning_rate_1?a7�֣�I       6%�	'�?���A�A*;


total_lossJ��@

error_RR�i?

learning_rate_1?a7o��ZI       6%�	T^�?���A�A*;


total_loss|�A

error_R^G?

learning_rate_1?a7y��%I       6%�	���?���A�A*;


total_loss�A

error_RÍW?

learning_rate_1?a7���I       6%�	�?���A�A*;


total_loss/ͭ@

error_RDa?

learning_rate_1?a7�4I       6%�	�8�?���A�A*;


total_losshX�@

error_RM�E?

learning_rate_1?a7�9��I       6%�	&�?���A�A*;


total_losslj�@

error_R�>?

learning_rate_1?a7ֻKAI       6%�	�?���A�A*;


total_loss:Z�@

error_R��W?

learning_rate_1?a72oI       6%�	.�?���A�A*;


total_loss��@

error_RM�/?

learning_rate_1?a7�I�I       6%�	~I�?���A�A*;


total_loss?A

error_R�NK?

learning_rate_1?a7V�I       6%�	Ꮛ?���A�A*;


total_loss#�@

error_RZ�O?

learning_rate_1?a7�ϊSI       6%�	�Ӌ?���A�A*;


total_lossV��@

error_R�(T?

learning_rate_1?a7�9��I       6%�	��?���A�A*;


total_loss_�@

error_R�<B?

learning_rate_1?a76��I       6%�	�]�?���A�A*;


total_loss���@

error_R$i9?

learning_rate_1?a7B��vI       6%�	_��?���A�A*;


total_loss�G�@

error_Rc�K?

learning_rate_1?a7h�m�I       6%�	��?���A�A*;


total_loss1��@

error_Ra4N?

learning_rate_1?a7HA�I       6%�	�.�?���A�B*;


total_loss��@

error_R �@?

learning_rate_1?a7v�^I       6%�	mu�?���A�B*;


total_losszȗ@

error_R|WM?

learning_rate_1?a7F���I       6%�	^��?���A�B*;


total_loss�Ƈ@

error_R�$Z?

learning_rate_1?a71�sI       6%�	{	�?���A�B*;


total_loss���@

error_RqUX?

learning_rate_1?a7����I       6%�	.P�?���A�B*;


total_lossZ��@

error_R=qM?

learning_rate_1?a7	@�#I       6%�	���?���A�B*;


total_loss���@

error_R�<C?

learning_rate_1?a7��7aI       6%�	>َ?���A�B*;


total_loss�߱@

error_R�P]?

learning_rate_1?a7�v4�I       6%�	%#�?���A�B*;


total_loss$��@

error_RV)Y?

learning_rate_1?a7[ө"I       6%�	$s�?���A�B*;


total_lossz8�@

error_R�D?

learning_rate_1?a7$��I       6%�	��?���A�B*;


total_lossr�@

error_R�CH?

learning_rate_1?a7�ߋ�I       6%�	��?���A�B*;


total_lossF�A

error_RC0Y?

learning_rate_1?a7!ɲI       6%�	�B�?���A�B*;


total_loss���@

error_R�GG?

learning_rate_1?a7YĕdI       6%�	 ��?���A�B*;


total_lossQ��@

error_R_�Y?

learning_rate_1?a7��$I       6%�	:ǐ?���A�B*;


total_loss�T�@

error_R:�P?

learning_rate_1?a7��"�I       6%�	)�?���A�B*;


total_loss���@

error_R�'Q?

learning_rate_1?a7�ÒfI       6%�	�_�?���A�B*;


total_loss���@

error_R��D?

learning_rate_1?a7]�I       6%�	���?���A�B*;


total_loss2Q�@

error_R�H?

learning_rate_1?a7�p;I       6%�	��?���A�B*;


total_loss!�@

error_R��Z?

learning_rate_1?a7�xD%I       6%�	�<�?���A�B*;


total_loss��@

error_R��??

learning_rate_1?a7�;�I       6%�	���?���A�B*;


total_loss[Q�@

error_Rv�I?

learning_rate_1?a7G� �I       6%�	@͒?���A�B*;


total_loss��h@

error_RA�C?

learning_rate_1?a7�,$I       6%�	��?���A�B*;


total_loss��	A

error_R�T?

learning_rate_1?a7;<�I       6%�	cj�?���A�B*;


total_lossڵ�@

error_Re|M?

learning_rate_1?a7t�i�I       6%�	$��?���A�B*;


total_loss{�@

error_R{�X?

learning_rate_1?a7�S��I       6%�	���?���A�B*;


total_loss3�A

error_R�^?

learning_rate_1?a7PAJI       6%�	�B�?���A�B*;


total_loss���@

error_R|F?

learning_rate_1?a7�)�I       6%�	���?���A�B*;


total_loss��@

error_R*]?

learning_rate_1?a7�k�I       6%�	̔?���A�B*;


total_lossя�@

error_R�[?

learning_rate_1?a7����I       6%�	��?���A�B*;


total_loss���@

error_R��P?

learning_rate_1?a7�?M=I       6%�	�Y�?���A�B*;


total_loss�E�@

error_Ri�R?

learning_rate_1?a7p�MI       6%�	���?���A�B*;


total_loss��@

error_R��E?

learning_rate_1?a7�$�YI       6%�	��?���A�B*;


total_loss���@

error_R�[F?

learning_rate_1?a7�c�I       6%�	�.�?���A�B*;


total_losse��@

error_R�K?

learning_rate_1?a7#��]I       6%�	�q�?���A�B*;


total_loss�(�@

error_R�\?

learning_rate_1?a74�]1I       6%�	�Ӗ?���A�B*;


total_loss�ʐ@

error_R�LQ?

learning_rate_1?a7�^�I       6%�	.�?���A�B*;


total_loss��~@

error_R=OM?

learning_rate_1?a7�^��I       6%�	v�?���A�B*;


total_lossϟ�@

error_RیH?

learning_rate_1?a7��nI       6%�	���?���A�B*;


total_lossх@

error_R�D?

learning_rate_1?a7�[�oI       6%�	��?���A�B*;


total_loss���@

error_R�F?

learning_rate_1?a7/��4I       6%�	�G�?���A�B*;


total_lossQ��@

error_R��W?

learning_rate_1?a7�p�LI       6%�	$��?���A�B*;


total_lossT��@

error_R��M?

learning_rate_1?a7pV�"I       6%�	ʘ?���A�B*;


total_lossz�@

error_R6�N?

learning_rate_1?a7�q��I       6%�	��?���A�B*;


total_loss���@

error_R�U?

learning_rate_1?a7��O�I       6%�	�S�?���A�B*;


total_lossq��@

error_R��M?

learning_rate_1?a7�|7�I       6%�	K��?���A�B*;


total_loss\��@

error_Rײ`?

learning_rate_1?a7��8I       6%�	$ؙ?���A�B*;


total_loss.�@

error_R��??

learning_rate_1?a7��I       6%�	��?���A�B*;


total_loss�?�@

error_R�M?

learning_rate_1?a7ƻ�I       6%�	"]�?���A�B*;


total_loss&׾@

error_R�O?

learning_rate_1?a7��QJI       6%�	���?���A�B*;


total_lossh5�@

error_R�pV?

learning_rate_1?a7����I       6%�	�ߚ?���A�B*;


total_loss|��@

error_Rv9?

learning_rate_1?a7�[�&I       6%�	�"�?���A�B*;


total_loss���@

error_RϖP?

learning_rate_1?a7��0�I       6%�	#e�?���A�B*;


total_lossч�@

error_R�0=?

learning_rate_1?a7c-,RI       6%�		��?���A�B*;


total_loss���@

error_R� Z?

learning_rate_1?a7A�I       6%�	��?���A�B*;


total_loss�`@

error_R�gH?

learning_rate_1?a7n��pI       6%�	l2�?���A�B*;


total_loss�9�@

error_RQ�2?

learning_rate_1?a7�I       6%�	du�?���A�B*;


total_loss���@

error_R�:?

learning_rate_1?a7��]vI       6%�	��?���A�B*;


total_loss���@

error_RlZM?

learning_rate_1?a7�˿�I       6%�	^�?���A�B*;


total_lossx��@

error_R�W?

learning_rate_1?a7`v I       6%�	5I�?���A�B*;


total_loss<�@

error_RE�N?

learning_rate_1?a7�(�|I       6%�	#��?���A�B*;


total_loss)��@

error_R�&_?

learning_rate_1?a7�Ǭ;I       6%�	��?���A�B*;


total_loss�.�@

error_R�{]?

learning_rate_1?a78SI       6%�	B9�?���A�B*;


total_loss���@

error_RiC?

learning_rate_1?a7�F�I       6%�	3~�?���A�B*;


total_loss]�@

error_Rn K?

learning_rate_1?a7>�?I       6%�	�Ğ?���A�B*;


total_loss�p�@

error_RρA?

learning_rate_1?a7D6�1I       6%�	K�?���A�B*;


total_lossT�@

error_ROs\?

learning_rate_1?a7�5�yI       6%�	BS�?���A�B*;


total_loss�'�@

error_RM�V?

learning_rate_1?a7�k,?I       6%�	k��?���A�B*;


total_loss&D�@

error_R��[?

learning_rate_1?a7{�I       6%�	X��?���A�B*;


total_loss���@

error_R�Y?

learning_rate_1?a7<)�I       6%�	,$�?���A�B*;


total_losse�@

error_R��]?

learning_rate_1?a7�nI       6%�	�f�?���A�B*;


total_lossd:A

error_Ri$L?

learning_rate_1?a7�*��I       6%�	J��?���A�B*;


total_lossC�@

error_R�V?

learning_rate_1?a7b6�I       6%�	m�?���A�B*;


total_loss73�@

error_R��G?

learning_rate_1?a7�D4I       6%�	M9�?���A�B*;


total_loss<A

error_Rz�O?

learning_rate_1?a7���I       6%�	��?���A�B*;


total_loss_�f@

error_R	�F?

learning_rate_1?a7G��I       6%�	�ʡ?���A�B*;


total_lossxS�@

error_R�'L?

learning_rate_1?a7P'�aI       6%�	��?���A�B*;


total_loss㫗@

error_R�8?

learning_rate_1?a7Yd=�I       6%�	�_�?���A�B*;


total_loss\4�@

error_R��;?

learning_rate_1?a7I^5�I       6%�	d��?���A�B*;


total_loss_��@

error_R��a?

learning_rate_1?a7|W�>I       6%�	��?���A�B*;


total_loss!�@

error_Rr�=?

learning_rate_1?a7u�K�I       6%�	.�?���A�B*;


total_loss��@

error_R�Pn?

learning_rate_1?a7�׶I       6%�	�q�?���A�B*;


total_loss���@

error_RŒa?

learning_rate_1?a7{��I       6%�	���?���A�B*;


total_loss���@

error_RZ=U?

learning_rate_1?a7��VI       6%�	��?���A�B*;


total_loss@J�@

error_RdA?

learning_rate_1?a7���I       6%�	_@�?���A�B*;


total_lossq�@

error_R�?Z?

learning_rate_1?a7�W I       6%�	키?���A�B*;


total_loss�`x@

error_REx[?

learning_rate_1?a7W��cI       6%�	�Ť?���A�B*;


total_loss �@

error_R�vU?

learning_rate_1?a7#�)�I       6%�	��?���A�B*;


total_loss��@

error_R�zR?

learning_rate_1?a7�*�+I       6%�	U�?���A�B*;


total_lossx�A

error_R��G?

learning_rate_1?a7�G��I       6%�	8��?���A�B*;


total_loss���@

error_R�N?

learning_rate_1?a7H=7I       6%�	��?���A�B*;


total_loss��@

error_R}�N?

learning_rate_1?a7�b$I       6%�	�%�?���A�B*;


total_loss*f�@

error_Rv�F?

learning_rate_1?a7���I       6%�	gh�?���A�B*;


total_loss/ �@

error_R��R?

learning_rate_1?a7i��/I       6%�	찦?���A�B*;


total_loss�~�@

error_R\�V?

learning_rate_1?a7xq8&I       6%�	�?���A�B*;


total_loss= �@

error_R�B?

learning_rate_1?a7[kI       6%�	8P�?���A�B*;


total_loss��@

error_R�tN?

learning_rate_1?a7(�I       6%�	w��?���A�B*;


total_lossie@

error_R�-F?

learning_rate_1?a7O�c-I       6%�	�ۧ?���A�B*;


total_loss�<�@

error_R��K?

learning_rate_1?a7�A�UI       6%�	�&�?���A�B*;


total_loss$�@

error_R.K?

learning_rate_1?a76�P�I       6%�	�r�?���A�B*;


total_loss)[�@

error_R
WJ?

learning_rate_1?a7���I       6%�	���?���A�B*;


total_loss��@

error_R �A?

learning_rate_1?a7)!6I       6%�	���?���A�B*;


total_loss3'�@

error_R��??

learning_rate_1?a7�Dv�I       6%�	�=�?���A�B*;


total_loss.IA

error_R`�O?

learning_rate_1?a7���I       6%�	���?���A�B*;


total_loss �@

error_R6E?

learning_rate_1?a7#�I       6%�	vʩ?���A�B*;


total_loss(�]@

error_R�O?

learning_rate_1?a7
L�I       6%�	_�?���A�B*;


total_loss���@

error_R��g?

learning_rate_1?a7��?I       6%�	�R�?���A�B*;


total_lossA'�@

error_R�]?

learning_rate_1?a7׷EI       6%�	'��?���A�B*;


total_loss&��@

error_R8KX?

learning_rate_1?a7k��CI       6%�	��?���A�B*;


total_loss���@

error_R��S?

learning_rate_1?a7�v��I       6%�	�)�?���A�B*;


total_loss���@

error_R�nS?

learning_rate_1?a7� ��I       6%�	k�?���A�B*;


total_lossox�@

error_Ri�R?

learning_rate_1?a7����I       6%�	��?���A�B*;


total_lossL!�@

error_R,:?

learning_rate_1?a7*:;I       6%�	 �?���A�B*;


total_lossz*�@

error_RE�X?

learning_rate_1?a7�ށ�I       6%�	9�?���A�B*;


total_loss�"�@

error_R�T?

learning_rate_1?a7s���I       6%�	%�?���A�B*;


total_loss%�@

error_R�C?

learning_rate_1?a7UQ��I       6%�	�ì?���A�B*;


total_lossC��@

error_R{ J?

learning_rate_1?a7��I       6%�	u�?���A�B*;


total_loss�f�@

error_R��E?

learning_rate_1?a7_r�xI       6%�	CJ�?���A�B*;


total_lossx��@

error_R�U?

learning_rate_1?a7&j�BI       6%�	;��?���A�B*;


total_loss�D�@

error_R��T?

learning_rate_1?a7�i�yI       6%�	�Э?���A�B*;


total_loss���@

error_R�)<?

learning_rate_1?a76�ZI       6%�	e�?���A�B*;


total_loss���@

error_R�N?

learning_rate_1?a7�W��I       6%�	FU�?���A�B*;


total_loss;��@

error_R�eO?

learning_rate_1?a7ɧ�I       6%�	뚮?���A�B*;


total_loss�6�@

error_R�R?

learning_rate_1?a7�F�I       6%�	�ݮ?���A�B*;


total_loss�f�@

error_R��J?

learning_rate_1?a7�w
I       6%�	��?���A�B*;


total_loss��@

error_R�U?

learning_rate_1?a7��DKI       6%�	+a�?���A�B*;


total_loss��@

error_R�S?

learning_rate_1?a7:s��I       6%�	 ��?���A�B*;


total_lossч�@

error_Rd�P?

learning_rate_1?a7���I       6%�	��?���A�B*;


total_lossA�@

error_Rn�X?

learning_rate_1?a7�N�I       6%�	^/�?���A�B*;


total_loss)�@

error_R_H?

learning_rate_1?a7���I       6%�	t�?���A�C*;


total_loss�H�@

error_R�F?

learning_rate_1?a7�Qq�I       6%�	���?���A�C*;


total_loss��@

error_R�7]?

learning_rate_1?a7m���I       6%�	���?���A�C*;


total_loss���@

error_R��B?

learning_rate_1?a7�͡�I       6%�	E�?���A�C*;


total_loss
�@

error_R��M?

learning_rate_1?a7q�P�I       6%�	���?���A�C*;


total_loss�b�@

error_R��[?

learning_rate_1?a7^�Z�I       6%�	�ױ?���A�C*;


total_loss4��@

error_Rz=N?

learning_rate_1?a7��"I       6%�	�#�?���A�C*;


total_loss���@

error_R��F?

learning_rate_1?a7�}j\I       6%�	�k�?���A�C*;


total_lossn�v@

error_Rs�C?

learning_rate_1?a7��I       6%�	%��?���A�C*;


total_lossRmx@

error_R.F`?

learning_rate_1?a7_���I       6%�	���?���A�C*;


total_lossJ��@

error_R��Q?

learning_rate_1?a7��ԈI       6%�	r<�?���A�C*;


total_loss�3�@

error_RM�C?

learning_rate_1?a7��I       6%�	]��?���A�C*;


total_loss�S�@

error_R8=N?

learning_rate_1?a7���|I       6%�	�ó?���A�C*;


total_loss�|�@

error_R��O?

learning_rate_1?a7��$I       6%�	r�?���A�C*;


total_loss{T�@

error_R��Z?

learning_rate_1?a7!V�I       6%�	J�?���A�C*;


total_loss��@

error_R�`O?

learning_rate_1?a7k`r�I       6%�	��?���A�C*;


total_lossNL�@

error_R.�F?

learning_rate_1?a7̖�'I       6%�	�Դ?���A�C*;


total_loss�˕@

error_R!bQ?

learning_rate_1?a7���I       6%�	��?���A�C*;


total_loss��@

error_R�QK?

learning_rate_1?a7��QI       6%�	)W�?���A�C*;


total_loss_'�@

error_R�;T?

learning_rate_1?a7֦cI       6%�	���?���A�C*;


total_loss��@

error_R��F?

learning_rate_1?a7����I       6%�	8޵?���A�C*;


total_loss���@

error_R}�J?

learning_rate_1?a7(�QI       6%�	�&�?���A�C*;


total_lossث�@

error_R�bU?

learning_rate_1?a7op�[I       6%�	�i�?���A�C*;


total_loss���@

error_R��H?

learning_rate_1?a7nĬZI       6%�	���?���A�C*;


total_loss̣�@

error_R#+[?

learning_rate_1?a7����I       6%�	�3�?���A�C*;


total_lossa0�@

error_R<TQ?

learning_rate_1?a7��ҜI       6%�	�|�?���A�C*;


total_loss�p�@

error_R�oN?

learning_rate_1?a7�hW�I       6%�	���?���A�C*;


total_lossL��@

error_R��S?

learning_rate_1?a7�c��I       6%�	��?���A�C*;


total_loss���@

error_R��V?

learning_rate_1?a7���XI       6%�	SK�?���A�C*;


total_loss��@

error_R��M?

learning_rate_1?a7�XpI       6%�	=��?���A�C*;


total_loss��@

error_R�E?

learning_rate_1?a7�i�I       6%�	TԸ?���A�C*;


total_loss��@

error_RژN?

learning_rate_1?a7�$I       6%�	��?���A�C*;


total_loss[ö@

error_R!�Z?

learning_rate_1?a7�K�NI       6%�	�[�?���A�C*;


total_loss���@

error_R��C?

learning_rate_1?a7YA*�I       6%�	���?���A�C*;


total_loss�A

error_RfmF?

learning_rate_1?a7�]SI       6%�	�޹?���A�C*;


total_loss =�@

error_R��c?

learning_rate_1?a7�֦�I       6%�	+"�?���A�C*;


total_loss��@

error_RcI?

learning_rate_1?a7c'.I       6%�	f�?���A�C*;


total_loss�9�@

error_R]?

learning_rate_1?a7�ٖ�I       6%�	J��?���A�C*;


total_loss͹�@

error_Rͱ&?

learning_rate_1?a7�P	I       6%�	��?���A�C*;


total_lossm��@

error_Ri�Y?

learning_rate_1?a7��q�I       6%�	#-�?���A�C*;


total_loss�2�@

error_RvTc?

learning_rate_1?a7h��I       6%�	�r�?���A�C*;


total_lossX�A

error_R��I?

learning_rate_1?a7��s�I       6%�	]��?���A�C*;


total_loss}��@

error_RȨe?

learning_rate_1?a7��I�I       6%�	��?���A�C*;


total_lossx�@

error_R
M?

learning_rate_1?a7L��hI       6%�	�Q�?���A�C*;


total_lossq܁@

error_R�SR?

learning_rate_1?a73��I       6%�	���?���A�C*;


total_losso�@

error_R�U?

learning_rate_1?a7�
�I       6%�	��?���A�C*;


total_loss�]�@

error_R []?

learning_rate_1?a7Ս�#I       6%�	�,�?���A�C*;


total_losss��@

error_Rx}d?

learning_rate_1?a7��(;I       6%�	]y�?���A�C*;


total_lossy�A

error_Rv�C?

learning_rate_1?a7+:[I       6%�	�˽?���A�C*;


total_loss�U�@

error_RzYN?

learning_rate_1?a7t�0�I       6%�	a�?���A�C*;


total_loss�Ţ@

error_R@ZK?

learning_rate_1?a7���VI       6%�	�_�?���A�C*;


total_lossrq�@

error_RlbK?

learning_rate_1?a7�eC�I       6%�	ͧ�?���A�C*;


total_loss|��@

error_R��V?

learning_rate_1?a7��8I       6%�	���?���A�C*;


total_loss7��@

error_Rq�e?

learning_rate_1?a7����I       6%�	�1�?���A�C*;


total_loss�]�@

error_RWV?

learning_rate_1?a7��(�I       6%�	pu�?���A�C*;


total_lossn��@

error_RO�c?

learning_rate_1?a7/F	I       6%�	{��?���A�C*;


total_lossl��@

error_R��C?

learning_rate_1?a7�a��I       6%�	���?���A�C*;


total_loss���@

error_R�P?

learning_rate_1?a7���I       6%�	�D�?���A�C*;


total_lossI��@

error_R�=?

learning_rate_1?a7�K�I       6%�	���?���A�C*;


total_loss[|�@

error_R�T?

learning_rate_1?a7b<I       6%�	���?���A�C*;


total_loss�A

error_R��U?

learning_rate_1?a7���I       6%�	��?���A�C*;


total_loss�/�@

error_RxuO?

learning_rate_1?a7@<'"I       6%�	|c�?���A�C*;


total_loss�@

error_R�U?

learning_rate_1?a7�8�qI       6%�	���?���A�C*;


total_loss��@

error_RtY?

learning_rate_1?a7���wI       6%�	h��?���A�C*;


total_loss�R�@

error_R)w??

learning_rate_1?a7��zI       6%�	�9�?���A�C*;


total_lossn̋@

error_R�_M?

learning_rate_1?a7����I       6%�	1��?���A�C*;


total_loss&�@

error_Ri�>?

learning_rate_1?a7\�Y�I       6%�	Y��?���A�C*;


total_lossn֯@

error_R�#C?

learning_rate_1?a7S�OI       6%�	n�?���A�C*;


total_loss@H�@

error_R�NZ?

learning_rate_1?a7�ë�I       6%�	"G�?���A�C*;


total_losstYo@

error_R�g]?

learning_rate_1?a7��,�I       6%�	���?���A�C*;


total_loss&֞@

error_RdiQ?

learning_rate_1?a7X�_�I       6%�	 ��?���A�C*;


total_lossۜA

error_R��G?

learning_rate_1?a7��@�I       6%�	�?���A�C*;


total_loss���@

error_R��8?

learning_rate_1?a7�Z�I       6%�	�T�?���A�C*;


total_loss���@

error_R��G?

learning_rate_1?a7��-�I       6%�	T��?���A�C*;


total_loss�oA

error_R�[\?

learning_rate_1?a7���I       6%�	C��?���A�C*;


total_loss�Qj@

error_R �6?

learning_rate_1?a7��'I       6%�	�$�?���A�C*;


total_loss�ޝ@

error_R�A?

learning_rate_1?a7�d�I       6%�	an�?���A�C*;


total_loss@ѧ@

error_Rt�K?

learning_rate_1?a7��fI       6%�	ǵ�?���A�C*;


total_loss��@

error_R�XZ?

learning_rate_1?a7�� I       6%�	G��?���A�C*;


total_loss24�@

error_R�l5?

learning_rate_1?a7�AH0I       6%�	oA�?���A�C*;


total_loss�Ht@

error_RqyL?

learning_rate_1?a7��7I       6%�	���?���A�C*;


total_loss�ӿ@

error_RA�f?

learning_rate_1?a7�
��I       6%�	_��?���A�C*;


total_loss���@

error_R,�b?

learning_rate_1?a7("�I       6%�	�A�?���A�C*;


total_loss�� A

error_R;�O?

learning_rate_1?a7���I       6%�	 ��?���A�C*;


total_loss=ƶ@

error_R\=O?

learning_rate_1?a7P��I       6%�	u��?���A�C*;


total_lossw��@

error_R�/F?

learning_rate_1?a7 u�I       6%�	O �?���A�C*;


total_loss��w@

error_R��D?

learning_rate_1?a7�.��I       6%�	i�?���A�C*;


total_loss���@

error_R�9P?

learning_rate_1?a7K�O�I       6%�	��?���A�C*;


total_loss<��@

error_Rf%@?

learning_rate_1?a7���-I       6%�	���?���A�C*;


total_loss���@

error_RT�U?

learning_rate_1?a7�L�hI       6%�	�6�?���A�C*;


total_loss
ܻ@

error_R�IC?

learning_rate_1?a7U=�_I       6%�	]w�?���A�C*;


total_loss�@

error_R2C?

learning_rate_1?a7�2ԠI       6%�	���?���A�C*;


total_loss��@

error_RW�S?

learning_rate_1?a7 uzhI       6%�	���?���A�C*;


total_loss���@

error_R�2O?

learning_rate_1?a7�SI       6%�	�E�?���A�C*;


total_loss���@

error_R�zR?

learning_rate_1?a7�C�I       6%�	׎�?���A�C*;


total_lossr?�@

error_R,Y?

learning_rate_1?a7�eI       6%�	���?���A�C*;


total_loss���@

error_R�\?

learning_rate_1?a7�!��I       6%�	U�?���A�C*;


total_loss��{@

error_R�D?

learning_rate_1?a7 �GKI       6%�	�]�?���A�C*;


total_lossT��@

error_R�3?

learning_rate_1?a7}�+I       6%�	Ԩ�?���A�C*;


total_lossMĽ@

error_RT?M?

learning_rate_1?a7D�=)I       6%�	��?���A�C*;


total_loss���@

error_R�a?

learning_rate_1?a7��[I       6%�	)>�?���A�C*;


total_loss=��@

error_R��A?

learning_rate_1?a7S7O�I       6%�	���?���A�C*;


total_loss϶�@

error_R�D?

learning_rate_1?a7��j&I       6%�	��?���A�C*;


total_loss�n�@

error_Rx�F?

learning_rate_1?a7�+]HI       6%�	��?���A�C*;


total_loss��@

error_R1�<?

learning_rate_1?a7���I       6%�	S�?���A�C*;


total_loss=�d@

error_RmP?

learning_rate_1?a7jmKI       6%�	z��?���A�C*;


total_loss�I�@

error_R��R?

learning_rate_1?a7����I       6%�	���?���A�C*;


total_loss]R�@

error_R_N?

learning_rate_1?a7�'+I       6%�	#�?���A�C*;


total_lossr٭@

error_R��Y?

learning_rate_1?a7F[�I       6%�	$_�?���A�C*;


total_loss��@

error_R�EE?

learning_rate_1?a7�Q��I       6%�	3��?���A�C*;


total_loss�*A

error_R��M?

learning_rate_1?a7끁I       6%�	���?���A�C*;


total_lossc��@

error_RRjA?

learning_rate_1?a7��4I       6%�	80�?���A�C*;


total_loss��A

error_R#�Z?

learning_rate_1?a7zy�I       6%�	�t�?���A�C*;


total_loss��@

error_R$kA?

learning_rate_1?a7�,I       6%�	J��?���A�C*;


total_loss��@

error_R�oW?

learning_rate_1?a7�NձI       6%�	y��?���A�C*;


total_loss�)�@

error_R�dO?

learning_rate_1?a7�o�6I       6%�	�E�?���A�C*;


total_loss3��@

error_R��G?

learning_rate_1?a7ꛯxI       6%�	��?���A�C*;


total_loss�d�@

error_R�=W?

learning_rate_1?a7���>I       6%�	3��?���A�C*;


total_loss��@

error_Rx�T?

learning_rate_1?a7K�GI       6%�	��?���A�C*;


total_loss���@

error_R�4O?

learning_rate_1?a7�D�I       6%�	�U�?���A�C*;


total_loss���@

error_R{�??

learning_rate_1?a7�hI       6%�	���?���A�C*;


total_loss���@

error_R��\?

learning_rate_1?a7�U� I       6%�	���?���A�C*;


total_lossa<�@

error_R�{W?

learning_rate_1?a7l�I       6%�	�/�?���A�C*;


total_loss_�@

error_RBO?

learning_rate_1?a7��YI       6%�		z�?���A�C*;


total_loss��@

error_R�9?

learning_rate_1?a7كj�I       6%�	���?���A�C*;


total_loss4�@

error_R�MZ?

learning_rate_1?a7�O<�I       6%�	f�?���A�C*;


total_loss�D�@

error_R�+R?

learning_rate_1?a7����I       6%�	�\�?���A�C*;


total_loss1�@

error_R�A?

learning_rate_1?a7��7�I       6%�	��?���A�C*;


total_loss�P4A

error_R�hG?

learning_rate_1?a7�1&�I       6%�	�?���A�D*;


total_loss��A

error_R��V?

learning_rate_1?a7��.�I       6%�	QO�?���A�D*;


total_losso A

error_R��M?

learning_rate_1?a7ݶ?I       6%�	I��?���A�D*;


total_loss1��@

error_R�W?

learning_rate_1?a7�yI       6%�	���?���A�D*;


total_loss�M�@

error_R�\?

learning_rate_1?a7C��I       6%�	�+�?���A�D*;


total_loss��@

error_R�KB?

learning_rate_1?a7f���I       6%�	8n�?���A�D*;


total_loss��@

error_R��E?

learning_rate_1?a7a
��I       6%�	k��?���A�D*;


total_losss�@

error_R��E?

learning_rate_1?a7[���I       6%�	���?���A�D*;


total_lossf��@

error_R�I?

learning_rate_1?a7���I       6%�	=�?���A�D*;


total_loss7��@

error_R!8Q?

learning_rate_1?a7p�$I       6%�	'��?���A�D*;


total_loss�ц@

error_R;�H?

learning_rate_1?a7��8�I       6%�	���?���A�D*;


total_loss�W�@

error_RAKE?

learning_rate_1?a7Z�n*I       6%�	q;�?���A�D*;


total_loss���@

error_R�??

learning_rate_1?a7[C�+I       6%�	��?���A�D*;


total_loss�B�@

error_R*�R?

learning_rate_1?a7 	�+I       6%�	���?���A�D*;


total_loss�\�@

error_R�=?

learning_rate_1?a7�uI       6%�	,�?���A�D*;


total_loss�%�@

error_RHOI?

learning_rate_1?a7��[ I       6%�	FM�?���A�D*;


total_loss��x@

error_R)�Q?

learning_rate_1?a7'�eI       6%�	Y��?���A�D*;


total_loss�}A

error_R\�W?

learning_rate_1?a7����I       6%�	��?���A�D*;


total_lossd<�@

error_RWtQ?

learning_rate_1?a7����I       6%�	��?���A�D*;


total_loss:f�@

error_RV?

learning_rate_1?a7��BI       6%�	vY�?���A�D*;


total_lossX�@

error_RO�N?

learning_rate_1?a7���I       6%�	���?���A�D*;


total_lossw؈@

error_RZFB?

learning_rate_1?a7qN��I       6%�	���?���A�D*;


total_lossG�@

error_R�X?

learning_rate_1?a7)-}I       6%�	$�?���A�D*;


total_loss{�@

error_R�S?

learning_rate_1?a7�h��I       6%�	i�?���A�D*;


total_loss&'�@

error_R״X?

learning_rate_1?a7�>�GI       6%�	��?���A�D*;


total_loss�v�@

error_RW�a?

learning_rate_1?a7f��I       6%�	���?���A�D*;


total_loss>�@

error_R�tM?

learning_rate_1?a7&p�I       6%�	�?�?���A�D*;


total_loss.�@

error_R۱>?

learning_rate_1?a7�5��I       6%�	r��?���A�D*;


total_loss� �@

error_RV�9?

learning_rate_1?a7�7��I       6%�	���?���A�D*;


total_loss�]�@

error_R�A?

learning_rate_1?a7�QS�I       6%�	K�?���A�D*;


total_loss.�@

error_R��`?

learning_rate_1?a7�bL�I       6%�	�[�?���A�D*;


total_loss�!�@

error_R��@?

learning_rate_1?a7ӌF�I       6%�	���?���A�D*;


total_loss���@

error_R� Q?

learning_rate_1?a7�Т�I       6%�	4��?���A�D*;


total_lossڠ@

error_R}FA?

learning_rate_1?a7&@9�I       6%�	lG�?���A�D*;


total_loss�S�@

error_R8)O?

learning_rate_1?a7���I       6%�	E��?���A�D*;


total_loss?�A

error_R��h?

learning_rate_1?a72b9I       6%�	���?���A�D*;


total_lossi��@

error_R��^?

learning_rate_1?a7v�X�I       6%�	j%�?���A�D*;


total_loss.�@

error_R��G?

learning_rate_1?a7�|�I       6%�	�l�?���A�D*;


total_loss�c�@

error_R��Y?

learning_rate_1?a75��I       6%�	:��?���A�D*;


total_losszb�@

error_R�=?

learning_rate_1?a7�>w�I       6%�	9�?���A�D*;


total_lossQ܆@

error_R�T?

learning_rate_1?a71�\9I       6%�	�H�?���A�D*;


total_loss���@

error_R��;?

learning_rate_1?a7��(�I       6%�	���?���A�D*;


total_lossf��@

error_R�@O?

learning_rate_1?a7|x�I       6%�	���?���A�D*;


total_loss���@

error_Ri6?

learning_rate_1?a7%s
I       6%�	FR�?���A�D*;


total_loss6�@

error_R�OC?

learning_rate_1?a7p��.I       6%�	���?���A�D*;


total_loss�#a@

error_R`iU?

learning_rate_1?a7���I       6%�	���?���A�D*;


total_loss���@

error_R��E?

learning_rate_1?a7nf�?I       6%�	�#�?���A�D*;


total_lossCP�@

error_RM3E?

learning_rate_1?a7r�I       6%�	�h�?���A�D*;


total_loss��@

error_R�Ua?

learning_rate_1?a7/��I       6%�	Ʊ�?���A�D*;


total_loss���@

error_R.Q?

learning_rate_1?a7��>�I       6%�	N��?���A�D*;


total_loss�=�@

error_Rc�B?

learning_rate_1?a7���I       6%�	2:�?���A�D*;


total_loss}g�@

error_R,2I?

learning_rate_1?a7��0�I       6%�	�~�?���A�D*;


total_loss��@

error_R��\?

learning_rate_1?a7�7��I       6%�	��?���A�D*;


total_loss�	�@

error_R�xF?

learning_rate_1?a7n���I       6%�	��?���A�D*;


total_loss��@

error_RWHS?

learning_rate_1?a7��I       6%�	�J�?���A�D*;


total_loss�}�@

error_R l?

learning_rate_1?a7ީ��I       6%�	��?���A�D*;


total_loss�4�@

error_R�J`?

learning_rate_1?a7%:0<I       6%�	���?���A�D*;


total_lossQ�@

error_RVkV?

learning_rate_1?a7�6�I       6%�	�?���A�D*;


total_loss�I�@

error_RmY?

learning_rate_1?a7σ9EI       6%�	�b�?���A�D*;


total_loss���@

error_RGX?

learning_rate_1?a7b&�I       6%�	ϩ�?���A�D*;


total_loss�c�@

error_RW�P?

learning_rate_1?a7_�I       6%�	���?���A�D*;


total_loss�n�@

error_R,�O?

learning_rate_1?a7B��II       6%�	�0�?���A�D*;


total_loss=k�@

error_R��U?

learning_rate_1?a7�r�I       6%�	;s�?���A�D*;


total_loss��@

error_R�U?

learning_rate_1?a7:xGI       6%�	X��?���A�D*;


total_loss�%u@

error_R��O?

learning_rate_1?a7d�~�I       6%�	�?���A�D*;


total_loss64�@

error_R�C?

learning_rate_1?a7>�;�I       6%�	@O�?���A�D*;


total_loss���@

error_R��O?

learning_rate_1?a7Y[ikI       6%�	��?���A�D*;


total_loss��@

error_R2V?

learning_rate_1?a7h^�HI       6%�	���?���A�D*;


total_loss|�@

error_R�pO?

learning_rate_1?a7���I       6%�	YF�?���A�D*;


total_losso��@

error_RC�B?

learning_rate_1?a7��EuI       6%�	���?���A�D*;


total_loss��@

error_R��S?

learning_rate_1?a7Q7��I       6%�	���?���A�D*;


total_lossVg�@

error_Rf�X?

learning_rate_1?a7V�1I       6%�	��?���A�D*;


total_lossQ��@

error_R�X?

learning_rate_1?a7y�O3I       6%�	�b�?���A�D*;


total_loss
F�@

error_R_�R?

learning_rate_1?a7����I       6%�	���?���A�D*;


total_loss_��@

error_Ri�F?

learning_rate_1?a7��H�I       6%�	3��?���A�D*;


total_loss�i�@

error_R�97?

learning_rate_1?a7��I       6%�	b1�?���A�D*;


total_loss��@

error_R��>?

learning_rate_1?a7q��I       6%�	Bu�?���A�D*;


total_loss�Q�@

error_R�bO?

learning_rate_1?a7��t�I       6%�	���?���A�D*;


total_loss�=g@

error_R)cD?

learning_rate_1?a7ZE`,I       6%�	��?���A�D*;


total_loss���@

error_Rh�[?

learning_rate_1?a7�$A�I       6%�	�C�?���A�D*;


total_loss� �@

error_R�5?

learning_rate_1?a7&�I       6%�	Ѕ�?���A�D*;


total_loss�h�@

error_R}>S?

learning_rate_1?a7-�4<I       6%�	(��?���A�D*;


total_loss��A

error_R��F?

learning_rate_1?a7�ĽUI       6%�	d�?���A�D*;


total_loss�!�@

error_RV<K?

learning_rate_1?a7�-�I       6%�	�O�?���A�D*;


total_loss���@

error_RsC?

learning_rate_1?a7z��LI       6%�	��?���A�D*;


total_loss�X�@

error_R�yX?

learning_rate_1?a7r��I       6%�	���?���A�D*;


total_loss���@

error_R)(K?

learning_rate_1?a7^<��I       6%�	$�?���A�D*;


total_lossi A

error_R�:N?

learning_rate_1?a7[c��I       6%�	�b�?���A�D*;


total_loss}C�@

error_R;-E?

learning_rate_1?a7֤�I       6%�	D��?���A�D*;


total_lossqy�@

error_R��N?

learning_rate_1?a7�ߑMI       6%�	���?���A�D*;


total_loss���@

error_Ra�K?

learning_rate_1?a7�LTLI       6%�	�3�?���A�D*;


total_loss��@

error_R#�R?

learning_rate_1?a7���I       6%�	Tx�?���A�D*;


total_loss�-�@

error_R��I?

learning_rate_1?a7��#�I       6%�	9��?���A�D*;


total_loss��@

error_R��D?

learning_rate_1?a7fd�I       6%�	e�?���A�D*;


total_lossz��@

error_R�9D?

learning_rate_1?a7��6�I       6%�	 K�?���A�D*;


total_loss���@

error_R�LR?

learning_rate_1?a7K��GI       6%�	���?���A�D*;


total_loss��@

error_RRzX?

learning_rate_1?a7B�w#I       6%�	2��?���A�D*;


total_loss�PW@

error_RsLH?

learning_rate_1?a7�ˮI       6%�	B�?���A�D*;


total_loss�#_@

error_R�H?

learning_rate_1?a7��E�I       6%�	�d�?���A�D*;


total_loss��@

error_Ra�U?

learning_rate_1?a7�­*I       6%�	$��?���A�D*;


total_lossa�1A

error_R�Wa?

learning_rate_1?a7'�&oI       6%�	y��?���A�D*;


total_loss���@

error_R�vP?

learning_rate_1?a7��|I       6%�	�3�?���A�D*;


total_loss���@

error_RE�>?

learning_rate_1?a7̮2I       6%�	�v�?���A�D*;


total_loss\B�@

error_R�G?

learning_rate_1?a7�I       6%�	u��?���A�D*;


total_lossS�@

error_R�B?

learning_rate_1?a7D��I       6%�	&��?���A�D*;


total_loss84�@

error_R��T?

learning_rate_1?a7}��NI       6%�	�B�?���A�D*;


total_loss\m�@

error_R#!=?

learning_rate_1?a7d�5I       6%�	��?���A�D*;


total_lossA�@

error_R��U?

learning_rate_1?a7�LnI       6%�	���?���A�D*;


total_loss��@

error_Rs9W?

learning_rate_1?a7;�[�I       6%�	P�?���A�D*;


total_loss6�@

error_R	�??

learning_rate_1?a76�sI       6%�	�W�?���A�D*;


total_lossaQ�@

error_R<�B?

learning_rate_1?a7ܘR�I       6%�	{��?���A�D*;


total_loss���@

error_Ri�D?

learning_rate_1?a7��I       6%�	���?���A�D*;


total_lossrq�@

error_R�iI?

learning_rate_1?a7[��4I       6%�	d%�?���A�D*;


total_loss	��@

error_R{�Q?

learning_rate_1?a7;�o�I       6%�	�i�?���A�D*;


total_loss��V@

error_R�K?

learning_rate_1?a7���I       6%�	��?���A�D*;


total_loss)�@

error_R_g;?

learning_rate_1?a7�bJI       6%�	���?���A�D*;


total_loss$l�@

error_R�M?

learning_rate_1?a7b�n�I       6%�	�8�?���A�D*;


total_loss�!�@

error_R��O?

learning_rate_1?a7���I       6%�	���?���A�D*;


total_loss�9�@

error_R�,E?

learning_rate_1?a7�t�!I       6%�	���?���A�D*;


total_loss8��@

error_RL�B?

learning_rate_1?a7���I       6%�	A�?���A�D*;


total_loss��@

error_R�EB?

learning_rate_1?a7vK�I       6%�	�P�?���A�D*;


total_loss��@

error_R��N?

learning_rate_1?a7�p��I       6%�	A��?���A�D*;


total_lossO�@

error_R��V?

learning_rate_1?a7.i)I       6%�	��?���A�D*;


total_loss?��@

error_R1)O?

learning_rate_1?a7òYMI       6%�	�_�?���A�D*;


total_loss⩖@

error_R$�F?

learning_rate_1?a7e]�I       6%�	��?���A�D*;


total_loss�<�@

error_RS�Z?

learning_rate_1?a75�pnI       6%�	�	�?���A�D*;


total_loss���@

error_R�
[?

learning_rate_1?a7:d޻I       6%�	dZ�?���A�D*;


total_lossO��@

error_R�[Q?

learning_rate_1?a7���TI       6%�	��?���A�D*;


total_loss���@

error_Rh]`?

learning_rate_1?a7���!I       6%�	���?���A�E*;


total_lossE_�@

error_R��K?

learning_rate_1?a7|i�I       6%�	�+�?���A�E*;


total_lossAw�@

error_R*�Z?

learning_rate_1?a78n(<I       6%�	�s�?���A�E*;


total_loss���@

error_R8R??

learning_rate_1?a7�Z��I       6%�	���?���A�E*;


total_lossat�@

error_R.w<?

learning_rate_1?a7x�BI       6%�	���?���A�E*;


total_loss٭@

error_R&�@?

learning_rate_1?a7S�D]I       6%�	�C�?���A�E*;


total_loss0��@

error_R��\?

learning_rate_1?a7��K0I       6%�	���?���A�E*;


total_loss���@

error_R�BJ?

learning_rate_1?a7u�N/I       6%�	a��?���A�E*;


total_loss��@

error_R��b?

learning_rate_1?a7(I1I       6%�	@�?���A�E*;


total_lossvѱ@

error_RawQ?

learning_rate_1?a7ynX4I       6%�	K�?���A�E*;


total_loss���@

error_R�=?

learning_rate_1?a71�]aI       6%�	���?���A�E*;


total_loss�@

error_R%�d?

learning_rate_1?a7���I       6%�	���?���A�E*;


total_loss.�@

error_R/R?

learning_rate_1?a7��%I       6%�	��?���A�E*;


total_loss	�@

error_Ro�R?

learning_rate_1?a7P�I       6%�	LX�?���A�E*;


total_loss��U@

error_RWK?

learning_rate_1?a7��>I       6%�	���?���A�E*;


total_loss6��@

error_R�:W?

learning_rate_1?a7���I       6%�	D��?���A�E*;


total_loss1��@

error_R��Y?

learning_rate_1?a7T�T_I       6%�	o4�?���A�E*;


total_loss.�@

error_R�G?

learning_rate_1?a7r��I       6%�	�|�?���A�E*;


total_loss��@

error_R��Z?

learning_rate_1?a7+6�	I       6%�	���?���A�E*;


total_loss�*�@

error_R��Y?

learning_rate_1?a7�p��I       6%�	��?���A�E*;


total_loss:Ֆ@

error_Rf�>?

learning_rate_1?a7�
P�I       6%�	�^�?���A�E*;


total_loss�gA

error_RW�T?

learning_rate_1?a7g�fI       6%�	T��?���A�E*;


total_loss#��@

error_RƮL?

learning_rate_1?a7+�I       6%�	��?���A�E*;


total_loss��f@

error_RC�F?

learning_rate_1?a7�̫NI       6%�	�)�?���A�E*;


total_loss�݌@

error_R.'F?

learning_rate_1?a7��9�I       6%�	fj�?���A�E*;


total_lossHO�@

error_R��J?

learning_rate_1?a7aWs�I       6%�	���?���A�E*;


total_loss�p�@

error_R�;?

learning_rate_1?a7S��I       6%�	��?���A�E*;


total_loss�"�@

error_R�`L?

learning_rate_1?a7m���I       6%�	�1�?���A�E*;


total_loss��@

error_Rf;?

learning_rate_1?a7?�WvI       6%�	�t�?���A�E*;


total_loss�@

error_RxXQ?

learning_rate_1?a7~"��I       6%�	`��?���A�E*;


total_loss��@

error_RW�O?

learning_rate_1?a7�}�I       6%�	���?���A�E*;


total_loss��@

error_R��@?

learning_rate_1?a7�ԺbI       6%�	�F @���A�E*;


total_loss��@

error_R��X?

learning_rate_1?a7vQ�WI       6%�	Ή @���A�E*;


total_lossw��@

error_Rf�C?

learning_rate_1?a7�_�I       6%�	d� @���A�E*;


total_loss�s�@

error_R�z:?

learning_rate_1?a7U��I       6%�	�@���A�E*;


total_loss���@

error_R3!P?

learning_rate_1?a7���I       6%�	^@���A�E*;


total_lossv"�@

error_R��T?

learning_rate_1?a7����I       6%�	�@���A�E*;


total_loss��@

error_Ri�S?

learning_rate_1?a7���I       6%�	w
@���A�E*;


total_lossq��@

error_R� K?

learning_rate_1?a7D[U1I       6%�	�Z@���A�E*;


total_loss["�@

error_R�r=?

learning_rate_1?a7d{BUI       6%�	��@���A�E*;


total_loss���@

error_R1?

learning_rate_1?a7�#7�I       6%�	��@���A�E*;


total_loss=�~@

error_R3�I?

learning_rate_1?a7�cI       6%�	�7@���A�E*;


total_lossd�@

error_R@?

learning_rate_1?a7m�+�I       6%�	�y@���A�E*;


total_loss���@

error_R�bS?

learning_rate_1?a7��I       6%�	(�@���A�E*;


total_loss���@

error_Rbg?

learning_rate_1?a7���I       6%�	@���A�E*;


total_loss|�@

error_RnE?

learning_rate_1?a7:}I       6%�	VH@���A�E*;


total_loss��@

error_R��g?

learning_rate_1?a7
��WI       6%�	<�@���A�E*;


total_loss%�@

error_R�??

learning_rate_1?a7&S�uI       6%�	��@���A�E*;


total_loss�@

error_R)0=?

learning_rate_1?a7��CI       6%�	�@���A�E*;


total_loss��@

error_R��K?

learning_rate_1?a7�{��I       6%�	fV@���A�E*;


total_loss�� A

error_R�2Z?

learning_rate_1?a7'kU�I       6%�	Ü@���A�E*;


total_loss�n�@

error_R��F?

learning_rate_1?a7�A�I       6%�	��@���A�E*;


total_loss*n�@

error_R�Y?

learning_rate_1?a7��W�I       6%�	�"@���A�E*;


total_loss��@

error_R��O?

learning_rate_1?a7�ʥI       6%�	g@���A�E*;


total_lossz��@

error_RM�S?

learning_rate_1?a7�
�I       6%�	��@���A�E*;


total_loss#K�@

error_R�f?

learning_rate_1?a7��zI       6%�	�@���A�E*;


total_loss��@

error_R�Z?

learning_rate_1?a7��9I       6%�	�U@���A�E*;


total_lossڻ�@

error_R!�A?

learning_rate_1?a7��*I       6%�	��@���A�E*;


total_loss��@

error_R�}-?

learning_rate_1?a7���I       6%�	�@���A�E*;


total_loss{ĭ@

error_R�pQ?

learning_rate_1?a7J>c�I       6%�	�-@���A�E*;


total_loss��@

error_R�G?

learning_rate_1?a7X�WLI       6%�	'r@���A�E*;


total_loss:��@

error_R�?U?

learning_rate_1?a7���~I       6%�	=�@���A�E*;


total_lossw��@

error_RV�J?

learning_rate_1?a7�e�|I       6%�	�	@���A�E*;


total_lossu�@

error_R`tJ?

learning_rate_1?a7�u,�I       6%�	�D	@���A�E*;


total_lossΘ�@

error_R��e?

learning_rate_1?a7H�UI       6%�	��	@���A�E*;


total_loss6�@

error_R�O?

learning_rate_1?a7u-�YI       6%�	��	@���A�E*;


total_loss��o@

error_RA&J?

learning_rate_1?a7,E��I       6%�	A
@���A�E*;


total_loss�έ@

error_Rx9U?

learning_rate_1?a7��` I       6%�	�e
@���A�E*;


total_lossӔ@

error_R��K?

learning_rate_1?a7R�I       6%�	�
@���A�E*;


total_loss�&�@

error_R{�V?

learning_rate_1?a7���I       6%�	�
@���A�E*;


total_loss:>�@

error_RhsQ?

learning_rate_1?a7yb�bI       6%�	9@���A�E*;


total_lossv��@

error_R
LA?

learning_rate_1?a7��y�I       6%�	�}@���A�E*;


total_loss�+�@

error_R�T?

learning_rate_1?a7}��$I       6%�	V�@���A�E*;


total_loss\8�@

error_R��D?

learning_rate_1?a7��0?I       6%�	+@���A�E*;


total_loss��@

error_R�L?

learning_rate_1?a7�V5I       6%�	�H@���A�E*;


total_lossQu�@

error_R
"I?

learning_rate_1?a7�	=I       6%�	��@���A�E*;


total_lossH�@

error_RʉL?

learning_rate_1?a7�I       6%�	N�@���A�E*;


total_loss=3A

error_RI?

learning_rate_1?a7�|�I       6%�	�@���A�E*;


total_lossf�@

error_R��F?

learning_rate_1?a7[u3!I       6%�	S^@���A�E*;


total_loss�K�@

error_R:�D?

learning_rate_1?a7�eH5I       6%�	��@���A�E*;


total_loss��@

error_R:X?

learning_rate_1?a7CP�I       6%�	��@���A�E*;


total_lossXܣ@

error_R�U?

learning_rate_1?a7��تI       6%�	�,@���A�E*;


total_loss��A

error_R��N?

learning_rate_1?a7z���I       6%�	Ku@���A�E*;


total_loss���@

error_R��D?

learning_rate_1?a7��bI       6%�	��@���A�E*;


total_loss`�@

error_R�	W?

learning_rate_1?a7�j��I       6%�	��@���A�E*;


total_loss"b�@

error_RÕV?

learning_rate_1?a7���I       6%�	^=@���A�E*;


total_loss�@

error_Rv�Z?

learning_rate_1?a7xb��I       6%�		}@���A�E*;


total_loss�v�@

error_RE�a?

learning_rate_1?a7 W.I       6%�	��@���A�E*;


total_lossj:�@

error_RR�V?

learning_rate_1?a7��I       6%�	E
@���A�E*;


total_loss�k�@

error_Rc�M?

learning_rate_1?a7S���I       6%�	�X@���A�E*;


total_loss��@

error_RC�G?

learning_rate_1?a7��D�I       6%�	}�@���A�E*;


total_lossډ�@

error_R<�??

learning_rate_1?a7�+��I       6%�	t�@���A�E*;


total_lossXA

error_Rr�S?

learning_rate_1?a7��#I       6%�	�7@���A�E*;


total_loss��z@

error_RÉR?

learning_rate_1?a7�V�KI       6%�	�~@���A�E*;


total_loss�q�@

error_R*`G?

learning_rate_1?a7JVp�I       6%�	��@���A�E*;


total_loss���@

error_R&W?

learning_rate_1?a7Ҽ�9I       6%�	8@���A�E*;


total_loss�%�@

error_R�=F?

learning_rate_1?a7���I       6%�	mZ@���A�E*;


total_lossC}�@

error_R�M?

learning_rate_1?a7H��mI       6%�	�@���A�E*;


total_loss[�@

error_R�PA?

learning_rate_1?a7-~�I       6%�	��@���A�E*;


total_loss��@

error_R�0R?

learning_rate_1?a7"�RI       6%�	**@���A�E*;


total_lossx��@

error_Rf\V?

learning_rate_1?a7sA�pI       6%�	Cn@���A�E*;


total_loss��@

error_RϸQ?

learning_rate_1?a7W��CI       6%�	 �@���A�E*;


total_loss��@

error_RE�E?

learning_rate_1?a7�ޔ�I       6%�	��@���A�E*;


total_loss]e�@

error_R�QV?

learning_rate_1?a7��* I       6%�	�0@���A�E*;


total_loss�#�@

error_R`&:?

learning_rate_1?a7	�̇I       6%�	�s@���A�E*;


total_loss���@

error_R,uE?

learning_rate_1?a7Fh�I       6%�	��@���A�E*;


total_loss�p�@

error_R�2T?

learning_rate_1?a7�E	�I       6%�	a�@���A�E*;


total_loss���@

error_RZ{I?

learning_rate_1?a7�<��I       6%�	KA@���A�E*;


total_loss'��@

error_R�<Z?

learning_rate_1?a7^��I       6%�	��@���A�E*;


total_loss��@

error_Rwj?

learning_rate_1?a7#�>�I       6%�	��@���A�E*;


total_loss�#�@

error_R�[?

learning_rate_1?a7*`"dI       6%�	`@���A�E*;


total_loss}}�@

error_R{1W?

learning_rate_1?a7sì�I       6%�	8g@���A�E*;


total_loss��@

error_R�A?

learning_rate_1?a7���I       6%�	�@���A�E*;


total_loss,��@

error_RtBC?

learning_rate_1?a72 �I       6%�	�@���A�E*;


total_lossd A

error_R�N?

learning_rate_1?a7���rI       6%�	�T@���A�E*;


total_lossZު@

error_R3Z?

learning_rate_1?a7`<bRI       6%�	x�@���A�E*;


total_loss.u�@

error_R��F?

learning_rate_1?a7�Q9qI       6%�	��@���A�E*;


total_lossXݺ@

error_R��Y?

learning_rate_1?a7�]��I       6%�	�@���A�E*;


total_loss8�@

error_Rs"P?

learning_rate_1?a7oF�LI       6%�	�a@���A�E*;


total_lossŲ@

error_R�E?

learning_rate_1?a7�#��I       6%�	��@���A�E*;


total_loss��@

error_R(9N?

learning_rate_1?a7�Dt�I       6%�	p�@���A�E*;


total_loss��@

error_R�S?

learning_rate_1?a7��I       6%�	*@���A�E*;


total_loss�ν@

error_R}[R?

learning_rate_1?a7�>��I       6%�	�n@���A�E*;


total_loss���@

error_R�cT?

learning_rate_1?a71��I       6%�	��@���A�E*;


total_loss��@

error_R�S?

learning_rate_1?a7߯�I       6%�	�@���A�E*;


total_loss,Ҥ@

error_R�Y?

learning_rate_1?a7,��^I       6%�	K[@���A�E*;


total_loss?|�@

error_R��a?

learning_rate_1?a7@���I       6%�	 �@���A�E*;


total_loss�`@

error_R O?

learning_rate_1?a7P�I       6%�	h�@���A�E*;


total_loss�$�@

error_R�AV?

learning_rate_1?a7Wd��I       6%�	�+@���A�F*;


total_loss�hA

error_Rv�W?

learning_rate_1?a7�8\�I       6%�	�n@���A�F*;


total_loss��@

error_RC,W?

learning_rate_1?a7�	I       6%�	Ű@���A�F*;


total_lossz�@

error_R�&N?

learning_rate_1?a7���*I       6%�	t�@���A�F*;


total_loss�b�@

error_Ra�=?

learning_rate_1?a7j0!MI       6%�	�<@���A�F*;


total_loss��@

error_REY2?

learning_rate_1?a7�Fh�I       6%�	�@���A�F*;


total_loss�L�@

error_RȪ[?

learning_rate_1?a7?9I�I       6%�	�@���A�F*;


total_loss���@

error_R{0Y?

learning_rate_1?a7#CWI       6%�	�@���A�F*;


total_lossڭ�@

error_RlB?

learning_rate_1?a7?q�NI       6%�	Ci@���A�F*;


total_lossd�@

error_Rc�=?

learning_rate_1?a7���I       6%�	F�@���A�F*;


total_loss�j�@

error_R�E?

learning_rate_1?a7����I       6%�	A�@���A�F*;


total_loss|��@

error_Rm�??

learning_rate_1?a7É'I       6%�	0C@���A�F*;


total_loss��~@

error_R_eT?

learning_rate_1?a7^��I       6%�	��@���A�F*;


total_lossܳ�@

error_RC;I?

learning_rate_1?a7����I       6%�	��@���A�F*;


total_loss_�@

error_R�SQ?

learning_rate_1?a7��&�I       6%�	�@���A�F*;


total_loss#�@

error_R��T?

learning_rate_1?a7�A?�I       6%�	�S@���A�F*;


total_loss�<�@

error_R��X?

learning_rate_1?a7�{z�I       6%�	��@���A�F*;


total_lossotg@

error_R�4I?

learning_rate_1?a7���I       6%�	i�@���A�F*;


total_loss�5�@

error_R=I?

learning_rate_1?a7�ݱ�I       6%�	y& @���A�F*;


total_lossv2�@

error_R�"A?

learning_rate_1?a7�|nI       6%�	�p @���A�F*;


total_loss�A

error_R�`A?

learning_rate_1?a7�9n�I       6%�	� @���A�F*;


total_loss2�@

error_R,�E?

learning_rate_1?a7���	I       6%�	�� @���A�F*;


total_lossڔ�@

error_RH�]?

learning_rate_1?a7��W_I       6%�	�@!@���A�F*;


total_loss�U�@

error_R��E?

learning_rate_1?a7(�I       6%�	ч!@���A�F*;


total_lossS�}@

error_R}^G?

learning_rate_1?a74Ħ	I       6%�	��!@���A�F*;


total_loss�~�@

error_R��V?

learning_rate_1?a7��-I       6%�	J"@���A�F*;


total_loss�J�@

error_R�?K?

learning_rate_1?a7'¤�I       6%�	%`"@���A�F*;


total_loss���@

error_R��Q?

learning_rate_1?a7�=�eI       6%�	��"@���A�F*;


total_loss�҇@

error_R3�^?

learning_rate_1?a7q�uoI       6%�	O�"@���A�F*;


total_loss���@

error_Rd�>?

learning_rate_1?a7�s�CI       6%�	,/#@���A�F*;


total_loss���@

error_R��C?

learning_rate_1?a7��[I       6%�	�t#@���A�F*;


total_loss�6�@

error_R EH?

learning_rate_1?a7��c�I       6%�	�#@���A�F*;


total_loss�`�@

error_R8�[?

learning_rate_1?a7�Q9I       6%�	�$@���A�F*;


total_lossh�A

error_Ri'K?

learning_rate_1?a7o"�OI       6%�	�O$@���A�F*;


total_loss���@

error_R�
J?

learning_rate_1?a7̡��I       6%�	8�$@���A�F*;


total_loss�8�@

error_R�R?

learning_rate_1?a7�@�iI       6%�	#�$@���A�F*;


total_loss_8�@

error_R�:Q?

learning_rate_1?a7�ˊI       6%�	�%@���A�F*;


total_loss�Ѡ@

error_R[�J?

learning_rate_1?a7o��I       6%�	g%@���A�F*;


total_loss[��@

error_R��J?

learning_rate_1?a7�HI       6%�	��%@���A�F*;


total_lossɥ�@

error_Ra�N?

learning_rate_1?a7n}�SI       6%�	��%@���A�F*;


total_loss���@

error_RR�c?

learning_rate_1?a7�(\I       6%�	�G&@���A�F*;


total_loss;z@

error_R�H?

learning_rate_1?a7�>�I       6%�	��)@���A�F*;


total_losshו@

error_R8�J?

learning_rate_1?a7G![�I       6%�	�*@���A�F*;


total_loss�Ύ@

error_R�MI?

learning_rate_1?a7�5�MI       6%�	�_*@���A�F*;


total_loss�'�@

error_RoO?

learning_rate_1?a7�t%I       6%�	@�*@���A�F*;


total_loss܆�@

error_R�>?

learning_rate_1?a7i�"I       6%�	��*@���A�F*;


total_loss��@

error_RCoS?

learning_rate_1?a7��AI       6%�	�8+@���A�F*;


total_loss��@

error_R��J?

learning_rate_1?a7��`�I       6%�	��+@���A�F*;


total_lossc�@

error_R�^<?

learning_rate_1?a7�fk\I       6%�	��+@���A�F*;


total_loss�#�@

error_RrEM?

learning_rate_1?a7�Y%I       6%�	F,@���A�F*;


total_lossdh�@

error_R__R?

learning_rate_1?a7����I       6%�	�U,@���A�F*;


total_loss���@

error_R�_??

learning_rate_1?a7jQ4I       6%�	ؤ,@���A�F*;


total_lossf	t@

error_R��\?

learning_rate_1?a7���I       6%�	X�,@���A�F*;


total_loss�J�@

error_R�+Q?

learning_rate_1?a7����I       6%�	�5-@���A�F*;


total_loss&�@

error_R�rd?

learning_rate_1?a7�R��I       6%�	{�-@���A�F*;


total_loss�o�@

error_R��K?

learning_rate_1?a7>�LkI       6%�	e�-@���A�F*;


total_lossI��@

error_R��^?

learning_rate_1?a7��S�I       6%�	�.@���A�F*;


total_loss�}@

error_R�+F?

learning_rate_1?a7��6�I       6%�	)X.@���A�F*;


total_losstZ�@

error_Rd~J?

learning_rate_1?a7�3_eI       6%�	��.@���A�F*;


total_loss�A

error_R��H?

learning_rate_1?a7E�MJI       6%�	o�.@���A�F*;


total_loss�@

error_R�vA?

learning_rate_1?a7{i�I       6%�	(6/@���A�F*;


total_loss��@

error_RE]M?

learning_rate_1?a7X��I       6%�	~�/@���A�F*;


total_loss�Ŷ@

error_R�@?

learning_rate_1?a7��u�I       6%�	��/@���A�F*;


total_loss��@

error_R�I?

learning_rate_1?a7��w"I       6%�	�0@���A�F*;


total_loss�8�@

error_R)CM?

learning_rate_1?a7b$��I       6%�	=Z0@���A�F*;


total_loss1;�@

error_R�|@?

learning_rate_1?a7�!I       6%�	^�0@���A�F*;


total_loss���@

error_R%5P?

learning_rate_1?a7Ct�I       6%�	&�0@���A�F*;


total_lossa�y@

error_R�9J?

learning_rate_1?a7�5��I       6%�	~91@���A�F*;


total_lossAJ�@

error_RJTW?

learning_rate_1?a7����I       6%�	�}1@���A�F*;


total_loss�G�@

error_R�(M?

learning_rate_1?a7d�|�I       6%�	��1@���A�F*;


total_loss\&�@

error_R�TK?

learning_rate_1?a75��I       6%�	�2@���A�F*;


total_loss�߹@

error_R�Kd?

learning_rate_1?a7�neiI       6%�	�W2@���A�F*;


total_loss`u�@

error_R3vO?

learning_rate_1?a7�_��I       6%�	 �2@���A�F*;


total_lossQ��@

error_R�V1?

learning_rate_1?a7�S�I       6%�	+�2@���A�F*;


total_loss�Z�@

error_R_[U?

learning_rate_1?a7q�,�I       6%�	�+3@���A�F*;


total_loss.Ӻ@

error_RWJf?

learning_rate_1?a7ޭ��I       6%�	tq3@���A�F*;


total_lossu�@

error_R��Q?

learning_rate_1?a7��I       6%�	}�3@���A�F*;


total_lossF�@

error_R�qO?

learning_rate_1?a7*�71I       6%�	��3@���A�F*;


total_loss�>�@

error_R�R?

learning_rate_1?a7���I       6%�	5]4@���A�F*;


total_loss2�@

error_Rv�H?

learning_rate_1?a7��>�I       6%�	J�4@���A�F*;


total_lossI9�@

error_R��K?

learning_rate_1?a7f/%�I       6%�	��4@���A�F*;


total_lossֱ�@

error_R�K?

learning_rate_1?a7�]u�I       6%�	:5@���A�F*;


total_loss�t�@

error_R��M?

learning_rate_1?a70�{�I       6%�	��5@���A�F*;


total_loss��{@

error_R�TP?

learning_rate_1?a7e���I       6%�	��5@���A�F*;


total_loss�1�@

error_R��_?

learning_rate_1?a7b;�I       6%�	BF6@���A�F*;


total_loss��@

error_R��S?

learning_rate_1?a7G;l�I       6%�	s�6@���A�F*;


total_loss��}@

error_RP?

learning_rate_1?a7����I       6%�	)&7@���A�F*;


total_loss�@

error_RLuZ?

learning_rate_1?a7�f\�I       6%�	,w7@���A�F*;


total_loss�� A

error_R�P?

learning_rate_1?a7�!�
I       6%�	��7@���A�F*;


total_loss3��@

error_R�>N?

learning_rate_1?a7�<2NI       6%�	 .8@���A�F*;


total_loss�
�@

error_R.�L?

learning_rate_1?a7��BcI       6%�	x8@���A�F*;


total_loss���@

error_R;�Q?

learning_rate_1?a7���I       6%�	�8@���A�F*;


total_loss`ن@

error_R�]?

learning_rate_1?a7���I       6%�	�9@���A�F*;


total_loss��@

error_RL�X?

learning_rate_1?a7�s�I       6%�	�M9@���A�F*;


total_loss;V�@

error_RW�:?

learning_rate_1?a7;�[I       6%�	�9@���A�F*;


total_loss�E�@

error_R��;?

learning_rate_1?a7Z���I       6%�	}:@���A�F*;


total_lossJ��@

error_RT�6?

learning_rate_1?a7&��=I       6%�	�z:@���A�F*;


total_lossa��@

error_R�I?

learning_rate_1?a7%�t�I       6%�	��:@���A�F*;


total_lossl�@

error_R�*P?

learning_rate_1?a7��I       6%�	y";@���A�F*;


total_loss�?�@

error_RS�^?

learning_rate_1?a79<DFI       6%�	y�;@���A�F*;


total_losse��@

error_RZ�H?

learning_rate_1?a7}��pI       6%�	K<@���A�F*;


total_loss��@

error_RSY?

learning_rate_1?a7vE��I       6%�	-x<@���A�F*;


total_loss�l�@

error_R�G?

learning_rate_1?a7y��:I       6%�	��<@���A�F*;


total_lossJ��@

error_RE�N?

learning_rate_1?a7�y�#I       6%�	?=@���A�F*;


total_loss�A

error_R�PX?

learning_rate_1?a7"���I       6%�	�=@���A�F*;


total_lossa��@

error_R�2V?

learning_rate_1?a7�.cI       6%�	B>@���A�F*;


total_loss���@

error_ROg?

learning_rate_1?a7yf�MI       6%�	�L>@���A�F*;


total_loss�.A

error_R�[E?

learning_rate_1?a7�~�I       6%�	��>@���A�F*;


total_loss���@

error_RcS9?

learning_rate_1?a72���I       6%�	��>@���A�F*;


total_lossw��@

error_Rn�b?

learning_rate_1?a7d�,�I       6%�	iG?@���A�F*;


total_lossl׿@

error_R\zG?

learning_rate_1?a7��/�I       6%�	B�?@���A�F*;


total_loss���@

error_R&\?

learning_rate_1?a7��A�I       6%�	�?@���A�F*;


total_loss��@

error_R=�H?

learning_rate_1?a7����I       6%�	Z/@@���A�F*;


total_loss���@

error_R�
K?

learning_rate_1?a7����I       6%�	�q@@���A�F*;


total_loss�D�@

error_R��K?

learning_rate_1?a7wh"I       6%�	/�@@���A�F*;


total_loss	�@

error_R;$_?

learning_rate_1?a7�C)jI       6%�	�A@���A�F*;


total_loss0�@

error_R�R?

learning_rate_1?a7@NEI       6%�	HXA@���A�F*;


total_loss��@

error_R��]?

learning_rate_1?a7]��mI       6%�	o�A@���A�F*;


total_loss��`@

error_RC+<?

learning_rate_1?a7����I       6%�	�A@���A�F*;


total_lossc#A

error_R@+5?

learning_rate_1?a7bjSHI       6%�	�CB@���A�F*;


total_loss��@

error_RqS?

learning_rate_1?a7��^�I       6%�	�B@���A�F*;


total_lossrs�@

error_REaG?

learning_rate_1?a7��{�I       6%�	�B@���A�F*;


total_lossH{�@

error_R�9E?

learning_rate_1?a7\}^�I       6%�	�7C@���A�F*;


total_loss� �@

error_RdmI?

learning_rate_1?a7���I       6%�	�C@���A�F*;


total_loss@

error_R!;8?

learning_rate_1?a7��LI       6%�	��C@���A�F*;


total_lossݱ@

error_Rc)P?

learning_rate_1?a7v���I       6%�	k<D@���A�F*;


total_loss�C�@

error_R�R?

learning_rate_1?a7�Q|sI       6%�	؎D@���A�F*;


total_losso_�@

error_Rs�J?

learning_rate_1?a7�V�I       6%�	��D@���A�F*;


total_loss{��@

error_R:$R?

learning_rate_1?a7f��I       6%�	/8E@���A�G*;


total_lossM��@

error_R��N?

learning_rate_1?a7����I       6%�	ۃE@���A�G*;


total_loss���@

error_R�\?

learning_rate_1?a7\a+YI       6%�	S�E@���A�G*;


total_loss���@

error_R��N?

learning_rate_1?a74��I       6%�	�F@���A�G*;


total_loss?��@

error_R&�L?

learning_rate_1?a7���I       6%�	�^F@���A�G*;


total_loss���@

error_R�K?

learning_rate_1?a7X[�@I       6%�	��F@���A�G*;


total_loss���@

error_R��O?

learning_rate_1?a7�%�I       6%�	�G@���A�G*;


total_loss���@

error_R6�A?

learning_rate_1?a7:5nI       6%�	eUG@���A�G*;


total_loss��@

error_RԃT?

learning_rate_1?a7��Z�I       6%�	��G@���A�G*;


total_lossA[�@

error_RM W?

learning_rate_1?a7���I       6%�	i�G@���A�G*;


total_loss��@

error_R��S?

learning_rate_1?a7}&�I       6%�	�!H@���A�G*;


total_loss���@

error_R�[R?

learning_rate_1?a7��I       6%�	�cH@���A�G*;


total_lossf5�@

error_RT9Y?

learning_rate_1?a7�&�I       6%�	r�H@���A�G*;


total_loss��@

error_R�MT?

learning_rate_1?a7
"�I       6%�	^�H@���A�G*;


total_lossNT@

error_R�4=?

learning_rate_1?a7���'I       6%�	v9I@���A�G*;


total_loss�!�@

error_Rd�]?

learning_rate_1?a7䧚�I       6%�	J�I@���A�G*;


total_loss8�k@

error_R�1W?

learning_rate_1?a7�_Z~I       6%�	o�I@���A�G*;


total_loss���@

error_Rx�T?

learning_rate_1?a7�}r$I       6%�	4
J@���A�G*;


total_loss�ɼ@

error_RԞV?

learning_rate_1?a7���I       6%�		NJ@���A�G*;


total_loss3�@

error_R�E?

learning_rate_1?a7�9e�I       6%�	�J@���A�G*;


total_loss:\�@

error_R��I?

learning_rate_1?a7�y�)I       6%�	z�J@���A�G*;


total_loss��@

error_R)�F?

learning_rate_1?a7EU�iI       6%�	�K@���A�G*;


total_loss&֐@

error_R��S?

learning_rate_1?a7*_��I       6%�	bK@���A�G*;


total_loss���@

error_R�W?

learning_rate_1?a7�6�I       6%�	3�K@���A�G*;


total_loss H@

error_R�A?

learning_rate_1?a7g=��I       6%�	]�K@���A�G*;


total_loss��@

error_R�TQ?

learning_rate_1?a7��ƔI       6%�	�3L@���A�G*;


total_loss���@

error_R%4\?

learning_rate_1?a74&��I       6%�	�|L@���A�G*;


total_loss}v�@

error_R�CN?

learning_rate_1?a7��I       6%�	R�L@���A�G*;


total_loss	��@

error_R R?

learning_rate_1?a7���I       6%�	gM@���A�G*;


total_loss���@

error_R��G?

learning_rate_1?a7�C�I       6%�	FM@���A�G*;


total_lossUT�@

error_R
�H?

learning_rate_1?a7�+j<I       6%�		�M@���A�G*;


total_loss��e@

error_R�zP?

learning_rate_1?a7`kz�I       6%�	��M@���A�G*;


total_lossv�@

error_R8�I?

learning_rate_1?a7�mI       6%�	�N@���A�G*;


total_lossc��@

error_R)�H?

learning_rate_1?a7Ȫ�I       6%�	�TN@���A�G*;


total_loss#H�@

error_R�cW?

learning_rate_1?a77$lI       6%�	��N@���A�G*;


total_lossm_�@

error_R��B?

learning_rate_1?a7G�w�I       6%�	��N@���A�G*;


total_loss�Y�@

error_R�2P?

learning_rate_1?a7���0I       6%�	�(O@���A�G*;


total_loss���@

error_R��D?

learning_rate_1?a7X��I       6%�	kO@���A�G*;


total_loss��@

error_R[�W?

learning_rate_1?a7�'I       6%�	,�O@���A�G*;


total_loss)�@

error_R�zP?

learning_rate_1?a7�\��I       6%�	��O@���A�G*;


total_loss�o�@

error_R��K?

learning_rate_1?a7�!�I       6%�	�KP@���A�G*;


total_loss���@

error_R�8X?

learning_rate_1?a7�C^I       6%�	P@���A�G*;


total_loss5(�@

error_R�/T?

learning_rate_1?a7)ʭI       6%�	��P@���A�G*;


total_loss��@

error_R�C?

learning_rate_1?a7�FG�I       6%�	$Q@���A�G*;


total_lossѣ�@

error_RŰK?

learning_rate_1?a7i��`I       6%�	RnQ@���A�G*;


total_loss��@

error_R�eT?

learning_rate_1?a7��:�I       6%�	��Q@���A�G*;


total_loss�,�@

error_Ra�F?

learning_rate_1?a7gj��I       6%�	�Q@���A�G*;


total_lossl�@

error_R�oP?

learning_rate_1?a7�C՜I       6%�	�DR@���A�G*;


total_loss?�@

error_R�H?

learning_rate_1?a7���I       6%�	��R@���A�G*;


total_lossrm�@

error_R��??

learning_rate_1?a7wLSI       6%�	�R@���A�G*;


total_loss�z�@

error_R�B?

learning_rate_1?a7E��I       6%�	�S@���A�G*;


total_lossA

error_R�1M?

learning_rate_1?a7���I       6%�	L]S@���A�G*;


total_loss;s\@

error_R=�T?

learning_rate_1?a7ҩ�xI       6%�	��S@���A�G*;


total_loss��n@

error_RqNV?

learning_rate_1?a7+9X�I       6%�	��S@���A�G*;


total_loss��@

error_RxHW?

learning_rate_1?a7�O�I       6%�	'T@���A�G*;


total_lossE��@

error_R�W?

learning_rate_1?a7�+�I       6%�	\sT@���A�G*;


total_loss/ �@

error_R�hK?

learning_rate_1?a7Nl]�I       6%�	�T@���A�G*;


total_loss���@

error_R$�7?

learning_rate_1?a7�ur�I       6%�	�U@���A�G*;


total_lossҊ@

error_R��H?

learning_rate_1?a7ߥ��I       6%�	STU@���A�G*;


total_lossqԘ@

error_R�LS?

learning_rate_1?a7|叒I       6%�	)�U@���A�G*;


total_lossC�@

error_R�^F?

learning_rate_1?a7�\xI       6%�	�U@���A�G*;


total_loss��@

error_R�N?

learning_rate_1?a7��CI       6%�	'V@���A�G*;


total_loss���@

error_R!^?

learning_rate_1?a74k�^I       6%�	�lV@���A�G*;


total_loss\��@

error_R�bG?

learning_rate_1?a7�!;�I       6%�	v�V@���A�G*;


total_loss:�@

error_R3�M?

learning_rate_1?a7c�I�I       6%�	�:W@���A�G*;


total_loss��@

error_R;\?

learning_rate_1?a73��I       6%�	KW@���A�G*;


total_loss��@

error_R4R?

learning_rate_1?a7���I       6%�	��W@���A�G*;


total_loss���@

error_R�v9?

learning_rate_1?a7��+�I       6%�	�X@���A�G*;


total_lossdv_@

error_RN	7?

learning_rate_1?a7��\I       6%�	pkX@���A�G*;


total_loss���@

error_R�D?

learning_rate_1?a7N���I       6%�	��X@���A�G*;


total_lossC2�@

error_R�Q?

learning_rate_1?a7�vI       6%�	��X@���A�G*;


total_lossϧ�@

error_RCA?

learning_rate_1?a7�;��I       6%�	wLY@���A�G*;


total_loss6��@

error_R��S?

learning_rate_1?a7��I       6%�	��Y@���A�G*;


total_lossX��@

error_R��D?

learning_rate_1?a7�gM?I       6%�	��Y@���A�G*;


total_loss;y�@

error_R2lV?

learning_rate_1?a7N��I       6%�	�/Z@���A�G*;


total_loss|�@

error_R��O?

learning_rate_1?a7�{A�I       6%�	^Z@���A�G*;


total_loss:��@

error_Rrd?

learning_rate_1?a7���nI       6%�	��Z@���A�G*;


total_loss��@

error_R�eS?

learning_rate_1?a7}7I       6%�	3[@���A�G*;


total_lossR1�@

error_R�C?

learning_rate_1?a7O)�