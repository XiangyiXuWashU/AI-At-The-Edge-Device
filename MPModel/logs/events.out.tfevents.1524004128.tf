       �K"	   H���Abrain.Event:2f�6>K     6�.	�dH���A"��
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
*weights/random_normal/RandomStandardNormalRandomStandardNormalweights/random_normal/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
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
weights/weight2/AssignAssignweights/weight2weights/random_normal_1*
use_locking(*
T0*"
_class
loc:@weights/weight2*
validate_shape(* 
_output_shapes
:
��
�
weights/weight2/readIdentityweights/weight2* 
_output_shapes
:
��*
T0*"
_class
loc:@weights/weight2
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
,weights/random_normal_2/RandomStandardNormalRandomStandardNormalweights/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2 *

seed 
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
weights/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
,weights/random_normal_3/RandomStandardNormalRandomStandardNormalweights/random_normal_3/shape*
T0*
dtype0*
_output_shapes

:d*
seed2 *

seed 
�
weights/random_normal_3/mulMul,weights/random_normal_3/RandomStandardNormalweights/random_normal_3/stddev*
_output_shapes

:d*
T0
�
weights/random_normal_3Addweights/random_normal_3/mulweights/random_normal_3/mean*
T0*
_output_shapes

:d
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
weights/weight_out/AssignAssignweights/weight_outweights/random_normal_3*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
biases/random_normal/mulMul)biases/random_normal/RandomStandardNormalbiases/random_normal/stddev*
_output_shapes	
:�*
T0
v
biases/random_normalAddbiases/random_normal/mulbiases/random_normal/mean*
T0*
_output_shapes	
:�
z
biases/bias1
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
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
biases/bias1/readIdentitybiases/bias1*
_output_shapes	
:�*
T0*
_class
loc:@biases/bias1
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
biases/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
biases/random_normal_1Addbiases/random_normal_1/mulbiases/random_normal_1/mean*
_output_shapes	
:�*
T0
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
biases/bias2/readIdentitybiases/bias2*
T0*
_class
loc:@biases/bias2*
_output_shapes	
:�
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
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
�
biases/bias3/AssignAssignbiases/bias3biases/random_normal_2*
T0*
_class
loc:@biases/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
biases/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
biases/bias_out/AssignAssignbiases/bias_outbiases/random_normal_3*
use_locking(*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:
z
biases/bias_out/readIdentitybiases/bias_out*
_output_shapes
:*
T0*"
_class
loc:@biases/bias_out
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
weights_1/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
weights_1/random_normalAddweights_1/random_normal/mulweights_1/random_normal/mean* 
_output_shapes
:
��*
T0
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
weights_1/weight1/AssignAssignweights_1/weight1weights_1/random_normal*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1
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
valueB"�   �   *
dtype0*
_output_shapes
:
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
VariableV2*
shape:
��*
shared_name *
dtype0* 
_output_shapes
:
��*
	container 
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
 weights_1/random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
.weights_1/random_normal_2/RandomStandardNormalRandomStandardNormalweights_1/random_normal_2/shape*
dtype0*
_output_shapes
:	�d*
seed2 *

seed *
T0
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
VariableV2*
shape:	�d*
shared_name *
dtype0*
_output_shapes
:	�d*
	container 
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
weights_1/weight3/readIdentityweights_1/weight3*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
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

seed *
T0*
dtype0*
_output_shapes

:d*
seed2 
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
biases_1/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+biases_1/random_normal/RandomStandardNormalRandomStandardNormalbiases_1/random_normal/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
�
biases_1/random_normal/mulMul+biases_1/random_normal/RandomStandardNormalbiases_1/random_normal/stddev*
T0*
_output_shapes	
:�
|
biases_1/random_normalAddbiases_1/random_normal/mulbiases_1/random_normal/mean*
_output_shapes	
:�*
T0
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
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
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
-biases_1/random_normal_2/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_2/shape*
dtype0*
_output_shapes
:d*
seed2 *

seed *
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
biases_1/bias_out/AssignAssignbiases_1/bias_outbiases_1/random_normal_3*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
�
biases_1/bias_out/readIdentitybiases_1/bias_out*
_output_shapes
:*
T0*$
_class
loc:@biases_1/bias_out
�
layer_1/MatMulMatMulinput/Spectrum-inputweights_1/weight1/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
j
layer_1/AddAddlayer_1/MatMulbiases_1/bias1/read*
T0*(
_output_shapes
:����������
T
layer_2/ReluRelulayer_1/Add*(
_output_shapes
:����������*
T0
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
result/MatMulMatMulresult/Reluweights_1/weight_out/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j

result/AddAddresult/MatMulbiases_1/bias_out/read*'
_output_shapes
:���������*
T0
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
SquareSquaresub_1*
T0*'
_output_shapes
:���������
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
Variable/AssignAssignVariableVariable/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
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
 *��L=*
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
'learning_rate/ExponentialDecay/Cast_1/xConst*
value
B :�'*
dtype0*
_output_shapes
: 
�
%learning_rate/ExponentialDecay/Cast_1Cast'learning_rate/ExponentialDecay/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
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
dtype0*
_output_shapes
:*
valueB"       
c
	loss/MeanMeanSquare
loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
$train/gradients/loss/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
l
&train/gradients/loss/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
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
: *

Tidx0*
	keep_dims( 
p
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
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
train/gradients/Square_grad/mulMul!train/gradients/Square_grad/mul/xsub_1*
T0*'
_output_shapes
:���������
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
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"train/gradients/sub_1_grad/ReshapeReshapetrain/gradients/sub_1_grad/Sum train/gradients/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
 train/gradients/sub_1_grad/Sum_1Sum!train/gradients/Square_grad/mul_12train/gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
3train/gradients/sub_1_grad/tuple/control_dependencyIdentity"train/gradients/sub_1_grad/Reshape,^train/gradients/sub_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*5
_class+
)'loc:@train/gradients/sub_1_grad/Reshape
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
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
8train/gradients/result/Add_grad/tuple/control_dependencyIdentity'train/gradients/result/Add_grad/Reshape1^train/gradients/result/Add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*:
_class0
.,loc:@train/gradients/result/Add_grad/Reshape
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
=train/gradients/result/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/result/MatMul_grad/MatMul_14^train/gradients/result/MatMul_grad/tuple/group_deps*
_output_shapes

:d*
T0*>
_class4
20loc:@train/gradients/result/MatMul_grad/MatMul_1
�
)train/gradients/result/Relu_grad/ReluGradReluGrad;train/gradients/result/MatMul_grad/tuple/control_dependencyresult/Relu*'
_output_shapes
:���������d*
T0
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
6train/gradients/layer_3/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/Add_grad/Shape(train/gradients/layer_3/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_3/Add_grad/SumSum)train/gradients/result/Relu_grad/ReluGrad6train/gradients/layer_3/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( *
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/Add_grad/tuple/control_dependencyweights_1/weight3/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
&train/gradients/layer_2/Add_grad/ShapeShapelayer_2/MatMul*
_output_shapes
:*
T0*
out_type0
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
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
;train/gradients/layer_2/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/Add_grad/Reshape_12^train/gradients/layer_2/Add_grad/tuple/group_deps*
_output_shapes	
:�*
T0*=
_class3
1/loc:@train/gradients/layer_2/Add_grad/Reshape_1
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/Add_grad/tuple/control_dependencyweights_1/weight2/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_2/Add_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( 
�
*train/gradients/layer_1/Add_grad/Reshape_1Reshape&train/gradients/layer_1/Add_grad/Sum_1(train/gradients/layer_1/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
1train/gradients/layer_1/Add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/Add_grad/Reshape+^train/gradients/layer_1/Add_grad/Reshape_1
�
9train/gradients/layer_1/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/Add_grad/Reshape2^train/gradients/layer_1/Add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/Add_grad/Reshape*(
_output_shapes
:����������
�
;train/gradients/layer_1/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/Add_grad/Reshape_12^train/gradients/layer_1/Add_grad/tuple/group_deps*
_output_shapes	
:�*
T0*=
_class3
1/loc:@train/gradients/layer_1/Add_grad/Reshape_1
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/Add_grad/tuple/control_dependencyweights_1/weight1/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Spectrum-input9train/gradients/layer_1/Add_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*!
_class
loc:@biases_1/bias1
�
train/beta1_power
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
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
y
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking(
y
train/beta2_power/readIdentitytrain/beta2_power*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
weights_1/weight1/Adam/AssignAssignweights_1/weight1/Adam(weights_1/weight1/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1
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
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight1*
	container 
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
weights_1/weight1/Adam_1/readIdentityweights_1/weight1/Adam_1* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight1
�
(weights_1/weight2/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
��*$
_class
loc:@weights_1/weight2*
valueB
��*    
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
weights_1/weight2/Adam/AssignAssignweights_1/weight2/Adam(weights_1/weight2/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��
�
weights_1/weight2/Adam/readIdentityweights_1/weight2/Adam*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
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
VariableV2*$
_class
loc:@weights_1/weight2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
weights_1/weight2/Adam_1/AssignAssignweights_1/weight2/Adam_1*weights_1/weight2/Adam_1/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
weights_1/weight2/Adam_1/readIdentityweights_1/weight2/Adam_1*
T0*$
_class
loc:@weights_1/weight2* 
_output_shapes
:
��
�
(weights_1/weight3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	�d*$
_class
loc:@weights_1/weight3*
valueB	�d*    
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
*weights_1/weight3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	�d*$
_class
loc:@weights_1/weight3*
valueB	�d*    
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
"weights_1/weight_out/Adam_1/AssignAssignweights_1/weight_out/Adam_1-weights_1/weight_out/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
 weights_1/weight_out/Adam_1/readIdentityweights_1/weight_out/Adam_1*
T0*'
_class
loc:@weights_1/weight_out*
_output_shapes

:d
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
biases_1/bias1/Adam/AssignAssignbiases_1/bias1/Adam%biases_1/bias1/Adam/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
VariableV2*
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
biases_1/bias2/Adam/AssignAssignbiases_1/bias2/Adam%biases_1/bias2/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias2/Adam/readIdentitybiases_1/bias2/Adam*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias2
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
biases_1/bias3/Adam/AssignAssignbiases_1/bias3/Adam%biases_1/bias3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out*
	container *
shape:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out*
	container *
shape:
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
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
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
0train/Adam/update_weights_1/weight_out/ApplyAdam	ApplyAdamweights_1/weight_outweights_1/weight_out/Adamweights_1/weight_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/result/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@weights_1/weight_out*
use_nesterov( *
_output_shapes

:d*
use_locking( 
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
*train/Adam/update_biases_1/bias3/ApplyAdam	ApplyAdambiases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/Add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@biases_1/bias3*
use_nesterov( *
_output_shapes
:d*
use_locking( 
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
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/updateNoOp.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
�
train/Adam/valueConst^train/Adam/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable
�

train/Adam	AssignAddVariabletrain/Adam/value*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
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
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
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
save/Assign_5Assignbiases_1/bias1save/RestoreV2_5*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
y
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBbiases_1/bias1/Adam
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
save/Assign_6Assignbiases_1/bias1/Adamsave/RestoreV2_6*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBbiases_1/bias2
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
save/Assign_9Assignbiases_1/bias2/Adamsave/RestoreV2_9*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias2
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
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assignbiases_1/bias2/Adam_1save/RestoreV2_10*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_11Assignbiases_1/bias3save/RestoreV2_11*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
save/RestoreV2_13/tensor_namesConst*
dtype0*
_output_shapes
:**
value!BBbiases_1/bias3/Adam_1
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
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assignbiases_1/bias_outsave/RestoreV2_14*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
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
save/RestoreV2_16/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bbiases_1/bias_out/Adam_1
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2
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
save/RestoreV2_18/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBtrain/beta2_power
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
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/RestoreV2_20/tensor_namesConst*
dtype0*
_output_shapes
:*$
valueBBweights/weight2
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
save/Assign_20Assignweights/weight2save/RestoreV2_20*
use_locking(*
T0*"
_class
loc:@weights/weight2*
validate_shape(* 
_output_shapes
:
��
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
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_25Assignweights_1/weight1/Adam_1save/RestoreV2_25*
use_locking(*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��
x
save/RestoreV2_26/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBweights_1/weight2
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
save/Assign_26Assignweights_1/weight2save/RestoreV2_26*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
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
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assignweights_1/weight_out/Adam_1save/RestoreV2_34*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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

total_lossScalarSummarytotal_loss/tags	loss/Mean*
_output_shapes
: *
T0
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
:*

Tidx0*
	keep_dims( 
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
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
T
error_R/tagsConst*
dtype0*
_output_shapes
: *
valueB Berror_R
V
error_RScalarSummaryerror_R/tagsstrided_slice*
T0*
_output_shapes
: 
d
learning_rate_1/tagsConst* 
valueB Blearning_rate_1*
dtype0*
_output_shapes
: 
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"�IUPh     6Ԇg	khH���AJ��
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
weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*weights/random_normal/RandomStandardNormalRandomStandardNormalweights/random_normal/shape*
dtype0* 
_output_shapes
:
��*
seed2 *

seed *
T0
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
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
��*
	container *
shape:
��
�
weights/weight1/AssignAssignweights/weight1weights/random_normal*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*"
_class
loc:@weights/weight1
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
weights/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
c
weights/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
weights/weight2/AssignAssignweights/weight2weights/random_normal_1*
use_locking(*
T0*"
_class
loc:@weights/weight2*
validate_shape(* 
_output_shapes
:
��
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
,weights/random_normal_2/RandomStandardNormalRandomStandardNormalweights/random_normal_2/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2 *

seed 
�
weights/random_normal_2/mulMul,weights/random_normal_2/RandomStandardNormalweights/random_normal_2/stddev*
T0*
_output_shapes
:	�d
�
weights/random_normal_2Addweights/random_normal_2/mulweights/random_normal_2/mean*
T0*
_output_shapes
:	�d
�
weights/weight3
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
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
weights/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
weights/random_normal_3Addweights/random_normal_3/mulweights/random_normal_3/mean*
T0*
_output_shapes

:d
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
weights/weight_out/AssignAssignweights/weight_outweights/random_normal_3*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
biases/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
biases/random_normalAddbiases/random_normal/mulbiases/random_normal/mean*
_output_shapes	
:�*
T0
z
biases/bias1
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
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
biases/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
biases/random_normal_1Addbiases/random_normal_1/mulbiases/random_normal_1/mean*
_output_shapes	
:�*
T0
z
biases/bias2
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
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
biases/random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
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
biases/bias3/readIdentitybiases/bias3*
_output_shapes
:d*
T0*
_class
loc:@biases/bias3
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
biases/bias_out/AssignAssignbiases/bias_outbiases/random_normal_3*
use_locking(*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:
z
biases/bias_out/readIdentitybiases/bias_out*
_output_shapes
:*
T0*"
_class
loc:@biases/bias_out
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
weights_1/weight1/AssignAssignweights_1/weight1weights_1/random_normal*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1
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
valueB"�   �   *
dtype0*
_output_shapes
:
c
weights_1/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
e
 weights_1/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
.weights_1/random_normal_1/RandomStandardNormalRandomStandardNormalweights_1/random_normal_1/shape*

seed *
T0*
dtype0* 
_output_shapes
:
��*
seed2 
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
VariableV2*
shape:
��*
shared_name *
dtype0* 
_output_shapes
:
��*
	container 
�
weights_1/weight2/AssignAssignweights_1/weight2weights_1/random_normal_1*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
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
weights_1/random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
weights_1/random_normal_2Addweights_1/random_normal_2/mulweights_1/random_normal_2/mean*
_output_shapes
:	�d*
T0
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
weights_1/weight3/readIdentityweights_1/weight3*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
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
 weights_1/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
VariableV2*
dtype0*
_output_shapes

:d*
	container *
shape
:d*
shared_name 
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
biases_1/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+biases_1/random_normal/RandomStandardNormalRandomStandardNormalbiases_1/random_normal/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
biases_1/bias2/AssignAssignbiases_1/bias2biases_1/random_normal_1*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias2
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
biases_1/bias3/readIdentitybiases_1/bias3*
_output_shapes
:d*
T0*!
_class
loc:@biases_1/bias3
h
biases_1/random_normal_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:
b
biases_1/random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
biases_1/random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
biases_1/bias_out/AssignAssignbiases_1/bias_outbiases_1/random_normal_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out
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
layer_3/SigmoidSigmoidlayer_2/Add*(
_output_shapes
:����������*
T0
�
layer_3/MatMulMatMullayer_3/Sigmoidweights_1/weight3/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
i
layer_3/AddAddlayer_3/MatMulbiases_1/bias3/read*'
_output_shapes
:���������d*
T0
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
SquareSquaresub_1*
T0*'
_output_shapes
:���������
X
Variable/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
l
Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
 *��L=*
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
%learning_rate/ExponentialDecay/Cast_1Cast'learning_rate/ExponentialDecay/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
l
'learning_rate/ExponentialDecay/Cast_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *��u?
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
"learning_rate/ExponentialDecay/PowPow'learning_rate/ExponentialDecay/Cast_2/x$learning_rate/ExponentialDecay/Floor*
T0*
_output_shapes
: 
�
learning_rate/ExponentialDecayMul,learning_rate/ExponentialDecay/learning_rate"learning_rate/ExponentialDecay/Pow*
_output_shapes
: *
T0
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
c
	loss/MeanMeanSquare
loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
$train/gradients/loss/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
l
&train/gradients/loss/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
!train/gradients/Square_grad/mul/xConst'^train/gradients/loss/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
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
result/Add*
T0*
out_type0*
_output_shapes
:
s
"train/gradients/sub_1_grad/Shape_1Shapeinput/Label-input*
_output_shapes
:*
T0*
out_type0
�
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"train/gradients/sub_1_grad/ReshapeReshapetrain/gradients/sub_1_grad/Sum train/gradients/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
 train/gradients/sub_1_grad/Sum_1Sum!train/gradients/Square_grad/mul_12train/gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
'train/gradients/result/Add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/result/Add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/result/Add_grad/Shape'train/gradients/result/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
)train/gradients/result/Add_grad/Reshape_1Reshape%train/gradients/result/Add_grad/Sum_1'train/gradients/result/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/result/Add_grad/tuple/group_depsNoOp(^train/gradients/result/Add_grad/Reshape*^train/gradients/result/Add_grad/Reshape_1
�
8train/gradients/result/Add_grad/tuple/control_dependencyIdentity'train/gradients/result/Add_grad/Reshape1^train/gradients/result/Add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*:
_class0
.,loc:@train/gradients/result/Add_grad/Reshape
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
+train/gradients/result/MatMul_grad/MatMul_1MatMulresult/Relu8train/gradients/result/Add_grad/tuple/control_dependency*
_output_shapes

:d*
transpose_a(*
transpose_b( *
T0
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
6train/gradients/layer_3/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/Add_grad/Shape(train/gradients/layer_3/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/Add_grad/SumSum)train/gradients/result/Relu_grad/ReluGrad6train/gradients/layer_3/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
&train/gradients/layer_3/Add_grad/Sum_1Sum)train/gradients/result/Relu_grad/ReluGrad8train/gradients/layer_3/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
;train/gradients/layer_3/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/Add_grad/Reshape_12^train/gradients/layer_3/Add_grad/tuple/group_deps*
_output_shapes
:d*
T0*=
_class3
1/loc:@train/gradients/layer_3/Add_grad/Reshape_1
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/Add_grad/tuple/control_dependencyweights_1/weight3/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
6train/gradients/layer_2/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/Add_grad/Shape(train/gradients/layer_2/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
*train/gradients/layer_2/Add_grad/Reshape_1Reshape&train/gradients/layer_2/Add_grad/Sum_1(train/gradients/layer_2/Add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1* 
_output_shapes
:
��
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
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
(train/gradients/layer_1/Add_grad/ReshapeReshape$train/gradients/layer_1/Add_grad/Sum&train/gradients/layer_1/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_1/Add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_1/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/Add_grad/tuple/control_dependencyweights_1/weight1/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1
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
VariableV2*!
_class
loc:@biases_1/bias1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
VariableV2*
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight1*
	container 
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
weights_1/weight1/Adam/readIdentityweights_1/weight1/Adam*
T0*$
_class
loc:@weights_1/weight1* 
_output_shapes
:
��
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
weights_1/weight1/Adam_1/AssignAssignweights_1/weight1/Adam_1*weights_1/weight1/Adam_1/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
VariableV2*$
_class
loc:@weights_1/weight2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
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
weights_1/weight2/Adam_1/AssignAssignweights_1/weight2/Adam_1*weights_1/weight2/Adam_1/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
weights_1/weight_out/Adam/readIdentityweights_1/weight_out/Adam*
_output_shapes

:d*
T0*'
_class
loc:@weights_1/weight_out
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
"weights_1/weight_out/Adam_1/AssignAssignweights_1/weight_out/Adam_1-weights_1/weight_out/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out
�
 weights_1/weight_out/Adam_1/readIdentityweights_1/weight_out/Adam_1*
T0*'
_class
loc:@weights_1/weight_out*
_output_shapes

:d
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias1
�
biases_1/bias1/Adam/AssignAssignbiases_1/bias1/Adam%biases_1/bias1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
�
biases_1/bias1/Adam/readIdentitybiases_1/bias1/Adam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes	
:�
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
VariableV2*
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
biases_1/bias1/Adam_1/AssignAssignbiases_1/bias1/Adam_1'biases_1/bias1/Adam_1/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
biases_1/bias1/Adam_1/readIdentitybiases_1/bias1/Adam_1*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes	
:�
�
%biases_1/bias2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias2*
valueB�*    
�
biases_1/bias2/Adam
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias2*
	container 
�
biases_1/bias2/Adam_1/AssignAssignbiases_1/bias2/Adam_1'biases_1/bias2/Adam_1/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
biases_1/bias2/Adam_1/readIdentitybiases_1/bias2/Adam_1*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias2
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
biases_1/bias3/Adam/AssignAssignbiases_1/bias3/Adam%biases_1/bias3/Adam/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
�
biases_1/bias3/Adam/readIdentitybiases_1/bias3/Adam*
_output_shapes
:d*
T0*!
_class
loc:@biases_1/bias3
�
'biases_1/bias3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:d*!
_class
loc:@biases_1/bias3*
valueBd*    
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
biases_1/bias3/Adam_1/AssignAssignbiases_1/bias3/Adam_1'biases_1/bias3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out*
	container *
shape:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out*
	container *
shape:
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
biases_1/bias_out/Adam_1/readIdentitybiases_1/bias_out/Adam_1*
T0*$
_class
loc:@biases_1/bias_out*
_output_shapes
:
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
-train/Adam/update_weights_1/weight1/ApplyAdam	ApplyAdamweights_1/weight1weights_1/weight1/Adamweights_1/weight1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@weights_1/weight1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
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
*train/Adam/update_biases_1/bias1/ApplyAdam	ApplyAdambiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias1*
use_nesterov( *
_output_shapes	
:�
�
*train/Adam/update_biases_1/bias2/ApplyAdam	ApplyAdambiases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/Add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@biases_1/bias2*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
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
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking( 
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

train/Adam	AssignAddVariabletrain/Adam/value*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
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
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
r
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBbiases/bias1
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_2/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBbiases/bias2
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
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
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
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assignbiases/bias_outsave/RestoreV2_4*
use_locking(*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:
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
save/Assign_5Assignbiases_1/bias1save/RestoreV2_5*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
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
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_7Assignbiases_1/bias1/Adam_1save/RestoreV2_7*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
t
save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBbiases_1/bias2
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assignbiases_1/bias2/Adamsave/RestoreV2_9*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
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
save/RestoreV2_12/tensor_namesConst*(
valueBBbiases_1/bias3/Adam*
dtype0*
_output_shapes
:
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
save/Assign_12Assignbiases_1/bias3/Adamsave/RestoreV2_12*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
|
save/RestoreV2_13/tensor_namesConst*
dtype0*
_output_shapes
:**
value!BBbiases_1/bias3/Adam_1
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_14Assignbiases_1/bias_outsave/RestoreV2_14*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
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
save/Assign_15Assignbiases_1/bias_out/Adamsave/RestoreV2_15*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:

save/RestoreV2_16/tensor_namesConst*-
value$B"Bbiases_1/bias_out/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assignbiases_1/bias_out/Adam_1save/RestoreV2_16*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
x
save/RestoreV2_17/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBtrain/beta1_power
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
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_18Assigntrain/beta2_powersave/RestoreV2_18*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
save/Assign_19Assignweights/weight1save/RestoreV2_19*
T0*"
_class
loc:@weights/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
v
save/RestoreV2_20/tensor_namesConst*
dtype0*
_output_shapes
:*$
valueBBweights/weight2
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_21Assignweights/weight3save/RestoreV2_21*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*"
_class
loc:@weights/weight3
y
save/RestoreV2_22/tensor_namesConst*
dtype0*
_output_shapes
:*'
valueBBweights/weight_out
k
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*%
_class
loc:@weights/weight_out
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
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_27/tensor_namesConst*
dtype0*
_output_shapes
:*+
value"B Bweights_1/weight2/Adam
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
save/Assign_28Assignweights_1/weight2/Adam_1save/RestoreV2_28*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
x
save/RestoreV2_29/tensor_namesConst*&
valueBBweights_1/weight3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_29Assignweights_1/weight3save/RestoreV2_29*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
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
save/Assign_30Assignweights_1/weight3/Adamsave/RestoreV2_30*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(

save/RestoreV2_31/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bweights_1/weight3/Adam_1
k
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Assign_32Assignweights_1/weight_outsave/RestoreV2_32*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assignweights_1/weight_out/Adamsave/RestoreV2_33*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
save/Assign_34Assignweights_1/weight_out/Adam_1save/RestoreV2_34*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
:*

Tidx0*
	keep_dims( 
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
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
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
T
error_R/tagsConst*
valueB Berror_R*
dtype0*
_output_shapes
: 
V
error_RScalarSummaryerror_R/tagsstrided_slice*
T0*
_output_shapes
: 
d
learning_rate_1/tagsConst* 
valueB Blearning_rate_1*
dtype0*
_output_shapes
: 
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
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0��;;F       r5��	1�H���A*;


total_loss#�@

error_R!�<?

learning_rate_1�%�8%���H       ��H�	$$H���A*;


total_loss�
�@

error_R@�I?

learning_rate_1�%�8˽eMH       ��H�	�uH���A*;


total_loss%�@

error_RA�N?

learning_rate_1�%�8f�*H       ��H�	�H���A*;


total_loss
�@

error_R��W?

learning_rate_1�%�8��
�H       ��H�	� H���A*;


total_loss�A

error_RORf?

learning_rate_1�%�8-x�H       ��H�	�NH���A*;


total_loss�yA

error_R��O?

learning_rate_1�%�8G5H       ��H�	��H���A*;


total_loss��@

error_R&cD?

learning_rate_1�%�8��)H       ��H�	�H���A*;


total_loss���@

error_R$\X?

learning_rate_1�%�8j���H       ��H�	�H���A*;


total_lossQ��@

error_R�e?

learning_rate_1�%�8ɭ5|H       ��H�	�WH���A	*;


total_lossM�@

error_RJF?

learning_rate_1�%�8��� H       ��H�	H���A
*;


total_loss.2�@

error_R2�K?

learning_rate_1�%�8��bH       ��H�	>�H���A*;


total_losszr�@

error_R;iI?

learning_rate_1�%�8y�Y�H       ��H�	\#H���A*;


total_lossQ��@

error_R&I?

learning_rate_1�%�8e��qH       ��H�	�hH���A*;


total_loss�a�@

error_R��O?

learning_rate_1�%�8���H       ��H�	Z�H���A*;


total_loss� �@

error_RH�H?

learning_rate_1�%�8�~@H       ��H�	��H���A*;


total_loss��z@

error_R��Y?

learning_rate_1�%�8�o�xH       ��H�	5H���A*;


total_loss�I�@

error_Rh�h?

learning_rate_1�%�8���?H       ��H�	yH���A*;


total_lossD��@

error_RO�I?

learning_rate_1�%�8��$H       ��H�	��H���A*;


total_loss\��@

error_R��@?

learning_rate_1�%�8n��H       ��H�	�H���A*;


total_lossw��@

error_R��G?

learning_rate_1�%�8���H       ��H�	�EH���A*;


total_lossIA

error_R��R?

learning_rate_1�%�8� H       ��H�	�H���A*;


total_lossTv�@

error_RcC?

learning_rate_1�%�8x=�H       ��H�	��H���A*;


total_loss�.�@

error_R��7?

learning_rate_1�%�8<��H       ��H�	�=H���A*;


total_loss㐚@

error_R6[L?

learning_rate_1�%�8v��H       ��H�	ЉH���A*;


total_lossl�n@

error_R}~A?

learning_rate_1�%�8tnH       ��H�	��H���A*;


total_lossZ�@

error_R)q;?

learning_rate_1�%�8�s|�H       ��H�	pE H���A*;


total_loss],�@

error_RL?

learning_rate_1�%�8�B��H       ��H�	� H���A*;


total_loss���@

error_R`�V?

learning_rate_1�%�8|풅H       ��H�	T� H���A*;


total_loss�@

error_RE�Z?

learning_rate_1�%�8���6H       ��H�	�!H���A*;


total_lossΉ~@

error_RW�P?

learning_rate_1�%�8�YFxH       ��H�	]!H���A*;


total_lossi��@

error_R��S?

learning_rate_1�%�8!� H       ��H�	(�!H���A*;


total_loss��@

error_RHW?

learning_rate_1�%�8�H       ��H�	2�!H���A *;


total_loss��@

error_R��S?

learning_rate_1�%�83S�yH       ��H�	�)"H���A!*;


total_lossJ��@

error_R�M?

learning_rate_1�%�8&?<}H       ��H�	m"H���A"*;


total_loss��@

error_R��U?

learning_rate_1�%�8D���H       ��H�	Ȳ"H���A#*;


total_loss�Ϯ@

error_R��C?

learning_rate_1�%�8ͻ�H       ��H�	��"H���A$*;


total_loss�ڲ@

error_RCUH?

learning_rate_1�%�8�*�H       ��H�	�:#H���A%*;


total_loss�3|@

error_R4%U?

learning_rate_1�%�8_�llH       ��H�	#H���A&*;


total_loss�x�@

error_R!Z?

learning_rate_1�%�8�Ru�H       ��H�	"�#H���A'*;


total_loss-A

error_RN
h?

learning_rate_1�%�8�˹�H       ��H�	!$H���A(*;


total_lossZ:�@

error_R��G?

learning_rate_1�%�8V��H       ��H�	sI$H���A)*;


total_loss�v�@

error_R�VM?

learning_rate_1�%�8���H       ��H�	��$H���A**;


total_loss<��@

error_R�@?

learning_rate_1�%�8�� �H       ��H�	��$H���A+*;


total_loss�ɩ@

error_R=[\?

learning_rate_1�%�8��RrH       ��H�	-%H���A,*;


total_lossW�@

error_R��S?

learning_rate_1�%�8�ms�H       ��H�	�b%H���A-*;


total_lossl�@

error_R��G?

learning_rate_1�%�8Q=n�H       ��H�	n�%H���A.*;


total_loss��@

error_R �??

learning_rate_1�%�8-ꬭH       ��H�	�&H���A/*;


total_lossl
A

error_RX�H?

learning_rate_1�%�8��bH       ��H�	�X&H���A0*;


total_loss��@

error_R��=?

learning_rate_1�%�8�ҩ�H       ��H�	�&H���A1*;


total_lossR��@

error_R�U?

learning_rate_1�%�8�� �H       ��H�	F�&H���A2*;


total_loss�a�@

error_Rc�C?

learning_rate_1�%�8!���H       ��H�	,'H���A3*;


total_loss��@

error_R�Q?

learning_rate_1�%�8e��H       ��H�	s'H���A4*;


total_lossh��@

error_R��T?

learning_rate_1�%�8��H       ��H�	p�'H���A5*;


total_lossql�@

error_Rs�`?

learning_rate_1�%�8�N��H       ��H�	�(H���A6*;


total_loss�@

error_R�*L?

learning_rate_1�%�8�'vH       ��H�	�H(H���A7*;


total_lossd��@

error_R�iO?

learning_rate_1�%�8 �4�H       ��H�	ʭ(H���A8*;


total_loss�^�@

error_R��??

learning_rate_1�%�8u��H       ��H�	�)H���A9*;


total_lossά�@

error_R�7P?

learning_rate_1�%�8�(WH       ��H�	N)H���A:*;


total_loss$��@

error_R��G?

learning_rate_1�%�8�yR�H       ��H�	%�)H���A;*;


total_loss%�@

error_R}�N?

learning_rate_1�%�87��qH       ��H�	�)H���A<*;


total_loss��@

error_Rn5[?

learning_rate_1�%�84�3H       ��H�	H*H���A=*;


total_loss�~�@

error_R�Y?

learning_rate_1�%�85e�H       ��H�	�]*H���A>*;


total_loss �@

error_R�G?

learning_rate_1�%�8�$�H       ��H�	m�*H���A?*;


total_loss܆�@

error_R�>?

learning_rate_1�%�8gNk�H       ��H�	��*H���A@*;


total_loss��@

error_R�E_?

learning_rate_1�%�8��;�H       ��H�	�,+H���AA*;


total_lossmo�@

error_RlW?

learning_rate_1�%�8c��H       ��H�	<u+H���AB*;


total_loss��A

error_R��K?

learning_rate_1�%�8�1�H       ��H�	Լ+H���AC*;


total_loss�!�@

error_Rڶ^?

learning_rate_1�%�8���|H       ��H�	��+H���AD*;


total_lossM�A

error_R�;R?

learning_rate_1�%�8d\�FH       ��H�	H,H���AE*;


total_lossc�s@

error_R�,L?

learning_rate_1�%�8ϥ�H       ��H�	�,H���AF*;


total_lossJ/�@

error_RCyD?

learning_rate_1�%�8<��H       ��H�	��,H���AG*;


total_loss���@

error_R�^?

learning_rate_1�%�8I�|�H       ��H�	-H���AH*;


total_loss�h�@

error_Rt>U?

learning_rate_1�%�8�inH       ��H�	�b-H���AI*;


total_loss���@

error_RzuF?

learning_rate_1�%�8Ҫ��H       ��H�	��-H���AJ*;


total_lossV��@

error_RO�b?

learning_rate_1�%�8��ʌH       ��H�	��-H���AK*;


total_lossi��@

error_R$�A?

learning_rate_1�%�8�ȁH       ��H�	�3.H���AL*;


total_loss*և@

error_RR�C?

learning_rate_1�%�8�?�H       ��H�	Sv.H���AM*;


total_loss�ӷ@

error_R��U?

learning_rate_1�%�8,���H       ��H�	\�.H���AN*;


total_lossaɺ@

error_RkM?

learning_rate_1�%�8���H       ��H�	&�.H���AO*;


total_loss��@

error_R�oO?

learning_rate_1�%�8��#�H       ��H�	�=/H���AP*;


total_loss�A

error_R_eU?

learning_rate_1�%�8��/H       ��H�	��/H���AQ*;


total_lossρ�@

error_R��Y?

learning_rate_1�%�8�/��H       ��H�	��/H���AR*;


total_loss*R�@

error_R_�B?

learning_rate_1�%�8��H       ��H�	T0H���AS*;


total_loss]� A

error_R�lE?

learning_rate_1�%�8�G�H       ��H�	�Z0H���AT*;


total_lossv A

error_Ri�L?

learning_rate_1�%�8��:�H       ��H�	^�0H���AU*;


total_loss��@

error_R��A?

learning_rate_1�%�8�H       ��H�	��0H���AV*;


total_lossl�@

error_RC�X?

learning_rate_1�%�8�V�H       ��H�	�21H���AW*;


total_loss��@

error_R�Z?

learning_rate_1�%�8J!�%H       ��H�	�z1H���AX*;


total_loss�g�@

error_R&�W?

learning_rate_1�%�8��~aH       ��H�	H�1H���AY*;


total_loss���@

error_ROFQ?

learning_rate_1�%�8-�z�H       ��H�	�2H���AZ*;


total_loss��@

error_RHU:?

learning_rate_1�%�8��DH       ��H�	tR2H���A[*;


total_lossѳ�@

error_R�S7?

learning_rate_1�%�8��H       ��H�	�2H���A\*;


total_lossNϱ@

error_RܭI?

learning_rate_1�%�8�
��H       ��H�	��2H���A]*;


total_loss+A

error_R6�L?

learning_rate_1�%�8_�>�H       ��H�	%3H���A^*;


total_loss�w�@

error_R��D?

learning_rate_1�%�8�
�FH       ��H�	;j3H���A_*;


total_loss4:�@

error_R��`?

learning_rate_1�%�8uc��H       ��H�	��3H���A`*;


total_loss�_�@

error_RqJX?

learning_rate_1�%�8�M!�H       ��H�	��3H���Aa*;


total_loss�̩@

error_Re�G?

learning_rate_1�%�8YˏH       ��H�	�84H���Ab*;


total_loss�A

error_R X?

learning_rate_1�%�8��pH       ��H�	�~4H���Ac*;


total_lossSd�@

error_RH_E?

learning_rate_1�%�8+���H       ��H�	��4H���Ad*;


total_loss�^�@

error_RRH?

learning_rate_1�%�8O�A�H       ��H�	�5H���Ae*;


total_loss`x�@

error_ROIY?

learning_rate_1�%�8�#5�H       ��H�		O5H���Af*;


total_loss:}�@

error_R8=V?

learning_rate_1�%�8��4FH       ��H�	9�5H���Ag*;


total_loss�m�@

error_R�DU?

learning_rate_1�%�8��D�H       ��H�	�5H���Ah*;


total_lossaCA

error_R�&Q?

learning_rate_1�%�8FH�H       ��H�	�6H���Ai*;


total_loss�\�@

error_R��N?

learning_rate_1�%�8��.�H       ��H�	A\6H���Aj*;


total_loss��@

error_R�)S?

learning_rate_1�%�8��B�H       ��H�	x�6H���Ak*;


total_loss���@

error_RԶL?

learning_rate_1�%�8f�WH       ��H�	!�6H���Al*;


total_loss᭝@

error_RÃM?

learning_rate_1�%�8^w�H       ��H�	�(7H���Am*;


total_loss`%�@

error_R��W?

learning_rate_1�%�8���BH       ��H�	�k7H���An*;


total_loss໧@

error_R�0d?

learning_rate_1�%�8�^LH       ��H�	��7H���Ao*;


total_lossq��@

error_R_ S?

learning_rate_1�%�8$�hH       ��H�	��7H���Ap*;


total_loss ��@

error_Rg?

learning_rate_1�%�8;�$�H       ��H�	�68H���Aq*;


total_loss���@

error_R�}V?

learning_rate_1�%�8�T��H       ��H�	j�8H���Ar*;


total_lossA

error_R!�N?

learning_rate_1�%�8	�7�H       ��H�	t�8H���As*;


total_loss��t@

error_R1K?

learning_rate_1�%�8�!�H       ��H�	�#9H���At*;


total_lossC�@

error_RQ$Q?

learning_rate_1�%�8K��RH       ��H�	fg9H���Au*;


total_loss��A

error_R�f/?

learning_rate_1�%�8�v�SH       ��H�	��9H���Av*;


total_loss|�A

error_R D?

learning_rate_1�%�8v�WH       ��H�	[�9H���Aw*;


total_loss	��@

error_R�:?

learning_rate_1�%�8��lH       ��H�	F::H���Ax*;


total_lossŎA

error_R
w?

learning_rate_1�%�8��tH       ��H�	��:H���Ay*;


total_lossLe@

error_R�]?

learning_rate_1�%�8`���H       ��H�	�:H���Az*;


total_lossx̵@

error_R�lO?

learning_rate_1�%�8!���H       ��H�	f;H���A{*;


total_loss���@

error_R��C?

learning_rate_1�%�8�U��H       ��H�	�E;H���A|*;


total_lossIo�@

error_R{B?

learning_rate_1�%�8�C~�H       ��H�	7�;H���A}*;


total_loss���@

error_R��O?

learning_rate_1�%�8����H       ��H�	>�;H���A~*;


total_lossI�@

error_R<�S?

learning_rate_1�%�8J���H       ��H�	[<H���A*;


total_lossHi�@

error_R�^Z?

learning_rate_1�%�8c�TDI       6%�	,Y<H���A�*;


total_loss ��@

error_R�NU?

learning_rate_1�%�8��-I       6%�	-�<H���A�*;


total_loss���@

error_R)�[?

learning_rate_1�%�8�E�aI       6%�	,�<H���A�*;


total_loss�H�@

error_R/2I?

learning_rate_1�%�8�X�I       6%�	�(=H���A�*;


total_loss#d�@

error_R�$U?

learning_rate_1�%�8�xH�I       6%�	q=H���A�*;


total_lossG��@

error_R;O?

learning_rate_1�%�8e��I       6%�	�=H���A�*;


total_lossG��@

error_R\G?

learning_rate_1�%�8�U�tI       6%�	�=H���A�*;


total_lossoܮ@

error_R�W?

learning_rate_1�%�8U��AI       6%�	�B>H���A�*;


total_loss(.�@

error_RFN?

learning_rate_1�%�8{>�I       6%�	��>H���A�*;


total_loss���@

error_R�=:?

learning_rate_1�%�8Ww��I       6%�	��>H���A�*;


total_lossHB�@

error_R�;?

learning_rate_1�%�8��y2I       6%�	�?H���A�*;


total_lossqY�@

error_R7lO?

learning_rate_1�%�8�a��I       6%�	�c?H���A�*;


total_losso��@

error_RZbB?

learning_rate_1�%�8ox�!I       6%�	�?H���A�*;


total_loss�Y�@

error_R��R?

learning_rate_1�%�8<yaI       6%�	i�?H���A�*;


total_loss ��@

error_R�y;?

learning_rate_1�%�8/�#I       6%�	�>@H���A�*;


total_loss���@

error_R�!U?

learning_rate_1�%�8J��eI       6%�	_�@H���A�*;


total_loss�u�@

error_R#�O?

learning_rate_1�%�8��tI       6%�	��@H���A�*;


total_loss��@

error_R�O?

learning_rate_1�%�8�:iI       6%�	AH���A�*;


total_loss�:�@

error_R�7U?

learning_rate_1�%�8ȟXI       6%�	�JAH���A�*;


total_loss1��@

error_R�]K?

learning_rate_1�%�8�r/�I       6%�	=�AH���A�*;


total_lossMS�@

error_Rq�L?

learning_rate_1�%�8�~��I       6%�	@�AH���A�*;


total_loss���@

error_R�8U?

learning_rate_1�%�8jOcI       6%�	HBH���A�*;


total_loss�#�@

error_RE�??

learning_rate_1�%�8f4��I       6%�	�XBH���A�*;


total_loss��
A

error_R�ZH?

learning_rate_1�%�8�S�NI       6%�	4�BH���A�*;


total_lossO)�@

error_R �O?

learning_rate_1�%�8L~I       6%�	��BH���A�*;


total_loss���@

error_R	�E?

learning_rate_1�%�8��iLI       6%�	"CH���A�*;


total_loss�qA

error_R=YJ?

learning_rate_1�%�8��iI       6%�	�eCH���A�*;


total_loss�-�@

error_R�??

learning_rate_1�%�8_lt�I       6%�	ƪCH���A�*;


total_lossj=�@

error_R#�V?

learning_rate_1�%�8mRI       6%�	R�CH���A�*;


total_loss���@

error_R��Z?

learning_rate_1�%�8�|�I       6%�	9DH���A�*;


total_loss��A

error_R��S?

learning_rate_1�%�8MO�BI       6%�	 |DH���A�*;


total_loss���@

error_RfQ?

learning_rate_1�%�8�&��I       6%�	˾DH���A�*;


total_loss�k�@

error_RC&G?

learning_rate_1�%�8��I       6%�	Y EH���A�*;


total_loss���@

error_REH?

learning_rate_1�%�8BQmI       6%�	>REH���A�*;


total_loss��q@

error_R�G?

learning_rate_1�%�8�,��I       6%�	��EH���A�*;


total_loss�rj@

error_R�>M?

learning_rate_1�%�8U�ޢI       6%�	N�EH���A�*;


total_loss�b�@

error_R�F?

learning_rate_1�%�8��q�I       6%�	�KFH���A�*;


total_loss��@

error_R��X?

learning_rate_1�%�8���I       6%�	��FH���A�*;


total_loss�˥@

error_R�dB?

learning_rate_1�%�8�T�/I       6%�	��FH���A�*;


total_loss73�@

error_Rs�=?

learning_rate_1�%�8�T�dI       6%�	6GH���A�*;


total_lossQ�x@

error_RF�G?

learning_rate_1�%�8۸��I       6%�	5|GH���A�*;


total_lossw��@

error_R!�U?

learning_rate_1�%�84�I       6%�	��GH���A�*;


total_loss�,"A

error_RC�K?

learning_rate_1�%�8�<�5I       6%�	�HH���A�*;


total_loss�&�@

error_R��`?

learning_rate_1�%�8^'�I       6%�	CXHH���A�*;


total_loss(��@

error_R�\?

learning_rate_1�%�8h�E�I       6%�	�HH���A�*;


total_lossOu�@

error_R�GR?

learning_rate_1�%�8�րI       6%�	�IH���A�*;


total_lossϗA

error_R=dI?

learning_rate_1�%�8�bb<I       6%�	UIH���A�*;


total_lossKZ�@

error_Rd�;?

learning_rate_1�%�8��I       6%�	מIH���A�*;


total_loss���@

error_R�a?

learning_rate_1�%�8����I       6%�	��IH���A�*;


total_loss�#�@

error_Rt�R?

learning_rate_1�%�8U��wI       6%�	Q2JH���A�*;


total_losse&�@

error_R��T?

learning_rate_1�%�8pw\JI       6%�	FwJH���A�*;


total_loss<��@

error_RH5U?

learning_rate_1�%�8O��XI       6%�	D�JH���A�*;


total_loss�؋@

error_R�W?

learning_rate_1�%�8Ì�I       6%�	# KH���A�*;


total_loss�Q�@

error_RS�M?

learning_rate_1�%�8b��I       6%�	UIKH���A�*;


total_loss�7A

error_RatP?

learning_rate_1�%�8D��)I       6%�	�KH���A�*;


total_lossE��@

error_RsNN?

learning_rate_1�%�8Yv)�I       6%�	��KH���A�*;


total_loss�6�@

error_RO�[?

learning_rate_1�%�8�b�WI       6%�	�LH���A�*;


total_lossh��@

error_R?\?

learning_rate_1�%�8 I       6%�	�^LH���A�*;


total_lossd�	A

error_RHER?

learning_rate_1�%�8�N�+I       6%�	��LH���A�*;


total_loss_Y�@

error_R �l?

learning_rate_1�%�8+)�I       6%�	��LH���A�*;


total_loss��@

error_R�XJ?

learning_rate_1�%�8��-�I       6%�	�-MH���A�*;


total_lossim�@

error_R��F?

learning_rate_1�%�8V_�I       6%�	zxMH���A�*;


total_lossH�A

error_R�X?

learning_rate_1�%�8��s*I       6%�	��MH���A�*;


total_loss�ޡ@

error_R��S?

learning_rate_1�%�8S�;I       6%�	NH���A�*;


total_loss,��@

error_R��2?

learning_rate_1�%�8�:��I       6%�	mPNH���A�*;


total_loss��@

error_RҦE?

learning_rate_1�%�8�]II       6%�	>�NH���A�*;


total_loss��A

error_RlSC?

learning_rate_1�%�8���iI       6%�	��NH���A�*;


total_loss���@

error_R�H?

learning_rate_1�%�8��{)I       6%�	�OH���A�*;


total_loss��@

error_R��4?

learning_rate_1�%�8���I       6%�	o]OH���A�*;


total_loss螹@

error_R�B?

learning_rate_1�%�8u9��I       6%�	�OH���A�*;


total_loss{FA

error_R��G?

learning_rate_1�%�8���XI       6%�	*�OH���A�*;


total_loss���@

error_R��Q?

learning_rate_1�%�8�a��I       6%�	�%PH���A�*;


total_loss���@

error_R=�X?

learning_rate_1�%�8QQT�I       6%�	�iPH���A�*;


total_loss�6A

error_R�EO?

learning_rate_1�%�8`,�mI       6%�	�PH���A�*;


total_loss<�@

error_R�	[?

learning_rate_1�%�8�o�I       6%�	��PH���A�*;


total_loss�"�@

error_RZL?

learning_rate_1�%�8��ՔI       6%�	_3QH���A�*;


total_loss<p�@

error_RJ�D?

learning_rate_1�%�8�v\�I       6%�	EvQH���A�*;


total_loss�m�@

error_R&�H?

learning_rate_1�%�8ץ~I       6%�	z�QH���A�*;


total_loss��A

error_RڲL?

learning_rate_1�%�8��zQI       6%�	��QH���A�*;


total_loss�Ҙ@

error_R��D?

learning_rate_1�%�8��X�I       6%�	�FRH���A�*;


total_loss_��@

error_R�R?

learning_rate_1�%�8q4aI       6%�	8�RH���A�*;


total_loss]�@

error_RV�Y?

learning_rate_1�%�8��WI       6%�	��RH���A�*;


total_lossAد@

error_R��S?

learning_rate_1�%�8�*�I       6%�	?SH���A�*;


total_loss� �@

error_R,tG?

learning_rate_1�%�8�p@I       6%�	�VSH���A�*;


total_loss�A

error_R��Q?

learning_rate_1�%�8�PI       6%�	��SH���A�*;


total_lossl}�@

error_R�K?

learning_rate_1�%�8+1��I       6%�	h�SH���A�*;


total_loss�@

error_RZ�D?

learning_rate_1�%�8us)�I       6%�	�)TH���A�*;


total_lossSg�@

error_R_FW?

learning_rate_1�%�8���I       6%�	�~TH���A�*;


total_losstV�@

error_R��O?

learning_rate_1�%�8�Ǜ,I       6%�	��TH���A�*;


total_loss21�@

error_RZmK?

learning_rate_1�%�8]�YI       6%�	�UH���A�*;


total_loss��@

error_R#iL?

learning_rate_1�%�8�n��I       6%�	�IUH���A�*;


total_lossO<�@

error_R�T?

learning_rate_1�%�8�C��I       6%�	9�UH���A�*;


total_loss!�@

error_R�KA?

learning_rate_1�%�8ms�zI       6%�	H�UH���A�*;


total_lossLn�@

error_Ra8L?

learning_rate_1�%�8Mk|�I       6%�	�VH���A�*;


total_loss��@

error_R�L?

learning_rate_1�%�8*�RI       6%�	�dVH���A�*;


total_loss2��@

error_R��H?

learning_rate_1�%�8�OrI       6%�	A�VH���A�*;


total_loss�7�@

error_R�O?

learning_rate_1�%�8( *7I       6%�	��VH���A�*;


total_loss4��@

error_Rr�8?

learning_rate_1�%�8_DnMI       6%�	 3WH���A�*;


total_loss�!�@

error_R;�X?

learning_rate_1�%�8��W�I       6%�	�zWH���A�*;


total_lossH\�@

error_R�:]?

learning_rate_1�%�8�VI       6%�	Z�WH���A�*;


total_lossJ��@

error_R�O?

learning_rate_1�%�8-7'I       6%�	BXH���A�*;


total_loss8?�@

error_R6�N?

learning_rate_1�%�8�I3I       6%�	eXXH���A�*;


total_lossvac@

error_R{�N?

learning_rate_1�%�8��RfI       6%�	@�XH���A�*;


total_loss},�@

error_RO�L?

learning_rate_1�%�8}(.I       6%�	�YH���A�*;


total_loss�m�@

error_R�J?

learning_rate_1�%�8&�]�I       6%�	aRYH���A�*;


total_loss�\�@

error_R.�J?

learning_rate_1�%�8ng�/I       6%�	ȘYH���A�*;


total_lossQ2�@

error_R�{V?

learning_rate_1�%�8C[�I       6%�	��YH���A�*;


total_loss���@

error_R&iB?

learning_rate_1�%�8	HyI       6%�	�#ZH���A�*;


total_lossl[A

error_RTY?

learning_rate_1�%�8����I       6%�	CkZH���A�*;


total_loss��@

error_R�|Z?

learning_rate_1�%�8�pq�I       6%�	u�ZH���A�*;


total_loss_щ@

error_R�,i?

learning_rate_1�%�8�˦\I       6%�	{ [H���A�*;


total_lossSǻ@

error_R��`?

learning_rate_1�%�87(|PI       6%�	�J[H���A�*;


total_loss@

error_R��i?

learning_rate_1�%�8��GI       6%�	��[H���A�*;


total_loss)'�@

error_R�'O?

learning_rate_1�%�8���I       6%�	��[H���A�*;


total_loss�W�@

error_R��K?

learning_rate_1�%�8�J�(I       6%�	`\H���A�*;


total_loss�]v@

error_R�G?

learning_rate_1�%�8m/	1I       6%�	n`\H���A�*;


total_loss*C�@

error_R�S?

learning_rate_1�%�8gZ0I       6%�	�\H���A�*;


total_lossW��@

error_R�^U?

learning_rate_1�%�8@�N�I       6%�	*�\H���A�*;


total_lossAQ�@

error_R�T?

learning_rate_1�%�8�WqI       6%�	�(]H���A�*;


total_lossYf�@

error_R��R?

learning_rate_1�%�8R�I       6%�	�m]H���A�*;


total_loss4�@

error_RlMN?

learning_rate_1�%�8�q�XI       6%�	2�]H���A�*;


total_loss���@

error_RZ�g?

learning_rate_1�%�8��D I       6%�	A�]H���A�*;


total_losss�@

error_R�T?

learning_rate_1�%�8�t�$I       6%�	:^H���A�*;


total_loss��@

error_R��W?

learning_rate_1�%�8Q�ޔI       6%�	�^H���A�*;


total_loss��@

error_R=Q?

learning_rate_1�%�8��j�I       6%�	��^H���A�*;


total_loss�[A

error_R�oL?

learning_rate_1�%�8����I       6%�	R_H���A�*;


total_loss
�@

error_RC1[?

learning_rate_1�%�8z޵I       6%�	�S_H���A�*;


total_loss5m�@

error_Rm�H?

learning_rate_1�%�8ݍ�0I       6%�	�_H���A�*;


total_lossi��@

error_Ra�M?

learning_rate_1�%�8ƙ�I       6%�	��_H���A�*;


total_lossx�A

error_R3�G?

learning_rate_1�%�8�UeI       6%�	. `H���A�*;


total_loss�@

error_R}Y?

learning_rate_1�%�8�Z��I       6%�	n`H���A�*;


total_loss�e�@

error_R��l?

learning_rate_1�%�87�bI       6%�	��`H���A�*;


total_loss�p�@

error_R�CV?

learning_rate_1�%�8l���I       6%�	�aH���A�*;


total_loss��@

error_R��M?

learning_rate_1�%�8ξ jI       6%�	$GaH���A�*;


total_loss��f@

error_RM�>?

learning_rate_1�%�8�$I       6%�	v�aH���A�*;


total_loss�N�@

error_RIOR?

learning_rate_1�%�8�9I       6%�	4�aH���A�*;


total_lossccA

error_R�"S?

learning_rate_1�%�8,�Q�I       6%�	�bH���A�*;


total_loss�@

error_R�W=?

learning_rate_1�%�8*�|�I       6%�	�YbH���A�*;


total_loss�ڠ@

error_R܁E?

learning_rate_1�%�8$C�2I       6%�	ӜbH���A�*;


total_lossq�@

error_R�)Z?

learning_rate_1�%�8F��I       6%�	��bH���A�*;


total_loss?��@

error_Rs9P?

learning_rate_1�%�8X�{8I       6%�	�"cH���A�*;


total_loss\��@

error_R�W?

learning_rate_1�%�8�Z�I       6%�	]fcH���A�*;


total_lossᦻ@

error_R��M?

learning_rate_1�%�8杖�I       6%�	v�cH���A�*;


total_loss�h�@

error_R׭T?

learning_rate_1�%�8w�g5I       6%�	��cH���A�*;


total_loss���@

error_R�R?

learning_rate_1�%�8L�I       6%�	4dH���A�*;


total_lossJ�@

error_R�~N?

learning_rate_1�%�8���I       6%�	mzdH���A�*;


total_loss��A

error_RO�g?

learning_rate_1�%�8Q��I       6%�	�dH���A�*;


total_lossV�@

error_R\�O?

learning_rate_1�%�8�#wEI       6%�	qeH���A�*;


total_lossؓ�@

error_RFN?

learning_rate_1�%�8{�&!I       6%�	
PeH���A�*;


total_loss��A

error_R!�N?

learning_rate_1�%�8�k}�I       6%�	��eH���A�*;


total_lossw=�@

error_RE[V?

learning_rate_1�%�8����I       6%�	��eH���A�*;


total_loss�!�@

error_R�G?

learning_rate_1�%�8��I       6%�	�"fH���A�*;


total_loss��@

error_R�I?

learning_rate_1�%�8n���I       6%�	�mfH���A�*;


total_loss Ň@

error_R�8?

learning_rate_1�%�8Y�-I       6%�	)�fH���A�*;


total_loss��@

error_R�[O?

learning_rate_1�%�8GG,pI       6%�	?gH���A�*;


total_loss�@

error_R�>W?

learning_rate_1�%�808�eI       6%�	�NgH���A�*;


total_loss�(�@

error_R�<?

learning_rate_1�%�8��Q�I       6%�	H�gH���A�*;


total_loss�@

error_R�,L?

learning_rate_1�%�82��(I       6%�	�gH���A�*;


total_loss�˦@

error_R��I?

learning_rate_1�%�8j�%AI       6%�	�*hH���A�*;


total_loss�w@

error_R�E?

learning_rate_1�%�8Ƃ�I       6%�	�uhH���A�*;


total_losst��@

error_R��J?

learning_rate_1�%�8I�lhI       6%�	�hH���A�*;


total_loss�=�@

error_Rl�M?

learning_rate_1�%�8��}I       6%�	�"iH���A�*;


total_loss̆A

error_Ri?

learning_rate_1�%�8�Fz�I       6%�	�jiH���A�*;


total_loss%L�@

error_R��E?

learning_rate_1�%�8f�\�I       6%�	��iH���A�*;


total_loss�9�@

error_R�_U?

learning_rate_1�%�8��I       6%�	��iH���A�*;


total_lossd��@

error_R�K?

learning_rate_1�%�8��i>I       6%�	)8jH���A�*;


total_loss�¥@

error_R-�T?

learning_rate_1�%�8E&=I       6%�	l~jH���A�*;


total_loss�A

error_R@�Q?

learning_rate_1�%�8���I       6%�	w�jH���A�*;


total_loss��@

error_R��A?

learning_rate_1�%�8xJ�I       6%�	kH���A�*;


total_loss��@

error_R��>?

learning_rate_1�%�8��I       6%�	"LkH���A�*;


total_lossD��@

error_R\Tb?

learning_rate_1�%�8e�>YI       6%�	@�kH���A�*;


total_loss.R�@

error_R@�G?

learning_rate_1�%�8'9�I       6%�	V�kH���A�*;


total_lossihA

error_RAR?

learning_rate_1�%�8,�RI       6%�	�lH���A�*;


total_loss��@

error_RN?

learning_rate_1�%�8��y
I       6%�	�ZlH���A�*;


total_loss�|�@

error_R��P?

learning_rate_1�%�8�;�I       6%�	�lH���A�*;


total_loss_�@

error_RiW?

learning_rate_1�%�8�z�MI       6%�	�lH���A�*;


total_loss�5�@

error_R3nS?

learning_rate_1�%�8�?lI       6%�	v1mH���A�*;


total_loss=�@

error_RѽB?

learning_rate_1�%�8����I       6%�	�mH���A�*;


total_lossOC�@

error_R�Z?

learning_rate_1�%�8Y�I       6%�	��mH���A�*;


total_loss�^�@

error_R�I?

learning_rate_1�%�8w[ޑI       6%�	�nH���A�*;


total_loss�)�@

error_R,yM?

learning_rate_1�%�8GG�I       6%�	 anH���A�*;


total_loss�&]@

error_R�[I?

learning_rate_1�%�8����I       6%�	�nH���A�*;


total_loss�l�@

error_R�A?

learning_rate_1�%�8z�I       6%�	q�nH���A�*;


total_lossvO�@

error_R��??

learning_rate_1�%�8S��I       6%�	�3oH���A�*;


total_loss�֞@

error_R�]?

learning_rate_1�%�8r���I       6%�	p~oH���A�*;


total_loss�d�@

error_R�;?

learning_rate_1�%�8?�I       6%�	o�oH���A�*;


total_loss$A

error_RV/a?

learning_rate_1�%�8 ��wI       6%�	Z pH���A�*;


total_loss|@�@

error_R�bW?

learning_rate_1�%�8�.��I       6%�	HDpH���A�*;


total_lossӧA

error_R��K?

learning_rate_1�%�8����I       6%�	�pH���A�*;


total_loss��@

error_Rf�N?

learning_rate_1�%�8�� �I       6%�	-�pH���A�*;


total_loss�SA

error_R��I?

learning_rate_1�%�8q��+I       6%�	qH���A�*;


total_loss5�@

error_R��A?

learning_rate_1�%�8Vf�I       6%�	�NqH���A�*;


total_loss���@

error_R��R?

learning_rate_1�%�8���I       6%�	�qH���A�*;


total_loss��@

error_R-�b?

learning_rate_1�%�8�O^�I       6%�	�qH���A�*;


total_loss!��@

error_R�A\?

learning_rate_1�%�8P\�%I       6%�	ZrH���A�*;


total_loss{9�@

error_R�P=?

learning_rate_1�%�8��[I       6%�	�\rH���A�*;


total_loss�:�@

error_RV?

learning_rate_1�%�8Nz�I       6%�	-�rH���A�*;


total_loss��@

error_R;�,?

learning_rate_1�%�8�д�I       6%�	��rH���A�*;


total_lossu	A

error_R7�F?

learning_rate_1�%�8����I       6%�	�+sH���A�*;


total_lossF�@

error_R`\I?

learning_rate_1�%�8���_I       6%�	nvsH���A�*;


total_lossj��@

error_R��=?

learning_rate_1�%�8O4UuI       6%�	\�sH���A�*;


total_loss�@

error_R�oI?

learning_rate_1�%�8����I       6%�	;�sH���A�*;


total_lossv��@

error_R��S?

learning_rate_1�%�8ǁ�I       6%�	�CtH���A�*;


total_lossC`�@

error_RsK?

learning_rate_1�%�8�<X}I       6%�	|�tH���A�*;


total_loss4�@

error_R��Q?

learning_rate_1�%�8�[m�I       6%�	��tH���A�*;


total_loss6gA

error_R��D?

learning_rate_1�%�8erztI       6%�	7>uH���A�*;


total_lossl�A

error_RaH?

learning_rate_1�%�8�__�I       6%�	�uH���A�*;


total_lossEH�@

error_R!3W?

learning_rate_1�%�8�߉�I       6%�	)�uH���A�*;


total_loss��@

error_R��G?

learning_rate_1�%�8G�[�I       6%�	�vH���A�*;


total_loss�́@

error_R:XG?

learning_rate_1�%�8"NUI       6%�	l_vH���A�*;


total_loss%�PA

error_R�Y?

learning_rate_1�%�8�#�I       6%�	6�vH���A�*;


total_loss��@

error_R�;T?

learning_rate_1�%�8V�+I       6%�	��vH���A�*;


total_loss� �@

error_R�(R?

learning_rate_1�%�8���AI       6%�	.wH���A�*;


total_lossl��@

error_R�"@?

learning_rate_1�%�8S�v�I       6%�	�qwH���A�*;


total_loss���@

error_REoQ?

learning_rate_1�%�8�K$�I       6%�	��wH���A�*;


total_loss�|�@

error_R�O?

learning_rate_1�%�8w�^�I       6%�	/xH���A�*;


total_loss�҉@

error_Rq�D?

learning_rate_1�%�8zӓI       6%�	[MxH���A�*;


total_loss��@

error_R�3\?

learning_rate_1�%�8��W�I       6%�	��xH���A�*;


total_loss(R�@

error_R�6?

learning_rate_1�%�8���I       6%�	��xH���A�*;


total_loss[h�@

error_R�h?

learning_rate_1�%�8i"��I       6%�	]IyH���A�*;


total_loss8�@

error_R��M?

learning_rate_1�%�8<��I       6%�	��yH���A�*;


total_loss�A

error_R��H?

learning_rate_1�%�8,h�,I       6%�	��yH���A�*;


total_loss,f�@

error_R_�Y?

learning_rate_1�%�8fK�3I       6%�	�!zH���A�*;


total_loss�ռ@

error_R��>?

learning_rate_1�%�8���3I       6%�	�hzH���A�*;


total_loss��@

error_R�9P?

learning_rate_1�%�80o6�I       6%�	@�zH���A�*;


total_loss���@

error_R�CN?

learning_rate_1�%�8�mI       6%�	�zH���A�*;


total_lossoF�@

error_R�@?

learning_rate_1�%�8�a:I       6%�	j;{H���A�*;


total_loss3r�@

error_RX?

learning_rate_1�%�8j��I       6%�	L�{H���A�*;


total_loss��@

error_R�E?

learning_rate_1�%�8M´jI       6%�	�{H���A�*;


total_loss���@

error_R$�B?

learning_rate_1�%�8-'��I       6%�	c|H���A�*;


total_lossT�A

error_R��K?

learning_rate_1�%�8����I       6%�	�U|H���A�*;


total_loss��@

error_R�oR?

learning_rate_1�%�8��;�I       6%�	��|H���A�*;


total_loss _}@

error_R܍C?

learning_rate_1�%�8PǗ�I       6%�	��|H���A�*;


total_loss
��@

error_R�CK?

learning_rate_1�%�8-(��I       6%�	'}H���A�*;


total_loss���@

error_RmbT?

learning_rate_1�%�8��c�I       6%�	�m}H���A�*;


total_loss�&�@

error_R&�J?

learning_rate_1�%�8}��I       6%�	�}H���A�*;


total_loss�� A

error_R߰O?

learning_rate_1�%�8�B�I       6%�	o~H���A�*;


total_loss�W�@

error_Rj�^?

learning_rate_1�%�8�#gI       6%�	K~H���A�*;


total_loss�,A

error_Rr�J?

learning_rate_1�%�8��{I       6%�	+�~H���A�*;


total_loss�$�@

error_R��N?

learning_rate_1�%�8�)�mI       6%�	P�~H���A�*;


total_loss̜�@

error_R,�P?

learning_rate_1�%�8[��I       6%�	�H���A�*;


total_lossؔ�@

error_R�I?

learning_rate_1�%�8�I       6%�	�`H���A�*;


total_loss���@

error_R{%Q?

learning_rate_1�%�8��	�I       6%�	��H���A�*;


total_loss��A

error_R�PP?

learning_rate_1�%�8��CnI       6%�	#�H���A�*;


total_lossa�@

error_R/�F?

learning_rate_1�%�84��zI       6%�	�1�H���A�*;


total_loss��8A

error_R['N?

learning_rate_1�%�8�걓I       6%�	u�H���A�*;


total_loss��@

error_R#�9?

learning_rate_1�%�8{<�_I       6%�	���H���A�*;


total_lossA�A

error_R%�P?

learning_rate_1�%�8�{cI       6%�	���H���A�*;


total_loss#}�@

error_R�[J?

learning_rate_1�%�8T���I       6%�	�>�H���A�*;


total_loss[{�@

error_R۽U?

learning_rate_1�%�8R��;I       6%�	��H���A�*;


total_lossh�@

error_R��C?

learning_rate_1�%�8�u�^I       6%�	�H���A�*;


total_loss_��@

error_R��L?

learning_rate_1�%�8�ьI       6%�	R�H���A�*;


total_lossa�@

error_R)�F?

learning_rate_1�%�8���vI       6%�	J�H���A�*;


total_loss���@

error_R�:?

learning_rate_1�%�8���I       6%�	���H���A�*;


total_loss]��@

error_R�ie?

learning_rate_1�%�8Z1�I       6%�	k҂H���A�*;


total_lossӽ�@

error_R��M?

learning_rate_1�%�8D�z�I       6%�	�H���A�*;


total_lossۧ@

error_R3BG?

learning_rate_1�%�8�$�gI       6%�	#Z�H���A�*;


total_loss���@

error_Rf�F?

learning_rate_1�%�85`�I       6%�	���H���A�*;


total_loss 0�@

error_R�H?

learning_rate_1�%�8^E�I       6%�	x߃H���A�*;


total_loss���@

error_R&�L?

learning_rate_1�%�8�O�GI       6%�	;&�H���A�*;


total_loss���@

error_R�M?

learning_rate_1�%�85gd�I       6%�	�j�H���A�*;


total_lossmF�@

error_R�;O?

learning_rate_1�%�8����I       6%�	��H���A�*;


total_loss�2�@

error_R�Q?

learning_rate_1�%�8�%�I       6%�	5�H���A�*;


total_loss��@

error_R��R?

learning_rate_1�%�8u	�I       6%�	�4�H���A�*;


total_loss�/�@

error_R� K?

learning_rate_1�%�8m+[I       6%�	�x�H���A�*;


total_loss;��@

error_R�Fh?

learning_rate_1�%�8���CI       6%�	û�H���A�*;


total_lossŏ�@

error_R1�:?

learning_rate_1�%�8��L�I       6%�	���H���A�*;


total_loss���@

error_R�U?

learning_rate_1�%�8���I       6%�	�C�H���A�*;


total_loss�0�@

error_R-R?

learning_rate_1�%�8cH��I       6%�	ⅆH���A�*;


total_loss�ƺ@

error_R�{T?

learning_rate_1�%�8��)CI       6%�	�͆H���A�*;


total_loss&]�@

error_R��H?

learning_rate_1�%�8|���I       6%�	N�H���A�*;


total_loss�NA

error_Ra.R?

learning_rate_1�%�8o�.I       6%�	�a�H���A�*;


total_loss��@

error_RhN?

learning_rate_1�%�8�C� I       6%�	ݩ�H���A�*;


total_loss�?�@

error_RA�V?

learning_rate_1�%�8^Ǡ]I       6%�	&�H���A�*;


total_loss�.�@

error_R�9O?

learning_rate_1�%�8�QI       6%�	:�H���A�*;


total_loss�`�@

error_RV�N?

learning_rate_1�%�8د�I       6%�	k��H���A�*;


total_loss���@

error_R�M?

learning_rate_1�%�8/���I       6%�	c��H���A�*;


total_loss���@

error_R�u@?

learning_rate_1�%�8���I       6%�	C�H���A�*;


total_loss	��@

error_R�F[?

learning_rate_1�%�8���I       6%�	��H���A�*;


total_loss�?	A

error_Ri!E?

learning_rate_1�%�8���I       6%�	_ЉH���A�*;


total_loss+�@

error_R�X?

learning_rate_1�%�8�,�I       6%�	 �H���A�*;


total_loss%��@

error_R)7@?

learning_rate_1�%�8��DI       6%�	kb�H���A�*;


total_loss�d�@

error_RTnP?

learning_rate_1�%�8P��AI       6%�	ɰ�H���A�*;


total_loss��@

error_Rs\@?

learning_rate_1�%�8��I       6%�	,��H���A�*;


total_loss���@

error_R�P?

learning_rate_1�%�8����I       6%�	�E�H���A�*;


total_loss���@

error_R��H?

learning_rate_1�%�8�r�I       6%�	I��H���A�*;


total_loss���@

error_R�3?

learning_rate_1�%�8L�+I       6%�	�̋H���A�*;


total_lossH��@

error_Rh�X?

learning_rate_1�%�8��I       6%�	��H���A�*;


total_lossI!�@

error_R�U?

learning_rate_1�%�8	ȹ"I       6%�	kU�H���A�*;


total_lossV��@

error_RE65?

learning_rate_1�%�8(r4�I       6%�	:��H���A�*;


total_losslا@

error_R=�I?

learning_rate_1�%�8�"gI       6%�	��H���A�*;


total_loss�޻@

error_R��[?

learning_rate_1�%�8���sI       6%�	;'�H���A�*;


total_lossTh�@

error_R_�c?

learning_rate_1�%�8�"ǉI       6%�	�k�H���A�*;


total_loss���@

error_R�V>?

learning_rate_1�%�8�$@�I       6%�	洍H���A�*;


total_loss�O�@

error_R��B?

learning_rate_1�%�8Ǹ��I       6%�	���H���A�*;


total_loss$��@

error_Rܻl?

learning_rate_1�%�8z��I       6%�	fK�H���A�*;


total_loss�˴@

error_R�F?

learning_rate_1�%�8��I       6%�	딎H���A�*;


total_lossv�@

error_R�H?

learning_rate_1�%�8��(I       6%�	aڎH���A�*;


total_loss1��@

error_R�4W?

learning_rate_1�%�8ϒ�I       6%�	|�H���A�*;


total_lossC�@

error_R�pI?

learning_rate_1�%�878I       6%�	ya�H���A�*;


total_loss�#�@

error_R�"[?

learning_rate_1�%�8b�0�I       6%�	累H���A�*;


total_loss��b@

error_R�g6?

learning_rate_1�%�8K��|I       6%�	��H���A�*;


total_loss�?�@

error_Rl�`?

learning_rate_1�%�8
œvI       6%�	�+�H���A�*;


total_losszA

error_R��@?

learning_rate_1�%�84E�I       6%�	Cn�H���A�*;


total_loss$0A

error_RH�H?

learning_rate_1�%�8K1�I       6%�	p��H���A�*;


total_loss_�@

error_R.�K?

learning_rate_1�%�8�I       6%�	��H���A�*;


total_loss�wy@

error_R�v8?

learning_rate_1�%�83ZI       6%�	�>�H���A�*;


total_loss�eP@

error_RM&8?

learning_rate_1�%�8�z
I       6%�	R��H���A�*;


total_lossj%�@

error_Rs�D?

learning_rate_1�%�8���I       6%�	�ƑH���A�*;


total_loss��@

error_R�C?

learning_rate_1�%�8au�:I       6%�	m�H���A�*;


total_loss2q�@

error_R-P?

learning_rate_1�%�8~��I       6%�	wW�H���A�*;


total_loss��@

error_R��F?

learning_rate_1�%�8
!bsI       6%�	0��H���A�*;


total_lossE��@

error_R�__?

learning_rate_1�%�8���pI       6%�	���H���A�*;


total_loss=�@

error_R�wM?

learning_rate_1�%�8�˾aI       6%�	�!�H���A�*;


total_loss,�A

error_R%2T?

learning_rate_1�%�8!���I       6%�	g�H���A�*;


total_loss7�@

error_R��T?

learning_rate_1�%�89N̿I       6%�	���H���A�*;


total_loss/|�@

error_RJ�^?

learning_rate_1�%�8%��I       6%�	��H���A�*;


total_losse�@

error_R}�V?

learning_rate_1�%�8�'c�I       6%�	f8�H���A�*;


total_loss�A

error_RvdX?

learning_rate_1�%�8Z��(I       6%�	���H���A�*;


total_lossQc�@

error_R��R?

learning_rate_1�%�8(�z�I       6%�	�ٔH���A�*;


total_loss���@

error_RO?

learning_rate_1�%�8զ�xI       6%�	��H���A�*;


total_loss�CA

error_R6rD?

learning_rate_1�%�8G@��I       6%�	I��H���A�*;


total_loss��8A

error_R��N?

learning_rate_1�%�8
�BI       6%�	?ՕH���A�*;


total_lossf�@

error_R�@?

learning_rate_1�%�8�,~�I       6%�	�#�H���A�*;


total_lossra�@

error_RD�J?

learning_rate_1�%�8A.;I       6%�	Im�H���A�*;


total_loss��@

error_R�q;?

learning_rate_1�%�8�Y`=I       6%�	R��H���A�*;


total_loss��@

error_Rv�_?

learning_rate_1�%�8�� �I       6%�	 ��H���A�*;


total_loss�N�@

error_R��U?

learning_rate_1�%�8o��vI       6%�	�8�H���A�*;


total_loss.��@

error_R��d?

learning_rate_1�%�8#��I       6%�	�|�H���A�*;


total_loss�1�@

error_R��f?

learning_rate_1�%�8����I       6%�	�×H���A�*;


total_lossC�@

error_Rn�L?

learning_rate_1�%�8pa_I       6%�	��H���A�*;


total_lossiH�@

error_RC�:?

learning_rate_1�%�8���I       6%�	!O�H���A�*;


total_loss���@

error_R�S<?

learning_rate_1�%�8sI��I       6%�	���H���A�*;


total_loss�@

error_R�X?

learning_rate_1�%�8����I       6%�	���H���A�*;


total_loss��@

error_RT�M?

learning_rate_1�%�8�d66I       6%�	D�H���A�*;


total_loss��@

error_RO�F?

learning_rate_1�%�86=,%I       6%�	���H���A�*;


total_lossm��@

error_R��N?

learning_rate_1�%�8VuI       6%�	0ҙH���A�*;


total_loss��@

error_R�CL?

learning_rate_1�%�8��MI       6%�	W�H���A�*;


total_loss��@

error_R�T?

learning_rate_1�%�8U�$�I       6%�	}c�H���A�*;


total_lossw��@

error_R��S?

learning_rate_1�%�8�s�FI       6%�	٨�H���A�*;


total_loss�$�@

error_R�x_?

learning_rate_1�%�8��j�I       6%�	B�H���A�*;


total_loss£@

error_Rz`?

learning_rate_1�%�8ƉUI       6%�	_;�H���A�*;


total_loss�"�@

error_R�8Q?

learning_rate_1�%�8Kr��I       6%�	��H���A�*;


total_loss�!�@

error_RcW?

learning_rate_1�%�8���pI       6%�	�ƛH���A�*;


total_loss�G�@

error_R��W?

learning_rate_1�%�8�e�@I       6%�	��H���A�*;


total_loss��@

error_R	;?

learning_rate_1�%�8(PvI       6%�	�X�H���A�*;


total_loss̚�@

error_R�UJ?

learning_rate_1�%�80~��I       6%�	S��H���A�*;


total_loss=+&A

error_R��_?

learning_rate_1�%�8��p�I       6%�	��H���A�*;


total_loss�W�@

error_R%�V?

learning_rate_1�%�8�B�I       6%�	62�H���A�*;


total_loss&q@

error_RW|W?

learning_rate_1�%�8?(.�I       6%�	(y�H���A�*;


total_loss(� A

error_RܖS?

learning_rate_1�%�8>*�I       6%�	B��H���A�*;


total_lossI�@

error_R!YO?

learning_rate_1�%�83���I       6%�	c�H���A�*;


total_loss�B�@

error_R�e?

learning_rate_1�%�8<Q �I       6%�	�N�H���A�*;


total_loss���@

error_R��=?

learning_rate_1�%�8y��I       6%�	I��H���A�*;


total_loss���@

error_R��H?

learning_rate_1�%�8p�-
I       6%�	K��H���A�*;


total_loss�e�@

error_R��>?

learning_rate_1�%�8���I       6%�	�)�H���A�*;


total_loss�i�@

error_R��[?

learning_rate_1�%�8���sI       6%�	�s�H���A�*;


total_lossDb�@

error_RΕ\?

learning_rate_1�%�8���I       6%�	��H���A�*;


total_loss#�@

error_R�eC?

learning_rate_1�%�8� I       6%�	���H���A�*;


total_loss�r�@

error_Rd2P?

learning_rate_1�%�8���I       6%�	F�H���A�*;


total_loss@c�@

error_R�.I?

learning_rate_1�%�8¤5I       6%�	Ƈ�H���A�*;


total_loss 
A

error_RțD?

learning_rate_1�%�8���qI       6%�	ˠH���A�*;


total_lossC�@

error_R�X^?

learning_rate_1�%�8O���I       6%�	c�H���A�*;


total_loss���@

error_R��L?

learning_rate_1�%�8�@GI       6%�	�S�H���A�*;


total_loss�m�@

error_Rl?U?

learning_rate_1�%�8�@ �I       6%�	N��H���A�*;


total_loss��@

error_R16\?

learning_rate_1�%�8�ؠI       6%�	�ޡH���A�*;


total_lossZ��@

error_R�eU?

learning_rate_1�%�8�KI       6%�	})�H���A�*;


total_loss�ϝ@

error_R.P?

learning_rate_1�%�8Յ%�I       6%�	�o�H���A�*;


total_loss	�@

error_R��G?

learning_rate_1�%�8D<�I       6%�	1��H���A�*;


total_loss�ڃ@

error_R��??

learning_rate_1�%�8��ЌI       6%�	���H���A�*;


total_loss��@

error_R[DQ?

learning_rate_1�%�88B��I       6%�	�E�H���A�*;


total_loss���@

error_R��V?

learning_rate_1�%�8���I       6%�	���H���A�*;


total_loss��@

error_R��O?

learning_rate_1�%�8O��4I       6%�	iԣH���A�*;


total_loss��@

error_RF?

learning_rate_1�%�8toq�I       6%�	��H���A�*;


total_loss���@

error_Rf5?

learning_rate_1�%�8]�`I       6%�	}d�H���A�*;


total_lossC��@

error_RJMP?

learning_rate_1�%�89�{XI       6%�	���H���A�*;


total_loss�o�@

error_RCV\?

learning_rate_1�%�8��~FI       6%�	��H���A�*;


total_loss:#�@

error_RTeC?

learning_rate_1�%�8��*HI       6%�	X:�H���A�*;


total_loss�]�@

error_R_�N?

learning_rate_1�%�8o��qI       6%�	�}�H���A�*;


total_loss?�@

error_R�<\?

learning_rate_1�%�8�D�I       6%�	�ǥH���A�*;


total_loss���@

error_R7o2?

learning_rate_1�%�8��$eI       6%�	��H���A�*;


total_loss�L�@

error_R��M?

learning_rate_1�%�8�f�I       6%�	�U�H���A�*;


total_loss�#�@

error_R�K?

learning_rate_1�%�8�5�|I       6%�	ś�H���A�*;


total_loss-A

error_RB?

learning_rate_1�%�8{8I       6%�	gݦH���A�*;


total_lossO�@

error_R6,F?

learning_rate_1�%�8��q�I       6%�	� �H���A�*;


total_loss�߲@

error_R�<A?

learning_rate_1�%�8�Q�.I       6%�	�d�H���A�*;


total_loss�X�@

error_RɱK?

learning_rate_1�%�8�n�I       6%�	W��H���A�*;


total_loss��
A

error_RsP?

learning_rate_1�%�8��,6I       6%�	&�H���A�*;


total_lossh��@

error_R��Y?

learning_rate_1�%�8T�ִI       6%�	w7�H���A�*;


total_loss���@

error_RgU?

learning_rate_1�%�8N�jI       6%�	���H���A�*;


total_loss��A

error_R߰c?

learning_rate_1�%�8���AI       6%�	ZߨH���A�*;


total_lossl�q@

error_R��Q?

learning_rate_1�%�8��(I       6%�	�%�H���A�*;


total_loss&[�@

error_R��I?

learning_rate_1�%�8��O�I       6%�	�h�H���A�*;


total_loss@�A

error_R�K?

learning_rate_1�%�8�J�9I       6%�	A��H���A�*;


total_lossa�@

error_R|�U?

learning_rate_1�%�8�r�?I       6%�	x�H���A�*;


total_loss��@

error_R�.c?

learning_rate_1�%�8�I�I       6%�	I9�H���A�*;


total_loss]�@

error_RD�J?

learning_rate_1�%�8��/�I       6%�	A|�H���A�*;


total_lossn�@

error_R��I?

learning_rate_1�%�8��üI       6%�	���H���A�*;


total_loss�@

error_R�L?

learning_rate_1�%�8�~0#I       6%�	Z��H���A�*;


total_loss��<@

error_R<lU?

learning_rate_1�%�8w���I       6%�	�B�H���A�*;


total_loss���@

error_R�W?

learning_rate_1�%�8�@ibI       6%�	���H���A�*;


total_lossD��@

error_R��B?

learning_rate_1�%�8�_�
I       6%�	�̫H���A�*;


total_lossG�@

error_R�G?

learning_rate_1�%�8���uI       6%�	��H���A�*;


total_loss�,�@

error_R1�F?

learning_rate_1�%�8Mw��I       6%�	D[�H���A�*;


total_loss���@

error_R��Y?

learning_rate_1�%�8��L#I       6%�	��H���A�*;


total_loss�LA

error_R%�B?

learning_rate_1�%�8�iEI       6%�	�H���A�*;


total_loss�^�@

error_R$�G?

learning_rate_1�%�8��hI       6%�	(�H���A�*;


total_lossA�@

error_R��Q?

learning_rate_1�%�8���I       6%�	;q�H���A�*;


total_loss�{�@

error_R�[?

learning_rate_1�%�8]�I       6%�	���H���A�*;


total_lossϗ{@

error_Re�P?

learning_rate_1�%�85t��I       6%�	���H���A�*;


total_loss!�@

error_R��c?

learning_rate_1�%�8fS[TI       6%�	"A�H���A�*;


total_lossnz�@

error_R+O?

learning_rate_1�%�8l)��I       6%�	ԅ�H���A�*;


total_loss:QA

error_R)+a?

learning_rate_1�%�8��jI       6%�	�ȮH���A�*;


total_lossi�@

error_Rf�K?

learning_rate_1�%�8���?I       6%�	h�H���A�*;


total_loss���@

error_R��H?

learning_rate_1�%�8!�uI       6%�	�P�H���A�*;


total_loss�۶@

error_R�dN?

learning_rate_1�%�8�i�[I       6%�	]��H���A�*;


total_loss2�@

error_R�J?

learning_rate_1�%�8����I       6%�	�ԯH���A�*;


total_loss�xA

error_R�T^?

learning_rate_1�%�8^��I       6%�	�H���A�*;


total_loss�kA

error_R_7S?

learning_rate_1�%�8��w�I       6%�	�[�H���A�*;


total_loss��@

error_REh??

learning_rate_1�%�8Iy1I       6%�	읰H���A�*;


total_loss/	�@

error_R�8?

learning_rate_1�%�8��T0I       6%�	+�H���A�*;


total_loss8�@

error_R��G?

learning_rate_1�%�8�ޘHI       6%�	�$�H���A�*;


total_loss�4�@

error_R��D?

learning_rate_1�%�8�bI       6%�	h�H���A�*;


total_loss`��@

error_RS�U?

learning_rate_1�%�8���I       6%�	r��H���A�*;


total_lossH4�@

error_R3�c?

learning_rate_1�%�8���I       6%�	B�H���A�*;


total_lossԯ�@

error_R��M?

learning_rate_1�%�8b�I       6%�	�:�H���A�*;


total_loss�H�@

error_Rs�K?

learning_rate_1�%�8��9I       6%�	}��H���A�*;


total_loss��A

error_Ra�^?

learning_rate_1�%�8�N��I       6%�	0̲H���A�*;


total_loss+�@

error_R�E>?

learning_rate_1�%�8�b�I       6%�	��H���A�*;


total_loss�ޟ@

error_R��T?

learning_rate_1�%�8t�"�I       6%�	NU�H���A�*;


total_loss��@

error_RHN?

learning_rate_1�%�8p��I       6%�	���H���A�*;


total_loss�ѭ@

error_RpX?

learning_rate_1�%�8=Z;YI       6%�	�H���A�*;


total_loss�pA

error_R�+K?

learning_rate_1�%�8�h#I       6%�	�4�H���A�*;


total_loss`�l@

error_RR�R?

learning_rate_1�%�8t���I       6%�	o��H���A�*;


total_loss�O A

error_R�i?

learning_rate_1�%�8��n6I       6%�	�ݴH���A�*;


total_loss���@

error_R��O?

learning_rate_1�%�8��|2I       6%�	\%�H���A�*;


total_loss׊�@

error_R��B?

learning_rate_1�%�8mc��I       6%�	�l�H���A�*;


total_loss��@

error_Ri�Z?

learning_rate_1�%�8`HbI       6%�	���H���A�*;


total_loss��@

error_RMU?

learning_rate_1�%�8q�YI       6%�	���H���A�*;


total_lossJ�@

error_R�S?

learning_rate_1�%�8�ofI       6%�	�C�H���A�*;


total_loss��@

error_RoC?

learning_rate_1�%�8��<\I       6%�	���H���A�*;


total_loss$��@

error_R8�a?

learning_rate_1�%�8E癬I       6%�	ڶH���A�*;


total_lossx�@

error_RC)N?

learning_rate_1�%�8�[VI       6%�	Q$�H���A�*;


total_loss�kA

error_R@ W?

learning_rate_1�%�8��+�I       6%�	�h�H���A�*;


total_loss���@

error_RZ]R?

learning_rate_1�%�8�y�I       6%�	4��H���A�*;


total_lossl2�@

error_R�1?

learning_rate_1�%�8<8�I       6%�	���H���A�*;


total_lossv��@

error_R1E^?

learning_rate_1�%�8/�8I       6%�	I�H���A�*;


total_loss�d�@

error_R�}_?

learning_rate_1�%�8���I       6%�	%��H���A�*;


total_loss���@

error_RT�M?

learning_rate_1�%�8d� �I       6%�	k�H���A�*;


total_loss���@

error_R�<?

learning_rate_1�%�8��0I       6%�	�7�H���A�*;


total_loss�&�@

error_R��O?

learning_rate_1�%�8�� I       6%�	Q~�H���A�*;


total_loss%��@

error_R�X?

learning_rate_1�%�8�).I       6%�	�ĹH���A�*;


total_loss�b�@

error_R��U?

learning_rate_1�%�8w+I       6%�	��H���A�*;


total_lossXr�@

error_Ra�H?

learning_rate_1�%�8֏��I       6%�	L�H���A�*;


total_loss���@

error_R>?

learning_rate_1�%�8�4OuI       6%�	�H���A�*;


total_loss�7�@

error_R!Z?

learning_rate_1�%�8�5�I       6%�	�ܺH���A�*;


total_loss�V�@

error_R&K?

learning_rate_1�%�8��!I       6%�	%#�H���A�*;


total_loss)G�@

error_R�k?

learning_rate_1�%�8N7�I       6%�	xk�H���A�*;


total_loss��@

error_R��B?

learning_rate_1�%�8��<I       6%�	���H���A�*;


total_lossj�@

error_R��W?

learning_rate_1�%�8d�3�I       6%�	o�H���A�*;


total_loss�@

error_R��@?

learning_rate_1�%�8!�/ I       6%�	qS�H���A�*;


total_loss��@

error_RaB?

learning_rate_1�%�8n%�I       6%�	���H���A�*;


total_loss��@

error_R��d?

learning_rate_1�%�8��YI       6%�	o�H���A�*;


total_lossq��@

error_R��T?

learning_rate_1�%�8K1W�I       6%�	�+�H���A�*;


total_loss�A

error_RQ|Q?

learning_rate_1�%�8�W�I       6%�	�n�H���A�*;


total_loss�@

error_R��M?

learning_rate_1�%�8��H�I       6%�	�H���A�*;


total_loss��@

error_R�cL?

learning_rate_1�%�8_|
BI       6%�	D��H���A�*;


total_loss���@

error_Ro�E?

learning_rate_1�%�8+&�TI       6%�	�D�H���A�*;


total_lossOm�@

error_R�%L?

learning_rate_1�%�8v��I       6%�	���H���A�*;


total_lossiΛ@

error_RHc?

learning_rate_1�%�8b�I       6%�	2ϾH���A�*;


total_loss�Q�@

error_R�[?

learning_rate_1�%�8g6I       6%�	��H���A�*;


total_loss	��@

error_R��N?

learning_rate_1�%�8<Y�I       6%�	tW�H���A�*;


total_lossβ�@

error_R�*O?

learning_rate_1�%�8�C!@I       6%�	&��H���A�*;


total_lossn�@

error_R�0E?

learning_rate_1�%�8(	I       6%�	(�H���A�*;


total_lossx�	A

error_R��U?

learning_rate_1�%�8X�BI       6%�	7�H���A�*;


total_lossε�@

error_R�[?

learning_rate_1�%�8��k�I       6%�	�H���A�*;


total_lossJ�@

error_RH7*?

learning_rate_1�%�8 '\�I       6%�	��H���A�*;


total_loss�G�@

error_R��T?

learning_rate_1�%�8
7�aI       6%�	��H���A�*;


total_loss���@

error_R!�]?

learning_rate_1�%�8�o�I       6%�	#W�H���A�*;


total_loss|�A

error_R��D?

learning_rate_1�%�8�OqI       6%�	���H���A�*;


total_loss��@

error_R�dS?

learning_rate_1�%�8�%M�I       6%�	L��H���A�*;


total_loss���@

error_RHB?

learning_rate_1�%�8Vk�I       6%�	�2�H���A�*;


total_loss_�@

error_RVX?

learning_rate_1�%�8�s��I       6%�	?w�H���A�*;


total_lossZ��@

error_R�S?

learning_rate_1�%�8x��!I       6%�	c��H���A�*;


total_lossi.�@

error_R�[?

learning_rate_1�%�8���kI       6%�	��H���A�*;


total_lossc��@

error_Ri�E?

learning_rate_1�%�8�96I       6%�	SO�H���A�*;


total_loss��@

error_R@vQ?

learning_rate_1�%�8��(�I       6%�	ɓ�H���A�*;


total_loss,��@

error_Ra�H?

learning_rate_1�%�8y���I       6%�	���H���A�*;


total_lossȳ@

error_RJOR?

learning_rate_1�%�8���I       6%�	"�H���A�*;


total_loss*v�@

error_RT�G?

learning_rate_1�%�8�uKI       6%�	Ch�H���A�*;


total_loss*��@

error_R@	L?

learning_rate_1�%�8�,�I       6%�	ի�H���A�*;


total_losst�@

error_RĹM?

learning_rate_1�%�8��5=I       6%�	���H���A�*;


total_loss�A�@

error_RJ�[?

learning_rate_1�%�8�R�<I       6%�	�7�H���A�*;


total_loss-��@

error_RS^?

learning_rate_1�%�8m}�I       6%�	4��H���A�*;


total_loss.�@

error_R�T?

learning_rate_1�%�8���I       6%�	v��H���A�*;


total_lossc\�@

error_R�`?

learning_rate_1�%�8Ȫ�GI       6%�	��H���A�*;


total_loss=��@

error_Rj�T?

learning_rate_1�%�8aF[�I       6%�	�X�H���A�*;


total_loss�u�@

error_R15N?

learning_rate_1�%�85�kkI       6%�	X��H���A�*;


total_loss�A

error_R!:M?

learning_rate_1�%�84��I       6%�	���H���A�*;


total_loss\Kk@

error_ReuO?

learning_rate_1�%�8��V�I       6%�	o*�H���A�*;


total_loss���@

error_R�PD?

learning_rate_1�%�8Ϋn}I       6%�	*n�H���A�*;


total_loss�\�@

error_R_(B?

learning_rate_1�%�8퀋�I       6%�	��H���A�*;


total_loss�@

error_RCP?

learning_rate_1�%�8��@�I       6%�	���H���A�*;


total_loss�W@

error_R?#C?

learning_rate_1�%�8z��4I       6%�	\?�H���A�*;


total_loss`�@

error_R��Z?

learning_rate_1�%�8�,%I       6%�	K��H���A�*;


total_loss8��@

error_R6�P?

learning_rate_1�%�87٩I       6%�	B��H���A�*;


total_loss���@

error_R��G?

learning_rate_1�%�8�Z&�I       6%�	�B�H���A�*;


total_loss �A

error_RZRH?

learning_rate_1�%�87R�I       6%�	�H���A�*;


total_loss��@

error_R;gc?

learning_rate_1�%�8�FII       6%�	��H���A�*;


total_loss�<A

error_R�OM?

learning_rate_1�%�8\���I       6%�	>�H���A�*;


total_lossl0�@

error_R�#X?

learning_rate_1�%�8�II       6%�	MX�H���A�*;


total_loss�f�@

error_Rl{X?

learning_rate_1�%�8�|dI       6%�	���H���A�*;


total_loss�:�@

error_R4�E?

learning_rate_1�%�8$dI       6%�	���H���A�*;


total_loss�?�@

error_R�kP?

learning_rate_1�%�8�I       6%�	X+�H���A�*;


total_lossao�@

error_R��G?

learning_rate_1�%�8(3�I       6%�	�r�H���A�*;


total_loss4��@

error_Rt�D?

learning_rate_1�%�8xe��I       6%�	ҵ�H���A�*;


total_loss�6�@

error_R �,?

learning_rate_1�%�8eV
6I       6%�	7��H���A�*;


total_loss���@

error_R �S?

learning_rate_1�%�8��PoI       6%�	�<�H���A�*;


total_loss�t�@

error_R�kM?

learning_rate_1�%�8�v�]I       6%�	��H���A�*;


total_loss��A

error_R�N?

learning_rate_1�%�8�q�'I       6%�	���H���A�*;


total_loss�F�@

error_R*�J?

learning_rate_1�%�8��Q�I       6%�	9�H���A�*;


total_loss���@

error_R�O?

learning_rate_1�%�8l��I       6%�	EU�H���A�*;


total_loss{Ϻ@

error_R&�P?

learning_rate_1�%�8����I       6%�	���H���A�*;


total_loss#�@

error_R!�`?

learning_rate_1�%�8Nr�$I       6%�	��H���A�*;


total_loss*7�@

error_R&gS?

learning_rate_1�%�8�I       6%�	H&�H���A�*;


total_loss
�@

error_R�J:?

learning_rate_1�%�8��I       6%�	�o�H���A�*;


total_lossƛ@

error_R$�J?

learning_rate_1�%�8z�~uI       6%�	X��H���A�*;


total_lossA

error_R��_?

learning_rate_1�%�8oh%I       6%�	u��H���A�*;


total_loss��@

error_R�I?

learning_rate_1�%�8p�iI       6%�	1A�H���A�*;


total_losshߐ@

error_R��C?

learning_rate_1�%�8�l�I       6%�	���H���A�*;


total_loss�{�@

error_R�PK?

learning_rate_1�%�80FA�I       6%�	3��H���A�*;


total_loss���@

error_R�V?

learning_rate_1�%�8;y�;I       6%�	�H���A�*;


total_loss�ȭ@

error_R�2L?

learning_rate_1�%�8����I       6%�	3c�H���A�*;


total_losse��@

error_RxBa?

learning_rate_1�%�8tR�I       6%�	��H���A�*;


total_loss�+�@

error_RL0D?

learning_rate_1�%�8f1�I       6%�	���H���A�*;


total_lossqR�@

error_Rf\?

learning_rate_1�%�8�L�+I       6%�	�7�H���A�*;


total_lossO޳@

error_R�-H?

learning_rate_1�%�8i�jI       6%�	�z�H���A�*;


total_loss�d@

error_RamB?

learning_rate_1�%�8�}C(I       6%�	ɿ�H���A�*;


total_loss��@

error_R�WA?

learning_rate_1�%�8��A�I       6%�	��H���A�*;


total_loss���@

error_R�&M?

learning_rate_1�%�8Wf��I       6%�	I�H���A�*;


total_loss�@

error_R�U?

learning_rate_1�%�8~1�I       6%�	Ս�H���A�*;


total_loss���@

error_Rz�A?

learning_rate_1�%�8�_�I       6%�	���H���A�*;


total_loss��@

error_R��S?

learning_rate_1�%�8��I       6%�	��H���A�*;


total_lossr��@

error_Rm�3?

learning_rate_1�%�8��6NI       6%�	�b�H���A�*;


total_loss�4�@

error_R�W?

learning_rate_1�%�8�o!�I       6%�	ȥ�H���A�*;


total_loss$�V@

error_R)�W?

learning_rate_1�%�804�I       6%�	I��H���A�*;


total_loss��A

error_R1�^?

learning_rate_1�%�8��&I       6%�	q.�H���A�*;


total_loss��@

error_R�vN?

learning_rate_1�%�8ФB3I       6%�	{��H���A�*;


total_loss���@

error_R�	Y?

learning_rate_1�%�8n�)]I       6%�	���H���A�*;


total_loss��@

error_R\�b?

learning_rate_1�%�8��� I       6%�	��H���A�*;


total_loss�r�@

error_R��K?

learning_rate_1�%�8����I       6%�	W_�H���A�*;


total_lossۋ�@

error_RT�`?

learning_rate_1�%�8�<�I       6%�	��H���A�*;


total_loss���@

error_R�SL?

learning_rate_1�%�8�фI       6%�	� �H���A�*;


total_loss�c�@

error_R��I?

learning_rate_1�%�81��I       6%�	�D�H���A�*;


total_loss��v@

error_Rf@?

learning_rate_1�%�8����I       6%�	 ��H���A�*;


total_loss�B�@

error_Rj�F?

learning_rate_1�%�8L�\I       6%�	���H���A�*;


total_loss���@

error_R!->?

learning_rate_1�%�8���I       6%�	��H���A�*;


total_lossȣ�@

error_Rf�b?

learning_rate_1�%�8�>4I       6%�	!a�H���A�*;


total_loss꫼@

error_RלS?

learning_rate_1�%�8�C<�I       6%�	���H���A�*;


total_loss�2�@

error_R��a?

learning_rate_1�%�8�3s�I       6%�	F��H���A�*;


total_loss�u�@

error_RnF?

learning_rate_1�%�8����I       6%�	|9�H���A�*;


total_loss���@

error_R6i?

learning_rate_1�%�8zK��I       6%�	��H���A�*;


total_loss8ҳ@

error_R��Y?

learning_rate_1�%�8�H�%I       6%�	��H���A�*;


total_loss�@

error_RL*R?

learning_rate_1�%�8F�MnI       6%�	O�H���A�*;


total_loss�[�@

error_R��K?

learning_rate_1�%�8� �I       6%�		��H���A�*;


total_loss�V�@

error_R�>?

learning_rate_1�%�8l��I       6%�	��H���A�*;


total_loss��s@

error_RMX[?

learning_rate_1�%�8�?�I       6%�	B(�H���A�*;


total_loss��y@

error_R8�D?

learning_rate_1�%�8[6�I       6%�	�o�H���A�*;


total_loss8��@

error_R�K?

learning_rate_1�%�8_�I       6%�	���H���A�*;


total_lossw9�@

error_R��G?

learning_rate_1�%�8�I       6%�	���H���A�*;


total_lossi<�@

error_R� G?

learning_rate_1�%�8�}�bI       6%�	�G�H���A�*;


total_loss#A

error_R�U?

learning_rate_1�%�8���I       6%�	e��H���A�*;


total_loss,��@

error_R��Q?

learning_rate_1�%�8�ǸRI       6%�	���H���A�*;


total_loss_8A

error_R8�U?

learning_rate_1�%�8[ePI       6%�	��H���A�*;


total_loss3��@

error_RJ�W?

learning_rate_1�%�8Ya&I       6%�	�U�H���A�*;


total_lossI��@

error_R�YS?

learning_rate_1�%�8fWW�I       6%�	���H���A�*;


total_loss���@

error_R�o^?

learning_rate_1�%�8]B��I       6%�	���H���A�*;


total_lossm�@

error_R�,\?

learning_rate_1�%�8�d��I       6%�	� �H���A�*;


total_loss.��@

error_R;�F?

learning_rate_1�%�8(���I       6%�	f�H���A�*;


total_losswL�@

error_R�\?

learning_rate_1�%�8���OI       6%�	��H���A�*;


total_loss�F�@

error_Ro�Y?

learning_rate_1�%�8�Y��I       6%�	!��H���A�*;


total_loss���@

error_R��W?

learning_rate_1�%�8J�kI       6%�	0�H���A�*;


total_lossfB�@

error_R�:G?

learning_rate_1�%�85�O_I       6%�	�s�H���A�*;


total_loss�{�@

error_R�F?

learning_rate_1�%�8���I       6%�	,��H���A�*;


total_lossr��@

error_R��E?

learning_rate_1�%�8{��I       6%�	���H���A�*;


total_loss `�@

error_R�fU?

learning_rate_1�%�8_A�SI       6%�	IE�H���A�*;


total_loss��@

error_R��D?

learning_rate_1�%�8��9I       6%�	���H���A�*;


total_loss1,�@

error_R��??

learning_rate_1�%�8f��uI       6%�	���H���A�*;


total_loss���@

error_R��??

learning_rate_1�%�89�I       6%�	��H���A�*;


total_loss�j�@

error_R�|??

learning_rate_1�%�8-9�"I       6%�	�U�H���A�*;


total_loss��A

error_R�zE?

learning_rate_1�%�8�[@NI       6%�	���H���A�*;


total_loss�A

error_R{N?

learning_rate_1�%�8�ض\I       6%�	b��H���A�*;


total_lossȶ�@

error_R7Q?

learning_rate_1�%�8a��I       6%�	��H���A�*;


total_loss$�>A

error_R_�D?

learning_rate_1�%�8�{woI       6%�	�a�H���A�*;


total_loss��@

error_RO�G?

learning_rate_1�%�8,罢I       6%�		��H���A�*;


total_lossʾA

error_R7_?

learning_rate_1�%�87u�iI       6%�	��H���A�*;


total_loss�͏@

error_R�O?

learning_rate_1�%�8�p I       6%�	 (�H���A�*;


total_loss���@

error_R��U?

learning_rate_1�%�8z�`I       6%�	El�H���A�*;


total_lossz�@

error_R_�I?

learning_rate_1�%�8���I       6%�	���H���A�*;


total_loss^A

error_R��=?

learning_rate_1�%�8�.=vI       6%�	q��H���A�*;


total_loss���@

error_R�M?

learning_rate_1�%�8���I       6%�	Z5�H���A�*;


total_loss_��@

error_R��P?

learning_rate_1�%�8��I       6%�	�x�H���A�*;


total_loss{�@

error_R�Q?

learning_rate_1�%�8��I       6%�	���H���A�*;


total_loss�7�@

error_R�iK?

learning_rate_1�%�8��UI       6%�	2�H���A�*;


total_loss���@

error_R�gX?

learning_rate_1�%�8��2�I       6%�	pD�H���A�*;


total_loss��A

error_R�ZI?

learning_rate_1�%�8{$�3I       6%�	"��H���A�*;


total_lossv�@

error_R�SN?

learning_rate_1�%�8�.}�I       6%�	u��H���A�*;


total_loss)�@

error_R��G?

learning_rate_1�%�8���I       6%�	��H���A�*;


total_loss���@

error_R6~N?

learning_rate_1�%�8rA�I       6%�	�Q�H���A�*;


total_loss��@

error_R��V?

learning_rate_1�%�8��SI       6%�	���H���A�*;


total_lossLA

error_Rv�S?

learning_rate_1�%�8����I       6%�	Q��H���A�*;


total_loss:�@

error_R6#X?

learning_rate_1�%�8�wj�I       6%�	��H���A�*;


total_lossq��@

error_RiNQ?

learning_rate_1�%�8f���I       6%�	r^�H���A�*;


total_loss���@

error_R�Q?

learning_rate_1�%�8gd��I       6%�	��H���A�*;


total_loss��@

error_R�vC?

learning_rate_1�%�8nKd�I       6%�	���H���A�*;


total_loss2��@

error_R-�N?

learning_rate_1�%�8��I       6%�	S,�H���A�*;


total_loss��$A

error_R�{Y?

learning_rate_1�%�85�:I       6%�	�t�H���A�*;


total_lossz��@

error_Ro{W?

learning_rate_1�%�8�C&I       6%�	̺�H���A�*;


total_loss�fA

error_R��=?

learning_rate_1�%�8Υ�BI       6%�	� �H���A�*;


total_lossN �@

error_R�F?

learning_rate_1�%�8�waqI       6%�	�K�H���A�*;


total_loss9g"A

error_R�V?

learning_rate_1�%�8��BI       6%�	A��H���A�*;


total_loss��@

error_R�d?

learning_rate_1�%�8�#�tI       6%�	���H���A�*;


total_loss鍴@

error_ReH?

learning_rate_1�%�8��^I       6%�	M>�H���A�*;


total_lossY�@

error_R�RI?

learning_rate_1�%�8L�a�I       6%�	*��H���A�*;


total_loss��@

error_R�DO?

learning_rate_1�%�8]*�I       6%�	T��H���A�*;


total_loss���@

error_R
Y?

learning_rate_1�%�8��%�I       6%�	�	�H���A�*;


total_loss���@

error_R{�U?

learning_rate_1�%�8��`I       6%�	�M�H���A�*;


total_loss���@

error_R�@.?

learning_rate_1�%�8Ӝ��I       6%�	��H���A�*;


total_loss�x�@

error_R�3?

learning_rate_1�%�8��NI       6%�	���H���A�*;


total_loss��@

error_RąB?

learning_rate_1�%�8	�I       6%�	-�H���A�*;


total_loss{g�@

error_R��H?

learning_rate_1�%�8�.�OI       6%�	�a�H���A�*;


total_lossqG�@

error_RÇ]?

learning_rate_1�%�8A�D|I       6%�	��H���A�*;


total_loss��@

error_R�L?

learning_rate_1�%�8tvՎI       6%�	���H���A�*;


total_loss��@

error_R�E?

learning_rate_1�%�8mw�I       6%�	C0�H���A�*;


total_loss���@

error_RC`?

learning_rate_1�%�8�#�wI       6%�	�x�H���A�*;


total_losso9�@

error_R�=H?

learning_rate_1�%�8�iI       6%�	��H���A�*;


total_loss�f�@

error_R��H?

learning_rate_1�%�8��%�I       6%�	5	�H���A�*;


total_loss�� A

error_R�{T?

learning_rate_1�%�8I��!I       6%�	!P�H���A�*;


total_lossMB�@

error_R{,8?

learning_rate_1�%�8Kw�I       6%�	��H���A�*;


total_loss��@

error_RD�T?

learning_rate_1�%�8f]I       6%�	e��H���A�*;


total_loss��@

error_R<�E?

learning_rate_1�%�8*W��I       6%�	y�H���A�*;


total_loss��@

error_R2�N?

learning_rate_1�%�8�h�+I       6%�	v]�H���A�*;


total_loss6��@

error_R�L?

learning_rate_1�%�8��wI       6%�	Ĩ�H���A�*;


total_loss��@

error_R�b?

learning_rate_1�%�8j�c�I       6%�	i��H���A�*;


total_loss���@

error_RĝI?

learning_rate_1�%�8'<ĈI       6%�	^?�H���A�*;


total_loss��@

error_R:�E?

learning_rate_1�%�8�DI       6%�	U��H���A�*;


total_loss��'A

error_RD�T?

learning_rate_1�%�8���I       6%�	���H���A�*;


total_loss�ء@

error_R�OT?

learning_rate_1�%�8�ãI       6%�		�H���A�*;


total_lossz��@

error_RڜW?

learning_rate_1�%�8P��I       6%�	�^�H���A�*;


total_loss!L�@

error_R�%J?

learning_rate_1�%�8�7�GI       6%�	���H���A�*;


total_lossx��@

error_RׇJ?

learning_rate_1�%�8�+e:I       6%�	4��H���A�*;


total_loss|��@

error_R "K?

learning_rate_1�%�8*�S�I       6%�	�)�H���A�*;


total_loss���@

error_RiX[?

learning_rate_1�%�8�G��I       6%�	�k�H���A�*;


total_loss衞@

error_R%/E?

learning_rate_1�%�8X���I       6%�	*��H���A�*;


total_loss7 �@

error_RW�Z?

learning_rate_1�%�8v,�I       6%�	���H���A�*;


total_loss(\�@

error_R��??

learning_rate_1�%�8KqNI       6%�	�3�H���A�*;


total_loss���@

error_R�E?

learning_rate_1�%�8ְd`I       6%�	�x�H���A�*;


total_loss���@

error_RҋY?

learning_rate_1�%�8�S-I       6%�	��H���A�*;


total_lossZ�@

error_R��[?

learning_rate_1�%�8+#FI       6%�	��H���A�*;


total_lossV��@

error_Rr@?

learning_rate_1�%�8'�I       6%�	�X�H���A�*;


total_loss�A

error_RD�^?

learning_rate_1�%�8�ъ�I       6%�	}��H���A�*;


total_lossvH�@

error_R��M?

learning_rate_1�%�8��F�I       6%�	-��H���A�*;


total_loss`²@

error_R�d>?

learning_rate_1�%�8�_�I       6%�	x:�H���A�*;


total_loss[��@

error_RW�V?

learning_rate_1�%�8�Qz�I       6%�	t��H���A�*;


total_loss�@

error_R�f?

learning_rate_1�%�8C��I       6%�	���H���A�*;


total_lossP�@

error_R/�Q?

learning_rate_1�%�8^@�eI       6%�	4$�H���A�*;


total_lossX=�@

error_R��>?

learning_rate_1�%�8D�)>I       6%�	�h�H���A�*;


total_lossO:A

error_R�J?

learning_rate_1�%�8�a}I       6%�	���H���A�*;


total_lossZ�@

error_R�lf?

learning_rate_1�%�8	XI       6%�	���H���A�*;


total_loss'A

error_R�r@?

learning_rate_1�%�8΋P�I       6%�	�4�H���A�*;


total_loss!�@

error_R�Q?

learning_rate_1�%�8�<1I       6%�	Ox�H���A�*;


total_loss��@

error_R!�B?

learning_rate_1�%�8M�ƟI       6%�	���H���A�*;


total_loss���@

error_RI?

learning_rate_1�%�8$q��I       6%�	��H���A�*;


total_lossS�A

error_R��Q?

learning_rate_1�%�8��t�I       6%�	CF�H���A�*;


total_lossn��@

error_R >P?

learning_rate_1�%�8�ӈ�I       6%�	A��H���A�*;


total_loss���@

error_R}QD?

learning_rate_1�%�8Ӆ=I       6%�	G��H���A�*;


total_lossKA

error_R�^?

learning_rate_1�%�8��I       6%�	��H���A�*;


total_loss���@

error_R�~F?

learning_rate_1�%�8�*I       6%�	$Y�H���A�*;


total_loss���@

error_R��M?

learning_rate_1�%�8����I       6%�	��H���A�*;


total_loss?�@

error_R�P?

learning_rate_1�%�8��,I       6%�	��H���A�*;


total_loss��@

error_R�F?

learning_rate_1�%�8��?I       6%�	�I�H���A�*;


total_loss��@

error_R�O?

learning_rate_1�%�8�*{oI       6%�	���H���A�*;


total_loss~��@

error_R?2Z?

learning_rate_1�%�8���I       6%�	3��H���A�*;


total_loss��@

error_R�U?

learning_rate_1�%�8��cI       6%�	��H���A�*;


total_lossR~@

error_R�G?

learning_rate_1�%�8B���I       6%�	C]�H���A�*;


total_loss�<�@

error_R��P?

learning_rate_1�%�8�K��I       6%�	���H���A�*;


total_loss*��@

error_R)eW?

learning_rate_1�%�8>~zI       6%�	z��H���A�*;


total_loss@��@

error_R�qV?

learning_rate_1�%�8�=MI       6%�		0�H���A�*;


total_loss��@

error_R`�J?

learning_rate_1�%�8z}�I       6%�	Mv�H���A�*;


total_loss��@

error_R)4W?

learning_rate_1�%�8f��	I       6%�	]��H���A�*;


total_lossa��@

error_R�J?

learning_rate_1�%�8.+I       6%�	���H���A�*;


total_loss��@

error_RTMJ?

learning_rate_1�%�8<�)I       6%�	�@�H���A�*;


total_loss�v@

error_R��V?

learning_rate_1�%�8q��I       6%�	���H���A�*;


total_loss���@

error_R�=?

learning_rate_1�%�8�2�yI       6%�	��H���A�*;


total_loss*��@

error_RQ�V?

learning_rate_1�%�8���CI       6%�	8�H���A�*;


total_loss;;�@

error_R�AZ?

learning_rate_1�%�8i23I       6%�	�X�H���A�*;


total_lossƓA

error_R$eY?

learning_rate_1�%�8Sj,�I       6%�	���H���A�*;


total_loss&4�@

error_R#S?

learning_rate_1�%�8w;��I       6%�	J��H���A�*;


total_loss���@

error_R?�9?

learning_rate_1�%�8���I       6%�	W*�H���A�*;


total_loss���@

error_Rd3Q?

learning_rate_1�%�8�!�I       6%�	�q�H���A�*;


total_loss���@

error_R<FP?

learning_rate_1�%�8wIwI       6%�	N��H���A�*;


total_lossL��@

error_R�]N?

learning_rate_1�%�8���I       6%�	m��H���A�*;


total_lossv��@

error_R�Y?

learning_rate_1�%�8�I       6%�	:9�H���A�*;


total_loss��@

error_R�M?

learning_rate_1�%�8ro�1I       6%�	rz�H���A�*;


total_lossUǢ@

error_R��>?

learning_rate_1�%�8�2�iI       6%�	Ѽ�H���A�*;


total_loss��@

error_REh??

learning_rate_1�%�8 ��>I       6%�	���H���A�*;


total_loss�<�@

error_R��I?

learning_rate_1�%�80��I       6%�	�? I���A�*;


total_loss���@

error_R��F?

learning_rate_1�%�8S���I       6%�	�� I���A�*;


total_loss=Q�@

error_R��O?

learning_rate_1�%�8w�XqI       6%�	�� I���A�*;


total_lossą�@

error_R7-`?

learning_rate_1�%�8W�(eI       6%�	�I���A�*;


total_loss���@

error_R�MU?

learning_rate_1�%�8�D��I       6%�	2YI���A�*;


total_loss)��@

error_RJ;?

learning_rate_1�%�8,��_I       6%�	t�I���A�*;


total_loss<ǭ@

error_R��O?

learning_rate_1�%�8x:�I       6%�	��I���A�*;


total_loss\-�@

error_R}�\?

learning_rate_1�%�8��I       6%�	�0I���A�*;


total_loss���@

error_R��F?

learning_rate_1�%�8�i7�I       6%�	\pI���A�*;


total_loss���@

error_R8�[?

learning_rate_1�%�8.y^I       6%�	?�I���A�*;


total_lossM��@

error_R��R?

learning_rate_1�%�8l,��I       6%�	F�I���A�*;


total_loss�ɉ@

error_R1�E?

learning_rate_1�%�8�s��I       6%�	dCI���A�*;


total_loss\i�@

error_RD?

learning_rate_1�%�8D�w�I       6%�	��I���A�*;


total_lossr��@

error_RfGK?

learning_rate_1�%�8�MI       6%�	�I���A�*;


total_loss��A

error_R@�V?

learning_rate_1�%�8�v�I       6%�	2I���A�*;


total_lossS3�@

error_RoMM?

learning_rate_1�%�8��R�I       6%�	�UI���A�*;


total_lossT�@

error_R�^?

learning_rate_1�%�8g[��I       6%�	��I���A�*;


total_lossO2�@

error_R,qK?

learning_rate_1�%�8/� �I       6%�	^�I���A�*;


total_lossjޞ@

error_R�1M?

learning_rate_1�%�8�~~I       6%�	I���A�*;


total_lossoa�@

error_R;�Y?

learning_rate_1�%�8�	�5I       6%�	�bI���A�*;


total_lossQf�@

error_R�/\?

learning_rate_1�%�8���}I       6%�	6�I���A�*;


total_loss���@

error_R�G?

learning_rate_1�%�8��zkI       6%�	��I���A�*;


total_loss�.�@

error_R�L]?

learning_rate_1�%�8��B�I       6%�	B2I���A�*;


total_loss�
�@

error_Rq�O?

learning_rate_1�%�8~ɱI       6%�	��I���A�*;


total_loss�l�@

error_RW|U?

learning_rate_1�%�8�ÎI       6%�	��I���A�*;


total_loss���@

error_R�I?

learning_rate_1�%�8)��I       6%�	>I���A�*;


total_loss*��@

error_R�^?

learning_rate_1�%�8#�7I       6%�	EOI���A�*;


total_loss�Y�@

error_R�eW?

learning_rate_1�%�848gI       6%�	o�I���A�*;


total_lossԮ�@

error_Rq�5?

learning_rate_1�%�8��lRI       6%�	 �I���A�*;


total_lossQr�@

error_R�:O?

learning_rate_1�%�8�я�I       6%�	�I���A�*;


total_lossX�@

error_R��E?

learning_rate_1�%�8d9اI       6%�	�]I���A�*;


total_loss�W�@

error_R/M?

learning_rate_1�%�8M}?I       6%�	ܼI���A�*;


total_loss*{�@

error_RT�J?

learning_rate_1�%�8�r�I       6%�	�	I���A�*;


total_lossߋ�@

error_RwT?

learning_rate_1�%�8{�&�I       6%�	�W	I���A�*;


total_losss�@

error_R"[?

learning_rate_1�%�8���uI       6%�	(�	I���A�*;


total_loss���@

error_R�U?

learning_rate_1�%�8��I       6%�	~�	I���A�*;


total_loss��A

error_Rf�J?

learning_rate_1�%�8�":�I       6%�	�3
I���A�*;


total_loss��A

error_R�-e?

learning_rate_1�%�8l�hI       6%�	e{
I���A�*;


total_loss�L�@

error_R�Q?

learning_rate_1�%�8bf(I       6%�	��
I���A�*;


total_lossn̘@

error_Rfxa?

learning_rate_1�%�8w��QI       6%�	�I���A�*;


total_loss]s�@

error_R�R?

learning_rate_1�%�8�Z�I       6%�	>MI���A�*;


total_loss�\�@

error_R��P?

learning_rate_1�%�87
�I       6%�	O�I���A�*;


total_lossQ�@

error_R7V?

learning_rate_1�%�8�Z�NI       6%�	��I���A�*;


total_lossl�U@

error_R�(G?

learning_rate_1�%�8d�J:I       6%�	�(I���A�*;


total_loss��@

error_R��Y?

learning_rate_1�%�8f�qI       6%�	�vI���A�*;


total_loss{5�@

error_R[�_?

learning_rate_1�%�8�Dw�I       6%�	M�I���A�*;


total_lossZ�@

error_Ri�I?

learning_rate_1�%�8�`�I       6%�	�I���A�*;


total_loss���@

error_Rm>E?

learning_rate_1�%�8?�oI       6%�	%PI���A�*;


total_loss_`�@

error_R�@?

learning_rate_1�%�8Ɓ�I       6%�	6�I���A�*;


total_loss�'�@

error_RNCV?

learning_rate_1�%�8�CA�I       6%�	��I���A�*;


total_loss턔@

error_R��=?

learning_rate_1�%�8A�J9I       6%�	�I���A�*;


total_loss�2�@

error_R҇I?

learning_rate_1�%�8�tX�I       6%�	�dI���A�*;


total_loss���@

error_RH�O?

learning_rate_1�%�8Eq��I       6%�	u�I���A�*;


total_lossD��@

error_RP]?

learning_rate_1�%�8��J�I       6%�	��I���A�*;


total_lossH��@

error_R�S?

learning_rate_1�%�8.�D&I       6%�	j<I���A�*;


total_loss���@

error_R�YB?

learning_rate_1�%�8�VCMI       6%�	�I���A�*;


total_lossLA�@

error_Rf?

learning_rate_1�%�8� �%I       6%�	��I���A�*;


total_loss�t�@

error_R�eA?

learning_rate_1�%�8��ՖI       6%�	`I���A�*;


total_lossl�@

error_R�G?

learning_rate_1�%�8���gI       6%�	�VI���A�*;


total_loss_��@

error_R}PE?

learning_rate_1�%�8=	�	I       6%�	͞I���A�*;


total_loss9�@

error_R�AD?

learning_rate_1�%�8J]��I       6%�	C�I���A�*;


total_loss���@

error_R�F?

learning_rate_1�%�8��PI       6%�	d&I���A�*;


total_loss4�@

error_R�\J?

learning_rate_1�%�8#�� I       6%�	�iI���A�*;


total_loss��@

error_RJ�N?

learning_rate_1�%�8lG�QI       6%�	�I���A�*;


total_loss�]�@

error_R\MF?

learning_rate_1�%�8��M�I       6%�	H�I���A�*;


total_loss���@

error_R��??

learning_rate_1�%�8���I       6%�	'<I���A�*;


total_loss%�@

error_R\qY?

learning_rate_1�%�8�aږI       6%�	�I���A�*;


total_loss��@

error_R�8Y?

learning_rate_1�%�8e���I       6%�	y�I���A�*;


total_lossn��@

error_R͜W?

learning_rate_1�%�8:	�yI       6%�	I���A�*;


total_lossRl�@

error_R�lX?

learning_rate_1�%�8�!xTI       6%�	LNI���A�*;


total_loss[�@

error_R�2V?

learning_rate_1�%�8���I       6%�	(�I���A�*;


total_losst�@

error_R��L?

learning_rate_1�%�8���I       6%�	o�I���A�*;


total_loss��(A

error_R�^?

learning_rate_1�%�8���hI       6%�	�I���A�*;


total_loss��A

error_Rt�X?

learning_rate_1�%�8c��I       6%�	�cI���A�*;


total_loss�� A

error_R�2P?

learning_rate_1�%�8���I       6%�	׷I���A�*;


total_loss�yA

error_R!�6?

learning_rate_1�%�8��` I       6%�	&�I���A�*;


total_lossx��@

error_RN�L?

learning_rate_1�%�8-��4I       6%�	DI���A�*;


total_loss��@

error_RE�V?

learning_rate_1�%�8y�j�I       6%�	��I���A�*;


total_loss�D�@

error_Ra�Q?

learning_rate_1�%�8>0�YI       6%�	��I���A�*;


total_loss��@

error_Rj�Y?

learning_rate_1�%�8*6{,I       6%�	�4I���A�*;


total_loss|��@

error_R@"V?

learning_rate_1�%�8�Kk�I       6%�	�zI���A�*;


total_loss�k�@

error_R�1V?

learning_rate_1�%�8�$��I       6%�	��I���A�*;


total_loss΢�@

error_R�1Y?

learning_rate_1�%�8�r�:I       6%�	�I���A�*;


total_loss;�@

error_R-�N?

learning_rate_1�%�8լ�I       6%�	gUI���A�*;


total_lossΏ@

error_R��J?

learning_rate_1�%�8�i��I       6%�	��I���A�*;


total_lossI��@

error_R�IQ?

learning_rate_1�%�8�׾�I       6%�	J�I���A�*;


total_loss�'�@

error_R`�G?

learning_rate_1�%�8���II       6%�	�'I���A�*;


total_loss��@

error_R��W?

learning_rate_1�%�8	P�I       6%�	�pI���A�*;


total_loss��@

error_R��J?

learning_rate_1�%�8��SI       6%�	��I���A�*;


total_loss��@

error_R�CJ?

learning_rate_1�%�8��"I       6%�	S*I���A�*;


total_loss?��@

error_R�aJ?

learning_rate_1�%�8u/ 4I       6%�	�sI���A�*;


total_loss
!�@

error_RT�`?

learning_rate_1�%�82l�I       6%�	�I���A�*;


total_loss3q�@

error_R��o?

learning_rate_1�%�80V�lI       6%�	�I���A�*;


total_lossHe�@

error_R%N?

learning_rate_1�%�8�wI       6%�	�LI���A�*;


total_loss�A�@

error_RŮH?

learning_rate_1�%�8
G��I       6%�	5�I���A�*;


total_loss�(�@

error_R��@?

learning_rate_1�%�8��C
I       6%�	~�I���A�*;


total_loss���@

error_R3<X?

learning_rate_1�%�8w�y�I       6%�	!I���A�*;


total_loss�UA

error_RO?

learning_rate_1�%�8�j�qI       6%�	n\I���A�*;


total_loss;1�@

error_RdFR?

learning_rate_1�%�8$�]wI       6%�	p�I���A�*;


total_loss&��@

error_R/ V?

learning_rate_1�%�8%U��I       6%�	��I���A�*;


total_loss�g@

error_R<?

learning_rate_1�%�8�e-I       6%�	�3I���A�*;


total_loss�n�@

error_R�K?

learning_rate_1�%�8�|�I       6%�	uI���A�*;


total_loss���@

error_R�'[?

learning_rate_1�%�8�VeI       6%�	U�I���A�*;


total_loss��@

error_R�gO?

learning_rate_1�%�8�V�I       6%�	XI���A�*;


total_lossͳ�@

error_RIyT?

learning_rate_1�%�8�S�I       6%�	oLI���A�*;


total_loss�^�@

error_R�N?

learning_rate_1�%�8�%��I       6%�	��I���A�*;


total_lossi�@

error_RIiJ?

learning_rate_1�%�8�b�hI       6%�	��I���A�*;


total_loss�]�@

error_R;�[?

learning_rate_1�%�86���I       6%�	�I���A�*;


total_lossoCA

error_R#�R?

learning_rate_1�%�8ᴄ�I       6%�	XYI���A�*;


total_lossR#�@

error_Rc/[?

learning_rate_1�%�8e���I       6%�	��I���A�*;


total_loss�3�@

error_Ra�S?

learning_rate_1�%�8���GI       6%�	��I���A�*;


total_loss���@

error_Ra?

learning_rate_1�%�8]1.�I       6%�	�!I���A�*;


total_loss�>�@

error_R�E?

learning_rate_1�%�8���I       6%�	kI���A�*;


total_lossH:�@

error_R��T?

learning_rate_1�%�8��I       6%�	L�I���A�*;


total_loss��3A

error_R؀B?

learning_rate_1�%�8�7��I       6%�	��I���A�*;


total_loss\�@

error_R�??

learning_rate_1�%�8c�G�I       6%�	\? I���A�*;


total_loss�e�@

error_R}RT?

learning_rate_1�%�87}��I       6%�	� I���A�*;


total_loss��@

error_RM7c?

learning_rate_1�%�8ٻI       6%�	�� I���A�*;


total_lossvsv@

error_R?B?

learning_rate_1�%�8=�MI       6%�	p!I���A�*;


total_loss�m�@

error_Rt�G?

learning_rate_1�%�8�q�MI       6%�	�P!I���A�*;


total_loss���@

error_R,�K?

learning_rate_1�%�8���sI       6%�	��!I���A�*;


total_loss�L�@

error_R�C?

learning_rate_1�%�8kƠI       6%�	�!I���A�*;


total_loss��@

error_R��V?

learning_rate_1�%�8l�7I       6%�	o"I���A�*;


total_loss���@

error_R��A?

learning_rate_1�%�8�.�I       6%�	5`"I���A�*;


total_loss��@

error_R�@\?

learning_rate_1�%�8�q�tI       6%�	B�"I���A�*;


total_loss���@

error_RMHF?

learning_rate_1�%�8��aI       6%�	��"I���A�*;


total_loss,�@

error_R_�V?

learning_rate_1�%�8�
�I       6%�	�6#I���A�*;


total_lossom�@

error_R��F?

learning_rate_1�%�8����I       6%�	�y#I���A�*;


total_lossf��@

error_R��G?

learning_rate_1�%�8��I       6%�	E�#I���A�*;


total_loss��A

error_R�Q?

learning_rate_1�%�8� �I       6%�	�$I���A�*;


total_loss��@

error_R�LA?

learning_rate_1�%�8��j�I       6%�	`D$I���A�*;


total_loss��@

error_Ri�I?

learning_rate_1�%�84B)6I       6%�	ņ$I���A�*;


total_loss3��@

error_R��L?

learning_rate_1�%�8�cI       6%�	��$I���A�*;


total_lossA��@

error_R�5A?

learning_rate_1�%�8ߏaI       6%�	-%I���A�*;


total_loss���@

error_R,xP?

learning_rate_1�%�8el I       6%�	�S%I���A�*;


total_lossjA�@

error_RR�@?

learning_rate_1�%�8^�%�I       6%�	ט%I���A�*;


total_lossf�@

error_R��P?

learning_rate_1�%�8���$I       6%�	��%I���A�*;


total_loss�ެ@

error_R��:?

learning_rate_1�%�8;��I       6%�	�(&I���A�*;


total_loss~�@

error_R�	W?

learning_rate_1�%�8'Lq5I       6%�	o&I���A�*;


total_loss\L�@

error_R�
\?

learning_rate_1�%�86�t�I       6%�	#�&I���A�*;


total_losss��@

error_R�xe?

learning_rate_1�%�8�x+PI       6%�	'I���A�*;


total_loss���@

error_R��R?

learning_rate_1�%�8�=
�I       6%�	�G'I���A�*;


total_lossl�@

error_Rn_?

learning_rate_1�%�8�g�~I       6%�	8�'I���A�*;


total_lossZ�@

error_RL�L?

learning_rate_1�%�8�7��I       6%�	`�'I���A�*;


total_loss�"�@

error_RIO?

learning_rate_1�%�8{�fI       6%�	r(I���A�*;


total_loss���@

error_R;+C?

learning_rate_1�%�8����I       6%�	�Z(I���A�*;


total_lossn��@

error_R
�L?

learning_rate_1�%�8m�i�I       6%�	'�(I���A�*;


total_loss��A

error_RRrC?

learning_rate_1�%�8X׷rI       6%�	c)I���A�*;


total_loss.�@

error_R��W?

learning_rate_1�%�8�qSFI       6%�	�N)I���A�*;


total_lossUxA

error_RET?

learning_rate_1�%�8�^�I       6%�	��)I���A�*;


total_loss�w�@

error_R��P?

learning_rate_1�%�8p�٫I       6%�	��)I���A�*;


total_lossDA

error_Rw�X?

learning_rate_1�%�8��7�I       6%�	�%*I���A�*;


total_loss�y�@

error_R�Lb?

learning_rate_1�%�8�aI       6%�	>l*I���A�*;


total_loss�NA

error_R��L?

learning_rate_1�%�8�~�I       6%�	ɱ*I���A�*;


total_loss
��@

error_R�]?

learning_rate_1�%�8����I       6%�	��*I���A�*;


total_loss��@

error_RR�Z?

learning_rate_1�%�8���I       6%�	1>+I���A�*;


total_loss�E�@

error_R,�S?

learning_rate_1�%�83-��I       6%�	M�+I���A�*;


total_loss�C�@

error_R�K?

learning_rate_1�%�8Ȉ�)I       6%�	��+I���A�*;


total_lossCm�@

error_R��S?

learning_rate_1�%�8���I       6%�	[,I���A�*;


total_loss�]�@

error_R�G?

learning_rate_1�%�82�=I       6%�	�J,I���A�*;


total_loss[�@

error_R��B?

learning_rate_1�%�8�B��I       6%�	��,I���A�*;


total_loss�*�@

error_R��=?

learning_rate_1�%�8@���I       6%�	�,I���A�*;


total_loss�{�@

error_R�bO?

learning_rate_1�%�8e�e�I       6%�	�%-I���A�*;


total_lossϸ@

error_R�A?

learning_rate_1�%�8�.(I       6%�	�o-I���A�*;


total_loss�$�@

error_Rco?

learning_rate_1�%�8Q�J�I       6%�	A�-I���A�*;


total_loss���@

error_R�Z?

learning_rate_1�%�8=�*�I       6%�	��-I���A�*;


total_lossSz�@

error_R��S?

learning_rate_1�%�8�̓�I       6%�	(E.I���A�*;


total_lossʜ@

error_Rd�J?

learning_rate_1�%�8��0I       6%�	U'1I���A�*;


total_lossM�v@

error_R�I?

learning_rate_1�%�8E��wI       6%�	Yl1I���A�*;


total_loss��@

error_R�E?

learning_rate_1�%�8���I       6%�	&�1I���A�*;


total_loss��@

error_R�ZP?

learning_rate_1�%�8
xy�I       6%�	g2I���A�*;


total_lossz��@

error_R�U?

learning_rate_1�%�8B��I       6%�	�I2I���A�*;


total_loss]g�@

error_R;�Y?

learning_rate_1�%�8�2mI       6%�	�2I���A�*;


total_loss`A

error_R�hE?

learning_rate_1�%�8�T�I       6%�	�2I���A�*;


total_lossF��@

error_Rs�`?

learning_rate_1�%�8:J3�I       6%�	�3I���A�*;


total_loss���@

error_R�tJ?

learning_rate_1�%�8	²I       6%�	�h3I���A�*;


total_loss��q@

error_R.�O?

learning_rate_1�%�8�=�I       6%�	��3I���A�*;


total_loss8CA

error_R� _?

learning_rate_1�%�8����I       6%�	��3I���A�*;


total_loss���@

error_R��H?

learning_rate_1�%�8�5�eI       6%�	.M4I���A�*;


total_loss��@

error_R�7C?

learning_rate_1�%�8����I       6%�	��4I���A�*;


total_loss2<�@

error_R�oK?

learning_rate_1�%�8:݇;I       6%�	0�4I���A�*;


total_loss]�@

error_RCIQ?

learning_rate_1�%�8�I       6%�	`D5I���A�*;


total_loss'�@

error_R�BH?

learning_rate_1�%�8�7�I       6%�	�5I���A�*;


total_loss���@

error_R�e?

learning_rate_1�%�8�.n�I       6%�	8�5I���A�*;


total_lossZ��@

error_R�[?

learning_rate_1�%�8o�0EI       6%�	 6I���A�*;


total_loss�@

error_R��??

learning_rate_1�%�8܊�@I       6%�	.Z6I���A�*;


total_losse�@

error_R1�X?

learning_rate_1�%�8�i�<I       6%�	�6I���A�*;


total_loss%�@

error_Rc�N?

learning_rate_1�%�8f�b?I       6%�	��6I���A�*;


total_loss��n@

error_R��L?

learning_rate_1�%�8�J�{I       6%�	�57I���A�*;


total_loss�L�@

error_Rhd?

learning_rate_1�%�8����I       6%�	�x7I���A�*;


total_loss_�@

error_R��J?

learning_rate_1�%�8�	��I       6%�	4�7I���A�*;


total_loss\�	A

error_R/�A?

learning_rate_1�%�8��DI       6%�	x8I���A�*;


total_loss?�A

error_R�V?

learning_rate_1�%�8⃨�I       6%�	�H8I���A�*;


total_loss���@

error_R��T?

learning_rate_1�%�8��M�I       6%�	ؕ8I���A�*;


total_loss�ϣ@

error_R��>?

learning_rate_1�%�8���I       6%�	��8I���A�*;


total_loss��@

error_Re�T?

learning_rate_1�%�8j��6I       6%�	�19I���A�*;


total_loss�(�@

error_R�XA?

learning_rate_1�%�8�U�I       6%�	eu9I���A�*;


total_loss� �@

error_R��Z?

learning_rate_1�%�8w��(I       6%�	��9I���A�*;


total_loss�x�@

error_R�:M?

learning_rate_1�%�8>�΂I       6%�	�9I���A�*;


total_lossl_�@

error_R!�E?

learning_rate_1�%�8�-͔I       6%�	�C:I���A�*;


total_loss���@

error_R�K?

learning_rate_1�%�8
b�@I       6%�	%�:I���A�*;


total_loss�"�@

error_R��Q?

learning_rate_1�%�8i�I       6%�	U�:I���A�*;


total_loss ħ@

error_Rt3W?

learning_rate_1�%�82�f�I       6%�	�;I���A�*;


total_loss���@

error_Rn�Z?

learning_rate_1�%�8K�:I       6%�	�[;I���A�*;


total_lossH��@

error_Rx�H?

learning_rate_1�%�8�K'�I       6%�	_�;I���A�*;


total_loss���@

error_RT�D?

learning_rate_1�%�8F�3I       6%�	��;I���A�*;


total_loss<ͥ@

error_R��R?

learning_rate_1�%�8�?�I       6%�	�J<I���A�*;


total_lossZ��@

error_RCC]?

learning_rate_1�%�8��)I       6%�	Ē<I���A�*;


total_loss���@

error_R�E?

learning_rate_1�%�8��NI       6%�	k�<I���A�*;


total_loss�L�@

error_RנO?

learning_rate_1�%�8޿��I       6%�	�=I���A�*;


total_loss�@

error_RP?

learning_rate_1�%�8�I       6%�	�c=I���A�*;


total_lossV�@

error_R�QR?

learning_rate_1�%�8�
\AI       6%�	߲=I���A�*;


total_lossL�@

error_RA�I?

learning_rate_1�%�8���I       6%�	�>I���A�*;


total_lossW��@

error_R.�J?

learning_rate_1�%�8�g�I       6%�	�a>I���A�*;


total_loss�j�@

error_R}FY?

learning_rate_1�%�8��`~I       6%�	�>I���A�*;


total_loss�_�@

error_R{�M?

learning_rate_1�%�8,���I       6%�	;�>I���A�*;


total_lossȚ�@

error_RVQ?

learning_rate_1�%�8'�9�I       6%�	O\?I���A�*;


total_lossW��@

error_RشE?

learning_rate_1�%�8����I       6%�	)�?I���A�*;


total_lossTi�@

error_R�L?

learning_rate_1�%�8��I       6%�	��?I���A�*;


total_loss6��@

error_R3�R?

learning_rate_1�%�8� .�I       6%�	,@I���A�*;


total_loss��k@

error_R�=?

learning_rate_1�%�8��c)I       6%�	{p@I���A�*;


total_loss�_�@

error_RH�O?

learning_rate_1�%�8Q�I       6%�	`�@I���A�*;


total_loss�A

error_R��N?

learning_rate_1�%�8�I�"I       6%�	NAI���A�*;


total_loss��A

error_R�K?

learning_rate_1�%�8�Ӊ�I       6%�	,yAI���A�*;


total_lossc4�@

error_R��P?

learning_rate_1�%�8�/b�I       6%�	��AI���A�*;


total_loss�:�@

error_Ra�W?

learning_rate_1�%�8�X�I       6%�	�BI���A�*;


total_lossH��@

error_R��K?

learning_rate_1�%�8��d�I       6%�	�bBI���A�*;


total_loss;S�@

error_R3xC?

learning_rate_1�%�8b�I       6%�	G�BI���A�*;


total_loss�c�@

error_R|�K?

learning_rate_1�%�8���I       6%�	f�BI���A�*;


total_loss ��@

error_R�[?

learning_rate_1�%�8�y��I       6%�	F@CI���A�*;


total_loss���@

error_R�=D?

learning_rate_1�%�8�Y	�I       6%�	O�CI���A�*;


total_loss���@

error_RM�F?

learning_rate_1�%�8/tkI       6%�	�CI���A�*;


total_loss3��@

error_R��W?

learning_rate_1�%�8����I       6%�	3DI���A�*;


total_lossA��@

error_R��<?

learning_rate_1�%�8�Y�I       6%�	5XDI���A�*;


total_losss�@

error_R��d?

learning_rate_1�%�8��I       6%�	��DI���A�*;


total_loss;�@

error_R�_?

learning_rate_1�%�8���vI       6%�	w�DI���A�*;


total_lossJR�@

error_R�^`?

learning_rate_1�%�8y��I       6%�	}(EI���A�*;


total_lossD��@

error_R/PV?

learning_rate_1�%�8`�5&I       6%�	ipEI���A�*;


total_losso��@

error_R�I?

learning_rate_1�%�8��wI       6%�	��EI���A�*;


total_lossv�u@

error_RC!B?

learning_rate_1�%�8�e��I       6%�	1�EI���A�*;


total_loss z�@

error_R��A?

learning_rate_1�%�8��FI       6%�	AFI���A�*;


total_loss6�@

error_R�u\?

learning_rate_1�%�86��I       6%�	F�FI���A�*;


total_loss}�@

error_R�HV?

learning_rate_1�%�8�E�II       6%�	I�FI���A�*;


total_lossq��@

error_R3�X?

learning_rate_1�%�8W��I       6%�	�GI���A�*;


total_loss;#�@

error_R��<?

learning_rate_1�%�8RV�I       6%�	/\GI���A�*;


total_loss���@

error_RùE?

learning_rate_1�%�8
KI       6%�	�GI���A�*;


total_loss:�@

error_R�@Y?

learning_rate_1�%�8��"�I       6%�	+�GI���A�*;


total_loss%YA

error_R(U?

learning_rate_1�%�8am�I       6%�	�"HI���A�*;


total_loss���@

error_RI ^?

learning_rate_1�%�8���fI       6%�	jHI���A�*;


total_loss)��@

error_R�E?

learning_rate_1�%�8�:�I       6%�	��HI���A�*;


total_loss�i@

error_R=�6?

learning_rate_1�%�8�Ir�I       6%�	�II���A�*;


total_loss�c�@

error_R�kJ?

learning_rate_1�%�8A�>1I       6%�	�hII���A�*;


total_loss	c�@

error_RvCH?

learning_rate_1�%�87IoqI       6%�	įII���A�*;


total_loss���@

error_RaIY?

learning_rate_1�%�8��I       6%�	��II���A�*;


total_loss���@

error_R��G?

learning_rate_1�%�8w���I       6%�	�>JI���A�*;


total_loss�(�@

error_RJ�=?

learning_rate_1�%�8���I       6%�	=�JI���A�*;


total_loss�B�@

error_R��W?

learning_rate_1�%�8��V�I       6%�	�JI���A�*;


total_loss�v�@

error_RX/A?

learning_rate_1�%�8J� �I       6%�	�KI���A�*;


total_lossۻA

error_R��J?

learning_rate_1�%�8��I       6%�	ILKI���A�*;


total_lossat�@

error_RRsW?

learning_rate_1�%�8����I       6%�	��KI���A�*;


total_loss+I�@

error_R�Z?

learning_rate_1�%�8^�;�I       6%�	L�KI���A�*;


total_loss�O�@

error_RH�@?

learning_rate_1�%�8A~T�I       6%�	�LI���A�*;


total_loss���@

error_R�zN?

learning_rate_1�%�8,6{�I       6%�	)jLI���A�*;


total_lossQv	A

error_R@_Q?

learning_rate_1�%�8ٜ�6I       6%�	�LI���A�*;


total_loss`)�@

error_RȣO?

learning_rate_1�%�8��x"I       6%�	�MI���A�*;


total_loss|��@

error_R$7I?

learning_rate_1�%�8��-\I       6%�	�IMI���A�*;


total_loss���@

error_R�?N?

learning_rate_1�%�8�z�}I       6%�	&�MI���A�*;


total_loss��@

error_RO�B?

learning_rate_1�%�8�|��I       6%�	u�MI���A�*;


total_lossB��@

error_RQ#Y?

learning_rate_1�%�8�-I       6%�	>NI���A�*;


total_loss��2A

error_R�KE?

learning_rate_1�%�8�W�I       6%�	gcNI���A�*;


total_loss3x�@

error_R�m?

learning_rate_1�%�8JB�#I       6%�	0�NI���A�*;


total_loss��@

error_R��M?

learning_rate_1�%�8�Xn^I       6%�	��NI���A�*;


total_lossC��@

error_RO?

learning_rate_1�%�8���I       6%�	X2OI���A�*;


total_loss_|�@

error_RݨJ?

learning_rate_1�%�8�[�I       6%�	NwOI���A�*;


total_lossM-A

error_Ra�S?

learning_rate_1�%�8�
��I       6%�	|�OI���A�*;


total_loss e�@

error_RV�_?

learning_rate_1�%�8㦰�I       6%�	 PI���A�*;


total_lossN�x@

error_R��@?

learning_rate_1�%�8�בoI       6%�	�IPI���A�*;


total_loss���@

error_R�W?

learning_rate_1�%�8RwH%I       6%�	׎PI���A�*;


total_lossh	A

error_R�{X?

learning_rate_1�%�8��I       6%�	}�PI���A�*;


total_loss��@

error_R�i?

learning_rate_1�%�8� $hI       6%�	�QI���A�*;


total_loss��@

error_R
�=?

learning_rate_1�%�8���I       6%�	�YQI���A�*;


total_loss���@

error_R��T?

learning_rate_1�%�8x�"#I       6%�	>�QI���A�*;


total_loss<ޱ@

error_R�W?

learning_rate_1�%�8��ЭI       6%�	��QI���A�*;


total_loss�i�@

error_R;fM?

learning_rate_1�%�8��GI       6%�	�,RI���A�*;


total_loss��@

error_R��R?

learning_rate_1�%�8�)I       6%�	�oRI���A�*;


total_lossV.A

error_R�[?

learning_rate_1�%�8��lI       6%�	o�RI���A�*;


total_loss�-�@

error_Ro.C?

learning_rate_1�%�84:QcI       6%�	�RI���A�*;


total_loss�K�@

error_R͑@?

learning_rate_1�%�8
r��I       6%�	,<SI���A�*;


total_lossW��@

error_R��E?

learning_rate_1�%�8Uó
I       6%�	7�SI���A�*;


total_loss�h�@

error_R?W?

learning_rate_1�%�8;��I       6%�	'�SI���A�*;


total_lossҔ�@

error_RZ�G?

learning_rate_1�%�8|�I       6%�	�TI���A�*;


total_loss��@

error_Rץd?

learning_rate_1�%�8#a�I       6%�	?mTI���A�*;


total_loss�Y�@

error_R1oW?

learning_rate_1�%�8��v�I       6%�	��TI���A�*;


total_loss�@

error_R��V?

learning_rate_1�%�8��T�I       6%�	UI���A�*;


total_loss㭴@

error_R��j?

learning_rate_1�%�8g]`�I       6%�	�aUI���A�*;


total_loss�A

error_R{�M?

learning_rate_1�%�8�$3GI       6%�	~�UI���A�*;


total_loss�ț@

error_RD}B?

learning_rate_1�%�8�Wz�I       6%�	��UI���A�*;


total_loss���@

error_R��>?

learning_rate_1�%�8�aI       6%�	9VI���A�*;


total_lossڹ�@

error_Rͧ[?

learning_rate_1�%�8_!��I       6%�	|�VI���A�*;


total_loss�z�@

error_R��R?

learning_rate_1�%�8�ČwI       6%�	��VI���A�*;


total_loss�?�@

error_R��Y?

learning_rate_1�%�8*�I       6%�	�WI���A�*;


total_lossĺ@

error_RsM?

learning_rate_1�%�8QLŗI       6%�	XWI���A�*;


total_lossuh�@

error_R�`;?

learning_rate_1�%�89���I       6%�	�WI���A�*;


total_loss�@

error_R��E?

learning_rate_1�%�8L'�I       6%�	�WI���A�*;


total_loss���@

error_R��T?

learning_rate_1�%�8Kl�WI       6%�	F)XI���A�*;


total_loss�OA

error_RE8D?

learning_rate_1�%�8����I       6%�	�rXI���A�*;


total_lossc��@

error_R,�]?

learning_rate_1�%�8�GЩI       6%�	r�XI���A�*;


total_loss��A

error_R�BT?

learning_rate_1�%�8��F�I       6%�	�"YI���A�*;


total_loss(��@

error_R43M?

learning_rate_1�%�8����I       6%�	�eYI���A�*;


total_loss�"�@

error_R�;?

learning_rate_1�%�8s�sI       6%�	2�YI���A�*;


total_loss���@

error_R6�W?

learning_rate_1�%�8'��$I       6%�	^�YI���A�*;


total_loss�͉@

error_R�sT?

learning_rate_1�%�8m�CI       6%�	�4ZI���A�*;


total_loss���@

error_R�nX?

learning_rate_1�%�8��1�I       6%�	:xZI���A�*;


total_loss���@

error_RȌb?

learning_rate_1�%�8G,�7I       6%�	��ZI���A�*;


total_loss@��@

error_R�W?

learning_rate_1�%�8֝�[I       6%�	�[I���A�*;


total_loss�V�@

error_Rag9?

learning_rate_1�%�8�8��I       6%�	�V[I���A�*;


total_lossC��@

error_R&�X?

learning_rate_1�%�8�n�nI       6%�	�[I���A�*;


total_loss��@

error_R&�B?

learning_rate_1�%�8��DCI       6%�	h�[I���A�*;


total_loss!�A

error_RĉO?

learning_rate_1�%�8���I       6%�	�'\I���A�	*;


total_loss䱄@

error_R�5G?

learning_rate_1�%�8�f�*I       6%�	�k\I���A�	*;


total_loss^{�@

error_R fG?

learning_rate_1�%�8�k�(I       6%�	��\I���A�	*;


total_loss���@

error_R�mB?

learning_rate_1�%�8�!��I       6%�	��\I���A�	*;


total_lossI��@

error_RͬJ?

learning_rate_1�%�8	�I       6%�	5D]I���A�	*;


total_loss��@

error_R�gY?

learning_rate_1�%�8�j�5I       6%�	��]I���A�	*;


total_loss���@

error_R!�K?

learning_rate_1�%�8쎉�I       6%�	��]I���A�	*;


total_loss���@

error_RWAR?

learning_rate_1�%�8�X2�I       6%�	~^I���A�	*;


total_losse��@

error_R�]?

learning_rate_1�%�8��14I       6%�	mM^I���A�	*;


total_loss�:�@

error_R%S?

learning_rate_1�%�8�y�I       6%�	��^I���A�	*;


total_lossF��@

error_RTP?

learning_rate_1�%�8�:L�I       6%�	U�^I���A�	*;


total_loss ��@

error_R��@?

learning_rate_1�%�8�F}�I       6%�	�_I���A�	*;


total_loss{�@

error_R@�X?

learning_rate_1�%�8� ��I       6%�	.`_I���A�	*;


total_loss)~�@

error_R�0N?

learning_rate_1�%�8b��I       6%�	�_I���A�	*;


total_loss�1�@

error_R�@X?

learning_rate_1�%�8�!�I       6%�	��_I���A�	*;


total_loss}�@

error_R�Y?

learning_rate_1�%�8��`}I       6%�	T&`I���A�	*;


total_loss�e�@

error_R�AM?

learning_rate_1�%�8�۪zI       6%�	&h`I���A�	*;


total_loss�۟@

error_R/n?

learning_rate_1�%�8
9��I       6%�	
�`I���A�	*;


total_loss�ܩ@

error_R�R?

learning_rate_1�%�8Z�\I       6%�	�`I���A�	*;


total_loss)�@

error_R��G?

learning_rate_1�%�8��IiI       6%�	�EaI���A�	*;


total_loss��@

error_R [?

learning_rate_1�%�8���6I       6%�	͌aI���A�	*;


total_lossaư@

error_R�@?

learning_rate_1�%�8j=I       6%�	}�aI���A�	*;


total_loss!t�@

error_R�^C?

learning_rate_1�%�8�T_I       6%�	bI���A�	*;


total_loss��@

error_R�V?

learning_rate_1�%�8J�I       6%�	[bI���A�	*;


total_lossxBA

error_RiQ?

learning_rate_1�%�8$AD�I       6%�	*�bI���A�	*;


total_loss<8�@

error_R�
W?

learning_rate_1�%�8�QAI       6%�	I�bI���A�	*;


total_loss"�@

error_RH/>?

learning_rate_1�%�8 d��I       6%�	A/cI���A�	*;


total_loss.��@

error_Rna\?

learning_rate_1�%�8�r�@I       6%�	�scI���A�	*;


total_loss�y�@

error_R8�T?

learning_rate_1�%�8�>m�I       6%�	B�cI���A�	*;


total_loss
�@

error_R��B?

learning_rate_1�%�86�I       6%�	o�cI���A�	*;


total_loss�"�@

error_Rf�V?

learning_rate_1�%�8�*}�I       6%�	�AdI���A�	*;


total_loss1�@

error_RҺM?

learning_rate_1�%�8��'�I       6%�	I�dI���A�	*;


total_loss�qA

error_R��Q?

learning_rate_1�%�8��w�I       6%�	#�dI���A�	*;


total_lossa,�@

error_R��L?

learning_rate_1�%�8��cCI       6%�	eI���A�	*;


total_loss��@

error_RQD?

learning_rate_1�%�8��BwI       6%�	�UeI���A�	*;


total_lossx1�@

error_R�uV?

learning_rate_1�%�8p�X�I       6%�	��eI���A�	*;


total_loss߲A

error_RN�Z?

learning_rate_1�%�8 ��I       6%�	�eI���A�	*;


total_loss�C�@

error_RJ�H?

learning_rate_1�%�8�t}I       6%�	"fI���A�	*;


total_lossF�@

error_R��8?

learning_rate_1�%�8
aS�I       6%�	�ffI���A�	*;


total_loss��@

error_RҭX?

learning_rate_1�%�8�U� I       6%�	�fI���A�	*;


total_loss��@

error_RX�]?

learning_rate_1�%�89�=�I       6%�	Q�fI���A�	*;


total_loss���@

error_R��:?

learning_rate_1�%�8��W�I       6%�	%AgI���A�	*;


total_loss���@

error_Rv`?

learning_rate_1�%�8V>�@I       6%�	��gI���A�	*;


total_loss1�n@

error_R��??

learning_rate_1�%�8�L��I       6%�	'�gI���A�	*;


total_losskA

error_RZ�^?

learning_rate_1�%�8E[I       6%�	�hI���A�	*;


total_loss֏�@

error_R��R?

learning_rate_1�%�8٪I       6%�	�dhI���A�	*;


total_loss�;�@

error_R$/N?

learning_rate_1�%�8��I       6%�	#�hI���A�	*;


total_lossF��@

error_R�!P?

learning_rate_1�%�8��_I       6%�	�iI���A�	*;


total_lossZ��@

error_R��A?

learning_rate_1�%�8�Σ%I       6%�	uPiI���A�	*;


total_loss8U�@

error_R��A?

learning_rate_1�%�8_��vI       6%�	b�iI���A�	*;


total_loss0�@

error_R.�>?

learning_rate_1�%�8�/εI       6%�	��iI���A�	*;


total_loss���@

error_RlsQ?

learning_rate_1�%�8��ډI       6%�	RjI���A�	*;


total_loss�	�@

error_R&FZ?

learning_rate_1�%�8�P��I       6%�	cjI���A�	*;


total_loss}*�@

error_R�ea?

learning_rate_1�%�8�KLXI       6%�	��jI���A�	*;


total_lossX�@

error_R)bV?

learning_rate_1�%�8vy��I       6%�	=�jI���A�	*;


total_loss�K�@

error_R(8?

learning_rate_1�%�8��RzI       6%�	�+kI���A�	*;


total_loss��@

error_R?jS?

learning_rate_1�%�8_��WI       6%�	brkI���A�	*;


total_loss�[A

error_R��S?

learning_rate_1�%�8i"0�I       6%�	��kI���A�	*;


total_lossd�c@

error_R��P?

learning_rate_1�%�8��ϗI       6%�	�lI���A�	*;


total_lossvz@

error_Rn�J?

learning_rate_1�%�8��dxI       6%�	�QlI���A�	*;


total_lossa)�@

error_R�*N?

learning_rate_1�%�8�7q�I       6%�	ߘlI���A�	*;


total_loss@��@

error_RoK?

learning_rate_1�%�8+���I       6%�	��lI���A�	*;


total_lossLn�@

error_RX?

learning_rate_1�%�8IdZI       6%�	�'mI���A�	*;


total_loss���@

error_RS�R?

learning_rate_1�%�8���1I       6%�	rmmI���A�	*;


total_lossS)�@

error_Rn�F?

learning_rate_1�%�8��HjI       6%�	��mI���A�	*;


total_loss���@

error_R�yY?

learning_rate_1�%�8�Ff�I       6%�	A�mI���A�	*;


total_loss��@

error_R;xR?

learning_rate_1�%�8t^ZOI       6%�	mEnI���A�	*;


total_loss�&s@

error_R�n??

learning_rate_1�%�8L!NmI       6%�	�nI���A�	*;


total_loss�J�@

error_R�G?

learning_rate_1�%�8];~�I       6%�	8�nI���A�	*;


total_loss��@

error_R�L?

learning_rate_1�%�8ȷʌI       6%�	~oI���A�	*;


total_loss|ܩ@

error_R�QM?

learning_rate_1�%�8_�d�I       6%�	�hoI���A�	*;


total_lossE��@

error_RRSW?

learning_rate_1�%�8*��I       6%�	�oI���A�	*;


total_loss�H�@

error_R��Z?

learning_rate_1�%�8YI       6%�	��oI���A�	*;


total_lossmJ�@

error_Rs�X?

learning_rate_1�%�8��oI       6%�	$JpI���A�	*;


total_lossh��@

error_R��??

learning_rate_1�%�8D�XI       6%�	G�pI���A�	*;


total_loss}{�@

error_R3R?

learning_rate_1�%�8 _RI       6%�	��pI���A�	*;


total_lossO�@

error_R�$Z?

learning_rate_1�%�8`�vI       6%�	�qI���A�	*;


total_loss�ƙ@

error_R��T?

learning_rate_1�%�8&�!CI       6%�	V^qI���A�	*;


total_loss�A

error_Rz`Z?

learning_rate_1�%�8)��KI       6%�	�qI���A�	*;


total_loss���@

error_R۷??

learning_rate_1�%�8�O�I       6%�	�qI���A�	*;


total_loss�R�@

error_R�!9?

learning_rate_1�%�8x㙽I       6%�	�.rI���A�	*;


total_loss|��@

error_R�'V?

learning_rate_1�%�8OfI       6%�	�rrI���A�	*;


total_loss���@

error_R��>?

learning_rate_1�%�8�](�I       6%�	}�rI���A�	*;


total_lossc��@

error_RQQC?

learning_rate_1�%�8��T�I       6%�	��rI���A�	*;


total_lossÒ�@

error_R)Nq?

learning_rate_1�%�8�m"oI       6%�	Q=sI���A�	*;


total_loss=)�@

error_R��N?

learning_rate_1�%�8��-?I       6%�	t�sI���A�	*;


total_loss��@

error_R.G?

learning_rate_1�%�8�T�>I       6%�	��sI���A�	*;


total_loss�p�@

error_R�3?

learning_rate_1�%�8"�]�I       6%�	tI���A�	*;


total_loss��@

error_RA�_?

learning_rate_1�%�8cFV�I       6%�	�HtI���A�	*;


total_loss��@

error_R
�V?

learning_rate_1�%�8�/�I       6%�	D�tI���A�	*;


total_lossj,�@

error_R�\[?

learning_rate_1�%�8"���I       6%�	�tI���A�	*;


total_lossi�@

error_RA_?

learning_rate_1�%�8,޹'I       6%�	,FuI���A�	*;


total_losshA

error_R �J?

learning_rate_1�%�8t"��I       6%�	�uI���A�	*;


total_loss�)A

error_RHIE?

learning_rate_1�%�89r�I       6%�	��uI���A�	*;


total_lossq��@

error_R	�T?

learning_rate_1�%�8�JE�I       6%�	�vI���A�	*;


total_loss.�|@

error_R��U?

learning_rate_1�%�8���I       6%�	BcvI���A�	*;


total_loss�ر@

error_R��J?

learning_rate_1�%�8�	��I       6%�	
�vI���A�	*;


total_loss��@

error_R��g?

learning_rate_1�%�8�%�I       6%�	��vI���A�	*;


total_loss���@

error_R#�`?

learning_rate_1�%�8�gFI       6%�	r6wI���A�	*;


total_loss�"A

error_R�^L?

learning_rate_1�%�8_��I       6%�	\~wI���A�	*;


total_loss#"�@

error_RV?

learning_rate_1�%�8��I       6%�	S�wI���A�	*;


total_loss�j�@

error_RinN?

learning_rate_1�%�86�x�I       6%�	>	xI���A�	*;


total_lossq!�@

error_R�AS?

learning_rate_1�%�8F��I       6%�	nQxI���A�	*;


total_losso��@

error_R�>L?

learning_rate_1�%�8|1��I       6%�	G�xI���A�	*;


total_loss�u�@

error_R�H?

learning_rate_1�%�8���ZI       6%�	yI���A�	*;


total_lossVʯ@

error_R��B?

learning_rate_1�%�8;]��I       6%�	,OyI���A�	*;


total_loss��@

error_R�~A?

learning_rate_1�%�8(���I       6%�	��yI���A�	*;


total_loss���@

error_RC�@?

learning_rate_1�%�8����I       6%�	��yI���A�	*;


total_lossNf�@

error_RA�P?

learning_rate_1�%�8�ZI       6%�	&,zI���A�	*;


total_lossI��@

error_R�uN?

learning_rate_1�%�8�5YI       6%�	~zzI���A�	*;


total_lossۅ�@

error_RODP?

learning_rate_1�%�8�},I       6%�	��zI���A�	*;


total_loss�9�@

error_R=�Q?

learning_rate_1�%�8k%q�I       6%�	�
{I���A�	*;


total_loss���@

error_R�1A?

learning_rate_1�%�8�ɼiI       6%�		N{I���A�	*;


total_lossv��@

error_R�;?

learning_rate_1�%�8?�:_I       6%�	Y�{I���A�	*;


total_loss���@

error_R�<I?

learning_rate_1�%�8ty�2I       6%�	R�{I���A�	*;


total_loss/ܝ@

error_RE�Q?

learning_rate_1�%�8�1_I       6%�	]|I���A�	*;


total_lossQ�@

error_R�}]?

learning_rate_1�%�8�@owI       6%�	"^|I���A�	*;


total_loss��@

error_R)1S?

learning_rate_1�%�8{_1�I       6%�	X�|I���A�	*;


total_loss��@

error_R:R?

learning_rate_1�%�8}���I       6%�	��|I���A�	*;


total_loss'��@

error_R��N?

learning_rate_1�%�8T���I       6%�	)}I���A�	*;


total_lossrc�@

error_RxS?

learning_rate_1�%�8)W�I       6%�	�n}I���A�	*;


total_loss��@

error_RM�^?

learning_rate_1�%�8�G�\I       6%�	,�}I���A�	*;


total_lossDA

error_R]hL?

learning_rate_1�%�8�i��I       6%�	��}I���A�	*;


total_lossX�A

error_R��R?

learning_rate_1�%�84�UI       6%�	qC~I���A�	*;


total_loss?�@

error_R%vM?

learning_rate_1�%�8�IB�I       6%�	~�~I���A�	*;


total_loss�d�@

error_R�KQ?

learning_rate_1�%�8؆�I       6%�	��~I���A�	*;


total_loss��M@

error_R6>?

learning_rate_1�%�8�l��I       6%�	�I���A�	*;


total_loss�!�@

error_R��a?

learning_rate_1�%�8 [)�I       6%�	�jI���A�	*;


total_loss��@

error_R.�`?

learning_rate_1�%�8��I       6%�	��I���A�
*;


total_loss��`@

error_R�Z?

learning_rate_1�%�8�U��I       6%�	��I���A�
*;


total_loss ��@

error_R�mC?

learning_rate_1�%�80��{I       6%�	�;�I���A�
*;


total_lossoo�@

error_R;PP?

learning_rate_1�%�8RL]I       6%�	���I���A�
*;


total_lossh��@

error_RԪj?

learning_rate_1�%�8{���I       6%�	ȀI���A�
*;


total_lossf#�@

error_R]�R?

learning_rate_1�%�8�:�	I       6%�	\�I���A�
*;


total_lossZͱ@

error_R�G?

learning_rate_1�%�8
��I       6%�	�U�I���A�
*;


total_loss<�@

error_RH�e?

learning_rate_1�%�8�+��I       6%�	���I���A�
*;


total_loss�3�@

error_R zL?

learning_rate_1�%�8
~I       6%�	���I���A�
*;


total_loss��@

error_R��b?

learning_rate_1�%�8�`֬I       6%�	�$�I���A�
*;


total_loss*��@

error_R��@?

learning_rate_1�%�8�&I       6%�	Lk�I���A�
*;


total_lossx��@

error_R��T?

learning_rate_1�%�8����I       6%�	���I���A�
*;


total_loss�p�@

error_R`xA?

learning_rate_1�%�8$��I       6%�	���I���A�
*;


total_loss�V�@

error_R!6J?

learning_rate_1�%�8�u�I       6%�	iI�I���A�
*;


total_loss�U�@

error_R@�^?

learning_rate_1�%�8��f�I       6%�	C��I���A�
*;


total_loss��@

error_R�;?

learning_rate_1�%�8�ॱI       6%�	:уI���A�
*;


total_loss�@

error_R��??

learning_rate_1�%�8��;I       6%�	��I���A�
*;


total_loss�A

error_R�Z\?

learning_rate_1�%�8c/��I       6%�	�b�I���A�
*;


total_loss?H�@

error_R�AP?

learning_rate_1�%�8���cI       6%�	쩄I���A�
*;


total_loss@

error_R}�G?

learning_rate_1�%�8.k��I       6%�	��I���A�
*;


total_loss�2DA

error_R�<M?

learning_rate_1�%�8�4DI       6%�	�=�I���A�
*;


total_lossjh�@

error_RZW?

learning_rate_1�%�8����I       6%�	���I���A�
*;


total_loss��@

error_RS�b?

learning_rate_1�%�8
���I       6%�	�ӅI���A�
*;


total_lossa��@

error_RFI?

learning_rate_1�%�8���I       6%�	@�I���A�
*;


total_lossT:�@

error_R��H?

learning_rate_1�%�83(�I       6%�	�W�I���A�
*;


total_loss��@

error_Rc?

learning_rate_1�%�8h��`I       6%�	�I���A�
*;


total_loss���@

error_R�xT?

learning_rate_1�%�8�3��I       6%�	�܆I���A�
*;


total_losseԱ@

error_R�0??

learning_rate_1�%�8��מI       6%�	��I���A�
*;


total_lossA��@

error_R�dS?

learning_rate_1�%�8�S�bI       6%�	�d�I���A�
*;


total_loss���@

error_R�\?

learning_rate_1�%�8�\O*I       6%�	ũ�I���A�
*;


total_loss8�@

error_R��J?

learning_rate_1�%�8��I       6%�	��I���A�
*;


total_loss�"�@

error_RXA?

learning_rate_1�%�8�0I       6%�	�1�I���A�
*;


total_lossI��@

error_RN�^?

learning_rate_1�%�8S�FHI       6%�	^x�I���A�
*;


total_loss$��@

error_R��C?

learning_rate_1�%�8�G�qI       6%�	S�I���A�
*;


total_loss�Y�@

error_R�T?

learning_rate_1�%�8/7�oI       6%�	�.�I���A�
*;


total_loss�@

error_R ?L?

learning_rate_1�%�8����I       6%�	6q�I���A�
*;


total_loss%�@

error_Rq�d?

learning_rate_1�%�8e�VI       6%�	���I���A�
*;


total_losstI�@

error_R��U?

learning_rate_1�%�8M�FI       6%�	���I���A�
*;


total_loss4I�@

error_R�Cr?

learning_rate_1�%�8�omI       6%�	�:�I���A�
*;


total_loss��
A

error_Rq�<?

learning_rate_1�%�8��*�I       6%�	�I���A�
*;


total_loss6�@

error_R�;9?

learning_rate_1�%�8'�#I       6%�	�ɊI���A�
*;


total_lossh�@

error_RO2L?

learning_rate_1�%�8����I       6%�	t�I���A�
*;


total_loss  �@

error_Rq|J?

learning_rate_1�%�8��!yI       6%�	)Z�I���A�
*;


total_losst}�@

error_R��G?

learning_rate_1�%�8��NI       6%�	���I���A�
*;


total_loss���@

error_R�A?

learning_rate_1�%�8v�^�I       6%�	$�I���A�
*;


total_lossHp�@

error_R7T?

learning_rate_1�%�8�y�8I       6%�	c&�I���A�
*;


total_loss�~�@

error_RaR4?

learning_rate_1�%�8���I       6%�	�k�I���A�
*;


total_loss#��@

error_R=O?

learning_rate_1�%�8����I       6%�	h��I���A�
*;


total_lossj�@

error_R�ZB?

learning_rate_1�%�87-2�I       6%�	@��I���A�
*;


total_loss�³@

error_RRP?

learning_rate_1�%�8WLI       6%�	�>�I���A�
*;


total_lossC)�@

error_R
�A?

learning_rate_1�%�8'��I       6%�	���I���A�
*;


total_loss5�@

error_R�r<?

learning_rate_1�%�8�$3I       6%�	E̍I���A�
*;


total_loss15�@

error_RұW?

learning_rate_1�%�8��FI       6%�	��I���A�
*;


total_lossM��@

error_R:uS?

learning_rate_1�%�86$��I       6%�	HV�I���A�
*;


total_loss2��@

error_R6�e?

learning_rate_1�%�8?D��I       6%�	��I���A�
*;


total_loss��@

error_R��O?

learning_rate_1�%�8�Y�I       6%�	��I���A�
*;


total_lossL�z@

error_R�7b?

learning_rate_1�%�8���I       6%�	v,�I���A�
*;


total_lossVL�@

error_R)�H?

learning_rate_1�%�8��7I       6%�	St�I���A�
*;


total_lossi��@

error_RڞC?

learning_rate_1�%�8
�RMI       6%�	6��I���A�
*;


total_loss8Ip@

error_R�J??

learning_rate_1�%�8�
�I       6%�	S�I���A�
*;


total_lossGf�@

error_Rw�M?

learning_rate_1�%�8�4SI       6%�	[L�I���A�
*;


total_loss��d@

error_R&�L?

learning_rate_1�%�83���I       6%�	S��I���A�
*;


total_loss>ׅ@

error_RD�A?

learning_rate_1�%�8�XnI       6%�	��I���A�
*;


total_loss���@

error_R�[o?

learning_rate_1�%�8�%�tI       6%�	r.�I���A�
*;


total_losse�@

error_R��??

learning_rate_1�%�89EI       6%�	>z�I���A�
*;


total_loss/�@

error_R�C?

learning_rate_1�%�8�~5I       6%�	;I���A�
*;


total_loss�"�@

error_R�sD?

learning_rate_1�%�8f&��I       6%�	U�I���A�
*;


total_lossh��@

error_RL�V?

learning_rate_1�%�8�P7�I       6%�		R�I���A�
*;


total_loss)*�@

error_R�L?

learning_rate_1�%�8YL�I       6%�	K��I���A�
*;


total_loss�g�@

error_R�:K?

learning_rate_1�%�8�ZWzI       6%�	 �I���A�
*;


total_loss��@

error_R�T?

learning_rate_1�%�8��I       6%�	0�I���A�
*;


total_lossn�@

error_R��O?

learning_rate_1�%�8���\I       6%�	�u�I���A�
*;


total_losshQ�@

error_R#�;?

learning_rate_1�%�8�pI       6%�	<��I���A�
*;


total_loss���@

error_R�L?

learning_rate_1�%�8�_��I       6%�	� �I���A�
*;


total_loss8��@

error_R�Z?

learning_rate_1�%�8s�XI       6%�	�E�I���A�
*;


total_loss�Q�@

error_RH�H?

learning_rate_1�%�8�k�	I       6%�	���I���A�
*;


total_loss��@

error_RJ�F?

learning_rate_1�%�8f��"I       6%�	���I���A�
*;


total_loss��@

error_R�~E?

learning_rate_1�%�8����I       6%�	�>�I���A�
*;


total_loss���@

error_R�LF?

learning_rate_1�%�8��mI       6%�	���I���A�
*;


total_loss���@

error_R�L?

learning_rate_1�%�8 Y&I       6%�	�˕I���A�
*;


total_lossz{�@

error_R@T?

learning_rate_1�%�8.��I       6%�	b�I���A�
*;


total_loss��@

error_R�TO?

learning_rate_1�%�8��EI       6%�	�T�I���A�
*;


total_loss�y�@

error_R�q_?

learning_rate_1�%�8m��I       6%�	瘖I���A�
*;


total_loss&�@

error_R�`a?

learning_rate_1�%�8t��/I       6%�	�ܖI���A�
*;


total_loss[B�@

error_R	�\?

learning_rate_1�%�8���I       6%�	.'�I���A�
*;


total_lossg�A

error_RԥI?

learning_rate_1�%�8�I       6%�	�o�I���A�
*;


total_lossE�@

error_R��S?

learning_rate_1�%�8D��I       6%�	}��I���A�
*;


total_loss��@

error_R��O?

learning_rate_1�%�88�:�I       6%�	��I���A�
*;


total_loss�#�@

error_R�!O?

learning_rate_1�%�8�د�I       6%�	 S�I���A�
*;


total_lossX!�@

error_R�gI?

learning_rate_1�%�8��bI       6%�	T��I���A�
*;


total_loss�4�@

error_R�\@?

learning_rate_1�%�8�H��I       6%�	B
�I���A�
*;


total_loss�@

error_Rw�I?

learning_rate_1�%�8�Sr�I       6%�	�T�I���A�
*;


total_lossJy�@

error_Rq�P?

learning_rate_1�%�8��,I       6%�	˗�I���A�
*;


total_lossl��@

error_Ri�<?

learning_rate_1�%�8�ޞ�I       6%�	cݙI���A�
*;


total_loss;\�@

error_R��??

learning_rate_1�%�8�|��I       6%�	�!�I���A�
*;


total_loss/�@

error_REM?

learning_rate_1�%�8>J+TI       6%�	Rg�I���A�
*;


total_loss\ʙ@

error_R�m?

learning_rate_1�%�8aW�I       6%�	���I���A�
*;


total_loss���@

error_R!RG?

learning_rate_1�%�8c�SI       6%�	�I���A�
*;


total_loss(�@

error_RZ�Z?

learning_rate_1�%�8�۝PI       6%�	kH�I���A�
*;


total_lossv8�@

error_R��??

learning_rate_1�%�8�\DI       6%�	��I���A�
*;


total_lossf�@

error_RȧL?

learning_rate_1�%�8�ٟ�I       6%�	@ӛI���A�
*;


total_loss���@

error_R��a?

learning_rate_1�%�8�p=bI       6%�	J�I���A�
*;


total_loss<��@

error_R�RI?

learning_rate_1�%�8[*<�I       6%�	�_�I���A�
*;


total_loss�L�@

error_R߶P?

learning_rate_1�%�8��I       6%�	d��I���A�
*;


total_lossZ��@

error_RN[?

learning_rate_1�%�8=�c]I       6%�	[�I���A�
*;


total_losszb�@

error_RA�=?

learning_rate_1�%�8�\i�I       6%�	9�I���A�
*;


total_loss,��@

error_R�xT?

learning_rate_1�%�8��GI       6%�	�I���A�
*;


total_loss��A

error_R��H?

learning_rate_1�%�8fJ�}I       6%�	!ÝI���A�
*;


total_loss=��@

error_R)cP?

learning_rate_1�%�8sI       6%�	��I���A�
*;


total_loss3�@

error_Rxf\?

learning_rate_1�%�8�
�eI       6%�	�Q�I���A�
*;


total_losst�@

error_R?}W?

learning_rate_1�%�8&�J�I       6%�	&��I���A�
*;


total_loss�-�@

error_RE8X?

learning_rate_1�%�8���I       6%�	ܞI���A�
*;


total_loss�l�@

error_R�GU?

learning_rate_1�%�8���?I       6%�	�(�I���A�
*;


total_loss��@

error_REjH?

learning_rate_1�%�8I���I       6%�	ps�I���A�
*;


total_loss�w�@

error_RvO?

learning_rate_1�%�8��!I       6%�	⷟I���A�
*;


total_loss�A

error_Rx�9?

learning_rate_1�%�8��,I       6%�	��I���A�
*;


total_lossMA

error_R��G?

learning_rate_1�%�8��I       6%�	�B�I���A�
*;


total_loss��@

error_R=I>?

learning_rate_1�%�8���)I       6%�	�I���A�
*;


total_loss��v@

error_R�[?

learning_rate_1�%�8�X�hI       6%�	�͠I���A�
*;


total_loss�%f@

error_R�(J?

learning_rate_1�%�8����I       6%�	��I���A�
*;


total_loss�ԕ@

error_R�`?

learning_rate_1�%�8�藊I       6%�	�]�I���A�
*;


total_losst�a@

error_Rs�Q?

learning_rate_1�%�8S(�I       6%�	���I���A�
*;


total_loss�x�@

error_RsX9?

learning_rate_1�%�8�r� I       6%�	��I���A�
*;


total_loss��@

error_R�lC?

learning_rate_1�%�8(�īI       6%�	6�I���A�
*;


total_lossH�@

error_R��`?

learning_rate_1�%�8!!I       6%�	�z�I���A�
*;


total_loss�@

error_RT�W?

learning_rate_1�%�8Oe��I       6%�	���I���A�
*;


total_lossٿ@

error_R�zK?

learning_rate_1�%�88s<9I       6%�	�I���A�
*;


total_loss���@

error_Ri�Z?

learning_rate_1�%�8&�~I       6%�	Q�I���A�
*;


total_loss���@

error_R$0T?

learning_rate_1�%�82�?�I       6%�	��I���A�*;


total_loss���@

error_R��V?

learning_rate_1�%�8���I       6%�	��I���A�*;


total_loss�g�@

error_R�mT?

learning_rate_1�%�8��6�I       6%�	p(�I���A�*;


total_loss���@

error_RME??

learning_rate_1�%�8Y8J(I       6%�	tn�I���A�*;


total_lossv��@

error_R�H?

learning_rate_1�%�8;���I       6%�	���I���A�*;


total_loss��A

error_RjC?

learning_rate_1�%�8�qR�I       6%�	<��I���A�*;


total_loss�3�@

error_R�H?

learning_rate_1�%�8T$hI       6%�	v;�I���A�*;


total_loss�9�@

error_R�eK?

learning_rate_1�%�8�C�"I       6%�	��I���A�*;


total_lossL��@

error_RDc?

learning_rate_1�%�8�q)I       6%�	ʥI���A�*;


total_loss�c�@

error_R�
P?

learning_rate_1�%�8��qI       6%�	{�I���A�*;


total_loss���@

error_R�A?

learning_rate_1�%�8�|�I       6%�	�U�I���A�*;


total_lossTa�@

error_Rv�L?

learning_rate_1�%�8!Sz�I       6%�	��I���A�*;


total_lossho�@

error_R��N?

learning_rate_1�%�8؀��I       6%�	�ۦI���A�*;


total_lossR8�@

error_R�wS?

learning_rate_1�%�8I�u6I       6%�	g�I���A�*;


total_lossCd�@

error_R@;R?

learning_rate_1�%�8z��I       6%�	�o�I���A�*;


total_loss�ɱ@

error_RRD\?

learning_rate_1�%�8�}f�I       6%�	޳�I���A�*;


total_loss���@

error_R�V?

learning_rate_1�%�8e!I       6%�	���I���A�*;


total_lossFS�@

error_R%F?

learning_rate_1�%�8i��I       6%�	CK�I���A�*;


total_loss�8�@

error_R�<N?

learning_rate_1�%�8��:I       6%�	
��I���A�*;


total_loss���@

error_RXqO?

learning_rate_1�%�8�/��I       6%�	� �I���A�*;


total_loss��A

error_R�\?

learning_rate_1�%�8���YI       6%�	�M�I���A�*;


total_loss��E@

error_R��F?

learning_rate_1�%�8��#+I       6%�	���I���A�*;


total_loss8��@

error_R7�W?

learning_rate_1�%�8B��I       6%�	GܩI���A�*;


total_lossD_�@

error_R�:c?

learning_rate_1�%�87>[kI       6%�	��I���A�*;


total_loss�{r@

error_R�1J?

learning_rate_1�%�8��ۼI       6%�	�^�I���A�*;


total_lossĤ@

error_R}�J?

learning_rate_1�%�8cI       6%�	բ�I���A�*;


total_loss�	�@

error_Ro�@?

learning_rate_1�%�8�x��I       6%�	��I���A�*;


total_lossQ��@

error_R��I?

learning_rate_1�%�8u��I       6%�	'.�I���A�*;


total_loss��@

error_R�%]?

learning_rate_1�%�8K!��I       6%�	v�I���A�*;


total_loss���@

error_R�SK?

learning_rate_1�%�8L�OI       6%�	|��I���A�*;


total_loss�ɭ@

error_R�5K?

learning_rate_1�%�8�v�yI       6%�	b�I���A�*;


total_lossv�@

error_RE�U?

learning_rate_1�%�8ȹ&pI       6%�	�j�I���A�*;


total_loss^�A

error_R�@S?

learning_rate_1�%�8א�&I       6%�	���I���A�*;


total_losst!A

error_R�jL?

learning_rate_1�%�8Q8��I       6%�	u��I���A�*;


total_lossᥡ@

error_R�S?

learning_rate_1�%�8x�AI       6%�	�;�I���A�*;


total_loss���@

error_RAE?

learning_rate_1�%�8x_�I       6%�	�}�I���A�*;


total_loss���@

error_R$QU?

learning_rate_1�%�8��yjI       6%�	���I���A�*;


total_loss�d�@

error_R��L?

learning_rate_1�%�8���CI       6%�	y�I���A�*;


total_loss2��@

error_R3?Z?

learning_rate_1�%�8I       6%�	�H�I���A�*;


total_loss�p�@

error_R�~V?

learning_rate_1�%�8qK.I       6%�	���I���A�*;


total_loss(]�@

error_R
�H?

learning_rate_1�%�8��u�I       6%�	�ҮI���A�*;


total_loss�	�@

error_RvW9?

learning_rate_1�%�8>�#I       6%�	*�I���A�*;


total_loss� �@

error_R��Y?

learning_rate_1�%�8UQ�PI       6%�	W�I���A�*;


total_lossDd�@

error_R*U?

learning_rate_1�%�8Y_��I       6%�	i��I���A�*;


total_lossaJ�@

error_R�H?

learning_rate_1�%�8ps&I       6%�	�ݯI���A�*;


total_loss�Ǒ@

error_R؉O?

learning_rate_1�%�8�K�I       6%�	�"�I���A�*;


total_loss��@

error_R�NH?

learning_rate_1�%�8	]dI       6%�	�e�I���A�*;


total_lossO��@

error_R�F?

learning_rate_1�%�8p�I       6%�	-��I���A�*;


total_loss`�@

error_RC+E?

learning_rate_1�%�8�ڧ�I       6%�	i�I���A�*;


total_loss�j�@

error_R�@?

learning_rate_1�%�8�~�I       6%�	�;�I���A�*;


total_loss �@

error_RwH?

learning_rate_1�%�8oA�I       6%�	,��I���A�*;


total_loss��@

error_R��F?

learning_rate_1�%�8�MI       6%�	 ıI���A�*;


total_loss���@

error_Rz[?

learning_rate_1�%�8_�nvI       6%�		�I���A�*;


total_lossF�@

error_R��J?

learning_rate_1�%�8�h�I       6%�	N�I���A�*;


total_loss�~�@

error_R�DI?

learning_rate_1�%�8&c-I       6%�	�I���A�*;


total_loss��@

error_RضK?

learning_rate_1�%�8���I       6%�	޲I���A�*;


total_lossA��@

error_R�bA?

learning_rate_1�%�8�J*�I       6%�	�&�I���A�*;


total_loss��r@

error_R	#A?

learning_rate_1�%�8N�ɠI       6%�	jp�I���A�*;


total_loss%�{@

error_RτU?

learning_rate_1�%�8���mI       6%�	ܷ�I���A�*;


total_loss���@

error_R!c<?

learning_rate_1�%�8�P,�I       6%�	��I���A�*;


total_loss|�@

error_R=/=?

learning_rate_1�%�8w�Y�I       6%�	<�I���A�*;


total_lossz�@

error_R�!E?

learning_rate_1�%�8����I       6%�	L��I���A�*;


total_loss�V�@

error_R�X;?

learning_rate_1�%�8y���I       6%�	�ݴI���A�*;


total_loss���@

error_R�nW?

learning_rate_1�%�8���I       6%�	-�I���A�*;


total_loss=��@

error_R��L?

learning_rate_1�%�8t���I       6%�	F�I���A�*;


total_loss�@

error_R,�Z?

learning_rate_1�%�8_�0I       6%�	�ĵI���A�*;


total_lossWި@

error_R�_?

learning_rate_1�%�8�f��I       6%�	��I���A�*;


total_loss�u�@

error_R��M?

learning_rate_1�%�8
� �I       6%�	&O�I���A�*;


total_loss���@

error_R}�N?

learning_rate_1�%�8G-FaI       6%�	2��I���A�*;


total_loss}uA

error_Rs�L?

learning_rate_1�%�8��ŬI       6%�	O۶I���A�*;


total_lossL��@

error_R�I?

learning_rate_1�%�8�'XI       6%�	 �I���A�*;


total_loss��@

error_R��F?

learning_rate_1�%�8)2TKI       6%�	�f�I���A�*;


total_loss��A

error_RSQ?

learning_rate_1�%�8���I       6%�	���I���A�*;


total_loss�a�@

error_R�^?

learning_rate_1�%�8��I\I       6%�	���I���A�*;


total_loss���@

error_R��I?

learning_rate_1�%�8�R�+I       6%�	�F�I���A�*;


total_loss7��@

error_RJ�U?

learning_rate_1�%�8��I       6%�	���I���A�*;


total_loss}U�@

error_R�B?

learning_rate_1�%�8����I       6%�	U�I���A�*;


total_losss$�@

error_R�s7?

learning_rate_1�%�8[ۘrI       6%�	�F�I���A�*;


total_loss��@

error_R!,K?

learning_rate_1�%�8 m$I       6%�	���I���A�*;


total_loss7��@

error_R��D?

learning_rate_1�%�85ݤ�I       6%�	�ϹI���A�*;


total_lossͮ�@

error_RKB?

learning_rate_1�%�8J�I       6%�	�I���A�*;


total_loss�_�@

error_R��Q?

learning_rate_1�%�8�t}�I       6%�	RX�I���A�*;


total_loss�D�@

error_R6>C?

learning_rate_1�%�8���I       6%�	雺I���A�*;


total_loss\`�@

error_RN?

learning_rate_1�%�8��BI       6%�	!�I���A�*;


total_lossa��@

error_R�#R?

learning_rate_1�%�8�i�DI       6%�	"�I���A�*;


total_lossq"�@

error_R{oQ?

learning_rate_1�%�8�7�fI       6%�	Cm�I���A�*;


total_loss�n�@

error_R,�^?

learning_rate_1�%�8��hcI       6%�	@��I���A�*;


total_lossVj�@

error_R{y=?

learning_rate_1�%�8�u�fI       6%�	3��I���A�*;


total_losso��@

error_R�Z?

learning_rate_1�%�8pj�{I       6%�	�=�I���A�*;


total_loss�h�@

error_R�B?

learning_rate_1�%�8�^�I       6%�	���I���A�*;


total_loss���@

error_R.vD?

learning_rate_1�%�8�c�I       6%�	�ͼI���A�*;


total_loss��@

error_R@C[?

learning_rate_1�%�8w�O�I       6%�	1�I���A�*;


total_lossUd�@

error_R��P?

learning_rate_1�%�8A��I       6%�	cb�I���A�*;


total_loss���@

error_Rh�8?

learning_rate_1�%�8&9ҹI       6%�	���I���A�*;


total_lossnRA

error_R3*Q?

learning_rate_1�%�8�ٔ�I       6%�	�I���A�*;


total_loss�f�@

error_R�P?

learning_rate_1�%�8ms��I       6%�	6�I���A�*;


total_lossE�@

error_R$YT?

learning_rate_1�%�8�~4I       6%�	�|�I���A�*;


total_lossPE#A

error_R�2\?

learning_rate_1�%�8�PL-I       6%�	���I���A�*;


total_loss���@

error_RT�M?

learning_rate_1�%�8���I       6%�	P�I���A�*;


total_loss��@

error_Rn�T?

learning_rate_1�%�8�I       6%�	�J�I���A�*;


total_lossl�@

error_R��P?

learning_rate_1�%�8T�I       6%�	���I���A�*;


total_lossS��@

error_R�;?

learning_rate_1�%�8�-�I       6%�	�ӿI���A�*;


total_loss�K�@

error_R�6?

learning_rate_1�%�8��gI       6%�	r�I���A�*;


total_loss֐@

error_R�7?

learning_rate_1�%�8HԠ�I       6%�	[^�I���A�*;


total_loss��@

error_R&ab?

learning_rate_1�%�8��?�I       6%�	c��I���A�*;


total_loss��@

error_RA�U?

learning_rate_1�%�80��[I       6%�	U��I���A�*;


total_losse+�@

error_R�R?

learning_rate_1�%�8�N�I       6%�	�)�I���A�*;


total_lossH�@

error_R_M?

learning_rate_1�%�8CwaTI       6%�	�l�I���A�*;


total_loss�@

error_R�?Q?

learning_rate_1�%�8�F�I       6%�	���I���A�*;


total_lossc��@

error_R�R>?

learning_rate_1�%�8��I       6%�	{��I���A�*;


total_loss��@

error_RWGG?

learning_rate_1�%�8F�okI       6%�	�9�I���A�*;


total_loss�v�@

error_R$�@?

learning_rate_1�%�8� HI       6%�	u��I���A�*;


total_lossl&y@

error_RCD?

learning_rate_1�%�8�eI       6%�	���I���A�*;


total_loss���@

error_R�6^?

learning_rate_1�%�8Ɲ?I       6%�	�I���A�*;


total_loss)U�@

error_R 5T?

learning_rate_1�%�8,#JI       6%�	U�I���A�*;


total_loss&h�@

error_Rv�<?

learning_rate_1�%�8�@F�I       6%�	̚�I���A�*;


total_loss	N�@

error_RabT?

learning_rate_1�%�8��*�I       6%�	���I���A�*;


total_lossސ@

error_R7;Z?

learning_rate_1�%�8�e�xI       6%�	P&�I���A�*;


total_loss["�@

error_R�%P?

learning_rate_1�%�8��9"I       6%�	�m�I���A�*;


total_loss�A�@

error_R}�L?

learning_rate_1�%�8�KI       6%�	4��I���A�*;


total_loss�	c@

error_R	-B?

learning_rate_1�%�8{���I       6%�	���I���A�*;


total_lossn$�@

error_RyE?

learning_rate_1�%�8R��4I       6%�	�:�I���A�*;


total_loss��A

error_RmT?

learning_rate_1�%�8X`hqI       6%�	��I���A�*;


total_lossov�@

error_RS�H?

learning_rate_1�%�8!S�I       6%�	 ��I���A�*;


total_loss\�@

error_Rl�I?

learning_rate_1�%�8/��OI       6%�	�I���A�*;


total_loss�� A

error_R��E?

learning_rate_1�%�8"���I       6%�	�G�I���A�*;


total_lossH3�@

error_R��W?

learning_rate_1�%�8�2>�I       6%�	 ��I���A�*;


total_loss���@

error_RDA?

learning_rate_1�%�8�	s�I       6%�	���I���A�*;


total_loss0�@

error_RU7?

learning_rate_1�%�8EK��I       6%�	��I���A�*;


total_lossV�@

error_R�Q?

learning_rate_1�%�8I��GI       6%�	_�I���A�*;


total_lossZ�@

error_RasU?

learning_rate_1�%�8���I       6%�	���I���A�*;


total_loss?��@

error_R��I?

learning_rate_1�%�8h��I       6%�	��I���A�*;


total_loss���@

error_RW�P?

learning_rate_1�%�8�T�I       6%�	a,�I���A�*;


total_loss�a@

error_RWV?

learning_rate_1�%�8��NNI       6%�	#q�I���A�*;


total_loss��w@

error_R�d\?

learning_rate_1�%�8go�I       6%�	z��I���A�*;


total_loss���@

error_R��k?

learning_rate_1�%�8%��I       6%�	�+�I���A�*;


total_loss�A

error_R�G?

learning_rate_1�%�89I       6%�	xr�I���A�*;


total_loss%��@

error_R �V?

learning_rate_1�%�8vc��I       6%�	��I���A�*;


total_loss�@

error_R��G?

learning_rate_1�%�8�/�I       6%�	���I���A�*;


total_lossȶ@

error_RC�Y?

learning_rate_1�%�8q��'I       6%�	�@�I���A�*;


total_lossD֏@

error_R��^?

learning_rate_1�%�8G�+]I       6%�	��I���A�*;


total_lossW�@

error_RWBC?

learning_rate_1�%�8�AI       6%�	���I���A�*;


total_loss�˻@

error_RP:?

learning_rate_1�%�8�ܕ�I       6%�	��I���A�*;


total_loss���@

error_R�EK?

learning_rate_1�%�8����I       6%�	mU�I���A�*;


total_lossi��@

error_Ri�W?

learning_rate_1�%�8?��I       6%�	���I���A�*;


total_loss��A

error_R�}8?

learning_rate_1�%�8�%aI       6%�	Z��I���A�*;


total_loss�A

error_RNK?

learning_rate_1�%�8G���I       6%�	|,�I���A�*;


total_loss�%A

error_R�SJ?

learning_rate_1�%�8D�*:I       6%�	$v�I���A�*;


total_loss?��@

error_R=c?

learning_rate_1�%�8o`��I       6%�	��I���A�*;


total_loss��@

error_R�B?

learning_rate_1�%�8��-^I       6%�	��I���A�*;


total_loss[�@

error_Rs�C?

learning_rate_1�%�8�}I       6%�	oB�I���A�*;


total_loss(�@

error_R2�V?

learning_rate_1�%�8`3�I       6%�	���I���A�*;


total_loss��@

error_R�cI?

learning_rate_1�%�8�Q3I       6%�	��I���A�*;


total_loss��@

error_R�JU?

learning_rate_1�%�8>�aI       6%�	��I���A�*;


total_loss&c�@

error_RD?

learning_rate_1�%�8�
��I       6%�	IS�I���A�*;


total_lossL9�@

error_Rn2^?

learning_rate_1�%�8qe*I       6%�	���I���A�*;


total_loss��@

error_RLld?

learning_rate_1�%�8��;I       6%�	C��I���A�*;


total_lossH��@

error_R�L?

learning_rate_1�%�8%PI       6%�	��I���A�*;


total_loss���@

error_R��Y?

learning_rate_1�%�8����I       6%�	@a�I���A�*;


total_lossq6�@

error_R�]?

learning_rate_1�%�8<Q��I       6%�	o��I���A�*;


total_loss=��@

error_RM4;?

learning_rate_1�%�8n��I       6%�	d��I���A�*;


total_lossa��@

error_R�RO?

learning_rate_1�%�8��>�I       6%�	*�I���A�*;


total_loss�܄@

error_R�bb?

learning_rate_1�%�8�2[&I       6%�	6l�I���A�*;


total_loss�A

error_RF�U?

learning_rate_1�%�8,!9I       6%�	���I���A�*;


total_loss2)�@

error_Rl-_?

learning_rate_1�%�8o �[I       6%�	a��I���A�*;


total_loss�õ@

error_R��U?

learning_rate_1�%�8����I       6%�	�8�I���A�*;


total_loss���@

error_R��`?

learning_rate_1�%�8�Y�I       6%�	i|�I���A�*;


total_loss�I�@

error_R�U6?

learning_rate_1�%�8����I       6%�	��I���A�*;


total_loss�[�@

error_R�WY?

learning_rate_1�%�8-�wI       6%�	��I���A�*;


total_loss&�@

error_R�T?

learning_rate_1�%�8�a�I       6%�	�I�I���A�*;


total_loss�^@

error_R�??

learning_rate_1�%�8�qCRI       6%�	���I���A�*;


total_loss�̓@

error_R$$O?

learning_rate_1�%�8x��I       6%�	���I���A�*;


total_loss,à@

error_R�6?

learning_rate_1�%�8nm��I       6%�	��I���A�*;


total_loss�#A

error_R��M?

learning_rate_1�%�8�U�I       6%�	�_�I���A�*;


total_loss���@

error_RJ?

learning_rate_1�%�8D_5�I       6%�	֦�I���A�*;


total_lossHT�@

error_R&�T?

learning_rate_1�%�8�j��I       6%�	���I���A�*;


total_loss�ɦ@

error_R�=]?

learning_rate_1�%�8�e�I       6%�	�.�I���A�*;


total_lossC�@

error_RxLI?

learning_rate_1�%�8��B�I       6%�	�t�I���A�*;


total_loss�T�@

error_Ra�_?

learning_rate_1�%�8a��I       6%�	}��I���A�*;


total_loss�\�@

error_R�BZ?

learning_rate_1�%�8�2b�I       6%�	-�I���A�*;


total_loss�K�@

error_R:�:?

learning_rate_1�%�8�]�I       6%�	L`�I���A�*;


total_lossD|�@

error_R,�b?

learning_rate_1�%�8��DaI       6%�	Z��I���A�*;


total_loss��@

error_R
�W?

learning_rate_1�%�8A2�;I       6%�	�I���A�*;


total_loss��@

error_R �p?

learning_rate_1�%�8�]�BI       6%�	W�I���A�*;


total_loss��@

error_R�5[?

learning_rate_1�%�8��I       6%�	p��I���A�*;


total_loss���@

error_R�g?

learning_rate_1�%�8���*I       6%�	���I���A�*;


total_loss���@

error_R�NK?

learning_rate_1�%�8o�|qI       6%�	6�I���A�*;


total_loss;Ԭ@

error_R�VD?

learning_rate_1�%�8ȂwI       6%�	~|�I���A�*;


total_loss*¬@

error_RIWb?

learning_rate_1�%�8x��TI       6%�	p��I���A�*;


total_loss���@

error_Rv�J?

learning_rate_1�%�8��:�I       6%�	3�I���A�*;


total_lossk�@

error_R
b`?

learning_rate_1�%�8���I       6%�	�W�I���A�*;


total_loss���@

error_RE�]?

learning_rate_1�%�8B���I       6%�	���I���A�*;


total_loss:�A

error_R�_?

learning_rate_1�%�8+4k�I       6%�	-�I���A�*;


total_losss��@

error_Rx�L?

learning_rate_1�%�8?�KI       6%�	�c�I���A�*;


total_lossҶ�@

error_R�D?

learning_rate_1�%�8Q��I       6%�	/��I���A�*;


total_loss!��@

error_R�<I?

learning_rate_1�%�8�`�
I       6%�	���I���A�*;


total_loss<9�@

error_R�N?

learning_rate_1�%�8$F*VI       6%�	�.�I���A�*;


total_loss��@

error_Rf<^?

learning_rate_1�%�8�I       6%�	Ur�I���A�*;


total_loss��@

error_Rا[?

learning_rate_1�%�8:ΧDI       6%�	!��I���A�*;


total_lossW�@

error_RxL?

learning_rate_1�%�8�L�I       6%�	$��I���A�*;


total_loss��@

error_RT&N?

learning_rate_1�%�8�C�I       6%�	A�I���A�*;


total_loss`.�@

error_R)>X?

learning_rate_1�%�8M(I       6%�	ڃ�I���A�*;


total_loss�0�@

error_R_}[?

learning_rate_1�%�8�"'�I       6%�	���I���A�*;


total_loss�/�@

error_RC�S?

learning_rate_1�%�8�L�I       6%�	��I���A�*;


total_loss*�@

error_R�KY?

learning_rate_1�%�8k�l�I       6%�		N�I���A�*;


total_loss;��@

error_R�A?

learning_rate_1�%�8��?<I       6%�	%��I���A�*;


total_loss(q�@

error_R]�R?

learning_rate_1�%�8)�MaI       6%�	#��I���A�*;


total_loss���@

error_R*n7?

learning_rate_1�%�8�N)9I       6%�	��I���A�*;


total_loss�H�@

error_R�[G?

learning_rate_1�%�8�̺�I       6%�	�g�I���A�*;


total_loss](�@

error_R
AO?

learning_rate_1�%�8dd)I       6%�	ӫ�I���A�*;


total_loss`wA

error_R��M?

learning_rate_1�%�8�V��I       6%�	Z��I���A�*;


total_loss�`�@

error_RAR?

learning_rate_1�%�8nLaI       6%�	M4�I���A�*;


total_loss��@

error_R,@;?

learning_rate_1�%�8n�EI       6%�	\}�I���A�*;


total_lossr�@

error_RJ�]?

learning_rate_1�%�81A:I       6%�	���I���A�*;


total_loss�L�@

error_R?5M?

learning_rate_1�%�8xI       6%�	��I���A�*;


total_loss<X�@

error_R��F?

learning_rate_1�%�8�'�I       6%�	>Z�I���A�*;


total_loss_��@

error_R6'I?

learning_rate_1�%�8-J-I       6%�	ˤ�I���A�*;


total_loss �@

error_Ro>a?

learning_rate_1�%�8�-�(I       6%�	J��I���A�*;


total_loss<�@

error_R6HU?

learning_rate_1�%�8Sw{wI       6%�	x7�I���A�*;


total_loss��@

error_R,�??

learning_rate_1�%�8�P\jI       6%�	'��I���A�*;


total_losstH�@

error_R��\?

learning_rate_1�%�8c4�I       6%�	���I���A�*;


total_loss�I�@

error_R�0Y?

learning_rate_1�%�8\Q��I       6%�	��I���A�*;


total_loss�@

error_R{;M?

learning_rate_1�%�8�BI       6%�	�X�I���A�*;


total_loss��@

error_R�>?

learning_rate_1�%�8X2I       6%�	���I���A�*;


total_loss{ A

error_R2�X?

learning_rate_1�%�8L�jI       6%�	���I���A�*;


total_lossq*�@

error_R��I?

learning_rate_1�%�8��&�I       6%�	!(�I���A�*;


total_loss|޴@

error_R/�I?

learning_rate_1�%�8<�B~I       6%�	,n�I���A�*;


total_loss)o�@

error_R��[?

learning_rate_1�%�8Rs�I       6%�	���I���A�*;


total_losst �@

error_RMn\?

learning_rate_1�%�8���I       6%�	���I���A�*;


total_loss`�@

error_R�qM?

learning_rate_1�%�8Eh�BI       6%�	�<�I���A�*;


total_loss!x�@

error_R�U?

learning_rate_1�%�8t�r�I       6%�	��I���A�*;


total_loss�
�@

error_R*Q?

learning_rate_1�%�8��NI       6%�	���I���A�*;


total_lossd�@

error_R��[?

learning_rate_1�%�8 ���I       6%�	7�I���A�*;


total_loss�Un@

error_R�ec?

learning_rate_1�%�8a|��I       6%�	ZU�I���A�*;


total_lossFP�@

error_R�SD?

learning_rate_1�%�8L"?�I       6%�	)��I���A�*;


total_loss��@

error_R�A?

learning_rate_1�%�8�ZvI       6%�	��I���A�*;


total_lossV��@

error_R�nZ?

learning_rate_1�%�8U�lEI       6%�	�#�I���A�*;


total_lossWu�@

error_R�XE?

learning_rate_1�%�8����I       6%�	k�I���A�*;


total_loss��@

error_R�X;?

learning_rate_1�%�8�|R�I       6%�	K��I���A�*;


total_loss���@

error_RjZ?

learning_rate_1�%�8To�oI       6%�	���I���A�*;


total_loss�	�@

error_R��Q?

learning_rate_1�%�8����I       6%�	�=�I���A�*;


total_loss�a�@

error_RZ^R?

learning_rate_1�%�8�W�I       6%�	��I���A�*;


total_loss�km@

error_R�sR?

learning_rate_1�%�8"y$�I       6%�	���I���A�*;


total_loss�
�@

error_R�aK?

learning_rate_1�%�8 �>�I       6%�	�I���A�*;


total_loss�A

error_R��Y?

learning_rate_1�%�8)�9=I       6%�	�I�I���A�*;


total_lossZt�@

error_R=�J?

learning_rate_1�%�8ܕ�I       6%�	ƌ�I���A�*;


total_loss�o�@

error_R1�W?

learning_rate_1�%�8�b�I       6%�	X��I���A�*;


total_lossh�@

error_R�P?

learning_rate_1�%�8Pb֧I       6%�	��I���A�*;


total_loss���@

error_Rw�\?

learning_rate_1�%�8��I       6%�	�W�I���A�*;


total_loss�ޗ@

error_R$Q?

learning_rate_1�%�89��I       6%�	���I���A�*;


total_loss6�@

error_RO�I?

learning_rate_1�%�8%��I       6%�	1�I���A�*;


total_loss��@

error_R��F?

learning_rate_1�%�8�GԅI       6%�	T�I���A�*;


total_lossXP�@

error_R��S?

learning_rate_1�%�8�Z�*I       6%�	���I���A�*;


total_loss!�@

error_Rf�I?

learning_rate_1�%�8��I       6%�	{��I���A�*;


total_loss�ģ@

error_R)3Z?

learning_rate_1�%�8����I       6%�	�*�I���A�*;


total_loss_L�@

error_R.�J?

learning_rate_1�%�8��I       6%�	�w�I���A�*;


total_loss�A

error_R�*F?

learning_rate_1�%�8�$(�I       6%�	���I���A�*;


total_loss?��@

error_R��P?

learning_rate_1�%�8g���I       6%�	Q�I���A�*;


total_loss�п@

error_R1>[?

learning_rate_1�%�8pW��I       6%�	[I�I���A�*;


total_losso��@

error_Rl�S?

learning_rate_1�%�8t��I       6%�	��I���A�*;


total_loss���@

error_R�WF?

learning_rate_1�%�8D�	CI       6%�	���I���A�*;


total_loss�V�@

error_R��e?

learning_rate_1�%�8��c�I       6%�	N�I���A�*;


total_losso��@

error_R�??

learning_rate_1�%�8��dI       6%�	\�I���A�*;


total_loss,��@

error_R8FF?

learning_rate_1�%�8���I       6%�	���I���A�*;


total_loss���@

error_R�\?

learning_rate_1�%�8O89I       6%�	
��I���A�*;


total_loss�H�@

error_R�{P?

learning_rate_1�%�8'w$�I       6%�	)�I���A�*;


total_loss#�A

error_R��Q?

learning_rate_1�%�8���CI       6%�	�j�I���A�*;


total_lossAc�@

error_Ry_?

learning_rate_1�%�8�fe�I       6%�	���I���A�*;


total_loss��@

error_R��:?

learning_rate_1�%�8���I       6%�	.��I���A�*;


total_loss$r�@

error_R��Q?

learning_rate_1�%�8�v�I       6%�	�5�I���A�*;


total_loss��@

error_Rs�]?

learning_rate_1�%�8]!a[I       6%�	 x�I���A�*;


total_loss_�A

error_R�Y?

learning_rate_1�%�8�<��I       6%�	���I���A�*;


total_loss�<	A

error_RJ?

learning_rate_1�%�8q���I       6%�	�I���A�*;


total_lossHڭ@

error_RFO?

learning_rate_1�%�8��u�I       6%�	�F�I���A�*;


total_loss��@

error_R�T`?

learning_rate_1�%�8���I       6%�	f��I���A�*;


total_loss1��@

error_RM�H?

learning_rate_1�%�8��s�I       6%�	���I���A�*;


total_loss/��@

error_R��D?

learning_rate_1�%�8Y��rI       6%�	��I���A�*;


total_loss&��@

error_R!eQ?

learning_rate_1�%�8f�!�I       6%�	{]�I���A�*;


total_loss��@

error_R߾G?

learning_rate_1�%�8C�bI       6%�	���I���A�*;


total_lossxÛ@

error_R��V?

learning_rate_1�%�8>��I       6%�	G��I���A�*;


total_loss_G�@

error_RR�I?

learning_rate_1�%�8x�1I       6%�	';�I���A�*;


total_loss��@

error_R�nK?

learning_rate_1�%�8<}B�I       6%�	��I���A�*;


total_loss�CA

error_R��B?

learning_rate_1�%�8�I�dI       6%�	���I���A�*;


total_loss;	{@

error_R��I?

learning_rate_1�%�8���JI       6%�	w�I���A�*;


total_lossh��@

error_R�Z?

learning_rate_1�%�8�ȉHI       6%�	�g�I���A�*;


total_loss��@

error_RR?

learning_rate_1�%�8�E��I       6%�	3��I���A�*;


total_loss�Mc@

error_R��S?

learning_rate_1�%�8�v�I       6%�	���I���A�*;


total_loss8W�@

error_R�??

learning_rate_1�%�8v��wI       6%�	�@�I���A�*;


total_lossfm�@

error_RbF?

learning_rate_1�%�8S��=I       6%�	��I���A�*;


total_loss}��@

error_R��N?

learning_rate_1�%�8�<��I       6%�	���I���A�*;


total_loss�:�@

error_R�G?

learning_rate_1�%�8<�&�I       6%�	K)�I���A�*;


total_lossQ��@

error_R�K?

learning_rate_1�%�8ŽVI       6%�	}r�I���A�*;


total_loss;��@

error_R�M?

learning_rate_1�%�8<<l3I       6%�	��I���A�*;


total_loss���@

error_Rȩ_?

learning_rate_1�%�8�EBI       6%�	y�I���A�*;


total_loss��@

error_R,,V?

learning_rate_1�%�8�2%�I       6%�	�W�I���A�*;


total_loss�ȥ@

error_R#+I?

learning_rate_1�%�8���I       6%�	|��I���A�*;


total_loss\�@

error_R�T?

learning_rate_1�%�8��zI       6%�	���I���A�*;


total_loss�C�@

error_Rj�K?

learning_rate_1�%�8�T�HI       6%�	)�I���A�*;


total_lossd��@

error_R��T?

learning_rate_1�%�8[ȢI       6%�	^p�I���A�*;


total_lossq��@

error_Rv�P?

learning_rate_1�%�8����I       6%�	ƴ�I���A�*;


total_loss���@

error_R�a?

learning_rate_1�%�8����I       6%�	��I���A�*;


total_lossD�@

error_R��N?

learning_rate_1�%�8s��I       6%�	<�I���A�*;


total_loss1�@

error_R=�L?

learning_rate_1�%�8���xI       6%�	 ~�I���A�*;


total_loss�K�@

error_R8�X?

learning_rate_1�%�8uJwI       6%�	��I���A�*;


total_loss\��@

error_R��@?

learning_rate_1�%�8m@�;I       6%�	��I���A�*;


total_loss.A

error_R�	Q?

learning_rate_1�%�8�vI       6%�	�D�I���A�*;


total_loss�h�@

error_Ra�E?

learning_rate_1�%�8-r��I       6%�	���I���A�*;


total_loss�gi@

error_R�wE?

learning_rate_1�%�8yvɂI       6%�	`��I���A�*;


total_loss>A

error_R}wH?

learning_rate_1�%�8��9I       6%�	{)�I���A�*;


total_loss�A

error_Rf�L?

learning_rate_1�%�8d��GI       6%�	�o�I���A�*;


total_loss��@

error_R)eW?

learning_rate_1�%�8�=I       6%�	��I���A�*;


total_loss�G�@

error_R8�T?

learning_rate_1�%�8��q+I       6%�	�I���A�*;


total_loss7~�@

error_R�m\?

learning_rate_1�%�8]��I       6%�	@J�I���A�*;


total_lossj�@

error_Rv�W?

learning_rate_1�%�84ۓ�I       6%�	=��I���A�*;


total_lossaڢ@

error_R��A?

learning_rate_1�%�8u}#I       6%�	���I���A�*;


total_loss_��@

error_R��G?

learning_rate_1�%�8䯃�I       6%�	��I���A�*;


total_loss�'�@

error_R��K?

learning_rate_1�%�8��d�I       6%�	\�I���A�*;


total_loss���@

error_R��T?

learning_rate_1�%�8���LI       6%�	
��I���A�*;


total_loss�ô@

error_R��R?

learning_rate_1�%�8�v�I       6%�	��I���A�*;


total_loss��@

error_R�xM?

learning_rate_1�%�8��,~I       6%�	�(�I���A�*;


total_loss���@

error_R�Q?

learning_rate_1�%�8pQ�I       6%�	�m�I���A�*;


total_lossWE�@

error_RπS?

learning_rate_1�%�8
�X�I       6%�	@��I���A�*;


total_loss�>�@

error_R��Q?

learning_rate_1�%�8��B-I       6%�	���I���A�*;


total_lossX��@

error_R��I?

learning_rate_1�%�8q5�
I       6%�	n5�I���A�*;


total_loss3�@

error_R��U?

learning_rate_1�%�8�MΚI       6%�	�z�I���A�*;


total_lossD��@

error_Rq�X?

learning_rate_1�%�8Z�#-I       6%�	v��I���A�*;


total_loss��@

error_R��T?

learning_rate_1�%�8��7{I       6%�	��I���A�*;


total_loss ý@

error_RxU?

learning_rate_1�%�8@an}I       6%�	T�I���A�*;


total_loss�@

error_R�M[?

learning_rate_1�%�8��CI       6%�	\��I���A�*;


total_loss�?�@

error_R��K?

learning_rate_1�%�8��r�I       6%�	��I���A�*;


total_loss�p�@

error_RK?

learning_rate_1�%�8�EVWI       6%�	- �I���A�*;


total_loss�A

error_R��E?

learning_rate_1�%�8�%��I       6%�	bc�I���A�*;


total_lossȭ@

error_R�{L?

learning_rate_1�%�8թ��I       6%�	���I���A�*;


total_loss7�@

error_R�I?

learning_rate_1�%�8��FI       6%�	���I���A�*;


total_lossm�@

error_R�a^?

learning_rate_1�%�8�6�I       6%�	0 J���A�*;


total_loss1S�@

error_R�:?

learning_rate_1�%�8��V�I       6%�	�r J���A�*;


total_loss�{�@

error_R��K?

learning_rate_1�%�8���I       6%�	W� J���A�*;


total_lossRa�@

error_R��V?

learning_rate_1�%�8_4t\I       6%�	�� J���A�*;


total_loss�rA

error_R��Z?

learning_rate_1�%�8�e�'I       6%�	�=J���A�*;


total_loss���@

error_R�P?

learning_rate_1�%�8ۃ}WI       6%�	�J���A�*;


total_loss���@

error_RiH?

learning_rate_1�%�8�/�I       6%�	N�J���A�*;


total_loss|JA

error_R
Q?

learning_rate_1�%�8|�`I       6%�	�J���A�*;


total_loss2q@

error_R[�H?

learning_rate_1�%�8�I       6%�	�TJ���A�*;


total_loss�|A

error_R��Y?

learning_rate_1�%�8�<�bI       6%�	U�J���A�*;


total_loss�@

error_R�K?

learning_rate_1�%�8}K��I       6%�	��J���A�*;


total_loss�s�@

error_R:U?

learning_rate_1�%�8c�"I       6%�	�1J���A�*;


total_lossh��@

error_R��W?

learning_rate_1�%�8<�I       6%�	6{J���A�*;


total_lossiW�@

error_RτU?

learning_rate_1�%�8�S�vI       6%�	y�J���A�*;


total_lossҸ�@

error_R��@?

learning_rate_1�%�8@���I       6%�	J���A�*;


total_loss��@

error_R�A?

learning_rate_1�%�8�	��I       6%�	qFJ���A�*;


total_loss���@

error_R.lk?

learning_rate_1�%�8��)I       6%�	�J���A�*;


total_loss��@

error_R��`?

learning_rate_1�%�8YO�.I       6%�	@�J���A�*;


total_lossj��@

error_RWH\?

learning_rate_1�%�8���I       6%�	hJ���A�*;


total_loss�"�@

error_R�zY?

learning_rate_1�%�8B��I       6%�	�SJ���A�*;


total_loss���@

error_R��_?

learning_rate_1�%�8���I       6%�	��J���A�*;


total_lossO�@

error_R�5I?

learning_rate_1�%�8e��I       6%�	��J���A�*;


total_loss�\�@

error_Rs�H?

learning_rate_1�%�8���I       6%�	�J���A�*;


total_loss*��@

error_R�.M?

learning_rate_1�%�8�':I       6%�	~`J���A�*;


total_loss�1�@

error_R�U?

learning_rate_1�%�8q�I       6%�	ΥJ���A�*;


total_loss���@

error_R��D?

learning_rate_1�%�8�b��I       6%�	�J���A�*;


total_loss��@

error_Rf�P?

learning_rate_1�%�8�1$�I       6%�	</J���A�*;


total_loss|�@

error_R�=P?

learning_rate_1�%�8@��I       6%�	tJ���A�*;


total_loss�/�@

error_R�b?

learning_rate_1�%�8��I       6%�	r�J���A�*;


total_lossfxz@

error_Rd�F?

learning_rate_1�%�8�uk�I       6%�	�	J���A�*;


total_loss��@

error_RZ�J?

learning_rate_1�%�8�1�1I       6%�	�QJ���A�*;


total_loss��@

error_R'R?

learning_rate_1�%�8�ݗXI       6%�	�J���A�*;


total_loss]4)A

error_R�XW?

learning_rate_1�%�8Kٳ�I       6%�	&	J���A�*;


total_lossO��@

error_RqH?

learning_rate_1�%�8�L��I       6%�	=_	J���A�*;


total_loss��@

error_RC�[?

learning_rate_1�%�8U?I       6%�	ڧ	J���A�*;


total_loss�B�@

error_RJB=?

learning_rate_1�%�8�iWAI       6%�	��	J���A�*;


total_loss�:A

error_R��H?

learning_rate_1�%�89��I       6%�	�7
J���A�*;


total_lossij@

error_R�@Q?

learning_rate_1�%�8�iI       6%�	t
J���A�*;


total_loss��@

error_REcS?

learning_rate_1�%�8����I       6%�	��
J���A�*;


total_loss�-�@

error_R�:_?

learning_rate_1�%�8�9I       6%�	WJ���A�*;


total_loss:j�@

error_R2�=?

learning_rate_1�%�8��,�I       6%�	wLJ���A�*;


total_loss�7�@

error_R)�N?

learning_rate_1�%�8��2I       6%�	ڏJ���A�*;


total_loss��@

error_RJ�U?

learning_rate_1�%�80��I       6%�	��J���A�*;


total_loss�ƥ@

error_R݊`?

learning_rate_1�%�8j�pI       6%�	) J���A�*;


total_loss#̀@

error_R�B?

learning_rate_1�%�8C�J�I       6%�	gJ���A�*;


total_loss(��@

error_RܵK?

learning_rate_1�%�8Cԕ�I       6%�	��J���A�*;


total_lossw��@

error_R��T?

learning_rate_1�%�8NK�}I       6%�	��J���A�*;


total_loss���@

error_R�M?

learning_rate_1�%�8(���I       6%�	R3J���A�*;


total_lossc�A

error_R�O?

learning_rate_1�%�8Jp�I       6%�	�uJ���A�*;


total_loss���@

error_R)�Y?

learning_rate_1�%�8�q%5I       6%�	��J���A�*;


total_lossu��@

error_Rv�R?

learning_rate_1�%�8\�'I       6%�	�J���A�*;


total_loss,2�@

error_R��I?

learning_rate_1�%�8@+��I       6%�	LJ���A�*;


total_loss��@

error_R�:X?

learning_rate_1�%�8M��I       6%�	��J���A�*;


total_loss;ſ@

error_R��Y?

learning_rate_1�%�8=�<�I       6%�	��J���A�*;


total_loss.��@

error_R_�I?

learning_rate_1�%�8��WI       6%�	�J���A�*;


total_loss��A

error_R$�3?

learning_rate_1�%�8MI.I       6%�	gJ���A�*;


total_loss��@

error_R��4?

learning_rate_1�%�8��z�I       6%�	��J���A�*;


total_loss�@�@

error_Rz?Q?

learning_rate_1�%�8'�I       6%�	(�J���A�*;


total_loss�@

error_R�yh?

learning_rate_1�%�8��hI       6%�	�.J���A�*;


total_loss��@

error_Rq�Y?

learning_rate_1�%�8|��7I       6%�	�zJ���A�*;


total_loss/�@

error_R�IU?

learning_rate_1�%�8+UzI       6%�	��J���A�*;


total_loss�]�@

error_R.�G?

learning_rate_1�%�8���1I       6%�	J���A�*;


total_loss�қ@

error_Rv�I?

learning_rate_1�%�8',w�I       6%�	`KJ���A�*;


total_loss��@

error_R�o@?

learning_rate_1�%�8	j9I       6%�	��J���A�*;


total_loss��@

error_R��I?

learning_rate_1�%�8�~��I       6%�	I�J���A�*;


total_loss4G�@

error_R��\?

learning_rate_1�%�8ayJ�I       6%�	�$J���A�*;


total_loss&��@

error_R ;B?

learning_rate_1�%�8�Դ9I       6%�	JiJ���A�*;


total_loss��@

error_R�cK?

learning_rate_1�%�8�a(nI       6%�	8�J���A�*;


total_loss���@

error_R\�N?

learning_rate_1�%�8 ڛ5I       6%�	 �J���A�*;


total_lossA��@

error_R�RV?

learning_rate_1�%�8��lI       6%�	 6J���A�*;


total_loss��A

error_R��K?

learning_rate_1�%�8�Y�fI       6%�	�yJ���A�*;


total_loss�]�@

error_R��N?

learning_rate_1�%�8v���I       6%�	��J���A�*;


total_loss.�@

error_R�#:?

learning_rate_1�%�8��YnI       6%�	TJ���A�*;


total_loss�@

error_R8sM?

learning_rate_1�%�8�5�I       6%�	�LJ���A�*;


total_lossY<�@

error_R�P?

learning_rate_1�%�8��I       6%�	ۜJ���A�*;


total_loss�؀@

error_R8L?

learning_rate_1�%�8�eD@I       6%�	��J���A�*;


total_lossɴA

error_R��G?

learning_rate_1�%�8��I       6%�	F?J���A�*;


total_loss{J�@

error_R�7?

learning_rate_1�%�8���I       6%�	��J���A�*;


total_loss8 �@

error_R}0K?

learning_rate_1�%�8vug=I       6%�	z�J���A�*;


total_loss��l@

error_R��W?

learning_rate_1�%�8��V|I       6%�	E=J���A�*;


total_loss�G�@

error_R��S?

learning_rate_1�%�8�e�II       6%�	w�J���A�*;


total_loss�'�@

error_R NO?

learning_rate_1�%�8qm�I       6%�	��J���A�*;


total_loss��@

error_Ri�H?

learning_rate_1�%�8��I       6%�	�#J���A�*;


total_losse��@

error_RabT?

learning_rate_1�%�8¬J�I       6%�	!hJ���A�*;


total_loss�tA

error_R�O?

learning_rate_1�%�8W'�I       6%�	)�J���A�*;


total_loss醩@

error_R��B?

learning_rate_1�%�8bT�I       6%�	H�J���A�*;


total_loss���@

error_R�'Q?

learning_rate_1�%�8 W�I       6%�	(8J���A�*;


total_lossr=�@

error_R��R?

learning_rate_1�%�8m51eI       6%�	�|J���A�*;


total_lossXA

error_R��V?

learning_rate_1�%�8��I       6%�	��J���A�*;


total_loss�J�@

error_R��S?

learning_rate_1�%�8{��8I       6%�	/1J���A�*;


total_loss*�@

error_R��R?

learning_rate_1�%�8�W�DI       6%�	�xJ���A�*;


total_loss^�A

error_R�#L?

learning_rate_1�%�8�4�iI       6%�	�J���A�*;


total_loss��@

error_R|:U?

learning_rate_1�%�8���}I       6%�	kJ���A�*;


total_lossA�@

error_R͟^?

learning_rate_1�%�8T�_�I       6%�	YFJ���A�*;


total_loss{��@

error_R��W?

learning_rate_1�%�8�E��I       6%�	7�J���A�*;


total_loss�5�@

error_RZ�N?

learning_rate_1�%�8��f�I       6%�	��J���A�*;


total_loss��@

error_R��E?

learning_rate_1�%�8 C�I       6%�	�J���A�*;


total_lossrѪ@

error_R;U?

learning_rate_1�%�8T���I       6%�	�XJ���A�*;


total_loss�җ@

error_R\�I?

learning_rate_1�%�8�w��I       6%�	�J���A�*;


total_loss�u�@

error_R��<?

learning_rate_1�%�8��rI       6%�	��J���A�*;


total_lossay�@

error_R��P?

learning_rate_1�%�8���I       6%�	�%J���A�*;


total_lossN@

error_R��B?

learning_rate_1�%�8��vcI       6%�	�jJ���A�*;


total_loss���@

error_Rɏ]?

learning_rate_1�%�8����I       6%�	��J���A�*;


total_loss/�@

error_R�T?

learning_rate_1�%�8\�oI       6%�	�J���A�*;


total_loss��A

error_R�I?

learning_rate_1�%�8��$I       6%�	�?J���A�*;


total_loss���@

error_R��\?

learning_rate_1�%�8F� $I       6%�	��J���A�*;


total_loss̴A

error_R�U?

learning_rate_1�%�8uf�BI       6%�	3�J���A�*;


total_lossrd�@

error_RwOG?

learning_rate_1�%�8d��I       6%�	�J���A�*;


total_loss�mm@

error_Ro1\?

learning_rate_1�%�8-I�I       6%�	*aJ���A�*;


total_loss�1�@

error_RA�I?

learning_rate_1�%�8X�uLI       6%�	�J���A�*;


total_loss���@

error_R�f?

learning_rate_1�%�8K�h�I       6%�	@�J���A�*;


total_loss�r�@

error_R(>9?

learning_rate_1�%�8S��sI       6%�	O,J���A�*;


total_lossE��@

error_RF�J?

learning_rate_1�%�8M.<I       6%�	EoJ���A�*;


total_loss��@

error_R��N?

learning_rate_1�%�8#��I       6%�	y�J���A�*;


total_lossX��@

error_Ra�S?

learning_rate_1�%�8�(�I       6%�	/�J���A�*;


total_loss���@

error_Ra�I?

learning_rate_1�%�8�Y�I       6%�	x? J���A�*;


total_lossMH�@

error_R��Y?

learning_rate_1�%�8�I       6%�	S� J���A�*;


total_loss��@

error_R��:?

learning_rate_1�%�8,���I       6%�	G� J���A�*;


total_loss���@

error_R�I\?

learning_rate_1�%�8�4�SI       6%�	!J���A�*;


total_losseE�@

error_R%�J?

learning_rate_1�%�8�EϭI       6%�	�T!J���A�*;


total_loss���@

error_R��R?

learning_rate_1�%�8��y?I       6%�	3�!J���A�*;


total_loss��@

error_R&�Q?

learning_rate_1�%�8H�I       6%�	��!J���A�*;


total_lossc�@

error_R;wM?

learning_rate_1�%�8PvI       6%�	+!"J���A�*;


total_loss|l�@

error_R_�M?

learning_rate_1�%�8�lTI       6%�	ef"J���A�*;


total_lossH�@

error_RW
S?

learning_rate_1�%�8��6�I       6%�	�"J���A�*;


total_loss�A�@

error_R�D?

learning_rate_1�%�8�B};I       6%�	5�"J���A�*;


total_loss;��@

error_R��M?

learning_rate_1�%�8/w�I       6%�	�3#J���A�*;


total_loss6�@

error_R�^H?

learning_rate_1�%�8%FRwI       6%�	#x#J���A�*;


total_loss���@

error_R��]?

learning_rate_1�%�8E�}\I       6%�	,�#J���A�*;


total_loss��@

error_R�R?

learning_rate_1�%�8Щ�I       6%�	�$J���A�*;


total_loss4��@

error_R�C?

learning_rate_1�%�82�I       6%�	�O$J���A�*;


total_loss��A

error_RA U?

learning_rate_1�%�8h�{NI       6%�	5�$J���A�*;


total_loss6��@

error_R@�d?

learning_rate_1�%�8TY�I       6%�	��$J���A�*;


total_loss%��@

error_R4�R?

learning_rate_1�%�8�^�I       6%�	$)%J���A�*;


total_loss��@

error_RL?

learning_rate_1�%�8�VڦI       6%�	:l%J���A�*;


total_loss�]�@

error_R�E?

learning_rate_1�%�8�i�I       6%�	-�%J���A�*;


total_lossȀA

error_R\DS?

learning_rate_1�%�81o5I       6%�	��%J���A�*;


total_lossxA

error_Ri�]?

learning_rate_1�%�83Y��I       6%�	�3&J���A�*;


total_lossQ�@

error_RTL?

learning_rate_1�%�8�MqI       6%�	�}&J���A�*;


total_lossJ@

error_R%oK?

learning_rate_1�%�8�5}+I       6%�	#�&J���A�*;


total_lossS�@

error_R��Q?

learning_rate_1�%�8�lI       6%�	�'J���A�*;


total_loss���@

error_R��R?

learning_rate_1�%�8�}�I       6%�	�W'J���A�*;


total_loss
�@

error_R�-I?

learning_rate_1�%�8��XXI       6%�	�'J���A�*;


total_loss�Z�@

error_Rn�J?

learning_rate_1�%�8���lI       6%�	��'J���A�*;


total_lossS�A

error_R�|`?

learning_rate_1�%�8�>�I       6%�	*(J���A�*;


total_loss�@

error_Rrj?

learning_rate_1�%�8�V��I       6%�	'k(J���A�*;


total_lossq��@

error_RlyZ?

learning_rate_1�%�8
��$I       6%�	��(J���A�*;


total_lossЭ@

error_R�EM?

learning_rate_1�%�8��o	I       6%�	|)J���A�*;


total_lossZw�@

error_R�{K?

learning_rate_1�%�8��*'I       6%�	-_)J���A�*;


total_lossD��@

error_R�]?

learning_rate_1�%�8�ejI       6%�	�)J���A�*;


total_loss��@

error_Rϫ??

learning_rate_1�%�8�a|�I       6%�	��)J���A�*;


total_loss��@

error_R�G?

learning_rate_1�%�8�M$�I       6%�	4+*J���A�*;


total_lossڲ�@

error_R��Q?

learning_rate_1�%�8WI       6%�	cq*J���A�*;


total_loss�W�@

error_R*mE?

learning_rate_1�%�8���I       6%�	��*J���A�*;


total_loss���@

error_RR�P?

learning_rate_1�%�8��C[I       6%�	�*J���A�*;


total_lossn��@

error_RimE?

learning_rate_1�%�8>�wI       6%�	�B+J���A�*;


total_loss�yA

error_R�hW?

learning_rate_1�%�8J=^I       6%�	�+J���A�*;


total_loss�@

error_RԆI?

learning_rate_1�%�8�QA�I       6%�	��+J���A�*;


total_loss(w�@

error_R�CP?

learning_rate_1�%�8o�ڛI       6%�	/,J���A�*;


total_loss�.�@

error_R�b[?

learning_rate_1�%�8�t�LI       6%�	Y,J���A�*;


total_lossF��@

error_RO�S?

learning_rate_1�%�8P]�I       6%�	��,J���A�*;


total_lossvx@

error_R)oP?

learning_rate_1�%�8$p�I       6%�	]�,J���A�*;


total_loss-��@

error_R%�I?

learning_rate_1�%�8�l8�I       6%�	�$-J���A�*;


total_loss�@

error_R_Q?

learning_rate_1�%�8�g�I       6%�	�i-J���A�*;


total_loss�<�@

error_R��E?

learning_rate_1�%�8�`ŗI       6%�	��-J���A�*;


total_lossQ�@

error_R
�;?

learning_rate_1�%�8tS'I       6%�	��-J���A�*;


total_loss�3A

error_RE.L?

learning_rate_1�%�8c��gI       6%�	K>.J���A�*;


total_loss̦�@

error_R/}J?

learning_rate_1�%�8h���I       6%�	Y�.J���A�*;


total_loss�­@

error_R�V?

learning_rate_1�%�8��ҟI       6%�	n�.J���A�*;


total_loss���@

error_Rt�h?

learning_rate_1�%�8�[_,I       6%�	G	/J���A�*;


total_loss[:�@

error_R��W?

learning_rate_1�%�8Iq�I       6%�	Q/J���A�*;


total_loss�F�@

error_R�P?

learning_rate_1�%�83���I       6%�	N�/J���A�*;


total_loss֛�@

error_Rd�K?

learning_rate_1�%�8LyI       6%�	��/J���A�*;


total_lossZY�@

error_R �a?

learning_rate_1�%�8��I       6%�	S0J���A�*;


total_loss��@

error_RaQB?

learning_rate_1�%�8�%d�I       6%�	_0J���A�*;


total_loss��@

error_R�F?

learning_rate_1�%�8��CPI       6%�	��0J���A�*;


total_loss���@

error_RN�S?

learning_rate_1�%�8�B^I       6%�	��0J���A�*;


total_loss���@

error_R$eL?

learning_rate_1�%�8$�E�I       6%�	x&1J���A�*;


total_loss@͎@

error_R�QC?

learning_rate_1�%�8n�_zI       6%�	'i1J���A�*;


total_loss�b�@

error_R1�X?

learning_rate_1�%�8NIZ�I       6%�	�1J���A�*;


total_loss�� A

error_R�X?

learning_rate_1�%�8�HݘI       6%�	M�1J���A�*;


total_loss�Z�@

error_R��U?

learning_rate_1�%�8t�G�I       6%�	s12J���A�*;


total_loss��@

error_RW�T?

learning_rate_1�%�8T-��I       6%�	�t2J���A�*;


total_lossD>�@

error_RT2<?

learning_rate_1�%�8��I       6%�	k�2J���A�*;


total_lossAW�@

error_Ro�O?

learning_rate_1�%�8��ĨI       6%�	�3J���A�*;


total_lossI��@

error_R�5Q?

learning_rate_1�%�8Ev݃I       6%�	mH3J���A�*;


total_loss:|y@

error_R*�C?

learning_rate_1�%�8���I       6%�	�3J���A�*;


total_loss���@

error_R�C?

learning_rate_1�%�8/1�I       6%�	��3J���A�*;


total_loss��@

error_R_CQ?

learning_rate_1�%�8�J��I       6%�	�4J���A�*;


total_loss@�@

error_R�)L?

learning_rate_1�%�8`��xI       6%�	�h4J���A�*;


total_loss��a@

error_R�zO?

learning_rate_1�%�8�0eI       6%�	׸4J���A�*;


total_loss�$�@

error_R�I?

learning_rate_1�%�8�(NI       6%�	�5J���A�*;


total_lossQ}�@

error_R�)J?

learning_rate_1�%�8���I       6%�	>]5J���A�*;


total_loss�@

error_R��L?

learning_rate_1�%�8�k��I       6%�	�5J���A�*;


total_loss ��@

error_Rs??

learning_rate_1�%�8��@I       6%�	q�5J���A�*;


total_loss'h�@

error_R,MO?

learning_rate_1�%�8N��?I       6%�	=86J���A�*;


total_loss�`�@

error_R�La?

learning_rate_1�%�8��zI       6%�	}�6J���A�*;


total_loss��@

error_R�C?

learning_rate_1�%�8�|.I       6%�	��6J���A�*;


total_loss�"�@

error_R]?

learning_rate_1�%�8}nO�I       6%�	7J���A�*;


total_lossc^�@

error_R��m?

learning_rate_1�%�8c�UI       6%�	uc7J���A�*;


total_loss�@

error_R�3U?

learning_rate_1�%�8�i��I       6%�	��7J���A�*;


total_loss��@

error_RHS?

learning_rate_1�%�8K���I       6%�	}�7J���A�*;


total_loss��s@

error_R��X?

learning_rate_1�%�8��@I       6%�	-<8J���A�*;


total_loss�o�@

error_R�Q?

learning_rate_1�%�8FE<I       6%�	8J���A�*;


total_loss��A

error_R�L?

learning_rate_1�%�8D�1�I       6%�	)�8J���A�*;


total_lossv A

error_R^Z?

learning_rate_1�%�8m�:I       6%�	j'9J���A�*;


total_loss�p A

error_Ro�T?

learning_rate_1�%�8���$I       6%�	�j9J���A�*;


total_lossm̘@

error_R1�=?

learning_rate_1�%�8 �ۚI       6%�	w�9J���A�*;


total_lossh)�@

error_ROAD?

learning_rate_1�%�8 �pvI       6%�	��9J���A�*;


total_loss��@

error_REKT?

learning_rate_1�%�8�U�oI       6%�	�@:J���A�*;


total_lossJR�@

error_R
�N?

learning_rate_1�%�8ՈՂI       6%�	M�:J���A�*;


total_loss��@

error_R��K?

learning_rate_1�%�80��yI       6%�	�:J���A�*;


total_lossR6�@

error_R�K?

learning_rate_1�%�8�ی�I       6%�	�;J���A�*;


total_loss@�@

error_R�SW?

learning_rate_1�%�8Eo�[I       6%�	�S;J���A�*;


total_loss@

error_R�P?

learning_rate_1�%�8��II       6%�	М;J���A�*;


total_lossK�@

error_R=�A?

learning_rate_1�%�8�IuiI       6%�	d�;J���A�*;


total_loss�VA

error_R�r?

learning_rate_1�%�8(��I       6%�	+<J���A�*;


total_lossx��@

error_R�Y?

learning_rate_1�%�8q��=I       6%�	�k<J���A�*;


total_loss���@

error_Rd�U?

learning_rate_1�%�8��)�I       6%�	k�<J���A�*;


total_loss,��@

error_R�q>?

learning_rate_1�%�8s�I       6%�	C�<J���A�*;


total_lossF��@

error_RaoE?

learning_rate_1�%�8��I       6%�	�:=J���A�*;


total_loss�.A

error_R��T?

learning_rate_1�%�8�_�I       6%�	�=J���A�*;


total_loss䰣@

error_R�P?

learning_rate_1�%�8=v��I       6%�	.�=J���A�*;


total_lossFmc@

error_R#D?

learning_rate_1�%�8�|��I       6%�	�>J���A�*;


total_loss��@

error_R\#O?

learning_rate_1�%�8�ry�I       6%�	|L>J���A�*;


total_loss���@

error_R) @?

learning_rate_1�%�8F�U]I       6%�	y�>J���A�*;


total_loss,[�@

error_R,xN?

learning_rate_1�%�8j�;xI       6%�	E�>J���A�*;


total_loss���@

error_RߡZ?

learning_rate_1�%�8]C`�I       6%�	�?J���A�*;


total_loss�h�@

error_R 	>?

learning_rate_1�%�8D>�6I       6%�	zc?J���A�*;


total_loss�aA

error_R;�W?

learning_rate_1�%�8 a�I       6%�	��?J���A�*;


total_lossq�@

error_R�V?

learning_rate_1�%�8м{SI       6%�	)�?J���A�*;


total_loss�f�@

error_R
YX?

learning_rate_1�%�8���6I       6%�	�.@J���A�*;


total_loss1t�@

error_Rj\U?

learning_rate_1�%�8@�WI       6%�	6r@J���A�*;


total_loss���@

error_R�gP?

learning_rate_1�%�8@\�I       6%�	��@J���A�*;


total_lossZ��@

error_R��<?

learning_rate_1�%�8���jI       6%�	>�@J���A�*;


total_loss���@

error_RR:Y?

learning_rate_1�%�8�.��I       6%�	�;AJ���A�*;


total_loss18�@

error_R��M?

learning_rate_1�%�8��NI       6%�	�~AJ���A�*;


total_loss�t�@

error_RE�S?

learning_rate_1�%�8@��pI       6%�	��AJ���A�*;


total_loss���@

error_R;�F?

learning_rate_1�%�85U8�I       6%�	�BJ���A�*;


total_lossA#	A

error_R4']?

learning_rate_1�%�8�~�I       6%�	�KBJ���A�*;


total_lossC"z@

error_R�C?

learning_rate_1�%�8���(I       6%�	�BJ���A�*;


total_loss@��@

error_R@Y?

learning_rate_1�%�8p�y"I       6%�	��BJ���A�*;


total_loss�h�@

error_R�	V?

learning_rate_1�%�8��O/I       6%�	�!CJ���A�*;


total_loss8]�@

error_R�9T?

learning_rate_1�%�8d|I       6%�	]iCJ���A�*;


total_lossz2�@

error_R�^?

learning_rate_1�%�8 _��I       6%�	�CJ���A�*;


total_loss?�@

error_RS0O?

learning_rate_1�%�8����I       6%�	 �CJ���A�*;


total_loss���@

error_R�M?

learning_rate_1�%�8s:�I       6%�	=FDJ���A�*;


total_lossL��@

error_R}Bg?

learning_rate_1�%�8���qI       6%�	��DJ���A�*;


total_lossD�A

error_R�T?

learning_rate_1�%�8	��I       6%�	��DJ���A�*;


total_lossZ^�@

error_RZ?

learning_rate_1�%�8�'GI       6%�	2EJ���A�*;


total_lossE��@

error_R �R?

learning_rate_1�%�8���I       6%�	#[EJ���A�*;


total_lossޟ@

error_R��E?

learning_rate_1�%�8v��I       6%�	.�EJ���A�*;


total_loss�-�@

error_R�&W?

learning_rate_1�%�8F���I       6%�	��EJ���A�*;


total_loss8�{@

error_R�NX?

learning_rate_1�%�8���I       6%�	�%FJ���A�*;


total_loss&`"A

error_R�rP?

learning_rate_1�%�8`�kI       6%�	jFJ���A�*;


total_lossڜ@

error_R,�T?

learning_rate_1�8'��I       6%�	��FJ���A�*;


total_loss���@

error_R�H?

learning_rate_1�8z3K�I       6%�	S�FJ���A�*;


total_lossa�@

error_R�#H?

learning_rate_1�8) GI       6%�	�9GJ���A�*;


total_loss�e�@

error_REm?

learning_rate_1�8���%I       6%�	_|GJ���A�*;


total_lossx��@

error_R�Lb?

learning_rate_1�8r�5�I       6%�	ٞJJ���A�*;


total_loss���@

error_R��X?

learning_rate_1�8w&��I       6%�	��JJ���A�*;


total_loss�@

error_R�bJ?

learning_rate_1�8W/5I       6%�	�4KJ���A�*;


total_loss�Ц@

error_R�K?

learning_rate_1�8ɬ�iI       6%�	
xKJ���A�*;


total_loss��A

error_Ro�a?

learning_rate_1�8�ϩI       6%�	[�KJ���A�*;


total_loss&�s@

error_Rq&E?

learning_rate_1�8�V�I       6%�	 LJ���A�*;


total_loss<��@

error_R}G?

learning_rate_1�8>��I       6%�	�MLJ���A�*;


total_losszP�@

error_R�*C?

learning_rate_1�8݋$I       6%�	��LJ���A�*;


total_loss��@

error_R�|R?

learning_rate_1�8��&YI       6%�	�LJ���A�*;


total_loss���@

error_R)�A?

learning_rate_1�8НҷI       6%�	fMJ���A�*;


total_losslPy@

error_R�/@?

learning_rate_1�8=��&I       6%�	ZMJ���A�*;


total_loss^�@

error_R�]P?

learning_rate_1�8�z!I       6%�	|�MJ���A�*;


total_lossݔ�@

error_R�D?

learning_rate_1�8��6I       6%�	8�MJ���A�*;


total_loss.�
A

error_R� K?

learning_rate_1�8��I       6%�	]-NJ���A�*;


total_loss�@

error_Rw[?

learning_rate_1�8Kì�I       6%�	pNJ���A�*;


total_loss�g�@

error_RϼT?

learning_rate_1�8J4�$I       6%�	�NJ���A�*;


total_loss0��@

error_R��=?

learning_rate_1�8ͿQI       6%�	�NJ���A�*;


total_loss�1�@

error_R�P?

learning_rate_1�8X���I       6%�	�COJ���A�*;


total_loss���@

error_R�<U?

learning_rate_1�8K��/I       6%�	R�OJ���A�*;


total_lossl��@

error_RC�M?

learning_rate_1�8����I       6%�	n�OJ���A�*;


total_loss�ǿ@

error_R=d?

learning_rate_1�8����I       6%�	mPJ���A�*;


total_lossE[c@

error_Rc�J?

learning_rate_1�8#��I       6%�	�fPJ���A�*;


total_loss��@

error_R��R?

learning_rate_1�8�S]AI       6%�	��PJ���A�*;


total_loss���@

error_R�U?

learning_rate_1�8sC�I       6%�	A�PJ���A�*;


total_loss<��@

error_RN,[?

learning_rate_1�8�!�*I       6%�	*GQJ���A�*;


total_loss��@

error_R�C?

learning_rate_1�8r<!I       6%�	�QJ���A�*;


total_loss8�@

error_R�^I?

learning_rate_1�8k�I       6%�	(�QJ���A�*;


total_lossL��@

error_RE&V?

learning_rate_1�8>���I       6%�	�RJ���A�*;


total_loss�@

error_R�yW?

learning_rate_1�8@��,I       6%�	�URJ���A�*;


total_loss�[�@

error_Rv�H?

learning_rate_1�8��_I       6%�	��RJ���A�*;


total_loss� �@

error_R�aQ?

learning_rate_1�8��t�I       6%�	�RJ���A�*;


total_loss8�@

error_R8k:?

learning_rate_1�8WV_I       6%�	�SJ���A�*;


total_loss��@

error_RMB?

learning_rate_1�8�.�I       6%�	8aSJ���A�*;


total_loss.��@

error_R͈L?

learning_rate_1�8�0ZCI       6%�	E�SJ���A�*;


total_loss֏�@

error_RR	M?

learning_rate_1�8�<��I       6%�	��SJ���A�*;


total_loss6��@

error_R��[?

learning_rate_1�8G���I       6%�	E=TJ���A�*;


total_lossE޶@

error_R��Z?

learning_rate_1�8I	lLI       6%�	��TJ���A�*;


total_lossXU�@

error_R�c_?

learning_rate_1�8�� I       6%�	
�TJ���A�*;


total_loss���@

error_R?
R?

learning_rate_1�8�>�I       6%�	�AUJ���A�*;


total_lossoF�@

error_R�K?

learning_rate_1�8Z���I       6%�	UJ���A�*;


total_loss6F�@

error_R��_?

learning_rate_1�8���I       6%�	��UJ���A�*;


total_loss��@

error_Rl�^?

learning_rate_1�8��'I       6%�	|VVJ���A�*;


total_lossD�@

error_R��a?

learning_rate_1�8�U%�I       6%�	&�VJ���A�*;


total_loss�@

error_R
�[?

learning_rate_1�8oR5jI       6%�	��VJ���A�*;


total_loss��@

error_RMN?

learning_rate_1�8��I       6%�	�QWJ���A�*;


total_loss�ܤ@

error_R2aI?

learning_rate_1�8��+�I       6%�	ҕWJ���A�*;


total_loss3�A

error_R�7?

learning_rate_1�8��X�I       6%�	#�WJ���A�*;


total_loss�@

error_R�,I?

learning_rate_1�8����I       6%�	�=XJ���A�*;


total_loss3ُ@

error_R�R?

learning_rate_1�8%���I       6%�	��XJ���A�*;


total_lossZ��@

error_RiDj?

learning_rate_1�8��(kI       6%�	��XJ���A�*;


total_lossx^�@

error_R�{??

learning_rate_1�8��4�I       6%�	DYJ���A�*;


total_lossO��@

error_R�A?

learning_rate_1�8 ��I       6%�	Z�YJ���A�*;


total_loss?ȯ@

error_R�F?

learning_rate_1�8:���I       6%�	��YJ���A�*;


total_lossX��@

error_R��K?

learning_rate_1�8�v�I       6%�	ZJ���A�*;


total_lossR~�@

error_R�W?

learning_rate_1�8#���I       6%�	�ZZJ���A�*;


total_loss��@

error_RS[N?

learning_rate_1�8��I       6%�	Q�ZJ���A�*;


total_lossw��@

error_R�R?

learning_rate_1�8v�e�I       6%�	
[J���A�*;


total_lossJ�@

error_R�<U?

learning_rate_1�8wD��I       6%�	MT[J���A�*;


total_loss4x�@

error_R��\?

learning_rate_1�8)�V~I       6%�	�[J���A�*;


total_loss���@

error_RT�F?

learning_rate_1�8E�1�I       6%�	u�[J���A�*;


total_loss�2�@

error_R4oP?

learning_rate_1�8+�I       6%�	�"\J���A�*;


total_lossXv�@

error_R|�B?

learning_rate_1�8��u�I       6%�	�f\J���A�*;


total_lossvh�@

error_RZNU?

learning_rate_1�8�I       6%�	N�\J���A�*;


total_loss�y@

error_R�H?

learning_rate_1�8�I       6%�	I�\J���A�*;


total_loss	�@

error_R �Z?

learning_rate_1�8�#  I       6%�	�:]J���A�*;


total_loss�A

error_R��n?

learning_rate_1�8Q/ʿI       6%�	O�]J���A�*;


total_loss}<�@

error_R�eI?

learning_rate_1�8hQUnI       6%�	P�]J���A�*;


total_loss�@

error_R��S?

learning_rate_1�8Ub}�I       6%�	^J���A�*;


total_lossò�@

error_RopC?

learning_rate_1�8����I       6%�	@\^J���A�*;


total_lossNp�@

error_R��S?

learning_rate_1�8�V�fI       6%�	��^J���A�*;


total_loss�a�@

error_R�g?

learning_rate_1�8Â�'I       6%�	��^J���A�*;


total_lossX��@

error_R��I?

learning_rate_1�8�,��I       6%�	�1_J���A�*;


total_lossɓ@

error_R�LG?

learning_rate_1�8�>�-I       6%�	jw_J���A�*;


total_lossG�@

error_R�SH?

learning_rate_1�8e	vtI       6%�	ü_J���A�*;


total_loss&��@

error_R.TF?

learning_rate_1�8����I       6%�	� `J���A�*;


total_lossfB�@

error_R�Y?

learning_rate_1�8����I       6%�	�D`J���A�*;


total_loss�S�@

error_R�}G?

learning_rate_1�8�xI       6%�	��`J���A�*;


total_loss$+	A

error_RqTP?

learning_rate_1�8ܳ�vI       6%�	6�`J���A�*;


total_loss�׮@

error_R� ]?

learning_rate_1�8�~I       6%�	qaJ���A�*;


total_loss�@

error_R�0F?

learning_rate_1�8��s�I       6%�	�RaJ���A�*;


total_loss�4A

error_R�]S?

learning_rate_1�8}z�YI       6%�	�aJ���A�*;


total_loss��@

error_R�]?

learning_rate_1�8&�I       6%�	 �aJ���A�*;


total_loss��@

error_R�&C?

learning_rate_1�8�or%I       6%�	)bJ���A�*;


total_loss��~@

error_Rv�A?

learning_rate_1�8���fI       6%�	�sbJ���A�*;


total_loss�N�@

error_R�K?

learning_rate_1�8iWlI       6%�	j�bJ���A�*;


total_loss��e@

error_Rw�F?

learning_rate_1�8��+2I       6%�	|cJ���A�*;


total_loss�M�@

error_RI?

learning_rate_1�8#U�I       6%�	�HcJ���A�*;


total_lossD�@

error_RZ|P?

learning_rate_1�8�+V�I       6%�	��cJ���A�*;


total_loss1�@

error_R�K?

learning_rate_1�8*��I       6%�	[�cJ���A�*;


total_loss��@

error_R�PT?

learning_rate_1�8�	|I       6%�	OdJ���A�*;


total_loss��@

error_RvsD?

learning_rate_1�8�+I       6%�	�YdJ���A�*;


total_lossq�@

error_R׌`?

learning_rate_1�8�3�I       6%�	W�dJ���A�*;


total_loss�\�@

error_R�F?

learning_rate_1�8�G'I       6%�	G�dJ���A�*;


total_loss�,�@

error_R.�T?

learning_rate_1�8L�f2I       6%�	�,eJ���A�*;


total_loss�� A

error_Rd�<?

learning_rate_1�8��-[I       6%�	�teJ���A�*;


total_loss���@

error_R�?B?

learning_rate_1�8	zY�I       6%�	��eJ���A�*;


total_loss��@

error_R�N?

learning_rate_1�8a�jUI       6%�	*fJ���A�*;


total_loss���@

error_R��B?

learning_rate_1�8npP�I       6%�	�IfJ���A�*;


total_loss@��@

error_R�X?

learning_rate_1�8�͍�I       6%�	<�fJ���A�*;


total_loss�A

error_R��U?

learning_rate_1�8���I       6%�	��fJ���A�*;


total_loss���@

error_Rx�D?

learning_rate_1�8:cBI       6%�	dgJ���A�*;


total_loss� A

error_R8�R?

learning_rate_1�8��RUI       6%�	obgJ���A�*;


total_loss�I�@

error_RϴD?

learning_rate_1�8ȧ��I       6%�	�gJ���A�*;


total_loss��@

error_R�:L?

learning_rate_1�8��+I       6%�	&�gJ���A�*;


total_loss:A

error_R�h?

learning_rate_1�8�v>\I       6%�	�+hJ���A�*;


total_lossR�@

error_RƉ=?

learning_rate_1�8��I       6%�	�phJ���A�*;


total_lossW��@

error_R��X?

learning_rate_1�8n-^I       6%�	�hJ���A�*;


total_loss:��@

error_Rl�E?

learning_rate_1�8Ü�UI       6%�	�iJ���A�*;


total_loss���@

error_R��C?

learning_rate_1�8�&:.I       6%�	ofiJ���A�*;


total_loss҇@

error_R�`?

learning_rate_1�8�I       6%�	]�iJ���A�*;


total_loss�^�@

error_R��E?

learning_rate_1�8I��I       6%�	w�iJ���A�*;


total_lossv�@

error_R�SR?

learning_rate_1�8���iI       6%�	�8jJ���A�*;


total_loss�@

error_RZ;?

learning_rate_1�8>T4I       6%�	|jJ���A�*;


total_loss:A

error_R7M?

learning_rate_1�8"	A�I       6%�	g�jJ���A�*;


total_loss-��@

error_R�k<?

learning_rate_1�8:z�I       6%�	� kJ���A�*;


total_loss�+A

error_R�|Y?

learning_rate_1�8RĚ�I       6%�	CkJ���A�*;


total_loss�|rA

error_R��D?

learning_rate_1�8���I       6%�	5�kJ���A�*;


total_loss�9�@

error_Rr�Z?

learning_rate_1�8N: �I       6%�	��kJ���A�*;


total_loss�@

error_R�U?

learning_rate_1�8�^$)I       6%�	hlJ���A�*;


total_loss�r|@

error_Rs�O?

learning_rate_1�8�o�I       6%�	�_lJ���A�*;


total_loss��A

error_R�n?

learning_rate_1�8�P yI       6%�	��lJ���A�*;


total_loss�Ǽ@

error_RÅL?

learning_rate_1�8@��I       6%�	k�lJ���A�*;


total_lossot�@

error_RN?

learning_rate_1�8��V�I       6%�	�4mJ���A�*;


total_loss$X�@

error_R�D^?

learning_rate_1�8�$QI       6%�	?}mJ���A�*;


total_loss���@

error_R?L?

learning_rate_1�8��c�I       6%�	��mJ���A�*;


total_lossAA

error_R=�Y?

learning_rate_1�8F�I       6%�		nJ���A�*;


total_lossFE�@

error_R8�G?

learning_rate_1�8��YI       6%�	�OnJ���A�*;


total_loss��@

error_R�lU?

learning_rate_1�8�IU
I       6%�	�nJ���A�*;


total_loss���@

error_Rr�K?

learning_rate_1�8�V�hI       6%�	��nJ���A�*;


total_loss��@

error_R�H?

learning_rate_1�86eoI       6%�	#oJ���A�*;


total_loss��@

error_RIGX?

learning_rate_1�8�0zI       6%�	�doJ���A�*;


total_lossOh�@

error_R�L?

learning_rate_1�8v�qI       6%�	�oJ���A�*;


total_loss���@

error_R
K?

learning_rate_1�8Ϋ�I       6%�	��oJ���A�*;


total_loss���@

error_R��[?

learning_rate_1�8����I       6%�	g=pJ���A�*;


total_loss���@

error_R$�Q?

learning_rate_1�8~j2�I       6%�	:�pJ���A�*;


total_loss�K�@

error_R$LG?

learning_rate_1�8���I       6%�	W�pJ���A�*;


total_loss@��@

error_R�BP?

learning_rate_1�8��OuI       6%�	1	qJ���A�*;


total_loss�}�@

error_R�N?

learning_rate_1�8���/I       6%�	�OqJ���A�*;


total_loss���@

error_R��C?

learning_rate_1�8}�kI       6%�	��qJ���A�*;


total_loss���@

error_R�!??

learning_rate_1�8y��bI       6%�	��qJ���A�*;


total_loss\(�@

error_Rc[?

learning_rate_1�8u)5�I       6%�	�#rJ���A�*;


total_loss�B{@

error_R<pS?

learning_rate_1�8��O�I       6%�	eirJ���A�*;


total_lossvh�@

error_Rr�K?

learning_rate_1�8�-�I       6%�	1�rJ���A�*;


total_loss8n�@

error_R�I?

learning_rate_1�8Ί��I       6%�	��rJ���A�*;


total_lossl��@

error_RNDS?

learning_rate_1�8zL�I       6%�	�GsJ���A�*;


total_loss�!�@

error_Rnna?

learning_rate_1�8FO%�I       6%�	�sJ���A�*;


total_loss&hA

error_RۊK?

learning_rate_1�8fN	�I       6%�	��sJ���A�*;


total_lossα@

error_R��Q?

learning_rate_1�8+���I       6%�	�tJ���A�*;


total_loss��~@

error_RQ�H?

learning_rate_1�8G��	I       6%�	�WtJ���A�*;


total_loss��@

error_R
g?

learning_rate_1�8<>��I       6%�	�tJ���A�*;


total_loss���@

error_RͮW?

learning_rate_1�8�V�zI       6%�	auJ���A�*;


total_lossx7�@

error_R��G?

learning_rate_1�8J��I       6%�	�XuJ���A�*;


total_lossM�@

error_R��J?

learning_rate_1�8��O"I       6%�	�uJ���A�*;


total_lossϠ@

error_R�??

learning_rate_1�8`�zI       6%�	��uJ���A�*;


total_loss��@

error_R<GU?

learning_rate_1�8 �lI       6%�	�YvJ���A�*;


total_loss^�@

error_R��G?

learning_rate_1�8 �$�I       6%�	��vJ���A�*;


total_loss� A

error_R��=?

learning_rate_1�8�@sII       6%�	�vJ���A�*;


total_loss�M�@

error_R=Y?

learning_rate_1�8�I       6%�	4wJ���A�*;


total_loss��@

error_R}6?

learning_rate_1�8+CpEI       6%�	yzwJ���A�*;


total_loss�R�@

error_R��W?

learning_rate_1�8h��I       6%�	=�wJ���A�*;


total_loss���@

error_RD�D?

learning_rate_1�8����I       6%�	xJ���A�*;


total_loss4MGA

error_R7 M?

learning_rate_1�8�mI       6%�	�BxJ���A�*;


total_loss��@

error_R�j?

learning_rate_1�8}fJI       6%�	�xJ���A�*;


total_loss��A

error_R��O?

learning_rate_1�8��I       6%�	��xJ���A�*;


total_lossF�@

error_R=0D?

learning_rate_1�87U�I       6%�	�:yJ���A�*;


total_lossL)�@

error_R{�D?

learning_rate_1�8�kdBI       6%�	�~yJ���A�*;


total_lossl;�@

error_R}W?

learning_rate_1�8 ���I       6%�	�yJ���A�*;


total_lossp�@

error_R�VV?

learning_rate_1�8�(�I       6%�	�zJ���A�*;


total_loss���@

error_R��U?

learning_rate_1�8����I       6%�	AKzJ���A�*;


total_loss�A

error_R�OR?

learning_rate_1�8��%I       6%�	#�zJ���A�*;


total_loss�g�@

error_R�YA?

learning_rate_1�80���I       6%�	8�zJ���A�*;


total_loss\��@

error_RJQA?

learning_rate_1�8$-�I       6%�	({J���A�*;


total_loss�K�@

error_R�PE?

learning_rate_1�8B�&�I       6%�	{X{J���A�*;


total_losse��@

error_RL;M?

learning_rate_1�8�$b�I       6%�	c�{J���A�*;


total_loss�=�@

error_R�O?

learning_rate_1�8?Mz�I       6%�	��{J���A�*;


total_loss�Q�@

error_R3uR?

learning_rate_1�8S�}{I       6%�	
#|J���A�*;


total_lossC%�@

error_R�rR?

learning_rate_1�8H�I       6%�	rh|J���A�*;


total_loss��@

error_R[�<?

learning_rate_1�8���1I       6%�	�|J���A�*;


total_loss�@

error_R�?R?

learning_rate_1�8=��kI       6%�	X�|J���A�*;


total_lossN��@

error_R�]J?

learning_rate_1�8=��I       6%�	�9}J���A�*;


total_loss@�@

error_R��P?

learning_rate_1�8��:NI       6%�	��}J���A�*;


total_loss�Cc@

error_RO:??

learning_rate_1�8n�VbI       6%�	��}J���A�*;


total_loss�(�@

error_R�V?

learning_rate_1�8���I       6%�	�~J���A�*;


total_loss��A

error_RM�D?

learning_rate_1�8�\eI       6%�	�_~J���A�*;


total_loss�XA

error_RLC^?

learning_rate_1�8�h�EI       6%�	�~J���A�*;


total_lossLҟ@

error_R4uW?

learning_rate_1�8]L�I       6%�	��~J���A�*;


total_loss��@

error_RD?

learning_rate_1�8�猱I       6%�	:-J���A�*;


total_losst�~@

error_R�]R?

learning_rate_1�8L�$I       6%�	-qJ���A�*;


total_loss)�@

error_REX?

learning_rate_1�8iB>�I       6%�	��J���A�*;


total_loss:;�@

error_R�T?

learning_rate_1�8,,�I       6%�	X�J���A�*;


total_loss$��@

error_R��R?

learning_rate_1�8BI       6%�	�@�J���A�*;


total_lossC��@

error_Rq�R?

learning_rate_1�8��բI       6%�	ш�J���A�*;


total_loss`��@

error_R$�L?

learning_rate_1�8���gI       6%�	�΀J���A�*;


total_loss�S�@

error_R��G?

learning_rate_1�8X�f�I       6%�	��J���A�*;


total_lossq��@

error_R�QQ?

learning_rate_1�8����I       6%�	\�J���A�*;


total_loss���@

error_RE�X?

learning_rate_1�8t�qI       6%�	磁J���A�*;


total_loss��@

error_Ra^G?

learning_rate_1�8�}LII       6%�	w�J���A�*;


total_loss���@

error_R1�W?

learning_rate_1�8�`�I       6%�	#/�J���A�*;


total_loss4��@

error_R Q?

learning_rate_1�8�n�I       6%�	�|�J���A�*;


total_lossRwA

error_R3�U?

learning_rate_1�8���I       6%�	���J���A�*;


total_lossQ��@

error_R�h?

learning_rate_1�8��wI       6%�	��J���A�*;


total_loss���@

error_R;�>?

learning_rate_1�8\���I       6%�	hU�J���A�*;


total_lossx��@

error_RSE?

learning_rate_1�8K�&&I       6%�	���J���A�*;


total_loss
 �@

error_RQ%D?

learning_rate_1�8qMd�I       6%�	+�J���A�*;


total_loss��@

error_R�M?

learning_rate_1�8N��I       6%�	�#�J���A�*;


total_loss���@

error_R@�L?

learning_rate_1�8��$I       6%�	�e�J���A�*;


total_loss��@

error_R��b?

learning_rate_1�8= ��I       6%�	]��J���A�*;


total_loss���@

error_R)�A?

learning_rate_1�8섪I       6%�	��J���A�*;


total_loss�t�@

error_R�G?

learning_rate_1�8 l�I       6%�	�;�J���A�*;


total_lossa
�@

error_R�8i?

learning_rate_1�8�/�I       6%�	Ȇ�J���A�*;


total_loss8��@

error_R�_?

learning_rate_1�8��=�I       6%�	�̅J���A�*;


total_loss��@

error_R<gE?

learning_rate_1�8��R,I       6%�	��J���A�*;


total_loss�q�@

error_R`kP?

learning_rate_1�84�)aI       6%�	hZ�J���A�*;


total_loss���@

error_R�#T?

learning_rate_1�8�ٰI       6%�	ᜆJ���A�*;


total_loss8�@

error_R�YE?

learning_rate_1�8�8��I       6%�	�ކJ���A�*;


total_loss���@

error_R�FU?

learning_rate_1�8+���I       6%�	�"�J���A�*;


total_loss៰@

error_R��H?

learning_rate_1�8�t�I       6%�	Bp�J���A�*;


total_loss�B�@

error_R�P?

learning_rate_1�8qOC�I       6%�	0��J���A�*;


total_loss@��@

error_R��U?

learning_rate_1�8�l��I       6%�	���J���A�*;


total_loss��@

error_R��j?

learning_rate_1�8�	z�I       6%�	|D�J���A�*;


total_loss�,�@

error_Rh�K?

learning_rate_1�8����I       6%�	��J���A�*;


total_loss�M�@

error_R��Y?

learning_rate_1�8.��I       6%�	��J���A�*;


total_loss�U�@

error_R4
j?

learning_rate_1�893��I       6%�	:�J���A�*;


total_loss�N�@

error_R6�T?

learning_rate_1�8D��~I       6%�	�{�J���A�*;


total_loss�q�@

error_R|�P?

learning_rate_1�8�wI       6%�	���J���A�*;


total_lossIA

error_RxB?

learning_rate_1�8��I       6%�	��J���A�*;


total_loss��t@

error_R�J?

learning_rate_1�8����I       6%�	|Q�J���A�*;


total_loss���@

error_R�:S?

learning_rate_1�8��I       6%�	͔�J���A�*;


total_loss��@

error_RZ�^?

learning_rate_1�8�I       6%�	�يJ���A�*;


total_loss鮾@

error_R��U?

learning_rate_1�8k|�MI       6%�	X�J���A�*;


total_loss �@

error_RCU?

learning_rate_1�8�9�I       6%�	n`�J���A�*;


total_loss���@

error_R}�Q?

learning_rate_1�8�ͲI       6%�	ݣ�J���A�*;


total_loss�޺@

error_RJ�_?

learning_rate_1�8C��"I       6%�	��J���A�*;


total_lossȺA

error_R��H?

learning_rate_1�8rU�I       6%�	6*�J���A�*;


total_loss�iA

error_Ri�G?

learning_rate_1�8_^��I       6%�	^m�J���A�*;


total_lossNx�@

error_R�]C?

learning_rate_1�8�x��I       6%�	#��J���A�*;


total_loss���@

error_R$�F?

learning_rate_1�8�v�I       6%�	���J���A�*;


total_loss�/�@

error_R�Y?

learning_rate_1�8���pI       6%�	�E�J���A�*;


total_loss�@

error_RA�M?

learning_rate_1�8����I       6%�	���J���A�*;


total_loss��@

error_R��S?

learning_rate_1�8�pOI       6%�	r΍J���A�*;


total_loss@�@

error_R�<?

learning_rate_1�8���I       6%�	��J���A�*;


total_lossl��@

error_Rl�K?

learning_rate_1�8>��I       6%�	GV�J���A�*;


total_loss̈�@

error_R��B?

learning_rate_1�8Z��I       6%�	���J���A�*;


total_loss��@

error_R��C?

learning_rate_1�8�7�]I       6%�	ێJ���A�*;


total_lossa��@

error_R��O?

learning_rate_1�8��N&I       6%�	��J���A�*;


total_lossM`�@

error_R�xR?

learning_rate_1�8
�ĦI       6%�	c�J���A�*;


total_loss���@

error_R
�N?

learning_rate_1�8�Tj�I       6%�	���J���A�*;


total_loss�Z}@

error_R�4S?

learning_rate_1�8*�<+I       6%�	��J���A�*;


total_loss�G�@

error_RFL@?

learning_rate_1�8�GqI       6%�	P9�J���A�*;


total_loss�;�@

error_RW8W?

learning_rate_1�8g�YMI       6%�	Q��J���A�*;


total_lossM��@

error_RM?

learning_rate_1�8��8(I       6%�	/ȐJ���A�*;


total_loss�0�@

error_R�XZ?

learning_rate_1�8F��WI       6%�	��J���A�*;


total_loss��A

error_R�?P?

learning_rate_1�8	�}pI       6%�	�\�J���A�*;


total_loss�?�@

error_R?2S?

learning_rate_1�8�TB�I       6%�	ꢑJ���A�*;


total_loss���@

error_Rv�B?

learning_rate_1�8��r�I       6%�	��J���A�*;


total_lossa��@

error_R�<U?

learning_rate_1�8(J-5I       6%�	~'�J���A�*;


total_loss��@

error_R��T?

learning_rate_1�8��I       6%�	Om�J���A�*;


total_lossn��@

error_R7�d?

learning_rate_1�8^�#�I       6%�	��J���A�*;


total_loss�PA

error_Rr�Y?

learning_rate_1�8���I       6%�	���J���A�*;


total_loss �@

error_R��e?

learning_rate_1�8�dY�I       6%�	kC�J���A�*;


total_loss%�@

error_R�Q?

learning_rate_1�8�L̃I       6%�	"��J���A�*;


total_loss�dx@

error_R�M?

learning_rate_1�8#&�I       6%�	RГJ���A�*;


total_loss8��@

error_R2M?

learning_rate_1�8f��bI       6%�	Z�J���A�*;


total_loss/�@

error_R��A?

learning_rate_1�8���I       6%�	$X�J���A�*;


total_loss5�@

error_R��P?

learning_rate_1�8>�D.I       6%�	��J���A�*;


total_lossFܥ@

error_R�G?

learning_rate_1�8��D�I       6%�	�
�J���A�*;


total_losscy�@

error_R��Q?

learning_rate_1�8��wI       6%�	�a�J���A�*;


total_lossmxA

error_R� Y?

learning_rate_1�8��OTI       6%�	.��J���A�*;


total_loss�@

error_R�L?

learning_rate_1�8�mI       6%�	��J���A�*;


total_losso�_@

error_R�fM?

learning_rate_1�8�;�AI       6%�	�p�J���A�*;


total_lossv��@

error_R/wH?

learning_rate_1�8�1�I       6%�	Ի�J���A�*;


total_loss���@

error_R�lI?

learning_rate_1�8���TI       6%�	��J���A�*;


total_lossN�@

error_R�p_?

learning_rate_1�8H��lI       6%�	4L�J���A�*;


total_lossI�@

error_R�R?

learning_rate_1�8�I       6%�	ʑ�J���A�*;


total_loss��A

error_RFL?

learning_rate_1�8�G�I       6%�	�ޗJ���A�*;


total_losstϸ@

error_R�\?

learning_rate_1�8�ןI       6%�	�*�J���A�*;


total_lossIӝ@

error_Rt�X?

learning_rate_1�8�G��I       6%�	rq�J���A�*;


total_loss��A

error_R�lZ?

learning_rate_1�8���0I       6%�	-��J���A�*;


total_lossC��@

error_R�L?

learning_rate_1�8k}v7I       6%�	�$�J���A�*;


total_loss�T�@

error_RW�a?

learning_rate_1�8÷�I       6%�	Zm�J���A�*;


total_lossY�@

error_R��e?

learning_rate_1�8�a�sI       6%�	���J���A�*;


total_loss���@

error_R��M?

learning_rate_1�8�T�I       6%�	���J���A�*;


total_loss��@

error_R�fA?

learning_rate_1�8��I       6%�	�B�J���A�*;


total_loss\�@

error_RVNE?

learning_rate_1�8(��I       6%�	���J���A�*;


total_loss!#�@

error_R)4M?

learning_rate_1�8��n�I       6%�	�͚J���A�*;


total_loss�|�@

error_RLJ?

learning_rate_1�8���I       6%�	V�J���A�*;


total_lossw� A

error_R��??

learning_rate_1�8 &��I       6%�	�]�J���A�*;


total_loss���@

error_Rx�d?

learning_rate_1�8��gI       6%�	���J���A�*;


total_loss/��@

error_R��J?

learning_rate_1�8�R3I       6%�	��J���A�*;


total_loss���@

error_R�fL?

learning_rate_1�8�a)I       6%�	G.�J���A�*;


total_loss�(�@

error_R��J?

learning_rate_1�8[�jI       6%�	�u�J���A�*;


total_loss� A

error_R��R?

learning_rate_1�8؂�I       6%�	鸜J���A�*;


total_loss�V�@

error_R]Z?

learning_rate_1�8��U(I       6%�	��J���A�*;


total_loss�qx@

error_R |A?

learning_rate_1�8����I       6%�	�A�J���A�*;


total_lossD|�@

error_R@F3?

learning_rate_1�8d�5oI       6%�	^��J���A�*;


total_loss�b�@

error_RsW?

learning_rate_1�8���I       6%�	VӝJ���A�*;


total_loss� 1A

error_Rl�H?

learning_rate_1�8�J?�I       6%�	>�J���A�*;


total_loss!*�@

error_R��\?

learning_rate_1�8eRf|I       6%�	`c�J���A�*;


total_loss�|�@

error_R4dG?

learning_rate_1�8֦͓I       6%�	0��J���A�*;


total_loss�Z�@

error_R\bN?

learning_rate_1�8m�OI       6%�	r�J���A�*;


total_loss���@

error_RVYh?

learning_rate_1�8F�\I       6%�	�2�J���A�*;


total_loss8��@

error_R�k<?

learning_rate_1�8���I       6%�	2u�J���A�*;


total_loss#o�@

error_R��=?

learning_rate_1�8��`BI       6%�	l��J���A�*;


total_loss�9�@

error_Rv�>?

learning_rate_1�8|r@I       6%�	��J���A�*;


total_loss�i�@

error_R��Q?

learning_rate_1�8{�tI       6%�	�J�J���A�*;


total_loss=dA

error_RKS?

learning_rate_1�8/��I       6%�	
��J���A�*;


total_lossj-�@

error_R �P?

learning_rate_1�8yы�I       6%�	�٠J���A�*;


total_loss���@

error_R��R?

learning_rate_1�8?�vI       6%�	 �J���A�*;


total_loss�A

error_R\?

learning_rate_1�8�U
�I       6%�	te�J���A�*;


total_lossC��@

error_R�D?

learning_rate_1�8��?I       6%�	��J���A�*;


total_loss�gA

error_RR4K?

learning_rate_1�8�:��I       6%�	��J���A�*;


total_loss9�@

error_R�Q?

learning_rate_1�8P+ӻI       6%�	�4�J���A�*;


total_lossI��@

error_R~:?

learning_rate_1�8��!�I       6%�	�w�J���A�*;


total_losscm�@

error_R͠Y?

learning_rate_1�8��aI       6%�	W��J���A�*;


total_loss_�@

error_R��b?

learning_rate_1�85��I       6%�	j�J���A�*;


total_lossoϪ@

error_R�J?

learning_rate_1�8k>�7I       6%�	M�J���A�*;


total_loss�	�@

error_R�d?

learning_rate_1�8�\8I       6%�	Β�J���A�*;


total_loss��"A

error_R}a?

learning_rate_1�8���TI       6%�	�գJ���A�*;


total_lossv֕@

error_R389?

learning_rate_1�8!{LI       6%�	&�J���A�*;


total_loss� �@

error_R��W?

learning_rate_1�8�D�lI       6%�	�[�J���A�*;


total_loss�[�@

error_Rf�P?

learning_rate_1�8n�I       6%�	���J���A�*;


total_loss���@

error_RsJX?

learning_rate_1�8�ɛ�I       6%�	A�J���A�*;


total_loss)�@

error_R�Ka?

learning_rate_1�8�^0I       6%�	m*�J���A�*;


total_loss��@

error_R3�f?

learning_rate_1�8��I       6%�	�n�J���A�*;


total_lossT��@

error_R8�S?

learning_rate_1�8�[x�I       6%�	S��J���A�*;


total_loss���@

error_R��O?

learning_rate_1�8$T�tI       6%�	Q��J���A�*;


total_loss|��@

error_RDjI?

learning_rate_1�8s�K�I       6%�	�@�J���A�*;


total_lossɨ@

error_R�]?

learning_rate_1�8R1?�I       6%�	Ą�J���A�*;


total_loss���@

error_R��\?

learning_rate_1�8�&��I       6%�	�ƦJ���A�*;


total_loss�*�@

error_R�P?

learning_rate_1�8m�q�I       6%�	�	�J���A�*;


total_loss���@

error_R�dL?

learning_rate_1�8�>��I       6%�	�M�J���A�*;


total_loss��@

error_R��K?

learning_rate_1�8n�WI       6%�	��J���A�*;


total_lossܨ�@

error_Rd�G?

learning_rate_1�8��	I       6%�	�էJ���A�*;


total_loss���@

error_R��<?

learning_rate_1�8���I       6%�	��J���A�*;


total_lossS��@

error_R�S?

learning_rate_1�8�:q�I       6%�	Wd�J���A�*;


total_loss.�@

error_R�?O?

learning_rate_1�8/v I       6%�	���J���A�*;


total_lossK�A

error_R)H?

learning_rate_1�8L��I       6%�	��J���A�*;


total_lossw�@

error_RXP?

learning_rate_1�8���vI       6%�	�]�J���A�*;


total_loss�A

error_Rl`?

learning_rate_1�8Ĭ5I       6%�	���J���A�*;


total_lossj��@

error_R�3W?

learning_rate_1�8��,I       6%�	K�J���A�*;


total_loss-Q�@

error_R�o\?

learning_rate_1�8��M`I       6%�	@2�J���A�*;


total_loss�ع@

error_R:H?

learning_rate_1�8�:o�I       6%�	�J���A�*;


total_lossV��@

error_Rt G?

learning_rate_1�8>�$I       6%�	�êJ���A�*;


total_loss�g�@

error_Rxj]?

learning_rate_1�8�V�wI       6%�	��J���A�*;


total_loss�@

error_R	tT?

learning_rate_1�8��9I       6%�	BK�J���A�*;


total_lossj�@

error_R��T?

learning_rate_1�8n���I       6%�	�J���A�*;


total_lossq�@

error_RPX?

learning_rate_1�8˽��I       6%�	nԫJ���A�*;


total_loss.��@

error_R
�U?

learning_rate_1�8����I       6%�	(�J���A�*;


total_loss/��@

error_R$�W?

learning_rate_1�8��2:I       6%�	,^�J���A�*;


total_loss
��@

error_R�Y?

learning_rate_1�8�_V@I       6%�	Ҡ�J���A�*;


total_loss�@

error_RE?=?

learning_rate_1�8�	�I       6%�	��J���A�*;


total_loss��@

error_R$�D?

learning_rate_1�8>%�I       6%�	*�J���A�*;


total_loss.��@

error_R�8Q?

learning_rate_1�8�U��I       6%�	/r�J���A�*;


total_loss\͔@

error_R��q?

learning_rate_1�8�u!�I       6%�	[��J���A�*;


total_loss&�w@

error_R$�E?

learning_rate_1�8�ׯ�I       6%�	;�J���A�*;


total_loss��@

error_R�8Q?

learning_rate_1�8��nI       6%�	aE�J���A�*;


total_lossk�@

error_Rx3N?

learning_rate_1�8Y��I       6%�	��J���A�*;


total_loss3�@

error_R��a?

learning_rate_1�8�W�I       6%�	JˮJ���A�*;


total_loss���@

error_Ra�H?

learning_rate_1�8D��I       6%�	��J���A�*;


total_loss��@

error_R�<O?

learning_rate_1�8s=�1I       6%�	NS�J���A�*;


total_lossh?�@

error_R�%W?

learning_rate_1�8��I       6%�	\��J���A�*;


total_loss���@

error_RRU?

learning_rate_1�8	!��I       6%�	ٯJ���A�*;


total_loss��A

error_R,g?

learning_rate_1�8�5��I       6%�	��J���A�*;


total_loss���@

error_R��Q?

learning_rate_1�8�6jhI       6%�	^�J���A�*;


total_lossĞ�@

error_R�<d?

learning_rate_1�8`�LI       6%�	�J���A�*;


total_loss:�@

error_R�PY?

learning_rate_1�8̠��I       6%�	��J���A�*;


total_loss@n�@

error_R;�U?

learning_rate_1�8�ʻxI       6%�	X+�J���A�*;


total_lossz��@

error_R�\<?

learning_rate_1�8l��QI       6%�	�t�J���A�*;


total_loss�@

error_R�D?

learning_rate_1�8�s��I       6%�	���J���A�*;


total_loss���@

error_R�0a?

learning_rate_1�8H��II       6%�	��J���A�*;


total_lossL��@

error_R��K?

learning_rate_1�8թ��I       6%�	�F�J���A�*;


total_loss�@

error_Rh-X?

learning_rate_1�8����I       6%�	���J���A�*;


total_loss��<A

error_R81S?

learning_rate_1�8/L�I       6%�	*ղJ���A�*;


total_lossڨ�@

error_Rj]H?

learning_rate_1�8��GKI       6%�	j�J���A�*;


total_loss��@

error_RγA?

learning_rate_1�8���I       6%�	hc�J���A�*;


total_lossG{@

error_R�TN?

learning_rate_1�8[4I       6%�	���J���A�*;


total_loss�y�@

error_R�T?

learning_rate_1�8�c�nI       6%�	I�J���A�*;


total_loss�%{@

error_R=�<?

learning_rate_1�8��K�I       6%�	�-�J���A�*;


total_loss�T�@

error_R.9?

learning_rate_1�8��3�I       6%�	!p�J���A�*;


total_loss���@

error_Rf�S?

learning_rate_1�8m��~I       6%�	ӵ�J���A�*;


total_loss�b�@

error_R�I?

learning_rate_1�8����I       6%�	u�J���A�*;


total_loss��@

error_R�PQ?

learning_rate_1�8@�{I       6%�	ZW�J���A�*;


total_loss~" A

error_R$sO?

learning_rate_1�8eh��I       6%�	(��J���A�*;


total_loss{��@

error_R�V?

learning_rate_1�8�t�%I       6%�	��J���A�*;


total_loss�aA

error_RXH?

learning_rate_1�8j�S�I       6%�	=&�J���A�*;


total_loss-2�@

error_R��D?

learning_rate_1�8o��+I       6%�	tn�J���A�*;


total_loss{�@

error_R�DL?

learning_rate_1�8�X:ZI       6%�	3��J���A�*;


total_loss���@

error_R`�R?

learning_rate_1�8y��EI       6%�	���J���A�*;


total_loss�}@

error_Rq�9?

learning_rate_1�8�q7dI       6%�	�?�J���A�*;


total_lossl@

error_R�W?

learning_rate_1�8V�+�I       6%�	��J���A�*;


total_loss��@

error_R�kO?

learning_rate_1�8}�EI       6%�	SַJ���A�*;


total_loss#��@

error_R�~I?

learning_rate_1�8�1ȄI       6%�	� �J���A�*;


total_loss�@

error_Rz�6?

learning_rate_1�8j�*I       6%�	l�J���A�*;


total_loss���@

error_R
�K?

learning_rate_1�8��"I       6%�	/��J���A�*;


total_loss���@

error_R��D?

learning_rate_1�8C�!�I       6%�	(,�J���A�*;


total_loss{�@

error_R��P?

learning_rate_1�8/�ADI       6%�	���J���A�*;


total_loss<A�@

error_R�EQ?

learning_rate_1�8j��I       6%�	߹J���A�*;


total_loss~�@

error_Rv�f?

learning_rate_1�8~jFI       6%�	�(�J���A�*;


total_loss���@

error_R��I?

learning_rate_1�8 ڦ�I       6%�	v�J���A�*;


total_loss�A

error_Rw�T?

learning_rate_1�8���I       6%�	��J���A�*;


total_lossË�@

error_R, K?

learning_rate_1�8�_�I       6%�	�-�J���A�*;


total_loss���@

error_R7WI?

learning_rate_1�8؁�I       6%�	^r�J���A�*;


total_lossͷA

error_R�T?

learning_rate_1�8��*I       6%�	6��J���A�*;


total_loss��@

error_RH?

learning_rate_1�8��XI       6%�	N�J���A�*;


total_loss�{�@

error_R?�E?

learning_rate_1�8��=I       6%�	n�J���A�*;


total_loss<��@

error_RSK?

learning_rate_1�8m8{�I       6%�	8��J���A�*;


total_loss���@

error_R2Q?

learning_rate_1�8��I       6%�	��J���A�*;


total_loss�#�@

error_R��Q?

learning_rate_1�8�@'�I       6%�	�M�J���A�*;


total_loss���@

error_R��Q?

learning_rate_1�8%�ovI       6%�	/��J���A�*;


total_loss�2}@

error_Rf�;?

learning_rate_1�8�ʪ[I       6%�	��J���A�*;


total_loss���@

error_R��V?

learning_rate_1�8x��9I       6%�	M�J���A�*;


total_loss��@

error_R��>?

learning_rate_1�8�-�BI       6%�	��J���A�*;


total_loss��@

error_Ro�V?

learning_rate_1�8���$I       6%�	�޾J���A�*;


total_loss��@

error_R!�N?

learning_rate_1�8�q��I       6%�	%�J���A�*;


total_loss4��@

error_R��K?

learning_rate_1�8����I       6%�	Ui�J���A�*;


total_lossY^�@

error_R��T?

learning_rate_1�8=㟧I       6%�	�J���A�*;


total_loss��@

error_R�K?

learning_rate_1�8���I       6%�	B��J���A�*;


total_loss��@

error_R��S?

learning_rate_1�8:4tCI       6%�	C<�J���A�*;


total_loss)�@

error_R��Q?

learning_rate_1�8�&I       6%�	k��J���A�*;


total_loss�Ĵ@

error_R��O?

learning_rate_1�8-X�I       6%�	f��J���A�*;


total_loss?�@

error_R��]?

learning_rate_1�8ĉBPI       6%�	��J���A�*;


total_losstuA

error_R��c?

learning_rate_1�8eDI       6%�	�O�J���A�*;


total_loss�̨@

error_R]?

learning_rate_1�8���I       6%�	���J���A�*;


total_loss_`�@

error_Rf??

learning_rate_1�8��k9I       6%�	:��J���A�*;


total_loss��A

error_R��U?

learning_rate_1�8��1I       6%�	E/�J���A�*;


total_loss�"�@

error_RpG?

learning_rate_1�88��>I       6%�		��J���A�*;


total_loss]S�@

error_R�x[?

learning_rate_1�8���sI       6%�	Z��J���A�*;


total_loss`��@

error_R�d^?

learning_rate_1�8v ^*I       6%�	2/�J���A�*;


total_lossYZA

error_R�AI?

learning_rate_1�8.�T�I       6%�	s�J���A�*;


total_loss���@

error_R�o_?

learning_rate_1�8��#�I       6%�	/��J���A�*;


total_loss`�@

error_RS?

learning_rate_1�8|�(I       6%�	��J���A�*;


total_loss��@

error_R��O?

learning_rate_1�8�I       6%�	O�J���A�*;


total_lossdR�@

error_R�vB?

learning_rate_1�8�[��I       6%�	ܘ�J���A�*;


total_loss���@

error_R��U?

learning_rate_1�8(�XbI       6%�	���J���A�*;


total_lossGԙ@

error_R�f=?

learning_rate_1�8g�٤I       6%�	�-�J���A�*;


total_loss*U�@

error_R��W?

learning_rate_1�8eZ�sI       6%�	���J���A�*;


total_loss`c�@

error_R�7G?

learning_rate_1�8�`#I       6%�	���J���A�*;


total_loss��@

error_R�@?

learning_rate_1�8#��I       6%�	�2�J���A�*;


total_lossԩ�@

error_REC?

learning_rate_1�8�%��I       6%�	�{�J���A�*;


total_loss}��@

error_Rw�E?

learning_rate_1�8F��qI       6%�	���J���A�*;


total_lossv�@

error_R7�f?

learning_rate_1�8C�u{I       6%�	��J���A�*;


total_loss@�@

error_R,VP?

learning_rate_1�8g�I       6%�	IL�J���A�*;


total_loss�E�@

error_R��I?

learning_rate_1�8L1I       6%�	���J���A�*;


total_lossS��@

error_R��L?

learning_rate_1�8B�u?I       6%�	E��J���A�*;


total_loss���@

error_RE�G?

learning_rate_1�8���VI       6%�	6�J���A�*;


total_loss��%A

error_R3�Y?

learning_rate_1�8
n-I       6%�	�a�J���A�*;


total_loss��@

error_R��S?

learning_rate_1�8-d�I       6%�	6��J���A�*;


total_loss�+�@

error_R��q?

learning_rate_1�8�4I       6%�	:�J���A�*;


total_loss �@

error_R�_?

learning_rate_1�8{�yI       6%�	l��J���A�*;


total_lossSSA

error_RD�N?

learning_rate_1�8L=܃I       6%�	���J���A�*;


total_losst��@

error_RHPR?

learning_rate_1�8�c�'I       6%�	�+�J���A�*;


total_loss
�@

error_R��K?

learning_rate_1�8��WtI       6%�	�x�J���A�*;


total_loss��t@

error_Rc�N?

learning_rate_1�8�M�QI       6%�	-��J���A�*;


total_loss��@

error_R�'>?

learning_rate_1�8�vVI       6%�	��J���A�*;


total_loss�T�@

error_R�R?

learning_rate_1�8Ȁ��I       6%�	^X�J���A�*;


total_loss���@

error_RD�H?

learning_rate_1�8�b��I       6%�	���J���A�*;


total_loss!|�@

error_R�>E?

learning_rate_1�8!?I       6%�	���J���A�*;


total_loss���@

error_R�>G?

learning_rate_1�8	�A�I       6%�	�<�J���A�*;


total_loss|�@

error_R��Q?

learning_rate_1�8s���I       6%�	��J���A�*;


total_loss�	�@

error_RldC?

learning_rate_1�8g,�dI       6%�	1��J���A�*;


total_loss�@

error_R��X?

learning_rate_1�8����I       6%�	Y�J���A�*;


total_loss�p�@

error_R}S?

learning_rate_1�8��UUI       6%�	�a�J���A�*;


total_loss���@

error_RE�H?

learning_rate_1�8ƌ�RI       6%�	��J���A�*;


total_loss��@

error_R��F?

learning_rate_1�8��UI       6%�	���J���A�*;


total_lossP��@

error_R��[?

learning_rate_1�8�T�I       6%�	�<�J���A�*;


total_loss��@

error_R��Y?

learning_rate_1�8@4�I       6%�	���J���A�*;


total_losse6A

error_R�cY?

learning_rate_1�8p&bI       6%�	���J���A�*;


total_loss:ڗ@

error_R��K?

learning_rate_1�8��8(I       6%�	N�J���A�*;


total_lossl�@

error_RHjT?

learning_rate_1�8I�8I       6%�	�\�J���A�*;


total_lossĨ�@

error_RH�I?

learning_rate_1�8�̻QI       6%�	���J���A�*;


total_loss��@

error_R��W?

learning_rate_1�87kI       6%�	���J���A�*;


total_lossO��@

error_Re�Y?

learning_rate_1�8��I       6%�	�1�J���A�*;


total_lossf��@

error_R��I?

learning_rate_1�8F��I       6%�	�u�J���A�*;


total_loss͖�@

error_R)pK?

learning_rate_1�8���I       6%�	s��J���A�*;


total_loss[5�@

error_R�1N?

learning_rate_1�8��mmI       6%�	#��J���A�*;


total_loss��@

error_R�i7?

learning_rate_1�8����I       6%�	�B�J���A�*;


total_loss��@

error_R<�R?

learning_rate_1�8m��I       6%�	d��J���A�*;


total_loss�^�@

error_R@NS?

learning_rate_1�8~�{I       6%�	���J���A�*;


total_loss{o�@

error_R��D?

learning_rate_1�8w��I       6%�	J�J���A�*;


total_lossX/�@

error_R�S?

learning_rate_1�8[��I       6%�	H\�J���A�*;


total_lossm�@

error_R�dN?

learning_rate_1�8����I       6%�	���J���A�*;


total_loss�{�@

error_R�K?

learning_rate_1�8����I       6%�	���J���A�*;


total_lossD`�@

error_R�^?

learning_rate_1�8Y��dI       6%�	�5�J���A�*;


total_loss��@

error_R|=d?

learning_rate_1�8�'EsI       6%�	�y�J���A�*;


total_loss���@

error_RJzX?

learning_rate_1�8�8�I       6%�	F��J���A�*;


total_loss8�@

error_R�eN?

learning_rate_1�8�{�I       6%�	� �J���A�*;


total_lossO
�@

error_R�SA?

learning_rate_1�8��g�I       6%�	G�J���A�*;


total_lossD�@

error_RvxI?

learning_rate_1�8H��I       6%�	���J���A�*;


total_loss�cA

error_R��P?

learning_rate_1�8�y_�I       6%�	\��J���A�*;


total_loss$4�@

error_R��O?

learning_rate_1�8C�XqI       6%�	�J���A�*;


total_loss��@

error_Rt&K?

learning_rate_1�8�R�I       6%�	cd�J���A�*;


total_lossc��@

error_R�SN?

learning_rate_1�8ťy�I       6%�	���J���A�*;


total_loss$��@

error_R7G?

learning_rate_1�8TNS_I       6%�	���J���A�*;


total_lossf��@

error_R��W?

learning_rate_1�8�PԠI       6%�	�3�J���A�*;


total_lossR��@

error_R�}V?

learning_rate_1�8'*"I       6%�	�w�J���A�*;


total_loss�h$A

error_RR9^?

learning_rate_1�8��A(I       6%�	p��J���A�*;


total_loss4P�@

error_R�L?

learning_rate_1�8[R��I       6%�	���J���A�*;


total_loss)��@

error_RASk?

learning_rate_1�8q~!�I       6%�	>B�J���A�*;


total_loss�5�@

error_R%�R?

learning_rate_1�8�LdI       6%�	O��J���A�*;


total_loss���@

error_R�M?

learning_rate_1�8���I       6%�	q��J���A�*;


total_loss��@

error_R�U?

learning_rate_1�8r/u=I       6%�	��J���A�*;


total_lossL�@

error_R�MS?

learning_rate_1�8���6I       6%�	)U�J���A�*;


total_loss1H�@

error_R@�=?

learning_rate_1�8��v?I       6%�	H��J���A�*;


total_loss��A

error_R�UI?

learning_rate_1�8D���I       6%�	
 �J���A�*;


total_loss��@

error_RF�E?

learning_rate_1�8�G�I       6%�	\K�J���A�*;


total_lossa�@

error_RE�L?

learning_rate_1�8I�F0I       6%�	g��J���A�*;


total_lossX��@

error_R��Z?

learning_rate_1�8�U�I       6%�	&��J���A�*;


total_loss��@

error_Ra?

learning_rate_1�8/�I       6%�	c�J���A�*;


total_loss��@

error_R�UR?

learning_rate_1�8-�PMI       6%�	�b�J���A�*;


total_loss�,�@

error_R�M?

learning_rate_1�8⸟�I       6%�	���J���A�*;


total_loss�ۯ@

error_R
�R?

learning_rate_1�8}~��I       6%�	C��J���A�*;


total_loss36�@

error_R�uf?

learning_rate_1�8�C�%I       6%�	K-�J���A�*;


total_loss���@

error_R!V`?

learning_rate_1�8�H�I       6%�	�p�J���A�*;


total_loss]�@

error_R�J?

learning_rate_1�8��-�I       6%�	S��J���A�*;


total_lossR<�@

error_R�.X?

learning_rate_1�8��DI       6%�	T��J���A�*;


total_loss*�@

error_Rx�??

learning_rate_1�8�f�I       6%�	�7�J���A�*;


total_loss<dA

error_Rh�c?

learning_rate_1�8���I       6%�	�}�J���A�*;


total_loss���@

error_R��V?

learning_rate_1�8��.I       6%�	��J���A�*;


total_loss�@

error_R�]?

learning_rate_1�8�=:�I       6%�	��J���A�*;


total_loss�^�@

error_R�}W?

learning_rate_1�8e��6I       6%�	�]�J���A�*;


total_loss�Y�@

error_R�eK?

learning_rate_1�8�C[3I       6%�	��J���A�*;


total_loss�s�@

error_R��U?

learning_rate_1�8�]bUI       6%�	n��J���A�*;


total_loss;��@

error_R�l?

learning_rate_1�8%��	I       6%�	�N�J���A�*;


total_lossM}�@

error_R=�P?

learning_rate_1�8AY��I       6%�	��J���A�*;


total_loss3��@

error_RM�I?

learning_rate_1�8�hI       6%�	��J���A�*;


total_loss(�@

error_R��D?

learning_rate_1�8�mI       6%�	*�J���A�*;


total_loss�̟@

error_R��W?

learning_rate_1�8u��I       6%�	�p�J���A�*;


total_loss���@

error_R
XE?

learning_rate_1�8�O I       6%�	��J���A�*;


total_loss�<�@

error_R��@?

learning_rate_1�8��:I       6%�	���J���A�*;


total_loss�4�@

error_R�dQ?

learning_rate_1�8�q?�I       6%�	�<�J���A�*;


total_loss�'A

error_R�Y?

learning_rate_1�8����I       6%�	ׇ�J���A�*;


total_lossb�@

error_R`j@?

learning_rate_1�8ԡ�I       6%�	���J���A�*;


total_loss<I�@

error_R�8U?

learning_rate_1�8s�ӋI       6%�	g�J���A�*;


total_lossx A

error_R�CY?

learning_rate_1�89Š�I       6%�	V�J���A�*;


total_loss([�@

error_R*�E?

learning_rate_1�8���I       6%�	L��J���A�*;


total_loss�0�@

error_R�ke?

learning_rate_1�8yY�I       6%�	o��J���A�*;


total_loss�G�@

error_R�_O?

learning_rate_1�8ٓ��I       6%�	a!�J���A�*;


total_loss=��@

error_R��F?

learning_rate_1�8�
%�I       6%�	�e�J���A�*;


total_loss���@

error_R�-e?

learning_rate_1�8�9K�I       6%�	��J���A�*;


total_lossJ�@

error_R��X?

learning_rate_1�8��I       6%�	h��J���A�*;


total_lossW��@

error_R.<?

learning_rate_1�8D��OI       6%�	�.�J���A�*;


total_loss]��@

error_R��_?

learning_rate_1�8#�zI       6%�	�r�J���A�*;


total_loss89�@

error_R�~3?

learning_rate_1�8亅�I       6%�	e��J���A�*;


total_loss���@

error_RDPJ?

learning_rate_1�8���I       6%�	x��J���A�*;


total_loss���@

error_R�	W?

learning_rate_1�8t�ގI       6%�	H@�J���A�*;


total_loss*c�@

error_R�9S?

learning_rate_1�8b-�I       6%�	B��J���A�*;


total_loss[�@

error_RMP?

learning_rate_1�8/#r�I       6%�	 ��J���A�*;


total_loss�A

error_R7\?

learning_rate_1�8�l��I       6%�	�J���A�*;


total_loss�>�@

error_R�@?

learning_rate_1�8]���I       6%�	 \�J���A�*;


total_loss�&�@

error_RHKW?

learning_rate_1�8�4ޮI       6%�	֢�J���A�*;


total_lossxW�@

error_R�B?

learning_rate_1�8�@}�I       6%�	O��J���A�*;


total_loss��@

error_RF�X?

learning_rate_1�8�h2|I       6%�		E�J���A�*;


total_loss�I�@

error_RZQ?

learning_rate_1�8�X��I       6%�	*��J���A�*;


total_lossD;�@

error_Rd>I?

learning_rate_1�8�J9I       6%�	��J���A�*;


total_lossA��@

error_R��9?

learning_rate_1�8���I       6%�	np�J���A�*;


total_lossE�@

error_R�Y??

learning_rate_1�8���I       6%�	���J���A�*;


total_loss
 �@

error_R�];?

learning_rate_1�8�搯I       6%�	�?�J���A�*;


total_loss�
�@

error_R3H?

learning_rate_1�8�W�@I       6%�	|��J���A�*;


total_loss7ܑ@

error_R�&H?

learning_rate_1�8!�}>I       6%�	d��J���A�*;


total_loss���@

error_Rv�\?

learning_rate_1�8� I�I       6%�	Ws�J���A�*;


total_lossڙ�@

error_R	�K?

learning_rate_1�8�NI       6%�	���J���A�*;


total_loss�!�@

error_Ra�C?

learning_rate_1�8��I       6%�	��J���A�*;


total_loss��K@

error_R<�8?

learning_rate_1�8}�?�I       6%�	d�J���A�*;


total_loss�d�@

error_RcD=?

learning_rate_1�89veI       6%�	��J���A�*;


total_loss-��@

error_RO#E?

learning_rate_1�8D:eI       6%�	��J���A�*;


total_loss���@

error_RFR?

learning_rate_1�8cNI       6%�	�z�J���A�*;


total_loss�c�@

error_R�9e?

learning_rate_1�8�r�EI       6%�	=��J���A�*;


total_loss�SA

error_R31Q?

learning_rate_1�8VG��I       6%�	&"�J���A�*;


total_loss�{�@

error_Rs�X?

learning_rate_1�8��I       6%�	�n�J���A�*;


total_loss�ֵ@

error_R��R?

learning_rate_1�8��jwI       6%�	���J���A�*;


total_losson�@

error_R)wI?

learning_rate_1�8�*�>I       6%�	��J���A�*;


total_loss���@

error_R��O?

learning_rate_1�8�P��I       6%�	�_�J���A�*;


total_loss��@

error_R��H?

learning_rate_1�8��GI       6%�	_��J���A�*;


total_losslƢ@

error_R�;?

learning_rate_1�8B��I       6%�	M��J���A�*;


total_lossRZ�@

error_R�bC?

learning_rate_1�8,CCI       6%�	�T�J���A�*;


total_loss���@

error_R�Z?

learning_rate_1�8��LI       6%�	���J���A�*;


total_loss%J�@

error_R��J?

learning_rate_1�8�/�I       6%�	��J���A�*;


total_lossF��@

error_R/�C?

learning_rate_1�8u�#I       6%�	4;�J���A�*;


total_loss�լ@

error_R�O@?

learning_rate_1�8�̞�I       6%�	���J���A�*;


total_loss�y�@

error_Rf�S?

learning_rate_1�8�P��I       6%�	���J���A�*;


total_loss��@

error_RUj?

learning_rate_1�8�7	kI       6%�	��J���A�*;


total_lossD��@

error_R�B?

learning_rate_1�8��^I       6%�	dU�J���A�*;


total_loss��A

error_R��Y?

learning_rate_1�8�K*RI       6%�	��J���A�*;


total_loss�G�@

error_RSFZ?

learning_rate_1�8�g#I       6%�	t��J���A�*;


total_loss�@

error_RwJ?

learning_rate_1�8�'�XI       6%�	�D�J���A�*;


total_lossJf�@

error_Rl�Y?

learning_rate_1�8<]�I       6%�	���J���A�*;


total_loss`��@

error_R�OZ?

learning_rate_1�8 �UI       6%�	b��J���A�*;


total_loss�(�@

error_R\f[?

learning_rate_1�8	�R-I       6%�	9�J���A�*;


total_loss�Ű@

error_R�.?

learning_rate_1�87Sz�I       6%�	�_�J���A�*;


total_loss.��@

error_RʃU?

learning_rate_1�8�&FI       6%�	��J���A�*;


total_loss|�@

error_R�>D?

learning_rate_1�8�-�uI       6%�	��J���A�*;


total_loss��@

error_RTOB?

learning_rate_1�8�*�#I       6%�	�R�J���A�*;


total_lossꛁ@

error_R!�R?

learning_rate_1�8լ�I       6%�	ܛ�J���A�*;


total_loss��A

error_R��P?

learning_rate_1�8� �I       6%�	���J���A�*;


total_lossqA�@

error_R�:U?

learning_rate_1�8���+I       6%�	%�J���A�*;


total_loss��@

error_R�XC?

learning_rate_1�8��1I       6%�	�j�J���A�*;


total_loss�/�@

error_R��Q?

learning_rate_1�8��PI       6%�	J��J���A�*;


total_loss��@

error_R��Q?

learning_rate_1�8O2�I       6%�	���J���A�*;


total_lossJ��@

error_RaEg?

learning_rate_1�8ĥI       6%�	S8�J���A�*;


total_lossW��@

error_R�]K?

learning_rate_1�8���I       6%�	�|�J���A�*;


total_loss��@

error_RD9S?

learning_rate_1�8FH��I       6%�	d��J���A�*;


total_loss�h�@

error_R�mB?

learning_rate_1�8t���I       6%�	z�J���A�*;


total_loss:��@

error_R��a?

learning_rate_1�8�_��I       6%�	jL�J���A�*;


total_loss�i�@

error_R��J?

learning_rate_1�8��#LI       6%�	{��J���A�*;


total_loss_E�@

error_RJ?

learning_rate_1�8Vp�I       6%�	���J���A�*;


total_loss��@

error_R��E?

learning_rate_1�8�9QQI       6%�	O*�J���A�*;


total_loss�R�@

error_R��D?

learning_rate_1�8�I       6%�	�m�J���A�*;


total_loss��@

error_Rt�H?

learning_rate_1�8N�F�I       6%�	��J���A�*;


total_loss�|�@

error_Rf4\?

learning_rate_1�8o��xI       6%�	B��J���A�*;


total_loss�@

error_R��M?

learning_rate_1�8��DI       6%�	P=�J���A�*;


total_loss��@

error_R&�I?

learning_rate_1�8��2I       6%�	�}�J���A�*;


total_loss��@

error_R6#_?

learning_rate_1�8���I       6%�	���J���A�*;


total_loss�>�@

error_ROG?

learning_rate_1�8^#e�I       6%�	��J���A�*;


total_loss�˚@

error_R�5M?

learning_rate_1�8,ٕ�I       6%�	�a�J���A�*;


total_loss�[�@

error_R��9?

learning_rate_1�8���I       6%�	A��J���A�*;


total_lossa��@

error_R��B?

learning_rate_1�8v��I       6%�	��J���A�*;


total_loss���@

error_R�fS?

learning_rate_1�8�Q�YI       6%�	�0�J���A�*;


total_loss7A

error_R�y[?

learning_rate_1�8	��I       6%�	st�J���A�*;


total_loss��A

error_R#C?

learning_rate_1�8+�I       6%�	d��J���A�*;


total_loss���@

error_R�R?

learning_rate_1�8 9�I       6%�	p �J���A�*;


total_loss���@

error_R�1J?

learning_rate_1�8B��I       6%�	E�J���A�*;


total_loss�X�@

error_R��H?

learning_rate_1�8p@��I       6%�	���J���A�*;


total_loss��@

error_R�H?

learning_rate_1�8�:�UI       6%�	]��J���A�*;


total_loss�į@

error_R�NZ?

learning_rate_1�8�hqI       6%�	%!�J���A�*;


total_lossEй@

error_R�9I?

learning_rate_1�8���uI       6%�	�a�J���A�*;


total_loss���@

error_RT�Z?

learning_rate_1�8��t�I       6%�	,��J���A�*;


total_loss�M�@

error_R��R?

learning_rate_1�8	\�I       6%�	���J���A�*;


total_loss[܄@

error_Rq�W?

learning_rate_1�8�m�I       6%�	(3�J���A�*;


total_loss,L�@

error_R�D?

learning_rate_1�8�܁I       6%�	}�J���A�*;


total_loss��,A

error_RdkU?

learning_rate_1�8���$I       6%�	���J���A�*;


total_loss{�@

error_Rr�W?

learning_rate_1�8�d�I       6%�	r�J���A�*;


total_losst�@

error_R�Q?

learning_rate_1�8r�I       6%�	�L�J���A�*;


total_lossͶ@

error_Rt[?

learning_rate_1�8:aXWI       6%�	"��J���A�*;


total_loss���@

error_R��L?

learning_rate_1�8T�'�I       6%�	d��J���A�*;


total_lossZ�@

error_R��D?

learning_rate_1�8��I       6%�	j �J���A�*;


total_losso�@

error_R�^L?

learning_rate_1�8ѳ�fI       6%�	we�J���A�*;


total_loss(��@

error_R��X?

learning_rate_1�8�)��I       6%�	��J���A�*;


total_loss��@

error_R�E?

learning_rate_1�8�>�I       6%�	� K���A�*;


total_loss�M�@

error_RD�O?

learning_rate_1�8�$�I       6%�	�J K���A�*;


total_loss��@

error_Rv�Q?

learning_rate_1�8�ܩ�I       6%�	� K���A�*;


total_lossy�A

error_R��T?

learning_rate_1�8`��:I       6%�	�� K���A�*;


total_loss:h�@

error_R�mV?

learning_rate_1�8��yI       6%�	�K���A�*;


total_lossm!�@

error_R��F?

learning_rate_1�8,k�-I       6%�	�bK���A�*;


total_loss�l�@

error_RђS?

learning_rate_1�8�¨8I       6%�	�K���A�*;


total_lossǏ�@

error_R��F?

learning_rate_1�8�c� I       6%�	��K���A�*;


total_loss!T�@

error_R&E[?

learning_rate_1�8�.�9I       6%�	;2K���A�*;


total_lossϛ�@

error_R�[?

learning_rate_1�8��2I       6%�	�uK���A�*;


total_loss꼤@

error_R�8?

learning_rate_1�8��� I       6%�	��K���A�*;


total_loss�J�@

error_R�.U?

learning_rate_1�8n�h�I       6%�	��K���A�*;


total_lossO3�@

error_R��a?

learning_rate_1�8��=I       6%�	�BK���A�*;


total_loss�e�@

error_RsQ[?

learning_rate_1�8�iQI       6%�	��K���A�*;


total_lossƁ�@

error_RDGX?

learning_rate_1�8g�YPI       6%�	��K���A�*;


total_loss��@

error_R�EM?

learning_rate_1�8�&��I       6%�	{K���A�*;


total_loss���@

error_RM�R?

learning_rate_1�89l�PI       6%�	@NK���A�*;


total_loss�$A

error_R$J?

learning_rate_1�8A�3I       6%�	��K���A�*;


total_loss)�@

error_R�tM?

learning_rate_1�8���I       6%�	^�K���A�*;


total_loss"Q�@

error_RDNU?

learning_rate_1�8���0I       6%�	K���A�*;


total_loss&i�@

error_R�T?

learning_rate_1�8�x¿I       6%�	ldK���A�*;


total_loss:(A

error_RvG?

learning_rate_1�8F>�I       6%�	[�K���A�*;


total_loss�7�@

error_R�P?

learning_rate_1�8��I       6%�	� K���A�*;


total_loss	[�@

error_R�TX?

learning_rate_1�8*�/I       6%�	�DK���A�*;


total_lossv{@

error_R�a?

learning_rate_1�8��2�I       6%�	s�K���A�*;


total_loss_�@

error_R3\?

learning_rate_1�8w�=I       6%�	c�K���A�*;


total_losso��@

error_R�[?

learning_rate_1�8���SI       6%�	�-K���A�*;


total_loss;"�@

error_R3|G?

learning_rate_1�8���4I       6%�	>rK���A�*;


total_lossڲ�@

error_RZ�O?

learning_rate_1�8B��I       6%�	�K���A�*;


total_loss�B�@

error_RMG\?

learning_rate_1�8�@s�I       6%�	Q�K���A�*;


total_loss��@

error_R\^K?

learning_rate_1�8B\�I       6%�	FK���A�*;


total_loss��@

error_R��T?

learning_rate_1�8�k��I       6%�	}�K���A�*;


total_loss�4�@

error_R�Z]?

learning_rate_1�8��I       6%�	#	K���A�*;


total_loss�	�@

error_RʫK?

learning_rate_1�8��M�I       6%�	\q	K���A�*;


total_lossV�K@

error_R�m;?

learning_rate_1�8�E|�I       6%�	��	K���A�*;


total_loss�ծ@

error_R��T?

learning_rate_1�8��^#I       6%�	��	K���A�*;


total_losst7�@

error_R��W?

learning_rate_1�8�T�I       6%�	@
K���A�*;


total_loss�8�@

error_R�mS?

learning_rate_1�8�!y3I       6%�	z�
K���A�*;


total_loss!l�@

error_R3^X?

learning_rate_1�8`k��I       6%�	�
K���A�*;


total_loss6@

error_R�D?

learning_rate_1�8�se�I       6%�	 K���A�*;


total_loss��A

error_RfBO?

learning_rate_1�8��bI       6%�	lRK���A�*;


total_loss!�@

error_RF�T?

learning_rate_1�8��¿I       6%�	1�K���A�*;


total_loss{y@

error_R��U?

learning_rate_1�8�I       6%�	��K���A�*;


total_loss�/�@

error_R��O?

learning_rate_1�8�KsEI       6%�	=2K���A�*;


total_lossS��@

error_RW�T?

learning_rate_1�8����I       6%�	={K���A�*;


total_loss���@

error_R��P?

learning_rate_1�8�Gf�I       6%�	v�K���A�*;


total_lossHO�@

error_R$�L?

learning_rate_1�8F�c�I       6%�	�K���A�*;


total_losso��@

error_R�S?

learning_rate_1�8�z2I       6%�	�YK���A�*;


total_loss�¬@

error_RMT?

learning_rate_1�8d�#nI       6%�	A�K���A�*;


total_loss���@

error_RneJ?

learning_rate_1�8��%�I       6%�	��K���A�*;


total_loss���@

error_R�U?

learning_rate_1�8�Z�*I       6%�	�2K���A�*;


total_loss��@

error_Rq9?

learning_rate_1�8 ���I       6%�	|tK���A�*;


total_loss��@

error_R�eJ?

learning_rate_1�8��3�I       6%�	7�K���A�*;


total_loss<P�@

error_RĜP?

learning_rate_1�8�e�I       6%�	��K���A�*;


total_loss���@

error_R�B_?

learning_rate_1�8�\ I       6%�	*EK���A�*;


total_loss�'�@

error_R�fU?

learning_rate_1�8-��I       6%�	f�K���A�*;


total_loss�H�@

error_R��R?

learning_rate_1�8���I       6%�	b�K���A�*;


total_loss�5�@

error_RרQ?

learning_rate_1�8d��I       6%�	vK���A�*;


total_lossN/�@

error_R#�M?

learning_rate_1�8�3>�I       6%�	�WK���A�*;


total_loss��@

error_R�AI?

learning_rate_1�8�7̂I       6%�	C�K���A�*;


total_loss�V�@

error_R�R?

learning_rate_1�8Lr�I       6%�	�K���A�*;


total_lossQ�@

error_RlSX?

learning_rate_1�8��I       6%�	�&K���A�*;


total_lossӷ�@

error_R�CA?

learning_rate_1�8��I       6%�	AlK���A�*;


total_loss���@

error_R{�[?

learning_rate_1�8�[/I       6%�	׵K���A�*;


total_loss���@

error_RԕM?

learning_rate_1�8�^M�I       6%�	�K���A�*;


total_lossz��@

error_RC�T?

learning_rate_1�8�N;�I       6%�	8<K���A�*;


total_loss}i�@

error_R�`?

learning_rate_1�8b!��I       6%�	Q�K���A�*;


total_loss�_@

error_R��F?

learning_rate_1�8�&"�I       6%�	��K���A�*;


total_loss
�@

error_RMP?

learning_rate_1�8(΄I       6%�	rK���A�*;


total_loss�M�@

error_R�6L?

learning_rate_1�8����I       6%�	;KK���A�*;


total_loss�d�@

error_RH�@?

learning_rate_1�8�s��I       6%�	ȎK���A�*;


total_lossچ�@

error_R_4R?

learning_rate_1�8����I       6%�	d�K���A�*;


total_loss�"�@

error_RjR?

learning_rate_1�8<��I       6%�	�K���A�*;


total_lossh��@

error_R�\?

learning_rate_1�8Lb��I       6%�	"dK���A�*;


total_loss1:�@

error_R��L?

learning_rate_1�8E\I       6%�	��K���A�*;


total_lossԢ�@

error_RTB?

learning_rate_1�8���I       6%�	�K���A�*;


total_loss�%�@

error_RC;?

learning_rate_1�8��I       6%�	�6K���A�*;


total_loss/�@

error_R�aK?

learning_rate_1�8���8I       6%�	�K���A�*;


total_loss�'�@

error_Rq�E?

learning_rate_1�8��^"I       6%�	��K���A�*;


total_loss�b�@

error_RX�D?

learning_rate_1�8�!�I       6%�	rK���A�*;


total_lossj��@

error_R�Da?

learning_rate_1�8���bI       6%�	UK���A�*;


total_lossV�@

error_RoV?

learning_rate_1�8���I       6%�	��K���A�*;


total_lossb�@

error_R�U?

learning_rate_1�8� �_I       6%�	��K���A�*;


total_loss�E�@

error_R��A?

learning_rate_1�8(/�LI       6%�	�,K���A�*;


total_loss@��@

error_R�~L?

learning_rate_1�8 �)I       6%�	�uK���A�*;


total_loss`��@

error_R1JX?

learning_rate_1�8n
I       6%�	�K���A�*;


total_loss8��@

error_R��L?

learning_rate_1�8V#�5I       6%�	��K���A�*;


total_loss�ک@

error_R,�G?

learning_rate_1�8�3O�I       6%�	*@K���A�*;


total_lossc�A

error_R�2[?

learning_rate_1�8�hT�I       6%�	T�K���A�*;


total_loss�̉@

error_R$�8?

learning_rate_1�8���I       6%�	H�K���A�*;


total_lossoa�@

error_R�Z?

learning_rate_1�8s�uI       6%�	D'K���A�*;


total_loss,� A

error_R�7A?

learning_rate_1�8�EvI       6%�	�mK���A�*;


total_loss�h�@

error_R��O?

learning_rate_1�8����I       6%�	��K���A�*;


total_loss��@

error_R3�C?

learning_rate_1�8��rI       6%�	��K���A�*;


total_loss�ڼ@

error_RӽT?

learning_rate_1�8�sI       6%�	�:K���A�*;


total_loss�t�@

error_R�OX?

learning_rate_1�8��� I       6%�	1�K���A�*;


total_loss���@

error_RhJ?

learning_rate_1�8ʉ�gI       6%�	��K���A�*;


total_losst@�@

error_R��8?

learning_rate_1�8�|�I       6%�	� K���A�*;


total_loss,��@

error_Rf�K?

learning_rate_1�8ܰ$I       6%�	�eK���A�*;


total_lossb�@

error_R|F?

learning_rate_1�8�UsI       6%�	��K���A�*;


total_lossW�A

error_R��G?

learning_rate_1�8��P�I       6%�	H�K���A�*;


total_loss��@

error_RR4G?

learning_rate_1�8���JI       6%�	�0K���A�*;


total_loss�G�@

error_R|S?

learning_rate_1�8�`�I       6%�	�tK���A�*;


total_loss7g�@

error_R�D?

learning_rate_1�8|���I       6%�	T�K���A�*;


total_loss�͗@

error_R.r=?

learning_rate_1�8H��I       6%�	 K���A�*;


total_loss�s@

error_R�Y?

learning_rate_1�8ݬ�I       6%�	PK���A�*;


total_loss�׈@

error_RdT?

learning_rate_1�8�r��I       6%�	!�K���A�*;


total_lossL��@

error_R4D>?

learning_rate_1�8��NI       6%�	�K���A�*;


total_lossw�@

error_Ra�E?

learning_rate_1�8&��I       6%�	$K���A�*;


total_loss_{�@

error_R�a?

learning_rate_1�8��y�I       6%�	�fK���A�*;


total_loss�;v@

error_R�H?

learning_rate_1�8�8��I       6%�	1�K���A�*;


total_loss���@

error_Ri�N?

learning_rate_1�8���I       6%�	��K���A�*;


total_loss�@

error_R��??

learning_rate_1�81�	�I       6%�	3,K���A�*;


total_loss�OA

error_R��D?

learning_rate_1�8���I       6%�	�mK���A�*;


total_loss�mA

error_R�O?

learning_rate_1�8z�I       6%�	��K���A�*;


total_loss4��@

error_R�GR?

learning_rate_1�85�7�I       6%�	��K���A�*;


total_loss*��@

error_RY:?

learning_rate_1�8<�I       6%�	�: K���A�*;


total_loss��@

error_R�I?

learning_rate_1�8��%I       6%�	҄ K���A�*;


total_loss�A

error_R$�E?

learning_rate_1�8�xvmI       6%�	o� K���A�*;


total_lossCt�@

error_R�??

learning_rate_1�8�FdI       6%�	!K���A�*;


total_lossɥ�@

error_R�X?

learning_rate_1�8�VKI       6%�	�b!K���A�*;


total_loss)k�@

error_Rʄc?

learning_rate_1�87S4�I       6%�	��!K���A�*;


total_loss_��@

error_R �W?

learning_rate_1�8��|�I       6%�	��!K���A�*;


total_loss&N�@

error_R��F?

learning_rate_1�8}�(�I       6%�	�@"K���A�*;


total_lossi�@

error_R�[:?

learning_rate_1�85��uI       6%�	T�"K���A�*;


total_loss`m�@

error_R�H?

learning_rate_1�81q�gI       6%�	��"K���A�*;


total_loss,�@

error_R;zX?

learning_rate_1�8���I       6%�	K)#K���A�*;


total_lossw�@

error_RM->?

learning_rate_1�8 ""`I       6%�	�r#K���A�*;


total_loss ��@

error_R.�F?

learning_rate_1�8�g�tI       6%�	6�#K���A�*;


total_loss��@

error_R��T?

learning_rate_1�8��eI       6%�	3�#K���A�*;


total_loss�A

error_R�>F?

learning_rate_1�8ԳfI       6%�	�=$K���A�*;


total_loss��/A

error_R��J?

learning_rate_1�8*��'I       6%�	Q�$K���A�*;


total_loss�Q�@

error_Rȫ;?

learning_rate_1�8kb�I       6%�	��$K���A�*;


total_loss�b�@

error_R��G?

learning_rate_1�8�B��I       6%�	�%K���A�*;


total_lossSK�@

error_R�^?

learning_rate_1�8�HI       6%�	�Z%K���A�*;


total_lossxp�@

error_Rd�N?

learning_rate_1�8�=8@I       6%�	p�%K���A�*;


total_loss1ǫ@

error_R@U?

learning_rate_1�8��I       6%�	�%K���A�*;


total_loss6�@

error_R��]?

learning_rate_1�8l1rI       6%�	�#&K���A�*;


total_loss8��@

error_R�=M?

learning_rate_1�8�(gI       6%�	�k&K���A�*;


total_loss@��@

error_RܖX?

learning_rate_1�8^ ��I       6%�	�&K���A�*;


total_loss���@

error_R��R?

learning_rate_1�8���=I       6%�	��&K���A�*;


total_lossf��@

error_R�D?

learning_rate_1�8�5��I       6%�	"A'K���A�*;


total_loss��@

error_R�c?

learning_rate_1�8>t,�I       6%�	�'K���A�*;


total_lossS_�@

error_RAR[?

learning_rate_1�8���I       6%�	:�'K���A�*;


total_loss�X�@

error_R@�O?

learning_rate_1�8P�I       6%�	0(K���A�*;


total_loss!��@

error_R��A?

learning_rate_1�8��I       6%�	�X(K���A�*;


total_loss�>�@

error_RR	J?

learning_rate_1�8�I��I       6%�	��(K���A�*;


total_lossDE�@

error_RJ�8?

learning_rate_1�8Y�+I       6%�	�(K���A�*;


total_loss�h�@

error_R�1E?

learning_rate_1�8a0EI       6%�	�M)K���A�*;


total_loss���@

error_R#P?

learning_rate_1�84��UI       6%�	C�)K���A�*;


total_loss���@

error_R�hG?

learning_rate_1�8�SI       6%�	S�)K���A�*;


total_loss���@

error_RE�U?

learning_rate_1�8W��I       6%�	%*K���A�*;


total_loss:�@

error_RZIM?

learning_rate_1�8&&��I       6%�	�h*K���A�*;


total_loss!W�@

error_R.�K?

learning_rate_1�8��&I       6%�	8�*K���A�*;


total_loss�w�@

error_R!`Q?

learning_rate_1�8���SI       6%�	��*K���A�*;


total_losso��@

error_R��V?

learning_rate_1�8���eI       6%�	�B+K���A�*;


total_loss�$�@

error_R�B?

learning_rate_1�8�j(�I       6%�	w�+K���A�*;


total_loss6J�@

error_R��E?

learning_rate_1�8kˇ�I       6%�	+�+K���A�*;


total_loss�4�@

error_R��H?

learning_rate_1�8s�2I       6%�	],K���A�*;


total_loss�a�@

error_R�"5?

learning_rate_1�8���eI       6%�	5_,K���A�*;


total_lossn��@

error_Rܼ]?

learning_rate_1�8A1�I       6%�	ե,K���A�*;


total_loss�U�@

error_R�JL?

learning_rate_1�8˾,�I       6%�	��,K���A�*;


total_loss�7�@

error_R�xR?

learning_rate_1�8����I       6%�	�,-K���A�*;


total_lossLdA

error_R�WL?

learning_rate_1�8EDȤI       6%�	Uo-K���A�*;


total_loss֎@

error_R
mO?

learning_rate_1�8�~u�I       6%�	H�-K���A�*;


total_loss���@

error_R�F?

learning_rate_1�8E��I       6%�	��-K���A�*;


total_loss���@

error_R�C?

learning_rate_1�8���I       6%�	�:.K���A�*;


total_loss˞@

error_R�W?

learning_rate_1�8���I       6%�	�}.K���A�*;


total_loss��@

error_RؽR?

learning_rate_1�8t�I       6%�	_�.K���A�*;


total_loss�K�@

error_R��N?

learning_rate_1�8��o�I       6%�	/K���A�*;


total_loss]��@

error_RV�b?

learning_rate_1�8�3FwI       6%�	+M/K���A�*;


total_lossz݌@

error_R �S?

learning_rate_1�8��I       6%�	��/K���A�*;


total_loss���@

error_R�Q?

learning_rate_1�8p�;I       6%�	��/K���A�*;


total_loss�h�@

error_R�sV?

learning_rate_1�8X���I       6%�	o0K���A�*;


total_loss.X�@

error_R��R?

learning_rate_1�8��DI       6%�	�X0K���A�*;


total_loss�*�@

error_R��R?

learning_rate_1�8d�1I       6%�	۝0K���A�*;


total_loss���@

error_R��U?

learning_rate_1�8[x�I       6%�	��0K���A�*;


total_lossvT�@

error_RrqB?

learning_rate_1�8Ƒ�XI       6%�	�)1K���A�*;


total_loss-�A

error_R��Y?

learning_rate_1�8fm�/I       6%�	op1K���A�*;


total_lossI��@

error_R��F?

learning_rate_1�8 N��I       6%�	5�1K���A�*;


total_loss��@

error_Rv�H?

learning_rate_1�8��X�I       6%�	A�1K���A�*;


total_loss�*�@

error_RV�\?

learning_rate_1�8��=I       6%�	v:2K���A�*;


total_loss��@

error_R�U?

learning_rate_1�8�	�ZI       6%�	!{2K���A�*;


total_lossn��@

error_R�:Z?

learning_rate_1�8��'�I       6%�	�2K���A�*;


total_lossc��@

error_R=�O?

learning_rate_1�8O��7I       6%�	�3K���A�*;


total_loss���@

error_R. O?

learning_rate_1�8�nJ@I       6%�	�V3K���A�*;


total_losslj�@

error_R�EK?

learning_rate_1�8P[��I       6%�	p�3K���A�*;


total_loss ��@

error_R��N?

learning_rate_1�8���I       6%�	��3K���A�*;


total_loss��@

error_R�WX?

learning_rate_1�8;ASI       6%�	�@4K���A�*;


total_lossP�@

error_R��F?

learning_rate_1�8)CeI       6%�	�4K���A�*;


total_loss<i�@

error_R��5?

learning_rate_1�8��2�I       6%�	)�4K���A�*;


total_lossT'�@

error_R��W?

learning_rate_1�8T{�PI       6%�	c5K���A�*;


total_loss, �@

error_R�]?

learning_rate_1�8�a�I       6%�	�]5K���A�*;


total_lossN)�@

error_R�KA?

learning_rate_1�8eo��I       6%�	�5K���A�*;


total_loss
��@

error_R��6?

learning_rate_1�8E�F�I       6%�	��5K���A�*;


total_loss{*�@

error_R�"U?

learning_rate_1�8���I       6%�	�-6K���A�*;


total_loss���@

error_R�
S?

learning_rate_1�8qK;eI       6%�	�r6K���A�*;


total_lossS�@

error_R�^?

learning_rate_1�8�K/�I       6%�	`�6K���A�*;


total_lossI��@

error_R��M?

learning_rate_1�8��\yI       6%�	��6K���A�*;


total_loss�;�@

error_Rt�H?

learning_rate_1�8�S,�I       6%�	B7K���A�*;


total_loss�ɨ@

error_R�&P?

learning_rate_1�8�4�I       6%�	��7K���A�*;


total_loss���@

error_R��G?

learning_rate_1�8��;I       6%�	��7K���A�*;


total_lossc��@

error_R�<T?

learning_rate_1�8w �I       6%�		8K���A�*;


total_loss��@

error_Rj�S?

learning_rate_1�8~�QqI       6%�	2X8K���A�*;


total_loss�0�@

error_R[S?

learning_rate_1�8;ȧUI       6%�	��8K���A�*;


total_lossM:�@

error_R�P?

learning_rate_1�8<��I       6%�	��8K���A�*;


total_lossn�@

error_R��]?

learning_rate_1�8��2ZI       6%�	�C9K���A�*;


total_loss#��@

error_R��W?

learning_rate_1�8"b �I       6%�	�9K���A�*;


total_loss�D�@

error_R\yR?

learning_rate_1�8�>�I       6%�	��9K���A�*;


total_loss7��@

error_R�\D?

learning_rate_1�8�HI       6%�	X:K���A�*;


total_lossZi�@

error_R��Q?

learning_rate_1�8��B7I       6%�	^:K���A�*;


total_loss�{�@

error_RoM?

learning_rate_1�8]��I       6%�	��:K���A�*;


total_lossfH�@

error_R�HY?

learning_rate_1�88���I       6%�	u�:K���A�*;


total_loss{��@

error_R�A?

learning_rate_1�8�ɝI       6%�	� ;K���A�*;


total_loss�V�@

error_R�~;?

learning_rate_1�8G�yI       6%�	�f;K���A�*;


total_loss���@

error_RȄS?

learning_rate_1�8ŵ"I       6%�	�;K���A�*;


total_loss��@

error_R`�F?

learning_rate_1�8�aG�I       6%�	i�;K���A�*;


total_loss�k	A

error_Rv�P?

learning_rate_1�8:��>I       6%�	�4<K���A�*;


total_loss�ζ@

error_R�V?

learning_rate_1�8�b��I       6%�	)y<K���A�*;


total_loss܌�@

error_R�ZG?

learning_rate_1�8y/x�I       6%�	�<K���A�*;


total_loss��A

error_Rz"Z?

learning_rate_1�8EH��I       6%�	�=K���A�*;


total_loss{��@

error_R�S?

learning_rate_1�8ҿI       6%�	�I=K���A�*;


total_loss@��@

error_R�@:?

learning_rate_1�8[OI       6%�	��=K���A�*;


total_lossI��@

error_R��R?

learning_rate_1�8HUI       6%�	��=K���A�*;


total_loss�Ũ@

error_R4�e?

learning_rate_1�8�h��I       6%�	&>K���A�*;


total_loss�m�@

error_R18T?

learning_rate_1�8�-�`I       6%�	[>K���A�*;


total_loss�`�@

error_R�
b?

learning_rate_1�8k�h2I       6%�	T�>K���A�*;


total_loss�N�@

error_RT\H?

learning_rate_1�8h�[I       6%�	��>K���A�*;


total_loss7��@

error_RA?

learning_rate_1�8Ji��I       6%�	!+?K���A�*;


total_loss�D�@

error_R��]?

learning_rate_1�8XV�UI       6%�	�n?K���A�*;


total_loss�´@

error_R{*H?

learning_rate_1�8�&I       6%�	Z�?K���A�*;


total_loss@%�@

error_R��=?

learning_rate_1�8���I       6%�	�@K���A�*;


total_loss�¦@

error_R��I?

learning_rate_1�8����I       6%�	�E@K���A�*;


total_loss��@

error_Rx�S?

learning_rate_1�8�w�+I       6%�	#�@K���A�*;


total_lossI�p@

error_R|�F?

learning_rate_1�8���lI       6%�	��@K���A�*;


total_loss�7�@

error_R_ZC?

learning_rate_1�8�<"�I       6%�	�AK���A�*;


total_loss���@

error_R�^?

learning_rate_1�8�$��I       6%�	XAK���A�*;


total_lossj�@

error_R��Q?

learning_rate_1�8݄��I       6%�	�AK���A�*;


total_lossaJ�@

error_RH�b?

learning_rate_1�8e�o�I       6%�	��AK���A�*;


total_loss�V�@

error_R.UU?

learning_rate_1�8z��RI       6%�	�&BK���A�*;


total_loss䊹@

error_R��K?

learning_rate_1�8�ޕI       6%�	�jBK���A�*;


total_losssҚ@

error_R=�Q?

learning_rate_1�8����I       6%�	��BK���A�*;


total_loss��@

error_RrXF?

learning_rate_1�8*HwI       6%�	�BK���A�*;


total_loss.��@

error_R��]?

learning_rate_1�8j��UI       6%�	<>CK���A�*;


total_loss�W�@

error_R�U?

learning_rate_1�8MS��I       6%�	��CK���A�*;


total_loss���@

error_R3/F?

learning_rate_1�8j�]JI       6%�	��CK���A�*;


total_loss��@

error_R��H?

learning_rate_1�8h��<I       6%�	 DK���A�*;


total_loss.G�@

error_R��E?

learning_rate_1�8�)I       6%�	;[DK���A�*;


total_lossI��@

error_RŁX?

learning_rate_1�8#8�I       6%�	l�DK���A�*;


total_lossxp�@

error_R-AC?

learning_rate_1�8�]I       6%�	��DK���A�*;


total_lossz�@

error_R]�R?

learning_rate_1�8��LI       6%�	�%EK���A�*;


total_loss��@

error_RHAT?

learning_rate_1�8�W�}I       6%�	�jEK���A�*;


total_loss��@

error_Rʸ??

learning_rate_1�8���I       6%�	޵EK���A�*;


total_loss��@

error_RE1B?

learning_rate_1�8�4�9I       6%�	��EK���A�*;


total_loss��@

error_RvGR?

learning_rate_1�8C
E�I       6%�	�BFK���A�*;


total_loss*��@

error_R�AR?

learning_rate_1�8��N�I       6%�	��FK���A�*;


total_loss�w�@

error_Ra�Q?

learning_rate_1�8ÄOAI       6%�	��FK���A�*;


total_loss/��@

error_R!�=?

learning_rate_1�88��I       6%�	�GK���A�*;


total_loss/w�@

error_R!�P?

learning_rate_1�8 �j�I       6%�	iiGK���A�*;


total_lossa��@

error_R�{E?

learning_rate_1�8>�W�I       6%�	ֱGK���A�*;


total_loss_d�@

error_RiCY?

learning_rate_1�8�l�7I       6%�	��GK���A�*;


total_loss�A

error_RaX=?

learning_rate_1�8��hnI       6%�	O?HK���A�*;


total_loss<��@

error_R�,P?

learning_rate_1�8;��AI       6%�	U�HK���A�*;


total_loss,{A

error_RnL?

learning_rate_1�8IB�KI       6%�	��HK���A�*;


total_loss���@

error_RCK?

learning_rate_1�8ҙI       6%�	5IK���A�*;


total_loss��@

error_R$�D?

learning_rate_1�8.�	�I       6%�	E{IK���A�*;


total_loss�@

error_R%�L?

learning_rate_1�8_B�KI       6%�	�IK���A�*;


total_loss���@

error_R�T?

learning_rate_1�8��Z@I       6%�	)JK���A�*;


total_loss���@

error_R@n\?

learning_rate_1�8|��I       6%�	�LJK���A�*;


total_loss��@

error_R��T?

learning_rate_1�87${?I       6%�	��JK���A�*;


total_loss�=�@

error_R�eV?

learning_rate_1�8M�",I       6%�	��JK���A�*;


total_losszR�@

error_R}�N?

learning_rate_1�8v֞`I       6%�	;7KK���A�*;


total_lossV�@

error_R@d??

learning_rate_1�8?��I       6%�	��KK���A�*;


total_lossq�@

error_R3�I?

learning_rate_1�8���8I       6%�	��KK���A�*;


total_lossxe�@

error_Rh�B?

learning_rate_1�8C�cI       6%�	�ALK���A�*;


total_loss��@

error_R��=?

learning_rate_1�8;rI       6%�	�LK���A�*;


total_loss�G�@

error_R�D?

learning_rate_1�8v\I       6%�	z�LK���A�*;


total_loss`�A

error_Rq�k?

learning_rate_1�8Ա!I       6%�	MK���A�*;


total_loss���@

error_R�+P?

learning_rate_1�8<�|�I       6%�	�`MK���A�*;


total_loss%��@

error_R�J?

learning_rate_1�8YV�I       6%�	��MK���A�*;


total_loss��@

error_R`rP?

learning_rate_1�886+�I       6%�	��MK���A�*;


total_loss��%A

error_R�oP?

learning_rate_1�8�s
lI       6%�	�9NK���A�*;


total_lossNS�@

error_R�K?

learning_rate_1�8�W�I       6%�	ބNK���A�*;


total_lossx��@

error_RsxY?

learning_rate_1�8II       6%�	d�NK���A�*;


total_loss�C�@

error_R(m)?

learning_rate_1�8!`��I       6%�	�OK���A�*;


total_loss. �@

error_R�f?

learning_rate_1�80(	�I       6%�	�dOK���A�*;


total_loss�9�@

error_R��L?

learning_rate_1�8�>RI       6%�	s�OK���A�*;


total_loss=i�@

error_R�lA?

learning_rate_1�8mȸI       6%�	t�OK���A�*;


total_lossd�@

error_RC�W?

learning_rate_1�8.��I       6%�	�<PK���A�*;


total_lossd.�@

error_R��N?

learning_rate_1�8����I       6%�	͈PK���A�*;


total_loss�ż@

error_R�e?

learning_rate_1�8����I       6%�	��PK���A�*;


total_loss:	A

error_R6�S?

learning_rate_1�8���AI       6%�	,QK���A�*;


total_lossO��@

error_R,UT?

learning_rate_1�8���5I       6%�	�[QK���A�*;


total_lossj��@

error_R��U?

learning_rate_1�8��NI       6%�	�QK���A�*;


total_loss8��@

error_R�$f?

learning_rate_1�8?FO�I       6%�	��QK���A�*;


total_loss�A�@

error_R�yW?

learning_rate_1�8`�I       6%�	0(RK���A�*;


total_loss;eA

error_R�h?

learning_rate_1�8�0+�I       6%�	!mRK���A�*;


total_loss���@

error_R�'=?

learning_rate_1�8!x�I       6%�	2�RK���A�*;


total_loss�<�@

error_R=�F?

learning_rate_1�8���\I       6%�	:�RK���A�*;


total_loss���@

error_R=UF?

learning_rate_1�8D¬�I       6%�	�JSK���A�*;


total_loss;U�@

error_R��K?

learning_rate_1�8��ČI       6%�	�SK���A�*;


total_loss�i�@

error_R`7J?

learning_rate_1�8�1�I       6%�	��SK���A�*;


total_loss�3�@

error_R�4G?

learning_rate_1�8ء[I       6%�	�TK���A�*;


total_loss��@

error_R;{k?

learning_rate_1�8����I       6%�	�RTK���A�*;


total_lossaX�@

error_R��R?

learning_rate_1�8{=G�I       6%�	�TK���A�*;


total_loss;#�@

error_R�O?

learning_rate_1�8�"��I       6%�	��TK���A�*;


total_lossԧ�@

error_Rc�>?

learning_rate_1�8��ԇI       6%�	X&UK���A�*;


total_loss��@

error_R ?R?

learning_rate_1�8w��I       6%�	�fUK���A�*;


total_lossi*�@

error_R��S?

learning_rate_1�8t�8UI       6%�	M�UK���A�*;


total_lossBA

error_ROIH?

learning_rate_1�8�W�I       6%�	m�UK���A�*;


total_lossI��@

error_R�	M?

learning_rate_1�8���I       6%�	=VK���A�*;


total_loss��A

error_R�|V?

learning_rate_1�86�Z�I       6%�	�VK���A�*;


total_loss��A

error_R��I?

learning_rate_1�8�"h4I       6%�	r�VK���A�*;


total_lossO��@

error_R��C?

learning_rate_1�8N�s�I       6%�	WK���A�*;


total_loss�C A

error_RS�Z?

learning_rate_1�8L�<�I       6%�	JWK���A�*;


total_loss��@

error_R�d?

learning_rate_1�8��nI       6%�	��WK���A�*;


total_loss@��@

error_RW�Y?

learning_rate_1�8�jV�I       6%�	@�WK���A�*;


total_loss
��@

error_R�H?

learning_rate_1�8����I       6%�	XK���A�*;


total_loss�X�@

error_R+P?

learning_rate_1�8�5�I       6%�	:\XK���A�*;


total_loss��f@

error_R��Q?

learning_rate_1�8,B��I       6%�	g�XK���A�*;


total_lossڸ�@

error_R�ld?

learning_rate_1�8����I       6%�	x�XK���A�*;


total_loss;�@

error_R�L?

learning_rate_1�8��v�I       6%�	�QYK���A�*;


total_loss*]�@

error_R�AO?

learning_rate_1�8�hTWI       6%�	8�YK���A�*;


total_loss�&�@

error_R|�K?

learning_rate_1�8�1/I       6%�	��YK���A�*;


total_loss�2R@

error_R8@?

learning_rate_1�8����I       6%�	."ZK���A�*;


total_loss4}�@

error_R	�]?

learning_rate_1�8��4�I       6%�	�gZK���A�*;


total_loss�@

error_RA�R?

learning_rate_1�82&��I       6%�	��ZK���A�*;


total_loss���@

error_R.qX?

learning_rate_1�8 �`-I       6%�	(�ZK���A�*;


total_loss�Ѣ@

error_R)�C?

learning_rate_1�8,O݉I       6%�	b2[K���A�*;


total_loss6�@

error_R\[P?

learning_rate_1�8G?��I       6%�	3v[K���A�*;


total_loss	��@

error_Rl�K?

learning_rate_1�8bײEI       6%�	}�[K���A�*;


total_loss�S�@

error_R��0?

learning_rate_1�8-y(9I       6%�	��[K���A�*;


total_loss���@

error_R�V?

learning_rate_1�8��>`I       6%�	C\K���A�*;


total_lossW��@

error_R -N?

learning_rate_1�8��I       6%�	�\K���A�*;


total_loss ί@

error_R��L?

learning_rate_1�8-%�=I       6%�	��\K���A�*;


total_loss@\�@

error_R:�U?

learning_rate_1�8�)�I       6%�	4]K���A�*;


total_lossO��@

error_R��K?

learning_rate_1�8P��I       6%�	�S]K���A�*;


total_loss���@

error_R��=?

learning_rate_1�8��c�I       6%�	З]K���A�*;


total_loss3�i@

error_RRoN?

learning_rate_1�8p�?�I       6%�	��]K���A�*;


total_loss�h�@

error_R�[P?

learning_rate_1�8?·�I       6%�	$!^K���A�*;


total_loss|V�@

error_R�G?

learning_rate_1�8�:.I       6%�	�f^K���A�*;


total_lossm��@

error_R�>:?

learning_rate_1�8�]&�I       6%�	�^K���A�*;


total_loss֘@

error_R}�W?

learning_rate_1�8��2�I       6%�	E�^K���A�*;


total_loss�|�@

error_R��N?

learning_rate_1�8��ZI       6%�	�;_K���A�*;


total_loss�%�@

error_Rh�H?

learning_rate_1�8~��^I       6%�	[�_K���A�*;


total_lossZ`�@

error_R��V?

learning_rate_1�8� UI       6%�	p�_K���A�*;


total_lossɳ�@

error_R}R?

learning_rate_1�8��|I       6%�	G`K���A�*;


total_loss��@

error_R�U?

learning_rate_1�8�}ajI       6%�	3j`K���A�*;


total_loss�X�@

error_R@rN?

learning_rate_1�8�T�I       6%�	.�`K���A�*;


total_lossה@

error_R=1O?

learning_rate_1�8��Q�I       6%�	��`K���A�*;


total_loss���@

error_RsI[?

learning_rate_1�8MJR�I       6%�	�7aK���A�*;


total_lossw�@

error_R��[?

learning_rate_1�8�@I       6%�	�yaK���A�*;


total_loss��A

error_R)N?

learning_rate_1�8�غI       6%�	�aK���A�*;


total_loss�^�@

error_Rf�;?

learning_rate_1�8�r�I       6%�	��aK���A�*;


total_loss1�@

error_R�M?

learning_rate_1�8Pd�ZI       6%�	(?bK���A�*;


total_loss��@

error_R��H?

learning_rate_1�8��k�I       6%�	ԂbK���A�*;


total_loss�@

error_R�ke?

learning_rate_1�8-���I       6%�	"�bK���A�*;


total_loss�f#A

error_R�I?

learning_rate_1�8LU��I       6%�	�
cK���A�*;


total_loss��U@

error_R�;?

learning_rate_1�8U�l�I       6%�	;UcK���A�*;


total_loss;ȏ@

error_R��Q?

learning_rate_1�8�II       6%�	A�cK���A�*;


total_loss�7�@

error_R:	N?

learning_rate_1�8�$[�I       6%�	��cK���A�*;


total_lossq��@

error_R��K?

learning_rate_1�8�]��I       6%�	 dK���A�*;


total_lossW�@

error_R��F?

learning_rate_1�8�EP_I       6%�	"bdK���A�*;


total_loss��@

error_RdIV?

learning_rate_1�8t��I       6%�	�tgK���A�*;


total_loss�n�@

error_R��??

learning_rate_1�8�Ϋ�I       6%�	0�gK���A�*;


total_loss��@

error_R�\U?

learning_rate_1�8lY�I       6%�	�hK���A�*;


total_lossx��@

error_R\�Q?

learning_rate_1�8���I       6%�	RhK���A�*;


total_loss��@

error_R�KO?

learning_rate_1�8b� �I       6%�	��hK���A�*;


total_losso�@

error_R�CR?

learning_rate_1�8�h�I       6%�	��hK���A�*;


total_loss�%�@

error_R�_I?

learning_rate_1�8|��I       6%�	QJiK���A�*;


total_loss�Ҁ@

error_R U?

learning_rate_1�8�o�I       6%�	i�iK���A�*;


total_loss �@

error_R|)Y?

learning_rate_1�8���5I       6%�	/�iK���A�*;


total_loss3��@

error_R�<?

learning_rate_1�8I�5I       6%�	�!jK���A�*;


total_loss�,�@

error_R��j?

learning_rate_1�8v�{�I       6%�	 pjK���A�*;


total_loss`�@

error_R<p>?

learning_rate_1�8IN��I       6%�	�jK���A�*;


total_loss)��@

error_RnT?

learning_rate_1�8��yI       6%�	= kK���A�*;


total_loss��@

error_R��Y?

learning_rate_1�8P��fI       6%�	xBkK���A�*;


total_lossNd�@

error_R��I?

learning_rate_1�8���I       6%�	��kK���A�*;


total_loss?�@

error_R_UK?

learning_rate_1�8����I       6%�	4�kK���A�*;


total_loss1��@

error_R�H?

learning_rate_1�8؟ �I       6%�	�
lK���A�*;


total_loss�.�@

error_R*oG?

learning_rate_1�8����I       6%�	�OlK���A�*;


total_lossQ�v@

error_Rq�T?

learning_rate_1�8rX��I       6%�	&�lK���A�*;


total_lossvF�@

error_R=K?

learning_rate_1�8�)�I       6%�	��lK���A�*;


total_loss�]�@

error_R4�B?

learning_rate_1�8��4�I       6%�	�mK���A�*;


total_loss�*y@

error_RҥN?

learning_rate_1�80�$0I       6%�	fmK���A�*;


total_loss���@

error_R)DO?

learning_rate_1�8vi&aI       6%�	��mK���A�*;


total_loss��@

error_R�VO?

learning_rate_1�8��3,I       6%�	$�mK���A�*;


total_lossa��@

error_RH!K?

learning_rate_1�8U!�.I       6%�	~?nK���A�*;


total_loss���@

error_R�5G?

learning_rate_1�8�wөI       6%�	z�nK���A�*;


total_loss$��@

error_R�M?

learning_rate_1�8O�EI       6%�	��nK���A�*;


total_loss��@

error_R$�=?

learning_rate_1�88)��I       6%�	�oK���A�*;


total_lossQ:�@

error_Rr�I?

learning_rate_1�81!�bI       6%�	eQoK���A�*;


total_loss���@

error_R��F?

learning_rate_1�8y�z�I       6%�	�oK���A�*;


total_lossSi�@

error_R�K?

learning_rate_1�8��a�I       6%�	��oK���A�*;


total_loss�=�@

error_R�7L?

learning_rate_1�8�Q�I       6%�	�#pK���A�*;


total_loss�0�@

error_RE�Y?

learning_rate_1�8S��I       6%�	jmpK���A�*;


total_lossn�@

error_RC�P?

learning_rate_1�8ka�UI       6%�	�pK���A�*;


total_lossք�@

error_R(L?

learning_rate_1�8&��I       6%�	wqK���A�*;


total_loss�+�@

error_R�Q?

learning_rate_1�8�W��I       6%�	rQqK���A�*;


total_loss42�@

error_R�R?

learning_rate_1�8�
�I       6%�	z�qK���A�*;


total_loss�@

error_R�i?

learning_rate_1�8�4I       6%�	��qK���A�*;


total_lossy��@

error_R��_?

learning_rate_1�8^|�I       6%�	/rK���A�*;


total_lossױ�@

error_RR[O?

learning_rate_1�8e�I       6%�	r�rK���A�*;


total_loss��@

error_R/ad?

learning_rate_1�8Z��~I       6%�	y�rK���A�*;


total_loss��@

error_RȆ:?

learning_rate_1�8W�^�I       6%�	+sK���A�*;


total_lossJz A

error_R�E?

learning_rate_1�8�]��I       6%�	:|sK���A�*;


total_loss�X�@

error_R�D?

learning_rate_1�8�)�I       6%�	'�sK���A�*;


total_lossss�@

error_R�M?

learning_rate_1�8����I       6%�	&tK���A�*;


total_loss��}@

error_R?NU?

learning_rate_1�8��X[I       6%�	intK���A�*;


total_loss�V�@

error_R<O?

learning_rate_1�8S:nWI       6%�	Z�tK���A�*;


total_lossچ�@

error_R�Q?

learning_rate_1�8�x��I       6%�		uK���A�*;


total_lossԨ�@

error_R��J?

learning_rate_1�8�e��I       6%�	@juK���A�*;


total_lossl�@

error_R8tE?

learning_rate_1�8�8��I       6%�	?�uK���A�*;


total_lossN��@

error_R&�X?

learning_rate_1�8I���I       6%�	 �uK���A�*;


total_lossٸ@

error_Rne?

learning_rate_1�8b1$4I       6%�	�<vK���A�*;


total_loss�r�@

error_RtL?

learning_rate_1�8�܏�I       6%�	��vK���A�*;


total_lossd�@

error_RL�T?

learning_rate_1�8�X=I       6%�	��vK���A�*;


total_lossL��@

error_R@[?

learning_rate_1�8�(�I       6%�	cwK���A�*;


total_loss��@

error_R�W?

learning_rate_1�8�ׯI       6%�	cwK���A�*;


total_loss���@

error_RVL?

learning_rate_1�8{��I       6%�	j�wK���A�*;


total_loss?��@

error_R��J?

learning_rate_1�8QX��I       6%�	�
xK���A�*;


total_loss�e�@

error_R�IS?

learning_rate_1�8�j �I       6%�	?OxK���A�*;


total_loss�@

error_RM_]?

learning_rate_1�8�#cKI       6%�	љxK���A�*;


total_lossn$�@

error_R
LX?

learning_rate_1�8y�I       6%�	��xK���A�*;


total_lossj\�@

error_R�R?

learning_rate_1�8��RfI       6%�	BMyK���A�*;


total_lossE'�@

error_R,�`?

learning_rate_1�8��	�I       6%�	��yK���A�*;


total_lossŠ�@

error_R�7O?

learning_rate_1�8�;*SI       6%�	��yK���A�*;


total_loss)�@

error_RL,K?

learning_rate_1�8܀��I       6%�	�(zK���A�*;


total_lossW�l@

error_R;:?

learning_rate_1�8�&�I       6%�	�mzK���A�*;


total_loss��@

error_R��M?

learning_rate_1�8�fN-I       6%�	 �zK���A�*;


total_loss��A

error_R]K?

learning_rate_1�8���XI       6%�	�{K���A�*;


total_loss�V�@

error_R�H?

learning_rate_1�8v��I       6%�	Q{K���A�*;


total_loss}?�@

error_R@�F?

learning_rate_1�8v�
(I       6%�	��{K���A�*;


total_loss���@

error_R?�K?

learning_rate_1�8��# I       6%�	H�{K���A�*;


total_lossr��@

error_R�\8?

learning_rate_1�8t�<EI       6%�	�"|K���A�*;


total_losss�A

error_RrW?

learning_rate_1�8�_�I       6%�	�f|K���A�*;


total_loss�r@

error_RnN?

learning_rate_1�8����I       6%�	T�|K���A�*;


total_loss�e�@

error_RR�=?

learning_rate_1�8iє�I       6%�	��|K���A�*;


total_loss�ʝ@

error_Rf�T?

learning_rate_1�8�nJ�I       6%�	b>}K���A�*;


total_loss���@

error_RW?

learning_rate_1�8��I       6%�	H�}K���A�*;


total_lossl��@

error_RH�L?

learning_rate_1�8`���I       6%�	B�}K���A�*;


total_lossX��@

error_R��N?

learning_rate_1�8"�R\I       6%�	YC~K���A�*;


total_loss�6�@

error_R�%d?

learning_rate_1�8 e�I       6%�	�~K���A�*;


total_loss��@

error_R[E?

learning_rate_1�8�6N�I       6%�	?�~K���A�*;


total_loss���@

error_R��X?

learning_rate_1�8���I       6%�	�K���A�*;


total_loss!�@

error_RcK?

learning_rate_1�8�\s]I       6%�	�]K���A�*;


total_loss`F�@

error_RW^\?

learning_rate_1�8ˇ�-I       6%�	2�K���A�*;


total_loss��@

error_R��M?

learning_rate_1�8�+u�I       6%�	��K���A�*;


total_loss�*�@

error_Rv�=?

learning_rate_1�8oK!�I       6%�	76�K���A�*;


total_loss�K�@

error_R��[?

learning_rate_1�8wb��I       6%�	�|�K���A�*;


total_lossJ��@

error_R��K?

learning_rate_1�86U��I       6%�	K���A�*;


total_loss�T�@

error_RfVS?

learning_rate_1�8.rI       6%�	��K���A�*;


total_loss\��@

error_Rl�U?

learning_rate_1�8���PI       6%�	�L�K���A�*;


total_lossd�@

error_R}R?

learning_rate_1�8���I       6%�	���K���A�*;


total_loss&��@

error_R?wO?

learning_rate_1�8���I       6%�	0ӁK���A�*;


total_loss�@

error_R�oI?

learning_rate_1�8��fVI       6%�	��K���A�*;


total_loss�g�@

error_RA�N?

learning_rate_1�8���I       6%�	�[�K���A�*;


total_loss,�@

error_R\n8?

learning_rate_1�8��^�I       6%�	k��K���A�*;


total_loss"d�@

error_R$�_?

learning_rate_1�8&�f`I       6%�	�K���A�*;


total_lossN�h@

error_R)�M?

learning_rate_1�8�mq�I       6%�	5/�K���A�*;


total_loss�1�@

error_R�wV?

learning_rate_1�8����I       6%�	�s�K���A�*;


total_loss#Q�@

error_R!N?

learning_rate_1�84�YI       6%�	5��K���A�*;


total_loss��@

error_R��F?

learning_rate_1�8���iI       6%�	��K���A�*;


total_loss�œ@

error_R��B?

learning_rate_1�8=A�I       6%�	�L�K���A�*;


total_lossg�@

error_R*hG?

learning_rate_1�8qk�I       6%�	���K���A�*;


total_lossQِ@

error_R�F?

learning_rate_1�8%WvI       6%�	ԄK���A�*;


total_loss��@

error_R�xW?

learning_rate_1�8Y�dAI       6%�	��K���A�*;


total_loss�B�@

error_R�Kd?

learning_rate_1�8mҟI       6%�	a�K���A�*;


total_loss�ۘ@

error_Rs�L?

learning_rate_1�8@_I       6%�	���K���A�*;


total_loss��@

error_R�VT?

learning_rate_1�8IoI       6%�	��K���A�*;


total_loss���@

error_R�M?

learning_rate_1�8����I       6%�	F2�K���A�*;


total_loss�>�@

error_RLO?

learning_rate_1�8�{�II       6%�	���K���A�*;


total_loss�@�@

error_R[=a?

learning_rate_1�8���I       6%�	�݆K���A�*;


total_loss@K�@

error_Rq�T?

learning_rate_1�8:��I       6%�	a#�K���A�*;


total_lossҘ�@

error_RJ�L?

learning_rate_1�8|��I       6%�	�l�K���A�*;


total_loss`��@

error_R��L?

learning_rate_1�8����I       6%�	ׇK���A�*;


total_loss�*�@

error_RsMC?

learning_rate_1�8h�I       6%�	!�K���A�*;


total_losss��@

error_R�M?

learning_rate_1�8�÷�I       6%�	�f�K���A�*;


total_loss���@

error_R��I?

learning_rate_1�8��XI       6%�	��K���A�*;


total_lossJ��@

error_R��=?

learning_rate_1�8�i��I       6%�	4	�K���A�*;


total_losso�A

error_Rh�M?

learning_rate_1�8�<�I       6%�	�^�K���A�*;


total_loss�z�@

error_R�L?

learning_rate_1�8�E�I       6%�	G��K���A�*;


total_lossN.�@

error_R�K?

learning_rate_1�8j�w�I       6%�	#�K���A�*;


total_losswV�@

error_RӧJ?

learning_rate_1�8h���I       6%�	�7�K���A�*;


total_loss�c�@

error_R��\?

learning_rate_1�8^�eI       6%�	�~�K���A�*;


total_loss��@

error_R��S?

learning_rate_1�8�tI       6%�	�ĊK���A�*;


total_loss,�'A

error_RQ?

learning_rate_1�8��I       6%�	F�K���A�*;


total_loss�g�@

error_R��Q?

learning_rate_1�8 �wI       6%�	/N�K���A�*;


total_loss`��@

error_R�Q?

learning_rate_1�8���I       6%�	ۖ�K���A�*;


total_loss��@

error_R�jU?

learning_rate_1�8�l I       6%�	x݋K���A�*;


total_lossW�A

error_R�R?

learning_rate_1�8��xuI       6%�	�!�K���A�*;


total_loss�@

error_R��>?

learning_rate_1�8�:�qI       6%�	f�K���A�*;


total_loss/f�@

error_R��C?

learning_rate_1�8��
I       6%�	y��K���A�*;


total_loss<"�@

error_R��F?

learning_rate_1�8�; I       6%�	�K���A�*;


total_lossm��@

error_R�lK?

learning_rate_1�8�.��I       6%�	}/�K���A�*;


total_loss�5�@

error_R@�3?

learning_rate_1�8}�5kI       6%�	�s�K���A�*;


total_lossc:�@

error_Re�O?

learning_rate_1�8�Z�I       6%�	꺍K���A�*;


total_lossz�@

error_R��A?

learning_rate_1�8c��I       6%�	�K���A�*;


total_loss�T}@

error_R��Q?

learning_rate_1�8�[~I       6%�	�F�K���A�*;


total_loss�~�@

error_RM=U?

learning_rate_1�8�n��I       6%�	���K���A�*;


total_loss���@

error_R �X?

learning_rate_1�8w$�KI       6%�	�ˎK���A�*;


total_losst�@

error_R�N?

learning_rate_1�8
I��I       6%�	��K���A�*;


total_loss@��@

error_RaR?

learning_rate_1�8�tI       6%�	�N�K���A�*;


total_losslƬ@

error_R>?

learning_rate_1�8���I       6%�	��K���A�*;


total_loss��@

error_RC�`?

learning_rate_1�8�Fj�I       6%�	ZޏK���A�*;


total_loss<��@

error_R�0E?

learning_rate_1�8����I       6%�	�(�K���A�*;


total_loss���@

error_R}�S?

learning_rate_1�8W5��I       6%�	 s�K���A�*;


total_loss'L�@

error_R&�W?

learning_rate_1�8��wI       6%�	w��K���A�*;


total_loss�Ɏ@

error_R�n=?

learning_rate_1�8�w�I       6%�	��K���A�*;


total_loss�_�@

error_R�?N?

learning_rate_1�8x��I       6%�	�T�K���A�*;


total_loss���@

error_R��K?

learning_rate_1�8��>�I       6%�	=��K���A�*;


total_loss-1�@

error_Ra�U?

learning_rate_1�8��}I       6%�	=�K���A�*;


total_loss��A

error_R�>O?

learning_rate_1�8ҫSI       6%�	p.�K���A�*;


total_loss���@

error_R2�M?

learning_rate_1�8���I       6%�	�t�K���A�*;


total_loss|�@

error_R��R?

learning_rate_1�8�_�mI       6%�	��K���A�*;


total_loss��@

error_RC&V?

learning_rate_1�8o��I       6%�	��K���A�*;


total_loss�u�@

error_R�>=?

learning_rate_1�8vf�1I       6%�	j@�K���A�*;


total_loss��@

error_R�eZ?

learning_rate_1�8�ākI       6%�	���K���A�*;


total_loss�R�@

error_R�@?

learning_rate_1�8��:�I       6%�	NٓK���A�*;


total_loss��@

error_R��K?

learning_rate_1�83R�I       6%�	�$�K���A�*;


total_loss@��@

error_R%W?

learning_rate_1�8�cr{I       6%�	Lk�K���A�*;


total_loss���@

error_R�[?

learning_rate_1�8��LI       6%�	���K���A�*;


total_lossȼA

error_R��^?

learning_rate_1�83_%`I       6%�	���K���A�*;


total_loss३@

error_R;KA?

learning_rate_1�8Q���I       6%�	$;�K���A�*;


total_lossi �@

error_R��T?

learning_rate_1�8��B[I       6%�	�|�K���A�*;


total_lossR��@

error_R��^?

learning_rate_1�8�5�>I       6%�	{��K���A�*;


total_loss&��@

error_R ??

learning_rate_1�8��euI       6%�	 �K���A�*;


total_lossWĜ@

error_R�'k?

learning_rate_1�8���qI       6%�	�K�K���A�*;


total_lossX�@

error_R�{V?

learning_rate_1�8��:�I       6%�	ʖ�K���A�*;


total_loss�׊@

error_R۬;?

learning_rate_1�89���I       6%�	5��K���A�*;


total_loss��@

error_R�F?

learning_rate_1�8K��@I       6%�	-�K���A�*;


total_loss(-�@

error_R1pR?

learning_rate_1�8+�eeI       6%�	]o�K���A�*;


total_loss���@

error_R�T?

learning_rate_1�8�.��I       6%�	\��K���A�*;


total_lossh�@

error_R�,a?

learning_rate_1�8��#�I       6%�	B��K���A�*;


total_loss�LA

error_R�,?

learning_rate_1�8��:vI       6%�	�E�K���A�*;


total_loss���@

error_R�JL?

learning_rate_1�80��I       6%�	���K���A�*;


total_loss��@

error_R��;?

learning_rate_1�8��bHI       6%�	���K���A�*;


total_loss�i�@

error_R�jZ?

learning_rate_1�8����I       6%�	;�K���A�*;


total_lossi�@

error_R��`?

learning_rate_1�8���I       6%�	��K���A�*;


total_lossW��@

error_R�\?

learning_rate_1�8p�gI       6%�	���K���A�*;


total_loss���@

error_R��H?

learning_rate_1�8�P!�I       6%�	��K���A�*;


total_loss�k�@

error_R��^?

learning_rate_1�8�*,qI       6%�	N�K���A�*;


total_loss�K�@

error_R#�@?

learning_rate_1�8�h�ZI       6%�	���K���A�*;


total_loss��@

error_R,XZ?

learning_rate_1�8����I       6%�	�ٚK���A�*;


total_losse�@

error_R�R?

learning_rate_1�8���I       6%�	� �K���A�*;


total_loss�$�@

error_RqqV?

learning_rate_1�8�mI       6%�	�e�K���A�*;


total_loss9<�@

error_R��b?

learning_rate_1�8t��I       6%�	l��K���A�*;


total_loss��@

error_R��M?

learning_rate_1�8%ϻ�I       6%�	��K���A�*;


total_loss�� A

error_R��G?

learning_rate_1�8t;o�I       6%�	5.�K���A�*;


total_loss1e�@

error_R�O?

learning_rate_1�8ے��I       6%�	�r�K���A�*;


total_lossAgA

error_R�@?

learning_rate_1�8S��I       6%�	���K���A�*;


total_loss��A

error_R��I?

learning_rate_1�8��I       6%�	(��K���A�*;


total_loss4B�@

error_R7}V?

learning_rate_1�8W�mI       6%�	�C�K���A�*;


total_loss���@

error_R��L?

learning_rate_1�8�;�I       6%�	��K���A�*;


total_loss��@

error_R*F?

learning_rate_1�8 -/I       6%�	�˝K���A�*;


total_loss?R�@

error_R��S?

learning_rate_1�81F�I       6%�	�K���A�*;


total_loss�7p@

error_R��C?

learning_rate_1�8Ӟ@�I       6%�	kZ�K���A�*;


total_loss��@

error_R�#J?

learning_rate_1�8ٲ��I       6%�	Q��K���A�*;


total_loss�*�@

error_R/*J?

learning_rate_1�8��I       6%�	��K���A�*;


total_lossM	�@

error_R��T?

learning_rate_1�8}�+�I       6%�	�,�K���A�*;


total_loss��%A

error_R_�_?

learning_rate_1�8u�lAI       6%�	3s�K���A�*;


total_lossw[�@

error_RE�O?

learning_rate_1�8�<�:I       6%�	ƻ�K���A�*;


total_lossf�@

error_Ri�\?

learning_rate_1�8�Ǟ/I       6%�	G�K���A�*;


total_loss| �@

error_R�E?

learning_rate_1�8�_(�I       6%�	AC�K���A�*;


total_loss _�@

error_R��L?

learning_rate_1�8��V�I       6%�	��K���A�*;


total_loss��@

error_R��G?

learning_rate_1�8{7M�I       6%�	tݠK���A�*;


total_loss	V�@

error_R�I?

learning_rate_1�8܆ngI       6%�	6�K���A�*;


total_loss�k@

error_R[Y;?

learning_rate_1�8��SI       6%�	/{�K���A�*;


total_lossw�A

error_R̇\?

learning_rate_1�8�"I       6%�	A¡K���A�*;


total_loss.@�@

error_R<�>?

learning_rate_1�8G� I       6%�	� �K���A�*;


total_loss�&�@

error_R��B?

learning_rate_1�8�d�I       6%�	Aj�K���A�*;


total_loss6��@

error_R�Y?

learning_rate_1�8 �λI       6%�	l��K���A�*;


total_loss���@

error_R��N?

learning_rate_1�8���I       6%�	�K���A�*;


total_loss�=�@

error_R@�K?

learning_rate_1�8W�eI       6%�	�P�K���A�*;


total_loss]�}@

error_R�G\?

learning_rate_1�8�v�I       6%�	Δ�K���A�*;


total_loss�j�@

error_R�b?

learning_rate_1�8 �ŝI       6%�	ߣK���A�*;


total_loss�a�@

error_R|0P?

learning_rate_1�8'�`�I       6%�	7�K���A�*;


total_loss�A

error_RZTO?

learning_rate_1�8�ΤxI       6%�	L��K���A�*;


total_loss�̣@

error_R�YP?

learning_rate_1�8X��I       6%�	4ɤK���A�*;


total_loss3<�@

error_R��@?

learning_rate_1�8�翚I       6%�	m�K���A�*;


total_loss�.�@

error_RҁK?

learning_rate_1�8t�܄I       6%�	�n�K���A�*;


total_loss{��@

error_RT|G?

learning_rate_1�8���I       6%�	ι�K���A�*;


total_loss
�@

error_R�d?

learning_rate_1�8
d `I       6%�	��K���A�*;


total_loss��@

error_R�=N?

learning_rate_1�8G���I       6%�	�g�K���A�*;


total_loss
�@

error_R��O?

learning_rate_1�8���B