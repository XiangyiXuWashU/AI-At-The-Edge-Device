       �K"	   \���Abrain.Event:2^�_�>K     6�.	�W\���A"��
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
weights/random_normalAddweights/random_normal/mulweights/random_normal/mean* 
_output_shapes
:
��*
T0
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
weights/weight1/readIdentityweights/weight1* 
_output_shapes
:
��*
T0*"
_class
loc:@weights/weight1
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
weights/random_normal_1Addweights/random_normal_1/mulweights/random_normal_1/mean* 
_output_shapes
:
��*
T0
�
weights/weight2
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
��*
	container *
shape:
��
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
weights/weight3/AssignAssignweights/weight3weights/random_normal_2*
T0*"
_class
loc:@weights/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
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
VariableV2*
shared_name *
dtype0*
_output_shapes

:d*
	container *
shape
:d
�
weights/weight_out/AssignAssignweights/weight_outweights/random_normal_3*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*%
_class
loc:@weights/weight_out
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
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
.weights_1/random_normal_1/RandomStandardNormalRandomStandardNormalweights_1/random_normal_1/shape*
dtype0* 
_output_shapes
:
��*
seed2 *

seed *
T0
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
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
��*
	container *
shape:
��
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
dtype0*
_output_shapes
:*
valueB"�   d   
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
.weights_1/random_normal_2/RandomStandardNormalRandomStandardNormalweights_1/random_normal_2/shape*
dtype0*
_output_shapes
:	�d*
seed2 *

seed *
T0
�
weights_1/random_normal_2/mulMul.weights_1/random_normal_2/RandomStandardNormal weights_1/random_normal_2/stddev*
_output_shapes
:	�d*
T0
�
weights_1/random_normal_2Addweights_1/random_normal_2/mulweights_1/random_normal_2/mean*
_output_shapes
:	�d*
T0
�
weights_1/weight3
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
�
weights_1/weight3/AssignAssignweights_1/weight3weights_1/random_normal_2*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
�
weights_1/weight3/readIdentityweights_1/weight3*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
p
weights_1/random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      
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
VariableV2*
dtype0*
_output_shapes

:d*
	container *
shape
:d*
shared_name 
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
biases_1/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
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
biases_1/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
biases_1/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
-biases_1/random_normal_1/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_1/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
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
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
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
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
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
biases_1/random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
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
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
q
,learning_rate/ExponentialDecay/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
ף;
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
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*'
_output_shapes
:���������*
T0
�
!train/gradients/Square_grad/mul/xConst'^train/gradients/loss/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
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
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"train/gradients/sub_1_grad/ReshapeReshapetrain/gradients/sub_1_grad/Sum train/gradients/sub_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
 train/gradients/sub_1_grad/Sum_1Sum!train/gradients/Square_grad/mul_12train/gradients/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
3train/gradients/sub_1_grad/tuple/control_dependencyIdentity"train/gradients/sub_1_grad/Reshape,^train/gradients/sub_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*5
_class+
)'loc:@train/gradients/sub_1_grad/Reshape
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
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
9train/gradients/layer_3/Add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/Add_grad/Reshape2^train/gradients/layer_3/Add_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*;
_class1
/-loc:@train/gradients/layer_3/Add_grad/Reshape
�
;train/gradients/layer_3/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/Add_grad/Reshape_12^train/gradients/layer_3/Add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/Add_grad/Reshape_1*
_output_shapes
:d
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/Add_grad/tuple/control_dependencyweights_1/weight3/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
0train/gradients/layer_3/Sigmoid_grad/SigmoidGradSigmoidGradlayer_3/Sigmoid<train/gradients/layer_3/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/Add_grad/tuple/control_dependencyweights_1/weight2/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
*train/gradients/layer_1/Add_grad/Reshape_1Reshape&train/gradients/layer_1/Add_grad/Sum_1(train/gradients/layer_1/Add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul
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
weights_1/weight2/Adam/AssignAssignweights_1/weight2/Adam(weights_1/weight2/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
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
weights_1/weight2/Adam_1/AssignAssignweights_1/weight2/Adam_1*weights_1/weight2/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��
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
"weights_1/weight_out/Adam_1/AssignAssignweights_1/weight_out/Adam_1-weights_1/weight_out/Adam_1/Initializer/zeros*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
�
 weights_1/weight_out/Adam_1/readIdentityweights_1/weight_out/Adam_1*
T0*'
_class
loc:@weights_1/weight_out*
_output_shapes

:d
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
biases_1/bias1/Adam_1/AssignAssignbiases_1/bias1/Adam_1'biases_1/bias1/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
�
biases_1/bias1/Adam_1/readIdentitybiases_1/bias1/Adam_1*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
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
'biases_1/bias2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias2*
valueB�*    
�
biases_1/bias2/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@biases_1/bias2
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
biases_1/bias_out/Adam/AssignAssignbiases_1/bias_out/Adam(biases_1/bias_out/Adam/Initializer/zeros*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
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
*train/Adam/update_biases_1/bias3/ApplyAdam	ApplyAdambiases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/Add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@biases_1/bias3*
use_nesterov( *
_output_shapes
:d*
use_locking( 
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
train/Adam/valueConst^train/Adam/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable
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
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
r
save/RestoreV2_1/tensor_namesConst*!
valueBBbiases/bias1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
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
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assignbiases/bias2save/RestoreV2_2*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@biases/bias2
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
save/Assign_3Assignbiases/bias3save/RestoreV2_3*
T0*
_class
loc:@biases/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Assign_8Assignbiases_1/bias2save/RestoreV2_8*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias2
y
save/RestoreV2_9/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBbiases_1/bias2/Adam
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
save/Assign_9Assignbiases_1/bias2/Adamsave/RestoreV2_9*
use_locking(*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�
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
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_13Assignbiases_1/bias3/Adam_1save/RestoreV2_13*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
save/RestoreV2_15/tensor_namesConst*
dtype0*
_output_shapes
:*+
value"B Bbiases_1/bias_out/Adam
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_19/tensor_namesConst*$
valueBBweights/weight1*
dtype0*
_output_shapes
:
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
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
_output_shapes
:*
dtypes
2
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
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*%
_class
loc:@weights/weight_out
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
save/Assign_27Assignweights_1/weight2/Adamsave/RestoreV2_27*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
save/Assign_31Assignweights_1/weight3/Adam_1save/RestoreV2_31*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
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
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assignweights_1/weight_outsave/RestoreV2_32*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
�
save/RestoreV2_33/tensor_namesConst*.
value%B#Bweights_1/weight_out/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_34Assignweights_1/weight_out/Adam_1save/RestoreV2_34*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
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
: *
Index0*
T0
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"
3�!Ph     6Ԇg	3�\���AJ��
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
weights/weight1/AssignAssignweights/weight1weights/random_normal*
T0*"
_class
loc:@weights/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
weights/random_normal_1Addweights/random_normal_1/mulweights/random_normal_1/mean* 
_output_shapes
:
��*
T0
�
weights/weight2
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
��*
	container *
shape:
��
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
weights/random_normal_2Addweights/random_normal_2/mulweights/random_normal_2/mean*
T0*
_output_shapes
:	�d
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
weights/weight3/AssignAssignweights/weight3weights/random_normal_2*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*"
_class
loc:@weights/weight3
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
weights/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *
dtype0*
_output_shapes

:d*
	container *
shape
:d
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
)biases/random_normal/RandomStandardNormalRandomStandardNormalbiases/random_normal/shape*
dtype0*
_output_shapes	
:�*
seed2 *

seed *
T0
�
biases/random_normal/mulMul)biases/random_normal/RandomStandardNormalbiases/random_normal/stddev*
_output_shapes	
:�*
T0
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
+biases/random_normal_1/RandomStandardNormalRandomStandardNormalbiases/random_normal_1/shape*
dtype0*
_output_shapes	
:�*
seed2 *

seed *
T0
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
biases/random_normal_2/shapeConst*
valueB:d*
dtype0*
_output_shapes
:
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
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
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
+biases/random_normal_3/RandomStandardNormalRandomStandardNormalbiases/random_normal_3/shape*
dtype0*
_output_shapes
:*
seed2 *

seed *
T0
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
weights_1/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
c
weights_1/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights_1/random_normal/RandomStandardNormalRandomStandardNormalweights_1/random_normal/shape*

seed *
T0*
dtype0* 
_output_shapes
:
��*
seed2 
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
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
��*
	container *
shape:
��
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
weights_1/weight2/readIdentityweights_1/weight2* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight2
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
weights_1/weight3/AssignAssignweights_1/weight3weights_1/random_normal_2*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
weights_1/weight3/readIdentityweights_1/weight3*
T0*$
_class
loc:@weights_1/weight3*
_output_shapes
:	�d
p
weights_1/random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      
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
weights_1/weight_out/AssignAssignweights_1/weight_outweights_1/random_normal_3*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
�
weights_1/weight_out/readIdentityweights_1/weight_out*
_output_shapes

:d*
T0*'
_class
loc:@weights_1/weight_out
g
biases_1/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
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
+biases_1/random_normal/RandomStandardNormalRandomStandardNormalbiases_1/random_normal/shape*
T0*
dtype0*
_output_shapes	
:�*
seed2 *

seed 
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
biases_1/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
biases_1/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
biases_1/bias2/AssignAssignbiases_1/bias2biases_1/random_normal_1*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
-biases_1/random_normal_2/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_2/shape*

seed *
T0*
dtype0*
_output_shapes
:d*
seed2 
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
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
�
biases_1/bias3/AssignAssignbiases_1/bias3biases_1/random_normal_2*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
-biases_1/random_normal_3/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_3/shape*
dtype0*
_output_shapes
:*
seed2 *

seed *
T0
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
layer_2/MatMulMatMullayer_2/Reluweights_1/weight2/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
layer_3/MatMulMatMullayer_3/Sigmoidweights_1/weight3/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
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
result/Addinput/Label-input*'
_output_shapes
:���������*
T0
A
AbsAbssub*'
_output_shapes
:���������*
T0
]
sub_1Sub
result/Addinput/Label-input*'
_output_shapes
:���������*
T0
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
Variable/AssignAssignVariableVariable/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
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
ף;*
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
loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
dtype0*
_output_shapes
:*
valueB"      
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
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
)train/gradients/result/MatMul_grad/MatMulMatMul8train/gradients/result/Add_grad/tuple/control_dependencyweights_1/weight_out/read*
transpose_b(*
T0*'
_output_shapes
:���������d*
transpose_a( 
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
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_3/Sigmoid9train/gradients/layer_3/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�d*
transpose_a(
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
(train/gradients/layer_2/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
6train/gradients/layer_2/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/Add_grad/Shape(train/gradients/layer_2/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/Add_grad/tuple/control_dependencyweights_1/weight2/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
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
shape: *
dtype0*
_output_shapes
: *
shared_name *!
_class
loc:@biases_1/bias1*
	container 
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
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight2
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
weights_1/weight3/Adam/AssignAssignweights_1/weight3/Adam(weights_1/weight3/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d
�
weights_1/weight3/Adam/readIdentityweights_1/weight3/Adam*
_output_shapes
:	�d*
T0*$
_class
loc:@weights_1/weight3
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
 weights_1/weight_out/Adam/AssignAssignweights_1/weight_out/Adam+weights_1/weight_out/Adam/Initializer/zeros*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
"weights_1/weight_out/Adam_1/AssignAssignweights_1/weight_out/Adam_1-weights_1/weight_out/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
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
biases_1/bias2/Adam/AssignAssignbiases_1/bias2/Adam%biases_1/bias2/Adam/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
biases_1/bias2/Adam/readIdentitybiases_1/bias2/Adam*
T0*!
_class
loc:@biases_1/bias2*
_output_shapes	
:�
�
'biases_1/bias2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias2*
valueB�*    
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
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *!
_class
loc:@biases_1/bias3*
	container 
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
biases_1/bias3/Adam_1/AssignAssignbiases_1/bias3/Adam_1'biases_1/bias3/Adam_1/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
�
biases_1/bias3/Adam_1/readIdentitybiases_1/bias3/Adam_1*
_output_shapes
:d*
T0*!
_class
loc:@biases_1/bias3
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
VariableV2*$
_class
loc:@biases_1/bias_out*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
biases_1/bias_out/Adam/AssignAssignbiases_1/bias_out/Adam(biases_1/bias_out/Adam/Initializer/zeros*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
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
biases_1/bias_out/Adam_1/AssignAssignbiases_1/bias_out/Adam_1*biases_1/bias_out/Adam_1/Initializer/zeros*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
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
*train/Adam/update_biases_1/bias2/ApplyAdam	ApplyAdambiases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@biases_1/bias2
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
-train/Adam/update_biases_1/bias_out/ApplyAdam	ApplyAdambiases_1/bias_outbiases_1/bias_out/Adambiases_1/bias_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/result/Add_grad/tuple/control_dependency_1*
T0*$
_class
loc:@biases_1/bias_out*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*!
_class
loc:@biases_1/bias1
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
train/Adam/valueConst^train/Adam/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable
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
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:#*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_1Assignbiases/bias1save/RestoreV2_1*
T0*
_class
loc:@biases/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
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
save/Assign_4Assignbiases/bias_outsave/RestoreV2_4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@biases/bias_out
t
save/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBbiases_1/bias1
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_7Assignbiases_1/bias1/Adam_1save/RestoreV2_7*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
save/Assign_11Assignbiases_1/bias3save/RestoreV2_11*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
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
save/RestoreV2_15/tensor_namesConst*
dtype0*
_output_shapes
:*+
value"B Bbiases_1/bias_out/Adam
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
save/Assign_16Assignbiases_1/bias_out/Adam_1save/RestoreV2_16*
use_locking(*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:
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
save/Assign_17Assigntrain/beta1_powersave/RestoreV2_17*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
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
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*%
_class
loc:@weights/weight_out
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
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_24Assignweights_1/weight1/Adamsave/RestoreV2_24*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
save/Assign_26Assignweights_1/weight2save/RestoreV2_26*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27Assignweights_1/weight2/Adamsave/RestoreV2_27*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��*
use_locking(

save/RestoreV2_28/tensor_namesConst*-
value$B"Bweights_1/weight2/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_28Assignweights_1/weight2/Adam_1save/RestoreV2_28*
use_locking(*
T0*$
_class
loc:@weights_1/weight2*
validate_shape(* 
_output_shapes
:
��
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
save/Assign_31Assignweights_1/weight3/Adam_1save/RestoreV2_31*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
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
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_34Assignweights_1/weight_out/Adam_1save/RestoreV2_34*
use_locking(*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d
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
Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B : 
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
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign""�
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
biases_1/bias_out:0biases_1/bias_out/Assignbiases_1/bias_out/read:0";
	summaries.
,
total_loss:0
	error_R:0
learning_rate_1:0"
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
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0S���F       r5��	J\���A*;


total_lossS��@

error_R�Q=?

learning_rate_1�O�6��H       ��H�	�:!\���A*;


total_lossz��@

error_R@cD?

learning_rate_1�O�6��� H       ��H�	��!\���A*;


total_losso!�@

error_R��S?

learning_rate_1�O�6��|H       ��H�	��!\���A*;


total_loss�;�@

error_R�mX?

learning_rate_1�O�6�ga�H       ��H�	�"\���A*;


total_lossqJ�@

error_R@�\?

learning_rate_1�O�6S5z�H       ��H�	_"\���A*;


total_loss<�@

error_R3�Q?

learning_rate_1�O�6t��H       ��H�	��"\���A*;


total_lossa�]@

error_R6�X?

learning_rate_1�O�6�̘�H       ��H�	x�"\���A*;


total_loss�6�@

error_R��S?

learning_rate_1�O�6��EhH       ��H�	?8#\���A*;


total_loss�2�@

error_R�;?

learning_rate_1�O�6�:K"H       ��H�	�}#\���A	*;


total_loss2�@

error_R�.P?

learning_rate_1�O�6��H       ��H�	��#\���A
*;


total_loss�5�@

error_Re9Q?

learning_rate_1�O�6����H       ��H�	}$\���A*;


total_loss�6�@

error_R�[S?

learning_rate_1�O�6?�0nH       ��H�	�J$\���A*;


total_loss�߆@

error_R$YO?

learning_rate_1�O�611BH       ��H�	M�$\���A*;


total_lossA�@

error_R��d?

learning_rate_1�O�6>
?H       ��H�	[�$\���A*;


total_lossJmz@

error_Rf/M?

learning_rate_1�O�6[\-H       ��H�	�%\���A*;


total_loss�*A

error_R��Q?

learning_rate_1�O�6�~�H       ��H�	C\%\���A*;


total_loss
��@

error_R �J?

learning_rate_1�O�6��3�H       ��H�	�%\���A*;


total_loss��@

error_R;�Y?

learning_rate_1�O�6�V�H       ��H�	�%\���A*;


total_loss�u�@

error_R�J?

learning_rate_1�O�6���H       ��H�	�3&\���A*;


total_loss���@

error_R@TP?

learning_rate_1�O�6<��H       ��H�	�&\���A*;


total_lossZv�@

error_R��H?

learning_rate_1�O�6E�z9H       ��H�	��&\���A*;


total_loss�2�@

error_R��_?

learning_rate_1�O�6�h�}H       ��H�	�+'\���A*;


total_loss��@

error_R��]?

learning_rate_1�O�6�q��H       ��H�	$q'\���A*;


total_lossֹ�@

error_R��M?

learning_rate_1�O�6�֞H       ��H�	��'\���A*;


total_loss3A

error_R4WN?

learning_rate_1�O�6<�եH       ��H�	��'\���A*;


total_loss�A�@

error_RxK<?

learning_rate_1�O�6#��H       ��H�	�B(\���A*;


total_loss1�@

error_R �A?

learning_rate_1�O�6�K�H       ��H�	х(\���A*;


total_loss���@

error_R�cF?

learning_rate_1�O�6�)��H       ��H�	��(\���A*;


total_loss�ʷ@

error_ReR?

learning_rate_1�O�6i��H       ��H�	�)\���A*;


total_loss���@

error_R��Z?

learning_rate_1�O�6����H       ��H�	�^)\���A*;


total_lossDܤ@

error_R�S?

learning_rate_1�O�6TTL-H       ��H�	(�)\���A*;


total_loss&��@

error_RG?

learning_rate_1�O�6��8H       ��H�	��)\���A *;


total_lossaw�@

error_R�S?

learning_rate_1�O�69���H       ��H�	�8*\���A!*;


total_lossz��@

error_Rx%Y?

learning_rate_1�O�6�`�H       ��H�	~*\���A"*;


total_loss`�@

error_RBK?

learning_rate_1�O�6��g+H       ��H�	q�*\���A#*;


total_loss�ȑ@

error_R��H?

learning_rate_1�O�6͇�RH       ��H�	�+\���A$*;


total_lossc7�@

error_R��^?

learning_rate_1�O�6Q?B�H       ��H�	�F+\���A%*;


total_lossR]�@

error_R8+M?

learning_rate_1�O�6[��kH       ��H�	R�+\���A&*;


total_loss���@

error_R�L?

learning_rate_1�O�6q��H       ��H�	;�+\���A'*;


total_loss4?�@

error_R}�V?

learning_rate_1�O�6�/�H       ��H�	9,\���A(*;


total_loss�A

error_R�gF?

learning_rate_1�O�6+J��H       ��H�	x,\���A)*;


total_loss\�@

error_Rd�[?

learning_rate_1�O�6Y���H       ��H�	�,\���A**;


total_loss/&�@

error_R��??

learning_rate_1�O�6RK�H       ��H�	��,\���A+*;


total_lossx&�@

error_R͕e?

learning_rate_1�O�6�w�H       ��H�	�D-\���A,*;


total_lossj@�@

error_RA�V?

learning_rate_1�O�6�]
�H       ��H�	Z�-\���A-*;


total_loss�A

error_R\K?

learning_rate_1�O�6*%NH       ��H�	��-\���A.*;


total_lossO�A

error_RaS?

learning_rate_1�O�6�ޮH       ��H�	5 .\���A/*;


total_loss
W�@

error_R\�c?

learning_rate_1�O�6�P�~H       ��H�	td.\���A0*;


total_lossຳ@

error_R��f?

learning_rate_1�O�6�� H       ��H�	ǰ.\���A1*;


total_loss7��@

error_R��U?

learning_rate_1�O�6���H       ��H�	��.\���A2*;


total_lossǻ@

error_R6�??

learning_rate_1�O�6nB�H       ��H�	=A/\���A3*;


total_loss[�@

error_R�^?

learning_rate_1�O�6���H       ��H�	��/\���A4*;


total_loss �@

error_R�,J?

learning_rate_1�O�6��q=H       ��H�	��/\���A5*;


total_loss��@

error_R$:O?

learning_rate_1�O�6���H       ��H�	~60\���A6*;


total_lossZ�@

error_RhY?

learning_rate_1�O�6ۗ9-H       ��H�	�}0\���A7*;


total_lossL�@

error_R�A?

learning_rate_1�O�62�	PH       ��H�	Z�0\���A8*;


total_loss�>�@

error_R$�9?

learning_rate_1�O�6��H       ��H�	(1\���A9*;


total_losso6�@

error_R�S?

learning_rate_1�O�6���H       ��H�	"J1\���A:*;


total_lossQ��@

error_R7�S?

learning_rate_1�O�6wĩH       ��H�	r�1\���A;*;


total_lossՉ@

error_R�pD?

learning_rate_1�O�6��H       ��H�	��1\���A<*;


total_loss.B�@

error_R?�E?

learning_rate_1�O�6f��H       ��H�	&2\���A=*;


total_lossl�@

error_R��K?

learning_rate_1�O�6턬?H       ��H�	&^2\���A>*;


total_loss|ِ@

error_RT�K?

learning_rate_1�O�6p��cH       ��H�	�2\���A?*;


total_lossX��@

error_R��X?

learning_rate_1�O�6���XH       ��H�	��2\���A@*;


total_loss	�@

error_RirR?

learning_rate_1�O�6k�)�H       ��H�	�23\���AA*;


total_losso�@

error_R�>X?

learning_rate_1�O�6�p�EH       ��H�	)3\���AB*;


total_loss��@

error_RhK?

learning_rate_1�O�6��\H       ��H�	�3\���AC*;


total_loss�O�@

error_RR�K?

learning_rate_1�O�6��9H       ��H�	�4\���AD*;


total_loss�,f@

error_R�=?

learning_rate_1�O�6<�*H       ��H�	�R4\���AE*;


total_loss���@

error_R�qD?

learning_rate_1�O�6�G�aH       ��H�	K�4\���AF*;


total_loss	N�@

error_R��[?

learning_rate_1�O�6N�$�H       ��H�	c�4\���AG*;


total_lossڪ�@

error_R�QY?

learning_rate_1�O�6�E�H       ��H�	�5\���AH*;


total_loss��@

error_R;�P?

learning_rate_1�O�6�jJ'H       ��H�	Sd5\���AI*;


total_lossڢ�@

error_Rd?

learning_rate_1�O�6)=P H       ��H�	o�5\���AJ*;


total_loss;)�@

error_R�g[?

learning_rate_1�O�67�֟H       ��H�	l�5\���AK*;


total_loss�U�@

error_RqBK?

learning_rate_1�O�6	�H       ��H�	�36\���AL*;


total_lossn��@

error_R/B?

learning_rate_1�O�6��SH       ��H�	܁6\���AM*;


total_loss�Ȅ@

error_RdR?

learning_rate_1�O�6+;ׇH       ��H�	�6\���AN*;


total_loss���@

error_R�AV?

learning_rate_1�O�6a�6kH       ��H�	�7\���AO*;


total_loss�� A

error_R��I?

learning_rate_1�O�6��_DH       ��H�	cU7\���AP*;


total_loss��@

error_R�wM?

learning_rate_1�O�6Ԩ�XH       ��H�	�7\���AQ*;


total_lossžA

error_R��Z?

learning_rate_1�O�6,bMH       ��H�	��7\���AR*;


total_lossC��@

error_R��Y?

learning_rate_1�O�6z��BH       ��H�	�8\���AS*;


total_loss�@

error_Rq9?

learning_rate_1�O�6�@�H       ��H�	K`8\���AT*;


total_lossI*�@

error_R�h:?

learning_rate_1�O�6Ku*H       ��H�	��8\���AU*;


total_lossiO�@

error_RA�I?

learning_rate_1�O�6��\6H       ��H�	��8\���AV*;


total_loss^�@

error_R�E?

learning_rate_1�O�6=t��H       ��H�	 +9\���AW*;


total_loss}��@

error_R��A?

learning_rate_1�O�6��&H       ��H�	ݗ9\���AX*;


total_loss�@

error_RmlH?

learning_rate_1�O�6�&rKH       ��H�	w�9\���AY*;


total_loss�#A

error_R��M?

learning_rate_1�O�6�u��H       ��H�	�A:\���AZ*;


total_loss�k�@

error_R�=D?

learning_rate_1�O�6*�*{H       ��H�	?�:\���A[*;


total_loss	�@

error_R�H_?

learning_rate_1�O�6�O��H       ��H�	~�:\���A\*;


total_lossy(�@

error_Rxn_?

learning_rate_1�O�6���-H       ��H�	�;\���A]*;


total_loss��A

error_Rq�Z?

learning_rate_1�O�6{^>�H       ��H�	�P;\���A^*;


total_losslz�@

error_RV�Q?

learning_rate_1�O�6ˊ�
H       ��H�	R�;\���A_*;


total_lossFU�@

error_R�P?

learning_rate_1�O�6^I��H       ��H�	C�;\���A`*;


total_loss.�@

error_RR�U?

learning_rate_1�O�6ԣH       ��H�	CE<\���Aa*;


total_loss��@

error_R�A?

learning_rate_1�O�6�x	uH       ��H�	��<\���Ab*;


total_loss*�@

error_R�Y?

learning_rate_1�O�6l���H       ��H�	q�<\���Ac*;


total_loss� �@

error_R�C?

learning_rate_1�O�6*���H       ��H�	�=\���Ad*;


total_loss��@

error_R�kU?

learning_rate_1�O�6��I,H       ��H�	"e=\���Ae*;


total_loss䣟@

error_R�<O?

learning_rate_1�O�6%:�NH       ��H�	��=\���Af*;


total_lossڛ�@

error_R�\?

learning_rate_1�O�6�C�%H       ��H�	��=\���Ag*;


total_loss�@

error_REO?

learning_rate_1�O�6[��@H       ��H�	l=>\���Ah*;


total_lossj��@

error_R ;9?

learning_rate_1�O�6Ң�H       ��H�	��>\���Ai*;


total_loss���@

error_R��B?

learning_rate_1�O�6�Z�H       ��H�	��>\���Aj*;


total_loss���@

error_R��T?

learning_rate_1�O�6��yLH       ��H�	"?\���Ak*;


total_losso��@

error_R)�B?

learning_rate_1�O�6*Ra�H       ��H�	�_?\���Al*;


total_lossen�@

error_RڌP?

learning_rate_1�O�6�!��H       ��H�	��?\���Am*;


total_loss�K�@

error_R�A?

learning_rate_1�O�6c��H       ��H�	��?\���An*;


total_loss�%�@

error_R�?K?

learning_rate_1�O�6�n�H       ��H�	{%@\���Ao*;


total_loss�ՠ@

error_R�N?

learning_rate_1�O�6ŃH       ��H�	�i@\���Ap*;


total_loss95�@

error_R8I?

learning_rate_1�O�6Ʊ<�H       ��H�	a�@\���Aq*;


total_loss��A

error_R`�;?

learning_rate_1�O�6�i&�H       ��H�	��@\���Ar*;


total_loss��@

error_R�O?

learning_rate_1�O�6�w�.H       ��H�	�1A\���As*;


total_loss���@

error_RL?

learning_rate_1�O�6Z�/�H       ��H�	�tA\���At*;


total_loss3�@

error_Ra�Q?

learning_rate_1�O�6�I�xH       ��H�	�A\���Au*;


total_lossnm�@

error_RC�:?

learning_rate_1�O�6��eH       ��H�	0B\���Av*;


total_loss���@

error_R�iP?

learning_rate_1�O�6dJ�)H       ��H�	iIB\���Aw*;


total_loss_��@

error_RcQ?

learning_rate_1�O�6�D�H       ��H�	D�B\���Ax*;


total_lossEQ�@

error_R�>?

learning_rate_1�O�6=���H       ��H�	��B\���Ay*;


total_loss��@

error_R��P?

learning_rate_1�O�6��ҀH       ��H�	�C\���Az*;


total_loss�m�@

error_Rw�W?

learning_rate_1�O�6]`qH       ��H�	ZeC\���A{*;


total_lossW�@

error_R�I?

learning_rate_1�O�6�6wiH       ��H�	�C\���A|*;


total_lossŐ�@

error_R��A?

learning_rate_1�O�6�vw�H       ��H�	m�C\���A}*;


total_loss�*�@

error_R��A?

learning_rate_1�O�6}6��H       ��H�	.0D\���A~*;


total_losse��@

error_RPF?

learning_rate_1�O�6�d�H       ��H�	�xD\���A*;


total_loss2��@

error_RZ�B?

learning_rate_1�O�6N���I       6%�	��D\���A�*;


total_loss��@

error_R�2Z?

learning_rate_1�O�6W�>I       6%�	/E\���A�*;


total_lossS�@

error_Rm�N?

learning_rate_1�O�6���I       6%�	.rE\���A�*;


total_loss���@

error_R�pN?

learning_rate_1�O�6_��I       6%�	��E\���A�*;


total_loss�*�@

error_R� c?

learning_rate_1�O�6�k�'I       6%�	f%F\���A�*;


total_lossv�@

error_R�HN?

learning_rate_1�O�6�h�7I       6%�	�lF\���A�*;


total_loss��@

error_R��8?

learning_rate_1�O�6+�r�I       6%�	9�F\���A�*;


total_lossmWk@

error_R�|??

learning_rate_1�O�6�/H�I       6%�	�F\���A�*;


total_lossM��@

error_R_
H?

learning_rate_1�O�6�2ѶI       6%�	�?G\���A�*;


total_loss�@

error_R#@Q?

learning_rate_1�O�64.>3I       6%�	�G\���A�*;


total_loss�A

error_R�{Q?

learning_rate_1�O�6&���I       6%�	b�G\���A�*;


total_loss|��@

error_RWQG?

learning_rate_1�O�6]�-�I       6%�	'H\���A�*;


total_loss�_�@

error_R�nA?

learning_rate_1�O�6�px?I       6%�	#ZH\���A�*;


total_lossR*A

error_R�JF?

learning_rate_1�O�6�eI       6%�	c�H\���A�*;


total_loss���@

error_R�sG?

learning_rate_1�O�6��@I       6%�	5�H\���A�*;


total_loss�o�@

error_R�0O?

learning_rate_1�O�6���I       6%�	_3I\���A�*;


total_loss)��@

error_RL4S?

learning_rate_1�O�6�lU�I       6%�	�xI\���A�*;


total_lossC�@

error_Rj�@?

learning_rate_1�O�6�$��I       6%�	��I\���A�*;


total_lossE��@

error_R�jU?

learning_rate_1�O�6�)�I       6%�	J\���A�*;


total_loss��@

error_R&<K?

learning_rate_1�O�6P\��I       6%�	�MJ\���A�*;


total_lossr��@

error_R�
J?

learning_rate_1�O�6B�S�I       6%�	O�J\���A�*;


total_loss�@

error_R�B?

learning_rate_1�O�6Ul��I       6%�	Q�J\���A�*;


total_lossn�@

error_Rڭ=?

learning_rate_1�O�6��{I       6%�	%K\���A�*;


total_loss��@

error_R�?T?

learning_rate_1�O�6I��I       6%�	gK\���A�*;


total_lossD��@

error_R�.@?

learning_rate_1�O�6��ԬI       6%�	�K\���A�*;


total_loss��l@

error_R�M?

learning_rate_1�O�6�X�|I       6%�	fL\���A�*;


total_loss��@

error_R)^?

learning_rate_1�O�6�S�I       6%�	
[L\���A�*;


total_loss:��@

error_RڨQ?

learning_rate_1�O�6��	yI       6%�	(�L\���A�*;


total_loss��@

error_R�Pa?

learning_rate_1�O�6^��I       6%�	��L\���A�*;


total_lossR�g@

error_R�L?

learning_rate_1�O�6R�ȖI       6%�	�+M\���A�*;


total_lossw�A

error_R�Q?

learning_rate_1�O�6���I       6%�	nnM\���A�*;


total_loss1ȷ@

error_RShW?

learning_rate_1�O�6�$}I       6%�	��M\���A�*;


total_loss��@

error_R(�J?

learning_rate_1�O�6YE.�I       6%�	��M\���A�*;


total_lossSa�@

error_R|Y?

learning_rate_1�O�6�$�.I       6%�	k8N\���A�*;


total_lossC��@

error_R,�5?

learning_rate_1�O�6!�~I       6%�	�N\���A�*;


total_loss1��@

error_R}2U?

learning_rate_1�O�6�f��I       6%�	Y�N\���A�*;


total_lossE�@

error_R \?

learning_rate_1�O�6�,�I       6%�	ZO\���A�*;


total_loss��@

error_R��B?

learning_rate_1�O�6Fx��I       6%�	2GO\���A�*;


total_loss�ۄ@

error_RԣN?

learning_rate_1�O�6�4��I       6%�	��O\���A�*;


total_loss�A�@

error_R�B?

learning_rate_1�O�6���dI       6%�	�O\���A�*;


total_loss�  A

error_R�fA?

learning_rate_1�O�6T�I       6%�	*,P\���A�*;


total_loss��A

error_R}�[?

learning_rate_1�O�6�%�I       6%�	1wP\���A�*;


total_loss[m�@

error_RS�A?

learning_rate_1�O�6Z��=I       6%�	o�P\���A�*;


total_losso��@

error_R(�Y?

learning_rate_1�O�6�jxI       6%�	e Q\���A�*;


total_loss���@

error_R,�`?

learning_rate_1�O�6N��I       6%�	�GQ\���A�*;


total_loss�P@

error_RE^:?

learning_rate_1�O�6��q�I       6%�	��Q\���A�*;


total_loss9�@

error_R�\?

learning_rate_1�O�6=���I       6%�	|�Q\���A�*;


total_loss�P�@

error_Rc�I?

learning_rate_1�O�6�|w�I       6%�	�%R\���A�*;


total_loss�b�@

error_R!|J?

learning_rate_1�O�6���	I       6%�	�jR\���A�*;


total_lossI�@

error_Rx�:?

learning_rate_1�O�6�T�I       6%�	�R\���A�*;


total_losst!A

error_R��F?

learning_rate_1�O�6d�2I       6%�	q�R\���A�*;


total_loss��@

error_Rl)G?

learning_rate_1�O�6"�R�I       6%�	�ES\���A�*;


total_lossA��@

error_R�
_?

learning_rate_1�O�6�T�MI       6%�	�S\���A�*;


total_loss��@

error_R��@?

learning_rate_1�O�6�RHI       6%�	��S\���A�*;


total_loss8�@

error_RqP?

learning_rate_1�O�6�I       6%�	oT\���A�*;


total_loss@�@

error_R��_?

learning_rate_1�O�6�ŗ�I       6%�	pgT\���A�*;


total_loss
#�@

error_R*M?

learning_rate_1�O�6\��I       6%�	�T\���A�*;


total_loss҈�@

error_R��N?

learning_rate_1�O�6d`��I       6%�	��T\���A�*;


total_loss<i�@

error_R��^?

learning_rate_1�O�6���jI       6%�	x<U\���A�*;


total_lossK��@

error_R�b?

learning_rate_1�O�6Ԁ,I       6%�	�U\���A�*;


total_loss���@

error_R�mH?

learning_rate_1�O�63���I       6%�	t�U\���A�*;


total_lossCY@

error_R��>?

learning_rate_1�O�6�,bcI       6%�	fV\���A�*;


total_lossQ��@

error_RQ�T?

learning_rate_1�O�6�A/I       6%�	�\V\���A�*;


total_loss�J�@

error_RM�R?

learning_rate_1�O�6��\I       6%�	ΠV\���A�*;


total_loss}A

error_R�!I?

learning_rate_1�O�6��k�I       6%�	k�V\���A�*;


total_loss�A

error_R��R?

learning_rate_1�O�6�%KI       6%�	�3W\���A�*;


total_loss!�RA

error_Rn�A?

learning_rate_1�O�6�$Y]I       6%�	�zW\���A�*;


total_loss���@

error_R/�U?

learning_rate_1�O�6_]8I       6%�	i�W\���A�*;


total_lossF�@

error_R{K?

learning_rate_1�O�6%s)�I       6%�	��W\���A�*;


total_loss׍�@

error_R)�>?

learning_rate_1�O�6���UI       6%�	�CX\���A�*;


total_loss���@

error_R�{0?

learning_rate_1�O�6���}I       6%�	H�X\���A�*;


total_lossjN�@

error_R�.T?

learning_rate_1�O�6���aI       6%�	`�X\���A�*;


total_loss}|�@

error_R8�L?

learning_rate_1�O�6�nPI       6%�	�Y\���A�*;


total_lossl��@

error_RJQ?

learning_rate_1�O�6�>*�I       6%�	�QY\���A�*;


total_lossw��@

error_R�Y?

learning_rate_1�O�66��I       6%�	��Y\���A�*;


total_loss���@

error_R�kQ?

learning_rate_1�O�6�zI       6%�	s�Y\���A�*;


total_lossOJ�@

error_R�E?

learning_rate_1�O�6�B�:I       6%�	sZ\���A�*;


total_loss��@

error_R�6T?

learning_rate_1�O�6�f	1I       6%�	y^Z\���A�*;


total_loss|��@

error_R҇Z?

learning_rate_1�O�6�KI       6%�	\�Z\���A�*;


total_loss/��@

error_R6bL?

learning_rate_1�O�6;�P�I       6%�	m�Z\���A�*;


total_loss�A

error_R)�H?

learning_rate_1�O�6�:��I       6%�	�=[\���A�*;


total_loss��@

error_R�yO?

learning_rate_1�O�6e� _I       6%�	w�[\���A�*;


total_lossQ[�@

error_R�z[?

learning_rate_1�O�6��?�I       6%�	��[\���A�*;


total_loss�ǚ@

error_R��V?

learning_rate_1�O�6'VI       6%�	 1\\���A�*;


total_loss.
A

error_R2�G?

learning_rate_1�O�6�w�I       6%�	�q\\���A�*;


total_loss4�@

error_R�??

learning_rate_1�O�6B�{I       6%�	B�\\���A�*;


total_losshO�@

error_R��C?

learning_rate_1�O�6��*�I       6%�	��\\���A�*;


total_loss`#�@

error_R?A^?

learning_rate_1�O�6'��I       6%�	�=]\���A�*;


total_lossFy�@

error_R��M?

learning_rate_1�O�6��I       6%�	?�]\���A�*;


total_loss���@

error_Rm�L?

learning_rate_1�O�6�T�I       6%�	��]\���A�*;


total_loss�4�@

error_R\�:?

learning_rate_1�O�6�>yI       6%�	a^\���A�*;


total_loss�Y�@

error_R�l?

learning_rate_1�O�6��V�I       6%�	�N^\���A�*;


total_loss&��@

error_R!�P?

learning_rate_1�O�6���I       6%�	?�^\���A�*;


total_loss���@

error_R��P?

learning_rate_1�O�6�F�I       6%�	J�^\���A�*;


total_loss��@

error_RT:?

learning_rate_1�O�6���JI       6%�	�_\���A�*;


total_loss���@

error_R��@?

learning_rate_1�O�6�~�I       6%�	�\_\���A�*;


total_loss��@

error_Rh�K?

learning_rate_1�O�6p��I       6%�	x�_\���A�*;


total_loss}�@

error_R;-I?

learning_rate_1�O�6���I       6%�	f�_\���A�*;


total_loss��@

error_R�Y?

learning_rate_1�O�6��GCI       6%�	62`\���A�*;


total_losssfA

error_R�GP?

learning_rate_1�O�6ؚk�I       6%�	nt`\���A�*;


total_loss�:�@

error_R��Z?

learning_rate_1�O�6��m�I       6%�	w�`\���A�*;


total_loss��(A

error_R�/b?

learning_rate_1�O�6Y�NI       6%�	��`\���A�*;


total_lossW��@

error_R�/\?

learning_rate_1�O�6�N'�I       6%�	�<a\���A�*;


total_loss��@

error_RZ�P?

learning_rate_1�O�6����I       6%�	�a\���A�*;


total_loss�r�@

error_RCY=?

learning_rate_1�O�6ls�I       6%�	�a\���A�*;


total_lossHʩ@

error_R�,O?

learning_rate_1�O�6E��]I       6%�	m	b\���A�*;


total_loss��@

error_R��V?

learning_rate_1�O�6�UuI       6%�	�Mb\���A�*;


total_loss��c@

error_R�vL?

learning_rate_1�O�61)w�I       6%�	ߒb\���A�*;


total_loss���@

error_R1X?

learning_rate_1�O�6�%3�I       6%�	,�b\���A�*;


total_loss�\�@

error_R�`?

learning_rate_1�O�6��K�I       6%�	c\���A�*;


total_loss��@

error_R��;?

learning_rate_1�O�6$�ŻI       6%�	m[c\���A�*;


total_loss���@

error_RԔ^?

learning_rate_1�O�6k�MSI       6%�	|�c\���A�*;


total_loss���@

error_R�FV?

learning_rate_1�O�6�퐺I       6%�	A�c\���A�*;


total_loss���@

error_R��C?

learning_rate_1�O�6���_I       6%�	�+d\���A�*;


total_loss�_�@

error_R@}_?

learning_rate_1�O�6z���I       6%�	�rd\���A�*;


total_loss��t@

error_R��J?

learning_rate_1�O�6-�`�I       6%�	�d\���A�*;


total_loss���@

error_RI�P?

learning_rate_1�O�6�UEEI       6%�	[e\���A�*;


total_loss�g@

error_R&rO?

learning_rate_1�O�6�y��I       6%�	Ye\���A�*;


total_losshe�@

error_R�aJ?

learning_rate_1�O�6�-rI       6%�	�e\���A�*;


total_lossO��@

error_R�M?

learning_rate_1�O�6 $MI       6%�	^�e\���A�*;


total_loss�#�@

error_R��D?

learning_rate_1�O�6*���I       6%�	-f\���A�*;


total_loss�!Y@

error_R��<?

learning_rate_1�O�6�ȺI       6%�	/tf\���A�*;


total_loss#�@

error_RpW?

learning_rate_1�O�63��I       6%�	\�f\���A�*;


total_loss%ҩ@

error_R\�_?

learning_rate_1�O�6V<<oI       6%�	;g\���A�*;


total_loss�?�@

error_R��d?

learning_rate_1�O�6|�W�I       6%�	�Bg\���A�*;


total_loss�h�@

error_R��f?

learning_rate_1�O�6�r��I       6%�	��g\���A�*;


total_loss��A

error_RsL?

learning_rate_1�O�6�;YYI       6%�	��g\���A�*;


total_loss(q�@

error_R��O?

learning_rate_1�O�6 ŚlI       6%�	�h\���A�*;


total_loss�p�@

error_R{�s?

learning_rate_1�O�6�)zI       6%�	vZh\���A�*;


total_loss�ʗ@

error_R��U?

learning_rate_1�O�6Vy
nI       6%�	P�h\���A�*;


total_loss?�@

error_Rl>C?

learning_rate_1�O�6��I       6%�	P�h\���A�*;


total_loss�n+A

error_RJ�=?

learning_rate_1�O�6+��-I       6%�	�&i\���A�*;


total_loss���@

error_R�76?

learning_rate_1�O�6q�=I       6%�	�gi\���A�*;


total_loss���@

error_Rv�R?

learning_rate_1�O�6U�ƪI       6%�	��i\���A�*;


total_loss���@

error_Rԍ4?

learning_rate_1�O�6c��+I       6%�	�i\���A�*;


total_loss_�o@

error_R��X?

learning_rate_1�O�6�QS�I       6%�	�4j\���A�*;


total_loss�@�@

error_RE�C?

learning_rate_1�O�61�I       6%�	vj\���A�*;


total_loss��@

error_R\E?

learning_rate_1�O�6�J�&I       6%�	"�j\���A�*;


total_lossSo�@

error_R�#R?

learning_rate_1�O�6m�I       6%�	m�j\���A�*;


total_loss4�@

error_Rs~G?

learning_rate_1�O�6C=�6I       6%�	-;k\���A�*;


total_loss�b@

error_R��C?

learning_rate_1�O�6vx�#I       6%�	�}k\���A�*;


total_loss֎�@

error_R�D?

learning_rate_1�O�6!�kI       6%�	I�k\���A�*;


total_loss�~@

error_R$�I?

learning_rate_1�O�66�_7I       6%�	�(l\���A�*;


total_lossv�@

error_R�V?

learning_rate_1�O�6Нk�I       6%�	zl\���A�*;


total_loss���@

error_RM@?

learning_rate_1�O�6MGS�I       6%�	!�l\���A�*;


total_loss���@

error_RoB?

learning_rate_1�O�6z��I       6%�	x	m\���A�*;


total_lossv�@

error_R��_?

learning_rate_1�O�6(�I       6%�	aNm\���A�*;


total_loss{Ф@

error_R�NW?

learning_rate_1�O�6(X��I       6%�	��m\���A�*;


total_lossA��@

error_R{QP?

learning_rate_1�O�6�б�I       6%�	��m\���A�*;


total_losscS�@

error_RdkE?

learning_rate_1�O�6�a�0I       6%�	Qn\���A�*;


total_loss!9�@

error_R�
[?

learning_rate_1�O�6�I��I       6%�	�dn\���A�*;


total_loss�@

error_RߤJ?

learning_rate_1�O�6���I       6%�	w�n\���A�*;


total_loss���@

error_R6TR?

learning_rate_1�O�6�2EI       6%�	No\���A�*;


total_lossw"�@

error_R;�R?

learning_rate_1�O�6Va�I       6%�	�No\���A�*;


total_loss-��@

error_R,�_?

learning_rate_1�O�6�0øI       6%�	.�o\���A�*;


total_loss1��@

error_R�V?

learning_rate_1�O�6�	AI       6%�	c�o\���A�*;


total_loss���@

error_R�N?

learning_rate_1�O�6��?I       6%�	%p\���A�*;


total_loss��@

error_RۆV?

learning_rate_1�O�6��I       6%�	�Xp\���A�*;


total_loss�lv@

error_R��^?

learning_rate_1�O�6��k�I       6%�	��p\���A�*;


total_loss��@

error_R� P?

learning_rate_1�O�6��FI       6%�	��p\���A�*;


total_lossWa@

error_RCY5?

learning_rate_1�O�6$KKI       6%�	8#q\���A�*;


total_loss�#�@

error_R��_?

learning_rate_1�O�6���I       6%�	�gq\���A�*;


total_loss�X�@

error_R{J?

learning_rate_1�O�6�F�I       6%�	�q\���A�*;


total_loss���@

error_R1w]?

learning_rate_1�O�6��zwI       6%�	i�q\���A�*;


total_loss�T�@

error_Rs�U?

learning_rate_1�O�6�@�)I       6%�	.7r\���A�*;


total_loss���@

error_R� X?

learning_rate_1�O�6X��I       6%�	�r\���A�*;


total_loss�:�@

error_R\[?

learning_rate_1�O�6��I       6%�	��r\���A�*;


total_lossqm�@

error_Rͣ`?

learning_rate_1�O�6|�ϩI       6%�	�s\���A�*;


total_lossg@

error_R�uW?

learning_rate_1�O�6��I       6%�	�Os\���A�*;


total_loss���@

error_R��K?

learning_rate_1�O�6�wt�I       6%�	W�s\���A�*;


total_loss���@

error_R��C?

learning_rate_1�O�6Sp*�I       6%�	}�s\���A�*;


total_loss�A

error_RdA?

learning_rate_1�O�6���I       6%�	�t\���A�*;


total_lossH4�@

error_R��G?

learning_rate_1�O�62~��I       6%�	�]t\���A�*;


total_loss��A

error_RM7?

learning_rate_1�O�6�a�I       6%�	p�t\���A�*;


total_loss2[�@

error_RśS?

learning_rate_1�O�6[��I       6%�	��t\���A�*;


total_loss��@

error_R @d?

learning_rate_1�O�6�a�I       6%�	�,u\���A�*;


total_loss��A

error_R<V?

learning_rate_1�O�6��|I       6%�	su\���A�*;


total_loss�4�@

error_R)U?

learning_rate_1�O�6=�REI       6%�	��u\���A�*;


total_lossj�@

error_Rl�<?

learning_rate_1�O�6$_�pI       6%�	� v\���A�*;


total_loss�-�@

error_R�W?

learning_rate_1�O�6/|rI       6%�	�Ov\���A�*;


total_loss�V�@

error_R�0W?

learning_rate_1�O�6./��I       6%�	�v\���A�*;


total_loss4��@

error_R�T?

learning_rate_1�O�6cgGI       6%�	W�v\���A�*;


total_lossX��@

error_R��M?

learning_rate_1�O�6��OQI       6%�	7w\���A�*;


total_loss��@

error_RC�^?

learning_rate_1�O�6�m{I       6%�	_cw\���A�*;


total_loss���@

error_R��Q?

learning_rate_1�O�6,=�5I       6%�	�w\���A�*;


total_loss�M�@

error_RVkG?

learning_rate_1�O�6R���I       6%�	D�w\���A�*;


total_loss:��@

error_R�_M?

learning_rate_1�O�644_�I       6%�	.x\���A�*;


total_loss
Ӷ@

error_R�P?

learning_rate_1�O�6$c?TI       6%�	�ox\���A�*;


total_lossؼ�@

error_R�??

learning_rate_1�O�6��KI       6%�	˳x\���A�*;


total_lossQ��@

error_R!�F?

learning_rate_1�O�6�g-�I       6%�	e�x\���A�*;


total_loss8C�@

error_REH?

learning_rate_1�O�6u�<I       6%�	�:y\���A�*;


total_lossoT@

error_R�&P?

learning_rate_1�O�6	m!ZI       6%�	�}y\���A�*;


total_lossXa�@

error_RڭA?

learning_rate_1�O�6��(I       6%�	a�y\���A�*;


total_loss��@

error_R_OP?

learning_rate_1�O�6���I       6%�	�z\���A�*;


total_loss�
�@

error_R��T?

learning_rate_1�O�6j܉�I       6%�	�Fz\���A�*;


total_loss�^A

error_R�)?

learning_rate_1�O�6ts��I       6%�	��z\���A�*;


total_lossyA

error_R��M?

learning_rate_1�O�6FI       6%�	��z\���A�*;


total_loss��@

error_RAe?

learning_rate_1�O�6���AI       6%�	{\���A�*;


total_loss�e
A

error_R[(H?

learning_rate_1�O�6��=I       6%�	�W{\���A�*;


total_loss;�
A

error_R�J?

learning_rate_1�O�6��myI       6%�	Ν{\���A�*;


total_loss|ڱ@

error_R �I?

learning_rate_1�O�6�D#I       6%�	�|\���A�*;


total_loss���@

error_R��R?

learning_rate_1�O�6za5�I       6%�	�J|\���A�*;


total_loss��@

error_R{%Z?

learning_rate_1�O�6�Ó�I       6%�	<�|\���A�*;


total_loss)�@

error_R��D?

learning_rate_1�O�6,�eI       6%�	�|\���A�*;


total_loss�Ԯ@

error_R��I?

learning_rate_1�O�6��Z�I       6%�	>}\���A�*;


total_loss�e�@

error_RF�S?

learning_rate_1�O�6�A��I       6%�	�d}\���A�*;


total_loss/�@

error_R[oA?

learning_rate_1�O�6[;|/I       6%�	��}\���A�*;


total_loss�j�@

error_R��K?

learning_rate_1�O�6�/&I       6%�	��}\���A�*;


total_loss__�@

error_RpP?

learning_rate_1�O�6|��1I       6%�	�3~\���A�*;


total_lossLa�@

error_ReTN?

learning_rate_1�O�6�_6I       6%�	-v~\���A�*;


total_lossc�@

error_RQ�\?

learning_rate_1�O�6G�ҥI       6%�	��~\���A�*;


total_loss�<�@

error_R�A?

learning_rate_1�O�6�̰�I       6%�	_�~\���A�*;


total_lossC�@

error_R��F?

learning_rate_1�O�6�Jp�I       6%�	$F\���A�*;


total_lossp��@

error_RثW?

learning_rate_1�O�6 �I       6%�	݋\���A�*;


total_losst�@

error_R��F?

learning_rate_1�O�6nw�I       6%�	|�\���A�*;


total_loss�$�@

error_R�vT?

learning_rate_1�O�6�l��I       6%�	,�\���A�*;


total_loss#�@

error_R8/=?

learning_rate_1�O�6�p��I       6%�	�b�\���A�*;


total_lossrw�@

error_R}>=?

learning_rate_1�O�6&��0I       6%�	��\���A�*;


total_lossT/�@

error_R�eU?

learning_rate_1�O�6��R
I       6%�	=�\���A�*;


total_lossL��@

error_R��E?

learning_rate_1�O�6#�kI       6%�	G2�\���A�*;


total_loss��d@

error_RA�K?

learning_rate_1�O�6S��I       6%�	�u�\���A�*;


total_loss|��@

error_R�d6?

learning_rate_1�O�6��)aI       6%�	���\���A�*;


total_lossb��@

error_R$�X?

learning_rate_1�O�6+���I       6%�	���\���A�*;


total_lossg�@

error_R��A?

learning_rate_1�O�6��ٰI       6%�	`D�\���A�*;


total_loss(ȹ@

error_Ri�L?

learning_rate_1�O�6���I       6%�	��\���A�*;


total_lossz��@

error_Rܸ@?

learning_rate_1�O�6xMPI       6%�	�т\���A�*;


total_loss�Ė@

error_R�L?

learning_rate_1�O�6�*�
I       6%�	��\���A�*;


total_loss9��@

error_R�"A?

learning_rate_1�O�63�I       6%�	VV�\���A�*;


total_loss�_�@

error_R�Z?

learning_rate_1�O�6b2�BI       6%�	$��\���A�*;


total_loss��@

error_Rt�Z?

learning_rate_1�O�6�+�I       6%�	)�\���A�*;


total_loss�A

error_Rnm\?

learning_rate_1�O�6��qI       6%�	bL�\���A�*;


total_loss�Y�@

error_R�yI?

learning_rate_1�O�6a�GeI       6%�	%��\���A�*;


total_loss���@

error_R��F?

learning_rate_1�O�6�^6*I       6%�	Iׄ\���A�*;


total_loss��@

error_RX\?

learning_rate_1�O�6�K��I       6%�	u�\���A�*;


total_lossR�@

error_R E?

learning_rate_1�O�6���I       6%�	b�\���A�*;


total_loss�.o@

error_R,�O?

learning_rate_1�O�6k�I       6%�	���\���A�*;


total_loss	��@

error_RqN?

learning_rate_1�O�6�(��I       6%�	�\���A�*;


total_loss��A

error_R��M?

learning_rate_1�O�6e��I       6%�	#K�\���A�*;


total_loss �@

error_R��M?

learning_rate_1�O�6��}I       6%�	$��\���A�*;


total_loss��@

error_R��F?

learning_rate_1�O�6�͉I       6%�	;ֆ\���A�*;


total_loss��@

error_R��C?

learning_rate_1�O�6�TI       6%�	S�\���A�*;


total_loss{��@

error_R��4?

learning_rate_1�O�6pD��I       6%�	v]�\���A�*;


total_loss���@

error_RKb?

learning_rate_1�O�6V7KI       6%�	���\���A�*;


total_loss�:�@

error_R�(H?

learning_rate_1�O�6��y�I       6%�	Y�\���A�*;


total_loss?r�@

error_R�Q?

learning_rate_1�O�6��NI       6%�	�&�\���A�*;


total_lossR�@

error_R_�S?

learning_rate_1�O�6Ĭ��I       6%�	Gh�\���A�*;


total_loss�~�@

error_R)�K?

learning_rate_1�O�6���I       6%�	��\���A�*;


total_loss�ڰ@

error_R`�A?

learning_rate_1�O�6�m�KI       6%�	m��\���A�*;


total_loss��@

error_RݔE?

learning_rate_1�O�6Gē�I       6%�	 E�\���A�*;


total_loss�o�@

error_Re�G?

learning_rate_1�O�6��jI       6%�	���\���A�*;


total_lossdNo@

error_RHz=?

learning_rate_1�O�6���I       6%�	Nщ\���A�*;


total_loss��@

error_Rf�Q?

learning_rate_1�O�6�N�0I       6%�	��\���A�*;


total_loss{��@

error_R`V?

learning_rate_1�O�66g�I       6%�	zd�\���A�*;


total_loss�~�@

error_R��E?

learning_rate_1�O�6�f�I       6%�	���\���A�*;


total_loss�v�@

error_R�	b?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss��@

error_R�V@?

learning_rate_1�O�6WG�I       6%�	~6�\���A�*;


total_lossS��@

error_R�yC?

learning_rate_1�O�600�I       6%�	|�\���A�*;


total_losstQA

error_Rj�D?

learning_rate_1�O�6��`I       6%�	�؋\���A�*;


total_loss�D�@

error_R*Q?

learning_rate_1�O�6���UI       6%�	�*�\���A�*;


total_loss���@

error_R=�P?

learning_rate_1�O�6�q�	I       6%�	�x�\���A�*;


total_loss|^�@

error_Ra�M?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss��@

error_R3�Y?

learning_rate_1�O�6��VI       6%�	a�\���A�*;


total_lossO3�@

error_RnT?

learning_rate_1�O�6����I       6%�	K�\���A�*;


total_loss�\�@

error_RiEN?

learning_rate_1�O�6��@WI       6%�	"��\���A�*;


total_loss-b�@

error_R�2P?

learning_rate_1�O�6�:��I       6%�	�Ս\���A�*;


total_loss�x�@

error_R=/Q?

learning_rate_1�O�6�i�I       6%�	,"�\���A�*;


total_loss��A

error_R�Y?

learning_rate_1�O�6$-]�I       6%�	�j�\���A�*;


total_loss���@

error_R��9?

learning_rate_1�O�6�¥-I       6%�	���\���A�*;


total_loss2ۄ@

error_R{6?

learning_rate_1�O�6�w��I       6%�	��\���A�*;


total_loss�1D@

error_R�W?

learning_rate_1�O�6y�hI       6%�	�>�\���A�*;


total_loss�J�@

error_RqnS?

learning_rate_1�O�6)
kI       6%�	P��\���A�*;


total_loss���@

error_R_%B?

learning_rate_1�O�6����I       6%�	~ʏ\���A�*;


total_lossW9�@

error_R1O?

learning_rate_1�O�6�'+I       6%�	2�\���A�*;


total_loss�V�@

error_R�TQ?

learning_rate_1�O�6�I       6%�	tS�\���A�*;


total_loss}&�@

error_RiB>?

learning_rate_1�O�6v��I       6%�	���\���A�*;


total_loss
�@

error_R{�T?

learning_rate_1�O�6PіI       6%�	��\���A�*;


total_lossgDA

error_RivQ?

learning_rate_1�O�6a���I       6%�	*$�\���A�*;


total_loss��@

error_R&Y?

learning_rate_1�O�6s�I       6%�	e�\���A�*;


total_lossC�A

error_R�%K?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss�k@

error_RNj@?

learning_rate_1�O�66�j�I       6%�	~�\���A�*;


total_loss�@

error_R��`?

learning_rate_1�O�6��I       6%�	�0�\���A�*;


total_loss]t�@

error_R�_@?

learning_rate_1�O�6S�ƚI       6%�	�w�\���A�*;


total_lossiv@

error_R��[?

learning_rate_1�O�6��O&I       6%�	ǿ�\���A�*;


total_lossV߯@

error_R]S?

learning_rate_1�O�6��:I       6%�	2�\���A�*;


total_loss�P�@

error_R�!E?

learning_rate_1�O�6J�
I       6%�	�F�\���A�*;


total_loss���@

error_Rq�V?

learning_rate_1�O�6�E�I       6%�	打\���A�*;


total_loss��@

error_R}M]?

learning_rate_1�O�6�)ziI       6%�	Q̓\���A�*;


total_loss�u�@

error_Rx�F?

learning_rate_1�O�6��)�I       6%�	��\���A�*;


total_lossn�@

error_R�eX?

learning_rate_1�O�6R.)cI       6%�	�U�\���A�*;


total_loss	A�@

error_R�Y?

learning_rate_1�O�6RV"mI       6%�	0��\���A�*;


total_loss¶A

error_R@vD?

learning_rate_1�O�6e ��I       6%�	�ݔ\���A�*;


total_loss5A�@

error_R gU?

learning_rate_1�O�6��GI       6%�	� �\���A�*;


total_lossa@�@

error_R I?

learning_rate_1�O�6�(�,I       6%�	�d�\���A�*;


total_loss?a�@

error_R�G?

learning_rate_1�O�6p��I       6%�	���\���A�*;


total_loss���@

error_R\�F?

learning_rate_1�O�6���CI       6%�	��\���A�*;


total_loss[�@

error_R�}W?

learning_rate_1�O�6Z�RI       6%�	k/�\���A�*;


total_loss[�@

error_RH�H?

learning_rate_1�O�6?��I       6%�	�s�\���A�*;


total_loss`��@

error_R^X?

learning_rate_1�O�6�]�I       6%�	#��\���A�*;


total_lossW��@

error_RϰD?

learning_rate_1�O�6!��I       6%�	���\���A�*;


total_loss�|�@

error_R�\Q?

learning_rate_1�O�6�I       6%�	�I�\���A�*;


total_loss��@

error_R�}M?

learning_rate_1�O�6P�זI       6%�	鏗\���A�*;


total_loss�1�@

error_R�K?

learning_rate_1�O�6�	�I       6%�	ؗ\���A�*;


total_lossq�A

error_RÞP?

learning_rate_1�O�6�I       6%�	!�\���A�*;


total_lossƎ�@

error_R�W?

learning_rate_1�O�6vKI       6%�	A^�\���A�*;


total_loss���@

error_R�K?

learning_rate_1�O�6��°I       6%�	䢘\���A�*;


total_loss֤�@

error_ROP?

learning_rate_1�O�6�}��I       6%�	u�\���A�*;


total_loss�,�@

error_R�K?

learning_rate_1�O�67�#�I       6%�	�*�\���A�*;


total_loss��@

error_R��K?

learning_rate_1�O�6�H��I       6%�	=t�\���A�*;


total_loss���@

error_R��C?

learning_rate_1�O�6��"}I       6%�	B��\���A�*;


total_loss
I�@

error_R�,N?

learning_rate_1�O�6Tt�%I       6%�	Z��\���A�*;


total_loss)��@

error_R��K?

learning_rate_1�O�6<0�I       6%�	EF�\���A�*;


total_loss���@

error_RH�X?

learning_rate_1�O�6Z�3+I       6%�	h��\���A�*;


total_loss���@

error_R�C?

learning_rate_1�O�6���bI       6%�	Rۚ\���A�*;


total_loss���@

error_R
=O?

learning_rate_1�O�6Ɂ[�I       6%�	�!�\���A�*;


total_lossj�@

error_R��Z?

learning_rate_1�O�6/n%oI       6%�	ie�\���A�*;


total_loss;��@

error_R�fZ?

learning_rate_1�O�6R@�'I       6%�	I��\���A�*;


total_loss_p�@

error_R|IA?

learning_rate_1�O�65= �I       6%�	� �\���A�*;


total_lossbǐ@

error_R�M?

learning_rate_1�O�6��4I       6%�	ak�\���A�*;


total_loss.�@

error_R��O?

learning_rate_1�O�6�v�	I       6%�	���\���A�*;


total_loss���@

error_RH�F?

learning_rate_1�O�6�Y�UI       6%�	��\���A�*;


total_loss�"�@

error_R
j8?

learning_rate_1�O�6Ӝ�I       6%�	�8�\���A�*;


total_losst�@

error_R�~O?

learning_rate_1�O�6��I       6%�	�~�\���A�*;


total_loss�A

error_Rw�U?

learning_rate_1�O�6n�7I       6%�	ĝ\���A�*;


total_lossnߨ@

error_R�+\?

learning_rate_1�O�6��I       6%�	�
�\���A�*;


total_lossXƜ@

error_R�GS?

learning_rate_1�O�6�c��I       6%�	�R�\���A�*;


total_loss��@

error_R��@?

learning_rate_1�O�6@ٓ�I       6%�	���\���A�*;


total_loss1�@

error_REL?

learning_rate_1�O�6YҝI       6%�	v��\���A�*;


total_loss�H�@

error_R�|E?

learning_rate_1�O�6����I       6%�	b9�\���A�*;


total_loss�	�@

error_R��[?

learning_rate_1�O�6���lI       6%�	F��\���A�*;


total_loss���@

error_Rv�A?

learning_rate_1�O�6�ؚI       6%�	�̟\���A�*;


total_loss��@

error_RT�\?

learning_rate_1�O�67��I       6%�	�"�\���A�*;


total_loss� �@

error_R;�Q?

learning_rate_1�O�6�i�I       6%�	@q�\���A�*;


total_loss1�@

error_RlK?

learning_rate_1�O�6����I       6%�	}��\���A�*;


total_lossH��@

error_RiFD?

learning_rate_1�O�6"4�I       6%�	 �\���A�*;


total_loss��@

error_R�L?

learning_rate_1�O�6���I       6%�	�D�\���A�*;


total_loss�K�@

error_R�}T?

learning_rate_1�O�6�I       6%�	5��\���A�*;


total_lossf�@

error_RE�S?

learning_rate_1�O�6r�W�I       6%�	Nϡ\���A�*;


total_loss���@

error_R��J?

learning_rate_1�O�6�CI       6%�	��\���A�*;


total_loss|�@

error_RIjK?

learning_rate_1�O�6�,��I       6%�	�[�\���A�*;


total_loss7~�@

error_R�0B?

learning_rate_1�O�6�#9WI       6%�	U��\���A�*;


total_loss$.�@

error_RT�A?

learning_rate_1�O�6�(�I       6%�	~�\���A�*;


total_loss�6�@

error_R4%A?

learning_rate_1�O�6��i�I       6%�	�,�\���A�*;


total_lossя�@

error_Rs"F?

learning_rate_1�O�6
��MI       6%�	�q�\���A�*;


total_loss�Z�@

error_RS}M?

learning_rate_1�O�6ك�XI       6%�	���\���A�*;


total_loss
� A

error_Ri�U?

learning_rate_1�O�6�<�I       6%�	_�\���A�*;


total_loss6�@

error_R��H?

learning_rate_1�O�6���I       6%�	AL�\���A�*;


total_loss���@

error_R�6W?

learning_rate_1�O�6+�;�I       6%�	䍤\���A�*;


total_loss �@

error_R�R?

learning_rate_1�O�6EnЧI       6%�	�Τ\���A�*;


total_loss{A

error_RdcM?

learning_rate_1�O�6n��)I       6%�	��\���A�*;


total_loss��@

error_R��W?

learning_rate_1�O�6��.�I       6%�	�R�\���A�*;


total_loss�=�@

error_R�)I?

learning_rate_1�O�66�I       6%�	��\���A�*;


total_lossV�@

error_R�E?

learning_rate_1�O�6i.l3I       6%�	ڥ\���A�*;


total_loss���@

error_R�
_?

learning_rate_1�O�6��zFI       6%�	��\���A�*;


total_loss|w�@

error_R��D?

learning_rate_1�O�6�%��I       6%�	�]�\���A�*;


total_lossT�w@

error_R.}L?

learning_rate_1�O�657˅I       6%�	���\���A�*;


total_loss�`�@

error_R��B?

learning_rate_1�O�6���]I       6%�	y�\���A�*;


total_loss�f�@

error_R��Y?

learning_rate_1�O�6C�VjI       6%�	)�\���A�*;


total_loss��@

error_R�P?

learning_rate_1�O�6igI�I       6%�	�o�\���A�*;


total_loss.�@

error_R�P?

learning_rate_1�O�6�C�tI       6%�	��\���A�*;


total_lossҗ�@

error_R�hQ?

learning_rate_1�O�6^��AI       6%�	^��\���A�*;


total_loss�X�@

error_R=hK?

learning_rate_1�O�6��˪I       6%�	�;�\���A�*;


total_loss�bA

error_R��N?

learning_rate_1�O�6��hI       6%�	�~�\���A�*;


total_loss���@

error_R�lI?

learning_rate_1�O�6�H��I       6%�	cè\���A�*;


total_loss���@

error_R:gI?

learning_rate_1�O�6��!�I       6%�	'�\���A�*;


total_lossb�@

error_R�rE?

learning_rate_1�O�6��I       6%�	�J�\���A�*;


total_loss.\�@

error_R}�G?

learning_rate_1�O�6淚+I       6%�	��\���A�*;


total_loss���@

error_Rl`?

learning_rate_1�O�6Gd�I       6%�	cѩ\���A�*;


total_lossĊ�@

error_R3T?

learning_rate_1�O�6R��I       6%�	��\���A�*;


total_loss9�
A

error_R/ZQ?

learning_rate_1�O�6J��KI       6%�	-_�\���A�*;


total_loss�`�@

error_R�d:?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss��@

error_R��S?

learning_rate_1�O�6�ׯ�I       6%�	��\���A�*;


total_lossH��@

error_R�3T?

learning_rate_1�O�6)�sI       6%�	'�\���A�*;


total_lossM�@

error_R{CW?

learning_rate_1�O�6.��HI       6%�	�k�\���A�*;


total_loss��@

error_R�pM?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss���@

error_RO4T?

learning_rate_1�O�6]���I       6%�	��\���A�*;


total_loss���@

error_R��N?

learning_rate_1�O�6A�I       6%�	MZ�\���A�*;


total_loss�0�@

error_R�8@?

learning_rate_1�O�6�㉁I       6%�	۠�\���A�*;


total_loss��@

error_RږM?

learning_rate_1�O�6l_v^I       6%�	��\���A�*;


total_loss��t@

error_R�B?

learning_rate_1�O�6�։�I       6%�	-�\���A�*;


total_loss�>�@

error_R��a?

learning_rate_1�O�6`PI       6%�	rn�\���A�*;


total_lossw�@

error_R��S?

learning_rate_1�O�64��[I       6%�	���\���A�*;


total_loss��@

error_R�_c?

learning_rate_1�O�6�-�I       6%�	��\���A�*;


total_loss���@

error_R�X?

learning_rate_1�O�6ƯK�I       6%�	:A�\���A�*;


total_loss��@

error_Rn}W?

learning_rate_1�O�6ɩ��I       6%�	.��\���A�*;


total_loss�-�@

error_R��Y?

learning_rate_1�O�6���I       6%�	�Ǯ\���A�*;


total_loss��@

error_R��K?

learning_rate_1�O�6�8��I       6%�	�
�\���A�*;


total_lossH}�@

error_R�Q?

learning_rate_1�O�6$�ڱI       6%�	�W�\���A�*;


total_loss��@

error_RV�M?

learning_rate_1�O�6�l$TI       6%�	��\���A�*;


total_loss���@

error_R-zc?

learning_rate_1�O�63x�I       6%�	��\���A�*;


total_loss���@

error_R�X?

learning_rate_1�O�6m���I       6%�	�3�\���A�*;


total_loss6J�@

error_R_X<?

learning_rate_1�O�6Q͢uI       6%�	Ev�\���A�*;


total_loss���@

error_R��K?

learning_rate_1�O�6��I       6%�	~��\���A�*;


total_loss�_�@

error_R�0c?

learning_rate_1�O�6(N�I       6%�	b�\���A�*;


total_lossV֨@

error_R�1U?

learning_rate_1�O�6�z�I       6%�	I�\���A�*;


total_loss`j�@

error_R��=?

learning_rate_1�O�6�P��I       6%�	���\���A�*;


total_loss
k@

error_Rl�K?

learning_rate_1�O�6H/p�I       6%�	Yͱ\���A�*;


total_loss�o�@

error_RҤE?

learning_rate_1�O�6���I       6%�	��\���A�*;


total_loss���@

error_R�]?

learning_rate_1�O�6#��fI       6%�	�V�\���A�*;


total_loss��@

error_RQ�Q?

learning_rate_1�O�6[Ta�I       6%�	��\���A�*;


total_loss�>�@

error_RC[?

learning_rate_1�O�6�d�#I       6%�	��\���A�*;


total_loss�@

error_R�jW?

learning_rate_1�O�6���eI       6%�	]&�\���A�*;


total_loss��@

error_R}8S?

learning_rate_1�O�6//�I       6%�	Ui�\���A�*;


total_lossq�@

error_R�`V?

learning_rate_1�O�6Y�U�I       6%�	��\���A�*;


total_loss��@

error_R�_?

learning_rate_1�O�60���I       6%�	�\���A�*;


total_losst�z@

error_R�T?

learning_rate_1�O�6�C�I       6%�	5�\���A�*;


total_lossP�@

error_R�M?

learning_rate_1�O�6�Ҡ�I       6%�	���\���A�*;


total_loss�u�@

error_R)�\?

learning_rate_1�O�6P�y�I       6%�	Kɴ\���A�*;


total_lossH}�@

error_R��P?

learning_rate_1�O�61�̕I       6%�	��\���A�*;


total_lossx��@

error_R,�S?

learning_rate_1�O�6B��I       6%�	R�\���A�*;


total_loss��@

error_RZ`g?

learning_rate_1�O�6�§zI       6%�	�\���A�*;


total_lossl��@

error_RV�J?

learning_rate_1�O�6��5^I       6%�	��\���A�*;


total_loss}�@

error_R�-<?

learning_rate_1�O�6i��#I       6%�	c1�\���A�*;


total_loss�|v@

error_RۚB?

learning_rate_1�O�6*~h`I       6%�	bw�\���A�*;


total_loss���@

error_RF�G?

learning_rate_1�O�6�`�/I       6%�	q��\���A�*;


total_loss6�@

error_R��P?

learning_rate_1�O�6��XI       6%�	�\���A�*;


total_loss�8�@

error_R&??

learning_rate_1�O�6��wJI       6%�	�C�\���A�*;


total_loss4�@

error_R�2_?

learning_rate_1�O�6�I       6%�	���\���A�*;


total_loss-��@

error_Rd;J?

learning_rate_1�O�6g/��I       6%�	�ɷ\���A�*;


total_loss���@

error_R�N?

learning_rate_1�O�61z�I       6%�	�\���A�*;


total_loss]K�@

error_Rt�L?

learning_rate_1�O�6j�W�I       6%�	yQ�\���A�*;


total_loss* A

error_RY?

learning_rate_1�O�6��NI       6%�	u��\���A�*;


total_loss�a�@

error_R��J?

learning_rate_1�O�6%��&I       6%�	 ָ\���A�*;


total_loss,E�@

error_R/?

learning_rate_1�O�6`��+I       6%�	��\���A�*;


total_loss#�@

error_RwOV?

learning_rate_1�O�6�k'�I       6%�	�[�\���A�*;


total_loss3��@

error_R�	S?

learning_rate_1�O�6mJ��I       6%�	۠�\���A�*;


total_loss��@

error_R�KT?

learning_rate_1�O�64�7I       6%�	��\���A�*;


total_losss߮@

error_Rn:?

learning_rate_1�O�6��B0I       6%�	�'�\���A�*;


total_loss�1�@

error_R��N?

learning_rate_1�O�6�[FI       6%�	�l�\���A�*;


total_loss��{@

error_RlS?

learning_rate_1�O�6�ks�I       6%�	���\���A�*;


total_loss���@

error_R��D?

learning_rate_1�O�6���I       6%�	^ �\���A�*;


total_loss���@

error_R�JN?

learning_rate_1�O�6��chI       6%�	gI�\���A�*;


total_lossV�@

error_R�0Q?

learning_rate_1�O�6����I       6%�	y��\���A�*;


total_loss�+�@

error_R�>?

learning_rate_1�O�6��%I       6%�	$��\���A�*;


total_loss���@

error_R.R?

learning_rate_1�O�68�5vI       6%�	�C�\���A�*;


total_loss�k�@

error_R�O?

learning_rate_1�O�6p�νI       6%�	��\���A�*;


total_loss߶�@

error_R��A?

learning_rate_1�O�6T��CI       6%�	μ\���A�*;


total_loss��@

error_R�EI?

learning_rate_1�O�6�σI       6%�	��\���A�*;


total_loss���@

error_R�!P?

learning_rate_1�O�6i��'I       6%�	7W�\���A�*;


total_lossDQ�@

error_R,^B?

learning_rate_1�O�6Yf�I       6%�	ʢ�\���A�*;


total_loss% �@

error_R��I?

learning_rate_1�O�6��ڥI       6%�	}�\���A�*;


total_loss�I�@

error_RߦN?

learning_rate_1�O�6^"�I       6%�	0�\���A�*;


total_loss�rA

error_RyM?

learning_rate_1�O�6����I       6%�	os�\���A�*;


total_loss���@

error_R-G?

learning_rate_1�O�6*-�[I       6%�	^��\���A�*;


total_loss\�@

error_R�U?

learning_rate_1�O�62���I       6%�	V�\���A�*;


total_loss��@

error_R�	I?

learning_rate_1�O�6L�E�I       6%�	�Z�\���A�*;


total_lossų�@

error_R��=?

learning_rate_1�O�6�uAI       6%�	A��\���A�*;


total_loss�!z@

error_R�N?

learning_rate_1�O�6.�W�I       6%�	~�\���A�*;


total_loss���@

error_R�lX?

learning_rate_1�O�63_�I       6%�	
N�\���A�*;


total_loss�R�@

error_R	�L?

learning_rate_1�O�6ؕEI       6%�	Y��\���A�*;


total_loss���@

error_R�kI?

learning_rate_1�O�6z��I       6%�	Z��\���A�*;


total_loss^�@

error_R��M?

learning_rate_1�O�6�5�I       6%�	"$�\���A�*;


total_loss,��@

error_Rn�]?

learning_rate_1�O�6A2
I       6%�	�l�\���A�*;


total_loss��q@

error_R�kh?

learning_rate_1�O�6׼e�I       6%�	ݲ�\���A�*;


total_loss�e�@

error_R;�J?

learning_rate_1�O�6��&�I       6%�	���\���A�*;


total_loss)��@

error_R��R?

learning_rate_1�O�6�A�I       6%�	�;�\���A�*;


total_lossʍ�@

error_R4�_?

learning_rate_1�O�6�w��I       6%�	���\���A�*;


total_loss��@

error_R��D?

learning_rate_1�O�6kXoI       6%�	���\���A�*;


total_lossj�@

error_R��Y?

learning_rate_1�O�6*��)I       6%�	��\���A�*;


total_loss��@

error_R)�V?

learning_rate_1�O�6�p�NI       6%�	�^�\���A�*;


total_lossO%�@

error_RRO?

learning_rate_1�O�6��",I       6%�	q��\���A�*;


total_loss���@

error_R�0W?

learning_rate_1�O�6��I       6%�	]��\���A�*;


total_loss�J�@

error_R��S?

learning_rate_1�O�6U:wSI       6%�	-*�\���A�*;


total_loss�@

error_R��M?

learning_rate_1�O�6��q"I       6%�	�k�\���A�*;


total_loss�X�@

error_R3�F?

learning_rate_1�O�6j�I       6%�	}��\���A�*;


total_lossĆ�@

error_R�O?

learning_rate_1�O�6�tPI       6%�	���\���A�*;


total_loss��@

error_R�z^?

learning_rate_1�O�6���I       6%�	�8�\���A�*;


total_loss�(�@

error_R��S?

learning_rate_1�O�6WYI       6%�	��\���A�*;


total_loss���@

error_RSM?

learning_rate_1�O�6j�?_I       6%�	��\���A�*;


total_loss ��@

error_RXwE?

learning_rate_1�O�6��B&I       6%�	��\���A�*;


total_loss���@

error_RJ�Q?

learning_rate_1�O�6�b�I       6%�	O�\���A�*;


total_lossI��@

error_RÖ^?

learning_rate_1�O�6�1y7I       6%�	n��\���A�*;


total_loss���@

error_RC?

learning_rate_1�O�6R�I       6%�	E��\���A�*;


total_lossHѻ@

error_Rl'b?

learning_rate_1�O�6�&�I       6%�	��\���A�*;


total_loss&�@

error_R�3]?

learning_rate_1�O�6���I       6%�	]a�\���A�*;


total_loss�@

error_R�oM?

learning_rate_1�O�6&���I       6%�	���\���A�*;


total_loss:��@

error_R�3?

learning_rate_1�O�6�5��I       6%�	���\���A�*;


total_loss�Ū@

error_R_�T?

learning_rate_1�O�6JaA�I       6%�	�-�\���A�*;


total_loss���@

error_R�I?

learning_rate_1�O�61|�I       6%�	Fs�\���A�*;


total_losss�@

error_R@�L?

learning_rate_1�O�6��I       6%�	L��\���A�*;


total_loss
#�@

error_R�ZV?

learning_rate_1�O�6��I       6%�	��\���A�*;


total_loss̀�@

error_RR?U?

learning_rate_1�O�62��I       6%�	�H�\���A�*;


total_lossE��@

error_Ra?

learning_rate_1�O�6����I       6%�	���\���A�*;


total_loss1��@

error_RS?

learning_rate_1�O�6 X��I       6%�	 ��\���A�*;


total_loss�@�@

error_R,�D?

learning_rate_1�O�6�TI       6%�	$�\���A�*;


total_loss�p�@

error_R2GE?

learning_rate_1�O�6ҩ��I       6%�	lm�\���A�*;


total_loss&�@

error_R-�U?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_lossi�@

error_R��O?

learning_rate_1�O�6e^ҘI       6%�	� �\���A�*;


total_loss���@

error_R�aV?

learning_rate_1�O�6V9ݾI       6%�	4K�\���A�*;


total_loss=��@

error_R�N?

learning_rate_1�O�6w�~I       6%�	~��\���A�*;


total_loss���@

error_RdbC?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss���@

error_Rd'[?

learning_rate_1�O�6�3ObI       6%�	=�\���A�*;


total_lossF��@

error_R�PO?

learning_rate_1�O�6oh��I       6%�	C~�\���A�*;


total_loss�A

error_R3!@?

learning_rate_1�O�6݂ԜI       6%�	N��\���A�*;


total_loss3��@

error_R&�@?

learning_rate_1�O�6ci�I       6%�	�\���A�*;


total_loss�c�@

error_R�*K?

learning_rate_1�O�66*��I       6%�	�F�\���A�*;


total_loss$i�@

error_R6AT?

learning_rate_1�O�6��I       6%�	
��\���A�*;


total_loss�B�@

error_R��Y?

learning_rate_1�O�6R��I       6%�	��\���A�*;


total_loss�G�@

error_R��[?

learning_rate_1�O�6f'�I       6%�	]�\���A�*;


total_loss�@

error_R��i?

learning_rate_1�O�6�I3�I       6%�	Jd�\���A�*;


total_loss��@

error_RuL?

learning_rate_1�O�6M�c�I       6%�	ߩ�\���A�*;


total_lossoc�@

error_R}H?

learning_rate_1�O�6���ZI       6%�	&��\���A�*;


total_lossOA

error_R�HX?

learning_rate_1�O�6����I       6%�	�6�\���A�*;


total_loss�@

error_R�\\?

learning_rate_1�O�6su��I       6%�	�|�\���A�*;


total_loss�@

error_R��H?

learning_rate_1�O�6I�I       6%�	���\���A�*;


total_loss}��@

error_R� <?

learning_rate_1�O�6t��\I       6%�	�\���A�*;


total_loss/�@

error_R2)X?

learning_rate_1�O�6"�}I       6%�	�C�\���A�*;


total_lossW�@

error_R��Q?

learning_rate_1�O�6ahs�I       6%�	��\���A�*;


total_loss17�@

error_R�@?

learning_rate_1�O�6�Ic�I       6%�	+��\���A�*;


total_loss�I�@

error_R�4?

learning_rate_1�O�66�Q�I       6%�	��\���A�*;


total_loss�@

error_R�T?

learning_rate_1�O�6��i�I       6%�	�^�\���A�*;


total_lossB�@

error_RÊE?

learning_rate_1�O�6��!?I       6%�	���\���A�*;


total_loss�D�@

error_R�N?

learning_rate_1�O�6R bI       6%�	���\���A�*;


total_loss�`�@

error_R�N?

learning_rate_1�O�6�(��I       6%�	�)�\���A�*;


total_loss���@

error_R�F?

learning_rate_1�O�6�y��I       6%�	�p�\���A�*;


total_lossQ�@

error_Rl�B?

learning_rate_1�O�6���wI       6%�	���\���A�*;


total_loss��@

error_Rh�G?

learning_rate_1�O�6���I       6%�	H��\���A�*;


total_loss,��@

error_R�N?

learning_rate_1�O�6��!I       6%�	q6�\���A�*;


total_loss][�@

error_R��N?

learning_rate_1�O�6%]#I       6%�	k�\���A�*;


total_lossE��@

error_R�8?

learning_rate_1�O�6��I       6%�	���\���A�*;


total_loss$I�@

error_R�#F?

learning_rate_1�O�67�_I       6%�	��\���A�*;


total_loss���@

error_RүE?

learning_rate_1�O�6\y
I       6%�	�P�\���A�*;


total_loss	U�@

error_R�M?

learning_rate_1�O�6ȐVVI       6%�	ٕ�\���A�*;


total_loss؊�@

error_R��K?

learning_rate_1�O�6\��I       6%�	���\���A�*;


total_losso��@

error_R�C?

learning_rate_1�O�6V˨�I       6%�	[�\���A�*;


total_lossC+�@

error_R�3=?

learning_rate_1�O�6�s�[I       6%�	�`�\���A�*;


total_loss�[�@

error_Rs^?

learning_rate_1�O�66~�I       6%�	���\���A�*;


total_lossjb�@

error_R�wX?

learning_rate_1�O�6�cX"I       6%�	���\���A�*;


total_loss\iA

error_R�W?

learning_rate_1�O�6�7ŜI       6%�	�(�\���A�*;


total_loss���@

error_R�AJ?

learning_rate_1�O�6n�II       6%�	�k�\���A�*;


total_loss#�@

error_R)@?

learning_rate_1�O�6px�I       6%�	|��\���A�*;


total_loss���@

error_RnW??

learning_rate_1�O�6g���I       6%�	%��\���A�*;


total_loss�˸@

error_Rf!V?

learning_rate_1�O�6XzqI       6%�	i9�\���A�*;


total_loss(6�@

error_R�P?

learning_rate_1�O�6�yy�I       6%�	���\���A�*;


total_loss�ڄ@

error_R�L?

learning_rate_1�O�6d�I       6%�	���\���A�*;


total_loss,f�@

error_R�F?

learning_rate_1�O�6!�b	I       6%�	��\���A�*;


total_losszRA

error_R;$N?

learning_rate_1�O�6,�6I       6%�	%S�\���A�*;


total_loss���@

error_R��V?

learning_rate_1�O�6�gH�I       6%�	ĝ�\���A�*;


total_loss���@

error_R��H?

learning_rate_1�O�6s
-I       6%�	��\���A�*;


total_loss�@

error_R�T?

learning_rate_1�O�61��I       6%�	$�\���A�*;


total_lossEY�@

error_R� I?

learning_rate_1�O�6~��[I       6%�	/g�\���A�*;


total_loss���@

error_RW?

learning_rate_1�O�6m�MI       6%�	��\���A�*;


total_loss���@

error_R�W?

learning_rate_1�O�6SS�&I       6%�	���\���A�*;


total_loss�o�@

error_Rf�Q?

learning_rate_1�O�6��#I       6%�	#6�\���A�*;


total_loss�J�@

error_R��O?

learning_rate_1�O�6�l8+I       6%�	Yy�\���A�*;


total_loss\ʋ@

error_R�O?

learning_rate_1�O�6p��I       6%�	��\���A�*;


total_loss0A

error_R�T?

learning_rate_1�O�69��I       6%�	
��\���A�*;


total_loss:�@

error_R@�J?

learning_rate_1�O�6��*I       6%�	�@�\���A�*;


total_loss�(�@

error_R��F?

learning_rate_1�O�6f�oI       6%�	���\���A�*;


total_lossvn�@

error_R G?

learning_rate_1�O�6a���I       6%�	���\���A�*;


total_lossfj�@

error_RNu6?

learning_rate_1�O�6�:I       6%�	�'�\���A�*;


total_loss�A�@

error_R��T?

learning_rate_1�O�6E�:I       6%�	Rm�\���A�*;


total_loss!��@

error_R.�[?

learning_rate_1�O�6�K) I       6%�	n��\���A�*;


total_losse��@

error_R�s9?

learning_rate_1�O�6�=�nI       6%�	���\���A�*;


total_loss�ۨ@

error_R
�H?

learning_rate_1�O�6Pc6jI       6%�	F=�\���A�*;


total_loss��@

error_RiT?

learning_rate_1�O�6?��@I       6%�	߅�\���A�*;


total_loss�.�@

error_RrEX?

learning_rate_1�O�6�l�I       6%�	���\���A�*;


total_loss��@

error_R$�4?

learning_rate_1�O�6�..I       6%�	��\���A�*;


total_loss�L�@

error_Reo^?

learning_rate_1�O�6.�sI       6%�	�]�\���A�*;


total_loss e�@

error_Rl�8?

learning_rate_1�O�63�z/I       6%�	���\���A�*;


total_loss�`�@

error_R��F?

learning_rate_1�O�6hy,sI       6%�	r��\���A�*;


total_loss'�@

error_R��\?

learning_rate_1�O�6${�I       6%�	aD�\���A�*;


total_lossS<�@

error_R�S?

learning_rate_1�O�6��/�I       6%�	��\���A�*;


total_loss�ȑ@

error_R&@T?

learning_rate_1�O�6!R�I       6%�	���\���A�*;


total_loss�~�@

error_RL�L?

learning_rate_1�O�6�35I       6%�	m(�\���A�*;


total_loss#��@

error_R�PL?

learning_rate_1�O�6��I       6%�	�o�\���A�*;


total_loss��@

error_R�P`?

learning_rate_1�O�6�u�I       6%�	ݲ�\���A�*;


total_loss�[�@

error_R�~U?

learning_rate_1�O�6��:I       6%�	B��\���A�*;


total_losss��@

error_R�}L?

learning_rate_1�O�6��R�I       6%�	;>�\���A�*;


total_loss��@

error_Rq�A?

learning_rate_1�O�6�1w�I       6%�	��\���A�*;


total_lossc1�@

error_R�OL?

learning_rate_1�O�6*ov�I       6%�	���\���A�*;


total_loss� �@

error_Rx�G?

learning_rate_1�O�6uA�AI       6%�	��\���A�*;


total_loss�Ҡ@

error_R�K?

learning_rate_1�O�6�_�tI       6%�	�V�\���A�*;


total_loss��@

error_RMA`?

learning_rate_1�O�6R���I       6%�	���\���A�*;


total_loss�y@

error_R/9?

learning_rate_1�O�6;=�^I       6%�	���\���A�*;


total_loss���@

error_R�	:?

learning_rate_1�O�6�.ѷI       6%�	"2�\���A�*;


total_loss�ȱ@

error_R�BT?

learning_rate_1�O�6���I       6%�	sx�\���A�*;


total_loss�j�@

error_Rt�O?

learning_rate_1�O�6G��I       6%�	!��\���A�*;


total_loss�2�@

error_R�ME?

learning_rate_1�O�6����I       6%�	R	�\���A�*;


total_lossFG�@

error_RD?

learning_rate_1�O�6p^6�I       6%�	�O�\���A�*;


total_lossz�@

error_R=�E?

learning_rate_1�O�6����I       6%�	���\���A�*;


total_loss�'�@

error_Rf9V?

learning_rate_1�O�6�5�eI       6%�	���\���A�*;


total_loss�y�@

error_R��U?

learning_rate_1�O�6�5dXI       6%�	��\���A�*;


total_loss�@

error_R}�\?

learning_rate_1�O�6~��I       6%�	9Z�\���A�*;


total_loss<�@

error_Ri"P?

learning_rate_1�O�6�%�5I       6%�	+��\���A�*;


total_loss$ʼ@

error_RR�:?

learning_rate_1�O�6����I       6%�	��\���A�*;


total_loss�� A

error_R&N=?

learning_rate_1�O�6�<]�I       6%�	!�\���A�*;


total_loss��@

error_R�<[?

learning_rate_1�O�6�h}8I       6%�	eh�\���A�*;


total_loss�u�@

error_Rf<V?

learning_rate_1�O�6Xѥ�I       6%�	���\���A�*;


total_loss�`�@

error_R�K?

learning_rate_1�O�6u�JqI       6%�	���\���A�*;


total_losse��@

error_R�R=?

learning_rate_1�O�6З��I       6%�	C6�\���A�*;


total_loss��@

error_R��H?

learning_rate_1�O�6ߞI       6%�	�z�\���A�*;


total_lossz�@

error_R��??

learning_rate_1�O�6��PI       6%�	���\���A�*;


total_loss�ޖ@

error_R�7W?

learning_rate_1�O�6�V �I       6%�	��\���A�*;


total_loss��@

error_R(�q?

learning_rate_1�O�6<�4`I       6%�	�N�\���A�*;


total_loss���@

error_R�O?

learning_rate_1�O�6H��I       6%�	ܒ�\���A�*;


total_loss��@

error_R��U?

learning_rate_1�O�6���I       6%�	^��\���A�*;


total_loss�ߧ@

error_R��Z?

learning_rate_1�O�6�Z^I       6%�	��\���A�*;


total_lossF�@

error_RL�??

learning_rate_1�O�6��_I       6%�	�\�\���A�*;


total_loss��@

error_R7zF?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_lossv1�@

error_R��G?

learning_rate_1�O�6��.�I       6%�	���\���A�*;


total_loss���@

error_RV�Z?

learning_rate_1�O�6��7�I       6%�	u%�\���A�*;


total_loss6�@

error_Rr�N?

learning_rate_1�O�6�%i�I       6%�	i�\���A�*;


total_lossG�@

error_R?N?

learning_rate_1�O�6T���I       6%�	8��\���A�*;


total_loss�A

error_R9Q?

learning_rate_1�O�6醡�I       6%�	6��\���A�*;


total_lossw�@

error_R��I?

learning_rate_1�O�6!�T<I       6%�	9�\���A�*;


total_lossR��@

error_R��W?

learning_rate_1�O�6�m�I       6%�	�z�\���A�*;


total_loss��@

error_R3�Q?

learning_rate_1�O�6	P�I       6%�	O��\���A�*;


total_loss���@

error_R��Y?

learning_rate_1�O�6�[s=I       6%�	'+�\���A�*;


total_lossڻ�@

error_R��D?

learning_rate_1�O�6�LDI       6%�	}�\���A�*;


total_lossFp�@

error_R1�H?

learning_rate_1�O�6Jf6&I       6%�	W��\���A�*;


total_loss���@

error_R,�V?

learning_rate_1�O�6�K�I       6%�	��\���A�*;


total_loss��@

error_RϼG?

learning_rate_1�O�64��I       6%�	W�\���A�*;


total_loss��@

error_R��[?

learning_rate_1�O�6�7�I       6%�	���\���A�*;


total_lossԒ@

error_R.NN?

learning_rate_1�O�6���I       6%�	%��\���A�*;


total_loss��@

error_Rm4[?

learning_rate_1�O�6���I       6%�	$�\���A�*;


total_loss��@

error_R�N?

learning_rate_1�O�6��1I       6%�	�h�\���A�*;


total_lossx#�@

error_Rv�>?

learning_rate_1�O�6	cE�I       6%�	���\���A�*;


total_loss�L�@

error_R״A?

learning_rate_1�O�6�K3�I       6%�	(��\���A�*;


total_lossԥ�@

error_RC"N?

learning_rate_1�O�6�c<I       6%�	z1�\���A�*;


total_loss "�@

error_R��E?

learning_rate_1�O�6fժ9I       6%�	�r�\���A�*;


total_loss�oU@

error_R�D?

learning_rate_1�O�6�J�{I       6%�	���\���A�*;


total_loss�@

error_R�fX?

learning_rate_1�O�6���I       6%�	F��\���A�*;


total_loss��@

error_RdS?

learning_rate_1�O�6X���I       6%�	�?�\���A�*;


total_loss*�@

error_R�\?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_loss�A

error_R Q?

learning_rate_1�O�6Ii��I       6%�	���\���A�*;


total_loss��@

error_R�nC?

learning_rate_1�O�65��I       6%�	G�\���A�*;


total_loss$��@

error_R�O?

learning_rate_1�O�6�k��I       6%�	8M�\���A�*;


total_loss�@

error_R��Z?

learning_rate_1�O�6��h�I       6%�	>��\���A�*;


total_loss¬@

error_R�'T?

learning_rate_1�O�6�b��I       6%�	���\���A�*;


total_loss�ѻ@

error_R�??

learning_rate_1�O�6�1�I       6%�	% �\���A�*;


total_loss�ē@

error_R:�X?

learning_rate_1�O�6����I       6%�	qf�\���A�*;


total_loss��v@

error_RJ�,?

learning_rate_1�O�6U��|I       6%�	���\���A�*;


total_loss��@

error_R�AK?

learning_rate_1�O�6�g�I       6%�	���\���A�*;


total_lossߋ�@

error_R=�H?

learning_rate_1�O�6��I       6%�	?�\���A�*;


total_loss� �@

error_R�	b?

learning_rate_1�O�6T&��I       6%�	��\���A�*;


total_loss��@

error_R{�H?

learning_rate_1�O�6�hI�I       6%�	���\���A�*;


total_lossdn�@

error_RW�M?

learning_rate_1�O�6��9�I       6%�	X�\���A�*;


total_loss�J�@

error_R�]E?

learning_rate_1�O�6�.�I       6%�	�a�\���A�*;


total_lossn�@

error_R��O?

learning_rate_1�O�6�GI       6%�	���\���A�*;


total_loss�g@

error_R�|L?

learning_rate_1�O�6�q&I       6%�	a��\���A�*;


total_loss#p@

error_R�aU?

learning_rate_1�O�6�玎I       6%�	p>�\���A�*;


total_loss`Ƃ@

error_R3C?

learning_rate_1�O�6� =BI       6%�	��\���A�*;


total_loss���@

error_Rx�M?

learning_rate_1�O�66�!�I       6%�	U��\���A�*;


total_loss��@

error_RE^K?

learning_rate_1�O�6L��I       6%�	`�\���A�*;


total_loss�Mm@

error_R:L?

learning_rate_1�O�6~lB�I       6%�	X�\���A�*;


total_loss�}�@

error_RuD?

learning_rate_1�O�6.�n~I       6%�	���\���A�*;


total_loss2��@

error_R�+T?

learning_rate_1�O�6/�'XI       6%�	���\���A�*;


total_loss8.�@

error_R�^N?

learning_rate_1�O�6�ȺI       6%�	�0�\���A�*;


total_lossDA

error_R-hW?

learning_rate_1�O�6E���I       6%�	v�\���A�*;


total_loss���@

error_Rn V?

learning_rate_1�O�6�!;<I       6%�	1��\���A�*;


total_loss�A

error_R�Y?

learning_rate_1�O�6�_��I       6%�	$ �\���A�*;


total_lossH�@

error_R(N?

learning_rate_1�O�6At��I       6%�	�A�\���A�*;


total_lossԔ�@

error_R@CS?

learning_rate_1�O�6ҏ��I       6%�	��\���A�*;


total_lossJ�@

error_R�.N?

learning_rate_1�O�6D(II       6%�	���\���A�*;


total_lossq��@

error_R�K?

learning_rate_1�O�6�#I       6%�	
�\���A�*;


total_loss���@

error_R�gY?

learning_rate_1�O�6�2NI       6%�	���\���A�*;


total_loss7w�@

error_R�hI?

learning_rate_1�O�6���I       6%�	���\���A�*;


total_lossJT�@

error_R��E?

learning_rate_1�O�6;9-I       6%�	�;�\���A�*;


total_loss7ƹ@

error_R��D?

learning_rate_1�O�6�d��I       6%�	~��\���A�*;


total_loss���@

error_R�]M?

learning_rate_1�O�6Z�I       6%�	z��\���A�*;


total_loss���@

error_R�e`?

learning_rate_1�O�6-�I       6%�	V�\���A�*;


total_loss���@

error_RS2G?

learning_rate_1�O�6r.�gI       6%�	�]�\���A�*;


total_loss���@

error_RjU?

learning_rate_1�O�6iС	I       6%�	���\���A�*;


total_loss?~�@

error_RX�I?

learning_rate_1�O�6gU!+I       6%�	�\���A�*;


total_lossS��@

error_R�(N?

learning_rate_1�O�6	��kI       6%�	�N�\���A�*;


total_loss_A�@

error_R&
]?

learning_rate_1�O�6�`jI       6%�	���\���A�*;


total_loss�BA

error_R�a?

learning_rate_1�O�6SjH>I       6%�	'��\���A�*;


total_loss(ͅ@

error_R� Z?

learning_rate_1�O�6��F�I       6%�	��\���A�*;


total_loss���@

error_RRV?

learning_rate_1�O�6��FMI       6%�	�c�\���A�*;


total_lossJD�@

error_R�![?

learning_rate_1�O�6S��`I       6%�	���\���A�*;


total_loss5�@

error_RHW?

learning_rate_1�O�6l@̙I       6%�	Y��\���A�*;


total_loss��@

error_R
�c?

learning_rate_1�O�6�O��I       6%�	14�\���A�*;


total_loss�!�@

error_R�%S?

learning_rate_1�O�6�)�I       6%�	�y�\���A�*;


total_loss�́@

error_R�_?

learning_rate_1�O�6n�#I       6%�	L��\���A�*;


total_loss���@

error_R�b@?

learning_rate_1�O�6�=2I       6%�	��\���A�*;


total_loss��@

error_RJ�_?

learning_rate_1�O�6oʯ�I       6%�	�w�\���A�*;


total_lossIt�@

error_R�P?

learning_rate_1�O�6�_avI       6%�	��\���A�*;


total_lossm'A

error_R�I?

learning_rate_1�O�6Y��I       6%�	
 ]���A�*;


total_loss���@

error_R �c?

learning_rate_1�O�6�<�rI       6%�	H ]���A�*;


total_loss�9�@

error_R�O?

learning_rate_1�O�6�*#gI       6%�	�� ]���A�*;


total_loss/M�@

error_R �X?

learning_rate_1�O�6�9��I       6%�	�]���A�*;


total_loss�V�@

error_RT�Y?

learning_rate_1�O�6�n�I       6%�	@]]���A�*;


total_loss�m�@

error_R�-Z?

learning_rate_1�O�6�*�LI       6%�	�]���A�*;


total_lossrԶ@

error_RE�V?

learning_rate_1�O�6�ͽ�I       6%�	��]���A�*;


total_loss�@

error_RʺD?

learning_rate_1�O�6� qI       6%�	2;]���A�*;


total_loss��@

error_R�@\?

learning_rate_1�O�6��I       6%�	��]���A�*;


total_lossak�@

error_RJ
P?

learning_rate_1�O�6pA,I       6%�	��]���A�*;


total_loss�9�@

error_Rj�T?

learning_rate_1�O�6�H4�I       6%�	"]���A�*;


total_loss��A

error_R�D?

learning_rate_1�O�6�h*I       6%�	�Q]���A�*;


total_loss�4A

error_R�X?

learning_rate_1�O�6�`��I       6%�	�]���A�*;


total_loss[�@

error_R�,V?

learning_rate_1�O�6p�e�I       6%�	N�]���A�*;


total_loss/^�@

error_R�NS?

learning_rate_1�O�6,�2�I       6%�	!]���A�*;


total_loss�7�@

error_R�J?

learning_rate_1�O�6��_�I       6%�	k]���A�*;


total_loss��@

error_R�Z?

learning_rate_1�O�6���I       6%�	C�]���A�*;


total_loss	��@

error_R#jN?

learning_rate_1�O�6�C�I       6%�	��]���A�*;


total_loss��@

error_R8�R?

learning_rate_1�O�6���II       6%�	�>]���A�*;


total_loss���@

error_R�W?

learning_rate_1�O�6���I       6%�	e�]���A�*;


total_loss�k�@

error_Rv:?

learning_rate_1�O�6x7ngI       6%�	<�]���A�*;


total_loss}O�@

error_RA�O?

learning_rate_1�O�6�"��I       6%�	�]���A�*;


total_loss���@

error_R��H?

learning_rate_1�O�6DN� I       6%�	]Z]���A�*;


total_loss��@

error_R��Y?

learning_rate_1�O�6���I       6%�	͜]���A�*;


total_lossF�@

error_RTS?

learning_rate_1�O�6woI       6%�	�]���A�*;


total_loss�(�@

error_R��??

learning_rate_1�O�6���mI       6%�	 +]���A�*;


total_loss��@

error_R�yL?

learning_rate_1�O�6�hI       6%�	�v]���A�*;


total_loss���@

error_R��/?

learning_rate_1�O�6C�z�I       6%�	˿]���A�*;


total_lossf��@

error_R�lQ?

learning_rate_1�O�6�n�I       6%�	�]���A�*;


total_loss���@

error_R��M?

learning_rate_1�O�6����I       6%�	tQ]���A�*;


total_loss��@

error_Rq$G?

learning_rate_1�O�6M��I       6%�	��]���A�*;


total_loss�߫@

error_R �H?

learning_rate_1�O�6
��TI       6%�	Q�]���A�*;


total_lossX�@

error_R=cU?

learning_rate_1�O�6���I       6%�	�	]���A�*;


total_loss),�@

error_RQ?

learning_rate_1�O�6��I       6%�	.g	]���A�*;


total_loss|@�@

error_R�	M?

learning_rate_1�O�6I�
I       6%�	[�	]���A�*;


total_loss��@

error_R�_?

learning_rate_1�O�6W~�I       6%�	�	]���A�*;


total_loss�S�@

error_R.kA?

learning_rate_1�O�6����I       6%�	�.
]���A�*;


total_loss�x�@

error_R��[?

learning_rate_1�O�6n��\I       6%�	�q
]���A�*;


total_loss��@

error_Ri�Y?

learning_rate_1�O�6��xI       6%�	õ
]���A�*;


total_loss2��@

error_R�J?

learning_rate_1�O�6b'ZI       6%�	��
]���A�*;


total_lossx)�@

error_RȭR?

learning_rate_1�O�6�	�%I       6%�	�@]���A�*;


total_loss`��@

error_R3-E?

learning_rate_1�O�6���I       6%�	/�]���A�*;


total_lossG[A

error_RwXT?

learning_rate_1�O�6��I       6%�	-�]���A�*;


total_loss2ؘ@

error_R�0K?

learning_rate_1�O�6T��I       6%�	�6]���A�*;


total_loss��@

error_RCF=?

learning_rate_1�O�6HPۀI       6%�	�{]���A�*;


total_lossI�@

error_R�I?

learning_rate_1�O�6����I       6%�	��]���A�*;


total_loss%��@

error_RM|K?

learning_rate_1�O�6GT4I       6%�	f]���A�*;


total_loss�8�@

error_R�7??

learning_rate_1�O�6�N�I       6%�	L]���A�*;


total_lossTY�@

error_R��G?

learning_rate_1�O�6Jl
II       6%�	<�]���A�*;


total_loss;
�@

error_R�Z?

learning_rate_1�O�6�W6I       6%�	7�]���A�*;


total_loss��l@

error_R�`O?

learning_rate_1�O�6��I       6%�	�]���A�*;


total_lossG8�@

error_R��M?

learning_rate_1�O�6��L�I       6%�	,b]���A�*;


total_loss�h�@

error_R�XT?

learning_rate_1�O�6pk��I       6%�	��]���A�*;


total_lossX+�@

error_R�P?

learning_rate_1�O�65��I       6%�	#�]���A�*;


total_loss�T�@

error_R��M?

learning_rate_1�O�6�[�I       6%�	�*]���A�*;


total_lossOv@

error_R� C?

learning_rate_1�O�6���0I       6%�	&l]���A�*;


total_loss{�@

error_R�:?

learning_rate_1�O�6tC2�I       6%�	_�]���A�*;


total_loss��@

error_R[0D?

learning_rate_1�O�61
�XI       6%�	E�]���A�*;


total_loss�x@

error_Rx�F?

learning_rate_1�O�6�څI       6%�	7]���A�*;


total_losso��@

error_R~D?

learning_rate_1�O�6��gOI       6%�	�{]���A�*;


total_loss�Ǥ@

error_RM?

learning_rate_1�O�6V���I       6%�	�]���A�*;


total_lossֱA

error_R3J?

learning_rate_1�O�6,���I       6%�	]���A�*;


total_lossL�@

error_R7'c?

learning_rate_1�O�6�tI       6%�	FK]���A�*;


total_loss��@

error_R8�L?

learning_rate_1�O�6R��mI       6%�	s�]���A�*;


total_loss�ի@

error_R��??

learning_rate_1�O�6_6�pI       6%�	��]���A�*;


total_loss<�@

error_RxWF?

learning_rate_1�O�6�TI       6%�	R]���A�*;


total_loss��@

error_RJ�W?

learning_rate_1�O�6�ѕI       6%�	]]���A�*;


total_loss� �@

error_R��C?

learning_rate_1�O�6b	/_I       6%�	�]���A�*;


total_lossa�@

error_RM�C?

learning_rate_1�O�6���I       6%�	��]���A�*;


total_loss�ɵ@

error_RtCK?

learning_rate_1�O�6�]�I       6%�	R3]���A�*;


total_loss!O�@

error_R��K?

learning_rate_1�O�6���WI       6%�	Q}]���A�*;


total_lossV$�@

error_R�]?

learning_rate_1�O�6ꢴ�I       6%�	��]���A�*;


total_loss�~�@

error_R�;[?

learning_rate_1�O�6���yI       6%�	j]���A�*;


total_loss$�A

error_RMT?

learning_rate_1�O�6R��I       6%�	�H]���A�*;


total_loss�[�@

error_R��;?

learning_rate_1�O�6�BI       6%�	3�]���A�*;


total_lossL��@

error_R
f?

learning_rate_1�O�6� �I       6%�	y�]���A�*;


total_loss
(�@

error_RJ;?

learning_rate_1�O�6��II       6%�	�]���A�*;


total_loss 6�@

error_R.�I?

learning_rate_1�O�6W$=vI       6%�	�X]���A�*;


total_loss��@

error_RMS?

learning_rate_1�O�6�Z�	I       6%�	L�]���A�*;


total_loss��@

error_R��A?

learning_rate_1�O�6�L��I       6%�	��]���A�*;


total_loss{�@

error_R��U?

learning_rate_1�O�6c���I       6%�	�&]���A�*;


total_loss���@

error_R�F?

learning_rate_1�O�6I       6%�	vr]���A�*;


total_loss�m�@

error_R�M?

learning_rate_1�O�6?I       6%�	�]���A�*;


total_lossC(�@

error_R!O\?

learning_rate_1�O�6i4wCI       6%�	�]���A�*;


total_lossW��@

error_R��I?

learning_rate_1�O�6
)(�I       6%�	�I]���A�*;


total_loss�s�@

error_R??

learning_rate_1�O�6�!�I       6%�	M�]���A�*;


total_loss���@

error_R��:?

learning_rate_1�O�6��I       6%�	��]���A�*;


total_loss+�	A

error_Ra�a?

learning_rate_1�O�6؊�rI       6%�	�]���A�*;


total_loss-��@

error_R�$H?

learning_rate_1�O�6, �lI       6%�	�d]���A�*;


total_lossͰ�@

error_RW~<?

learning_rate_1�O�6����I       6%�	�]���A�*;


total_loss2	�@

error_R{�G?

learning_rate_1�O�6��#�I       6%�	��]���A�*;


total_loss��@

error_Rw�Z?

learning_rate_1�O�6KwW7I       6%�	�6]���A�*;


total_loss��@

error_R�LU?

learning_rate_1�O�6�FhI       6%�	'{]���A�*;


total_loss/��@

error_R�B?

learning_rate_1�O�6��I       6%�	�]���A�*;


total_loss�P�@

error_R�sL?

learning_rate_1�O�6xi?eI       6%�	I]���A�*;


total_loss�-�@

error_RLFM?

learning_rate_1�O�6�)� I       6%�	�E]���A�*;


total_loss]��@

error_RpX?

learning_rate_1�O�6<.\I       6%�	Ȉ]���A�*;


total_loss�ǚ@

error_R��G?

learning_rate_1�O�6(��I       6%�	�]���A�*;


total_loss`u�@

error_R��H?

learning_rate_1�O�6α��I       6%�	�]���A�*;


total_lossO,�@

error_Rd�D?

learning_rate_1�O�63)�	I       6%�	�W]���A�*;


total_loss���@

error_R�wI?

learning_rate_1�O�6k�@�I       6%�	d�]���A�*;


total_loss#HA

error_R�[?

learning_rate_1�O�6R�(�I       6%�	�]���A�*;


total_lossv �@

error_RŤF?

learning_rate_1�O�6ޅI�I       6%�	�Y]���A�*;


total_loss��@

error_R�N?

learning_rate_1�O�6)�SI       6%�	\�]���A�*;


total_loss\X�@

error_RHS?

learning_rate_1�O�6d�3@I       6%�	f�]���A�*;


total_loss�Q�@

error_R�sR?

learning_rate_1�O�6Ӫd�I       6%�	�']���A�*;


total_lossh;�@

error_R,2P?

learning_rate_1�O�6��@I       6%�	�k]���A�*;


total_loss�&�@

error_R�9E?

learning_rate_1�O�6'�ÿI       6%�	G�]���A�*;


total_loss�-�@

error_R��Q?

learning_rate_1�O�6g:8I       6%�	)�]���A�*;


total_lossOP�@

error_R��W?

learning_rate_1�O�6u�1pI       6%�	�:]���A�*;


total_loss ��@

error_R��C?

learning_rate_1�O�6%]��I       6%�	��]���A�*;


total_loss -�@

error_RN�H?

learning_rate_1�O�6݁�I       6%�	��]���A�*;


total_loss<YA

error_R�DL?

learning_rate_1�O�6���I       6%�	k]���A�*;


total_loss�i�@

error_R��O?

learning_rate_1�O�6�g�I       6%�	�e]���A�*;


total_loss�\�@

error_R�tN?

learning_rate_1�O�6d+�GI       6%�	��]���A�*;


total_loss�w�@

error_R�U?

learning_rate_1�O�6^b��I       6%�	��]���A�*;


total_loss�H�@

error_R�JG?

learning_rate_1�O�6G�I       6%�	�U ]���A�*;


total_lossth�@

error_R�L?

learning_rate_1�O�6����I       6%�	4� ]���A�*;


total_loss���@

error_R��<?

learning_rate_1�O�6�25�I       6%�	0� ]���A�*;


total_loss�@

error_R�Y?

learning_rate_1�O�6n�&�I       6%�	�1!]���A�*;


total_loss��@

error_R�_G?

learning_rate_1�O�6NmQ�I       6%�	�s!]���A�*;


total_loss|v�@

error_R�*N?

learning_rate_1�O�6��+&I       6%�	r�!]���A�*;


total_loss��@

error_R�R?

learning_rate_1�O�6�w@|I       6%�	-�!]���A�*;


total_loss���@

error_R�4N?

learning_rate_1�O�6sp�$I       6%�	�>"]���A�*;


total_loss�$�@

error_RJKA?

learning_rate_1�O�6��:I       6%�	��"]���A�*;


total_loss*�@

error_RR
>?

learning_rate_1�O�6{T��I       6%�	�"]���A�*;


total_loss�	�@

error_R;^O?

learning_rate_1�O�6����I       6%�	[	#]���A�*;


total_loss�@�@

error_R�]F?

learning_rate_1�O�6}{��I       6%�	
L#]���A�*;


total_loss��@

error_R��S?

learning_rate_1�O�6"��DI       6%�	��#]���A�*;


total_loss�'�@

error_R��I?

learning_rate_1�O�6:,!I       6%�	��#]���A�*;


total_loss��@

error_R�+M?

learning_rate_1�O�6����I       6%�	�$]���A�*;


total_loss,=�@

error_R�[?

learning_rate_1�O�6�E�I       6%�	b$]���A�*;


total_loss��@

error_R��Y?

learning_rate_1�O�68>��I       6%�	�$]���A�*;


total_loss��@

error_R��G?

learning_rate_1�O�6�اI       6%�	��$]���A�*;


total_loss���@

error_R[U?

learning_rate_1�O�6�bEI       6%�	&1%]���A�*;


total_loss��@

error_R�l6?

learning_rate_1�O�6@%w~I       6%�	Xv%]���A�*;


total_loss��@

error_R�R?

learning_rate_1�O�6�*+�I       6%�	�%]���A�*;


total_loss<h�@

error_R��>?

learning_rate_1�O�6�bt�I       6%�	� &]���A�*;


total_loss��z@

error_Rz�S?

learning_rate_1�O�6~.�I       6%�	�H&]���A�*;


total_loss���@

error_R��^?

learning_rate_1�O�6�ۦBI       6%�	�&]���A�*;


total_loss��@

error_R�G?

learning_rate_1�O�6���JI       6%�	�&]���A�*;


total_loss
��@

error_R<�@?

learning_rate_1�O�6�-��I       6%�	�']���A�*;


total_loss�V�@

error_RO`W?

learning_rate_1�O�6��:I       6%�	�^']���A�*;


total_loss��@

error_RG?

learning_rate_1�O�6* ~&I       6%�	¤']���A�*;


total_loss���@

error_R7�L?

learning_rate_1�O�6 �PI       6%�	��']���A�*;


total_loss�J�@

error_R��V?

learning_rate_1�O�6y��I       6%�	;.(]���A�*;


total_loss�g�@

error_R�BU?

learning_rate_1�O�6�U~I       6%�	Tr(]���A�*;


total_loss;֩@

error_R�\g?

learning_rate_1�O�6�ޡuI       6%�	ɷ(]���A�*;


total_loss��@

error_RE?

learning_rate_1�O�6���tI       6%�	��(]���A�*;


total_loss�W�@

error_R�>?

learning_rate_1�O�6`|�HI       6%�	@=)]���A�*;


total_loss��@

error_R/�G?

learning_rate_1�O�6,a�I       6%�	I�)]���A�*;


total_loss���@

error_R�C?

learning_rate_1�O�6�Lu�I       6%�	��)]���A�*;


total_lossN+�@

error_RfvV?

learning_rate_1�O�6Im�aI       6%�	d*]���A�*;


total_loss�Ė@

error_R��W?

learning_rate_1�O�6��6I       6%�	�_*]���A�*;


total_loss#m�@

error_R��D?

learning_rate_1�O�6Fg�dI       6%�	ۧ*]���A�*;


total_loss� �@

error_R%�V?

learning_rate_1�O�6vS�I       6%�	0�*]���A�*;


total_loss�Ƣ@

error_R��A?

learning_rate_1�O�6��F�I       6%�	_5+]���A�*;


total_lossU��@

error_RR�8?

learning_rate_1�O�6~Ȅ�I       6%�	�y+]���A�*;


total_loss���@

error_RHU?

learning_rate_1�O�6��?�I       6%�	0�+]���A�*;


total_lossoʾ@

error_R{WD?

learning_rate_1�O�6�ūTI       6%�	c',]���A�*;


total_lossS��@

error_R�S?

learning_rate_1�O�6��]I       6%�	�n,]���A�*;


total_loss���@

error_R�F?

learning_rate_1�O�6�_�I       6%�	��,]���A�*;


total_loss#H�@

error_R�I?

learning_rate_1�O�6�bI       6%�	x�,]���A�*;


total_lossK[ A

error_R�WW?

learning_rate_1�O�6�qwI       6%�	�8-]���A�*;


total_loss���@

error_RI�I?

learning_rate_1�O�6��%�I       6%�	�-]���A�*;


total_loss��@

error_R7EL?

learning_rate_1�O�6"s�I       6%�	��-]���A�*;


total_lossծ@

error_R�II?

learning_rate_1�O�6�u%�I       6%�	!.]���A�*;


total_loss��@

error_R�1Z?

learning_rate_1�O�6��dI       6%�	"P.]���A�*;


total_loss�w�@

error_R*XZ?

learning_rate_1�O�6�]o@I       6%�	��.]���A�*;


total_loss/Z�@

error_R��W?

learning_rate_1�O�6Y��I       6%�	��.]���A�*;


total_loss�[�@

error_RI�N?

learning_rate_1�O�6 #��I       6%�	�(/]���A�*;


total_loss�s�@

error_R�^K?

learning_rate_1�O�6�B�UI       6%�	�p/]���A�*;


total_lossW�@

error_R��S?

learning_rate_1�O�6�mr�I       6%�	�/]���A�*;


total_loss���@

error_R�PN?

learning_rate_1�O�6zx#I       6%�	70]���A�*;


total_loss���@

error_R��K?

learning_rate_1�O�6ќeI       6%�	KG0]���A�*;


total_lossv��@

error_RZZV?

learning_rate_1�O�6)US}I       6%�	�0]���A�*;


total_lossS�6@

error_RӈG?

learning_rate_1�O�6��7I       6%�	��0]���A�*;


total_loss��@

error_R8�H?

learning_rate_1�O�6�4��I       6%�	�1]���A�*;


total_lossI�@

error_R!#_?

learning_rate_1�O�6����I       6%�	t]1]���A�*;


total_loss]~�@

error_Rxd4?

learning_rate_1�O�69�kI       6%�	/�1]���A�*;


total_loss�A

error_R[QS?

learning_rate_1�O�6T�2�I       6%�	��1]���A�*;


total_lossM��@

error_RiDI?

learning_rate_1�O�6p��I       6%�	�#2]���A�*;


total_loss4�@

error_Rm�G?

learning_rate_1�O�6!B�TI       6%�	�e2]���A�*;


total_loss�@

error_R�)G?

learning_rate_1�O�6(��I       6%�	?�2]���A�*;


total_loss�t�@

error_R�R?

learning_rate_1�O�6�QO�I       6%�	�2]���A�*;


total_loss�V�@

error_R�nN?

learning_rate_1�O�6e-��I       6%�	�A3]���A�*;


total_loss�,�@

error_R:�K?

learning_rate_1�O�6H�w�I       6%�	��3]���A�*;


total_loss.#�@

error_Rq�_?

learning_rate_1�O�6"�hI       6%�	��3]���A�*;


total_loss3��@

error_R
^i?

learning_rate_1�O�6Zc#I       6%�	�4]���A�*;


total_loss���@

error_R�wL?

learning_rate_1�O�6[ӅI       6%�	�S4]���A�*;


total_loss5�@

error_R�;V?

learning_rate_1�O�6Q�SI       6%�	a�4]���A�*;


total_loss���@

error_R��X?

learning_rate_1�O�6��o�I       6%�	k�4]���A�*;


total_loss�8A

error_RډK?

learning_rate_1�O�6���I       6%�	L5]���A�*;


total_loss�@

error_R��U?

learning_rate_1�O�6�(tI       6%�	�]5]���A�*;


total_loss=<�@

error_R ^?

learning_rate_1�O�6]�I       6%�	��5]���A�*;


total_loss�v�@

error_R�TM?

learning_rate_1�O�6��E�I       6%�	C�5]���A�*;


total_loss_�@

error_RMH?

learning_rate_1�O�6/�ȠI       6%�	(�8]���A�*;


total_loss1U�@

error_R)6B?

learning_rate_1�O�6��FI       6%�	�9]���A�*;


total_loss���@

error_RZ4M?

learning_rate_1�O�6����I       6%�	@i9]���A�*;


total_loss2�M@

error_R��N?

learning_rate_1�O�6����I       6%�	ö9]���A�*;


total_loss���@

error_R��P?

learning_rate_1�O�6s�ȰI       6%�	:]���A�*;


total_loss�H�@

error_R�gN?

learning_rate_1�O�6�j�I       6%�	�D:]���A�*;


total_loss���@

error_Rm�Y?

learning_rate_1�O�6���I       6%�	.�:]���A�*;


total_lossv^�@

error_R[�Y?

learning_rate_1�O�6��xZI       6%�	
�:]���A�*;


total_loss2A�@

error_R$�M?

learning_rate_1�O�6s��I       6%�	+;]���A�*;


total_loss���@

error_R��c?

learning_rate_1�O�6�j�}I       6%�	>h;]���A�*;


total_loss��@

error_R��L?

learning_rate_1�O�6�0WI       6%�	2�;]���A�*;


total_lossq�@

error_R/,c?

learning_rate_1�O�6�``oI       6%�	m<]���A�*;


total_lossd�@

error_R�M?

learning_rate_1�O�6D��I       6%�	�b<]���A�*;


total_loss!6�@

error_RnK?

learning_rate_1�O�6����I       6%�	��<]���A�*;


total_loss8�@

error_R��[?

learning_rate_1�O�6�=�[I       6%�	0�<]���A�*;


total_loss�@

error_Ro�N?

learning_rate_1�O�6���I       6%�	xB=]���A�*;


total_lossq��@

error_R�*6?

learning_rate_1�O�6�;t9I       6%�	��=]���A�*;


total_loss�h@

error_R$uK?

learning_rate_1�O�6tGI       6%�	>�=]���A�*;


total_loss� A

error_RCd??

learning_rate_1�O�6}�I       6%�	�>]���A�*;


total_loss7��@

error_RH�K?

learning_rate_1�O�62�	�I       6%�	�]>]���A�*;


total_loss\�@

error_RM�Q?

learning_rate_1�O�6�@�I       6%�	T�>]���A�*;


total_loss�տ@

error_R�F?

learning_rate_1�O�6�bkCI       6%�	��>]���A�*;


total_loss��@

error_R�HJ?

learning_rate_1�O�6�|ߐI       6%�	�=?]���A�*;


total_loss�[�@

error_R�G;?

learning_rate_1�O�6����I       6%�	O�?]���A�*;


total_loss��@

error_R�IL?

learning_rate_1�O�6�3�pI       6%�	��?]���A�*;


total_loss�Ǌ@

error_R)�O?

learning_rate_1�O�6���I       6%�		@]���A�*;


total_loss��@

error_Rܘ:?

learning_rate_1�O�6���I       6%�	�{@]���A�*;


total_lossEA

error_Rs�1?

learning_rate_1�O�6LWI       6%�	��@]���A�*;


total_loss`~�@

error_R�VM?

learning_rate_1�O�6F8%�I       6%�	BA]���A�*;


total_loss��@

error_R��A?

learning_rate_1�O�6{�I       6%�	rKA]���A�*;


total_loss�i@

error_R3�O?

learning_rate_1�O�6_���I       6%�	9�A]���A�*;


total_lossx޲@

error_R*Jf?

learning_rate_1�O�6��S�I       6%�	��A]���A�*;


total_loss9�@

error_Rha?

learning_rate_1�O�6��sI       6%�	�"B]���A�*;


total_loss��@

error_R[�X?

learning_rate_1�O�6I!?�I       6%�	�nB]���A�*;


total_loss�Q�@

error_R��E?

learning_rate_1�O�6)b@NI       6%�	�B]���A�*;


total_loss���@

error_RaaQ?

learning_rate_1�O�6���BI       6%�	wC]���A�*;


total_loss@��@

error_R��@?

learning_rate_1�O�6���BI       6%�	xYC]���A�*;


total_loss��A

error_R�oC?

learning_rate_1�O�6�(�I       6%�	_�C]���A�*;


total_loss�CW@

error_R{KH?

learning_rate_1�O�6��j�I       6%�	�D]���A�*;


total_loss
n�@

error_R�T?

learning_rate_1�O�6lL|I       6%�	ldD]���A�*;


total_loss���@

error_R�vS?

learning_rate_1�O�6�,I       6%�	�D]���A�*;


total_lossZV�@

error_R�U?

learning_rate_1�O�6��C�I       6%�	��D]���A�*;


total_lossD��@

error_RR?

learning_rate_1�O�6�ԴI       6%�	�>E]���A�*;


total_loss�T�@

error_R=�G?

learning_rate_1�O�6vFTI       6%�	5�E]���A�*;


total_lossZ��@

error_R��R?

learning_rate_1�O�6���	I       6%�	�E]���A�*;


total_loss�9�@

error_R�N?

learning_rate_1�O�6o�=I       6%�	�AF]���A�*;


total_loss1�@

error_R�s:?

learning_rate_1�O�6����I       6%�	�F]���A�*;


total_loss���@

error_R!lO?

learning_rate_1�O�6$οI       6%�	��F]���A�*;


total_loss���@

error_R	�V?

learning_rate_1�O�6�~�aI       6%�	�$G]���A�*;


total_loss�Ǟ@

error_R��B?

learning_rate_1�O�6PÃ1I       6%�	SnG]���A�*;


total_lossyA

error_R +M?

learning_rate_1�O�6��2WI       6%�	*�G]���A�*;


total_loss]��@

error_R�J?

learning_rate_1�O�6J���I       6%�	��G]���A�*;


total_loss�=�@

error_R�gW?

learning_rate_1�O�6� � I       6%�	�@H]���A�*;


total_lossd��@

error_R�8<?

learning_rate_1�O�6��I       6%�	��H]���A�*;


total_loss�ӭ@

error_Rl�H?

learning_rate_1�O�6a�D�I       6%�	��H]���A�*;


total_loss���@

error_R�N?

learning_rate_1�O�6��k�I       6%�	bGI]���A�*;


total_loss��@

error_RJ�D?

learning_rate_1�O�6v4<�I       6%�	o�I]���A�*;


total_loss��A

error_R�TT?

learning_rate_1�O�6�"e�I       6%�	H�I]���A�*;


total_losso��@

error_R[(O?

learning_rate_1�O�6��V�I       6%�	8J]���A�*;


total_loss��@

error_R\iR?

learning_rate_1�O�6���I       6%�	[J]���A�*;


total_loss���@

error_RhK?

learning_rate_1�O�6&\�I       6%�	̣J]���A�*;


total_loss�2�@

error_Rf�O?

learning_rate_1�O�6��7I       6%�	A�J]���A�*;


total_loss��A

error_R1X?

learning_rate_1�O�6� -MI       6%�	0K]���A�*;


total_loss]�@

error_R�oJ?

learning_rate_1�O�6���I       6%�	�wK]���A�*;


total_loss	�@

error_R��J?

learning_rate_1�O�6H5צI       6%�	��K]���A�*;


total_loss7��@

error_R��N?

learning_rate_1�O�6��+�I       6%�	�)L]���A�*;


total_loss|P�@

error_R2�J?

learning_rate_1�O�6�e#I       6%�	qL]���A�*;


total_loss�8�@

error_R�`C?

learning_rate_1�O�6wjMyI       6%�	��L]���A�*;


total_loss�6�@

error_R��V?

learning_rate_1�O�6]w$�I       6%�	5M]���A�*;


total_lossO��@

error_R��O?

learning_rate_1�O�6n�skI       6%�	]LM]���A�*;


total_lossTh�@

error_R��U?

learning_rate_1�O�6�VuAI       6%�	E�M]���A�*;


total_loss�@�@

error_R�O?

learning_rate_1�O�6,x�0I       6%�	 �M]���A�*;


total_lossq��@

error_R��7?

learning_rate_1�O�6+�[I       6%�	�N]���A�*;


total_loss�Ϣ@

error_R҇b?

learning_rate_1�O�6���I       6%�	0\N]���A�*;


total_loss�8�@

error_RZub?

learning_rate_1�O�6��8�I       6%�	�N]���A�*;


total_losseG�@

error_RC?

learning_rate_1�O�6�$ bI       6%�	H�N]���A�*;


total_loss%�@

error_RI�b?

learning_rate_1�O�6U��~I       6%�	�0O]���A�*;


total_loss�آ@

error_R�*[?

learning_rate_1�O�6��UpI       6%�	�uO]���A�*;


total_loss-�d@

error_Rx@?

learning_rate_1�O�6U�8�I       6%�	ۼO]���A�*;


total_loss1��@

error_R8�G?

learning_rate_1�O�6g���I       6%�	g�O]���A�*;


total_loss}�T@

error_RaP?

learning_rate_1�O�60c�!I       6%�	�DP]���A�*;


total_lossmD�@

error_R�$>?

learning_rate_1�O�6,�I       6%�	��P]���A�*;


total_losse�@

error_R�=?

learning_rate_1�O�6߷ppI       6%�	q�P]���A�*;


total_loss4x�@

error_R}�B?

learning_rate_1�O�6T�%I       6%�	bQ]���A�*;


total_loss�y�@

error_RM�2?

learning_rate_1�O�6q�I       6%�	[Q]���A�*;


total_loss���@

error_Rr7?

learning_rate_1�O�6Îd I       6%�	f�Q]���A�*;


total_loss���@

error_R��9?

learning_rate_1�O�6�$��I       6%�	w�Q]���A�*;


total_loss���@

error_R��R?

learning_rate_1�O�6�?�"I       6%�	�R]���A�*;


total_loss ��@

error_R��X?

learning_rate_1�O�64��I       6%�	�eR]���A�*;


total_lossz�A

error_RD�E?

learning_rate_1�O�61FEI       6%�	̥R]���A�*;


total_losst�@

error_R��R?

learning_rate_1�O�6��J{I       6%�	��R]���A�*;


total_loss iw@

error_ROS?

learning_rate_1�O�6��II       6%�	�0S]���A�*;


total_loss8k�@

error_Rr�M?

learning_rate_1�O�6L~/�I       6%�	zS]���A�*;


total_loss[�@

error_RÞL?

learning_rate_1�O�6&I�XI       6%�	
�S]���A�*;


total_loss6��@

error_R�R?

learning_rate_1�O�6,�I       6%�	6	T]���A�*;


total_loss��@

error_R|>S?

learning_rate_1�O�6���I       6%�	+OT]���A�*;


total_loss-�@

error_Rd�L?

learning_rate_1�O�6l�(I       6%�	J�T]���A�*;


total_loss��@

error_Rx�^?

learning_rate_1�O�6�#'�I       6%�	��T]���A�*;


total_loss	[�@

error_R�[?

learning_rate_1�O�6[��pI       6%�	�U]���A�*;


total_lossJ�@

error_R�p>?

learning_rate_1�O�6--�7I       6%�	�]U]���A�*;


total_loss!k�@

error_R'O?

learning_rate_1�O�6Bҡ5I       6%�	��U]���A�*;


total_loss�:RA

error_R��K?

learning_rate_1�O�6M�ZI       6%�	��U]���A�*;


total_loss(:�@

error_R�Rd?

learning_rate_1�O�6�SJ�I       6%�	e'V]���A�*;


total_loss-O�@

error_R.W?

learning_rate_1�O�6wfa�I       6%�	~lV]���A�*;


total_loss�s�@

error_Ro�C?

learning_rate_1�O�6���I       6%�	߰V]���A�*;


total_lossq��@

error_R��O?

learning_rate_1�O�6U> I       6%�	Q�V]���A�*;


total_loss���@

error_R�[;?

learning_rate_1�O�6z�oWI       6%�	�7W]���A�*;


total_lossS��@

error_R��K?

learning_rate_1�O�6���I       6%�	�yW]���A�*;


total_lossn�@

error_R֘X?

learning_rate_1�O�6q�I       6%�	
�W]���A�*;


total_loss���@

error_R��P?

learning_rate_1�O�63#]�I       6%�	��W]���A�*;


total_loss��@

error_R4 @?

learning_rate_1�O�6�d��I       6%�	BX]���A�*;


total_loss�C�@

error_RHV?

learning_rate_1�O�6-h]�I       6%�	�X]���A�*;


total_lossQĚ@

error_R�L?

learning_rate_1�O�6�|��I       6%�	��X]���A�*;


total_lossW��@

error_REO_?

learning_rate_1�O�6ƶ[�I       6%�	*Y]���A�*;


total_loss��@

error_R�[G?

learning_rate_1�O�61TI       6%�	�SY]���A�*;


total_lossLIA

error_R��_?

learning_rate_1�O�6�I       6%�	��Y]���A�*;


total_loss/�A

error_R�LU?

learning_rate_1�O�6n�g�I       6%�	��Y]���A�*;


total_loss*�z@

error_Rs�J?

learning_rate_1�O�6���I       6%�	cZ]���A�*;


total_loss���@

error_R�$Q?

learning_rate_1�O�6a��I       6%�	n`Z]���A�*;


total_lossJ�@

error_R�L?

learning_rate_1�O�6҇��I       6%�	�Z]���A�*;


total_loss�{�@

error_R��L?

learning_rate_1�O�6��CI       6%�	��Z]���A�*;


total_loss�7�@

error_R�[?

learning_rate_1�O�6fW��I       6%�	S)[]���A�*;


total_loss�M�@

error_R��W?

learning_rate_1�O�6d]��I       6%�	�k[]���A�*;


total_lossZ��@

error_R/�=?

learning_rate_1�O�6��,�I       6%�	ڳ[]���A�*;


total_loss�%�@

error_RoVU?

learning_rate_1�O�6�yV�I       6%�	�\]���A�*;


total_loss���@

error_R�nc?

learning_rate_1�O�6|m�I       6%�	�[\]���A�*;


total_loss$�@

error_R]J?

learning_rate_1�O�6(suI       6%�	 �\]���A�*;


total_loss��@

error_R)=Y?

learning_rate_1�O�6��I       6%�	��\]���A�*;


total_lossRR�@

error_RqwL?

learning_rate_1�O�6��I       6%�	�)]]���A�*;


total_loss��`@

error_Ra�P?

learning_rate_1�O�6�x�I       6%�	�k]]���A�*;


total_loss��@

error_R\�L?

learning_rate_1�O�6�I       6%�	f�]]���A�*;


total_loss�9�@

error_RJ�U?

learning_rate_1�O�6���I       6%�	��]]���A�*;


total_loss�6�@

error_R,\<?

learning_rate_1�O�69���I       6%�	M=^]���A�*;


total_loss$`�@

error_R�J?

learning_rate_1�O�6� �I       6%�	U�^]���A�*;


total_loss�!�@

error_R��L?

learning_rate_1�O�6=D	I       6%�	��^]���A�*;


total_loss�2�@

error_R�"_?

learning_rate_1�O�6%w��I       6%�	_]���A�*;


total_loss{>�@

error_R��N?

learning_rate_1�O�6#�H,I       6%�	O�_]���A�*;


total_loss�m@

error_R(R?

learning_rate_1�O�64�ʟI       6%�	Y�_]���A�*;


total_loss-��@

error_R�B@?

learning_rate_1�O�6/6jI       6%�	/`]���A�*;


total_loss&��@

error_RH�I?

learning_rate_1�O�6e�k�I       6%�	
�`]���A�*;


total_loss���@

error_R`W?

learning_rate_1�O�6��?mI       6%�	L�`]���A�*;


total_loss���@

error_R��K?

learning_rate_1�O�6��ֈI       6%�	�a]���A�*;


total_lossl/�@

error_R�V?

learning_rate_1�O�65[�uI       6%�	�fa]���A�*;


total_loss��@

error_R�V?

learning_rate_1�O�6�#I       6%�	��a]���A�*;


total_loss���@

error_R��g?

learning_rate_1�O�6i@�I       6%�	��a]���A�*;


total_lossn��@

error_R�qC?

learning_rate_1�O�6��II       6%�	Db]���A�*;


total_loss$2�@

error_R�9J?

learning_rate_1�O�6Z��'I       6%�	T�b]���A�*;


total_lossAي@

error_R\�W?

learning_rate_1�O�6��JmI       6%�	��b]���A�*;


total_lossh��@

error_R��C?

learning_rate_1�O�6=�%�I       6%�	�c]���A�*;


total_loss���@

error_R��E?

learning_rate_1�O�6T�I       6%�	�Uc]���A�*;


total_loss��@

error_R�J?

learning_rate_1�O�6��fLI       6%�	U�c]���A�*;


total_loss���@

error_RE�K?

learning_rate_1�O�66�SI       6%�	*�c]���A�	*;


total_loss��c@

error_R��4?

learning_rate_1�O�6�LI       6%�	4d]���A�	*;


total_loss�Ŧ@

error_R@A?

learning_rate_1�O�6�q��I       6%�	�~d]���A�	*;


total_loss¥@

error_R$K?

learning_rate_1�O�6��I       6%�	�d]���A�	*;


total_loss���@

error_R�]L?

learning_rate_1�O�6��gI       6%�	�e]���A�	*;


total_loss�@

error_R�tN?

learning_rate_1�O�6J��I       6%�	�Re]���A�	*;


total_loss��@

error_R��U?

learning_rate_1�O�6�4�5I       6%�	C�e]���A�	*;


total_loss�l�@

error_RCA>?

learning_rate_1�O�6)P� I       6%�	�e]���A�	*;


total_loss�p�@

error_R=F?

learning_rate_1�O�6��}(I       6%�	�!f]���A�	*;


total_loss�s�@

error_R�F?

learning_rate_1�O�6ͰH�I       6%�	odf]���A�	*;


total_lossk�A

error_R��I?

learning_rate_1�O�66N�I       6%�	 �f]���A�	*;


total_loss�D�@

error_R$M?

learning_rate_1�O�6dmdI       6%�	��f]���A�	*;


total_lossV�@

error_Ro?@?

learning_rate_1�O�6�^@I       6%�	�9g]���A�	*;


total_loss�ƃ@

error_R8�X?

learning_rate_1�O�6�_\DI       6%�	K~g]���A�	*;


total_loss�"m@

error_RpD?

learning_rate_1�O�6�KI�I       6%�	��g]���A�	*;


total_lossJU|@

error_R� J?

learning_rate_1�O�6&6(�I       6%�	=h]���A�	*;


total_loss��@

error_R�G?

learning_rate_1�O�6k��I       6%�	Hh]���A�	*;


total_loss��@

error_R�V?

learning_rate_1�O�6�X��I       6%�	-�h]���A�	*;


total_loss�h�@

error_R�I?

learning_rate_1�O�6�)l�I       6%�	�h]���A�	*;


total_loss���@

error_Rf�L?

learning_rate_1�O�6r�5<I       6%�	�i]���A�	*;


total_lossN1q@

error_R=�F?

learning_rate_1�O�6G&�}I       6%�	ai]���A�	*;


total_losst��@

error_RX_I?

learning_rate_1�O�6(�	vI       6%�	A�i]���A�	*;


total_loss#��@

error_RWqJ?

learning_rate_1�O�6�(<DI       6%�	��i]���A�	*;


total_loss���@

error_R�O?

learning_rate_1�O�6�I       6%�	�8j]���A�	*;


total_loss���@

error_R)�??

learning_rate_1�O�6D��I       6%�	ԅj]���A�	*;


total_loss��@

error_R 2??

learning_rate_1�O�6�;yRI       6%�	r�j]���A�	*;


total_loss�@

error_R�VN?

learning_rate_1�O�6�A2#I       6%�	�k]���A�	*;


total_lossDG�@

error_R� ,?

learning_rate_1�O�6��~5I       6%�	!ak]���A�	*;


total_loss�K�@

error_R=�W?

learning_rate_1�O�6�{��I       6%�	�k]���A�	*;


total_loss�>�@

error_R�??

learning_rate_1�O�6�)�I       6%�	�l]���A�	*;


total_lossQ:�@

error_R]=?

learning_rate_1�O�6�|t�I       6%�	�]l]���A�	*;


total_lossi�@

error_R��N?

learning_rate_1�O�6��EI       6%�	��l]���A�	*;


total_loss��@

error_R�AL?

learning_rate_1�O�6�_��I       6%�	��l]���A�	*;


total_lossZ��@

error_R�I^?

learning_rate_1�O�6w�#I       6%�	k-m]���A�	*;


total_loss.��@

error_RJJ?

learning_rate_1�O�6��K$I       6%�	�pm]���A�	*;


total_loss�D"A

error_Rf�Q?

learning_rate_1�O�6`ݐ�I       6%�	�m]���A�	*;


total_loss��A

error_R �T?

learning_rate_1�O�6A;w�I       6%�	��m]���A�	*;


total_loss!{@

error_R�E?

learning_rate_1�O�6�rt)I       6%�	�>n]���A�	*;


total_losss��@

error_R�)O?

learning_rate_1�O�6��9tI       6%�	!�n]���A�	*;


total_loss�8�@

error_R�i?

learning_rate_1�O�6��I       6%�	��n]���A�	*;


total_loss�@

error_R�[V?

learning_rate_1�O�6���I       6%�	�o]���A�	*;


total_loss�|�@

error_RtV?

learning_rate_1�O�6iKGI       6%�		Qo]���A�	*;


total_lossq��@

error_R5S?

learning_rate_1�O�6˿E]I       6%�	��o]���A�	*;


total_loss���@

error_R
�h?

learning_rate_1�O�6
!n�I       6%�	�o]���A�	*;


total_lossi��@

error_R�(U?

learning_rate_1�O�6qGI       6%�	o*p]���A�	*;


total_loss�ʉ@

error_R[W?

learning_rate_1�O�6ӞNI       6%�	Kvp]���A�	*;


total_lossZR�@

error_R$dI?

learning_rate_1�O�6d�&SI       6%�	��p]���A�	*;


total_loss�@

error_R=lN?

learning_rate_1�O�6��НI       6%�	q]���A�	*;


total_loss$l�@

error_R��??

learning_rate_1�O�6�,��I       6%�	�Oq]���A�	*;


total_lossY*�@

error_R�cJ?

learning_rate_1�O�6�5��I       6%�	��q]���A�	*;


total_loss��@

error_R�P?

learning_rate_1�O�6"L9�I       6%�	��q]���A�	*;


total_loss4�
A

error_R�tS?

learning_rate_1�O�6dίrI       6%�	�&r]���A�	*;


total_loss���@

error_R��M?

learning_rate_1�O�6҇��I       6%�	mmr]���A�	*;


total_loss!�@

error_R��k?

learning_rate_1�O�6q�]�I       6%�	P�r]���A�	*;


total_loss6Nu@

error_R��H?

learning_rate_1�O�6�I       6%�	K�r]���A�	*;


total_lossߙ�@

error_R}=Q?

learning_rate_1�O�6���yI       6%�	�?s]���A�	*;


total_loss�6�@

error_RFoO?

learning_rate_1�O�6��	&I       6%�	q�s]���A�	*;


total_lossM��@

error_R�VL?

learning_rate_1�O�6yP��I       6%�	��s]���A�	*;


total_loss.��@

error_R1qR?

learning_rate_1�O�6�z�I       6%�	�t]���A�	*;


total_loss7��@

error_R��P?

learning_rate_1�O�65q.I       6%�	�St]���A�	*;


total_losss��@

error_R�MH?

learning_rate_1�O�6|_�zI       6%�	�t]���A�	*;


total_loss�8�@

error_R�uQ?

learning_rate_1�O�6��FI       6%�	L�t]���A�	*;


total_loss�j�@

error_R�K8?

learning_rate_1�O�6 �^I       6%�	Bu]���A�	*;


total_loss�h�@

error_R\�M?

learning_rate_1�O�6㴜�I       6%�	�_u]���A�	*;


total_loss�`�@

error_Rs�N?

learning_rate_1�O�63R9I       6%�	`�u]���A�	*;


total_loss_��@

error_R�*a?

learning_rate_1�O�6'ޠI       6%�	��u]���A�	*;


total_lossC��@

error_RT�=?

learning_rate_1�O�6���@I       6%�	/2v]���A�	*;


total_lossc%!A

error_R	Q?

learning_rate_1�O�6�J��I       6%�	7{v]���A�	*;


total_lossZ\�@

error_R��_?

learning_rate_1�O�6�"I       6%�	��v]���A�	*;


total_lossne@

error_RH�??

learning_rate_1�O�6���I       6%�	�	w]���A�	*;


total_loss���@

error_R�!_?

learning_rate_1�O�6�eTKI       6%�	;Nw]���A�	*;


total_loss�#A

error_Rn�I?

learning_rate_1�O�6UyDnI       6%�	�w]���A�	*;


total_loss�o�@

error_R�?L?

learning_rate_1�O�6�y�I       6%�	a�w]���A�	*;


total_lossኟ@

error_R�F?

learning_rate_1�O�6m�d�I       6%�	�x]���A�	*;


total_losse-�@

error_Rq_Y?

learning_rate_1�O�66d"I       6%�	�`x]���A�	*;


total_loss�մ@

error_R�F?

learning_rate_1�O�6�g�}I       6%�	4�x]���A�	*;


total_lossh˴@

error_R��F?

learning_rate_1�O�6����I       6%�	��x]���A�	*;


total_loss��@

error_R��D?

learning_rate_1�O�6��)I       6%�	#3y]���A�	*;


total_lossɆ�@

error_R=�_?

learning_rate_1�O�6����I       6%�	��y]���A�	*;


total_lossHE�@

error_R<P?

learning_rate_1�O�6�8_�I       6%�	3�y]���A�	*;


total_loss�s�@

error_R�N?

learning_rate_1�O�6��I       6%�	~z]���A�	*;


total_loss�@

error_RY@?

learning_rate_1�O�6�!�I       6%�	JXz]���A�	*;


total_loss"�@

error_R�HI?

learning_rate_1�O�6]��kI       6%�	w�z]���A�	*;


total_loss�̬@

error_R�S?

learning_rate_1�O�6�P_I       6%�	��z]���A�	*;


total_lossjކ@

error_R�nC?

learning_rate_1�O�6��-XI       6%�	�.{]���A�	*;


total_lossr4�@

error_R
�U?

learning_rate_1�O�6\r�I       6%�	}r{]���A�	*;


total_lossVG�@

error_RE�H?

learning_rate_1�O�6[�!cI       6%�	�{]���A�	*;


total_lossl�A

error_R�D?

learning_rate_1�O�6���I       6%�	�|]���A�	*;


total_loss�@

error_RL�1?

learning_rate_1�O�6o��I       6%�	�j|]���A�	*;


total_loss��@

error_RnI?

learning_rate_1�O�6���jI       6%�	.�|]���A�	*;


total_loss�Z�@

error_R�E?

learning_rate_1�O�6��I       6%�	O}]���A�	*;


total_loss�?�@

error_R8�T?

learning_rate_1�O�6�Q �I       6%�	�E}]���A�	*;


total_losspϏ@

error_R�=?

learning_rate_1�O�6�3�
I       6%�	D�}]���A�	*;


total_loss�"�@

error_R%�E?

learning_rate_1�O�6���I       6%�	+�}]���A�	*;


total_loss�A

error_R,zP?

learning_rate_1�O�6�I�I       6%�	�~]���A�	*;


total_loss��@

error_R I?

learning_rate_1�O�6b��I       6%�	�l~]���A�	*;


total_loss{��@

error_Rq3E?

learning_rate_1�O�6��SI       6%�	��~]���A�	*;


total_loss��@

error_R��F?

learning_rate_1�O�61W�I       6%�	�]���A�	*;


total_lossx��@

error_R�J?

learning_rate_1�O�6Ȅ}"I       6%�	]���A�	*;


total_loss���@

error_R��8?

learning_rate_1�O�6�N�I       6%�	��]���A�	*;


total_lossל�@

error_R�s;?

learning_rate_1�O�6@�P�I       6%�	>�]���A�	*;


total_lossj�@

error_R��=?

learning_rate_1�O�6���I       6%�	L��]���A�	*;


total_loss� �@

error_R��\?

learning_rate_1�O�6}�=I       6%�	�׀]���A�	*;


total_losst�@

error_R�N?

learning_rate_1�O�6*�AsI       6%�	��]���A�	*;


total_lossX}�@

error_R$h??

learning_rate_1�O�6 ��8I       6%�	�e�]���A�	*;


total_loss��@

error_R��N?

learning_rate_1�O�6Đ��I       6%�	E��]���A�	*;


total_loss:y�@

error_R��D?

learning_rate_1�O�6�2}�I       6%�	D�]���A�	*;


total_loss^��@

error_R�C?

learning_rate_1�O�6�"�I       6%�	f4�]���A�	*;


total_lossmf�@

error_R7HE?

learning_rate_1�O�6Փ�I       6%�	1y�]���A�	*;


total_loss��@

error_R��>?

learning_rate_1�O�6���fI       6%�	ܿ�]���A�	*;


total_loss�[�@

error_R��N?

learning_rate_1�O�6��I       6%�	��]���A�	*;


total_lossh��@

error_R�\D?

learning_rate_1�O�69���I       6%�	@K�]���A�	*;


total_loss[ہ@

error_R�+A?

learning_rate_1�O�6��l�I       6%�	���]���A�	*;


total_loss3�t@

error_RDWZ?

learning_rate_1�O�6��I       6%�	,݃]���A�	*;


total_lossLO�@

error_R,O?

learning_rate_1�O�6�E�I       6%�	�#�]���A�	*;


total_loss��@

error_R8�^?

learning_rate_1�O�6w޼�I       6%�	�m�]���A�	*;


total_loss@

error_R�cP?

learning_rate_1�O�6���I       6%�	0��]���A�	*;


total_loss���@

error_R-�E?

learning_rate_1�O�6MȲcI       6%�	��]���A�	*;


total_loss��@

error_R�??

learning_rate_1�O�60' \I       6%�	F�]���A�	*;


total_loss.�A

error_R�~Q?

learning_rate_1�O�6���bI       6%�	���]���A�	*;


total_loss�o�@

error_R��B?

learning_rate_1�O�6�Y�I       6%�	6Յ]���A�	*;


total_loss�@

error_R��I?

learning_rate_1�O�6�#I       6%�	I�]���A�	*;


total_loss���@

error_R�=?

learning_rate_1�O�6N�dI       6%�	c�]���A�	*;


total_loss	8�@

error_Rl�D?

learning_rate_1�O�6��I       6%�	#��]���A�	*;


total_loss�@ A

error_R��I?

learning_rate_1�O�6���I       6%�	6��]���A�	*;


total_lossj�@

error_RJ�H?

learning_rate_1�O�6G�_I       6%�	�9�]���A�	*;


total_loss�Z@

error_R��A?

learning_rate_1�O�6�I-I       6%�	&~�]���A�	*;


total_loss�@

error_R�zW?

learning_rate_1�O�6u8{I       6%�	�ˇ]���A�	*;


total_loss�~�@

error_R) T?

learning_rate_1�O�6���I       6%�	��]���A�
*;


total_loss�L�@

error_R�W?

learning_rate_1�O�6�T�NI       6%�	I\�]���A�
*;


total_lossq�@

error_R��S?

learning_rate_1�O�6�7 aI       6%�	ˣ�]���A�
*;


total_loss��@

error_R�eO?

learning_rate_1�O�6p�yI       6%�	(�]���A�
*;


total_lossa'�@

error_R&�Y?

learning_rate_1�O�63�:I       6%�	Q.�]���A�
*;


total_loss2ߦ@

error_RW�E?

learning_rate_1�O�6��Y I       6%�	�p�]���A�
*;


total_loss,�@

error_RR�K?

learning_rate_1�O�6/U�I       6%�	���]���A�
*;


total_loss&��@

error_R&5?

learning_rate_1�O�6�%��I       6%�	� �]���A�
*;


total_lossh�@

error_RTI?

learning_rate_1�O�6@��I       6%�	�H�]���A�
*;


total_loss(�@

error_Ri$F?

learning_rate_1�O�6RElI       6%�	9��]���A�
*;


total_loss���@

error_RN�G?

learning_rate_1�O�6,��I       6%�	/֊]���A�
*;


total_loss�"�@

error_R*�9?

learning_rate_1�O�6���]I       6%�	��]���A�
*;


total_loss��@

error_R&�T?

learning_rate_1�O�6r��4I       6%�	�g�]���A�
*;


total_loss���@

error_R&O>?

learning_rate_1�O�68*�pI       6%�	«�]���A�
*;


total_loss{HA

error_R�`?

learning_rate_1�O�6�Y�uI       6%�	�]���A�
*;


total_loss��@

error_RsW?

learning_rate_1�O�6�e-I       6%�	�X�]���A�
*;


total_loss�8�@

error_R4B?

learning_rate_1�O�6���I       6%�	_��]���A�
*;


total_loss@�@

error_R�S>?

learning_rate_1�O�6�_�I       6%�	A�]���A�
*;


total_loss$*�@

error_RA�T?

learning_rate_1�O�6>�ǽI       6%�	�*�]���A�
*;


total_losswG�@

error_R6?6?

learning_rate_1�O�6c�QI       6%�	,r�]���A�
*;


total_lossT�@

error_RϕN?

learning_rate_1�O�6����I       6%�	 ��]���A�
*;


total_lossQ�@

error_R��J?

learning_rate_1�O�6w�0�I       6%�	��]���A�
*;


total_lossᒺ@

error_R�[?

learning_rate_1�O�6���KI       6%�	�L�]���A�
*;


total_loss#<�@

error_R�'e?

learning_rate_1�O�6SL?I       6%�	ْ�]���A�
*;


total_loss�t�@

error_R��S?

learning_rate_1�O�6�觤I       6%�	�܎]���A�
*;


total_loss��@

error_R�rK?

learning_rate_1�O�6cP�I       6%�	$"�]���A�
*;


total_lossr��@

error_RɽQ?

learning_rate_1�O�6�GM�I       6%�	�d�]���A�
*;


total_loss\�@

error_R_?

learning_rate_1�O�6��XI       6%�	槏]���A�
*;


total_loss�N�@

error_R��W?

learning_rate_1�O�6��eI       6%�	��]���A�
*;


total_loss 2�@

error_R�V?

learning_rate_1�O�6ߛ�I       6%�	.?�]���A�
*;


total_loss��@

error_R}�E?

learning_rate_1�O�6$�I       6%�	]��]���A�
*;


total_loss��@

error_R��P?

learning_rate_1�O�6/��HI       6%�	�Ӑ]���A�
*;


total_loss�|�@

error_R=�Q?

learning_rate_1�O�6�øI       6%�	P�]���A�
*;


total_loss�(�@

error_R�`M?

learning_rate_1�O�6�l��I       6%�	1c�]���A�
*;


total_loss�_�@

error_Rx	R?

learning_rate_1�O�6"��BI       6%�	Y��]���A�
*;


total_loss�͊@

error_R�5X?

learning_rate_1�O�60VȰI       6%�	��]���A�
*;


total_loss�ә@

error_R\�R?

learning_rate_1�O�6���I       6%�	0�]���A�
*;


total_loss���@

error_R�[?

learning_rate_1�O�6-vRI       6%�	�t�]���A�
*;


total_loss#��@

error_R�{V?

learning_rate_1�O�6���$I       6%�	㸒]���A�
*;


total_loss��@

error_Rl�R?

learning_rate_1�O�6!m�I       6%�	{��]���A�
*;


total_loss�H�@

error_Rv�E?

learning_rate_1�O�6�c�:I       6%�	�>�]���A�
*;


total_lossQe�@

error_R��Q?

learning_rate_1�O�6 <��I       6%�	-��]���A�
*;


total_loss�@

error_R��S?

learning_rate_1�O�6)�)�I       6%�	�ʓ]���A�
*;


total_loss�^�@

error_R֒H?

learning_rate_1�O�6��=I       6%�	��]���A�
*;


total_loss�̝@

error_R�lN?

learning_rate_1�O�6���I       6%�	�Y�]���A�
*;


total_loss��A

error_RJW?

learning_rate_1�O�6ʤ��I       6%�	\��]���A�
*;


total_loss���@

error_RԞJ?

learning_rate_1�O�6�t)0I       6%�	��]���A�
*;


total_loss-4�@

error_R�o:?

learning_rate_1�O�6����I       6%�	�,�]���A�
*;


total_lossY�@

error_R�ID?

learning_rate_1�O�6[��I       6%�	p�]���A�
*;


total_loss.�@

error_R@2G?

learning_rate_1�O�6�μDI       6%�	l��]���A�
*;


total_lossn�@

error_R$*Z?

learning_rate_1�O�6��fI       6%�	i��]���A�
*;


total_loss���@

error_R��C?

learning_rate_1�O�6��I       6%�	�>�]���A�
*;


total_lossZn�@

error_Rw.O?

learning_rate_1�O�6�=I       6%�	���]���A�
*;


total_loss�@�@

error_R��>?

learning_rate_1�O�6谇�I       6%�	fŖ]���A�
*;


total_loss�8�@

error_Rs�S?

learning_rate_1�O�6sW��I       6%�	��]���A�
*;


total_loss�|A

error_R1�U?

learning_rate_1�O�6�|b!I       6%�	uQ�]���A�
*;


total_loss=�@

error_R�U?

learning_rate_1�O�6B�ODI       6%�	��]���A�
*;


total_loss#Q A

error_R�f?

learning_rate_1�O�6�ȩ�I       6%�	�ؗ]���A�
*;


total_losss+�@

error_RE1H?

learning_rate_1�O�6��7aI       6%�	2�]���A�
*;


total_loss�A

error_R?*Q?

learning_rate_1�O�6*4�OI       6%�	�]�]���A�
*;


total_loss�Y�@

error_RZ�b?

learning_rate_1�O�6�+��I       6%�	���]���A�
*;


total_loss�E�@

error_R�Q?

learning_rate_1�O�6��XI       6%�	��]���A�
*;


total_loss�F�@

error_R�[R?

learning_rate_1�O�6V�~I       6%�	2'�]���A�
*;


total_loss���@

error_RFC?

learning_rate_1�O�6�5�$I       6%�	�j�]���A�
*;


total_loss%��@

error_R�4;?

learning_rate_1�O�6��~I       6%�	���]���A�
*;


total_loss}��@

error_R��D?

learning_rate_1�O�6G�D�I       6%�	C��]���A�
*;


total_lossC6A

error_R{S?

learning_rate_1�O�6F1�I       6%�	;�]���A�
*;


total_loss���@

error_R(�K?

learning_rate_1�O�66��I       6%�	��]���A�
*;


total_lossm��@

error_R�oK?

learning_rate_1�O�6�{�I       6%�	TŚ]���A�
*;


total_loss�8�@

error_R�LV?

learning_rate_1�O�6��c�I       6%�	��]���A�
*;


total_loss���@

error_R#�T?

learning_rate_1�O�6YR�HI       6%�	�O�]���A�
*;


total_loss
��@

error_R�S?

learning_rate_1�O�6���RI       6%�	���]���A�
*;


total_loss�>�@

error_R�i?

learning_rate_1�O�62�XcI       6%�	q�]���A�
*;


total_lossnA

error_R�0>?

learning_rate_1�O�6�hI       6%�	D�]���A�
*;


total_loss��A

error_RZ�G?

learning_rate_1�O�6+%�zI       6%�	U��]���A�
*;


total_loss,)�@

error_R�~<?

learning_rate_1�O�64��I       6%�	�ќ]���A�
*;


total_loss8ȥ@

error_R�dS?

learning_rate_1�O�6� !.I       6%�	J�]���A�
*;


total_loss�q�@

error_R�8R?

learning_rate_1�O�6k@��I       6%�	b_�]���A�
*;


total_loss���@

error_R�MP?

learning_rate_1�O�6�6#I       6%�	ܦ�]���A�
*;


total_lossA� A

error_R�J?

learning_rate_1�O�6��I       6%�	I�]���A�
*;


total_loss��@

error_RhG?

learning_rate_1�O�6fn{�I       6%�	�9�]���A�
*;


total_loss��@

error_R��^?

learning_rate_1�O�6 �%I       6%�	ҁ�]���A�
*;


total_loss%Uo@

error_R�FO?

learning_rate_1�O�6��5�I       6%�	�ʞ]���A�
*;


total_loss
�e@

error_RW�W?

learning_rate_1�O�6��q�I       6%�	'�]���A�
*;


total_loss�@

error_RW&K?

learning_rate_1�O�6�o�I       6%�	�o�]���A�
*;


total_loss��@

error_R}RS?

learning_rate_1�O�6��%tI       6%�	���]���A�
*;


total_loss�\�@

error_Rn3Y?

learning_rate_1�O�6�Y�I       6%�	0�]���A�
*;


total_lossNA

error_R�a?

learning_rate_1�O�6��I       6%�	�c�]���A�
*;


total_loss�0�@

error_RϩJ?

learning_rate_1�O�6;'?I       6%�	箠]���A�
*;


total_lossE��@

error_R`RU?

learning_rate_1�O�6�V֪I       6%�	��]���A�
*;


total_losso+�@

error_R��>?

learning_rate_1�O�6I��I       6%�	w9�]���A�
*;


total_loss�T�@

error_R��R?

learning_rate_1�O�6!��I       6%�	9~�]���A�
*;


total_loss�	�@

error_R�EV?

learning_rate_1�O�6�p	HI       6%�	š]���A�
*;


total_loss�#�@

error_R��b?

learning_rate_1�O�6o��I       6%�	�	�]���A�
*;


total_loss߷�@

error_R�U?

learning_rate_1�O�6.4�I       6%�	%N�]���A�
*;


total_lossnn�@

error_RF�\?

learning_rate_1�O�6�ŵ;I       6%�	��]���A�
*;


total_loss�
�@

error_R�X?

learning_rate_1�O�6BuI       6%�	�բ]���A�
*;


total_loss�CA

error_R�eT?

learning_rate_1�O�6;R�I       6%�	��]���A�
*;


total_loss߯�@

error_Ri�V?

learning_rate_1�O�6���AI       6%�	�]�]���A�
*;


total_loss�e@

error_RŁ??

learning_rate_1�O�6cw&I       6%�	p��]���A�
*;


total_lossvay@

error_RjNI?

learning_rate_1�O�6DMAZI       6%�	*�]���A�
*;


total_loss^�@

error_R�#R?

learning_rate_1�O�6�
I       6%�	u*�]���A�
*;


total_loss���@

error_Rl�M?

learning_rate_1�O�6|y�I       6%�	:p�]���A�
*;


total_loss��@

error_R��F?

learning_rate_1�O�6F�,�I       6%�	2��]���A�
*;


total_lossڷ�@

error_RԺZ?

learning_rate_1�O�6�OZ�I       6%�	X��]���A�
*;


total_loss���@

error_R3�Q?

learning_rate_1�O�6�0^I       6%�	�E�]���A�
*;


total_loss��@

error_R�uO?

learning_rate_1�O�6��ZI       6%�	���]���A�
*;


total_lossod�@

error_R�hQ?

learning_rate_1�O�6�FUHI       6%�	�˥]���A�
*;


total_loss�{@

error_R/�M?

learning_rate_1�O�6���wI       6%�	c�]���A�
*;


total_loss��@

error_R9O?

learning_rate_1�O�6���I       6%�	2S�]���A�
*;


total_loss���@

error_Rj�=?

learning_rate_1�O�6j��I       6%�	���]���A�
*;


total_loss:��@

error_RtJ?

learning_rate_1�O�6i�I       6%�	�ئ]���A�
*;


total_loss�S�@

error_R�\`?

learning_rate_1�O�6��I       6%�	j�]���A�
*;


total_lossD�@

error_R�jV?

learning_rate_1�O�6����I       6%�	nd�]���A�
*;


total_loss�.�@

error_R�P?

learning_rate_1�O�6ߵ�I       6%�	L��]���A�
*;


total_loss8"A

error_Re�N?

learning_rate_1�O�6�}I       6%�	��]���A�
*;


total_lossT�@

error_R�S?

learning_rate_1�O�6G�ivI       6%�	�0�]���A�
*;


total_loss�5�@

error_R�@Q?

learning_rate_1�O�6�<!CI       6%�	_s�]���A�
*;


total_loss��@

error_RʅG?

learning_rate_1�O�66�-�I       6%�	��]���A�
*;


total_loss���@

error_R�"C?

learning_rate_1�O�6����I       6%�	���]���A�
*;


total_loss�V�@

error_R��G?

learning_rate_1�O�6��I       6%�	�D�]���A�
*;


total_loss��@

error_RϭN?

learning_rate_1�O�6%N�I       6%�	��]���A�
*;


total_loss
��@

error_RsPC?

learning_rate_1�O�6���I       6%�	l̩]���A�
*;


total_loss4M�@

error_RӤ??

learning_rate_1�O�6]�9I       6%�	��]���A�
*;


total_loss��@

error_R�H?

learning_rate_1�O�6U�I       6%�	^W�]���A�
*;


total_loss�@

error_R��L?

learning_rate_1�O�6g�mI       6%�	\��]���A�
*;


total_loss��@

error_R
?X?

learning_rate_1�O�6��I       6%�	8ߪ]���A�
*;


total_loss��A

error_R܎P?

learning_rate_1�O�6KO�I       6%�	"�]���A�
*;


total_lossT��@

error_R��5?

learning_rate_1�O�6����I       6%�	`f�]���A�*;


total_loss@

error_RnE?

learning_rate_1�O�6m_�I       6%�	��]���A�*;


total_loss��@

error_R@T?

learning_rate_1�O�6
��LI       6%�	��]���A�*;


total_loss9�@

error_Rj�M?

learning_rate_1�O�6ޱ�WI       6%�	e�]���A�*;


total_loss���@

error_R/�T?

learning_rate_1�O�6��*I       6%�	K��]���A�*;


total_lossdgA

error_R�N?

learning_rate_1�O�6��j.I       6%�	��]���A�*;


total_loss���@

error_RT�P?

learning_rate_1�O�6��KI       6%�	I4�]���A�*;


total_loss��@

error_RLY?

learning_rate_1�O�6/)wI       6%�	y�]���A�*;


total_loss���@

error_Rz�O?

learning_rate_1�O�6�%�I       6%�	λ�]���A�*;


total_loss��@

error_R�ZG?

learning_rate_1�O�6���I       6%�	���]���A�*;


total_loss(��@

error_RڋM?

learning_rate_1�O�6@@�I       6%�	�A�]���A�*;


total_loss�d�@

error_R)J6?

learning_rate_1�O�60�{�I       6%�	o��]���A�*;


total_loss���@

error_R>L?

learning_rate_1�O�6mz�I       6%�	�ɮ]���A�*;


total_loss��@

error_RN�A?

learning_rate_1�O�6��#(I       6%�	��]���A�*;


total_lossm�A

error_RnF?

learning_rate_1�O�6�|�4I       6%�	y_�]���A�*;


total_loss@��@

error_R��J?

learning_rate_1�O�6��YI       6%�	׭�]���A�*;


total_loss���@

error_RH�C?

learning_rate_1�O�6�$�'I       6%�	b��]���A�*;


total_lossז�@

error_R.�S?

learning_rate_1�O�6��I       6%�	wA�]���A�*;


total_loss�(�@

error_RZYX?

learning_rate_1�O�6X+~I       6%�	<��]���A�*;


total_lossf8�@

error_Rn�I?

learning_rate_1�O�6e�r�I       6%�	ΰ]���A�*;


total_loss�ˡ@

error_R�J@?

learning_rate_1�O�6����I       6%�	��]���A�*;


total_loss^��@

error_R�JQ?

learning_rate_1�O�6-��,I       6%�	�[�]���A�*;


total_loss�c�@

error_R.$C?

learning_rate_1�O�6Z�7�I       6%�	W��]���A�*;


total_loss���@

error_R[EV?

learning_rate_1�O�6k{1�I       6%�	��]���A�*;


total_loss���@

error_R��P?

learning_rate_1�O�6�#��I       6%�	i8�]���A�*;


total_loss���@

error_R�YT?

learning_rate_1�O�6)�. I       6%�	��]���A�*;


total_loss�A

error_R��X?

learning_rate_1�O�6R.�kI       6%�	rͲ]���A�*;


total_loss¶@

error_R�sQ?

learning_rate_1�O�6����I       6%�	��]���A�*;


total_lossJ�@

error_R�sD?

learning_rate_1�O�6Q���I       6%�	S�]���A�*;


total_loss?��@

error_RѵQ?

learning_rate_1�O�6ԸF`I       6%�	���]���A�*;


total_loss3�@

error_R6P?

learning_rate_1�O�6����I       6%�	E߳]���A�*;


total_loss���@

error_R��9?

learning_rate_1�O�6����I       6%�	�%�]���A�*;


total_losso��@

error_R��7?

learning_rate_1�O�6�*(�I       6%�	h�]���A�*;


total_loss�,�@

error_R�L?

learning_rate_1�O�6�N�I       6%�	H��]���A�*;


total_lossH�@

error_R�7G?

learning_rate_1�O�6�qI       6%�	�]���A�*;


total_loss��@

error_R��X?

learning_rate_1�O�6�1i�I       6%�	�3�]���A�*;


total_loss٤@

error_R� ]?

learning_rate_1�O�6^��`I       6%�	x�]���A�*;


total_lossp@

error_R�\?

learning_rate_1�O�6\��I       6%�	���]���A�*;


total_loss3f�@

error_RX?

learning_rate_1�O�6swK\I       6%�	G	�]���A�*;


total_loss K�@

error_Rd�T?

learning_rate_1�O�6�/��I       6%�	O�]���A�*;


total_loss��@

error_R�V\?

learning_rate_1�O�65QX�I       6%�	e��]���A�*;


total_lossk�@

error_R�\?

learning_rate_1�O�6�T��I       6%�	!ն]���A�*;


total_loss���@

error_R 5\?

learning_rate_1�O�6��II       6%�	?�]���A�*;


total_loss$��@

error_R�.a?

learning_rate_1�O�6 �ZI       6%�	�^�]���A�*;


total_lossQ�@

error_R�CA?

learning_rate_1�O�65��I       6%�	��]���A�*;


total_loss��@

error_R�VD?

learning_rate_1�O�6�,�I       6%�	5�]���A�*;


total_lossO&�@

error_R��_?

learning_rate_1�O�6��+KI       6%�	�9�]���A�*;


total_loss�Q�@

error_R͋M?

learning_rate_1�O�6,�I       6%�	���]���A�*;


total_lossz��@

error_R�jY?

learning_rate_1�O�6}�PI       6%�	�Ÿ]���A�*;


total_loss]f�@

error_R��N?

learning_rate_1�O�6Q��I       6%�	��]���A�*;


total_lossH��@

error_R\�L?

learning_rate_1�O�6p�?�I       6%�	�l�]���A�*;


total_loss���@

error_R��S?

learning_rate_1�O�6��0�I       6%�	�ٹ]���A�*;


total_lossMe�@

error_R��J?

learning_rate_1�O�6��lcI       6%�	� �]���A�*;


total_loss(˱@

error_RX8;?

learning_rate_1�O�6U��I       6%�	k�]���A�*;


total_loss��A

error_RfBO?

learning_rate_1�O�6OX��I       6%�	���]���A�*;


total_loss�Ŧ@

error_R)�U?

learning_rate_1�O�6��6I       6%�	�]���A�*;


total_loss���@

error_R��F?

learning_rate_1�O�6�]y�I       6%�	�N�]���A�*;


total_lossz�@

error_R΍J?

learning_rate_1�O�6���I       6%�	���]���A�*;


total_loss3��@

error_R�HU?

learning_rate_1�O�6���I       6%�	4��]���A�*;


total_loss���@

error_R��K?

learning_rate_1�O�6�aDCI       6%�	L�]���A�*;


total_lossQ�A

error_RȨU?

learning_rate_1�O�6N�_I       6%�	���]���A�*;


total_loss�ɇ@

error_R2j>?

learning_rate_1�O�6W>��I       6%�	�ܼ]���A�*;


total_lossF*�@

error_R�$J?

learning_rate_1�O�6~U�gI       6%�	T!�]���A�*;


total_lossFL�@

error_R��=?

learning_rate_1�O�6�1P�I       6%�	Fd�]���A�*;


total_loss%�c@

error_R$�@?

learning_rate_1�O�6��<8I       6%�	���]���A�*;


total_loss6.�@

error_R��>?

learning_rate_1�O�6ݜ+|I       6%�	��]���A�*;


total_lossv�@

error_Rm�I?

learning_rate_1�O�6r���I       6%�	�1�]���A�*;


total_loss
4�@

error_R��^?

learning_rate_1�O�6$�4I       6%�	�w�]���A�*;


total_loss]tZ@

error_Rx�N?

learning_rate_1�O�6�=*I       6%�	���]���A�*;


total_loss*,�@

error_RL�W?

learning_rate_1�O�60���I       6%�	[�]���A�*;


total_lossi�@

error_R\�8?

learning_rate_1�O�6`l�8I       6%�	.z�]���A�*;


total_loss?@�@

error_R�>?

learning_rate_1�O�6^���I       6%�	0ƿ]���A�*;


total_loss�Y�@

error_R��N?

learning_rate_1�O�6֥��I       6%�	S�]���A�*;


total_loss�@

error_RiH?

learning_rate_1�O�6_�qI       6%�	��]���A�*;


total_loss}��@

error_R��k?

learning_rate_1�O�6HP�I       6%�	���]���A�*;


total_lossI��@

error_Rm�N?

learning_rate_1�O�6�0�.I       6%�	�]���A�*;


total_lossdR�@

error_RW T?

learning_rate_1�O�6�)�I       6%�	�e�]���A�*;


total_lossn��@

error_R��S?

learning_rate_1�O�6���I       6%�	��]���A�*;


total_lossa��@

error_R�?B?

learning_rate_1�O�6�HI       6%�	� �]���A�*;


total_loss���@

error_R@Y?

learning_rate_1�O�6��sI       6%�	�J�]���A�*;


total_loss�k�@

error_R�V?

learning_rate_1�O�6ԫ 9I       6%�	7��]���A�*;


total_lossV{�@

error_RZ�[?

learning_rate_1�O�6��B�I       6%�	��]���A�*;


total_loss;e�@

error_RR�O?

learning_rate_1�O�6[�v�I       6%�	+�]���A�*;


total_lossݵ�@

error_Ra�\?

learning_rate_1�O�6�I       6%�	�p�]���A�*;


total_loss��@

error_RTkQ?

learning_rate_1�O�6n#�I       6%�	���]���A�*;


total_loss[o�@

error_RMxT?

learning_rate_1�O�6q ��I       6%�	���]���A�*;


total_lossv�A

error_R��K?

learning_rate_1�O�6�5
+I       6%�	�H�]���A�*;


total_lossD�@

error_RT�Y?

learning_rate_1�O�6�n�I       6%�	���]���A�*;


total_loss� A

error_R��Q?

learning_rate_1�O�6�ɯJI       6%�	,��]���A�*;


total_loss(�
A

error_RJTR?

learning_rate_1�O�63qɗI       6%�	��]���A�*;


total_loss��@

error_R�i9?

learning_rate_1�O�6���9I       6%�	\�]���A�*;


total_lossVl@

error_R,�J?

learning_rate_1�O�6-CI       6%�	-��]���A�*;


total_lossZ�@

error_R��N?

learning_rate_1�O�6��M�I       6%�	���]���A�*;


total_lossN �@

error_R��L?

learning_rate_1�O�6���II       6%�	�0�]���A�*;


total_lossv$�@

error_R RV?

learning_rate_1�O�6'�j�I       6%�	3r�]���A�*;


total_loss�E�@

error_R	[?

learning_rate_1�O�6�bƞI       6%�	��]���A�*;


total_loss:�@

error_R��S?

learning_rate_1�O�6%��I       6%�	���]���A�*;


total_lossIa�@

error_R*�H?

learning_rate_1�O�6-a?�I       6%�	�C�]���A�*;


total_loss��@

error_R�'P?

learning_rate_1�O�65�#�I       6%�	���]���A�*;


total_loss[0�@

error_RM�H?

learning_rate_1�O�6W���I       6%�	Y��]���A�*;


total_loss�ױ@

error_R&?H?

learning_rate_1�O�6���kI       6%�	��]���A�*;


total_lossB�@

error_R�[Z?

learning_rate_1�O�6�Lq)I       6%�	uV�]���A�*;


total_loss
�@

error_R�X?

learning_rate_1�O�6<�R|I       6%�	��]���A�*;


total_lossk�@

error_R��@?

learning_rate_1�O�6s`2I       6%�	X��]���A�*;


total_loss܇�@

error_Rv^U?

learning_rate_1�O�6��8I       6%�	�)�]���A�*;


total_loss� �@

error_Rt9?

learning_rate_1�O�6!�(I       6%�	pm�]���A�*;


total_loss땡@

error_R}'V?

learning_rate_1�O�6�ȃpI       6%�	���]���A�*;


total_loss���@

error_R�8??

learning_rate_1�O�6Ջ��I       6%�	���]���A�*;


total_loss-��@

error_R�?T?

learning_rate_1�O�6�@��I       6%�	yC�]���A�*;


total_loss�;�@

error_Rq�Z?

learning_rate_1�O�6�*��I       6%�	i��]���A�*;


total_loss���@

error_R͝W?

learning_rate_1�O�6Y�RI       6%�	#��]���A�*;


total_lossac�@

error_R�[B?

learning_rate_1�O�6s$I       6%�	�]���A�*;


total_lossܧ�@

error_Rt�\?

learning_rate_1�O�6�<u�I       6%�	yj�]���A�*;


total_loss���@

error_R��T?

learning_rate_1�O�6�HCI       6%�	o��]���A�*;


total_loss��A

error_R߆Q?

learning_rate_1�O�665JI       6%�	��]���A�*;


total_loss�D�@

error_R�HV?

learning_rate_1�O�6���%I       6%�	X�]���A�*;


total_loss ݬ@

error_RԊT?

learning_rate_1�O�66Y�I       6%�	f��]���A�*;


total_lossL��@

error_R��E?

learning_rate_1�O�6�U>I       6%�	o��]���A�*;


total_loss��z@

error_R��Z?

learning_rate_1�O�6�r��I       6%�	�&�]���A�*;


total_loss]Q�@

error_Rc??

learning_rate_1�O�6�XI       6%�	�m�]���A�*;


total_loss��@

error_R@�L?

learning_rate_1�O�6���I       6%�	��]���A�*;


total_loss���@

error_RS7?

learning_rate_1�O�6 ��@I       6%�	���]���A�*;


total_lossh�@

error_R�=R?

learning_rate_1�O�6��;�I       6%�	LC�]���A�*;


total_loss��f@

error_R��a?

learning_rate_1�O�6iΟI       6%�	���]���A�*;


total_lossx�@

error_R;J?

learning_rate_1�O�6�bI       6%�	 ��]���A�*;


total_loss)}�@

error_R�nQ?

learning_rate_1�O�6�+g=I       6%�	3�]���A�*;


total_lossi�A

error_RX?

learning_rate_1�O�6OF{�I       6%�	R`�]���A�*;


total_loss6��@

error_R��X?

learning_rate_1�O�6�B�VI       6%�	 ��]���A�*;


total_loss[3�@

error_Rq:N?

learning_rate_1�O�6���I       6%�	���]���A�*;


total_lossn��@

error_R�\J?

learning_rate_1�O�6�X�zI       6%�	W;�]���A�*;


total_loss���@

error_R\�T?

learning_rate_1�O�6̛��I       6%�	��]���A�*;


total_loss*��@

error_R��J?

learning_rate_1�O�6��jI       6%�	��]���A�*;


total_loss�RA

error_RxDG?

learning_rate_1�O�6K��I       6%�	��]���A�*;


total_loss�˕@

error_RŭL?

learning_rate_1�O�6SI�I       6%�	�I�]���A�*;


total_lossb�@

error_R�*N?

learning_rate_1�O�6�6�I       6%�	\��]���A�*;


total_lossA	A

error_R��S?

learning_rate_1�O�6?�I       6%�	 ��]���A�*;


total_loss�r�@

error_R��C?

learning_rate_1�O�6�5��I       6%�	 �]���A�*;


total_loss���@

error_R�Q?

learning_rate_1�O�6��uI       6%�	�g�]���A�*;


total_lossl��@

error_R1xE?

learning_rate_1�O�6��(�I       6%�	Ÿ�]���A�*;


total_loss���@

error_RWN??

learning_rate_1�O�6�:"xI       6%�	g �]���A�*;


total_loss���@

error_R��H?

learning_rate_1�O�6��9GI       6%�	(I�]���A�*;


total_lossn%�@

error_R��O?

learning_rate_1�O�6�RM�I       6%�	)��]���A�*;


total_loss ou@

error_RR;?

learning_rate_1�O�6���tI       6%�	���]���A�*;


total_loss��@

error_R��I?

learning_rate_1�O�6K��I       6%�	�&�]���A�*;


total_loss�@

error_R;�R?

learning_rate_1�O�68�{JI       6%�	�m�]���A�*;


total_loss3^A

error_R�M?

learning_rate_1�O�6�&#�I       6%�	l��]���A�*;


total_loss�>�@

error_R3�K?

learning_rate_1�O�6�6�I       6%�	���]���A�*;


total_loss�
�@

error_RI]_?

learning_rate_1�O�6BÊI       6%�	�7�]���A�*;


total_loss��@

error_R�/L?

learning_rate_1�O�6���I       6%�	G}�]���A�*;


total_losss�l@

error_R�&O?

learning_rate_1�O�6^CA-I       6%�	��]���A�*;


total_lossTZ�@

error_Ro�`?

learning_rate_1�O�6=�I       6%�	<	�]���A�*;


total_lossF+�@

error_R�L?

learning_rate_1�O�6̺pI       6%�	�T�]���A�*;


total_loss,A

error_R��X?

learning_rate_1�O�6����I       6%�	��]���A�*;


total_lossV��@

error_RCMP?

learning_rate_1�O�62d��I       6%�	���]���A�*;


total_loss�R�@

error_RET?

learning_rate_1�O�6�8�PI       6%�	g,�]���A�*;


total_loss��@

error_R;K?

learning_rate_1�O�6�l%I       6%�	�o�]���A�*;


total_loss�[@

error_R$>N?

learning_rate_1�O�6P���I       6%�	V��]���A�*;


total_loss��@

error_R�VP?

learning_rate_1�O�6���I       6%�	5 �]���A�*;


total_loss�l�@

error_R��a?

learning_rate_1�O�6���mI       6%�	6M�]���A�*;


total_loss��t@

error_R[BP?

learning_rate_1�O�6���I       6%�	ړ�]���A�*;


total_loss?�A

error_R�
K?

learning_rate_1�O�6{䟯I       6%�	.��]���A�*;


total_loss�)�@

error_R#.^?

learning_rate_1�O�63��I       6%�	�#�]���A�*;


total_loss���@

error_R��V?

learning_rate_1�O�6�y)�I       6%�	?m�]���A�*;


total_loss��Z@

error_R�sN?

learning_rate_1�O�65�΅I       6%�	5��]���A�*;


total_lossDk�@

error_R7�T?

learning_rate_1�O�6��<fI       6%�	F��]���A�*;


total_loss�7�@

error_R�U?

learning_rate_1�O�6�"��I       6%�	�<�]���A�*;


total_loss�X�@

error_R�c1?

learning_rate_1�O�65�xXI       6%�	���]���A�*;


total_lossع�@

error_R�eV?

learning_rate_1�O�6M��I       6%�	���]���A�*;


total_lossg�@

error_R��>?

learning_rate_1�O�6�P�I       6%�	_�]���A�*;


total_loss?��@

error_Rq�K?

learning_rate_1�O�6ޱ�nI       6%�	�P�]���A�*;


total_loss���@

error_R��Y?

learning_rate_1�O�6�H�I       6%�	���]���A�*;


total_loss}��@

error_R;�M?

learning_rate_1�O�6�i|I       6%�	���]���A�*;


total_loss�oj@

error_RJ�\?

learning_rate_1�O�6Oܢ�I       6%�		@�]���A�*;


total_loss�_�@

error_R8P?

learning_rate_1�O�6|S��I       6%�	΅�]���A�*;


total_loss�t@

error_R�6L?

learning_rate_1�O�6j�V�I       6%�	���]���A�*;


total_loss�5�@

error_R��P?

learning_rate_1�O�6�kR�I       6%�	)�]���A�*;


total_loss���@

error_RTS?

learning_rate_1�O�6ؿI       6%�	\Y�]���A�*;


total_loss<��@

error_R1;K?

learning_rate_1�O�6i��DI       6%�	���]���A�*;


total_loss���@

error_R��O?

learning_rate_1�O�6k�cI       6%�	H��]���A�*;


total_loss�ڰ@

error_R�\`?

learning_rate_1�O�6��nNI       6%�	A�]���A�*;


total_lossI@�@

error_R�C?

learning_rate_1�O�6fB�8I       6%�	���]���A�*;


total_loss
J�@

error_R��T?

learning_rate_1�O�6���@I       6%�	���]���A�*;


total_lossz(�@

error_RќQ?

learning_rate_1�O�6���I       6%�	�!�]���A�*;


total_lossQ�@

error_R��T?

learning_rate_1�O�6���NI       6%�	��]���A�*;


total_loss��A

error_Rm�b?

learning_rate_1�O�6��l~I       6%�	��]���A�*;


total_loss���@

error_RL7P?

learning_rate_1�O�6�g�2I       6%�	!�]���A�*;


total_loss���@

error_R�Z?

learning_rate_1�O�6UQ�"I       6%�	�e�]���A�*;


total_loss��@

error_RM�S?

learning_rate_1�O�6I�M�I       6%�	���]���A�*;


total_lossÒ@

error_RcY?

learning_rate_1�O�6�itI       6%�	K�]���A�*;


total_loss
i�@

error_R��Y?

learning_rate_1�O�6��s�I       6%�	�X�]���A�*;


total_loss*��@

error_R,J?

learning_rate_1�O�6�6+�I       6%�	;��]���A�*;


total_loss&�	A

error_R�wA?

learning_rate_1�O�6��qI       6%�	���]���A�*;


total_loss�^�@

error_R�N?

learning_rate_1�O�6*z�eI       6%�	�)�]���A�*;


total_loss�:�@

error_R�1C?

learning_rate_1�O�6�N�I       6%�	Lt�]���A�*;


total_loss�ގ@

error_R�4?

learning_rate_1�O�6sk��I       6%�	)��]���A�*;


total_loss���@

error_R�`W?

learning_rate_1�O�6a2I       6%�	���]���A�*;


total_loss�g�@

error_R=�L?

learning_rate_1�O�6�A�I       6%�	3H�]���A�*;


total_lossTs�@

error_R7�Y?

learning_rate_1�O�6����I       6%�	���]���A�*;


total_lossz�]@

error_R��C?

learning_rate_1�O�6E��\I       6%�	���]���A�*;


total_loss�i�@

error_R��X?

learning_rate_1�O�6���zI       6%�	��]���A�*;


total_loss�7�@

error_R��J?

learning_rate_1�O�6W[$�I       6%�	�c�]���A�*;


total_loss�0�@

error_R�Q?

learning_rate_1�O�6�;iI       6%�	}��]���A�*;


total_loss�b�@

error_R�Z?

learning_rate_1�O�6e ˿I       6%�	J��]���A�*;


total_lossth�@

error_R��R?

learning_rate_1�O�6���fI       6%�	p/�]���A�*;


total_loss8��@

error_R��C?

learning_rate_1�O�6x�z�I       6%�	�q�]���A�*;


total_loss׃�@

error_R0I?

learning_rate_1�O�6+�I       6%�	=��]���A�*;


total_lossQ��@

error_Rd�O?

learning_rate_1�O�6�'�I       6%�	��]���A�*;


total_lossJޗ@

error_R.hB?

learning_rate_1�O�6Yr�AI       6%�	\?�]���A�*;


total_loss�ԭ@

error_R�"D?

learning_rate_1�O�6��rjI       6%�	���]���A�*;


total_loss�=�@

error_R��I?

learning_rate_1�O�6Z��I       6%�	f��]���A�*;


total_lossS�@

error_R��W?

learning_rate_1�O�6��VI       6%�	|�]���A�*;


total_loss�d�@

error_RÂM?

learning_rate_1�O�6$���I       6%�	:_�]���A�*;


total_loss�n�@

error_R��U?

learning_rate_1�O�6i\�I       6%�	��]���A�*;


total_loss�N�@

error_R;.h?

learning_rate_1�O�6�B]I       6%�	���]���A�*;


total_lossz��@

error_R��P?

learning_rate_1�O�6���4I       6%�	A;�]���A�*;


total_loss���@

error_R&lP?

learning_rate_1�O�64AXAI       6%�	%��]���A�*;


total_loss-��@

error_R��H?

learning_rate_1�O�6�pTI       6%�	��]���A�*;


total_loss��@

error_R��N?

learning_rate_1�O�6���I       6%�	Z�]���A�*;


total_loss��@

error_R֏S?

learning_rate_1�O�6[Q��I       6%�	B^�]���A�*;


total_loss[u@

error_R��U?

learning_rate_1�O�6�.�I       6%�	å�]���A�*;


total_loss��@

error_R�E?

learning_rate_1�O�6=��I       6%�	���]���A�*;


total_loss��@

error_R��F?

learning_rate_1�O�6��;I       6%�	A/�]���A�*;


total_loss���@

error_Rh#J?

learning_rate_1�O�6�q�I       6%�	�r�]���A�*;


total_loss�G�@

error_R={A?

learning_rate_1�O�6��-I       6%�	۶�]���A�*;


total_loss9&�@

error_R�fh?

learning_rate_1�O�6�%��I       6%�	���]���A�*;


total_loss�A�@

error_R K?

learning_rate_1�O�6i��I       6%�	�?�]���A�*;


total_loss�̏@

error_R=?G?

learning_rate_1�O�6ט�2I       6%�	|��]���A�*;


total_loss�@

error_R�lL?

learning_rate_1�O�6H�%(I       6%�	���]���A�*;


total_loss\�@

error_RcU?

learning_rate_1�O�6#VKI       6%�	'+�]���A�*;


total_loss��@

error_R6a?

learning_rate_1�O�6�-�rI       6%�	Tw�]���A�*;


total_loss&��@

error_R�!F?

learning_rate_1�O�6�F0�I       6%�	���]���A�*;


total_loss.ǖ@

error_Rx�P?

learning_rate_1�O�6:�I       6%�	J�]���A�*;


total_lossC2�@

error_R._?

learning_rate_1�O�6�n1�I       6%�	T�]���A�*;


total_loss�%�@

error_R�*I?

learning_rate_1�O�6�RcI       6%�	��]���A�*;


total_loss�+�@

error_R��E?

learning_rate_1�O�6�M؎I       6%�	%��]���A�*;


total_loss�M�@

error_R�cJ?

learning_rate_1�O�6��~I       6%�	H-�]���A�*;


total_loss�k�@

error_R@�O?

learning_rate_1�O�6���I       6%�	qp�]���A�*;


total_loss���@

error_R�\?

learning_rate_1�O�6�i�}I       6%�	[��]���A�*;


total_lossT1�@

error_R�gM?

learning_rate_1�O�6j8XI       6%�	���]���A�*;


total_loss��@

error_Ri�B?

learning_rate_1�O�6�\/�I       6%�	�9�]���A�*;


total_loss �@

error_R��L?

learning_rate_1�O�6
|d�I       6%�	���]���A�*;


total_lossa��@

error_R��W?

learning_rate_1�O�6^m�I       6%�	���]���A�*;


total_lossSD�@

error_R��8?

learning_rate_1�O�6>�rI       6%�	$�]���A�*;


total_loss��@

error_R��a?

learning_rate_1�O�6�؜�I       6%�	�X�]���A�*;


total_loss�5�@

error_R��C?

learning_rate_1�O�6;Q�I       6%�	ל�]���A�*;


total_loss��@

error_R�R?

learning_rate_1�O�6A�4I       6%�	���]���A�*;


total_loss�!�@

error_R�L?

learning_rate_1�O�6� ��I       6%�	5 �]���A�*;


total_loss} �@

error_R�|H?

learning_rate_1�O�6Ŷ�#I       6%�	�e�]���A�*;


total_lossR��@

error_R��H?

learning_rate_1�O�6�9aI       6%�	���]���A�*;


total_lossj��@

error_R@�@?

learning_rate_1�O�6�G�I       6%�	O��]���A�*;


total_lossh��@

error_Rq55?

learning_rate_1�O�6��LI       6%�	n3�]���A�*;


total_loss�b�@

error_R�BV?

learning_rate_1�O�6-�I       6%�	hz�]���A�*;


total_loss���@

error_RR�X?

learning_rate_1�O�6U9�I       6%�	��]���A�*;


total_loss6��@

error_RM�e?

learning_rate_1�O�6��I       6%�	;�]���A�*;


total_loss�
n@

error_R�P?

learning_rate_1�O�6���I       6%�	�Q�]���A�*;


total_loss B�@

error_R�lX?

learning_rate_1�O�6�W	�I       6%�	y��]���A�*;


total_loss���@

error_R��U?

learning_rate_1�O�6`?gI       6%�	���]���A�*;


total_loss�7�@

error_R6�>?

learning_rate_1�O�6%Uu�I       6%�	�&�]���A�*;


total_loss��@

error_RlIf?

learning_rate_1�O�6t�vI       6%�	�p�]���A�*;


total_losslBA

error_R�W?

learning_rate_1�O�6��I       6%�	��]���A�*;


total_loss*��@

error_R�O?

learning_rate_1�O�6�aA�I       6%�	t�]���A�*;


total_loss���@

error_R� O?

learning_rate_1�O�6����I       6%�	�N�]���A�*;


total_lossl��@

error_R)�A?

learning_rate_1�O�6ݜ>I       6%�	L��]���A�*;


total_loss�@�@

error_R��U?

learning_rate_1�O�6?E�I       6%�	���]���A�*;


total_loss���@

error_R��W?

learning_rate_1�O�6ӋE�I       6%�	�'�]���A�*;


total_loss�V�@

error_R=1L?

learning_rate_1�O�6�ޠI       6%�	�r�]���A�*;


total_lossR2�@

error_RQ�=?

learning_rate_1�O�6@��TI       6%�	;��]���A�*;


total_loss��@

error_R��U?

learning_rate_1�O�6�FI       6%�	���]���A�*;


total_loss��@

error_R��D?

learning_rate_1�O�6��LI       6%�	�D�]���A�*;


total_loss8/�@

error_R�S?

learning_rate_1�O�6��4LI       6%�	r��]���A�*;


total_loss�%�@

error_R�ZH?

learning_rate_1�O�6�j��I       6%�	��]���A�*;


total_loss��@

error_Ra�E?

learning_rate_1�O�6���=I       6%�	�]���A�*;


total_loss�"�@

error_R�T?

learning_rate_1�O�6��ޡI       6%�	7h�]���A�*;


total_loss
��@

error_R}'J?

learning_rate_1�O�6�U~I       6%�	��]���A�*;


total_loss�{A

error_RJ�W?

learning_rate_1�O�6�%$�I       6%�	���]���A�*;


total_loss���@

error_RH?

learning_rate_1�O�6�.�I       6%�	�8�]���A�*;


total_loss�ղ@

error_R� =?

learning_rate_1�O�6�YDrI       6%�	�y�]���A�*;


total_loss���@

error_R�7V?

learning_rate_1�O�6e�fI       6%�	A��]���A�*;


total_loss�ڦ@

error_R��N?

learning_rate_1�O�6�6�I       6%�	���]���A�*;


total_loss_�@

error_R=bO?

learning_rate_1�O�6&K�I       6%�	�B�]���A�*;


total_loss	;]@

error_R�hC?

learning_rate_1�O�6�pUI       6%�	���]���A�*;


total_lossm�A

error_R-yS?

learning_rate_1�O�6QLm�I       6%�	6��]���A�*;


total_loss���@

error_Rl�S?

learning_rate_1�O�6���I       6%�	��]���A�*;


total_loss�1A

error_R!fR?

learning_rate_1�O�6���nI       6%�	�^�]���A�*;


total_lossJ�A

error_R<�a?

learning_rate_1�O�6"D�I       6%�	I��]���A�*;


total_loss/]�@

error_Rד[?

learning_rate_1�O�6ψq�I       6%�	y��]���A�*;


total_loss�_�@

error_R��P?

learning_rate_1�O�6N�I       6%�	_P�]���A�*;


total_loss��@

error_R��F?

learning_rate_1�O�6
�TI       6%�	��]���A�*;


total_loss�O�@

error_R�%P?

learning_rate_1�O�6��-I       6%�	x��]���A�*;


total_loss쓚@

error_R�YV?

learning_rate_1�O�6��s$I       6%�	�-�]���A�*;


total_loss ��@

error_R��U?

learning_rate_1�O�6|�O�I       6%�	br�]���A�*;


total_lossz��@

error_R)jP?

learning_rate_1�O�6�]-�I       6%�	ʳ�]���A�*;


total_loss5v A

error_RZ�;?

learning_rate_1�O�6�ap�I       6%�	L��]���A�*;


total_loss���@

error_R�'T?

learning_rate_1�O�6!dxI       6%�	eJ�]���A�*;


total_loss��XA

error_R}HA?

learning_rate_1�O�6�LA�I       6%�	Ֆ�]���A�*;


total_loss<zA

error_R��e?

learning_rate_1�O�6eh%I       6%�	���]���A�*;


total_lossݓ@

error_R�L?

learning_rate_1�O�6͟~I       6%�	�!�]���A�*;


total_loss�@

error_R(�C?

learning_rate_1�O�6ׅ �I       6%�	���]���A�*;


total_loss�j�@

error_R��6?

learning_rate_1�O�6X�i{I       6%�	L��]���A�*;


total_loss���@

error_RaRB?

learning_rate_1�O�6���2I       6%�	z ^���A�*;


total_loss��@

error_R'_?

learning_rate_1�O�6�1տI       6%�	yY ^���A�*;


total_loss��@

error_R4tN?

learning_rate_1�O�6��$mI       6%�	޴ ^���A�*;


total_loss&��@

error_R�OY?

learning_rate_1�O�6�yI       6%�	Y� ^���A�*;


total_loss���@

error_R3kL?

learning_rate_1�O�6�sI       6%�	�;^���A�*;


total_loss��A

error_RúS?

learning_rate_1�O�6�Vd$I       6%�	��^���A�*;


total_loss�~�@

error_RF]?

learning_rate_1�O�6�c� I       6%�	��^���A�*;


total_losss��@

error_RTx]?

learning_rate_1�O�6�I       6%�	3	^���A�*;


total_lossu9A

error_RiHU?

learning_rate_1�O�6a���I       6%�	�L^���A�*;


total_lossX�@

error_R7U?

learning_rate_1�O�6mu�|I       6%�	��^���A�*;


total_loss
v�@

error_R��R?

learning_rate_1�O�6'�e�I       6%�	��^���A�*;


total_lossqԏ@

error_R_>M?

learning_rate_1�O�6�B�bI       6%�	�^���A�*;


total_lossT_�@

error_RjpG?

learning_rate_1�O�6Eo��I       6%�	f`^���A�*;


total_loss��@

error_RR�W?

learning_rate_1�O�6�(n�I       6%�	=�^���A�*;


total_loss�*A

error_RngU?

learning_rate_1�O�6�c"I       6%�	��^���A�*;


total_loss��@

error_RZ\?

learning_rate_1�O�6���I       6%�	B1^���A�*;


total_lossm�@

error_RV?

learning_rate_1�O�6Q�F�I       6%�	�w^���A�*;


total_lossc��@

error_R�9[?

learning_rate_1�O�6�Ƕ�I       6%�	��^���A�*;


total_loss��@

error_R��P?

learning_rate_1�O�6��}'I       6%�	��^���A�*;


total_loss��A

error_R��P?

learning_rate_1�O�6#�L�I       6%�	F^���A�*;


total_loss���@

error_Rr�a?

learning_rate_1�O�6̳�I       6%�	Չ^���A�*;


total_loss�ġ@

error_R�ge?

learning_rate_1�O�6c�[I       6%�	��^���A�*;


total_loss�7�@

error_R��S?

learning_rate_1�O�6���VI       6%�	(^���A�*;


total_loss�A

error_R�`T?

learning_rate_1�O�6Q(b�I       6%�	XR^���A�*;


total_loss��@

error_R�eR?

learning_rate_1�O�6�lAyI       6%�	��^���A�*;


total_loss���@

error_R}O?

learning_rate_1�O�6_M]0I       6%�	��^���A�*;


total_loss�и@

error_R��M?

learning_rate_1�O�6����I       6%�	�^���A�*;


total_lossq.�@

error_RV�V?

learning_rate_1�O�6S��qI       6%�	�d^���A�*;


total_loss�n�@

error_R�KE?

learning_rate_1�O�6`q��I       6%�	��^���A�*;


total_loss�	�@

error_R��J?

learning_rate_1�O�6EfN3I       6%�	��^���A�*;


total_loss���@

error_R��P?

learning_rate_1�O�6�B�%I       6%�	�2^���A�*;


total_loss���@

error_Rq�Q?

learning_rate_1�O�6�4dLI       6%�	�r^���A�*;


total_loss���@

error_R��K?

learning_rate_1�O�6s�o-I       6%�	ʵ^���A�*;


total_loss8ȵ@

error_RofP?

learning_rate_1�O�6���|I       6%�	\�^���A�*;


total_loss�F�@

error_RHoT?

learning_rate_1�O�6Ǭ_�I       6%�	x<	^���A�*;


total_loss��@

error_R��c?

learning_rate_1�O�6Y�UI       6%�	�	^���A�*;


total_loss8�@

error_R{�M?

learning_rate_1�O�6b!�I       6%�	��	^���A�*;


total_lossqڰ@

error_R\eW?

learning_rate_1�O�6\6��I       6%�	m
^���A�*;


total_loss���@

error_R$V?

learning_rate_1�O�6MIy�I       6%�	�U
^���A�*;


total_lossC��@

error_R�=O?

learning_rate_1�O�6fX#`I       6%�	��
^���A�*;


total_loss���@

error_R}�=?

learning_rate_1�O�6w�`PI       6%�	o�
^���A�*;


total_loss
�@

error_R��[?

learning_rate_1�O�6RydI       6%�	�,^���A�*;


total_loss{	�@

error_RZW?

learning_rate_1�O�6x_�I       6%�	.s^���A�*;


total_loss��A

error_R�R?

learning_rate_1�O�6�wI       6%�	q�^���A�*;


total_loss!�a@

error_R�KE?

learning_rate_1�O�6��,�I       6%�	q#^���A�*;


total_loss/��@

error_R��U?

learning_rate_1�O�6]��I       6%�	�r^���A�*;


total_loss���@

error_R�O?

learning_rate_1�O�6+��6I       6%�	̻^���A�*;


total_lossX�@

error_R�4?

learning_rate_1�O�6���I       6%�	V^���A�*;


total_loss��@

error_R{�O?

learning_rate_1�O�6��I       6%�	mG^���A�*;


total_loss{|�@

error_RQ+E?

learning_rate_1�O�6}��I       6%�	��^���A�*;


total_loss���@

error_R`�X?

learning_rate_1�O�6+��I       6%�	#�^���A�*;


total_loss��@

error_RWLY?

learning_rate_1�O�6ܥ��I       6%�	� ^���A�*;


total_loss!9h@

error_R
J9?

learning_rate_1�O�6��_I       6%�	�e^���A�*;


total_loss���@

error_R;L?

learning_rate_1�O�6_̱I       6%�	Ӫ^���A�*;


total_losstk�@

error_R#JL?

learning_rate_1�O�6�gmI       6%�	��^���A�*;


total_loss���@

error_RJ8H?

learning_rate_1�O�6�rqgI       6%�	F3^���A�*;


total_lossҏ�@

error_Ri
H?

learning_rate_1�O�6g� lI       6%�	�v^���A�*;


total_lossU�@

error_R\�R?

learning_rate_1�O�6���]I       6%�	�^���A�*;


total_loss�@

error_R�pT?

learning_rate_1�O�6X�&I       6%�	��^���A�*;


total_loss=��@

error_R�;F?

learning_rate_1�O�6�	��I       6%�	�>^���A�*;


total_loss�۵@

error_R�ZR?

learning_rate_1�O�6ˊo�I       6%�	F�^���A�*;


total_loss��@

error_R�!I?

learning_rate_1�O�6k8)�I       6%�	��^���A�*;


total_lossRw�@

error_R��O?

learning_rate_1�O�6��˷I       6%�	]^���A�*;


total_loss��@

error_R��D?

learning_rate_1�O�6cv��I       6%�	�^^���A�*;


total_loss��@

error_R-#L?

learning_rate_1�O�6I��I       6%�	_�^���A�*;


total_loss@w�@

error_R�
A?

learning_rate_1�O�6.���I       6%�	u�^���A�*;


total_loss�O�@

error_R��N?

learning_rate_1�O�6�c�I       6%�	:^���A�*;


total_losshE�@

error_RV�N?

learning_rate_1�O�61i@�I       6%�	��^���A�*;


total_loss�,�@

error_R3�O?

learning_rate_1�O�6��)I       6%�	��^���A�*;


total_loss�@�@

error_R�*S?

learning_rate_1�O�6��R>I       6%�	0^���A�*;


total_loss���@

error_RBT?

learning_rate_1�O�6
�I       6%�	zP^���A�*;


total_loss�� A

error_R��N?

learning_rate_1�O�6L�4I       6%�	�^���A�*;


total_loss�4�@

error_Ra1P?

learning_rate_1�O�6�L�I       6%�	�^���A�*;


total_loss�[�@

error_R�tT?

learning_rate_1�O�6,Pi�I       6%�	V^���A�*;


total_lossh��@

error_R.�K?

learning_rate_1�O�67lp�I       6%�	a^���A�*;


total_lossQ�&A

error_Rb?

learning_rate_1�O�6��AI       6%�	M�^���A�*;


total_loss���@

error_RLT?

learning_rate_1�O�6���I       6%�	��^���A�*;


total_loss꿑@

error_R�b?

learning_rate_1�O�6�Yh�I       6%�	�;^���A�*;


total_loss@J�@

error_R��O?

learning_rate_1�O�6m
I       6%�	r�^���A�*;


total_loss�l�@

error_ROM?

learning_rate_1�O�6��I       6%�	��^���A�*;


total_loss���@

error_R�C?

learning_rate_1�O�6�j��I       6%�	�^���A�*;


total_loss�2�@

error_R��V?

learning_rate_1�O�6���6I       6%�	V^���A�*;


total_loss�I
A

error_R��N?

learning_rate_1�O�6Ȣ�nI       6%�	��^���A�*;


total_lossܥ�@

error_R?%@?

learning_rate_1�O�6���I       6%�	��^���A�*;


total_lossf�@

error_R��D?

learning_rate_1�O�6�{�I       6%�	�7^���A�*;


total_lossaA

error_R�3w?

learning_rate_1�O�6,�+CI       6%�	�{^���A�*;


total_loss℣@

error_R2�??

learning_rate_1�O�6�čCI       6%�	x�^���A�*;


total_lossR��@

error_Rd4_?

learning_rate_1�O�6��5I       6%�	�^���A�*;


total_loss*��@

error_R�gO?

learning_rate_1�O�6�צsI       6%�	�H^���A�*;


total_loss�+�@

error_R8�7?

learning_rate_1�O�6�M1�I       6%�	�^���A�*;


total_loss�@

error_R�#/?

learning_rate_1�O�6 0�0I       6%�	��^���A�*;


total_loss|��@

error_Rvb?

learning_rate_1�O�6:zI       6%�	<^���A�*;


total_lossw@

error_R)�S?

learning_rate_1�O�6�{��I       6%�	�Y^���A�*;


total_lossX˨@

error_R6?

learning_rate_1�O�6�e�I       6%�	�^���A�*;


total_loss�v�@

error_R��O?

learning_rate_1�O�6�O�`I       6%�	��^���A�*;


total_loss;��@

error_R�HB?

learning_rate_1�O�6S�	XI       6%�	�)^���A�*;


total_loss/s@

error_R\�X?

learning_rate_1�O�69��I       6%�	�v^���A�*;


total_lossRm`@

error_R��K?

learning_rate_1�O�6� �AI       6%�	g�^���A�*;


total_loss:i�@

error_Ri�;?

learning_rate_1�O�6%�j�I       6%�	�^���A�*;


total_loss���@

error_R9K?

learning_rate_1�O�67.rI       6%�	�U^���A�*;


total_loss���@

error_R�%M?

learning_rate_1�O�6�d	9I       6%�	�^���A�*;


total_loss���@

error_R�U?

learning_rate_1�O�6�I       6%�	o�^���A�*;


total_loss���@

error_R2.\?

learning_rate_1�O�6G)'mI       6%�	G^���A�*;


total_loss���@

error_RArN?

learning_rate_1�O�6�Ʌ�I       6%�	�^���A�*;


total_loss�M�@

error_R��S?

learning_rate_1�O�6 �lI       6%�	g�^���A�*;


total_loss��@

error_R�T?

learning_rate_1�O�6,��I       6%�	_^���A�*;


total_loss(|�@

error_R�C?

learning_rate_1�O�6�~9 I       6%�	�V^���A�*;


total_lossUw�@

error_R�DD?

learning_rate_1�O�6E��I       6%�	P�^���A�*;


total_lossD�@

error_R�2P?

learning_rate_1�O�6�%�I       6%�	z�^���A�*;


total_loss�l�@

error_RlGX?

learning_rate_1�O�6�!�I       6%�	n0^���A�*;


total_loss|��@

error_R�wS?

learning_rate_1�O�6rPII       6%�	ls^���A�*;


total_loss�X�@

error_R�sW?

learning_rate_1�O�6���$I       6%�	x�^���A�*;


total_lossL��@

error_R�A?

learning_rate_1�O�6br�I       6%�	�^���A�*;


total_lossA��@

error_R�dX?

learning_rate_1�O�6�oO�I       6%�	�A^���A�*;


total_lossT��@

error_R��N?

learning_rate_1�O�6�{�I       6%�	��^���A�*;


total_loss�X�@

error_R��C?

learning_rate_1�O�6F���I       6%�	�^���A�*;


total_loss8q�@

error_R��J?

learning_rate_1�O�6�~�eI       6%�	�' ^���A�*;


total_loss�s�@

error_R�EZ?

learning_rate_1�O�6��ǤI       6%�	�m ^���A�*;


total_loss�)�@

error_Rj�7?

learning_rate_1�O�6q��I       6%�	,� ^���A�*;


total_loss�k�@

error_R��T?

learning_rate_1�O�6���^I       6%�	1!^���A�*;


total_loss�@

error_R�\X?

learning_rate_1�O�6�V$�I       6%�	]]!^���A�*;


total_loss���@

error_R�%\?

learning_rate_1�O�6	V��I       6%�	�!^���A�*;


total_loss^?�@

error_R�V?

learning_rate_1�O�6D���I       6%�	��!^���A�*;


total_loss[��@

error_REHS?

learning_rate_1�O�6od��I       6%�	6"^���A�*;


total_loss$^�@

error_RtU?

learning_rate_1�O�6u��~I       6%�	�}"^���A�*;


total_loss��@

error_R)�I?

learning_rate_1�O�6d���I       6%�	B�"^���A�*;


total_loss�^�@

error_R�X?

learning_rate_1�O�6p���I       6%�	R
#^���A�*;


total_lossB��@

error_RɢY?

learning_rate_1�O�6���I       6%�	rM#^���A�*;


total_losso��@

error_R�KM?

learning_rate_1�O�6����I       6%�	˔#^���A�*;


total_loss��@

error_RH^X?

learning_rate_1�O�6����I       6%�	��#^���A�*;


total_lossps�@

error_R!8?

learning_rate_1�O�6\�+HI       6%�	Q$^���A�*;


total_loss�@

error_R&�\?

learning_rate_1�O�6�l�I       6%�	Md$^���A�*;


total_losset�@

error_Rq�??

learning_rate_1�O�6�:�I       6%�	Q�$^���A�*;


total_lossD��@

error_RҌU?

learning_rate_1�O�6ࢷI       6%�	��$^���A�*;


total_loss{x�@

error_RoiT?

learning_rate_1�O�6�s;�I       6%�	.%^���A�*;


total_loss��@

error_R��I?

learning_rate_1�O�6��5fI       6%�	�r%^���A�*;


total_loss���@

error_R�$K?

learning_rate_1�O�6R��I       6%�	1�%^���A�*;


total_loss���@

error_R݋H?

learning_rate_1�O�61az�I       6%�	��%^���A�*;


total_loss���@

error_RO�S?

learning_rate_1�O�6K��2I       6%�	�B&^���A�*;


total_lossR@�@

error_R�G?

learning_rate_1�O�6���jI       6%�	�&^���A�*;


total_lossﻺ@

error_R�j?

learning_rate_1�O�6�?I       6%�	'�&^���A�*;


total_loss)ڽ@

error_R1�I?

learning_rate_1�O�6��nI       6%�	�'^���A�*;


total_lossZg�@

error_R N?

learning_rate_1�O�6F��I       6%�	�^'^���A�*;


total_lossE��@

error_R�e?

learning_rate_1�O�6H	I       6%�	�'^���A�*;


total_losslm�@

error_RV I?

learning_rate_1�O�6.HI       6%�	��'^���A�*;


total_loss�.�@

error_R�U?

learning_rate_1�O�6>ͺJI       6%�	�.(^���A�*;


total_lossC�A

error_Rr=D?

learning_rate_1�O�6�˜�I       6%�	p(^���A�*;


total_loss:��@

error_R�xJ?

learning_rate_1�O�6L��I       6%�	�(^���A�*;


total_lossꃝ@

error_R�JM?

learning_rate_1�O�6_3m
I       6%�	��(^���A�*;


total_loss��@

error_R@/P?

learning_rate_1�O�6o��yI       6%�	I=)^���A�*;


total_loss|�@

error_R�NX?

learning_rate_1�O�6R�AI       6%�	��)^���A�*;


total_loss�@

error_R�O?

learning_rate_1�O�6};q�I       6%�	U�)^���A�*;


total_loss�p�@

error_R��Z?

learning_rate_1�O�6��q'I       6%�	v
*^���A�*;


total_loss���@

error_R��H?

learning_rate_1�O�6'�,I       6%�	�L*^���A�*;


total_lossOv�@

error_R�ha?

learning_rate_1�O�6u�f@I       6%�	,�*^���A�*;


total_loss�&�@

error_RT�X?

learning_rate_1�O�6	��I       6%�	E�*^���A�*;


total_lossy�@

error_R�I?

learning_rate_1�O�6W���I       6%�	W+^���A�*;


total_loss��@

error_R��`?

learning_rate_1�O�6eCvnI       6%�	b_+^���A�*;


total_loss���@

error_R	�V?

learning_rate_1�O�6f���I       6%�	��+^���A�*;


total_loss	xA

error_RR�N?

learning_rate_1�O�6�ǓI       6%�	��+^���A�*;


total_loss��r@

error_R)`N?

learning_rate_1�O�6��s�I       6%�	IP,^���A�*;


total_lossw��@

error_R��D?

learning_rate_1�O�6��gI       6%�	6�,^���A�*;


total_loss�ͮ@

error_R=4S?

learning_rate_1�O�6��k|I       6%�	��,^���A�*;


total_lossݓ@

error_R�(D?

learning_rate_1�O�6S}j�I       6%�	Z--^���A�*;


total_loss�V�@

error_R�OO?

learning_rate_1�O�6����I       6%�	Hr-^���A�*;


total_loss,��@

error_RS�A?

learning_rate_1�O�6��I       6%�	t�-^���A�*;


total_lossUsA

error_R�1<?

learning_rate_1�O�6 �*I       6%�	��-^���A�*;


total_loss!�@

error_R��B?

learning_rate_1�O�6B]�\I       6%�	.C.^���A�*;


total_loss��@

error_R	]6?

learning_rate_1�O�6е��I       6%�	7�.^���A�*;


total_loss60�@

error_RXG?

learning_rate_1�O�6*��%I       6%�	�.^���A�*;


total_lossu5A

error_R��_?

learning_rate_1�O�6G��I       6%�	�/^���A�*;


total_loss��@

error_RT�V?

learning_rate_1�O�6 z{I       6%�	�\/^���A�*;


total_lossV��@

error_R\�U?

learning_rate_1�O�6П�6I       6%�	a�/^���A�*;


total_loss��@

error_R
�L?

learning_rate_1�O�6L�'bI       6%�	�/^���A�*;


total_loss���@

error_Rx�N?

learning_rate_1�O�6�d��I       6%�	�,0^���A�*;


total_loss��@

error_R�I?

learning_rate_1�O�6��zI       6%�	�r0^���A�*;


total_lossُ�@

error_R��X?

learning_rate_1�O�6$&^�I       6%�	:�0^���A�*;


total_loss�K�@

error_RZ=]?

learning_rate_1�O�6o�OI       6%�	�0^���A�*;


total_lossF��@

error_RJ�H?

learning_rate_1�O�6t	�SI       6%�	�81^���A�*;


total_loss�}�@

error_R4�T?

learning_rate_1�O�6l/
�I       6%�	�z1^���A�*;


total_loss��@

error_Ra�Y?

learning_rate_1�O�6��GoI       6%�	/�1^���A�*;


total_lossM��@

error_R�R?

learning_rate_1�O�6	qr�I       6%�	�2^���A�*;


total_loss�U�@

error_R�B?

learning_rate_1�O�6���I       6%�	U2^���A�*;


total_loss�Ň@

error_R��7?

learning_rate_1�O�6jސXI       6%�	��2^���A�*;


total_lossϠ�@

error_RƒK?

learning_rate_1�O�63~A�I       6%�	�2^���A�*;


total_loss�N�@

error_R3H?

learning_rate_1�O�6i�G�I       6%�	"3^���A�*;


total_loss��A

error_R�{B?

learning_rate_1�O�6��E�I       6%�	�h3^���A�*;


total_lossw�@

error_R\?

learning_rate_1�O�6#�z3I       6%�	Ŭ3^���A�*;


total_loss���@

error_R�G?

learning_rate_1�O�6�!3PI       6%�	��3^���A�*;


total_lossa��@

error_R�C`?

learning_rate_1�O�66p5uI       6%�	�04^���A�*;


total_loss�u�@

error_RnI?

learning_rate_1�O�6�MeI       6%�	�t4^���A�*;


total_loss:%�@

error_RxQ?

learning_rate_1�O�6�.�I       6%�	�4^���A�*;


total_losss�@

error_RʹL?

learning_rate_1�O�6� �I       6%�	��4^���A�*;


total_loss���@

error_R��B?

learning_rate_1�O�6���I       6%�	jB5^���A�*;


total_loss���@

error_R��R?

learning_rate_1�O�6��:I       6%�	e�5^���A�*;


total_loss�%�@

error_R��B?

learning_rate_1�O�6�Q �I       6%�	,�5^���A�*;


total_loss���@

error_Rn�R?

learning_rate_1�O�6��C�I       6%�	\6^���A�*;


total_loss�ذ@

error_R��N?

learning_rate_1�O�6uuRI       6%�	f6^���A�*;


total_loss�b�@

error_R\�L?

learning_rate_1�O�6:k��I       6%�	�6^���A�*;


total_loss4t�@

error_R6�d?

learning_rate_1�O�67E$hI       6%�	��6^���A�*;


total_loss��@

error_R �T?

learning_rate_1�O�6;<N�I       6%�	�67^���A�*;


total_lossE��@

error_RL?

learning_rate_1�O�6��.�I       6%�	v{7^���A�*;


total_lossv�A

error_R?M?

learning_rate_1�O�6�Fz�I       6%�	�7^���A�*;


total_loss��@

error_Rs�N?

learning_rate_1�O�6vDt�I       6%�	#8^���A�*;


total_loss�	�@

error_R�8?

learning_rate_1�O�6l�7�I       6%�	|G8^���A�*;


total_loss��A

error_R��`?

learning_rate_1�O�6��R�I       6%�	�8^���A�*;


total_loss���@

error_RS�J?

learning_rate_1�O�6n��;I       6%�	+�8^���A�*;


total_loss�Ǥ@

error_R�S?

learning_rate_1�O�6
7��I       6%�	9^���A�*;


total_loss���@

error_R��T?

learning_rate_1�O�6���*I       6%�	�b9^���A�*;


total_loss�@

error_R{�C?

learning_rate_1�O�6��Z�I       6%�	ѩ9^���A�*;


total_lossQP�@

error_Rj)I?

learning_rate_1�O�6�iv�I       6%�	��9^���A�*;


total_lossW[�@

error_R�re?

learning_rate_1�O�6�U��I       6%�	V5:^���A�*;


total_loss��@

error_Ri�<?

learning_rate_1�O�6T6�I       6%�	�w:^���A�*;


total_loss��@

error_Rx�_?

learning_rate_1�O�6�(Y�I       6%�	!�:^���A�*;


total_lossҗ�@

error_R�K?

learning_rate_1�O�67�#�I       6%�	� ;^���A�*;


total_lossZ�@

error_R�Q?

learning_rate_1�O�6�k��I       6%�	DN;^���A�*;


total_lossy�A

error_RD�Z?

learning_rate_1�O�6M�I       6%�	l�;^���A�*;


total_loss ߙ@

error_R��G?

learning_rate_1�O�6Vs0*I       6%�	2�;^���A�*;


total_loss6�@

error_R�(W?

learning_rate_1�O�6,�rI       6%�	�F<^���A�*;


total_loss(?�@

error_RaS?

learning_rate_1�O�6?��I       6%�	T�<^���A�*;


total_loss�=�@

error_R�V?

learning_rate_1�O�6�P��I       6%�	��<^���A�*;


total_loss#��@

error_RO�I?

learning_rate_1�O�6��_�I       6%�	�,=^���A�*;


total_loss���@

error_R�YG?

learning_rate_1�O�6�蕱I       6%�	�r=^���A�*;


total_loss4��@

error_R΀J?

learning_rate_1�O�6��E�I       6%�	j�=^���A�*;


total_loss�`�@

error_R�M?

learning_rate_1�O�6:�+I       6%�	��=^���A�*;


total_loss@��@

error_R��N?

learning_rate_1�O�6a��YI       6%�	f=>^���A�*;


total_loss�!�@

error_R��L?

learning_rate_1�O�6R���I       6%�	>^���A�*;


total_loss�I�@

error_R�HK?

learning_rate_1�O�6�*�RI       6%�	��>^���A�*;


total_loss$�@

error_R5U?

learning_rate_1�O�6��AI       6%�	9?^���A�*;


total_loss�L�@

error_R<GZ?

learning_rate_1�O�6-+��I       6%�	��?^���A�*;


total_lossc�
A

error_Rc�@?

learning_rate_1�O�6�oPI       6%�	��?^���A�*;


total_loss�r�@

error_Rt�h?

learning_rate_1�O�6#��WI       6%�	 @^���A�*;


total_loss�ԉ@

error_RD�E?

learning_rate_1�O�6��CDI       6%�	�n@^���A�*;


total_loss�A

error_R��T?

learning_rate_1�O�6���sI       6%�	��@^���A�*;


total_lossR�@

error_R:�O?

learning_rate_1�O�6����I       6%�	#A^���A�*;


total_loss��A

error_R�DH?

learning_rate_1�O�6/���I       6%�	dRA^���A�*;


total_loss�O�@

error_Rj�H?

learning_rate_1�O�6ũI       6%�	�A^���A�*;


total_loss�v	A

error_RR�7?

learning_rate_1�O�6g��I       6%�	#�A^���A�*;


total_loss� A

error_R��M?

learning_rate_1�O�6��SI       6%�	�B^���A�*;


total_loss���@

error_R8�T?

learning_rate_1�O�6{I       6%�	cB^���A�*;


total_loss��@

error_RSS?

learning_rate_1�O�6���I       6%�	+�B^���A�*;


total_loss���@

error_R�CP?

learning_rate_1�O�6��1cI       6%�	��B^���A�*;


total_loss*��@

error_R��O?

learning_rate_1�O�6x�R�I       6%�	)?C^���A�*;


total_lossEP�@

error_R�[?

learning_rate_1�O�61v��I       6%�	��C^���A�*;


total_loss�	A

error_R�??

learning_rate_1�O�6�XW�I       6%�	��C^���A�*;


total_loss��@

error_R�sI?

learning_rate_1�O�6f
k�I       6%�	D^���A�*;


total_loss̔�@

error_RSB?

learning_rate_1�O�66wYI       6%�	<_D^���A�*;


total_loss�S�@

error_R��X?

learning_rate_1�O�6{�I       6%�	L�D^���A�*;


total_loss�)�@

error_Rx^J?

learning_rate_1�O�6B��I       6%�	��D^���A�*;


total_loss��@

error_RԖ>?

learning_rate_1�O�6�i��I       6%�	�&E^���A�*;


total_loss�Զ@

error_RX�G?

learning_rate_1�O�6���I       6%�	BjE^���A�*;


total_loss�� A

error_R�Y?

learning_rate_1�O�6$�'I       6%�	��E^���A�*;


total_lossk_	A

error_R �D?

learning_rate_1�O�6`/�I       6%�	C�E^���A�*;


total_loss�h�@

error_Rl�P?

learning_rate_1�O�6Z��.I       6%�	q>F^���A�*;


total_lossM�A

error_R �S?

learning_rate_1�O�6�T�I       6%�	ӂF^���A�*;


total_losss��@

error_R�=\?

learning_rate_1�O�6��4I       6%�	T�F^���A�*;


total_loss ��@

error_Rs,R?

learning_rate_1�O�6[�zI       6%�	�G^���A�*;


total_loss<� A

error_R��F?

learning_rate_1�O�6C!ǙI       6%�	H[G^���A�*;


total_loss�t�@

error_R.jB?

learning_rate_1�O�6֢�yI       6%�	��G^���A�*;


total_loss��@

error_R�TU?

learning_rate_1�O�6��aI       6%�	��G^���A�*;


total_loss��@

error_R7�Y?

learning_rate_1�O�6	
|�I       6%�	$H^���A�*;


total_loss%T�@

error_R��Q?

learning_rate_1�O�6��lI       6%�	�eH^���A�*;


total_loss���@

error_R�OK?

learning_rate_1�O�6,�J<I       6%�	�H^���A�*;


total_loss���@

error_R��P?

learning_rate_1�O�6�SI       6%�	��H^���A�*;


total_loss�k�@

error_R_�(?

learning_rate_1�O�6V_-�I       6%�	�CI^���A�*;


total_loss��@

error_R2�L?

learning_rate_1�O�6wuk^I       6%�	>�I^���A�*;


total_lossx�@

error_R�ca?

learning_rate_1�O�6g��I       6%�	��I^���A�*;


total_lossx)�@

error_R8PK?

learning_rate_1�O�6��I�I       6%�	EJ^���A�*;


total_loss*8�@

error_R��J?

learning_rate_1�O�6��I       6%�		UJ^���A�*;


total_loss�t�@

error_R��]?

learning_rate_1�O�6fR��I       6%�	��J^���A�*;


total_losss��@

error_RxoS?

learning_rate_1�O�6�G4�I       6%�	�J^���A�*;


total_loss;A

error_R\nL?

learning_rate_1�O�6�RI       6%�	L&K^���A�*;


total_loss�c�@

error_R�L7?

learning_rate_1�O�6�j�I       6%�	@qK^���A�*;


total_loss��@

error_R��W?

learning_rate_1�O�6��PI       6%�	�K^���A�*;


total_loss��jA

error_RR�C?

learning_rate_1�O�6n���I       6%�	�L^���A�*;


total_loss�e�@

error_R�bZ?

learning_rate_1�O�6�b�`I       6%�	�jL^���A�*;


total_loss�Ь@

error_R�A?

learning_rate_1�O�6�@I       6%�	�L^���A�*;


total_loss��@

error_R�bJ?

learning_rate_1�O�6.�z�I       6%�	��L^���A�*;


total_loss���@

error_R��O?

learning_rate_1�O�6x�ݺI       6%�	8M^���A�*;


total_loss�/�@

error_R;E?

learning_rate_1�O�6@��xI       6%�	R~M^���A�*;


total_loss�F�@

error_R*�K?

learning_rate_1�O�6�@�I       6%�	��M^���A�*;


total_loss\g�@

error_R��T?

learning_rate_1�O�6�2�I       6%�	pN^���A�*;


total_loss�l�@

error_Rj*I?

learning_rate_1�O�6�.� I       6%�	�LN^���A�*;


total_loss}�@

error_Rۄ]?

learning_rate_1�O�6�lI       6%�	��N^���A�*;


total_loss�P�@

error_R��K?

learning_rate_1�O�6�,d I       6%�	��N^���A�*;


total_loss��@

error_R �L?

learning_rate_1�O�6Oۖ�I       6%�	�O^���A�*;


total_loss`�@

error_R��C?

learning_rate_1�O�6׶R�I       6%�	�XO^���A�*;


total_loss��x@

error_R�HG?

learning_rate_1먷6� ��I       6%�	$�O^���A�*;


total_lossfԸ@

error_RCuH?

learning_rate_1먷6�5H�I       6%�	�O^���A�*;


total_loss�� A

error_R��E?

learning_rate_1먷6_�!I       6%�	%P^���A�*;


total_loss��@

error_Rc&6?

learning_rate_1먷6f��I       6%�	5oP^���A�*;


total_loss�@

error_R��T?

learning_rate_1먷6i�c�I       6%�	N�P^���A�*;


total_lossV�@

error_R8jH?

learning_rate_1먷6�MI       6%�	��P^���A�*;


total_loss_�@

error_R�6?

learning_rate_1먷6�V:I       6%�	�>Q^���A�*;


total_loss��@

error_R��J?

learning_rate_1먷6!�n+I       6%�	�8T^���A�*;


total_loss?"�@

error_R�	>?

learning_rate_1먷6ګ�nI       6%�	}�T^���A�*;


total_lossx�@

error_R�Y?

learning_rate_1먷6h���I       6%�	��T^���A�*;


total_lossX�A

error_RQU?

learning_rate_1먷6U��I       6%�	�"U^���A�*;


total_loss}I�@

error_R$�C?

learning_rate_1먷6�c,2I       6%�	yhU^���A�*;


total_lossK�@

error_R4�Q?

learning_rate_1먷6���I       6%�	I�U^���A�*;


total_losshBh@

error_R(D?

learning_rate_1먷6�25�I       6%�	�U^���A�*;


total_loss#G�@

error_R�N?

learning_rate_1먷6Y�]I       6%�	�9V^���A�*;


total_lossE��@

error_RX;?

learning_rate_1먷6~�UI       6%�	�}V^���A�*;


total_loss��A

error_R�A?

learning_rate_1먷6 ��(I       6%�	e�V^���A�*;


total_loss�jA

error_R��S?

learning_rate_1먷6�>I       6%�	��V^���A�*;


total_lossM��@

error_R dT?

learning_rate_1먷6��I       6%�	^@W^���A�*;


total_lossͫn@

error_R��X?

learning_rate_1먷6�Q��I       6%�	�W^���A�*;


total_lossoc�@

error_R>`?

learning_rate_1먷6�U�iI       6%�	7�W^���A�*;


total_loss�ӥ@

error_R�WS?

learning_rate_1먷6
d�I       6%�	�X^���A�*;


total_loss�.�@

error_R�I?

learning_rate_1먷6�^��I       6%�	�IX^���A�*;


total_loss6��@

error_Rf�M?

learning_rate_1먷6��,CI       6%�	��X^���A�*;


total_loss6�v@

error_R�\?

learning_rate_1먷6 I��I       6%�	3�X^���A�*;


total_loss]�@

error_RiR?

learning_rate_1먷6���I       6%�	Y^���A�*;


total_loss�@

error_RawL?

learning_rate_1먷6*�I       6%�	MYY^���A�*;


total_lossoɸ@

error_R�YN?

learning_rate_1먷6cGTI       6%�	ϛY^���A�*;


total_loss�@

error_R�8?

learning_rate_1먷6/�I       6%�	��Y^���A�*;


total_lossQS�@

error_Rr�>?

learning_rate_1먷6�@V�I       6%�	"Z^���A�*;


total_lossZǲ@

error_R�qR?

learning_rate_1먷6vooI       6%�	dZ^���A�*;


total_lossܛ�@

error_RvZV?

learning_rate_1먷6�9I       6%�	�Z^���A�*;


total_lossҚ@

error_R�r?

learning_rate_1먷6���I       6%�	��Z^���A�*;


total_loss ��@

error_R��Q?

learning_rate_1먷6OԐI       6%�	�-[^���A�*;


total_loss�Z�@

error_R�.c?

learning_rate_1먷6TIDlI       6%�	�s[^���A�*;


total_loss���@

error_R�L:?

learning_rate_1먷6��S�I       6%�	��[^���A�*;


total_loss7��@

error_R��Z?

learning_rate_1먷6�M3vI       6%�	%\^���A�*;


total_loss���@

error_R�rf?

learning_rate_1먷6���I       6%�	�q\^���A�*;


total_loss(�@

error_R�V?

learning_rate_1먷6���I       6%�	��\^���A�*;


total_lossӸ�@

error_Rl&5?

learning_rate_1먷6��I       6%�	��\^���A�*;


total_loss�6�@

error_R,�\?

learning_rate_1먷6�� �I       6%�	B]^���A�*;


total_loss��@

error_Rl[?

learning_rate_1먷63�֭I       6%�	ǎ]^���A�*;


total_loss>�@

error_RJ�J?

learning_rate_1먷6t��I       6%�	�]^���A�*;


total_lossݍ@

error_R�FF?

learning_rate_1먷61ʥ�I       6%�	}^^���A�*;


total_loss�p�@

error_RJ�U?

learning_rate_1먷6�gI       6%�	�l^^���A�*;


total_loss.��@

error_R��b?

learning_rate_1먷6�oA�I       6%�	\�^^���A�*;


total_loss�~@

error_R�iE?

learning_rate_1먷6ٿ_MI       6%�	�_^���A�*;


total_loss�=�@

error_R�hT?

learning_rate_1먷6�$�I       6%�	�o_^���A�*;


total_loss�G�@

error_R�L?

learning_rate_1먷6�:�EI       6%�	ϻ_^���A�*;


total_loss��@

error_R� S?

learning_rate_1먷6����I       6%�	�`^���A�*;


total_loss�D|@

error_RA�H?

learning_rate_1먷6���I       6%�	Iu`^���A�*;


total_lossx۽@

error_R��Y?

learning_rate_1먷6���I       6%�	�`^���A�*;


total_lossC��@

error_R�]?

learning_rate_1먷6��I       6%�	�	a^���A�*;


total_loss@+�@

error_RܝF?

learning_rate_1먷6eŹ=I       6%�	��a^���A�*;


total_loss1��@

error_R��O?

learning_rate_1먷6>���I       6%�	`�a^���A�*;


total_loss��@

error_R͇M?

learning_rate_1먷6�ބ�I       6%�	Mb^���A�*;


total_loss@

error_R]�X?

learning_rate_1먷6�}�I       6%�	��b^���A�*;


total_losst�q@

error_Rf{M?

learning_rate_1먷6���I       6%�	;�b^���A�*;


total_lossw�@

error_R\Z?

learning_rate_1먷6 cTI       6%�	�c^���A�*;


total_loss:��@

error_R�nR?

learning_rate_1먷6O)�YI       6%�	?`c^���A�*;


total_loss5y�@

error_R��L?

learning_rate_1먷6�уI       6%�	��c^���A�*;


total_lossVy�@

error_R�B?

learning_rate_1먷6���I       6%�	�c^���A�*;


total_loss-��@

error_R��??

learning_rate_1먷6])[\I       6%�	�Ad^���A�*;


total_loss\_�@

error_Rf�M?

learning_rate_1먷6��tI       6%�	��d^���A�*;


total_loss�O�@

error_R�dH?

learning_rate_1먷6qsw:I       6%�	��d^���A�*;


total_lossT�@

error_R̣b?

learning_rate_1먷6 �]�I       6%�	F.e^���A�*;


total_lossƕ�@

error_R��B?

learning_rate_1먷6�̌�I       6%�	�ue^���A�*;


total_loss���@

error_R��=?

learning_rate_1먷6vؓ@I       6%�	�e^���A�*;


total_loss�;�@

error_R;VS?

learning_rate_1먷6��^I       6%�	�f^���A�*;


total_lossjN�@

error_R�^J?

learning_rate_1먷6ud�#I       6%�	FHf^���A�*;


total_loss֬�@

error_ROfS?

learning_rate_1먷69)��I       6%�	��f^���A�*;


total_lossȎ�@

error_R�CM?

learning_rate_1먷6�A�I       6%�	�f^���A�*;


total_loss
t�@

error_R�EJ?

learning_rate_1먷6���I       6%�	Rg^���A�*;


total_lossH��@

error_RfCp?

learning_rate_1먷6�wF�I       6%�	jg^���A�*;


total_lossh��@

error_R]�F?

learning_rate_1먷6�3wBI       6%�	�g^���A�*;


total_lossR� A

error_R�K?

learning_rate_1먷6�)��I       6%�	��g^���A�*;


total_loss��@

error_R�UE?

learning_rate_1먷6�Va�I       6%�	pCh^���A�*;


total_loss��@

error_R��L?

learning_rate_1먷6���I       6%�	>�h^���A�*;


total_loss\�@

error_RV�Z?

learning_rate_1먷6��d&I       6%�	 �h^���A�*;


total_lossn�n@

error_R�ZO?

learning_rate_1먷6#�\�I       6%�	�i^���A�*;


total_lossȫ�@

error_Rd�H?

learning_rate_1먷6�Ĝ+I       6%�	�bi^���A�*;


total_loss�nA

error_R�~S?

learning_rate_1먷6R�@I       6%�	&�i^���A�*;


total_loss���@

error_R �R?

learning_rate_1먷6+�y~I       6%�	��i^���A�*;


total_loss�ޏ@

error_R��W?

learning_rate_1먷6@�|�I       6%�	;7j^���A�*;


total_loss2F�@

error_R�S?

learning_rate_1먷6�{!�I       6%�	�{j^���A�*;


total_loss��@

error_R�K?

learning_rate_1먷6)*�I       6%�	]�j^���A�*;


total_lossN�@

error_R��K?

learning_rate_1먷6����I       6%�	k^���A�*;


total_loss�dA

error_R�%T?

learning_rate_1먷6��@�I       6%�	�Ok^���A�*;


total_loss
8�@

error_RF7?

learning_rate_1먷6����I       6%�	��k^���A�*;


total_loss�o�@

error_R��L?

learning_rate_1먷6\@��I       6%�	��k^���A�*;


total_loss��@

error_R�WP?

learning_rate_1먷6鄟LI       6%�	=Wl^���A�*;


total_loss�#�@

error_R��X?

learning_rate_1먷6Ue;I       6%�	^�l^���A�*;


total_loss�z�@

error_RfN?

learning_rate_1먷6��I       6%�	$�l^���A�*;


total_loss�@

error_R�|C?

learning_rate_1먷6i���I       6%�	K0m^���A�*;


total_lossڎ�@

error_Rd�6?

learning_rate_1먷6U�I       6%�	{qm^���A�*;


total_loss1:�@

error_R�e?

learning_rate_1먷6͒��I       6%�	z�m^���A�*;


total_loss;L�@

error_R��G?

learning_rate_1먷6&KY>I       6%�	�m^���A�*;


total_loss#i�@

error_R��S?

learning_rate_1먷6 �4xI       6%�	[Hn^���A�*;


total_lossX֓@

error_R�,X?

learning_rate_1먷6���YI       6%�	͎n^���A�*;


total_loss�K�@

error_R�^?

learning_rate_1먷6[��QI       6%�	W�n^���A�*;


total_lossD�@

error_R�O?

learning_rate_1먷6]���I       6%�	-o^���A�*;


total_loss诡@

error_R�(T?

learning_rate_1먷6Б"I       6%�	*co^���A�*;


total_loss���@

error_R��E?

learning_rate_1먷6S7��I       6%�	�o^���A�*;


total_lossE߲@

error_ROa?

learning_rate_1먷6��I       6%�	��o^���A�*;


total_loss(�@

error_R��D?

learning_rate_1먷6r���I       6%�	R8p^���A�*;


total_loss�Θ@

error_R�y\?

learning_rate_1먷6���I       6%�	�zp^���A�*;


total_loss*G�@

error_R�j@?

learning_rate_1먷6/��AI       6%�	o�p^���A�*;


total_losse�@

error_Rs�]?

learning_rate_1먷6b)�I       6%�	�q^���A�*;


total_loss�҈@

error_R�/O?

learning_rate_1먷6x�I       6%�	Vq^���A�*;


total_loss?��@

error_R�<J?

learning_rate_1먷6�^I       6%�	��q^���A�*;


total_loss#^�@

error_R��U?

learning_rate_1먷6n�5�I       6%�	�q^���A�*;


total_loss��@

error_R�KA?

learning_rate_1먷6���I       6%�	�3r^���A�*;


total_loss�O A

error_RE�H?

learning_rate_1먷6P0h�I       6%�	r^���A�*;


total_loss.�@

error_RҸ^?

learning_rate_1먷6@

�I       6%�	X�r^���A�*;


total_loss�w�@

error_R�P?

learning_rate_1먷67,4�I       6%�	�s^���A�*;


total_loss ��@

error_R��^?

learning_rate_1먷6�e�VI       6%�	}Us^���A�*;


total_loss�D�@

error_RC�A?

learning_rate_1먷6��u�I       6%�	��s^���A�*;


total_loss���@

error_R1�7?

learning_rate_1먷6p�xmI       6%�	��s^���A�*;


total_loss��@

error_R��S?

learning_rate_1먷60> I       6%�	3"t^���A�*;


total_loss(NA

error_R)xT?

learning_rate_1먷6��^UI       6%�	�gt^���A�*;


total_loss[¨@

error_Rv�T?

learning_rate_1먷6֫�I       6%�	Ԫt^���A�*;


total_loss\d�@

error_R�A?

learning_rate_1먷6��I       6%�	��t^���A�*;


total_loss���@

error_Rq	F?

learning_rate_1먷6��CI       6%�	*0u^���A�*;


total_loss�@

error_RO�W?

learning_rate_1먷6T�B�I       6%�	tru^���A�*;


total_lossJ��@

error_RT�X?

learning_rate_1먷6:2��I       6%�	0�u^���A�*;


total_lossC��@

error_R;Z?

learning_rate_1먷6��2I       6%�	}�u^���A�*;


total_loss���@

error_R��M?

learning_rate_1먷6���!I       6%�	�=v^���A�*;


total_loss�A

error_Rs>R?

learning_rate_1먷6���I       6%�	>�v^���A�*;


total_loss�o�@

error_R��F?

learning_rate_1먷6��)�I       6%�	��v^���A�*;


total_lossE��@

error_Rz�I?

learning_rate_1먷6$��I       6%�	}w^���A�*;


total_lossLH�@

error_RZ\d?

learning_rate_1먷6n*uoI       6%�	�Xw^���A�*;


total_loss��@

error_R��\?

learning_rate_1먷6��ٯI       6%�	:�w^���A�*;


total_loss��@

error_R�jV?

learning_rate_1먷6�O��I       6%�	b�w^���A�*;


total_loss4�A

error_RV�V?

learning_rate_1먷6�__�I       6%�	�rx^���A�*;


total_loss��@

error_Ra�^?

learning_rate_1먷6�^�I       6%�	��x^���A�*;


total_loss?�@

error_R��D?

learning_rate_1먷6��I       6%�	�y^���A�*;


total_loss���@

error_R�|C?

learning_rate_1먷6Ɗ-�I       6%�	
^y^���A�*;


total_loss��@

error_R�BQ?

learning_rate_1먷6���I       6%�	��y^���A�*;


total_loss3��@

error_R�VL?

learning_rate_1먷6�1"�I       6%�	�"z^���A�*;


total_loss���@

error_R�6L?

learning_rate_1먷6#�jOI       6%�	�iz^���A�*;


total_lossn��@

error_R��S?

learning_rate_1먷6����I       6%�	��z^���A�*;


total_loss�L�@

error_R?�Y?

learning_rate_1먷6�Ċ�I       6%�	*�z^���A�*;


total_lossF�@

error_RH�S?

learning_rate_1먷6BI9I       6%�	-;{^���A�*;


total_loss�̠@

error_R��K?

learning_rate_1먷6�<0�I       6%�	]�{^���A�*;


total_loss1A

error_R�"I?

learning_rate_1먷6��.�I       6%�	��{^���A�*;


total_lossB �@

error_R�Q?

learning_rate_1먷6�,�I       6%�	�0|^���A�*;


total_loss�]�@

error_R
�L?

learning_rate_1먷6M|wI       6%�	;y|^���A�*;


total_loss���@

error_R�2?

learning_rate_1먷6�њI       6%�	{�|^���A�*;


total_loss36�@

error_Rs�I?

learning_rate_1먷6iC�oI       6%�	n}^���A�*;


total_lossh�@

error_R}ue?

learning_rate_1먷6����I       6%�	�K}^���A�*;


total_loss��@

error_REbX?

learning_rate_1먷6�ۚI       6%�	��}^���A�*;


total_loss���@

error_Ro:`?

learning_rate_1먷6隰�I       6%�	��}^���A�*;


total_lossi��@

error_R�O?

learning_rate_1먷6��I       6%�	�~^���A�*;


total_loss� �@

error_R�V?

learning_rate_1먷6��~I       6%�	�h~^���A�*;


total_lossd%�@

error_RK?

learning_rate_1먷6<�dNI       6%�	'�~^���A�*;


total_losst��@

error_R�BA?

learning_rate_1먷6ϖ�RI       6%�	'�~^���A�*;


total_loss$[�@

error_RO�??

learning_rate_1먷6��]I       6%�	�S^���A�*;


total_loss -�@

error_R��a?

learning_rate_1먷6���I       6%�	'�^���A�*;


total_loss&̇@

error_R
NQ?

learning_rate_1먷6b�B�I       6%�	R�^���A�*;


total_lossF�@

error_R�#J?

learning_rate_1먷6lڙ�I       6%�	F�^���A�*;


total_loss+ԛ@

error_RF�T?

learning_rate_1먷6����I       6%�	u��^���A�*;


total_lossS��@

error_R��B?

learning_rate_1먷6R4��I       6%�	��^���A�*;


total_loss�*�@

error_Rs/<?

learning_rate_1먷6+�I       6%�	4;�^���A�*;


total_loss_�@

error_R�D?

learning_rate_1먷6ě�I       6%�	I�^���A�*;


total_loss��@

error_R�`@?

learning_rate_1먷6%F��I       6%�	Ɂ^���A�*;


total_loss�@

error_R86U?

learning_rate_1먷6Fg��I       6%�	��^���A�*;


total_loss.�@

error_R�P?

learning_rate_1먷6����I       6%�	�]�^���A�*;


total_loss��@

error_R�*T?

learning_rate_1먷6�88I       6%�	���^���A�*;


total_loss4��@

error_R�AW?

learning_rate_1먷6x�3?I       6%�	�^���A�*;


total_loss8��@

error_R�AQ?

learning_rate_1먷69�0sI       6%�	�@�^���A�*;


total_loss��A

error_Rܜ`?

learning_rate_1먷6; �9I       6%�	���^���A�*;


total_loss$��@

error_R҆I?

learning_rate_1먷6h�<I       6%�	P̓^���A�*;


total_losst �@

error_RLA?

learning_rate_1먷6���I       6%�	��^���A�*;


total_lossd��@

error_R�K\?

learning_rate_1먷6AdRI       6%�	}S�^���A�*;


total_loss`%�@

error_R�TM?

learning_rate_1먷6�=�I       6%�	���^���A�*;


total_loss#�@

error_R!X?

learning_rate_1먷6�.>&I       6%�	�^���A�*;


total_loss��@

error_R��??

learning_rate_1먷6�S6I       6%�	�)�^���A�*;


total_lossե�@

error_R�u@?

learning_rate_1먷6Z!+I       6%�	n�^���A�*;


total_loss3!�@

error_RÝY?

learning_rate_1먷6���I       6%�	���^���A�*;


total_lossԣA

error_R{�D?

learning_rate_1먷6��I       6%�	U��^���A�*;


total_loss���@

error_R6�<?

learning_rate_1먷6�6Z�I       6%�	8=�^���A�*;


total_lossO�@

error_R�M?

learning_rate_1먷6���>I       6%�	%��^���A�*;


total_loss���@

error_R��_?

learning_rate_1먷6�s��I       6%�	�Ȇ^���A�*;


total_lossC��@

error_Rf�I?

learning_rate_1먷6�FTnI       6%�	��^���A�*;


total_lossT��@

error_R!CL?

learning_rate_1먷6U��gI       6%�	�Z�^���A�*;


total_loss��@

error_R��J?

learning_rate_1먷6���I       6%�	f��^���A�*;


total_lossy��@

error_R&�M?

learning_rate_1먷6�!�{I       6%�	��^���A�*;


total_loss?�%A

error_RLba?

learning_rate_1먷6��x�I       6%�	�-�^���A�*;


total_lossr�@

error_RHpR?

learning_rate_1먷6�0�~I       6%�	�t�^���A�*;


total_loss��@

error_RO�[?

learning_rate_1먷6�E�I       6%�	���^���A�*;


total_loss��@

error_R�MR?

learning_rate_1먷6#p�6I       6%�	�^���A�*;


total_losss��@

error_Rx�R?

learning_rate_1먷6�F��I       6%�	IE�^���A�*;


total_lossXd�@

error_RפN?

learning_rate_1먷6x�I       6%�	߄�^���A�*;


total_loss�p�@

error_R��J?

learning_rate_1먷6���WI       6%�	�ȉ^���A�*;


total_loss�Ժ@

error_R��\?

learning_rate_1먷6N EI       6%�	�^���A�*;


total_loss�4�@

error_R�pP?

learning_rate_1먷6%���I       6%�	R�^���A�*;


total_lossy�A

error_RM�E?

learning_rate_1먷6�MZI       6%�	#��^���A�*;


total_lossبe@

error_R�D?

learning_rate_1먷6�rX�I       6%�	؊^���A�*;


total_loss�G�@

error_RuS?

learning_rate_1먷6dM�I       6%�	�^���A�*;


total_loss<��@

error_Re�M?

learning_rate_1먷6|�S�I       6%�	5_�^���A�*;


total_loss`��@

error_RZ�R?

learning_rate_1먷6�}I       6%�	ǧ�^���A�*;


total_lossn�A

error_R1P?

learning_rate_1먷6���4I       6%�	<��^���A�*;


total_loss���@

error_R�9I?

learning_rate_1먷6y�"�I       6%�	XQ�^���A�*;


total_loss�%A

error_Ri!O?

learning_rate_1먷6.� 	I       6%�	K��^���A�*;


total_loss�U�@

error_R)�N?

learning_rate_1먷6#᧼I       6%�	�،^���A�*;


total_loss"�@

error_R{1T?

learning_rate_1먷6#ĔgI       6%�	�"�^���A�*;


total_loss
,�@

error_R�B?

learning_rate_1먷6��(�I       6%�	m�^���A�*;


total_loss�%�@

error_R~P?

learning_rate_1먷6��/�I       6%�	?��^���A�*;


total_loss�e�@

error_R;LO?

learning_rate_1먷6�ZI       6%�	A �^���A�*;


total_lossț�@

error_R� V?

learning_rate_1먷6�[��I       6%�	I�^���A�*;


total_loss�_�@

error_R��F?

learning_rate_1먷6���I       6%�	l��^���A�*;


total_loss��@

error_R�U?

learning_rate_1먷6i� �I       6%�	a֎^���A�*;


total_lossg�@

error_R��I?

learning_rate_1먷6n#ϿI       6%�	��^���A�*;


total_loss��@

error_RT�C?

learning_rate_1먷6yG�
I       6%�	a�^���A�*;


total_loss�j�@

error_R�M?

learning_rate_1먷6!��kI       6%�	ũ�^���A�*;


total_lossHG�@

error_R��R?

learning_rate_1먷6鉦�I       6%�	��^���A�*;


total_loss���@

error_RאH?

learning_rate_1먷6�˄I       6%�	39�^���A�*;


total_loss���@

error_R}�K?

learning_rate_1먷6��=@I       6%�	�z�^���A�*;


total_lossq�@

error_RÖK?

learning_rate_1먷6u�4=I       6%�	ϼ�^���A�*;


total_loss�HA

error_R�qM?

learning_rate_1먷6��!�I       6%�	���^���A�*;


total_loss!��@

error_R�rK?

learning_rate_1먷6rx�I       6%�	cD�^���A�*;


total_loss�X�@

error_R�>L?

learning_rate_1먷6�w�I       6%�	c��^���A�*;


total_lossW{�@

error_RTdS?

learning_rate_1먷6I��I       6%�	�֑^���A�*;


total_lossߩA

error_RjEI?

learning_rate_1먷6~�YI       6%�	��^���A�*;


total_lossb�@

error_R�T?

learning_rate_1먷6{��I       6%�	Oi�^���A�*;


total_loss�@

error_RW�T?

learning_rate_1먷6�sQ�I       6%�	���^���A�*;


total_loss��@

error_RW�`?

learning_rate_1먷61g�I       6%�	]��^���A�*;


total_loss�5�@

error_R��Q?

learning_rate_1먷6�R3I       6%�	NH�^���A�*;


total_lossX�@

error_R��??

learning_rate_1먷6z~�kI       6%�	���^���A�*;


total_loss�	�@

error_R�CC?

learning_rate_1먷6�2{�I       6%�	#̓^���A�*;


total_loss�5�@

error_RϘ[?

learning_rate_1먷6�*ѮI       6%�	a�^���A�*;


total_loss�H�@

error_R
�L?

learning_rate_1먷6���I       6%�	/W�^���A�*;


total_loss�E�@

error_RJ�T?

learning_rate_1먷6�f�7I       6%�	N��^���A�*;


total_loss�p�@

error_R�F?

learning_rate_1먷6$ǗwI       6%�	Cޔ^���A�*;


total_loss�L�@

error_RW�R?

learning_rate_1먷6��7�I       6%�	U!�^���A�*;


total_lossҪ�@

error_R�??

learning_rate_1먷6ǅ�HI       6%�	�f�^���A�*;


total_loss��@

error_Rf�J?

learning_rate_1먷6-�zI       6%�	���^���A�*;


total_loss�I�@

error_Rn�??

learning_rate_1먷6�� I       6%�	M�^���A�*;


total_loss�)�@

error_R�E?

learning_rate_1먷6�4=�I       6%�	�7�^���A�*;


total_loss8�@

error_Ra�V?

learning_rate_1먷6���,I       6%�	|��^���A�*;


total_loss[в@

error_R�???

learning_rate_1먷6�#`�I       6%�	sɖ^���A�*;


total_loss7�@

error_R��^?

learning_rate_1먷6��I       6%�	��^���A�*;


total_loss�yA

error_RD�Q?

learning_rate_1먷6�[S�I       6%�	�L�^���A�*;


total_loss�Ѯ@

error_R,�S?

learning_rate_1먷6��I       6%�	珗^���A�*;


total_loss)h�@

error_R��R?

learning_rate_1먷6��hI       6%�	�җ^���A�*;


total_loss��@

error_Rf�L?

learning_rate_1먷6���I       6%�	��^���A�*;


total_lossml�@

error_R�I?

learning_rate_1먷6�)I       6%�	�`�^���A�*;


total_loss��d@

error_R��H?

learning_rate_1먷6�%�I       6%�	���^���A�*;


total_lossz6�@

error_R��W?

learning_rate_1먷6#���I       6%�	�^���A�*;


total_loss��@

error_Rq�[?

learning_rate_1먷6��I       6%�	�-�^���A�*;


total_loss���@

error_R��O?

learning_rate_1먷6�F��I       6%�	s�^���A�*;


total_loss!��@

error_R�rF?

learning_rate_1먷6l��I       6%�	���^���A�*;


total_loss{vA

error_R�'V?

learning_rate_1먷6#Խ�I       6%�	"��^���A�*;


total_lossQKA

error_R�G?

learning_rate_1먷6��8I       6%�	�@�^���A�*;


total_loss�i�@

error_R��D?

learning_rate_1먷646I       6%�	���^���A�*;


total_loss���@

error_R�;P?

learning_rate_1먷60&Y�I       6%�	�Ӛ^���A�*;


total_loss���@

error_RCbF?

learning_rate_1먷6a�DI       6%�	�^���A�*;


total_loss�A

error_R�qa?

learning_rate_1먷6T�6I       6%�	�a�^���A�*;


total_loss���@

error_R�W?

learning_rate_1먷6D�d�I       6%�	���^���A�*;


total_loss�2�@

error_Rm�R?

learning_rate_1먷6� ?�I       6%�	8�^���A�*;


total_lossi��@

error_R�L?

learning_rate_1먷6��&�I       6%�	�W�^���A�*;


total_loss<q�@

error_Rx�Y?

learning_rate_1먷6��ϜI       6%�	�^���A�*;


total_loss�(�@

error_R�J?

learning_rate_1먷6��XI       6%�	'��^���A�*;


total_loss8��@

error_R�}T?

learning_rate_1먷6�G�yI       6%�	�*�^���A�*;


total_loss���@

error_R��^?

learning_rate_1먷6��2EI       6%�	�t�^���A�*;


total_loss��@

error_Rd�R?

learning_rate_1먷6���I       6%�	���^���A�*;


total_loss�wy@

error_R<ME?

learning_rate_1먷6��d�I       6%�	a�^���A�*;


total_loss\}A

error_R��I?

learning_rate_1먷6CtS�I       6%�	mN�^���A�*;


total_loss��@

error_R��I?

learning_rate_1먷6}T�I       6%�	���^���A�*;


total_loss��@

error_R�O?

learning_rate_1먷6%�/�I       6%�	pݞ^���A�*;


total_loss�0�@

error_R%KI?

learning_rate_1먷6�Q��I       6%�	�%�^���A�*;


total_loss;�UA

error_R�J?

learning_rate_1먷6�SI       6%�	�k�^���A�*;


total_loss:~�@

error_R�&T?

learning_rate_1먷67y��I       6%�	�ҟ^���A�*;


total_loss�n�@

error_R��S?

learning_rate_1먷6�VYiI       6%�	��^���A�*;


total_loss�(�@

error_RVf?

learning_rate_1먷6>�G�I       6%�	�c�^���A�*;


total_loss�d�@

error_R�`g?

learning_rate_1먷6K��I       6%�	K͠^���A�*;


total_loss��@

error_R7�P?

learning_rate_1먷6=8��I       6%�	>!�^���A�*;


total_loss�)�@

error_Rm�]?

learning_rate_1먷6����I       6%�	�e�^���A�*;


total_loss��@

error_Rn;J?

learning_rate_1먷6�NYI       6%�	���^���A�*;


total_losslp@

error_Ra�K?

learning_rate_1먷6����I       6%�	&�^���A�*;


total_loss���@

error_R66V?

learning_rate_1먷65%�iI       6%�	T9�^���A�*;


total_loss��@

error_RH�N?

learning_rate_1먷6�_�I       6%�	���^���A�*;


total_loss}2�@

error_R<zB?

learning_rate_1먷6!�F�I       6%�	R̢^���A�*;


total_lossר�@

error_R��;?

learning_rate_1먷6P�C�I       6%�	��^���A�*;


total_loss�k�@

error_R��Q?

learning_rate_1먷6�-�<I       6%�	*[�^���A�*;


total_lossE�@

error_R}9:?

learning_rate_1먷6G�) I       6%�	%��^���A�*;


total_loss�@

error_R��K?

learning_rate_1먷6�b�I       6%�	x�^���A�*;


total_lossf/�@

error_R�mD?

learning_rate_1먷6R���I       6%�	�)�^���A�*;


total_loss�	�@

error_RJ�L?

learning_rate_1먷6Ț�I       6%�	p�^���A�*;


total_loss
Y�@

error_R�;?

learning_rate_1먷6�tI       6%�	k��^���A�*;


total_loss��A

error_RZ�V?

learning_rate_1먷6R�5lI       6%�	^��^���A�*;


total_lossjw�@

error_R�V?

learning_rate_1먷6�&�2I       6%�	�@�^���A�*;


total_loss���@

error_R�;[?

learning_rate_1먷6r�oI       6%�	��^���A�*;


total_loss��@

error_R$�S?

learning_rate_1먷6����I       6%�	�Υ^���A�*;


total_lossJˏ@

error_R��R?

learning_rate_1먷6��I       6%�	?�^���A�*;


total_loss&x�@

error_R�]?

learning_rate_1먷6�Sn�I       6%�	�Z�^���A�*;


total_loss=��@

error_R4�P?

learning_rate_1먷6�*oI       6%�	h��^���A�*;


total_lossXA

error_R��P?

learning_rate_1먷6}�O�I       6%�	��^���A�*;


total_lossS��@

error_RD}Q?

learning_rate_1먷6��FI       6%�	�/�^���A�*;


total_loss0�@

error_R�i]?

learning_rate_1먷6y�yI       6%�	�{�^���A�*;


total_loss���@

error_R�I?

learning_rate_1먷6�wYcI       6%�	=ɧ^���A�*;


total_loss�-�@

error_R�W?

learning_rate_1먷6��R	I       6%�	��^���A�*;


total_loss}ͬ@

error_Ro??

learning_rate_1먷6<�(I       6%�	�R�^���A�*;


total_losse2A

error_Rs2W?

learning_rate_1먷6%�6I       6%�	6��^���A�*;


total_loss��@

error_R<[?

learning_rate_1먷6���II       6%�	lݨ^���A�*;


total_loss.�@

error_R|V?

learning_rate_1먷6s�2�I       6%�	� �^���A�*;


total_loss���@

error_R��T?

learning_rate_1먷6e�|I       6%�	ed�^���A�*;


total_loss��@

error_R�7K?

learning_rate_1먷6�d��I       6%�	B��^���A�*;


total_loss�ݯ@

error_R�T?

learning_rate_1먷6L��}I       6%�	L�^���A�*;


total_lossJÁ@

error_RRTG?

learning_rate_1먷6K!�KI       6%�	�.�^���A�*;


total_loss�/k@

error_RE�O?

learning_rate_1먷6��*VI       6%�	�r�^���A�*;


total_losslɫ@

error_Rs�\?

learning_rate_1먷6�5mI       6%�	\��^���A�*;


total_loss�4�@

error_R�T?

learning_rate_1먷6D��I       6%�	���^���A�*;


total_loss�@

error_RqeY?

learning_rate_1먷6�RZRI       6%�	_;�^���A�*;


total_loss��@

error_R��F?

learning_rate_1먷6hI\�I       6%�	�}�^���A�*;


total_lossQ�&A

error_R��O?

learning_rate_1먷6�C/�I       6%�	e��^���A�*;


total_loss{�@

error_R�S?

learning_rate_1먷6Ǐ�>I       6%�	� �^���A�*;


total_loss��@

error_R��U?

learning_rate_1먷6:�.�I       6%�	$y�^���A�*;


total_loss;��@

error_R�I?

learning_rate_1먷6���ZI       6%�	/��^���A�*;


total_loss/	�@

error_R/�D?

learning_rate_1먷6x&I       6%�	��^���A�*;


total_loss=+�@

error_R�qH?

learning_rate_1먷6��]�I       6%�	�C�^���A�*;


total_loss�#�@

error_R{YT?

learning_rate_1먷6p�>�I       6%�	���^���A�*;


total_loss�C�@

error_R�|B?

learning_rate_1먷6�0�SI       6%�	Gԭ^���A�*;


total_loss�2�@

error_R��V?

learning_rate_1먷6���I       6%�	�"�^���A�*;


total_loss���@

error_Rv�N?

learning_rate_1먷6��Z�I       6%�	�m�^���A�*;


total_loss"}�@

error_R6�V?

learning_rate_1먷6C7�.I       6%�	ζ�^���A�*;


total_loss���@

error_R
�X?

learning_rate_1먷65mI       6%�	P�^���A�*;


total_loss��@

error_R�!T?

learning_rate_1먷6,���I       6%�	�J�^���A�*;


total_loss���@

error_R�zL?

learning_rate_1먷6Qa��I       6%�	;��^���A�*;


total_loss RH@

error_R&NG?

learning_rate_1먷6����I       6%�	{֯^���A�*;


total_loss[E�@

error_R�M?

learning_rate_1먷6�$W�I       6%�	F�^���A�*;


total_lossQyA

error_R�*V?

learning_rate_1먷6��"XI       6%�	�c�^���A�*;


total_loss���@

error_R�>?

learning_rate_1먷6�Y�gI       6%�	���^���A�*;


total_loss�%�@

error_R�q;?

learning_rate_1먷6��ϽI       6%�	���^���A�*;


total_lossߧ@

error_Rs]B?

learning_rate_1먷6dќqI       6%�	A=�^���A�*;


total_loss#��@

error_R�Y?

learning_rate_1먷6b���I       6%�	ၱ^���A�*;


total_loss8��@

error_RNE?

learning_rate_1먷6�d��I       6%�	�ı^���A�*;


total_loss
Υ@

error_R��j?

learning_rate_1먷6S�XWI       6%�	��^���A�*;


total_loss�ƺ@

error_R��`?

learning_rate_1먷6��SI       6%�	OL�^���A�*;


total_loss#W@

error_R;zO?

learning_rate_1먷6כl�I       6%�	���^���A�*;


total_loss��@

error_R/�b?

learning_rate_1먷6��'I       6%�	-�^���A�*;


total_losst5�@

error_RnpL?

learning_rate_1먷6p�-�I       6%�	?.�^���A�*;


total_loss�pe@

error_R;ZL?

learning_rate_1먷6�;o�I       6%�	}v�^���A�*;


total_loss��@

error_R;8J?

learning_rate_1먷6Zң:I       6%�	ٻ�^���A�*;


total_lossʟ�@

error_R��V?

learning_rate_1먷6�s#I       6%�	=��^���A�*;


total_losse�A

error_R� W?

learning_rate_1먷6�ƧI       6%�	f@�^���A�*;


total_loss�0�@

error_R�H?

learning_rate_1먷6��$pI       6%�	���^���A�*;


total_loss�o�@

error_R�dI?

learning_rate_1먷6��U�I       6%�	(Ǵ^���A�*;


total_lossV��@

error_Ri�\?

learning_rate_1먷67�/pI       6%�	��^���A�*;


total_loss�3�@

error_R� K?

learning_rate_1먷6��jrI       6%�	N�^���A�*;


total_loss���@

error_R�=O?

learning_rate_1먷6��'I       6%�	;��^���A�*;


total_loss¨@

error_R3�V?

learning_rate_1먷6z��fI       6%�	ٵ^���A�*;


total_lossf��@

error_R��B?

learning_rate_1먷6�I       6%�	�(�^���A�*;


total_loss�>	A

error_R�QP?

learning_rate_1먷6�5�dI       6%�	|u�^���A�*;


total_loss���@

error_R�c?

learning_rate_1먷6N\ 
I       6%�	���^���A�*;


total_loss�*�@

error_R�sF?

learning_rate_1먷6쇪�I       6%�	@�^���A�*;


total_loss�@�@

error_R��I?

learning_rate_1먷6����I       6%�	(U�^���A�*;


total_loss��@

error_RJ�O?

learning_rate_1먷6��Q�I       6%�	���^���A�*;


total_loss���@

error_RJ{d?

learning_rate_1먷6\��I       6%�	��^���A�*;


total_lossң@

error_R]�??

learning_rate_1먷6H>��I       6%�	�5�^���A�*;


total_loss�1�@

error_R�vT?

learning_rate_1먷6��8I       6%�	ق�^���A�*;


total_lossE�@

error_RvCG?

learning_rate_1먷6N
�.I       6%�	s͸^���A�*;


total_loss��A

error_R�XJ?

learning_rate_1먷6C�T�I       6%�	��^���A�*;


total_loss<��@

error_R�K?

learning_rate_1먷6yQ vI       6%�	�g�^���A�*;


total_loss�g�@

error_R!�H?

learning_rate_1먷6~7I       6%�	ƴ�^���A�*;


total_loss�@

error_R
�E?

learning_rate_1먷6�^�I       6%�	� �^���A�*;


total_loss�;�@

error_R�b?

learning_rate_1먷6�g��I       6%�	dM�^���A�*;


total_loss�ތ@

error_R��M?

learning_rate_1먷6l��I       6%�	u��^���A�*;


total_lossWe�@

error_R�6>?

learning_rate_1먷6��I       6%�	Gغ^���A�*;


total_lossN��@

error_Rԟp?

learning_rate_1먷6�U;I       6%�	1�^���A�*;


total_loss���@

error_R6[T?

learning_rate_1먷6�gnI       6%�	[a�^���A�*;


total_loss���@

error_R�eN?

learning_rate_1먷6 >�I       6%�	`��^���A�*;


total_loss�1�@

error_R�F?

learning_rate_1먷6Y'��I       6%�	���^���A�*;


total_loss�[�@

error_R�HK?

learning_rate_1먷6)���I       6%�	#e�^���A�*;


total_loss�?�@

error_R� O?

learning_rate_1먷6�14I       6%�	���^���A�*;


total_loss�A

error_R� H?

learning_rate_1먷6���0I       6%�	r �^���A�*;


total_lossR�@

error_R�I?

learning_rate_1먷6�� ZI       6%�	G�^���A�*;


total_lossl��@

error_R�.G?

learning_rate_1먷6���|I       6%�	���^���A�*;


total_loss=b�@

error_R!<]?

learning_rate_1먷6��bNI       6%�	4ӽ^���A�*;


total_lossW�@

error_R\VX?

learning_rate_1먷6V�3I       6%�	��^���A�*;


total_loss_ֿ@

error_R)H?

learning_rate_1먷6:Ә�I       6%�	�c�^���A�*;


total_loss���@

error_R6�M?

learning_rate_1먷6_�'ZI       6%�	e��^���A�*;


total_loss�A

error_RjR?

learning_rate_1먷6}��I       6%�	�^���A�*;


total_losshA

error_R�j?

learning_rate_1먷6�hI       6%�	KY�^���A�*;


total_lossʌ�@

error_R�5\?

learning_rate_1먷6�`ieI       6%�	���^���A�*;


total_losst$�@

error_RP>?

learning_rate_1먷6���jI       6%�	��^���A�*;


total_loss4�@

error_R�f?

learning_rate_1먷6�$}�I       6%�	o5�^���A�*;


total_loss�l�@

error_R|�h?

learning_rate_1먷6����I       6%�	�|�^���A�*;


total_losswp�@

error_RT�K?

learning_rate_1먷6��I       6%�	5��^���A�*;


total_loss,�@

error_R&@L?

learning_rate_1먷6k���I       6%�	��^���A�*;


total_lossHv�@

error_R��G?

learning_rate_1먷6��NI       6%�	�S�^���A�*;


total_loss�$�@

error_R� >?

learning_rate_1먷6E��I       6%�	���^���A�*;


total_loss��A

error_R;UQ?

learning_rate_1먷6�7�I       6%�	`��^���A�*;


total_lossm��@

error_RI�R?

learning_rate_1먷6����I       6%�	1(�^���A�*;


total_lossH�@

error_R��O?

learning_rate_1먷66A�I       6%�	�k�^���A�*;


total_lossw��@

error_RMW?

learning_rate_1먷6! �I       6%�	���^���A�*;


total_loss���@

error_R@�G?

learning_rate_1먷6�g�fI       6%�	P�^���A�*;


total_loss(A

error_R
�`?

learning_rate_1먷6<T$~I       6%�	wQ�^���A�*;


total_loss�@

error_R
}W?

learning_rate_1먷6�I       6%�	���^���A�*;


total_loss�x�@

error_R��E?

learning_rate_1먷6�ޔ�I       6%�	o��^���A�*;


total_loss�@�@

error_RR�K?

learning_rate_1먷6JW �I       6%�	�.�^���A�*;


total_loss�T�@

error_R��U?

learning_rate_1먷6��\uI       6%�	�y�^���A�*;


total_loss��@

error_R�KG?

learning_rate_1먷6R�lI       6%�	7��^���A�*;


total_loss��A

error_R
�N?

learning_rate_1먷6��@�I       6%�	J�^���A�*;


total_lossDݶ@

error_R!#L?

learning_rate_1먷6%փ�I       6%�	�P�^���A�*;


total_lossc�@

error_R�W?

learning_rate_1먷6_�I       6%�	P��^���A�*;


total_loss���@

error_R��F?

learning_rate_1먷6�=�5I       6%�	Y��^���A�*;


total_lossp�@

error_RW}A?

learning_rate_1먷6{-&6I       6%�	d�^���A�*;


total_lossF��@

error_R�9T?

learning_rate_1먷6��'�I       6%�	�a�^���A�*;


total_lossR�@

error_R/�I?

learning_rate_1먷6����I       6%�	4��^���A�*;


total_lossq�@

error_R�bO?

learning_rate_1먷6}s^�I       6%�	���^���A�*;


total_loss�A

error_R�GR?

learning_rate_1먷61}�I       6%�	�5�^���A�*;


total_loss,��@

error_R��]?

learning_rate_1먷6D�tI       6%�	P}�^���A�*;


total_lossi~�@

error_RJ�E?

learning_rate_1먷6m"-I       6%�	���^���A�*;


total_lossw��@

error_R�S?

learning_rate_1먷6ty*I       6%�	��^���A�*;


total_loss L�@

error_R��??

learning_rate_1먷6+)�I       6%�	�Q�^���A�*;


total_losskjA

error_RjV?

learning_rate_1먷6���I       6%�	���^���A�*;


total_loss���@

error_R.�G?

learning_rate_1먷6��]I       6%�	���^���A�*;


total_lossF
A

error_R�tI?

learning_rate_1먷6���6I       6%�	n�^���A�*;


total_lossʃ�@

error_R,UH?

learning_rate_1먷6�?��I       6%�	�h�^���A�*;


total_loss�#�@

error_R�*L?

learning_rate_1먷6��I       6%�	���^���A�*;


total_loss�c�@

error_R�S;?

learning_rate_1먷6���I       6%�	� �^���A�*;


total_loss�δ@

error_R�!N?

learning_rate_1먷6�s6�I       6%�	BK�^���A�*;


total_loss��@

error_Rq�T?

learning_rate_1먷6�{,`I       6%�	���^���A�*;


total_loss���@

error_RCP?

learning_rate_1먷6��	4I       6%�	?��^���A�*;


total_loss��@

error_R�+X?

learning_rate_1먷6�BG�I       6%�	��^���A�*;


total_loss2��@

error_R�0e?

learning_rate_1먷6��2I       6%�	�_�^���A�*;


total_loss�p�@

error_R&�W?

learning_rate_1먷6'�I       6%�	?��^���A�*;


total_loss W�@

error_R�dN?

learning_rate_1먷6��1�I       6%�	t�^���A�*;


total_loss�A

error_R.wM?

learning_rate_1먷6���I       6%�	'd�^���A�*;


total_loss�@

error_R$�I?

learning_rate_1먷6���I       6%�	a��^���A�*;


total_loss\Ԅ@

error_R_|O?

learning_rate_1먷6��4<I       6%�	��^���A�*;


total_loss���@

error_R��V?

learning_rate_1먷68b��I       6%�	�=�^���A�*;


total_loss��@

error_R3�Y?

learning_rate_1먷6���I       6%�	���^���A�*;


total_loss�j�@

error_R�1J?

learning_rate_1먷6 ^�I       6%�	��^���A�*;


total_loss0�@

error_R�^R?

learning_rate_1먷6���I       6%�	� �^���A�*;


total_loss�v�@

error_RJ�Y?

learning_rate_1먷6-�1�I       6%�	�f�^���A�*;


total_lossc�@

error_RQ�H?

learning_rate_1먷6Mb��I       6%�	���^���A�*;


total_lossT��@

error_R�W?

learning_rate_1먷6��dII       6%�	���^���A�*;


total_loss�ٴ@

error_Rn!V?

learning_rate_1먷6H{��I       6%�	�F�^���A�*;


total_loss�@

error_R�OU?

learning_rate_1먷6�~��I       6%�	:��^���A�*;


total_loss�̗@

error_R�MW?

learning_rate_1먷6k�PI       6%�	2��^���A�*;


total_loss̅�@

error_Rxi=?

learning_rate_1먷6��S�I       6%�	v�^���A�*;


total_loss��@

error_R�W^?

learning_rate_1먷6��
I       6%�	�]�^���A�*;


total_loss7��@

error_R��D?

learning_rate_1먷6��_�I       6%�	)��^���A�*;


total_loss�I�@

error_RM?

learning_rate_1먷6��G�I       6%�	B��^���A�*;


total_lossl.�@

error_RZ�C?

learning_rate_1먷6*�I�I       6%�	0�^���A�*;


total_lossW@

error_R)�\?

learning_rate_1먷6��ZI       6%�	@t�^���A�*;


total_loss���@

error_R{\?

learning_rate_1먷6z���I       6%�	c��^���A�*;


total_loss��@

error_R�?W?

learning_rate_1먷6�o�I       6%�	��^���A�*;


total_loss8��@

error_RX^U?

learning_rate_1먷6���I       6%�	RP�^���A�*;


total_loss1K�@

error_RsIL?

learning_rate_1먷6�aI       6%�	}��^���A�*;


total_loss�}�@

error_RđG?

learning_rate_1먷6:`�$I       6%�	���^���A�*;


total_loss�S|@

error_R�I?

learning_rate_1먷6�߱�I       6%�	-!�^���A�*;


total_loss���@

error_Rf�J?

learning_rate_1먷6���'I       6%�	Ri�^���A�*;


total_loss�i�@

error_RgY?

learning_rate_1먷6ca�I       6%�	��^���A�*;


total_loss�p�@

error_R�sO?

learning_rate_1먷6��I       6%�	4��^���A�*;


total_lossi�V@

error_R\�V?

learning_rate_1먷6�E�I       6%�	8�^���A�*;


total_loss�͔@

error_R�T?

learning_rate_1먷6a��eI       6%�	�~�^���A�*;


total_loss��	A

error_R�~@?

learning_rate_1먷6�drGI       6%�	 ��^���A�*;


total_loss{�@

error_R�JE?

learning_rate_1먷6���I       6%�	5�^���A�*;


total_lossl��@

error_R�q??

learning_rate_1먷6D!I       6%�	`O�^���A�*;


total_loss*!�@

error_R۰F?

learning_rate_1먷6F���I       6%�	1��^���A�*;


total_losss��@

error_R[�S?

learning_rate_1먷6��`I       6%�	:��^���A�*;


total_loss��@

error_R�Z?

learning_rate_1먷6����I       6%�	�$�^���A�*;


total_loss��@

error_R�l<?

learning_rate_1먷6`L�I       6%�	_m�^���A�*;


total_loss;ǻ@

error_R={E?

learning_rate_1먷6�e��I       6%�	~��^���A�*;


total_loss4��@

error_RҦM?

learning_rate_1먷6�'�I       6%�	
��^���A�*;


total_loss�M�@

error_R.�M?

learning_rate_1먷6{�ٶI       6%�	>�^���A�*;


total_lossK�@

error_R�!9?

learning_rate_1먷6��LhI       6%�	��^���A�*;


total_loss
I�@

error_RE?

learning_rate_1먷6���I       6%�	(��^���A�*;


total_loss�Y�@

error_Rf�H?

learning_rate_1먷6�s��I       6%�	8�^���A�*;


total_loss��@

error_R��L?

learning_rate_1먷69#�}I       6%�	S�^���A�*;


total_loss���@

error_R�<?

learning_rate_1먷6n>s�I       6%�	���^���A�*;


total_loss�n�@

error_R�Q`?

learning_rate_1먷6��ܡI       6%�	!��^���A�*;


total_loss���@

error_R_�8?

learning_rate_1먷6 �0�I       6%�	6"�^���A�*;


total_loss6 �@

error_RH�[?

learning_rate_1먷6UhҌI       6%�	h�^���A�*;


total_loss.B�@

error_R{�Z?

learning_rate_1먷6�(I       6%�	��^���A�*;


total_loss$��@

error_R,J?

learning_rate_1먷6Uc�I       6%�	a��^���A�*;


total_loss�X�@

error_Rbk?

learning_rate_1먷6 �ĞI       6%�	;�^���A�*;


total_loss���@

error_RQ�B?

learning_rate_1먷6�IɞI       6%�	���^���A�*;


total_loss$�`@

error_R�&??

learning_rate_1먷6�aY4I       6%�	���^���A�*;


total_loss��@

error_Rs�b?

learning_rate_1먷6Q�=�I       6%�	��^���A�*;


total_lossx��@

error_REUP?

learning_rate_1먷6L��I       6%�	�^�^���A�*;


total_loss�~�@

error_R�E?

learning_rate_1먷6?NٺI       6%�	��^���A�*;


total_loss
�@

error_R��Z?

learning_rate_1먷6dt̤I       6%�	x��^���A�*;


total_loss\�@

error_ReP?

learning_rate_1먷6�6��I       6%�	�E�^���A�*;


total_loss�'�@

error_R:K?

learning_rate_1먷6�}s!I       6%�	>��^���A�*;


total_loss�S�@

error_R	;R?

learning_rate_1먷6�7�I       6%�	3��^���A�*;


total_lossAt�@

error_R�NY?

learning_rate_1먷6:�&I       6%�	��^���A�*;


total_loss�@

error_R�%H?

learning_rate_1먷6�	�I       6%�	�V�^���A�*;


total_loss���@

error_R(�V?

learning_rate_1먷68�S�I       6%�	���^���A�*;


total_lossm��@

error_R�c^?

learning_rate_1먷6^�-�I       6%�	_��^���A�*;


total_loss�2�@

error_RqI?

learning_rate_1먷6�9g�I       6%�	�"�^���A�*;


total_loss�ߒ@

error_RT�Q?

learning_rate_1먷6C4I       6%�	_p�^���A�*;


total_loss��@

error_Rh>K?

learning_rate_1먷6��I       6%�	��^���A�*;


total_loss�_�@

error_RS�\?

learning_rate_1먷6���WI       6%�	c�^���A�*;


total_loss%�@

error_R�P?

learning_rate_1먷6k��I       6%�	�V�^���A�*;


total_lossr��@

error_R�@?

learning_rate_1먷61?>I       6%�	���^���A�*;


total_loss ~�@

error_RZ�V?

learning_rate_1먷6���I       6%�	U��^���A�*;


total_loss��G@

error_R��S?

learning_rate_1먷6���I       6%�	�&�^���A�*;


total_loss1T�@

error_Rl�Y?

learning_rate_1먷6���DI       6%�	�p�^���A�*;


total_loss��@

error_R��I?

learning_rate_1먷6���<I       6%�	e��^���A�*;


total_loss#��@

error_RfO?

learning_rate_1먷6�ۮ�I       6%�	l	�^���A�*;


total_lossvWA

error_RώR?

learning_rate_1먷6:l��I       6%�	aO�^���A�*;


total_lossA�@

error_R�[?

learning_rate_1먷6 fDI       6%�	Ȓ�^���A�*;


total_loss	��@

error_RW�J?

learning_rate_1먷6��u�I       6%�	���^���A�*;


total_loss���@

error_R�
Y?

learning_rate_1먷6y�I       6%�	-�^���A�*;


total_loss��u@

error_R�wJ?

learning_rate_1먷6�:e�I       6%�	�c�^���A�*;


total_lossz<�@

error_R��`?

learning_rate_1먷6RR�|I       6%�	z��^���A�*;


total_lossT֏@

error_R�X?

learning_rate_1먷6h�(I       6%�	���^���A�*;


total_loss��@

error_R��J?

learning_rate_1먷6��I       6%�	�5�^���A�*;


total_loss��@

error_R<oK?

learning_rate_1먷6��K�I       6%�	)~�^���A�*;


total_loss�[�@

error_R�4U?

learning_rate_1먷6,���I       6%�	��^���A�*;


total_loss��@

error_R�??

learning_rate_1먷6�!l�I       6%�	�^���A�*;


total_loss�@

error_R�N?

learning_rate_1먷6K�H�I       6%�	yI�^���A�*;


total_loss�M�@

error_R�F?

learning_rate_1먷6��I       6%�	S��^���A�*;


total_loss)��@

error_R��I?

learning_rate_1먷6Dz�I       6%�	���^���A�*;


total_loss�A

error_R��_?

learning_rate_1먷6���AI       6%�	� �^���A�*;


total_loss�j�@

error_R\$S?

learning_rate_1먷6ş_�I       6%�	�e�^���A�*;


total_lossC$�@

error_R��I?

learning_rate_1먷6C�ݷI       6%�	a��^���A�*;


total_loss:y�@

error_R�vE?

learning_rate_1먷6]ĭ�I       6%�	� �^���A�*;


total_lossֈ�@

error_R�W?

learning_rate_1먷6���I       6%�	�G�^���A�*;


total_loss.��@

error_R�U?

learning_rate_1먷6:[-�I       6%�	
��^���A�*;


total_lossc&�@

error_RqJ?

learning_rate_1먷6q^QmI       6%�	k��^���A�*;


total_loss?��@

error_R��O?

learning_rate_1먷6{�_I       6%�	�^���A�*;


total_loss$�@

error_R=�Q?

learning_rate_1먷6`8�I       6%�	zU�^���A�*;


total_loss8T�@

error_R�gj?

learning_rate_1먷6y��I       6%�	��^���A�*;


total_loss�@

error_RzO?

learning_rate_1먷6c
�sI       6%�	j��^���A�*;


total_loss�m�@

error_R�UP?

learning_rate_1먷65��qI       6%�	� �^���A�*;


total_loss���@

error_R3�X?

learning_rate_1먷6���I       6%�	c�^���A�*;


total_loss�a�@

error_R[B?

learning_rate_1먷6M˟5I       6%�	ק�^���A�*;


total_lossOb�@

error_R6�j?

learning_rate_1먷6`nu�I       6%�	���^���A�*;


total_loss���@

error_R�P?

learning_rate_1먷6r1��I       6%�	N-�^���A�*;


total_loss���@

error_R�HM?

learning_rate_1먷6�z��I       6%�	7r�^���A�*;


total_lossv��@

error_R�B?

learning_rate_1먷6��2\I       6%�	\��^���A�*;


total_lossϜ@

error_R�YO?

learning_rate_1먷6¯̫I       6%�	���^���A�*;


total_loss�ٲ@

error_R�
[?

learning_rate_1먷6�FI       6%�	l?�^���A�*;


total_lossD�A

error_Rd�J?

learning_rate_1먷6��I       6%�	���^���A�*;


total_loss΄@

error_R�W?

learning_rate_1먷6��lI       6%�	���^���A�*;


total_loss�j�@

error_RE.C?

learning_rate_1먷66�\�I       6%�	��^���A�*;


total_loss�5A

error_R��H?

learning_rate_1먷6%���I       6%�	\Q�^���A�*;


total_lossC6A

error_Rr�\?

learning_rate_1먷6�2!I       6%�	-��^���A�*;


total_loss���@

error_R�M?

learning_rate_1먷6��zI       6%�	���^���A�*;


total_lossR=�@

error_R6.E?

learning_rate_1먷6�.��I       6%�	H@�^���A�*;


total_loss7��@

error_R�xX?

learning_rate_1먷6�eoI       6%�	��^���A�*;


total_loss7�@

error_RN�??

learning_rate_1먷6�|�3I       6%�	I��^���A�*;


total_loss<��@

error_R8�C?

learning_rate_1먷6�R��I       6%�	G�^���A�*;


total_loss��@

error_R��d?

learning_rate_1먷6�#1�I       6%�	�Q�^���A�*;


total_losse�@

error_R81G?

learning_rate_1먷6I�98I       6%�	��^���A�*;


total_lossڵ�@

error_R�Y?

learning_rate_1먷6V��KI       6%�	���^���A�*;


total_loss}��@

error_R�U:?

learning_rate_1먷6�J I       6%�	��^���A�*;


total_lossd�@

error_R�X?

learning_rate_1먷6�� I       6%�	]d�^���A�*;


total_loss3k�@

error_R�J?

learning_rate_1먷6(hW1I       6%�	ȫ�^���A�*;


total_loss��@

error_R�`?

learning_rate_1먷6��I       6%�	v��^���A�*;


total_lossc��@

error_R�L?

learning_rate_1먷6l��I       6%�	F�^���A�*;


total_loss�^A

error_R��N?

learning_rate_1먷6����I       6%�	-��^���A�*;


total_loss���@

error_REc?

learning_rate_1먷6rC��I       6%�	���^���A�*;


total_loss8v�@

error_R CQ?

learning_rate_1먷6kǯ+I       6%�	��^���A�*;


total_lossO��@

error_R%�>?

learning_rate_1먷6�/�I       6%�	]�^���A�*;


total_loss���@

error_R�eS?

learning_rate_1먷6�l#I       6%�	.��^���A�*;


total_losso��@

error_R��J?

learning_rate_1먷6��[I       6%�	���^���A�*;


total_loss���@

error_R��R?

learning_rate_1먷6�rŃI       6%�	�,�^���A�*;


total_lossds�@

error_R{�E?

learning_rate_1먷6�d�*I       6%�	r�^���A�*;


total_lossL�@

error_RL�V?

learning_rate_1먷6	xYI       6%�	��^���A�*;


total_losslM�@

error_R@�J?

learning_rate_1먷6�\cqI       6%�	��^���A�*;


total_loss%@A

error_R3BM?

learning_rate_1먷6�Z7�I       6%�	HM�^���A�*;


total_lossۜ@

error_R�5>?

learning_rate_1먷6,���I       6%�	E��^���A�*;


total_loss�Ӝ@

error_RTqM?

learning_rate_1먷66��DI       6%�	P��^���A�*;


total_loss���@

error_R�E?

learning_rate_1먷6��d�I       6%�	C �^���A�*;


total_loss_{�@

error_R
8Q?

learning_rate_1먷6zGOI       6%�	�e�^���A�*;


total_loss,�@

error_R��[?

learning_rate_1먷6#J�I       6%�	��^���A�*;


total_lossi� A

error_R��K?

learning_rate_1먷6��Y�I       6%�	���^���A�*;


total_loss��@

error_Re�F?

learning_rate_1먷6�6��I       6%�	l+�^���A�*;


total_lossc>w@

error_R�(Z?

learning_rate_1먷6�8,5I       6%�	�m�^���A�*;


total_loss酒@

error_R&�L?

learning_rate_1먷6Z�I       6%�	���^���A�*;


total_loss�C{@

error_R�yS?

learning_rate_1먷6���I       6%�	s��^���A�*;


total_lossY�@

error_R�F?

learning_rate_1먷6>=6I       6%�	�P�^���A�*;


total_loss[��@

error_R�,G?

learning_rate_1먷6^�I$I       6%�	+��^���A�*;


total_loss2��@

error_R7?

learning_rate_1먷6�O��I       6%�	���^���A�*;


total_loss�@

error_R4Z?

learning_rate_1먷6�HI       6%�	m6�^���A�*;


total_loss�4�@

error_RW�P?

learning_rate_1먷6�bH�I       6%�	�|�^���A�*;


total_loss��@

error_Rd�N?

learning_rate_1먷6�V�I       6%�	���^���A�*;


total_loss��@

error_R�W?

learning_rate_1먷6Sx�OI       6%�	��^���A�*;


total_loss�5�@

error_RhQ?

learning_rate_1먷6���I       6%�	6Z�^���A�*;


total_lossWN�@

error_R�;N?

learning_rate_1먷6���/I       6%�	��^���A�*;


total_loss֋�@

error_R�4X?

learning_rate_1먷6s�,I       6%�	��^���A�*;


total_loss�^�@

error_R��g?

learning_rate_1먷6���I       6%�	�\�^���A�*;


total_loss�)�@

error_RM`?

learning_rate_1먷6��qI       6%�	5��^���A�*;


total_loss�	A

error_R��E?

learning_rate_1먷6��%I       6%�	G �^���A�*;


total_losstU=A

error_R�qX?

learning_rate_1먷6���I       6%�	U^�^���A�*;


total_loss4&A

error_RJ�M?

learning_rate_1먷6?��I       6%�	���^���A�*;


total_lossZ��@

error_R7h>?

learning_rate_1먷6��p�I       6%�	��^���A�*;


total_lossSފ@

error_R�&O?

learning_rate_1먷6< qQI       6%�	�C�^���A�*;


total_loss�5�@

error_R�E?

learning_rate_1먷6���I       6%�	h��^���A�*;


total_loss$��@

error_R��@?

learning_rate_1먷6)�Z�I       6%�	"��^���A�*;


total_lossਘ@

error_R)�H?

learning_rate_1먷6����I       6%�	�A�^���A�*;


total_loss�؇@

error_R1�`?

learning_rate_1먷6G��I       6%�	ê�^���A�*;


total_lossS��@

error_R��W?

learning_rate_1먷6ߜfI       6%�	���^���A�*;


total_loss��@

error_RcJ?

learning_rate_1먷6/��%I       6%�	Sf�^���A�*;


total_loss
��@

error_R��L?

learning_rate_1먷6Y�I       6%�	$��^���A�*;


total_loss,ڱ@

error_R�;?

learning_rate_1먷6�)��I       6%�	��^���A�*;


total_losss)K@

error_Rd�<?

learning_rate_1먷6���I       6%�	�c�^���A�*;


total_loss�Ǡ@

error_R��H?

learning_rate_1먷6H�vI       6%�	i��^���A�*;


total_loss��@

error_R�Y??

learning_rate_1먷6쥀�I       6%�	��^���A�*;


total_lossR��@

error_Rn�7?

learning_rate_1먷6�ۜcI       6%�		n�^���A�*;


total_loss���@

error_RiU?

learning_rate_1먷6C�"�I       6%�	���^���A�*;


total_loss���@

error_RܱE?

learning_rate_1먷6K"�TI       6%�	�,�^���A�*;


total_loss�*A

error_R��b?

learning_rate_1먷6Wz�I       6%�	8w�^���A�*;


total_lossHŻ@

error_R�;?

learning_rate_1먷6��8�I       6%�	 ��^���A�*;


total_loss��@

error_R;O?

learning_rate_1먷6���I       6%�	�0 _���A�*;


total_loss��@

error_R�??

learning_rate_1먷6����I       6%�	e| _���A�*;


total_loss(�@

error_R(3T?

learning_rate_1먷6I�<I       6%�	�� _���A�*;


total_lossx��@

error_R�R?

learning_rate_1먷6��ͬI       6%�	�D_���A�*;


total_loss�~�@

error_R��D?

learning_rate_1먷6ڤ�oI       6%�	��_���A�*;


total_loss8(A

error_RT�N?

learning_rate_1먷62��I       6%�	(�_���A�*;


total_loss;�A

error_R��M?

learning_rate_1먷6S}�I       6%�	�N_���A�*;


total_loss�W�@

error_R��H?

learning_rate_1먷6����I       6%�	V�_���A�*;


total_loss�k�@

error_R-�1?

learning_rate_1먷6q�U"I       6%�	��_���A�*;


total_loss���@

error_Re�H?

learning_rate_1먷6 'b�I       6%�	�9_���A�*;


total_lossɰ�@

error_RNP?

learning_rate_1먷6>��2I       6%�	�{_���A�*;


total_lossh��@

error_R�sF?

learning_rate_1먷6WHTDI       6%�	"�_���A�*;


total_loss���@

error_RO�T?

learning_rate_1먷6��I       6%�	�_���A�*;


total_lossA��@

error_R8�>?

learning_rate_1먷6i��7I       6%�	N_���A�*;


total_loss`��@

error_RX[?

learning_rate_1먷6�}RI       6%�	��_���A�*;


total_loss&�@

error_R��D?

learning_rate_1먷6?��I       6%�	��_���A�*;


total_lossǶ@

error_R�Rf?

learning_rate_1먷6+`��I       6%�	._���A�*;


total_loss���@

error_R�:W?

learning_rate_1먷6��I       6%�	fe_���A�*;


total_lossfX�@

error_R�AG?

learning_rate_1먷6�1�tI       6%�	�_���A�*;


total_lossz�@

error_R��I?

learning_rate_1먷6�bˌI       6%�	+�_���A�*;


total_loss�.�@

error_RXL?

learning_rate_1먷6�u�[I       6%�	ZC_���A�*;


total_lossR�@

error_R�LH?

learning_rate_1먷6�.|�I       6%�	I�_���A�*;


total_lossTd�@

error_R��N?

learning_rate_1먷6)��GI       6%�	��_���A�*;


total_loss��@

error_R|eC?

learning_rate_1먷6��aI       6%�	$G_���A�*;


total_loss��@

error_Rܻ=?

learning_rate_1먷6[�F�I       6%�	��_���A�*;


total_lossu)�@

error_Rf=L?

learning_rate_1먷6,�a�I       6%�	 �_���A�*;


total_loss���@

error_R�O_?

learning_rate_1먷67 V�I       6%�	6_���A�*;


total_lossb|�@

error_R)B?

learning_rate_1먷6�`I       6%�	��_���A�*;


total_lossL��@

error_R��^?

learning_rate_1먷6�6u]I       6%�	O�_���A�*;


total_loss�k�@

error_R�5\?

learning_rate_1먷6r���I       6%�	�4	_���A�*;


total_lossfe�@

error_R3�R?

learning_rate_1먷6$��I       6%�	fy	_���A�*;


total_loss�@

error_R=f8?

learning_rate_1먷6u|��I       6%�	#�	_���A�*;


total_loss��@

error_R�#L?

learning_rate_1먷6�+��I       6%�	 "
_���A�*;


total_loss�c�@

error_RR�=?

learning_rate_1먷6fK��I       6%�	Pm
_���A�*;


total_loss5�@

error_R��J?

learning_rate_1먷6)���I       6%�	ɷ
_���A�*;


total_lossXa�@

error_R�ER?

learning_rate_1먷6-�TI       6%�	� _���A�*;


total_loss<UA

error_R�??

learning_rate_1먷6HdE�I       6%�	~e_���A�*;


total_loss{F�@

error_RdMO?

learning_rate_1먷6���lI       6%�	�_���A�*;


total_loss�
r@

error_R��U?

learning_rate_1먷61�*I       6%�	��_���A�*;


total_loss�_�@

error_R��W?

learning_rate_1먷67H`�I       6%�	�_���A�*;


total_lossyd@

error_R#@?

learning_rate_1먷6���HI       6%�	��_���A�*;


total_loss�� A

error_R0P?

learning_rate_1먷6]´,I       6%�	V_���A�*;


total_loss��@

error_R��D?

learning_rate_1먷6�B8�I       6%�	�U_���A�*;


total_loss�_�@

error_R��Z?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss�@

error_R�'3?

learning_rate_1먷6X���I       6%�	�_���A�*;


total_loss��@

error_RlF?

learning_rate_1먷6���I       6%�	<%_���A�*;


total_loss�c�@

error_R9?

learning_rate_1먷6�o��I       6%�	�i_���A�*;


total_loss��@

error_RMvE?

learning_rate_1먷60��BI       6%�	e�_���A�*;


total_loss+�@

error_R@T?

learning_rate_1먷6WCa�I       6%�	k�_���A�*;


total_loss-��@

error_R�uK?

learning_rate_1먷6h�'�I       6%�	�9_���A�*;


total_loss��@

error_RD?

learning_rate_1먷6���I       6%�	�}_���A�*;


total_lossZh�@

error_R�#A?

learning_rate_1먷6]�W�I       6%�	f�_���A�*;


total_loss�<�@

error_R��L?

learning_rate_1먷6' �I       6%�	�_���A�*;


total_loss�@

error_R�[?

learning_rate_1먷6�R=5I       6%�	�M_���A�*;


total_loss�7�@

error_Rs�5?

learning_rate_1먷6נ��I       6%�	�_���A�*;


total_loss4��@

error_R�XV?

learning_rate_1먷6�C˙I       6%�	��_���A�*;


total_loss���@

error_R�(L?

learning_rate_1먷6l�iI       6%�	�_���A�*;


total_losss|�@

error_R�O[?

learning_rate_1먷6��I       6%�	�[_���A�*;


total_loss��@

error_RF�L?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss���@

error_R�cD?

learning_rate_1먷6��PI       6%�	��_���A�*;


total_loss��@

error_R8TI?

learning_rate_1먷6&*�TI       6%�	�&_���A�*;


total_loss���@

error_RR�J?

learning_rate_1먷6ZቑI       6%�	�k_���A�*;


total_lossO��@

error_R�G?

learning_rate_1먷6���I       6%�	M�_���A�*;


total_losst��@

error_R��Z?

learning_rate_1먷66�oI       6%�	&�_���A�*;


total_loss˽@

error_Re�U?

learning_rate_1먷6�cjI       6%�	�7_���A�*;


total_loss�@

error_R�>?

learning_rate_1먷6�� 2I       6%�	�y_���A�*;


total_loss�Q�@

error_RJdU?

learning_rate_1먷6;���I       6%�	.�_���A�*;


total_loss�^YA

error_R �A?

learning_rate_1먷6�kL�I       6%�	a�_���A�*;


total_lossw5�@

error_RjV?

learning_rate_1먷6��&[I       6%�	�F_���A�*;


total_loss��@

error_R7QH?

learning_rate_1먷6�97ZI       6%�	g�_���A�*;


total_lossC3�@

error_R&^?

learning_rate_1먷6p^{&I       6%�	��_���A�*;


total_loss�r�@

error_R�V?

learning_rate_1먷6Y��I       6%�		"_���A�*;


total_loss_7�@

error_R)�M?

learning_rate_1먷6
l�I       6%�	�k_���A�*;


total_loss#�A

error_RÜZ?

learning_rate_1먷6�~�I       6%�	>�_���A�*;


total_loss���@

error_R�H?

learning_rate_1먷6}"��I       6%�	\_���A�*;


total_loss�;�@

error_Ra�X?

learning_rate_1먷6O���I       6%�	�h_���A�*;


total_lossoq�@

error_R��]?

learning_rate_1먷6����I       6%�	,�_���A�*;


total_lossɰ�@

error_R�eR?

learning_rate_1먷6{U�BI       6%�	��_���A�*;


total_loss�Ƌ@

error_R@�W?

learning_rate_1먷6�;��I       6%�	"5_���A�*;


total_loss��@

error_RɄS?

learning_rate_1먷6?1=I       6%�	�z_���A�*;


total_loss�f�@

error_R�\\?

learning_rate_1먷6�Ja�I       6%�	Ŀ_���A�*;


total_loss<�@

error_R*�S?

learning_rate_1먷6�vbI       6%�	e_���A�*;


total_loss(ܟ@

error_R�OO?

learning_rate_1먷6n_�I       6%�	�E_���A�*;


total_loss��@

error_R�8?

learning_rate_1먷6oX^I       6%�	�_���A�*;


total_lossw�@

error_R�-b?

learning_rate_1먷6.��I       6%�	��_���A�*;


total_lossz��@

error_R��C?

learning_rate_1먷6���I       6%�	0_���A�*;


total_lossY�@

error_R?�J?

learning_rate_1먷6�d��I       6%�	�T_���A�*;


total_loss(=�@

error_R΄Q?

learning_rate_1먷6ޫ.QI       6%�	��_���A�*;


total_loss3/�@

error_R�P?

learning_rate_1먷6�`��I       6%�	?�_���A�*;


total_loss�7�@

error_Rd�L?

learning_rate_1먷6 g��I       6%�	|/_���A�*;


total_loss4v�@

error_R�xT?

learning_rate_1먷6%j�I       6%�	�s_���A�*;


total_lossd��@

error_R�Q?

learning_rate_1먷6<��fI       6%�	��_���A�*;


total_loss���@

error_R�P?

learning_rate_1먷6���I       6%�	��_���A�*;


total_lossM*�@

error_R�vI?

learning_rate_1먷6n;	I       6%�	�<_���A�*;


total_loss)��@

error_R�P;?

learning_rate_1먷6���)I       6%�	�_���A�*;


total_loss�.�@

error_R��C?

learning_rate_1먷6�\uI       6%�	��_���A�*;


total_lossӷ@

error_R��L?

learning_rate_1먷6WH6I       6%�	�_���A�*;


total_loss���@

error_R��c?

learning_rate_1먷6<V|�I       6%�	�j_���A�*;


total_loss	ȴ@

error_R�!J?

learning_rate_1먷6W���I       6%�	��_���A�*;


total_loss�&�@

error_RAa?

learning_rate_1먷6�	9�I       6%�	�_���A�*;


total_loss솩@

error_R1�W?

learning_rate_1먷6y���I       6%�	[V_���A�*;


total_loss���@

error_R�U?

learning_rate_1먷6��I       6%�	$�_���A�*;


total_loss���@

error_R��S?

learning_rate_1먷6�ELI       6%�	�_���A�*;


total_lossf7�@

error_R�Rj?

learning_rate_1먷6�ݍ�I       6%�	8%_���A�*;


total_lossȬ�@

error_RwB?

learning_rate_1먷6� T1I       6%�	�l_���A�*;


total_loss)��@

error_RN?

learning_rate_1먷6G2��I       6%�	H�_���A�*;


total_loss"�@

error_R�|R?

learning_rate_1먷6�
V}I       6%�	��_���A�*;


total_loss]��@

error_R�dL?

learning_rate_1먷6v��pI       6%�	�<_���A�*;


total_loss�;�@

error_R}�S?

learning_rate_1먷6��XI       6%�	#�_���A�*;


total_loss�o�@

error_R$�N?

learning_rate_1먷6�&(�I       6%�	J�_���A�*;


total_loss� �@

error_R��U?

learning_rate_1먷6�y�I       6%�	� _���A�*;


total_loss�"�@

error_R�PB?

learning_rate_1먷6J:�I       6%�	fP _���A�*;


total_loss�6�@

error_RZqI?

learning_rate_1먷6L���I       6%�	N� _���A�*;


total_loss�[�@

error_Rf�R?

learning_rate_1먷6�}�I       6%�	f� _���A�*;


total_loss�"�@

error_R��O?

learning_rate_1먷6���I       6%�	� !_���A�*;


total_lossڣ�@

error_R@�Y?

learning_rate_1먷6���~I       6%�	h!_���A�*;


total_lossݷ�@

error_RJcP?

learning_rate_1먷6�,I       6%�	��!_���A�*;


total_loss�2�@

error_R�B?

learning_rate_1먷6K��I       6%�	"_���A�*;


total_loss�I�@

error_R8�9?

learning_rate_1먷6�$MI       6%�	U"_���A�*;


total_loss��@

error_R��>?

learning_rate_1먷6� �4I       6%�	�"_���A�*;


total_loss#A

error_R]Oa?

learning_rate_1먷6��0�I       6%�	��"_���A�*;


total_loss쐩@

error_R�bA?

learning_rate_1먷6U�I       6%�	� #_���A�*;


total_loss:�@

error_R�^H?

learning_rate_1먷6L�6I       6%�	�d#_���A�*;


total_loss��@

error_RB?

learning_rate_1먷6� �I       6%�	8�#_���A�*;


total_loss�T
A

error_R��7?

learning_rate_1먷6T���I       6%�	N�#_���A�*;


total_lossQ�@

error_R��a?

learning_rate_1먷6	���I       6%�	�2$_���A�*;


total_loss���@

error_R�O?

learning_rate_1먷6=B�TI       6%�	�t$_���A�*;


total_lossd,�@

error_RřP?

learning_rate_1먷68Н/I       6%�	��$_���A�*;


total_loss���@

error_R�v@?

learning_rate_1먷6��\I       6%�	.�$_���A�*;


total_loss�m�@

error_R��I?

learning_rate_1먷6�II       6%�	�D%_���A�*;


total_loss�@�@

error_R=2P?

learning_rate_1먷6��I       6%�	�%_���A�*;


total_loss�?�@

error_R�6R?

learning_rate_1먷6D�kI       6%�	��%_���A�*;


total_lossr�@

error_R��f?

learning_rate_1먷6���I       6%�	8&_���A�*;


total_loss(k�@

error_RJ\?

learning_rate_1먷6��9�I       6%�	�e&_���A�*;


total_lossa��@

error_Ri�d?

learning_rate_1먷6-�I       6%�	��&_���A�*;


total_loss��@

error_R2�H?

learning_rate_1먷6�Q�I       6%�	��&_���A�*;


total_loss�@

error_R�-G?

learning_rate_1먷6�`qtI       6%�	�2'_���A�*;


total_loss��@

error_RŽF?

learning_rate_1먷6���BI       6%�	�v'_���A�*;


total_loss�9�@

error_RJ�b?

learning_rate_1먷6�k@�I       6%�	��'_���A�*;


total_loss���@

error_R�U?

learning_rate_1먷6���3I       6%�	p�'_���A�*;


total_loss���@

error_R]�X?

learning_rate_1먷6r�j^I       6%�	L?(_���A�*;


total_loss�E�@

error_Ri�>?

learning_rate_1먷6�3`BI       6%�	�(_���A�*;


total_loss)��@

error_R�F?

learning_rate_1먷6Cd�I       6%�	&�(_���A�*;


total_loss�
�@

error_R�d?

learning_rate_1먷6(�wI       6%�	5)_���A�*;


total_loss:��@

error_R��X?

learning_rate_1먷6��B�I       6%�	"W)_���A�*;


total_loss:�@

error_RNT?

learning_rate_1먷6���I       6%�	��)_���A�*;


total_lossFu�@

error_Rv�R?

learning_rate_1먷6�X׳I       6%�	�)_���A�*;


total_loss���@

error_R�D?

learning_rate_1먷6�x�vI       6%�	�$*_���A�*;


total_lossF�@

error_R��I?

learning_rate_1먷6�~<I       6%�	�h*_���A�*;


total_loss�&�@

error_R _?

learning_rate_1먷67C�I       6%�	�*_���A�*;


total_loss�}�@

error_R�<?

learning_rate_1먷6{�b$I       6%�	��*_���A�*;


total_loss���@

error_R3�P?

learning_rate_1먷6㩦\I       6%�	�-+_���A�*;


total_loss��@

error_R;�N?

learning_rate_1먷6vէI       6%�	jq+_���A�*;


total_loss�V�@

error_R��N?

learning_rate_1먷6���TI       6%�	��+_���A�*;


total_loss���@

error_R��<?

learning_rate_1먷6P+A�I       6%�	��+_���A�*;


total_loss��A

error_RŅT?

learning_rate_1먷6#/l�I       6%�	�b,_���A�*;


total_loss�@

error_R6t^?

learning_rate_1먷6���QI       6%�	�,_���A�*;


total_loss��@

error_R��D?

learning_rate_1먷6o|�EI       6%�	��,_���A�*;


total_loss�@

error_Rf�L?

learning_rate_1먷6���)I       6%�	/-_���A�*;


total_loss`˩@

error_R)�P?

learning_rate_1먷6텪�I       6%�	u-_���A�*;


total_losseg�@

error_R,�S?

learning_rate_1먷6�GL@I       6%�	@�-_���A�*;


total_loss�"�@

error_RIP?

learning_rate_1먷6Ā�I       6%�	�._���A�*;


total_lossq�@

error_R��R?

learning_rate_1먷6�U�*I       6%�	/Q._���A�*;


total_lossO^�@

error_R��??

learning_rate_1먷6��`I       6%�	`�._���A�*;


total_loss���@

error_R��W?

learning_rate_1먷6��IHI       6%�	��._���A�*;


total_lossʵ@

error_R�'A?

learning_rate_1먷6�[��I       6%�	�-/_���A�*;


total_loss�P�@

error_R\�K?

learning_rate_1먷6!ZgI       6%�	�y/_���A�*;


total_lossx�@

error_R�Q?

learning_rate_1먷6ݚ�bI       6%�	E�/_���A�*;


total_lossϷ�@

error_R�T?

learning_rate_1먷6_I       6%�	0_���A�*;


total_loss�s�@

error_R��;?

learning_rate_1먷6Zc�I       6%�	eL0_���A�*;


total_loss�K�@

error_R�UZ?

learning_rate_1먷6��MI       6%�	\�0_���A�*;


total_loss�X�@

error_R`�N?

learning_rate_1먷6R�QcI       6%�	��0_���A�*;


total_loss�_�@

error_RM�X?

learning_rate_1먷6FW�CI       6%�	o1_���A�*;


total_loss5A

error_R�N?

learning_rate_1먷6�۸I       6%�	}b1_���A�*;


total_loss�K�@

error_RtSW?

learning_rate_1먷6 V��I       6%�	�1_���A�*;


total_loss�o�@

error_R6U?

learning_rate_1먷6��ԩI       6%�	�1_���A�*;


total_lossLP�@

error_RM�L?

learning_rate_1먷6W��I       6%�	3:2_���A�*;


total_loss	G�@

error_RE�U?

learning_rate_1먷6[,t�I       6%�	�2_���A�*;


total_loss���@

error_R�B?

learning_rate_1먷6��a(I       6%�	��2_���A�*;


total_loss��@

error_Rؤ7?

learning_rate_1먷6����I       6%�	�3_���A�*;


total_lossN��@

error_RbU?

learning_rate_1먷6���KI       6%�	,T3_���A�*;


total_loss�@

error_R��N?

learning_rate_1먷6�[#.I       6%�	�3_���A�*;


total_loss��@

error_R�X\?

learning_rate_1먷6�ǵ%I       6%�	��3_���A�*;


total_lossi�G@

error_R�R?

learning_rate_1먷6��I       6%�	"4_���A�*;


total_loss;�@

error_Rn�O?

learning_rate_1먷6��lI       6%�	�f4_���A�*;


total_loss��H@

error_R��G?

learning_rate_1먷6|A+I       6%�	�4_���A�*;


total_lossV��@

error_R��K?

learning_rate_1먷6�[>I       6%�	��4_���A�*;


total_lossx��@

error_R�6K?

learning_rate_1먷6r��I       6%�	�15_���A�*;


total_lossʭ�@

error_R�2R?

learning_rate_1먷6^��I       6%�	�5_���A�*;


total_loss��"A

error_R]�O?

learning_rate_1먷6\|�I       6%�	s�5_���A�*;


total_loss]��@

error_R��<?

learning_rate_1먷6�|��I       6%�	6_���A�*;


total_loss���@

error_R��W?

learning_rate_1먷6B�ߣI       6%�	e_6_���A�*;


total_lossȜ@

error_Rv{F?

learning_rate_1먷6 ��I       6%�	�6_���A�*;


total_loss���@

error_Ra�H?

learning_rate_1먷6�Z%I       6%�	R�6_���A�*;


total_loss�qA

error_R��S?

learning_rate_1먷6$���I       6%�	�,7_���A�*;


total_lossx��@

error_R��4?

learning_rate_1먷6�(f*I       6%�	�n7_���A�*;


total_loss�~�@

error_R��>?

learning_rate_1먷6�iw�I       6%�	��7_���A�*;


total_loss��@

error_Rq�F?

learning_rate_1먷6Q	�I       6%�	��7_���A�*;


total_loss|K�@

error_R�SF?

learning_rate_1먷6T+��I       6%�	uA8_���A�*;


total_loss6��@

error_R�SO?

learning_rate_1먷6�Ks3I       6%�	�8_���A�*;


total_loss4[A

error_R�1M?

learning_rate_1먷6�6}PI       6%�	=�8_���A�*;


total_loss���@

error_R�Ui?

learning_rate_1먷6s5��I       6%�	f9_���A�*;


total_loss�Y�@

error_R=C?

learning_rate_1먷6����I       6%�	�9_���A�*;


total_loss��@

error_R��H?

learning_rate_1먷6配�I       6%�	|�9_���A�*;


total_loss�Ю@

error_R��A?

learning_rate_1먷6قdI       6%�	�5:_���A�*;


total_lossi��@

error_R�sF?

learning_rate_1먷6�@iI       6%�	�z:_���A�*;


total_loss�Q�@

error_R(e??

learning_rate_1먷6�8�I       6%�	ؾ:_���A�*;


total_loss1�@

error_R��U?

learning_rate_1먷6���BI       6%�	�;_���A�*;


total_losse��@

error_R�w@?

learning_rate_1먷6�R��I       6%�	6J;_���A�*;


total_lossQ��@

error_R?�T?

learning_rate_1먷6�;�I       6%�	Q�;_���A�*;


total_lossĺ�@

error_R,�T?

learning_rate_1먷6ʠ(DI       6%�	��;_���A�*;


total_loss)<�@

error_R�5S?

learning_rate_1먷6Z�ͺI       6%�	5<_���A�*;


total_loss.��@

error_Re�I?

learning_rate_1먷6��iI       6%�	3�<_���A�*;


total_loss;��@

error_R��U?

learning_rate_1먷6��1aI       6%�	B�<_���A�*;


total_lossq��@

error_R�dJ?

learning_rate_1먷6[���I       6%�	�=_���A�*;


total_loss�E�@

error_R��R?

learning_rate_1먷6�f�I       6%�	Z\=_���A�*;


total_loss�X�@

error_R�4W?

learning_rate_1먷6���UI       6%�	��=_���A�*;


total_loss�Ȱ@

error_R��D?

learning_rate_1먷6%̰�I       6%�	.�=_���A�*;


total_loss���@

error_R(,M?

learning_rate_1먷6�'1�I       6%�	�$>_���A�*;


total_loss��}@

error_R;�J?

learning_rate_1먷6�Z)eI       6%�	_m>_���A�*;


total_lossAZ�@

error_R�U?

learning_rate_1먷6�0<�I       6%�	@�>_���A�*;


total_loss|x A

error_RC{A?

learning_rate_1먷6�J �I       6%�	'�>_���A�*;


total_lossVq�@

error_R�.R?

learning_rate_1먷6�VcI       6%�	j8?_���A�*;


total_loss{@

error_R�9>?

learning_rate_1먷6���I       6%�	�~?_���A�*;


total_loss��v@

error_RL=?

learning_rate_1먷6��I       6%�	��?_���A�*;


total_loss8�@

error_RXY?

learning_rate_1먷6d�S�I       6%�	�
@_���A�*;


total_loss(c�@

error_Re`?

learning_rate_1먷6]�SI       6%�	�O@_���A�*;


total_lossoU�@

error_R3\G?

learning_rate_1먷6GUǒI       6%�	��@_���A�*;


total_loss�ˈ@

error_Rò_?

learning_rate_1먷6�@n$I       6%�	��@_���A�*;


total_lossh1�@

error_R��F?

learning_rate_1먷6�M�I       6%�	jA_���A�*;


total_lossg�@

error_R�K?

learning_rate_1먷6et�eI       6%�	�ZA_���A�*;


total_losstb�@

error_R_�L?

learning_rate_1먷6�cGBI       6%�	��A_���A�*;


total_loss��@

error_R}A?

learning_rate_1먷6f�)I       6%�	��A_���A�*;


total_lossR��@

error_R
P?

learning_rate_1먷6��7I       6%�	O(B_���A�*;


total_loss�"�@

error_R�G?

learning_rate_1먷6��CI       6%�	�kB_���A�*;


total_loss��T@

error_R3�O?

learning_rate_1먷6��Y`I       6%�	o�B_���A�*;


total_loss���@

error_RX�Q?

learning_rate_1먷6oo�iI       6%�	��B_���A�*;


total_loss�ۤ@

error_R�MQ?

learning_rate_1먷6��`�I       6%�	�IC_���A�*;


total_loss���@

error_R��@?

learning_rate_1먷6N!�I       6%�	W�C_���A�*;


total_lossF=�@

error_Rd�S?

learning_rate_1먷6��q�I       6%�	�D_���A�*;


total_loss�_�@

error_RL�M?

learning_rate_1먷6ka�DI       6%�	_D_���A�*;


total_losswV�@

error_R$vO?

learning_rate_1먷6�� �I       6%�	��D_���A�*;


total_losso�@

error_R�cV?

learning_rate_1먷6��1I       6%�	�D_���A�*;


total_lossH�@

error_Rq�N?

learning_rate_1먷6�7��I       6%�	HAE_���A�*;


total_lossC��@

error_Rc�;?

learning_rate_1먷6eu@�I       6%�	��E_���A�*;


total_loss:�s@

error_R�U?

learning_rate_1먷6��I       6%�	��E_���A�*;


total_loss�ΰ@

error_R��I?

learning_rate_1먷6��=I       6%�	F_���A�*;


total_lossz��@

error_R��X?

learning_rate_1먷6^3��I       6%�	�PF_���A�*;


total_lossl5�@

error_R��G?

learning_rate_1먷6�1R�I       6%�	єF_���A�*;


total_loss=��@

error_R֨E?

learning_rate_1먷6Ls@I       6%�	��F_���A�*;


total_loss��@

error_R,K?

learning_rate_1먷6�F��I       6%�	DG_���A�*;


total_loss� �@

error_R}BN?

learning_rate_1먷6e�z�I       6%�	}aG_���A�*;


total_lossFۆ@

error_RJ�T?

learning_rate_1먷6��^XI       6%�	f�G_���A�*;


total_lossԌ�@

error_R�g?

learning_rate_1먷6�S�JI       6%�	�G_���A�*;


total_lossf��@

error_R��R?

learning_rate_1먷6�#p�I       6%�	�3H_���A�*;


total_loss���@

error_R2hR?

learning_rate_1먷6���I       6%�	p�H_���A�*;


total_loss���@

error_R�^?

learning_rate_1먷6w�z�I       6%�	�H_���A�*;


total_loss��@

error_RSA?

learning_rate_1먷6|v�I       6%�	�I_���A�*;


total_loss�Z�@

error_R;FG?

learning_rate_1먷6=��I       6%�	]I_���A�*;


total_loss!��@

error_RiV?

learning_rate_1먷6���I       6%�	:�I_���A�*;


total_loss�7�@

error_Rf�H?

learning_rate_1먷6�5I       6%�	��I_���A�*;


total_lossc�@

error_R=�M?

learning_rate_1먷6p��I       6%�	�#J_���A�*;


total_loss�U[@

error_R$t;?

learning_rate_1먷6��W�I       6%�	fJ_���A�*;


total_loss���@

error_R�ML?

learning_rate_1먷6.�4I       6%�	�J_���A�*;


total_lossa�@

error_RdFO?

learning_rate_1먷6�@rI       6%�	��J_���A�*;


total_loss���@

error_R �Q?

learning_rate_1먷6�u�I       6%�	�.K_���A�*;


total_loss߃�@

error_R�6?

learning_rate_1먷63L)I       6%�	cqK_���A�*;


total_loss�5�@

error_R��F?

learning_rate_1먷6�Y�I       6%�	D�K_���A�*;


total_loss��@

error_R$�V?

learning_rate_1먷6#�2�I       6%�	f
L_���A�*;


total_lossrZ�@

error_R,FZ?

learning_rate_1먷6��EXI       6%�	hL_���A�*;


total_loss���@

error_Rd�R?

learning_rate_1먷6�H�\I       6%�	0�L_���A�*;


total_loss��@

error_R��??

learning_rate_1먷6/��I       6%�	<�L_���A�*;


total_loss�-�@

error_RX?

learning_rate_1먷6^(%I       6%�	 .M_���A�*;


total_lossA�@

error_Rn6Z?

learning_rate_1먷6L�_eI       6%�	=qM_���A�*;


total_lossk�@

error_R�r??

learning_rate_1먷6z[�I       6%�	��M_���A�*;


total_loss-�@

error_R�O??

learning_rate_1먷6�(�I       6%�	��M_���A�*;


total_lossڦ�@

error_RW
K?

learning_rate_1먷6z˲]I       6%�	GBN_���A�*;


total_loss�P�@

error_R�@?

learning_rate_1먷6��"I       6%�	�N_���A�*;


total_loss};�@

error_R�gU?

learning_rate_1먷6/�U�I       6%�	o�N_���A�*;


total_loss���@

error_R�[?

learning_rate_1먷6�Կ4I       6%�	�O_���A�*;


total_loss�Ƶ@

error_R�L?

learning_rate_1먷6c�(�I       6%�	�UO_���A�*;


total_lossT��@

error_RX\P?

learning_rate_1먷6���KI       6%�	��O_���A�*;


total_loss�G�@

error_ROp`?

learning_rate_1먷6��?I       6%�	��O_���A�*;


total_loss��@

error_R��F?

learning_rate_1먷6$�>�I       6%�	�#P_���A�*;


total_loss���@

error_RL5]?

learning_rate_1먷6���=I       6%�	dP_���A�*;


total_loss䮨@

error_R��K?

learning_rate_1먷6q�2ZI       6%�	>�P_���A�*;


total_loss���@

error_R�rE?

learning_rate_1먷6����I       6%�	��P_���A�*;


total_lossX��@

error_R�\?

learning_rate_1먷6M8��I       6%�	,AQ_���A�*;


total_loss!?T@

error_R�}F?

learning_rate_1먷6���BI       6%�	6�Q_���A�*;


total_loss.��@

error_R�O?

learning_rate_1먷6��)_I       6%�	��Q_���A�*;


total_lossu��@

error_RܥG?

learning_rate_1먷6zMK`I       6%�	�R_���A�*;


total_loss���@

error_RV?

learning_rate_1먷6e�]�I       6%�	�VR_���A�*;


total_loss���@

error_R��P?

learning_rate_1먷63�w�I       6%�	V�R_���A�*;


total_loss,ͬ@

error_R3�T?

learning_rate_1먷6�װI       6%�	R�R_���A�*;


total_lossF^�@

error_Rq�^?

learning_rate_1먷6'p�I       6%�	�5S_���A�*;


total_loss9VA

error_R$d0?

learning_rate_1먷6���I       6%�	_S_���A�*;


total_lossڧ�@

error_R��G?

learning_rate_1먷6$�?I       6%�	6�S_���A�*;


total_lossj��@

error_R�vW?

learning_rate_1먷6A�@I       6%�	1T_���A�*;


total_lossV��@

error_R�D?

learning_rate_1먷69�1PI       6%�	ST_���A�*;


total_loss9�@

error_RJ�K?

learning_rate_1먷6�=��I       6%�	�T_���A�*;


total_lossW�@

error_Rq�Y?

learning_rate_1먷6{�D-I       6%�	:�T_���A�*;


total_loss���@

error_R38I?

learning_rate_1먷6y�^�I       6%�	x!U_���A�*;


total_loss��C@

error_R��H?

learning_rate_1먷6V���I       6%�	`eU_���A�*;


total_loss�P�@

error_R(i9?

learning_rate_1먷6�1��I       6%�	:�U_���A�*;


total_loss
X�@

error_Rڧ1?

learning_rate_1먷6����I       6%�	��U_���A�*;


total_loss�g�@

error_RԊP?

learning_rate_1먷6+4I       6%�	�GV_���A�*;


total_lossf�c@

error_R�#P?

learning_rate_1먷6I0�I       6%�		�V_���A�*;


total_loss�%A

error_R
�P?

learning_rate_1먷6�&+I       6%�	I�V_���A�*;


total_loss�۵@

error_R(K?

learning_rate_1먷6	Ɓ�I       6%�	�W_���A�*;


total_loss���@

error_R!H??

learning_rate_1먷6(�B�I       6%�	�UW_���A�*;


total_lossa2�@

error_Rl�W?

learning_rate_1먷6B�c�I       6%�	a�W_���A�*;


total_loss���@

error_R�3P?

learning_rate_1먷6Ex@OI       6%�	��W_���A�*;


total_loss� �@

error_R�V?

learning_rate_1먷6�I       6%�	�#X_���A�*;


total_loss<��@

error_R�US?

learning_rate_1먷6n�oI       6%�	�mX_���A�*;


total_lossl1�@

error_R�8N?

learning_rate_1먷6ww��I       6%�	��X_���A�*;


total_loss�;�@

error_R�Y?

learning_rate_1먷6ݛ2I       6%�	�X_���A�*;


total_loss�n�@

error_R�kZ?

learning_rate_1먷6���CI       6%�	�>Y_���A�*;


total_loss ��@

error_R�8F?

learning_rate_1먷6�7.
I       6%�	
�Y_���A�*;


total_loss��@

error_R�D[?

learning_rate_1먷6���I       6%�	��Y_���A�*;


total_loss�@

error_R�~K?

learning_rate_1먷6H/	wI       6%�	�Z_���A�*;


total_loss%��@

error_RdaV?

learning_rate_1먷6z�}I       6%�	�^Z_���A�*;


total_lossxO�@

error_R�O?

learning_rate_1먷6
��&I       6%�	9�Z_���A�*;


total_loss��@

error_R�S?

learning_rate_1먷6.c2�I       6%�	/�Z_���A�*;


total_loss�vA

error_R��J?

learning_rate_1먷6+x[I       6%�	�*[_���A�*;


total_loss�O�@

error_R�O;?

learning_rate_1먷6��6rI       6%�	{l[_���A�*;


total_loss���@

error_R1�J?

learning_rate_1먷6�c�I       6%�	+�[_���A�*;


total_lossN��@

error_R�M?

learning_rate_1먷6��iI       6%�	��[_���A�*;


total_loss��@

error_RI�H?

learning_rate_1먷6����I       6%�	�Y\_���A�*;


total_loss�
�@

error_RqAS?

learning_rate_1먷6,�I       6%�	�\_���A�*;


total_lossN��@

error_R�R?

learning_rate_1먷6�:dI       6%�	�\_���A�*;


total_loss�5c@

error_R�<D?

learning_rate_1먷6���SI       6%�	�)]_���A�*;


total_loss�y�@

error_R�w(?

learning_rate_1먷6��XI       6%�	n]_���A�*;


total_lossq�@

error_R�H?

learning_rate_1먷6��$I       6%�	��]_���A�*;


total_lossz��@

error_R�gU?

learning_rate_1먷6��I       6%�	>�]_���A�*;


total_loss]Ձ@

error_R/�@?

learning_rate_1먷6�喴I       6%�	�@^_���A�*;


total_loss�չ@

error_Re�A?

learning_rate_1먷6�$��I       6%�	:�^_���A�*;


total_loss��@

error_R��D?

learning_rate_1먷6�;�cI       6%�	��^_���A�*;


total_loss��v@

error_R�B?

learning_rate_1먷6GqI       6%�	�__���A�*;


total_loss=C�@

error_RùR?

learning_rate_1먷6�(�AI       6%�	�___���A�*;


total_loss���@

error_R�cY?

learning_rate_1먷6�`�I       6%�	��__���A�*;


total_loss"�@

error_R]�X?

learning_rate_1먷6��uI       6%�	�	`_���A�*;


total_loss�P�@

error_R�cR?

learning_rate_1먷6�6DFI       6%�	1N`_���A�*;


total_loss��@

error_RVR?

learning_rate_1먷6���I       6%�	��`_���A�*;


total_loss���@

error_R�r8?

learning_rate_1먷6OH2|I       6%�	a_���A�*;


total_loss���@

error_R�_]?

learning_rate_1먷6���I       6%�	�Ja_���A�*;


total_loss�a�@

error_R��Q?

learning_rate_1먷6�� ,I       6%�	؎a_���A�*;


total_loss���@

error_Rc�T?

learning_rate_1먷6P�HdI       6%�	C�a_���A�*;


total_loss���@

error_RdD<?

learning_rate_1먷6ꭎ�I       6%�	�b_���A�*;


total_loss���@

error_R��@?

learning_rate_1먷6g��I       6%�	�ab_���A�*;


total_loss=H�@

error_RT\G?

learning_rate_1먷6f��I       6%�	>�b_���A�*;


total_loss���@

error_R��F?

learning_rate_1먷6���I       6%�	m�b_���A�*;


total_loss�¾@

error_Rd�c?

learning_rate_1먷6�X�)I       6%�	�<c_���A�*;


total_loss�t�@

error_R@�D?

learning_rate_1먷6�T �I       6%�	]�c_���A�*;


total_loss	״@

error_R-U?

learning_rate_1먷6��[fI       6%�	��c_���A�*;


total_loss<�@

error_RjK?

learning_rate_1먷6��R�I       6%�	d_���A�*;


total_loss���@

error_R.Q]?

learning_rate_1먷6��HyI       6%�	�Md_���A�*;


total_loss��w@

error_R{�G?

learning_rate_1먷6��p2I       6%�	��d_���A�*;


total_loss�8�@

error_R�AD?

learning_rate_1먷6�(�<I       6%�	D�d_���A�*;


total_lossj��@

error_R!�>?

learning_rate_1먷6�E~�I       6%�	�e_���A�*;


total_loss�E�@

error_R($O?

learning_rate_1먷60��I       6%�	�_e_���A�*;


total_loss�W�@

error_R�R?

learning_rate_1먷6�_II       6%�	�e_���A�*;


total_lossmɌ@

error_R��O?

learning_rate_1먷6*��I       6%�	�e_���A�*;


total_loss���@

error_R� R?

learning_rate_1먷6Β�I       6%�	�*f_���A�*;


total_loss_�A

error_R�c`?

learning_rate_1먷6C�^I       6%�	Oof_���A�*;


total_loss�<�@

error_R��S?

learning_rate_1먷6r{V�I       6%�	��f_���A�*;


total_loss�Й@

error_R�G?

learning_rate_1먷6��pI       6%�	��f_���A�*;


total_loss�?�@

error_R�a?

learning_rate_1먷6�/RI       6%�	�?g_���A�*;


total_loss�<�@

error_R��J?

learning_rate_1먷6�=E/I       6%�	΃g_���A�*;


total_loss	�g@

error_R��B?

learning_rate_1먷6�:�I       6%�	��g_���A�*;


total_loss�Y�@

error_RM�L?

learning_rate_1먷6�}J�I       6%�	�h_���A�*;


total_loss���@

error_R��P?

learning_rate_1먷6ѐBI       6%�	�Xh_���A�*;


total_loss�@

error_R͔K?

learning_rate_1먷6���"I       6%�	�h_���A�*;


total_loss�� A

error_Rm�Q?

learning_rate_1먷6���I       6%�	��h_���A�*;


total_lossS��@

error_R�T?

learning_rate_1먷6|iG�I       6%�	�(i_���A�*;


total_loss!s�@

error_R1�O?

learning_rate_1먷6>�P�I       6%�	mi_���A�*;


total_loss��@

error_R��A?

learning_rate_1먷6�*�-I       6%�	ձi_���A�*;


total_lossL�@

error_REZ?

learning_rate_1먷6��_�I       6%�	�i_���A�*;


total_loss�0�@

error_Rߏa?

learning_rate_1먷6:�!�I       6%�	H:j_���A�*;


total_loss�Ϡ@

error_R �I?

learning_rate_1먷6.vE�I       6%�	}j_���A�*;


total_loss�+A

error_RVCU?

learning_rate_1먷6Ғ��I       6%�	��j_���A�*;


total_lossr۩@

error_R�xO?

learning_rate_1먷6l�1nI       6%�	hk_���A�*;


total_loss���@

error_Rf�R?

learning_rate_1먷6/힂I       6%�	�Ik_���A�*;


total_loss@�@

error_R�dD?

learning_rate_1먷6�<�I       6%�	ْk_���A�*;


total_loss)��@

error_RW�S?

learning_rate_1먷6,���I       6%�	��k_���A�*;


total_loss`��@

error_R�lj?

learning_rate_1먷6:�~tI       6%�	�@l_���A�*;


total_lossA�@

error_R<Q?

learning_rate_1먷6���I       6%�	]�l_���A�*;


total_loss��}@

error_R�V?

learning_rate_1먷6��6gI       6%�	.�l_���A�*;


total_loss��@

error_Rl;?

learning_rate_1먷6Ɔ�nI       6%�	Sm_���A�*;


total_loss��@

error_R�<H?

learning_rate_1먷68_�I       6%�	�_m_���A�*;


total_loss�{�@

error_R��=?

learning_rate_1먷6i�eI       6%�	v�m_���A�*;


total_loss-��@

error_R�T?

learning_rate_1먷69�GI       6%�	B�m_���A�*;


total_loss�@

error_R��8?

learning_rate_1먷6}�\�I       6%�	/n_���A�*;


total_loss�̀@

error_RE�7?

learning_rate_1먷6���I       6%�	3sn_���A�*;


total_loss��@

error_R_oM?

learning_rate_1먷6rJ�sI       6%�	׶n_���A�*;


total_loss.v�@

error_RQ�`?

learning_rate_1먷6�$I�I       6%�	��n_���A�*;


total_loss�)�@

error_R1ES?

learning_rate_1먷6��
�I       6%�	?o_���A�*;


total_loss�$�@

error_R�V?

learning_rate_1먷6���I       6%�	(�o_���A�*;


total_loss-ѣ@

error_RC�Q?

learning_rate_1먷6���I       6%�	�r_���A�*;


total_lossV�@

error_R� D?

learning_rate_1먷6�0��I       6%�	�r_���A�*;


total_loss��A

error_R36W?

learning_rate_1먷6b6d�I       6%�	s_���A�*;


total_loss��@

error_R��:?

learning_rate_1먷6�0<�I       6%�	�js_���A�*;


total_lossRP�@

error_R�K?

learning_rate_1먷6-�pI       6%�	 �s_���A�*;


total_loss!jA

error_Ri�I?

learning_rate_1먷6EV�I       6%�	��s_���A�*;


total_lossϥ�@

error_RO�T?

learning_rate_1먷6Zn�0I       6%�	�=t_���A�*;


total_lossJ0�@

error_R,Z?

learning_rate_1먷6��5 I       6%�	Y�t_���A�*;


total_loss�Þ@

error_RïZ?

learning_rate_1먷6���I       6%�	��t_���A�*;


total_loss�'�@

error_R��[?

learning_rate_1먷6"jEpI       6%�	�u_���A�*;


total_lossc��@

error_R�U?

learning_rate_1먷6�c#�I       6%�	`u_���A�*;


total_loss��@

error_R�Q^?

learning_rate_1먷6=­wI       6%�	ޣu_���A�*;


total_loss�ĩ@

error_R�Z=?

learning_rate_1먷6ѶI       6%�	��u_���A�*;


total_loss���@

error_R.�I?

learning_rate_1먷6Lߵ�I       6%�	><v_���A�*;


total_lossI��@

error_R&V?

learning_rate_1먷6V�I       6%�	4�v_���A�*;


total_lossQ�@

error_R��T?

learning_rate_1먷6�T�I       6%�	��v_���A�*;


total_losse�@

error_RR1Z?

learning_rate_1먷6~q5�I       6%�	w_���A�*;


total_lossJf�@

error_R&f??

learning_rate_1먷6!lnHI       6%�	�Pw_���A�*;


total_lossr�@

error_R�l\?

learning_rate_1먷6���I       6%�	j�w_���A�*;


total_loss���@

error_R�E?

learning_rate_1먷6J���I       6%�	}�w_���A�*;


total_lossᑠ@

error_R�CJ?

learning_rate_1먷6M zAI       6%�	u#x_���A�*;


total_loss���@

error_R�r>?

learning_rate_1먷6/�	#I       6%�	�jx_���A�*;


total_loss̚�@

error_R��W?

learning_rate_1먷6�C�I       6%�	��x_���A�*;


total_loss흌@

error_R_UJ?

learning_rate_1먷6;�^I       6%�	��x_���A�*;


total_lossNƬ@

error_R�F\?

learning_rate_1먷6'3kI       6%�	a=y_���A�*;


total_loss閴@

error_R{�`?

learning_rate_1먷6�d�I       6%�	��y_���A�*;


total_lossA�@

error_R8�M?

learning_rate_1먷6��bI       6%�	}�y_���A�*;


total_lossM6�@

error_RLw^?

learning_rate_1먷6�6!I       6%�	�
z_���A�*;


total_loss�E�@

error_R�??

learning_rate_1먷6��SI       6%�	�Nz_���A�*;


total_lossץ�@

error_R1D?

learning_rate_1먷6�I       6%�	V�z_���A�*;


total_lossb�@

error_R�[?

learning_rate_1먷6[C�I       6%�	��z_���A�*;


total_loss�4�@

error_R*V?

learning_rate_1먷6f�ĽI       6%�	{#{_���A�*;


total_loss�;�@

error_R�D?

learning_rate_1먷6|�J�I       6%�	j{_���A�*;


total_loss��@

error_R�aZ?

learning_rate_1먷6�x��I       6%�		�{_���A�*;


total_loss�]�@

error_R�)<?

learning_rate_1먷6��S"I       6%�	��{_���A�*;


total_loss�ە@

error_R3�O?

learning_rate_1먷6��-I       6%�	�P|_���A�*;


total_loss�x�@

error_R&�E?

learning_rate_1먷6�V^oI       6%�	6�|_���A�*;


total_lossS��@

error_RdmQ?

learning_rate_1먷6�0�I       6%�	F�|_���A�*;


total_loss�"�@

error_R$yG?

learning_rate_1먷6캾WI       6%�	�)}_���A�*;


total_loss��@

error_R�F?

learning_rate_1먷6�-QI       6%�	�}_���A�*;


total_losse��@

error_R�I?

learning_rate_1먷6��$I       6%�	�}_���A�*;


total_loss�`�@

error_R iT?

learning_rate_1먷6���I       6%�	�$~_���A�*;


total_loss�y�@

error_R��??

learning_rate_1먷6��]�I       6%�	6�~_���A�*;


total_loss�Nb@

error_Rm�@?

learning_rate_1먷6e��xI       6%�	��~_���A�*;


total_lossʧ�@

error_R�#X?

learning_rate_1먷6�@�I       6%�	c"_���A�*;


total_loss
��@

error_R
>C?

learning_rate_1먷6��I       6%�	�n_���A�*;


total_loss��@

error_RlC?

learning_rate_1먷6���I       6%�	#�_���A�*;


total_loss��@

error_Rn�c?

learning_rate_1먷6�*I       6%�	�)�_���A�*;


total_loss�6�@

error_R�>G?

learning_rate_1먷6Q��I       6%�	�t�_���A�*;


total_lossώ�@

error_R�V?

learning_rate_1먷6>��vI       6%�	b�_���A�*;


total_loss�0�@

error_Rd>?

learning_rate_1먷6��"�I       6%�	�2�_���A�*;


total_loss���@

error_RJ�P?

learning_rate_1먷6Z�JI       6%�	���_���A�*;


total_loss���@

error_R��N?

learning_rate_1먷6��Q�I       6%�	�́_���A�*;


total_loss�y�@

error_R��N?

learning_rate_1먷6���~I       6%�	��_���A�*;


total_loss���@

error_R�J?

learning_rate_1먷6�=&ZI       6%�	�i�_���A�*;


total_lossOu�@

error_RZ�P?

learning_rate_1먷6e�$I       6%�	?Ղ_���A�*;


total_losssZ�@

error_R��C?

learning_rate_1먷6���I       6%�	q$�_���A�*;


total_loss�G�@

error_RVbS?

learning_rate_1먷6d�r�I       6%�	ki�_���A�*;


total_loss��s@

error_R)5_?

learning_rate_1먷6�h�?I       6%�	>��_���A�*;


total_loss �v@

error_R�"=?

learning_rate_1먷6W/�I       6%�	g�_���A�*;


total_loss*�@

error_R�C?

learning_rate_1먷6���	I       6%�	9�_���A�*;


total_loss���@

error_RݢS?

learning_rate_1먷6,Vu�I       6%�	]~�_���A�*;


total_lossRh�@

error_R�X?

learning_rate_1먷6�PE�I       6%�	iĄ_���A�*;


total_loss�dj@

error_R�TL?

learning_rate_1먷6=��FI       6%�	�	�_���A�*;


total_loss���@

error_R��1?

learning_rate_1먷6%
QRI       6%�	`L�_���A�*;


total_loss C�@

error_RF�A?

learning_rate_1먷6C��VI       6%�	���_���A�*;


total_loss�q�@

error_R�jM?

learning_rate_1먷6 }C�I       6%�	.Յ_���A�*;


total_loss�E�@

error_Ro�O?

learning_rate_1먷6A�-I       6%�	��_���A�*;


total_lossrq�@

error_R�^?

learning_rate_1먷6J�m�I       6%�	�Y�_���A�*;


total_loss�tz@

error_R�NA?

learning_rate_1먷6s9�?I       6%�	_���A�*;


total_loss��@

error_R%2N?

learning_rate_1먷6�.�I       6%�	��_���A�*;


total_loss�@

error_R�=?

learning_rate_1먷6;���I       6%�	�,�_���A�*;


total_lossZ#�@

error_Ri�>?

learning_rate_1먷6�9%�I       6%�	�t�_���A�*;


total_loss �|@

error_R��L?

learning_rate_1먷6�1��I       6%�	;Ƈ_���A�*;


total_lossʊ�@

error_R�4A?

learning_rate_1먷6���-I       6%�	�_���A�*;


total_lossN]�@

error_Rn�X?

learning_rate_1먷6�\/I       6%�	fT�_���A�*;


total_lossN@�@

error_R��Q?

learning_rate_1먷6�I       6%�	었_���A�*;


total_loss�m�@

error_R�J?

learning_rate_1먷6���I       6%�	݈_���A�*;


total_loss�8�@

error_RWC?

learning_rate_1먷6u���I       6%�	)�_���A�*;


total_loss�6�@

error_R�DK?

learning_rate_1먷68�CI       6%�	�k�_���A�*;


total_losseȠ@

error_R�R?

learning_rate_1먷6#HR�I       6%�	2��_���A�*;


total_lossӟ�@

error_R=u??

learning_rate_1먷6�� �I       6%�	��_���A�*;


total_loss\W�@

error_R3�M?

learning_rate_1먷6WBG�I       6%�	�0�_���A�*;


total_loss�i�@

error_R�hQ?

learning_rate_1먷60>I       6%�	�x�_���A�*;


total_lossT��@

error_R�EI?

learning_rate_1먷6�E�I       6%�	x��_���A�*;


total_loss>�A

error_R�R?

learning_rate_1먷6(��I       6%�	}�_���A�*;


total_loss.�@

error_R��>?

learning_rate_1먷62��I       6%�	F�_���A�*;


total_loss��@

error_R��L?

learning_rate_1먷6�I�I       6%�	戋_���A�*;


total_loss���@

error_R�dY?

learning_rate_1먷6���I       6%�	�̋_���A�*;


total_loss]�@

error_R�&B?

learning_rate_1먷6�	'I       6%�	� �_���A�*;


total_loss��!A

error_R��J?

learning_rate_1먷6z<�4I       6%�	-|�_���A�*;


total_lossH-�@

error_RO�U?

learning_rate_1먷60((�I       6%�	%Č_���A�*;


total_lossD��@

error_R]Q?

learning_rate_1먷6��}�I       6%�	\�_���A�*;


total_loss��@

error_R�1M?

learning_rate_1먷6�B�I       6%�	`J�_���A�*;


total_loss�4�@

error_R��Q?

learning_rate_1먷6����I       6%�	���_���A�*;


total_loss|�@

error_Rv�B?

learning_rate_1먷69 S,I       6%�	OՍ_���A�*;


total_loss{3y@

error_R�NG?

learning_rate_1먷6Z{I       6%�	6�_���A�*;


total_loss���@

error_R�5O?

learning_rate_1먷6�gK�I       6%�	�_�_���A�*;


total_loss�֕@

error_R;<N?

learning_rate_1먷6�7N�I       6%�	ӣ�_���A�*;


total_lossF�@

error_R��]?

learning_rate_1먷6�w��I       6%�	j�_���A�*;


total_loss�d�@

error_R/F?

learning_rate_1먷6�@I       6%�	j;�_���A�*;


total_loss{�@

error_R�G?

learning_rate_1먷6vJ�eI       6%�	`��_���A�*;


total_lossM)�@

error_Rx�T?

learning_rate_1먷6�N�I       6%�	Ώ_���A�*;


total_loss]@�@

error_R�H?

learning_rate_1먷6gC�I       6%�	��_���A�*;


total_lossm �@

error_RF�Q?

learning_rate_1먷6��-�I       6%�	�[�_���A�*;


total_loss�qv@

error_R��S?

learning_rate_1먷6@+�I       6%�	{��_���A�*;


total_loss d�@

error_ReA?

learning_rate_1먷6�mHEI       6%�	"�_���A�*;


total_loss(�@

error_R�i?

learning_rate_1먷6���I       6%�	;5�_���A�*;


total_losss�@

error_R��D?

learning_rate_1먷6��WI       6%�	||�_���A�*;


total_loss���@

error_R�T?

learning_rate_1먷6�
ͤI       6%�	�Ƒ_���A�*;


total_lossn�@

error_R��E?

learning_rate_1먷6o�I       6%�	��_���A�*;


total_loss�0�@

error_R�N?

learning_rate_1먷6��aI       6%�	�V�_���A�*;


total_losspA

error_R�P?

learning_rate_1먷6I���I       6%�	\��_���A�*;


total_loss|�@

error_R�gE?

learning_rate_1먷6�� I       6%�	9ߒ_���A�*;


total_loss�پ@

error_R_�T?

learning_rate_1먷6O�ZJI       6%�	�!�_���A�*;


total_loss#�@

error_RZ�V?

learning_rate_1먷6����I       6%�	�c�_���A�*;


total_loss�l�@

error_R�R?

learning_rate_1먷6�0��I       6%�	嫓_���A�*;


total_lossB��@

error_RO X?

learning_rate_1먷6���I       6%�	���_���A�*;


total_lossE�@

error_Rj\P?

learning_rate_1먷6%�qZI       6%�	�@�_���A�*;


total_loss\��@

error_Rl.b?

learning_rate_1먷6��>I       6%�	S��_���A�*;


total_loss��@

error_R�H?

learning_rate_1먷61$�I       6%�	�Ȕ_���A�*;


total_lossH��@

error_R��S?

learning_rate_1먷6*x��I       6%�	'�_���A�*;


total_loss/�X@

error_RFu??

learning_rate_1먷6�I       6%�	�M�_���A�*;


total_loss#y�@

error_R��[?

learning_rate_1먷6���nI       6%�	���_���A�*;


total_loss!��@

error_REc[?

learning_rate_1먷6�r�I       6%�	�Օ_���A�*;


total_loss�]�@

error_R��G?

learning_rate_1먷6>d0I       6%�	��_���A�*;


total_loss���@

error_R80E?

learning_rate_1먷6n�;/I       6%�	�g�_���A�*;


total_loss���@

error_RHj?

learning_rate_1먷6|D�I       6%�	cϖ_���A�*;


total_loss߀d@

error_R`@K?

learning_rate_1먷6�B�I       6%�	�_���A�*;


total_loss�y�@

error_R�H?

learning_rate_1먷6d	A*I       6%�	�d�_���A�*;


total_loss�v\@

error_RD�H?

learning_rate_1먷6iw��I       6%�	���_���A�*;


total_lossT'�@

error_R��G?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss(��@

error_RT�M?

learning_rate_1먷6�?��I       6%�	�M�_���A�*;


total_loss��@

error_R�HO?

learning_rate_1먷6��%I       6%�	
��_���A�*;


total_loss5�@

error_R]>?

learning_rate_1먷6���AI       6%�	���_���A�*;


total_loss���@

error_R\~G?

learning_rate_1먷60�<�I       6%�	/�_���A�*;


total_loss�ݦ@

error_R�F?

learning_rate_1먷6]��QI       6%�	�z�_���A�*;


total_loss ��@

error_R��E?

learning_rate_1먷6���I       6%�	���_���A�*;


total_loss&��@

error_R�'H?

learning_rate_1먷6���$I       6%�	��_���A�*;


total_loss��@

error_R��T?

learning_rate_1먷6a��I       6%�	�J�_���A�*;


total_loss��p@

error_R��J?

learning_rate_1먷6�`��I       6%�	᎚_���A�*;


total_loss�k�@

error_R/�=?

learning_rate_1먷63Ү�I       6%�	Қ_���A�*;


total_lossc�@

error_R�/d?

learning_rate_1먷6��XI       6%�	9�_���A�*;


total_lossF��@

error_R�5S?

learning_rate_1먷6��-mI       6%�	�`�_���A�*;


total_loss=�A

error_R��T?

learning_rate_1먷6��"-I       6%�	z��_���A�*;


total_loss���@

error_R�U?

learning_rate_1먷6X�_I       6%�	P�_���A�*;


total_lossfԳ@

error_R�F?

learning_rate_1먷6�V�I       6%�	UK�_���A�*;


total_lossh�@

error_R��F?

learning_rate_1먷6��|CI       6%�	[��_���A�*;


total_loss��@

error_R}�S?

learning_rate_1먷6鶪I       6%�	؜_���A�*;


total_loss���@

error_R��<?

learning_rate_1먷6!��,I       6%�	��_���A�*;


total_loss�@

error_R;I?

learning_rate_1먷6rǕ�I       6%�	U`�_���A�*;


total_loss`�}@

error_R2 F?

learning_rate_1먷6`Ke�I       6%�	ޤ�_���A�*;


total_loss �@

error_R==Y?

learning_rate_1먷6�g YI       6%�	��_���A�*;


total_lossi�@

error_R��Y?

learning_rate_1먷6g�.�I       6%�	y-�_���A�*;


total_loss �@

error_R�\8?

learning_rate_1먷6*E�XI       6%�	2r�_���A�*;


total_loss6�@

error_R1QN?

learning_rate_1먷6�UK�I       6%�	��_���A�*;


total_loss\��@

error_R-T?

learning_rate_1먷6iP��I       6%�	���_���A�*;


total_lossc�@

error_R��F?

learning_rate_1먷6�ƛVI       6%�	�>�_���A�*;


total_loss+A

error_R
CP?

learning_rate_1먷6Y��HI       6%�	6��_���A�*;


total_loss�6�@

error_R:`C?

learning_rate_1먷6�WI       6%�	zğ_���A�*;


total_loss��@

error_R�rV?

learning_rate_1먷6|��I       6%�	�	�_���A�*;


total_loss���@

error_RxU?

learning_rate_1먷6)�#�I       6%�	^N�_���A�*;


total_loss���@

error_RX.g?

learning_rate_1먷6��I       6%�	���_���A�*;


total_loss���@

error_RWf]?

learning_rate_1먷6s�z:I       6%�	�ՠ_���A�*;


total_loss
A

error_Rԣ??

learning_rate_1먷6�Y�=I       6%�	��_���A�*;


total_lossAۧ@

error_RΣ>?

learning_rate_1먷6�s�qI       6%�	Z�_���A�*;


total_loss�d�@

error_R1�U?

learning_rate_1먷6-�oI       6%�	O��_���A�*;


total_loss��@

error_R?/Y?

learning_rate_1먷6!+p�I       6%�	��_���A�*;


total_loss�Ʊ@

error_R6�\?

learning_rate_1먷6foLI       6%�	i/�_���A�*;


total_loss��@

error_R�bB?

learning_rate_1먷6oh�I       6%�	�r�_���A�*;


total_lossR�@

error_Rd�a?

learning_rate_1먷6ВWKI       6%�	f��_���A�*;


total_loss�ż@

error_RNZ?

learning_rate_1먷6�S��I       6%�	� �_���A�*;


total_loss䚪@

error_R	�K?

learning_rate_1먷6�F I       6%�	L�_���A�*;


total_loss}��@

error_R�N?

learning_rate_1먷6��XI       6%�	d��_���A�*;


total_loss��@

error_R�NN?

learning_rate_1먷6�h*I       6%�	�ݣ_���A�*;


total_lossh��@

error_R�I?

learning_rate_1먷6�U�TI       6%�	X*�_���A�*;


total_loss�(�@

error_R�VO?

learning_rate_1먷6#��_I       6%�		q�_���A�*;


total_loss�D�@

error_R�Xc?

learning_rate_1먷6N��I       6%�	���_���A�*;


total_loss��@

error_R��D?

learning_rate_1먷6��I       6%�	���_���A�*;


total_lossx"�@

error_R�GS?

learning_rate_1먷6j�P�I       6%�	;�_���A�*;


total_loss��@

error_R�7;?

learning_rate_1먷6b��I       6%�	~��_���A�*;


total_loss���@

error_RlVL?

learning_rate_1먷6�+�RI       6%�	ť_���A�*;


total_lossA�@

error_Rv�@?

learning_rate_1먷6/��I       6%�	��_���A�*;


total_loss}��@

error_R�<??

learning_rate_1먷6[��I       6%�	�N�_���A�*;


total_lossا�@

error_R� H?

learning_rate_1먷6�ySI       6%�	�_���A�*;


total_loss�˩@

error_RW�K?

learning_rate_1먷6��OI       6%�	�צ_���A�*;


total_loss�$�@

error_RHgV?

learning_rate_1먷6(W�cI       6%�	��_���A�*;


total_loss�@

error_R�GU?

learning_rate_1먷6��aI       6%�	�`�_���A�*;


total_loss.*�@

error_R�^I?

learning_rate_1먷6#z��I       6%�	o��_���A�*;


total_loss��@

error_R��N?

learning_rate_1먷6AU%�I       6%�	f�_���A�*;


total_loss���@

error_R��M?

learning_rate_1먷6��ȏI       6%�	�+�_���A�*;


total_loss���@

error_R2�R?

learning_rate_1먷60�άI       6%�	�u�_���A�*;


total_lossuĒ@

error_R3�H?

learning_rate_1먷6U��I       6%�	e��_���A�*;


total_lossaj�@

error_R-`?

learning_rate_1먷6��O$I       6%�	n	�_���A�*;


total_loss�|�@

error_RSnH?

learning_rate_1먷6�U��I       6%�	*T�_���A�*;


total_losslڲ@

error_R�VX?

learning_rate_1먷6���*I       6%�	���_���A�*;


total_loss��A

error_R\�a?

learning_rate_1먷6^ԚI       6%�	�ܩ_���A�*;


total_loss僔@

error_RzZ?

learning_rate_1먷6����I       6%�	��_���A�*;


total_loss���@

error_R�"_?

learning_rate_1먷6�r?�I       6%�	kg�_���A�*;


total_loss�F�@

error_R�Y?

learning_rate_1먷6�v�,I       6%�	%��_���A�*;


total_lossZG�@

error_RNN?

learning_rate_1먷6��LI       6%�	���_���A�*;


total_lossw��@

error_R�:P?

learning_rate_1먷6h�;I       6%�	@�_���A�*;


total_loss�G�@

error_R��R?

learning_rate_1먷6��I       6%�	���_���A�*;


total_loss���@

error_R13[?

learning_rate_1먷6�7�I       6%�	�˫_���A�*;


total_loss�@

error_Ri�U?

learning_rate_1먷6M�VI       6%�	�"�_���A�*;


total_loss���@

error_R�[?

learning_rate_1먷6�T>�I       6%�	���_���A�*;


total_loss��@

error_R�<S?

learning_rate_1먷6=qm�I       6%�	�ɬ_���A�*;


total_loss���@

error_R�5R?

learning_rate_1먷6p2:I       6%�	-�_���A�*;


total_loss��@

error_R1NF?

learning_rate_1먷6�	�I       6%�	�U�_���A�*;


total_loss}��@

error_R�L?

learning_rate_1먷6�b-�I       6%�	3��_���A�*;


total_loss�r�@

error_Rvt\?

learning_rate_1먷6�E�QI       6%�	��_���A�*;


total_loss2�@

error_R��5?

learning_rate_1먷6��I       6%�	�*�_���A�*;


total_loss�@

error_R� S?

learning_rate_1먷6�	�<I       6%�	8r�_���A�*;


total_lossV��@

error_RE8?

learning_rate_1먷6�A��I       6%�	T��_���A�*;


total_loss�ڒ@

error_R��U?

learning_rate_1먷6M�yI       6%�	���_���A�*;


total_loss���@

error_R��H?

learning_rate_1먷6�9�I       6%�	�A�_���A�*;


total_lossܸ�@

error_RZdd?

learning_rate_1먷6FH�6I       6%�	���_���A�*;


total_lossM{�@

error_RT_A?

learning_rate_1먷6 �9I       6%�	dȯ_���A�*;


total_loss;a�@

error_R1�>?

learning_rate_1먷6`-��I       6%�	��_���A�*;


total_loss�Px@

error_R(�G?

learning_rate_1먷6�Ox[I       6%�	�T�_���A�*;


total_loss�#�@

error_R	p1?

learning_rate_1먷6���&I       6%�	c��_���A�*;


total_loss���@

error_R
�D?

learning_rate_1먷6�> I       6%�	o�_���A�*;


total_lossƉ�@

error_R�S?

learning_rate_1먷6(�}I       6%�	�.�_���A�*;


total_lossL��@

error_RsV?

learning_rate_1먷6Y@�LI       6%�	�{�_���A�*;


total_loss�(�@

error_R�}K?

learning_rate_1먷6���I       6%�	1ű_���A�*;


total_loss{=�@

error_RZ�M?

learning_rate_1먷6�h�bI       6%�	��_���A�*;


total_loss�@

error_R6W@?

learning_rate_1먷6N��kI       6%�	
Z�_���A�*;


total_lossi��@

error_R�	O?

learning_rate_1먷6��I       6%�	V��_���A�*;


total_loss���@

error_R�IC?

learning_rate_1먷6��I       6%�	���_���A�*;


total_loss���@

error_R�%K?

learning_rate_1먷6�S�EI       6%�	AH�_���A�*;


total_loss���@

error_R��F?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss�@

error_RC?

learning_rate_1먷6Փ@8I       6%�	1ֳ_���A�*;


total_loss�ع@

error_RH8U?

learning_rate_1먷6��9~I       6%�	P�_���A�*;


total_loss��@

error_Rݹ@?

learning_rate_1먷6�SS�I       6%�	_�_���A�*;


total_loss_��@

error_R8m=?

learning_rate_1먷6M띿I       6%�	���_���A�*;


total_loss�.�@

error_R��D?

learning_rate_1먷6���I       6%�	�_���A�*;


total_loss�;�@

error_R16H?

learning_rate_1먷6� ��I       6%�	�/�_���A�*;


total_loss �@

error_R��T?

learning_rate_1먷6��2�I       6%�	�s�_���A�*;


total_lossfO�@

error_R�VC?

learning_rate_1먷6Ym��I       6%�	ܸ�_���A�*;


total_loss@�@

error_R��V?

learning_rate_1먷6Tn��I       6%�	�_���A�*;


total_lossw��@

error_R��A?

learning_rate_1먷6��ˍI       6%�	�E�_���A�*;


total_loss�8�@

error_R�L?

learning_rate_1먷6u�I       6%�	���_���A�*;


total_loss�K�@

error_RE*@?

learning_rate_1먷6h�DBI       6%�	�Ѷ_���A�*;


total_lossm}�@

error_R��h?

learning_rate_1먷6o��}I       6%�	��_���A�*;


total_lossib�@

error_R!x>?

learning_rate_1먷6I9�gI       6%�	�a�_���A�*;


total_lossN�@

error_R�]??

learning_rate_1먷6K��I       6%�	f��_���A�*;


total_loss��@

error_R�J?

learning_rate_1먷61TI       6%�	���_���A�*;


total_loss���@

error_RӶJ?

learning_rate_1먷6�I       6%�	;�_���A�*;


total_loss2��@

error_R�J?

learning_rate_1먷6��I       6%�	�}�_���A�*;


total_loss���@

error_ReG?

learning_rate_1먷6Q�|�I       6%�	_ø_���A�*;


total_loss�ֵ@

error_RňK?

learning_rate_1먷6X�%�I       6%�	��_���A�*;


total_losstP�@

error_Ra�B?

learning_rate_1먷6��**I       6%�	�J�_���A�*;


total_loss*��@

error_R3J\?

learning_rate_1먷6V��I       6%�	���_���A�*;


total_loss$]�@

error_Rv�7?

learning_rate_1먷6�J�I       6%�	�߹_���A�*;


total_loss1�@

error_RV?

learning_rate_1먷6B3�I       6%�	�)�_���A�*;


total_loss���@

error_R?IG?

learning_rate_1먷6���I       6%�	�t�_���A�*;


total_lossjA

error_R��J?

learning_rate_1먷6r�0:I       6%�	���_���A�*;


total_lossD��@

error_R��U?

learning_rate_1먷6��I       6%�	P�_���A�*;


total_lossќ�@

error_R:�H?

learning_rate_1먷6}�1
I       6%�	SE�_���A�*;


total_loss�@

error_R)cF?

learning_rate_1먷6ZA��I       6%�	n��_���A�*;


total_loss��@

error_R�KT?

learning_rate_1먷6���I       6%�	�ͻ_���A�*;


total_loss�	�@

error_R(9U?

learning_rate_1먷6�#A=I       6%�	"�_���A�*;


total_loss�@

error_R��Y?

learning_rate_1먷6k]I       6%�	�w�_���A�*;


total_loss��@

error_R�]?

learning_rate_1먷6�$I       6%�		��_���A�*;


total_lossې@

error_Rm�M?

learning_rate_1먷6��UI       6%�	: �_���A�*;


total_loss���@

error_R:H?

learning_rate_1먷6Qd�VI       6%�	�E�_���A�*;


total_loss�|�@

error_R�D?

learning_rate_1먷6��g/I       6%�	ܒ�_���A�*;


total_loss��@

error_RԱN?

learning_rate_1먷6��"I       6%�	!�_���A�*;


total_losss��@

error_R3�>?

learning_rate_1먷6��\I       6%�	�'�_���A�*;


total_lossS��@

error_Rr�F?

learning_rate_1먷6 �I       6%�	5p�_���A�*;


total_lossZg{@

error_R%48?

learning_rate_1먷67UTI       6%�	���_���A�*;


total_loss���@

error_R\�[?

learning_rate_1먷6�o�I       6%�	&��_���A�*;


total_losst��@

error_R��O?

learning_rate_1먷6�t	0I       6%�	�=�_���A�*;


total_loss�O�@

error_RZAX?

learning_rate_1먷6���2I       6%�	���_���A�*;


total_loss�}�@

error_Rwpa?

learning_rate_1먷6Ò��I       6%�	�ǿ_���A�*;


total_loss,�@

error_R��G?

learning_rate_1먷6�aw�I       6%�	�_���A�*;


total_loss���@

error_R�{X?

learning_rate_1먷6�xiI       6%�	X�_���A�*;


total_loss�JA

error_RRL?

learning_rate_1먷6� �I       6%�	ߚ�_���A�*;


total_loss��@

error_R�&G?

learning_rate_1먷6[g��I       6%�	X��_���A�*;


total_loss`��@

error_RmUJ?

learning_rate_1먷6F�m�I       6%�	^"�_���A�*;


total_loss�cA

error_RD�X?

learning_rate_1먷6�\1I       6%�	Ad�_���A�*;


total_loss�9�@

error_R��F?

learning_rate_1먷6K�� I       6%�	��_���A�*;


total_loss��@

error_R��7?

learning_rate_1먷6ц�I       6%�	r��_���A�*;


total_loss�WH@

error_Ra[H?

learning_rate_1먷6I�;+I       6%�	�,�_���A�*;


total_loss[d�@

error_RڙZ?

learning_rate_1먷6�bVI       6%�	$q�_���A�*;


total_lossy��@

error_R�oR?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss���@

error_RO�7?

learning_rate_1먷6X��0I       6%�	T��_���A�*;


total_lossI��@

error_R)R?

learning_rate_1먷6�wZI       6%�	�<�_���A�*;


total_losso*�@

error_R�]D?

learning_rate_1먷6�ڷ+I       6%�	y��_���A�*;


total_loss�h�@

error_RV?

learning_rate_1먷6��9ZI       6%�	���_���A�*;


total_loss"�@

error_R�{A?

learning_rate_1먷6�:�I       6%�	��_���A�*;


total_loss���@

error_R��\?

learning_rate_1먷6��8yI       6%�	Y�_���A�*;


total_loss82�@

error_RʲW?

learning_rate_1먷6A�џI       6%�	Ԟ�_���A�*;


total_loss1�@

error_Rf�[?

learning_rate_1먷6�I�(I       6%�	I��_���A�*;


total_lossj�@

error_R��N?

learning_rate_1먷6��-�I       6%�	�$�_���A�*;


total_loss?�R@

error_R�EE?

learning_rate_1먷6=�I       6%�	Mi�_���A�*;


total_loss��A

error_RJ�C?

learning_rate_1먷6q!X�I       6%�	��_���A�*;


total_loss�w�@

error_R�p\?

learning_rate_1먷6�j8�I       6%�	���_���A�*;


total_lossiN�@

error_R3�Q?

learning_rate_1먷6LO>*I       6%�	S8�_���A�*;


total_loss��A

error_R�|M?

learning_rate_1먷6�kI       6%�	|�_���A�*;


total_loss6a�@

error_R��I?

learning_rate_1먷6�Ҽ�I       6%�	��_���A�*;


total_lossr�@

error_R��L?

learning_rate_1먷6�4{�I       6%�	��_���A�*;


total_lossi�@

error_Rvy9?

learning_rate_1먷6TI       6%�	}I�_���A�*;


total_loss�gy@

error_R�G?

learning_rate_1먷6���PI       6%�	���_���A�*;


total_lossߍ�@

error_R�D?

learning_rate_1먷6���I       6%�	���_���A�*;


total_loss�ĵ@

error_RËN?

learning_rate_1먷6��FI       6%�	�_���A�*;


total_loss���@

error_R;^R?

learning_rate_1먷6���I       6%�	�V�_���A�*;


total_loss���@

error_R1wX?

learning_rate_1먷6���lI       6%�	���_���A�*;


total_loss[��@

error_R�
A?

learning_rate_1먷6��'I       6%�	~��_���A�*;


total_loss	C�@

error_R?�K?

learning_rate_1먷6����I       6%�	).�_���A�*;


total_lossӨ�@

error_RÆD?

learning_rate_1먷6E�5I       6%�	Sq�_���A�*;


total_loss�M�@

error_R�)I?

learning_rate_1먷6�Z=�I       6%�	|��_���A�*;


total_losss�c@

error_R8�2?

learning_rate_1먷6�tI       6%�	U��_���A�*;


total_loss�A�@

error_RvR?

learning_rate_1먷6���I       6%�	B�_���A�*;


total_loss�c�@

error_R{�N?

learning_rate_1먷68��gI       6%�	��_���A�*;


total_loss���@

error_R�'V?

learning_rate_1먷6ކ|jI       6%�	A��_���A�*;


total_loss�v�@

error_RJ�X?

learning_rate_1먷6�b�I       6%�	��_���A�*;


total_loss`]�@

error_R��B?

learning_rate_1먷65>� I       6%�	�L�_���A�*;


total_loss��@

error_R�2J?

learning_rate_1먷6���I       6%�	ǐ�_���A�*;


total_loss{\�@

error_R�P?

learning_rate_1먷6��BI       6%�	���_���A�*;


total_lossC˖@

error_Rz3f?

learning_rate_1먷6� q
I       6%�	�#�_���A�*;


total_losslW�@

error_RM�U?

learning_rate_1먷6��I       6%�	W��_���A�*;


total_lossQ_�@

error_R�HS?

learning_rate_1먷68��I       6%�	���_���A�*;


total_loss&�A

error_R1S?

learning_rate_1먷6ܭ~I       6%�	y�_���A�*;


total_loss� A

error_R��V?

learning_rate_1먷6-hPI       6%�	�b�_���A�*;


total_loss��P@

error_R��D?

learning_rate_1먷6�Y��I       6%�	���_���A�*;


total_losss�@

error_R��Q?

learning_rate_1먷6,��rI       6%�	P��_���A�*;


total_lossY�@

error_R�KL?

learning_rate_1먷6�&�1I       6%�	C9�_���A�*;


total_loss�3x@

error_R�J?

learning_rate_1먷6��-SI       6%�	s|�_���A�*;


total_loss���@

error_R��^?

learning_rate_1먷6��3 I       6%�	���_���A�*;


total_loss�@

error_R�Z?

learning_rate_1먷6�zW'I       6%�	O�_���A�*;


total_loss/�@

error_RI�S?

learning_rate_1먷6�hcI       6%�	�N�_���A�*;


total_lossʝ�@

error_RWc8?

learning_rate_1먷6�q.jI       6%�	%��_���A�*;


total_loss=>�@

error_R��P?

learning_rate_1먷6zRr4I       6%�	_��_���A�*;


total_lossc|�@

error_R��:?

learning_rate_1먷6>�R�I       6%�	��_���A�*;


total_loss���@

error_R
jL?

learning_rate_1먷6 q�I       6%�	U`�_���A�*;


total_loss�ٴ@

error_R�W?

learning_rate_1먷6"��oI       6%�	ʣ�_���A�*;


total_lossA��@

error_R��D?

learning_rate_1먷6HJ��I       6%�	���_���A�*;


total_loss/��@

error_R��K?

learning_rate_1먷6- �I       6%�	�/�_���A�*;


total_loss�%A

error_R��??

learning_rate_1먷6�yp�I       6%�	�u�_���A�*;


total_loss!&�@

error_R�<W?

learning_rate_1먷6ȟ1�I       6%�	n��_���A�*;


total_lossi$�@

error_R&�D?

learning_rate_1먷6<�.I       6%�	H��_���A�*;


total_loss X�@

error_R!�=?

learning_rate_1먷6(q7I       6%�	�@�_���A�*;


total_loss�ǵ@

error_RW�M?

learning_rate_1먷6g�>I       6%�	h��_���A�*;


total_loss��@

error_R4rX?

learning_rate_1먷6Ų�cI       6%�	���_���A�*;


total_lossM�@

error_Ri�Z?

learning_rate_1먷6Q��I       6%�	�C�_���A�*;


total_loss�1�@

error_RD�e?

learning_rate_1먷6k���I       6%�	C��_���A�*;


total_loss�@

error_R6�F?

learning_rate_1먷6��%I       6%�	���_���A�*;


total_lossͣ@

error_R�B?

learning_rate_1먷6y;�I       6%�	��_���A�*;


total_lossE��@

error_R�b?

learning_rate_1먷6n^	HI       6%�	�^�_���A�*;


total_loss�@

error_R�H?

learning_rate_1먷6l���I       6%�	I��_���A�*;


total_loss�@�@

error_R��G?

learning_rate_1먷6�.a�I       6%�	��_���A�*;


total_loss�̱@

error_R�R?

learning_rate_1먷6n�q\I       6%�	h0�_���A�*;


total_loss��@

error_R��J?

learning_rate_1먷6/J��I       6%�	#~�_���A�*;


total_lossC�@

error_Rm~O?

learning_rate_1먷6��k,I       6%�	��_���A�*;


total_loss��v@

error_R ~B?

learning_rate_1먷6��neI       6%�	�	�_���A�*;


total_loss�l�@

error_R��T?

learning_rate_1먷6��jI       6%�	3N�_���A�*;


total_loss�Nv@

error_Rd4O?

learning_rate_1먷66�9�I       6%�	ۑ�_���A�*;


total_loss�l�@

error_RJ\?

learning_rate_1먷6�7YQI       6%�	F��_���A�*;


total_loss�H�@

error_R8�T?

learning_rate_1먷6�2QI       6%�	h�_���A�*;


total_loss�ϵ@

error_RlJ?

learning_rate_1먷6�.��I       6%�	_`�_���A�*;


total_loss�1�@

error_RV|g?

learning_rate_1먷6�+]I       6%�	���_���A�*;


total_loss{J�@

error_RlMR?

learning_rate_1먷6{Du�I       6%�	���_���A�*;


total_loss��@

error_R�B?

learning_rate_1먷6�!>8I       6%�	�+�_���A�*;


total_loss���@

error_RIL?

learning_rate_1먷6�bkI       6%�	�o�_���A�*;


total_loss��_@

error_Rw�K?

learning_rate_1먷6����I       6%�	��_���A�*;


total_loss���@

error_R,M?

learning_rate_1먷6��SI       6%�	(��_���A�*;


total_loss��^@

error_R�47?

learning_rate_1먷6q��vI       6%�	�<�_���A�*;


total_lossڰ�@

error_R�F?

learning_rate_1먷6�N�I       6%�	b��_���A�*;


total_lossrH�@

error_R��U?

learning_rate_1먷6�[�I       6%�	��_���A�*;


total_loss�b�@

error_R�[E?

learning_rate_1먷6��I       6%�	��_���A�*;


total_lossH-�@

error_R�j<?

learning_rate_1먷6E�HI       6%�	�G�_���A�*;


total_loss9l�@

error_R��??

learning_rate_1먷6��FI       6%�	���_���A�*;


total_loss�U�@

error_R�=P?

learning_rate_1먷6Y���I       6%�	���_���A�*;


total_loss�R�@

error_R��D?

learning_rate_1먷6�]�I       6%�	��_���A�*;


total_loss6k�@

error_R�9V?

learning_rate_1먷60�KI       6%�	$_�_���A�*;


total_loss �@

error_R�BD?

learning_rate_1먷6�t�I       6%�	���_���A�*;


total_loss��@

error_R�'\?

learning_rate_1먷6r1�I       6%�	M��_���A�*;


total_loss``�@

error_R2rU?

learning_rate_1먷6~��I       6%�	�K�_���A�*;


total_loss{и@

error_R�4?

learning_rate_1먷6�r��I       6%�	ܔ�_���A�*;


total_lossߺ@

error_R,�F?

learning_rate_1먷6��t�I       6%�	;��_���A�*;


total_loss=G�@

error_R�XG?

learning_rate_1먷6���I       6%�	�#�_���A�*;


total_loss�֎@

error_R�9H?

learning_rate_1먷6�kI       6%�	�g�_���A�*;


total_lossH�@

error_Rs�??

learning_rate_1먷6�9D7I       6%�	���_���A�*;


total_loss?�@

error_RT?

learning_rate_1먷6�  I       6%�	���_���A�*;


total_lossO*�@

error_RۏO?

learning_rate_1먷6��֪I       6%�	.;�_���A�*;


total_loss���@

error_R�V?

learning_rate_1먷6���2I       6%�	��_���A�*;


total_loss(r!A

error_R	(U?

learning_rate_1먷6Ζ��I       6%�	d��_���A�*;


total_loss�q�@

error_R=
Q?

learning_rate_1먷6��DzI       6%�	��_���A�*;


total_loss���@

error_R֘??

learning_rate_1먷6��p�I       6%�	"T�_���A�*;


total_lossh�A

error_R�)B?

learning_rate_1먷6Y)J�I       6%�	k��_���A�*;


total_loss���@

error_R��W?

learning_rate_1먷6�va%I       6%�	Z��_���A�*;


total_lossS+�@

error_RjL?

learning_rate_1먷6D:N�I       6%�	� �_���A�*;


total_loss�/�@

error_R�b?

learning_rate_1먷6ñ�I       6%�	�d�_���A�*;


total_loss셳@

error_R�7?

learning_rate_1먷6C)�I       6%�	q��_���A�*;


total_loss[��@

error_Ri�T?

learning_rate_1먷6a�ŊI       6%�	2��_���A�*;


total_loss육@

error_R/�L?

learning_rate_1먷6�g�'I       6%�	�7�_���A�*;


total_loss@��@

error_R�0R?

learning_rate_1먷6�=5I       6%�	y��_���A�*;


total_loss�B�@

error_R��C?

learning_rate_1먷6X���I       6%�	R��_���A�*;


total_loss7�@

error_R�L?

learning_rate_1먷6��N�I       6%�	��_���A�*;


total_loss���@

error_RT?

learning_rate_1먷6pxI       6%�	�S�_���A�*;


total_loss�@

error_R�?C?

learning_rate_1먷6(ӏ�I       6%�	���_���A�*;


total_loss���@

error_R��S?

learning_rate_1먷6q��WI       6%�	���_���A�*;


total_loss���@

error_RTsa?

learning_rate_1먷6���I       6%�	�#�_���A�*;


total_loss�Gk@

error_R�7?

learning_rate_1먷6���I       6%�	�j�_���A�*;


total_loss�e�@

error_R��N?

learning_rate_1먷6�<�I       6%�	��_���A�*;


total_loss ��@

error_R�S?

learning_rate_1먷6�B}�I       6%�	P��_���A�*;


total_loss&��@

error_R1�D?

learning_rate_1먷6�~�_I       6%�	?�_���A�*;


total_lossOq�@

error_R�,q?

learning_rate_1먷6�݇6I       6%�	��_���A�*;


total_loss#ɥ@

error_R�Q?

learning_rate_1먷6z:7I       6%�	3��_���A�*;


total_loss���@

error_R�`P?

learning_rate_1먷6�A��I       6%�	��_���A�*;


total_loss�(�@

error_R�Z?

learning_rate_1먷6���I       6%�	nW�_���A�*;


total_lossf��@

error_RWs5?

learning_rate_1먷6�ɀ^I       6%�	\��_���A�*;


total_lossXl�@

error_R}�F?

learning_rate_1먷6 ���I       6%�	���_���A�*;


total_lossnM]@

error_R�UO?

learning_rate_1먷6δ�I       6%�	�'�_���A�*;


total_loss�#A

error_RW?

learning_rate_1먷6�|#�I       6%�	Gr�_���A�*;


total_loss_w�@

error_R��X?

learning_rate_1먷6;ԪI       6%�	���_���A�*;


total_loss��@

error_R�?L?

learning_rate_1먷6���I       6%�	H�_���A�*;


total_lossx��@

error_R�E?

learning_rate_1먷6��NlI       6%�	)K�_���A�*;


total_loss��@

error_Rw�K?

learning_rate_1먷6���I       6%�	 ��_���A�*;


total_loss8f�@

error_R6�d?

learning_rate_1먷6`^�I       6%�	���_���A�*;


total_loss��	A

error_RA9?

learning_rate_1먷6�0�KI       6%�	�_���A�*;


total_loss�Ŋ@

error_R��P?

learning_rate_1먷6A�I       6%�	^�_���A�*;


total_lossh��@

error_R��S?

learning_rate_1먷6%��I       6%�	ߢ�_���A�*;


total_lossI��@

error_RW�E?

learning_rate_1먷6Zq��I       6%�	���_���A�*;


total_loss�.�@

error_R��A?

learning_rate_1먷6D�~�I       6%�	�.�_���A�*;


total_loss���@

error_R��Q?

learning_rate_1먷6H��I       6%�	�x�_���A�*;


total_lossi��@

error_R�2E?

learning_rate_1먷6}�D�I       6%�	u��_���A�*;


total_loss��@

error_R
nN?

learning_rate_1먷6�GI       6%�	��_���A�*;


total_losseN�@

error_R:�L?

learning_rate_1먷6����I       6%�	�I�_���A�*;


total_loss�M�@

error_R�$M?

learning_rate_1먷6�ЄI       6%�	`��_���A�*;


total_loss���@

error_Rö^?

learning_rate_1먷6�I       6%�	@��_���A�*;


total_lossh�@

error_R_(<?

learning_rate_1먷6��}PI       6%�	��_���A�*;


total_loss�@

error_R�{G?

learning_rate_1먷6)P'I       6%�	�_�_���A�*;


total_loss{�@

error_R�7[?

learning_rate_1먷6�a�#I       6%�	���_���A�*;


total_loss���@

error_R.N@?

learning_rate_1먷6��&�I       6%�	���_���A�*;


total_loss"H�@

error_R3B?

learning_rate_1먷6�~0I       6%�	Y<�_���A�*;


total_loss�˼@

error_R�o_?

learning_rate_1먷6I       6%�	���_���A�*;


total_loss�t]@

error_RW�W?

learning_rate_1먷6�wI       6%�	3��_���A�*;


total_loss��v@

error_R��T?

learning_rate_1먷6�tI       6%�	K-�_���A�*;


total_lossOI�@

error_R��C?

learning_rate_1먷6m��RI       6%�	gp�_���A�*;


total_loss�ֲ@

error_RL�/?

learning_rate_1먷6�f�I       6%�	i��_���A�*;


total_loss�G�@

error_RCrE?

learning_rate_1먷6$;(DI       6%�	{��_���A�*;


total_loss�Ͳ@

error_R@�7?

learning_rate_1먷6���I       6%�	~B�_���A�*;


total_lossJ�A

error_R��Q?

learning_rate_1먷6�o��I       6%�	,��_���A�*;


total_loss���@

error_R)�[?

learning_rate_1먷6�2�<I       6%�	���_���A�*;


total_loss�?�@

error_R��@?

learning_rate_1먷6����I       6%�	��_���A�*;


total_loss�@

error_R�O?

learning_rate_1먷6�'�I       6%�	zO�_���A�*;


total_lossJE�@

error_R��W?

learning_rate_1먷6_Wx�I       6%�	�_���A�*;


total_loss��@

error_R(�C?

learning_rate_1먷6/s�I       6%�	���_���A�*;


total_loss,��@

error_RCS?

learning_rate_1먷6?�7�I       6%�	�_���A�*;


total_loss{@

error_Ro�Q?

learning_rate_1먷6y�:�I       6%�	�_�_���A�*;


total_lossf��@

error_RN�H?

learning_rate_1먷6"yOSI       6%�	`��_���A�*;


total_loss*�@

error_R�[?

learning_rate_1먷6��8I       6%�	��_���A�*;


total_loss�5�@

error_R�T>?

learning_rate_1먷6���I       6%�	�<�_���A�*;


total_lossE��@

error_R�L?

learning_rate_1먷6X���I       6%�	X��_���A�*;


total_lossM��@

error_R��S?

learning_rate_1먷6z>�I       6%�	V��_���A�*;


total_lossq4�@

error_R��6?

learning_rate_1먷6(��RI       6%�	�_���A�*;


total_loss���@

error_R��Y?

learning_rate_1먷6-�V�I       6%�	�a�_���A�*;


total_loss�.#A

error_R,�a?

learning_rate_1먷6&¬�I       6%�	���_���A�*;


total_lossv��@

error_R�\N?

learning_rate_1먷6�/j4I       6%�	Z�_���A�*;


total_lossE�n@

error_R�Y?

learning_rate_1먷6Pk��I       6%�	�V�_���A�*;


total_loss�=�@

error_R�G?

learning_rate_1먷6�b�qI       6%�	���_���A�*;


total_loss�D�@

error_R�}E?

learning_rate_1먷6�,bmI       6%�	���_���A�*;


total_lossV,�@

error_RZ�P?

learning_rate_1먷6j��9I       6%�	�.�_���A�*;


total_loss���@

error_Rl�7?

learning_rate_1먷6��I       6%�	�v�_���A�*;


total_loss��A

error_R�i??

learning_rate_1먷6$�I       6%�	���_���A�*;


total_losss�@

error_ReRU?

learning_rate_1먷6�?��I       6%�	I�_���A�*;


total_loss6t�@

error_R�Z?

learning_rate_1먷6zEI       6%�	ZF�_���A�*;


total_loss�.A

error_R�H>?

learning_rate_1먷6��/�I       6%�	i��_���A�*;


total_lossmP�@

error_R4�^?

learning_rate_1먷6{��DI       6%�	���_���A�*;


total_loss���@

error_R�H?

learning_rate_1먷6�๖I       6%�	,�_���A�*;


total_loss�[�@

error_R�M?

learning_rate_1먷6���I       6%�	Fu�_���A�*;


total_loss�d@

error_R.�G?

learning_rate_1먷6���uI       6%�	���_���A�*;


total_lossaaK@

error_R�<?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss�V�@

error_Rh�V?

learning_rate_1먷6�ФkI       6%�	�[�_���A�*;


total_lossq�@

error_R�K?

learning_rate_1먷6���I       6%�	��_���A�*;


total_loss��@

error_R��W?

learning_rate_1먷6G��I       6%�	 ��_���A�*;


total_lossm��@

error_R��<?

learning_rate_1먷6TWr>I       6%�	�4�_���A�*;


total_loss���@

error_Rc??

learning_rate_1먷6@��xI       6%�	��_���A�*;


total_loss��@

error_Rj�N?

learning_rate_1먷6oI�uI       6%�	w��_���A�*;


total_loss�˦@

error_RX�T?

learning_rate_1먷6����I       6%�	��_���A�*;


total_loss�n@

error_R��O?

learning_rate_1먷6n�f�I       6%�	a�_���A�*;


total_loss<�@

error_R{�P?

learning_rate_1먷6�<��I       6%�	^��_���A�*;


total_loss�9�@

error_R8L?

learning_rate_1먷6fd�I       6%�	x,�_���A�*;


total_lossƫ@

error_R,�D?

learning_rate_1먷6X�m�I       6%�	�t�_���A�*;


total_loss-/w@

error_R��@?

learning_rate_1먷6�Ѓ�I       6%�	a��_���A�*;


total_loss�Ǟ@

error_R�M??

learning_rate_1먷6=���I       6%�	U�_���A�*;


total_loss��@

error_Rܼ_?

learning_rate_1먷6+}{�I       6%�	�K�_���A�*;


total_loss��@

error_R,�e?

learning_rate_1먷6�~�ZI       6%�	���_���A�*;


total_loss_x�@

error_R÷`?

learning_rate_1먷6q���I       6%�	��_���A�*;


total_loss�W�@

error_R
R?

learning_rate_1먷65��I       6%�	�%�_���A�*;


total_lossA��@

error_R�VJ?

learning_rate_1먷6$�I       6%�	r�_���A�*;


total_loss{��@

error_R�
`?

learning_rate_1먷6̶��I       6%�	���_���A�*;


total_loss�s�@

error_R�\?

learning_rate_1먷6���=I       6%�	R)�_���A�*;


total_loss�>w@

error_R��T?

learning_rate_1먷6�n�I       6%�	jq�_���A�*;


total_loss̣�@

error_Rf�T?

learning_rate_1먷6.�eEI       6%�	��_���A�*;


total_loss�q�@

error_R<�A?

learning_rate_1먷6'2T`I       6%�		��_���A�*;


total_loss�)�@

error_RQ?

learning_rate_1먷6�vŜI       6%�	�C�_���A�*;


total_loss�^�@

error_R�6J?

learning_rate_1먷6O��I       6%�	���_���A�*;


total_loss嬼@

error_RCP?

learning_rate_1먷6�a.tI       6%�	��_���A�*;


total_losso�@

error_R�J?

learning_rate_1먷6�"�I       6%�	�<�_���A�*;


total_lossM|�@

error_R�K?

learning_rate_1먷6l��I       6%�	��_���A�*;


total_loss�l�@

error_R��L?

learning_rate_1먷62��HI       6%�	��_���A�*;


total_lossn�@

error_R�~[?

learning_rate_1먷6��c�I       6%�	{
 `���A�*;


total_loss��@

error_Rl?^?

learning_rate_1먷6k���I       6%�	~O `���A�*;


total_loss�g�@

error_R��P?

learning_rate_1먷60�
MI       6%�	֒ `���A�*;


total_loss�T�@

error_R��[?

learning_rate_1먷6%�dFI       6%�	�� `���A�*;


total_lossz��@

error_R$�H?

learning_rate_1먷6g� 6I       6%�	s `���A�*;


total_lossƊ�@

error_R�3<?

learning_rate_1먷6�fI       6%�	�h`���A�*;


total_loss��@

error_R��P?

learning_rate_1먷6�r��I       6%�	R�`���A�*;


total_loss�V�@

error_R`>??

learning_rate_1먷6����I       6%�	,�`���A�*;


total_loss�=A

error_R��N?

learning_rate_1먷6:��I       6%�	�9`���A�*;


total_lossک�@

error_R[�C?

learning_rate_1먷6�5��I       6%�	T�`���A�*;


total_loss\R�@

error_R��W?

learning_rate_1먷6��JI       6%�	��`���A�*;


total_loss�!�@

error_R�rS?

learning_rate_1먷6�a)�I       6%�	`���A�*;


total_lossϳ�@

error_Rx�C?

learning_rate_1먷6���wI       6%�	�^`���A�*;


total_losssȮ@

error_RL?

learning_rate_1먷6CōI       6%�	��`���A�*;


total_loss�W�@

error_R��F?

learning_rate_1먷6��GI       6%�	��`���A�*;


total_loss�<�@

error_R��F?

learning_rate_1먷6�@K�I       6%�	D`���A�*;


total_loss��@

error_R�L?

learning_rate_1먷6��#I       6%�	�`���A�*;


total_loss4]�@

error_R�`K?

learning_rate_1먷6�&�I       6%�	��`���A�*;


total_loss�@�@

error_R�sJ?

learning_rate_1먷6�Z�XI       6%�	�)`���A�*;


total_loss�.�@

error_R={K?

learning_rate_1먷6��6�I       6%�	�`���A�*;


total_loss?�@

error_R]lT?

learning_rate_1먷6��3I       6%�	��`���A�*;


total_loss �@

error_R��G?

learning_rate_1먷6�k�mI       6%�	�"`���A�*;


total_loss���@

error_R�8P?

learning_rate_1먷6b���I       6%�	�h`���A�*;


total_lossɇ�@

error_RM�=?

learning_rate_1먷6�L�I       6%�	B�`���A�*;


total_loss�A

error_R�5S?

learning_rate_1먷6��7hI       6%�	(�`���A�*;


total_lossZU�@

error_R=ke?

learning_rate_1먷6U0Z�I       6%�	�A`���A�*;


total_loss��w@

error_R�<?

learning_rate_1먷6��3�I       6%�	��`���A�*;


total_lossX��@

error_R=[=?

learning_rate_1먷6�3r�I       6%�	��`���A�*;


total_loss���@

error_R!oN?

learning_rate_1먷6��B=I       6%�	�`���A�*;


total_loss�T�@

error_R��A?

learning_rate_1먷6զ I       6%�	�f`���A�*;


total_loss�?@

error_R�K?

learning_rate_1먷6�]0'I       6%�	l�`���A�*;


total_loss���@

error_R��H?

learning_rate_1먷6��h�I       6%�	��`���A�*;


total_loss�۷@

error_R�3U?

learning_rate_1먷6��{tI       6%�	�4	`���A�*;


total_lossn�@

error_R(�Q?

learning_rate_1먷6s��I       6%�	�	`���A�*;


total_loss��@

error_R��K?

learning_rate_1먷6���II       6%�	��	`���A�*;


total_loss��@

error_R�UM?

learning_rate_1먷6�{I       6%�	�
`���A�*;


total_loss�B�@

error_R�>[?

learning_rate_1먷6|�o�I       6%�	�R
`���A�*;


total_loss�ͤ@

error_R�W?

learning_rate_1먷60�	I       6%�	��
`���A�*;


total_loss�'�@

error_R��S?

learning_rate_1먷6�rŢI       6%�	��
`���A�*;


total_lossO��@

error_R�*_?

learning_rate_1먷63*ښI       6%�	�&`���A�*;


total_loss��@

error_R\�G?

learning_rate_1먷6�W�I       6%�	�s`���A�*;


total_lossj��@

error_R��9?

learning_rate_1먷6RQ��I       6%�	��`���A�*;


total_loss���@

error_RZ2M?

learning_rate_1먷6#�I       6%�	a3`���A�*;


total_loss�|�@

error_R��3?

learning_rate_1먷6R~�WI       6%�	�{`���A�*;


total_loss�Ƅ@

error_RZfR?

learning_rate_1먷6?�W[I       6%�	�`���A�*;


total_lossZ/�@

error_R�<?

learning_rate_1먷6c�/�I       6%�	C%`���A�*;


total_lossF��@

error_RV�X?

learning_rate_1먷6s|I       6%�	�j`���A�*;


total_loss�:�@

error_R�
Q?

learning_rate_1먷6 �aI       6%�	r�`���A�*;


total_lossq�@

error_R�D?

learning_rate_1먷6���I       6%�	\�`���A�*;


total_loss�,�@

error_R��@?

learning_rate_1먷6@MűI       6%�	�4`���A�*;


total_loss��@

error_RvPV?

learning_rate_1먷6Ct`I       6%�	Jw`���A�*;


total_loss;��@

error_R%OT?

learning_rate_1먷6E�*I       6%�	��`���A�*;


total_loss6��@

error_R��5?

learning_rate_1먷6�F\�I       6%�	6`���A�*;


total_loss���@

error_R/aG?

learning_rate_1먷6���I       6%�	�I`���A�*;


total_loss�K�@

error_RaX<?

learning_rate_1먷6�g�I       6%�	��`���A�*;


total_loss�ov@

error_R�.B?

learning_rate_1먷61aI       6%�	��`���A�*;


total_loss.�@

error_R{�>?

learning_rate_1먷6�)MI       6%�	�`���A�*;


total_loss���@

error_RD4G?

learning_rate_1먷6�vW�I       6%�	X`���A�*;


total_lossAG�@

error_R8q<?

learning_rate_1먷6h�M�I       6%�	t�`���A�*;


total_loss@<�@

error_Rq�a?

learning_rate_1먷6�?��I       6%�	3�`���A�*;


total_loss��@

error_R\\_?

learning_rate_1먷6��PI       6%�	�)`���A�*;


total_loss�t�@

error_R�LN?

learning_rate_1먷6tI[�I       6%�	�`���A�*;


total_loss�J�@

error_R�}S?

learning_rate_1먷6ŧӸI       6%�	�`���A�*;


total_lossv��@

error_RʵS?

learning_rate_1먷6�T�0I       6%�	[`���A�*;


total_loss��@

error_R6�G?

learning_rate_1먷6 �	I       6%�	ta`���A�*;


total_loss�@

error_R�]T?

learning_rate_1먷6����I       6%�	?�`���A�*;


total_loss!��@

error_R��[?

learning_rate_1먷6�SabI       6%�	^�`���A�*;


total_loss��@

error_R�`_?

learning_rate_1먷6�ۉI       6%�	�3`���A�*;


total_loss���@

error_R�I?

learning_rate_1먷6�;��I       6%�	�x`���A�*;


total_loss��@

error_R��<?

learning_rate_1먷6Ȩ��I       6%�	��`���A�*;


total_lossw��@

error_R��l?

learning_rate_1먷6��69I       6%�	,`���A�*;


total_loss&��@

error_R��d?

learning_rate_1먷6��I       6%�	^N`���A�*;


total_loss��@

error_R�.E?

learning_rate_1먷6j��I       6%�	k�`���A�*;


total_lossO2�@

error_R�E?

learning_rate_1먷6��,I       6%�	�`���A�*;


total_loss�ˬ@

error_R��N?

learning_rate_1먷6G@I       6%�	 `���A�*;


total_loss�r@

error_R��:?

learning_rate_1먷6d�AI       6%�	�d`���A�*;


total_loss4*�@

error_RJf?

learning_rate_1먷6<`�I       6%�	��`���A�*;


total_lossk�A

error_R=xU?

learning_rate_1먷6��cI       6%�	��`���A�*;


total_lossl�@

error_R��Y?

learning_rate_1먷6�%&JI       6%�	H7`���A�*;


total_loss��@

error_RvnZ?

learning_rate_1먷6o�_I       6%�	��`���A�*;


total_loss�c�@

error_RII?

learning_rate_1먷6(��I       6%�	J�`���A�*;


total_loss���@

error_R��J?

learning_rate_1먷6cХ6I       6%�	�%`���A�*;


total_loss���@

error_R�|C?

learning_rate_1먷6  pI       6%�	do`���A�*;


total_loss-�@

error_R_(M?

learning_rate_1먷6wt��I       6%�	j�`���A�*;


total_loss�4�@

error_RE�V?

learning_rate_1먷6|8لI       6%�	d�`���A�*;


total_lossA��@

error_RB?

learning_rate_1먷6�]~fI       6%�	:`���A�*;


total_loss�%�@

error_RW�T?

learning_rate_1먷6���I       6%�	�`���A�*;


total_losst[o@

error_R)[F?

learning_rate_1먷6���I       6%�		�`���A�*;


total_loss*�|@

error_Rz�L?

learning_rate_1먷6S�/I       6%�	L`���A�*;


total_loss��@

error_RR;?

learning_rate_1먷6
��I       6%�	2L`���A�*;


total_lossZ��@

error_R�jZ?

learning_rate_1먷6��I       6%�	u�`���A�*;


total_loss(��@

error_R�F<?

learning_rate_1먷6b�XI       6%�	��`���A�*;


total_lossͣ�@

error_RE�E?

learning_rate_1먷6%^2�I       6%�	�`���A�*;


total_loss&�m@

error_RhA>?

learning_rate_1먷6�P�]I       6%�	�^`���A�*;


total_loss��A

error_R�S?

learning_rate_1먷6��I       6%�	��`���A�*;


total_loss�@

error_R2�T?

learning_rate_1먷6tfy�I       6%�	n�`���A�*;


total_loss�e�@

error_R�M?

learning_rate_1먷6R�F�I       6%�	�1`���A�*;


total_loss{��@

error_RI?

learning_rate_1먷6�ˢ9I       6%�	�w`���A�*;


total_loss�8�@

error_RsUL?

learning_rate_1먷6X�,I       6%�	=�`���A�*;


total_loss�T�@

error_RM�T?

learning_rate_1먷6_p�uI       6%�	M`���A�*;


total_loss��@

error_RC_G?

learning_rate_1먷6�vL�I       6%�	nQ`���A�*;


total_lossd$�@

error_R1�i?

learning_rate_1먷6���I       6%�	�`���A�*;


total_lossד�@

error_R��T?

learning_rate_1먷6�q��I       6%�	&`���A�*;


total_lossdQ�@

error_R}LI?

learning_rate_1먷6����I       6%�	�N`���A�*;


total_lossqT�@

error_R;WQ?

learning_rate_1먷6��I       6%�	֔`���A�*;


total_loss%G�@

error_R3pN?

learning_rate_1먷6b���I       6%�	j�`���A�*;


total_loss��@

error_R�Y?

learning_rate_1먷6��,�I       6%�	(`���A�*;


total_loss���@

error_RExc?

learning_rate_1먷6¯��I       6%�	�q`���A�*;


total_loss�T�@

error_R�XS?

learning_rate_1먷6 ��#I       6%�	ڵ`���A�*;


total_lossm��@

error_RJcD?

learning_rate_1먷6�Y��I       6%�	��`���A�*;


total_loss궢@

error_R=u9?

learning_rate_1먷6�M^	I       6%�	NA`���A�*;


total_loss�_A

error_R)-F?

learning_rate_1먷6S��I       6%�	�`���A�*;


total_loss�)�@

error_R4iH?

learning_rate_1먷6eL=I       6%�	V�`���A�*;


total_loss�ڰ@

error_R�QK?

learning_rate_1먷60[�I       6%�	=2 `���A�*;


total_loss�#�@

error_R!d?

learning_rate_1먷6�@,RI       6%�	!w `���A�*;


total_loss1h�@

error_R[�h?

learning_rate_1먷6�'G\I       6%�	ü `���A�*;


total_loss�?�@

error_R��R?

learning_rate_1먷6�@�I       6%�	P!`���A�*;


total_loss1�@

error_RŀL?

learning_rate_1먷6-'e�I       6%�	�J!`���A�*;


total_loss�2�@

error_RECA?

learning_rate_1먷6��QrI       6%�	��!`���A�*;


total_loss�˿@

error_R�;V?

learning_rate_1먷6~�RUI       6%�	��!`���A�*;


total_loss���@

error_R�a?

learning_rate_1먷6�PnI       6%�	�"`���A�*;


total_loss���@

error_RX�N?

learning_rate_1먷6�J(aI       6%�	w["`���A�*;


total_loss�;�@

error_R�B?

learning_rate_1먷6�ꌺI       6%�	��"`���A�*;


total_loss��@

error_R�F?

learning_rate_1먷6�z�KI       6%�	\�"`���A�*;


total_loss�ƽ@

error_Rr�G?

learning_rate_1먷6��=I       6%�	m9#`���A�*;


total_lossFfA

error_R�Y`?

learning_rate_1먷6.�]�I       6%�	��#`���A�*;


total_loss�,�@

error_RȈS?

learning_rate_1먷6�$;I       6%�	��#`���A�*;


total_loss��@

error_R�A??

learning_rate_1먷6&�I       6%�	+$`���A�*;


total_loss���@

error_R�IQ?

learning_rate_1먷6��eI       6%�	�J$`���A�*;


total_loss�@

error_RD�N?

learning_rate_1먷6Fns�I       6%�	��$`���A�*;


total_lossFެ@

error_R#dC?

learning_rate_1먷6��L�I       6%�	w�$`���A�*;


total_lossUA

error_RE:L?

learning_rate_1먷6]!�I       6%�	~%`���A�*;


total_loss+6�@

error_R��Q?

learning_rate_1먷6`JF�I       6%�	:a%`���A�*;


total_loss�q�@

error_R��H?

learning_rate_1먷6��$cI       6%�	��%`���A�*;


total_loss�Ӕ@

error_R��M?

learning_rate_1먷60��I       6%�	��%`���A�*;


total_loss���@

error_R;�\?

learning_rate_1먷6Qd��I       6%�	e-&`���A�*;


total_lossj(�@

error_Rŏ@?

learning_rate_1먷6s���I       6%�	[r&`���A�*;


total_lossM<�@

error_R�d>?

learning_rate_1먷6=�bI       6%�	��&`���A�*;


total_loss.�R@

error_R��D?

learning_rate_1먷6[��I       6%�	� '`���A�*;


total_loss�a�@

error_R:�J?

learning_rate_1먷6᱂�I       6%�	�E'`���A�*;


total_loss5�@

error_Rf�??

learning_rate_1먷6V�"I       6%�	��'`���A�*;


total_loss:g�@

error_R�8?

learning_rate_1먷6�D$,I       6%�	��'`���A�*;


total_lossq��@

error_R��K?

learning_rate_1먷6E��[I       6%�	�(`���A�*;


total_loss��A

error_RߝK?

learning_rate_1먷6�aI       6%�	ub(`���A�*;


total_lossf��@

error_R�M?

learning_rate_1먷6��cI       6%�	ڧ(`���A�*;


total_loss$��@

error_R]�W?

learning_rate_1먷6'3dI       6%�	�(`���A�*;


total_lossQ��@

error_R��E?

learning_rate_1먷6F��I       6%�	y/)`���A�*;


total_loss�@

error_R�,c?

learning_rate_1먷6%.I       6%�	�w)`���A�*;


total_lossn��@

error_R�%C?

learning_rate_1먷6�"
I       6%�	�)`���A�*;


total_lossW3�@

error_R �_?

learning_rate_1먷6r[�I       6%�	n*`���A�*;


total_loss`y�@

error_RF}[?

learning_rate_1먷6(���I       6%�	sM*`���A�*;


total_loss*�@

error_R2W?

learning_rate_1먷6�1!�I       6%�	��*`���A�*;


total_loss�A

error_R��_?

learning_rate_1먷6��I       6%�	?�*`���A�*;


total_loss� �@

error_R��N?

learning_rate_1먷6z�2I       6%�	�L+`���A�*;


total_lossA��@

error_R�XV?

learning_rate_1먷6�6��I       6%�	j�+`���A�*;


total_loss�D�@

error_R��X?

learning_rate_1먷6�l,I       6%�	��+`���A�*;


total_loss���@

error_R�N?

learning_rate_1먷6����I       6%�	1M,`���A�*;


total_loss忴@

error_RTY7?

learning_rate_1먷6�=�I       6%�	i�,`���A�*;


total_lossk�A

error_R}�R?

learning_rate_1먷6�$UI       6%�	3�,`���A�*;


total_lossV�@

error_RN�H?

learning_rate_1먷6݂�I       6%�	"D-`���A�*;


total_lossR��@

error_Rd�U?

learning_rate_1먷6b���I       6%�	f�-`���A�*;


total_lossr#�@

error_R&DH?

learning_rate_1먷6J�ZVI       6%�	"�-`���A�*;


total_lossٜ@

error_R��K?

learning_rate_1먷6���HI       6%�	�2.`���A�*;


total_loss=B�@

error_Rr�A?

learning_rate_1먷6��?0I       6%�	�.`���A�*;


total_loss�.�@

error_R)gL?

learning_rate_1먷6)T�I       6%�	��.`���A�*;


total_loss��@

error_R�xB?

learning_rate_1먷6o
�I       6%�	4B/`���A�*;


total_loss��@

error_R@NP?

learning_rate_1먷6)J�I       6%�	?�/`���A�*;


total_lossMo�@

error_R�kH?

learning_rate_1먷6ƞ[I       6%�	x�/`���A�*;


total_loss�߃@

error_R�9?

learning_rate_1먷6�ڢ�I       6%�	{A0`���A�*;


total_lossdof@

error_R�D?

learning_rate_1먷6/�\�I       6%�	��0`���A�*;


total_loss��A

error_R� >?

learning_rate_1먷6w��I       6%�	�0`���A�*;


total_loss2��@

error_R��L?

learning_rate_1먷6��32I       6%�	�:1`���A�*;


total_loss�	�@

error_R!�B?

learning_rate_1먷6�[��I       6%�	ڈ1`���A�*;


total_loss��@

error_R{'[?

learning_rate_1먷6D�S�I       6%�	��1`���A�*;


total_loss���@

error_R
M?

learning_rate_1먷6LA�I       6%�	�02`���A�*;


total_loss��@

error_R@_K?

learning_rate_1먷6G�bI       6%�	Yt2`���A�*;


total_loss�h�@

error_Rf�i?

learning_rate_1먷6��Y�I       6%�	0�2`���A�*;


total_loss���@

error_R�KY?

learning_rate_1먷6�$|I       6%�	��2`���A�*;


total_lossA�@

error_R}@??

learning_rate_1먷6�*�I       6%�	8D3`���A�*;


total_lossA=�@

error_R(}E?

learning_rate_1먷6�0FBI       6%�	5�3`���A�*;


total_loss�k�@

error_R�O?

learning_rate_1먷6N��sI       6%�	�3`���A�*;


total_loss�i�@

error_Ra�N?

learning_rate_1먷6ێ]I       6%�	["4`���A�*;


total_loss��@

error_R�J?

learning_rate_1먷6�0�I       6%�	�m4`���A�*;


total_lossc��@

error_R)�8?

learning_rate_1먷6O��OI       6%�	��4`���A�*;


total_loss_��@

error_R�H?

learning_rate_1먷6q\�I       6%�	A 5`���A�*;


total_loss���@

error_RҼL?

learning_rate_1먷6ra�PI       6%�	TC5`���A�*;


total_loss ��@

error_R41Q?

learning_rate_1먷6⎠�I       6%�	�5`���A�*;


total_lossW�@

error_R=�T?

learning_rate_1먷6���dI       6%�	v�5`���A�*;


total_loss��@

error_R6�E?

learning_rate_1먷6=�vI       6%�	?6`���A�*;


total_loss�V�@

error_R�J?

learning_rate_1먷6�!�I       6%�	8Z6`���A�*;


total_loss��@

error_R�]?

learning_rate_1먷6�Z��I       6%�	��6`���A�*;


total_losss��@

error_RŕE?

learning_rate_1먷6���CI       6%�	��6`���A�*;


total_lossEy�@

error_R17\?

learning_rate_1먷6VD1`I       6%�	W$7`���A�*;


total_loss�m�@

error_RZH?

learning_rate_1먷6X߰I       6%�	�e7`���A�*;


total_loss;�u@

error_R��;?

learning_rate_1먷6g�{�I       6%�	�7`���A�*;


total_lossݙ@

error_R�c?

learning_rate_1먷6K�,�I       6%�	R�7`���A�*;


total_loss�@

error_R��Z?

learning_rate_1먷6��~�I       6%�	�38`���A�*;


total_loss�	�@

error_R��B?

learning_rate_1먷6���I       6%�	�u8`���A�*;


total_lossv��@

error_RDU?

learning_rate_1먷6�d�#I       6%�	��8`���A�*;


total_loss�{@

error_RXwK?

learning_rate_1먷6V]W�I       6%�	�8`���A�*;


total_loss\��@

error_R��H?

learning_rate_1먷6���I       6%�	�=9`���A�*;


total_loss,�@

error_R*�Q?

learning_rate_1먷6�O�I       6%�	��9`���A�*;


total_loss�~�@

error_R�JQ?

learning_rate_1먷6�z�I       6%�	��9`���A�*;


total_loss�0�@

error_R2�a?

learning_rate_1먷6�3GI       6%�	g:`���A�*;


total_loss�}�@

error_R�HF?

learning_rate_1먷6��gI       6%�	�H:`���A�*;


total_loss��@

error_R�uZ?

learning_rate_1먷6�G�^I       6%�	̍:`���A�*;


total_loss�@

error_Rl�N?

learning_rate_1먷6��qqI       6%�	�:`���A�*;


total_lossT�%A

error_Rt4?

learning_rate_1먷6D�YI       6%�	y;`���A�*;


total_loss��@

error_R{qT?

learning_rate_1먷6YhI       6%�	�X;`���A�*;


total_loss�(�@

error_R�%A?

learning_rate_1먷6*?��I       6%�	T�;`���A�*;


total_loss6�<@

error_RG?

learning_rate_1먷6�a��I       6%�	��;`���A�*;


total_lossL��@

error_R�iH?

learning_rate_1먷6	��I       6%�	�)<`���A�*;


total_lossC߮@

error_R��W?

learning_rate_1먷6n�*I       6%�	�m<`���A�*;


total_loss,�@

error_R$�<?

learning_rate_1먷6W4��I       6%�	��<`���A�*;


total_loss/e�@

error_R.nG?

learning_rate_1먷6I�C�I       6%�	�=`���A�*;


total_lossS��@

error_R��R?

learning_rate_1먷6#(��I       6%�	�W=`���A�*;


total_loss���@

error_R�A?

learning_rate_1먷6xg�`I       6%�	��=`���A�*;


total_lossi��@

error_R)�V?

learning_rate_1먷6�|fsI       6%�	��=`���A�*;


total_loss�
�@

error_RM�a?

learning_rate_1먷6��nI       6%�	 4>`���A�*;


total_loss���@

error_Rl�Y?

learning_rate_1먷6K?x�I       6%�	��>`���A�*;


total_loss3F@

error_R�}J?

learning_rate_1먷6��AI       6%�	�>`���A�*;


total_loss���@

error_R*Z7?

learning_rate_1먷6W]�wI       6%�	)?`���A�*;


total_loss�\�@

error_R_tB?

learning_rate_1먷6��ӡI       6%�	�q?`���A�*;


total_loss�RA

error_R�KH?

learning_rate_1먷6��2'I       6%�	��?`���A�*;


total_loss���@

error_R��X?

learning_rate_1먷6ݵ� I       6%�	�?`���A�*;


total_loss��@

error_R�)K?

learning_rate_1먷6ل�I       6%�	-a@`���A�*;


total_losseù@

error_R<eK?

learning_rate_1먷6ߺB�I       6%�	��@`���A�*;


total_loss��@

error_R�wM?

learning_rate_1먷6}Q#5I       6%�	��@`���A�*;


total_loss��@

error_R�g?

learning_rate_1먷6���/I       6%�	/LA`���A�*;


total_loss�K�@

error_R��5?

learning_rate_1먷6�)�I       6%�	g�A`���A�*;


total_lossD0�@

error_R%%e?

learning_rate_1먷6�f�I       6%�	��A`���A�*;


total_loss��@

error_Rv�@?

learning_rate_1먷6���I       6%�	e!B`���A�*;


total_loss��q@

error_R�PN?

learning_rate_1먷6���I       6%�	]iB`���A�*;


total_loss=��@

error_Rw�Q?

learning_rate_1먷6r�Q�I       6%�	�B`���A�*;


total_loss��@

error_R!/J?

learning_rate_1먷6�\UI       6%�	��B`���A�*;


total_loss��@

error_R,�J?

learning_rate_1먷6��G�I       6%�	8;C`���A�*;


total_loss:��@

error_RڔB?

learning_rate_1먷6׾��I       6%�	6�C`���A�*;


total_lossD��@

error_Rl�Y?

learning_rate_1먷6,�]I       6%�	��C`���A�*;


total_lossTď@

error_R�M?

learning_rate_1먷64|�yI       6%�	 D`���A�*;


total_loss?�@

error_Rt5[?

learning_rate_1먷6)a�I       6%�	�SD`���A�*;


total_loss�K�@

error_RX?

learning_rate_1먷6KZ�I       6%�	��D`���A�*;


total_loss�#�@

error_R�C?

learning_rate_1먷6�u�)I       6%�	��D`���A�*;


total_lossݠ�@

error_R��@?

learning_rate_1먷6;)�rI       6%�	Z0E`���A�*;


total_lossï�@

error_R�vX?

learning_rate_1먷6̭+�I       6%�	�xE`���A�*;


total_loss�j�@

error_Riv\?

learning_rate_1먷6QA-ZI       6%�	�E`���A�*;


total_loss?͋@

error_RO�Z?

learning_rate_1먷6�"�3I       6%�	�F`���A�*;


total_lossC�@

error_R��E?

learning_rate_1먷6ZT��I       6%�	GEF`���A�*;


total_loss�B�@

error_R�{T?

learning_rate_1먷6Qm�~I       6%�	�F`���A�*;


total_lossa�A

error_R�B?

learning_rate_1먷6�f7�I       6%�	��F`���A�*;


total_loss���@

error_R�:Y?

learning_rate_1먷6�~�I       6%�	fG`���A�*;


total_loss���@

error_R,�D?

learning_rate_1먷6��I       6%�	��G`���A�*;


total_lossrO�@

error_R��R?

learning_rate_1먷6S ��I       6%�	T�G`���A�*;


total_loss$��@

error_R�wN?

learning_rate_1먷6A��I       6%�	�H`���A�*;


total_lossj�@

error_RE�D?

learning_rate_1먷6ѷ]�I       6%�	V`H`���A�*;


total_loss�4�@

error_R��N?

learning_rate_1먷6�Z<I       6%�	��H`���A�*;


total_lossv�`@

error_R��T?

learning_rate_1먷6*��I       6%�	a�H`���A�*;


total_loss��@

error_R�cJ?

learning_rate_1먷6b-�I       6%�	�*I`���A�*;


total_loss88�@

error_R|BS?

learning_rate_1먷6�J�uI       6%�	�mI`���A�*;


total_loss�&�@

error_R�P?

learning_rate_1먷6f��I       6%�	��I`���A�*;


total_lossn�@

error_R f[?

learning_rate_1먷6�,�I       6%�	�I`���A�*;


total_loss&��@

error_Rx0b?

learning_rate_1먷6��ElI       6%�	t9J`���A�*;


total_loss=P�@

error_R�L?

learning_rate_1먷6Q��yI       6%�	֋J`���A�*;


total_loss�S�@

error_R8=P?

learning_rate_1먷6BusXI       6%�	}�J`���A�*;


total_lossx^�@

error_R%Tg?

learning_rate_1먷6�m�I       6%�	8LK`���A�*;


total_loss�C�@

error_R�K?

learning_rate_1먷6�c�I       6%�	ΗK`���A�*;


total_lossWd�@

error_R�TE?

learning_rate_1먷6��`qI       6%�	�K`���A�*;


total_loss�;�@

error_R#�L?

learning_rate_1먷6Y��\I       6%�	TQL`���A�*;


total_loss���@

error_R�\?

learning_rate_1먷6X�RI       6%�	S�L`���A�*;


total_loss��@

error_R�xW?

learning_rate_1먷66��I       6%�	4M`���A�*;


total_loss�n�@

error_R��A?

learning_rate_1먷6��BI       6%�	��M`���A�*;


total_loss���@

error_R�HF?

learning_rate_1먷6�}��I       6%�	��M`���A�*;


total_loss�z@

error_R�FI?

learning_rate_1먷6����I       6%�	\6N`���A�*;


total_losssbg@

error_R[�J?

learning_rate_1먷6�u�*I       6%�	ݝN`���A�*;


total_lossx��@

error_R�^?

learning_rate_1먷6�h1I       6%�	��N`���A�*;


total_loss�Y�@

error_R�F?

learning_rate_1먷6���I       6%�	)O`���A�*;


total_loss���@

error_R�S?

learning_rate_1먷6Q��I       6%�	�kO`���A�*;


total_loss�C�@

error_RIX?

learning_rate_1먷6Y�6�I       6%�	
�O`���A�*;


total_loss2�@

error_R�5E?

learning_rate_1먷6���I       6%�	�O`���A�*;


total_loss�*�@

error_R�!R?

learning_rate_1먷6:���I       6%�	%?P`���A�*;


total_loss/C�@

error_R�U?

learning_rate_1먷6���I       6%�	�P`���A�*;


total_lossR��@

error_R��W?

learning_rate_1먷6DۡI       6%�	��P`���A�*;


total_loss��@

error_Ra�T?

learning_rate_1먷6F�WbI       6%�	�Q`���A�*;


total_lossV=A

error_RvM]?

learning_rate_1먷6@�K�I       6%�	wYQ`���A�*;


total_loss�3�@

error_R��P?

learning_rate_1먷6Ʌ�I       6%�	��Q`���A�*;


total_lossh��@

error_R��K?

learning_rate_1먷6���I       6%�	��Q`���A�*;


total_loss�# A

error_R�c[?

learning_rate_1먷6"�#�I       6%�	0R`���A�*;


total_loss�ֹ@

error_R7Q?

learning_rate_1먷6���I       6%�	�vR`���A�*;


total_loss���@

error_RA�\?

learning_rate_1먷6����I       6%�	��R`���A�*;


total_loss~��@

error_R��Z?

learning_rate_1먷6|���I       6%�	1S`���A�*;


total_lossR�@

error_Rv�S?

learning_rate_1먷6�3UdI       6%�	|CS`���A�*;


total_lossv~�@

error_R�eS?

learning_rate_1먷6f���I       6%�	�S`���A�*;


total_loss&`�@

error_R�U?

learning_rate_1먷6���I       6%�	|�S`���A�*;


total_loss���@

error_RƆ>?

learning_rate_1먷6�UI       6%�		T`���A�*;


total_loss��@

error_RqVU?

learning_rate_1먷6a':�I       6%�	S\T`���A�*;


total_lossIm�@

error_R6�4?

learning_rate_1먷6�$�SI       6%�	$�T`���A�*;


total_loss���@

error_R��G?

learning_rate_1먷6��I       6%�	-�T`���A�*;


total_loss
f�@

error_R�WB?

learning_rate_1먷6�=I       6%�	�+U`���A�*;


total_loss��@

error_R�[;?

learning_rate_1먷6�I       6%�	�pU`���A�*;


total_loss�x�@

error_R7�C?

learning_rate_1먷6�Z_�I       6%�	J�U`���A�*;


total_loss>W�@

error_RJ�H?

learning_rate_1먷6@�`�I       6%�	@�U`���A�*;


total_lossI��@

error_R�N?

learning_rate_1먷6D��I       6%�	�>V`���A�*;


total_lossF��@

error_Rn�W?

learning_rate_1먷6���I       6%�	�V`���A�*;


total_loss���@

error_R�pB?

learning_rate_1먷6I�I       6%�	��V`���A�*;


total_lossO�@

error_R�6?

learning_rate_1먷6��W�I       6%�	�W`���A�*;


total_loss�~�@

error_R�M?

learning_rate_1먷6�7��I       6%�	�^W`���A�*;


total_loss�o@

error_R�QY?

learning_rate_1먷6J�'4I       6%�	��W`���A�*;


total_loss��@

error_R&HP?

learning_rate_1먷6%d�;I       6%�	T�W`���A�*;


total_loss�D�@

error_Rw3G?

learning_rate_1먷6���9I       6%�	V1X`���A�*;


total_loss�(�@

error_R��T?

learning_rate_1먷66�TMI       6%�	sX`���A�*;


total_loss��@

error_R�eU?

learning_rate_1먷6Gin�I       6%�	�X`���A�*;


total_losse�@

error_R�S[?

learning_rate_1먷6C-��I       6%�	SY`���A�*;


total_loss�n�@

error_R�Sf?

learning_rate_1먷6�	�I       6%�	�EY`���A�*;


total_loss���@

error_R�5?

learning_rate_1먷6�|\I       6%�	ׅY`���A�*;


total_loss���@

error_R�jJ?

learning_rate_1먷6I       6%�	d�Y`���A�*;


total_loss0�@

error_R7	J?

learning_rate_1먷62)��I       6%�	�
Z`���A�*;


total_loss�~@

error_R��Q?

learning_rate_1먷6�q9�I       6%�	QPZ`���A�*;


total_losst�@

error_R�K?

learning_rate_1먷6�A.I       6%�	H�Z`���A�*;


total_loss�M�@

error_R�[?

learning_rate_1먷6�}VI       6%�	
[`���A�*;


total_lossJP�@

error_Rt�@?

learning_rate_1먷6G��+I       6%�	�H[`���A�*;


total_lossӜ�@

error_RE�I?

learning_rate_1먷6��-I       6%�	:�[`���A�*;


total_loss��A

error_RŸH?

learning_rate_1먷6���I       6%�	4�[`���A�*;


total_losseJ�@

error_R�\M?

learning_rate_1먷63j��I       6%�	pC\`���A�*;


total_lossS>�@

error_R�B?

learning_rate_1먷6��YhI       6%�	�\`���A�*;


total_loss}��@

error_Rq�P?

learning_rate_1먷6Ėf�I       6%�	��\`���A�*;


total_losscՑ@

error_R&>S?

learning_rate_1먷6�ᐮI       6%�		E]`���A�*;


total_loss2�R@

error_R�N?

learning_rate_1먷6��SFI       6%�	f�]`���A�*;


total_lossw*�@

error_RT�E?

learning_rate_1먷6�t�I       6%�	��]`���A�*;


total_lossVc�@

error_R�Z?

learning_rate_1먷6Y7ٜI       6%�	�'^`���A�*;


total_lossLQ�@

error_R��H?

learning_rate_1먷6(�� I       6%�	t^`���A�*;


total_loss3�@

error_R�aR?

learning_rate_1먷6���I       6%�	��^`���A�*;


total_loss`0A

error_R
�I?

learning_rate_1먷6����I       6%�	��^`���A�*;


total_lossIr�@

error_R��@?

learning_rate_1먷6��'WI       6%�	�A_`���A�*;


total_loss���@

error_R��D?

learning_rate_1먷6����I       6%�	5�_`���A�*;


total_loss���@

error_R�V?

learning_rate_1먷6vx��I       6%�		�_`���A�*;


total_loss�2s@

error_R�2:?

learning_rate_1먷6nU�I       6%�	�``���A�*;


total_loss���@

error_RO�J?

learning_rate_1먷6���I       6%�	�^``���A�*;


total_loss|��@

error_R�;6?

learning_rate_1먷6N���I       6%�	{�``���A�*;


total_loss���@

error_R��D?

learning_rate_1먷6Ȳo�I       6%�	��``���A�*;


total_loss��@

error_RsfF?

learning_rate_1먷6��I       6%�		1a`���A�*;


total_lossW��@

error_R�R?

learning_rate_1먷6�Kc�I       6%�	�ta`���A�*;


total_losso��@

error_R�d?

learning_rate_1먷6���I       6%�	l�a`���A�*;


total_loss�=�@

error_ReSQ?

learning_rate_1먷61�(�I       6%�	s�a`���A�*;


total_loss:��@

error_R
E?

learning_rate_1먷6�y�I       6%�	T<b`���A�*;


total_lossC8�@

error_R=�W?

learning_rate_1먷6m��I       6%�	�b`���A�*;


total_lossT�A

error_R#O?

learning_rate_1먷6��I       6%�	��b`���A�*;


total_lossR�@

error_R��U?

learning_rate_1먷6��fI       6%�	�c`���A�*;


total_loss�4�@

error_R��K?

learning_rate_1먷6�}�2I       6%�	isc`���A�*;


total_lossZ��@

error_R�TH?

learning_rate_1먷6�8ƤI       6%�	x�c`���A�*;


total_lossЋ@

error_R�.N?

learning_rate_1먷6Z\˓I       6%�	e�c`���A�*;


total_loss�_A

error_R4�G?

learning_rate_1먷6��DI       6%�	Ed`���A�*;


total_loss��@

error_R\�@?

learning_rate_1먷65l��I       6%�	��d`���A�*;


total_loss-��@

error_R
N?

learning_rate_1먷6�X<�I       6%�	��d`���A�*;


total_lossq��@

error_R�3??

learning_rate_1먷6G��;I       6%�	>e`���A�*;


total_loss6��@

error_R��Z?

learning_rate_1먷6�a�I       6%�	[�e`���A�*;


total_loss�@

error_R��D?

learning_rate_1먷6M2�8I       6%�	��e`���A�*;


total_lossf��@

error_R�t\?

learning_rate_1먷6h�:I       6%�	�f`���A�*;


total_loss2�@

error_Rd7?

learning_rate_1먷6��]I       6%�	3Zf`���A�*;


total_lossm|�@

error_R,9C?

learning_rate_1먷6��WzI       6%�	ݝf`���A�*;


total_loss���@

error_R&�<?

learning_rate_1먷6�-ӿI       6%�	E�f`���A�*;


total_loss�Z�@

error_R�Q?

learning_rate_1먷6%�6iI       6%�	�^g`���A�*;


total_loss��@

error_R]�J?

learning_rate_1먷6�{e�I       6%�	B�g`���A�*;


total_loss�N`@

error_Rof?

learning_rate_1먷6�z�I       6%�	��g`���A�*;


total_loss���@

error_Ri+U?

learning_rate_1먷6/k*I       6%�	 2h`���A�*;


total_loss���@

error_R}O?

learning_rate_1먷6]7NEI       6%�	r�h`���A�*;


total_loss��v@

error_R,=F?

learning_rate_1먷6�V�FI       6%�	2�h`���A�*;


total_loss�v�@

error_R
"K?

learning_rate_1먷6��rI       6%�	�'i`���A�*;


total_loss\�@

error_R4�X?

learning_rate_1먷6�Y[8I       6%�	>ki`���A�*;


total_loss8��@

error_R�[?

learning_rate_1먷6����I       6%�	�i`���A�*;


total_lossl1A

error_R��??

learning_rate_1먷6ĞZI       6%�	��i`���A�*;


total_loss{ر@

error_R�5W?

learning_rate_1먷6#�!�I       6%�	:j`���A�*;


total_loss�A

error_R��T?

learning_rate_1먷6�Zi�I       6%�	3~j`���A�*;


total_lossx��@

error_Ro�M?

learning_rate_1먷6Uv��I       6%�	�j`���A�*;


total_lossG�@

error_R�M?

learning_rate_1먷6X�uI       6%�	�
k`���A�*;


total_loss�Ծ@

error_RpX?

learning_rate_1먷6x܊I       6%�	)Pk`���A�*;


total_loss$��@

error_R&�R?

learning_rate_1먷6��?`I       6%�	u�k`���A�*;


total_loss�͓@

error_R�O?

learning_rate_1먷6�i;I       6%�	I�k`���A�*;


total_lossĿ�@

error_Rf�O?

learning_rate_1먷6,��I       6%�	� l`���A�*;


total_loss�"A

error_R;T?

learning_rate_1먷6��
�I       6%�	�hl`���A�*;


total_loss���@

error_R��K?

learning_rate_1먷6G��I       6%�	�l`���A�*;


total_lossy��@

error_R�??

learning_rate_1먷6�[
I       6%�	`m`���A�*;


total_loss���@

error_R-�Q?

learning_rate_1먷6���
I       6%�	�am`���A�*;


total_lossV|�@

error_R|sM?

learning_rate_1먷6.�ϮI       6%�	�m`���A�*;


total_loss;2�@

error_RԅW?

learning_rate_1먷6�]0BI       6%�	5�m`���A�*;


total_loss�@

error_Rs\S?

learning_rate_1먷6n��I       6%�	�2n`���A�*;


total_loss�5�@

error_RcE?

learning_rate_1먷6K��I       6%�	vn`���A�*;


total_lossΈ�@

error_R�7>?

learning_rate_1먷6n�vI       6%�	�n`���A�*;


total_lossN��@

error_R�IA?

learning_rate_1먷6���I       6%�	9�n`���A�*;


total_lossL��@

error_Ro4X?

learning_rate_1먷6z�1I       6%�	�=o`���A�*;


total_lossF�@

error_R� B?

learning_rate_1먷6v;h!I       6%�	ރo`���A�*;


total_loss�CA

error_R.�G?

learning_rate_1먷6:I       6%�	3�o`���A�*;


total_lossR��@

error_R��S?

learning_rate_1먷6�ֺ�I       6%�	�p`���A�*;


total_loss�A�@

error_R1�K?

learning_rate_1먷6���"I       6%�	�Rp`���A�*;


total_loss�Y�@

error_R�GK?

learning_rate_1먷6� I       6%�	#�p`���A�*;


total_lossL��@

error_R�W?

learning_rate_1먷6�r��I       6%�	E�p`���A�*;


total_lossx�Y@

error_R��G?

learning_rate_1먷6�Q�I       6%�	�&q`���A�*;


total_lossA�@

error_RA�L?

learning_rate_1먷6�wZaI       6%�	Wjq`���A�*;


total_loss���@

error_R�N?

learning_rate_1먷6�}�;I       6%�	b�q`���A�*;


total_loss)7�@

error_Rמ]?

learning_rate_1먷6⬗:I       6%�	3�q`���A�*;


total_loss���@

error_R�^?

learning_rate_1먷6�K��I       6%�	kDr`���A�*;


total_loss��@

error_R
�U?

learning_rate_1먷6�v�I       6%�	��r`���A�*;


total_loss���@

error_R:EV?

learning_rate_1먷6
~��I       6%�	P�r`���A�*;


total_loss���@

error_R��V?

learning_rate_1먷6�_�I       6%�	�s`���A�*;


total_loss��@

error_RZ7F?

learning_rate_1먷6��DjI       6%�	1Ys`���A�*;


total_loss���@

error_RJA?

learning_rate_1먷6���I       6%�	%�s`���A�*;


total_loss餺@

error_R_M?

learning_rate_1먷6hY�I       6%�	a�s`���A�*;


total_lossj��@

error_R��;?

learning_rate_1먷6���I       6%�	�'t`���A�*;


total_loss%�o@

error_R6jK?

learning_rate_1먷6�וI       6%�	lt`���A�*;


total_lossQ��@

error_R@�S?

learning_rate_1먷6@8m�I       6%�	��t`���A�*;


total_lossh��@

error_Ri�R?

learning_rate_1먷6X���I       6%�	W�t`���A�*;


total_loss�@�@

error_R�E?

learning_rate_1먷6�#I       6%�	�>u`���A�*;


total_lossW�A

error_R�yL?

learning_rate_1먷6��w�I       6%�	�u`���A�*;


total_loss6�@

error_R��Q?

learning_rate_1먷6���WI       6%�	\�u`���A�*;


total_loss]��@

error_R1�E?

learning_rate_1먷6fI       6%�	,0v`���A�*;


total_lossQ~�@

error_R�OS?

learning_rate_1먷64�YI       6%�	�~v`���A�*;


total_loss`C�@

error_R��T?

learning_rate_1먷6W8�bI       6%�	_�v`���A�*;


total_loss4�@

error_R{zF?

learning_rate_1먷6v�6JI       6%�	�w`���A�*;


total_lossӲ�@

error_R�#>?

learning_rate_1먷6ON�\I       6%�	�Ow`���A�*;


total_loss(�@

error_R�G?

learning_rate_1먷6��I       6%�	p�w`���A�*;


total_lossv�@

error_R�AX?

learning_rate_1먷6%)�fI       6%�	��w`���A�*;


total_loss8�-A

error_R�A?

learning_rate_1먷6���I       6%�	 x`���A�*;


total_losse�U@

error_R�b8?

learning_rate_1먷6\��,I       6%�	�dx`���A�*;


total_loss11�@

error_RQ�_?

learning_rate_1먷6���I       6%�	��x`���A�*;


total_loss�ߥ@

error_R�5T?

learning_rate_1먷6ە�I       6%�	��x`���A�*;


total_loss��@

error_R��K?

learning_rate_1먷6�\�0I       6%�	56y`���A�*;


total_loss�h�@

error_Rq�Z?

learning_rate_1먷6��l�I       6%�	�xy`���A�*;


total_lossz/�@

error_R�CE?

learning_rate_1먷6H8,I       6%�	�y`���A�*;


total_loss<�@

error_R.�<?

learning_rate_1먷6�ő�I       6%�	m�y`���A�*;


total_loss��@

error_R�O?

learning_rate_1먷6��I       6%�	�>z`���A�*;


total_lossa��@

error_ROyd?

learning_rate_1먷6�j�{I       6%�	<�z`���A�*;


total_loss�y�@

error_R��M?

learning_rate_1먷6� s�I       6%�	P�z`���A�*;


total_loss�\�@

error_R�y`?

learning_rate_1먷6�X�I       6%�	B8{`���A�*;


total_loss9�@

error_R@ug?

learning_rate_1먷6^���I       6%�	�}{`���A�*;


total_loss�@

error_R��R?

learning_rate_1먷6�GΚI       6%�	��{`���A�*;


total_loss���@

error_RsC?

learning_rate_1먷6��dI       6%�	�|`���A�*;


total_loss�u�@

error_R@SS?

learning_rate_1먷6��-LI       6%�	O[|`���A�*;


total_loss��@

error_RzgJ?

learning_rate_1먷6N)��I       6%�	A�|`���A�*;


total_loss�0�@

error_R&b`?

learning_rate_1먷6��$�I       6%�	�	}`���A�*;


total_loss���@

error_R�=?

learning_rate_1먷6G0u�I       6%�	�S}`���A�*;


total_losss��@

error_R�3U?

learning_rate_1먷6���I       6%�	ʙ}`���A�*;


total_loss_Hx@

error_R�,?

learning_rate_1먷61�NI       6%�	��}`���A�*;


total_loss@

error_R�Q?

learning_rate_1먷6����I       6%�	J'~`���A�*;


total_loss�r�@

error_R��>?

learning_rate_1먷6P��I       6%�	0s~`���A�*;


total_loss���@

error_R� R?

learning_rate_1먷6U��I       6%�	��~`���A�*;


total_loss��@

error_R��K?

learning_rate_1먷6���I       6%�	��~`���A�*;


total_loss�
�@

error_RÜ=?

learning_rate_1먷6>�65I       6%�	J`���A�*;


total_loss�D�@

error_RrW?

learning_rate_1먷6Ы��I       6%�	��`���A�*;


total_loss�k�@

error_R}E?

learning_rate_1먷6���;I       6%�	F�`���A�*;


total_loss?��@

error_R!�Z?

learning_rate_1먷6��|I       6%�	�.�`���A�*;


total_loss���@

error_Rŕ`?

learning_rate_1먷67�6I       6%�	q�`���A�*;


total_loss\�@

error_RMX>?

learning_rate_1먷6o���I       6%�	���`���A�*;


total_loss��@

error_RMQJ?

learning_rate_1먷6�z�I       6%�	��`���A�*;


total_loss�>�@

error_RCQ?

learning_rate_1먷6�J<]I       6%�	6h�`���A�*;


total_loss���@

error_R��F?

learning_rate_1먷6��(I       6%�	³�`���A�*;


total_loss4��@

error_R��B?

learning_rate_1먷6��3�I       6%�	��`���A�*;


total_loss���@

error_R�J?

learning_rate_1먷6�I��I       6%�	�@�`���A�*;


total_loss�k�@

error_R��I?

learning_rate_1먷6����I       6%�	ↂ`���A�*;


total_loss�o�@

error_RH4[?

learning_rate_1먷6��-I       6%�	˂`���A�*;


total_loss���@

error_Rs�T?

learning_rate_1먷6�y�I       6%�	��`���A�*;


total_lossq��@

error_RmJR?

learning_rate_1먷6�vT`I       6%�	V�`���A�*;


total_loss���@

error_R��a?

learning_rate_1먷6��I       6%�	잃`���A�*;


total_loss���@

error_R��D?

learning_rate_1먷6 b�I       6%�	��`���A�*;


total_loss�ξ@

error_R\
H?

learning_rate_1먷6z#l�I       6%�	o+�`���A�*;


total_lossܙ�@

error_R_�A?

learning_rate_1먷6`�%I       6%�	�t�`���A�*;


total_loss*��@

error_RW�]?

learning_rate_1먷69��zI       6%�	z��`���A�*;


total_loss� �@

error_R
xG?

learning_rate_1먷6//HI       6%�	�
�`���A�*;


total_loss�@

error_R��M?

learning_rate_1먷67�V�I       6%�	vT�`���A�*;


total_loss�L�@

error_R/�W?

learning_rate_1먷6�>��I       6%�	��`���A�*;


total_loss���@

error_R&�X?

learning_rate_1먷6��dOI       6%�	��`���A�*;


total_loss��@

error_RR�V?

learning_rate_1먷6/=�I       6%�	:%�`���A�*;


total_loss�R�@

error_R[|Q?

learning_rate_1먷6�g�I       6%�	�i�`���A�*;


total_loss!�q@

error_R�_?

learning_rate_1먷6!�5I       6%�	'��`���A�*;


total_loss���@

error_RwD?

learning_rate_1먷6�$@�I       6%�	��`���A�*;


total_loss�'�@

error_R�U?

learning_rate_1먷69�I       6%�	]3�`���A�*;


total_loss���@

error_R�\?

learning_rate_1먷69�E�I       6%�	�w�`���A�*;


total_loss*ŧ@

error_R:!R?

learning_rate_1먷6�@^I       6%�	ž�`���A�*;


total_loss��@

error_R_�o?

learning_rate_1먷6���pI       6%�	,�`���A�*;


total_lossL��@

error_R<I?

learning_rate_1먷6��@I       6%�	O�`���A�*;


total_lossLk�@

error_R�cZ?

learning_rate_1먷6N�)aI       6%�	5��`���A�*;


total_loss8��@

error_R�BS?

learning_rate_1먷6*D��I       6%�	y�`���A�*;


total_loss���@

error_RS?

learning_rate_1먷6�-��I       6%�	a+�`���A�*;


total_lossC��@

error_R��V?

learning_rate_1먷6���I       6%�	�u�`���A�*;


total_loss���@

error_Rr%W?

learning_rate_1먷6s��I       6%�	�`���A�*;


total_loss�n`@

error_R�iW?

learning_rate_1먷6#wHI       6%�	��`���A�*;


total_lossHZ�@

error_R�VC?

learning_rate_1먷6�vK�I       6%�	Y�`���A�*;


total_loss2#k@

error_R��I?

learning_rate_1먷6�~�I       6%�	,��`���A�*;


total_loss��
A

error_R�U?

learning_rate_1먷6b�^�I       6%�	��`���A�*;


total_losssҔ@

error_R!�I?

learning_rate_1먷6Ȁ�5I       6%�	b6�`���A�*;


total_loss���@

error_RxNJ?

learning_rate_1먷6_�?�I       6%�	c{�`���A�*;


total_loss��@

error_R�A?

learning_rate_1먷6]%_6I       6%�	��`���A�*;


total_lossi�f@

error_R}�??

learning_rate_1먷6�$|*I       6%�	_�`���A�*;


total_loss���@

error_R �b?

learning_rate_1먷6u\��I       6%�	J�`���A�*;


total_lossf��@

error_RnqM?

learning_rate_1먷6�Jq�I       6%�	䒌`���A�*;


total_loss�	�@

error_R��L?

learning_rate_1먷6�G�sI       6%�	��`���A�*;


total_lossq��@

error_RC�;?

learning_rate_1먷6	R^QI       6%�	W�`���A�*;


total_loss4�kA

error_R�2D?

learning_rate_1먷65��^I       6%�	韍`���A�*;


total_loss��@

error_R?k>?

learning_rate_1먷67��I       6%�	��`���A�*;


total_loss&_�@

error_Rq�P?

learning_rate_1먷6�5U<I       6%�	��`���A�*;


total_loss���@

error_RW/8?

learning_rate_1먷6�n��I       6%�	9�`���A�*;


total_lossH�V@

error_R�F8?

learning_rate_1먷6�H�"I       6%�	v��`���A�*;


total_lossѪ�@

error_R��L?

learning_rate_1먷6m���I       6%�	ˑ`���A�*;


total_loss�@

error_RnlJ?

learning_rate_1먷6
$��I       6%�	��`���A�*;


total_lossx)�@

error_R��??

learning_rate_1먷6�'>�I       6%�	�Z�`���A�*;


total_lossP �@

error_RFY?

learning_rate_1먷6� MhI       6%�	��`���A�*;


total_loss�s�@

error_R(&[?

learning_rate_1먷6^I       6%�	W�`���A�*;


total_lossJ5�@

error_R
�S?

learning_rate_1먷6���I       6%�	=*�`���A�*;


total_loss~DA

error_RW�Q?

learning_rate_1먷6��%I       6%�	�m�`���A�*;


total_loss�D�@

error_R�X?

learning_rate_1먷6�M�}I       6%�	괓`���A�*;


total_loss��@

error_R,�Z?

learning_rate_1먷6E(��I       6%�	���`���A�*;


total_loss|�@

error_RZ^??

learning_rate_1먷6HF�oI       6%�	�B�`���A�*;


total_loss�5�@

error_R��\?

learning_rate_1먷6�k�bI       6%�	҈�`���A�*;


total_loss�L�@

error_Rq�D?

learning_rate_1먷6zTlI       6%�	6Ӕ`���A�*;


total_lossM}@

error_R��V?

learning_rate_1먷6���!I       6%�	O!�`���A�*;


total_lossߝ�@

error_R��W?

learning_rate_1먷6ƚ;I       6%�	6n�`���A�*;


total_lossݣ�@

error_R�Z?

learning_rate_1먷6HN̘I       6%�	]��`���A�*;


total_loss��@

error_R2E?

learning_rate_1먷6�]�I       6%�	*��`���A�*;


total_loss��@

error_R�kX?

learning_rate_1먷6�I6�I       6%�	P>�`���A�*;


total_loss"�@

error_R�S?

learning_rate_1먷6�t,{I       6%�	i��`���A�*;


total_loss�@

error_R�J?

learning_rate_1먷6G*��I       6%�	ږ`���A�*;


total_loss�@

error_R�,T?

learning_rate_1먷6Iv��I       6%�	+�`���A�*;


total_loss��@

error_R�O?

learning_rate_1먷6���VI       6%�	�s�`���A�*;


total_loss���@

error_R�D?

learning_rate_1먷6�I
�I       6%�	蹗`���A�*;


total_losso��@

error_RH�K?

learning_rate_1먷6��I       6%�	I�`���A�*;


total_loss�@

error_RheA?

learning_rate_1먷6i�DI       6%�	�]�`���A�*;


total_loss1Լ@

error_R��V?

learning_rate_1먷6�cEI       6%�	"��`���A�*;


total_loss^mA

error_R*�N?

learning_rate_1먷6G�+MI       6%�	k�`���A�*;


total_loss�޴@

error_R=�m?

learning_rate_1먷6Z	��I       6%�	�.�`���A�*;


total_loss̝�@

error_R{�R?

learning_rate_1먷6���I       6%�	:p�`���A�*;


total_lossf��@

error_Rh�C?

learning_rate_1먷6�ȠcI       6%�	7��`���A�*;


total_lossE�X@

error_R7�H?

learning_rate_1먷6E��rI       6%�	���`���A�*;


total_loss;Y�@

error_RI'>?

learning_rate_1먷6Ǌ�I       6%�	�;�`���A�*;


total_loss1�@

error_RͨC?

learning_rate_1먷6]A�I       6%�	C��`���A�*;


total_loss)��@

error_R߯A?

learning_rate_1먷6T�6�I       6%�	(ƚ`���A�*;


total_loss�5�@

error_ROM?

learning_rate_1먷6΄69I       6%�	$�`���A�*;


total_lossָ�@

error_R �T?

learning_rate_1먷6�q��I       6%�	VV�`���A�*;


total_loss�F�@

error_R�A?

learning_rate_1먷6g�>I       6%�	��`���A�*;


total_lossqK�@

error_R�BZ?

learning_rate_1먷6�rܞI       6%�	p�`���A�*;


total_loss���@

error_Rl�N?

learning_rate_1먷6'��I       6%�	AO�`���A�*;


total_loss�|�@

error_R%@J?

learning_rate_1먷6Y�I       6%�	���`���A�*;


total_lossv�A

error_R�sg?

learning_rate_1먷6Íw�I       6%�	*�`���A�*;


total_loss�mA

error_R&�S?

learning_rate_1먷6��kLI       6%�	�m�`���A�*;


total_loss��@

error_R�b=?

learning_rate_1먷6�
��I       6%�	@��`���A�*;


total_lossS��@

error_R�Y?

learning_rate_1먷6(Ki�I       6%�	��`���A�*;


total_loss�l�@

error_R�Z?

learning_rate_1먷6b���I       6%�	�I�`���A�*;


total_loss��z@

error_RO�B?

learning_rate_1먷6�m�I       6%�	Y��`���A�*;


total_loss?��@

error_R��T?

learning_rate_1먷6�]I       6%�	��`���A�*;


total_lossd�@

error_R�k?

learning_rate_1먷6(֗�I       6%�	�H�`���A�*;


total_loss5�@

error_R�Q?

learning_rate_1먷6����I       6%�	đ�`���A�*;


total_losso��@

error_R)T?

learning_rate_1먷6H��I       6%�	���`���A�*;


total_loss7�@

error_R$AR?

learning_rate_1먷6P�x~I       6%�	oF�`���A�*;


total_lossS�@

error_R��6?

learning_rate_1먷6��II       6%�	ꉠ`���A�*;


total_lossZ��@

error_R{�T?

learning_rate_1먷6WjG�I       6%�	�̠`���A�*;


total_loss귣@

error_Rҧ>?

learning_rate_1먷6AI       6%�	=�`���A�*;


total_lossS�A

error_R�M?

learning_rate_1먷6&�'{I       6%�	��`���A�*;


total_loss*�@

error_RqV?

learning_rate_1먷6(��aI       6%�	С`���A�*;


total_loss��@

error_RwQ?

learning_rate_1먷6yo�;I       6%�	��`���A�*;


total_loss-��@

error_R�E?

learning_rate_1먷6(�N�I       6%�	/Z�`���A�*;


total_loss@

error_R��K?

learning_rate_1먷6ޓ'eI       6%�	F��`���A�*;


total_lossT֦@

error_R DX?

learning_rate_1먷6�.�I       6%�	@�`���A�*;


total_loss���@

error_R��P?

learning_rate_1먷6Q��I       6%�	�\�`���A�*;


total_loss��A

error_Rc�N?

learning_rate_1먷6� ��I       6%�	_��`���A�*;


total_loss�RA

error_R�8\?

learning_rate_1먷6���I       6%�	J�`���A�*;


total_loss���@

error_R�-F?

learning_rate_1먷6X�-I       6%�	p4�`���A�*;


total_loss--�@

error_R�;?

learning_rate_1먷6W^�8I       6%�	�y�`���A�*;


total_loss�ђ@

error_R��`?

learning_rate_1먷6HeI       6%�	ѽ�`���A�*;


total_loss�H�@

error_R��H?

learning_rate_1먷6��x0I       6%�	��`���A�*;


total_loss�v�@

error_R�-Y?

learning_rate_1먷6?��I       6%�	�N�`���A�*;


total_loss���@

error_R.@I?

learning_rate_1먷6y}¤I       6%�	��`���A�*;


total_lossR �@

error_R1�J?

learning_rate_1먷6EТ%I       6%�	��`���A�*;


total_loss��@

error_R<�N?

learning_rate_1먷6b��I       6%�	�%�`���A�*;


total_lossJ�h@

error_Ra�H?

learning_rate_1먷6y�I       6%�	2n�`���A�*;


total_loss#Y�@

error_R�kT?

learning_rate_1먷6x'0I       6%�	l��`���A�*;


total_loss�_�@

error_R_�Z?

learning_rate_1먷6��3I       6%�	��`���A�*;


total_loss;�@

error_R�N?

learning_rate_1먷6"��6I       6%�	LN�`���A�*;


total_loss$��@

error_R7G?

learning_rate_1먷6�pZI       6%�	���`���A�*;


total_lossF��@

error_R��E?

learning_rate_1먷6=Z�LI       6%�	kէ`���A�*;


total_lossS�c@

error_R�Iq?

learning_rate_1먷6�7�I       6%�	��`���A�*;


total_lossqj�@

error_R��;?

learning_rate_1먷6i.�I       6%�	�\�`���A�*;


total_loss3�@

error_R_�I?

learning_rate_1먷6��^�I       6%�	���`���A�*;


total_loss�V�@

error_R�6^?

learning_rate_1먷6;l�yI       6%�	A�`���A�*;


total_loss/�@

error_Rx�Z?

learning_rate_1먷6�s�rI       6%�	p)�`���A�*;


total_lossvL�@

error_R\lU?

learning_rate_1먷6?v��I       6%�	n�`���A�*;


total_loss
��@

error_Rx/L?

learning_rate_1먷6�'3I       6%�	���`���A�*;


total_lossZ׸@

error_R�L?

learning_rate_1먷6ٰO�I       6%�		��`���A�*;


total_loss��@

error_R,�J?

learning_rate_1먷6��
I       6%�	�:�`���A�*;


total_loss$�{@

error_R��M?

learning_rate_1먷63B�I       6%�	���`���A�*;


total_loss�+�@

error_R��M?

learning_rate_1먷6��)�I       6%�	Ū`���A�*;


total_loss�,�@

error_R�M?

learning_rate_1먷6�OKI       6%�	��`���A�*;


total_loss�ԩ@

error_R��Y?

learning_rate_1먷6�3��I       6%�	<N�`���A�*;


total_loss$��@

error_R��I?

learning_rate_1먷6��I       6%�	3��`���A�*;


total_lossz��@

error_R�x7?

learning_rate_1먷6W�5gI       6%�	K׫`���A�*;


total_lossj$<A

error_R��J?

learning_rate_1먷6����I       6%�	�`���A�*;


total_loss�ɭ@

error_Rh`C?

learning_rate_1먷6~�|I       6%�	�`�`���A� *;


total_loss:��@

error_R��L?

learning_rate_1먷6��xqI       6%�	A��`���A� *;


total_loss�1�@

error_Rx�M?

learning_rate_1먷6�"r�I       6%�	"�`���A� *;


total_loss�z@

error_Ro�@?

learning_rate_1먷6�ٺ�I       6%�	�j�`���A� *;


total_loss��@

error_R;@S?

learning_rate_1먷6�O�4I       6%�	'��`���A� *;


total_losse��@

error_R<M?

learning_rate_1먷6�.�I       6%�	���`���A� *;


total_loss���@

error_Rx5R?

learning_rate_1먷6�_�GI       6%�	�;�`���A� *;


total_lossA��@

error_R��M?

learning_rate_1먷6�`�I       6%�	��`���A� *;


total_loss`)�@

error_R[�;?

learning_rate_1먷6�'wI       6%�	�®`���A� *;


total_loss	A

error_Rԃa?

learning_rate_1먷6L�#�I       6%�	(�`���A� *;


total_loss)~�@

error_R�K?

learning_rate_1먷6:�I       6%�	�K�`���A� *;


total_loss���@

error_RڕP?

learning_rate_1먷6n�m|I       6%�	N��`���A� *;


total_loss�n�@

error_R��\?

learning_rate_1먷6=�I*I       6%�	lݯ`���A� *;


total_loss�!�@

error_R�ng?

learning_rate_1먷6�I�I       6%�	p$�`���A� *;


total_loss!��@

error_R��N?

learning_rate_1먷6:�I       6%�	�j�`���A� *;


total_loss���@

error_RZ;Q?

learning_rate_1먷6�=I       6%�	���`���A� *;


total_loss���@

error_R��[?

learning_rate_1먷6't��I       6%�	��`���A� *;


total_lossMK�@

error_R��J?

learning_rate_1먷6Dh��I       6%�	�:�`���A� *;


total_loss?R�@

error_R%�C?

learning_rate_1먷6�(h_I       6%�	r��`���A� *;


total_loss�ר@

error_R�0Y?

learning_rate_1먷6�6ݬI       6%�	�ɱ`���A� *;


total_lossxb�@

error_R��E?

learning_rate_1먷6�"�I       6%�	u�`���A� *;


total_loss�F�@

error_R�T?

learning_rate_1먷6����I       6%�	R�`���A� *;


total_lossļ�@

error_R��J?

learning_rate_1먷6�!��I       6%�	���`���A� *;


total_loss�1�@

error_R��<?

learning_rate_1먷63�rI       6%�	�߲`���A� *;


total_lossY�@

error_R��G?

learning_rate_1먷6'�ivI       6%�	P$�`���A� *;


total_loss��@

error_R`Z?

learning_rate_1먷6j�SI       6%�	@f�`���A� *;


total_lossO��@

error_R�IS?

learning_rate_1먷6�7I       6%�	���`���A� *;


total_loss�U�@

error_R�^<?

learning_rate_1먷6/cjI       6%�	��`���A� *;


total_loss� A

error_Rlmg?

learning_rate_1먷6�zMI       6%�	+�`���A� *;


total_lossy�@

error_R8�F?

learning_rate_1먷6��hI       6%�	�n�`���A� *;


total_lossl�@

error_RA�8?

learning_rate_1먷6 ��WI       6%�	J��`���A� *;


total_loss�@

error_RvUD?

learning_rate_1먷6��%+I       6%�	��`���A� *;


total_loss�E�@

error_R�yH?

learning_rate_1먷6�y��I       6%�	H7�`���A� *;


total_lossH߲@

error_R�yR?

learning_rate_1먷6����I       6%�	���`���A� *;


total_loss
&A

error_R;�B?

learning_rate_1먷6��dSI       6%�	�̵`���A� *;


total_loss ��@

error_R�XS?

learning_rate_1먷6�n<�I       6%�	#�`���A� *;


total_loss��@

error_R�X?

learning_rate_1먷6����I       6%�	S�`���A� *;


total_loss�O�@

error_R}
>?

learning_rate_1먷6��9AI       6%�	���`���A� *;


total_lossdA�@

error_R�J?

learning_rate_1먷6-`8I       6%�	�׶`���A� *;


total_loss�؜@

error_R��E?

learning_rate_1먷6ݣ`I       6%�	7�`���A� *;


total_lossɦ�@

error_R��`?

learning_rate_1먷6�,I       6%�	�^�`���A� *;


total_loss��@

error_R�kM?

learning_rate_1먷626I       6%�	���`���A� *;


total_loss��@

error_R)�5?

learning_rate_1먷6޺��I       6%�	��`���A� *;


total_loss,��@

error_RO�N?

learning_rate_1먷6�%-I       6%�	g5�`���A� *;


total_loss�4�@

error_R2	G?

learning_rate_1먷6�5�cI       6%�	Յ�`���A� *;


total_loss�݀@

error_Ra�O?

learning_rate_1먷64(�I       6%�	�θ`���A� *;


total_loss�+�@

error_Rj:P?

learning_rate_1먷6<���I       6%�	�E�`���A� *;


total_loss٭@

error_R�8N?

learning_rate_1먷6����I       6%�	���`���A� *;


total_loss��@

error_R��J?

learning_rate_1먷6�5�I       6%�	��`���A� *;


total_loss��@

error_Rc�M?

learning_rate_1먷6�E0I       6%�	cu�`���A� *;


total_lossN{�@

error_R@RS?

learning_rate_1먷6�<_�I       6%�	޺`���A� *;


total_lossz��@

error_R�3>?

learning_rate_1먷64�!6I       6%�	#$�`���A� *;


total_loss�V�@

error_R@�R?

learning_rate_1먷6��0I       6%�	�o�`���A� *;


total_loss��@

error_RCDZ?

learning_rate_1먷6��ХI       6%�	�ڻ`���A� *;


total_loss�z�@

error_R��T?

learning_rate_1먷6�;�I       6%�	�`���A� *;


total_lossN�A

error_R�nP?

learning_rate_1먷61��I       6%�	�^�`���A� *;


total_lossE־@

error_R��V?

learning_rate_1먷6��[,I       6%�	9��`���A� *;


total_lossN��@

error_R_�W?

learning_rate_1먷6���I       6%�	E�`���A� *;


total_lossqK�@

error_R�+Y?

learning_rate_1먷6��8�I       6%�	 R�`���A� *;


total_lossm��@

error_R(xN?

learning_rate_1먷6�m�I       6%�	���`���A� *;


total_lossr��@

error_R��^?

learning_rate_1먷6��RqI       6%�	�޽`���A� *;


total_loss���@

error_R��R?

learning_rate_1먷6Yd)�I       6%�	�#�`���A� *;


total_loss�3�@

error_R4�J?

learning_rate_1먷6aX�LI       6%�	.r�`���A� *;


total_lossIK}@

error_R88?

learning_rate_1먷6~|�PI       6%�	���`���A� *;


total_loss�r�@

error_R�??

learning_rate_1먷6u)�I       6%�	�,�`���A� *;


total_loss$��@

error_RZ^?

learning_rate_1먷6_�I       6%�	"��`���A� *;


total_loss�8�@

error_R�<R?

learning_rate_1먷6���,I       6%�	�ǿ`���A� *;


total_loss{�@

error_R�-O?

learning_rate_1먷6VTOI       6%�	I!�`���A� *;


total_loss@�s@

error_R�N?

learning_rate_1먷6�u��I       6%�	�{�`���A� *;


total_loss	3�@

error_R��Y?

learning_rate_1먷6x�I       6%�	��`���A� *;


total_losslc�@

error_RZ�G?

learning_rate_1먷6;q<�I       6%�	�`���A� *;


total_lossw�@

error_R$HQ?

learning_rate_1먷64�m�I       6%�	K�`���A� *;


total_lossǷ�@

error_R��Q?

learning_rate_1먷68ΡI       6%�	���`���A� *;


total_loss}�@

error_R��??

learning_rate_1먷6}���I       6%�	9��`���A� *;


total_loss��@

error_R�??

learning_rate_1먷6,�t0I       6%�	�6�`���A� *;


total_loss�L�@

error_RLL?

learning_rate_1먷6�I       6%�	_{�`���A� *;


total_loss���@

error_R]�V?

learning_rate_1먷6�+W�I       6%�	���`���A� *;


total_loss<�A

error_R�P?

learning_rate_1먷6����I       6%�	�`���A� *;


total_loss* �@

error_R�;?

learning_rate_1먷6���I       6%�	�H�`���A� *;


total_lossv��@

error_R`(G?

learning_rate_1먷6
{۠I       6%�	|��`���A� *;


total_loss���@

error_R ;A?

learning_rate_1먷6c,h�I       6%�	���`���A� *;


total_loss ��@

error_R��D?

learning_rate_1먷6+�L�I       6%�	�`���A� *;


total_loss�W�@

error_R<�c?

learning_rate_1먷6��2�I       6%�	mY�`���A� *;


total_loss6�@

error_R��F?

learning_rate_1먷6-%�)I       6%�	^��`���A� *;


total_loss�^A

error_R�`?

learning_rate_1먷6�MuI       6%�	x��`���A� *;


total_loss���@

error_R)vd?

learning_rate_1먷6��eI       6%�	
"�`���A� *;


total_losstd�@

error_R��U?

learning_rate_1먷6?HV�I       6%�	i�`���A� *;


total_loss�@

error_R�,X?

learning_rate_1먷6aYb�I       6%�	(��`���A� *;


total_losst޻@

error_R��M?

learning_rate_1먷68��I       6%�	� �`���A� *;


total_loss �@

error_RlD?

learning_rate_1먷6���`I       6%�	�Q�`���A� *;


total_loss&��@

error_Ri�W?

learning_rate_1먷6����I       6%�	Ș�`���A� *;


total_loss�Az@

error_R�GT?

learning_rate_1먷6��%I       6%�	q��`���A� *;


total_loss��@

error_R�%V?

learning_rate_1먷6��z�I       6%�	(:�`���A� *;


total_lossS��@

error_Ra�>?

learning_rate_1먷6��K�I       6%�	���`���A� *;


total_lossn�|@

error_RhM?

learning_rate_1먷6�ǎI       6%�	?��`���A� *;


total_loss\\�@

error_R��E?

learning_rate_1먷6�!�I       6%�	�`���A� *;


total_loss���@

error_R�P?

learning_rate_1먷6�5�I       6%�	�d�`���A� *;


total_loss�^�@

error_R�5?

learning_rate_1먷6cE�YI       6%�	A��`���A� *;


total_loss�@

error_R�O?

learning_rate_1먷66�rlI       6%�	���`���A� *;


total_loss*ѷ@

error_RoS?

learning_rate_1먷6�Dc�I       6%�	n2�`���A� *;


total_loss���@

error_R�'W?

learning_rate_1먷6��d%I       6%�	�{�`���A� *;


total_lossӃ�@

error_R��D?

learning_rate_1먷6\YI       6%�	���`���A� *;


total_loss�d�@

error_R��V?

learning_rate_1먷6��{�I       6%�	#-�`���A� *;


total_lossO�@

error_R�dV?

learning_rate_1먷6�k}I       6%�	>q�`���A� *;


total_lossst�@

error_R��J?

learning_rate_1먷6	y6.I       6%�	 ��`���A� *;


total_loss!�@

error_R�YF?

learning_rate_1먷6��=II       6%�	J��`���A� *;


total_loss��WA

error_R*B?

learning_rate_1먷6vm/�I       6%�	9�`���A� *;


total_loss���@

error_R/�Y?

learning_rate_1먷6��n�I       6%�	�|�`���A� *;


total_loss!��@

error_R��D?

learning_rate_1먷6���nI       6%�	��`���A� *;


total_lossS8A

error_R�`N?

learning_rate_1먷6@�!I       6%�	��`���A� *;


total_loss�K�@

error_R��H?

learning_rate_1먷6�Ѡ.I       6%�	�V�`���A� *;


total_lossۼ�@

error_R�C?

learning_rate_1먷6���I       6%�	��`���A� *;


total_loss$G�@

error_R�<?

learning_rate_1먷6��EI       6%�	IT�`���A� *;


total_loss��@

error_R�R?

learning_rate_1먷6��_JI       6%�	��`���A� *;


total_loss7s~@

error_RWAG?

learning_rate_1먷6%_I       6%�	9&�`���A� *;


total_lossX�@

error_R�{4?

learning_rate_1먷6`�ȆI       6%�	s�`���A� *;


total_loss�7�@

error_R�S?

learning_rate_1먷65�I       6%�	���`���A� *;


total_losss� A

error_R�MY?

learning_rate_1먷6u�9I       6%�	s�`���A� *;


total_lossV.�@

error_R��V?

learning_rate_1먷6N��DI       6%�	�X�`���A� *;


total_lossp�@

error_R�@?

learning_rate_1먷6�r��I       6%�	P��`���A� *;


total_loss3��@

error_R�E?

learning_rate_1먷6ގdI       6%�	���`���A� *;


total_loss?�@

error_R�uK?

learning_rate_1먷6U
��I       6%�	7H�`���A� *;


total_loss�/�@

error_RO�T?

learning_rate_1먷6$a�I       6%�	E��`���A� *;


total_lossr8�@

error_R`�=?

learning_rate_1먷6|H~�I       6%�	�
�`���A� *;


total_loss�6�@

error_RR?

learning_rate_1먷6����I       6%�	�X�`���A� *;


total_loss�_�@

error_R[JS?

learning_rate_1먷6f%jfI       6%�	���`���A� *;


total_loss�@

error_R�;L?

learning_rate_1먷6Q̰AI       6%�	I,�`���A� *;


total_loss6|�@

error_R[�j?

learning_rate_1먷6�+nI       6%�	��`���A� *;


total_loss��@

error_R1�P?

learning_rate_1먷6��$?I       6%�	���`���A�!*;


total_loss���@

error_R�ac?

learning_rate_1먷6ը�I       6%�	h�`���A�!*;


total_loss���@

error_R�S?

learning_rate_1먷6���I       6%�	�n�`���A�!*;


total_loss�A

error_Rq�M?

learning_rate_1먷6\��SI       6%�	ܹ�`���A�!*;


total_loss�[A

error_R��l?

learning_rate_1먷6�-��I       6%�	��`���A�!*;


total_loss}��@

error_RM�H?

learning_rate_1먷6�`��I       6%�	�r�`���A�!*;


total_loss��@

error_RJ�C?

learning_rate_1먷6�{��I       6%�	(��`���A�!*;


total_loss�W�@

error_R`�H?

learning_rate_1먷6���I       6%�	�	�`���A�!*;


total_loss� �@

error_R}�@?

learning_rate_1먷6`F�I       6%�	vU�`���A�!*;


total_loss��@

error_R&�\?

learning_rate_1먷6@ ��I       6%�	|��`���A�!*;


total_loss!��@

error_R�%\?

learning_rate_1먷6��!I       6%�	��`���A�!*;


total_loss��@

error_RF<?

learning_rate_1먷6l\I       6%�	=6�`���A�!*;


total_lossÿ�@

error_R��R?

learning_rate_1먷6k��aI       6%�	-��`���A�!*;


total_loss֖�@

error_R��Q?

learning_rate_1먷6� ��I       6%�	���`���A�!*;


total_loss�x�@

error_RHpa?

learning_rate_1먷6��p	I       6%�	�+�`���A�!*;


total_loss�}@

error_RFS?

learning_rate_1먷6XzI       6%�	Fq�`���A�!*;


total_loss���@

error_RJ'I?

learning_rate_1먷6-*$�I       6%�	޴�`���A�!*;


total_loss�ʰ@

error_R��N?

learning_rate_1먷6V)�I       6%�	%��`���A�!*;


total_loss��@

error_R�+P?

learning_rate_1먷6h�G�I       6%�	;�`���A�!*;


total_loss\��@

error_RZ�Y?

learning_rate_1먷6����I       6%�	Ԁ�`���A�!*;


total_losssc�@

error_R)�J?

learning_rate_1먷6� >I       6%�	m��`���A�!*;


total_lossFQ�@

error_RϺS?

learning_rate_1먷6�i�I       6%�	��`���A�!*;


total_losscЬ@

error_R��I?

learning_rate_1먷6v_�I       6%�	uL�`���A�!*;


total_lossᏖ@

error_R��P?

learning_rate_1먷6�DuI       6%�	��`���A�!*;


total_lossμ�@

error_R�C?

learning_rate_1먷6���yI       6%�	\��`���A�!*;


total_loss�g�@

error_R�S?

learning_rate_1먷6���pI       6%�	J:�`���A�!*;


total_loss�N�@

error_R�C?

learning_rate_1먷6܆��I       6%�	і�`���A�!*;


total_loss[��@

error_R��N?

learning_rate_1먷6>Җ�I       6%�	���`���A�!*;


total_loss-�@

error_RC?

learning_rate_1먷6Jِ�I       6%�	�"�`���A�!*;


total_loss&a�@

error_R$�a?

learning_rate_1먷6���jI       6%�	�h�`���A�!*;


total_loss�Y�@

error_R��U?

learning_rate_1먷6/MwI       6%�	��`���A�!*;


total_loss_9�@

error_R�R?

learning_rate_1먷6�SgI       6%�	���`���A�!*;


total_loss
�@

error_R�.?

learning_rate_1먷6��$�I       6%�	a3�`���A�!*;


total_loss5G�@

error_R�|R?

learning_rate_1먷67�~I       6%�	�w�`���A�!*;


total_loss���@

error_R{WO?

learning_rate_1먷6C�8�I       6%�	��`���A�!*;


total_lossnk�@

error_R�Z?

learning_rate_1먷6_\<!I       6%�	�S�`���A�!*;


total_loss���@

error_R��N?

learning_rate_1먷6�1�I       6%�	���`���A�!*;


total_loss�1�@

error_RңD?

learning_rate_1먷6�I��I       6%�	���`���A�!*;


total_loss���@

error_Ra�W?

learning_rate_1먷6� f�I       6%�	{;�`���A�!*;


total_loss��@

error_RRkI?

learning_rate_1먷6T�I       6%�	��`���A�!*;


total_loss_[�@

error_Rq�N?

learning_rate_1먷6G>�I       6%�	���`���A�!*;


total_loss<7�@

error_R`?

learning_rate_1먷6�gI       6%�	h)�`���A�!*;


total_loss^�@

error_RnA9?

learning_rate_1먷6O�	ZI       6%�	9p�`���A�!*;


total_loss��@

error_Rq�L?

learning_rate_1먷6A�/�I       6%�	���`���A�!*;


total_loss}$�@

error_R�"X?

learning_rate_1먷6����I       6%�	���`���A�!*;


total_lossXL�@

error_R�7?

learning_rate_1먷6�);I       6%�	�<�`���A�!*;


total_loss���@

error_R_O?

learning_rate_1먷6Z
�I       6%�	���`���A�!*;


total_loss@�@

error_R_�Y?

learning_rate_1먷6���I       6%�	T��`���A�!*;


total_loss�_�@

error_R�aZ?

learning_rate_1먷6�I       6%�	��`���A�!*;


total_loss���@

error_R_�G?

learning_rate_1먷6����I       6%�	�L�`���A�!*;


total_loss���@

error_R&fV?

learning_rate_1먷6�/�;I       6%�	��`���A�!*;


total_loss �@

error_R6�Q?

learning_rate_1먷6g�3I       6%�	���`���A�!*;


total_loss��@

error_R��O?

learning_rate_1먷6�ݾ�I       6%�	��`���A�!*;


total_loss���@

error_R��T?

learning_rate_1먷6J?�pI       6%�	\`�`���A�!*;


total_lossX٠@

error_R;�E?

learning_rate_1먷6�ީ�I       6%�	/��`���A�!*;


total_lossּ�@

error_R\E?

learning_rate_1먷6�d�I       6%�	���`���A�!*;


total_loss�M�@

error_RJ]c?

learning_rate_1먷6N=!I       6%�	�1�`���A�!*;


total_lossls$A

error_R��L?

learning_rate_1먷6�'�I       6%�	nu�`���A�!*;


total_lossdA�@

error_R�;?

learning_rate_1먷6_k�I       6%�	̽�`���A�!*;


total_loss6@}@

error_R��[?

learning_rate_1먷6$��SI       6%�	p �`���A�!*;


total_lossr��@

error_R� `?

learning_rate_1먷6ՠ�!I       6%�	�A�`���A�!*;


total_loss6(�@

error_R�}I?

learning_rate_1먷62)�I       6%�	��`���A�!*;


total_loss���@

error_R3yO?

learning_rate_1먷6���I       6%�	���`���A�!*;


total_loss�I�@

error_R�H?

learning_rate_1먷6�)�(I       6%�	��`���A�!*;


total_lossf�A

error_R)e]?

learning_rate_1먷6�C�I       6%�	IT�`���A�!*;


total_lossq��@

error_R*WS?

learning_rate_1먷6c���I       6%�	K��`���A�!*;


total_loss��@

error_RŻH?

learning_rate_1먷6n��]I       6%�	���`���A�!*;


total_loss!�@

error_R�U`?

learning_rate_1먷63v-kI       6%�	��`���A�!*;


total_loss:�A

error_R_�X?

learning_rate_1먷6�aI       6%�	^�`���A�!*;


total_loss�#�@

error_R�uD?

learning_rate_1먷6&D��I       6%�	Y��`���A�!*;


total_loss���@

error_RJ�S?

learning_rate_1먷6Hv_�I       6%�	g��`���A�!*;


total_loss�Yk@

error_R�(U?

learning_rate_1먷6WX��I       6%�	�'�`���A�!*;


total_loss���@

error_R�dI?

learning_rate_1먷6V@I       6%�	�i�`���A�!*;


total_lossv��@

error_RT�>?

learning_rate_1먷6�u��I       6%�	հ�`���A�!*;


total_loss���@

error_R@�R?

learning_rate_1먷6�>��I       6%�	c��`���A�!*;


total_lossa�@

error_R��@?

learning_rate_1먷6�=FYI       6%�	�9�`���A�!*;


total_lossk=�@

error_R�!O?

learning_rate_1먷6�?�I       6%�	�~�`���A�!*;


total_loss��@

error_R�\G?

learning_rate_1먷6�-@HI       6%�	���`���A�!*;


total_loss�p�@

error_RJxP?

learning_rate_1먷6h�W/I       6%�	��`���A�!*;


total_loss���@

error_R=KL?

learning_rate_1먷6��I       6%�	MH�`���A�!*;


total_lossL�@

error_Rڈb?

learning_rate_1먷6�1NI       6%�	ˌ�`���A�!*;


total_loss?i�@

error_RUL?

learning_rate_1먷6Ń&I       6%�	���`���A�!*;


total_loss̛�@

error_R�F?

learning_rate_1먷6��jI       6%�	��`���A�!*;


total_loss�]�@

error_RR�L?

learning_rate_1먷6@m�pI       6%�	p`�`���A�!*;


total_loss]!�@

error_RN�X?

learning_rate_1먷6� I       6%�	I��`���A�!*;


total_lossJ��@

error_R�dE?

learning_rate_1먷6ŧ��I       6%�	�`���A�!*;


total_loss��@

error_Ra�I?

learning_rate_1먷6E�N}I       6%�	�X�`���A�!*;


total_loss�@

error_R��M?

learning_rate_1먷6S�-I       6%�	=��`���A�!*;


total_lossle�@

error_R:5G?

learning_rate_1먷6���I       6%�	��`���A�!*;


total_loss��v@

error_R.�S?

learning_rate_1먷6lh*�I       6%�	Xh�`���A�!*;


total_loss�u�@

error_R�M?

learning_rate_1먷6�wOI       6%�	,��`���A�!*;


total_loss�#�@

error_R-$F?

learning_rate_1먷6�l�I       6%�	��`���A�!*;


total_loss�׺@

error_R�$S?

learning_rate_1먷6���I       6%�	1Y�`���A�!*;


total_loss��@

error_R"]?

learning_rate_1먷6����I       6%�	`��`���A�!*;


total_loss3ݒ@

error_R[lJ?

learning_rate_1먷6�[��I       6%�	c��`���A�!*;


total_lossI�@

error_RW�H?

learning_rate_1먷60�rI       6%�	~&�`���A�!*;


total_loss;��@

error_R�J?

learning_rate_1먷6�V�nI       6%�	l�`���A�!*;


total_loss`��@

error_R/KU?

learning_rate_1먷6y=��I       6%�	X��`���A�!*;


total_loss.��@

error_R��B?

learning_rate_1먷6��I       6%�	|��`���A�!*;


total_lossd�@

error_R�F?

learning_rate_1먷6]L9�I       6%�	#8�`���A�!*;


total_lossS��@

error_R��N?

learning_rate_1먷6�}NI       6%�	z�`���A�!*;


total_lossິ@

error_RB`?

learning_rate_1먷6�_7I       6%�	���`���A�!*;


total_loss�`�@

error_RԖ;?

learning_rate_1먷6�+�YI       6%�	���`���A�!*;


total_loss�lKA

error_R.�a?

learning_rate_1먷6jtJ�I       6%�	�r�`���A�!*;


total_losst�p@

error_Rm�K?

learning_rate_1먷6S�I       6%�	��`���A�!*;


total_loss�l�@

error_R�U?

learning_rate_1먷66��I       6%�	Z\�`���A�!*;


total_losse�@

error_R(Y?

learning_rate_1먷6 ��I       6%�	ߨ�`���A�!*;


total_loss��@

error_RP?

learning_rate_1먷6#�6tI       6%�	�`���A�!*;


total_loss%`�@

error_R�E?

learning_rate_1먷6Ƭ�OI       6%�	�i�`���A�!*;


total_losse�@

error_R;�X?

learning_rate_1먷6�~�=I       6%�	ҷ�`���A�!*;


total_loss1��@

error_R�ph?

learning_rate_1먷6%[rI       6%�	��`���A�!*;


total_lossH{�@

error_R}dU?

learning_rate_1먷6��زI       6%�	'O�`���A�!*;


total_loss�l@

error_Rx�C?

learning_rate_1먷64�I       6%�	X��`���A�!*;


total_lossm��@

error_R�OO?

learning_rate_1먷6k�_@I       6%�	�`���A�!*;


total_loss?Jc@

error_R$�I?

learning_rate_1먷6aeI       6%�	Pl�`���A�!*;


total_loss�a�@

error_R��M?

learning_rate_1먷6A�/fI       6%�	½�`���A�!*;


total_loss;�_@

error_R��O?

learning_rate_1먷6ݙ��I       6%�	W*�`���A�!*;


total_lossCu@

error_R�VI?

learning_rate_1먷6F�TI       6%�	�}�`���A�!*;


total_lossT��@

error_RN�F?

learning_rate_1먷6�e��I       6%�	n��`���A�!*;


total_lossf: A

error_R�PV?

learning_rate_1먷6cv��I       6%�	@�`���A�!*;


total_lossZ�@

error_ROZ?

learning_rate_1먷6ʲ�I       6%�	Q[�`���A�!*;


total_loss`��@

error_R.�E?

learning_rate_1먷6�Q��I       6%�	ި�`���A�!*;


total_loss�D�@

error_R�A?

learning_rate_1먷6-�6�I       6%�	���`���A�!*;


total_loss�$�@

error_Rf�U?

learning_rate_1먷6��%�I       6%�	"C�`���A�!*;


total_lossr7�@

error_Rd�J?

learning_rate_1먷6��I�I       6%�	i��`���A�!*;


total_loss)��@

error_R��=?

learning_rate_1먷6��g�I       6%�	f��`���A�!*;


total_loss?��@

error_RNpQ?

learning_rate_1먷6���?I       6%�	�F�`���A�!*;


total_loss���@

error_R�,c?

learning_rate_1먷6A���I       6%�	2��`���A�!*;


total_loss��@

error_RD�W?

learning_rate_1먷6��jhI       6%�	y��`���A�"*;


total_loss��@

error_R%�;?

learning_rate_1먷6��{I       6%�	S<�`���A�"*;


total_loss���@

error_Rh]M?

learning_rate_1먷6=Z�JI       6%�	���`���A�"*;


total_loss�A

error_R*�E?

learning_rate_1먷6��pKI       6%�	���`���A�"*;


total_loss��@

error_R�Y?

learning_rate_1먷6�Z.:I       6%�	{/�`���A�"*;


total_loss&B�@

error_R�1Z?

learning_rate_1먷6�n�I       6%�	o�`���A�"*;


total_lossw:�@

error_R?�<?

learning_rate_1먷6���EI       6%�	���`���A�"*;


total_loss���@

error_R�]K?

learning_rate_1먷6u�VI       6%�	���`���A�"*;


total_loss���@

error_R3Z?

learning_rate_1먷6��XpI       6%�	�5�`���A�"*;


total_loss�-�@

error_R�^H?

learning_rate_1먷6i�"�I       6%�	�y�`���A�"*;


total_loss��@

error_R{W?

learning_rate_1먷6�+��I       6%�	=��`���A�"*;


total_loss���@

error_R4�W?

learning_rate_1먷6����I       6%�	s��`���A�"*;


total_loss&�@

error_R�^?

learning_rate_1먷6��^xI       6%�	@I�`���A�"*;


total_lossY��@

error_RsMM?

learning_rate_1먷6�t��I       6%�	���`���A�"*;


total_loss���@

error_R�W?

learning_rate_1먷6@>6>I       6%�	��`���A�"*;


total_loss]��@

error_R@�K?

learning_rate_1먷6�%z�I       6%�	RN�`���A�"*;


total_loss$��@

error_Rn%>?

learning_rate_1먷6�c��I       6%�	z��`���A�"*;


total_loss&H�@

error_R 3U?

learning_rate_1먷6ݖ��I       6%�	P��`���A�"*;


total_loss�l�@

error_R�i<?

learning_rate_1먷6�r��I       6%�	�4�`���A�"*;


total_loss�`�@

error_R�^?

learning_rate_1먷6 ���I       6%�	�x�`���A�"*;


total_lossOʻ@

error_R��<?

learning_rate_1먷6l�U,I       6%�	c��`���A�"*;


total_lossC��@

error_RdeR?

learning_rate_1먷6�PL�I       6%�	���`���A�"*;


total_lossk�@

error_R�GL?

learning_rate_1먷6R�o�I       6%�	�U�`���A�"*;


total_loss��@

error_R��S?

learning_rate_1먷60���I       6%�	1��`���A�"*;


total_loss��@

error_R��A?

learning_rate_1먷6��ZI       6%�	���`���A�"*;


total_loss�x�@

error_R1N?

learning_rate_1먷6J0��I       6%�	�! a���A�"*;


total_loss%��@

error_R��G?

learning_rate_1먷6g"��I       6%�	&b a���A�"*;


total_losshX�@

error_R_�;?

learning_rate_1먷6VdrI       6%�	B� a���A�"*;


total_loss��@

error_R�<`?

learning_rate_1먷6;�ԝI       6%�	�� a���A�"*;


total_loss��&A

error_R)�b?

learning_rate_1먷6���pI       6%�	$a���A�"*;


total_loss���@

error_RN�S?

learning_rate_1먷6&=��I       6%�	�ea���A�"*;


total_loss�#A

error_R/�W?

learning_rate_1먷6�f|I       6%�	¨a���A�"*;


total_loss���@

error_R�/?

learning_rate_1먷6\FUI       6%�	�a���A�"*;


total_loss[��@

error_R��G?

learning_rate_1먷6�I�I       6%�	�,a���A�"*;


total_loss�}�@

error_R��B?

learning_rate_1먷6�Z4�I       6%�	la���A�"*;


total_loss��@

error_R�PM?

learning_rate_1먷6�0=I       6%�	��a���A�"*;


total_lossֳ�@

error_R�I?

learning_rate_1먷6gZD0I       6%�	��a���A�"*;


total_loss���@

error_R=�I?

learning_rate_1먷6͎�*I       6%�	�-a���A�"*;


total_loss(��@

error_R�Je?

learning_rate_1먷6��UI       6%�	]oa���A�"*;


total_loss�'�@

error_R��U?

learning_rate_1먷6�Jp�I       6%�	�a���A�"*;


total_loss�@

error_R�N?

learning_rate_1먷6��ЍI       6%�	}�a���A�"*;


total_loss���@

error_Ri�??

learning_rate_1먷6�E�I       6%�	�1a���A�"*;


total_loss;c�@

error_R�y_?

learning_rate_1먷61A�mI       6%�	@xa���A�"*;


total_loss�� A

error_R%V?

learning_rate_1먷6-kx�I       6%�	��a���A�"*;


total_loss�B�@

error_R�D?

learning_rate_1먷6g%/+I       6%�	Fa���A�"*;


total_loss	��@

error_RO�W?

learning_rate_1먷6V�)I       6%�	EFa���A�"*;


total_loss� �@

error_R�mM?

learning_rate_1먷6R8gFI       6%�	O�a���A�"*;


total_lossj8�@

error_R�QQ?

learning_rate_1먷6h�x�I       6%�	��a���A�"*;


total_lossȸ�@

error_R$�7?

learning_rate_1먷6���I       6%�	0a���A�"*;


total_loss��~@

error_R��U?

learning_rate_1먷6-U�I       6%�	�^a���A�"*;


total_lossW��@

error_RX�c?

learning_rate_1먷6����I       6%�	N�a���A�"*;


total_loss��@

error_R�S?

learning_rate_1먷69'�$I       6%�	��a���A�"*;


total_loss�}�@

error_R$=?

learning_rate_1먷61�q�I       6%�	5-a���A�"*;


total_loss���@

error_R�K?

learning_rate_1먷6�;n*I       6%�	�oa���A�"*;


total_loss6��@

error_RL,V?

learning_rate_1먷67�4I       6%�	ڵa���A�"*;


total_loss{T�@

error_R�D?

learning_rate_1먷6�v8�I       6%�	��a���A�"*;


total_lossof�@

error_R�	T?

learning_rate_1먷6��I       6%�	�Ha���A�"*;


total_loss���@

error_RTXY?

learning_rate_1먷6!=ZgI       6%�	L�a���A�"*;


total_lossJ��@

error_RZWY?

learning_rate_1먷6��mjI       6%�	��a���A�"*;


total_losshB�@

error_R��G?

learning_rate_1먷68DI       6%�	D	a���A�"*;


total_loss���@

error_R�<?

learning_rate_1먷6��hkI       6%�	{�	a���A�"*;


total_loss;��@

error_R�;X?

learning_rate_1먷6V�{�I       6%�	�	a���A�"*;


total_loss���@

error_RC�W?

learning_rate_1먷6S�3�I       6%�	C*
a���A�"*;


total_loss�P�@

error_R��=?

learning_rate_1먷6� >�I       6%�	�r
a���A�"*;


total_loss��@

error_R��R?

learning_rate_1먷6�{YI       6%�	��
a���A�"*;


total_loss�@

error_R�yU?

learning_rate_1먷6�E�I       6%�	8�
a���A�"*;


total_lossS��@

error_R�N?

learning_rate_1먷6[�X"I       6%�	fAa���A�"*;


total_loss
��@

error_Rd�D?

learning_rate_1먷6��_�I       6%�	Q�a���A�"*;


total_loss��@

error_R��V?

learning_rate_1먷6�t �I       6%�	g�a���A�"*;


total_loss�f�@

error_RX!F?

learning_rate_1먷6���I       6%�	Oa���A�"*;


total_loss��A

error_R �U?

learning_rate_1먷6�1`�I       6%�	mSa���A�"*;


total_lossj��@

error_R	�h?

learning_rate_1먷6��#I       6%�	�a���A�"*;


total_loss���@

error_R�gI?

learning_rate_1먷6���;I       6%�	�a���A�"*;


total_loss�ʍ@

error_R��X?

learning_rate_1먷6��V�I       6%�	Sa���A�"*;


total_loss�@

error_Rv9[?

learning_rate_1먷6�[�I       6%�	<�a���A�"*;


total_loss��.A

error_R�uL?

learning_rate_1먷6R$I       6%�	��a���A�"*;


total_loss�!�@

error_R��a?

learning_rate_1먷6��:FI       6%�	�"a���A�"*;


total_loss�;�@

error_R��O?

learning_rate_1먷6���I       6%�	dha���A�"*;


total_loss�j�@

error_RnqW?

learning_rate_1먷6�a�I       6%�	>�a���A�"*;


total_lossW��@

error_Rj=O?

learning_rate_1먷62��I       6%�	.�a���A�"*;


total_lossp��@

error_R��K?

learning_rate_1먷6��I       6%�	6a���A�"*;


total_loss�-�@

error_R��U?

learning_rate_1먷6�?\!I       6%�	s{a���A�"*;


total_loss�9�@

error_R�:Z?

learning_rate_1먷6���I       6%�	i�a���A�"*;


total_loss_�@

error_R��M?

learning_rate_1먷6��&�I       6%�	�a���A�"*;


total_loss���@

error_R!6Z?

learning_rate_1먷6\Qm9I       6%�	�Ha���A�"*;


total_loss��A

error_RnD?

learning_rate_1먷6�0hrI       6%�	d�a���A�"*;


total_loss�� A

error_R�^0?

learning_rate_1먷6�D�I       6%�	��a���A�"*;


total_loss���@

error_R��N?

learning_rate_1먷6��5I       6%�	"a���A�"*;


total_loss�A

error_R$^H?

learning_rate_1먷6�q��I       6%�	�Ya���A�"*;


total_lossm
�@

error_RW
>?

learning_rate_1먷6��P�I       6%�	9�a���A�"*;


total_loss^-�@

error_R!�Z?

learning_rate_1먷6�V@~I       6%�	>�a���A�"*;


total_loss= �@

error_RCd?

learning_rate_1먷6���I       6%�	�'a���A�"*;


total_loss��@

error_R�Y?

learning_rate_1먷6T�lI       6%�	�la���A�"*;


total_lossNF�@

error_R�W?

learning_rate_1먷62ɫ{I       6%�	$�a���A�"*;


total_loss���@

error_RZ�g?

learning_rate_1먷6ޒ�I       6%�	�a���A�"*;


total_loss���@

error_R��X?

learning_rate_1먷6]KN�I       6%�	y:a���A�"*;


total_lossZg�@

error_R;�S?

learning_rate_1먷6=�lI       6%�	�a���A�"*;


total_loss$q�@

error_RVP?

learning_rate_1먷6UadI       6%�	��a���A�"*;


total_loss��@

error_R!;<?

learning_rate_1먷6W,,I       6%�	a���A�"*;


total_loss?��@

error_R�aS?

learning_rate_1먷6:.;�I       6%�	2Pa���A�"*;


total_loss��|@

error_R�U?

learning_rate_1먷6]��I       6%�	K�a���A�"*;


total_lossl�@

error_RoP?

learning_rate_1먷64�#�I       6%�	}�a���A�"*;


total_lossEj�@

error_RZV`?

learning_rate_1먷6�QܸI       6%�	�a���A�"*;


total_loss�3$A

error_R��L?

learning_rate_1먷6Rsa(I       6%�	"ca���A�"*;


total_lossݨ�@

error_R��L?

learning_rate_1먷6[��I       6%�	Q�a���A�"*;


total_loss쳑@

error_RMZ?

learning_rate_1먷68N�jI       6%�	��a���A�"*;


total_lossԒ~@

error_R��]?

learning_rate_1먷6Dk�I       6%�	�2a���A�"*;


total_loss�@

error_R�ui?

learning_rate_1먷6�P�I       6%�	Lwa���A�"*;


total_loss���@

error_R&XG?

learning_rate_1먷6�X�I       6%�	/�a���A�"*;


total_loss�!�@

error_R�H?

learning_rate_1먷6ꚤ�I       6%�	�a���A�"*;


total_loss�̤@

error_R��C?

learning_rate_1먷6�=�I       6%�	`Ea���A�"*;


total_loss���@

error_Rj�J?

learning_rate_1먷6���I       6%�	e�a���A�"*;


total_lossew�@

error_R�U?

learning_rate_1먷6�[�I       6%�	��a���A�"*;


total_loss�j�@

error_R2f^?

learning_rate_1먷6��[�I       6%�	a���A�"*;


total_loss�Ò@

error_RX�U?

learning_rate_1먷6�I       6%�	�Sa���A�"*;


total_loss�ܕ@

error_R�"S?

learning_rate_1먷6)�ģI       6%�	��a���A�"*;


total_lossj��@

error_R_[A?

learning_rate_1먷67�0II       6%�	��a���A�"*;


total_loss&�@

error_R��M?

learning_rate_1먷6���HI       6%�	�a���A�"*;


total_loss���@

error_R�YM?

learning_rate_1먷6�!�I       6%�	aa���A�"*;


total_loss�ـ@

error_R�EX?

learning_rate_1먷6�n��I       6%�	��a���A�"*;


total_loss���@

error_R6�G?

learning_rate_1먷6k�L�I       6%�	I�a���A�"*;


total_loss���@

error_R�Q?

learning_rate_1먷6����I       6%�	�a���A�"*;


total_lossc<�@

error_R#�L?

learning_rate_1먷6�h|I       6%�	�`a���A�"*;


total_loss=^�@

error_R@�B?

learning_rate_1먷6ߟ�TI       6%�	�a���A�"*;


total_lossf5�@

error_ROBL?

learning_rate_1먷6���I       6%�	�a���A�"*;


total_lossE��@

error_RH%R?

learning_rate_1먷6}?�CI       6%�	o!a���A�"*;


total_loss�kA

error_R��F?

learning_rate_1먷6�#�I       6%�	Wea���A�"*;


total_loss�*�@

error_R��F?

learning_rate_1먷6�kI       6%�	ͦa���A�"*;


total_loss���@

error_RdVR?

learning_rate_1먷6�݇�I       6%�	&�a���A�#*;


total_loss��@

error_R�IS?

learning_rate_1먷6~���I       6%�	
'a���A�#*;


total_loss�?�@

error_RIS?

learning_rate_1먷6�#X�I       6%�	Kha���A�#*;


total_loss��A

error_RO�??

learning_rate_1먷6����I       6%�	��a���A�#*;


total_lossf�t@

error_Rگ@?

learning_rate_1먷6�D˅I       6%�	#a���A�#*;


total_lossLZ�@

error_R��O?

learning_rate_1먷6	���I       6%�	�Na���A�#*;


total_loss��@

error_Ri�Y?

learning_rate_1먷6��qI       6%�	F�a���A�#*;


total_loss�J�@

error_RoxE?

learning_rate_1먷6����I       6%�	�a���A�#*;


total_lossp,�@

error_RC�F?

learning_rate_1먷6&l�I       6%�	  a���A�#*;


total_loss�H
A

error_R��>?

learning_rate_1먷6#�I       6%�	�fa���A�#*;


total_loss���@

error_R�T?

learning_rate_1먷6yd�HI       6%�	�a���A�#*;


total_loss	��@

error_R�*]?

learning_rate_1먷6cR��I       6%�	�a���A�#*;


total_loss��@

error_R\%T?

learning_rate_1먷6��8QI       6%�	�-a���A�#*;


total_loss��@

error_R��O?

learning_rate_1먷6�O�I       6%�	ppa���A�#*;


total_lossn�@

error_R�@?

learning_rate_1먷6T�I       6%�	ݯa���A�#*;


total_loss��@

error_Rl�V?

learning_rate_1먷6�*�I       6%�	�a���A�#*;


total_loss���@

error_R��G?

learning_rate_1먷6�}*+I       6%�	�2 a���A�#*;


total_loss�_�@

error_R�l;?

learning_rate_1먷6���UI       6%�	�� a���A�#*;


total_lossR��@

error_R2�G?

learning_rate_1먷6���qI       6%�	�� a���A�#*;


total_loss@�@

error_R$�??

learning_rate_1먷6�,1�I       6%�	�!a���A�#*;


total_loss���@

error_RAM?

learning_rate_1먷6�x�UI       6%�	�^!a���A�#*;


total_loss��
A

error_R�K?

learning_rate_1먷6Y�d�I       6%�	��!a���A�#*;


total_loss��@

error_R��O?

learning_rate_1먷6@�}VI       6%�	g "a���A�#*;


total_lossQ�@

error_R�:H?

learning_rate_1먷6n?�I       6%�	�A"a���A�#*;


total_loss;т@

error_RTfL?

learning_rate_1먷6��	�I       6%�	S�"a���A�#*;


total_loss{�@

error_R�:`?

learning_rate_1먷6X�2OI       6%�	�"a���A�#*;


total_lossT˓@

error_R*�S?

learning_rate_1먷6��9I       6%�	w#a���A�#*;


total_lossp�@

error_RzV?

learning_rate_1먷6��=#I       6%�	�C#a���A�#*;


total_loss��@

error_R)�Z?

learning_rate_1먷68pa�I       6%�	�#a���A�#*;


total_loss�Q�@

error_Rv#=?

learning_rate_1먷6�BN�I       6%�	��#a���A�#*;


total_lossdJ�@

error_R��I?

learning_rate_1먷6�G[I       6%�	�$a���A�#*;


total_loss?�@

error_R�>?

learning_rate_1먷6T��I       6%�	�R$a���A�#*;


total_loss?�m@

error_R�M?

learning_rate_1먷6��ٺI       6%�	 �$a���A�#*;


total_loss[��@

error_R-W?

learning_rate_1먷6��I       6%�	y�$a���A�#*;


total_loss���@

error_R(�4?

learning_rate_1먷6�
"[I       6%�	�%a���A�#*;


total_loss�b�@

error_R��a?

learning_rate_1먷6�]�I       6%�	W%a���A�#*;


total_lossMD�@

error_R%Q?

learning_rate_1먷6Jh��I       6%�	ѕ%a���A�#*;


total_lossu�A

error_Rx4U?

learning_rate_1먷6'��I       6%�	�%a���A�#*;


total_loss�@

error_R��A?

learning_rate_1먷6V@V�I       6%�	�&a���A�#*;


total_lossW��@

error_R��Q?

learning_rate_1먷6��7zI       6%�	[&a���A�#*;


total_loss�C�@

error_R6�^?

learning_rate_1먷6#��I       6%�	+�&a���A�#*;


total_loss�W�@

error_Rn�q?

learning_rate_1먷6q ��I       6%�	��&a���A�#*;


total_loss�һ@

error_R(]<?

learning_rate_1먷6#�i�I       6%�	�'a���A�#*;


total_loss�)�@

error_R��8?

learning_rate_1먷6yOPI       6%�	\]'a���A�#*;


total_loss�8�@

error_R�vY?

learning_rate_1먷6&��oI       6%�	��'a���A�#*;


total_loss�V�@

error_R-8Q?

learning_rate_1먷6�K�I       6%�	��'a���A�#*;


total_loss��@

error_R��??

learning_rate_1먷6QR�sI       6%�	�#(a���A�#*;


total_loss�n�@

error_R��I?

learning_rate_1먷6/�I       6%�	hh(a���A�#*;


total_loss���@

error_R��A?

learning_rate_1먷6��9I       6%�	��(a���A�#*;


total_loss1��@

error_R��[?

learning_rate_1먷6��uI       6%�	�(a���A�#*;


total_loss�{�@

error_R\AT?

learning_rate_1먷6^:�I       6%�	'')a���A�#*;


total_loss�]{@

error_Rx�S?

learning_rate_1먷6�.�`I       6%�	|i)a���A�#*;


total_loss���@

error_R�;?

learning_rate_1먷6~��I       6%�	ĭ)a���A�#*;


total_lossMJm@

error_R��T?

learning_rate_1먷6���I       6%�	K�)a���A�#*;


total_lossJ��@

error_RM�;?

learning_rate_1먷6���EI       6%�	l3*a���A�#*;


total_loss=��@

error_R�P?

learning_rate_1먷6J@�AI       6%�	3r*a���A�#*;


total_loss�{�@

error_R��]?

learning_rate_1먷6���I       6%�	��*a���A�#*;


total_loss�M�@

error_R1�V?

learning_rate_1먷6�`^I       6%�	b�*a���A�#*;


total_loss�?�@

error_R�\Y?

learning_rate_1먷6��%aI       6%�	�5+a���A�#*;


total_loss�e�@

error_R�S?

learning_rate_1먷6���I       6%�	�v+a���A�#*;


total_lossw��@

error_R�zO?

learning_rate_1먷6�'6I       6%�	ڶ+a���A�#*;


total_loss�K�@

error_RqvW?

learning_rate_1먷6����I       6%�	��+a���A�#*;


total_loss*�@

error_R�fK?

learning_rate_1먷6��JI       6%�	l=,a���A�#*;


total_loss_��@

error_R͢c?

learning_rate_1먷6א�I       6%�	s�,a���A�#*;


total_loss���@

error_R��6?

learning_rate_1먷6W+�I       6%�	 �,a���A�#*;


total_loss���@

error_R��Y?

learning_rate_1먷6X�͍I       6%�	�!-a���A�#*;


total_lossWZ�@

error_R��N?

learning_rate_1먷6@��\I       6%�	�h-a���A�#*;


total_loss��@

error_R|�O?

learning_rate_1먷6�҈I       6%�	
�-a���A�#*;


total_lossT��@

error_R�<?

learning_rate_1먷6��I       6%�	t�-a���A�#*;


total_lossH/�@

error_Rs�:?

learning_rate_1먷6��NI       6%�	�*.a���A�#*;


total_lossI��@

error_R��I?

learning_rate_1먷6����I       6%�	m.a���A�#*;


total_loss�f�@

error_R�bV?

learning_rate_1먷6�I�I       6%�	ë.a���A�#*;


total_loss�<�@

error_R-*D?

learning_rate_1먷6ay��I       6%�	��.a���A�#*;


total_loss#@�@

error_R��T?

learning_rate_1먷6g���I       6%�	�-/a���A�#*;


total_loss8��@

error_R/�I?

learning_rate_1먷6P�g�I       6%�	m/a���A�#*;


total_loss}�!A

error_R)�U?

learning_rate_1먷6|�(2I       6%�	��/a���A�#*;


total_loss��@

error_R#�L?

learning_rate_1먷6�ƊI       6%�	T�/a���A�#*;


total_lossD^�@

error_R�N?

learning_rate_1먷6`?I       6%�	�-0a���A�#*;


total_loss�̆@

error_R��I?

learning_rate_1먷6��٫I       6%�	�~0a���A�#*;


total_loss��@

error_R(B?

learning_rate_1먷6�^E�I       6%�	��0a���A�#*;


total_loss}��@

error_R�B?

learning_rate_1먷6�iI       6%�	G1a���A�#*;


total_loss�E�@

error_R.�X?

learning_rate_1먷6·�3I       6%�	<N1a���A�#*;


total_loss�c�@

error_Rw�e?

learning_rate_1먷6�� I       6%�	�1a���A�#*;


total_loss���@

error_R�#1?

learning_rate_1먷6��I       6%�	��1a���A�#*;


total_loss���@

error_R`]X?

learning_rate_1먷6E�I       6%�	k2a���A�#*;


total_loss1ƕ@

error_R��T?

learning_rate_1먷6���0I       6%�	�`2a���A�#*;


total_loss�@

error_R��Q?

learning_rate_1먷6��7�I       6%�	�2a���A�#*;


total_lossh��@

error_R��Q?

learning_rate_1먷62(�I       6%�	r�2a���A�#*;


total_lossY/A

error_R��_?

learning_rate_1먷6ܟU�I       6%�	�,3a���A�#*;


total_loss.�@

error_ROEU?

learning_rate_1먷6�s��I       6%�	gm3a���A�#*;


total_lossXk�@

error_R�`<?

learning_rate_1먷6���+I       6%�	�3a���A�#*;


total_lossm��@

error_RC�X?

learning_rate_1먷6i�R=I       6%�	|�3a���A�#*;


total_lossd�@

error_R=F?

learning_rate_1먷6m�$I       6%�	424a���A�#*;


total_loss�0�@

error_R�6G?

learning_rate_1먷6�6��I       6%�	Yw4a���A�#*;


total_lossT�@

error_R�JS?

learning_rate_1먷6���I       6%�	Ÿ4a���A�#*;


total_loss�3�@

error_R/vA?

learning_rate_1먷6�I       6%�	��4a���A�#*;


total_loss�`�@

error_R_�<?

learning_rate_1먷6i�I       6%�	[:5a���A�#*;


total_loss���@

error_R�DP?

learning_rate_1먷6F��I       6%�	�z5a���A�#*;


total_loss�/�@

error_RS�C?

learning_rate_1먷6�)^�I       6%�	�5a���A�#*;


total_lossf��@

error_REDY?

learning_rate_1먷6�8-�I       6%�	��5a���A�#*;


total_loss���@

error_R4I?

learning_rate_1먷6�ۯI       6%�	�=6a���A�#*;


total_lossr��@

error_R��F?

learning_rate_1먷6��r�I       6%�	�~6a���A�#*;


total_losst��@

error_Rci?

learning_rate_1먷6��\�I       6%�	�6a���A�#*;


total_loss{:�@

error_R��V?

learning_rate_1먷6��I       6%�	7a���A�#*;


total_lossP�@

error_RX�I?

learning_rate_1먷6�Z�	I       6%�	
H7a���A�#*;


total_loss:,�@

error_ROr<?

learning_rate_1먷6-��I       6%�	u�7a���A�#*;


total_loss7��@

error_R2O?

learning_rate_1먷6#VBI       6%�	��7a���A�#*;


total_lossO��@

error_R�??

learning_rate_1먷6�qf�I       6%�	�8a���A�#*;


total_loss[ y@

error_RΛK?

learning_rate_1먷6 J<I       6%�	�J8a���A�#*;


total_loss8"�@

error_R�B?

learning_rate_1먷6�x �I       6%�	M�8a���A�#*;


total_lossȔ�@

error_R��M?

learning_rate_1먷6��v�I       6%�	5�8a���A�#*;


total_loss��@

error_Rn�W?

learning_rate_1먷6}ޚ=I       6%�	9a���A�#*;


total_loss!��@

error_R1cO?

learning_rate_1먷6+��I       6%�	W9a���A�#*;


total_loss�g@

error_RM�D?

learning_rate_1먷6dI I       6%�	:�9a���A�#*;


total_loss2A

error_R8�G?

learning_rate_1먷6Ms��I       6%�	��9a���A�#*;


total_lossKڡ@

error_R��I?

learning_rate_1먷6O+��I       6%�	Y:a���A�#*;


total_loss�g�@

error_R�'G?

learning_rate_1먷6 O4rI       6%�	Z:a���A�#*;


total_loss<��@

error_R_�V?

learning_rate_1먷6[��;I       6%�	�:a���A�#*;


total_lossxP�@

error_R�DW?

learning_rate_1먷6p?�I       6%�	��:a���A�#*;


total_loss�m�@

error_R8�H?

learning_rate_1먷6�WrI       6%�	;a���A�#*;


total_lossc�@

error_R�`C?

learning_rate_1먷6_�G�I       6%�	V;a���A�#*;


total_loss�4�@

error_R��]?

learning_rate_1먷6��sFI       6%�	�;a���A�#*;


total_lossli�@

error_R��W?

learning_rate_1먷6� ��I       6%�	��;a���A�#*;


total_loss�Ǝ@

error_R6�^?

learning_rate_1먷6����I       6%�	<a���A�#*;


total_loss�X�@

error_R�&a?

learning_rate_1먷6��;I       6%�	�`<a���A�#*;


total_lossW�X@

error_R�qH?

learning_rate_1먷6]���I       6%�	��<a���A�#*;


total_loss���@

error_RτI?

learning_rate_1먷6��I       6%�	J�<a���A�#*;


total_loss�ؙ@

error_R�wG?

learning_rate_1먷6o�I       6%�	�F=a���A�#*;


total_loss���@

error_Rc!Q?

learning_rate_1먷6>F�qI       6%�	b�=a���A�$*;


total_loss�A

error_Ra�X?

learning_rate_1먷6��o�I       6%�	&�=a���A�$*;


total_loss�n�@

error_R$�C?

learning_rate_1먷6!��I       6%�	�>a���A�$*;


total_losspF�@

error_R�D?

learning_rate_1먷6����I       6%�	�Q>a���A�$*;


total_loss���@

error_R��M?

learning_rate_1먷6�!�0I       6%�	˖>a���A�$*;


total_loss�@

error_RZzR?

learning_rate_1먷6�ƼI       6%�	��>a���A�$*;


total_lossAҦ@

error_R{5N?

learning_rate_1먷6�
�LI       6%�	#?a���A�$*;


total_loss�k�@

error_R� >?

learning_rate_1먷6��[I       6%�	EX?a���A�$*;


total_loss,�@

error_RE??

learning_rate_1먷6~�lI       6%�	;�?a���A�$*;


total_losscU
A

error_R��8?

learning_rate_1먷62��I       6%�	��?a���A�$*;


total_loss;�@

error_R�JY?

learning_rate_1먷6���I       6%�	�6@a���A�$*;


total_loss��@

error_R��W?

learning_rate_1먷6��LI       6%�	�@a���A�$*;


total_lossM��@

error_RE�;?

learning_rate_1먷6�ϕpI       6%�	��@a���A�$*;


total_loss�Q�@

error_R��Z?

learning_rate_1먷6W��I       6%�	Aa���A�$*;


total_loss�3�@

error_RyJ?

learning_rate_1먷6�=r�I       6%�	k]Aa���A�$*;


total_loss��@

error_R�FW?

learning_rate_1먷6��I       6%�	��Aa���A�$*;


total_lossz�@

error_R6�J?

learning_rate_1먷6O��I       6%�	�Aa���A�$*;


total_loss���@

error_R݇C?

learning_rate_1먷6�|b=I       6%�	�$Ba���A�$*;


total_loss�"�@

error_RԯI?

learning_rate_1먷6����I       6%�	�hBa���A�$*;


total_loss�.�@

error_RW"_?

learning_rate_1먷64{6�I       6%�	�Ba���A�$*;


total_loss��@

error_R��`?

learning_rate_1먷6��-�I       6%�	��Ba���A�$*;


total_loss�/�@

error_Rn~R?

learning_rate_1먷6J2�zI       6%�	-Ca���A�$*;


total_loss��@

error_R{�]?

learning_rate_1먷60`�I       6%�	�mCa���A�$*;


total_loss|c�@

error_R5:?

learning_rate_1먷6�M �I       6%�	P�Ca���A�$*;


total_loss�k�@

error_R�??

learning_rate_1먷6����I       6%�	��Ca���A�$*;


total_loss�tq@

error_R�#[?

learning_rate_1먷6	m�I       6%�	�/Da���A�$*;


total_lossߘ�@

error_R�MS?

learning_rate_1먷6�y��I       6%�	�oDa���A�$*;


total_lossჁ@

error_R��@?

learning_rate_1먷6�BHI       6%�	��Da���A�$*;


total_loss�ʂ@

error_R�O?

learning_rate_1먷6��I       6%�	��Da���A�$*;


total_loss�E�@

error_Rc(T?

learning_rate_1먷6�u�cI       6%�	K1Ea���A�$*;


total_loss�2�@

error_R��H?

learning_rate_1먷6u���I       6%�	�pEa���A�$*;


total_loss���@

error_R��X?

learning_rate_1먷6눉�I       6%�	��Ea���A�$*;


total_loss��@

error_R�N?

learning_rate_1먷6�j<I       6%�	��Ea���A�$*;


total_lossT��@

error_R]�M?

learning_rate_1먷6Or2SI       6%�	Y4Fa���A�$*;


total_loss�I�@

error_R�R?

learning_rate_1먷6��I       6%�	5uFa���A�$*;


total_loss]�@

error_R�tV?

learning_rate_1먷6��I       6%�	X�Fa���A�$*;


total_loss��@

error_R8�M?

learning_rate_1먷6���I       6%�	|�Fa���A�$*;


total_lossS��@

error_RH{R?

learning_rate_1먷6ƁiI       6%�	�5Ga���A�$*;


total_loss���@

error_RF�L?

learning_rate_1먷6��6tI       6%�	̇Ga���A�$*;


total_loss�/�@

error_R�R?

learning_rate_1먷6f�� I       6%�	��Ga���A�$*;


total_loss6V�@

error_R�UH?

learning_rate_1먷6�?��I       6%�	 $Ha���A�$*;


total_loss��@

error_RoT?

learning_rate_1먷6�P �I       6%�	@eHa���A�$*;


total_loss\y�@

error_R�N?

learning_rate_1먷6��\I       6%�	(�Ha���A�$*;


total_loss�\�@

error_R�@T?

learning_rate_1먷6���I       6%�	��Ha���A�$*;


total_loss3;�@

error_R�^T?

learning_rate_1먷6�q��I       6%�	�2Ia���A�$*;


total_loss�m�@

error_Rx_?

learning_rate_1먷6)KA=I       6%�	�wIa���A�$*;


total_losso��@

error_RMaY?

learning_rate_1먷66(��I       6%�	̾Ia���A�$*;


total_loss��@

error_R@�??

learning_rate_1먷6�\L6I       6%�	-Ja���A�$*;


total_lossI�@

error_R�'Z?

learning_rate_1먷6��I�I       6%�	�FJa���A�$*;


total_loss���@

error_R�\N?

learning_rate_1먷6����I       6%�	܇Ja���A�$*;


total_loss��@

error_ROo/?

learning_rate_1먷6jp?|I       6%�	��Ja���A�$*;


total_loss;��@

error_R{�S?

learning_rate_1먷6����I       6%�	�Ka���A�$*;


total_loss�t�@

error_R&YT?

learning_rate_1먷6��+I       6%�	KKa���A�$*;


total_lossV_�@

error_Rf�=?

learning_rate_1먷6|�I       6%�	׋Ka���A�$*;


total_loss)[@

error_Rlz\?

learning_rate_1먷6��hI       6%�	E�Ka���A�$*;


total_loss �@

error_R�yP?

learning_rate_1먷6��yEI       6%�	fLa���A�$*;


total_lossez@

error_R|�R?

learning_rate_1먷6�)��I       6%�	�SLa���A�$*;


total_loss��@

error_R6�O?

learning_rate_1먷6n;I       6%�	��La���A�$*;


total_loss�/�@

error_Rc�<?

learning_rate_1먷6_2�}I       6%�	��La���A�$*;


total_loss��a@

error_R�ED?

learning_rate_1먷6-x�I       6%�	V?Ma���A�$*;


total_loss�B�@

error_R1E?

learning_rate_1먷6v#��I       6%�		�Ma���A�$*;


total_lossX�@

error_R�A<?

learning_rate_1먷6N��TI       6%�	4�Ma���A�$*;


total_loss�Ҿ@

error_R�W?

learning_rate_1먷6�.vI       6%�	Na���A�$*;


total_loss�ō@

error_R��c?

learning_rate_1먷6/N"�I       6%�	[Na���A�$*;


total_loss5�@

error_R��G?

learning_rate_1먷6:I       6%�	̫Na���A�$*;


total_loss$r@

error_R��N?

learning_rate_1먷6� �~I       6%�	�'Oa���A�$*;


total_loss���@

error_R��V?

learning_rate_1먷6/��I       6%�	XoOa���A�$*;


total_lossᡗ@

error_R�N?

learning_rate_1먷6��=I       6%�	��Oa���A�$*;


total_loss=��@

error_R�eM?

learning_rate_1먷6~���I       6%�	��Oa���A�$*;


total_lossa��@

error_R۳;?

learning_rate_1먷6�(U�I       6%�	7Pa���A�$*;


total_loss2�@

error_RF4T?

learning_rate_1먷6��UOI       6%�	>zPa���A�$*;


total_lossՏ�@

error_R��V?

learning_rate_1먷6�)�oI       6%�	ѽPa���A�$*;


total_loss4�@

error_R�[?

learning_rate_1먷6@��I       6%�	� Qa���A�$*;


total_loss���@

error_R��:?

learning_rate_1먷6}��I       6%�	ZGQa���A�$*;


total_loss�Г@

error_R�ig?

learning_rate_1먷6�GUI       6%�	�Qa���A�$*;


total_lossL0�@

error_R~M?

learning_rate_1먷6j�I       6%�	��Qa���A�$*;


total_loss�/�@

error_R[(X?

learning_rate_1먷6}(�I       6%�	vRa���A�$*;


total_lossYp�@

error_R]*T?

learning_rate_1먷6��VbI       6%�	ZRa���A�$*;


total_loss���@

error_R_�<?

learning_rate_1먷6H�׬I       6%�	r�Ra���A�$*;


total_loss�I�@

error_RmuL?

learning_rate_1먷6�~��I       6%�	��Ra���A�$*;


total_lossw��@

error_R�@?

learning_rate_1먷6����I       6%�	�Sa���A�$*;


total_lossv��@

error_RI�Q?

learning_rate_1먷67RHZI       6%�	�dSa���A�$*;


total_lossr�@

error_R�E?

learning_rate_1먷6�iu�I       6%�	��Sa���A�$*;


total_loss"�@

error_R�%A?

learning_rate_1먷6{JOI       6%�	�Ta���A�$*;


total_lossMԷ@

error_R�L?

learning_rate_1먷6��mEI       6%�	�[Ta���A�$*;


total_lossce�@

error_R�XB?

learning_rate_1먷6Ŵ�I       6%�	��Ta���A�$*;


total_loss���@

error_RQ1R?

learning_rate_1먷6��NI       6%�	RUa���A�$*;


total_lossTqf@

error_R�=?

learning_rate_1먷65�EqI       6%�	N�Ua���A�$*;


total_loss�^�@

error_R��6?

learning_rate_1먷6qP�I       6%�	&�Ua���A�$*;


total_lossQ�	A

error_Rv�P?

learning_rate_1먷6k03�I       6%�	%|Va���A�$*;


total_loss w�@

error_R��F?

learning_rate_1먷6ԖI       6%�	�(Wa���A�$*;


total_loss8�A

error_R�*D?

learning_rate_1먷6h�M�I       6%�	�yWa���A�$*;


total_lossw|�@

error_RRK?

learning_rate_1먷6���8I       6%�	��Wa���A�$*;


total_loss�&�@

error_R�I?

learning_rate_1먷63�h�I       6%�	yXa���A�$*;


total_loss�@

error_R��D?

learning_rate_1먷6ΥC�I       6%�	`sXa���A�$*;


total_loss<��@

error_R qU?

learning_rate_1먷6Rl�kI       6%�	��Xa���A�$*;


total_loss�@

error_Ro�^?

learning_rate_1먷6�WI       6%�	�$Ya���A�$*;


total_loss?I�@

error_R�TH?

learning_rate_1먷6�$&I       6%�	�xYa���A�$*;


total_loss��@

error_RZ&E?

learning_rate_1먷6b��xI       6%�	2�Ya���A�$*;


total_loss�į@

error_R �G?

learning_rate_1먷6��grI       6%�	�;Za���A�$*;


total_loss|�@

error_RT�E?

learning_rate_1먷6���oI       6%�	��Za���A�$*;


total_loss��@

error_R_�R?

learning_rate_1먷6;�I       6%�	��Za���A�$*;


total_loss�q_@

error_RJ(B?

learning_rate_1먷6�'xI       6%�	�[a���A�$*;


total_loss��@

error_R��A?

learning_rate_1먷6h�YI       6%�	�j[a���A�$*;


total_lossr�A

error_R�H?

learning_rate_1먷6�^�'I       6%�	�[a���A�$*;


total_loss�O@

error_R,�7?

learning_rate_1먷6��I       6%�	�[a���A�$*;


total_loss!&�@

error_R��J?

learning_rate_1먷6���
I       6%�	�F\a���A�$*;


total_losst�@

error_R8�U?

learning_rate_1먷6�(I       6%�	`�\a���A�$*;


total_loss�Z�@

error_R�PL?

learning_rate_1먷6Y�/I       6%�	�\a���A�$*;


total_lossf��@

error_R��D?

learning_rate_1먷6@�e�I       6%�	�N]a���A�$*;


total_loss*ܾ@

error_R?O?

learning_rate_1먷65�h�I       6%�	��]a���A�$*;


total_loss� �@

error_R�8J?

learning_rate_1먷6jm�xI       6%�	v�]a���A�$*;


total_loss{�@

error_R.�X?

learning_rate_1먷6j��BI       6%�	�,^a���A�$*;


total_loss��@

error_R8R?

learning_rate_1먷6Zb��I       6%�	�^a���A�$*;


total_loss�ܤ@

error_R3(S?

learning_rate_1먷6"��hI       6%�	c�^a���A�$*;


total_loss+�@

error_R�@S?

learning_rate_1먷6q4=I       6%�	"_a���A�$*;


total_loss=��@

error_R�HV?

learning_rate_1먷6�m�YI       6%�	�f_a���A�$*;


total_loss��@

error_R�DP?

learning_rate_1먷6+O�ZI       6%�	�_a���A�$*;


total_loss� �@

error_R��E?

learning_rate_1먷6&��I       6%�	Q`a���A�$*;


total_loss'�@

error_Rs�N?

learning_rate_1먷6�T�I       6%�	�K`a���A�$*;


total_loss�]�@

error_R&�U?

learning_rate_1먷6��oI       6%�	��`a���A�$*;


total_lossZX�@

error_R�N?

learning_rate_1먷6r�h�I       6%�	b�`a���A�$*;


total_lossڀ�@

error_R��I?

learning_rate_1먷67lMxI       6%�	#aa���A�$*;


total_loss4��@

error_RH�@?

learning_rate_1먷6ѯm5I       6%�	�_aa���A�$*;


total_lossd�A

error_R��R?

learning_rate_1먷6�*�EI       6%�	��aa���A�$*;


total_loss��A

error_R��R?

learning_rate_1먷6+zUI       6%�	9$ba���A�$*;


total_lossD�@

error_RW�N?

learning_rate_1먷6I�L�I       6%�	�jba���A�$*;


total_loss�ؽ@

error_RN�W?

learning_rate_1먷6Lff�I       6%�	��ba���A�$*;


total_loss$n�@

error_R�HF?

learning_rate_1먷6E��I       6%�	9ca���A�%*;


total_lossj�@

error_R��L?

learning_rate_1먷6���I       6%�	ZXca���A�%*;


total_loss��@

error_RO}X?

learning_rate_1먷6�h=`I       6%�	U�ca���A�%*;


total_loss.Yb@

error_R
�f?

learning_rate_1먷6����I       6%�	��ca���A�%*;


total_loss�!�@

error_R��H?

learning_rate_1먷6۳%NI       6%�	QJda���A�%*;


total_loss�ܛ@

error_R�3K?

learning_rate_1먷6�hZKI       6%�	=�da���A�%*;


total_loss��@

error_R4O>?

learning_rate_1먷6��I       6%�	��da���A�%*;


total_loss�|�@

error_RP?

learning_rate_1먷6A�@�I       6%�	�ea���A�%*;


total_loss2��@

error_R�<?

learning_rate_1먷6]�)}I       6%�	�[ea���A�%*;


total_loss���@

error_R�E?

learning_rate_1먷6���nI       6%�	��ea���A�%*;


total_lossBH�@

error_R��9?

learning_rate_1먷6o���I       6%�	"�ea���A�%*;


total_lossZA�@

error_RѧS?

learning_rate_1먷6�z�I       6%�	!*fa���A�%*;


total_loss�F�@

error_R��@?

learning_rate_1먷6\���I       6%�	Emfa���A�%*;


total_loss�B�@

error_R�lS?

learning_rate_1먷6�[�;I       6%�	j�fa���A�%*;


total_loss��Q@

error_RM]F?

learning_rate_1먷6�+I       6%�	��fa���A�%*;


total_loss]ʕ@

error_R>?

learning_rate_1먷6��.I       6%�	�8ga���A�%*;


total_lossg�@

error_R+Q?

learning_rate_1먷6��O�I       6%�	Y}ga���A�%*;


total_loss�2�@

error_R�X?

learning_rate_1먷6�`�I       6%�	 �ga���A�%*;


total_loss�8�@

error_RT�K?

learning_rate_1먷6���~I       6%�	pha���A�%*;


total_loss���@

error_R��F?

learning_rate_1먷6l�	kI       6%�	vLha���A�%*;


total_loss��@

error_R�b?

learning_rate_1먷6j�jcI       6%�	��ha���A�%*;


total_lossRA

error_RM�P?

learning_rate_1먷6�dI       6%�	V�ha���A�%*;


total_lossvf�@

error_R�K?

learning_rate_1먷6�/q�I       6%�	�ia���A�%*;


total_loss��@

error_R�I?

learning_rate_1먷6%��I       6%�	�Via���A�%*;


total_loss���@

error_R��<?

learning_rate_1먷69퍘I       6%�	��ia���A�%*;


total_loss���@

error_RoA<?

learning_rate_1먷6w�5I       6%�	��ia���A�%*;


total_loss���@

error_R��6?

learning_rate_1먷6usGI       6%�	m$ja���A�%*;


total_loss���@

error_R��K?

learning_rate_1먷6��:*I       6%�	Sfja���A�%*;


total_loss�G�@

error_R��D?

learning_rate_1먷6�1фI       6%�	C�ja���A�%*;


total_loss'�@

error_R��L?

learning_rate_1먷6ݩHcI       6%�	C�ja���A�%*;


total_loss!��@

error_RMdI?

learning_rate_1먷6P�/I       6%�	.ka���A�%*;


total_loss�n�@

error_R�T?

learning_rate_1먷6���I       6%�	oka���A�%*;


total_lossD^�@

error_RZHY?

learning_rate_1먷6�VL�I       6%�	��ka���A�%*;


total_loss��@

error_R�eL?

learning_rate_1먷6���I       6%�	��ka���A�%*;


total_loss�	�@

error_R��L?

learning_rate_1먷6�~
1I       6%�	�:la���A�%*;


total_loss�@

error_R��I?

learning_rate_1먷6�=�?I       6%�	�~la���A�%*;


total_loss��@

error_R�X?

learning_rate_1먷6VUNI       6%�	�la���A�%*;


total_loss :A

error_RtM?

learning_rate_1먷6���oI       6%�	�'ma���A�%*;


total_loss��A

error_R�K?

learning_rate_1먷6rD��I       6%�	�mma���A�%*;


total_lossֺ�@

error_Rԩ@?

learning_rate_1먷6~�OI       6%�	0�ma���A�%*;


total_loss`��@

error_Rd�6?

learning_rate_1먷6=��nI       6%�	�ma���A�%*;


total_loss��r@

error_R�Q_?

learning_rate_1먷6	��I       6%�	a7na���A�%*;


total_loss1��@

error_R�T?

learning_rate_1먷6&̓�I       6%�	#|na���A�%*;


total_loss���@

error_R$QH?

learning_rate_1먷6��ТI       6%�	��na���A�%*;


total_loss%p�@

error_RTG>?

learning_rate_1먷6���I       6%�	(+oa���A�%*;


total_loss#U�@

error_R�D?

learning_rate_1먷6�~�I       6%�	zoa���A�%*;


total_lossR�@

error_R��V?

learning_rate_1먷6�M!�I       6%�	l�oa���A�%*;


total_loss�-�@

error_R4�<?

learning_rate_1먷6(hI       6%�	�:pa���A�%*;


total_loss�=�@

error_RɝQ?

learning_rate_1먷6XURWI       6%�	N�pa���A�%*;


total_lossh�@

error_R��F?

learning_rate_1먷6�[l�I       6%�	H�pa���A�%*;


total_loss8��@

error_R%�b?

learning_rate_1먷6��JI       6%�	~!qa���A�%*;


total_loss�0�@

error_Rj�[?

learning_rate_1먷6��KI       6%�	Khqa���A�%*;


total_lossz�@

error_RDVF?

learning_rate_1먷6��LI       6%�	��qa���A�%*;


total_loss��@

error_R��_?

learning_rate_1먷6�)u�I       6%�	��qa���A�%*;


total_lossF��@

error_R/xH?

learning_rate_1먷6�m��I       6%�		8ra���A�%*;


total_lossd��@

error_R�X?

learning_rate_1먷6��|�I       6%�	R}ra���A�%*;


total_loss���@

error_R
bF?

learning_rate_1먷67ć�I       6%�	��ra���A�%*;


total_lossO`�@

error_R1pN?

learning_rate_1먷6���I       6%�	hsa���A�%*;


total_lossv~�@

error_R/IN?

learning_rate_1먷6e��I       6%�	;Jsa���A�%*;


total_lossm3A

error_R�.S?

learning_rate_1먷6|���I       6%�	p�sa���A�%*;


total_loss�6A

error_R]%X?

learning_rate_1먷6=<�I       6%�	��sa���A�%*;


total_lossVA

error_R��O?

learning_rate_1먷6�_�kI       6%�	�ta���A�%*;


total_lossܛ@

error_RI2K?

learning_rate_1먷6'��6I       6%�	�Yta���A�%*;


total_loss
��@

error_RDgP?

learning_rate_1먷6��׶I       6%�	C�ta���A�%*;


total_loss)�@

error_Ra�F?

learning_rate_1먷6y`�I       6%�	j�ta���A�%*;


total_lossh��@

error_R�ZU?

learning_rate_1먷6�uUZI       6%�	 Mua���A�%*;


total_losslo�@

error_RY?

learning_rate_1먷6�)o�I       6%�	��ua���A�%*;


total_loss:��@

error_R�MZ?

learning_rate_1먷6�jI       6%�	Nva���A�%*;


total_loss�_�@

error_RͨE?

learning_rate_1먷6?%:I       6%�	�`va���A�%*;


total_loss�u�@

error_R�@?

learning_rate_1먷6�=�I       6%�	�va���A�%*;


total_loss��A

error_R��Q?

learning_rate_1먷6�X�OI       6%�	swa���A�%*;


total_loss�(�@

error_R��W?

learning_rate_1먷69�&jI       6%�	Kbwa���A�%*;


total_loss@V�@

error_Ri�R?

learning_rate_1먷6>���I       6%�	�wa���A�%*;


total_lossAx�@

error_R�2]?

learning_rate_1먷6]I�7I       6%�	S�wa���A�%*;


total_lossq�@

error_R_M?

learning_rate_1먷6��3�I       6%�	�Gxa���A�%*;


total_loss4��@

error_R/9I?

learning_rate_1먷6X89�I       6%�	/�xa���A�%*;


total_loss��@

error_R�#J?

learning_rate_1먷6s�I       6%�	#ya���A�%*;


total_lossvA

error_Rtz4?

learning_rate_1먷6��ޮI       6%�	Q�ya���A�%*;


total_lossa2�@

error_RW?

learning_rate_1먷6���I       6%�	bAza���A�%*;


total_loss���@

error_R�$6?

learning_rate_1먷6�^��I       6%�	�za���A�%*;


total_loss��A

error_R��_?

learning_rate_1먷6�x�FI       6%�	j�za���A�%*;


total_lossq�@

error_R�"M?

learning_rate_1먷6���I       6%�	*{a���A�%*;


total_loss�U�@

error_R.[O?

learning_rate_1먷6F
�I       6%�	q{a���A�%*;


total_loss��@

error_R�lI?

learning_rate_1먷6��I       6%�	.�{a���A�%*;


total_loss���@

error_Rq�??

learning_rate_1먷6i<��I       6%�	.|a���A�%*;


total_loss��@

error_RnL?

learning_rate_1먷6���kI       6%�	�||a���A�%*;


total_loss�]�@

error_R��B?

learning_rate_1먷6�H�I       6%�	C�|a���A�%*;


total_loss�i�@

error_R�F?

learning_rate_1먷6G"��I       6%�	�@}a���A�%*;


total_loss�S�@

error_R��A?

learning_rate_1먷6��`LI       6%�	��}a���A�%*;


total_loss]��@

error_R�jM?

learning_rate_1먷6ي1�I       6%�	��}a���A�%*;


total_lossz3�@

error_R��H?

learning_rate_1먷6cJ[hI       6%�	/~a���A�%*;


total_loss�x�@

error_R�pI?

learning_rate_1먷6�D`VI       6%�	�f~a���A�%*;


total_loss*E�@

error_R!BD?

learning_rate_1먷6Sbv�I       6%�	��~a���A�%*;


total_loss���@

error_R��R?

learning_rate_1먷6p�ɎI       6%�	}(a���A�%*;


total_loss��@

error_R��:?

learning_rate_1먷6y*6}I       6%�	�la���A�%*;


total_loss�'~@

error_R! G?

learning_rate_1먷6݌��I       6%�	��a���A�%*;


total_losszM�@

error_R:�S?

learning_rate_1먷6�P��I       6%�	��a���A�%*;


total_lossC�@

error_R%�K?

learning_rate_1먷6�bPI       6%�	�2�a���A�%*;


total_lossz'�@

error_R�R?

learning_rate_1먷6wj��I       6%�	eu�a���A�%*;


total_loss�-�@

error_R4�N?

learning_rate_1먷6d1�I       6%�	K��a���A�%*;


total_loss��@

error_R�?J?

learning_rate_1먷6g�[�I       6%�	b��a���A�%*;


total_loss�@

error_Rn0A?

learning_rate_1먷6�<�"I       6%�	37�a���A�%*;


total_loss�{�@

error_R�ZV?

learning_rate_1먷6�;$�I       6%�	�y�a���A�%*;


total_loss�@�@

error_R�G?

learning_rate_1먷6���I       6%�	¼�a���A�%*;


total_loss]��@

error_R�\Q?

learning_rate_1먷6�O�I       6%�	�+�a���A�%*;


total_loss�Ȫ@

error_R9V?

learning_rate_1먷6t��
I       6%�	�r�a���A�%*;


total_loss���@

error_R�vA?

learning_rate_1먷6#ifxI       6%�	3��a���A�%*;


total_lossJ��@

error_R�O?

learning_rate_1먷6�fS�I       6%�	1��a���A�%*;


total_lossO��@

error_R$C?

learning_rate_1먷6��I       6%�	7�a���A�%*;


total_loss�E�@

error_R ZX?

learning_rate_1먷6́�I       6%�	�{�a���A�%*;


total_loss���@

error_R��G?

learning_rate_1먷6f� �I       6%�	|��a���A�%*;


total_loss��@

error_R��>?

learning_rate_1먷6c��I       6%�	���a���A�%*;


total_loss蚀@

error_R�rR?

learning_rate_1먷6!���I       6%�	�<�a���A�%*;


total_loss,+�@

error_R��]?

learning_rate_1먷6V�wkI       6%�	�|�a���A�%*;


total_loss�I�@

error_R,G?

learning_rate_1먷6�~��I       6%�	1��a���A�%*;


total_loss��@

error_R��F?

learning_rate_1먷6�0��I       6%�	3 �a���A�%*;


total_loss�$�@

error_R�A?

learning_rate_1먷6��^I       6%�	6D�a���A�%*;


total_loss���@

error_Rf�7?

learning_rate_1먷6�mT�I       6%�	u��a���A�%*;


total_loss��@

error_R�u_?

learning_rate_1먷6ޫ�^I       6%�	zυa���A�%*;


total_loss!> A

error_R�:A?

learning_rate_1먷6�7�,I       6%�	F�a���A�%*;


total_loss�V�@

error_RjE?

learning_rate_1먷6��cI       6%�	AR�a���A�%*;


total_loss�z@

error_R��S?

learning_rate_1먷6��F�I       6%�	X��a���A�%*;


total_loss��@

error_RCrg?

learning_rate_1먷6c�M�I       6%�	�Ԇa���A�%*;


total_loss�@

error_Rl�H?

learning_rate_1먷6�I       6%�	��a���A�%*;


total_loss��@

error_R��K?

learning_rate_1먷6�W��I       6%�	�W�a���A�%*;


total_loss���@

error_R��@?

learning_rate_1먷6c#pI       6%�	���a���A�%*;


total_loss���@

error_R\�F?

learning_rate_1먷6�Y�I       6%�	�هa���A�%*;


total_loss�!�@

error_RHPG?

learning_rate_1먷6��b�I       6%�	W8�a���A�%*;


total_loss8��@

error_R<P?

learning_rate_1먷6=��xI       6%�	�{�a���A�&*;


total_loss(̗@

error_Rh�O?

learning_rate_1먷62�I       6%�	很a���A�&*;


total_loss�*�@

error_RR�M?

learning_rate_1먷6��e�I       6%�	2�a���A�&*;


total_loss}��@

error_R�O?

learning_rate_1먷6��?~I       6%�	�b�a���A�&*;


total_loss��@

error_RE F?

learning_rate_1먷6�9�I       6%�	���a���A�&*;


total_lossÅ�@

error_R�Q?

learning_rate_1먷6�u�I       6%�	��a���A�&*;


total_loss�u�@

error_RA?

learning_rate_1먷6�h?�I       6%�	z.�a���A�&*;


total_loss��@

error_R�,C?

learning_rate_1먷6��zI       6%�	�p�a���A�&*;


total_losslXA

error_R��N?

learning_rate_1먷6}J2I       6%�	ϲ�a���A�&*;


total_loss��@

error_Ra�J?

learning_rate_1먷6⸗fI       6%�	��a���A�&*;


total_loss��@

error_R�7G?

learning_rate_1먷6�iq[I       6%�	�7�a���A�&*;


total_loss���@

error_R3;H?

learning_rate_1먷6�\�MI       6%�	�w�a���A�&*;


total_loss�J�@

error_RvR?

learning_rate_1먷6	T),I       6%�	ʷ�a���A�&*;


total_loss?{�@

error_R�Q\?

learning_rate_1먷6վa�I       6%�	i��a���A�&*;


total_lossd:�@

error_R�&P?

learning_rate_1먷6�᳹I       6%�	);�a���A�&*;


total_lossץ�@

error_R�ST?

learning_rate_1먷6紌PI       6%�	�{�a���A�&*;


total_loss6}�@

error_R�V?

learning_rate_1먷6^��JI       6%�	S��a���A�&*;


total_lossȺ�@

error_R&�E?

learning_rate_1먷66��I       6%�	$�a���A�&*;


total_loss�Vf@

error_R��G?

learning_rate_1먷6�n	I       6%�	zg�a���A�&*;


total_loss�W�@

error_R��Y?

learning_rate_1먷6\G*?I       6%�	��a���A�&*;


total_loss<n�@

error_R[�I?

learning_rate_1먷6���I       6%�	Y�a���A�&*;


total_loss�t�@

error_R�J?

learning_rate_1먷6��iI       6%�	M,�a���A�&*;


total_loss@��@

error_R@�R?

learning_rate_1먷6�89�I       6%�	1l�a���A�&*;


total_loss�SA

error_R�`R?

learning_rate_1먷6��I       6%�	~��a���A�&*;


total_losst[@

error_R��\?

learning_rate_1먷6_��tI       6%�	}�a���A�&*;


total_loss7��@

error_R1�g?

learning_rate_1먷6�]��I       6%�	�6�a���A�&*;


total_loss�=�@

error_R��U?

learning_rate_1먷6��TI       6%�	ky�a���A�&*;


total_lossG<�@

error_R�v[?

learning_rate_1먷6�+�hI       6%�	���a���A�&*;


total_loss7K�@

error_R�$R?

learning_rate_1먷62�I       6%�	���a���A�&*;


total_loss~�A

error_R�R?

learning_rate_1먷6H��I       6%�	?�a���A�&*;


total_loss��@

error_R|O?

learning_rate_1먷6#�LI       6%�	���a���A�&*;


total_loss���@

error_R� P?

learning_rate_1먷6�˾�I       6%�	���a���A�&*;


total_lossRu�@

error_R\*J?

learning_rate_1먷6�}R�I       6%�	��a���A�&*;


total_loss�"�@

error_R��S?

learning_rate_1먷6�RrI       6%�	A�a���A�&*;


total_loss@��@

error_R*�_?

learning_rate_1먷62��5I       6%�	���a���A�&*;


total_loss��@

error_R��S?

learning_rate_1먷6kN0I       6%�	�a���A�&*;


total_loss�1 A

error_RmL?

learning_rate_1먷6_2k7I       6%�	\�a���A�&*;


total_loss�g�@

error_R�F?

learning_rate_1먷6�!{�I       6%�	+E�a���A�&*;


total_loss���@

error_R��8?

learning_rate_1먷6�ZtGI       6%�	r��a���A�&*;


total_loss�w�@

error_Rf�K?

learning_rate_1먷6_�iI       6%�	�˒a���A�&*;


total_loss	�@

error_R�U?

learning_rate_1먷6z��I       6%�	�a���A�&*;


total_loss4x�@

error_R$KU?

learning_rate_1먷6,kյI       6%�	n[�a���A�&*;


total_loss.f�@

error_R�fT?

learning_rate_1먷6���AI       6%�	a��a���A�&*;


total_loss���@

error_R-*F?

learning_rate_1먷6S�5$I       6%�	���a���A�&*;


total_loss�=�@

error_R`&4?

learning_rate_1먷6�H`�I       6%�	L!�a���A�&*;


total_loss�qA

error_RQzI?

learning_rate_1먷6��V"I       6%�	�c�a���A�&*;


total_loss�@

error_RʩN?

learning_rate_1먷6�PлI       6%�	���a���A�&*;


total_loss��@

error_R6�I?

learning_rate_1먷6>mI       6%�	��a���A�&*;


total_losse��@

error_R��C?

learning_rate_1먷6(�eNI       6%�	:�a���A�&*;


total_lossf�@

error_R&oH?

learning_rate_1먷6���I       6%�	?��a���A�&*;


total_loss_b�@

error_R1�H?

learning_rate_1먷6�B�*I       6%�	��a���A�&*;


total_lossU�@

error_R�(D?

learning_rate_1먷6�ƸI       6%�	*�a���A�&*;


total_loss���@

error_R�^K?

learning_rate_1먷6���VI       6%�	��a���A�&*;


total_loss���@

error_R��H?

learning_rate_1먷6F\�+I       6%�	��a���A�&*;


total_loss���@

error_RKO?

learning_rate_1먷6o���I       6%�	�'�a���A�&*;


total_loss$��@

error_R��K?

learning_rate_1먷6�P�XI       6%�	�j�a���A�&*;


total_loss�@�@

error_R��I?

learning_rate_1먷6d�p<I       6%�	���a���A�&*;


total_loss�@

error_Rq4F?

learning_rate_1먷6s�'�I       6%�	f�a���A�&*;


total_loss�S�@

error_R_5F?

learning_rate_1먷6��AnI       6%�	�4�a���A�&*;


total_loss���@

error_R�I?

learning_rate_1먷6�mkwI       6%�	x�a���A�&*;


total_loss�*�@

error_R��_?

learning_rate_1먷6��?I       6%�	���a���A�&*;


total_loss-r�@

error_RvwN?

learning_rate_1먷6�O�SI       6%�	� �a���A�&*;


total_loss�?�@

error_R�G?

learning_rate_1먷6�w�SI       6%�	MF�a���A�&*;


total_lossn<NA

error_Rb?

learning_rate_1먷6:S�oI       6%�	��a���A�&*;


total_loss�@

error_R�^?

learning_rate_1먷6%ŢzI       6%�	͙a���A�&*;


total_lossx��@

error_R��a?

learning_rate_1먷6�~I       6%�	��a���A�&*;


total_loss2�@

error_R[�I?

learning_rate_1먷6X���I       6%�	�Q�a���A�&*;


total_lossi��@

error_R�u>?

learning_rate_1먷6dW4�I       6%�	���a���A�&*;


total_loss�A

error_Rq�R?

learning_rate_1먷6�R��I       6%�	U֚a���A�&*;


total_loss��A

error_R��>?

learning_rate_1먷6�F�I       6%�	+�a���A�&*;


total_loss@��@

error_R��D?

learning_rate_1먷6��(�I       6%�	���a���A�&*;


total_lossl��@

error_R��f?

learning_rate_1먷6z8�I       6%�	W��a���A�&*;


total_lossQ��@

error_R_�d?

learning_rate_1먷6I��I       6%�	�K�a���A�&*;


total_loss\�\@

error_R'X?

learning_rate_1먷6מuI       6%�	&��a���A�&*;


total_lossS��@

error_R1�N?

learning_rate_1먷6X��VI       6%�	&��a���A�&*;


total_loss��@

error_RG?

learning_rate_1먷6+���I       6%�	�]�a���A�&*;


total_loss�[�@

error_RȩU?

learning_rate_1먷6#��I       6%�	���a���A�&*;


total_loss?�@

error_R��T?

learning_rate_1먷6� �I       6%�	���a���A�&*;


total_loss�`�@

error_R�\?

learning_rate_1먷6�� �I       6%�	�B�a���A�&*;


total_loss�v�@

error_RW�O?

learning_rate_1먷62�S�I       6%�	#��a���A�&*;


total_lossf��@

error_R3bR?

learning_rate_1먷6O�jFI       6%�	O�a���A�&*;


total_loss��@

error_Ra?

learning_rate_1먷6���I       6%�	3e�a���A�&*;


total_loss�ڔ@

error_RdAD?

learning_rate_1먷6o	X�I       6%�	ȱ�a���A�&*;


total_lossDΏ@

error_R_�P?

learning_rate_1먷6H&PI       6%�	: �a���A�&*;


total_lossw.�@

error_R��K?

learning_rate_1먷6{�S�I       6%�	�b�a���A�&*;


total_lossѨ@

error_R�_?

learning_rate_1먷61I�%I       6%�	O��a���A�&*;


total_loss��@

error_R�K?

learning_rate_1먷6^�I       6%�	���a���A�&*;


total_loss�8�@

error_Ra�C?

learning_rate_1먷62�grI       6%�	A�a���A�&*;


total_loss�B�@

error_R�P?

learning_rate_1먷6�WI       6%�	&��a���A�&*;


total_lossҒ�@

error_R�K?

learning_rate_1먷6�I��I       6%�	�סa���A�&*;


total_loss��m@

error_R$�D?

learning_rate_1먷6�aI       6%�	�*�a���A�&*;


total_loss��A

error_Rf ^?

learning_rate_1먷6����I       6%�	A��a���A�&*;


total_loss���@

error_R�4?

learning_rate_1먷6#�uaI       6%�	�ۢa���A�&*;


total_lossȟ�@

error_R��X?

learning_rate_1먷6���I       6%�	^'�a���A�&*;


total_loss�&�@

error_R�kN?

learning_rate_1먷6�I       6%�	�q�a���A�&*;


total_loss��@

error_R�}S?

learning_rate_1먷6�@��I       6%�	�£a���A�&*;


total_lossT��@

error_Re=?

learning_rate_1먷6��DI       6%�	��a���A�&*;


total_losstH�@

error_RX�[?

learning_rate_1먷66n�I       6%�	Pz�a���A�&*;


total_loss�v@

error_R}�U?

learning_rate_1먷6:�nUI       6%�	Ťa���A�&*;


total_loss��j@

error_R
2D?

learning_rate_1먷6	|rkI       6%�	�
�a���A�&*;


total_loss�'�@

error_R��??

learning_rate_1먷6񦹯I       6%�	�^�a���A�&*;


total_loss��@

error_Rȇ;?

learning_rate_1먷60D�I       6%�	��a���A�&*;


total_loss�I�@

error_R�:?

learning_rate_1먷6�sg�I       6%�	�a���A�&*;


total_loss�@

error_R�F?

learning_rate_1먷6�̀@I       6%�	�%�a���A�&*;


total_loss�ݍ@

error_R.K?

learning_rate_1먷6��2�I       6%�	-g�a���A�&*;


total_lossњ�@

error_R��K?

learning_rate_1먷6��_I       6%�	���a���A�&*;


total_loss���@

error_R�Sc?

learning_rate_1먷6�2	RI       6%�	���a���A�&*;


total_loss��@

error_RTV?

learning_rate_1먷6��0I       6%�	C�a���A�&*;


total_loss���@

error_R>?

learning_rate_1먷6/��4I       6%�	���a���A�&*;


total_loss���@

error_R�Ue?

learning_rate_1먷6��+�I       6%�	Lɧa���A�&*;


total_loss�
@

error_R�Q?

learning_rate_1먷6J"��I       6%�	��a���A�&*;


total_loss�t�@

error_R��G?

learning_rate_1먷6ӝ�lI       6%�	�M�a���A�&*;


total_lossZ��@

error_RC�G?

learning_rate_1먷6'��I       6%�	#��a���A�&*;


total_loss '�@

error_R85E?

learning_rate_1먷6��(I       6%�	��a���A�&*;


total_lossc�@

error_R�|D?

learning_rate_1먷6���I       6%�	(*�a���A�&*;


total_loss�O�@

error_R��M?

learning_rate_1먷6��ccI       6%�	�h�a���A�&*;


total_loss���@

error_R�LS?

learning_rate_1먷6�J�I       6%�	���a���A�&*;


total_loss�h�@

error_R�D@?

learning_rate_1먷6�0�}I       6%�	A�a���A�&*;


total_lossɗ�@

error_R<OO?

learning_rate_1먷6����I       6%�	5-�a���A�&*;


total_loss@Z�@

error_R��e?

learning_rate_1먷6!x��I       6%�	�n�a���A�&*;


total_loss���@

error_R�#R?

learning_rate_1먷6����I       6%�	���a���A�&*;


total_loss��@

error_R�<P?

learning_rate_1먷6ہ�,I       6%�	�a���A�&*;


total_lossm4�@

error_RLT?

learning_rate_1먷6��tkI       6%�	�6�a���A�&*;


total_loss}�@

error_R�*I?

learning_rate_1먷6�09I       6%�	|y�a���A�&*;


total_losse�@

error_RWyR?

learning_rate_1먷6���I       6%�	ݺ�a���A�&*;


total_loss�@

error_R�/A?

learning_rate_1먷6�UZ
I       6%�	,��a���A�&*;


total_lossf<�@

error_R��[?

learning_rate_1먷6(�?CI       6%�	 A�a���A�&*;


total_loss���@

error_R�sC?

learning_rate_1먷6���I       6%�	���a���A�&*;


total_loss(2�@

error_R�B=?

learning_rate_1먷6��jI       6%�	�Ƭa���A�'*;


total_loss/@

error_R�]O?

learning_rate_1먷6�`-I       6%�	�(�a���A�'*;


total_loss���@

error_R��I?

learning_rate_1먷69�|I       6%�	 p�a���A�'*;


total_loss]��@

error_RH�G?

learning_rate_1먷6ݑQI       6%�	-��a���A�'*;


total_loss��@

error_R��O?

learning_rate_1먷6D/�I       6%�	��a���A�'*;


total_loss�@

error_R�R?

learning_rate_1먷6�q�CI       6%�	�3�a���A�'*;


total_loss/��@

error_RZ�`?

learning_rate_1먷6����I       6%�	�s�a���A�'*;


total_loss_��@

error_Rv:?

learning_rate_1먷6�_��I       6%�	���a���A�'*;


total_lossr��@

error_Rn�L?

learning_rate_1먷6�j��I       6%�	W��a���A�'*;


total_lossT�@

error_R�L?

learning_rate_1먷6��I       6%�	G=�a���A�'*;


total_loss�;�@

error_RѶ_?

learning_rate_1먷6�:I       6%�	��a���A�'*;


total_lossD�@

error_R�U?

learning_rate_1먷6�k�]I       6%�	�߲a���A�'*;


total_loss&�@

error_R��??

learning_rate_1먷6mC(�I       6%�	�:�a���A�'*;


total_loss�(�@

error_R�XT?

learning_rate_1먷6n��VI       6%�	��a���A�'*;


total_loss�f�@

error_R��P?

learning_rate_1먷6�]�I       6%�	!ȳa���A�'*;


total_lossֈ�@

error_Rx?<?

learning_rate_1먷6�H-�I       6%�	!�a���A�'*;


total_loss���@

error_Rq�T?

learning_rate_1먷6���I       6%�	�k�a���A�'*;


total_loss�i�@

error_R4pG?

learning_rate_1먷6l�oiI       6%�	���a���A�'*;


total_loss�)A

error_R��d?

learning_rate_1먷68�LI       6%�	��a���A�'*;


total_loss�eA

error_R�X?

learning_rate_1먷6��Q-I       6%�	#0�a���A�'*;


total_loss���@

error_RZVX?

learning_rate_1먷6<��~I       6%�	v�a���A�'*;


total_loss��@

error_R��F?

learning_rate_1먷6��OLI       6%�	���a���A�'*;


total_loss3��@

error_R}�E?

learning_rate_1먷6*D&(I       6%�	#��a���A�'*;


total_loss�y�@

error_R*�9?

learning_rate_1먷6����I       6%�	�5�a���A�'*;


total_loss��@

error_RV�R?

learning_rate_1먷6ʇM�I       6%�	Ot�a���A�'*;


total_loss;$�@

error_R��Z?

learning_rate_1먷6�4�I       6%�	v��a���A�'*;


total_loss~��@

error_Rd�U?

learning_rate_1먷6sb�I       6%�	���a���A�'*;


total_loss�{�@

error_Rl�Q?

learning_rate_1먷6e�`eI       6%�	�7�a���A�'*;


total_loss�P�@

error_R�>E?

learning_rate_1먷6�agI       6%�	�x�a���A�'*;


total_loss�C�@

error_R��N?

learning_rate_1먷6���_I       6%�	븷a���A�'*;


total_loss���@

error_R�l?

learning_rate_1먷6��HI       6%�	\��a���A�'*;


total_loss$�@

error_Rx�U?

learning_rate_1먷6���I       6%�	�H�a���A�'*;


total_loss�[�@

error_R�DS?

learning_rate_1먷6��I       6%�	T��a���A�'*;


total_loss�]^@

error_R�E?

learning_rate_1먷6�W�rI       6%�	��a���A�'*;


total_loss�Ε@

error_R�R?

learning_rate_1먷6���I       6%�	�3�a���A�'*;


total_loss! �@

error_Ro�L?

learning_rate_1먷6GvI       6%�	�x�a���A�'*;


total_loss���@

error_R��G?

learning_rate_1먷6���I       6%�	���a���A�'*;


total_lossyI�@

error_R�[?

learning_rate_1먷6��0�I       6%�	R��a���A�'*;


total_loss\At@

error_R\�U?

learning_rate_1먷6+"	I       6%�	�<�a���A�'*;


total_loss�b�@

error_Rڧ\?

learning_rate_1먷6r��^I       6%�	}�a���A�'*;


total_loss��@

error_R��[?

learning_rate_1먷6���I       6%�	a��a���A�'*;


total_loss�:�@

error_RS?

learning_rate_1먷6w�I       6%�	&�a���A�'*;


total_loss�q�@

error_R��H?

learning_rate_1먷6���I       6%�	�B�a���A�'*;


total_loss{�@

error_R�>?

learning_rate_1먷6����I       6%�	���a���A�'*;


total_loss���@

error_R�C?

learning_rate_1먷6����I       6%�	�Ļa���A�'*;


total_loss���@

error_R��[?

learning_rate_1먷6���I       6%�	�a���A�'*;


total_loss�9A

error_R��M?

learning_rate_1먷6C���I       6%�	G�a���A�'*;


total_lossa=�@

error_R��f?

learning_rate_1먷6�M�FI       6%�	��a���A�'*;


total_loss�J�@

error_R�0J?

learning_rate_1먷6���qI       6%�	�a���A�'*;


total_loss-T�@

error_R:-:?

learning_rate_1먷6A���I       6%�	ld�a���A�'*;


total_loss��@

error_RoP?

learning_rate_1먷6ㅘ�I       6%�	c��a���A�'*;


total_loss�JA

error_R8�M?

learning_rate_1먷6���I       6%�	�a���A�'*;


total_loss���@

error_R*�R?

learning_rate_1먷6��T�I       6%�	�b�a���A�'*;


total_lossx�I@

error_R��F?

learning_rate_1먷6�t"�I       6%�	���a���A�'*;


total_lossA �@

error_Ra�A?

learning_rate_1먷6u��I       6%�	��a���A�'*;


total_loss��o@

error_R{	_?

learning_rate_1먷6J"�I       6%�	.�a���A�'*;


total_loss֏@

error_RX	F?

learning_rate_1먷6���I       6%�	\��a���A�'*;


total_loss�2�@

error_RI�A?

learning_rate_1먷6}s��I       6%�	J�a���A�'*;


total_loss
3�@

error_R@�M?

learning_rate_1먷6��>kI       6%�	�1�a���A�'*;


total_loss�c�@

error_R<X?

learning_rate_1먷6Ͷ;�I       6%�	w�a���A�'*;


total_loss�&�@

error_R$�L?

learning_rate_1먷6AxÙI       6%�	���a���A�'*;


total_lossLO�@

error_R�TF?

learning_rate_1먷6�)'I       6%�	�'�a���A�'*;


total_lossͤ�@

error_R�@?

learning_rate_1먷6'��I       6%�	�g�a���A�'*;


total_losshc�@

error_R�,=?

learning_rate_1먷6��ĩI       6%�	��a���A�'*;


total_loss" A

error_R��M?

learning_rate_1먷6��fwI       6%�	�a���A�'*;


total_lossd��@

error_R�uK?

learning_rate_1먷6�D�I       6%�	`�a���A�'*;


total_loss|��@

error_R��E?

learning_rate_1먷6^� I       6%�	K��a���A�'*;


total_loss���@

error_Ri�G?

learning_rate_1먷6k���I       6%�	g��a���A�'*;


total_lossG�@

error_R�YH?

learning_rate_1먷6;��I       6%�	�1�a���A�'*;


total_lossDY�@

error_R�DI?

learning_rate_1먷6����I       6%�	r�a���A�'*;


total_loss��@

error_R��Q?

learning_rate_1먷6�˒I       6%�	,��a���A�'*;


total_losst��@

error_R��L?

learning_rate_1먷6���I       6%�	i��a���A�'*;


total_loss��@

error_R3c?

learning_rate_1먷6P"�#I       6%�	�4�a���A�'*;


total_loss�pA

error_R_iI?

learning_rate_1먷6�ZI       6%�	x�a���A�'*;


total_loss��@

error_R29M?

learning_rate_1먷6f"(�I       6%�	"��a���A�'*;


total_loss-��@

error_R�a?

learning_rate_1먷6�pEJI       6%�	^��a���A�'*;


total_lossw��@

error_R}�:?

learning_rate_1먷6-�+I       6%�	�:�a���A�'*;


total_loss�(�@

error_RfS?

learning_rate_1먷6z�/I       6%�	8{�a���A�'*;


total_loss���@

error_R��U?

learning_rate_1먷6�m6I       6%�	��a���A�'*;


total_loss���@

error_Rc�_?

learning_rate_1먷6��sII       6%�	� �a���A�'*;


total_loss4Ԫ@

error_R�wG?

learning_rate_1먷6 ��3I       6%�	�C�a���A�'*;


total_lossCg�@

error_R�P?

learning_rate_1먷6mҿMI       6%�	���a���A�'*;


total_loss�@

error_R��X?

learning_rate_1먷6�zG-I       6%�	��a���A�'*;


total_loss�@

error_R�kH?

learning_rate_1먷6��7DI       6%�	��a���A�'*;


total_loss]i�@

error_R{�G?

learning_rate_1먷6ӡԖI       6%�	�F�a���A�'*;


total_loss��@

error_RJ$=?

learning_rate_1먷63谘I       6%�	`��a���A�'*;


total_loss˗@

error_R�m=?

learning_rate_1먷64�I       6%�	���a���A�'*;


total_loss��@

error_RVlB?

learning_rate_1먷6D�x�I       6%�	*	�a���A�'*;


total_lossa�@

error_R�jL?

learning_rate_1먷6V��qI       6%�	J�a���A�'*;


total_loss���@

error_R$�H?

learning_rate_1먷6��AI       6%�	R��a���A�'*;


total_loss�`�@

error_RV�A?

learning_rate_1먷6�d�I       6%�	��a���A�'*;


total_loss{+�@

error_RT�@?

learning_rate_1먷6��JI       6%�	��a���A�'*;


total_loss�Y�@

error_RO?

learning_rate_1먷6߃�7I       6%�	�U�a���A�'*;


total_losst6�@

error_R�O?

learning_rate_1먷6�ð.I       6%�	��a���A�'*;


total_lossd��@

error_R�a?

learning_rate_1먷6͞�nI       6%�	���a���A�'*;


total_loss��@

error_R�fE?

learning_rate_1먷6ۦ�@I       6%�	U�a���A�'*;


total_loss*�@

error_RO&E?

learning_rate_1먷6sc��I       6%�	$a�a���A�'*;


total_loss�A

error_RS�D?

learning_rate_1먷6��<�I       6%�	���a���A�'*;


total_lossȧ�@

error_R�K?

learning_rate_1먷6��I       6%�	���a���A�'*;


total_loss��z@

error_RopO?

learning_rate_1먷6�N{I       6%�	 (�a���A�'*;


total_lossh>�@

error_R2�R?

learning_rate_1먷6D5��I       6%�	`h�a���A�'*;


total_loss���@

error_Rf�=?

learning_rate_1먷6���I       6%�	��a���A�'*;


total_loss��@

error_Rv�7?

learning_rate_1먷6��@�I       6%�	2��a���A�'*;


total_lossW��@

error_R��L?

learning_rate_1먷6��ۥI       6%�	�=�a���A�'*;


total_lossI��@

error_R%*?

learning_rate_1먷62��I       6%�	l��a���A�'*;


total_lossV|�@

error_R@R8?

learning_rate_1먷6��I       6%�	���a���A�'*;


total_loss�f�@

error_R��??

learning_rate_1먷6�[�I       6%�	gQ�a���A�'*;


total_loss
7�@

error_R;@?

learning_rate_1먷6�[I       6%�	x��a���A�'*;


total_lossf��@

error_RM_?

learning_rate_1먷6��g2I       6%�	���a���A�'*;


total_loss ��@

error_R��Q?

learning_rate_1먷6��dI       6%�	��a���A�'*;


total_loss��A

error_R��_?

learning_rate_1먷6 e�I       6%�	r[�a���A�'*;


total_lossn�@

error_R$�K?

learning_rate_1먷6\��I       6%�	���a���A�'*;


total_loss8�A

error_RrhS?

learning_rate_1먷6B�?*I       6%�	���a���A�'*;


total_loss�ߒ@

error_R�X?

learning_rate_1먷6?�I       6%�	��a���A�'*;


total_loss��@

error_RayX?

learning_rate_1먷6-��I       6%�	]�a���A�'*;


total_loss��@

error_R�vS?

learning_rate_1먷6���I       6%�	"��a���A�'*;


total_loss�F�@

error_R��W?

learning_rate_1먷6Q ��I       6%�	���a���A�'*;


total_loss	Qi@

error_R��T?

learning_rate_1먷6\���I       6%�	N�a���A�'*;


total_loss���@

error_R�[T?

learning_rate_1먷6����I       6%�	�_�a���A�'*;


total_lossʻ�@

error_R� K?

learning_rate_1먷6�2�,I       6%�	k��a���A�'*;


total_loss�~A

error_R� V?

learning_rate_1먷6{y\�I       6%�	~��a���A�'*;


total_loss���@

error_R�#I?

learning_rate_1먷6x�B�I       6%�	�"�a���A�'*;


total_lossF#�@

error_R�C?

learning_rate_1먷6D�\�I       6%�	�d�a���A�'*;


total_loss�.�@

error_R��H?

learning_rate_1먷6��pI       6%�	5��a���A�'*;


total_loss�̨@

error_R��K?

learning_rate_1먷6���yI       6%�	��a���A�'*;


total_lossq%�@

error_R��o?

learning_rate_1먷6�Z�jI       6%�	[%�a���A�'*;


total_loss_)�@

error_R�8?

learning_rate_1먷6�#ƒI       6%�	�f�a���A�'*;


total_loss�l�@

error_R�$;?

learning_rate_1먷6�V�I       6%�	e��a���A�'*;


total_lossޔ�@

error_R�G?

learning_rate_1먷6,��I       6%�	7��a���A�(*;


total_loss��@

error_R�L?

learning_rate_1먷6�T0I       6%�	J&�a���A�(*;


total_lossV��@

error_RH�B?

learning_rate_1먷6�Sf�I       6%�	ee�a���A�(*;


total_loss�)�@

error_R8�E?

learning_rate_1먷6j��I       6%�	��a���A�(*;


total_loss���@

error_R1�N?

learning_rate_1먷6���:I       6%�	B��a���A�(*;


total_losseV�@

error_R�TF?

learning_rate_1먷6đ9�I       6%�	0�a���A�(*;


total_loss���@

error_R}`?

learning_rate_1먷6ռ\�I       6%�	�p�a���A�(*;


total_loss���@

error_R��V?

learning_rate_1먷6�[>�I       6%�	���a���A�(*;


total_loss�B�@

error_R.{J?

learning_rate_1먷6�|`HI       6%�	k��a���A�(*;


total_loss���@

error_R�@C?

learning_rate_1먷6_J��I       6%�	m.�a���A�(*;


total_loss�S	A

error_Rc^E?

learning_rate_1먷6`��~I       6%�	1n�a���A�(*;


total_loss��@

error_Rð3?

learning_rate_1먷6|M4�I       6%�	���a���A�(*;


total_loss��@

error_R�B@?

learning_rate_1먷6�H.7I       6%�	s��a���A�(*;


total_lossN��@

error_R�[?

learning_rate_1먷6���I       6%�	{0�a���A�(*;


total_loss{J�@

error_RC?

learning_rate_1먷6G~I       6%�	&q�a���A�(*;


total_lossԖ�@

error_R�eG?

learning_rate_1먷6*M�I       6%�	@��a���A�(*;


total_loss$Rt@

error_R��M?

learning_rate_1먷6��I       6%�	��a���A�(*;


total_loss��@

error_R��B?

learning_rate_1먷6��ӗI       6%�	0�a���A�(*;


total_loss c�@

error_R�	g?

learning_rate_1먷6�<*�I       6%�	Uq�a���A�(*;


total_loss���@

error_Ri>L?

learning_rate_1먷6�1u?I       6%�	��a���A�(*;


total_lossἵ@

error_R�hP?

learning_rate_1먷6+ꝼI       6%�	���a���A�(*;


total_loss�,�@

error_Rs�:?

learning_rate_1먷6j���I       6%�	�@�a���A�(*;


total_loss��@

error_R��a?

learning_rate_1먷60W�bI       6%�	���a���A�(*;


total_lossxx�@

error_RHP?

learning_rate_1먷6tٯ[I       6%�	���a���A�(*;


total_loss�ڼ@

error_R�Z?

learning_rate_1먷6��bI       6%�	���a���A�(*;


total_loss1�@

error_R��>?

learning_rate_1먷6�e�I       6%�	_>�a���A�(*;


total_loss�	�@

error_Rn�V?

learning_rate_1먷6��$I       6%�	���a���A�(*;


total_loss���@

error_R|�>?

learning_rate_1먷6�k]�I       6%�	���a���A�(*;


total_loss$�@

error_R��W?

learning_rate_1먷6`��DI       6%�	 �a���A�(*;


total_loss���@

error_R��G?

learning_rate_1먷6�&�)I       6%�	�?�a���A�(*;


total_loss7��@

error_R��j?

learning_rate_1먷6���I       6%�	�~�a���A�(*;


total_lossX}�@

error_R��q?

learning_rate_1먷6,.J�I       6%�	���a���A�(*;


total_loss��@

error_R�(T?

learning_rate_1먷6,:�I       6%�	� �a���A�(*;


total_lossfƚ@

error_R�=b?

learning_rate_1먷6�0��I       6%�	
?�a���A�(*;


total_loss�g|@

error_R�qF?

learning_rate_1먷6�8L?I       6%�	y~�a���A�(*;


total_loss���@

error_R��N?

learning_rate_1먷6B�L�I       6%�	���a���A�(*;


total_loss�FA

error_R�[?

learning_rate_1먷6��(ZI       6%�	D�a���A�(*;


total_loss@y�@

error_R��Q?

learning_rate_1먷6V���I       6%�	�G�a���A�(*;


total_loss���@

error_R��M?

learning_rate_1먷6C!�ZI       6%�	���a���A�(*;


total_loss��@

error_RȋE?

learning_rate_1먷6����I       6%�	���a���A�(*;


total_lossm]�@

error_R�Y?

learning_rate_1먷6,@��I       6%�	�.�a���A�(*;


total_loss���@

error_R�S?

learning_rate_1먷6s��I       6%�	�p�a���A�(*;


total_loss�'�@

error_Rn�C?

learning_rate_1먷6���I       6%�	��a���A�(*;


total_loss��@

error_R��X?

learning_rate_1먷68͉I       6%�	D�a���A�(*;


total_loss͜�@

error_R
�@?

learning_rate_1먷6#�H�I       6%�	�m�a���A�(*;


total_loss@�@

error_Rq�T?

learning_rate_1먷6�hD�I       6%�	,��a���A�(*;


total_loss�.�@

error_R=�D?

learning_rate_1먷6w��I       6%�	���a���A�(*;


total_loss2΋@

error_R[�6?

learning_rate_1먷6��K�I       6%�	�4�a���A�(*;


total_lossڻ�@

error_R�}b?

learning_rate_1먷6�[I       6%�	�u�a���A�(*;


total_loss���@

error_R�1V?

learning_rate_1먷6Rm_I       6%�	���a���A�(*;


total_losso��@

error_R�F?

learning_rate_1먷65w��I       6%�	O��a���A�(*;


total_losss�@

error_RA�V?

learning_rate_1먷6����I       6%�	K;�a���A�(*;


total_loss�oA

error_R_�]?

learning_rate_1먷6���%I       6%�	���a���A�(*;


total_lossd �@

error_R*9?

learning_rate_1먷6X��I       6%�	���a���A�(*;


total_loss���@

error_R!�R?

learning_rate_1먷6��w�I       6%�	r�a���A�(*;


total_loss=BA

error_R�=I?

learning_rate_1먷6H��yI       6%�	�H�a���A�(*;


total_loss0�@

error_RM M?

learning_rate_1먷6���iI       6%�	
��a���A�(*;


total_loss5�@

error_R��V?

learning_rate_1먷6�I       6%�	���a���A�(*;


total_loss�o@

error_R��H?

learning_rate_1먷6tN��I       6%�	��a���A�(*;


total_lossV��@

error_Rd?

learning_rate_1먷6 ��I       6%�	�q�a���A�(*;


total_loss�	�@

error_R��D?

learning_rate_1먷6D�C�I       6%�	��a���A�(*;


total_lossތ@

error_R�V?

learning_rate_1먷6$!LOI       6%�	���a���A�(*;


total_loss��@

error_R��X?

learning_rate_1먷6���I       6%�	�=�a���A�(*;


total_loss4[�@

error_R!B?

learning_rate_1먷6D�|3I       6%�	8�a���A�(*;


total_lossѣ�@

error_R�T6?

learning_rate_1먷6���I       6%�	޿�a���A�(*;


total_loss���@

error_R2�U?

learning_rate_1먷6�,*�I       6%�	� �a���A�(*;


total_loss̡�@

error_R��O?

learning_rate_1먷6Z;Q'I       6%�	gD�a���A�(*;


total_loss,��@

error_RM�S?

learning_rate_1먷6�N��I       6%�	���a���A�(*;


total_loss&m�@

error_R.�U?

learning_rate_1먷6��7JI       6%�	t��a���A�(*;


total_lossB�	A

error_R�M?

learning_rate_1먷6�q I       6%�	��a���A�(*;


total_lossÄ�@

error_RlK<?

learning_rate_1먷6<��I       6%�	{G�a���A�(*;


total_lossa��@

error_R�s?

learning_rate_1먷66C[I       6%�	���a���A�(*;


total_lossJv@

error_R��G?

learning_rate_1먷6]��2I       6%�	���a���A�(*;


total_lossy��@

error_R;�E?

learning_rate_1먷6�S~�I       6%�	��a���A�(*;


total_loss��@

error_RTFK?

learning_rate_1먷6+�I       6%�	�R�a���A�(*;


total_loss��@

error_R��_?

learning_rate_1먷6�u�I       6%�	��a���A�(*;


total_loss1�@

error_R�E?

learning_rate_1먷6lI�I       6%�	Z��a���A�(*;


total_loss	�@

error_R��A?

learning_rate_1먷6U�
I       6%�	`�a���A�(*;


total_lossIA

error_R��D?

learning_rate_1먷6J+��I       6%�	ހ�a���A�(*;


total_loss���@

error_R��M?

learning_rate_1먷6�(;�I       6%�	���a���A�(*;


total_loss@S�@

error_RW/C?

learning_rate_1먷6ع-I       6%�	o�a���A�(*;


total_loss�A�@

error_RZ�E?

learning_rate_1먷6&#MnI       6%�	)`�a���A�(*;


total_loss|%�@

error_R{�X?

learning_rate_1먷6,��|I       6%�	��a���A�(*;


total_lossV�@

error_R0I?

learning_rate_1먷6Ҁ�AI       6%�	���a���A�(*;


total_lossTAA

error_R�ZM?

learning_rate_1먷6
a�sI       6%�	$+�a���A�(*;


total_loss���@

error_R�N?

learning_rate_1먷6��g�I       6%�	�i�a���A�(*;


total_lossY��@

error_Ra�K?

learning_rate_1먷6j��?I       6%�	���a���A�(*;


total_loss8��@

error_R3Y?

learning_rate_1먷6Ft�I       6%�	���a���A�(*;


total_lossW�{@

error_R��V?

learning_rate_1먷6��I       6%�	W*�a���A�(*;


total_loss�e�@

error_R��O?

learning_rate_1먷6��6�I       6%�	�k�a���A�(*;


total_loss� �@

error_R��X?

learning_rate_1먷6�jZGI       6%�	���a���A�(*;


total_loss{�@

error_Ro�C?

learning_rate_1먷6�֒uI       6%�	���a���A�(*;


total_lossa��@

error_R|
O?

learning_rate_1먷6N�N�I       6%�	�8�a���A�(*;


total_loss8)�@

error_R�JW?

learning_rate_1먷6���I       6%�	�y�a���A�(*;


total_loss��@

error_R�LD?

learning_rate_1먷6)mrI       6%�	L��a���A�(*;


total_loss�t�@

error_R&�Y?

learning_rate_1먷6�O�wI       6%�	���a���A�(*;


total_loss�t@

error_R��Z?

learning_rate_1먷64��I       6%�	0A�a���A�(*;


total_loss���@

error_R��J?

learning_rate_1먷6�OlI       6%�	z��a���A�(*;


total_loss�4�@

error_R:�E?

learning_rate_1먷6�,>=I       6%�	���a���A�(*;


total_lossxq�@

error_R}�L?

learning_rate_1먷6���7I       6%�	!�a���A�(*;


total_loss8��@

error_Rq%P?

learning_rate_1먷6�'�xI       6%�	|o�a���A�(*;


total_loss�:�@

error_RHM?

learning_rate_1먷6�y��I       6%�	ҳ�a���A�(*;


total_loss62�@

error_R=�K?

learning_rate_1먷6h=�-I       6%�	<��a���A�(*;


total_lossߘ�@

error_RW�B?

learning_rate_1먷6'��/I       6%�	P9�a���A�(*;


total_losso��@

error_REOh?

learning_rate_1먷6�n�#I       6%�	Iy�a���A�(*;


total_loss}��@

error_R��T?

learning_rate_1먷6�5aI       6%�	p��a���A�(*;


total_loss���@

error_Ra`2?

learning_rate_1먷6��d I       6%�	��a���A�(*;


total_loss�I�@

error_RRKE?

learning_rate_1먷6���I       6%�	�6�a���A�(*;


total_loss�i�@

error_R$F?

learning_rate_1먷6!J�@I       6%�	�v�a���A�(*;


total_loss}��@

error_R�$V?

learning_rate_1먷6� ��I       6%�	K��a���A�(*;


total_loss�n�@

error_R��[?

learning_rate_1먷6<
��I       6%�	��a���A�(*;


total_loss���@

error_R�LA?

learning_rate_1먷653}JI       6%�	X6�a���A�(*;


total_lossSR�@

error_RZ�Q?

learning_rate_1먷6�sNTI       6%�	�w�a���A�(*;


total_losst��@

error_R$Q?

learning_rate_1먷6�
��I       6%�	r��a���A�(*;


total_loss ��@

error_R=�V?

learning_rate_1먷6f٨�I       6%�	���a���A�(*;


total_lossJ��@

error_R�V\?

learning_rate_1먷6[S�I       6%�	%B�a���A�(*;


total_loss�U�@

error_R >=?

learning_rate_1먷6��F�I       6%�	8��a���A�(*;


total_lossf͖@

error_R3R?

learning_rate_1먷6xz�uI       6%�	o��a���A�(*;


total_loss঵@

error_R��Q?

learning_rate_1먷6Ʃ��I       6%�	��a���A�(*;


total_lossd��@

error_R��K?

learning_rate_1먷6н�I       6%�	sN�a���A�(*;


total_loss�(l@

error_R�Z?

learning_rate_1먷6�gI       6%�	w��a���A�(*;


total_loss�ї@

error_R<J?

learning_rate_1먷6��2I       6%�	#��a���A�(*;


total_loss�y�@

error_R�N?

learning_rate_1먷6���I       6%�	��a���A�(*;


total_loss�9�@

error_R�P?

learning_rate_1먷6DmI       6%�	�[�a���A�(*;


total_lossMA

error_R�yO?

learning_rate_1먷6��("I       6%�	)��a���A�(*;


total_loss���@

error_R
�R?

learning_rate_1먷6��l�I       6%�	���a���A�(*;


total_loss���@

error_RdnH?

learning_rate_1먷6�`4I       6%�	.�a���A�(*;


total_loss���@

error_RTM?

learning_rate_1먷6W}�I       6%�	?o�a���A�(*;


total_loss{��@

error_R}�U?

learning_rate_1먷6�9��I       6%�	���a���A�)*;


total_loss���@

error_R�|a?

learning_rate_1먷6�	_I       6%�	��a���A�)*;


total_loss���@

error_R�C?

learning_rate_1먷6%k�I       6%�	�4�a���A�)*;


total_lossĊ�@

error_R �B?

learning_rate_1먷6�VaI       6%�	�t�a���A�)*;


total_loss�@�@

error_R3Y?

learning_rate_1먷6���I       6%�	ø�a���A�)*;


total_loss���@

error_RL�a?

learning_rate_1먷62�@�I       6%�	��a���A�)*;


total_loss!��@

error_R#
>?

learning_rate_1먷6�tI       6%�	�B�a���A�)*;


total_loss�"{@

error_R��@?

learning_rate_1먷6��TI       6%�	��a���A�)*;


total_loss��@

error_R��a?

learning_rate_1먷6�N��I       6%�	8��a���A�)*;


total_loss��@

error_R�B?

learning_rate_1먷6/t]�I       6%�	Q	�a���A�)*;


total_loss���@

error_R�H?

learning_rate_1먷6f�2�I       6%�	�L�a���A�)*;


total_loss-��@

error_R��U?

learning_rate_1먷6J�O�I       6%�	��a���A�)*;


total_loss��q@

error_R
�\?

learning_rate_1먷6��5�I       6%�	���a���A�)*;


total_loss�p�@

error_RD�R?

learning_rate_1먷6����I       6%�	��a���A�)*;


total_loss�L�@

error_R��F?

learning_rate_1먷6���I       6%�	O�a���A�)*;


total_loss_�@

error_R��L?

learning_rate_1먷6�E�I       6%�	��a���A�)*;


total_loss�A

error_R|\?

learning_rate_1먷6����I       6%�	$��a���A�)*;


total_loss�=�@

error_R��[?

learning_rate_1먷6�W_!I       6%�	z�a���A�)*;


total_loss���@

error_R�OP?

learning_rate_1먷60��I       6%�	�X�a���A�)*;


total_loss��@

error_R��I?

learning_rate_1먷6�g^�I       6%�	���a���A�)*;


total_loss�@

error_R�L?

learning_rate_1먷6�h�I       6%�	H��a���A�)*;


total_loss)1�@

error_R�J?

learning_rate_1먷6�B�I       6%�	y�a���A�)*;


total_loss���@

error_R?\I?

learning_rate_1먷6F� �I       6%�	z[�a���A�)*;


total_loss{��@

error_RI?

learning_rate_1먷6yhg I       6%�	��a���A�)*;


total_lossc��@

error_RJ�S?

learning_rate_1먷6��WI       6%�	b��a���A�)*;


total_loss��@

error_R�f?

learning_rate_1먷6�9��I       6%�	��a���A�)*;


total_loss$݆@

error_R�M?

learning_rate_1먷6�2r�I       6%�	e�a���A�)*;


total_loss1W�@

error_RT�F?

learning_rate_1먷6A�jI       6%�	h��a���A�)*;


total_loss��@

error_RTS?

learning_rate_1먷6��BI       6%�	"��a���A�)*;


total_lossf1�@

error_R��H?

learning_rate_1먷6���I       6%�	�4�a���A�)*;


total_loss�;�@

error_Rm�O?

learning_rate_1먷6;���I       6%�	?w�a���A�)*;


total_loss�O�@

error_R��@?

learning_rate_1먷6B��I       6%�	k��a���A�)*;


total_loss�{�@

error_R8�;?

learning_rate_1먷6��I       6%�	��a���A�)*;


total_loss���@

error_RߧU?

learning_rate_1먷6:;mI       6%�	�]�a���A�)*;


total_loss3a�@

error_R�/>?

learning_rate_1먷6����I       6%�	u��a���A�)*;


total_lossi�@

error_R�N?

learning_rate_1먷6u�[�I       6%�	���a���A�)*;


total_loss�5A

error_Rf�G?

learning_rate_1먷6� �I       6%�	�!�a���A�)*;


total_loss�Q�@

error_R,�`?

learning_rate_1먷6p�I       6%�	�g�a���A�)*;


total_loss$�@

error_R2�G?

learning_rate_1먷6TK�I       6%�	���a���A�)*;


total_loss�\�@

error_R��H?

learning_rate_1먷6�Y+7I       6%�	���a���A�)*;


total_loss
�@

error_Rq�[?

learning_rate_1먷6�w�I       6%�	�4�a���A�)*;


total_loss���@

error_R�!K?

learning_rate_1먷61���I       6%�	�x�a���A�)*;


total_loss�ػ@

error_R�,Q?

learning_rate_1먷6~���I       6%�	��a���A�)*;


total_loss�I�@

error_R�lV?

learning_rate_1먷6x�I       6%�	H��a���A�)*;


total_loss?V�@

error_R�I?

learning_rate_1먷6�SX�I       6%�	AF b���A�)*;


total_loss��@

error_R�%??

learning_rate_1먷6���I       6%�	�� b���A�)*;


total_loss�A

error_RJ�F?

learning_rate_1먷6���hI       6%�	/� b���A�)*;


total_lossI�]@

error_R��;?

learning_rate_1먷6��I       6%�	�'b���A�)*;


total_loss���@

error_RR
J?

learning_rate_1먷6�i��I       6%�	�nb���A�)*;


total_loss8��@

error_R�uT?

learning_rate_1먷6$8I       6%�	��b���A�)*;


total_lossI1�@

error_R3�??

learning_rate_1먷6�܆�I       6%�	e�b���A�)*;


total_loss� A

error_R�]?

learning_rate_1먷6��"I       6%�	�8b���A�)*;


total_lossL��@

error_R�E?

learning_rate_1먷6�7)�I       6%�	�|b���A�)*;


total_lossł�@

error_R)�c?

learning_rate_1먷6lRp�I       6%�	��b���A�)*;


total_lossXu�@

error_R<F?

learning_rate_1먷6I��	I       6%�	b���A�)*;


total_loss{n�@

error_R;R?

learning_rate_1먷6���4I       6%�	`Ib���A�)*;


total_loss('�@

error_R�v<?

learning_rate_1먷6ے�KI       6%�	��b���A�)*;


total_lossI�@

error_R��L?

learning_rate_1먷6��,I       6%�	��b���A�)*;


total_lossM-'A

error_R��Z?

learning_rate_1먷6�݉�I       6%�	7b���A�)*;


total_loss��@

error_Rjl6?

learning_rate_1먷69b�I       6%�	LWb���A�)*;


total_loss��@

error_RDCK?

learning_rate_1먷66�-I       6%�	��b���A�)*;


total_loss�i�@

error_R��C?

learning_rate_1먷6H��@I       6%�	#�b���A�)*;


total_loss�g�@

error_R��F?

learning_rate_1먷6A�4�I       6%�	�b���A�)*;


total_loss�	�@

error_R�/D?

learning_rate_1먷6.N�EI       6%�	Vb���A�)*;


total_loss��@

error_R�_Q?

learning_rate_1먷6//sI       6%�	��b���A�)*;


total_loss�P�@

error_R_,L?

learning_rate_1먷6\��7I       6%�	 �b���A�)*;


total_loss%r�@

error_R�Y?

learning_rate_1먷6���I       6%�	�b���A�)*;


total_loss@F�@

error_R�P?

learning_rate_1먷6w�SLI       6%�	�Xb���A�)*;


total_loss���@

error_R�GM?

learning_rate_1먷6��U�I       6%�	Śb���A�)*;


total_lossJ�@

error_RA�E?

learning_rate_1먷6��]I       6%�	b�b���A�)*;


total_loss���@

error_R��V?

learning_rate_1먷6���I       6%�	b���A�)*;


total_loss�<�@

error_R��;?

learning_rate_1먷6C_�FI       6%�	�Yb���A�)*;


total_loss︶@

error_Rv�X?

learning_rate_1먷6�
�I       6%�	;�b���A�)*;


total_loss�lA

error_R��T?

learning_rate_1먷6�	KVI       6%�	��b���A�)*;


total_losszTw@

error_R�V?

learning_rate_1먷6Ҁ�NI       6%�	�b���A�)*;


total_loss���@

error_R<\?

learning_rate_1먷6��6�I       6%�	�]b���A�)*;


total_loss�q�@

error_R�vK?

learning_rate_1먷6�54�I       6%�	K�b���A�)*;


total_loss�XA

error_R��O?

learning_rate_1먷6c��I       6%�	��b���A�)*;


total_lossn¼@

error_R3�R?

learning_rate_1먷6��I       6%�	 	b���A�)*;


total_lossT��@

error_RqTa?

learning_rate_1먷6���-I       6%�	�f	b���A�)*;


total_loss���@

error_R6E?

learning_rate_1먷6�xI       6%�	��	b���A�)*;


total_loss�A

error_RΗE?

learning_rate_1먷6y��II       6%�	��	b���A�)*;


total_loss��@

error_R��A?

learning_rate_1먷6�5;I       6%�	o'
b���A�)*;


total_loss�*�@

error_R$�=?

learning_rate_1먷6Z�I       6%�	�f
b���A�)*;


total_loss�|�@

error_R=�I?

learning_rate_1먷6�9��I       6%�	��
b���A�)*;


total_lossA+�@

error_R��B?

learning_rate_1먷6_!NmI       6%�	��
b���A�)*;


total_loss���@

error_R�aH?

learning_rate_1먷6[�7>I       6%�	�2b���A�)*;


total_loss��@

error_R�H\?

learning_rate_1먷6���@I       6%�	vb���A�)*;


total_loss��@

error_RiLI?

learning_rate_1먷6�~P�I       6%�	�b���A�)*;


total_loss��@

error_Rj>Z?

learning_rate_1먷6ehI       6%�	b b���A�)*;


total_lossx��@

error_R��>?

learning_rate_1먷6�r9I       6%�	�@b���A�)*;


total_lossF
�@

error_RN&O?

learning_rate_1먷6�� �I       6%�	w�b���A�)*;


total_loss��@

error_R20>?

learning_rate_1먷6A�ߧI       6%�	��b���A�)*;


total_loss�R�@

error_R�M?

learning_rate_1먷6p��I       6%�	�%b���A�)*;


total_loss�L�@

error_R��H?

learning_rate_1먷6�n]I       6%�	�nb���A�)*;


total_lossh0�@

error_R��G?

learning_rate_1먷6�]hCI       6%�	��b���A�)*;


total_lossS��@

error_R��N?

learning_rate_1먷6}���I       6%�	��b���A�)*;


total_lossR�2A

error_R��K?

learning_rate_1먷6�6I       6%�	~2b���A�)*;


total_loss:��@

error_Rc�E?

learning_rate_1먷6漻�I       6%�	�sb���A�)*;


total_loss�u�@

error_RCdH?

learning_rate_1먷68���I       6%�	!�b���A�)*;


total_loss���@

error_R!m@?

learning_rate_1먷6��?�I       6%�	��b���A�)*;


total_loss�|@

error_RR�R?

learning_rate_1먷6��I       6%�	'0b���A�)*;


total_lossj5�@

error_R6�T?

learning_rate_1먷6c���I       6%�	�pb���A�)*;


total_loss��@

error_RZ�K?

learning_rate_1먷6��II       6%�	ʹb���A�)*;


total_loss�i�@

error_R,;G?

learning_rate_1먷6�I       6%�	��b���A�)*;


total_loss���@

error_R�>?

learning_rate_1먷60K��I       6%�	�7b���A�)*;


total_loss���@

error_R��I?

learning_rate_1먷6{�LfI       6%�	gxb���A�)*;


total_loss�q�@

error_RWP?

learning_rate_1먷6��F�I       6%�	�b���A�)*;


total_loss��@

error_R�JK?

learning_rate_1먷6g%W�I       6%�	�b���A�)*;


total_loss��A

error_R�"Y?

learning_rate_1먷6��DI       6%�	�<b���A�)*;


total_loss�U�@

error_RvP?

learning_rate_1먷6b�>�I       6%�	l{b���A�)*;


total_loss6ܓ@

error_R�P?

learning_rate_1먷6�y�bI       6%�	�b���A�)*;


total_loss[��@

error_R6�<?

learning_rate_1먷6��I       6%�	n�b���A�)*;


total_lossM��@

error_R�M[?

learning_rate_1먷6gݒ�I       6%�	�?b���A�)*;


total_loss	��@

error_R�L?

learning_rate_1먷6��CI       6%�	Y�b���A�)*;


total_lossW�@

error_R��e?

learning_rate_1먷6�aII       6%�	c�b���A�)*;


total_loss�}�@

error_R�YV?

learning_rate_1먷6 F�*I       6%�	rb���A�)*;


total_loss���@

error_R�G?

learning_rate_1먷6�I�I       6%�	�Rb���A�)*;


total_loss�ж@

error_R��X?

learning_rate_1먷6�E�I       6%�		�b���A�)*;


total_lossl��@

error_R��E?

learning_rate_1먷6x��I       6%�	��b���A�)*;


total_loss��@

error_R�mP?

learning_rate_1먷6�s$I       6%�	0b���A�)*;


total_loss��@

error_R)�Q?

learning_rate_1먷6���NI       6%�	�Vb���A�)*;


total_loss�x�@

error_RW�_?

learning_rate_1먷6tzII       6%�	�b���A�)*;


total_loss�o@

error_R��O?

learning_rate_1먷6�{�^I       6%�	�b���A�)*;


total_loss�n@

error_R��>?

learning_rate_1먷6��VI       6%�	"6b���A�)*;


total_loss���@

error_RzPC?

learning_rate_1먷6�`�I       6%�	�wb���A�)*;


total_loss}��@

error_R�:I?

learning_rate_1먷6���I       6%�	)�b���A�)*;


total_loss���@

error_R��V?

learning_rate_1먷6<�;HI       6%�	r�b���A�)*;


total_loss#��@

error_R�x]?

learning_rate_1먷6zިHI       6%�	�?b���A�**;


total_loss.^�@

error_R��L?

learning_rate_1먷6����I       6%�	ցb���A�**;


total_lossm��@

error_R�qS?

learning_rate_1먷6���I       6%�	��b���A�**;


total_loss.��@

error_RۄC?

learning_rate_1먷6�P�I       6%�	�
b���A�**;


total_loss](�@

error_RM�V?

learning_rate_1먷6S��I       6%�	Ob���A�**;


total_loss=�@

error_R�SE?

learning_rate_1먷6��YI       6%�	�b���A�**;


total_loss��@

error_Rm�=?

learning_rate_1먷6E�A�I       6%�	+�b���A�**;


total_lossۚ�@

error_R3�Z?

learning_rate_1먷6��FHI       6%�	�b���A�**;


total_loss��@

error_R,�O?

learning_rate_1먷6���I       6%�	�Pb���A�**;


total_loss#��@

error_RfYB?

learning_rate_1먷6��[�I       6%�	�b���A�**;


total_loss���@

error_R�8U?

learning_rate_1먷6=�DI       6%�	z�b���A�**;


total_loss��@

error_R� T?

learning_rate_1먷6�%P#I       6%�	tb���A�**;


total_loss{č@

error_R��T?

learning_rate_1먷6�R��I       6%�	.bb���A�**;


total_loss��@

error_REN?

learning_rate_1먷6�=�0I       6%�	ԣb���A�**;


total_loss��@

error_R7�S?

learning_rate_1먷6e[�I       6%�	W�b���A�**;


total_lossN�A

error_RC�F?

learning_rate_1먷6����I       6%�	&b���A�**;


total_loss�@

error_R��S?

learning_rate_1먷6:��OI       6%�	�eb���A�**;


total_loss�7�@

error_R�>Z?

learning_rate_1먷6>{9DI       6%�	X�b���A�**;


total_loss
��@

error_R�@?

learning_rate_1먷6�X�I       6%�	��b���A�**;


total_loss���@

error_R��M?

learning_rate_1먷6v'��I       6%�	�&b���A�**;


total_lossL�}@

error_Rx P?

learning_rate_1먷6hv��I       6%�	gb���A�**;


total_loss���@

error_R�:?

learning_rate_1먷6�CSI       6%�	��b���A�**;


total_loss���@

error_RTvI?

learning_rate_1먷6�IY�I       6%�	
�b���A�**;


total_loss��@

error_R(�Y?

learning_rate_1먷6��PI       6%�	`(b���A�**;


total_loss�6�@

error_R_rL?

learning_rate_1먷6ʚ�|I       6%�	�gb���A�**;


total_lossA�@

error_R��h?

learning_rate_1먷6�=5I       6%�	1�b���A�**;


total_loss}>�@

error_R��E?

learning_rate_1먷6��I       6%�	)�b���A�**;


total_loss�T�@

error_R�2@?

learning_rate_1먷6yUDI       6%�	Jb���A�**;


total_loss��@

error_R��A?

learning_rate_1먷6��I       6%�	+�b���A�**;


total_loss���@

error_R�}B?

learning_rate_1먷6��5I       6%�	��b���A�**;


total_loss��@

error_Ri�W?

learning_rate_1먷6��o�I       6%�	�b���A�**;


total_loss�ܑ@

error_R�>Q?

learning_rate_1먷6m��~I       6%�	�ab���A�**;


total_loss{l�@

error_R�U?

learning_rate_1먷6q���I       6%�	��b���A�**;


total_loss[��@

error_R�=?

learning_rate_1먷61�@�I       6%�	��b���A�**;


total_loss��@

error_R�4G?

learning_rate_1먷6�a�SI       6%�	�/b���A�**;


total_lossT=�@

error_R��I?

learning_rate_1먷6QK8zI       6%�	rqb���A�**;


total_loss��@

error_R� J?

learning_rate_1먷6���I       6%�	��b���A�**;


total_loss��@

error_R��B?

learning_rate_1먷6(҂I       6%�	Q�b���A�**;


total_loss���@

error_R@�H?

learning_rate_1먷6v��yI       6%�	4 b���A�**;


total_loss��@

error_R�b?

learning_rate_1먷6>�6+I       6%�	�y b���A�**;


total_loss�w�@

error_R;�G?

learning_rate_1먷6 ^ZI       6%�	J� b���A�**;


total_loss:�@

error_R�CM?

learning_rate_1먷6����I       6%�	�!b���A�**;


total_loss�b�@

error_R�FR?

learning_rate_1먷6"�I       6%�	NC!b���A�**;


total_loss ��@

error_R�@X?

learning_rate_1먷6U�I       6%�	~�!b���A�**;


total_loss��@

error_R��[?

learning_rate_1먷6/Č�I       6%�	�!b���A�**;


total_loss���@

error_R�H?

learning_rate_1먷6�Ն�I       6%�	�"b���A�**;


total_loss,�	A

error_R��E?

learning_rate_1먷6�#I       6%�	�J"b���A�**;


total_lossڜ�@

error_Rq�E?

learning_rate_1먷6&v�I       6%�	<�"b���A�**;


total_loss�e�@

error_RȎW?

learning_rate_1먷6�L��I       6%�	��"b���A�**;


total_loss%�@

error_Ra�<?

learning_rate_1먷6�=
I       6%�	�#b���A�**;


total_lossZ�@

error_R�#M?

learning_rate_1먷6�+65I       6%�	K#b���A�**;


total_loss�W�@

error_R��G?

learning_rate_1먷6Td�I       6%�	��#b���A�**;


total_loss�ح@

error_R�C?

learning_rate_1먷6<|x�I       6%�	��#b���A�**;


total_loss$Q�@

error_R��P?

learning_rate_1먷6v��AI       6%�	�
$b���A�**;


total_loss{��@

error_R�D?

learning_rate_1먷6�އI       6%�	�M$b���A�**;


total_loss���@

error_RO�X?

learning_rate_1먷6%K�;I       6%�	��$b���A�**;


total_loss��m@

error_R�N?

learning_rate_1먷6��	?I       6%�	��$b���A�**;


total_lossv��@

error_RZ�O?

learning_rate_1먷6���0I       6%�	�%b���A�**;


total_lossL�@

error_R�I?

learning_rate_1먷6��I       6%�	6X%b���A�**;


total_lossO��@

error_R\}??

learning_rate_1먷6���HI       6%�	��%b���A�**;


total_loss ��@

error_R.�[?

learning_rate_1먷6�Go�I       6%�	��%b���A�**;


total_loss���@

error_Rx&L?

learning_rate_1먷6�xyI       6%�	#&b���A�**;


total_lossQ|�@

error_R�G?

learning_rate_1먷6���<I       6%�	�Z&b���A�**;


total_lossT\�@

error_R�W?

learning_rate_1먷6
&�I       6%�	_�&b���A�**;


total_lossM��@

error_R�0[?

learning_rate_1먷6�-�I       6%�	X�&b���A�**;


total_loss�|�@

error_R#�Q?

learning_rate_1먷6��U�I       6%�	@'b���A�**;


total_loss�C�@

error_RZ�P?

learning_rate_1먷6��8jI       6%�	r]'b���A�**;


total_loss��A

error_R�
P?

learning_rate_1먷6�w�tI       6%�	��'b���A�**;


total_loss$�x@

error_R�P?

learning_rate_1먷6J��I       6%�	F�'b���A�**;


total_loss��@

error_R��V?

learning_rate_1먷6�,bI       6%�	�"(b���A�**;


total_lossT��@

error_RX�H?

learning_rate_1먷6���I       6%�	e(b���A�**;


total_loss�~�@

error_R��Q?

learning_rate_1먷6�J�;I       6%�	�(b���A�**;


total_loss�x�@

error_Ra�C?

learning_rate_1먷6+�Q^I       6%�	{�(b���A�**;


total_loss�g�@

error_R/@?

learning_rate_1먷6T��I       6%�	[K)b���A�**;


total_lossd��@

error_R�8F?

learning_rate_1먷6�I       6%�	�)b���A�**;


total_loss���@

error_R�1B?

learning_rate_1먷6��aI       6%�	$�)b���A�**;


total_loss]�@

error_R�I?

learning_rate_1먷6����I       6%�	�-*b���A�**;


total_lossH�@

error_RC�I?

learning_rate_1먷6_@�mI       6%�	_p*b���A�**;


total_loss���@

error_RNE?

learning_rate_1먷6K�f�I       6%�	T�*b���A�**;


total_loss��@

error_R��E?

learning_rate_1먷6�*&I       6%�	��*b���A�**;


total_loss���@

error_R��7?

learning_rate_1먷6�*�eI       6%�	/>+b���A�**;


total_lossl+�@

error_R
hG?

learning_rate_1먷6<�I       6%�	ڃ+b���A�**;


total_loss{��@

error_R8=?

learning_rate_1먷6cu�I       6%�		�+b���A�**;


total_loss�F�@

error_RnBP?

learning_rate_1먷6&���I       6%�	R,b���A�**;


total_loss��@

error_R\?

learning_rate_1먷6|@�I       6%�	=^,b���A�**;


total_lossHܽ@

error_RLK?

learning_rate_1먷6oJI       6%�	A�,b���A�**;


total_loss�T�@

error_R��U?

learning_rate_1먷6W��CI       6%�	��,b���A�**;


total_loss���@

error_RIQ?

learning_rate_1먷6^���I       6%�	�M-b���A�**;


total_loss�b�@

error_R��T?

learning_rate_1먷6QM�I       6%�	��-b���A�**;


total_loss���@

error_R�RE?

learning_rate_1먷6g^�UI       6%�	��-b���A�**;


total_loss��A

error_R��l?

learning_rate_1먷6_�"6I       6%�	�.b���A�**;


total_loss��@

error_R��V?

learning_rate_1먷6�nI       6%�	Y.b���A�**;


total_lossls�@

error_R�VJ?

learning_rate_1먷6��]CI       6%�	C�.b���A�**;


total_loss@��@

error_R_?

learning_rate_1먷6��ԑI       6%�	��.b���A�**;


total_loss�ϻ@

error_R8Y?

learning_rate_1먷6���I       6%�	�/b���A�**;


total_lossi�@

error_R\!I?

learning_rate_1먷6��yI       6%�	^/b���A�**;


total_loss1��@

error_R�Mi?

learning_rate_1먷6�!��I       6%�	��/b���A�**;


total_loss�@

error_RZ�O?

learning_rate_1먷6��I       6%�	��/b���A�**;


total_loss�	�@

error_R_�D?

learning_rate_1먷6���I       6%�	3#0b���A�**;


total_loss�6�@

error_R�-I?

learning_rate_1먷6���I       6%�	΂0b���A�**;


total_loss���@

error_R�P?

learning_rate_1먷6�ҠI       6%�	��0b���A�**;


total_loss�H�@

error_R�2Q?

learning_rate_1먷6�-�%I       6%�	k1b���A�**;


total_loss�]�@

error_R2�\?

learning_rate_1먷6Z��lI       6%�	-S1b���A�**;


total_loss׬�@

error_R�<E?

learning_rate_1먷6���I       6%�	Г1b���A�**;


total_lossd�@

error_R_K?

learning_rate_1먷6 �~�I       6%�	!�1b���A�**;


total_loss��@

error_R�oE?

learning_rate_1먷62s��I       6%�	�2b���A�**;


total_loss��@

error_R�yL?

learning_rate_1먷6=.��I       6%�	�]2b���A�**;


total_loss=��@

error_R�sM?

learning_rate_1먷6�wU`I       6%�	r�2b���A�**;


total_lossOw�@

error_RWqE?

learning_rate_1먷6���kI       6%�	b�2b���A�**;


total_loss��@

error_R�(S?

learning_rate_1먷69mv�I       6%�	N"3b���A�**;


total_loss���@

error_RHOS?

learning_rate_1먷6f�I       6%�	�f3b���A�**;


total_lossZ��@

error_Rx�N?

learning_rate_1먷6���I       6%�	��3b���A�**;


total_loss��@

error_R�uM?

learning_rate_1먷6�s>I       6%�	��3b���A�**;


total_lossh �@

error_R�k_?

learning_rate_1먷6�E�I       6%�	e&4b���A�**;


total_losss�@

error_R�3S?

learning_rate_1먷6e'm�I       6%�	�f4b���A�**;


total_loss#e~@

error_RÎg?

learning_rate_1먷6�$M�I       6%�	b�4b���A�**;


total_loss���@

error_R�)O?

learning_rate_1먷6t�I       6%�	��4b���A�**;


total_loss!��@

error_R��O?

learning_rate_1먷6�L�I       6%�	%5b���A�**;


total_loss]'�@

error_R�U?

learning_rate_1먷6����I       6%�	:e5b���A�**;


total_loss20�@

error_R�eS?

learning_rate_1먷66:% I       6%�	̥5b���A�**;


total_loss��@

error_RT9[?

learning_rate_1먷6�p|I       6%�	J�5b���A�**;


total_loss��@

error_R/]I?

learning_rate_1먷6�^&�I       6%�	Q(6b���A�**;


total_loss �@

error_R��H?

learning_rate_1먷6��I       6%�	�z6b���A�**;


total_loss&�A

error_R�G?

learning_rate_1먷6�1��I       6%�	��6b���A�**;


total_loss� A

error_R��J?

learning_rate_1먷6���I       6%�	�47b���A�**;


total_loss�@�@

error_R��F?

learning_rate_1먷6��QI       6%�	��7b���A�**;


total_loss�S�@

error_R�jD?

learning_rate_1먷6�U?�I       6%�	��7b���A�**;


total_lossJy�@

error_RVWC?

learning_rate_1먷6`�"�I       6%�	�&8b���A�**;


total_loss/�|@

error_RB?

learning_rate_1먷6/��I       6%�	hl8b���A�+*;


total_lossqc@

error_R��L?

learning_rate_1먷6��S�I       6%�	+�8b���A�+*;


total_lossMˬ@

error_R�0]?

learning_rate_1먷6�5РI       6%�	��8b���A�+*;


total_loss!�@

error_Rq=L?

learning_rate_1먷6i)��I       6%�	�59b���A�+*;


total_loss	8�@

error_R��J?

learning_rate_1먷6�=��I       6%�	,�9b���A�+*;


total_lossg 
A

error_R��O?

learning_rate_1먷6�)�I       6%�	�:b���A�+*;


total_lossS��@

error_R�@?

learning_rate_1먷6�BlDI       6%�	O_:b���A�+*;


total_lossQ!A

error_R6�N?

learning_rate_1먷6��f�I       6%�	W�:b���A�+*;


total_loss��@

error_R`aO?

learning_rate_1먷6X��I       6%�	��:b���A�+*;


total_loss�9�@

error_R�|a?

learning_rate_1먷6I;�WI       6%�	9+;b���A�+*;


total_loss_¦@

error_R%V?

learning_rate_1먷6@�`NI       6%�	bn;b���A�+*;


total_loss��@

error_R�jS?

learning_rate_1먷6!S�I       6%�	8�;b���A�+*;


total_loss���@

error_Rn�A?

learning_rate_1먷6ﺿ�I       6%�	��;b���A�+*;


total_loss���@

error_R:[N?

learning_rate_1먷6��[xI       6%�	o:<b���A�+*;


total_loss��@

error_R�3O?

learning_rate_1먷6O9qdI       6%�	�<b���A�+*;


total_loss��@

error_Rs�;?

learning_rate_1먷6�ADI       6%�	�<b���A�+*;


total_lossϧ@

error_R*�>?

learning_rate_1먷6yjs�I       6%�	Ǟ=b���A�+*;


total_loss y�@

error_R��D?

learning_rate_1먷6�ϹmI       6%�	<�=b���A�+*;


total_loss�Ƙ@

error_R pJ?

learning_rate_1먷6d�,cI       6%�	
_>b���A�+*;


total_loss�͵@

error_R��G?

learning_rate_1먷6�E%>I       6%�	��>b���A�+*;


total_lossu�A

error_R��F?

learning_rate_1먷6�L��I       6%�	�>b���A�+*;


total_loss��@

error_R��A?

learning_rate_1먷6�K�I       6%�	D?b���A�+*;


total_lossQ��@

error_R�H?

learning_rate_1먷6dI       6%�	�?b���A�+*;


total_loss�P�@

error_R�T?

learning_rate_1먷6@�_I       6%�	��?b���A�+*;


total_loss���@

error_R;�F?

learning_rate_1먷6�b,�I       6%�	�(@b���A�+*;


total_loss��@

error_R�R?

learning_rate_1먷6�'I       6%�	�w@b���A�+*;


total_loss`��@

error_R�=?

learning_rate_1먷6�&��I       6%�	>�@b���A�+*;


total_lossj��@

error_R��E?

learning_rate_1먷6�l�I       6%�	CyAb���A�+*;


total_loss��V@

error_R�O?

learning_rate_1먷6Z0�I       6%�	5�Ab���A�+*;


total_loss/F�@

error_R#V?

learning_rate_1먷6`��XI       6%�	+OBb���A�+*;


total_loss�A

error_R(�_?

learning_rate_1먷6�.qI       6%�	ƚBb���A�+*;


total_lossc��@

error_R�n??

learning_rate_1먷6)��I       6%�	�Bb���A�+*;


total_lossc�A

error_R`M?

learning_rate_1먷6�e��I       6%�	[KCb���A�+*;


total_lossN�@

error_RQQ?

learning_rate_1먷6�x�I       6%�	��Cb���A�+*;


total_loss��@

error_R�E?

learning_rate_1먷6��TI       6%�	m�Cb���A�+*;


total_loss�t�@

error_R*�O?

learning_rate_1먷6݂�:I       6%�	�>Db���A�+*;


total_loss���@

error_RԝM?

learning_rate_1먷6�r I       6%�	h�Db���A�+*;


total_loss#p�@

error_RO?

learning_rate_1먷6��+I       6%�	��Db���A�+*;


total_loss}?�@

error_ROE?

learning_rate_1먷6̒�dI       6%�	�!Eb���A�+*;


total_loss
�@

error_R�K?

learning_rate_1먷6� �I       6%�	oEb���A�+*;


total_loss���@

error_R��I?

learning_rate_1먷6�ћI       6%�	кEb���A�+*;


total_loss���@

error_RJ?Z?

learning_rate_1먷65�H2I       6%�	�/Fb���A�+*;


total_loss���@

error_R#Ma?

learning_rate_1먷6���I       6%�	7�Fb���A�+*;


total_loss���@

error_Rf2R?

learning_rate_1먷6
@��I       6%�	�Fb���A�+*;


total_loss�"�@

error_R�kL?

learning_rate_1먷6�h�I       6%�	#Gb���A�+*;


total_loss�T�@

error_R@�D?

learning_rate_1먷64Y�I       6%�	�hGb���A�+*;


total_loss���@

error_Re�=?

learning_rate_1먷6���II       6%�	��Gb���A�+*;


total_loss�F�@

error_RnZY?

learning_rate_1먷6��$�I       6%�	"�Gb���A�+*;


total_loss�?�@

error_R��B?

learning_rate_1먷6 ;I       6%�	33Hb���A�+*;


total_loss2�@

error_R#B?

learning_rate_1먷6n�#;I       6%�	�sHb���A�+*;


total_lossa��@

error_RN?

learning_rate_1먷6A�J�I       6%�	ԶHb���A�+*;


total_loss�F�@

error_Rw�C?

learning_rate_1먷6�aJ�I       6%�	��Hb���A�+*;


total_lossӶ�@

error_R|^A?

learning_rate_1먷6ĄX�I       6%�	U;Ib���A�+*;


total_loss-]�@

error_R&�O?

learning_rate_1먷6�pI       6%�	.�Ib���A�+*;


total_loss���@

error_R�Q?

learning_rate_1먷6��-I       6%�	@�Ib���A�+*;


total_loss蓔@

error_Rs�O?

learning_rate_1먷6��6'I       6%�	�'Jb���A�+*;


total_loss�/�@

error_R�/T?

learning_rate_1먷6F0��I       6%�	�kJb���A�+*;


total_loss�c�@

error_R�~Q?

learning_rate_1먷6�`�fI       6%�	M�Jb���A�+*;


total_loss�ݖ@

error_RZ�T?

learning_rate_1먷6ϙ�RI       6%�	��Jb���A�+*;


total_loss���@

error_RH�F?

learning_rate_1먷64q��I       6%�	�5Kb���A�+*;


total_lossH��@

error_R�1H?

learning_rate_1먷6*�G�I       6%�	yKb���A�+*;


total_loss%�@

error_R.M?

learning_rate_1먷6("�I       6%�	��Kb���A�+*;


total_loss�@

error_R��B?

learning_rate_1먷6�+�I       6%�	�Lb���A�+*;


total_loss$N�@

error_R�Q?

learning_rate_1먷6����I       6%�	�GLb���A�+*;


total_loss)��@

error_R�B?

learning_rate_1먷68��I       6%�	��Lb���A�+*;


total_loss���@

error_RN�3?

learning_rate_1먷6�Al|I       6%�	��Lb���A�+*;


total_loss�Z�@

error_R��G?

learning_rate_1먷6҄��I       6%�	F/Mb���A�+*;


total_loss���@

error_R]a?

learning_rate_1먷6E�3RI       6%�	v�Mb���A�+*;


total_loss�"�@

error_Rc�[?

learning_rate_1먷6娰hI       6%�	�Mb���A�+*;


total_loss��@

error_R�`N?

learning_rate_1먷6�{��I       6%�	�Nb���A�+*;


total_loss���@

error_R{]?

learning_rate_1먷6�Zd�I       6%�	�LNb���A�+*;


total_lossA��@

error_R�RM?

learning_rate_1먷6̳Y5I       6%�	t�Nb���A�+*;


total_loss�)�@

error_Rn�L?

learning_rate_1먷6�'NI       6%�	E�Nb���A�+*;


total_loss��@

error_R�kW?

learning_rate_1먷6���I       6%�	(Ob���A�+*;


total_lossh?m@

error_R�K?

learning_rate_1먷6�էLI       6%�	�[Ob���A�+*;


total_loss��@

error_R.Q?

learning_rate_1먷6|CNI       6%�	��Ob���A�+*;


total_loss1�v@

error_R�2K?

learning_rate_1먷6�$�I       6%�	��Ob���A�+*;


total_lossֲ�@

error_R�K?

learning_rate_1먷6�:�5I       6%�	�#Pb���A�+*;


total_loss\4�@

error_R�vZ?

learning_rate_1먷6�HmI       6%�	JiPb���A�+*;


total_loss���@

error_R|hS?

learning_rate_1먷6����I       6%�	V�Pb���A�+*;


total_loss軲@

error_R��;?

learning_rate_1먷6݂O�I       6%�	�Pb���A�+*;


total_loss�é@

error_R%!V?

learning_rate_1먷6��-I       6%�	�<Qb���A�+*;


total_lossя�@

error_R�CL?

learning_rate_1먷6$��I       6%�	:�Qb���A�+*;


total_loss�*�@

error_RV?

learning_rate_1먷6s��I       6%�	f�Qb���A�+*;


total_loss��@

error_RW�N?

learning_rate_1먷6 �U�I       6%�	�Rb���A�+*;


total_loss���@

error_R)Z_?

learning_rate_1먷6Lhm�I       6%�	�_Rb���A�+*;


total_loss�9�@

error_RmyL?

learning_rate_1먷6]�fI       6%�	��Rb���A�+*;


total_loss���@

error_R��K?

learning_rate_1먷6?ίI       6%�	��Rb���A�+*;


total_loss�_�@

error_Ri�M?

learning_rate_1먷6��-�I       6%�	�5Sb���A�+*;


total_lossR��@

error_R�B?

learning_rate_1먷6��.I       6%�	�|Sb���A�+*;


total_loss6�@

error_R4e9?

learning_rate_1먷6�َ�I       6%�	i�Sb���A�+*;


total_loss���@

error_Rb?

learning_rate_1먷6�?2�I       6%�	kTb���A�+*;


total_lossغ�@

error_RR?

learning_rate_1먷6Um��I       6%�	�YTb���A�+*;


total_lossm"�@

error_R�XF?

learning_rate_1먷6��TI       6%�	�Tb���A�+*;


total_lossM�@

error_R�YE?

learning_rate_1먷6�'I       6%�	�Tb���A�+*;


total_loss&r�@

error_R��Z?

learning_rate_1먷6���I       6%�	e+Ub���A�+*;


total_loss�P�@

error_RR?

learning_rate_1먷6C���I       6%�	<zUb���A�+*;


total_lossw��@

error_R		J?

learning_rate_1먷6���I       6%�	B�Ub���A�+*;


total_lossϽA

error_R��S?

learning_rate_1먷6�L6BI       6%�	G'Vb���A�+*;


total_loss���@

error_RsID?

learning_rate_1먷6��I       6%�	AiVb���A�+*;


total_lossÖ�@

error_R��U?

learning_rate_1먷6ٯL�I       6%�	f�Vb���A�+*;


total_loss��@

error_R��e?

learning_rate_1먷6�S\I       6%�	k�Vb���A�+*;


total_loss���@

error_RW\S?

learning_rate_1먷6��@KI       6%�	�/Wb���A�+*;


total_loss�̢@

error_R��O?

learning_rate_1먷65+ �I       6%�	rWb���A�+*;


total_loss�#�@

error_R	oN?

learning_rate_1먷6�P6)I       6%�	�Wb���A�+*;


total_loss��@

error_R�#D?

learning_rate_1먷6��&I       6%�	9�Wb���A�+*;


total_loss�߂@

error_R��F?

learning_rate_1먷6��4�I       6%�	,3Xb���A�+*;


total_loss��@

error_R��F?

learning_rate_1먷6���xI       6%�	AtXb���A�+*;


total_loss׽�@

error_RH�B?

learning_rate_1먷6���I       6%�	ѴXb���A�+*;


total_loss��@

error_R��c?

learning_rate_1먷6KW�#I       6%�	��Xb���A�+*;


total_loss�*A

error_R��O?

learning_rate_1먷6I       6%�	�7Yb���A�+*;


total_lossA

error_R��O?

learning_rate_1먷6e�rI       6%�	�vYb���A�+*;


total_loss�_�@

error_R�tO?

learning_rate_1먷6���8I       6%�	ظYb���A�+*;


total_lossA

error_R��T?

learning_rate_1먷60 ,�I       6%�	��Yb���A�+*;


total_loss�P�@

error_R��C?

learning_rate_1먷6ЭߧI       6%�	z@Zb���A�+*;


total_loss�G�@

error_R9L?

learning_rate_1먷6�[]�I       6%�	�Zb���A�+*;


total_loss$A

error_R�I?

learning_rate_1먷6�h9�I       6%�	��Zb���A�+*;


total_loss���@

error_RfHZ?

learning_rate_1먷6�/��I       6%�	�![b���A�+*;


total_lossֆ�@

error_Rl�Y?

learning_rate_1먷6/��I       6%�	�e[b���A�+*;


total_losso�@

error_R\K?

learning_rate_1먷6%�!I       6%�	��[b���A�+*;


total_loss�6�@

error_RtgJ?

learning_rate_1먷6�!_�I       6%�	��[b���A�+*;


total_lossے@

error_R��B?

learning_rate_1먷6�F&I       6%�	1\b���A�+*;


total_loss�g�@

error_RʑW?

learning_rate_1먷6��bRI       6%�	�w\b���A�+*;


total_loss૨@

error_R&�O?

learning_rate_1먷66Q*�I       6%�	�\b���A�+*;


total_loss(��@

error_RnHR?

learning_rate_1먷6VG��I       6%�	�"]b���A�+*;


total_loss#��@

error_R8�R?

learning_rate_1먷6�~u�I       6%�	�p]b���A�+*;


total_loss��@

error_R��g?

learning_rate_1먷6[';I       6%�	l�]b���A�+*;


total_lossj��@

error_R�F?

learning_rate_1먷6qdPMI       6%�	(�]b���A�+*;


total_loss�̫@

error_RV?

learning_rate_1먷6�:�I       6%�	#>^b���A�,*;


total_loss*j�@

error_R�>?

learning_rate_1먷6�/ҧI       6%�	E�^b���A�,*;


total_loss�i�@

error_R�W?

learning_rate_1먷6��I       6%�	S�^b���A�,*;


total_loss�y�@

error_RdVV?

learning_rate_1먷6k���I       6%�	@_b���A�,*;


total_loss�\�@

error_R�P?

learning_rate_1먷6���I       6%�	�U_b���A�,*;


total_lossq8�@

error_R K=?

learning_rate_1먷6��I       6%�	��_b���A�,*;


total_loss� �@

error_R��@?

learning_rate_1먷6�P� I       6%�	,�_b���A�,*;


total_lossܒ@

error_R��=?

learning_rate_1먷6�2�I       6%�	n!`b���A�,*;


total_lossI.�@

error_RO!U?

learning_rate_1먷6��;I       6%�	�d`b���A�,*;


total_lossoî@

error_RK?

learning_rate_1먷6 ��I       6%�	��`b���A�,*;


total_loss4��@

error_R�~Q?

learning_rate_1먷6�S�~I       6%�	��`b���A�,*;


total_loss�'�@

error_R�DL?

learning_rate_1먷6"O�zI       6%�	0ab���A�,*;


total_loss&/�@

error_RC�X?

learning_rate_1먷6Y^ӲI       6%�	spab���A�,*;


total_loss��@

error_RP?

learning_rate_1먷6�WuI       6%�	�ab���A�,*;


total_lossm��@

error_R,�R?

learning_rate_1먷6����I       6%�	��ab���A�,*;


total_loss�b�@

error_R�P?

learning_rate_1먷6WZ?I       6%�	6bb���A�,*;


total_lossخ@

error_RjPU?

learning_rate_1먷6)ܮI       6%�	�{bb���A�,*;


total_loss3
�@

error_RT�@?

learning_rate_1먷6�{iI       6%�	 �bb���A�,*;


total_loss�h�@

error_R1RU?

learning_rate_1먷6ߑ%FI       6%�	c cb���A�,*;


total_loss��@

error_RiuS?

learning_rate_1먷6T��2I       6%�	�Hcb���A�,*;


total_lossˑ@

error_RT�=?

learning_rate_1먷6���I       6%�	��cb���A�,*;


total_loss���@

error_R��]?

learning_rate_1먷6	��_I       6%�	��cb���A�,*;


total_lossmM�@

error_R\�??

learning_rate_1먷6�^��I       6%�	'2db���A�,*;


total_loss�@

error_R��:?

learning_rate_1먷6Y�(I       6%�	��db���A�,*;


total_loss�ǝ@

error_RV�\?

learning_rate_1먷6��CI       6%�	)�db���A�,*;


total_lossdX�@

error_R��T?

learning_rate_1먷66.�I       6%�	eb���A�,*;


total_losse��@

error_R�V?

learning_rate_1먷6�ۛI       6%�	'Yeb���A�,*;


total_lossv�@

error_R��8?

learning_rate_1먷6̠g9I       6%�	{�eb���A�,*;


total_loss�أ@

error_RJ?

learning_rate_1먷6G���I       6%�	��eb���A�,*;


total_lossj��@

error_R��-?

learning_rate_1먷6����I       6%�	y#fb���A�,*;


total_loss�B�@

error_R�Q?

learning_rate_1먷6<}>I       6%�	yffb���A�,*;


total_losse��@

error_R	[?

learning_rate_1먷6�*�I       6%�	ʩfb���A�,*;


total_lossbT�@

error_Rϡ8?

learning_rate_1먷6���xI       6%�	>�fb���A�,*;


total_loss�ϭ@

error_RF�S?

learning_rate_1먷6�'��I       6%�	2gb���A�,*;


total_lossy��@

error_R��[?

learning_rate_1먷6`��I       6%�	�wgb���A�,*;


total_lossFp�@

error_RAB?

learning_rate_1먷6r�C�I       6%�	��gb���A�,*;


total_loss�x�@

error_RMK?

learning_rate_1먷67�DI       6%�	"hb���A�,*;


total_loss�i�@

error_R��P?

learning_rate_1먷65�DI       6%�	&Ohb���A�,*;


total_loss��y@

error_R�59?

learning_rate_1먷64�I       6%�	��hb���A�,*;


total_loss���@

error_RV1J?

learning_rate_1먷6uMI       6%�	I�hb���A�,*;


total_lossD˸@

error_R !J?

learning_rate_1먷6���I       6%�	�!ib���A�,*;


total_lossD�@

error_R��]?

learning_rate_1먷6���KI       6%�	eib���A�,*;


total_loss�	�@

error_RCY?

learning_rate_1먷6	h�I       6%�	�ib���A�,*;


total_loss�M�@

error_R�TJ?

learning_rate_1먷6�=4I       6%�	��ib���A�,*;


total_loss;��@

error_R��R?

learning_rate_1먷6�ƬZI       6%�	�0jb���A�,*;


total_loss�e�@

error_R��D?

learning_rate_1먷6-n�I       6%�	�sjb���A�,*;


total_loss�ڳ@

error_R��B?

learning_rate_1먷6Y�H@I       6%�	��jb���A�,*;


total_lossC��@

error_R�SY?

learning_rate_1먷6��
I       6%�	 kb���A�,*;


total_loss�h�@

error_R<�Q?

learning_rate_1먷6@R�fI       6%�	�Ekb���A�,*;


total_loss ��@

error_R��L?

learning_rate_1먷6f7P�I       6%�	a�kb���A�,*;


total_loss\ �@

error_R��S?

learning_rate_1먷6����I       6%�	`�kb���A�,*;


total_loss�@

error_R�a?

learning_rate_1먷6��քI       6%�	�lb���A�,*;


total_loss�y�@

error_R�IQ?

learning_rate_1먷6�E��I       6%�	M\lb���A�,*;


total_loss�
�@

error_R��O?

learning_rate_1먷6a��I       6%�	�lb���A�,*;


total_loss|��@

error_R��[?

learning_rate_1먷6�V��I       6%�	e�lb���A�,*;


total_loss�x@

error_R��F?

learning_rate_1먷6c�tI       6%�	�Fmb���A�,*;


total_lossD��@

error_R��O?

learning_rate_1먷6�^U�I       6%�	͕mb���A�,*;


total_loss���@

error_R	K?

learning_rate_1먷6Px�I       6%�	}�mb���A�,*;


total_loss,��@

error_R��K?

learning_rate_1먷6E ��I       6%�	� nb���A�,*;


total_loss�v�@

error_R}P`?

learning_rate_1먷6B��I       6%�	fnb���A�,*;


total_lossl�@

error_R�Db?

learning_rate_1먷6��`�I       6%�	իnb���A�,*;


total_loss�A

error_RhQ?

learning_rate_1먷6o!.�I       6%�	L�nb���A�,*;


total_loss���@

error_R1�D?

learning_rate_1먷6����I       6%�	�5ob���A�,*;


total_loss�̂@

error_Rv�D?

learning_rate_1먷6��#�I       6%�	�yob���A�,*;


total_lossԠ�@

error_R�C;?

learning_rate_1먷6��$I       6%�	O�ob���A�,*;


total_loss!��@

error_R��^?

learning_rate_1먷6R���I       6%�	T�ob���A�,*;


total_lossrv�@

error_R�Z?

learning_rate_1먷6��pI       6%�	Epb���A�,*;


total_loss�*A

error_R�V?

learning_rate_1먷6��JI       6%�	w�pb���A�,*;


total_loss��@

error_Rv=@?

learning_rate_1먷6�0�I       6%�	y�pb���A�,*;


total_loss��@

error_R�C;?

learning_rate_1먷6j���I       6%�	�qb���A�,*;


total_loss��@

error_RP?

learning_rate_1먷6��I       6%�	�Zqb���A�,*;


total_loss��@

error_R�?L?

learning_rate_1먷6+���I       6%�	��qb���A�,*;


total_loss���@

error_RV7[?

learning_rate_1먷6�Pk"I       6%�	@�qb���A�,*;


total_loss�W�@

error_R�XA?

learning_rate_1먷6i���I       6%�	�-rb���A�,*;


total_loss���@

error_R@�<?

learning_rate_1먷6��	�I       6%�	�orb���A�,*;


total_loss�.�@

error_Rq%R?

learning_rate_1먷6�l�I       6%�	��rb���A�,*;


total_loss ��@

error_R��f?

learning_rate_1먷6)HC�I       6%�	!�rb���A�,*;


total_loss�pl@

error_R�aH?

learning_rate_1먷6�ш�I       6%�	�5sb���A�,*;


total_loss�5�@

error_R�@P?

learning_rate_1먷6�ʙ�I       6%�	�|sb���A�,*;


total_loss�^�@

error_R��R?

learning_rate_1먷6��̀I       6%�	�sb���A�,*;


total_loss�س@

error_R{L[?

learning_rate_1먷6:��I       6%�	Ftb���A�,*;


total_loss��@

error_R��M?

learning_rate_1먷6?O-�I       6%�	�Ltb���A�,*;


total_loss3Ɣ@

error_RʋZ?

learning_rate_1먷6���,I       6%�	��tb���A�,*;


total_loss���@

error_R��E?

learning_rate_1먷6s��I       6%�	��tb���A�,*;


total_lossZ�@

error_Rɗ_?

learning_rate_1먷6A�K%I       6%�	�ub���A�,*;


total_loss|��@

error_R_�R?

learning_rate_1먷6FM��I       6%�	�[ub���A�,*;


total_loss���@

error_R�^=?

learning_rate_1먷6Ww`�I       6%�	̞ub���A�,*;


total_losst��@

error_Ra�T?

learning_rate_1먷6S�#I       6%�	��ub���A�,*;


total_loss]Լ@

error_R�Q?

learning_rate_1먷6"i5I       6%�	�&vb���A�,*;


total_lossN�@

error_R:�U?

learning_rate_1먷6˺��I       6%�	�kvb���A�,*;


total_loss��@

error_RE�O?

learning_rate_1먷6o\��I       6%�	ʯvb���A�,*;


total_loss�c�@

error_R$F?

learning_rate_1먷6
8:-I       6%�	��vb���A�,*;


total_loss�g�@

error_RD2R?

learning_rate_1먷6��8�I       6%�	(6wb���A�,*;


total_loss�F�@

error_R�K?

learning_rate_1먷6'�LI       6%�	[}wb���A�,*;


total_loss/ �@

error_R�/C?

learning_rate_1먷6��f5I       6%�	"�wb���A�,*;


total_loss�L�@

error_R�^N?

learning_rate_1먷6X�٧I       6%�	�xb���A�,*;


total_lossI�@

error_RE�N?

learning_rate_1먷6�v�I       6%�	!Oxb���A�,*;


total_loss���@

error_R��K?

learning_rate_1먷66
��I       6%�	:�xb���A�,*;


total_loss���@

error_R��U?

learning_rate_1먷6�g�I       6%�	��xb���A�,*;


total_loss���@

error_R�8A?

learning_rate_1먷6����I       6%�	Jyb���A�,*;


total_loss��@

error_RH�K?

learning_rate_1먷6cI�I       6%�	1[yb���A�,*;


total_loss'�@

error_R�=?

learning_rate_1먷6�hHDI       6%�	�yb���A�,*;


total_loss���@

error_R��K?

learning_rate_1먷6 E�'I       6%�	��yb���A�,*;


total_loss
~�@

error_R��*?

learning_rate_1먷64���I       6%�	+!zb���A�,*;


total_lossar�@

error_R�E?

learning_rate_1먷6O��I       6%�	bzb���A�,*;


total_loss�_@

error_RNWH?

learning_rate_1먷6�Ov(I       6%�	'�zb���A�,*;


total_loss��@

error_R@�>?

learning_rate_1먷6��,I       6%�	��zb���A�,*;


total_loss�+�@

error_R.�L?

learning_rate_1먷6�.�I       6%�	'{b���A�,*;


total_loss�ί@

error_R4
[?

learning_rate_1먷6,~�SI       6%�	:i{b���A�,*;


total_lossLͰ@

error_R�5R?

learning_rate_1먷6�&��I       6%�	��{b���A�,*;


total_loss���@

error_R��@?

learning_rate_1먷6�!��I       6%�	��{b���A�,*;


total_loss<�@

error_R�T?

learning_rate_1먷6���I       6%�	�.|b���A�,*;


total_loss%J�@

error_R^@?

learning_rate_1먷6~�SI       6%�	�n|b���A�,*;


total_loss� �@

error_R\I?

learning_rate_1먷6��I       6%�	��|b���A�,*;


total_lossjW�@

error_R�*U?

learning_rate_1먷69C7	I       6%�	P�|b���A�,*;


total_loss흌@

error_R)H9?

learning_rate_1먷6�a�,I       6%�	�R}b���A�,*;


total_losszޢ@

error_R&^B?

learning_rate_1먷6`�,�I       6%�	m�}b���A�,*;


total_loss	�@

error_R�M?

learning_rate_1먷6��i�I       6%�	1�}b���A�,*;


total_loss.�@

error_R�QL?

learning_rate_1먷6S\lI       6%�	�~b���A�,*;


total_lossϙ�@

error_R$�L?

learning_rate_1먷6�UtI       6%�	�c~b���A�,*;


total_loss�.�@

error_R1n@?

learning_rate_1먷6�z)nI       6%�	��~b���A�,*;


total_loss���@

error_RX?

learning_rate_1먷6��I       6%�	��~b���A�,*;


total_loss�j�@

error_Rq�J?

learning_rate_1먷66�>I       6%�	�Db���A�,*;


total_losseF�@

error_R.�^?

learning_rate_1먷6ϣ"I       6%�	�b���A�,*;


total_lossL��@

error_R�X?

learning_rate_1먷6�!ۧI       6%�	 �b���A�,*;


total_loss%��@

error_RHI?

learning_rate_1먷6yt@I       6%�	s�b���A�,*;


total_loss-��@

error_R��B?

learning_rate_1먷6�}��I       6%�	�V�b���A�,*;


total_loss�5�@

error_R8MX?

learning_rate_1먷6��>�I       6%�	���b���A�,*;


total_loss���@

error_Rm	e?

learning_rate_1먷6��I       6%�	E�b���A�-*;


total_loss2i�@

error_Rl>^?

learning_rate_1먷6#�I       6%�	�Q�b���A�-*;


total_loss��@

error_R�DM?

learning_rate_1먷6]h�I       6%�	d��b���A�-*;


total_lossj!Z@

error_RQI?

learning_rate_1먷6'�<MI       6%�	��b���A�-*;


total_loss���@

error_RV[?

learning_rate_1먷6W�e�I       6%�	�,�b���A�-*;


total_loss]N�@

error_Rc�E?

learning_rate_1먷6�>��I       6%�	q�b���A�-*;


total_lossX �@

error_R!�S?

learning_rate_1먷6�"�I       6%�	=��b���A�-*;


total_loss�b�@

error_R��B?

learning_rate_1먷6
e�I       6%�	'��b���A�-*;


total_loss�P�@

error_RaG?

learning_rate_1먷6���I       6%�	#>�b���A�-*;


total_loss�]@

error_ROpD?

learning_rate_1먷6s9�I       6%�	W��b���A�-*;


total_loss���@

error_R� M?

learning_rate_1먷6���oI       6%�	Ƀb���A�-*;


total_loss�ɞ@

error_R�H?

learning_rate_1먷6p�SI       6%�	�
�b���A�-*;


total_loss���@

error_R3�K?

learning_rate_1먷6ó�^I       6%�	�K�b���A�-*;


total_loss�Ǜ@

error_R�>?

learning_rate_1먷6�s�I       6%�	�b���A�-*;


total_loss��@

error_R`�Q?

learning_rate_1먷6�mnI       6%�	Єb���A�-*;


total_loss�f�@

error_R%A?

learning_rate_1먷67TR�I       6%�	�b���A�-*;


total_loss��@

error_R��D?

learning_rate_1먷6o�_�I       6%�	AQ�b���A�-*;


total_lossq�@

error_RF�\?

learning_rate_1먷6P�|XI       6%�	(��b���A�-*;


total_loss���@

error_R�E?

learning_rate_1먷6<؛I       6%�	>Յb���A�-*;


total_lossY��@

error_R�qQ?

learning_rate_1먷6��N\I       6%�	��b���A�-*;


total_loss�֚@

error_R��O?

learning_rate_1먷6��I       6%�	gX�b���A�-*;


total_loss���@

error_RT�a?

learning_rate_1먷6��eI       6%�	?��b���A�-*;


total_loss�ώ@

error_R��5?

learning_rate_1먷6��^�I       6%�	Q܆b���A�-*;


total_loss�ް@

error_R V?

learning_rate_1먷6�4
I       6%�	S�b���A�-*;


total_loss�U�@

error_R�_?

learning_rate_1먷6x<�8I       6%�	`_�b���A�-*;


total_loss?��@

error_R?�C?

learning_rate_1먷6nYc�I       6%�	c��b���A�-*;


total_loss�r�@

error_Rj=H?

learning_rate_1먷6b��hI       6%�	��b���A�-*;


total_loss��@

error_RA�^?

learning_rate_1먷62uW�I       6%�	�8�b���A�-*;


total_loss{��@

error_R�J^?

learning_rate_1먷6Q�F�I       6%�	��b���A�-*;


total_loss�?�@

error_R�T?

learning_rate_1먷6m�q�I       6%�	�ǈb���A�-*;


total_loss�p�@

error_R��A?

learning_rate_1먷6`��PI       6%�	v�b���A�-*;


total_lossx��@

error_RC?

learning_rate_1먷6iLc(I       6%�	=P�b���A�-*;


total_loss݉�@

error_Rl}V?

learning_rate_1먷6ߵ�]I       6%�	s��b���A�-*;


total_lossW��@

error_R��A?

learning_rate_1먷6����I       6%�	G׉b���A�-*;


total_loss�щ@

error_R�AL?

learning_rate_1먷65ɰ�I       6%�	�b���A�-*;


total_loss��E@

error_R��H?

learning_rate_1먷6
*��I       6%�	�\�b���A�-*;


total_lossa^�@

error_R�nK?

learning_rate_1먷6����I       6%�	Þ�b���A�-*;


total_loss�Ky@

error_R�(A?

learning_rate_1먷6R�]II       6%�	"ߊb���A�-*;


total_loss;��@

error_R�b?

learning_rate_1먷6�B�?I       6%�	}!�b���A�-*;


total_loss�d�@

error_REG?

learning_rate_1먷6(#dyI       6%�	�c�b���A�-*;


total_loss��@

error_R��J?

learning_rate_1먷6�B%I       6%�	���b���A�-*;


total_loss�O�@

error_R��\?

learning_rate_1먷6L�1�I       6%�	��b���A�-*;


total_loss��@

error_R=�F?

learning_rate_1먷6�3�CI       6%�	�$�b���A�-*;


total_loss��A

error_R�9E?

learning_rate_1먷6sOqI       6%�	�d�b���A�-*;


total_lossNպ@

error_R%*H?

learning_rate_1먷6\�iI       6%�	���b���A�-*;


total_loss_��@

error_RD�Q?

learning_rate_1먷6��F�I       6%�	��b���A�-*;


total_loss�܃@

error_R��+?

learning_rate_1먷6�m�I       6%�	TV�b���A�-*;


total_loss�9�@

error_R�6O?

learning_rate_1먷6�Ϩ�I       6%�	Ĝ�b���A�-*;


total_loss_*�@

error_R�J?

learning_rate_1먷6�ǩI       6%�	�ߍb���A�-*;


total_loss��@

error_R�F?

learning_rate_1먷6�D�7I       6%�	�%�b���A�-*;


total_loss���@

error_R�jM?

learning_rate_1먷65{��I       6%�	�k�b���A�-*;


total_loss���@

error_RTV?

learning_rate_1먷6�p�jI       6%�	���b���A�-*;


total_loss���@

error_RH�W?

learning_rate_1먷6^���I       6%�	V�b���A�-*;


total_loss@ٯ@

error_R&qu?

learning_rate_1먷6�~�I       6%�	�V�b���A�-*;


total_loss���@

error_R�|X?

learning_rate_1먷6}sbZI       6%�	9��b���A�-*;


total_loss#��@

error_RT4?

learning_rate_1먷6 fuI       6%�	��b���A�-*;


total_loss�	�@

error_R�W?

learning_rate_1먷6۩½I       6%�	{1�b���A�-*;


total_loss�	�@

error_R|
L?

learning_rate_1먷6�f�hI       6%�	,w�b���A�-*;


total_loss���@

error_R�0H?

learning_rate_1먷6A�݆I       6%�	��b���A�-*;


total_loss
�@

error_R��O?

learning_rate_1먷67���I       6%�	���b���A�-*;


total_lossMr@

error_RW"@?

learning_rate_1먷6۟��I       6%�	�C�b���A�-*;


total_lossa*�@

error_R��B?

learning_rate_1먷6f� �I       6%�	f��b���A�-*;


total_loss�@

error_R�J?

learning_rate_1먷6,�I       6%�	/ȑb���A�-*;


total_loss�q�@

error_R��g?

learning_rate_1먷6����I       6%�	_�b���A�-*;


total_loss��@

error_R͂a?

learning_rate_1먷6(~�hI       6%�	CR�b���A�-*;


total_loss3�@

error_R$M?

learning_rate_1먷6�&��I       6%�	b���A�-*;


total_loss��@

error_R��C?

learning_rate_1먷6'HmI       6%�	�גb���A�-*;


total_loss: �@

error_Rq|E?

learning_rate_1먷6�b$�I       6%�	��b���A�-*;


total_lossz��@

error_R�wV?

learning_rate_1먷6u���I       6%�	�U�b���A�-*;


total_lossl��@

error_R�H?

learning_rate_1먷6�eW�I       6%�	���b���A�-*;


total_loss���@

error_R��I?

learning_rate_1먷6�I�]I       6%�	zܓb���A�-*;


total_lossO�@

error_R�O?

learning_rate_1먷6ͼ�iI       6%�	��b���A�-*;


total_loss-�@

error_R�tK?

learning_rate_1먷6���I       6%�	u_�b���A�-*;


total_lossi�@

error_R�XX?

learning_rate_1먷6���SI       6%�	U��b���A�-*;


total_lossS��@

error_R��V?

learning_rate_1먷6��I       6%�	j�b���A�-*;


total_loss�@

error_Rf�Q?

learning_rate_1먷6to��I       6%�	u#�b���A�-*;


total_loss\Z�@

error_R,3?

learning_rate_1먷6Ψ�I       6%�	zd�b���A�-*;


total_lossX��@

error_R�F?

learning_rate_1먷6�x�?I       6%�	 ��b���A�-*;


total_lossr��@

error_Rn�A?

learning_rate_1먷6�>��I       6%�	g�b���A�-*;


total_loss�j@

error_R��i?

learning_rate_1먷6p4��I       6%�	�(�b���A�-*;


total_loss'�@

error_R�<Z?

learning_rate_1먷6�0/I       6%�	�j�b���A�-*;


total_loss&D�@

error_R_Y5?

learning_rate_1먷6;=PcI       6%�	ҫ�b���A�-*;


total_loss ��@

error_R�q[?

learning_rate_1먷6����I       6%�	��b���A�-*;


total_loss�U�@

error_R*�X?

learning_rate_1먷6'��(I       6%�	C1�b���A�-*;


total_loss��@

error_R)uS?

learning_rate_1먷64#�I       6%�	Qr�b���A�-*;


total_loss��@

error_R�1V?

learning_rate_1먷6�Ǘ-I       6%�	���b���A�-*;


total_loss$��@

error_R�{<?

learning_rate_1먷6'"�fI       6%�	��b���A�-*;


total_loss�ʫ@

error_R�MQ?

learning_rate_1먷6N��[I       6%�	�5�b���A�-*;


total_loss<`}@

error_RO�O?

learning_rate_1먷6.�{.I       6%�	nu�b���A�-*;


total_loss]��@

error_RTlI?

learning_rate_1먷6 �%?I       6%�	ᶘb���A�-*;


total_loss��@

error_R_pH?

learning_rate_1먷6oiX�I       6%�	F��b���A�-*;


total_loss��@

error_R7TD?

learning_rate_1먷6��VI       6%�	�9�b���A�-*;


total_loss���@

error_RʺY?

learning_rate_1먷6���I       6%�	�z�b���A�-*;


total_loss�I�@

error_Ro�E?

learning_rate_1먷6��nUI       6%�	���b���A�-*;


total_loss���@

error_R)dU?

learning_rate_1먷6��EI       6%�	���b���A�-*;


total_loss<+�@

error_R{DP?

learning_rate_1먷6���I       6%�	^A�b���A�-*;


total_loss��@

error_RI�D?

learning_rate_1먷6.��I       6%�	d��b���A�-*;


total_lossjQ�@

error_R�-V?

learning_rate_1먷6N��I       6%�	:Ěb���A�-*;


total_loss��@

error_RH�K?

learning_rate_1먷6��tI       6%�	��b���A�-*;


total_loss)�@

error_R
�J?

learning_rate_1먷6N�:I       6%�	M�b���A�-*;


total_lossq��@

error_R��E?

learning_rate_1먷6X8�I       6%�	���b���A�-*;


total_loss���@

error_R��^?

learning_rate_1먷6��X4I       6%�	�ԛb���A�-*;


total_loss��@

error_Rs�X?

learning_rate_1먷6���MI       6%�	I(�b���A�-*;


total_loss%�@

error_R�L?

learning_rate_1먷6�`N�I       6%�	��b���A�-*;


total_loss鄿@

error_R��B?

learning_rate_1먷6Ԩ��I       6%�	�̜b���A�-*;


total_loss�=�@

error_RZO=?

learning_rate_1먷67-�FI       6%�	9�b���A�-*;


total_loss���@

error_R��R?

learning_rate_1먷68Y�I       6%�	w��b���A�-*;


total_loss>�@

error_R��V?

learning_rate_1먷6S	I       6%�	��b���A�-*;


total_loss�@�@

error_R�	O?

learning_rate_1먷6t�+_I       6%�	�/�b���A�-*;


total_lossJ
A

error_R�ya?

learning_rate_1먷6�esPI       6%�	�u�b���A�-*;


total_loss��A

error_Rn�M?

learning_rate_1먷6T�6�I       6%�	���b���A�-*;


total_loss�S�@

error_R�MM?

learning_rate_1먷6�J
I       6%�	�!�b���A�-*;


total_lossz�@

error_R��L?

learning_rate_1먷6o<�I       6%�	9a�b���A�-*;


total_lossﱦ@

error_R}�:?

learning_rate_1먷6���I       6%�	)��b���A�-*;


total_lossv�A

error_R�9^?

learning_rate_1먷6���I       6%�	��b���A�-*;


total_loss���@

error_R�cL?

learning_rate_1먷6'��I       6%�	 &�b���A�-*;


total_lossE-�@

error_R�O?

learning_rate_1먷6͞I       6%�	Vj�b���A�-*;


total_loss��@

error_R��V?

learning_rate_1먷6�쾾I       6%�	Ī�b���A�-*;


total_loss���@

error_RήV?

learning_rate_1먷6�&-�I       6%�	���b���A�-*;


total_lossh݇@

error_R��Q?

learning_rate_1먷6�u=�I       6%�	)0�b���A�-*;


total_loss8��@

error_R�<?

learning_rate_1먷6j(��I       6%�	t�b���A�-*;


total_loss�gA

error_R��H?

learning_rate_1먷6D�lI       6%�	m��b���A�-*;


total_lossZ�A

error_R�bH?

learning_rate_1먷6AM��I       6%�	���b���A�-*;


total_loss26�@

error_R@�H?

learning_rate_1먷6I�mI       6%�	�7�b���A�-*;


total_loss�@

error_R��Z?

learning_rate_1먷6�i��I       6%�	y�b���A�-*;


total_loss���@

error_R3�B?

learning_rate_1먷6���I       6%�	���b���A�-*;


total_loss���@

error_R�.O?

learning_rate_1먷6#��I       6%�	��b���A�-*;


total_loss��@

error_R�G?

learning_rate_1먷6�=|I       6%�	f9�b���A�-*;


total_loss�.�@

error_RuS?

learning_rate_1먷6 �OI       6%�	6z�b���A�.*;


total_loss���@

error_R�WU?

learning_rate_1먷6�uIZI       6%�	Һ�b���A�.*;


total_lossO��@

error_R2=?

learning_rate_1먷6�.��I       6%�	r��b���A�.*;


total_loss��A

error_RH�Y?

learning_rate_1먷6@H�'I       6%�	T=�b���A�.*;


total_loss�{A

error_Rj*N?

learning_rate_1먷6sɭ9I       6%�	�|�b���A�.*;


total_loss���@

error_R��a?

learning_rate_1먷6��M�I       6%�	���b���A�.*;


total_loss�@

error_R�DO?

learning_rate_1먷6+���I       6%�	�b���A�.*;


total_loss�[}@

error_R,xO?

learning_rate_1먷6���I       6%�	ZC�b���A�.*;


total_loss���@

error_R��T?

learning_rate_1먷6��OI       6%�	}��b���A�.*;


total_loss�O�@

error_R�Y?

learning_rate_1먷6����I       6%�	�åb���A�.*;


total_lossc\A

error_R��W?

learning_rate_1먷6=V�}I       6%�	��b���A�.*;


total_loss��@

error_RaTW?

learning_rate_1먷6�X=HI       6%�	�I�b���A�.*;


total_lossi��@

error_R�O?

learning_rate_1먷6�a[fI       6%�	<��b���A�.*;


total_loss���@

error_RlM?

learning_rate_1먷6�@�I       6%�	��b���A�.*;


total_loss�3�@

error_Rwi=?

learning_rate_1먷6���5I       6%�	%2�b���A�.*;


total_lossс@

error_R��G?

learning_rate_1먷6~l�I       6%�	vw�b���A�.*;


total_loss5ڗ@

error_R�S?

learning_rate_1먷6�N
I       6%�	3��b���A�.*;


total_loss=f�@

error_R�tB?

learning_rate_1먷6�p
sI       6%�	���b���A�.*;


total_lossS�@

error_Ra�Z?

learning_rate_1먷6��Z�I       6%�	�6�b���A�.*;


total_loss9A

error_RŔJ?

learning_rate_1먷6��u�I       6%�	Lw�b���A�.*;


total_loss��A

error_R��G?

learning_rate_1먷6i�:/I       6%�	���b���A�.*;


total_loss��@

error_R;=?

learning_rate_1먷6�@V�I       6%�	���b���A�.*;


total_lossQ�@

error_R�K?

learning_rate_1먷6��Z�I       6%�	pA�b���A�.*;


total_loss���@

error_R��4?

learning_rate_1먷6�*>ZI       6%�	�b���A�.*;


total_loss��R@

error_R��O?

learning_rate_1먷6G�n�I       6%�	ũb���A�.*;


total_loss(��@

error_RC�M?

learning_rate_1먷6RL��I       6%�	��b���A�.*;


total_loss���@

error_R�xV?

learning_rate_1먷6�)�
I       6%�	�M�b���A�.*;


total_loss�A

error_R�?O?

learning_rate_1먷6��u[