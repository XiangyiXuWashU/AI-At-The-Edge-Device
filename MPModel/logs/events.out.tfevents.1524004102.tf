       �K"	  �A���Abrain.Event:2x:7>K     6�.	�	�A���A"��
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
,weights/random_normal_1/RandomStandardNormalRandomStandardNormalweights/random_normal_1/shape*

seed *
T0*
dtype0* 
_output_shapes
:
��*
seed2 
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

seed *
T0*
dtype0*
_output_shapes
:	�d*
seed2 
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
weights/random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
c
weights/random_normal_3/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,weights/random_normal_3/RandomStandardNormalRandomStandardNormalweights/random_normal_3/shape*
T0*
dtype0*
_output_shapes

:d*
seed2 *

seed 
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
biases/bias1/AssignAssignbiases/bias1biases/random_normal*
T0*
_class
loc:@biases/bias1*
validate_shape(*
_output_shapes	
:�*
use_locking(
r
biases/bias1/readIdentitybiases/bias1*
_output_shapes	
:�*
T0*
_class
loc:@biases/bias1
g
biases/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
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
biases/bias2/AssignAssignbiases/bias2biases/random_normal_1*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@biases/bias2
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
+biases/random_normal_2/RandomStandardNormalRandomStandardNormalbiases/random_normal_2/shape*

seed *
T0*
dtype0*
_output_shapes
:d*
seed2 
�
biases/random_normal_2/mulMul+biases/random_normal_2/RandomStandardNormalbiases/random_normal_2/stddev*
_output_shapes
:d*
T0
{
biases/random_normal_2Addbiases/random_normal_2/mulbiases/random_normal_2/mean*
_output_shapes
:d*
T0
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
biases/bias3/AssignAssignbiases/bias3biases/random_normal_2*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@biases/bias3
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
biases/random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
biases/random_normal_3Addbiases/random_normal_3/mulbiases/random_normal_3/mean*
_output_shapes
:*
T0
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
dtype0*
_output_shapes
:*
valueB"�   d   
c
weights_1/random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
weights_1/random_normal_3Addweights_1/random_normal_3/mulweights_1/random_normal_3/mean*
_output_shapes

:d*
T0
�
weights_1/weight_out
VariableV2*
shared_name *
dtype0*
_output_shapes

:d*
	container *
shape
:d
�
weights_1/weight_out/AssignAssignweights_1/weight_outweights_1/random_normal_3*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
biases_1/random_normalAddbiases_1/random_normal/mulbiases_1/random_normal/mean*
_output_shapes	
:�*
T0
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
biases_1/bias1/AssignAssignbiases_1/bias1biases_1/random_normal*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
biases_1/random_normal_1/mulMul-biases_1/random_normal_1/RandomStandardNormalbiases_1/random_normal_1/stddev*
T0*
_output_shapes	
:�
�
biases_1/random_normal_1Addbiases_1/random_normal_1/mulbiases_1/random_normal_1/mean*
_output_shapes	
:�*
T0
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
biases_1/random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
biases_1/random_normal_2/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
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
biases_1/random_normal_3/mulMul-biases_1/random_normal_3/RandomStandardNormalbiases_1/random_normal_3/stddev*
T0*
_output_shapes
:
�
biases_1/random_normal_3Addbiases_1/random_normal_3/mulbiases_1/random_normal_3/mean*
_output_shapes
:*
T0
}
biases_1/bias_out
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
layer_3/SigmoidSigmoidlayer_2/Add*
T0*(
_output_shapes
:����������
�
layer_3/MatMulMatMullayer_3/Sigmoidweights_1/weight3/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
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
result/MatMulMatMulresult/Reluweights_1/weight_out/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
ף<*
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
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
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
"train/gradients/sub_1_grad/Shape_1Shapeinput/Label-input*
_output_shapes
:*
T0*
out_type0
�
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
(train/gradients/layer_3/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:d
�
6train/gradients/layer_3/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/Add_grad/Shape(train/gradients/layer_3/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/Add_grad/SumSum)train/gradients/result/Relu_grad/ReluGrad6train/gradients/layer_3/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
6train/gradients/layer_1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/Add_grad/Shape(train/gradients/layer_1/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/Add_grad/tuple/control_dependencyweights_1/weight1/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking(
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
VariableV2*$
_class
loc:@weights_1/weight1*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
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
weights_1/weight2/Adam_1/readIdentityweights_1/weight2/Adam_1* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight2
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
VariableV2*
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name *$
_class
loc:@weights_1/weight3*
	container 
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
VariableV2*'
_class
loc:@weights_1/weight_out*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
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
biases_1/bias1/Adam_1/readIdentitybiases_1/bias1/Adam_1*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
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
%biases_1/bias3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:d*!
_class
loc:@biases_1/bias3*
valueBd*    
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
biases_1/bias3/Adam_1/readIdentitybiases_1/bias3/Adam_1*
_output_shapes
:d*
T0*!
_class
loc:@biases_1/bias3
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
*biases_1/bias_out/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@biases_1/bias_out*
valueB*    
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
-train/Adam/update_weights_1/weight2/ApplyAdam	ApplyAdamweights_1/weight2weights_1/weight2/Adamweights_1/weight2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*$
_class
loc:@weights_1/weight2
�
-train/Adam/update_weights_1/weight3/ApplyAdam	ApplyAdamweights_1/weight3weights_1/weight3/Adamweights_1/weight3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@weights_1/weight3*
use_nesterov( *
_output_shapes
:	�d*
use_locking( 
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
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
save/Assign_2Assignbiases/bias2save/RestoreV2_2*
use_locking(*
T0*
_class
loc:@biases/bias2*
validate_shape(*
_output_shapes	
:�
r
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBbiases/bias3
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
save/Assign_4Assignbiases/bias_outsave/RestoreV2_4*
use_locking(*
T0*"
_class
loc:@biases/bias_out*
validate_shape(*
_output_shapes
:
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
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assignbiases_1/bias2/Adam_1save/RestoreV2_10*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias2
u
save/RestoreV2_11/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBbiases_1/bias3
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
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assignbiases_1/bias3/Adam_1save/RestoreV2_13*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
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
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_16Assignbiases_1/bias_out/Adam_1save/RestoreV2_16*
T0*$
_class
loc:@biases_1/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_17Assigntrain/beta1_powersave/RestoreV2_17*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking(
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
save/Assign_23Assignweights_1/weight1save/RestoreV2_23*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1
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
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2
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
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
_output_shapes
:*
dtypes
2
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
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
MeanMeanAbsMean/reduction_indices*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"����Ph     6Ԇg	[�A���AJ��
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
shape:����������*
dtype0*(
_output_shapes
:����������
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
weights/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
weights/random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
biases/bias1/AssignAssignbiases/bias1biases/random_normal*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@biases/bias1
r
biases/bias1/readIdentitybiases/bias1*
_output_shapes	
:�*
T0*
_class
loc:@biases/bias1
g
biases/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
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
+biases/random_normal_2/RandomStandardNormalRandomStandardNormalbiases/random_normal_2/shape*

seed *
T0*
dtype0*
_output_shapes
:d*
seed2 
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
biases/random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
,weights_1/random_normal/RandomStandardNormalRandomStandardNormalweights_1/random_normal/shape*
dtype0* 
_output_shapes
:
��*
seed2 *

seed *
T0
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
weights_1/random_normal_1Addweights_1/random_normal_1/mulweights_1/random_normal_1/mean* 
_output_shapes
:
��*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
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
biases_1/random_normal_2Addbiases_1/random_normal_2/mulbiases_1/random_normal_2/mean*
_output_shapes
:d*
T0
z
biases_1/bias3
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
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
layer_1/MatMulMatMulinput/Spectrum-inputweights_1/weight1/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
j
layer_1/AddAddlayer_1/MatMulbiases_1/bias1/read*(
_output_shapes
:����������*
T0
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
result/MatMulMatMulresult/Reluweights_1/weight_out/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
ף<*
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
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
(train/gradients/layer_3/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:d
�
6train/gradients/layer_3/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/Add_grad/Shape(train/gradients/layer_3/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/Add_grad/SumSum)train/gradients/result/Relu_grad/ReluGrad6train/gradients/layer_3/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
(train/gradients/layer_3/Add_grad/ReshapeReshape$train/gradients/layer_3/Add_grad/Sum&train/gradients/layer_3/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
&train/gradients/layer_3/Add_grad/Sum_1Sum)train/gradients/result/Relu_grad/ReluGrad8train/gradients/layer_3/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
*train/gradients/layer_3/Add_grad/Reshape_1Reshape&train/gradients/layer_3/Add_grad/Sum_1(train/gradients/layer_3/Add_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
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
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_3/Sigmoid9train/gradients/layer_3/Add_grad/tuple/control_dependency*
_output_shapes
:	�d*
transpose_a(*
transpose_b( *
T0
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
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes
:	�d*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
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
*train/gradients/layer_2/Add_grad/Reshape_1Reshape&train/gradients/layer_2/Add_grad/Sum_1(train/gradients/layer_2/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
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
VariableV2*$
_class
loc:@weights_1/weight1*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
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
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight2*
	container 
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
(weights_1/weight3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	�d*$
_class
loc:@weights_1/weight3*
valueB	�d*    
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
weights_1/weight3/Adam/AssignAssignweights_1/weight3/Adam(weights_1/weight3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
�
weights_1/weight3/Adam/readIdentityweights_1/weight3/Adam*
_output_shapes
:	�d*
T0*$
_class
loc:@weights_1/weight3
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
VariableV2*
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name *$
_class
loc:@weights_1/weight3*
	container 
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
-weights_1/weight_out/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:d*'
_class
loc:@weights_1/weight_out*
valueBd*    
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
%biases_1/bias1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias1*
valueB�*    
�
biases_1/bias1/Adam
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
%biases_1/bias3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:d*!
_class
loc:@biases_1/bias3*
valueBd*    
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
biases_1/bias3/Adam/AssignAssignbiases_1/bias3/Adam%biases_1/bias3/Adam/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
0train/Adam/update_weights_1/weight_out/ApplyAdam	ApplyAdamweights_1/weight_outweights_1/weight_out/Adamweights_1/weight_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/result/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@weights_1/weight_out*
use_nesterov( *
_output_shapes

:d
�
*train/Adam/update_biases_1/bias1/ApplyAdam	ApplyAdambiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/Add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@biases_1/bias1*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
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
-train/Adam/update_biases_1/bias_out/ApplyAdam	ApplyAdambiases_1/bias_outbiases_1/bias_out/Adambiases_1/bias_out/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/result/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*$
_class
loc:@biases_1/bias_out
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*!
_class
loc:@biases_1/bias1
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
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
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
save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
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
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_9/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBbiases_1/bias2/Adam
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/RestoreV2_16/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bbiases_1/bias_out/Adam_1
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
save/Assign_18Assigntrain/beta2_powersave/RestoreV2_18*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
save/RestoreV2_21/tensor_namesConst*
dtype0*
_output_shapes
:*$
valueBBweights/weight3
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
save/Assign_21Assignweights/weight3save/RestoreV2_21*
T0*"
_class
loc:@weights/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
y
save/RestoreV2_22/tensor_namesConst*
dtype0*
_output_shapes
:*'
valueBBweights/weight_out
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
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
save/Assign_23Assignweights_1/weight1save/RestoreV2_23*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
}
save/RestoreV2_24/tensor_namesConst*
dtype0*
_output_shapes
:*+
value"B Bweights_1/weight1/Adam
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_30/tensor_namesConst*
dtype0*
_output_shapes
:*+
value"B Bweights_1/weight3/Adam
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_34/tensor_namesConst*
dtype0*
_output_shapes
:*0
value'B%Bweights_1/weight_out/Adam_1
k
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0��<�F       r5��	~��A���A*;


total_loss�Ë@

error_RL�E?

learning_rate_1?�7�TH       ��H�	OܧA���A*;


total_lossR$�@

error_RvuQ?

learning_rate_1?�79�n�H       ��H�	0�A���A*;


total_loss��@

error_RרW?

learning_rate_1?�7��8H       ��H�	9~�A���A*;


total_lossE�@

error_R�[??

learning_rate_1?�7�:kH       ��H�	�ʨA���A*;


total_loss�'�@

error_RQLE?

learning_rate_1?�7G1�H       ��H�	H�A���A*;


total_loss$,�@

error_R1�l?

learning_rate_1?�7�MlH       ��H�	of�A���A*;


total_loss�	�@

error_R�mQ?

learning_rate_1?�7��]�H       ��H�	��A���A*;


total_loss�n�@

error_R�(U?

learning_rate_1?�7����H       ��H�	E��A���A*;


total_lossV�@

error_R�;@?

learning_rate_1?�7G@=H       ��H�	~1�A���A	*;


total_loss�$�@

error_R�$I?

learning_rate_1?�7��}�H       ��H�	{s�A���A
*;


total_lossyؖ@

error_RrK?

learning_rate_1?�7Y"�H       ��H�	ɶ�A���A*;


total_loss� �@

error_R@@i?

learning_rate_1?�7�Տ�H       ��H�	���A���A*;


total_loss���@

error_R�9?

learning_rate_1?�7���H       ��H�	<�A���A*;


total_loss$+�@

error_Rn<b?

learning_rate_1?�7�W~�H       ��H�	��A���A*;


total_loss��@

error_R�O?

learning_rate_1?�7�_��H       ��H�	=«A���A*;


total_loss&�@

error_R��@?

learning_rate_1?�7ٍ��H       ��H�	i�A���A*;


total_loss�6�@

error_R�N?

learning_rate_1?�7��H       ��H�	�I�A���A*;


total_lossߝ�@

error_RMR?

learning_rate_1?�7��sH       ��H�	c��A���A*;


total_loss�fA

error_RHF?

learning_rate_1?�7�*�RH       ��H�	sԬA���A*;


total_lossŝ�@

error_R��;?

learning_rate_1?�7بH       ��H�	��A���A*;


total_loss�}�@

error_R��X?

learning_rate_1?�7SPH       ��H�	�A���A*;


total_lossQ��@

error_Rߙ<?

learning_rate_1?�7
V{H       ��H�	�׭A���A*;


total_loss�A�@

error_RCwR?

learning_rate_1?�7�}�H       ��H�	&�A���A*;


total_lossZ�@

error_R
[G?

learning_rate_1?�7��mH       ��H�	�n�A���A*;


total_loss���@

error_R��W?

learning_rate_1?�7y=�KH       ��H�	N��A���A*;


total_lossρ�@

error_R�\?

learning_rate_1?�79�H       ��H�	J��A���A*;


total_loss2��@

error_RL�S?

learning_rate_1?�7�u�H       ��H�	:�A���A*;


total_loss6r�@

error_R\�8?

learning_rate_1?�7[���H       ��H�	\��A���A*;


total_loss(��@

error_R��??

learning_rate_1?�7c�7�H       ��H�	ίA���A*;


total_loss!�q@

error_R�C?

learning_rate_1?�7Y֜�H       ��H�	2�A���A*;


total_loss#P�@

error_R�\?

learning_rate_1?�7��lH       ��H�	�_�A���A*;


total_lossY+�@

error_R��P?

learning_rate_1?�7�W�H       ��H�	ɢ�A���A *;


total_losse��@

error_R��]?

learning_rate_1?�7���H       ��H�	�A���A!*;


total_loss��@

error_R�C8?

learning_rate_1?�7.�uBH       ��H�	�'�A���A"*;


total_lossr�@

error_R��J?

learning_rate_1?�7u���H       ��H�	[o�A���A#*;


total_loss���@

error_R�CO?

learning_rate_1?�7����H       ��H�	
��A���A$*;


total_lossL�@

error_Rr�Y?

learning_rate_1?�7G�l�H       ��H�	���A���A%*;


total_lossl�@

error_R�sM?

learning_rate_1?�7GV|H       ��H�	D?�A���A&*;


total_loss�M�@

error_Rf�G?

learning_rate_1?�7὿SH       ��H�		��A���A'*;


total_loss@�@

error_ReP?

learning_rate_1?�7D5=SH       ��H�	�ʲA���A(*;


total_loss���@

error_R=dE?

learning_rate_1?�7~��vH       ��H�	��A���A)*;


total_loss���@

error_R$,Q?

learning_rate_1?�7N]H       ��H�	/[�A���A**;


total_loss+��@

error_R�]Y?

learning_rate_1?�7U�}H       ��H�	ǟ�A���A+*;


total_loss��@

error_RJ�??

learning_rate_1?�7�v�wH       ��H�	w�A���A,*;


total_loss$�@

error_R��8?

learning_rate_1?�7�R$H       ��H�	IG�A���A-*;


total_loss���@

error_R��G?

learning_rate_1?�7A�kH       ��H�	`��A���A.*;


total_loss��A

error_RT�H?

learning_rate_1?�7a�E�H       ��H�	�ٴA���A/*;


total_losss�@

error_R��>?

learning_rate_1?�7LE�H       ��H�	� �A���A0*;


total_loss)��@

error_R�x@?

learning_rate_1?�7t�mkH       ��H�	c�A���A1*;


total_loss=��@

error_R4�X?

learning_rate_1?�7yfTH       ��H�	��A���A2*;


total_loss���@

error_R�.[?

learning_rate_1?�7xܳ�H       ��H�	��A���A3*;


total_lossLk|@

error_RӭQ?

learning_rate_1?�7�)A0H       ��H�	$4�A���A4*;


total_loss̔�@

error_RJ<U?

learning_rate_1?�7g�=eH       ��H�	Ut�A���A5*;


total_loss��@

error_R��P?

learning_rate_1?�79��H       ��H�	+��A���A6*;


total_lossz��@

error_RO�R?

learning_rate_1?�7�H       ��H�	 �A���A7*;


total_lossj�_@

error_RL�Z?

learning_rate_1?�7Z�,H       ��H�	�g�A���A8*;


total_lossw�@

error_R�I?

learning_rate_1?�7�?-�H       ��H�	粷A���A9*;


total_loss���@

error_Rl�Q?

learning_rate_1?�7A�k4H       ��H�	���A���A:*;


total_loss���@

error_R�J?

learning_rate_1?�7γ�oH       ��H�	�C�A���A;*;


total_loss���@

error_R�`L?

learning_rate_1?�7�:�H       ��H�	/��A���A<*;


total_loss6t�@

error_R�bJ?

learning_rate_1?�7l�ZxH       ��H�	2ϸA���A=*;


total_loss�@�@

error_R�e:?

learning_rate_1?�7�!H       ��H�	��A���A>*;


total_loss`|@

error_R�J?

learning_rate_1?�7���VH       ��H�	*Y�A���A?*;


total_lossݜ@

error_R#�L?

learning_rate_1?�7e|]H       ��H�	�A���A@*;


total_lossDA�@

error_Rt>R?

learning_rate_1?�7�r�H       ��H�	�޹A���AA*;


total_loss���@

error_RV@?

learning_rate_1?�7���_H       ��H�	�!�A���AB*;


total_loss��@

error_R�L?

learning_rate_1?�7��/H       ��H�	�c�A���AC*;


total_lossW��@

error_RiiJ?

learning_rate_1?�7��;H       ��H�	 ��A���AD*;


total_loss<�@

error_R�bX?

learning_rate_1?�7(_�H       ��H�	�A���AE*;


total_loss��@

error_R�D?

learning_rate_1?�7��AH       ��H�	�0�A���AF*;


total_lossj��@

error_RW�Q?

learning_rate_1?�7��H       ��H�	�x�A���AG*;


total_loss��@

error_R��N?

learning_rate_1?�7?C^�H       ��H�	�ŻA���AH*;


total_loss�-�@

error_R��Y?

learning_rate_1?�7��H       ��H�	(�A���AI*;


total_lossFk�@

error_Rt�;?

learning_rate_1?�7���H       ��H�	�V�A���AJ*;


total_loss�|�@

error_R�dY?

learning_rate_1?�7�ȤH       ��H�	A��A���AK*;


total_loss�q�@

error_R�??

learning_rate_1?�7�g��H       ��H�	8�A���AL*;


total_loss�P�@

error_R�>?

learning_rate_1?�7�J_5H       ��H�	�(�A���AM*;


total_loss-#�@

error_RHW`?

learning_rate_1?�7��
�H       ��H�	hl�A���AN*;


total_loss*�\@

error_R��E?

learning_rate_1?�7nĔ�H       ��H�	���A���AO*;


total_loss3ʡ@

error_RV�H?

learning_rate_1?�7@�/H       ��H�	p�A���AP*;


total_loss�I�@

error_R�K?

learning_rate_1?�7g�@sH       ��H�	b9�A���AQ*;


total_losso��@

error_RtC?

learning_rate_1?�7�J]KH       ��H�	e��A���AR*;


total_lossH�@

error_R�I?

learning_rate_1?�7�u�H       ��H�	6˾A���AS*;


total_lossqH�@

error_R�pS?

learning_rate_1?�7y%�H       ��H�	��A���AT*;


total_loss#�@

error_RY?

learning_rate_1?�7G��H       ��H�	�N�A���AU*;


total_loss�s�@

error_R�=O?

learning_rate_1?�7}l�4H       ��H�	���A���AV*;


total_loss,d�@

error_Rl�S?

learning_rate_1?�78+0�H       ��H�	�ڿA���AW*;


total_loss�h�@

error_R�hD?

learning_rate_1?�7;|�yH       ��H�	�A���AX*;


total_loss��L@

error_R 9?

learning_rate_1?�7&_��H       ��H�	b�A���AY*;


total_loss�	�@

error_RamV?

learning_rate_1?�7r~b_H       ��H�	J��A���AZ*;


total_loss���@

error_R��G?

learning_rate_1?�7�~��H       ��H�	���A���A[*;


total_loss�v�@

error_RWZ?

learning_rate_1?�7�5�H       ��H�	+7�A���A\*;


total_loss=��@

error_RʮL?

learning_rate_1?�7�j��H       ��H�	���A���A]*;


total_lossn�@

error_R��n?

learning_rate_1?�7^���H       ��H�	s��A���A^*;


total_loss��@

error_R�S?

learning_rate_1?�7m�eoH       ��H�	 �A���A_*;


total_lossD��@

error_R��U?

learning_rate_1?�7�M��H       ��H�	�h�A���A`*;


total_loss���@

error_R�{E?

learning_rate_1?�7�]��H       ��H�	;��A���Aa*;


total_lossū�@

error_R�>?

learning_rate_1?�7�0��H       ��H�	���A���Ab*;


total_loss�߲@

error_R��??

learning_rate_1?�7d�KLH       ��H�	�9�A���Ac*;


total_lossq�@

error_R�?A?

learning_rate_1?�7�֤H       ��H�	+��A���Ad*;


total_lossH&�@

error_R�aH?

learning_rate_1?�7_��WH       ��H�	C��A���Ae*;


total_losse��@

error_R�GQ?

learning_rate_1?�7y�Y�H       ��H�	��A���Af*;


total_loss�|�@

error_R��P?

learning_rate_1?�7��!H       ��H�	[�A���Ag*;


total_loss��@

error_R17P?

learning_rate_1?�7��/H       ��H�	��A���Ah*;


total_loss�f@

error_R�(O?

learning_rate_1?�7�Z#H       ��H�	���A���Ai*;


total_loss`��@

error_R=�H?

learning_rate_1?�7��FH       ��H�	i'�A���Aj*;


total_loss[!�@

error_RW�^?

learning_rate_1?�7)�?H       ��H�	%p�A���Ak*;


total_loss��@

error_RX�]?

learning_rate_1?�7(b�H       ��H�	ʴ�A���Al*;


total_loss�@

error_R@�<?

learning_rate_1?�7j�r�H       ��H�	+��A���Am*;


total_loss��@

error_R�@?

learning_rate_1?�7&���H       ��H�	�D�A���An*;


total_loss��@

error_R�3?

learning_rate_1?�7�d�tH       ��H�	k��A���Ao*;


total_lossQޞ@

error_RO-S?

learning_rate_1?�7��_�H       ��H�	���A���Ap*;


total_lossd(�@

error_Rr[?

learning_rate_1?�7��KH       ��H�	S.�A���Aq*;


total_loss��@

error_R��@?

learning_rate_1?�7\H       ��H�	�t�A���Ar*;


total_lossnޮ@

error_R��R?

learning_rate_1?�7�S��H       ��H�	C��A���As*;


total_loss[��@

error_Rse?

learning_rate_1?�7>�UH       ��H�	���A���At*;


total_loss<ۣ@

error_R�NH?

learning_rate_1?�7}\�\H       ��H�	A�A���Au*;


total_loss̓�@

error_R��`?

learning_rate_1?�7qހ�H       ��H�	��A���Av*;


total_loss�\�@

error_R�'H?

learning_rate_1?�7�AC�H       ��H�	���A���Aw*;


total_loss�.�@

error_R�L?

learning_rate_1?�7���H       ��H�	��A���Ax*;


total_loss@��@

error_RϥO?

learning_rate_1?�7�J�H       ��H�	�Y�A���Ay*;


total_loss]ۅ@

error_R��M?

learning_rate_1?�7�dZH       ��H�	5��A���Az*;


total_loss�@

error_RqEJ?

learning_rate_1?�7\���H       ��H�	���A���A{*;


total_loss��e@

error_R�LB?

learning_rate_1?�7|[�H       ��H�	�3�A���A|*;


total_loss_��@

error_RE;?

learning_rate_1?�7B#�H       ��H�	�y�A���A}*;


total_lossO*�@

error_RlG?

learning_rate_1?�7(�NTH       ��H�	Ǽ�A���A~*;


total_loss�(�@

error_R,D`?

learning_rate_1?�7��H       ��H�	; �A���A*;


total_loss!a�@

error_Rq5\?

learning_rate_1?�7Sݲ�I       6%�	�F�A���A�*;


total_loss���@

error_RϾZ?

learning_rate_1?�7�8�I       6%�	��A���A�*;


total_loss�ʘ@

error_R��L?

learning_rate_1?�7y6\I       6%�	��A���A�*;


total_loss��@

error_R��K?

learning_rate_1?�7���8I       6%�	��A���A�*;


total_loss���@

error_R��S?

learning_rate_1?�7��I       6%�	�V�A���A�*;


total_loss�A�@

error_R��S?

learning_rate_1?�7��I       6%�	љ�A���A�*;


total_loss���@

error_R<Q?

learning_rate_1?�7�)�sI       6%�	_��A���A�*;


total_loss_�@

error_RON?

learning_rate_1?�7�x�2I       6%�	�#�A���A�*;


total_loss��@

error_R�i?

learning_rate_1?�7��SI       6%�	*p�A���A�*;


total_losslN�@

error_R��P?

learning_rate_1?�7�uvI       6%�	���A���A�*;


total_loss�E�@

error_Ri(<?

learning_rate_1?�7�?�iI       6%�	��A���A�*;


total_loss	�@

error_R��L?

learning_rate_1?�7�$��I       6%�	I�A���A�*;


total_loss�7�@

error_R��a?

learning_rate_1?�7 O]�I       6%�	��A���A�*;


total_lossm��@

error_R\~7?

learning_rate_1?�7��$>I       6%�	?��A���A�*;


total_loss�մ@

error_RR�[?

learning_rate_1?�7���I       6%�	)�A���A�*;


total_lossAX�@

error_Rq�I?

learning_rate_1?�7�5)_I       6%�	�l�A���A�*;


total_lossC�@

error_R}�c?

learning_rate_1?�7x؏sI       6%�	���A���A�*;


total_lossx��@

error_RM�V?

learning_rate_1?�7�B��I       6%�	��A���A�*;


total_loss{=�@

error_R�N?

learning_rate_1?�7���RI       6%�	�6�A���A�*;


total_loss?�@

error_RÜN?

learning_rate_1?�7K*��I       6%�	�z�A���A�*;


total_loss���@

error_R�0I?

learning_rate_1?�7,��cI       6%�	���A���A�*;


total_loss ��@

error_R�[?

learning_rate_1?�7[�j]I       6%�	 �A���A�*;


total_lossg�@

error_R��J?

learning_rate_1?�7���mI       6%�	�T�A���A�*;


total_loss�� A

error_R��G?

learning_rate_1?�7���I       6%�	2��A���A�*;


total_loss���@

error_R�I?

learning_rate_1?�7�M��I       6%�	���A���A�*;


total_loss�I�@

error_R�F?

learning_rate_1?�7��VI       6%�	�1�A���A�*;


total_loss?Eu@

error_R��>?

learning_rate_1?�7���I       6%�	et�A���A�*;


total_loss�,�@

error_R��d?

learning_rate_1?�7��PHI       6%�	��A���A�*;


total_loss�1�@

error_R�JX?

learning_rate_1?�7��BI       6%�	m��A���A�*;


total_loss�԰@

error_R��X?

learning_rate_1?�7�8pI       6%�	=�A���A�*;


total_loss��@

error_R��;?

learning_rate_1?�7�wߊI       6%�	ہ�A���A�*;


total_loss>�@

error_R �T?

learning_rate_1?�7�M�iI       6%�	*��A���A�*;


total_lossaP�@

error_RRM?

learning_rate_1?�7��WI       6%�	c�A���A�*;


total_loss�
�@

error_RsP?

learning_rate_1?�7�/pI       6%�	�[�A���A�*;


total_loss�s�@

error_RSV??

learning_rate_1?�7���I       6%�	K��A���A�*;


total_loss3z�@

error_R��O?

learning_rate_1?�7��D�I       6%�	K��A���A�*;


total_loss�F�@

error_R��J?

learning_rate_1?�7�C�ZI       6%�	�(�A���A�*;


total_loss]�@

error_R!@?

learning_rate_1?�7�ҢI       6%�	�l�A���A�*;


total_loss=>�@

error_R�"T?

learning_rate_1?�7�\f�I       6%�	)��A���A�*;


total_lossZ=�@

error_Ro�:?

learning_rate_1?�7)�%3I       6%�	`�A���A�*;


total_lossst�@

error_Re�;?

learning_rate_1?�7�M��I       6%�	sQ�A���A�*;


total_loss�cA

error_R�J?

learning_rate_1?�7REI       6%�	��A���A�*;


total_loss ��@

error_Rl�U?

learning_rate_1?�7/E^�I       6%�	$��A���A�*;


total_loss��@

error_R -8?

learning_rate_1?�7|Ɓ�I       6%�	RK�A���A�*;


total_loss}þ@

error_Ro�M?

learning_rate_1?�7\xέI       6%�	ǖ�A���A�*;


total_lossm��@

error_R2�R?

learning_rate_1?�7�V�I       6%�	��A���A�*;


total_loss�@

error_R�D?

learning_rate_1?�7»EI       6%�	�%�A���A�*;


total_loss���@

error_R�U?

learning_rate_1?�7t��I       6%�	�i�A���A�*;


total_loss�A�@

error_RoXT?

learning_rate_1?�7����I       6%�	���A���A�*;


total_loss;��@

error_R�9J?

learning_rate_1?�7[��I       6%�	,��A���A�*;


total_lossdK�@

error_R��U?

learning_rate_1?�7����I       6%�	�5�A���A�*;


total_loss�e�@

error_R�:F?

learning_rate_1?�7�	'I       6%�	y�A���A�*;


total_lossV�@

error_R\�>?

learning_rate_1?�7��~I       6%�	���A���A�*;


total_loss�`�@

error_R3�W?

learning_rate_1?�7t	vEI       6%�	n�A���A�*;


total_lossޏ@

error_R��^?

learning_rate_1?�7���}I       6%�	E�A���A�*;


total_lossW��@

error_RݥG?

learning_rate_1?�7h
3hI       6%�	:��A���A�*;


total_loss��@

error_Rx�O?

learning_rate_1?�7��KI       6%�	'��A���A�*;


total_loss&"�@

error_R
�N?

learning_rate_1?�7��c�I       6%�	��A���A�*;


total_loss���@

error_R��Q?

learning_rate_1?�7s�2NI       6%�	pO�A���A�*;


total_loss�G�@

error_RYA?

learning_rate_1?�7SQWtI       6%�	,��A���A�*;


total_lossƺ�@

error_RW>;?

learning_rate_1?�7�G�RI       6%�	���A���A�*;


total_loss@}�@

error_R{�T?

learning_rate_1?�75/I       6%�	1)�A���A�*;


total_loss��@

error_R��P?

learning_rate_1?�7�~��I       6%�	_r�A���A�*;


total_loss&�@

error_R6Z`?

learning_rate_1?�7s>6II       6%�	���A���A�*;


total_losst�@

error_R��J?

learning_rate_1?�7{F�2I       6%�	��A���A�*;


total_loss�%�@

error_R{+f?

learning_rate_1?�7�%�I       6%�	X�A���A�*;


total_loss���@

error_R��I?

learning_rate_1?�7��(I       6%�	֜�A���A�*;


total_loss�E�@

error_RJ-P?

learning_rate_1?�7����I       6%�	]��A���A�*;


total_loss���@

error_R�;?

learning_rate_1?�7C�CiI       6%�	"+�A���A�*;


total_lossm��@

error_R#�P?

learning_rate_1?�7�B� I       6%�	�r�A���A�*;


total_loss���@

error_R�BC?

learning_rate_1?�7G�_I       6%�	���A���A�*;


total_loss2�@

error_R=�I?

learning_rate_1?�7T��I       6%�	z�A���A�*;


total_loss�M�@

error_R-8H?

learning_rate_1?�7�u��I       6%�	�E�A���A�*;


total_loss!��@

error_R[�W?

learning_rate_1?�77�I       6%�	��A���A�*;


total_loss�U�@

error_R�_E?

learning_rate_1?�7,�,�I       6%�	L��A���A�*;


total_lossi7�@

error_R�oO?

learning_rate_1?�7�'I       6%�	&�A���A�*;


total_losshV�@

error_R�9[?

learning_rate_1?�7�>�I       6%�	�V�A���A�*;


total_lossҲ�@

error_R,�O?

learning_rate_1?�70A/uI       6%�	��A���A�*;


total_loss�4�@

error_Rw??

learning_rate_1?�7)��I       6%�		��A���A�*;


total_lossz�@

error_R<U?

learning_rate_1?�7a[��I       6%�	/�A���A�*;


total_losstr�@

error_RםR?

learning_rate_1?�7���I       6%�	�b�A���A�*;


total_loss	V�@

error_R�AE?

learning_rate_1?�7*�n�I       6%�	��A���A�*;


total_loss�h�@

error_RR�N?

learning_rate_1?�7�֊�I       6%�	{��A���A�*;


total_loss���@

error_R��g?

learning_rate_1?�7�8L�I       6%�	*�A���A�*;


total_lossf�@

error_R�J?

learning_rate_1?�7�Q��I       6%�	!o�A���A�*;


total_loss�ۉ@

error_R+@?

learning_rate_1?�7h���I       6%�	|��A���A�*;


total_loss�T�@

error_Ra$K?

learning_rate_1?�7|#KI       6%�	���A���A�*;


total_loss�r�@

error_RT�N?

learning_rate_1?�7h�I       6%�	�>�A���A�*;


total_losszx�@

error_R�^?

learning_rate_1?�7*���I       6%�	݂�A���A�*;


total_loss{��@

error_Rd?

learning_rate_1?�7W/n�I       6%�	#��A���A�*;


total_loss��@

error_R&+D?

learning_rate_1?�7�!�I       6%�	J�A���A�*;


total_loss�V�@

error_R��@?

learning_rate_1?�7�,G�I       6%�	�H�A���A�*;


total_loss��@

error_R�;L?

learning_rate_1?�7�T?I       6%�	���A���A�*;


total_loss�Do@

error_R��J?

learning_rate_1?�7�gI       6%�	���A���A�*;


total_loss�@

error_RW�N?

learning_rate_1?�7* ��I       6%�	��A���A�*;


total_loss��@

error_R�B?

learning_rate_1?�7�C��I       6%�	�U�A���A�*;


total_loss2V�@

error_RL�H?

learning_rate_1?�7�S��I       6%�	o��A���A�*;


total_lossկ@

error_R��H?

learning_rate_1?�7-ktnI       6%�	q��A���A�*;


total_loss���@

error_R��=?

learning_rate_1?�7 ���I       6%�	l6�A���A�*;


total_lossthi@

error_R	8g?

learning_rate_1?�71|$�I       6%�	J�A���A�*;


total_lossɬ�@

error_R$B?

learning_rate_1?�7A��I       6%�	'��A���A�*;


total_loss|��@

error_R� G?

learning_rate_1?�7	�HI       6%�	!�A���A�*;


total_loss���@

error_R1�I?

learning_rate_1?�7i)#�I       6%�	�h�A���A�*;


total_loss��	A

error_R��I?

learning_rate_1?�71zI       6%�	��A���A�*;


total_loss|d�@

error_RD�@?

learning_rate_1?�7�l*LI       6%�	���A���A�*;


total_loss�@

error_R��K?

learning_rate_1?�7P���I       6%�	�:�A���A�*;


total_loss �@

error_R�eR?

learning_rate_1?�7A�RI       6%�	?��A���A�*;


total_loss���@

error_RVkO?

learning_rate_1?�7�"}�I       6%�	"��A���A�*;


total_losst#�@

error_RiV?

learning_rate_1?�7���JI       6%�	�A���A�*;


total_loss�ܶ@

error_R��[?

learning_rate_1?�7z�n]I       6%�	iR�A���A�*;


total_loss�$�@

error_R�8L?

learning_rate_1?�7h�D5I       6%�	Ǘ�A���A�*;


total_loss�+�@

error_RS�B?

learning_rate_1?�7�~��I       6%�	#��A���A�*;


total_loss#&�@

error_R�'G?

learning_rate_1?�7|��;I       6%�	-0�A���A�*;


total_loss�@

error_R�R?

learning_rate_1?�7g�?�I       6%�	�u�A���A�*;


total_loss�-�@

error_R�3T?

learning_rate_1?�7A|�I       6%�	���A���A�*;


total_loss]un@

error_R�?K?

learning_rate_1?�7����I       6%�	� �A���A�*;


total_lossr.�@

error_R��U?

learning_rate_1?�7'��gI       6%�	$G�A���A�*;


total_lossĺ@

error_RA�U?

learning_rate_1?�7��@[I       6%�	��A���A�*;


total_lossmN�@

error_R7H?

learning_rate_1?�7A@�eI       6%�	���A���A�*;


total_lossF��@

error_R�J?

learning_rate_1?�7bҕ�I       6%�	��A���A�*;


total_loss���@

error_R;�B?

learning_rate_1?�7�;I       6%�	�X�A���A�*;


total_loss�;@

error_RWL?

learning_rate_1?�7)�D�I       6%�	���A���A�*;


total_loss��@

error_R�I?

learning_rate_1?�7F]�QI       6%�	6��A���A�*;


total_loss���@

error_R[V?

learning_rate_1?�7^66AI       6%�	�$�A���A�*;


total_loss;�`@

error_R\�B?

learning_rate_1?�7�XWI       6%�	ko�A���A�*;


total_loss��	A

error_RM?

learning_rate_1?�7��^I       6%�	��A���A�*;


total_loss�R�@

error_R�,^?

learning_rate_1?�7�O,�I       6%�	� �A���A�*;


total_lossA:�@

error_R%�N?

learning_rate_1?�7eM:�I       6%�	WC�A���A�*;


total_loss��@

error_R_�d?

learning_rate_1?�7s;��I       6%�	��A���A�*;


total_lossd��@

error_R��@?

learning_rate_1?�7<�P�I       6%�	d��A���A�*;


total_loss��q@

error_R�+a?

learning_rate_1?�7�Ҥ�I       6%�	��A���A�*;


total_loss[��@

error_R
�W?

learning_rate_1?�7���I       6%�	�N�A���A�*;


total_lossR��@

error_R�ze?

learning_rate_1?�7�'�PI       6%�	���A���A�*;


total_loss��@

error_RI?

learning_rate_1?�7%^��I       6%�	S��A���A�*;


total_loss���@

error_RE�O?

learning_rate_1?�7�(_I       6%�	�"�A���A�*;


total_loss�S�@

error_R��F?

learning_rate_1?�7h��I       6%�	�h�A���A�*;


total_loss	%�@

error_R�P?

learning_rate_1?�7#��I       6%�	a��A���A�*;


total_loss��@

error_Rv5[?

learning_rate_1?�75_1cI       6%�	���A���A�*;


total_loss�#�@

error_R�??

learning_rate_1?�7��h�I       6%�	E;�A���A�*;


total_loss��@

error_R��[?

learning_rate_1?�7L��2I       6%�	�|�A���A�*;


total_loss|��@

error_R��H?

learning_rate_1?�7�%I       6%�	���A���A�*;


total_loss^�@

error_R
IX?

learning_rate_1?�7�V��I       6%�	��A���A�*;


total_loss6,�@

error_R$�L?

learning_rate_1?�7-F�I       6%�	�P�A���A�*;


total_loss��@

error_R��[?

learning_rate_1?�7pD�I       6%�	C��A���A�*;


total_loss��@

error_R}vV?

learning_rate_1?�7V��I       6%�	���A���A�*;


total_lossH#�@

error_RiMU?

learning_rate_1?�7Ѹ��I       6%�	s!�A���A�*;


total_loss�&A

error_R��A?

learning_rate_1?�7�I       6%�	<l�A���A�*;


total_lossh��@

error_Ri�M?

learning_rate_1?�7�\Y I       6%�	ǯ�A���A�*;


total_loss��@

error_R��K?

learning_rate_1?�7H��I       6%�	&��A���A�*;


total_lossxh�@

error_R,�Q?

learning_rate_1?�7ACKI       6%�	;�A���A�*;


total_loss���@

error_R�xc?

learning_rate_1?�7�p�I       6%�	\�A���A�*;


total_loss���@

error_R�GZ?

learning_rate_1?�7a��QI       6%�	���A���A�*;


total_loss[�@

error_R�lR?

learning_rate_1?�7��l�I       6%�	��A���A�*;


total_loss���@

error_R�X?

learning_rate_1?�7�m�CI       6%�	dK�A���A�*;


total_loss��@

error_R�~Q?

learning_rate_1?�7ͅKKI       6%�	+��A���A�*;


total_loss�Һ@

error_R�E?

learning_rate_1?�7��YDI       6%�	N��A���A�*;


total_lossa��@

error_RlQ?

learning_rate_1?�7i�OI       6%�	v�A���A�*;


total_loss��@

error_Rw�M?

learning_rate_1?�7�[^(I       6%�	�c�A���A�*;


total_loss�ە@

error_R �I?

learning_rate_1?�7\�P�I       6%�	��A���A�*;


total_loss���@

error_R�;?

learning_rate_1?�7���I       6%�	���A���A�*;


total_loss��@

error_R&Z?

learning_rate_1?�7��ppI       6%�	wN�A���A�*;


total_loss��@

error_RR�T?

learning_rate_1?�7�:AGI       6%�	`��A���A�*;


total_losse֙@

error_R\?

learning_rate_1?�7�>�HI       6%�	���A���A�*;


total_loss��@

error_R��S?

learning_rate_1?�7C{�7I       6%�	��A���A�*;


total_loss,�A

error_RsHU?

learning_rate_1?�7�^�I       6%�	X\�A���A�*;


total_loss���@

error_R�lB?

learning_rate_1?�7]��I       6%�	!��A���A�*;


total_loss�o�@

error_R�_?

learning_rate_1?�7��NI       6%�	���A���A�*;


total_loss���@

error_RW�Z?

learning_rate_1?�7����I       6%�	v#�A���A�*;


total_lossxA

error_R_ZM?

learning_rate_1?�7�HGI       6%�	}e�A���A�*;


total_loss���@

error_R@|I?

learning_rate_1?�7mK�BI       6%�	���A���A�*;


total_loss	+�@

error_R�IS?

learning_rate_1?�7{��I       6%�	���A���A�*;


total_loss�ר@

error_Rxrc?

learning_rate_1?�7ząI       6%�	v3�A���A�*;


total_lossP�@

error_R$�E?

learning_rate_1?�7p�0�I       6%�	y�A���A�*;


total_loss�)�@

error_R%mV?

learning_rate_1?�7u�b�I       6%�	��A���A�*;


total_lossC>
A

error_Rl�D?

learning_rate_1?�7��^VI       6%�	��A���A�*;


total_loss}�@

error_R��[?

learning_rate_1?�7p�0I       6%�	�f�A���A�*;


total_loss&��@

error_R�mO?

learning_rate_1?�7�ݕ�I       6%�	ȫ�A���A�*;


total_lossSY�@

error_R��N?

learning_rate_1?�7��7xI       6%�	J��A���A�*;


total_loss��@

error_RC!Z?

learning_rate_1?�7P� gI       6%�	�4�A���A�*;


total_loss�rm@

error_R�?O?

learning_rate_1?�7���.I       6%�	�{�A���A�*;


total_loss�{�@

error_RL-T?

learning_rate_1?�7`[y�I       6%�	���A���A�*;


total_loss���@

error_R��E?

learning_rate_1?�7:�&�I       6%�	�	�A���A�*;


total_loss���@

error_R)�N?

learning_rate_1?�7e��<I       6%�	�O�A���A�*;


total_lossA@�@

error_R(?T?

learning_rate_1?�7xFI       6%�	���A���A�*;


total_loss�Iw@

error_R��K?

learning_rate_1?�7�*�I       6%�	���A���A�*;


total_loss��@

error_Rl�M?

learning_rate_1?�7e��DI       6%�	��A���A�*;


total_lossu�@

error_RJM?

learning_rate_1?�7Y�"�I       6%�	u`�A���A�*;


total_lossR�@

error_R�e/?

learning_rate_1?�7�ǠI       6%�	R��A���A�*;


total_loss���@

error_R^J?

learning_rate_1?�7��xI       6%�	Q��A���A�*;


total_loss,��@

error_R��E?

learning_rate_1?�7�Q��I       6%�	�;�A���A�*;


total_lossa�@

error_R2�C?

learning_rate_1?�7soFkI       6%�	ـ�A���A�*;


total_lossT��@

error_R{Z?

learning_rate_1?�7��q	I       6%�	f��A���A�*;


total_loss�e@

error_R_eA?

learning_rate_1?�7[�pEI       6%�	G B���A�*;


total_loss\,�@

error_R� D?

learning_rate_1?�7�"�hI       6%�	�J B���A�*;


total_loss��@

error_R��R?

learning_rate_1?�7�cS>I       6%�	�� B���A�*;


total_lossm�@

error_R3�P?

learning_rate_1?�7lz�II       6%�	�� B���A�*;


total_loss�q�@

error_R��O?

learning_rate_1?�7��I       6%�	B���A�*;


total_loss�	�@

error_Ro�Y?

learning_rate_1?�7�E��I       6%�	WB���A�*;


total_loss @

error_Ro_?

learning_rate_1?�7bCI       6%�	�B���A�*;


total_loss£@

error_R��P?

learning_rate_1?�7�S�I       6%�	��B���A�*;


total_loss��@

error_Rn[?

learning_rate_1?�7F�M�I       6%�	'B���A�*;


total_loss�y�@

error_ReyN?

learning_rate_1?�7�D�I       6%�	jnB���A�*;


total_loss2��@

error_R3�I?

learning_rate_1?�7��RI       6%�	$�B���A�*;


total_loss9�A

error_R��7?

learning_rate_1?�7���I       6%�	!�B���A�*;


total_loss���@

error_R̡H?

learning_rate_1?�7_/+�I       6%�	�BB���A�*;


total_lossx1�@

error_R��Q?

learning_rate_1?�7�7I       6%�	�B���A�*;


total_loss�F�@

error_REQQ?

learning_rate_1?�7p:3RI       6%�	\�B���A�*;


total_lossq-�@

error_R��Y?

learning_rate_1?�7��I       6%�	�
B���A�*;


total_losse�Y@

error_R��E?

learning_rate_1?�7���I       6%�	�OB���A�*;


total_loss3�@

error_R;�J?

learning_rate_1?�7�o �I       6%�	��B���A�*;


total_lossI�@

error_R�T?

learning_rate_1?�7y��aI       6%�	<�B���A�*;


total_losst�@

error_R9Y?

learning_rate_1?�7���-I       6%�	B���A�*;


total_loss���@

error_R�M_?

learning_rate_1?�7\�7�I       6%�	�]B���A�*;


total_loss:��@

error_R�P?

learning_rate_1?�7�ѡ�I       6%�	T�B���A�*;


total_lossexA

error_R.H?

learning_rate_1?�7���6I       6%�	��B���A�*;


total_lossh-A

error_RD2R?

learning_rate_1?�7�`�BI       6%�	n'B���A�*;


total_loss�|�@

error_R��K?

learning_rate_1?�7Z�LI       6%�	�lB���A�*;


total_loss�i�@

error_R��G?

learning_rate_1?�7>�K I       6%�	Q�B���A�*;


total_lossYʘ@

error_R�F?

learning_rate_1?�7���I       6%�	B���A�*;


total_lossFmR@

error_RI:?

learning_rate_1?�7�)�I       6%�	{fB���A�*;


total_loss��{@

error_R��L?

learning_rate_1?�7�g�I       6%�	ѪB���A�*;


total_loss�.�@

error_R�dV?

learning_rate_1?�7'T��I       6%�	#�B���A�*;


total_loss4�@

error_R��I?

learning_rate_1?�7����I       6%�	i/B���A�*;


total_loss8g�@

error_R��L?

learning_rate_1?�7�4!II       6%�	KrB���A�*;


total_loss�I�@

error_R!f?

learning_rate_1?�7�*��I       6%�	ʵB���A�*;


total_loss*��@

error_R�1A?

learning_rate_1?�7ɔ)�I       6%�	��B���A�*;


total_loss���@

error_R��R?

learning_rate_1?�7%�I       6%�	�?	B���A�*;


total_losse�@

error_Rv�:?

learning_rate_1?�7��ǨI       6%�	?	B���A�*;


total_loss��@

error_R�N?

learning_rate_1?�7h�F�I       6%�	8�	B���A�*;


total_loss�0�@

error_R��S?

learning_rate_1?�7��n�I       6%�	�	
B���A�*;


total_lossTs�@

error_R�OI?

learning_rate_1?�7LHg<I       6%�	ON
B���A�*;


total_loss���@

error_RN�a?

learning_rate_1?�7.|�I       6%�	̑
B���A�*;


total_loss�� A

error_R �F?

learning_rate_1?�7s�i�I       6%�	��
B���A�*;


total_loss�d�@

error_Ra�E?

learning_rate_1?�7=��I       6%�	+B���A�*;


total_loss���@

error_R��G?

learning_rate_1?�7/oxI       6%�	�oB���A�*;


total_losse�@

error_Rag?

learning_rate_1?�7���fI       6%�	F�B���A�*;


total_lossv��@

error_R�4K?

learning_rate_1?�7��`dI       6%�	w�B���A�*;


total_lossA�@

error_R_�T?

learning_rate_1?�7[ԗ�I       6%�	t<B���A�*;


total_losssB�@

error_R�(L?

learning_rate_1?�7�*%I       6%�	��B���A�*;


total_loss�@

error_Rw:>?

learning_rate_1?�7�,�!I       6%�	D�B���A�*;


total_losse*�@

error_R@�U?

learning_rate_1?�7�e�I       6%�	4	B���A�*;


total_lossR��@

error_R��K?

learning_rate_1?�7_�ЙI       6%�	MB���A�*;


total_lossO�@

error_R6�1?

learning_rate_1?�7!�a�I       6%�	4�B���A�*;


total_loss��@

error_RC�L?

learning_rate_1?�7t�SI       6%�	��B���A�*;


total_loss?�@

error_R��9?

learning_rate_1?�7�Cd�I       6%�	�#B���A�*;


total_lossS� A

error_R��F?

learning_rate_1?�7�
�HI       6%�	�oB���A�*;


total_loss�A

error_R�+J?

learning_rate_1?�7�S7I       6%�	*�B���A�*;


total_loss|��@

error_RsCf?

learning_rate_1?�7��^�I       6%�	��B���A�*;


total_loss��@

error_R�A7?

learning_rate_1?�7���I       6%�	�<B���A�*;


total_loss�A�@

error_R�VN?

learning_rate_1?�7|DhI       6%�	X�B���A�*;


total_loss���@

error_R�,S?

learning_rate_1?�7��'�I       6%�	��B���A�*;


total_lossq��@

error_Rl�B?

learning_rate_1?�7$���I       6%�	�B���A�*;


total_lossX|A

error_R?�M?

learning_rate_1?�7y�;I       6%�	�_B���A�*;


total_loss�[�@

error_R.+Q?

learning_rate_1?�7���I       6%�	�B���A�*;


total_loss�iA

error_R��W?

learning_rate_1?�7����I       6%�	��B���A�*;


total_losssA

error_R)Q?

learning_rate_1?�7:2u�I       6%�	�?B���A�*;


total_lossA��@

error_R�0?

learning_rate_1?�7���I       6%�	�B���A�*;


total_loss_��@

error_R�V?

learning_rate_1?�7���I       6%�	��B���A�*;


total_loss��@

error_R�P?

learning_rate_1?�7P#'I       6%�	�	B���A�*;


total_loss`�{@

error_R�RP?

learning_rate_1?�7(5i�I       6%�	wMB���A�*;


total_lossw��@

error_R��A?

learning_rate_1?�7]�B�I       6%�	
�B���A�*;


total_lossV�@

error_R�NP?

learning_rate_1?�7��dLI       6%�	��B���A�*;


total_loss���@

error_R�A?

learning_rate_1?�7��+)I       6%�	�-B���A�*;


total_loss�V�@

error_RH�J?

learning_rate_1?�7�]�I       6%�	tB���A�*;


total_loss��@

error_R��P?

learning_rate_1?�7`��{I       6%�	��B���A�*;


total_loss�o�@

error_R�mH?

learning_rate_1?�71�
�I       6%�	�B���A�*;


total_loss��@

error_Rz`?

learning_rate_1?�7<���I       6%�	IB���A�*;


total_loss?3�@

error_R�dX?

learning_rate_1?�7���I       6%�	K�B���A�*;


total_loss���@

error_R�5?

learning_rate_1?�7բ6�I       6%�	��B���A�*;


total_lossb�@

error_R$�E?

learning_rate_1?�7�jh�I       6%�	�B���A�*;


total_lossf��@

error_R�r;?

learning_rate_1?�7�oz�I       6%�	�YB���A�*;


total_loss��h@

error_R߱B?

learning_rate_1?�7E�K�I       6%�	l�B���A�*;


total_lossܽ�@

error_R=�G?

learning_rate_1?�7�9�I       6%�	A�B���A�*;


total_loss��\@

error_R7�O?

learning_rate_1?�7�"��I       6%�	�%B���A�*;


total_loss �@

error_R��K?

learning_rate_1?�7��mI       6%�	�hB���A�*;


total_loss�d�@

error_R,G?

learning_rate_1?�7Wb��I       6%�	8�B���A�*;


total_loss1�e@

error_R�O?

learning_rate_1?�7{0�I       6%�	��B���A�*;


total_lossV,�@

error_R�VM?

learning_rate_1?�7�uI       6%�	IXB���A�*;


total_lossҲ�@

error_R�GJ?

learning_rate_1?�7�J}JI       6%�	ȜB���A�*;


total_loss/��@

error_RTT?

learning_rate_1?�7�zK�I       6%�	�B���A�*;


total_loss=��@

error_Rx"E?

learning_rate_1?�7e
O�I       6%�	�'B���A�*;


total_loss��@

error_Rl�J?

learning_rate_1?�7�ܖI       6%�	~kB���A�*;


total_loss,�@

error_RsjF?

learning_rate_1?�7����I       6%�	ôB���A�*;


total_loss���@

error_R��b?

learning_rate_1?�7~/o�I       6%�	�B���A�*;


total_lossR;�@

error_R �P?

learning_rate_1?�7#�d�I       6%�	�?B���A�*;


total_loss�@�@

error_R�MJ?

learning_rate_1?�7,<�!I       6%�	u�B���A�*;


total_loss�(�@

error_RjW?

learning_rate_1?�7��@I       6%�	��B���A�*;


total_loss��@

error_R�U?

learning_rate_1?�7�?y�I       6%�	�B���A�*;


total_lossvF�@

error_R*�D?

learning_rate_1?�7�F�gI       6%�	$ZB���A�*;


total_loss�s�@

error_Ra�F?

learning_rate_1?�7�p�I       6%�	��B���A�*;


total_loss�g�@

error_R�B6?

learning_rate_1?�75�<I       6%�	��B���A�*;


total_loss!��@

error_R��X?

learning_rate_1?�7ϸ �I       6%�	6&B���A�*;


total_lossQn@

error_R�>U?

learning_rate_1?�7Q��XI       6%�	�hB���A�*;


total_lossX��@

error_RMa?

learning_rate_1?�7&�K;I       6%�	>�B���A�*;


total_loss:�@

error_R��S?

learning_rate_1?�7� s�I       6%�	��B���A�*;


total_lossjQ�@

error_Rx	O?

learning_rate_1?�7�t�I       6%�	�4B���A�*;


total_lossLzA

error_R�)V?

learning_rate_1?�7B��bI       6%�	YzB���A�*;


total_loss0�@

error_R3�Q?

learning_rate_1?�7�и�I       6%�	��B���A�*;


total_loss���@

error_Rf�N?

learning_rate_1?�7/_��I       6%�	�	B���A�*;


total_lossᾯ@

error_R�"E?

learning_rate_1?�71���I       6%�	<RB���A�*;


total_loss���@

error_R��W?

learning_rate_1?�7o�}�I       6%�	˚B���A�*;


total_loss�=�@

error_RI]?

learning_rate_1?�73�#I       6%�	A�B���A�*;


total_loss�@�@

error_RH�Z?

learning_rate_1?�7�t�DI       6%�	�&B���A�*;


total_lossϏ�@

error_R{[Q?

learning_rate_1?�7�KxI       6%�	<jB���A�*;


total_lossW5�@

error_RHH?

learning_rate_1?�7�\kI       6%�	+�B���A�*;


total_loss��@

error_RC�=?

learning_rate_1?�7s��I       6%�	`�B���A�*;


total_loss.Y�@

error_R�P?

learning_rate_1?�7�$�!I       6%�	�1B���A�*;


total_loss��@

error_R�Ob?

learning_rate_1?�7��N�I       6%�	WuB���A�*;


total_loss��@

error_Rn�S?

learning_rate_1?�7��޼I       6%�	��B���A�*;


total_lossܜ�@

error_R�cT?

learning_rate_1?�7w `�I       6%�	��B���A�*;


total_loss�MLA

error_RE�\?

learning_rate_1?�7
��I       6%�	�? B���A�*;


total_loss%��@

error_RҾ@?

learning_rate_1?�7-���I       6%�	:� B���A�*;


total_loss��@

error_R�%J?

learning_rate_1?�7v>�I       6%�	S� B���A�*;


total_lossS��@

error_R*�B?

learning_rate_1?�7�]ۦI       6%�	:!B���A�*;


total_loss�@

error_R jD?

learning_rate_1?�7�D[;I       6%�	Ie!B���A�*;


total_loss�ݭ@

error_R�R?

learning_rate_1?�7[�I       6%�	��!B���A�*;


total_loss�f�@

error_R�(R?

learning_rate_1?�7_W,I       6%�	��!B���A�*;


total_loss� �@

error_RNN?

learning_rate_1?�7&�l�I       6%�	X6"B���A�*;


total_loss���@

error_R.�b?

learning_rate_1?�7�P�I       6%�	V~"B���A�*;


total_lossk�@

error_R�x=?

learning_rate_1?�7[�x�I       6%�	f�"B���A�*;


total_loss໸@

error_RzR?

learning_rate_1?�7�-�I       6%�	�#B���A�*;


total_loss}P�@

error_R
@_?

learning_rate_1?�7����I       6%�	�R#B���A�*;


total_loss�@

error_Rh�<?

learning_rate_1?�7�eI       6%�	��#B���A�*;


total_loss�_�@

error_R�BS?

learning_rate_1?�7G�H�I       6%�	�#B���A�*;


total_loss�0�@

error_R/D?

learning_rate_1?�7�Ն�I       6%�	 $B���A�*;


total_loss�K�@

error_R�kY?

learning_rate_1?�7�K�I       6%�	�c$B���A�*;


total_lossK�@

error_R{�W?

learning_rate_1?�7BG~I       6%�	q�$B���A�*;


total_loss�V�@

error_R�]U?

learning_rate_1?�7��OI       6%�	��$B���A�*;


total_lossx��@

error_R��9?

learning_rate_1?�7����I       6%�	)2%B���A�*;


total_lossj&A

error_R�_K?

learning_rate_1?�7�X�.I       6%�	+x%B���A�*;


total_loss7��@

error_R��I?

learning_rate_1?�7��+�I       6%�	4�%B���A�*;


total_loss�D�@

error_R6�O?

learning_rate_1?�7���!I       6%�	��%B���A�*;


total_loss���@

error_R�|U?

learning_rate_1?�7��"I       6%�	/A&B���A�*;


total_loss�C|@

error_RRO?

learning_rate_1?�7�+5I       6%�	݄&B���A�*;


total_lossF�@

error_R{�W?

learning_rate_1?�7���I       6%�	��&B���A�*;


total_lossP�@

error_R�3Y?

learning_rate_1?�7o�yI       6%�	�'B���A�*;


total_loss}0A

error_Rn�N?

learning_rate_1?�7c��lI       6%�	0j'B���A�*;


total_lossJ�@

error_R/�^?

learning_rate_1?�7����I       6%�	B�'B���A�*;


total_loss���@

error_R��b?

learning_rate_1?�7�?�I       6%�	��'B���A�*;


total_loss��@

error_R�T?

learning_rate_1?�7ө {I       6%�	I9(B���A�*;


total_lossi�@

error_R��I?

learning_rate_1?�7-�I       6%�	�}(B���A�*;


total_loss�q�@

error_R�V?

learning_rate_1?�7�K�I       6%�	�(B���A�*;


total_loss�Xa@

error_RH@G?

learning_rate_1?�7�H��I       6%�	`)B���A�*;


total_lossXbA

error_R�2H?

learning_rate_1?�78���I       6%�	�K)B���A�*;


total_lossnm@

error_R�/g?

learning_rate_1?�7[�hI       6%�	z�)B���A�*;


total_lossg�@

error_R%hU?

learning_rate_1?�7�΢RI       6%�	��)B���A�*;


total_loss�R�@

error_RlW?

learning_rate_1?�7�7��I       6%�	�*B���A�*;


total_loss���@

error_R+I?

learning_rate_1?�7;�XI       6%�	�j*B���A�*;


total_lossr&�@

error_R�;P?

learning_rate_1?�7�72I       6%�	`�*B���A�*;


total_lossϏ�@

error_RHhA?

learning_rate_1?�7O�u3I       6%�	��*B���A�*;


total_loss-N�@

error_R7C?

learning_rate_1?�7�A��I       6%�	�E+B���A�*;


total_loss;�@

error_RӓF?

learning_rate_1?�7	s��I       6%�	!�+B���A�*;


total_lossMA�@

error_R�mM?

learning_rate_1?�7w�QVI       6%�	��+B���A�*;


total_lossv�@

error_R6�W?

learning_rate_1?�7�pٕI       6%�	�*,B���A�*;


total_loss@��@

error_R7[?

learning_rate_1?�7���I       6%�	�,B���A�*;


total_lossO��@

error_R��,?

learning_rate_1?�7��FII       6%�	�,B���A�*;


total_loss���@

error_R�<?

learning_rate_1?�7ΩE�I       6%�	P-B���A�*;


total_loss�O�@

error_R�J?

learning_rate_1?�7p���I       6%�	a[-B���A�*;


total_loss��@

error_R�B?

learning_rate_1?�7z�I       6%�	G�-B���A�*;


total_loss�e�@

error_R�M?

learning_rate_1?�7���CI       6%�	=�-B���A�*;


total_loss�٫@

error_R��c?

learning_rate_1?�7&�~�I       6%�	�'.B���A�*;


total_lossdװ@

error_R@V?

learning_rate_1?�7m�E�I       6%�	�k.B���A�*;


total_lossak�@

error_R@�S?

learning_rate_1?�7���"I       6%�	ͯ.B���A�*;


total_loss3�z@

error_R&�J?

learning_rate_1?�7�
fyI       6%�	��.B���A�*;


total_lossqŤ@

error_RT�D?

learning_rate_1?�7�ѥI       6%�	�8/B���A�*;


total_loss� A

error_RBc?

learning_rate_1?�7��bI       6%�	�|/B���A�*;


total_loss�=�@

error_R�vN?

learning_rate_1?�7pflI       6%�	E�/B���A�*;


total_loss���@

error_R��C?

learning_rate_1?�7Gم�I       6%�	|0B���A�*;


total_loss�P�@

error_R�E?

learning_rate_1?�7�׏�I       6%�	}H0B���A�*;


total_loss]]�@

error_R6ZZ?

learning_rate_1?�76�I�I       6%�	��0B���A�*;


total_loss�9�@

error_Rօ6?

learning_rate_1?�7�=/�I       6%�	@�0B���A�*;


total_lossnD�@

error_R �P?

learning_rate_1?�7פ�I       6%�	�1B���A�*;


total_loss���@

error_R�X[?

learning_rate_1?�7��MI       6%�	N1B���A�*;


total_loss�(�@

error_R�Z?

learning_rate_1?�7����I       6%�	J�1B���A�*;


total_lossa|�@

error_R��K?

learning_rate_1?�7��I       6%�	#�1B���A�*;


total_loss���@

error_R��d?

learning_rate_1?�7A$�I       6%�	 2B���A�*;


total_loss���@

error_R!�L?

learning_rate_1?�7P6�I       6%�	=n2B���A�*;


total_loss%�x@

error_R�XL?

learning_rate_1?�7�a�I       6%�	3�2B���A�*;


total_loss8�@

error_R �M?

learning_rate_1?�7�^m�I       6%�	� 3B���A�*;


total_lossO֨@

error_R �V?

learning_rate_1?�7�h)0I       6%�	�H3B���A�*;


total_loss���@

error_R2QT?

learning_rate_1?�7+��PI       6%�	�3B���A�*;


total_lossO�@

error_R4�N?

learning_rate_1?�7��t�I       6%�	��3B���A�*;


total_loss�@

error_R��C?

learning_rate_1?�7���I       6%�	�4B���A�*;


total_loss���@

error_R&�T?

learning_rate_1?�7���I       6%�	�]4B���A�*;


total_loss���@

error_R��:?

learning_rate_1?�7j	�I       6%�	��4B���A�*;


total_loss���@

error_R.�B?

learning_rate_1?�7Q�f�I       6%�	�4B���A�*;


total_loss&��@

error_R��N?

learning_rate_1?�7 RI       6%�	�$5B���A�*;


total_lossWA A

error_R�LJ?

learning_rate_1?�7�N�I       6%�	hm5B���A�*;


total_lossާ@

error_R%'R?

learning_rate_1?�7�ЩtI       6%�	J�5B���A�*;


total_loss�B�@

error_R�8>?

learning_rate_1?�7�nI       6%�	�5B���A�*;


total_loss��@

error_R�\?

learning_rate_1?�7�!�KI       6%�	a:6B���A�*;


total_loss��@

error_R)�K?

learning_rate_1?�7�t�XI       6%�	�6B���A�*;


total_lossN�@

error_R��C?

learning_rate_1?�7!�bI       6%�	c�6B���A�*;


total_loss�@

error_RW�I?

learning_rate_1?�7���SI       6%�	�7B���A�*;


total_lossO[�@

error_R��Q?

learning_rate_1?�7\ۃ�I       6%�	$k7B���A�*;


total_losslʮ@

error_Rq�S?

learning_rate_1?�7��I       6%�	ǰ7B���A�*;


total_lossv��@

error_R��H?

learning_rate_1?�7�oGwI       6%�	��7B���A�*;


total_loss$�@

error_RC�R?

learning_rate_1?�7*�q�I       6%�	M68B���A�*;


total_lossCZ�@

error_R�Q?

learning_rate_1?�7�V�[I       6%�	�x8B���A�*;


total_loss���@

error_R�Sh?

learning_rate_1?�7#ԻI       6%�	�8B���A�*;


total_loss	F�@

error_R?`?

learning_rate_1?�77��aI       6%�	� 9B���A�*;


total_loss���@

error_RRjZ?

learning_rate_1?�7�'ZI       6%�	|G9B���A�*;


total_lossϮ�@

error_ROdT?

learning_rate_1?�7[��[I       6%�	g�9B���A�*;


total_loss���@

error_Ra�g?

learning_rate_1?�7��xI       6%�	]�9B���A�*;


total_lossr�@

error_R��^?

learning_rate_1?�7ޠ�I       6%�	?:B���A�*;


total_lossg�@

error_R�S?

learning_rate_1?�7���I       6%�	�e:B���A�*;


total_loss���@

error_R�V?

learning_rate_1?�7�}��I       6%�	E�:B���A�*;


total_loss�A

error_RJ�`?

learning_rate_1?�7�v6I       6%�	��:B���A�*;


total_loss�,�@

error_R�E?

learning_rate_1?�7'��dI       6%�	�5;B���A�*;


total_loss �@

error_R 1L?

learning_rate_1?�7��҈I       6%�	w;B���A�*;


total_lossm��@

error_R!M?

learning_rate_1?�7�=ڟI       6%�	�;B���A�*;


total_lossMj�@

error_RM�R?

learning_rate_1?�7<6�oI       6%�	� <B���A�*;


total_loss!K�@

error_RC�G?

learning_rate_1?�7ƣH*I       6%�	xF<B���A�*;


total_loss��@

error_Rl*I?

learning_rate_1?�7���I       6%�	��<B���A�*;


total_loss�yA

error_R�A?

learning_rate_1?�7}<#�I       6%�	Y�<B���A�*;


total_loss���@

error_R_|8?

learning_rate_1?�7�6
�I       6%�	�=B���A�*;


total_lossVٚ@

error_R��C?

learning_rate_1?�7	u�I       6%�	�[=B���A�*;


total_loss���@

error_R�F?

learning_rate_1?�7���I       6%�	$�=B���A�*;


total_lossD�o@

error_R��M?

learning_rate_1?�7f)��I       6%�	�=B���A�*;


total_loss��@

error_RR}N?

learning_rate_1?�7�@XI       6%�	A4>B���A�*;


total_lossV@�@

error_R� T?

learning_rate_1?�7�l�I       6%�	�x>B���A�*;


total_loss\��@

error_R=";?

learning_rate_1?�7G���I       6%�	��>B���A�*;


total_lossn�@

error_Rx�V?

learning_rate_1?�7����I       6%�	"
?B���A�*;


total_loss�I�@

error_RdV?

learning_rate_1?�7g�!JI       6%�	T?B���A�*;


total_loss��@

error_R�U@?

learning_rate_1?�7����I       6%�	��?B���A�*;


total_loss�d�@

error_R
~K?

learning_rate_1?�7	"SI       6%�	v�?B���A�*;


total_loss�W�@

error_R��M?

learning_rate_1?�7a��I       6%�	�4@B���A�*;


total_loss׆�@

error_RT�??

learning_rate_1?�7�\�I       6%�	y@B���A�*;


total_losssF�@

error_R_.G?

learning_rate_1?�7����I       6%�	Ĺ@B���A�*;


total_loss���@

error_R#M?

learning_rate_1?�7�A�I       6%�	��@B���A�*;


total_loss@

error_R��8?

learning_rate_1?�7�L��I       6%�	nAAB���A�*;


total_loss��o@

error_R�@K?

learning_rate_1?�7���I       6%�	/�AB���A�*;


total_lossß�@

error_R!HT?

learning_rate_1?�7��kNI       6%�	'�AB���A�*;


total_lossƯ�@

error_R8�U?

learning_rate_1?�7$��I       6%�	�BB���A�*;


total_loss	��@

error_Ra�??

learning_rate_1?�7���I       6%�	�RBB���A�*;


total_loss�hb@

error_R��C?

learning_rate_1?�7�m|%I       6%�	a�BB���A�*;


total_loss�7A

error_R��R?

learning_rate_1?�7���I       6%�	��BB���A�*;


total_lossHP�@

error_R��R?

learning_rate_1?�7��*�I       6%�	�&CB���A�*;


total_loss��@

error_R��F?

learning_rate_1?�7Ah�I       6%�	�oCB���A�*;


total_loss��@

error_RjlM?

learning_rate_1?�7鞕�I       6%�	V�CB���A�*;


total_loss<��@

error_Rl^?

learning_rate_1?�7�a��I       6%�	��CB���A�*;


total_loss��@

error_R�vS?

learning_rate_1?�7�N�RI       6%�	�DDB���A�*;


total_loss���@

error_R��I?

learning_rate_1?�7Z���I       6%�	��DB���A�*;


total_loss��@

error_R��P?

learning_rate_1?�7�2��I       6%�	��DB���A�*;


total_loss�u�@

error_R��\?

learning_rate_1?�7�I       6%�	]EB���A�*;


total_loss���@

error_R��E?

learning_rate_1?�7��BKI       6%�	�iEB���A�*;


total_lossxp�@

error_RS~P?

learning_rate_1?�7�}I       6%�	��EB���A�*;


total_loss���@

error_R)�@?

learning_rate_1?�7oz�I       6%�	�EB���A�*;


total_loss��@

error_R�E?

learning_rate_1?�7����I       6%�	�@FB���A�*;


total_lossCv@

error_R��S?

learning_rate_1?�7z�n�I       6%�	Y�FB���A�*;


total_lossx|�@

error_R�S?

learning_rate_1?�7��,�I       6%�	y�FB���A�*;


total_loss$��@

error_R�Q?

learning_rate_1?�7H�6I       6%�	�$GB���A�*;


total_lossL��@

error_R�T?

learning_rate_1?�7��!FI       6%�	�wGB���A�*;


total_lossIظ@

error_R�M?

learning_rate_1?�7;�I       6%�	4�GB���A�*;


total_loss�Y�@

error_R�uE?

learning_rate_1?�7��L�I       6%�	vHB���A�*;


total_loss̪�@

error_R�S?

learning_rate_1?�7���I       6%�	�FHB���A�*;


total_loss��@

error_R�8?

learning_rate_1?�7��oI       6%�	�HB���A�*;


total_loss���@

error_Rw�A?

learning_rate_1?�7���I       6%�	��HB���A�*;


total_lossVe�@

error_R$�W?

learning_rate_1?�7��qI       6%�	3IB���A�*;


total_loss�a�@

error_R� Y?

learning_rate_1?�7��I       6%�	&YIB���A�*;


total_loss��@

error_R&U?

learning_rate_1?�7� �I       6%�	ߣIB���A�*;


total_lossN�@

error_Rs<??

learning_rate_1?�7���VI       6%�	��IB���A�*;


total_loss?��@

error_R��Y?

learning_rate_1?�7^ƅI       6%�	K1JB���A�*;


total_loss���@

error_R�V?

learning_rate_1?�7
lZ�I       6%�	[uJB���A�*;


total_loss���@

error_R4D?

learning_rate_1?�7�l��I       6%�	�JB���A�*;


total_loss�R�@

error_R�2L?

learning_rate_1?�7���`I       6%�	O�JB���A�*;


total_loss�ؒ@

error_R8�P?

learning_rate_1?�7+3��I       6%�	�BKB���A�*;


total_loss��
A

error_R0[?

learning_rate_1?�7��Z@I       6%�	ٜKB���A�*;


total_loss@�@

error_R�I?

learning_rate_1?�7����I       6%�	��KB���A�*;


total_lossTh�@

error_R�N?

learning_rate_1?�7�v�YI       6%�	^0LB���A�*;


total_lossE��@

error_RȔH?

learning_rate_1?�7��H�I       6%�	&�LB���A�*;


total_loss���@

error_RN�>?

learning_rate_1?�7���>I       6%�	��LB���A�*;


total_loss��@

error_Rd�L?

learning_rate_1?�7����I       6%�	QMB���A�*;


total_loss���@

error_R�D?

learning_rate_1?�7�!�I       6%�	�\MB���A�*;


total_loss��@

error_R��A?

learning_rate_1?�73A�I       6%�	.�MB���A�*;


total_loss8��@

error_Rт\?

learning_rate_1?�7	��yI       6%�	��MB���A�*;


total_loss�@

error_R�ON?

learning_rate_1?�7#�7MI       6%�	�-NB���A�*;


total_loss���@

error_RlwF?

learning_rate_1?�7nY�iI       6%�	�rNB���A�*;


total_lossn��@

error_R��B?

learning_rate_1?�7AUv�I       6%�	ڴNB���A�*;


total_loss|K�@

error_Ra�F?

learning_rate_1?�7��TcI       6%�	��NB���A�*;


total_loss8v�@

error_R��R?

learning_rate_1?�70���I       6%�	h8OB���A�*;


total_lossY�@

error_R�lW?

learning_rate_1?�7(�~�I       6%�	E�OB���A�*;


total_loss���@

error_R��??

learning_rate_1?�7_;(I       6%�	"�OB���A�*;


total_loss|��@

error_R��F?

learning_rate_1?�7f))I       6%�	PB���A�*;


total_loss�t@

error_R�V?

learning_rate_1?�7;$��I       6%�	7UPB���A�*;


total_losseԎ@

error_RQ9P?

learning_rate_1?�7̱I       6%�	��PB���A�*;


total_loss�2�@

error_Rf�P?

learning_rate_1?�7�H,I       6%�	��PB���A�*;


total_lossHA�@

error_RݖV?

learning_rate_1?�7��|EI       6%�	(QB���A�*;


total_loss���@

error_RD�J?

learning_rate_1?�7�A�I       6%�	0qQB���A�*;


total_loss܉�@

error_RC�Q?

learning_rate_1?�7�lY�I       6%�	\�QB���A�*;


total_loss��@

error_R�I?

learning_rate_1?�7��	I       6%�	tRB���A�*;


total_lossi��@

error_Rj�e?

learning_rate_1?�7:�I       6%�	�MRB���A�*;


total_loss�g�@

error_Rf�G?

learning_rate_1?�7�B�I       6%�	͘RB���A�*;


total_loss$��@

error_R�%b?

learning_rate_1?�7����I       6%�	�RB���A�*;


total_loss���@

error_R��H?

learning_rate_1?�7����I       6%�	%SB���A�*;


total_lossH,�@

error_R�R?

learning_rate_1?�7
�NWI       6%�	}pSB���A�*;


total_loss�%�@

error_R�D?

learning_rate_1?�7�}I       6%�	F�SB���A�*;


total_loss���@

error_Rs=?

learning_rate_1?�7��I       6%�	�	TB���A�*;


total_loss@a�@

error_R�"@?

learning_rate_1?�7ht�I       6%�	JQTB���A�*;


total_loss�g@

error_R��I?

learning_rate_1?�7$'��I       6%�	ٙTB���A�*;


total_loss�T�@

error_R,VH?

learning_rate_1?�7,�ȇI       6%�	��TB���A�*;


total_loss���@

error_R�~@?

learning_rate_1?�7�n�I       6%�	�)UB���A�*;


total_loss q�@

error_R\�O?

learning_rate_1?�7d:b�I       6%�	�sUB���A�*;


total_loss_j�@

error_R�G\?

learning_rate_1?�7�h��I       6%�	�UB���A�*;


total_losss(A

error_RE�J?

learning_rate_1?�7�1�I       6%�	��UB���A�*;


total_loss�@

error_RN�H?

learning_rate_1?�7Ӟ�!I       6%�	GHVB���A�*;


total_loss3�@

error_R,k?

learning_rate_1?�7
��I       6%�	��VB���A�*;


total_loss�ud@

error_RnW?

learning_rate_1?�7�B�I       6%�	��VB���A�*;


total_loss�c�@

error_R��V?

learning_rate_1?�7���kI       6%�	WB���A�*;


total_lossV~�@

error_R�yT?

learning_rate_1?�7.��]I       6%�	�xWB���A�*;


total_lossʛ@

error_R��D?

learning_rate_1?�7CE/�I       6%�	�WB���A�*;


total_loss�DA

error_R��H?

learning_rate_1?�7���<I       6%�	�WB���A�*;


total_lossb�@

error_R�I?

learning_rate_1?�7���jI       6%�	FAXB���A�*;


total_loss���@

error_R�5?

learning_rate_1?�7���'I       6%�	K�XB���A�*;


total_loss-~�@

error_R��E?

learning_rate_1?�7���I       6%�	��XB���A�*;


total_lossfH�@

error_R��h?

learning_rate_1?�7�
QI       6%�	�YB���A�*;


total_loss��A

error_R��Y?

learning_rate_1?�7�-QI       6%�	�NYB���A�*;


total_loss|�@

error_R��E?

learning_rate_1?�7	�}I       6%�	ԏYB���A�*;


total_lossڃ�@

error_R��J?

learning_rate_1?�7~��I       6%�	��YB���A�*;


total_lossiJ�@

error_R�U?

learning_rate_1?�7;JdI       6%�	4$ZB���A�*;


total_loss(��@

error_R֠I?

learning_rate_1?�7�JR I       6%�	�mZB���A�*;


total_loss,R�@

error_RO?

learning_rate_1?�7�6�I       6%�	1�ZB���A�*;


total_loss���@

error_R�cR?

learning_rate_1?�7�A�I       6%�	w [B���A�*;


total_loss3я@

error_Rs4B?

learning_rate_1?�7��	�I       6%�	H[B���A�*;


total_loss���@

error_R=�i?

learning_rate_1?�7Ԑw�I       6%�	��[B���A�*;


total_loss%A

error_R�Q?

learning_rate_1?�7	��DI       6%�	��[B���A�*;


total_loss���@

error_R�PR?

learning_rate_1?�7Z%HI       6%�	=!\B���A�*;


total_loss��@

error_R&HQ?

learning_rate_1?�70HtMI       6%�	�c\B���A�*;


total_loss/��@

error_R3�T?

learning_rate_1?�7�!OjI       6%�	/�\B���A�*;


total_loss�uA

error_R��E?

learning_rate_1?�7/�"VI       6%�	�\B���A�*;


total_losse7A

error_R�P?

learning_rate_1?�7.�I       6%�	�4]B���A�*;


total_lossD�@

error_R�xN?

learning_rate_1?�7Dm�dI       6%�	!�]B���A�*;


total_loss���@

error_Rx[M?

learning_rate_1?�7�+�I       6%�	��]B���A�*;


total_loss$��@

error_R�_?

learning_rate_1?�7���>I       6%�	m^B���A�*;


total_loss�#�@

error_R�:?

learning_rate_1?�7,�8I       6%�	�U^B���A�*;


total_losso~q@

error_R�%M?

learning_rate_1?�7V���I       6%�	ٟ^B���A�*;


total_loss*v�@

error_R�tV?

learning_rate_1?�7�۵�I       6%�	��^B���A�*;


total_loss鮾@

error_Re�O?

learning_rate_1?�7T���I       6%�	�(_B���A�*;


total_loss.�@

error_R;pI?

learning_rate_1?�7�|�gI       6%�	�l_B���A�*;


total_loss�L�@

error_R�U?

learning_rate_1?�7˰�I       6%�	o�_B���A�*;


total_loss�Zn@

error_R�uK?

learning_rate_1?�7EWBkI       6%�	��_B���A�*;


total_loss�	�@

error_R�rJ?

learning_rate_1?�7��9�I       6%�	�@`B���A�*;


total_lossT�@

error_R��T?

learning_rate_1?�7��GNI       6%�	#�`B���A�*;


total_loss���@

error_Rq�J?

learning_rate_1?�7]!d�I       6%�	��`B���A�*;


total_lossb�@

error_RɰE?

learning_rate_1?�7U�P�I       6%�	�aB���A�*;


total_loss�:�@

error_R�i<?

learning_rate_1?�7���I       6%�	�[aB���A�*;


total_loss-��@

error_RωB?

learning_rate_1?�7�K(I       6%�	�aB���A�*;


total_loss�8�@

error_R��=?

learning_rate_1?�77�%I       6%�	�aB���A�*;


total_loss<�@

error_R�NX?

learning_rate_1?�7��API       6%�	�"bB���A�*;


total_loss�L�@

error_R�mJ?

learning_rate_1?�7|��I       6%�	fbB���A�*;


total_loss�f�@

error_R�lT?

learning_rate_1?�7� +I       6%�	ƭbB���A�*;


total_loss��@

error_RT?

learning_rate_1?�7�=��I       6%�	Z�bB���A�*;


total_lossȐ@

error_RRE?

learning_rate_1?�7#|v�I       6%�	�AcB���A�*;


total_loss ��@

error_R,�`?

learning_rate_1?�7����I       6%�	��cB���A�*;


total_loss7�A

error_R�eF?

learning_rate_1?�7A��,I       6%�	��cB���A�*;


total_loss��@

error_RvU?

learning_rate_1?�7�'��I       6%�	�dB���A�*;


total_loss���@

error_Rc�W?

learning_rate_1?�7v�_jI       6%�	�TdB���A�*;


total_lossi�|@

error_RqH?

learning_rate_1?�73�I       6%�	<�dB���A�*;


total_loss;�@

error_Ri�Q?

learning_rate_1?�7_I       6%�	��dB���A�*;


total_loss!�@

error_R�aC?

learning_rate_1?�7���II       6%�	,eB���A�*;


total_loss�(A

error_R�[P?

learning_rate_1?�7�'�I       6%�	�qeB���A�*;


total_loss䨃@

error_R�^R?

learning_rate_1?�7T)��I       6%�	��eB���A�*;


total_loss#�@

error_R��V?

learning_rate_1?�7��wI       6%�	n�eB���A�*;


total_loss_�A

error_R�.M?

learning_rate_1?�7v6�I       6%�	�7fB���A�*;


total_loss���@

error_R�-Y?

learning_rate_1?�7�,��I       6%�	D}fB���A�*;


total_loss��@

error_RŐJ?

learning_rate_1?�7p=�~I       6%�	�fB���A�*;


total_loss�@

error_R��O?

learning_rate_1?�7U��I       6%�	�gB���A�*;


total_loss,�@

error_R$ U?

learning_rate_1?�7�a�I       6%�	�kgB���A�*;


total_loss�*�@

error_R�PO?

learning_rate_1?�7�!��I       6%�	�gB���A�*;


total_loss�И@

error_R�d?

learning_rate_1?�7�-xI       6%�	��gB���A�*;


total_losssm�@

error_R�*U?

learning_rate_1?�7�w�I       6%�	`7hB���A�*;


total_lossɶ@

error_R�OO?

learning_rate_1?�7g�I       6%�	�xhB���A�*;


total_loss5f�@

error_R@RA?

learning_rate_1?�7� ��I       6%�	վhB���A�*;


total_loss��@

error_R�L?

learning_rate_1?�7�*6�I       6%�	� iB���A�*;


total_loss�ǂ@

error_R��a?

learning_rate_1?�7��qDI       6%�	*EiB���A�*;


total_loss�}�@

error_R��W?

learning_rate_1?�7�Vm�I       6%�	F�iB���A�*;


total_loss�'�@

error_R��U?

learning_rate_1?�7����I       6%�	��iB���A�*;


total_loss_�@

error_Rd,K?

learning_rate_1?�7���I       6%�	�!jB���A�*;


total_loss�+�@

error_RצA?

learning_rate_1?�7-ĳfI       6%�	�ejB���A�*;


total_loss�
�@

error_R�]F?

learning_rate_1?�7<�D�I       6%�	�jB���A�*;


total_lossC�@

error_R �U?

learning_rate_1?�7<�N�I       6%�	�jB���A�*;


total_loss�}�@

error_R܌I?

learning_rate_1?�7�bI       6%�	-3kB���A�*;


total_lossXJ�@

error_R�mD?

learning_rate_1?�70S4RI       6%�	5�kB���A�*;


total_lossmk�@

error_R-V?

learning_rate_1?�7�3WI       6%�	�kB���A�*;


total_loss(޲@

error_RקV?

learning_rate_1?�7�pXI       6%�	�lB���A�*;


total_loss4/�@

error_R~K?

learning_rate_1?�7;3�I       6%�	�TlB���A�*;


total_loss�?�@

error_R
�L?

learning_rate_1?�7`��I       6%�	ʼlB���A�*;


total_loss �@

error_R=�Q?

learning_rate_1?�7o��WI       6%�	YmB���A�*;


total_loss���@

error_R��R?

learning_rate_1?�7E��I       6%�	�SmB���A�*;


total_loss��@

error_R*:Q?

learning_rate_1?�7��CI       6%�	ӝmB���A�*;


total_lossl��@

error_R��X?

learning_rate_1?�7f��I       6%�	�mB���A�*;


total_loss���@

error_Rdb?

learning_rate_1?�7���I       6%�	�,nB���A�*;


total_loss)�@

error_RM�I?

learning_rate_1?�7Q�U�I       6%�	9snB���A�*;


total_loss�X�@

error_R�sJ?

learning_rate_1?�7�`��I       6%�	ܷnB���A�*;


total_losst�@

error_R.�U?

learning_rate_1?�7�NA8I       6%�	A�nB���A�*;


total_loss�i�@

error_R�]?

learning_rate_1?�7��I       6%�	2EoB���A�*;


total_loss[�@

error_R=�X?

learning_rate_1?�75y�AI       6%�	ňoB���A�*;


total_loss�r�@

error_R�K?

learning_rate_1?�7�o1I       6%�	Y�oB���A�*;


total_loss���@

error_RXuS?

learning_rate_1?�7"AɮI       6%�	pB���A�*;


total_loss㺸@

error_R�V[?

learning_rate_1?�7A6��I       6%�	NVpB���A�*;


total_loss���@

error_R�wL?

learning_rate_1?�7ޔiZI       6%�	��pB���A�*;


total_loss̸�@

error_RE�@?

learning_rate_1?�7Ǝ2,I       6%�	1�pB���A�*;


total_loss��@

error_RƆA?

learning_rate_1?�7�z�QI       6%�	�qB���A�*;


total_loss]��@

error_R�~R?

learning_rate_1?�7~�<�I       6%�	^bqB���A�*;


total_loss@

error_R}G]?

learning_rate_1?�7�!K4I       6%�	��qB���A�*;


total_loss@�@

error_RRDO?

learning_rate_1?�7	A�FI       6%�	F�qB���A�*;


total_lossl��@

error_Rs_=?

learning_rate_1?�7���I       6%�	�7rB���A�*;


total_loss�O�@

error_R�T?

learning_rate_1?�7�[߄I       6%�	.}rB���A�*;


total_loss��@

error_R?m>?

learning_rate_1?�7 9I       6%�	0�rB���A�*;


total_loss_��@

error_R�H?

learning_rate_1?�7ޚ �I       6%�	�sB���A�*;


total_loss<�@

error_RRlL?

learning_rate_1?�7n�ٺI       6%�	uHsB���A�*;


total_lossVL�@

error_R�dR?

learning_rate_1?�7V���I       6%�	�sB���A�*;


total_loss�N�@

error_R �=?

learning_rate_1?�7����I       6%�	!�sB���A�*;


total_loss�{�@

error_R8�Q?

learning_rate_1?�7iۑ�I       6%�	tB���A�*;


total_lossX��@

error_R�O?

learning_rate_1?�7�*��I       6%�	�UtB���A�*;


total_loss;��@

error_R�6\?

learning_rate_1?�7g�bI       6%�	�tB���A�*;


total_loss�A

error_R��O?

learning_rate_1?�7k�YEI       6%�	�tB���A�*;


total_loss ��@

error_R�i@?

learning_rate_1?�7���I       6%�	�$uB���A�*;


total_loss���@

error_R�.Z?

learning_rate_1?�7�ۜuI       6%�	TiuB���A�*;


total_loss���@

error_R=XK?

learning_rate_1?�7�5�I       6%�	a�uB���A�*;


total_lossw'�@

error_Ro�W?

learning_rate_1?�7��>�I       6%�	��uB���A�*;


total_lossS��@

error_R�F?

learning_rate_1?�7��ܜI       6%�	.vB���A�*;


total_loss[�@

error_R�zf?

learning_rate_1?�7���I       6%�	�svB���A�*;


total_loss��@

error_R�c?

learning_rate_1?�7�9PI       6%�	��vB���A�*;


total_loss��@

error_R��i?

learning_rate_1?�7Ť<�I       6%�	PwB���A�*;


total_loss �@

error_R�hY?

learning_rate_1?�7o�sI       6%�	�]wB���A�*;


total_loss���@

error_R�;?

learning_rate_1?�7�C�,I       6%�	@�wB���A�*;


total_loss㯶@

error_RT?

learning_rate_1?�7��Y�I       6%�	)�wB���A�*;


total_lossn��@

error_R-K?

learning_rate_1?�7��%�I       6%�	5xB���A�*;


total_lossI��@

error_R|S?

learning_rate_1?�7�ZI       6%�	�xxB���A�*;


total_lossN��@

error_ReT?

learning_rate_1?�7.9�@I       6%�	
�xB���A�*;


total_loss�
�@

error_R�	G?

learning_rate_1?�7� �I       6%�	�yB���A�*;


total_loss��@

error_RH�]?

learning_rate_1?�7{i1I       6%�	HyB���A�*;


total_loss��@

error_R�K?

learning_rate_1?�7��C�I       6%�	C�yB���A�*;


total_lossD�@

error_R`F?

learning_rate_1?�7~��qI       6%�	��yB���A�*;


total_loss{A�@

error_R�`?

learning_rate_1?�7C?�I       6%�	� zB���A�*;


total_loss�1�@

error_R��A?

learning_rate_1?�7R֭�I       6%�	�hzB���A�*;


total_loss\��@

error_Rd�9?

learning_rate_1?�7Z��I       6%�	�zB���A�*;


total_loss@

error_RA�;?

learning_rate_1?�7�Y�I       6%�	@�zB���A�*;


total_loss�?A

error_RMb??

learning_rate_1?�7�t�aI       6%�	�6{B���A�*;


total_loss��@

error_R��F?

learning_rate_1?�7`E��I       6%�	=|{B���A�*;


total_loss�¿@

error_R��T?

learning_rate_1?�7D�WI       6%�	�{B���A�*;


total_loss7K�@

error_R,F?

learning_rate_1?�7���I       6%�	a
|B���A�*;


total_loss�I�@

error_R�}R?

learning_rate_1?�7���I       6%�	�L|B���A�*;


total_loss\6�@

error_RJ�I?

learning_rate_1?�7,�?I       6%�	�|B���A�*;


total_lossR��@

error_R��M?

learning_rate_1?�7MN�I       6%�	5�|B���A�*;


total_loss2K�@

error_R_Z?

learning_rate_1?�7B���I       6%�	}B���A�*;


total_lossr@�@

error_R�9V?

learning_rate_1?�7t�;CI       6%�	�a}B���A�*;


total_loss��	A

error_RTK?

learning_rate_1?�7TL̞I       6%�	�}B���A�*;


total_loss��	A

error_Ri4Y?

learning_rate_1?�7
LJ�I       6%�	,�}B���A�*;


total_loss �@

error_R��<?

learning_rate_1?�7s��I       6%�	�8~B���A�*;


total_loss��A

error_R��I?

learning_rate_1?�7N6�I       6%�	�{~B���A�*;


total_loss�a�@

error_R�>L?

learning_rate_1?�7D#I       6%�	�~B���A�*;


total_loss@�@

error_R�G?

learning_rate_1?�7�@	I       6%�	�B���A�*;


total_lossM��@

error_R��I?

learning_rate_1?�7���I       6%�	�EB���A�*;


total_loss�;�@

error_R�>[?

learning_rate_1?�7υK�I       6%�	��B���A�*;


total_loss ��@

error_R��<?

learning_rate_1?�7αA�I       6%�	[�B���A�*;


total_loss�2�@

error_R��L?

learning_rate_1?�7���9I       6%�	��B���A�*;


total_lossq_�@

error_RvBJ?

learning_rate_1?�7c4*OI       6%�	4d�B���A�*;


total_loss��@

error_R��P?

learning_rate_1?�7�z|I       6%�	ƥ�B���A�*;


total_lossL�@

error_R�I?

learning_rate_1?�7q(�+I       6%�	'�B���A�*;


total_loss���@

error_R)M?

learning_rate_1?�72Y��I       6%�	<0�B���A�*;


total_loss�۽@

error_RsE?

learning_rate_1?�7{�]�I       6%�	 s�B���A�*;


total_lossa��@

error_R��R?

learning_rate_1?�7~�O�I       6%�	���B���A�*;


total_loss�q�@

error_Rtm\?

learning_rate_1?�73%��I       6%�	� �B���A�*;


total_loss:��@

error_R�VH?

learning_rate_1?�7�|=)I       6%�	�J�B���A�*;


total_loss1J�@

error_R��\?

learning_rate_1?�7JKO�I       6%�	H��B���A�*;


total_loss��@

error_R_�[?

learning_rate_1?�7xR�I       6%�	قB���A�*;


total_lossF�@

error_R�nO?

learning_rate_1?�7���	I       6%�	��B���A�*;


total_loss���@

error_R��O?

learning_rate_1?�7�i�UI       6%�	Cj�B���A�*;


total_lossso�@

error_R
(W?

learning_rate_1?�7t��I       6%�	g��B���A�*;


total_loss���@

error_R��J?

learning_rate_1?�7!��I       6%�	)��B���A�*;


total_loss��@

error_R�	J?

learning_rate_1?�7t��_I       6%�	GB�B���A�*;


total_loss�y�@

error_R$�R?

learning_rate_1?�7e7 �I       6%�	b��B���A�*;


total_loss2��@

error_R$�\?

learning_rate_1?�7>E��I       6%�	�ƄB���A�*;


total_loss�@

error_R��V?

learning_rate_1?�7z�QnI       6%�	��B���A�*;


total_loss��@

error_R�Y?

learning_rate_1?�7�2D�I       6%�	�K�B���A�*;


total_loss/�@

error_R�4V?

learning_rate_1?�7��� I       6%�	���B���A�*;


total_loss�@

error_R׮d?

learning_rate_1?�7Wl�4I       6%�	ׅB���A�*;


total_lossv��@

error_Rf&L?

learning_rate_1?�7�>�tI       6%�	��B���A�*;


total_losss�@

error_RJ�C?

learning_rate_1?�7)�\I       6%�	�_�B���A�*;


total_loss��@

error_R��W?

learning_rate_1?�7����I       6%�	���B���A�*;


total_loss(�@

error_R�tF?

learning_rate_1?�7�sŃI       6%�	��B���A�*;


total_lossx�@

error_R}�??

learning_rate_1?�7pb<I       6%�	iI�B���A�*;


total_loss�2�@

error_R!,I?

learning_rate_1?�7�nWNI       6%�	`��B���A�*;


total_loss�~�@

error_R�D?

learning_rate_1?�7�#�0I       6%�	�ԇB���A�*;


total_loss�t�@

error_R�_?

learning_rate_1?�7Z���I       6%�	}�B���A�*;


total_loss�Ֆ@

error_R�R?

learning_rate_1?�7�z|I       6%�	�\�B���A�*;


total_loss� X@

error_R��F?

learning_rate_1?�7,ݾ�I       6%�	+��B���A�*;


total_loss��@

error_RR�E?

learning_rate_1?�7%I'I       6%�	)�B���A�*;


total_lossћ�@

error_R��U?

learning_rate_1?�7�9I�I       6%�	7�B���A�*;


total_lossc��@

error_RF�Z?

learning_rate_1?�7wS�I       6%�	+{�B���A�*;


total_loss6Z�@

error_R��P?

learning_rate_1?�7ԆhDI       6%�	�ǉB���A�*;


total_losseF�@

error_R�O?

learning_rate_1?�7�U�I       6%�	e�B���A�*;


total_loss̢W@

error_R]�N?

learning_rate_1?�7Lup�I       6%�	�R�B���A�*;


total_losss��@

error_RdGJ?

learning_rate_1?�7'�I       6%�	���B���A�*;


total_loss���@

error_R�H?

learning_rate_1?�7��9�I       6%�	�B���A�*;


total_loss��@

error_RS?

learning_rate_1?�7a�YI       6%�	�,�B���A�*;


total_loss/(�@

error_R�g??

learning_rate_1?�7��I       6%�	��B���A�*;


total_lossT��@

error_R�pJ?

learning_rate_1?�7�$*6I       6%�	�̋B���A�*;


total_loss� �@

error_R@O?

learning_rate_1?�7��7I       6%�	$�B���A�*;


total_lossX,�@

error_R)�@?

learning_rate_1?�7�ҐI       6%�	q\�B���A�*;


total_loss�p�@

error_R��X?

learning_rate_1?�7��w�I       6%�	���B���A�*;


total_loss!��@

error_RLK?

learning_rate_1?�7T��OI       6%�	e�B���A�*;


total_loss��@

error_R/�J?

learning_rate_1?�7y��lI       6%�	}L�B���A�*;


total_lossh��@

error_R\Fd?

learning_rate_1?�7f(I       6%�	ݑ�B���A�*;


total_loss1Ԅ@

error_R�???

learning_rate_1?�7w��I       6%�	�ԍB���A�*;


total_lossJZp@

error_R�\?

learning_rate_1?�7���I       6%�	��B���A�*;


total_loss��A

error_R�K?

learning_rate_1?�7Dh'�I       6%�	�b�B���A�*;


total_loss]̆@

error_RF�J?

learning_rate_1?�7����I       6%�	���B���A�*;


total_loss�<�@

error_R.RX?

learning_rate_1?�7:���I       6%�	���B���A�*;


total_loss� �@

error_R�yQ?

learning_rate_1?�7��I       6%�	>;�B���A�*;


total_lossc �@

error_R�KD?

learning_rate_1?�7ǡ��I       6%�	��B���A�*;


total_lossIA

error_R
KU?

learning_rate_1?�7�Go}I       6%�	��B���A�*;


total_loss��@

error_Ra�T?

learning_rate_1?�7U��I       6%�	�B���A�*;


total_loss���@

error_RW�L?

learning_rate_1?�7G�t,I       6%�	'J�B���A�*;


total_loss�F�@

error_R;�_?

learning_rate_1?�7�n�+I       6%�	a��B���A�*;


total_loss
N�@

error_Rr�V?

learning_rate_1?�7�.I       6%�	֐B���A�*;


total_lossN��@

error_RC,L?

learning_rate_1?�7覬MI       6%�	C�B���A�*;


total_loss��@

error_R��I?

learning_rate_1?�7�J�]I       6%�	C]�B���A�*;


total_lossYP�@

error_Rŏd?

learning_rate_1?�71�I       6%�	���B���A�*;


total_loss;��@

error_Rr�8?

learning_rate_1?�7t���I       6%�	��B���A�*;


total_lossEt�@

error_R��E?

learning_rate_1?�70
ܗI       6%�	�,�B���A�*;


total_loss\ �@

error_R{�O?

learning_rate_1?�7P�B\I       6%�	�q�B���A�*;


total_loss ��@

error_R�zT?

learning_rate_1?�7�+V�I       6%�	���B���A�*;


total_loss���@

error_R6�i?

learning_rate_1?�7d��I       6%�	���B���A�*;


total_loss=�@

error_R$=?

learning_rate_1?�7F���I       6%�	�7�B���A�*;


total_loss�p�@

error_R}-g?

learning_rate_1?�7�S��I       6%�	�y�B���A�*;


total_loss��@

error_R��N?

learning_rate_1?�7ȸ��I       6%�	R��B���A�*;


total_loss�EA

error_R�*=?

learning_rate_1?�7��SI       6%�	���B���A�*;


total_loss�x@

error_ROP?

learning_rate_1?�7��6I       6%�	J@�B���A�*;


total_loss��@

error_Ri�R?

learning_rate_1?�7EtLI       6%�	!��B���A�*;


total_loss{�@

error_R�Q?

learning_rate_1?�75�ȘI       6%�	�ΔB���A�*;


total_lossNm�@

error_R��R?

learning_rate_1?�7:�'CI       6%�	p�B���A�*;


total_loss!��@

error_R	�V?

learning_rate_1?�7Y .jI       6%�	/W�B���A�*;


total_lossň�@

error_R�IC?

learning_rate_1?�7�.1I       6%�	G��B���A�*;


total_lossW^�@

error_R8�O?

learning_rate_1?�7��II       6%�	�ߕB���A�*;


total_loss�z�@

error_R
,N?

learning_rate_1?�7O�#�I       6%�	-$�B���A�*;


total_loss\�@

error_R]jK?

learning_rate_1?�7�ln:I       6%�	�f�B���A�*;


total_loss?�@

error_R$`?

learning_rate_1?�7�:�I       6%�	�B���A�*;


total_loss)��@

error_R�L?

learning_rate_1?�7��
I       6%�	<��B���A�*;


total_lossf��@

error_R�T?

learning_rate_1?�7Ս3�I       6%�	�S�B���A�*;


total_losscݝ@

error_R$%M?

learning_rate_1?�7�/iI       6%�	c��B���A�*;


total_lossĕ�@

error_R�U?

learning_rate_1?�7�|�I       6%�	#�B���A�*;


total_loss���@

error_R��d?

learning_rate_1?�7e�I       6%�	�0�B���A�*;


total_loss��@

error_RS�H?

learning_rate_1?�7��=�I       6%�	cx�B���A�*;


total_loss �@

error_R��Q?

learning_rate_1?�7w��I       6%�	���B���A�*;


total_loss��@

error_R.FI?

learning_rate_1?�7���I       6%�	� �B���A�*;


total_loss$W�@

error_R1�A?

learning_rate_1?�7���I       6%�	!D�B���A�*;


total_loss6M�@

error_RlS?

learning_rate_1?�7lnF8I       6%�	M��B���A�*;


total_lossfZ�@

error_RIO?

learning_rate_1?�7�(7�I       6%�	nʙB���A�*;


total_loss�-�@

error_RA�O?

learning_rate_1?�7�=I       6%�	��B���A�*;


total_loss��@

error_R��-?

learning_rate_1?�7���I       6%�	O�B���A�*;


total_loss�Ǚ@

error_R�J?

learning_rate_1?�7��iI       6%�	���B���A�*;


total_loss���@

error_R��[?

learning_rate_1?�7�9�I       6%�	�ҚB���A�*;


total_loss���@

error_RےC?

learning_rate_1?�7Y��I       6%�	;�B���A�*;


total_loss�u�@

error_R! ]?

learning_rate_1?�7	s��I       6%�	e]�B���A�*;


total_lossߚ�@

error_R|�I?

learning_rate_1?�7UCI       6%�	���B���A�*;


total_loss���@

error_R8�g?

learning_rate_1?�7yO�'I       6%�	��B���A�*;


total_loss��A

error_RΥ8?

learning_rate_1?�7qM�I       6%�	�)�B���A�*;


total_lossAָ@

error_R�L?

learning_rate_1?�7�z��I       6%�	�j�B���A�*;


total_loss E�@

error_R��P?

learning_rate_1?�7�t�I       6%�	&��B���A�*;


total_lossS��@

error_RR�D?

learning_rate_1?�7�!CDI       6%�	2��B���A�*;


total_lossZ�@

error_R[�R?

learning_rate_1?�7�h��I       6%�	?A�B���A�*;


total_loss���@

error_R�}V?

learning_rate_1?�7w��CI       6%�	���B���A�*;


total_loss��@

error_Rdz[?

learning_rate_1?�7(���I       6%�	�ѝB���A�*;


total_loss���@

error_Ri�J?

learning_rate_1?�7��I       6%�	��B���A�*;


total_loss�$�@

error_R�9B?

learning_rate_1?�7JK�}I       6%�	h�B���A�*;


total_lossL��@

error_R�3V?

learning_rate_1?�7�z��I       6%�	��B���A�*;


total_loss�zA

error_R-Q?

learning_rate_1?�7�3maI       6%�	d��B���A�*;


total_lossd�@

error_R��<?

learning_rate_1?�7�*�QI       6%�	$=�B���A�*;


total_loss�'�@

error_RF	U?

learning_rate_1?�7)lUI       6%�	���B���A�*;


total_loss�#�@

error_R6(W?

learning_rate_1?�7�:��I       6%�	hşB���A�*;


total_loss�0�@

error_RL�T?

learning_rate_1?�7j�QI       6%�	K	�B���A�*;


total_lossJ�@

error_R�G?

learning_rate_1?�7�ݿ7I       6%�	�L�B���A�*;


total_loss\@A

error_R�@W?

learning_rate_1?�7�<��I       6%�	���B���A�*;


total_loss��@

error_RN>i?

learning_rate_1?�7/߈QI       6%�	�ҠB���A�*;


total_lossVw@

error_R$�C?

learning_rate_1?�7�h%ZI       6%�		�B���A�*;


total_lossN7�@

error_RȟR?

learning_rate_1?�7k}��I       6%�	�X�B���A�*;


total_loss?�@

error_R��5?

learning_rate_1?�7S��I       6%�	��B���A�*;


total_loss<-�@

error_R�fF?

learning_rate_1?�7����I       6%�	sݡB���A�*;


total_lossz�@

error_R�[?

learning_rate_1?�7m���I       6%�	�B���A�*;


total_loss�;�@

error_R�M?

learning_rate_1?�7mCI       6%�	�c�B���A�*;


total_loss���@

error_R�fJ?

learning_rate_1?�7m���I       6%�	���B���A�*;


total_loss\�@

error_R
�\?

learning_rate_1?�7Y��I       6%�	��B���A�*;


total_loss���@

error_Rz�P?

learning_rate_1?�7��%�I       6%�	�0�B���A�*;


total_loss���@

error_R#�V?

learning_rate_1?�7���SI       6%�	�u�B���A�*;


total_lossTi�@

error_R<d?

learning_rate_1?�7�8� I       6%�	���B���A�*;


total_loss�ͺ@

error_R�D?

learning_rate_1?�7| j�I       6%�	���B���A�*;


total_loss�"h@

error_RR	C?

learning_rate_1?�7l� I       6%�	�A�B���A�*;


total_loss��h@

error_R7�@?

learning_rate_1?�7z2I       6%�	1��B���A�*;


total_lossm[BA

error_R�K?

learning_rate_1?�7��A�I       6%�	hȤB���A�*;


total_lossʎ�@

error_RA�N?

learning_rate_1?�7���TI       6%�	��B���A�*;


total_loss�J�@

error_R�P?

learning_rate_1?�7Dj��I       6%�	JM�B���A�*;


total_loss���@

error_R�F^?

learning_rate_1?�7G;��I       6%�	N��B���A�*;


total_loss���@

error_R3�d?

learning_rate_1?�7�$�I       6%�	vѥB���A�*;


total_loss���@

error_R�)R?

learning_rate_1?�7���WI       6%�	��B���A�*;


total_lossqvA

error_R�R?

learning_rate_1?�7bn-I       6%�	|X�B���A�*;


total_loss�T�@

error_R��N?

learning_rate_1?�7���I       6%�	)��B���A�*;


total_loss��@

error_R�b]?

learning_rate_1?�7V�~@I       6%�	ަB���A�*;


total_losst�@

error_ROuI?

learning_rate_1?�7Ӱ�I       6%�	�2�B���A�*;


total_loss��A

error_R|�K?

learning_rate_1?�7�\�eI       6%�	���B���A�*;


total_loss�ǥ@

error_R�\L?

learning_rate_1?�7�0�I       6%�	ԧB���A�*;


total_lossC��@

error_R�u\?

learning_rate_1?�7�]��I       6%�	��B���A�*;


total_loss��@

error_R.-U?

learning_rate_1?�7%u?�I       6%�	Kf�B���A�*;


total_loss`*�@

error_Rv�S?

learning_rate_1?�7܎� I       6%�	���B���A�*;


total_lossW̹@

error_R��L?

learning_rate_1?�7�G��I       6%�	�B���A�*;


total_lossן�@

error_Rl�U?

learning_rate_1?�7���^I       6%�	�1�B���A�*;


total_loss�2�@

error_R��U?

learning_rate_1?�7���WI       6%�	�t�B���A�*;


total_loss�_�@

error_R��E?

learning_rate_1?�7����I       6%�	ʿ�B���A�*;


total_loss\�@

error_Ra�:?

learning_rate_1?�7��4�I       6%�	�B���A�*;


total_loss�>�@

error_R(_W?

learning_rate_1?�7��&I       6%�	�S�B���A�*;


total_loss%�@

error_R�b?

learning_rate_1?�7ǡI       6%�	��B���A�*;


total_loss}�A

error_RߡB?

learning_rate_1?�7���ZI       6%�	�تB���A�*;


total_loss4}�@

error_RT�n?

learning_rate_1?�7\���I       6%�	��B���A�*;


total_loss=H�@

error_RvSm?

learning_rate_1?�7[ǍI       6%�	�h�B���A�*;


total_loss�}�@

error_R@GL?

learning_rate_1?�7��ĽI       6%�	#˫B���A�*;


total_loss��1A

error_R��I?

learning_rate_1?�7�Q0I       6%�	;�B���A�*;


total_loss�F�@

error_Rh�U?

learning_rate_1?�7�0�I       6%�	w^�B���A�*;


total_lossS��@

error_R@U?

learning_rate_1?�7�%OI       6%�	^��B���A�*;


total_losss��@

error_R�Y=?

learning_rate_1?�7P��	I       6%�	���B���A�*;


total_loss�tA

error_R;^M?

learning_rate_1?�7SqSNI       6%�	�@�B���A�*;


total_loss���@

error_R3=D?

learning_rate_1?�72|ӷI       6%�	@��B���A�*;


total_loss�	�@

error_RúA?

learning_rate_1?�7'/�I       6%�	ȭB���A�*;


total_lossڛ�@

error_R �L?

learning_rate_1?�7!꩚I       6%�	��B���A�*;


total_loss� =A

error_R)�E?

learning_rate_1?�7�ψyI       6%�	*Q�B���A�*;


total_loss��A

error_R�I?

learning_rate_1?�7��0I       6%�	,��B���A�*;


total_loss�#�@

error_R)�[?

learning_rate_1?�7$��I       6%�	yخB���A�*;


total_lossf�@

error_R
_?

learning_rate_1?�70H�6I       6%�	\�B���A�*;


total_lossH�@

error_RɕM?

learning_rate_1?�7ڃ�I       6%�	�b�B���A�*;


total_loss��@

error_Rw�R?

learning_rate_1?�7h8�I       6%�	���B���A�*;


total_loss��u@

error_RԓD?

learning_rate_1?�7cPI       6%�	��B���A�*;


total_loss�@

error_R�iC?

learning_rate_1?�7��|I       6%�	j3�B���A�*;


total_lossV��@

error_Rn�H?

learning_rate_1?�7�:HI       6%�	Jx�B���A�*;


total_loss!�@

error_Rd*7?

learning_rate_1?�7���I       6%�	_��B���A�*;


total_loss�h�@

error_R7JY?

learning_rate_1?�7��7gI       6%�	���B���A�*;


total_lossN(�@

error_Rm@?

learning_rate_1?�7w�g�I       6%�	�D�B���A�*;


total_loss���@

error_RO�P?

learning_rate_1?�70}c^I       6%�	���B���A�*;


total_lossOA

error_RduG?

learning_rate_1?�77�I       6%�	�ӱB���A�*;


total_loss9mA

error_R��K?

learning_rate_1?�7P�1I       6%�	K�B���A�*;


total_loss]j�@

error_R��D?

learning_rate_1?�7�!�I       6%�	�]�B���A�*;


total_lossH��@

error_R�>R?

learning_rate_1?�7f�(�I       6%�	���B���A�*;


total_lossL��@

error_R�O?

learning_rate_1?�7��=YI       6%�	g�B���A�*;


total_loss�P�@

error_R�AB?

learning_rate_1?�7�f�I       6%�	6'�B���A�*;


total_lossLi�@

error_R��K?

learning_rate_1?�7�jI       6%�	jp�B���A�*;


total_loss�*�@

error_R�HR?

learning_rate_1?�7��;AI       6%�	޵�B���A�*;


total_loss:C�@

error_R.�O?

learning_rate_1?�74���I       6%�	��B���A�*;


total_loss���@

error_Rl e?

learning_rate_1?�7��I       6%�	�>�B���A�*;


total_loss+�@

error_R��G?

learning_rate_1?�7RNAI       6%�	O��B���A�*;


total_loss���@

error_R��U?

learning_rate_1?�7�;{�I       6%�	�ʴB���A�*;


total_loss��@

error_RJvT?

learning_rate_1?�7����I       6%�	s�B���A�*;


total_lossH��@

error_R� W?

learning_rate_1?�7��/I       6%�	S�B���A�*;


total_lossİ�@

error_R�N?

learning_rate_1?�7P�rI       6%�	e��B���A�*;


total_loss_0�@

error_R�R?

learning_rate_1?�7�`I       6%�	�ݵB���A�*;


total_loss��@

error_R�J?

learning_rate_1?�7-#9I       6%�	4(�B���A�*;


total_loss�@

error_R��j?

learning_rate_1?�7J��I       6%�	�o�B���A�*;


total_loss�\�@

error_R)Q?

learning_rate_1?�7�߮�I       6%�	^��B���A�*;


total_loss�@

error_R��A?

learning_rate_1?�7����I       6%�	���B���A�*;


total_lossʆ�@

error_R{iE?

learning_rate_1?�7�/��I       6%�	QW�B���A�*;


total_loss��A

error_R�wI?

learning_rate_1?�7\+�I       6%�	���B���A�*;


total_lossO��@

error_R�EI?

learning_rate_1?�7T*xqI       6%�	��B���A�*;


total_loss$r�@

error_R�JG?

learning_rate_1?�70kI       6%�	:-�B���A�*;


total_loss~�@

error_R X?

learning_rate_1?�7b�uI       6%�	�v�B���A�*;


total_loss.$�@

error_R��.?

learning_rate_1?�7�d�I       6%�	���B���A�*;


total_loss�۪@

error_R��S?

learning_rate_1?�7A��I       6%�	[�B���A�*;


total_lossve�@

error_R&G`?

learning_rate_1?�7�z��I       6%�	cF�B���A�*;


total_loss(��@

error_RDNV?

learning_rate_1?�7P���I       6%�	���B���A�*;


total_loss�>�@

error_R�;?

learning_rate_1?�70!.FI       6%�	�׹B���A�*;


total_loss���@

error_R.#F?

learning_rate_1?�7��I       6%�	^�B���A�*;


total_loss&<�@

error_Rx�<?

learning_rate_1?�7:˴�I       6%�	{^�B���A�*;


total_loss1��@

error_R��6?

learning_rate_1?�7���I       6%�	���B���A�*;


total_loss�n@

error_R�	V?

learning_rate_1?�7�4�I       6%�	�B���A�*;


total_loss�A

error_RɟD?

learning_rate_1?�7�~�I       6%�	�-�B���A�*;


total_lossl��@

error_R��^?

learning_rate_1?�7�Q��I       6%�	��B���A�*;


total_loss���@

error_R�I?

learning_rate_1?�7��r�I       6%�	�X�B���A�*;


total_loss�5�@

error_R��N?

learning_rate_1?�7X��xI       6%�	a��B���A�*;


total_loss��@

error_R�u:?

learning_rate_1?�7�+��I       6%�	,��B���A�*;


total_loss!��@

error_R��V?

learning_rate_1?�7e��I       6%�	�7�B���A�*;


total_lossCR�@

error_R!�[?

learning_rate_1?�7����I       6%�	4z�B���A�*;


total_loss꛿@

error_Rd�Y?

learning_rate_1?�7.p��I       6%�	���B���A�*;


total_loss1M�@

error_R�g?

learning_rate_1?�7��I       6%�	��B���A�*;


total_loss� A

error_R�{O?

learning_rate_1?�7M�J�I       6%�	6F�B���A�*;


total_loss�}�@

error_Rw}M?

learning_rate_1?�7;���I       6%�	��B���A�*;


total_loss d�@

error_R�<Q?

learning_rate_1?�7w��eI       6%�	o��B���A�*;


total_loss燕@

error_RD}M?

learning_rate_1?�79�F�I       6%�	.�B���A�*;


total_loss��@

error_R ^X?

learning_rate_1?�7��|I       6%�	�W�B���A�*;


total_lossjz�@

error_R��A?

learning_rate_1?�7�1]�I       6%�	���B���A�*;


total_loss"n@

error_R�nY?

learning_rate_1?�7���pI       6%�	���B���A�*;


total_loss�>R@

error_R�jA?

learning_rate_1?�7_��I       6%�	�!�B���A�*;


total_loss���@

error_R=�I?

learning_rate_1?�7q���I       6%�	6h�B���A�*;


total_loss�"�@

error_RT�;?

learning_rate_1?�7.�(:I       6%�	t��B���A�*;


total_loss}�@

error_R�HL?

learning_rate_1?�7�՜I       6%�	���B���A�*;


total_loss�˩@

error_RJ?

learning_rate_1?�7��3�I       6%�	�-�B���A�*;


total_lossUA

error_R�F?

learning_rate_1?�7�%R�I       6%�	�r�B���A�*;


total_lossI�@

error_Rd�B?

learning_rate_1?�7�*�I       6%�	���B���A�*;


total_loss���@

error_Rh?

learning_rate_1?�7ab-0I       6%�	�	�B���A�*;


total_loss!��@

error_R�[?

learning_rate_1?�7�Z}I       6%�	nR�B���A�*;


total_lossei@

error_R��M?

learning_rate_1?�7��.�I       6%�	��B���A�*;


total_loss�@

error_Rf�>?

learning_rate_1?�7;�I       6%�	���B���A�*;


total_loss���@

error_R�rW?

learning_rate_1?�7<�A_I       6%�	
)�B���A�*;


total_loss)�@

error_ROH?

learning_rate_1?�7Π}I       6%�	?r�B���A�*;


total_losslA

error_R�W?

learning_rate_1?�7S��iI       6%�	��B���A�*;


total_loss��@

error_R�T?

learning_rate_1?�7��o�I       6%�	`�B���A�*;


total_loss�+�@

error_R��N?

learning_rate_1?�7*;�I       6%�	�I�B���A�*;


total_lossu$�@

error_R��E?

learning_rate_1?�70(��I       6%�	Z��B���A�*;


total_loss˸@

error_RŠA?

learning_rate_1?�7�.��I       6%�	���B���A�*;


total_loss�A

error_R}�_?

learning_rate_1?�7�)q�I       6%�	l-�B���A�*;


total_loss8}�@

error_R�6L?

learning_rate_1?�7Y��I       6%�	݁�B���A�*;


total_loss�%�@

error_R
�8?

learning_rate_1?�7[�~I       6%�	��B���A�*;


total_loss�-�@

error_R�(O?

learning_rate_1?�7��FI       6%�	m�B���A�*;


total_loss,:�@

error_R��N?

learning_rate_1?�7�늑I       6%�	�L�B���A�*;


total_lossJS�@

error_R�L?

learning_rate_1?�78T�I       6%�	5��B���A�*;


total_loss@D�@

error_R8�G?

learning_rate_1?�77cz;I       6%�	��B���A�*;


total_loss��@

error_R�\?

learning_rate_1?�7C��I       6%�	�,�B���A�*;


total_lossx�@

error_R�EW?

learning_rate_1?�7�B��I       6%�	���B���A�*;


total_loss���@

error_RsB??

learning_rate_1?�7��5|I       6%�	b��B���A�*;


total_losssڍ@

error_Rn�G?

learning_rate_1?�7�S�)I       6%�	y�B���A�*;


total_loss�(�@

error_R�0Y?

learning_rate_1?�7�E!�I       6%�	|y�B���A�*;


total_loss���@

error_R��H?

learning_rate_1?�70��I       6%�	2��B���A�*;


total_lossn��@

error_R!�L?

learning_rate_1?�7}���I       6%�	�B���A�*;


total_loss��@

error_R��N?

learning_rate_1?�7����I       6%�	in�B���A�*;


total_lossW��@

error_RdUY?

learning_rate_1?�7T�qI       6%�	���B���A�*;


total_lossf�1@

error_R��E?

learning_rate_1?�7O� I       6%�	���B���A�*;


total_loss���@

error_RW�G?

learning_rate_1?�73���I       6%�	�B�B���A�*;


total_loss�P�@

error_R�hD?

learning_rate_1?�7��TI       6%�	~��B���A�*;


total_loss���@

error_R�K?

learning_rate_1?�7�1O2I       6%�	��B���A�*;


total_loss茛@

error_R{�\?

learning_rate_1?�7g��I       6%�	-�B���A�*;


total_loss��A

error_R��V?

learning_rate_1?�7�||I       6%�	+Y�B���A�*;


total_loss(��@

error_R��H?

learning_rate_1?�7�9o�I       6%�	���B���A�*;


total_loss�A

error_RJ.M?

learning_rate_1?�7�s�`I       6%�	��B���A�*;


total_loss�4�@

error_R��W?

learning_rate_1?�7���nI       6%�	e[�B���A�*;


total_loss��@

error_R
(U?

learning_rate_1?�7͢�I       6%�	ӥ�B���A�*;


total_lossRZi@

error_R�-?

learning_rate_1?�7��PI       6%�	!��B���A�*;


total_loss��@

error_RB?

learning_rate_1?�7Y��DI       6%�	9�B���A�*;


total_loss�~�@

error_RHhE?

learning_rate_1?�7��l"I       6%�	b}�B���A�*;


total_loss���@

error_R��D?

learning_rate_1?�7��KI       6%�	���B���A�*;


total_loss+�@

error_R�X?

learning_rate_1?�7w�%I       6%�	��B���A�*;


total_lossn��@

error_R�CZ?

learning_rate_1?�7���I       6%�	�N�B���A�*;


total_lossأ�@

error_R��??

learning_rate_1?�74_��I       6%�	���B���A�*;


total_loss�
*A

error_R$T?

learning_rate_1?�7�D'I       6%�	���B���A�*;


total_lossl�@

error_Rq�P?

learning_rate_1?�70*�I       6%�	$�B���A�*;


total_loss|�@

error_R�[R?

learning_rate_1?�7��I       6%�	<g�B���A�*;


total_loss,��@

error_R�jP?

learning_rate_1?�7ex�I       6%�	��B���A�*;


total_loss��@

error_RR�\?

learning_rate_1?�7C��RI       6%�	T��B���A�*;


total_loss�@

error_R��b?

learning_rate_1?�7���,I       6%�	X4�B���A�*;


total_loss�݀@

error_R�^H?

learning_rate_1?�7�jI       6%�	�z�B���A�*;


total_losshڲ@

error_R�>?

learning_rate_1?�74�2�I       6%�	6��B���A�*;


total_loss�"�@

error_R�;S?

learning_rate_1?�7�Z��I       6%�	L�B���A�*;


total_loss���@

error_R��L?

learning_rate_1?�7
=<uI       6%�	BO�B���A�*;


total_loss��@

error_R??

learning_rate_1?�7��g�I       6%�	Z��B���A�*;


total_lossy0�@

error_R��D?

learning_rate_1?�7�PdI       6%�	!��B���A�*;


total_loss���@

error_R��>?

learning_rate_1?�7�!�6I       6%�	��B���A�*;


total_loss��A

error_RQFX?

learning_rate_1?�7���nI       6%�	�V�B���A�*;


total_loss��_@

error_R�7I?

learning_rate_1?�7�8ܾI       6%�	j��B���A�*;


total_loss� A

error_R�VG?

learning_rate_1?�7j���I       6%�	��B���A�*;


total_loss��@

error_R�E?

learning_rate_1?�7I�DMI       6%�	�*�B���A�*;


total_loss	�@

error_R�E?

learning_rate_1?�7�+*^I       6%�	�r�B���A�*;


total_loss�I�@

error_R�O?

learning_rate_1?�7��I       6%�	���B���A�*;


total_loss~�@

error_RuI?

learning_rate_1?�7���I       6%�	��B���A�*;


total_loss�z�@

error_Rv�O?

learning_rate_1?�7��~I       6%�	>@�B���A�*;


total_loss��@

error_R;}P?

learning_rate_1?�7��,II       6%�	X��B���A�*;


total_lossC$�@

error_R;�b?

learning_rate_1?�7���JI       6%�	:��B���A�*;


total_loss���@

error_R�S?

learning_rate_1?�7�\5I       6%�	��B���A�*;


total_loss��@

error_RqK[?

learning_rate_1?�7�2�xI       6%�	=r�B���A�*;


total_loss�W�@

error_RR�M?

learning_rate_1?�7l��I       6%�	\��B���A�*;


total_loss3�@

error_Rj�S?

learning_rate_1?�7(�8�I       6%�	���B���A�*;


total_loss�@

error_R��G?

learning_rate_1?�7%p�$I       6%�	vB�B���A�*;


total_loss	�@

error_R�NM?

learning_rate_1?�7k��I       6%�	��B���A�*;


total_loss���@

error_R�,??

learning_rate_1?�7��
�I       6%�	���B���A�*;


total_loss��@

error_R�U?

learning_rate_1?�7��I       6%�	D�B���A�*;


total_loss@

error_Rl�I?

learning_rate_1?�7p��I       6%�	�V�B���A�*;


total_loss�`@

error_RSrA?

learning_rate_1?�7[��^I       6%�	ƛ�B���A�*;


total_lossf�@

error_R	Qb?

learning_rate_1?�7@^��I       6%�	 ��B���A�*;


total_loss���@

error_R!�P?

learning_rate_1?�7� �yI       6%�	<+�B���A�*;


total_loss�l�@

error_R϶W?

learning_rate_1?�7��Z�I       6%�	l�B���A�*;


total_loss1��@

error_R#�d?

learning_rate_1?�7W��I       6%�	۲�B���A�*;


total_lossӂ@

error_R<�G?

learning_rate_1?�7��� I       6%�	@��B���A�*;


total_loss
n�@

error_R1�A?

learning_rate_1?�7��I       6%�	"<�B���A�*;


total_lossN(�@

error_RV�L?

learning_rate_1?�7l#7�I       6%�	��B���A�*;


total_loss��@

error_R�-\?

learning_rate_1?�7Ƽ�I       6%�	��B���A�*;


total_loss�e�@

error_Rx$V?

learning_rate_1?�7�^�I       6%�	9�B���A�*;


total_loss8�@

error_R�M?

learning_rate_1?�7�ʣ]I       6%�	�D�B���A�*;


total_losso@

error_R��B?

learning_rate_1?�7q��I       6%�	��B���A�*;


total_loss�\�@

error_R[QI?

learning_rate_1?�7ǣ�I       6%�	���B���A�*;


total_loss�L�@

error_R�>Q?

learning_rate_1?�7Z�MI       6%�	�B���A�*;


total_loss���@

error_R�0B?

learning_rate_1?�7`�G�I       6%�	IW�B���A�*;


total_loss?$�@

error_R`@?

learning_rate_1?�7�� �I       6%�	���B���A�*;


total_loss�6�@

error_R�m_?

learning_rate_1?�7nI|�I       6%�	��B���A�*;


total_loss�;�@

error_R�I?

learning_rate_1?�7cm�fI       6%�	��B���A�*;


total_loss2^�@

error_RpY?

learning_rate_1?�7`��I       6%�	�f�B���A�*;


total_loss�"�@

error_R`�P?

learning_rate_1?�7��BvI       6%�	'��B���A�*;


total_lossUH�@

error_RE�J?

learning_rate_1?�7��X�I       6%�	(��B���A�*;


total_loss���@

error_R�u<?

learning_rate_1?�7�,ƚI       6%�	!/�B���A�*;


total_loss�8�@

error_R�N?

learning_rate_1?�7��~I       6%�	/r�B���A�*;


total_loss�A�@

error_R�P?

learning_rate_1?�7�{jI       6%�	���B���A�*;


total_loss�!�@

error_Rs9O?

learning_rate_1?�7f�[eI       6%�	 �B���A�*;


total_loss1n�@

error_RCLC?

learning_rate_1?�7T�G�I       6%�	JK�B���A�*;


total_loss[$�@

error_R��L?

learning_rate_1?�7B�I       6%�	��B���A�*;


total_lossM�@

error_R�K?

learning_rate_1?�7쵌�I       6%�	���B���A�*;


total_lossJv@

error_R��C?

learning_rate_1?�7����I       6%�	a �B���A�*;


total_loss�	�@

error_R��R?

learning_rate_1?�7�fI       6%�	>k�B���A�*;


total_loss�ޚ@

error_R=�H?

learning_rate_1?�7�"I       6%�	���B���A�*;


total_loss��@

error_RE!^?

learning_rate_1?�7d�z4I       6%�	%��B���A�*;


total_loss��A

error_R�Z?

learning_rate_1?�7��VWI       6%�	_5�B���A�*;


total_loss/��@

error_R�KV?

learning_rate_1?�7��D�I       6%�	�u�B���A�*;


total_loss$� A

error_RE1P?

learning_rate_1?�7�7�I       6%�	��B���A�*;


total_loss�rA

error_R��Y?

learning_rate_1?�7`��&I       6%�	>�B���A�*;


total_loss��@

error_RM�B?

learning_rate_1?�7���I       6%�	�N�B���A�*;


total_loss�q�@

error_R<E?

learning_rate_1?�7��I       6%�	Y��B���A�*;


total_loss�l�@

error_R��H?

learning_rate_1?�7TI�I       6%�	���B���A�*;


total_loss2��@

error_R[�[?

learning_rate_1?�7�~R�I       6%�	O�B���A�*;


total_loss��@

error_RڽL?

learning_rate_1?�7����I       6%�	`b�B���A�*;


total_loss1́@

error_R��U?

learning_rate_1?�7v}sDI       6%�	ժ�B���A�*;


total_loss|A�@

error_RW�T?

learning_rate_1?�7�՞�I       6%�	���B���A�*;


total_lossqqj@

error_R�	Q?

learning_rate_1?�7�`X'I       6%�	j6�B���A�*;


total_lossQ��@

error_Rt�Y?

learning_rate_1?�7H���I       6%�	�w�B���A�*;


total_loss��@

error_R��>?

learning_rate_1?�7g�QI       6%�	P��B���A�*;


total_lossqRA

error_R@kK?

learning_rate_1?�7n�KII       6%�	?�B���A�*;


total_loss�Ҝ@

error_R� Q?

learning_rate_1?�7�\f�I       6%�	�H�B���A�*;


total_lossl~�@

error_R��I?

learning_rate_1?�7o�0�I       6%�	��B���A�*;


total_loss��@

error_R8sP?

learning_rate_1?�7B�Y�I       6%�	���B���A�*;


total_loss�<�@

error_RR�I?

learning_rate_1?�7�'�-I       6%�	-"�B���A�*;


total_lossĜA

error_R�Y?

learning_rate_1?�7����I       6%�	�~�B���A�*;


total_lossm
�@

error_R_ p?

learning_rate_1?�7�ܺ I       6%�	���B���A�*;


total_loss&�@

error_R�	D?

learning_rate_1?�7|��I       6%�	��B���A�	*;


total_loss���@

error_R��4?

learning_rate_1?�7�Է$I       6%�	�M�B���A�	*;


total_lossݲ�@

error_R;DJ?

learning_rate_1?�7��<�I       6%�	���B���A�	*;


total_loss��@

error_RqS?

learning_rate_1?�7�KI       6%�	 ��B���A�	*;


total_loss�?�@

error_R��S?

learning_rate_1?�76�I       6%�	��B���A�	*;


total_lossa A

error_RzS?

learning_rate_1?�7���I       6%�	�Z�B���A�	*;


total_loss柸@

error_RؘJ?

learning_rate_1?�7��Z�I       6%�	���B���A�	*;


total_loss���@

error_Rz�Z?

learning_rate_1?�7��R�I       6%�	{��B���A�	*;


total_lossP��@

error_R�<?

learning_rate_1?�7�f��I       6%�	�'�B���A�	*;


total_lossjc�@

error_R3�Y?

learning_rate_1?�7:� I       6%�	Il�B���A�	*;


total_lossZ��@

error_R3]?

learning_rate_1?�7��I       6%�	���B���A�	*;


total_loss���@

error_R�[?

learning_rate_1?�7'7��I       6%�	��B���A�	*;


total_loss���@

error_R�YL?

learning_rate_1?�7�`�JI       6%�	>2�B���A�	*;


total_loss(c�@

error_Rv�O?

learning_rate_1?�7n�I       6%�	�{�B���A�	*;


total_loss$��@

error_R͘N?

learning_rate_1?�7�8��I       6%�	���B���A�	*;


total_lossV(�@

error_R� g?

learning_rate_1?�7���,I       6%�	=*�B���A�	*;


total_lossp��@

error_R�wN?

learning_rate_1?�7�w�I       6%�	N��B���A�	*;


total_loss�T�@

error_RaR?

learning_rate_1?�7]2cI       6%�	���B���A�	*;


total_loss,�@

error_RM@Q?

learning_rate_1?�7�6I       6%�	�0�B���A�	*;


total_loss枛@

error_Rw<?

learning_rate_1?�7��5I       6%�	u�B���A�	*;


total_loss}��@

error_R�M?

learning_rate_1?�7�UII       6%�	���B���A�	*;


total_loss�Q�@

error_R��W?

learning_rate_1?�78��uI       6%�	��B���A�	*;


total_lossG��@

error_R�V?

learning_rate_1?�7Xg��I       6%�	cP�B���A�	*;


total_loss��@

error_R�P?

learning_rate_1?�7��u@I       6%�	M��B���A�	*;


total_loss8�A

error_RcEI?

learning_rate_1?�7�(�_I       6%�	���B���A�	*;


total_loss���@

error_R�9f?

learning_rate_1?�7���XI       6%�	]!�B���A�	*;


total_loss.�@

error_R@�a?

learning_rate_1?�7̷"�I       6%�	�f�B���A�	*;


total_loss.��@

error_R�TO?

learning_rate_1?�7#)ːI       6%�	��B���A�	*;


total_loss��A

error_R&`?

learning_rate_1?�7�XQ�I       6%�	<��B���A�	*;


total_loss��@

error_RFa?

learning_rate_1?�7��I       6%�	�3�B���A�	*;


total_loss��@

error_R�Q?

learning_rate_1?�7$�:pI       6%�	v�B���A�	*;


total_loss���@

error_R7+W?

learning_rate_1?�7`U1dI       6%�	?��B���A�	*;


total_lossʇ�@

error_R��Q?

learning_rate_1?�7=��I       6%�	��B���A�	*;


total_loss&o�@

error_R��E?

learning_rate_1?�7i#N�I       6%�	�D�B���A�	*;


total_loss=�@

error_R'H?

learning_rate_1?�7g��I       6%�	*��B���A�	*;


total_loss��@

error_RܤS?

learning_rate_1?�7�ޔI       6%�	���B���A�	*;


total_lossl�A

error_RQV?

learning_rate_1?�7���!I       6%�	C�B���A�	*;


total_loss�c�@

error_RI�c?

learning_rate_1?�7���I       6%�	�X�B���A�	*;


total_loss�.�@

error_RԦI?

learning_rate_1?�7�4��I       6%�	���B���A�	*;


total_lossi�p@

error_R��C?

learning_rate_1?�7&W>I       6%�	��B���A�	*;


total_loss:U�@

error_R:�<?

learning_rate_1?�7p��I       6%�	+�B���A�	*;


total_loss�ӗ@

error_R�L?

learning_rate_1?�7@i�I       6%�	�t�B���A�	*;


total_loss�C�@

error_RŜS?

learning_rate_1?�7��TI       6%�	ݹ�B���A�	*;


total_loss�!�@

error_Ri->?

learning_rate_1?�7���"I       6%�	R��B���A�	*;


total_loss�[�@

error_R$�I?

learning_rate_1?�7>��I       6%�	�C�B���A�	*;


total_loss�ɣ@

error_R!�M?

learning_rate_1?�7��#�I       6%�	���B���A�	*;


total_loss�A

error_RQ`?

learning_rate_1?�7Yh��I       6%�	M��B���A�	*;


total_loss�>�@

error_R��U?

learning_rate_1?�7|P(�I       6%�	�B���A�	*;


total_loss�`�@

error_RZ�Q?

learning_rate_1?�7�A�I       6%�	ad�B���A�	*;


total_loss墅@

error_R�P?

learning_rate_1?�7A��I       6%�	���B���A�	*;


total_lossn�@

error_R�'Y?

learning_rate_1?�7%Y	�I       6%�	���B���A�	*;


total_loss�!�@

error_R1�M?

learning_rate_1?�7[��I       6%�	7�B���A�	*;


total_loss���@

error_RX�P?

learning_rate_1?�7���I       6%�	Sx�B���A�	*;


total_loss�U�@

error_RO�J?

learning_rate_1?�7�t��I       6%�	5��B���A�	*;


total_lossmb�@

error_R��X?

learning_rate_1?�7Y��I       6%�	@��B���A�	*;


total_loss]9�@

error_R�~=?

learning_rate_1?�7�!I       6%�	)\�B���A�	*;


total_loss��A

error_RC�T?

learning_rate_1?�7�=1I       6%�	 ��B���A�	*;


total_losshP�@

error_R
/>?

learning_rate_1?�7�6��I       6%�	���B���A�	*;


total_loss?�@

error_R
0S?

learning_rate_1?�7����I       6%�	Q3�B���A�	*;


total_loss8��@

error_R�E?

learning_rate_1?�7B�ԲI       6%�	5x�B���A�	*;


total_lossŰ�@

error_R1:C?

learning_rate_1?�7��TI       6%�	���B���A�	*;


total_loss"�@

error_R܍K?

learning_rate_1?�7�,bI       6%�	L�B���A�	*;


total_loss�ؽ@

error_R6�H?

learning_rate_1?�7�R�I       6%�	oR�B���A�	*;


total_loss���@

error_R�4?

learning_rate_1?�7��AI       6%�	���B���A�	*;


total_loss �A

error_RL?

learning_rate_1?�7A�I       6%�	���B���A�	*;


total_loss���@

error_R�B?

learning_rate_1?�7c�!I       6%�	h �B���A�	*;


total_lossN��@

error_R �O?

learning_rate_1?�7�%�HI       6%�	c�B���A�	*;


total_lossh\�@

error_Rq�S?

learning_rate_1?�7� ��I       6%�	r��B���A�	*;


total_loss�M�@

error_R@�E?

learning_rate_1?�7���~I       6%�	���B���A�	*;


total_loss�I�@

error_R�J?

learning_rate_1?�7NV��I       6%�	�+�B���A�	*;


total_lossn?�@

error_RxXQ?

learning_rate_1?�7��̉I       6%�	>p�B���A�	*;


total_lossv�|@

error_R��N?

learning_rate_1?�7�R�xI       6%�	��B���A�	*;


total_losst>�@

error_Rd�V?

learning_rate_1?�7�_O�I       6%�	T�B���A�	*;


total_loss��@

error_R��H?

learning_rate_1?�7��a�I       6%�	�D�B���A�	*;


total_lossT�@

error_R�E?

learning_rate_1?�7[��QI       6%�	���B���A�	*;


total_loss��@

error_RSOI?

learning_rate_1?�7�7�nI       6%�	B��B���A�	*;


total_loss���@

error_Rx�S?

learning_rate_1?�7����I       6%�	N%�B���A�	*;


total_loss�Xr@

error_RzM?

learning_rate_1?�7~�tnI       6%�	�k�B���A�	*;


total_loss{ε@

error_R�7[?

learning_rate_1?�7���I       6%�	���B���A�	*;


total_loss�@

error_R߿G?

learning_rate_1?�7NxEI       6%�	_��B���A�	*;


total_lossc��@

error_R��I?

learning_rate_1?�7���I       6%�	*;�B���A�	*;


total_lossݔ�@

error_RܰV?

learning_rate_1?�74xI       6%�	�~�B���A�	*;


total_lossע�@

error_R!eG?

learning_rate_1?�7iTDI       6%�	���B���A�	*;


total_loss��@

error_R;�;?

learning_rate_1?�7��.GI       6%�	(�B���A�	*;


total_loss�w�@

error_R��X?

learning_rate_1?�7���I       6%�	'P�B���A�	*;


total_loss��@

error_R��B?

learning_rate_1?�7bI       6%�	���B���A�	*;


total_loss�q�@

error_R�>?

learning_rate_1?�7+�X�I       6%�	7��B���A�	*;


total_loss��@

error_R�>M?

learning_rate_1?�7YQ��I       6%�	) C���A�	*;


total_loss=��@

error_R�6V?

learning_rate_1?�7/sII       6%�	Um C���A�	*;


total_loss@��@

error_R�X?

learning_rate_1?�7��GUI       6%�	׶ C���A�	*;


total_loss��@

error_R��9?

learning_rate_1?�7�0�.I       6%�	�C���A�	*;


total_loss̺�@

error_RXG?

learning_rate_1?�7�߇�I       6%�	�LC���A�	*;


total_loss1BA

error_R��H?

learning_rate_1?�7���I       6%�	ۛC���A�	*;


total_lossv��@

error_R��L?

learning_rate_1?�7:"�YI       6%�	l�C���A�	*;


total_lossږ�@

error_R�nV?

learning_rate_1?�7&��iI       6%�	�,C���A�	*;


total_loss�J�@

error_R�m^?

learning_rate_1?�7��b�I       6%�	uqC���A�	*;


total_loss��h@

error_R{�[?

learning_rate_1?�7�
a�I       6%�	��C���A�	*;


total_lossq��@

error_R�z_?

learning_rate_1?�7����I       6%�	V�C���A�	*;


total_loss��@

error_R��d?

learning_rate_1?�7�9I       6%�	�EC���A�	*;


total_lossmk�@

error_R��F?

learning_rate_1?�7����I       6%�	��C���A�	*;


total_loss(p�@

error_RH?

learning_rate_1?�77��I       6%�	��C���A�	*;


total_lossY�@

error_R�Q?

learning_rate_1?�7��I       6%�	�C���A�	*;


total_losshQ�@

error_R�J?

learning_rate_1?�7���I       6%�	{UC���A�	*;


total_lossAǯ@

error_R�RS?

learning_rate_1?�7>Z`�I       6%�	��C���A�	*;


total_loss-��@

error_R�:d?

learning_rate_1?�7_�{I       6%�	X�C���A�	*;


total_lossǘ�@

error_R��d?

learning_rate_1?�7��w�I       6%�	b%C���A�	*;


total_loss=�@

error_R�ZD?

learning_rate_1?�7�(�I       6%�	�gC���A�	*;


total_loss&i�@

error_R�F]?

learning_rate_1?�7�|�I       6%�	��C���A�	*;


total_lossI�@

error_Rn I?

learning_rate_1?�7!nI       6%�	Z�C���A�	*;


total_loss^�A

error_RTzJ?

learning_rate_1?�7��X�I       6%�	�/C���A�	*;


total_losss��@

error_R�xD?

learning_rate_1?�7��05I       6%�	>qC���A�	*;


total_loss��@

error_Rx=?

learning_rate_1?�7�I       6%�	��C���A�	*;


total_loss�2�@

error_R�ZO?

learning_rate_1?�7�l�TI       6%�	�C���A�	*;


total_lossz��@

error_Roa?

learning_rate_1?�7����I       6%�	mVC���A�	*;


total_loss���@

error_R\�Q?

learning_rate_1?�7,�|kI       6%�	��C���A�	*;


total_loss�:�@

error_R�K?

learning_rate_1?�7����I       6%�	.�C���A�	*;


total_loss���@

error_R)�U?

learning_rate_1?�7U��QI       6%�	f*C���A�	*;


total_loss3�@

error_R�F?

learning_rate_1?�7�(dI       6%�	znC���A�	*;


total_loss�_A

error_R8qR?

learning_rate_1?�7;o�UI       6%�	�C���A�	*;


total_loss��@

error_R�Y^?

learning_rate_1?�7'}I       6%�	"�C���A�	*;


total_lossdȅ@

error_R�kT?

learning_rate_1?�7ሾ-I       6%�	$8	C���A�	*;


total_loss�@

error_Rs�N?

learning_rate_1?�7>z�YI       6%�	'�	C���A�	*;


total_loss���@

error_R�O?

learning_rate_1?�7���I       6%�	��	C���A�	*;


total_loss�i�@

error_R�VT?

learning_rate_1?�7���I       6%�	
C���A�	*;


total_lossQ.�@

error_RoFQ?

learning_rate_1?�7XQ�7I       6%�	�Q
C���A�	*;


total_loss���@

error_RM�L?

learning_rate_1?�7p��I       6%�	y�
C���A�	*;


total_lossڣ�@

error_R��@?

learning_rate_1?�7�ωI       6%�	��
C���A�	*;


total_loss1��@

error_R��V?

learning_rate_1?�7-8�I       6%�	�C���A�	*;


total_loss ��@

error_RE?

learning_rate_1?�7�#cI       6%�	�cC���A�
*;


total_loss�d�@

error_R��\?

learning_rate_1?�7�ɲ#I       6%�	f�C���A�
*;


total_lossP�@

error_RrW:?

learning_rate_1?�7Ƶ�I       6%�	aC���A�
*;


total_lossf�@

error_R4�E?

learning_rate_1?�7���I       6%�	�ZC���A�
*;


total_loss��@

error_R��L?

learning_rate_1?�7����I       6%�	w�C���A�
*;


total_loss8jA

error_R@�R?

learning_rate_1?�7x-�bI       6%�	
C���A�
*;


total_lossN�@

error_R��C?

learning_rate_1?�7
��wI       6%�	eC���A�
*;


total_lossQ�@

error_R $I?

learning_rate_1?�7��ZI       6%�	J�C���A�
*;


total_loss���@

error_Rm6S?

learning_rate_1?�7��#I       6%�	��C���A�
*;


total_loss��@

error_R�*O?

learning_rate_1?�7��p�I       6%�	�=C���A�
*;


total_loss���@

error_R�H?

learning_rate_1?�7,��QI       6%�	H�C���A�
*;


total_loss��@

error_Rd�]?

learning_rate_1?�7�$I       6%�	��C���A�
*;


total_lossF�@

error_R��U?

learning_rate_1?�7G��iI       6%�	i	C���A�
*;


total_loss#��@

error_RwA?

learning_rate_1?�7����I       6%�	�LC���A�
*;


total_lossA5�@

error_R.�C?

learning_rate_1?�7qP�I       6%�	��C���A�
*;


total_loss���@

error_R԰N?

learning_rate_1?�7O,d�I       6%�	8�C���A�
*;


total_loss���@

error_R��I?

learning_rate_1?�7"Oc�I       6%�	]C���A�
*;


total_loss��@

error_RLB?

learning_rate_1?�7H`�HI       6%�	�cC���A�
*;


total_losstZ�@

error_R��^?

learning_rate_1?�7�}�4I       6%�	C�C���A�
*;


total_loss�k�@

error_R�X?

learning_rate_1?�7s�kI       6%�	�C���A�
*;


total_lossԦA

error_R�MS?

learning_rate_1?�7�U�I       6%�	�,C���A�
*;


total_loss�P�@

error_R�nT?

learning_rate_1?�7��'I       6%�	�oC���A�
*;


total_loss���@

error_R��Y?

learning_rate_1?�7b:�I       6%�	O�C���A�
*;


total_loss���@

error_R.�B?

learning_rate_1?�7,@;I       6%�	��C���A�
*;


total_loss���@

error_RiUQ?

learning_rate_1?�7��ĹI       6%�	Y;C���A�
*;


total_losso��@

error_R��S?

learning_rate_1?�7��I       6%�	%~C���A�
*;


total_loss�4�@

error_Rd]N?

learning_rate_1?�7
9I       6%�	�C���A�
*;


total_loss��@

error_Rx�K?

learning_rate_1?�7lz~UI       6%�	�C���A�
*;


total_loss��@

error_R�M?

learning_rate_1?�7�/~I       6%�	)LC���A�
*;


total_loss�J�@

error_R	iT?

learning_rate_1?�7�(��I       6%�	p�C���A�
*;


total_loss�3�@

error_R^?

learning_rate_1?�7Y��I       6%�	K�C���A�
*;


total_losss	�@

error_R�Y?

learning_rate_1?�7��tI       6%�	�C���A�
*;


total_loss��@

error_R�;G?

learning_rate_1?�7{�]I       6%�	0_C���A�
*;


total_loss���@

error_R�S?

learning_rate_1?�7Q�M�I       6%�	��C���A�
*;


total_loss���@

error_R{�F?

learning_rate_1?�7b�o�I       6%�	��C���A�
*;


total_loss΄@

error_Rx�U?

learning_rate_1?�7`"�!I       6%�	R,C���A�
*;


total_loss#�A

error_R�&I?

learning_rate_1?�7[��cI       6%�	PnC���A�
*;


total_loss�@

error_RS?

learning_rate_1?�7�ܢI       6%�	��C���A�
*;


total_lossx�@

error_R��S?

learning_rate_1?�7��v	I       6%�	��C���A�
*;


total_loss���@

error_R��V?

learning_rate_1?�7�Y�\I       6%�	y4C���A�
*;


total_loss��@

error_RL�X?

learning_rate_1?�7[�I       6%�	�wC���A�
*;


total_lossC��@

error_R��F?

learning_rate_1?�7��՘I       6%�	�C���A�
*;


total_lossX�@

error_R�NY?

learning_rate_1?�74:��I       6%�	��C���A�
*;


total_losss��@

error_R�SW?

learning_rate_1?�7��,pI       6%�	hC���A�
*;


total_lossτ�@

error_R,�Z?

learning_rate_1?�7�C\[I       6%�	p�C���A�
*;


total_loss_��@

error_R/T?

learning_rate_1?�7� GI       6%�	Y�C���A�
*;


total_loss�>�@

error_R�&S?

learning_rate_1?�7I|%BI       6%�	6C���A�
*;


total_loss-|A

error_R��H?

learning_rate_1?�7�e	aI       6%�	:{C���A�
*;


total_loss�\�@

error_R$/8?

learning_rate_1?�7�xlUI       6%�	��C���A�
*;


total_loss5�@

error_RsFU?

learning_rate_1?�7{,�I       6%�	yC���A�
*;


total_lossNM�@

error_RW�F?

learning_rate_1?�7�üI       6%�	|GC���A�
*;


total_loss"�@

error_R�{??

learning_rate_1?�7���I       6%�	k�C���A�
*;


total_loss��@

error_RƌK?

learning_rate_1?�7n��I       6%�	�C���A�
*;


total_lossSg�@

error_R��V?

learning_rate_1?�7[��I       6%�	�C���A�
*;


total_loss�9�@

error_R�Nc?

learning_rate_1?�7Rsc�I       6%�	�[C���A�
*;


total_loss
<�@

error_Rj6?

learning_rate_1?�7|��I       6%�	��C���A�
*;


total_loss鬩@

error_Rn�R?

learning_rate_1?�7����I       6%�	�C���A�
*;


total_lossM_�@

error_R�V7?

learning_rate_1?�7�7��I       6%�	�+C���A�
*;


total_loss�)�@

error_R��L?

learning_rate_1?�7	Y�I       6%�	nC���A�
*;


total_loss�΀@

error_R$K?

learning_rate_1?�7��8I       6%�	ٯC���A�
*;


total_loss3��@

error_R��L?

learning_rate_1?�7���I       6%�	U�C���A�
*;


total_loss�A

error_RR�O?

learning_rate_1?�7��<I       6%�	�5C���A�
*;


total_loss3��@

error_R�X]?

learning_rate_1?�7&s:�I       6%�	�yC���A�
*;


total_loss�}�@

error_R�Jc?

learning_rate_1?�7��(LI       6%�	W�C���A�
*;


total_loss�
�@

error_R(�`?

learning_rate_1?�7����I       6%�	0C���A�
*;


total_loss�ٳ@

error_R��@?

learning_rate_1?�7�I�I       6%�	�OC���A�
*;


total_loss.��@

error_R��>?

learning_rate_1?�7��I       6%�	�C���A�
*;


total_loss���@

error_RE�W?

learning_rate_1?�7�V��I       6%�	��C���A�
*;


total_loss3��@

error_R߆P?

learning_rate_1?�7L{2zI       6%�	�C���A�
*;


total_lossQ�@

error_R�-F?

learning_rate_1?�7�R�I       6%�	�dC���A�
*;


total_loss���@

error_R�T?

learning_rate_1?�7���I       6%�	��C���A�
*;


total_lossN��@

error_R@W?

learning_rate_1?�7+:�I       6%�	�C���A�
*;


total_loss�P�@

error_R%%S?

learning_rate_1?�7"��I       6%�	:3C���A�
*;


total_loss�H A

error_RvS?

learning_rate_1?�7럖�I       6%�	hvC���A�
*;


total_loss�rA

error_R��=?

learning_rate_1?�7�F9I       6%�	��C���A�
*;


total_loss���@

error_R;vG?

learning_rate_1?�73�"^I       6%�	( C���A�
*;


total_loss���@

error_R�@?

learning_rate_1?�7F�I       6%�	BJ C���A�
*;


total_lossdt�@

error_R�9?

learning_rate_1?�7�G:�I       6%�	:� C���A�
*;


total_lossl�A

error_R��Y?

learning_rate_1?�7gDNI       6%�	_� C���A�
*;


total_lossԗ�@

error_RҳG?

learning_rate_1?�7��NI       6%�	%!C���A�
*;


total_lossv��@

error_RF`Y?

learning_rate_1?�7���I       6%�	;r!C���A�
*;


total_loss<�@

error_R�lU?

learning_rate_1?�7�S��I       6%�	f�!C���A�
*;


total_loss���@

error_RZXI?

learning_rate_1?�7�o�wI       6%�	��!C���A�
*;


total_loss��r@

error_R��I?

learning_rate_1?�7��	�I       6%�	�9"C���A�
*;


total_loss���@

error_R@�J?

learning_rate_1?�7����I       6%�	�"C���A�
*;


total_lossw�A

error_R�Q?

learning_rate_1?�7�%�I       6%�	�"C���A�
*;


total_loss�xA

error_RҚk?

learning_rate_1?�7U�E�I       6%�	a#C���A�
*;


total_loss�d�@

error_R.�9?

learning_rate_1?�7I�{�I       6%�	;S#C���A�
*;


total_lossr�@

error_R=U?

learning_rate_1?�7@��I       6%�	(�#C���A�
*;


total_loss��@

error_RHnG?

learning_rate_1?�7s��I       6%�	��#C���A�
*;


total_loss&�v@

error_Ro�>?

learning_rate_1?�7�(�^I       6%�	J6$C���A�
*;


total_loss-I�@

error_R�/I?

learning_rate_1?�7a,�I       6%�	��$C���A�
*;


total_lossct�@

error_R��>?

learning_rate_1?�7��cI       6%�	��$C���A�
*;


total_lossl�@

error_R��@?

learning_rate_1?�7[b�I       6%�	�%C���A�
*;


total_loss���@

error_RڽG?

learning_rate_1?�7��fI       6%�	-V%C���A�
*;


total_loss~'�@

error_R�EE?

learning_rate_1?�7[��I       6%�	8�%C���A�
*;


total_loss��@

error_R)�L?

learning_rate_1?�7���I       6%�	�%C���A�
*;


total_loss�\�@

error_R�0I?

learning_rate_1?�7R��8I       6%�	F#&C���A�
*;


total_loss$��@

error_R��W?

learning_rate_1?�7��I       6%�	�f&C���A�
*;


total_loss�@

error_R,�S?

learning_rate_1?�7�)'�I       6%�	��&C���A�
*;


total_loss`Ή@

error_R��E?

learning_rate_1?�7����I       6%�	��&C���A�
*;


total_loss���@

error_RnQ_?

learning_rate_1?�74GB�I       6%�	�K'C���A�
*;


total_lossIU�@

error_Rn�F?

learning_rate_1?�7nv�&I       6%�	4�'C���A�
*;


total_loss��@

error_R��U?

learning_rate_1?�7XbI       6%�	H�'C���A�
*;


total_loss��@

error_R�Y?

learning_rate_1?�7���I       6%�	�9(C���A�
*;


total_losss��@

error_R�hd?

learning_rate_1?�7�`�I       6%�	|~(C���A�
*;


total_loss���@

error_Rl�T?

learning_rate_1?�7:�<*I       6%�	�(C���A�
*;


total_loss�A

error_R�^X?

learning_rate_1?�7g N�I       6%�	g)C���A�
*;


total_loss��A

error_R��S?

learning_rate_1?�7Kqx�I       6%�	_F)C���A�
*;


total_loss���@

error_R��D?

learning_rate_1?�7���I       6%�	f�)C���A�
*;


total_loss?k�@

error_R��O?

learning_rate_1?�7��I       6%�	 �)C���A�
*;


total_loss���@

error_R�H?

learning_rate_1?�7���I       6%�	�*C���A�
*;


total_lossU�@

error_R��O?

learning_rate_1?�7S��I       6%�	Q*C���A�
*;


total_loss�A�@

error_R\�U?

learning_rate_1?�7Jd��I       6%�	d�*C���A�
*;


total_loss�@

error_R6S?

learning_rate_1?�7��)I       6%�	��*C���A�
*;


total_loss���@

error_R K?

learning_rate_1?�7I,ҡI       6%�	�+C���A�
*;


total_loss���@

error_R�O??

learning_rate_1?�7�k9I       6%�	�e+C���A�
*;


total_lossj��@

error_RĳI?

learning_rate_1?�7ZO�I       6%�	��+C���A�
*;


total_lossCd�@

error_R��S?

learning_rate_1?�7hlq�I       6%�	 ,C���A�
*;


total_loss�A

error_R��J?

learning_rate_1?�7�o�I       6%�	�S,C���A�
*;


total_loss��@

error_R�LE?

learning_rate_1?�7�Wd�I       6%�	��,C���A�
*;


total_loss$�@

error_R�_K?

learning_rate_1?�7 Z�I       6%�	�-C���A�
*;


total_loss��s@

error_Ràc?

learning_rate_1?�74���I       6%�	0I-C���A�
*;


total_loss�`�@

error_Rt�@?

learning_rate_1?�7~^�DI       6%�	S�-C���A�
*;


total_loss&��@

error_R��Z?

learning_rate_1?�7���I       6%�	��-C���A�
*;


total_loss���@

error_RN?

learning_rate_1?�7�S:�I       6%�	.C���A�
*;


total_loss]:�@

error_Rc[?

learning_rate_1?�7�x7I       6%�	�c.C���A�
*;


total_loss��A

error_R-TH?

learning_rate_1?�7�b3gI       6%�	T�.C���A�
*;


total_loss!��@

error_R�V?

learning_rate_1?�7i���I       6%�	|�.C���A�*;


total_loss��@

error_RF�L?

learning_rate_1?�7k��I       6%�	{</C���A�*;


total_loss��@

error_R:P?

learning_rate_1?�7N/fI       6%�	��/C���A�*;


total_loss��@

error_Rv_?

learning_rate_1?�7b��I       6%�	��/C���A�*;


total_loss���@

error_RR0B?

learning_rate_1?�7Rlu�I       6%�	40C���A�*;


total_loss=�@

error_R�wL?

learning_rate_1?�7F��kI       6%�	U0C���A�*;


total_loss��@

error_ReeB?

learning_rate_1?�7��NI       6%�	Ә0C���A�*;


total_loss�Z�@

error_R��9?

learning_rate_1?�7"j3I       6%�	��0C���A�*;


total_loss��@

error_R��W?

learning_rate_1?�7�<w�I       6%�	�!1C���A�*;


total_loss�;�@

error_R
�K?

learning_rate_1?�7��1I       6%�	uk1C���A�*;


total_loss
��@

error_R�>C?

learning_rate_1?�7uJY�I       6%�	��1C���A�*;


total_loss6�l@

error_Rx�C?

learning_rate_1?�7�B?�I       6%�	��1C���A�*;


total_loss�l�@

error_R��O?

learning_rate_1?�7D���I       6%�	;2C���A�*;


total_loss��
A

error_RڊU?

learning_rate_1?�7���I       6%�	�|2C���A�*;


total_lossh6�@

error_RZ�U?

learning_rate_1?�7XU\I       6%�	T�2C���A�*;


total_loss���@

error_Rq`U?

learning_rate_1?�7PY}I       6%�	�3C���A�*;


total_loss 0�@

error_RA�Z?

learning_rate_1?�7�!�kI       6%�	8E3C���A�*;


total_lossHP�@

error_R�4K?

learning_rate_1?�7;mP�I       6%�	��3C���A�*;


total_loss�\�@

error_RxU?

learning_rate_1?�7?���I       6%�	�3C���A�*;


total_lossV�@

error_R��I?

learning_rate_1?�7Q��I       6%�	�4C���A�*;


total_loss�z�@

error_R��E?

learning_rate_1?�7sp8I       6%�	Ac4C���A�*;


total_loss��@

error_R�M?

learning_rate_1?�7���I       6%�	D�4C���A�*;


total_loss]�@

error_R�8?

learning_rate_1?�7��1�I       6%�	 �4C���A�*;


total_loss�}@

error_R?EK?

learning_rate_1?�7��I       6%�	�)5C���A�*;


total_loss_��@

error_Rl�b?

learning_rate_1?�7���I       6%�	�l5C���A�*;


total_loss�4�@

error_R�(B?

learning_rate_1?�7 Y5FI       6%�	�5C���A�*;


total_loss�a�@

error_R��=?

learning_rate_1?�7�@�I       6%�	[�5C���A�*;


total_loss4��@

error_R��S?

learning_rate_1?�7&I�=I       6%�	
56C���A�*;


total_loss���@

error_RO�P?

learning_rate_1?�7��.�I       6%�	�v6C���A�*;


total_loss��~@

error_Rs�]?

learning_rate_1?�7���ZI       6%�	5�6C���A�*;


total_lossK��@

error_R*�P?

learning_rate_1?�7�*�/I       6%�	��6C���A�*;


total_loss�-�@

error_R�G?

learning_rate_1?�7Ɠ?�I       6%�	bZ7C���A�*;


total_lossj�@

error_R�1A?

learning_rate_1?�7<��I       6%�	:�7C���A�*;


total_loss1e�@

error_R�O?

learning_rate_1?�7�|�I       6%�	#�7C���A�*;


total_loss?�@

error_R�?Y?

learning_rate_1?�7Ө��I       6%�	�'8C���A�*;


total_loss8�@

error_R,"H?

learning_rate_1?�7��-GI       6%�	"j8C���A�*;


total_loss��@

error_R�Y?

learning_rate_1?�7R�I       6%�	��8C���A�*;


total_loss���@

error_R{�D?

learning_rate_1?�7'�`�I       6%�	3�8C���A�*;


total_losss��@

error_Ro2??

learning_rate_1?�7ƭ��I       6%�	349C���A�*;


total_loss1Vw@

error_RID?

learning_rate_1?�7n��UI       6%�	6y9C���A�*;


total_lossJ��@

error_RM�P?

learning_rate_1?�7O*E�I       6%�	��9C���A�*;


total_loss�^�@

error_Rv�I?

learning_rate_1?�7K֜�I       6%�	" :C���A�*;


total_loss�g�@

error_RgF?

learning_rate_1?�7w%�I       6%�	BB:C���A�*;


total_loss!w@

error_R�FA?

learning_rate_1?�7	e�I       6%�	&�:C���A�*;


total_loss�i�@

error_R�aN?

learning_rate_1?�7��o[I       6%�	��:C���A�*;


total_lossaA

error_R{�??

learning_rate_1?�7�2I       6%�	�;C���A�*;


total_lossD��@

error_R��O?

learning_rate_1?�7�ˢ�I       6%�	�S;C���A�*;


total_loss[Z�@

error_R��e?

learning_rate_1?�7H��I       6%�	��;C���A�*;


total_loss��@

error_R�H?

learning_rate_1?�7X�T�I       6%�	o�;C���A�*;


total_loss3s�@

error_Rq�<?

learning_rate_1?�7~~�mI       6%�	k(<C���A�*;


total_loss���@

error_R�QV?

learning_rate_1?�7�?cI       6%�	Il<C���A�*;


total_loss.�@

error_R.A?

learning_rate_1?�7��SI       6%�	�<C���A�*;


total_loss��@

error_RU?

learning_rate_1?�7�8�(I       6%�	U =C���A�*;


total_loss��@

error_RaFB?

learning_rate_1?�7,2�WI       6%�	�H=C���A�*;


total_loss,��@

error_R{�F?

learning_rate_1?�7�x/�I       6%�	b�=C���A�*;


total_loss��b@

error_RT�N?

learning_rate_1?�7n��I       6%�	��=C���A�*;


total_lossD��@

error_R��V?

learning_rate_1?�7i![�I       6%�	�>C���A�*;


total_loss��@

error_R{cO?

learning_rate_1?�7���)I       6%�	]>C���A�*;


total_loss�6�@

error_R�O?

learning_rate_1?�7��/�I       6%�	̞>C���A�*;


total_loss*�A

error_R�^N?

learning_rate_1?�7�fI       6%�	4�>C���A�*;


total_loss��@

error_R�IN?

learning_rate_1?�7#��I       6%�	m&?C���A�*;


total_loss���@

error_R�Q?

learning_rate_1?�7�\n�I       6%�	s?C���A�*;


total_lossA��@

error_R_)\?

learning_rate_1?�7�ԧ�I       6%�	�?C���A�*;


total_loss0�@

error_R.eE?

learning_rate_1?�7�N;%I       6%�	�@C���A�*;


total_loss���@

error_R�[K?

learning_rate_1?�7f�-nI       6%�	�J@C���A�*;


total_loss�y�@

error_R��P?

learning_rate_1?�7��N�I       6%�	ؑ@C���A�*;


total_lossm��@

error_R��S?

learning_rate_1?�7N2qI       6%�	��@C���A�*;


total_lossL��@

error_RV�T?

learning_rate_1?�7���I       6%�	]AC���A�*;


total_loss�`�@

error_R�=M?

learning_rate_1?�7h�I       6%�	NcAC���A�*;


total_lossK/�@

error_R�G?

learning_rate_1?�7��O�I       6%�	��AC���A�*;


total_loss��@

error_R�yJ?

learning_rate_1?�7��I       6%�	��AC���A�*;


total_loss걤@

error_R�C?

learning_rate_1?�7w�Q,I       6%�	-BC���A�*;


total_loss2��@

error_R�mR?

learning_rate_1?�7�	�I       6%�	yBC���A�*;


total_loss���@

error_R�V?

learning_rate_1?�7��n9I       6%�	��BC���A�*;


total_loss@�@

error_RjVE?

learning_rate_1?�7��d^I       6%�	��BC���A�*;


total_loss��@

error_RT�S?

learning_rate_1?�7c�PI       6%�	H;CC���A�*;


total_loss�v�@

error_R.�A?

learning_rate_1?�7"~rwI       6%�	}CC���A�*;


total_loss��@

error_R��W?

learning_rate_1?�7��"�I       6%�	�CC���A�*;


total_lossO�@

error_R�f?

learning_rate_1?�7�EO�I       6%�	��CC���A�*;


total_lossֽ�@

error_R=1X?

learning_rate_1?�7�UZ�I       6%�	�@DC���A�*;


total_loss��@

error_R(W?

learning_rate_1?�7��8,I       6%�	��DC���A�*;


total_loss�Q�@

error_R�T?

learning_rate_1?�7�i��I       6%�	��DC���A�*;


total_loss>��@

error_Ri�B?

learning_rate_1?�79���I       6%�	�EC���A�*;


total_loss�P�@

error_RR?

learning_rate_1?�7�RzI       6%�	0WEC���A�*;


total_loss-��@

error_Ra�U?

learning_rate_1?�70��FI       6%�	�EC���A�*;


total_loss� �@

error_R\oP?

learning_rate_1?�7��r I       6%�	��EC���A�*;


total_loss�+�@

error_RxA6?

learning_rate_1?�7C�S�I       6%�	�$FC���A�*;


total_lossz8�@

error_R�lU?

learning_rate_1?�7O��I       6%�	eiFC���A�*;


total_loss�E�@

error_Rn�G?

learning_rate_1?�7�$BI       6%�	��FC���A�*;


total_loss�@

error_R�X?

learning_rate_1?�7{,l�I       6%�	��FC���A�*;


total_loss�ސ@

error_RH F?

learning_rate_1?�7�K\�I       6%�	�WGC���A�*;


total_loss�۬@

error_R;UT?

learning_rate_1?�7�ݹ�I       6%�	��GC���A�*;


total_loss&��@

error_Rx�??

learning_rate_1?�7.Dx�I       6%�	��GC���A�*;


total_loss�Q�@

error_Rt�^?

learning_rate_1?�7S`�>I       6%�	�)HC���A�*;


total_lossE	�@

error_RT�J?

learning_rate_1?�7η�bI       6%�	mHC���A�*;


total_loss��@

error_R�yN?

learning_rate_1?�7D��9I       6%�	��HC���A�*;


total_lossZ�@

error_R��A?

learning_rate_1?�7.�5yI       6%�	�HC���A�*;


total_lossH1�@

error_R��V?

learning_rate_1?�7���JI       6%�	<4IC���A�*;


total_loss/��@

error_R�U??

learning_rate_1?�7�@?�I       6%�	�yIC���A�*;


total_loss]!�@

error_RC	I?

learning_rate_1?�7OWobI       6%�	��IC���A�*;


total_lossw�A

error_Rr*J?

learning_rate_1?�78L�I       6%�	rJC���A�*;


total_lossf��@

error_R��S?

learning_rate_1?�7��'I       6%�	LJC���A�*;


total_loss�V�@

error_R�K[?

learning_rate_1?�7}+.�I       6%�	k�JC���A�*;


total_lossM��@

error_RU?

learning_rate_1?�7�S�+I       6%�	U�JC���A�*;


total_loss�܆@

error_R��W?

learning_rate_1?�7\�U�I       6%�	� KC���A�*;


total_loss���@

error_R��V?

learning_rate_1?�7G5��I       6%�	�hKC���A�*;


total_loss�@

error_R��[?

learning_rate_1?�7���I       6%�	��KC���A�*;


total_loss��@

error_R�G?

learning_rate_1?�7x��CI       6%�	pLC���A�*;


total_loss��@

error_R��P?

learning_rate_1?�7X+I       6%�	�RLC���A�*;


total_loss�a�@

error_R�O?

learning_rate_1?�7"�iI       6%�	:�LC���A�*;


total_loss��@

error_R�<?

learning_rate_1?�7n�ЇI       6%�	2�LC���A�*;


total_loss���@

error_Rܯ]?

learning_rate_1?�7K�-I       6%�		9MC���A�*;


total_loss��@

error_R>U?

learning_rate_1?�7�<\jI       6%�	}MC���A�*;


total_loss���@

error_R��D?

learning_rate_1?�7C�0ZI       6%�	��MC���A�*;


total_lossd��@

error_R�d?

learning_rate_1?�7��e�I       6%�	NC���A�*;


total_loss���@

error_R<zP?

learning_rate_1?�7r��I       6%�	UNC���A�*;


total_loss<@�@

error_Rs\?

learning_rate_1?�74�۬I       6%�	�NC���A�*;


total_loss��A

error_R�^?

learning_rate_1?�7HMI       6%�	
�NC���A�*;


total_loss�̓@

error_R�NI?

learning_rate_1?�7G�_�I       6%�	aOC���A�*;


total_lossF��@

error_R;�T?

learning_rate_1?�7�`��I       6%�	KmOC���A�*;


total_loss���@

error_R_EO?

learning_rate_1?�7"�״I       6%�	j�OC���A�*;


total_loss��}@

error_R`US?

learning_rate_1?�7��1�I       6%�	�PC���A�*;


total_loss���@

error_RO?

learning_rate_1?�7XUI       6%�	�LPC���A�*;


total_lossՆ�@

error_R�B?

learning_rate_1?�7���*I       6%�	�PC���A�*;


total_lossXH�@

error_R,�K?

learning_rate_1?�7	P�I       6%�	��PC���A�*;


total_lossS��@

error_R7^?

learning_rate_1?�7���$I       6%�	�"QC���A�*;


total_loss�4�@

error_RVP?

learning_rate_1?�7�7hI       6%�	�hQC���A�*;


total_loss��%A

error_R�?I?

learning_rate_1?�7��ܛI       6%�	��QC���A�*;


total_loss�=�@

error_R,�H?

learning_rate_1?�7��I       6%�	a�QC���A�*;


total_loss���@

error_RM�X?

learning_rate_1?�7R�aI       6%�	.ARC���A�*;


total_loss<�@

error_R_S?

learning_rate_1?�7����I       6%�	]�RC���A�*;


total_loss�_�@

error_R�T?

learning_rate_1?�78��I       6%�	z�RC���A�*;


total_loss	��@

error_R��4?

learning_rate_1?�7���I       6%�	SC���A�*;


total_loss���@

error_R��Y?

learning_rate_1?�7�j:I       6%�	USC���A�*;


total_lossl`�@

error_R{i7?

learning_rate_1?�7�,�I       6%�	��SC���A�*;


total_loss���@

error_R��K?

learning_rate_1?�7C�I       6%�	��SC���A�*;


total_lossK�@

error_R��M?

learning_rate_1?�7��X�I       6%�	�&TC���A�*;


total_loss�3u@

error_R�P?

learning_rate_1?�7n��LI       6%�	�kTC���A�*;


total_loss##�@

error_R�UP?

learning_rate_1?�71&�bI       6%�	��TC���A�*;


total_loss���@

error_R�BU?

learning_rate_1?�7z�X�I       6%�	i�TC���A�*;


total_loss��^@

error_R��5?

learning_rate_1?�7kp�I       6%�	+5UC���A�*;


total_loss�
�@

error_Rn*9?

learning_rate_1?�7p\רI       6%�	�uUC���A�*;


total_loss��[@

error_RWZE?

learning_rate_1?�7-P7I       6%�	.�UC���A�*;


total_loss&x�@

error_R1rH?

learning_rate_1?�7�/؞I       6%�	!�UC���A�*;


total_lossD��@

error_Ra!J?

learning_rate_1?�7�\��I       6%�	7DVC���A�*;


total_loss�7�@

error_RO�>?

learning_rate_1?�7f��I       6%�	��VC���A�*;


total_loss��@

error_R�-Y?

learning_rate_1?�7�ldFI       6%�	-�VC���A�*;


total_loss��@

error_RE�T?

learning_rate_1?�7����I       6%�	[WC���A�*;


total_lossȨ�@

error_R�mS?

learning_rate_1?�7�SI       6%�	XwWC���A�*;


total_loss<��@

error_R��R?

learning_rate_1?�7uf5I       6%�	��WC���A�*;


total_lossq��@

error_R�Q?

learning_rate_1?�7�ז�I       6%�	sXC���A�*;


total_loss�ک@

error_R�>?

learning_rate_1?�7*�KI       6%�	�DXC���A�*;


total_loss%��@

error_R,ZV?

learning_rate_1?�7�C(gI       6%�	�XC���A�*;


total_lossH��@

error_R�eZ?

learning_rate_1?�7Y�lI       6%�	��XC���A�*;


total_loss��@

error_R��F?

learning_rate_1?�7���I       6%�	Z6YC���A�*;


total_lossxv�@

error_R�gA?

learning_rate_1?�7L��I       6%�	K~YC���A�*;


total_loss!��@

error_R8�R?

learning_rate_1?�7�ҤI       6%�	��YC���A�*;


total_lossWA

error_Rq�K?

learning_rate_1?�7캀I       6%�	 ZC���A�*;


total_loss6{�@

error_R�7^?

learning_rate_1?�7b�jpI       6%�	�^ZC���A�*;


total_loss���@

error_RD`G?

learning_rate_1?�7� I       6%�	q�ZC���A�*;


total_loss�{y@

error_R,�[?

learning_rate_1?�7"Yp�I       6%�	��ZC���A�*;


total_loss�>�@

error_R��C?

learning_rate_1?�7*�@~I       6%�	<A[C���A�*;


total_loss���@

error_R��R?

learning_rate_1?�7uw�1I       6%�	7�[C���A�*;


total_loss Ԗ@

error_R��U?

learning_rate_1?�7�iR�I       6%�	�[C���A�*;


total_lossoA

error_R�S?

learning_rate_1?�7���I       6%�	l"\C���A�*;


total_loss��@

error_R��U?

learning_rate_1?�7��\6I       6%�	�l\C���A�*;


total_loss�rj@

error_R[�W?

learning_rate_1?�7ɣ
�I       6%�	��\C���A�*;


total_loss�7�@

error_R��R?

learning_rate_1?�7�I       6%�	� ]C���A�*;


total_loss���@

error_RE�7?

learning_rate_1?�7Z�/+I       6%�	�O]C���A�*;


total_loss3j�@

error_R<J?

learning_rate_1?�7DJ��I       6%�	�]C���A�*;


total_lossH�@

error_R��K?

learning_rate_1?�7��\I       6%�	d�]C���A�*;


total_lossZX�@

error_R��Y?

learning_rate_1?�7��4~I       6%�	{6^C���A�*;


total_loss�l�@

error_RxJE?

learning_rate_1?�7sy�#I       6%�	�^C���A�*;


total_loss�ǟ@

error_R�Z?

learning_rate_1?�7��q{I       6%�	��^C���A�*;


total_loss�.A

error_R��V?

learning_rate_1?�7Xq?PI       6%�	s_C���A�*;


total_lossŹ�@

error_R�S?

learning_rate_1?�7���I       6%�	O_C���A�*;


total_loss�@

error_R�*G?

learning_rate_1?�7��בI       6%�	��_C���A�*;


total_lossFm�@

error_R,�V?

learning_rate_1?�7���I       6%�	!�_C���A�*;


total_loss[�}@

error_R 4O?

learning_rate_1?�7�^�I       6%�	�4`C���A�*;


total_loss:�@

error_Rn�??

learning_rate_1?�7$v��I       6%�	`C���A�*;


total_lossn
�@

error_R\NQ?

learning_rate_1?�77��=I       6%�	�`C���A�*;


total_loss� �@

error_R�8f?

learning_rate_1?�7%���I       6%�	�aC���A�*;


total_loss���@

error_RZ�K?

learning_rate_1?�7}6ݖI       6%�	PRaC���A�*;


total_lossf�@

error_R\�J?

learning_rate_1?�7D
2I       6%�	2�aC���A�*;


total_loss��@

error_R?b?

learning_rate_1?�7�o�'I       6%�	S�aC���A�*;


total_loss���@

error_Rd�F?

learning_rate_1?�7�F�!I       6%�	�%bC���A�*;


total_loss\i A

error_RRT?

learning_rate_1?�7�^w~I       6%�	�lbC���A�*;


total_loss���@

error_R��R?

learning_rate_1?�7e�ؑI       6%�	�bC���A�*;


total_loss�R�@

error_Rh�^?

learning_rate_1?�7QR�4I       6%�	��bC���A�*;


total_loss�t�@

error_R�<?

learning_rate_1?�7�|��I       6%�	W8cC���A�*;


total_loss#A

error_R�@?

learning_rate_1?�7�7�I       6%�	�|cC���A�*;


total_loss/��@

error_RiR?

learning_rate_1?�7��AbI       6%�	��cC���A�*;


total_loss�>�@

error_R�9?

learning_rate_1?�7T;&�I       6%�	+dC���A�*;


total_loss�G�@

error_Rz�\?

learning_rate_1?�7�F4I       6%�	q]dC���A�*;


total_loss��A

error_R�V?

learning_rate_1?�7�p9iI       6%�	��dC���A�*;


total_loss�UA

error_R�%`?

learning_rate_1?�7u�?�I       6%�	1�dC���A�*;


total_loss��@

error_RC`G?

learning_rate_1?�7���I       6%�	A7eC���A�*;


total_lossk�@

error_R�bC?

learning_rate_1?�7�? �I       6%�	z{eC���A�*;


total_loss�ػ@

error_R��I?

learning_rate_1?�7[�}I       6%�	 �eC���A�*;


total_lossD��@

error_R�JN?

learning_rate_1?�74CƴI       6%�		fC���A�*;


total_loss�4m@

error_R�yU?

learning_rate_1?�7�dI       6%�	nNfC���A�*;


total_loss�E�@

error_R�N?

learning_rate_1?�7i���I       6%�	�fC���A�*;


total_loss���@

error_RR�E?

learning_rate_1?�7��=KI       6%�	��fC���A�*;


total_lossA

error_R!H?

learning_rate_1?�7�h^I       6%�	R0gC���A�*;


total_loss�c�@

error_R!�B?

learning_rate_1?�7�F�I       6%�	'�gC���A�*;


total_loss���@

error_R��G?

learning_rate_1?�7���I       6%�	��gC���A�*;


total_lossV9�@

error_R�O?

learning_rate_1?�7pexI       6%�	�hC���A�*;


total_loss��@

error_R�5G?

learning_rate_1?�7:���I       6%�	�ghC���A�*;


total_loss㡺@

error_Rq�W?

learning_rate_1?�74#;�I       6%�	��hC���A�*;


total_loss`��@

error_Rۣ[?

learning_rate_1?�7�i��I       6%�	o�hC���A�*;


total_loss���@

error_RH?

learning_rate_1?�7���I       6%�	k-iC���A�*;


total_loss4*�@

error_RCT?

learning_rate_1?�7�O�I       6%�	MriC���A�*;


total_loss���@

error_R��Q?

learning_rate_1?�7����I       6%�	D�iC���A�*;


total_lossQrl@

error_R��??

learning_rate_1?�7�>�I       6%�	t�iC���A�*;


total_loss��`@

error_R��E?

learning_rate_1?�7J[`�I       6%�	1<jC���A�*;


total_loss#��@

error_RڥI?

learning_rate_1?�7Ϛ�I       6%�	]jC���A�*;


total_loss$�A

error_R�&R?

learning_rate_1?�7Iu��I       6%�	��jC���A�*;


total_loss-1 A

error_RR�5?

learning_rate_1?�7�HI       6%�	U
kC���A�*;


total_losst��@

error_R�J?

learning_rate_1?�7=��@I       6%�	�MkC���A�*;


total_loss��@

error_RFV1?

learning_rate_1?�7��wI       6%�	��kC���A�*;


total_loss)�-A

error_R�
V?

learning_rate_1?�7���VI       6%�	��kC���A�*;


total_lossHh�@

error_R&�G?

learning_rate_1?�7P�>4I       6%�	mlC���A�*;


total_loss���@

error_R�%7?

learning_rate_1?�7e=�I       6%�	F]lC���A�*;


total_loss���@

error_R7L?

learning_rate_1?�7���I       6%�	��lC���A�*;


total_loss8��@

error_R�|U?

learning_rate_1?�7V"��I       6%�	��lC���A�*;


total_loss���@

error_Rf�O?

learning_rate_1?�7z0sEI       6%�	*mC���A�*;


total_loss��@

error_R�iS?

learning_rate_1?�7�D1:I       6%�	�pmC���A�*;


total_loss��@

error_RZ�D?

learning_rate_1?�7NdI       6%�	e�mC���A�*;


total_lossw��@

error_R�aR?

learning_rate_1?�7?N��I       6%�	qnC���A�*;


total_loss5�@

error_RZ�M?

learning_rate_1?�7��I       6%�	|JnC���A�*;


total_lossO��@

error_R�Q?

learning_rate_1?�7v��tI       6%�	Q�nC���A�*;


total_loss&�@

error_RW*e?

learning_rate_1?�7X��I       6%�	��nC���A�*;


total_lossc!�@

error_Rc�N?

learning_rate_1?�7��I       6%�	oC���A�*;


total_loss3�@

error_R��M?

learning_rate_1?�7���I       6%�	boC���A�*;


total_loss�"�@

error_R�&??

learning_rate_1?�7�>��I       6%�	V�oC���A�*;


total_loss�P�@

error_R�l9?

learning_rate_1?�7J�RI       6%�	r�oC���A�*;


total_loss�D�@

error_RqwF?

learning_rate_1?�7?���I       6%�	P7pC���A�*;


total_lossFq2A

error_R��B?

learning_rate_1?�7�'xI       6%�	�~pC���A�*;


total_lossC��@

error_R�:\?

learning_rate_1?�75C�I       6%�	-�pC���A�*;


total_loss!�@

error_R��A?

learning_rate_1?�7;�ʳI       6%�	�qC���A�*;


total_lossw(�@

error_RE�D?

learning_rate_1?�7a*�I       6%�	RSqC���A�*;


total_loss��@

error_R�JC?

learning_rate_1?�7�ö�I       6%�	��qC���A�*;


total_loss�>�@

error_R�??

learning_rate_1?�7��O�I       6%�	^�qC���A�*;


total_loss�"�@

error_RC�@?

learning_rate_1?�7��jzI       6%�	�#rC���A�*;


total_loss��@

error_R�R?

learning_rate_1?�7�,�I       6%�	�irC���A�*;


total_lossv�@

error_R��T?

learning_rate_1?�7$I��I       6%�	��rC���A�*;


total_lossn�@

error_R��_?

learning_rate_1?�7�VC�I       6%�	m�rC���A�*;


total_loss�F�@

error_R�S?

learning_rate_1?�7�"��I       6%�	:sC���A�*;


total_loss�H�@

error_R�Ed?

learning_rate_1?�7V���I       6%�	��sC���A�*;


total_losslķ@

error_R8D?

learning_rate_1?�7u/OnI       6%�	��sC���A�*;


total_loss��@

error_R�L?

learning_rate_1?�7:	1XI       6%�	�tC���A�*;


total_loss�d�@

error_Rq�[?

learning_rate_1?�7��{I       6%�	�[tC���A�*;


total_lossDA�@

error_RSL?

learning_rate_1?�7���I       6%�	��tC���A�*;


total_loss�x�@

error_R�D?

learning_rate_1?�7+p|�I       6%�	?�tC���A�*;


total_loss�@

error_RP?

learning_rate_1?�7�[q�I       6%�	�1uC���A�*;


total_loss镱@

error_R��Q?

learning_rate_1?�7Bz<�I       6%�	�suC���A�*;


total_loss��@

error_R8I?

learning_rate_1?�7���I       6%�	<�uC���A�*;


total_lossa�@

error_R&eW?

learning_rate_1?�7v��BI       6%�	*	vC���A�*;


total_lossT��@

error_R��H?

learning_rate_1?�7�ӫI       6%�	�TvC���A�*;


total_losstο@

error_R<OU?

learning_rate_1?�7�cI       6%�	��vC���A�*;


total_loss���@

error_R�f?

learning_rate_1?�7��i�I       6%�	�vC���A�*;


total_loss�ɩ@

error_RgN?

learning_rate_1?�7�`ՁI       6%�	42wC���A�*;


total_loss�y�@

error_R��I?

learning_rate_1?�7:��jI       6%�	��wC���A�*;


total_loss�!�@

error_RL+>?

learning_rate_1?�7J�gI       6%�	��wC���A�*;


total_loss�:�@

error_RZ-W?

learning_rate_1?�7�s<I       6%�	�xC���A�*;


total_loss���@

error_R�S?

learning_rate_1?�7��*KI       6%�	�]xC���A�*;


total_loss�d�@

error_R�LI?

learning_rate_1?�7�\�I       6%�	o�xC���A�*;


total_loss��@

error_R��\?

learning_rate_1?�7��rI       6%�	t�xC���A�*;


total_lossv�@

error_R�88?

learning_rate_1?�7	N��I       6%�	&'yC���A�*;


total_lossH��@

error_R��F?

learning_rate_1?�7�y*tI       6%�	�iyC���A�*;


total_lossا@

error_RMN?

learning_rate_1?�7����I       6%�	��yC���A�*;


total_lossZ�@

error_R��V?

learning_rate_1?�7C�ȓI       6%�	��yC���A�*;


total_loss��@

error_R6eB?

learning_rate_1?�7���I       6%�	�9zC���A�*;


total_loss�Ϻ@

error_Rɧ\?

learning_rate_1?�7�T�=I       6%�	�|zC���A�*;


total_lossDO�@

error_R��S?

learning_rate_1?�7?�&�I       6%�	��zC���A�*;


total_loss�@

error_R??

learning_rate_1?�7�Z�DI       6%�	�{C���A�*;


total_loss2j�@

error_Rs5[?

learning_rate_1?�7a�I       6%�	EF{C���A�*;


total_loss߷�@

error_R�N?

learning_rate_1?�7j�I       6%�	�{C���A�*;


total_loss���@

error_R)B:?

learning_rate_1?�7}��I       6%�	��{C���A�*;


total_lossC �@

error_R�CV?

learning_rate_1?�7�H�4I       6%�	|C���A�*;


total_loss���@

error_RM�=?

learning_rate_1?�7�ͭI       6%�	�Z|C���A�*;


total_loss�A

error_R$�N?

learning_rate_1?�7��k�I       6%�	��|C���A�*;


total_loss&��@

error_R��n?

learning_rate_1?�7z��TI       6%�	 �|C���A�*;


total_loss�)�@

error_R��X?

learning_rate_1?�7��8dI       6%�	�:}C���A�*;


total_lossHN�@

error_R/�S?

learning_rate_1?�7w�q;I       6%�	��}C���A�*;


total_loss�}�@

error_Ri0Q?

learning_rate_1?�7QkrI       6%�	��}C���A�*;


total_loss���@

error_R�L?

learning_rate_1?�7}��I       6%�	)~C���A�*;


total_lossRt�@

error_RC�J?

learning_rate_1?�7�)ScI       6%�	p~C���A�*;


total_lossݓ�@

error_R�3?

learning_rate_1?�7��L�I       6%�	+�~C���A�*;


total_loss�A�@

error_R��S?

learning_rate_1?�7ٙv�I       6%�	��~C���A�*;


total_loss@

error_R�Z;?

learning_rate_1?�7�S�I       6%�	cDC���A�*;


total_loss �@

error_R}�W?

learning_rate_1?�7:8V�I       6%�	��C���A�*;


total_loss?�@

error_R�yC?

learning_rate_1?�7�PʍI       6%�	��C���A�*;


total_lossl�@

error_R�E?

learning_rate_1?�7I1I       6%�	��C���A�*;


total_loss(�@

error_R�oT?

learning_rate_1?�7F��)I       6%�	Te�C���A�*;


total_loss��@

error_R�'Q?

learning_rate_1?�7�	
I       6%�	A��C���A�*;


total_loss}(A

error_R(�H?

learning_rate_1?�7��%(I       6%�	5�C���A�*;


total_loss� A

error_R��H?

learning_rate_1?�79��xI       6%�	3�C���A�*;


total_loss}�@

error_R��V?

learning_rate_1?�7n��I       6%�	�x�C���A�*;


total_loss1ӥ@

error_Rm�Q?

learning_rate_1?�7�W�WI       6%�	���C���A�*;


total_lossE��@

error_RR�N?

learning_rate_1?�7�:BI       6%�	���C���A�*;


total_loss�9�@

error_R:�U?

learning_rate_1?�7@��	I       6%�	H�C���A�*;


total_loss�Y�@

error_RhT??

learning_rate_1?�7��I       6%�	Џ�C���A�*;


total_loss���@

error_R��Y?

learning_rate_1?�7uK`I       6%�	�ՂC���A�*;


total_loss���@

error_RZ?

learning_rate_1?�7A���I       6%�	<�C���A�*;


total_loss���@

error_R6�J?

learning_rate_1?�7��I       6%�	 `�C���A�*;


total_loss���@

error_Rn0<?

learning_rate_1?�7"��9I       6%�	C���A�*;


total_loss�[A

error_R<S?

learning_rate_1?�7g+jI       6%�	H�C���A�*;


total_loss���@

error_R��U?

learning_rate_1?�706*�I       6%�	�,�C���A�*;


total_loss��@

error_R3�S?

learning_rate_1?�7n��I       6%�	>u�C���A�*;


total_loss:~@

error_R) Y?

learning_rate_1?�7B`��I       6%�	m��C���A�*;


total_loss[U�@

error_R`�5?

learning_rate_1?�7���aI       6%�	���C���A�*;


total_loss��@

error_R&`Y?

learning_rate_1?�7LO��I       6%�	�A�C���A�*;


total_loss�:�@

error_R�E8?

learning_rate_1?�7��k�I       6%�	k��C���A�*;


total_loss�A

error_R��V?

learning_rate_1?�7U�c!I       6%�	�ƅC���A�*;


total_losswl�@

error_RR#C?

learning_rate_1?�7Lg��I       6%�	�C���A�*;


total_loss�o�@

error_R�c?

learning_rate_1?�7�I@�I       6%�	L�C���A�*;


total_lossi[l@

error_R
MI?

learning_rate_1?�7���I       6%�	G��C���A�*;


total_loss��@

error_R�gI?

learning_rate_1?�7ŏ�4I       6%�	�چC���A�*;


total_loss/B�@

error_RʩJ?

learning_rate_1?�7s���I       6%�	�/�C���A�*;


total_loss	(A

error_R$yO?

learning_rate_1?�7hFSAI       6%�	瑇C���A�*;


total_lossT��@

error_Rm�F?

learning_rate_1?�7���&I       6%�	�ڇC���A�*;


total_lossƧA

error_RF?

learning_rate_1?�7p!j�I       6%�	��C���A�*;


total_loss���@

error_R/SA?

learning_rate_1?�7y���I       6%�	]_�C���A�*;


total_loss�˻@

error_R�V?

learning_rate_1?�7�}/I       6%�	���C���A�*;


total_loss���@

error_R=�J?

learning_rate_1?�7��W�I       6%�	��C���A�*;


total_loss�L�@

error_Rl�R?

learning_rate_1?�7 ���I       6%�	+*�C���A�*;


total_loss�&�@

error_R�J?

learning_rate_1?�7�7I       6%�	�p�C���A�*;


total_loss!�@

error_R�H?

learning_rate_1?�7z���I       6%�	���C���A�*;


total_loss��@

error_R;�F?

learning_rate_1?�7��PuI       6%�	G��C���A�*;


total_lossJI�@

error_R��F?

learning_rate_1?�7��=|I       6%�	W>�C���A�*;


total_loss�P@

error_R�]N?

learning_rate_1?�7�)��I       6%�	;��C���A�*;


total_loss=�_@

error_R�Y?

learning_rate_1?�7C��I       6%�	�ȊC���A�*;


total_lossSh�@

error_R�IE?

learning_rate_1?�754�I       6%�	��C���A�*;


total_lossa]�@

error_R��E?

learning_rate_1?�7���>I       6%�	_b�C���A�*;


total_loss���@

error_RjV?

learning_rate_1?�7��I       6%�	���C���A�*;


total_loss�l�@

error_R_ U?

learning_rate_1?�7��Q�I       6%�	���C���A�*;


total_loss���@

error_R|f?

learning_rate_1?�7!��I       6%�	�;�C���A�*;


total_lossM��@

error_R3h?

learning_rate_1?�7}�XI       6%�	�~�C���A�*;


total_loss࿮@

error_RH?

learning_rate_1?�7����I       6%�	QɌC���A�*;


total_lossaث@

error_RvEL?

learning_rate_1?�7X��^I       6%�	��C���A�*;


total_loss��A

error_R�Dn?

learning_rate_1?�7�ƧI       6%�	[�C���A�*;


total_lossz �@

error_RԈ=?

learning_rate_1?�7��Z�I       6%�	���C���A�*;


total_lossS{�@

error_R�SR?

learning_rate_1?�7��C�I       6%�	�C���A�*;


total_loss̕�@

error_R7-G?

learning_rate_1?�7��-�I       6%�	�)�C���A�*;


total_loss
�A

error_R��@?

learning_rate_1?�7��wI       6%�	�s�C���A�*;


total_loss悃@

error_R�mV?

learning_rate_1?�7n6NLI       6%�	X��C���A�*;


total_loss�n�@

error_R�+=?

learning_rate_1?�7!-��I       6%�	���C���A�*;


total_loss��@

error_R�??

learning_rate_1?�7nH�II       6%�	t@�C���A�*;


total_lossV�@

error_R�CH?

learning_rate_1?�7�{�I       6%�	Ć�C���A�*;


total_loss:��@

error_R��K?

learning_rate_1?�7ݯ�I       6%�	]ɏC���A�*;


total_loss�0�@

error_R� O?

learning_rate_1?�7�c�>I       6%�	�C���A�*;


total_loss���@

error_R�,]?

learning_rate_1?�7�J��I       6%�	3R�C���A�*;


total_loss%(�@

error_R��O?

learning_rate_1?�7A�^�I       6%�	���C���A�*;


total_loss��@

error_RM�J?

learning_rate_1?�7Y� CI       6%�	�ېC���A�*;


total_lossچ�@

error_Rs�P?

learning_rate_1?�7��G>I       6%�	~�C���A�*;


total_loss�&A

error_RWM?

learning_rate_1?�7 i&I       6%�	�e�C���A�*;


total_lossmA

error_RHT?

learning_rate_1?�7.~�zI       6%�	l��C���A�*;


total_loss!�A

error_R�@?

learning_rate_1?�7��B�I       6%�	��C���A�*;


total_loss� �@

error_R��B?

learning_rate_1?�7��( I       6%�	�-�C���A�*;


total_loss���@

error_RlB?

learning_rate_1?�7��r:I       6%�	t�C���A�*;


total_lossS2�@

error_R�lY?

learning_rate_1?�7�w|�I       6%�	Ҹ�C���A�*;


total_loss��@

error_RN�J?

learning_rate_1?�7��I       6%�	���C���A�*;


total_loss�ɞ@

error_R�/f?

learning_rate_1?�7���+I       6%�	�D�C���A�*;


total_loss�E�@

error_R�6P?

learning_rate_1?�7�I       6%�	���C���A�*;


total_loss�*g@

error_R�	[?

learning_rate_1?�7���I       6%�	�ГC���A�*;


total_loss���@

error_RZ�=?

learning_rate_1?�7~�RI       6%�	�C���A�*;


total_lossX
�@

error_RaWI?

learning_rate_1?�7��κI       6%�	�W�C���A�*;


total_lossq\�@

error_R��A?

learning_rate_1?�7���[I       6%�	���C���A�*;


total_losst�@

error_R��L?

learning_rate_1?�7�MI       6%�	��C���A�*;


total_lossRՄ@

error_R$~:?

learning_rate_1?�7�i��I       6%�	j*�C���A�*;


total_lossA(�@

error_Rd8M?

learning_rate_1?�7#�e=I       6%�	�n�C���A�*;


total_lossͭ�@

error_Rl�Z?

learning_rate_1?�7 7II       6%�	6��C���A�*;


total_lossL A

error_R�M?

learning_rate_1?�7��DrI       6%�	>��C���A�*;


total_loss�ɍ@

error_R�<M?

learning_rate_1?�73�k�I       6%�	A�C���A�*;


total_loss5��@

error_Rs>M?

learning_rate_1?�7ގyI       6%�	���C���A�*;


total_loss�f A

error_RIP?

learning_rate_1?�7c�@�I       6%�	�ǖC���A�*;


total_lossS��@

error_R�W?

learning_rate_1?�7�VnWI       6%�	��C���A�*;


total_lossB��@

error_R�	I?

learning_rate_1?�7�?}�I       6%�	�|�C���A�*;


total_loss;O�@

error_RK\?

learning_rate_1?�7�(�`I       6%�	�×C���A�*;


total_loss��@

error_R�ZM?

learning_rate_1?�7�J�I       6%�	a�C���A�*;


total_losse�@

error_R�=?

learning_rate_1?�7�T��I       6%�	Ui�C���A�*;


total_loss�F�@

error_R��a?

learning_rate_1?�7�~�I       6%�	���C���A�*;


total_lossi۩@

error_R��Y?

learning_rate_1?�7U�c�I       6%�	s �C���A�*;


total_loss1m�@

error_R��d?

learning_rate_1?�7T)G;I       6%�	�H�C���A�*;


total_loss�F�@

error_R��a?

learning_rate_1?�7�±�I       6%�	Ћ�C���A�*;


total_loss#�c@

error_Rx�??

learning_rate_1?�7��E�I       6%�	�͙C���A�*;


total_lossh�@

error_R�uN?

learning_rate_1?�7�V)I       6%�	��C���A�*;


total_loss��A

error_R
�R?

learning_rate_1?�7�tGOI       6%�	�W�C���A�*;


total_loss���@

error_R�
I?

learning_rate_1?�7�=	kI       6%�	���C���A�*;


total_loss�S�@

error_RI�B?

learning_rate_1?�7��I       6%�	�ߚC���A�*;


total_loss<�A

error_R�JY?

learning_rate_1?�7y33NI       6%�	� �C���A�*;


total_loss`v�@

error_R��Z?

learning_rate_1?�7p,^EI       6%�	af�C���A�*;


total_lossS��@

error_R6�E?

learning_rate_1?�7W���I       6%�	Ԩ�C���A�*;


total_loss�z�@

error_R�U?

learning_rate_1?�72X�I       6%�	��C���A�*;


total_lossn�@

error_RdL?

learning_rate_1?�7CֵI       6%�	#;�C���A�*;


total_loss�.�@

error_Rf;?

learning_rate_1?�7�`��I       6%�	N��C���A�*;


total_loss��@

error_R�|P?

learning_rate_1?�7���I       6%�	8ҜC���A�*;


total_loss�ڛ@

error_R�	F?

learning_rate_1?�7���I       6%�	'�C���A�*;


total_loss���@

error_Rq=?

learning_rate_1?�7~�\�I       6%�	�r�C���A�*;


total_loss|��@

error_R�YT?

learning_rate_1?�7��I       6%�	s��C���A�*;


total_loss��@

error_R��O?

learning_rate_1?�7�SI       6%�	H
�C���A�*;


total_lossS>�@

error_R��Q?

learning_rate_1?�7U�I       6%�	U�C���A�*;


total_loss� �@

error_R߬E?

learning_rate_1?�7:;��I       6%�	L��C���A�*;


total_loss��@

error_R;D?

learning_rate_1?�7۷JAI       6%�	0ݞC���A�*;


total_loss���@

error_R
�T?

learning_rate_1?�7�*��I       6%�	��C���A�*;


total_loss��@

error_R�W`?

learning_rate_1?�7i1�,I       6%�	e�C���A�*;


total_loss���@

error_RcF?

learning_rate_1?�7i�I       6%�	=��C���A�*;


total_loss��A

error_R3DM?

learning_rate_1?�7Q��LI       6%�	��C���A�*;


total_loss[�@

error_Rד<?

learning_rate_1?�7.��I       6%�	 1�C���A�*;


total_loss�~A

error_R�K?

learning_rate_1?�71�I       6%�	�w�C���A�*;


total_lossO��@

error_Rvp?

learning_rate_1?�7i��YI       6%�	���C���A�*;


total_lossř�@

error_R{d?

learning_rate_1?�7�EI       6%�	� �C���A�*;


total_loss�Q�@

error_R��Q?

learning_rate_1?�7�i[�I       6%�	sG�C���A�*;


total_loss��@

error_R�|P?

learning_rate_1?�7dl3�I       6%�	�C���A�*;


total_loss�ض@

error_R��W?

learning_rate_1?�7�7j/I       6%�	�ҡC���A�*;


total_loss�
�@

error_R�yb?

learning_rate_1?�7es_�I       6%�	f�C���A�*;


total_loss{:�@

error_RȩO?

learning_rate_1?�7e��I       6%�	�]�C���A�*;


total_loss��@

error_R��K?

learning_rate_1?�7�mdPI       6%�	���C���A�*;


total_loss��@

error_R;O?

learning_rate_1?�7��I       6%�	j�C���A�*;


total_loss�� A

error_R�{`?

learning_rate_1?�7��'SI       6%�	t4�C���A�*;


total_loss�b�@

error_R�"B?

learning_rate_1?�7�6UHI       6%�	\{�C���A�*;


total_loss.�@

error_R��K?

learning_rate_1?�7M�gI       6%�	ģC���A�*;


total_loss���@

error_R�|L?

learning_rate_1?�7���I       6%�	T	�C���A�*;


total_loss�G�@

error_R�@??

learning_rate_1?�7)J�8I       6%�	�U�C���A�*;


total_loss��@

error_R��O?

learning_rate_1?�7��I       6%�	���C���A�*;


total_loss
��@

error_R��Z?

learning_rate_1?�73D`I       6%�	#�C���A�*;


total_loss�w�@

error_R=lS?

learning_rate_1?�7�R�I       6%�	y-�C���A�*;


total_lossXA�@

error_R�B?

learning_rate_1?�7�pj(I       6%�	�z�C���A�*;


total_loss�M�@

error_Rv�X?

learning_rate_1?�7�kȴI       6%�	��C���A�*;


total_lossH��@

error_Ro�Q?

learning_rate_1?�7*�L;I       6%�	��C���A�*;


total_loss�@�@

error_R��L?

learning_rate_1?�77���I       6%�	�X�C���A�*;


total_loss�u�@

error_RE�U?

learning_rate_1?�7�sWmI       6%�	㝦C���A�*;


total_loss���@

error_R�:W?

learning_rate_1?�7��&oI       6%�	�C���A�*;


total_loss���@

error_R�Ad?

learning_rate_1?�7���7I       6%�	_'�C���A�*;


total_loss��J@

error_R�cS?

learning_rate_1?�7��OI       6%�	4��C���A�*;


total_lossZ��@

error_R�L?

learning_rate_1?�7�b�I       6%�	�ԧC���A�*;


total_loss��@

error_Ri\\?

learning_rate_1?�7t��I       6%�	��C���A�*;


total_losscr�@

error_Rc~Q?

learning_rate_1?�7��,�I       6%�	�X�C���A�*;


total_loss�/�@

error_R;�M?

learning_rate_1?�7j+|I       6%�	��C���A�*;


total_lossc�@

error_R�'B?

learning_rate_1?�7_�I       6%�	$ߨC���A�*;


total_loss�1�@

error_R��Y?

learning_rate_1?�7��5I       6%�	Y!�C���A�*;


total_loss�Ӈ@

error_R_�I?

learning_rate_1?�7Y(�VI       6%�	�d�C���A�*;


total_loss`�@

error_RFbL?

learning_rate_1?�7� V�I       6%�	n��C���A�*;


total_loss/7�@

error_R�bX?

learning_rate_1?�7��NnI       6%�	B��C���A�*;


total_loss<��@

error_R
�G?

learning_rate_1?�7wK˃I       6%�	�.�C���A�*;


total_loss�*A

error_R2e?

learning_rate_1?�7':��I       6%�	�{�C���A�*;


total_loss�@

error_R��B?

learning_rate_1?�7ep¨I       6%�	�êC���A�*;


total_loss���@

error_R�OX?

learning_rate_1?�7��I       6%�	��C���A�*;


total_loss�A

error_R��W?

learning_rate_1?�7��I       6%�	�J�C���A�*;


total_loss�@

error_Rc�D?

learning_rate_1?�7ĜI       6%�	���C���A�*;


total_loss\>�@

error_RxU?

learning_rate_1?�7��J_I       6%�	cܫC���A�*;


total_loss��@

error_RL??

learning_rate_1?�7*ĆI       6%�	O$�C���A�*;


total_loss�A

error_R1�R?

learning_rate_1?�7�lܙI       6%�		l�C���A�*;


total_lossŖ�@

error_R6UA?

learning_rate_1?�7��hAI       6%�	���C���A�*;


total_loss�u�@

error_R��M?

learning_rate_1?�7&_�bI       6%�	��C���A�*;


total_lossL�g@

error_R8???

learning_rate_1?�7�%��I       6%�	4�C���A�*;


total_lossLa�@

error_R�PE?

learning_rate_1?�7����I       6%�	[w�C���A�*;


total_loss���@

error_R��Q?

learning_rate_1?�7�;��I       6%�	T��C���A�*;


total_loss���@

error_R��L?

learning_rate_1?�7�<�LI       6%�	��C���A�*;


total_loss�b�@

error_Rv�W?

learning_rate_1?�7�t�uI       6%�	8@�C���A�*;


total_loss�4�@

error_R��R?

learning_rate_1?�7'� %I       6%�	ł�C���A�*;


total_loss�0�@

error_R V?

learning_rate_1?�7A��I       6%�	�ǮC���A�*;


total_loss�]�@

error_R_@?

learning_rate_1?�7XcphI       6%�	��C���A�*;


total_loss�:�@

error_R��K?

learning_rate_1?�7ǋ��I       6%�	(P�C���A�*;


total_lossH4�@

error_R�D?

learning_rate_1?�7��� I       6%�	R��C���A�*;


total_loss��@

error_R
\F?

learning_rate_1?�77݉TI       6%�	�կC���A�*;


total_loss2�@

error_R�hH?

learning_rate_1?�7�n5�I       6%�	2�C���A�*;


total_lossޣ�@

error_R=�C?

learning_rate_1?�7��tI       6%�	�Z�C���A�*;


total_loss,4�@

error_R�hY?

learning_rate_1?�7�y�I       6%�	��C���A�*;


total_lossSb�@

error_RQ�;?

learning_rate_1?�7{;.�I       6%�	&�C���A�*;


total_losso�@

error_R�a5?

learning_rate_1?�7$��I       6%�	�4�C���A�*;


total_loss�@

error_R��H?

learning_rate_1?�7���I       6%�	fy�C���A�*;


total_lossS�@

error_Rs�V?

learning_rate_1?�7 K�I       6%�	h��C���A�*;


total_loss��@

error_R#IJ?

learning_rate_1?�7���I       6%�	�C���A�*;


total_loss�
�@

error_R4�M?

learning_rate_1?�75sbI       6%�	�J�C���A�*;


total_loss
�v@

error_RW_?

learning_rate_1?�7
�^I       6%�	씲C���A�*;


total_loss�e�@

error_R��L?

learning_rate_1?�7��+I       6%�	ZײC���A�*;


total_loss|�@

error_RW�F?

learning_rate_1?�7�ޒI       6%�	�C���A�*;


total_loss0�@

error_R AV?

learning_rate_1?�7��I       6%�	�c�C���A�*;


total_loss�A�@

error_R4�5?

learning_rate_1?�7�ӀI       6%�	���C���A�*;


total_loss�d�@

error_R69S?

learning_rate_1?�7�z��I       6%�	,�C���A�*;


total_loss ��@

error_R�9K?

learning_rate_1?�7(�3�I       6%�	�<�C���A�*;


total_loss��v@

error_RI�;?

learning_rate_1?�7^�|I       6%�	��C���A�*;


total_lossڢa@

error_R �<?

learning_rate_1?�7�ShI       6%�	ǴC���A�*;


total_loss	��@

error_R�`Y?

learning_rate_1?�7�!!�I       6%�	J�C���A�*;


total_loss\�@

error_R��H?

learning_rate_1?�7N-�vI       6%�	7I�C���A�*;


total_loss<��@

error_R�T?

learning_rate_1?�7��f3I       6%�	Ǌ�C���A�*;


total_lossEL�@

error_R�XP?

learning_rate_1?�7��!I       6%�	�͵C���A�*;


total_lossx��@

error_R� U?

learning_rate_1?�7;�>�I       6%�	��C���A�*;


total_loss	jA

error_R:�P?

learning_rate_1?�7�ZfI       6%�	�Y�C���A�*;


total_loss�Q�@

error_R��I?

learning_rate_1?�7�!`�I       6%�	䣶C���A�*;


total_loss��@

error_R��L?

learning_rate_1?�7��v�I       6%�	��C���A�*;


total_loss�Jn@

error_RI�8?

learning_rate_1?�7��TuI       6%�	M�C���A�*;


total_loss�ߩ@

error_Ri�G?

learning_rate_1?�7�q�I       6%�	]ɷC���A�*;


total_lossz�@

error_R�<??

learning_rate_1?�7�tHLI       6%�	V�C���A�*;


total_loss���@

error_RxT_?

learning_rate_1?�7|�ĳI       6%�	S�C���A�*;


total_loss���@

error_R1g?

learning_rate_1?�7��|)I       6%�	<��C���A�*;


total_loss���@

error_R!W?

learning_rate_1?�7>#jbI       6%�	|�C���A�*;


total_lossk]@

error_R�D?

learning_rate_1?�7�"�I       6%�	�*�C���A�*;


total_loss�Wh@

error_Rf�@?

learning_rate_1?�7K?RhI       6%�	�y�C���A�*;


total_loss	G�@

error_R�=H?

learning_rate_1?�7�uqEI       6%�	FĹC���A�*;


total_loss!1A

error_R�Hd?

learning_rate_1?�7O�C�I       6%�	��C���A�*;


total_loss�B�@

error_R)cZ?

learning_rate_1?�7]��I       6%�	�S�C���A�*;


total_loss�Z�@

error_R�tW?

learning_rate_1?�7B1_�I       6%�	ܚ�C���A�*;


total_loss��A

error_R�G?

learning_rate_1?�7��E"I       6%�	��C���A�*;


total_lossv;�@

error_R�r[?

learning_rate_1?�7��I       6%�	)�C���A�*;


total_loss r�@

error_R��K?

learning_rate_1?�7O�r�I       6%�	3k�C���A�*;


total_lossTi�@

error_R��H?

learning_rate_1?�7:��I       6%�	�C���A�*;


total_lossy3�@

error_R��O?

learning_rate_1?�79���I       6%�	���C���A�*;


total_loss��@

error_R;�@?

learning_rate_1?�7-ʷI       6%�	�C�C���A�*;


total_loss�<�@

error_R)�Z?

learning_rate_1?�7�.�bI       6%�	,��C���A�*;


total_loss=[U@

error_R�#B?

learning_rate_1?�7E��'I       6%�	8мC���A�*;


total_loss�l�@

error_R:�Z?

learning_rate_1?�7�HrI       6%�	8�C���A�*;


total_loss���@

error_Rl�@?

learning_rate_1?�7���I       6%�	ke�C���A�*;


total_loss�w
A

error_R
LN?

learning_rate_1?�7�q�9I       6%�	Z��C���A�*;


total_loss�i�@

error_R��U?

learning_rate_1?�7~mB�I       6%�	���C���A�*;


total_loss4� A

error_Rl�L?

learning_rate_1?�7q�I       6%�	�.�C���A�*;


total_loss�7�@

error_R@T?

learning_rate_1?�7)a�I       6%�	�o�C���A�*;


total_loss)��@

error_R��Z?

learning_rate_1?�7 q� I       6%�	B��C���A�*;


total_lossS�@

error_Rn\Y?

learning_rate_1?�7dc��I       6%�	��C���A�*;


total_lossa�@

error_Rs B?

learning_rate_1?�7ȜSI       6%�	�8�C���A�*;


total_loss|�@

error_R�%\?

learning_rate_1?�7��CI       6%�	|�C���A�*;


total_loss��@

error_RWj9?

learning_rate_1?�7?\�VI       6%�	�¿C���A�*;


total_loss���@

error_R��H?

learning_rate_1?�7��lMI       6%�	��C���A�*;


total_loss޳@

error_R�5?

learning_rate_1?�7
�?iI       6%�	K�C���A�*;


total_losso�@

error_Rf�X?

learning_rate_1?�7r��#I       6%�	��C���A�*;


total_loss��@

error_R�E?

learning_rate_1?�7#�I       6%�	���C���A�*;


total_losss6�@

error_R�`?

learning_rate_1?�7.�I       6%�	�"�C���A�*;


total_loss��@

error_RiMO?

learning_rate_1?�7���I       6%�	|l�C���A�*;


total_loss a�@

error_R/>c?

learning_rate_1?�7!�mjI       6%�	f��C���A�*;


total_loss�*�@

error_R1~@?

learning_rate_1?�7fQ{PI       6%�	���C���A�*;


total_loss�_�@

error_R��[?

learning_rate_1?�7����I       6%�	�;�C���A�*;


total_loss<��@

error_RwT?

learning_rate_1?�7ju�mI       6%�	}��C���A�*;


total_loss���@

error_R_iB?

learning_rate_1?�7Y��I       6%�	+��C���A�*;


total_lossҧ�@

error_RZ+T?

learning_rate_1?�7�ԑI       6%�	��C���A�*;


total_losss%�@

error_R��R?

learning_rate_1?�7!<�I       6%�	�_�C���A�*;


total_lossVo@

error_R�%>?

learning_rate_1?�7~` �I       6%�	��C���A�*;


total_loss�=�@

error_R�3X?

learning_rate_1?�7+���I       6%�	���C���A�*;


total_losst��@

error_R�Z?

learning_rate_1?�7�zO	I       6%�	?=�C���A�*;


total_loss��A

error_R�?A?

learning_rate_1?�7��v�I       6%�	�C���A�*;


total_lossJOBA

error_RsnI?

learning_rate_1?�78�)XI       6%�	���C���A�*;


total_loss\q�@

error_R�XE?

learning_rate_1?�7��i�I       6%�	�C���A�*;


total_loss��@

error_Rʫ[?

learning_rate_1?�7f�@�I       6%�	�`�C���A�*;


total_loss��@

error_R6�J?

learning_rate_1?�7QiT\I       6%�	Y��C���A�*;


total_lossg� A

error_R`�L?

learning_rate_1?�7��zKI       6%�	L��C���A�*;


total_loss�j�@

error_R{�<?

learning_rate_1?�7~E�fI       6%�	�7�C���A�*;


total_loss���@

error_R�1Y?

learning_rate_1?�7��I       6%�	�~�C���A�*;


total_loss�7�@

error_R��B?

learning_rate_1?�7Fז�I       6%�	#��C���A�*;


total_lossjx�@

error_R�`K?

learning_rate_1?�7^N��I       6%�		�C���A�*;


total_loss�#�@

error_RitE?

learning_rate_1?�7%˓�I       6%�	]s�C���A�*;


total_loss�q�@

error_R��N?

learning_rate_1?�7P���I       6%�	B��C���A�*;


total_loss�Ұ@

error_R*�Y?

learning_rate_1?�7��I       6%�	H��C���A�*;


total_loss:C�@

error_REZ?

learning_rate_1?�7+�N�I       6%�	�A�C���A�*;


total_loss���@

error_R�V?

learning_rate_1?�7�f�I       6%�	֌�C���A�*;


total_lossl�~@

error_R�O?

learning_rate_1?�7��VI       6%�	��C���A�*;


total_loss�@

error_R��P?

learning_rate_1?�7�*�I       6%�	*"�C���A�*;


total_lossN�]@

error_R�hE?

learning_rate_1?�7��tI       6%�	7l�C���A�*;


total_loss]�@

error_R�1l?

learning_rate_1?�7 �JI       6%�	q��C���A�*;


total_lossNH�@

error_R$�O?

learning_rate_1?�76�I       6%�	
��C���A�*;


total_loss���@

error_R�DW?

learning_rate_1?�7ͭCI       6%�	�E�C���A�*;


total_lossL��@

error_Rf#C?

learning_rate_1?�79>�QI       6%�	��C���A�*;


total_loss?�A

error_R�?>?

learning_rate_1?�7��o�I       6%�	���C���A�*;


total_loss8��@

error_Rc�I?

learning_rate_1?�7�>q�I       6%�	��C���A�*;


total_lossS�A

error_R\oP?

learning_rate_1?�7��\I       6%�	�\�C���A�*;


total_lossh3�@

error_Rl%N?

learning_rate_1?�7���3I       6%�	Y��C���A�*;


total_lossz��@

error_RanM?

learning_rate_1?�7w�
I       6%�	���C���A�*;


total_lossXʑ@

error_R,�N?

learning_rate_1?�72�z�I       6%�	z;�C���A�*;


total_lossT�}@

error_R1�C?

learning_rate_1?�7��a�I       6%�	���C���A�*;


total_loss��x@

error_RjU?

learning_rate_1?�7��%�I       6%�	E��C���A�*;


total_loss@ʨ@

error_R�Q?

learning_rate_1?�7�=6rI       6%�	�+�C���A�*;


total_loss
B�@

error_RMT?

learning_rate_1?�7��aGI       6%�	>v�C���A�*;


total_loss�(�@

error_RXtY?

learning_rate_1?�7��]I       6%�	ؼ�C���A�*;


total_loss���@

error_Rle@?

learning_rate_1?�7޷�I       6%�	��C���A�*;


total_loss��@

error_R��P?

learning_rate_1?�7˕�I       6%�	!H�C���A�*;


total_loss�m�@

error_RII?

learning_rate_1?�7�HI       6%�	��C���A�*;


total_loss�f	A

error_Ra�[?

learning_rate_1?�7��6&I       6%�	��C���A�*;


total_loss���@

error_R�(S?

learning_rate_1?�7n"I�I       6%�	&�C���A�*;


total_loss�`�@

error_Rq;?

learning_rate_1?�7c�i`I       6%�	u`�C���A�*;


total_loss\Y�@

error_R�D?

learning_rate_1?�7xk�I       6%�	���C���A�*;


total_lossҦ�@

error_R�T?

learning_rate_1?�7�@�I       6%�	Z��C���A�*;


total_loss�O�@

error_RH�V?

learning_rate_1?�7x��WI       6%�	�-�C���A�*;


total_loss2C�@

error_R��L?

learning_rate_1?�7]~�I       6%�	Ft�C���A�*;


total_loss)�@

error_Ra-X?

learning_rate_1?�7��3I       6%�	���C���A�*;


total_lossf�y@

error_R�#b?

learning_rate_1?�7ϵ�I       6%�	 �C���A�*;


total_loss>��@

error_R]?

learning_rate_1?�7�X~�I       6%�	�B�C���A�*;


total_loss�8�@

error_R,~C?

learning_rate_1?�7�N&I       6%�	>��C���A�*;


total_loss���@

error_R�[?

learning_rate_1?�7����I       6%�	7��C���A�*;


total_lossm:�@

error_R��I?

learning_rate_1?�7:�B�I       6%�	f�C���A�*;


total_lossj�@

error_R��P?

learning_rate_1?�7���RI       6%�	}U�C���A�*;


total_loss.z�@

error_R.G@?

learning_rate_1�<�7���yI       6%�	��C���A�*;


total_loss���@

error_R?&G?

learning_rate_1�<�7>��I       6%�	
��C���A�*;


total_loss�s@

error_RR6O?

learning_rate_1�<�7��~�I       6%�	!�C���A�*;


total_loss$N�@

error_R�dU?

learning_rate_1�<�7
�\4I       6%�	��C���A�*;


total_loss��@

error_R�X?

learning_rate_1�<�7�;��I       6%�	S�C���A�*;


total_lossW�@

error_R �Y?

learning_rate_1�<�779�I       6%�	���C���A�*;


total_loss�@

error_R�S?

learning_rate_1�<�7|���I       6%�	���C���A�*;


total_lossĴ�@

error_R�XK?

learning_rate_1�<�7��Z�I       6%�	VA�C���A�*;


total_lossAU�@

error_R��Q?

learning_rate_1�<�7+��I       6%�	���C���A�*;


total_lossj�@

error_R�VG?

learning_rate_1�<�7�hNI       6%�	��C���A�*;


total_loss��@

error_R��=?

learning_rate_1�<�7W��I       6%�	�%�C���A�*;


total_loss*�@

error_R)W?

learning_rate_1�<�7C���I       6%�	!g�C���A�*;


total_loss|�@

error_R8(K?

learning_rate_1�<�7H��I       6%�	۪�C���A�*;


total_loss���@

error_R�J?

learning_rate_1�<�7���_I       6%�	���C���A�*;


total_loss8�A

error_R�"H?

learning_rate_1�<�7)��I       6%�	�.�C���A�*;


total_loss��@

error_R��G?

learning_rate_1�<�7t���I       6%�	r�C���A�*;


total_loss/�@

error_R��V?

learning_rate_1�<�7ͦ8dI       6%�	���C���A�*;


total_loss�s0A

error_R�V??

learning_rate_1�<�7��;�I       6%�	&��C���A�*;


total_lossX^A

error_RdG>?

learning_rate_1�<�7UH�I       6%�	x<�C���A�*;


total_loss���@

error_R��W?

learning_rate_1�<�7�{7}I       6%�	��C���A�*;


total_loss��@

error_R�R?

learning_rate_1�<�7�hGI       6%�	���C���A�*;


total_lossi��@

error_RVK?

learning_rate_1�<�7���mI       6%�	��C���A�*;


total_loss��@

error_R�$O?

learning_rate_1�<�7��*tI       6%�	�E�C���A�*;


total_lossЩ@

error_R�+\?

learning_rate_1�<�7�ۑwI       6%�	E��C���A�*;


total_loss���@

error_R:\?

learning_rate_1�<�7�2mI       6%�	e��C���A�*;


total_loss�\�@

error_R�L?

learning_rate_1�<�7}%�2I       6%�	��C���A�*;


total_loss��@

error_Rj�N?

learning_rate_1�<�7�x-I       6%�	�_�C���A�*;


total_loss��@

error_R2?

learning_rate_1�<�7�RI       6%�	���C���A�*;


total_loss���@

error_R
P?

learning_rate_1�<�7��%I       6%�	L �C���A�*;


total_loss1�p@

error_R�Y?

learning_rate_1�<�7S��DI       6%�	�F�C���A�*;


total_loss�{�@

error_R��I?

learning_rate_1�<�76ol[I       6%�	���C���A�*;


total_lossݓ�@

error_R +U?

learning_rate_1�<�7��2I       6%�	���C���A�*;


total_loss��A

error_Rq�K?

learning_rate_1�<�7m��0I       6%�	� �C���A�*;


total_loss)��@

error_R�3T?

learning_rate_1�<�7�� [I       6%�	b�C���A�*;


total_loss���@

error_R�K?

learning_rate_1�<�7�4F�I       6%�	���C���A�*;


total_lossF��@

error_R��F?

learning_rate_1�<�7G7�EI       6%�	���C���A�*;


total_loss�*�@

error_R�Rd?

learning_rate_1�<�7��I       6%�	N.�C���A�*;


total_loss���@

error_RE�8?

learning_rate_1�<�7�t,�I       6%�	Ls�C���A�*;


total_loss`c�@

error_R&�N?

learning_rate_1�<�7J62iI       6%�	ٳ�C���A�*;


total_lossRf�@

error_R��U?

learning_rate_1�<�7����I       6%�	F��C���A�*;


total_lossvږ@

error_RT=?

learning_rate_1�<�7��I       6%�	8�C���A�*;


total_loss�ſ@

error_R%�F?

learning_rate_1�<�7�敓I       6%�	\z�C���A�*;


total_loss��e@

error_R�M?

learning_rate_1�<�7jT�_I       6%�	���C���A�*;


total_lossq5�@

error_R��T?

learning_rate_1�<�7���I       6%�	��C���A�*;


total_loss�ٵ@

error_R�N?

learning_rate_1�<�74��I       6%�	dd�C���A�*;


total_loss4y�@

error_R�ZA?

learning_rate_1�<�7��p�I       6%�	3��C���A�*;


total_losszc�@

error_RD�??

learning_rate_1�<�7�yI       6%�	x��C���A�*;


total_loss$Z�@

error_R��Z?

learning_rate_1�<�7�9A�I       6%�	�`�C���A�*;


total_loss�g�@

error_RfSD?

learning_rate_1�<�7��i,I       6%�	���C���A�*;


total_loss%-�@

error_R��E?

learning_rate_1�<�7��xI       6%�	���C���A�*;


total_loss@�r@

error_R#/G?

learning_rate_1�<�7�+MI       6%�	�X�C���A�*;


total_loss#St@

error_R��P?

learning_rate_1�<�77�sjI       6%�	v��C���A�*;


total_loss��@

error_R/�O?

learning_rate_1�<�7m֊_I       6%�	R��C���A�*;


total_lossq��@

error_R��J?

learning_rate_1�<�7d��I       6%�	�Q�C���A�*;


total_loss�չ@

error_R��U?

learning_rate_1�<�7>�ȵI       6%�	��C���A�*;


total_loss�4�@

error_R�hM?

learning_rate_1�<�7�L��I       6%�	��C���A�*;


total_loss6l�@

error_R{uE?

learning_rate_1�<�7NY��I       6%�	�#�C���A�*;


total_lossmL�@

error_R��C?

learning_rate_1�<�7u�w9I       6%�	�h�C���A�*;


total_loss�y�@

error_R�$8?

learning_rate_1�<�7Ç��I       6%�	2��C���A�*;


total_loss$"�@

error_Rl�C?

learning_rate_1�<�7D�7�I       6%�	���C���A�*;


total_loss*�A

error_R�U?

learning_rate_1�<�7��=�I       6%�	uU�C���A�*;


total_loss/@�@

error_R�lQ?

learning_rate_1�<�7�}I3I       6%�	8��C���A�*;


total_lossEr�@

error_R7�T?

learning_rate_1�<�7" ��I       6%�	��C���A�*;


total_loss�A�@

error_R��`?

learning_rate_1�<�7�C�eI       6%�	W)�C���A�*;


total_loss�ٚ@

error_R�oJ?

learning_rate_1�<�7�{��I       6%�	��C���A�*;


total_loss���@

error_R�I?

learning_rate_1�<�7�6#
I       6%�	>��C���A�*;


total_loss|��@

error_Rd�6?

learning_rate_1�<�7���I       6%�	A�C���A�*;


total_lossn�@

error_R� N?

learning_rate_1�<�7҂D�I       6%�	o��C���A�*;


total_loss}5x@

error_Rs�S?

learning_rate_1�<�7���5I       6%�	���C���A�*;


total_loss��x@

error_R�C?

learning_rate_1�<�7�D=�I       6%�	��C���A�*;


total_loss��@

error_R�R?

learning_rate_1�<�7���I       6%�	lh�C���A�*;


total_loss�U�@

error_R�KT?

learning_rate_1�<�7�HI       6%�	Y��C���A�*;


total_loss<��@

error_R$lR?

learning_rate_1�<�7Aq?wI       6%�	&��C���A�*;


total_loss��@

error_RdrJ?

learning_rate_1�<�7�0q~I       6%�	,5�C���A�*;


total_loss�\�@

error_R�2A?

learning_rate_1�<�7�@I       6%�	}y�C���A�*;


total_lossP�@

error_R@S?

learning_rate_1�<�74���I       6%�	u��C���A�*;


total_loss�?�@

error_R��T?

learning_rate_1�<�7��$I       6%�	�C���A�*;


total_lossP#�@

error_R*T?

learning_rate_1�<�7�k_�I       6%�	2Z�C���A�*;


total_loss,m�@

error_RJ�H?

learning_rate_1�<�7~VJI       6%�	'��C���A�*;


total_loss���@

error_RZOC?

learning_rate_1�<�7X�LI       6%�	���C���A�*;


total_loss4L�@

error_R�UH?

learning_rate_1�<�7�j-RI       6%�	d9�C���A�*;


total_lossEq�@

error_RT_X?

learning_rate_1�<�7�K�XI       6%�	��C���A�*;


total_lossD`~@

error_R`M?

learning_rate_1�<�7H�U�I       6%�	���C���A�*;


total_loss�U�@

error_R�;?

learning_rate_1�<�7�or#I       6%�	��C���A�*;


total_loss<կ@

error_R�pD?

learning_rate_1�<�7nb�I       6%�	�\�C���A�*;


total_loss� z@

error_R�R8?

learning_rate_1�<�7��&�I       6%�	|��C���A�*;


total_lossJ�@

error_R$4@?

learning_rate_1�<�7`uj*I       6%�	���C���A�*;


total_lossr��@

error_R�Ef?

learning_rate_1�<�7)��I       6%�	�4�C���A�*;


total_loss[o�@

error_R��G?

learning_rate_1�<�7��}I       6%�	��C���A�*;


total_lossQG�@

error_R�K?

learning_rate_1�<�7E�II       6%�	���C���A�*;


total_loss���@

error_R�<?

learning_rate_1�<�7��}AI       6%�	l�C���A�*;


total_loss ��@

error_R�B?

learning_rate_1�<�7s-�I       6%�	MI�C���A�*;


total_loss�:�@

error_R?W>?

learning_rate_1�<�7�{@�I       6%�	�C���A�*;


total_loss$A

error_R�E?

learning_rate_1�<�7fF��I       6%�	���C���A�*;


total_loss8k�@

error_R)�e?

learning_rate_1�<�7#V�I       6%�	[�C���A�*;


total_loss\2�@

error_R�P?

learning_rate_1�<�7�`حI       6%�	�S�C���A�*;


total_loss�1�@

error_Rw�J?

learning_rate_1�<�7��CeI       6%�	��C���A�*;


total_loss[��@

error_R�O?

learning_rate_1�<�7���I       6%�	_��C���A�*;


total_loss��@

error_R�~S?

learning_rate_1�<�7?u6I       6%�	��C���A�*;


total_loss�+�@

error_R��L?

learning_rate_1�<�7j�[3I       6%�	kc�C���A�*;


total_lossE�@

error_R1$M?

learning_rate_1�<�7/�<KI       6%�	���C���A�*;


total_loss$��@

error_R�tb?

learning_rate_1�<�7�.dI       6%�	��C���A�*;


total_loss�%�@

error_R��V?

learning_rate_1�<�7g4�rI       6%�	/-�C���A�*;


total_lossf��@

error_R8bD?

learning_rate_1�<�7��;I       6%�	t�C���A�*;


total_lossl\�@

error_R��I?

learning_rate_1�<�7/�(�I       6%�	z��C���A�*;


total_loss��@

error_RjlO?

learning_rate_1�<�7T��_I       6%�	���C���A�*;


total_loss4g�@

error_RhhG?

learning_rate_1�<�7��PI       6%�	�>�C���A�*;


total_loss�f@

error_R��G?

learning_rate_1�<�7 ���I       6%�	i��C���A�*;


total_loss���@

error_R�Y?

learning_rate_1�<�7���I       6%�	��C���A�*;


total_lossrv�@

error_R��F?

learning_rate_1�<�7�gN(I       6%�	��C���A�*;


total_loss��A

error_RF�J?

learning_rate_1�<�7<��bI       6%�	$\�C���A�*;


total_loss/A�@

error_R�G?

learning_rate_1�<�7����I       6%�	ʥ�C���A�*;


total_loss�V�@

error_RC�P?

learning_rate_1�<�7H���I       6%�	���C���A�*;


total_lossLה@

error_R��N?

learning_rate_1�<�7��фI       6%�	�5�C���A�*;


total_loss���@

error_R� U?

learning_rate_1�<�7��I       6%�	�w�C���A�*;


total_loss���@

error_R��E?

learning_rate_1�<�7⬓(I       6%�	w��C���A�*;


total_loss�@

error_R�Q?

learning_rate_1�<�7�&¢I       6%�	-��C���A�*;


total_loss���@

error_R�TS?

learning_rate_1�<�7��ȏI       6%�	�C�C���A�*;


total_loss�B�@

error_R��E?

learning_rate_1�<�7��{\I       6%�	���C���A�*;


total_lossMA

error_RfT?

learning_rate_1�<�7i��I       6%�	���C���A�*;


total_loss:w�@

error_RQ;?

learning_rate_1�<�7�с_I       6%�	��C���A�*;


total_lossx`�@

error_R3I?

learning_rate_1�<�7¦I       6%�	/|�C���A�*;


total_loss|��@

error_R|}7?

learning_rate_1�<�7���I       6%�	���C���A�*;


total_loss�B�@

error_R�b>?

learning_rate_1�<�7W�I       6%�	��C���A�*;


total_loss��@

error_R�yK?

learning_rate_1�<�7@[hI       6%�	�]�C���A�*;


total_loss,��@

error_R��P?

learning_rate_1�<�7d�A)I       6%�	I��C���A�*;


total_loss���@

error_R��C?

learning_rate_1�<�7ƸV�I       6%�	|��C���A�*;


total_loss7�A

error_R82D?

learning_rate_1�<�7�.�I       6%�	�2�C���A�*;


total_loss��@

error_R�a?

learning_rate_1�<�7�^I       6%�	�y�C���A�*;


total_loss?+�@

error_R�Q?

learning_rate_1�<�7c�_:I       6%�	;��C���A�*;


total_lossZ�@

error_R$gW?

learning_rate_1�<�7�npI       6%�	�C���A�*;


total_lossg�@

error_R��R?

learning_rate_1�<�7rX�LI       6%�	VI�C���A�*;


total_loss�Z�@

error_R-K?

learning_rate_1�<�7�ǁ�I       6%�	;��C���A�*;


total_loss�W�@

error_R��>?

learning_rate_1�<�7`��I       6%�	=��C���A�*;


total_loss��@

error_Rv�Z?

learning_rate_1�<�7�0RdI       6%�	e�C���A�*;


total_lossJ�c@

error_R�4@?

learning_rate_1�<�7ʼ��I       6%�	^�C���A�*;


total_loss��@

error_R�d?

learning_rate_1�<�7{�� I       6%�	c��C���A�*;


total_loss߮@

error_R ET?

learning_rate_1�<�7�69JI       6%�	���C���A�*;


total_loss�-�@

error_R8^]?

learning_rate_1�<�7��I       6%�	u0�C���A�*;


total_loss!nA

error_R�b?

learning_rate_1�<�7㳓�I       6%�	fs�C���A�*;


total_lossW�z@

error_R��^?

learning_rate_1�<�7�#�I       6%�	��C���A�*;


total_loss�N�@

error_R*�??

learning_rate_1�<�7�_��I       6%�	:��C���A�*;


total_loss�n�@

error_R�VC?

learning_rate_1�<�7�F�OI       6%�	�A�C���A�*;


total_loss-��@

error_R�L?

learning_rate_1�<�7ic�GI       6%�	j��C���A�*;


total_loss?��@

error_R-?

learning_rate_1�<�7���?I       6%�		��C���A�*;


total_loss��@

error_R1@?

learning_rate_1�<�7����I       6%�	�
�C���A�*;


total_loss4��@

error_R6�O?

learning_rate_1�<�7%�gJI       6%�	�N�C���A�*;


total_loss�'�@

error_R�2N?

learning_rate_1�<�7��pI       6%�	��C���A�*;


total_loss.�@

error_RϹD?

learning_rate_1�<�7���TI       6%�	o��C���A�*;


total_lossؾ@

error_R�J?

learning_rate_1�<�7銴�I       6%�	j�C���A�*;


total_loss(jp@

error_R��P?

learning_rate_1�<�7�%̛I       6%�	�]�C���A�*;


total_loss��@

error_R�Q?

learning_rate_1�<�7S��I       6%�	\��C���A�*;


total_loss��@

error_R�C?

learning_rate_1�<�7�$HI       6%�	���C���A�*;


total_loss �@

error_R�ZW?

learning_rate_1�<�7��tI       6%�	�* D���A�*;


total_loss��@

error_R��\?

learning_rate_1�<�7��>I       6%�	�n D���A�*;


total_lossU�@

error_R��Z?

learning_rate_1�<�7�j�I       6%�	�� D���A�*;


total_loss)��@

error_R��R?

learning_rate_1�<�7t�EXI       6%�	�� D���A�*;


total_loss��@

error_R�V?

learning_rate_1�<�7Yʡ�I       6%�	:D���A�*;


total_losss�@

error_R�KJ?

learning_rate_1�<�7���I       6%�	[�D���A�*;


total_loss���@

error_R��W?

learning_rate_1�<�7�I       6%�	`�D���A�*;


total_loss&��@

error_R�k`?

learning_rate_1�<�7I��I       6%�	�D���A�*;


total_loss� �@

error_R/�W?

learning_rate_1�<�7��יI       6%�	:KD���A�*;


total_loss�E�@

error_Rd�F?

learning_rate_1�<�7�?�+I       6%�	w�D���A�*;


total_lossϗ�@

error_R.L?

learning_rate_1�<�7\DzI       6%�	l�D���A�*;


total_loss(5�@

error_R��D?

learning_rate_1�<�7g���I       6%�	�D���A�*;


total_loss��v@

error_R��D?

learning_rate_1�<�7�E�I       6%�	mZD���A�*;


total_loss���@

error_R�"=?

learning_rate_1�<�7&{�:I       6%�	ʤD���A�*;


total_loss���@

error_Rq@?

learning_rate_1�<�7U��1I       6%�	��D���A�*;


total_loss��@

error_RԿ]?

learning_rate_1�<�7A��JI       6%�	8D���A�*;


total_loss�y�@

error_R�Kg?

learning_rate_1�<�7����I       6%�	��D���A�*;


total_loss���@

error_R�>,?

learning_rate_1�<�7.NI       6%�	�D���A�*;


total_lossR#�@

error_R��I?

learning_rate_1�<�7=��I       6%�	bD���A�*;


total_loss#&�@

error_R�X?

learning_rate_1�<�7�m;#I       6%�	�WD���A�*;


total_lossfo�@

error_R�B?

learning_rate_1�<�7c�7I       6%�	<�D���A�*;


total_lossn�@

error_R��[?

learning_rate_1�<�76`��I       6%�	��D���A�*;


total_lossh��@

error_R�R?

learning_rate_1�<�72;I       6%�	A(D���A�*;


total_loss���@

error_RR�G?

learning_rate_1�<�7��UOI       6%�	�D���A�*;


total_loss���@

error_R�a?

learning_rate_1�<�7�
��I       6%�	��D���A�*;


total_loss��@

error_R��P?

learning_rate_1�<�7}+�SI       6%�	�D���A�*;


total_loss��o@

error_R)�V?

learning_rate_1�<�7E�N�I       6%�	�sD���A�*;


total_loss��@

error_R��^?

learning_rate_1�<�7,��I       6%�	��D���A�*;


total_loss��@

error_RR�\?

learning_rate_1�<�7c֍I       6%�	"D���A�*;


total_lossρ�@

error_R�~H?

learning_rate_1�<�7���I       6%�	�dD���A�*;


total_loss�4�@

error_RazI?

learning_rate_1�<�7�v�EI       6%�	9�D���A�*;


total_loss6o�@

error_R�M?

learning_rate_1�<�7
��kI       6%�	�	D���A�*;


total_loss��@

error_R��\?

learning_rate_1�<�7-�I       6%�	�N	D���A�*;


total_loss��@

error_RTk5?

learning_rate_1�<�7{[�I       6%�	��	D���A�*;


total_loss�Q�@

error_R8)>?

learning_rate_1�<�7���eI       6%�	��	D���A�*;


total_losst�@

error_RҖM?

learning_rate_1�<�7ˡ3I       6%�	&
D���A�*;


total_loss�\�@

error_R�F?

learning_rate_1�<�7�H2I       6%�	Z
D���A�*;


total_loss���@

error_R�;Z?

learning_rate_1�<�7O�a�I       6%�	8�
D���A�*;


total_loss�b�@

error_Rs>?

learning_rate_1�<�7����I       6%�	��
D���A�*;


total_lossHZ�@

error_R3�P?

learning_rate_1�<�7K�X	I       6%�	�&D���A�*;


total_loss��J@

error_R�VL?

learning_rate_1�<�7 N)*I       6%�	kD���A�*;


total_loss:��@

error_R��e?

learning_rate_1�<�7=�(�I       6%�	M�D���A�*;


total_lossE�@

error_Rȶ_?

learning_rate_1�<�7GE0jI       6%�	��D���A�*;


total_loss�t�@

error_REkP?

learning_rate_1�<�7���I       6%�	�7D���A�*;


total_loss�&�@

error_R�I?

learning_rate_1�<�7!�J�I       6%�	�yD���A�*;


total_loss�@

error_R|G@?

learning_rate_1�<�7�t�I       6%�	i�D���A�*;


total_lossX��@

error_R�P?

learning_rate_1�<�7��O�I       6%�	0�D���A�*;


total_loss��@

error_RR�A?

learning_rate_1�<�7k<�mI       6%�	3?D���A�*;


total_loss�s�@

error_R
�W?

learning_rate_1�<�7���FI       6%�	��D���A�*;


total_loss(�@

error_R�H?

learning_rate_1�<�7�^1%I       6%�	��D���A�*;


total_loss�#�@

error_R�f=?

learning_rate_1�<�7Su`�I       6%�	�
D���A�*;


total_loss�ى@

error_R�C?

learning_rate_1�<�7��P%I       6%�	qQD���A�*;


total_loss�/�@

error_R�O?

learning_rate_1�<�7���I       6%�	��D���A�*;


total_loss��A

error_R�`Y?

learning_rate_1�<�7�ۥ$I       6%�	%�D���A�*;


total_loss�GA

error_R�wJ?

learning_rate_1�<�7�K��I       6%�	/%D���A�*;


total_lossī�@

error_R�vK?

learning_rate_1�<�7���qI       6%�	�hD���A�*;


total_loss�L�@

error_R��^?

learning_rate_1�<�7}�Y�I       6%�	ݫD���A�*;


total_loss�~�@

error_R��J?

learning_rate_1�<�7��F�I       6%�	��D���A�*;


total_lossK_�@

error_R�V?

learning_rate_1�<�70qj\I       6%�	4D���A�*;


total_loss���@

error_RZ a?

learning_rate_1�<�7���vI       6%�	�uD���A�*;


total_loss�3�@

error_RVeH?

learning_rate_1�<�7r�I       6%�	U�D���A�*;


total_lossI[�@

error_R��B?

learning_rate_1�<�7�y�vI       6%�	�D���A�*;


total_loss��@

error_R&D?

learning_rate_1�<�7�IE�I       6%�	�ED���A�*;


total_loss���@

error_R�R?

learning_rate_1�<�7ݔ0�I       6%�	̋D���A�*;


total_lossI�@

error_R_e?

learning_rate_1�<�7�/��I       6%�	��D���A�*;


total_loss���@

error_R��D?

learning_rate_1�<�7��cmI       6%�	:D���A�*;


total_loss�ë@

error_R�=?

learning_rate_1�<�7�k��I       6%�	>\D���A�*;


total_loss��@

error_R�eO?

learning_rate_1�<�7��LI       6%�	?�D���A�*;


total_loss���@

error_R|O;?

learning_rate_1�<�7�U��I       6%�	3�D���A�*;


total_loss�M�@

error_R\�J?

learning_rate_1�<�75�LI       6%�	c2D���A�*;


total_loss}F�@

error_R�FN?

learning_rate_1�<�7����I       6%�	�|D���A�*;


total_loss��@

error_R��M?

learning_rate_1�<�7.��I       6%�	��D���A�*;


total_loss��@

error_R��P?

learning_rate_1�<�7�	a�I       6%�	KD���A�*;


total_lossRs�@

error_R`�V?

learning_rate_1�<�7ކ�,I       6%�	@GD���A�*;


total_loss�.�@

error_RYS?

learning_rate_1�<�7��Y�I       6%�	1�D���A�*;


total_loss�w�@

error_R�E?

learning_rate_1�<�7)*fLI       6%�	�D���A�*;


total_loss3�@

error_R��X?

learning_rate_1�<�7��7TI       6%�	�/D���A�*;


total_lossl�@

error_R.gO?

learning_rate_1�<�7jx�LI       6%�	otD���A�*;


total_loss���@

error_R 2X?

learning_rate_1�<�7Bn+�I       6%�	��D���A�*;


total_loss��@

error_Rq&??

learning_rate_1�<�7z\��I       6%�	�D���A�*;


total_loss@��@

error_Rd�X?

learning_rate_1�<�7�v��I       6%�	LD���A�*;


total_loss�u�@

error_R�M?

learning_rate_1�<�7�T��I       6%�	��D���A�*;


total_loss��@

error_R@;@?

learning_rate_1�<�7F��AI       6%�	��D���A�*;


total_loss��@

error_R�\A?

learning_rate_1�<�7���I       6%�	{&D���A�*;


total_loss�|�@

error_R��L?

learning_rate_1�<�7�G)�I       6%�	�D���A�*;


total_loss�0�@

error_R&IW?

learning_rate_1�<�7�{�I       6%�	�D���A�*;


total_loss�t�@

error_R�U?

learning_rate_1�<�7�A�I       6%�	�#D���A�*;


total_loss�rA

error_RR"P?

learning_rate_1�<�7�DXI       6%�	�fD���A�*;


total_loss���@

error_RdyE?

learning_rate_1�<�7q�ΒI       6%�	éD���A�*;


total_loss�]�@

error_R�'=?

learning_rate_1�<�7ϐ^I       6%�	d�D���A�*;


total_loss�8�@

error_R1V?

learning_rate_1�<�7��I       6%�	[1D���A�*;


total_loss�v@

error_Rx�R?

learning_rate_1�<�7�;N�I       6%�	%yD���A�*;


total_loss���@

error_R�uC?

learning_rate_1�<�7;ۿI       6%�	��D���A�*;


total_lossRi�@

error_RHxM?

learning_rate_1�<�7�kZI       6%�	�D���A�*;


total_lossX��@

error_R6�V?

learning_rate_1�<�7oA��I       6%�	RJD���A�*;


total_loss�5�@

error_R]�T?

learning_rate_1�<�7�? )I       6%�	��D���A�*;


total_lossذ@

error_R�?O?

learning_rate_1�<�7�Pw�I       6%�	��D���A�*;


total_loss��@

error_R �V?

learning_rate_1�<�7n��I       6%�	6D���A�*;


total_loss��A

error_RWC?

learning_rate_1�<�7�ĺI       6%�	�]D���A�*;


total_lossE��@

error_R	U?

learning_rate_1�<�7�0sI       6%�	?�D���A�*;


total_loss_[�@

error_R�6O?

learning_rate_1�<�7[59:I       6%�	��D���A�*;


total_lossh�@

error_R.pL?

learning_rate_1�<�7b��sI       6%�	�(D���A�*;


total_loss��@

error_R�@i?

learning_rate_1�<�7�I       6%�	�nD���A�*;


total_loss��@

error_R}wS?

learning_rate_1�<�7q23I       6%�	�D���A�*;


total_loss?Ќ@

error_R��C?

learning_rate_1�<�7cv��I       6%�	��D���A�*;


total_loss�&�@

error_R��V?

learning_rate_1�<�7'�+%I       6%�	�JD���A�*;


total_lossTu�@

error_R��c?

learning_rate_1�<�7TC�I       6%�	I�D���A�*;


total_loss��@

error_R��M?

learning_rate_1�<�7j�	I       6%�	��D���A�*;


total_loss��@

error_R�c?

learning_rate_1�<�7#5I       6%�	]&D���A�*;


total_loss�/�@

error_Rv�I?

learning_rate_1�<�7v��I       6%�	�hD���A�*;


total_lossv1�@

error_RxrB?

learning_rate_1�<�7�c~II       6%�	��D���A�*;


total_lossF<�@

error_RX\D?

learning_rate_1�<�7j�I       6%�	��D���A�*;


total_loss�D�@

error_RHq`?

learning_rate_1�<�7%l�*I       6%�	1D���A�*;


total_loss���@

error_RdJ:?

learning_rate_1�<�7p �oI       6%�	�sD���A�*;


total_loss;jA

error_R&�[?

learning_rate_1�<�7���oI       6%�	ȵD���A�*;


total_lossN��@

error_R;?Z?

learning_rate_1�<�7�~��I       6%�	`�D���A�*;


total_loss;�@

error_R�DA?

learning_rate_1�<�7�H�kI       6%�	:B D���A�*;


total_loss}@

error_R�5U?

learning_rate_1�<�72���I       6%�	.� D���A�*;


total_losst2�@

error_R�4?

learning_rate_1�<�7��~oI       6%�	9� D���A�*;


total_lossM��@

error_Rj:?

learning_rate_1�<�7�$	I       6%�	�!!D���A�*;


total_loss���@

error_Rϥ]?

learning_rate_1�<�7ܸiI       6%�	�h!D���A�*;


total_lossd�@

error_RO'G?

learning_rate_1�<�7FgCI       6%�	��!D���A�*;


total_loss���@

error_RvL?

learning_rate_1�<�7��
rI       6%�	��!D���A�*;


total_lossdޠ@

error_Rw�W?

learning_rate_1�<�7�vC+I       6%�	�2"D���A�*;


total_loss��@

error_R�NH?

learning_rate_1�<�7��2I       6%�	ny"D���A�*;


total_loss�"�@

error_R��R?

learning_rate_1�<�7�E�^I       6%�	K�"D���A�*;


total_loss�{�@

error_R�3?

learning_rate_1�<�7�&9I       6%�	2#D���A�*;


total_loss�A

error_RfP?

learning_rate_1�<�7-Հ$I       6%�	�V#D���A�*;


total_loss �@

error_R�V?

learning_rate_1�<�7f�3�I       6%�	S�#D���A�*;


total_lossHq�@

error_R2M?

learning_rate_1�<�7��&/I       6%�	��#D���A�*;


total_lossC��@

error_Rl�=?

learning_rate_1�<�7m�I.I       6%�	),$D���A�*;


total_loss?%�@

error_RE+W?

learning_rate_1�<�7��S�I       6%�	~p$D���A�*;


total_loss6��@

error_RT�g?

learning_rate_1�<�7�- �I       6%�	Y�$D���A�*;


total_loss_c�@

error_R�&;?

learning_rate_1�<�7{�-}I       6%�	�$D���A�*;


total_loss��@

error_RoTO?

learning_rate_1�<�74>�SI       6%�	66%D���A�*;


total_losst�A

error_R�=?

learning_rate_1�<�7~�dI       6%�	Tx%D���A�*;


total_loss�n�@

error_R\D?

learning_rate_1�<�73��I       6%�	��%D���A�*;


total_loss��@

error_RL??

learning_rate_1�<�7�[�I       6%�	(�%D���A�*;


total_loss��A

error_R�VI?

learning_rate_1�<�7�C�I       6%�	�<&D���A�*;


total_loss�4�@

error_R��F?

learning_rate_1�<�7���I       6%�	�&D���A�*;


total_loss�7�@

error_RJyN?

learning_rate_1�<�7R���I       6%�	��&D���A�*;


total_loss���@

error_R�%H?

learning_rate_1�<�7�_I       6%�	U'D���A�*;


total_loss{\�@

error_R�V^?

learning_rate_1�<�7�B�OI       6%�	�Q'D���A�*;


total_loss�ۋ@

error_R��T?

learning_rate_1�<�7wb6�I       6%�	O�'D���A�*;


total_loss�pA

error_RCCR?

learning_rate_1�<�7���WI       6%�	��'D���A�*;


total_loss�@

error_R��*?

learning_rate_1�<�7���I       6%�	,H(D���A�*;


total_loss.2�@

error_RZ�N?

learning_rate_1�<�7��hI       6%�	ُ(D���A�*;


total_loss:L�@

error_RW>U?

learning_rate_1�<�7����I       6%�	r�(D���A�*;


total_lossH�A

error_R��@?

learning_rate_1�<�7�p�I       6%�	x)D���A�*;


total_loss�ћ@

error_R�R?

learning_rate_1�<�7*]d�I       6%�	\)D���A�*;


total_loss��@

error_RLjF?

learning_rate_1�<�7����I       6%�	n�)D���A�*;


total_loss��@

error_R��S?

learning_rate_1�<�7R��I       6%�	�)D���A�*;


total_loss ��@

error_R��]?

learning_rate_1�<�7+G%LI       6%�	c0*D���A�*;


total_loss��@

error_R{�h?

learning_rate_1�<�7>�4oI       6%�	qt*D���A�*;


total_loss�`A

error_RK??

learning_rate_1�<�7J輱I       6%�	��*D���A�*;


total_loss�J�@

error_R!3G?

learning_rate_1�<�7:�-{I       6%�	�+D���A�*;


total_loss}��@

error_R��e?

learning_rate_1�<�7���I       6%�	gD+D���A�*;


total_loss��@

error_R��P?

learning_rate_1�<�7�DiI       6%�	K�+D���A�*;


total_loss���@

error_Rz�D?

learning_rate_1�<�7�q`�I       6%�	t�+D���A�*;


total_loss�@

error_R�=n?

learning_rate_1�<�7d>GnI       6%�	$
,D���A�*;


total_loss�@

error_R��X?

learning_rate_1�<�7v�I       6%�	�M,D���A�*;


total_loss�@

error_R�[?

learning_rate_1�<�7�U JI       6%�	��,D���A�*;


total_loss�>�@

error_R �F?

learning_rate_1�<�7��P�I       6%�	��,D���A�*;


total_lossO�@

error_R�[d?

learning_rate_1�<�7a��I       6%�	�-D���A�*;


total_losss7�@

error_RJBS?

learning_rate_1�<�7����I       6%�	�g-D���A�*;


total_loss���@

error_R$+W?

learning_rate_1�<�7����I       6%�	F�-D���A�*;


total_loss��@

error_R��L?

learning_rate_1�<�7��:�I       6%�	��-D���A�*;


total_loss���@

error_R/�R?

learning_rate_1�<�7�)�oI       6%�	�8.D���A�*;


total_loss���@

error_R��D?

learning_rate_1�<�7��_I       6%�	W|.D���A�*;


total_loss��A

error_R�X<?

learning_rate_1�<�7�5�2I       6%�	W�.D���A�*;


total_loss�q�@

error_R6(J?

learning_rate_1�<�7C%�I       6%�	�/D���A�*;


total_loss}"A

error_R�mZ?

learning_rate_1�<�7n�RhI       6%�	�L/D���A�*;


total_loss�.�@

error_R�S?

learning_rate_1�<�7I,��I       6%�	��/D���A�*;


total_loss>2�@

error_R3@L?

learning_rate_1�<�7v�E�I       6%�	��/D���A�*;


total_loss�ѓ@

error_R��U?

learning_rate_1�<�7�c-I       6%�	�0D���A�*;


total_loss��@

error_RxC;?

learning_rate_1�<�7��E4I       6%�	�[0D���A�*;


total_lossZ�b@

error_R?�h?

learning_rate_1�<�78rI       6%�	�0D���A�*;


total_loss��@

error_RO�G?

learning_rate_1�<�7�7��I       6%�	_�0D���A�*;


total_loss��@

error_RrD?

learning_rate_1�<�7W>��I       6%�	�(1D���A�*;


total_loss7c�@

error_R�|D?

learning_rate_1�<�7��OVI       6%�	�p1D���A�*;


total_loss_��@

error_RT�U?

learning_rate_1�<�7���MI       6%�	��1D���A�*;


total_loss��@

error_R�V?

learning_rate_1�<�7���hI       6%�	��1D���A�*;


total_loss��@

error_R��T?

learning_rate_1�<�7��U�I       6%�	?2D���A�*;


total_loss��@

error_R�SK?

learning_rate_1�<�7�lc$I       6%�	�2D���A�*;


total_lossi��@

error_R��P?

learning_rate_1�<�7�U�I       6%�	��2D���A�*;


total_loss��@

error_R|�=?

learning_rate_1�<�7���QI       6%�	�3D���A�*;


total_loss���@

error_R�N?

learning_rate_1�<�7毮�I       6%�	K3D���A�*;


total_loss4/�@

error_R_�_?

learning_rate_1�<�7��ߏI       6%�	��3D���A�*;


total_loss�{p@

error_R�:R?

learning_rate_1�<�7`�I       6%�	��3D���A�*;


total_lossk�@

error_R�!Z?

learning_rate_1�<�7���I       6%�	K4D���A�*;


total_lossFɓ@

error_R��N?

learning_rate_1�<�7@�lnI       6%�	vZ4D���A�*;


total_loss�$�@

error_R��M?

learning_rate_1�<�7U�v�I       6%�	��4D���A�*;


total_loss�p�@

error_R��V?

learning_rate_1�<�7��[I       6%�	�4D���A�*;


total_lossh��@

error_R��X?

learning_rate_1�<�7�9�I       6%�	�+5D���A�*;


total_loss��@

error_RWFC?

learning_rate_1�<�7�eWHI       6%�	�n5D���A�*;


total_loss���@

error_Rd�K?

learning_rate_1�<�7�(WI       6%�	L�5D���A�*;


total_loss��@

error_R6�Z?

learning_rate_1�<�7{?mgI       6%�	��5D���A�*;


total_lossͧ�@

error_R`L]?

learning_rate_1�<�7�+�<I       6%�	�66D���A�*;


total_loss�|�@

error_R\\D?

learning_rate_1�<�7�t~�I       6%�	Dz6D���A�*;


total_lossN�@

error_R��L?

learning_rate_1�<�7��ړI       6%�	��6D���A�*;


total_loss��@

error_RҮ;?

learning_rate_1�<�76&*^I       6%�	�6D���A�*;


total_loss�i�@

error_R�tM?

learning_rate_1�<�7ɞjI       6%�	tY7D���A�*;


total_lossf0�@

error_R�*H?

learning_rate_1�<�7]�eI       6%�	d�7D���A�*;


total_lossta�@

error_Rx�S?

learning_rate_1�<�7��I       6%�	t8D���A�*;


total_loss:�@

error_Rnb?

learning_rate_1�<�7��rI       6%�	YJ8D���A�*;


total_loss-�@

error_RZG?

learning_rate_1�<�7
W]I       6%�	��8D���A�*;


total_lossJ��@

error_Rf�E?

learning_rate_1�<�75�FKI       6%�	"�8D���A�*;


total_lossW�
A

error_R�VU?

learning_rate_1�<�7nj!mI       6%�	�9D���A�*;


total_loss��@

error_RLR?

learning_rate_1�<�7D2�I       6%�	�`9D���A�*;


total_lossԿ	A

error_Rt?O?

learning_rate_1�<�7��m�I       6%�	��9D���A�*;


total_loss(#e@

error_R)>O?

learning_rate_1�<�7���I       6%�	��9D���A�*;


total_loss���@

error_R=�R?

learning_rate_1�<�7H�;�I       6%�	�,:D���A�*;


total_loss& �@

error_R�zA?

learning_rate_1�<�7N��QI       6%�	�p:D���A�*;


total_loss�̱@

error_R��I?

learning_rate_1�<�7�S�I       6%�	8�:D���A�*;


total_loss	�{@

error_RA	G?

learning_rate_1�<�7[�PI       6%�	"�:D���A�*;


total_loss���@

error_Rc�G?

learning_rate_1�<�7�L��I       6%�	'>;D���A�*;


total_loss1�@

error_R��[?

learning_rate_1�<�7%}I       6%�	0�;D���A�*;


total_loss)�|@

error_Rv�U?

learning_rate_1�<�7�x#gI       6%�	5�;D���A�*;


total_loss�M�@

error_R�yI?

learning_rate_1�<�7���I       6%�	�<D���A�*;


total_losswW�@

error_RR�Q?

learning_rate_1�<�7F�+I       6%�	�O<D���A�*;


total_loss�@�@

error_R�5Q?

learning_rate_1�<�7�[�tI       6%�	$�<D���A�*;


total_loss ��@

error_R�<g?

learning_rate_1�<�7I�@�I       6%�	:�<D���A�*;


total_loss��@

error_RJ<B?

learning_rate_1�<�7c���I       6%�	H=D���A�*;


total_loss���@

error_R�[?

learning_rate_1�<�78���I       6%�	�_=D���A�*;


total_lossH�@

error_RX�>?

learning_rate_1�<�7Q-I       6%�	=�=D���A�*;


total_loss�A

error_RL�O?

learning_rate_1�<�7C�vI       6%�	E�=D���A�*;


total_loss1S�@

error_R��J?

learning_rate_1�<�7�g�GI       6%�	�;>D���A�*;


total_loss��@

error_R@�>?

learning_rate_1�<�7R�. I       6%�	Ǎ>D���A�*;


total_loss���@

error_R�P?

learning_rate_1�<�7(�,�I       6%�	�>D���A�*;


total_loss,Q�@

error_RH�F?

learning_rate_1�<�7]�7�I       6%�	�?D���A�*;


total_lossW"�@

error_R��[?

learning_rate_1�<�7y��]I       6%�	�b?D���A�*;


total_loss�`�@

error_R �C?

learning_rate_1�<�7��G�I       6%�	i�?D���A�*;


total_loss��@

error_R�@?

learning_rate_1�<�7"JW�I       6%�	��?D���A�*;


total_loss���@

error_RM�W?

learning_rate_1�<�7���I       6%�	�-@D���A�*;


total_loss���@

error_R��C?

learning_rate_1�<�7]�GI       6%�	�q@D���A�*;


total_loss�m�@

error_Rx:?

learning_rate_1�<�7
"�I       6%�	��@D���A�*;


total_loss]D�@

error_R�&Y?

learning_rate_1�<�7}�?%I       6%�	��@D���A�*;


total_loss��@

error_RmV?

learning_rate_1�<�7����I       6%�	�CAD���A�*;


total_lossdX�@

error_R��K?

learning_rate_1�<�7��b�I       6%�	J�AD���A�*;


total_lossX�
A

error_R
�N?

learning_rate_1�<�7�#�\I       6%�	w�AD���A�*;


total_loss�A

error_R�N?

learning_rate_1�<�7��8II       6%�	�BD���A�*;


total_lossw�@

error_R��C?

learning_rate_1�<�7��
I       6%�	�SBD���A�*;


total_loss���@

error_RHB?

learning_rate_1�<�7Қ��I       6%�	��BD���A�*;


total_loss�ے@

error_Rf�O?

learning_rate_1�<�7�6�"I       6%�	��BD���A�*;


total_loss�J�@

error_R�M?

learning_rate_1�<�7���RI       6%�	"CD���A�*;


total_loss�7�@

error_R�6?

learning_rate_1�<�7��T�I       6%�	|hCD���A�*;


total_loss���@

error_R.t\?

learning_rate_1�<�7��0�I       6%�	«CD���A�*;


total_loss�;�@

error_R�|\?

learning_rate_1�<�7���I       6%�	��CD���A�*;


total_loss���@

error_R��0?

learning_rate_1�<�7���LI       6%�	P7DD���A�*;


total_loss��@

error_R
1[?

learning_rate_1�<�7�l(I       6%�	DD���A�*;


total_loss��@

error_R�d]?

learning_rate_1�<�7���.I       6%�	)�DD���A�*;


total_loss��@

error_R�<[?

learning_rate_1�<�7�o��I       6%�	ED���A�*;


total_loss8ӌ@

error_R�U?

learning_rate_1�<�7���8I       6%�	�]ED���A�*;


total_lossz�@

error_R,D@?

learning_rate_1�<�76؝I       6%�	�ED���A�*;


total_loss��@

error_Rr�F?

learning_rate_1�<�7�!F5I       6%�	Y�ED���A�*;


total_loss�p�@

error_R��M?

learning_rate_1�<�7�{7jI       6%�	�/FD���A�*;


total_loss(=�@

error_R`FV?

learning_rate_1�<�7@bI       6%�	EtFD���A�*;


total_loss��@

error_R��Y?

learning_rate_1�<�7x�`�I       6%�	$�FD���A�*;


total_loss=��@

error_RO�<?

learning_rate_1�<�7�)�I       6%�	Z�FD���A�*;


total_loss�C�@

error_RҎK?

learning_rate_1�<�7L���I       6%�	�CGD���A�*;


total_lossʥ�@

error_R�K?

learning_rate_1�<�7Ҫ�I       6%�	ӲGD���A�*;


total_loss���@

error_R��]?

learning_rate_1�<�7В˶I       6%�	 HD���A�*;


total_loss�ږ@

error_R�^Y?

learning_rate_1�<�7��,I       6%�	�IHD���A�*;


total_losseCg@

error_R=�B?

learning_rate_1�<�7z?pLI       6%�	��HD���A�*;


total_lossb1A

error_RCG\?

learning_rate_1�<�7^�[I       6%�	��HD���A�*;


total_loss:��@

error_RF?

learning_rate_1�<�7�?O�I       6%�	)ID���A�*;


total_loss�*�@

error_R�aF?

learning_rate_1�<�7�Q�I       6%�	}ID���A�*;


total_loss,D�@

error_R3�G?

learning_rate_1�<�7�>��I       6%�	��ID���A�*;


total_loss���@

error_R�-M?

learning_rate_1�<�7Fe~QI       6%�		JD���A�*;


total_lossDv�@

error_R��L?

learning_rate_1�<�7{�O�I       6%�	�YJD���A�*;


total_loss�J�@

error_R��N?

learning_rate_1�<�7S�+�I       6%�	��JD���A�*;


total_loss��@

error_R{8B?

learning_rate_1�<�7��DcI       6%�	��JD���A�*;


total_loss�W�@

error_R�oI?

learning_rate_1�<�7σ�I       6%�	�+KD���A�*;


total_loss>��@

error_Ro�T?

learning_rate_1�<�7���I       6%�	�pKD���A�*;


total_loss�h�@

error_RT�S?

learning_rate_1�<�7!�6�I       6%�	;�KD���A�*;


total_losso:�@

error_R.??

learning_rate_1�<�7�@I       6%�	��KD���A�*;


total_loss��@

error_Re�V?

learning_rate_1�<�7B���I       6%�	�GLD���A�*;


total_lossV��@

error_R��_?

learning_rate_1�<�7b��I       6%�	(�LD���A�*;


total_loss��@

error_R��I?

learning_rate_1�<�7ʣ��I       6%�	��LD���A�*;


total_lossFm�@

error_R�R?

learning_rate_1�<�7�s�I       6%�	iMD���A�*;


total_loss��@

error_R�:C?

learning_rate_1�<�7|�3I       6%�	]MD���A�*;


total_losskQ�@

error_RA�Q?

learning_rate_1�<�7vAI       6%�	��MD���A�*;


total_loss���@

error_R8/K?

learning_rate_1�<�7�r�I       6%�	��MD���A�*;


total_lossI�@

error_R&C?

learning_rate_1�<�7Y{�I       6%�	�+ND���A�*;


total_loss�&�@

error_R�T?

learning_rate_1�<�7�I       6%�	3wND���A�*;


total_loss,�@

error_R$�P?

learning_rate_1�<�7���I       6%�	S�ND���A�*;


total_loss���@

error_RyO?

learning_rate_1�<�79�8I       6%�	pOD���A�*;


total_loss��r@

error_R3?\?

learning_rate_1�<�7�b"I       6%�	�OOD���A�*;


total_lossF	A

error_Rab?

learning_rate_1�<�7��{�I       6%�	��OD���A�*;


total_loss���@

error_R��N?

learning_rate_1�<�7���5I       6%�	B�OD���A�*;


total_loss�>�@

error_RZU?

learning_rate_1�<�7���I       6%�	iPD���A�*;


total_loss���@

error_R��e?

learning_rate_1�<�72�gI       6%�	*\PD���A�*;


total_loss� A

error_R��U?

learning_rate_1�<�7Q3GI       6%�	��PD���A�*;


total_lossn��@

error_R1z2?

learning_rate_1�<�7��lI       6%�	?�PD���A�*;


total_loss��@

error_R{�R?

learning_rate_1�<�7�C��I       6%�	?&QD���A�*;


total_loss���@

error_R��O?

learning_rate_1�<�7q�rI       6%�	�iQD���A�*;


total_loss1�@

error_R
�S?

learning_rate_1�<�7�7cI       6%�	��QD���A�*;


total_loss�7�@

error_R&6T?

learning_rate_1�<�7�5I       6%�	S�QD���A�*;


total_loss�d�@

error_R��N?

learning_rate_1�<�7o��kI       6%�	�5RD���A�*;


total_loss2C�@

error_R3�B?

learning_rate_1�<�7�:�I       6%�	�xRD���A�*;


total_loss�n�@

error_R�D?

learning_rate_1�<�7�wd�I       6%�	üRD���A�*;


total_loss�ȕ@

error_R
H?

learning_rate_1�<�72�I       6%�	��RD���A�*;


total_loss;��@

error_RO�D?

learning_rate_1�<�7�7�I       6%�	6DSD���A�*;


total_loss��@

error_R� R?

learning_rate_1�<�7]�S�I       6%�	%�SD���A�*;


total_loss���@

error_R�-Q?

learning_rate_1�<�7�i�I       6%�	��SD���A�*;


total_lossE��@

error_Rq�S?

learning_rate_1�<�7�NZI       6%�	j$TD���A�*;


total_loss#�@

error_R@�H?

learning_rate_1�<�7d�M�I       6%�	sjTD���A�*;


total_loss���@

error_RFii?

learning_rate_1�<�7]*�I       6%�	��TD���A�*;


total_loss2��@

error_R�W?

learning_rate_1�<�7k�LI       6%�	�UD���A�*;


total_lossH-�@

error_RN�O?

learning_rate_1�<�7�AS�I       6%�	IQUD���A�*;


total_loss`��@

error_R}5W?

learning_rate_1�<�7d"!I       6%�	�UD���A�*;


total_loss���@

error_RȃK?

learning_rate_1�<�7\9(�I       6%�	��UD���A�*;


total_loss;:�@

error_R:	L?

learning_rate_1�<�7fiK�I       6%�	P VD���A�*;


total_lossn$�@

error_R��W?

learning_rate_1�<�7�E��I       6%�	=iVD���A�*;


total_loss�j�@

error_R�:[?

learning_rate_1�<�7��_KI       6%�	̯VD���A�*;


total_loss���@

error_R�SI?

learning_rate_1�<�7���I       6%�	��VD���A�*;


total_loss���@

error_R�@\?

learning_rate_1�<�7i���I       6%�	�aWD���A�*;


total_loss��p@

error_R��O?

learning_rate_1�<�7����I       6%�	^�WD���A�*;


total_loss-;�@

error_R�MP?

learning_rate_1�<�7�HvI       6%�	��WD���A�*;


total_lossܨ�@

error_R��I?

learning_rate_1�<�7��>I       6%�	�;XD���A�*;


total_lossW~�@

error_Ri�W?

learning_rate_1�<�7�5��I       6%�	||XD���A�*;


total_loss|F�@

error_R�{Y?

learning_rate_1�<�7�/�I       6%�	T�XD���A�*;


total_loss�A

error_R��c?

learning_rate_1�<�7F���I       6%�	�YD���A�*;


total_lossӑ@

error_R�	E?

learning_rate_1�<�7Q���I       6%�	�HYD���A�*;


total_lossh�|@

error_R��I?

learning_rate_1�<�7�T]II       6%�	��YD���A�*;


total_lossҦ�@

error_Rv&D?

learning_rate_1�<�7d<jI       6%�	L�YD���A�*;


total_loss���@

error_R��[?

learning_rate_1�<�7D��I       6%�	5ZD���A�*;


total_loss7v�@

error_R,�O?

learning_rate_1�<�7�F	I       6%�	�bZD���A�*;


total_loss��@

error_R-�9?

learning_rate_1�<�7�F>I       6%�	��ZD���A�*;


total_lossVB�@

error_R.�C?

learning_rate_1�<�7���I       6%�	�ZD���A�*;


total_loss;�@

error_R�<I?

learning_rate_1�<�7�!��I       6%�	�-[D���A�*;


total_loss���@

error_RJOL?

learning_rate_1�<�7���I       6%�	Ct[D���A�*;


total_loss�+�@

error_R��G?

learning_rate_1�<�7NabI       6%�	��[D���A�*;


total_losst�@

error_R��k?

learning_rate_1�<�7zAXI       6%�	�\D���A�*;


total_loss��@

error_R��W?

learning_rate_1�<�7�ڲ�I       6%�	L\D���A�*;


total_loss���@

error_R�E??

learning_rate_1�<�7?�vGI       6%�	��\D���A�*;


total_loss�g�@

error_R��Q?

learning_rate_1�<�7�N$I       6%�	��\D���A�*;


total_loss�8�@

error_R�D2?

learning_rate_1�<�7 ?�$I       6%�	(]D���A�*;


total_loss#��@

error_R�}i?

learning_rate_1�<�7*��
I       6%�	p]D���A�*;


total_loss��@

error_R��N?

learning_rate_1�<�7g�k�I       6%�	��]D���A�*;


total_loss���@

error_Rw�\?

learning_rate_1�<�7;f��I       6%�	�^D���A�*;


total_lossO�@

error_RNV?

learning_rate_1�<�7�X��I       6%�	3K^D���A�*;


total_loss��@

error_RZnX?

learning_rate_1�<�7��xyI       6%�	D�^D���A�*;


total_loss@Z�@

error_RԵc?

learning_rate_1�<�7� #sI       6%�	�^D���A�*;


total_loss��@

error_R<�U?

learning_rate_1�<�7��ݎI       6%�	�_D���A�*;


total_lossT��@

error_R%^K?

learning_rate_1�<�7��qI       6%�	(W_D���A�*;


total_lossԷ�@

error_R��>?

learning_rate_1�<�7�L9I       6%�	S�_D���A�*;


total_loss 3�@

error_R
XJ?

learning_rate_1�<�7���_I       6%�	��_D���A�*;


total_lossD�@

error_R�@?

learning_rate_1�<�7dbeI       6%�	%`D���A�*;


total_loss���@

error_R��M?

learning_rate_1�<�7�UZ�I       6%�	�b`D���A�*;


total_loss,X�@

error_R�G^?

learning_rate_1�<�7��8�I       6%�	s�`D���A�*;


total_loss�@

error_R�"S?

learning_rate_1�<�7�(˂I       6%�	=�`D���A�*;


total_lossa�@

error_R��P?

learning_rate_1�<�7���zI       6%�	87aD���A�*;


total_lossΠ@

error_R�MN?

learning_rate_1�<�7�I       6%�	{aD���A�*;


total_loss�@

error_R�:H?

learning_rate_1�<�7s���I       6%�	��aD���A�*;


total_loss��@

error_R349?

learning_rate_1�<�7�jeoI       6%�	bD���A�*;


total_loss|Ô@

error_R%UQ?

learning_rate_1�<�7�wi�I       6%�	�DbD���A�*;


total_loss�/�@

error_R��F?

learning_rate_1�<�7x�($I       6%�	ÅbD���A�*;


total_lossjݓ@

error_R2eS?

learning_rate_1�<�7�d�I       6%�	��bD���A�*;


total_loss�@

error_Rc[B?

learning_rate_1�<�7���I       6%�	.cD���A�*;


total_loss� �@

error_Rj<L?

learning_rate_1�<�7���I       6%�	'LcD���A�*;


total_loss#^�@

error_R1R?

learning_rate_1�<�7�E@�I       6%�	��cD���A�*;


total_lossN��@

error_R�#P?

learning_rate_1�<�7E�]�I       6%�	U�cD���A�*;


total_loss�,|@

error_RL�W?

learning_rate_1�<�7%IE�I       6%�	MdD���A�*;


total_loss�s�@

error_RdCY?

learning_rate_1�<�7�ŷGI       6%�	�`dD���A�*;


total_loss��@

error_RX�G?

learning_rate_1�<�7,���I       6%�	��dD���A�*;


total_loss���@

error_R��L?

learning_rate_1�<�7f���I       6%�	��dD���A�*;


total_lossJݗ@

error_R�BE?

learning_rate_1�<�7t�0
I       6%�	v-eD���A�*;


total_loss�*�@

error_R�%J?

learning_rate_1�<�7ٛ!I       6%�	"�eD���A�*;


total_lossv�@

error_R��^?

learning_rate_1�<�7P�1�I       6%�	��eD���A�*;


total_loss��@

error_R�Q?

learning_rate_1�<�7ǘxI       6%�	pfD���A�*;


total_lossNY�@

error_RR^4?

learning_rate_1�<�7 6wI       6%�	_fD���A�*;


total_loss�|A

error_R�5D?

learning_rate_1�<�7��I       6%�	�fD���A�*;


total_loss��@

error_R��M?

learning_rate_1�<�7{2ӚI       6%�	��fD���A�*;


total_loss�u�@

error_R)�C?

learning_rate_1�<�7,"�<I       6%�	�9gD���A�*;


total_lossT��@

error_R;�J?

learning_rate_1�<�7�
�iI       6%�	��gD���A�*;


total_loss���@

error_R(�S?

learning_rate_1�<�7>\�I       6%�	��gD���A�*;


total_lossxA

error_R��N?

learning_rate_1�<�78��aI       6%�	�)hD���A�*;


total_loss��@

error_R��I?

learning_rate_1�<�7���I       6%�	�lhD���A�*;


total_loss3�@

error_R�PQ?

learning_rate_1�<�7��I       6%�	T�hD���A�*;


total_lossR�@

error_R�)M?

learning_rate_1�<�7��!I       6%�	�hD���A�*;


total_loss7�1A

error_Rh�9?

learning_rate_1�<�7 }]%I       6%�	-7iD���A�*;


total_loss�@

error_R��J?

learning_rate_1�<�7(���I       6%�	�iD���A�*;


total_loss8�@

error_R�RP?

learning_rate_1�<�7���I       6%�	/�iD���A�*;


total_loss��@

error_R%�`?

learning_rate_1�<�7eS~I       6%�	�8jD���A�*;


total_loss��@

error_RxdY?

learning_rate_1�<�7�1δI       6%�	�jD���A�*;


total_loss�|�@

error_Rl{n?

learning_rate_1�<�7t�JI       6%�	D�jD���A�*;


total_loss�\A

error_R� a?

learning_rate_1�<�7���I       6%�	�8kD���A�*;


total_lossN߫@

error_R,�N?

learning_rate_1�<�7>��qI       6%�	i�kD���A�*;


total_loss���@

error_R#�C?

learning_rate_1�<�7���I       6%�	s�kD���A�*;


total_loss1l@

error_RWL?

learning_rate_1�<�7��HI       6%�	E.lD���A�*;


total_loss�/�@

error_R �C?

learning_rate_1�<�7VI       6%�	�}lD���A�*;


total_lossC�@

error_R�J?

learning_rate_1�<�7�-I       6%�	��lD���A�*;


total_loss�d�@

error_RD�B?

learning_rate_1�<�7�+�I       6%�	�3mD���A�*;


total_loss�M�@

error_R�}X?

learning_rate_1�<�7k%�I       6%�	�|mD���A�*;


total_loss�?�@

error_R=R?

learning_rate_1�<�7���I       6%�	0�mD���A�*;


total_loss�C�@

error_R1Q?

learning_rate_1�<�7W�rI       6%�	�
nD���A�*;


total_loss[י@

error_R�#F?

learning_rate_1�<�7@RVI       6%�	EQnD���A�*;


total_loss#�
A

error_R@C[?

learning_rate_1�<�7��9I       6%�	��nD���A�*;


total_loss�,�@

error_RZC?

learning_rate_1�<�7��I       6%�	��nD���A�*;


total_loss A�@

error_R��T?

learning_rate_1�<�7�8�I       6%�	�!oD���A�*;


total_loss<��@

error_RE�d?

learning_rate_1�<�7pF��I       6%�	�goD���A�*;


total_loss�$�@

error_R�R?

learning_rate_1�<�7���I       6%�	��oD���A�*;


total_loss���@

error_RiL?

learning_rate_1�<�7�byI       6%�	R�oD���A�*;


total_loss���@

error_R��L?

learning_rate_1�<�7W�/I       6%�	Z>pD���A�*;


total_losse�@

error_Rs(F?

learning_rate_1�<�7Oс~I       6%�	ۢpD���A�*;


total_loss���@

error_Rf�G?

learning_rate_1�<�7B��I       6%�	Q�pD���A�*;


total_loss��M@

error_Rj\O?

learning_rate_1�<�7��GEI       6%�	7qD���A�*;


total_loss���@

error_RfF7?

learning_rate_1�<�7���I       6%�	g�qD���A�*;


total_loss3��@

error_Rs�S?

learning_rate_1�<�7�s�VI       6%�	n�qD���A�*;


total_loss�"A

error_R�o_?

learning_rate_1�<�7d��I       6%�	$rD���A�*;


total_lossQ��@

error_R��B?

learning_rate_1�<�7[:I       6%�	�irD���A�*;


total_loss�f�@

error_Rq=Y?

learning_rate_1�<�7K�
I       6%�	�rD���A�*;


total_loss�pq@

error_R!X?

learning_rate_1�<�7�I��I       6%�	��rD���A�*;


total_losst�@

error_R�1f?

learning_rate_1�<�773|�I       6%�	�>sD���A�*;


total_loss)�@

error_R_�P?

learning_rate_1�<�7���I       6%�	 �sD���A�*;


total_loss컸@

error_R��m?

learning_rate_1�<�7�`bI       6%�	�sD���A�*;


total_loss��@

error_R�N?

learning_rate_1�<�70(|I       6%�	"tD���A�*;


total_loss�
�@

error_R8�U?

learning_rate_1�<�7��n�I       6%�	�dtD���A�*;


total_loss���@

error_R��<?

learning_rate_1�<�7.�U�I       6%�	&�tD���A�*;


total_loss��A

error_RJ�Y?

learning_rate_1�<�7�&�I       6%�	\�tD���A�*;


total_loss/_�@

error_RV?

learning_rate_1�<�7�	�=I       6%�	�0uD���A�*;


total_loss]��@

error_R��H?

learning_rate_1�<�7�q�I       6%�	dsuD���A�*;


total_loss1��@

error_R�Ui?

learning_rate_1�<�7`�wI       6%�	:�uD���A�*;


total_loss�)�@

error_RM�G?

learning_rate_1�<�7Ʌ�I       6%�	�vD���A�*;


total_loss���@

error_R��I?

learning_rate_1�<�7iB��I       6%�		FvD���A�*;


total_loss:1�@

error_R2lJ?

learning_rate_1�<�7m�I       6%�	�vD���A�*;


total_loss��@

error_R��P?

learning_rate_1�<�7�@I       6%�	j�vD���A�*;


total_losse�@

error_R�O?

learning_rate_1�<�7G�Q|I       6%�	p+wD���A�*;


total_loss���@

error_R}�T?

learning_rate_1�<�79R��I       6%�	��wD���A�*;


total_loss���@

error_R�
N?

learning_rate_1�<�7%&�I       6%�	'xD���A�*;


total_lossOy�@

error_R��9?

learning_rate_1�<�7��^I       6%�	�\xD���A�*;


total_loss<y�@

error_Ra`[?

learning_rate_1�<�7;�g�I       6%�	��xD���A�*;


total_loss�|�@

error_RM]J?

learning_rate_1�<�7�g�I       6%�	�yD���A�*;


total_loss�r�@

error_R�I?

learning_rate_1�<�7��O�I       6%�	byD���A�*;


total_loss�z�@

error_R�tO?

learning_rate_1�<�7�� �I       6%�	��yD���A�*;


total_loss�]�@

error_R��W?

learning_rate_1�<�7�U��I       6%�	E�yD���A�*;


total_loss�7�@

error_R�T?

learning_rate_1�<�7E���I       6%�	�5zD���A�*;


total_loss�n�@

error_R!�C?

learning_rate_1�<�7����I       6%�	�{zD���A�*;


total_loss���@

error_REE?

learning_rate_1�<�7�PݟI       6%�	��zD���A�*;


total_lossT`�@

error_R�kE?

learning_rate_1�<�7a&�xI       6%�	�{D���A�*;


total_loss�@

error_R�nI?

learning_rate_1�<�7<
�I       6%�	�G{D���A�*;


total_loss�+�@

error_RV�=?

learning_rate_1�<�7
�jI       6%�	/�{D���A�*;


total_loss7$�@

error_R�G?

learning_rate_1�<�7��!I       6%�	Y�{D���A�*;


total_loss*��@

error_R�P?

learning_rate_1�<�7��%�I       6%�	||D���A�*;


total_loss�J�@

error_R�+Z?

learning_rate_1�<�7x�AI       6%�	BT|D���A�*;


total_lossa�@

error_RNV=?

learning_rate_1�<�7���I       6%�	�|D���A�*;


total_losszE�@

error_R�(M?

learning_rate_1�<�7#?�I       6%�	��|D���A�*;


total_loss��@

error_RC�E?

learning_rate_1�<�7Y�I       6%�	�&}D���A�*;


total_loss�k�@

error_R��=?

learning_rate_1�<�7�:�I       6%�	yj}D���A�*;


total_loss-��@

error_R{�@?

learning_rate_1�<�7
���I       6%�	r�}D���A�*;


total_loss�n�@

error_RHsJ?

learning_rate_1�<�7�! I       6%�	r�}D���A�*;


total_loss��@

error_R�[I?

learning_rate_1�<�7�|�I       6%�	EL~D���A�*;


total_loss���@

error_R�_?

learning_rate_1�<�7e#нI       6%�	i�~D���A�*;


total_lossV��@

error_R8�V?

learning_rate_1�<�7�}�vI       6%�	V�~D���A�*;


total_lossc$�@

error_R��:?

learning_rate_1�<�7���oI       6%�	�D���A�*;


total_losse��@

error_Rr6?

learning_rate_1�<�7��I�I       6%�	 fD���A�*;


total_lossW@A

error_R�D?

learning_rate_1�<�7�v�I       6%�	'�D���A�*;


total_lossΉ@

error_R__I?

learning_rate_1�<�7m�&�I       6%�	��D���A�*;


total_losse4 A

error_RmwO?

learning_rate_1�<�7���I       6%�	�/�D���A�*;


total_loss�ـ@

error_R&�U?

learning_rate_1�<�7�֗�I       6%�	�t�D���A�*;


total_loss�@

error_R�Y?

learning_rate_1�<�7щ�yI       6%�	���D���A�*;


total_lossI�@

error_R\|T?

learning_rate_1�<�7!��I       6%�	y	�D���A�*;


total_loss֌�@

error_R6rM?

learning_rate_1�<�7�;s�I       6%�	AM�D���A�*;


total_loss�8�@

error_RC�T?

learning_rate_1�<�7E<I       6%�	:��D���A�*;


total_loss�b�@

error_R��L?

learning_rate_1�<�7�R��I       6%�	ԁD���A�*;


total_loss[��@

error_RCm^?

learning_rate_1�<�7�[�I       6%�	� �D���A�*;


total_lossMu�@

error_R��0?

learning_rate_1�<�7D��?I       6%�	`g�D���A�*;


total_loss��@

error_R�GK?

learning_rate_1�<�7=W"MI       6%�	%��D���A�*;


total_loss�K�@

error_R��P?

learning_rate_1�<�7Ƽ�EI       6%�	���D���A�*;


total_loss���@

error_R]�??

learning_rate_1�<�7��),I       6%�	A�D���A�*;


total_loss�-�@

error_R[f`?

learning_rate_1�<�7��xI       6%�	.��D���A�*;


total_loss׆@

error_RxlH?

learning_rate_1�<�7�g��I       6%�	�ɃD���A�*;


total_loss(�@

error_R�/O?

learning_rate_1�<�7T�I       6%�	��D���A�*;


total_lossk��@

error_R��N?

learning_rate_1�<�7G,c)I       6%�	�M�D���A�*;


total_lossc �@

error_RO+=?

learning_rate_1�<�7	�)	I       6%�	��D���A�*;


total_lossH��@

error_R��[?

learning_rate_1�<�7e�I       6%�	�؄D���A�*;


total_loss�[�@

error_R3iU?

learning_rate_1�<�7�ު!I       6%�	��D���A�*;


total_lossC�A

error_R6{B?

learning_rate_1�<�7B��\I       6%�	 ^�D���A�*;


total_loss|��@

error_RI�C?

learning_rate_1�<�7V�;lI       6%�	��D���A�*;


total_loss��A

error_R�+I?

learning_rate_1�<�7#/��I       6%�	s�D���A�*;


total_lossLƵ@

error_R��U?

learning_rate_1�<�7a�_�I       6%�	�-�D���A�*;


total_lossC~�@

error_R@FY?

learning_rate_1�<�7��I       6%�	w�D���A�*;


total_loss���@

error_R�`:?

learning_rate_1�<�7f�4rI       6%�	<��D���A�*;


total_loss!�@

error_R�TR?

learning_rate_1�<�7s�o�I       6%�	{	�D���A�*;


total_lossTk�@

error_R��d?

learning_rate_1�<�7lظGI       6%�	�W�D���A�*;


total_loss2�@

error_R��S?

learning_rate_1�<�7'll�I       6%�	e��D���A�*;


total_losslʹ@

error_RV7?

learning_rate_1�<�7�e�I       6%�	C�D���A�*;


total_loss���@

error_R��K?

learning_rate_1�<�7U��I       6%�	�6�D���A�*;


total_lossl��@

error_R-�[?

learning_rate_1�<�7i��dI       6%�	�z�D���A�*;


total_loss|�@

error_R,�D?

learning_rate_1�<�72.AhI       6%�	���D���A�*;


total_lossc�@

error_R�C?

learning_rate_1�<�7#F�5I       6%�	G�D���A�*;


total_loss�%�@

error_R}�N?

learning_rate_1�<�7N�/DI       6%�	�G�D���A�*;


total_loss6�@

error_RקM?

learning_rate_1�<�7B^I       6%�	}��D���A�*;


total_lossoYp@

error_Rnmd?

learning_rate_1�<�7�lZ�I       6%�	ԉD���A�*;


total_loss�b�@

error_R4[?

learning_rate_1�<�7�O��I       6%�	��D���A�*;


total_loss�A

error_RkY?

learning_rate_1�<�7<N�I       6%�	�_�D���A�*;


total_loss�:�@

error_R�K?

learning_rate_1�<�7�CI       6%�	���D���A�*;


total_loss�@

error_R�M?

learning_rate_1�<�7'���I       6%�	��D���A�*;


total_loss��@

error_R��B?

learning_rate_1�<�7<d$gI       6%�	!6�D���A�*;


total_loss#�@

error_R��<?

learning_rate_1�<�7��4�I       6%�	|�D���A�*;


total_loss�P�@

error_R[I?

learning_rate_1�<�7�zK�I       6%�	���D���A�*;


total_loss�R�@

error_R��Y?

learning_rate_1�<�79	*I       6%�	�	�D���A�*;


total_losslӿ@

error_R��O?

learning_rate_1�<�7�hMI       6%�	J�D���A�*;


total_loss��q@

error_R?Q?

learning_rate_1�<�7�<��I       6%�	ƍ�D���A�*;


total_loss$Ӂ@

error_RS(??

learning_rate_1�<�7�\��I       6%�	MόD���A�*;


total_lossdg�@

error_R7}m?

learning_rate_1�<�7lJ<I       6%�	��D���A�*;


total_loss1ٝ@

error_R��L?

learning_rate_1�<�7���I       6%�	�l�D���A�*;


total_loss�6�@

error_RړZ?

learning_rate_1�<�75<�I       6%�	���D���A�*;


total_losss�<A

error_RfPC?

learning_rate_1�<�7*q�I       6%�	���D���A�*;


total_loss^�@

error_R��8?

learning_rate_1�<�7a�:�I       6%�	B�D���A�*;


total_lossZ�@

error_R)O?

learning_rate_1�<�7n/��I       6%�	=��D���A�*;


total_loss=�@

error_R�CR?

learning_rate_1�<�7�$;�I       6%�	ˎD���A�*;


total_loss���@

error_R4V?

learning_rate_1�<�7@�W�I       6%�	��D���A�*;


total_loss��@

error_Rz�G?

learning_rate_1�<�7���I       6%�	R�D���A�*;


total_loss;g�@

error_RMNL?

learning_rate_1�<�7�t�I       6%�	.��D���A�*;


total_loss��@

error_R*B?

learning_rate_1�<�7����I       6%�	�ߏD���A�*;


total_lossO=m@

error_R��9?

learning_rate_1�<�7�jX�I       6%�	�'�D���A�*;


total_loss��{@

error_R=PX?

learning_rate_1�<�7�p@I       6%�	�o�D���A�*;


total_loss�,�@

error_R8nG?

learning_rate_1�<�7��i�I       6%�	��D���A�*;


total_loss���@

error_R�Q]?

learning_rate_1�<�7"n�MI       6%�	���D���A�*;


total_loss�P�@

error_R
R@?

learning_rate_1�<�7�1�II       6%�	l:�D���A�*;


total_loss�p�@

error_RY?

learning_rate_1�<�7��I       6%�	<��D���A�*;


total_loss2�A

error_R�8b?

learning_rate_1�<�7
�k�I       6%�	�ˑD���A�*;


total_lossO�A

error_RZ�[?

learning_rate_1�<�7�e��I       6%�	�D���A�*;


total_lossܮ}@

error_R1w??

learning_rate_1�<�7Em,I       6%�	�a�D���A�*;


total_loss�ɝ@

error_R�X>?

learning_rate_1�<�7?x�I       6%�	B��D���A�*;


total_loss��@

error_RλK?

learning_rate_1�<�7���I       6%�	��D���A�*;


total_loss!��@

error_ROVE?

learning_rate_1�<�7�INjI       6%�	�0�D���A�*;


total_loss�y�@

error_RSl_?

learning_rate_1�<�7"��I       6%�	�x�D���A�*;


total_lossԅ@

error_R�#T?

learning_rate_1�<�7B�'I       6%�	ý�D���A�*;


total_lossد�@

error_R��G?

learning_rate_1�<�7ЌI       6%�	P�D���A�*;


total_loss���@

error_R��^?

learning_rate_1�<�7�D�I       6%�	6P�D���A�*;


total_loss���@

error_R�tZ?

learning_rate_1�<�7���I       6%�	䐔D���A�*;


total_loss�G�@

error_R�@J?

learning_rate_1�<�7rjLI       6%�	�הD���A�*;


total_loss���@

error_Rv�[?

learning_rate_1�<�7[��I       6%�	��D���A�*;


total_lossnR�@

error_R-�B?

learning_rate_1�<�7��LI       6%�	�a�D���A�*;


total_loss�*�@

error_R��H?

learning_rate_1�<�7w,-I       6%�	/��D���A�*;


total_loss!�@

error_R��<?

learning_rate_1�<�7�+��I       6%�	��D���A�*;


total_lossV:�@

error_RݰI?

learning_rate_1�<�7�W"I       6%�	�9�D���A�*;


total_loss�r�@

error_R{HN?

learning_rate_1�<�7��4]I       6%�	w��D���A�*;


total_loss���@

error_Rl=<?

learning_rate_1�<�7��nI       6%�	ʖD���A�*;


total_loss���@

error_R��??

learning_rate_1�<�7! cI       6%�	)�D���A�*;


total_loss�Hw@

error_R{�j?

learning_rate_1�<�7F�A�I       6%�	^�D���A�*;


total_loss��@

error_R�Q?

learning_rate_1�<�7��}I       6%�	�͗D���A�*;


total_loss?��@

error_Rs??

learning_rate_1�<�7e�{lI       6%�	��D���A�*;


total_loss���@

error_RH�X?

learning_rate_1�<�7�.�I       6%�	�\�D���A�*;


total_lossH+�@

error_R�2b?

learning_rate_1�<�7T�I       6%�	T��D���A�*;


total_loss�v�@

error_R�Z?

learning_rate_1�<�79θI       6%�	�D���A�*;


total_loss<�@

error_R�H?

learning_rate_1�<�7Ƀ��I       6%�	>'�D���A�*;


total_loss66�@

error_Rq�W?

learning_rate_1�<�7��c�I       6%�	$p�D���A�*;


total_loss��@

error_R[�V?

learning_rate_1�<�7gSv I       6%�	���D���A�*;


total_loss1x�@

error_R$�G?

learning_rate_1�<�7/wI       6%�	��D���A�*;


total_lossŖ�@

error_R��D?

learning_rate_1�<�7��I       6%�	G�D���A�*;


total_loss��@

error_R�2<?

learning_rate_1�<�7(�C�I       6%�	L��D���A�*;


total_loss]�@

error_R�~V?

learning_rate_1�<�7T�PI       6%�	X˚D���A�*;


total_loss�7�@

error_R�U;?

learning_rate_1�<�7U�G�I       6%�	9�D���A�*;


total_loss7��@

error_R��O?

learning_rate_1�<�7u�^=I       6%�	;R�D���A�*;


total_loss�C�@

error_R��Y?

learning_rate_1�<�7W�qWI       6%�	��D���A�*;


total_loss�A

error_RƒM?

learning_rate_1�<�7
�I       6%�	tٛD���A�*;


total_loss	IA

error_R�5B?

learning_rate_1�<�7���I       6%�	��D���A�*;


total_loss-3�@

error_RiS?

learning_rate_1�<�7�O?)I       6%�	�_�D���A�*;


total_loss�D�@

error_R6_?

learning_rate_1�<�7ˮ��I       6%�	���D���A�*;


total_loss$��@

error_R�W:?

learning_rate_1�<�7n�I       6%�	�D���A�*;


total_loss�Q�@

error_R�T?

learning_rate_1�<�7�eI       6%�	/�D���A�*;


total_loss�=�@

error_R�>?

learning_rate_1�<�7bn�I       6%�	�r�D���A�*;


total_loss��@

error_R��;?

learning_rate_1�<�7/�BI       6%�	?��D���A�*;


total_loss���@

error_R��X?

learning_rate_1�<�71d�I       6%�	V�D���A�*;


total_loss��@

error_R�(N?

learning_rate_1�<�74��BI       6%�	�T�D���A�*;


total_lossA��@

error_R�W?

learning_rate_1�<�7`;i�I       6%�	���D���A�*;


total_loss�W�@

error_R�eC?

learning_rate_1�<�7�*bI       6%�	��D���A�*;


total_loss�!@

error_Rx�X?

learning_rate_1�<�7K�pI       6%�	I4�D���A�*;


total_loss���@

error_R��E?

learning_rate_1�<�7y~}I       6%�	�v�D���A�*;


total_loss��@

error_RcdS?

learning_rate_1�<�7Nb�tI       6%�	ܻ�D���A�*;


total_loss���@

error_RaOB?

learning_rate_1�<�7(�I       6%�	[��D���A�*;


total_loss�֬@

error_R:R?

learning_rate_1�<�7��I       6%�	`A�D���A�*;


total_loss�u�@

error_R$@?

learning_rate_1�<�7N��I       6%�	���D���A�*;


total_loss�Is@

error_Rr<b?

learning_rate_1�<�7@��zI       6%�	�ʠD���A�*;


total_lossf�A

error_RF`T?

learning_rate_1�<�7��I       6%�	��D���A�*;


total_loss�E�@

error_R�C?

learning_rate_1�<�79mmI       6%�	�R�D���A�*;


total_loss_i�@

error_R�sU?

learning_rate_1�<�79��6I       6%�	���D���A�*;


total_loss#��@

error_R*�N?

learning_rate_1�<�7"��TI       6%�	�סD���A�*;


total_loss1'�@

error_R�GS?

learning_rate_1�<�7�(�I       6%�	��D���A�*;


total_loss�ߔ@

error_R�|S?

learning_rate_1�<�7V	
�I       6%�	�c�D���A�*;


total_loss�q�@

error_R��W?

learning_rate_1�<�7OL��I       6%�	j��D���A�*;


total_loss���@

error_R�b?

learning_rate_1�<�7.֩/I       6%�	^�D���A�*;


total_loss���@

error_R<?

learning_rate_1�<�7���I       6%�	;�D���A�*;


total_loss���@

error_R��=?

learning_rate_1�<�7($3I       6%�	�{�D���A�*;


total_loss�Ƅ@

error_RN�D?

learning_rate_1�<�7d���I       6%�	���D���A�*;


total_loss3��@

error_R,�N?

learning_rate_1�<�7`K'HI       6%�	��D���A�*;


total_lossҵ@

error_R:�O?

learning_rate_1�<�7��OI       6%�	MG�D���A�*;


total_loss��@

error_RnM?

learning_rate_1�<�7&>�I       6%�	X��D���A�*;


total_lossC��@

error_R�N?

learning_rate_1�<�7^
O$I       6%�	ѤD���A�*;


total_loss�;�@

error_RT�[?

learning_rate_1�<�7�A�7I       6%�	|�D���A�*;


total_loss	��@

error_R.�T?

learning_rate_1�<�7�]W�I       6%�	�[�D���A�*;


total_loss�\�@

error_R��K?

learning_rate_1�<�7M�ުI       6%�	���D���A�*;


total_loss��l@

error_R�IQ?

learning_rate_1�<�7�,tI       6%�	��D���A�*;


total_loss�H�@

error_RJT?

learning_rate_1�<�7�%�I       6%�	�*�D���A�*;


total_loss
��@

error_R�`_?

learning_rate_1�<�7��qI       6%�	:n�D���A�*;


total_loss:��@

error_R��1?

learning_rate_1�<�728>�I       6%�	O��D���A�*;


total_loss�@

error_RIv;?

learning_rate_1�<�7�R�.I       6%�	���D���A�*;


total_loss��A

error_R��W?

learning_rate_1�<�7s��I       6%�	eV�D���A�*;


total_loss{v�@

error_R�X?

learning_rate_1�<�7U�c�I       6%�	�D���A�*;


total_loss�L�@

error_R�!V?

learning_rate_1�<�7_OR�I       6%�	<�D���A�*;


total_loss́�@

error_R��R?

learning_rate_1�<�7��>I       6%�	s6�D���A�*;


total_loss��@

error_R�L?

learning_rate_1�<�7~�(tI       6%�	
}�D���A�*;


total_loss���@

error_R��K?

learning_rate_1�<�7i8��I       6%�	�ȨD���A�*;


total_loss4��@

error_Rh&D?

learning_rate_1�<�7�/SI       6%�	��D���A�*;


total_lossC�@

error_R�}R?

learning_rate_1�<�7���I       6%�	T�D���A�*;


total_loss�͕@

error_R�O?

learning_rate_1�<�7�TE^I       6%�	s��D���A�*;


total_loss�@

error_R.�P?

learning_rate_1�<�7�ˈ I       6%�	|ةD���A�*;


total_loss_�@

error_Rl�J?

learning_rate_1�<�7W6�I       6%�	��D���A�*;


total_loss�@A

error_R�N?

learning_rate_1�<�7r�I       6%�	�[�D���A�*;


total_lossiA

error_R�E?

learning_rate_1�<�7��UI       6%�	���D���A�*;


total_loss�ә@

error_R�H?

learning_rate_1�<�7I���I       6%�	�ݪD���A�*;


total_loss�j�@

error_R�W?

learning_rate_1�<�7�I��I       6%�	�!�D���A�*;


total_loss�3A

error_R,^e?

learning_rate_1�<�7/�I       6%�	7j�D���A�*;


total_lossȏ@

error_R�J?

learning_rate_1�<�7nz�8I       6%�	A��D���A�*;


total_lossjx�@

error_RS�A?

learning_rate_1�<�7�.��I       6%�	/ �D���A�*;


total_loss]��@

error_R��R?

learning_rate_1�<�7���mI       6%�	/G�D���A�*;


total_loss���@

error_R_Q?

learning_rate_1�<�7B�zI       6%�	���D���A�*;


total_loss<f�@

error_R�I?

learning_rate_1�<�7ɦ��I       6%�	�٬D���A�*;


total_loss��@

error_RvR?

learning_rate_1�<�78M�I       6%�	�#�D���A�*;


total_loss\�a@

error_R�rM?

learning_rate_1�<�7d�I       6%�	jo�D���A�*;


total_loss.��@

error_RHfN?

learning_rate_1�<�79�4�I       6%�	���D���A�*;


total_lossA�@

error_R��A?

learning_rate_1�<�7.��I       6%�	���D���A�*;


total_lossf�@

error_R��W?

learning_rate_1�<�7��+SI       6%�	�<�D���A�*;


total_loss���@

error_R*�<?

learning_rate_1�<�7��F�I       6%�	;|�D���A�*;


total_loss�̠@

error_RfM?

learning_rate_1�<�7��F�I       6%�	���D���A�*;


total_loss��@

error_R)e?

learning_rate_1�<�7n�_0I       6%�	���D���A�*;


total_loss�$�@

error_R�"S?

learning_rate_1�<�7���I       6%�	B�D���A�*;


total_loss@�@

error_RD�N?

learning_rate_1�<�7^�52I       6%�	�D���A�*;


total_loss
?�@

error_R;�D?

learning_rate_1�<�7÷&�I       6%�	zͯD���A�*;


total_loss% �@

error_R�N?

learning_rate_1�<�7U��I       6%�	^�D���A�*;


total_loss���@

error_R�K?

learning_rate_1�<�7H���I       6%�	`�D���A�*;


total_loss��@

error_R�:?

learning_rate_1�<�7.1oI       6%�	>��D���A�*;


total_loss��@

error_RظI?

learning_rate_1�<�7�Xf-I       6%�	g�D���A�*;


total_loss��@

error_R�L@?

learning_rate_1�<�7l��/I       6%�	�0�D���A�*;


total_losstb�@

error_Rn�D?

learning_rate_1�<�7�K�I       6%�	�y�D���A�*;


total_loss��@

error_Rs�d?

learning_rate_1�<�7 d$qI       6%�	{��D���A�*;


total_loss�ݠ@

error_R IF?

learning_rate_1�<�7S[V�I       6%�	�
�D���A�*;


total_loss�و@

error_R$zW?

learning_rate_1�<�7K�$I       6%�	�O�D���A�*;


total_loss��@

error_R�Y\?

learning_rate_1�<�7��7jI       6%�	���D���A�*;


total_loss���@

error_R�_?

learning_rate_1�<�7��lI       6%�	�߲D���A�*;


total_loss3W�@

error_R�gO?

learning_rate_1�<�7^X2sI       6%�	+&�D���A�*;


total_loss��@

error_R�3Y?

learning_rate_1�<�7GE�+I       6%�	$o�D���A�*;


total_lossԫ�@

error_R��]?

learning_rate_1�<�7�:�I       6%�	���D���A�*;


total_loss�Ҳ@

error_Rl�S?

learning_rate_1�<�79�&�I       6%�	��D���A�*;


total_loss,�@

error_RMHO?

learning_rate_1�<�7|��I       6%�	�E�D���A�*;


total_loss��@

error_R��J?

learning_rate_1�<�7��7I       6%�	��D���A�*;


total_loss暐@

error_R��a?

learning_rate_1�<�7m��I       6%�	δD���A�*;


total_lossФ@

error_RUQ?

learning_rate_1�<�7�k�:I       6%�	��D���A�*;


total_lossӳ�@

error_R,�V?

learning_rate_1�<�7�e�I       6%�	 U�D���A�*;


total_lossAc�@

error_R:\?

learning_rate_1�<�7��I       6%�	旵D���A�*;


total_loss���@

error_Rn<?

learning_rate_1�<�7ϐyI       6%�	?޵D���A�*;


total_loss�>A

error_R�e?

learning_rate_1�<�7B>��I       6%�	 �D���A�*;


total_loss��@

error_R`nS?

learning_rate_1�<�7����I       6%�	Mf�D���A�*;


total_lossO��@

error_R�@?

learning_rate_1�<�7�hI       6%�	0��D���A�*;


total_loss?H�@

error_R��T?

learning_rate_1�<�7͈*�I       6%�	���D���A�*;


total_loss���@

error_R�cT?

learning_rate_1�<�7�� �I       6%�	�C�D���A�*;


total_lossy�@

error_R��P?

learning_rate_1�<�7m��I       6%�	'��D���A�*;


total_lossmn�@

error_R�:J?

learning_rate_1�<�7�9=iI       6%�	L�D���A�*;


total_loss:�@

error_R�pM?

learning_rate_1�<�7���I       6%�	?9�D���A�*;


total_loss�V�@

error_R�\?

learning_rate_1�<�7��:I       6%�	x{�D���A�*;


total_loss���@

error_R�6L?

learning_rate_1�<�7�i��I       6%�	���D���A�*;


total_lossv��@

error_R�w@?

learning_rate_1�<�7����I       6%�	��D���A�*;


total_loss��@

error_Rq�Q?

learning_rate_1�<�7�뢟I       6%�	�L�D���A�*;


total_lossr��@

error_R��F?

learning_rate_1�<�7U�N�I       6%�	T��D���A�*;


total_lossI�@

error_RTRC?

learning_rate_1�<�7�>
I       6%�	�չD���A�*;


total_loss��@

error_R��K?

learning_rate_1�<�7(��I       6%�	J8�D���A�*;


total_loss:
�@

error_R�	Z?

learning_rate_1�<�7��N�I       6%�	Ă�D���A�*;


total_lossE��@

error_R�bf?

learning_rate_1�<�7zNI       6%�	�кD���A�*;


total_loss��@

error_Rn�7?

learning_rate_1�<�7��L�I       6%�	�D���A�*;


total_lossQj�@

error_R X?

learning_rate_1�<�7���I       6%�	bf�D���A�*;


total_loss�n@

error_RHcL?

learning_rate_1�<�7��I       6%�	@��D���A�*;


total_loss)Л@

error_RؤF?

learning_rate_1�<�7yѠI       6%�	���D���A�*;


total_loss���@

error_R3QD?

learning_rate_1�<�7Rx<I       6%�	�4�D���A�*;


total_loss��@

error_R��V?

learning_rate_1�<�7P�I       6%�	�x�D���A�*;


total_loss�k�@

error_R�7M?

learning_rate_1�<�7�1]I       6%�	n��D���A�*;


total_loss�}@

error_R$"Y?

learning_rate_1�<�7�ĠI       6%�	}�D���A�*;


total_loss��@

error_R�R?

learning_rate_1�<�7v(I       6%�	�J�D���A�*;


total_lossQ��@

error_R��R?

learning_rate_1�<�7!(`�I       6%�	s��D���A�*;


total_loss욌@

error_R�BM?

learning_rate_1�<�7�iJI       6%�	�ҽD���A�*;


total_loss�:�@

error_R��Y?

learning_rate_1�<�7���#I       6%�	��D���A�*;


total_loss6Ѯ@

error_R��N?

learning_rate_1�<�7���I       6%�	�X�D���A�*;


total_loss���@

error_R��Z?

learning_rate_1�<�77k^�I       6%�	D���A�*;


total_lossc��@

error_R_�W?

learning_rate_1�<�7��<zI       6%�	l�D���A�*;


total_loss_�@

error_RlST?

learning_rate_1�<�7�lʢI       6%�	�-�D���A�*;


total_loss8ڼ@

error_RԬP?

learning_rate_1�<�70�BI       6%�	�u�D���A�*;


total_loss|ڵ@

error_R��R?

learning_rate_1�<�7� r�I       6%�	���D���A�*;


total_loss��@

error_R��>?

learning_rate_1�<�7kc�iI       6%�	1��D���A�*;


total_loss�e�@

error_R��@?

learning_rate_1�<�7	��I       6%�	�<�D���A�*;


total_loss`�@

error_R�O?

learning_rate_1�<�78��I       6%�	���D���A�*;


total_lossT��@

error_R[�S?

learning_rate_1�<�7
�XI       6%�	���D���A�*;


total_loss�`�@

error_R�bC?

learning_rate_1�<�7hd�I       6%�	�	�D���A�*;


total_lossS��@

error_R�H?

learning_rate_1�<�7U�2I       6%�	�I�D���A�*;


total_loss}��@

error_R �Q?

learning_rate_1�<�7��JI       6%�	��D���A�*;


total_loss�g�@

error_R�sL?

learning_rate_1�<�7~خ�I       6%�	Q��D���A�*;


total_loss	|�@

error_R!�c?

learning_rate_1�<�7�8I       6%�	�$�D���A�*;


total_lossϲ�@

error_R��E?

learning_rate_1�<�7�	�I       6%�	�l�D���A�*;


total_loss ��@

error_RU?

learning_rate_1�<�7.
6I       6%�	r��D���A�*;


total_lossXI�@

error_R�L?

learning_rate_1�<�7KRt�I       6%�	F��D���A�*;


total_loss�;�@

error_RR>?

learning_rate_1�<�7O��I       6%�	�8�D���A�*;


total_loss�/A

error_RM@V?

learning_rate_1�<�7I �I       6%�	�z�D���A�*;


total_loss��@

error_R�R?

learning_rate_1�<�7f���I       6%�	��D���A�*;


total_loss�(�@

error_R�S@?

learning_rate_1�<�78�2�I       6%�	�D���A�*;


total_loss�L�@

error_R��H?

learning_rate_1�<�7��I�I       6%�	�K�D���A�*;


total_loss/�A

error_REJT?

learning_rate_1�<�7��N�I       6%�	���D���A�*;


total_lossI��@

error_R�D?

learning_rate_1�<�7:I       6%�	���D���A�*;


total_loss60�@

error_R*�M?

learning_rate_1�<�7�}KI       6%�	�'�D���A�*;


total_loss=]�@

error_R��9?

learning_rate_1�<�7�z�I       6%�	�q�D���A�*;


total_lossk��@

error_R�KR?

learning_rate_1�<�7-)��I       6%�	U��D���A�*;


total_lossE��@

error_R�D?

learning_rate_1�<�7.��I       6%�	;�D���A�*;


total_lossDϲ@

error_R�J_?

learning_rate_1�<�7���I       6%�	eL�D���A�*;


total_loss��@

error_R@�N?

learning_rate_1�<�7�yɸI       6%�	���D���A�*;


total_loss\S�@

error_R8I?

learning_rate_1�<�7 [�eI       6%�	���D���A�*;


total_loss�f�@

error_R��Z?

learning_rate_1�<�7� ��I       6%�	��D���A�*;


total_loss�~�@

error_R��<?

learning_rate_1�<�7�;��I       6%�	���D���A�*;


total_loss�U�@

error_R ~K?

learning_rate_1�<�7�/0�I       6%�	���D���A�*;


total_loss@V�@

error_Rr�L?

learning_rate_1�<�7��I       6%�	"�D���A�*;


total_loss1��@

error_R�JC?

learning_rate_1�<�7;#kI       6%�	he�D���A�*;


total_loss�~@

error_R%�N?

learning_rate_1�<�7��.�I       6%�	7��D���A�*;


total_loss���@

error_RZ`?

learning_rate_1�<�7H�7I       6%�	���D���A�*;


total_loss��A

error_R�M?

learning_rate_1�<�7A�_�I       6%�	�9�D���A�*;


total_lossF��@

error_R��I?

learning_rate_1�<�7��mI       6%�	-}�D���A�*;


total_loss<9�@

error_R�S?

learning_rate_1�<�7GUI       6%�	���D���A�*;


total_loss7f�@

error_R�B?

learning_rate_1�<�7ÝP�I       6%�	��D���A�*;


total_loss��@

error_R��G?

learning_rate_1�<�7TQ��I       6%�	�I�D���A�*;


total_loss���@

error_R6x`?

learning_rate_1�<�7T HyI       6%�	���D���A�*;


total_lossh��@

error_R�TT?

learning_rate_1�<�7���I       6%�	S��D���A�*;


total_loss�Ο@

error_R�J?

learning_rate_1�<�7:��I       6%�	�D���A�*;


total_loss��@

error_R/T?

learning_rate_1�<�7�L-I       6%�	`P�D���A�*;


total_losszE�@

error_R`+L?

learning_rate_1�<�775��I       6%�	���D���A�*;


total_lossl��@

error_R�sR?

learning_rate_1�<�7�A�pI       6%�	���D���A�*;


total_loss:��@

error_R�L?

learning_rate_1�<�7Z�II       6%�	F�D���A�*;


total_loss�x�@

error_R�;?

learning_rate_1�<�7�Z�I       6%�	m_�D���A�*;


total_lossX7�@

error_R�YM?

learning_rate_1�<�7����I       6%�	��D���A�*;


total_loss���@

error_R�N?

learning_rate_1�<�7��CGI       6%�	���D���A�*;


total_lossCf�@

error_R��L?

learning_rate_1�<�7�Bg�I       6%�	E&�D���A�*;


total_loss]т@

error_R�W?

learning_rate_1�<�7�^�I       6%�	'g�D���A�*;


total_lossS(�@

error_R�[?

learning_rate_1�<�7��BI       6%�	���D���A�*;


total_lossm��@

error_Rl6I?

learning_rate_1�<�7!�lI       6%�	���D���A�*;


total_loss��@

error_R��S?

learning_rate_1�<�7�p�rI       6%�	q0�D���A�*;


total_loss�'�@

error_R��X?

learning_rate_1�<�7OP}�I       6%�	ew�D���A�*;


total_loss�/	A

error_RV�G?

learning_rate_1�<�7�}��I       6%�	���D���A�*;


total_lossί�@

error_R��9?

learning_rate_1�<�7��q�I       6%�	��D���A�*;


total_loss��@

error_R,�H?

learning_rate_1�<�7��I       6%�	aP�D���A�*;


total_lossH�@

error_R��U?

learning_rate_1�<�7��TrI       6%�	Ř�D���A�*;


total_loss��@

error_R��Q?

learning_rate_1�<�7�>DI       6%�	���D���A�*;


total_loss�Ox@

error_R�\A?

learning_rate_1�<�7�>�I       6%�	7*�D���A�*;


total_loss\��@

error_RST[?

learning_rate_1�<�71��pI       6%�	 n�D���A�*;


total_loss߮�@

error_R!RT?

learning_rate_1�<�7����I       6%�	���D���A�*;


total_loss���@

error_R=�X?

learning_rate_1�<�7��eI       6%�	���D���A�*;


total_loss���@

error_R�8[?

learning_rate_1�<�7���II       6%�	 :�D���A�*;


total_lossH��@

error_RXth?

learning_rate_1�<�7z�I       6%�	��D���A�*;


total_loss�ܫ@

error_R��G?

learning_rate_1�<�7lJ��I       6%�	���D���A�*;


total_loss9�A

error_R�PM?

learning_rate_1�<�7��^I       6%�	�
�D���A�*;


total_loss���@

error_R,�]?

learning_rate_1�<�7��I       6%�	O�D���A�*;


total_loss���@

error_R��V?

learning_rate_1�<�7Е��I       6%�	ϒ�D���A�*;


total_loss�$�@

error_R��I?

learning_rate_1�<�7�dT~I       6%�	 ��D���A�*;


total_loss���@

error_RW^E?

learning_rate_1�<�7��17I       6%�	��D���A�*;


total_lossJ��@

error_Rj�g?

learning_rate_1�<�7��2I       6%�	G`�D���A�*;


total_loss�+�@

error_Rq�U?

learning_rate_1�<�7���I       6%�	���D���A�*;


total_loss�_�@

error_R�;?

learning_rate_1�<�7�P�)I       6%�	���D���A�*;


total_lossO	�@

error_R@~_?

learning_rate_1�<�7~�w�I       6%�	�&�D���A�*;


total_lossր�@

error_Re{T?

learning_rate_1�<�7&2AdI       6%�	�i�D���A�*;


total_lossƤ@

error_R�J?

learning_rate_1�<�7�i>�I       6%�	���D���A�*;


total_loss�@

error_R�X?

learning_rate_1�<�7P��I       6%�	�
�D���A�*;


total_lossO��@

error_RL�@?

learning_rate_1�<�7�	'�I       6%�	xU�D���A�*;


total_lossT��@

error_R��J?

learning_rate_1�<�7���I       6%�	Ϝ�D���A�*;


total_lossa�@

error_Ro�J?

learning_rate_1�<�7S�FI       6%�	���D���A�*;


total_lossdl�@

error_Rz5?

learning_rate_1�<�7����I       6%�	�%�D���A�*;


total_loss=m�@

error_RC R?

learning_rate_1�<�7nQ[I       6%�	�l�D���A�*;


total_loss�ݛ@

error_RO�>?

learning_rate_1�<�7(��_I       6%�	���D���A�*;


total_lossd A

error_R�?]?

learning_rate_1�<�7n��BI       6%�	���D���A�*;


total_loss�"�@

error_R�FP?

learning_rate_1�<�7��tI       6%�	�D�D���A�*;


total_loss��@

error_R7*`?

learning_rate_1�<�7/�I       6%�	��D���A�*;


total_loss���@

error_RT-S?

learning_rate_1�<�7�D;I       6%�	V��D���A�*;


total_lossH\�@

error_R [?

learning_rate_1�<�7�w��I       6%�	+>�D���A�*;


total_loss�n�@

error_R6F?

learning_rate_1�<�7F��I       6%�	��D���A�*;


total_loss��@

error_Rn�]?

learning_rate_1�<�7�r�kI       6%�	���D���A�*;


total_loss��@

error_R�4O?

learning_rate_1�<�7*�gI       6%�	�
�D���A�*;


total_loss�p�@

error_R)zT?

learning_rate_1�<�7!�'�I       6%�	�K�D���A�*;


total_loss���@

error_R��M?

learning_rate_1�<�7���I       6%�	��D���A�*;


total_lossxZ�@

error_RP?

learning_rate_1�<�7��ͭI       6%�	���D���A�*;


total_loss6�@

error_R��@?

learning_rate_1�<�7��PiI       6%�	"�D���A�*;


total_loss��@

error_R�JI?

learning_rate_1�<�7f��rI       6%�	�R�D���A�*;


total_loss��@

error_Rv�H?

learning_rate_1�<�7�JvI       6%�	q��D���A�*;


total_loss�0�@

error_R��H?

learning_rate_1�<�7hf,I       6%�	��D���A�*;


total_loss���@

error_RqU?

learning_rate_1�<�7�{�+I       6%�	��D���A�*;


total_lossa��@

error_R��K?

learning_rate_1�<�7[{�I       6%�	�[�D���A�*;


total_loss�e�@

error_R��O?

learning_rate_1�<�7� �I       6%�	���D���A�*;


total_loss���@

error_R�,T?

learning_rate_1�<�7�"��I       6%�	���D���A�*;


total_lossn��@

error_RIgU?

learning_rate_1�<�7X���I       6%�	}&�D���A�*;


total_loss�α@

error_R��Q?

learning_rate_1�<�7�BR�I       6%�	�j�D���A�*;


total_loss�r�@

error_RZ.:?

learning_rate_1�<�7I/bGI       6%�	-��D���A�*;


total_loss*2�@

error_R �Q?

learning_rate_1�<�7(W�I       6%�	]��D���A�*;


total_loss��s@

error_R��D?

learning_rate_1�<�7��$I       6%�	�@�D���A�*;


total_loss|��@

error_R��O?

learning_rate_1�<�7�R@;I       6%�	=��D���A�*;


total_loss�k�@

error_RMA?

learning_rate_1�<�7	���I       6%�	(��D���A�*;


total_loss���@

error_R��M?

learning_rate_1�<�7#��I       6%�	�!�D���A�*;


total_loss�e�@

error_R��R?

learning_rate_1�<�7I5�@I       6%�	ve�D���A�*;


total_loss6ί@

error_RQ�S?

learning_rate_1�<�7OG�7I       6%�	a��D���A�*;


total_lossr��@

error_R�B??

learning_rate_1�<�7-D;I       6%�	��D���A�*;


total_loss��@

error_R��b?

learning_rate_1�<�7����I       6%�	�1�D���A�*;


total_loss�M A

error_Rs7?

learning_rate_1�<�7hkI       6%�	.t�D���A�*;


total_lossM<�@

error_R�SH?

learning_rate_1�<�7,8�1I       6%�	���D���A�*;


total_losscv�@

error_RA�A?

learning_rate_1�<�7�$2I       6%�	#�D���A�*;


total_lossd�@

error_R�V?

learning_rate_1�<�7cVn�I       6%�	rM�D���A�*;


total_loss"Ȅ@

error_R�R?

learning_rate_1�<�7��^�I       6%�	ŕ�D���A�*;


total_loss���@

error_R��G?

learning_rate_1�<�7], eI       6%�	r��D���A�*;


total_loss��@

error_RJ�Q?

learning_rate_1�<�7�M=�I       6%�	"(�D���A�*;


total_loss,�@

error_R��W?

learning_rate_1�<�7*IϐI       6%�	?l�D���A�*;


total_loss��1A

error_R�@?

learning_rate_1�<�7v��I       6%�	ͳ�D���A�*;


total_loss�2�@

error_R��R?

learning_rate_1�<�7�^*NI       6%�	���D���A�*;


total_loss��A

error_R8�O?

learning_rate_1�<�7qS�lI       6%�	�9�D���A�*;


total_loss��@

error_R1�T?

learning_rate_1�<�7$�@�I       6%�	�{�D���A�*;


total_lossET�@

error_RW�??

learning_rate_1�<�7;�j{I       6%�	J��D���A�*;


total_loss3*�@

error_R�!B?

learning_rate_1�<�7|�8'I       6%�	��D���A�*;


total_loss�B�@

error_R)�Z?

learning_rate_1�<�7�>|~I       6%�	�L�D���A�*;


total_loss��@

error_R�>?

learning_rate_1�<�7�^`-I       6%�	͏�D���A�*;


total_lossGwA

error_R�g?

learning_rate_1�<�7��}�I       6%�	��D���A�*;


total_loss﷦@

error_R$�E?

learning_rate_1�<�720�I       6%�	��D���A�*;


total_loss�b�@

error_R�/L?

learning_rate_1�<�7�I       6%�	]�D���A�*;


total_lossS��@

error_R@/e?

learning_rate_1�<�7A	pI       6%�	��D���A�*;


total_loss8Cj@

error_R��K?

learning_rate_1�<�7�D�yI       6%�	'��D���A�*;


total_loss�T�@

error_R/~J?

learning_rate_1�<�7e���I       6%�	�3�D���A�*;


total_loss�;�@

error_R�2X?

learning_rate_1�<�7X>��I       6%�	g}�D���A�*;


total_loss�&�@

error_R�O?

learning_rate_1�<�7����I       6%�	���D���A�*;


total_loss��@

error_R�^L?

learning_rate_1�<�7\��I       6%�	% �D���A�*;


total_loss���@

error_R�IV?

learning_rate_1�<�7�5�II       6%�	�A�D���A�*;


total_loss�H�@

error_RJ�9?

learning_rate_1�<�7���I       6%�	<��D���A�*;


total_loss��@

error_RC3\?

learning_rate_1�<�75�I       6%�	���D���A�*;


total_loss�/�@

error_R�DF?

learning_rate_1�<�7|�$�I       6%�	-�D���A�*;


total_lossxas@

error_Rl)I?

learning_rate_1�<�7�C~I       6%�	k�D���A�*;


total_loss�2�@

error_R��a?

learning_rate_1�<�7�4��I       6%�	���D���A�*;


total_loss �1A

error_R��B?

learning_rate_1�<�7+ö�I       6%�	��D���A�*;


total_loss4��@

error_Rtd?

learning_rate_1�<�7-`{I       6%�	�^�D���A�*;


total_lossN;�@

error_R�D?

learning_rate_1�<�7�(�I       6%�	���D���A�*;


total_loss�Ka@

error_RstN?

learning_rate_1�<�7�ēI       6%�	���D���A�*;


total_loss�@

error_R�^Y?

learning_rate_1�<�7�~�]I       6%�	�(�D���A�*;


total_loss��@

error_RםJ?

learning_rate_1�<�7v��I       6%�	�i�D���A�*;


total_loss#�@

error_Ra�W?

learning_rate_1�<�7�p�I       6%�	L��D���A�*;


total_loss�+�@

error_R�R?

learning_rate_1�<�7�MȌI       6%�	N��D���A�*;


total_loss�xA

error_R,$Q?

learning_rate_1�<�7�yw�I       6%�	�3�D���A�*;


total_loss��f@

error_R��[?

learning_rate_1�<�7;��^I       6%�	�w�D���A�*;


total_loss���@

error_R�E?

learning_rate_1�<�7$���I       6%�	t��D���A�*;


total_loss��@

error_R,�Q?

learning_rate_1�<�7<�Y+I       6%�	|��D���A�*;


total_lossћ@

error_R��`?

learning_rate_1�<�7��T	I       6%�	�H�D���A�*;


total_loss,h�@

error_R)D?

learning_rate_1�<�7�zh�I       6%�	��D���A�*;


total_loss���@

error_R̓M?

learning_rate_1�<�7��ސI       6%�	��D���A�*;


total_lossd��@

error_R�^?

learning_rate_1�<�7Y+,QI       6%�	j��D���A�*;


total_lossݽ�@

error_R�ci?

learning_rate_1�<�7uh��I       6%�	c9�D���A�*;


total_loss4�@

error_R��Q?

learning_rate_1�<�7}�\oI       6%�	3��D���A�*;


total_loss��@

error_R�"L?

learning_rate_1�<�7�O�I       6%�	D��D���A�*;


total_loss-(�@

error_R�LN?

learning_rate_1�<�7T�elI       6%�	%�D���A�*;


total_lossY�@

error_R�MP?

learning_rate_1�<�7�a�I       6%�	�F�D���A�*;


total_loss�~@

error_R1sR?

learning_rate_1�<�7��I       6%�	"��D���A�*;


total_loss,[�@

error_R��R?

learning_rate_1�<�7�@YPI       6%�	l��D���A�*;


total_loss%�@

error_R1pL?

learning_rate_1�<�7]�݉I       6%�	�/�D���A�*;


total_loss<d�@

error_R��:?

learning_rate_1�<�7�-d�I       6%�	'q�D���A�*;


total_loss��|@

error_R�XR?

learning_rate_1�<�7&�G1I       6%�	ʱ�D���A�*;


total_loss���@

error_Rv�B?

learning_rate_1�<�7�ϗ&I       6%�	���D���A�*;


total_loss��A

error_R��R?

learning_rate_1�<�7�q�I       6%�	�7�D���A�*;


total_lossX�A

error_Rn�M?

learning_rate_1�<�7�'I�I       6%�	�|�D���A�*;


total_loss���@

error_R×M?

learning_rate_1�<�7�/<I       6%�	g��D���A�*;


total_loss�f�@

error_R!1\?

learning_rate_1�<�7}��I       6%�	��D���A�*;


total_loss��A

error_R�E?

learning_rate_1�<�7P��"I       6%�	�J�D���A�*;


total_lossSq�@

error_R��D?

learning_rate_1�<�7���I       6%�	��D���A�*;


total_lossTA�@

error_R��7?

learning_rate_1�<�7-q*�I       6%�	���D���A�*;


total_loss��@

error_R��K?

learning_rate_1�<�7� �PI       6%�	��D���A�*;


total_loss�է@

error_RϛD?

learning_rate_1�<�7Ao3�I       6%�	]�D���A�*;


total_loss�7�@

error_RdP?

learning_rate_1�<�7�v�I       6%�	Ϣ�D���A�*;


total_loss�,�@

error_Rn�g?

learning_rate_1�<�7��EI       6%�	+��D���A�*;


total_loss�n�@

error_R��J?

learning_rate_1�<�7ǈ��I       6%�	�&�D���A�*;


total_lossD��@

error_R,�E?

learning_rate_1�<�7�=*I       6%�	\m�D���A�*;


total_lossvs�@

error_R8�H?

learning_rate_1�<�7S+f�I       6%�	Ƕ�D���A�*;


total_loss��@

error_R�AM?

learning_rate_1�<�7�6�I       6%�	���D���A�*;


total_loss/��@

error_RZ�M?

learning_rate_1�<�78&dI       6%�	�D�D���A�*;


total_loss���@

error_R�3M?

learning_rate_1�<�7�j=�I       6%�	��D���A�*;


total_loss��@

error_R�,S?

learning_rate_1�<�7#��LI       6%�	C��D���A�*;


total_loss�@

error_Ro�M?

learning_rate_1�<�7�&6I       6%�	�D���A�*;


total_loss��@

error_R�V?

learning_rate_1�<�7g�I       6%�	�O�D���A�*;


total_loss!}�@

error_R��Q?

learning_rate_1�<�7H�
uI       6%�	���D���A�*;


total_lossh��@

error_RM�M?

learning_rate_1�<�7�G�I       6%�	m��D���A�*;


total_loss'4�@

error_R�U?

learning_rate_1�<�7*�	�I       6%�	$8�D���A�*;


total_loss���@

error_R��K?

learning_rate_1�<�7t׍�I       6%�	�z�D���A�*;


total_loss���@

error_R��`?

learning_rate_1�<�7S�]I       6%�	��D���A�*;


total_loss���@

error_R�G@?

learning_rate_1�<�7��"dI       6%�	�	�D���A�*;


total_loss8�i@

error_R�SG?

learning_rate_1�<�7'v��I       6%�	?O�D���A�*;


total_loss�1A

error_RxL?

learning_rate_1�<�7�8��I       6%�	���D���A�*;


total_lossD��@

error_R
&_?

learning_rate_1�<�7;.�I       6%�	���D���A�*;


total_loss"V�@

error_RvjJ?

learning_rate_1�<�7�X;�I       6%�	J�D���A�*;


total_lossd�@

error_Rq�N?

learning_rate_1�<�7���;I       6%�	:��D���A�*;


total_loss���@

error_RIJ?

learning_rate_1�<�7��tI       6%�	���D���A�*;


total_loss���@

error_R�?U?

learning_rate_1�<�7�Y��I       6%�	�G�D���A�*;


total_loss��@

error_R#XF?

learning_rate_1�<�7���;I       6%�	��D���A�*;


total_loss!��@

error_RO�E?

learning_rate_1�<�7f@�GI       6%�	��D���A�*;


total_loss�h�@

error_R[�^?

learning_rate_1�<�7J$I       6%�	�:�D���A�*;


total_loss�!�@

error_R�uA?

learning_rate_1�<�77
��I       6%�	���D���A�*;


total_loss)WA

error_R�\?

learning_rate_1�<�7=���I       6%�	U��D���A�*;


total_loss��@

error_R�D?

learning_rate_1�<�7=+DI       6%�	3+�D���A�*;


total_loss�>�@

error_RaDQ?

learning_rate_1�<�7�.I       6%�	Bt�D���A�*;


total_loss�͟@

error_Rn}6?

learning_rate_1�<�7�D��I       6%�	���D���A�*;


total_loss��y@

error_RY?

learning_rate_1�<�7{xauI       6%�	��D���A�*;


total_lossߴ�@

error_R�)_?

learning_rate_1�<�7��~I       6%�	Fb�D���A�*;


total_loss6�@

error_R��N?

learning_rate_1�<�7�h�I       6%�	t��D���A�*;


total_loss�A

error_R�^I?

learning_rate_1�<�7�(�I       6%�	��D���A�*;


total_lossR�@

error_R8U?

learning_rate_1�<�7̓��I       6%�	�Q�D���A�*;


total_loss��@

error_R�c?

learning_rate_1�<�7���+I       6%�	R��D���A�*;


total_loss4%�@

error_R,�N?

learning_rate_1�<�7d*�I       6%�	��D���A�*;


total_loss#��@

error_R�{B?

learning_rate_1�<�7�A��I       6%�	z. E���A�*;


total_loss3A�@

error_R�RT?

learning_rate_1�<�7�E@�I       6%�	x E���A�*;


total_loss���@

error_Rv-n?

learning_rate_1�<�7�5�7I       6%�	�� E���A�*;


total_loss�g�@

error_RS?

learning_rate_1�<�71���I       6%�	�E���A�*;


total_loss�	�@

error_Rs�C?

learning_rate_1�<�7RK3I       6%�	�DE���A�*;


total_lossا@

error_R�D^?

learning_rate_1�<�7��h�I       6%�	��E���A�*;


total_loss�@

error_Rn@?

learning_rate_1�<�7 2��I       6%�	>�E���A�*;


total_loss�˸@

error_RCvP?

learning_rate_1�<�7�
�I       6%�	E���A�*;


total_loss#��@

error_Rj�@?

learning_rate_1�<�7+�@DI       6%�	�VE���A�*;


total_loss�c�@

error_R�X?

learning_rate_1�<�7^��I       6%�	z�E���A�*;


total_loss��@

error_RSI?

learning_rate_1�<�7E�I       6%�	K�E���A�*;


total_loss${�@

error_R�mF?

learning_rate_1�<�7iW�I       6%�	.E���A�*;


total_loss���@

error_R6ML?

learning_rate_1�<�7Q�+�I       6%�	1xE���A�*;


total_loss���@

error_R�N?

learning_rate_1�<�7P��I       6%�	<�E���A�*;


total_lossä�@

error_R��@?

learning_rate_1�<�7��� I       6%�	7�E���A�*;


total_lossl�@

error_R�nO?

learning_rate_1�<�7Iǁ�I       6%�	�DE���A�*;


total_lossc�p@

error_R�
P?

learning_rate_1�<�7��;I       6%�	o�E���A�*;


total_loss���@

error_R�C?

learning_rate_1�<�77�I       6%�	3�E���A�*;


total_lossE�@

error_R�dM?

learning_rate_1�<�7.7��I       6%�	E���A�*;


total_lossvM�@

error_R�FI?

learning_rate_1�<�7�s�I       6%�	CTE���A�*;


total_loss��q@

error_R��;?

learning_rate_1�<�7�rj�I       6%�	=�E���A�*;


total_loss�L�@

error_R�<@?

learning_rate_1�<�7%��wI       6%�	(�E���A�*;


total_loss]�@

error_R=>B?

learning_rate_1�<�7k��I       6%�	-E���A�*;


total_lossp]�@

error_Rx�4?

learning_rate_1�<�7Я��I       6%�	XtE���A�*;


total_loss�]�@

error_R��J?

learning_rate_1�<�72���I       6%�	ɸE���A�*;


total_loss j�@

error_R�E?

learning_rate_1�<�7��z�I       6%�	��E���A�*;


total_loss���@

error_RڬS?

learning_rate_1�<�7�!xI       6%�	AE���A�*;


total_loss���@

error_R*Fa?

learning_rate_1�<�732�I       6%�	9�E���A�*;


total_loss���@

error_R�*H?

learning_rate_1�<�7�kNI       6%�	 �E���A�*;


total_loss���@

error_R�]N?

learning_rate_1�<�7�2?I       6%�	T4E���A�*;


total_loss�i�@

error_R�B?

learning_rate_1�<�75/��I       6%�	wE���A�*;


total_loss�:�@

error_R�}P?

learning_rate_1�<�7�V�I       6%�	��E���A�*;


total_lossZn�@

error_RM2E?

learning_rate_1�<�7��|I       6%�	�	E���A�*;


total_loss1k�@

error_R�wT?

learning_rate_1�<�7q�Q�I       6%�	�I	E���A�*;


total_loss��@

error_R�b?

learning_rate_1�<�7p5r�I       6%�	6�	E���A�*;


total_loss�C�@

error_Rl�O?

learning_rate_1�<�7G�)I       6%�	��	E���A�*;


total_loss���@

error_R�6K?

learning_rate_1�<�7�%]0I       6%�	�
E���A�*;


total_loss�@

error_R/�S?

learning_rate_1�<�7(�6I       6%�	2`
E���A�*;


total_loss��@

error_R�M?

learning_rate_1�<�7W6dI       6%�	ݫ
E���A�*;


total_loss�Z�@

error_RW-E?

learning_rate_1�<�7�4�I       6%�	�
E���A�*;


total_lossN��@

error_Rq%R?

learning_rate_1�<�73��oI       6%�	l9E���A�*;


total_lossn(|@

error_R�E?

learning_rate_1�<�7�1@I       6%�	�|E���A�*;


total_loss���@

error_R,A?

learning_rate_1�<�7��S>I       6%�	��E���A�*;


total_loss���@

error_R.F?

learning_rate_1�<�7�M��I       6%�	�	E���A�*;


total_loss���@

error_R��V?

learning_rate_1�<�7�GI       6%�	/NE���A�*;


total_lossW��@

error_RwK?

learning_rate_1�<�7E��I       6%�	�E���A�*;


total_loss���@

error_RC*I?

learning_rate_1�<�7��v�I       6%�	��E���A�*;


total_loss���@

error_R�/N?

learning_rate_1�<�7����I       6%�	/E���A�*;


total_loss���@

error_R=K?

learning_rate_1�<�7��!I       6%�	$dE���A�*;


total_loss���@

error_R��L?

learning_rate_1�<�7�T�NI       6%�	ۭE���A�*;


total_loss���@

error_R)�O?

learning_rate_1�<�7���I       6%�	��E���A�*;


total_loss��@

error_R	SQ?

learning_rate_1�<�7��BI       6%�	�7E���A�*;


total_lossq��@

error_R�Q?

learning_rate_1�<�7�&�jI       6%�	�{E���A�*;


total_loss.��@

error_R�"<?

learning_rate_1�<�7&�K�I       6%�	\�E���A�*;


total_loss���@

error_R.�N?

learning_rate_1�<�7� �I       6%�	�E���A�*;


total_loss���@

error_R\D?

learning_rate_1�<�7���I       6%�	�EE���A�*;


total_loss�@

error_R��N?

learning_rate_1�<�7����I       6%�	؉E���A�*;


total_lossd��@

error_Rv�C?

learning_rate_1�<�7��f�I       6%�	 �E���A�*;


total_loss:[c@

error_R[;T?

learning_rate_1�<�7�F��I       6%�	NE���A�*;


total_loss2�@

error_RA�T?

learning_rate_1�<�7�F�{I       6%�	TE���A�*;


total_loss��@

error_R�X?

learning_rate_1�<�7�A|�I       6%�	�E���A�*;


total_loss���@

error_Rf�B?

learning_rate_1�<�73N�I       6%�	��E���A�*;


total_loss�t@

error_R�:?

learning_rate_1�<�7�6U6I       6%�	oE���A�*;


total_loss���@

error_R��E?

learning_rate_1�<�7��PI       6%�	�aE���A�*;


total_loss$X�@

error_R�gm?

learning_rate_1�<�7H���I       6%�	��E���A�*;


total_loss�
G@

error_R��E?

learning_rate_1�<�7�B�}I       6%�	�E���A�*;


total_loss
¬@

error_R:;?

learning_rate_1�<�7��^I       6%�	*E���A�*;


total_loss���@

error_R�M?

learning_rate_1�<�7I���I       6%�	�rE���A�*;


total_loss���@

error_R�s;?

learning_rate_1�<�7���I       6%�	U�E���A�*;


total_loss��@

error_R,�T?

learning_rate_1�<�7A�z�I       6%�	�E���A�*;


total_lossI��@

error_R=�F?

learning_rate_1�<�7�*��I       6%�	�CE���A�*;


total_lossw`�@

error_R
�Y?

learning_rate_1�<�7DT�I       6%�	�E���A�*;


total_loss3�@

error_RD�S?

learning_rate_1�<�7�+O3I       6%�	!�E���A�*;


total_loss1��@

error_R�OM?

learning_rate_1�<�7��]�I       6%�	�E���A�*;


total_lossy�@

error_R:kK?

learning_rate_1�<�7��o|I       6%�	�YE���A�*;


total_lossp�@

error_R߱e?

learning_rate_1�<�7 @I       6%�	a�E���A�*;


total_lossϢA

error_R@ K?

learning_rate_1�<�7)R�9I       6%�	��E���A�*;


total_loss�1�@

error_R�F?

learning_rate_1�<�7Gg�GI       6%�	"E���A�*;


total_loss���@

error_R6�L?

learning_rate_1�<�7��I       6%�	9eE���A�*;


total_lossF A

error_RjCP?

learning_rate_1�<�7���I       6%�	b�E���A�*;


total_loss�M�@

error_R��h?

learning_rate_1�<�7Е�TI       6%�	�E���A�*;


total_loss<>�@

error_R<EB?

learning_rate_1�<�7u�z_I       6%�	�4E���A�*;


total_lossM��@

error_R�O7?

learning_rate_1�<�7�~K\I       6%�	�}E���A�*;


total_loss�o�@

error_R*VQ?

learning_rate_1�<�7@P�|I       6%�	H�E���A�*;


total_loss���@

error_RazL?

learning_rate_1�<�7�^.I       6%�	z E���A�*;


total_loss���@

error_R}�Q?

learning_rate_1�<�7��<I       6%�	dDE���A�*;


total_loss)��@

error_RUL?

learning_rate_1�<�7�bDcI       6%�	��E���A�*;


total_loss�ܴ@

error_R8�P?

learning_rate_1�<�7I��I       6%�	��E���A�*;


total_lossܔ�@

error_R�Y?

learning_rate_1�<�7$�'�I       6%�	�1E���A�*;


total_loss��@

error_R�\?

learning_rate_1�<�7�=�tI       6%�	&wE���A�*;


total_loss�-0A

error_R�OB?

learning_rate_1�<�7ac��I       6%�	[�E���A�*;


total_loss��@

error_R�{U?

learning_rate_1�<�7Q|!I       6%�	 E���A�*;


total_lossW�@

error_R�9A?

learning_rate_1�<�7o�6I       6%�	�DE���A�*;


total_loss���@

error_R�+L?

learning_rate_1�<�7����I       6%�	ɊE���A�*;


total_loss჊@

error_R
�Y?

learning_rate_1�<�7=I       6%�	(�E���A�*;


total_loss��@

error_R/KC?

learning_rate_1�<�7�67jI       6%�	UE���A�*;


total_loss�N�@

error_R��U?

learning_rate_1�<�7~i|I       6%�	�aE���A�*;


total_loss-�@

error_Rc(U?

learning_rate_1�<�7�ǡ�I       6%�	(�E���A�*;


total_loss �A

error_RwN?

learning_rate_1�<�7��+�I       6%�	��E���A�*;


total_loss�@

error_R�KJ?

learning_rate_1�<�7�.�I       6%�	,E���A�*;


total_lossssA

error_R*K?

learning_rate_1�<�7���rI       6%�	��E���A�*;


total_loss��@

error_R7�Z?

learning_rate_1�<�7+9X�I       6%�	R�E���A�*;


total_lossg7A

error_R �J?

learning_rate_1�<�7���I       6%�	�'E���A�*;


total_losstQ�@

error_R�{F?

learning_rate_1�<�7+i�I       6%�	MjE���A�*;


total_lossE�@

error_R� E?

learning_rate_1�<�7�� �I       6%�	��E���A�*;


total_loss�y@

error_R�D?

learning_rate_1�<�7��mI       6%�	�E���A�*;


total_loss6q�@

error_R�i?

learning_rate_1�<�7�)��I       6%�	�TE���A�*;


total_loss��@

error_R�a?

learning_rate_1�<�7ڄJ�I       6%�	��E���A�*;


total_lossh޶@

error_RO�N?

learning_rate_1�<�7\���I       6%�	��E���A�*;


total_loss��A

error_R��g?

learning_rate_1�<�7��ܻI       6%�	%E���A�*;


total_loss\��@

error_R6�@?

learning_rate_1�<�7�˯xI       6%�	gE���A�*;


total_losst�@

error_R�e?

learning_rate_1�<�7�>��I       6%�	$�E���A�*;


total_loss�A�@

error_R�=W?

learning_rate_1�<�7����I       6%�	+�E���A�*;


total_loss)��@

error_RS�W?

learning_rate_1�<�73H��I       6%�	t4E���A�*;


total_loss?~@

error_RmT?

learning_rate_1�<�7�K8SI       6%�	wxE���A�*;


total_loss�+�@

error_R_�[?

learning_rate_1�<�7��jGI       6%�	ۼE���A�*;


total_lossݷ@

error_R�N^?

learning_rate_1�<�7��!�I       6%�	]  E���A�*;


total_loss	�;A

error_Rs�F?

learning_rate_1�<�7{���I       6%�	�D E���A�*;


total_loss A

error_RR?

learning_rate_1�<�7��I       6%�	ԉ E���A�*;


total_lossM�@

error_R�T?

learning_rate_1�<�7`��I       6%�	�� E���A�*;


total_loss
��@

error_R
O?

learning_rate_1�<�7<��I       6%�	�!E���A�*;


total_loss]�t@

error_R�K?

learning_rate_1�<�7�#�I       6%�	�V!E���A�*;


total_loss�-�@

error_R��S?

learning_rate_1�<�7�GvCI       6%�	�!E���A�*;


total_losso�@

error_R@[?

learning_rate_1�<�7�k��I       6%�	��!E���A�*;


total_lossq9�@

error_R�,P?

learning_rate_1�<�7�R��I       6%�	B1"E���A�*;


total_loss>�@

error_R��P?

learning_rate_1�<�7KH*>I       6%�	)z"E���A�*;


total_loss�6�@

error_R��U?

learning_rate_1�<�7zD�:I       6%�		�"E���A�*;


total_losse��@

error_R8 [?

learning_rate_1�<�7��©I       6%�	�#E���A�*;


total_loss��@

error_RJ�E?

learning_rate_1�<�7°�dI       6%�	eT#E���A�*;


total_loss7��@

error_Rc�>?

learning_rate_1�<�7<|zI       6%�	]�#E���A�*;


total_loss��@

error_R�R?

learning_rate_1�<�7����I       6%�	+�#E���A�*;


total_loss�â@

error_R3�>?

learning_rate_1�<�7+JnI       6%�	'-$E���A�*;


total_loss=�@

error_Rs.C?

learning_rate_1�<�7��I       6%�	[u$E���A�*;


total_loss
�@

error_R$�I?

learning_rate_1�<�7�. ZI       6%�	Y�$E���A�*;


total_loss ��@

error_R�_I?

learning_rate_1�<�7��_I       6%�	X�$E���A�*;


total_loss%�@

error_R��S?

learning_rate_1�<�7�?7I       6%�	"@%E���A�*;


total_loss�LA

error_R��N?

learning_rate_1�<�7֕pI       6%�	��%E���A�*;


total_lossҜ�@

error_R �F?

learning_rate_1�<�7���I       6%�	��%E���A�*;


total_loss��@

error_RR�[?

learning_rate_1�<�7 �O�I       6%�	�&E���A�*;


total_loss!9�@

error_R�T?

learning_rate_1�<�7�Q��I       6%�	�[&E���A�*;


total_loss��A

error_RWmW?

learning_rate_1�<�7���I       6%�	��&E���A�*;


total_lossR+�@

error_R={U?

learning_rate_1�<�7 ���I       6%�	��&E���A�*;


total_loss�t�@

error_R=�L?

learning_rate_1�<�7y�˾I       6%�	�0'E���A�*;


total_lossIo�@

error_RMS?

learning_rate_1�<�7�+I       6%�	�s'E���A�*;


total_lossߖ@

error_RlIP?

learning_rate_1�<�7��I       6%�	3�'E���A�*;


total_loss�p�@

error_R#�K?

learning_rate_1�<�7��\I       6%�	(E���A�*;


total_loss�G�@

error_R`�K?

learning_rate_1�<�7P�_lI       6%�	b(E���A�*;


total_loss�O�@

error_R�E5?

learning_rate_1�<�7�A�I       6%�	c�(E���A�*;


total_lossdA

error_RO�E?

learning_rate_1�<�7��bI       6%�	��(E���A�*;


total_loss���@

error_R�I?

learning_rate_1�<�7���I       6%�	�+)E���A�*;


total_loss&a�@

error_R��B?

learning_rate_1�<�7d~�I       6%�	�o)E���A�*;


total_loss,�@

error_RnKV?

learning_rate_1�<�7-��I       6%�	�)E���A�*;


total_losstL�@

error_Rq�>?

learning_rate_1�<�7$�L�I       6%�	��)E���A�*;


total_lossQ�@

error_R��T?

learning_rate_1�<�7�MI       6%�	 @*E���A�*;


total_loss���@

error_R�[?

learning_rate_1�<�7{��I       6%�	4�*E���A�*;


total_loss�*z@

error_R�R?

learning_rate_1�<�7X5QI       6%�	j�*E���A�*;


total_loss��@

error_R�6Z?

learning_rate_1�<�7f�V�I       6%�	�+E���A�*;


total_loss=�@

error_R��E?

learning_rate_1�<�7�,��I       6%�	Rh+E���A�*;


total_loss���@

error_R4�^?

learning_rate_1�<�7"�.VI       6%�	l�+E���A�*;


total_loss�*�@

error_R29_?

learning_rate_1�<�7!�?DI       6%�	��+E���A�*;


total_loss��@

error_R��W?

learning_rate_1�<�7�x�I       6%�	�>,E���A�*;


total_lossv��@

error_R�{V?

learning_rate_1�<�7��/�I       6%�	�,E���A�*;


total_loss�t�@

error_R��]?

learning_rate_1�<�7KR��I       6%�	��,E���A�*;


total_losss��@

error_R��M?

learning_rate_1�<�7;#�?I       6%�	_-E���A�*;


total_lossD��@

error_R�PH?

learning_rate_1�<�7K���I       6%�	�b-E���A�*;


total_loss-�@

error_R@�C?

learning_rate_1�<�7��s�I       6%�	��-E���A�*;


total_loss��@

error_R��[?

learning_rate_1�<�7!�&�I       6%�	��-E���A�*;


total_loss��|@

error_R�&U?

learning_rate_1�<�7�H�UI       6%�	�3.E���A�*;


total_loss�a�@

error_R��7?

learning_rate_1�<�7�ctI       6%�	�{.E���A�*;


total_loss��@

error_R%&N?

learning_rate_1�<�7��n)I       6%�	��.E���A�*;


total_loss��@

error_R��E?

learning_rate_1�<�7 u?�I       6%�	^/E���A�*;


total_loss�6�@

error_R[~\?

learning_rate_1�<�7]ڿlI       6%�	WG/E���A�*;


total_lossK��@

error_R�vT?

learning_rate_1�<�7�wmI       6%�	؇/E���A�*;


total_loss�bA

error_R��V?

learning_rate_1�<�7��փI       6%�	�/E���A�*;


total_loss�{�@

error_R<�H?

learning_rate_1�<�7>�s6I       6%�	0E���A�*;


total_loss���@

error_R�Q?

learning_rate_1�<�7���I       6%�	W0E���A�*;


total_loss�(�@

error_R�T?

learning_rate_1�<�7�<�I       6%�	��0E���A�*;


total_loss�-�@

error_R�A?

learning_rate_1�<�7�:�I       6%�	��0E���A�*;


total_lossNV�@

error_R�b?

learning_rate_1�<�7Q�HgI       6%�	d1E���A�*;


total_loss䕑@

error_R6?

learning_rate_1�<�7��iI       6%�	6_1E���A�*;


total_lossfT�@

error_R��K?

learning_rate_1�<�7����I       6%�	�1E���A�*;


total_loss�ގ@

error_RCvO?

learning_rate_1�<�7e� _I       6%�	��1E���A�*;


total_loss��@

error_R��T?

learning_rate_1�<�7���I       6%�	@,2E���A�*;


total_loss��@

error_R�Q?

learning_rate_1�<�7}���I       6%�	Rp2E���A�*;


total_loss#�~@

error_RCrE?

learning_rate_1�<�7�#�nI       6%�	U�2E���A�*;


total_lossq��@

error_R��T?

learning_rate_1�<�7����I       6%�	��2E���A�*;


total_lossl��@

error_R�_A?

learning_rate_1�<�7
=��I       6%�	53E���A�*;


total_loss��@

error_R�D?

learning_rate_1�<�7��,�I       6%�	\x3E���A�*;


total_lossi� A

error_R��J?

learning_rate_1�<�7x-ހI       6%�	w�3E���A�*;


total_lossf�A

error_R��W?

learning_rate_1�<�7�H�AI       6%�	8�3E���A�*;


total_loss�l�@

error_R�
K?

learning_rate_1�<�7`
�3I       6%�	�F4E���A�*;


total_lossD�@

error_RZ?

learning_rate_1�<�7��I       6%�	p�4E���A�*;


total_loss���@

error_RS�H?

learning_rate_1�<�7H�J/I       6%�	�4E���A�*;


total_loss7�@

error_R�[?

learning_rate_1�<�7d	҄I       6%�	%'5E���A�*;


total_loss3IA

error_RNI?

learning_rate_1�<�7�X��I       6%�	�m5E���A�*;


total_loss@1�@

error_R�0C?

learning_rate_1�<�7�<�I       6%�	:�5E���A�*;


total_loss�ߗ@

error_RRM?

learning_rate_1�<�7�|�I       6%�	!�5E���A�*;


total_lossǄ�@

error_RvCS?

learning_rate_1�<�7TCjI       6%�	�76E���A�*;


total_loss��@

error_RD?

learning_rate_1�<�7�I       6%�	�|6E���A�*;


total_lossu�@

error_R\R?

learning_rate_1�<�7fm7I       6%�	�6E���A�*;


total_loss�U�@

error_R,zJ?

learning_rate_1�<�7	�vI       6%�	:7E���A�*;


total_loss�6�@

error_R�M?

learning_rate_1�<�7�[i�I       6%�	�F7E���A�*;


total_loss��@

error_R��V?

learning_rate_1�<�7��>I       6%�	��7E���A�*;


total_lossR��@

error_RE�n?

learning_rate_1�<�7��RI       6%�	��7E���A�*;


total_loss�ڲ@

error_R3sQ?

learning_rate_1�<�7��9GI       6%�	�68E���A�*;


total_loss��@

error_R�hX?

learning_rate_1�<�7c�G_I       6%�	\z8E���A�*;


total_loss�a�@

error_R�W?

learning_rate_1�<�7C�:JI       6%�	�8E���A�*;


total_lossOD�@

error_R��:?

learning_rate_1�<�7��.I       6%�	,9E���A�*;


total_loss`�@

error_RT�M?

learning_rate_1�<�7F��I       6%�	�M9E���A�*;


total_loss��@

error_R�F?

learning_rate_1�<�7�gZ�I       6%�	��9E���A�*;


total_loss7��@

error_R��e?

learning_rate_1�<�7�COI       6%�	��9E���A�*;


total_loss4�e@

error_RJ�K?

learning_rate_1�<�7���FI       6%�	H:E���A�*;


total_loss��A

error_R_�T?

learning_rate_1�<�7�XeI       6%�	_h:E���A�*;


total_loss�o�@

error_R�L?

learning_rate_1�<�7vˉiI       6%�	l�:E���A�*;


total_loss� �@

error_R�M?

learning_rate_1�<�7m��wI       6%�	�:E���A�*;


total_lossZ��@

error_Rq�X?

learning_rate_1�<�7�%�I       6%�	D;E���A�*;


total_loss��@

error_Rf�F?

learning_rate_1�<�7*��I       6%�	*�;E���A�*;


total_loss��@

error_R��Z?

learning_rate_1�<�7w�,I       6%�	�;E���A�*;


total_loss2Ƣ@

error_R��Q?

learning_rate_1�<�7z�ݣI       6%�	�3<E���A�*;


total_loss�H�@

error_R�QJ?

learning_rate_1�<�7\ۃ�I       6%�	6}<E���A�*;


total_loss�:�@

error_Rn>X?

learning_rate_1�<�7Z�;�I       6%�	�<E���A�*;


total_lossΣ�@

error_RS3F?

learning_rate_1�<�7�L��I       6%�	�=E���A�*;


total_loss_h�@

error_R�TY?

learning_rate_1�<�7�Sc�I       6%�	�W=E���A�*;


total_loss>�@

error_RZX?

learning_rate_1�<�7��AI       6%�	��=E���A�*;


total_loss��@

error_R��[?

learning_rate_1�<�7jKM�I       6%�	��=E���A�*;


total_loss���@

error_R�F[?

learning_rate_1�<�7z2��I       6%�	�1>E���A�*;


total_loss���@

error_Rf?B?

learning_rate_1�<�7^�@xI       6%�	*u>E���A�*;


total_loss
x�@

error_R��K?

learning_rate_1�<�7	�?�I       6%�	�>E���A�*;


total_loss��@

error_RC�T?

learning_rate_1�<�7��`I       6%�	?E���A�*;


total_lossc��@

error_Ra&8?

learning_rate_1�<�7�	��I       6%�	%S?E���A�*;


total_loss&�
A

error_R8�M?

learning_rate_1�<�7���qI       6%�	��?E���A�*;


total_lossk�@

error_R�]?

learning_rate_1�<�7-dPII       6%�	2�?E���A�*;


total_loss��@

error_RD�R?

learning_rate_1�<�7O9�I       6%�	�@E���A�*;


total_loss��@

error_Rl][?

learning_rate_1�<�7Af�I       6%�	_@E���A�*;


total_loss�Ԥ@

error_R��;?

learning_rate_1�<�7Y6��I       6%�	&�@E���A�*;


total_losshA

error_R�LE?

learning_rate_1�<�7��@PI       6%�	��@E���A�*;


total_lossb�@

error_R�]R?

learning_rate_1�<�7NB�>I       6%�	�,AE���A�*;


total_losse�@

error_RɘG?

learning_rate_1�<�7�G�tI       6%�	_uAE���A�*;


total_loss�\�@

error_R�TM?

learning_rate_1�<�7��I       6%�	�AE���A�*;


total_loss�$�@

error_R��>?

learning_rate_1�<�7{� I       6%�	��AE���A�*;


total_loss�l�@

error_R=�K?

learning_rate_1�<�7��VI       6%�	�BBE���A�*;


total_loss?��@

error_R�@^?

learning_rate_1�<�7��XnI       6%�	]�BE���A�*;


total_loss,'�@

error_R�YJ?

learning_rate_1�<�71��I       6%�	��BE���A�*;


total_lossۼ�@

error_R��H?

learning_rate_1�<�7˨��I       6%�	�CE���A�*;


total_loss|7�@

error_R
�c?

learning_rate_1�<�7��e|I       6%�	JNCE���A�*;


total_loss{��@

error_R�F?

learning_rate_1�<�7�d�I       6%�	�CE���A�*;


total_loss��@

error_R�uH?

learning_rate_1�<�7���I       6%�	��CE���A�*;


total_loss9m�@

error_R�G?

learning_rate_1�<�7q��I       6%�	�DE���A�*;


total_lossR��@

error_R�I??

learning_rate_1�<�7��H�I       6%�	CXDE���A�*;


total_loss%�@

error_R��e?

learning_rate_1�<�74lv~I       6%�	k�DE���A�*;


total_loss���@

error_R�C?

learning_rate_1�<�7���MI       6%�	��DE���A�*;


total_loss�B�@

error_RqR?

learning_rate_1�<�7�l��I       6%�	{-EE���A�*;


total_loss\�q@

error_Rh#K?

learning_rate_1�<�7����I       6%�	vEE���A�*;


total_loss�3�@

error_R�P?

learning_rate_1�<�7�˚RI       6%�	4�EE���A�*;


total_loss���@

error_RVA?

learning_rate_1�<�7�E��I       6%�	q FE���A�*;


total_loss;�@

error_R�8\?

learning_rate_1�<�7 ׂ�I       6%�	KHFE���A�*;


total_loss2��@

error_R�'B?

learning_rate_1�<�7�,	I       6%�	N�FE���A�*;


total_lossf�@

error_Rr�S?

learning_rate_1�<�7�*.�I       6%�	\�FE���A�*;


total_loss�'�@

error_Rs�7?

learning_rate_1�<�7�>�I       6%�	JGE���A�*;


total_loss�@

error_R�DK?

learning_rate_1�<�7�:�I       6%�	aGE���A�*;


total_loss�xA

error_R�OX?

learning_rate_1�<�7s@]I       6%�	��GE���A�*;


total_loss��s@

error_R�,P?

learning_rate_1�<�7W���I       6%�	�HE���A�*;


total_losso�@

error_RT�J?

learning_rate_1�<�7ߛ�I       6%�	|VHE���A�*;


total_loss3�@

error_Ro�W?

learning_rate_1�<�7F�ݘI       6%�	p�HE���A�*;


total_loss�w�@

error_R�xX?

learning_rate_1�<�7x�ETI       6%�	?�HE���A�*;


total_loss�y�@

error_R��A?

learning_rate_1�<�7!�NI       6%�	&IE���A�*;


total_lossR��@

error_R�C?

learning_rate_1�<�7��%I       6%�	�kIE���A�*;


total_lossT�J@

error_R��@?

learning_rate_1�<�7�b�I       6%�	��IE���A�*;


total_losstD�@

error_R6�M?

learning_rate_1�<�7�M�I       6%�	n�IE���A�*;


total_lossM�o@

error_R��e?

learning_rate_1�<�7�4�cI       6%�	(8JE���A�*;


total_lossa��@

error_R�iO?

learning_rate_1�<�7�I       6%�	zJE���A�*;


total_lossc	�@

error_R}BO?

learning_rate_1�<�7t#8XI       6%�	��JE���A�*;


total_loss��@

error_R�+^?

learning_rate_1�<�7#`��I       6%�	JKE���A�*;


total_loss!��@

error_Rx}U?

learning_rate_1�<�7<v3I       6%�	�DKE���A�*;


total_loss���@

error_R��Z?

learning_rate_1�<�77$7�I       6%�	��KE���A�*;


total_loss���@

error_R6�S?

learning_rate_1�<�71���I       6%�	��KE���A�*;


total_lossq"�@

error_RZ�]?

learning_rate_1�<�7l�DI       6%�	�LE���A�*;


total_loss���@

error_R��G?

learning_rate_1�<�7ś�I       6%�	�ULE���A�*;


total_loss��@

error_R2�??

learning_rate_1�<�7O�'`I       6%�	�LE���A�*;


total_loss&M�@

error_R��i?

learning_rate_1�<�7�Ч�I       6%�	��LE���A�*;


total_loss���@

error_R��C?

learning_rate_1�<�7��T�I       6%�	�,ME���A�*;


total_lossp��@

error_R�uL?

learning_rate_1�<�72L��I       6%�	�vME���A�*;


total_loss#P�@

error_R��Y?

learning_rate_1�<�7QY��I       6%�	_�ME���A�*;


total_lossҚ�@

error_R�9a?

learning_rate_1�<�7�c�.I       6%�	�NE���A�*;


total_lossP�A

error_Rl�I?

learning_rate_1�<�7�C#I       6%�	KNE���A�*;


total_lossJ��@

error_R_�T?

learning_rate_1�<�7C�ڌI       6%�	��NE���A�*;


total_lossc.�@

error_R�P?

learning_rate_1�<�7�+� I       6%�	�NE���A�*;


total_loss��@

error_R�i?

learning_rate_1�<�7�� �I       6%�	N'OE���A�*;


total_loss �@

error_R\�K?

learning_rate_1�<�7��܍I       6%�	�OE���A�*;


total_loss�r�@

error_Rd�T?

learning_rate_1�<�71/I       6%�	��OE���A�*;


total_loss�s�@

error_R�]\?

learning_rate_1�<�7���>I       6%�	nPE���A�*;


total_loss�I�@

error_R��0?

learning_rate_1�<�7ܗ�dI       6%�	�dPE���A�*;


total_loss�e�@

error_R;zq?

learning_rate_1�<�7K��LI       6%�	��PE���A�*;


total_loss;@�@

error_R�Z?

learning_rate_1�<�7Z���I       6%�	�PE���A�*;


total_loss���@

error_RfG?

learning_rate_1�<�7���I       6%�	1QE���A�*;


total_loss`Z�@

error_R��R?

learning_rate_1�<�7Ww
�I       6%�	�yQE���A�*;


total_loss#��@

error_R��T?

learning_rate_1�<�7����I       6%�	��QE���A�*;


total_loss�~�@

error_R@�U?

learning_rate_1�<�7��
�I       6%�	kRE���A�*;


total_loss���@

error_R4�W?

learning_rate_1�<�7_��I       6%�	JRE���A�*;


total_losso\�@

error_RN�S?

learning_rate_1�<�7�^�I       6%�	��RE���A�*;


total_lossII�@

error_R�[?

learning_rate_1�<�7���I       6%�	��RE���A�*;


total_loss��@

error_R�]M?

learning_rate_1�<�7�I       6%�	�SE���A�*;


total_loss�Y�@

error_R��O?

learning_rate_1�<�7x��OI       6%�	ffSE���A�*;


total_loss{�@

error_R�{Z?

learning_rate_1�<�7�E�I       6%�	�SE���A�*;


total_loss8��@

error_RE�U?

learning_rate_1�<�7c׷YI       6%�	��SE���A�*;


total_losse{�@

error_R�2G?

learning_rate_1�<�7��qI       6%�	*:TE���A�*;


total_lossB�@

error_R�K?

learning_rate_1�<�7CE5I       6%�	?}TE���A�*;


total_losse��@

error_RcBK?

learning_rate_1�<�7�@��I       6%�	A�TE���A�*;


total_loss���@

error_RK?

learning_rate_1�<�7���3I       6%�	UE���A�*;


total_loss\I�@

error_R��W?

learning_rate_1�<�7��) I       6%�	>HUE���A�*;


total_lossRd�@

error_R�[?

learning_rate_1�<�7��v�I       6%�	��UE���A�*;


total_loss���@

error_RRT?

learning_rate_1�<�7�b8I       6%�	e�UE���A�*;


total_loss�?�@

error_RW�T?

learning_rate_1�<�7�B�I       6%�	�<VE���A�*;


total_loss�z�@

error_R�V/?

learning_rate_1�<�7��I       6%�	 �VE���A�*;


total_loss{c�@

error_R��E?

learning_rate_1�<�7(��2I       6%�	�VE���A�*;


total_lossza�@

error_R�rF?

learning_rate_1�<�7sbNI       6%�	*WE���A�*;


total_lossEћ@

error_R?D?

learning_rate_1�<�7o�`�I       6%�	�UWE���A�*;


total_loss��@

error_R�|E?

learning_rate_1�<�7�&I       6%�	�WE���A�*;


total_loss-�@

error_R6�J?

learning_rate_1�<�7k�]I       6%�	E+XE���A�*;


total_loss��@

error_R`�R?

learning_rate_1�<�7R�}I       6%�	�lXE���A�*;


total_loss��@

error_R��V?

learning_rate_1�<�70'�I       6%�	�XE���A�*;


total_loss枰@

error_R��E?

learning_rate_1�<�7��=I       6%�	��XE���A�*;


total_loss�G�@

error_R=�P?

learning_rate_1�<�71 ^I       6%�	�BYE���A�*;


total_loss#k�@

error_Rd"O?

learning_rate_1�<�7d6��I       6%�	r�YE���A�*;


total_loss���@

error_R�iJ?

learning_rate_1�<�7���I       6%�	��YE���A�*;


total_lossҪ@

error_R@RQ?

learning_rate_1�<�7A'G�I       6%�	BZE���A�*;


total_loss�:A

error_R�Zl?

learning_rate_1�<�7(n��I       6%�	@RZE���A�*;


total_lossγ@

error_R1�R?

learning_rate_1�<�7��@aI       6%�	�ZE���A�*;


total_loss�h�@

error_R;�L?

learning_rate_1�<�7A��3I       6%�	�ZE���A�*;


total_losso�@

error_RM�j?

learning_rate_1�<�7'i��I       6%�	�[E���A�*;


total_loss��@

error_R��D?

learning_rate_1�<�7So��I       6%�	�a[E���A�*;


total_lossLX�@

error_R��N?

learning_rate_1�<�7d�LPI       6%�	��[E���A�*;


total_loss$Y�@

error_R}zH?

learning_rate_1�<�7��̏I       6%�	��[E���A�*;


total_loss�mG@

error_R��Q?

learning_rate_1�<�7��I       6%�	�,\E���A�*;


total_loss��A

error_R�e>?

learning_rate_1�<�7,U��I       6%�	�p\E���A�*;


total_lossl�@

error_R}hL?

learning_rate_1�<�7ְCI       6%�	Y�\E���A�*;


total_loss��@

error_R��W?

learning_rate_1�<�7 (gI       6%�	~�\E���A�*;


total_lossnn�@

error_R[�V?

learning_rate_1�<�7\�
GI       6%�	�@]E���A�*;


total_loss���@

error_R��O?

learning_rate_1�<�7E��I       6%�	��]E���A�*;


total_loss/l�@

error_RHW??

learning_rate_1�<�7�zшI       6%�	�]E���A�*;


total_loss�S�@

error_R<yU?

learning_rate_1�<�7�=(�I       6%�	G!^E���A�*;


total_loss�'�@

error_RM�a?

learning_rate_1�<�7ŲI       6%�	hj^E���A�*;


total_lossΒ�@

error_R�J?

learning_rate_1�<�7�xT�I       6%�	\�^E���A�*;


total_loss�0�@

error_R�`N?

learning_rate_1�<�7�L#I       6%�	�_E���A�*;


total_loss�^�@

error_R�Q?

learning_rate_1�<�7A1:5I       6%�	yR_E���A�*;


total_loss�0�@

error_R�ZQ?

learning_rate_1�<�7���I       6%�	%�_E���A�*;


total_loss���@

error_R��\?

learning_rate_1�<�7#�aQI       6%�	1�_E���A�*;


total_loss��@

error_R{�Z?

learning_rate_1�<�7�,L�I       6%�	�/`E���A�*;


total_lossi�@

error_R�7?

learning_rate_1�<�7D��II       6%�	u`E���A�*;


total_loss=�@

error_R�P?

learning_rate_1�<�7����I       6%�	��`E���A�*;


total_loss	��@

error_RܼD?

learning_rate_1�<�7(�]�I       6%�	3�`E���A�*;


total_loss�|�@

error_R�TM?

learning_rate_1�<�7! ��I       6%�	:DaE���A�*;


total_losshA

error_R&�_?

learning_rate_1�<�7 /|�I       6%�	��aE���A�*;


total_losso��@

error_Rj�N?

learning_rate_1�<�7VR;aI       6%�	�aE���A�*;


total_loss,��@

error_R� O?

learning_rate_1�<�7.��kI       6%�	�bE���A�*;


total_loss
_�@

error_RL�H?

learning_rate_1�<�7�tI       6%�	'WbE���A�*;


total_lossA��@

error_R-[?

learning_rate_1�<�7�X;�I       6%�	��bE���A�*;


total_lossl"�@

error_R�.R?

learning_rate_1�<�7HC�I       6%�	<�bE���A�*;


total_lossϳ�@

error_R�G?

learning_rate_1�<�7��I       6%�	"cE���A�*;


total_loss}ɓ@

error_RÂN?

learning_rate_1�<�7I5I       6%�	�ncE���A�*;


total_loss��@

error_R�Ze?

learning_rate_1�<�7]!R�I       6%�	#�cE���A�*;


total_lossr�@

error_R�e<?

learning_rate_1�<�7Y`�I       6%�	�cE���A�*;


total_loss�}�@

error_R�P?

learning_rate_1�<�7�M��I       6%�	4;dE���A�*;


total_loss���@

error_R�KU?

learning_rate_1�<�7�1RxI       6%�	�dE���A�*;


total_loss�B�@

error_RdvZ?

learning_rate_1�<�7�ѪI       6%�	�dE���A�*;


total_lossx~�@

error_R�RP?

learning_rate_1�<�7q*��I       6%�	�	eE���A�*;


total_loss���@

error_R��A?

learning_rate_1�<�7�DBI       6%�	JeE���A�*;


total_loss�F�@

error_R��D?

learning_rate_1�<�7��r:I       6%�	�eE���A�*;


total_loss�a@

error_R��M?

learning_rate_1�<�7s���I       6%�	�eE���A�*;


total_loss1t�@

error_RxAX?

learning_rate_1�<�7OKj�I       6%�	_fE���A�*;


total_lossoO�@

error_R�UX?

learning_rate_1�<�7&߆hI       6%�	�XfE���A�*;


total_loss� �@

error_R�&K?

learning_rate_1�<�7v()I       6%�	��fE���A�*;


total_loss֨�@

error_R�U?

learning_rate_1�<�7����I       6%�	�fE���A�*;


total_loss7�X@

error_RCB?

learning_rate_1�<�7lS�I       6%�	�"gE���A�*;


total_loss�@�@

error_R��W?

learning_rate_1�<�7����I       6%�	ydgE���A�*;


total_loss��!A

error_R��F?

learning_rate_1�<�7.!�I       6%�	��gE���A�*;


total_lossȁ�@

error_R��H?

learning_rate_1�<�7M�P�I       6%�	<
hE���A�*;


total_loss%n@

error_Rn�g?

learning_rate_1�<�7��bSI       6%�	@NhE���A�*;


total_loss^�@

error_R��F?

learning_rate_1�<�7����I       6%�	��hE���A�*;


total_loss��@

error_R�N?

learning_rate_1�<�7[=�\I       6%�	w�hE���A�*;


total_loss@t�@

error_R�4]?

learning_rate_1�<�7��KhI       6%�		iE���A�*;


total_lossT��@

error_Rr�P?

learning_rate_1�<�78�5�I       6%�	/aiE���A�*;


total_lossx��@

error_Rf1I?

learning_rate_1�<�7��{�I       6%�	#�iE���A�*;


total_loss��@

error_R3�H?

learning_rate_1�<�7zn�I       6%�	��iE���A�*;


total_loss{.�@

error_R@�S?

learning_rate_1�<�7�T��I       6%�	�.jE���A�*;


total_loss��@

error_R��N?

learning_rate_1�<�7��&I       6%�	wrjE���A�*;


total_lossɵ@

error_R,J?

learning_rate_1�<�7����I       6%�	ƶjE���A�*;


total_loss���@

error_Rn�W?

learning_rate_1�<�7-�'I       6%�	��jE���A�*;


total_loss��@

error_R37X?

learning_rate_1�<�7���4I       6%�	�;kE���A�*;


total_loss���@

error_R)D?

learning_rate_1�<�7b��I       6%�	`�kE���A�*;


total_loss$��@

error_R�,R?

learning_rate_1�<�7T���I       6%�	o�kE���A�*;


total_loss�8�@

error_R[�M?

learning_rate_1�<�7�NI       6%�	/lE���A�*;


total_loss�H�@

error_Rőj?

learning_rate_1�<�7&B�2I       6%�	�JlE���A�*;


total_lossJ��@

error_R-�a?

learning_rate_1�<�7�V�I       6%�	��lE���A�*;


total_loss�\�@

error_RX'S?

learning_rate_1�<�7��r�I       6%�	��lE���A�*;


total_lossS��@

error_R\[?

learning_rate_1�<�7o�a�I       6%�	mE���A�*;


total_loss���@

error_R�O?

learning_rate_1�<�7#!r[I       6%�	SXmE���A�*;


total_lossd�@

error_R{bM?

learning_rate_1�<�7E��I       6%�	l�mE���A�*;


total_loss[T�@

error_R&�J?

learning_rate_1�<�7-�I       6%�	��mE���A�*;


total_lossq(�@

error_RqsH?

learning_rate_1�<�7��.�I       6%�	�nE���A�*;


total_loss��@

error_RTG?

learning_rate_1�<�7��0�I       6%�	�dnE���A�*;


total_lossjٻ@

error_R�?_?

learning_rate_1�<�7'���I       6%�	ɬnE���A�*;


total_loss��@

error_RxQ?

learning_rate_1�<�7K_G�I       6%�	k�nE���A�*;


total_loss���@

error_R�UU?

learning_rate_1�<�7�ڕ�I       6%�	j6oE���A�*;


total_loss�
�@

error_R�J?

learning_rate_1�<�7.��;I       6%�	
voE���A�*;


total_loss9�@

error_R��Y?

learning_rate_1�<�7B���I       6%�	��oE���A�*;


total_loss��@

error_R�@?

learning_rate_1�<�7´KI       6%�	��oE���A�*;


total_lossV7�@

error_RƆ5?

learning_rate_1�<�7���gI       6%�	�8pE���A�*;


total_lossA��@

error_R*�X?

learning_rate_1�<�7cn�I       6%�	�xpE���A�*;


total_loss���@

error_R2R?

learning_rate_1�<�7�[�I       6%�	a�pE���A�*;


total_lossej�@

error_RnI?

learning_rate_1�<�7-RZ�I       6%�	;�pE���A�*;


total_loss��@

error_R8RC?

learning_rate_1�<�7�?Q�I       6%�	=qE���A�*;


total_loss���@

error_R��L?

learning_rate_1�<�7i�;yI       6%�	K}qE���A�*;


total_loss�@

error_R!�I?

learning_rate_1�<�7+��I       6%�	��qE���A�*;


total_lossh�@

error_R��[?

learning_rate_1�<�7��I       6%�	{rE���A�*;


total_loss�
A

error_R�[F?

learning_rate_1�<�7�U�I       6%�	JrE���A�*;


total_loss���@

error_R��B?

learning_rate_1�<�7�ǪkI       6%�	h�rE���A�*;


total_loss��@

error_R�]B?

learning_rate_1�<�7�PXI       6%�	��rE���A�*;


total_loss��@

error_R�{@?

learning_rate_1�<�7���I       6%�	usE���A�*;


total_loss^â@

error_R��J?

learning_rate_1�<�7����I       6%�	�bsE���A�*;


total_loss@w�@

error_R)W?

learning_rate_1�<�7�X�I       6%�	=�sE���A�*;


total_loss��@

error_R�@8?

learning_rate_1�<�7e�ɢI       6%�	��sE���A�*;


total_loss�E�@

error_R��]?

learning_rate_1�<�74��I       6%�	U+tE���A�*;


total_loss�@

error_R�*_?

learning_rate_1�<�7�юtI       6%�	�itE���A�*;


total_loss���@

error_R�"E?

learning_rate_1�<�7���I       6%�	U�tE���A�*;


total_loss��@

error_RM?

learning_rate_1�<�7 �&I       6%�	��tE���A�*;


total_loss��E@

error_R.vH?

learning_rate_1�<�7��/�I       6%�	�GuE���A�*;


total_lossN��@

error_R�O?

learning_rate_1�<�7��<vI       6%�	�uE���A�*;


total_lossv��@

error_R�CU?

learning_rate_1�<�7�αI       6%�	��uE���A�*;


total_losscޥ@

error_R�"G?

learning_rate_1�<�7��fNI       6%�	vE���A�*;


total_loss�`�@

error_R�1M?

learning_rate_1�<�7�W��I       6%�	YvE���A�*;


total_loss��@

error_R8@G?

learning_rate_1�<�7JcF�I       6%�	؛vE���A�*;


total_loss�i�@

error_REjY?

learning_rate_1�<�7�`YI       6%�	��vE���A�*;


total_loss���@

error_R��]?

learning_rate_1�<�7��{I       6%�	�!wE���A�*;


total_loss���@

error_R(�I?

learning_rate_1�<�7�)��I       6%�	ewE���A�*;


total_loss���@

error_R�nH?

learning_rate_1�<�7uѺPI       6%�	C�wE���A�*;


total_loss��@

error_R1]?

learning_rate_1�<�7v��I       6%�	�xE���A�*;


total_loss���@

error_RaJ?

learning_rate_1�<�7�9I       6%�	`xE���A�*;


total_loss�|�@

error_R�!O?

learning_rate_1�<�7��	�I       6%�	b�xE���A�*;


total_lossl"�@

error_Rd�L?

learning_rate_1�<�7���I       6%�	��xE���A�*;


total_lossꔛ@

error_R FU?

learning_rate_1�<�7�&I       6%�	V0yE���A�*;


total_loss�h�@

error_R� L?

learning_rate_1�<�7�B�I       6%�	syE���A�*;


total_loss��@

error_R�;?

learning_rate_1�<�7 E}AI       6%�	��yE���A�*;


total_lossv�@

error_RÅX?

learning_rate_1�<�7�y �I       6%�	��yE���A�*;


total_loss��@

error_R�P?

learning_rate_1�<�7��I       6%�	�;zE���A�*;


total_loss���@

error_R�H?

learning_rate_1�<�76{;�I       6%�	5}zE���A�*;


total_loss��@

error_R��J?

learning_rate_1�<�7���I       6%�	��zE���A�*;


total_lossJ�@

error_R�`Z?

learning_rate_1�<�7;��I       6%�	�{E���A�*;


total_loss���@

error_R߈a?

learning_rate_1�<�7���I       6%�	�G{E���A�*;


total_loss��@

error_R��L?

learning_rate_1�<�7�.�	I       6%�	~�{E���A�*;


total_loss���@

error_Ra�I?

learning_rate_1�<�7�{>I       6%�	��{E���A�*;


total_loss
z�@

error_Ra�L?

learning_rate_1�<�7x��dI       6%�	e|E���A�*;


total_loss=a�@

error_Rv�X?

learning_rate_1�<�7Ǥ�'I       6%�	hP|E���A�*;


total_lossf��@

error_RGW?

learning_rate_1�<�7�-�I       6%�	�|E���A�*;


total_lossʗ�@

error_R8�6?

learning_rate_1�<�7�@B]I       6%�	�|E���A�*;


total_loss6Ï@

error_R��g?

learning_rate_1�<�7��#uI       6%�	.}E���A�*;


total_loss��@

error_R�~[?

learning_rate_1�<�7�pI       6%�	O`}E���A�*;


total_loss���@

error_R�R?

learning_rate_1�<�7ȋ�I       6%�	T�}E���A�*;


total_loss ��@

error_Rl0U?

learning_rate_1�<�7�`E�I       6%�	��}E���A�*;


total_lossញ@

error_RE�M?

learning_rate_1�<�7ϴ�I       6%�	+~E���A�*;


total_loss`
�@

error_R�C?

learning_rate_1�<�7�t1I       6%�	�n~E���A�*;


total_loss��@

error_RԔf?

learning_rate_1�<�7J�V�I       6%�	6�~E���A�*;


total_loss��@

error_R��L?

learning_rate_1�<�7��K�I       6%�	h�~E���A�*;


total_loss��A

error_R}�R?

learning_rate_1�<�7jmoI       6%�	z9E���A�*;


total_loss�|@

error_RR?

learning_rate_1�<�752AeI       6%�		{E���A�*;


total_loss�@

error_RڬG?

learning_rate_1�<�7���CI       6%�	/�E���A�*;


total_loss]�@

error_R�N?

learning_rate_1�<�7�A1I       6%�	��E���A�*;


total_lossvK�@

error_Rx_Q?

learning_rate_1�<�71}��I       6%�	�I�E���A�*;


total_loss�M�@

error_R��U?

learning_rate_1�<�7E��'I       6%�	Ћ�E���A�*;


total_loss��@

error_R�A?

learning_rate_1�<�7���I       6%�	tˀE���A�*;


total_loss�ܙ@

error_RR�U?

learning_rate_1�<�7�m_I       6%�	�
�E���A�*;


total_loss���@

error_Rz;W?

learning_rate_1�<�7�B��I       6%�	�J�E���A�*;


total_loss�Z�@

error_R��Q?

learning_rate_1�<�7O�'{I       6%�	̊�E���A�*;


total_loss���@

error_R
h;?

learning_rate_1�<�7&S�I       6%�	ʁE���A�*;


total_loss��@

error_R�Vl?

learning_rate_1�<�7i��I       6%�	�	�E���A�*;


total_lossml�@

error_R�$H?

learning_rate_1�<�7K?I       6%�	�I�E���A�*;


total_loss�IA

error_R7�K?

learning_rate_1�<�7��{I       6%�	4��E���A�*;


total_lossѥ@

error_R�mP?

learning_rate_1�<�7��PI       6%�	̂E���A�*;


total_loss���@

error_R��Y?

learning_rate_1�<�7�j�I       6%�	��E���A�*;


total_loss ��@

error_R�OX?

learning_rate_1�<�7(�)I       6%�	vd�E���A�*;


total_lossa��@

error_RoQ?

learning_rate_1�<�7��n�I       6%�	��E���A�*;


total_loss���@

error_R8sN?

learning_rate_1�<�7&$��I       6%�	�E���A�*;


total_lossWw�@

error_R!�R?

learning_rate_1�<�7�gNI       6%�	5,�E���A�*;


total_loss/ݼ@

error_R��n?

learning_rate_1�<�7���I       6%�	p�E���A�*;


total_lossTa�@

error_R�\V?

learning_rate_1�<�7��]HI       6%�	6��E���A�*;


total_loss�GA

error_R�']?

learning_rate_1�<�7�I       6%�	+��E���A�*;


total_loss���@

error_Rn�S?

learning_rate_1�<�7'���I       6%�	�4�E���A�*;


total_losso܊@

error_R��M?

learning_rate_1�<�7�|_�I       6%�	Uv�E���A�*;


total_loss�L�@

error_R_�U?

learning_rate_1�<�7��QI       6%�	�E���A�*;


total_lossA.�@

error_RJ`G?

learning_rate_1�<�7�u�I       6%�	� �E���A�*;


total_lossTƜ@

error_R��@?

learning_rate_1�<�7]��gI       6%�	�C�E���A�*;


total_loss���@

error_R �D?

learning_rate_1�<�7p?�mI       6%�	���E���A�*;


total_loss�@

error_Ra�@?

learning_rate_1�<�7���I       6%�	>ΆE���A�*;


total_loss��@

error_R��X?

learning_rate_1�<�7���I       6%�	��E���A�*;


total_lossci�@

error_R��H?

learning_rate_1�<�7��f�I       6%�	(Q�E���A�*;


total_lossr@�@

error_R�A?

learning_rate_1�<�7���I       6%�	E��E���A�*;


total_loss���@

error_R��J?

learning_rate_1�<�7��I       6%�	��E���A�*;


total_loss*"�@

error_R�0V?

learning_rate_1�<�7�[��I       6%�	t8�E���A�*;


total_loss_�@

error_RdDB?

learning_rate_1�<�7_#dbI       6%�	�y�E���A�*;


total_loss�@

error_Re�F?

learning_rate_1�<�7�_&�I       6%�	���E���A�*;


total_lossmb�@

error_Rd<?

learning_rate_1�<�7r<Q�I       6%�	3��E���A�*;


total_loss�x@

error_R��P?

learning_rate_1�<�7㾹�I       6%�	_8�E���A�*;


total_loss��@

error_R�M[?

learning_rate_1�<�74/g�I       6%�	gz�E���A�*;


total_lossW��@

error_R��P?

learning_rate_1�<�7`�,I       6%�	8��E���A�*;


total_lossJ@�@

error_R��g?

learning_rate_1�<�7�
�I       6%�	��E���A�*;


total_loss߲�@

error_R C?

learning_rate_1�<�7���I       6%�	&C�E���A�*;


total_loss�r�@

error_R��L?

learning_rate_1�<�7d��I       6%�	���E���A�*;


total_loss*�@

error_R�IX?

learning_rate_1�<�7�E͘I       6%�	�ȊE���A�*;


total_loss7V�@

error_RiBc?

learning_rate_1�<�7��*AI       6%�		�E���A�*;


total_lossU�@

error_R�P?

learning_rate_1�<�7q8�I       6%�	�I�E���A�*;


total_loss��@

error_RRI?

learning_rate_1�<�7���I       6%�	B��E���A�*;


total_loss�l�@

error_R��a?

learning_rate_1�<�7�g-GI       6%�	~ʋE���A�*;


total_loss��+A

error_R-"A?

learning_rate_1�<�7��{�I       6%�	h
�E���A�*;


total_lossd,�@

error_R?_X?

learning_rate_1�<�7�L�I       6%�	wJ�E���A�*;


total_loss��A

error_R$iU?

learning_rate_1�<�7+�"�I       6%�	ǉ�E���A�*;


total_lossʚ�@

error_R�SX?

learning_rate_1�<�7P�ڈI       6%�	<ɌE���A�*;


total_loss(��@

error_R.�V?

learning_rate_1�<�7D���I       6%�	�	�E���A�*;


total_loss��@

error_R8�I?

learning_rate_1�<�7��I       6%�	�J�E���A�*;


total_lossN��@

error_R�F?

learning_rate_1�<�7�wV�I       6%�	苍E���A�*;


total_loss1Nd@

error_R��I?

learning_rate_1�<�7���7I       6%�	lˍE���A�*;


total_loss���@

error_RFT?

learning_rate_1�<�7(�Z�I       6%�	c�E���A�*;


total_lossL�@

error_RT#Z?

learning_rate_1�<�7㪛�I       6%�	�N�E���A�*;


total_loss<�;@

error_R$�Q?

learning_rate_1�<�7z*�jI       6%�	$��E���A�*;


total_lossrΥ@

error_R:qT?

learning_rate_1�<�7ؔ�uI       6%�	5̎E���A�*;


total_loss�5�@

error_R`�C?

learning_rate_1�<�7�(dI       6%�	l	�E���A�*;


total_loss ��@

error_R��Q?

learning_rate_1�<�7eI�I       6%�	I�E���A�*;


total_loss�{�@

error_R�Xa?

learning_rate_1�<�7����I       6%�	��E���A�*;


total_loss&�@

error_R�3D?

learning_rate_1�<�7İ.�I       6%�	�ʏE���A�*;


total_loss*��@

error_R�|>?

learning_rate_1�<�7���I       6%�	J�E���A�*;


total_loss1��@

error_R4.X?

learning_rate_1�<�7�^��I       6%�	N�E���A�*;


total_loss�O�@

error_R��\?

learning_rate_1�<�7�aMI       6%�	E���A�*;


total_losst�@

error_R=�N?

learning_rate_1�<�7��tI       6%�	R͐E���A�*;


total_loss�@

error_R��E?

learning_rate_1�<�7�^��I       6%�	'�E���A�*;


total_loss�5�@

error_R�\>?

learning_rate_1�<�7�qA�I       6%�	M�E���A�*;


total_loss�
�@

error_R|g:?

learning_rate_1�<�7U��xI       6%�	���E���A�*;


total_loss��@

error_R��@?

learning_rate_1�<�7�.�^I       6%�	ӑE���A�*;


total_lossC3y@

error_R�B?

learning_rate_1�<�7Ay9#I       6%�	��E���A�*;


total_loss3h�@

error_R��G?

learning_rate_1�<�7Hđ#I       6%�	{Q�E���A�*;


total_loss���@

error_R�7T?

learning_rate_1�<�7ܓxCI       6%�	ޒ�E���A�*;


total_loss���@

error_R2Q?

learning_rate_1�<�7�1� I       6%�	bђE���A�*;


total_loss�L�@

error_R�h=?

learning_rate_1�<�7#�/I       6%�	��E���A�*;


total_loss_7�@

error_RWI?

learning_rate_1�<�7Z��I       6%�	�Q�E���A�*;


total_loss���@

error_R}BP?

learning_rate_1�<�7ӂa�I       6%�	'��E���A�*;


total_loss1m�@

error_R�N?

learning_rate_1�<�7���vI       6%�	�ؓE���A�*;


total_lossH��@

error_RT�F?

learning_rate_1�<�7V��1I       6%�	��E���A�*;


total_lossMP�@

error_R�O?

learning_rate_1�<�7S�y�I       6%�	�[�E���A�*;


total_loss��A

error_R��l?

learning_rate_1�<�7�'�I       6%�	���E���A�*;


total_lossM(�@

error_R�Y?

learning_rate_1�<�7RV��I       6%�	s�E���A�*;


total_loss�w�@

error_R�YS?

learning_rate_1�<�7}�8�I       6%�	}&�E���A�*;


total_loss_P�@

error_R�OW?

learning_rate_1�<�7mC}�I       6%�	�k�E���A�*;


total_loss�6�@

error_REgO?

learning_rate_1�<�7��i�I       6%�	���E���A�*;


total_loss
��@

error_R�^?

learning_rate_1�<�7�a-I       6%�	p��E���A�*;


total_lossm*�@

error_R��I?

learning_rate_1�<�7U�T�I       6%�	�9�E���A�*;


total_loss�Ҍ@

error_R�H7?

learning_rate_1�<�7ӏ��I       6%�	%~�E���A�*;


total_loss���@

error_R�_?

learning_rate_1�<�7��E�I       6%�	��E���A�*;


total_loss�m�@

error_R�T?

learning_rate_1�<�7�ا�I       6%�	���E���A�*;


total_lossα�@

error_RDd?

learning_rate_1�<�7RD��I       6%�	�:�E���A�*;


total_loss�6�@

error_R��F?

learning_rate_1�<�7@mI       6%�	�z�E���A�*;


total_lossQi�@

error_R��W?

learning_rate_1�<�7��I       6%�	�֗E���A�*;


total_lossiM�@

error_RV`K?

learning_rate_1�<�7�>�I       6%�	?�E���A�*;


total_lossp�@

error_R}�X?

learning_rate_1�<�7Ŕ�I       6%�	q`�E���A�*;


total_loss���@

error_R�>?

learning_rate_1�<�7�ߎvI       6%�	���E���A�*;


total_losso�@

error_RE�C?

learning_rate_1�<�7�߷I       6%�	��E���A�*;


total_loss4A

error_R�bW?

learning_rate_1�<�7��I       6%�	�;�E���A�*;


total_loss�,�@

error_RrR?

learning_rate_1�<�7�MWjI       6%�	���E���A�*;


total_loss��@

error_R�Y?

learning_rate_1�<�7�݆I       6%�	/͙E���A�*;


total_loss��w@

error_R�,Y?

learning_rate_1�<�7�"�I       6%�	�E���A�*;


total_loss�ޡ@

error_R�FP?

learning_rate_1�<�7�;�+I       6%�	 P�E���A�*;


total_lossrb�@

error_R��b?

learning_rate_1�<�7xY�-I       6%�	j��E���A�*;


total_lossq�@

error_R�N?

learning_rate_1�<�7P�II       6%�	T՚E���A�*;


total_loss��A

error_RC[?

learning_rate_1�<�7-��+I       6%�	D�E���A�*;


total_loss_}�@

error_R�R??

learning_rate_1�<�7^<�I       6%�	�Y�E���A�*;


total_lossAО@

error_Ri�N?

learning_rate_1�<�7�?ZdI       6%�	g��E���A�*;


total_lossRZ�@

error_RH!U?

learning_rate_1�<�7W�DI       6%�	tٛE���A�*;


total_loss���@

error_R�4?

learning_rate_1�<�7����I       6%�	�E���A�*;


total_loss��@

error_R�'R?

learning_rate_1�<�7����I       6%�	�\�E���A�*;


total_loss�7�@

error_R\Y?

learning_rate_1�<�7Tx$�I       6%�	F��E���A�*;


total_loss4,�@

error_R_O?

learning_rate_1�<�7���4I       6%�	9ޜE���A�*;


total_loss�۩@

error_REb\?

learning_rate_1�<�7rl�I       6%�	!�E���A�*;


total_loss ��@

error_RC�M?

learning_rate_1�<�7B�}I       6%�	�`�E���A�*;


total_loss6d�@

error_R�zW?

learning_rate_1�<�7p�I       6%�	ퟝE���A�*;


total_loss��@

error_R�]?

learning_rate_1�<�7�v'YI       6%�	��E���A�*;


total_loss%��@

error_Rv<K?

learning_rate_1�<�7!w�I       6%�	l%�E���A�*;


total_loss4��@

error_R�[?

learning_rate_1�<�7��>�I       6%�	i�E���A�*;


total_lossd��@

error_R`�P?

learning_rate_1�<�7�p�I       6%�	Ψ�E���A�*;


total_lossTg�@

error_R��J?

learning_rate_1�<�7�R�I       6%�	��E���A�*;


total_loss�}�@

error_RV�G?

learning_rate_1�<�7[�;�I       6%�	�-�E���A�*;


total_loss4{�@

error_RWP?

learning_rate_1�<�7艦I       6%�	&q�E���A�*;


total_loss���@

error_R��/?

learning_rate_1�<�7-�*I       6%�	0��E���A�*;


total_loss�@

error_RJ�P?

learning_rate_1�<�7'���I       6%�	���E���A�*;


total_loss���@

error_R�b?

learning_rate_1�<�71�/�I       6%�	|;�E���A�*;


total_loss|��@

error_R�-D?

learning_rate_1�<�7�g�wI       6%�	6��E���A�*;


total_loss�5�@

error_R�HW?

learning_rate_1�<�7PM~�I       6%�	���E���A�*;


total_loss)��@

error_R�c?

learning_rate_1�<�7;T)�I       6%�	���E���A�*;


total_loss�ч@

error_R��P?

learning_rate_1�<�7ą�cI       6%�	�>�E���A�*;


total_loss!�@

error_RZ�9?

learning_rate_1�<�7�l04I       6%�	��E���A�*;


total_loss]" A

error_RW_N?

learning_rate_1�<�7qH�3I       6%�	�ǡE���A�*;


total_loss$9�@

error_R-+M?

learning_rate_1�<�7���I       6%�	��E���A�*;


total_loss�1�@

error_R8^Y?

learning_rate_1�<�7;`JaI       6%�	+T�E���A�*;


total_loss�(�@

error_R�aL?

learning_rate_1�<�7�BI       6%�	���E���A�*;


total_loss��@

error_RQ�J?

learning_rate_1�<�7h��9I       6%�	�ݢE���A�*;


total_loss��@

error_R�i@?

learning_rate_1�<�7�ᶲI       6%�	��E���A�*;


total_loss��@

error_RԺI?

learning_rate_1�<�7k?�I       6%�	*^�E���A�*;


total_loss���@

error_R�S?

learning_rate_1�<�7��I       6%�	���E���A�*;


total_loss\=�@

error_R;O?

learning_rate_1�<�7��T�I       6%�	��E���A�*;


total_loss��A

error_R1�l?

learning_rate_1�<�7"��pI       6%�	}%�E���A�*;


total_loss.qA

error_Rl$=?

learning_rate_1�<�7m@>�I       6%�	i�E���A�*;


total_loss��@

error_Rנa?

learning_rate_1�<�7?_Q�I       6%�	��E���A�*;


total_loss��@

error_R�4a?

learning_rate_1�<�7��z�I       6%�	n��E���A�*;


total_loss7�@

error_R�Z]?

learning_rate_1�<�7� <�I       6%�	�;�E���A�*;


total_loss���@

error_R�]?

learning_rate_1�<�7���I       6%�	�}�E���A�*;


total_lossř@

error_R�S?

learning_rate_1�<�7HS;I       6%�	|��E���A�*;


total_loss]^�@

error_R��X?

learning_rate_1�<�7��I       6%�	w�E���A�*;


total_loss�D�@

error_R��P?

learning_rate_1�<�7<NF�I       6%�	J�E���A�*;


total_losslP�@

error_R�^?

learning_rate_1�<�7���I       6%�	Z��E���A�*;


total_loss(�A

error_Rlgi?

learning_rate_1�<�7��]I       6%�	�ЦE���A�*;


total_lossl�@

error_RR�V?

learning_rate_1�<�7:�SI       6%�	K�E���A�*;


total_loss�П@

error_R��\?

learning_rate_1�<�7ÖF�I       6%�	�R�E���A�*;


total_loss��@

error_RY?

learning_rate_1�<�7�%��I       6%�	��E���A�*;


total_loss���@

error_R8�G?

learning_rate_1�<�7��I       6%�	T��E���A�*;


total_loss�=A

error_R��Q?

learning_rate_1�<�7Ղ:�I       6%�	9�E���A�*;


total_loss(�@

error_RmPH?

learning_rate_1�<�72�S�I       6%�	�|�E���A�*;


total_loss�"A

error_R�1O?

learning_rate_1�<�7s��I       6%�	KШE���A�*;


total_loss��@

error_R��J?

learning_rate_1�<�77ҪI       6%�	��E���A�*;


total_loss�P�@

error_R�*P?

learning_rate_1�<�7y�vI       6%�	1V�E���A�*;


total_lossa�A

error_RM 9?

learning_rate_1�<�7���I       6%�	v��E���A�*;


total_lossc<�@

error_R�	F?

learning_rate_1�<�7A�ƘI       6%�		֩E���A�*;


total_loss�&A

error_R]?

learning_rate_1�<�7E���I       6%�	��E���A�*;


total_loss�A

error_RJ~J?

learning_rate_1�<�7��"I       6%�	�W�E���A�*;


total_loss�P�@

error_Rح\?

learning_rate_1�<�7mVp5I       6%�	S��E���A�*;


total_lossdH�@

error_R�I?

learning_rate_1�<�7l��I       6%�	2ߪE���A�*;


total_loss�#�@

error_R��Y?

learning_rate_1�<�7��I       6%�	"�E���A�*;


total_lossJvA

error_R�G?

learning_rate_1�<�7Ǧ�WI       6%�	�e�E���A�*;


total_loss݁�@

error_R�K?

learning_rate_1�<�7Y��I       6%�	h��E���A�*;


total_lossJ�@

error_R�TG?

learning_rate_1�<�7��!I       6%�	T�E���A�*;


total_loss;�@

error_R%�I?

learning_rate_1�<�7t��I       6%�	h*�E���A�*;


total_loss��@

error_R��C?

learning_rate_1�<�7z��9I       6%�	rl�E���A�*;


total_loss���@

error_R�~8?

learning_rate_1�<�7��I       6%�	���E���A�*;


total_loss��@

error_R��X?

learning_rate_1�<�7ӧ^dI       6%�	��E���A�*;


total_loss(� A

error_Ri�O?

learning_rate_1�<�7m5w�I       6%�	�3�E���A�*;


total_loss;A

error_R��U?

learning_rate_1�<�7��I       6%�	�x�E���A�*;


total_loss�4�@

error_R�E?

learning_rate_1�<�7�d�(I       6%�	仭E���A�*;


total_loss��A

error_R��V?

learning_rate_1�<�70��I       6%�	 �E���A�*;


total_loss�^ A

error_RڪS?

learning_rate_1�<�7�J5UI       6%�	QA�E���A�*;


total_lossCx�@

error_R��P?

learning_rate_1�<�7PZv�I       6%�	���E���A�*;


total_loss�
�@

error_R�E?

learning_rate_1�<�7et�I       6%�	�ƮE���A�*;


total_loss�[�@

error_RT�Q?

learning_rate_1�<�7�W��I       6%�	a�E���A�*;


total_loss�ٯ@

error_R��R?

learning_rate_1�<�7ǲ@�I       6%�	�G�E���A�*;


total_lossΩ�@

error_R�XH?

learning_rate_1�<�7M.BI       6%�	���E���A�*;


total_loss���@

error_R�C?

learning_rate_1�<�7��(gI       6%�	�ɯE���A�*;


total_lossQ�@

error_R,�J?

learning_rate_1�<�7H�X�I       6%�	��E���A�*;


total_lossx.�@

error_RO�_?

learning_rate_1�<�7E�LgI       6%�	�N�E���A�*;


total_loss�b@

error_R�\K?

learning_rate_1�<�7`��@I       6%�	;��E���A�*;


total_lossl��@

error_R��O?

learning_rate_1�<�7%�FI       6%�	�ְE���A�*;


total_lossf��@

error_R NM?

learning_rate_1�<�7��{I       6%�	��E���A�*;


total_loss�rA

error_R^?

learning_rate_1�<�7��OI       6%�	<V�E���A�*;


total_loss�8�@

error_R�j_?

learning_rate_1�<�7��}�I       6%�	 ��E���A�*;


total_loss�ƈ@

error_R��V?

learning_rate_1�<�77�yI       6%�	*ֱE���A�*;


total_loss�|�@

error_R$3U?

learning_rate_1�<�7�%��I       6%�	@�E���A�*;


total_lossᶍ@

error_R�05?

learning_rate_1�<�7t�q�I       6%�	�Z�E���A�*;


total_lossV��@

error_R�:M?

learning_rate_1�<�7�ǴI       6%�	���E���A�*;


total_loss䳴@

error_R�I?

learning_rate_1�<�7���I       6%�	0�E���A�*;


total_lossE��@

error_R�L?

learning_rate_1�<�7Z�N�I       6%�	�,�E���A�*;


total_loss\Q�@

error_Rܹ\?

learning_rate_1�<�7w'�xI       6%�	�q�E���A�*;


total_loss�b�@

error_R�xK?

learning_rate_1�<�7\t�I       6%�	���E���A�*;


total_losscj�@

error_R�UJ?

learning_rate_1�<�7gM<HI       6%�	���E���A�*;


total_loss�@

error_R_9S?

learning_rate_1�<�7d�+I       6%�	s:�E���A�*;


total_lossH��@

error_R@?

learning_rate_1�<�7x��I       6%�	0}�E���A�*;


total_loss�3�@

error_RCzK?

learning_rate_1�<�7����I       6%�	6��E���A�*;


total_loss��A

error_R�0P?

learning_rate_1�<�7�yNI       6%�	]�E���A�*;


total_loss�³@

error_RRD?

learning_rate_1�<�7�$��I       6%�	�C�E���A�*;


total_loss� �@

error_RnL?

learning_rate_1�<�72>I       6%�	%��E���A�*;


total_loss��@

error_R*6?

learning_rate_1�<�7
��I       6%�	�ŵE���A�*;


total_loss��@

error_R�e?

learning_rate_1�<�7���GI       6%�	��E���A�*;


total_lossӇ�@

error_R�
L?

learning_rate_1�<�7� I       6%�	�E�E���A�*;


total_loss�@

error_R�&O?

learning_rate_1�<�7����I       6%�	m��E���A�*;


total_loss|�@

error_R�S?

learning_rate_1�<�7��fI       6%�	ȶE���A�*;


total_loss�S�@

error_R,�Q?

learning_rate_1�<�7:�>�I       6%�	�E���A�*;


total_loss���@

error_Rx�=?

learning_rate_1�<�7�}��I       6%�	�E�E���A�*;


total_lossQO�@

error_R1�S?

learning_rate_1�<�7�B�RI       6%�	E��E���A�*;


total_loss�m@

error_RS-?

learning_rate_1�<�7Fm�I       6%�	�E���A�*;


total_losss��@

error_R�0Z?

learning_rate_1�<�7|�o�I       6%�	�5�E���A�*;


total_loss��@

error_R�G?

learning_rate_1�<�7)v�I       6%�	xv�E���A�*;


total_loss>�@

error_R1�G?

learning_rate_1�<�7��VI       6%�	���E���A�*;


total_loss��@

error_R
�D?

learning_rate_1�<�7J��*I       6%�	���E���A�*;


total_loss`��@

error_R3}@?

learning_rate_1�<�7��$�I       6%�	�@�E���A�*;


total_lossz��@

error_R�|d?

learning_rate_1�<�7ă�EI       6%�	���E���A�*;


total_loss��@

error_R�^W?

learning_rate_1�<�7[���I       6%�	uɹE���A�*;


total_loss}��@

error_Rv�T?

learning_rate_1�<�7�%�3I       6%�	L�E���A�*;


total_losslv�@

error_R�{P?

learning_rate_1�<�7�к�I       6%�	�W�E���A�*;


total_lossO��@

error_RqP?

learning_rate_1�<�7'���I       6%�	���E���A�*;


total_loss��@

error_R�H?

learning_rate_1�<�7�G}�I       6%�	�޺E���A�*;


total_losso~�@

error_R�Y?

learning_rate_1�<�7Ux��I       6%�	v!�E���A�*;


total_loss�@

error_R��L?

learning_rate_1�<�7���I       6%�	�a�E���A�*;


total_lossDb�@

error_R4bK?

learning_rate_1�<�7�+EI       6%�	��E���A�*;


total_loss���@

error_R!�G?

learning_rate_1�<�7��0�I       6%�	
�E���A�*;


total_loss��A

error_RRqO?

learning_rate_1�<�7�۪YI       6%�	j<�E���A�*;


total_loss��@

error_R�~O?

learning_rate_1�<�7!M�zI       6%�	��E���A�*;


total_loss4�@

error_RQ;?

learning_rate_1�<�7Q�� I       6%�	B��E���A�*;


total_lossf;�@

error_RW�>?

learning_rate_1�<�7���I       6%�	�
�E���A�*;


total_lossz��@

error_Rd�P?

learning_rate_1�<�7��GI       6%�	1R�E���A�*;


total_loss{��@

error_R}A?

learning_rate_1�<�7D��TI       6%�	ǔ�E���A�*;


total_loss�U�@

error_R��.?

learning_rate_1�<�7N���I       6%�	�ֽE���A�*;


total_lossnu@

error_R�vO?

learning_rate_1�<�7P���I       6%�	 �E���A�*;


total_loss�Ζ@

error_RdyO?

learning_rate_1�<�7(]�4I       6%�	tW�E���A�*;


total_loss1k�@

error_R�Y?

learning_rate_1�<�7����I       6%�	���E���A�*;


total_loss83�@

error_R�%X?

learning_rate_1�<�7��B�I       6%�	�۾E���A�*;


total_loss7
�@

error_RZ�K?

learning_rate_1�<�7_ABI       6%�	��E���A�*;


total_loss���@

error_RجQ?

learning_rate_1�<�7E�8�I       6%�	$_�E���A�*;


total_lossDf�@

error_R�e\?

learning_rate_1�<�7��9rI       6%�	���E���A�*;


total_lossĿA

error_R��=?

learning_rate_1�<�7�кLI       6%�	K�E���A�*;


total_loss�A

error_Ri�D?

learning_rate_1�<�7�V}I       6%�	1�E���A�*;


total_loss��A

error_R!�O?

learning_rate_1�<�7/귇I       6%�	�x�E���A�*;


total_loss�}�@

error_R�y@?

learning_rate_1�<�7���QI       6%�	۽�E���A�*;


total_loss8��@

error_R�+I?

learning_rate_1�<�7�m.�I       6%�	���E���A�*;


total_loss�?�@

error_R��U?

learning_rate_1�<�7��Y�I       6%�	nA�E���A�*;


total_loss{��@

error_R��\?

learning_rate_1�<�7�R��I       6%�	%��E���A�*;


total_loss���@

error_R��[?

learning_rate_1�<�7� I       6%�	��E���A�*;


total_lossrÌ@

error_R�KN?

learning_rate_1�<�7MM£I       6%�	��E���A�*;


total_loss�v�@

error_Rw6I?

learning_rate_1�<�7ލx<I       6%�	C�E���A�*;


total_loss�@

error_RR�I?

learning_rate_1�<�7�I       6%�	��E���A�*;


total_loss���@

error_R.�??

learning_rate_1�<�7��@I       6%�	���E���A�*;


total_loss��@

error_R*�X?

learning_rate_1�<�76�I       6%�	�	�E���A�*;


total_lossӤ@

error_R��D?

learning_rate_1�<�7�J1�I       6%�	�K�E���A�*;


total_lossS��@

error_R��X?

learning_rate_1�<�7�cI       6%�	L��E���A�*;


total_lossa7�@

error_R��K?

learning_rate_1�<�7���I       6%�	x��E���A�*;


total_loss��@

error_Rs�D?

learning_rate_1�<�72J6I       6%�	_�E���A�*;


total_loss���@

error_R`]:?

learning_rate_1�<�7n%�I       6%�	�L�E���A�*;


total_loss�g�@

error_RJ�I?

learning_rate_1�<�7�2w%I       6%�	6��E���A�*;


total_loss���@

error_R�P?

learning_rate_1�<�7��<I       6%�	��E���A�*;


total_losse�@

error_R_�F?

learning_rate_1�<�7^NsbI       6%�	��E���A�*;


total_lossm0h@

error_R��A?

learning_rate_1�<�77���I       6%�	�N�E���A�*;


total_loss1��@

error_Ri80?

learning_rate_1�<�7���uI       6%�	��E���A�*;


total_loss쉺@

error_RT�S?

learning_rate_1�<�77}1I       6%�	���E���A�*;


total_lossÓ�@

error_R	�C?

learning_rate_1�<�7�a�I       6%�	��E���A�*;


total_loss�RA

error_R�N?

learning_rate_1�<�7b���I       6%�	XY�E���A�*;


total_loss�A

error_RV_?

learning_rate_1�<�7��I       6%�	ϛ�E���A�*;


total_loss�_�@

error_RC�Z?

learning_rate_1�<�7vo�:I       6%�	]��E���A�*;


total_loss@

error_R��E?

learning_rate_1�<�7��fNI       6%�	'�E���A�*;


total_loss���@

error_R��b?

learning_rate_1�<�7	�I       6%�	i^�E���A�*;


total_loss���@

error_R��G?

learning_rate_1�<�7���gI       6%�	ֲ�E���A�*;


total_lossa�@

error_RE�G?

learning_rate_1�<�7��'I       6%�	L�E���A�*;


total_loss��@

error_R�fR?

learning_rate_1�<�7C��I       6%�	�O�E���A�*;


total_loss���@

error_R�>?

learning_rate_1�<�7{}}�I       6%�	E��E���A�*;


total_lossw2�@

error_R�(L?

learning_rate_1�<�7��h�I       6%�	���E���A�*;


total_loss2��@

error_R�*d?

learning_rate_1�<�7͗��I       6%�	|�E���A�*;


total_lossF �@

error_R��B?

learning_rate_1�<�7�w�I       6%�	T�E���A�*;


total_loss}��@

error_R*�X?

learning_rate_1�<�7n�)I       6%�	4��E���A�*;


total_loss��A

error_R��B?

learning_rate_1�<�7���nI       6%�	���E���A�*;


total_loss1f@

error_RđK?

learning_rate_1�<�75T4I       6%�	��E���A�*;


total_lossꃪ@

error_R
,J?

learning_rate_1�<�7�fʠI       6%�	4Y�E���A�*;


total_loss���@

error_RS�J?

learning_rate_1�<�7O�fI       6%�	ܙ�E���A�*;


total_lossn4�@

error_R�VE?

learning_rate_1�<�7ђ�yI       6%�	C��E���A�*;


total_loss�R�@

error_R�m[?

learning_rate_1�<�7��s�I       6%�	O�E���A�*;


total_lossH��@

error_R��]?

learning_rate_1�<�7����I       6%�	{_�E���A�*;


total_loss{��@

error_R�`S?

learning_rate_1�<�7ɘ�I       6%�	z��E���A�*;


total_loss�7�@

error_RL�F?

learning_rate_1�<�74�|I       6%�	#��E���A�*;


total_loss$��@

error_R�&@?

learning_rate_1�<�7e���I       6%�	J%�E���A�*;


total_loss�!1A

error_R��B?

learning_rate_1�<�7��MI       6%�	�d�E���A�*;


total_lossA��@

error_R��V?

learning_rate_1�<�7  y�I       6%�	S��E���A�*;


total_loss��@

error_R0?

learning_rate_1�<�7�?�I       6%�	1��E���A�*;


total_loss�W�@

error_R{�c?

learning_rate_1�<�7��koI       6%�	�+�E���A�*;


total_lossI��@

error_RwW?

learning_rate_1�<�7�q��I       6%�	�p�E���A�*;


total_lossؿ�@

error_R�W?

learning_rate_1�<�7��I       6%�	k��E���A�*;


total_loss�$�@

error_R�d?

learning_rate_1�<�7�8"�I       6%�	���E���A�*;


total_lossI�A

error_R�<@?

learning_rate_1�<�7���I       6%�	�7�E���A�*;


total_loss)��@

error_R�-]?

learning_rate_1�<�7����I       6%�	Uz�E���A�*;


total_loss��@

error_R��N?

learning_rate_1�<�7����I       6%�	��E���A�*;


total_lossv��@

error_ReJS?

learning_rate_1�<�7'�qI       6%�	���E���A�*;


total_losso�@

error_R!�K?

learning_rate_1�<�7�?)I       6%�	�=�E���A�*;


total_loss�B�@

error_RL?

learning_rate_1�<�7�1t
I       6%�	��E���A�*;


total_loss�+�@

error_R�~C?

learning_rate_1�<�7�JŭI       6%�	���E���A�*;


total_loss���@

error_R�MV?

learning_rate_1�<�7�Pv�I       6%�	s�E���A�*;


total_loss<	A

error_R?)]?

learning_rate_1�<�7��l|I       6%�	�I�E���A�*;


total_loss���@

error_Rň@?

learning_rate_1�<�7f�I       6%�	W��E���A�*;


total_loss��@

error_RɲC?

learning_rate_1�<�7w��PI       6%�	���E���A�*;


total_loss[u�@

error_RS�S?

learning_rate_1�<�7�?|I       6%�	��E���A�*;


total_loss���@

error_R��:?

learning_rate_1�<�7��I       6%�	YJ�E���A�*;


total_loss�3�@

error_R\?

learning_rate_1�<�7?`L.I       6%�	���E���A�*;


total_loss�V�@

error_Ri$_?

learning_rate_1�<�7���jI       6%�	4��E���A�*;


total_loss�3y@

error_R�~\?

learning_rate_1�<�7�I       6%�	��E���A�*;


total_lossC��@

error_RR.A?

learning_rate_1�<�7��ΛI       6%�	�J�E���A�*;


total_lossh �@

error_R��P?

learning_rate_1�<�7>�I       6%�	���E���A�*;


total_lossH��@

error_R��M?

learning_rate_1�<�7j��I       6%�	���E���A�*;


total_lossEr�@

error_R}IV?

learning_rate_1�<�7�qI       6%�	��E���A�*;


total_loss��@

error_R��E?

learning_rate_1�<�7D`΃I       6%�	�M�E���A�*;


total_lossֽ�@

error_R �L?

learning_rate_1�<�7�K�I       6%�	���E���A�*;


total_loss�=�@

error_R�P?

learning_rate_1�<�7����I       6%�	���E���A�*;


total_loss���@

error_RO[?

learning_rate_1�<�7�Ä�I       6%�	�"�E���A�*;


total_loss���@

error_RV�??

learning_rate_1�<�7�`��I       6%�	�u�E���A�*;


total_loss! �@

error_R��M?

learning_rate_1�<�7��s�I       6%�	{��E���A�*;


total_lossl�@

error_R�[?

learning_rate_1�<�7�7I       6%�	+��E���A�*;


total_loss_�@

error_R?L?

learning_rate_1�<�7:�XEI       6%�	~R�E���A�*;


total_lossվ@

error_R��M?

learning_rate_1�<�7)�vI       6%�	?��E���A�*;


total_loss*�@

error_R�&O?

learning_rate_1�<�7w�I       6%�	n��E���A�*;


total_loss`b�@

error_R,lK?

learning_rate_1�<�7Z �MI       6%�	21�E���A�*;


total_losss��@

error_R��O?

learning_rate_1�<�7��B�I       6%�	�t�E���A�*;


total_loss���@

error_R7�M?

learning_rate_1�<�7�%�I       6%�	m��E���A�*;


total_loss�~�@

error_Rl5S?

learning_rate_1�<�79��&I       6%�	t��E���A�*;


total_loss��@

error_R T?

learning_rate_1�<�7��
 I       6%�	r;�E���A�*;


total_loss��@

error_R��J?

learning_rate_1�<�7�7�I       6%�	@}�E���A�*;


total_loss���@

error_RQJ?

learning_rate_1�<�7*���I       6%�	Z��E���A�*;


total_loss��A

error_Rf�V?

learning_rate_1�<�7���I       6%�	�#�E���A�*;


total_loss�1�@

error_RQ?

learning_rate_1�<�7�ˈ�I       6%�	Fm�E���A�*;


total_loss���@

error_R�S?

learning_rate_1�<�7��<�I       6%�	B��E���A�*;


total_lossH�@

error_R�WO?

learning_rate_1�<�7�1GlI       6%�	��E���A�*;


total_loss�@�@

error_RJEG?

learning_rate_1�<�7����I       6%�	�4�E���A�*;


total_loss�b�@

error_R{n[?

learning_rate_1�<�7�B#�I       6%�	�u�E���A�*;


total_loss�;�@

error_R�C?

learning_rate_1�<�7m'��I       6%�	��E���A�*;


total_loss��@

error_R�L?

learning_rate_1�<�7�)I       6%�	���E���A�*;


total_loss!�j@

error_R�G@?

learning_rate_1�<�7`o05I       6%�	;�E���A�*;


total_loss�~�@

error_R�cN?

learning_rate_1�<�7J�lI       6%�	)}�E���A�*;


total_loss�P�@

error_R��T?

learning_rate_1�<�7��ѮI       6%�	��E���A�*;


total_lossa�@

error_RHWW?

learning_rate_1�<�7����I       6%�	C�E���A�*;


total_loss��@

error_R��O?

learning_rate_1�<�7�U4QI       6%�	�E�E���A�*;


total_loss�j
A

error_R\uL?

learning_rate_1�<�7�W��I       6%�	���E���A�*;


total_loss��@

error_R�^L?

learning_rate_1�<�7;�HI       6%�	���E���A�*;


total_lossJZ�@

error_R1�<?

learning_rate_1�<�7����I       6%�	�E���A�*;


total_loss���@

error_RV�V?

learning_rate_1�<�7Q/hvI       6%�	EI�E���A�*;


total_loss�<�@

error_R,C?

learning_rate_1�<�7��ХI       6%�	���E���A�*;


total_loss�{m@

error_Rv�D?

learning_rate_1�<�7a��ZI       6%�	���E���A�*;


total_lossMӶ@

error_R��H?

learning_rate_1�<�7���KI       6%�	@�E���A�*;


total_loss�E�@

error_RF#X?

learning_rate_1�<�7���I       6%�	h�E���A�*;


total_loss���@

error_R{�D?

learning_rate_1�<�7�{��I       6%�	֩�E���A�*;


total_loss�i�@

error_R/aG?

learning_rate_1�<�7A�?�I       6%�	���E���A�*;


total_loss��@

error_R*V?

learning_rate_1�<�7L{`]I       6%�	*�E���A�*;


total_loss�P�@

error_R��D?

learning_rate_1�<�7��@sI       6%�	�h�E���A�*;


total_loss���@

error_R�SA?

learning_rate_1�<�7���#I       6%�	 ��E���A�*;


total_loss���@

error_R�^Z?

learning_rate_1�<�7���wI       6%�	^��E���A�*;


total_loss��@

error_Ri�b?

learning_rate_1�<�7\�I       6%�	84�E���A�*;


total_loss�F�@

error_RnDJ?

learning_rate_1�<�7Nۂ2I       6%�	�w�E���A�*;


total_loss���@

error_R��G?

learning_rate_1�<�7� b�I       6%�	���E���A�*;


total_lossx,�@

error_R%�l?

learning_rate_1�<�7��z�I       6%�	m �E���A�*;


total_loss��A

error_Ri+I?

learning_rate_1�<�7�d�I       6%�	�C�E���A�*;


total_loss<t�@

error_RsuF?

learning_rate_1�<�76�8�I       6%�	U��E���A�*;


total_loss/��@

error_R��_?

learning_rate_1�<�7�0cZI       6%�	v��E���A�*;


total_loss#�@

error_R{M?

learning_rate_1�<�7̜^I       6%�	;�E���A�*;


total_loss.��@

error_R=5Z?

learning_rate_1�<�72Yr�I       6%�	tP�E���A�*;


total_lossF�@

error_R@UT?

learning_rate_1�<�7[/7I       6%�	��E���A�*;


total_lossLhA

error_R�@C?

learning_rate_1�<�7v�oI       6%�	'��E���A�*;


total_loss׍�@

error_RHfI?

learning_rate_1�<�7�#zI       6%�	��E���A�*;


total_loss�s@

error_R�R?

learning_rate_1�<�7u�jI       6%�	�R�E���A�*;


total_loss�W�@

error_Ru@?

learning_rate_1�<�7�"I       6%�	���E���A�*;


total_loss��@

error_R��Z?

learning_rate_1�<�7ž�I       6%�	���E���A�*;


total_loss��@

error_R�rJ?

learning_rate_1�<�7�nlrI       6%�	<�E���A�*;


total_loss�ր@

error_Rӡ??

learning_rate_1�<�7�sI       6%�	�P�E���A�*;


total_lossc4�@

error_R�J?

learning_rate_1�<�7���>I       6%�	{��E���A�*;


total_loss�^�@

error_Ra�N?

learning_rate_1�<�7�g�I       6%�	��E���A�*;


total_loss�@

error_R��F?

learning_rate_1�<�7�4�I       6%�	5�E���A�*;


total_loss'm�@

error_R�A?

learning_rate_1�<�7l
�I       6%�	;S�E���A�*;


total_loss�@

error_R�Q?

learning_rate_1�<�7vzd2I       6%�	��E���A�*;


total_loss���@

error_R�AK?

learning_rate_1�<�7�Y�2I       6%�	*��E���A�*;


total_lossF��@

error_R`yG?

learning_rate_1�<�7��5�I       6%�	(�E���A�*;


total_loss�A

error_R3�L?

learning_rate_1�<�7T���I       6%�	�c�E���A�*;


total_loss�2�@

error_RX�=?

learning_rate_1�<�7���0I       6%�	���E���A�*;


total_loss���@

error_R-lY?

learning_rate_1�<�7����I       6%�	��E���A�*;


total_loss���@

error_R%M?

learning_rate_1�<�7��KI       6%�	�.�E���A�*;


total_loss���@

error_R\�\?

learning_rate_1�<�7���}I       6%�	�o�E���A�*;


total_loss�+�@

error_R�AH?

learning_rate_1�<�75`�I       6%�	���E���A�*;


total_loss�4�@

error_R��S?

learning_rate_1�<�7����I       6%�	���E���A�*;


total_loss!I�@

error_R�E?

learning_rate_1�<�7��`�I       6%�	@:�E���A�*;


total_loss�'�@

error_R��=?

learning_rate_1�<�7��8I       6%�	�}�E���A�*;


total_loss_�@

error_R��L?

learning_rate_1�<�7&CI       6%�	C��E���A�*;


total_loss�$�@

error_R;�_?

learning_rate_1�<�7{�cI       6%�	&�E���A�*;


total_loss�M�@

error_R�af?

learning_rate_1�<�7�
�I       6%�	Rf�E���A�*;


total_loss�u�@

error_RŘO?

learning_rate_1�<�7�j��I       6%�	!��E���A�*;


total_loss*��@

error_R�NR?

learning_rate_1�<�7u0��I       6%�	���E���A�*;


total_loss��@

error_R��Y?

learning_rate_1�<�74���I       6%�	B0�E���A�*;


total_lossTG�@

error_R�G?

learning_rate_1�<�7J	��I       6%�	�t�E���A�*;


total_lossٝ@

error_R��A?

learning_rate_1�<�7��_I       6%�	��E���A�*;


total_loss��z@

error_RL.C?

learning_rate_1�<�7�d�	I       6%�	%��E���A�*;


total_lossf��@

error_RI�U?

learning_rate_1�<�7��kI       6%�	�7�E���A�*;


total_loss�o�@

error_R@2G?

learning_rate_1�<�7$��I       6%�	gw�E���A�*;


total_loss���@

error_R��F?

learning_rate_1�<�7�?PYI       6%�	���E���A�*;


total_loss���@

error_R�9[?

learning_rate_1�<�7�@I       6%�	M��E���A�*;


total_lossu�@

error_R�=Q?

learning_rate_1�<�7ϛOI       6%�	;=�E���A�*;


total_loss*��@

error_R��M?

learning_rate_1�<�7k�I       6%�	�|�E���A�*;


total_loss	�@

error_R��_?

learning_rate_1�<�7�(?�I       6%�	*��E���A�*;


total_loss`��@

error_R�F?

learning_rate_1�<�7�V.�I       6%�	f�E���A�*;


total_loss���@

error_R�OA?

learning_rate_1�<�7د��I       6%�	�I�E���A�*;


total_loss���@

error_R�U?

learning_rate_1�<�7���II       6%�	���E���A�*;


total_loss���@

error_R��T?

learning_rate_1�<�7e
��I       6%�	���E���A�*;


total_loss�s�@

error_R�@M?

learning_rate_1�<�7.`�I       6%�	x�E���A�*;


total_loss,�@

error_Ri�H?

learning_rate_1�<�7&Zx�I       6%�	<Z�E���A�*;


total_loss��@

error_RJI?

learning_rate_1�<�7�#BI       6%�	ڞ�E���A�*;


total_loss�A�@

error_R$/R?

learning_rate_1�<�7�^ZsI       6%�	���E���A�*;


total_loss�n�@

error_R�ra?

learning_rate_1�<�7�ԞI       6%�	��E���A�*;


total_lossR�@

error_RtT?

learning_rate_1�<�7�9�I       6%�	�\�E���A�*;


total_loss� �@

error_R��U?

learning_rate_1�<�7T;�I       6%�	���E���A�*;


total_lossA

error_R
lU?

learning_rate_1�<�7Ьh�I       6%�	)��E���A�*;


total_loss(��@

error_R��T?

learning_rate_1�<�7
���I       6%�	f�E���A�*;


total_loss,��@

error_R��O?

learning_rate_1�<�70�I       6%�	�X�E���A�*;


total_loss�:[@

error_RMSS?

learning_rate_1�<�7S|�I       6%�	)��E���A�*;


total_loss���@

error_R��N?

learning_rate_1�<�7���!I       6%�	Z��E���A�*;


total_loss�6�@

error_Rܞ[?

learning_rate_1�<�7L��I       6%�	6�E���A�*;


total_lossj�@

error_R�F?

learning_rate_1�<�7�
	�I       6%�	�]�E���A�*;


total_loss���@

error_RQX?

learning_rate_1�<�7��I       6%�		��E���A�*;


total_loss���@

error_R:S?

learning_rate_1�<�7���YI       6%�	���E���A�*;


total_loss��@

error_R�X?

learning_rate_1�<�7i�הI       6%�	 �E���A�*;


total_loss�Ş@

error_R3K?

learning_rate_1�<�7�o�I       6%�	�a�E���A�*;


total_loss=�@

error_RO6?

learning_rate_1�<�79ƭI       6%�	͢�E���A�*;


total_loss�
�@

error_RW�5?

learning_rate_1�<�7�+t�I       6%�	���E���A�*;


total_loss�ܘ@

error_RAZ?

learning_rate_1�<�7�:��I       6%�	�+�E���A�*;


total_loss���@

error_R�6?

learning_rate_1�<�7�bBI       6%�	�p�E���A�*;


total_loss�F�@

error_R��5?

learning_rate_1�<�7u�@I       6%�	���E���A�*;


total_loss���@

error_R�U??

learning_rate_1�<�7�,/�I       6%�	�E���A�*;


total_loss�a�@

error_R��T?

learning_rate_1�<�7��HI       6%�	D�E���A�*;


total_loss���@

error_RM�A?

learning_rate_1�<�7�u�wI       6%�	%��E���A�*;


total_lossfu�@

error_R]�H?

learning_rate_1�<�7*��I       6%�	W��E���A�*;


total_loss�#A

error_R��^?

learning_rate_1�<�7��9I       6%�	8�E���A�*;


total_lossl��@

error_R�mF?

learning_rate_1�<�7�9�I       6%�	CR�E���A�*;


total_lossM��@

error_RZ�P?

learning_rate_1�<�7���bI       6%�	���E���A�*;


total_loss��@

error_R͈M?

learning_rate_1�<�7�k��I       6%�	���E���A�*;


total_loss"��@

error_RΛ9?

learning_rate_1�<�7���I       6%�	U�E���A�*;


total_loss���@

error_R�d?

learning_rate_1�<�7�L�WI       6%�	�^�E���A�*;


total_losssT�@

error_RӜ;?

learning_rate_1�<�7��,	I       6%�	K��E���A�*;


total_loss��@

error_Ra�k?

learning_rate_1�<�7c3�I       6%�	���E���A�*;


total_loss|��@

error_RnO=?

learning_rate_1�<�7���;I       6%�	1+�E���A�*;


total_loss���@

error_R�3O?

learning_rate_1�<�7Q6�I       6%�	!p�E���A�*;


total_loss�^�@

error_R�P?

learning_rate_1�<�7v��I       6%�	���E���A�*;


total_loss�;�@

error_R��P?

learning_rate_1�<�7RAI       6%�	���E���A�*;


total_loss�T�@

error_R�>V?

learning_rate_1�<�7���0I       6%�	L:�E���A�*;


total_loss���@

error_R�dE?

learning_rate_1�<�7�֖I       6%�	�}�E���A�*;


total_loss�@�@

error_RʋQ?

learning_rate_1�<�7\�̛I       6%�	���E���A�*;


total_loss0�@

error_RjeG?

learning_rate_1�<�7��D�I       6%�	)�E���A�*;


total_loss�4A

error_R��b?

learning_rate_1�<�7�;�I       6%�	�i�E���A�*;


total_loss��@

error_R�V?

learning_rate_1�<�7�#ԈI       6%�	m��E���A�*;


total_loss��@

error_R	V?

learning_rate_1�<�7�/BtI       6%�	���E���A�*;


total_loss/h�@

error_RbN?

learning_rate_1�<�7G�Z�I       6%�	H,�E���A�*;


total_lossLi�@

error_R��O?

learning_rate_1�<�7�I       6%�	1m�E���A�*;


total_loss���@

error_Ri�:?

learning_rate_1�<�7k�\�I       6%�	��E���A�*;


total_loss�@�@

error_R��6?

learning_rate_1�<�7vdBI       6%�	��E���A�*;


total_loss|�@

error_RH�T?

learning_rate_1�<�7¦�I       6%�	�:�E���A�*;


total_loss���@

error_R8�T?

learning_rate_1�<�7�EbfI       6%�	Z~�E���A�*;


total_loss.1�@

error_R}�O?

learning_rate_1�<�7@sI       6%�	���E���A�*;


total_loss���@

error_R��N?

learning_rate_1�<�7��f�I       6%�	'�E���A�*;


total_loss� �@

error_R��<?

learning_rate_1�<�7�@��I       6%�	UG�E���A�*;


total_loss+��@

error_RJtW?

learning_rate_1�<�7®� I       6%�	Ɋ�E���A�*;


total_lossl&�@

error_RI?

learning_rate_1�<�7�+��I       6%�	D��E���A�*;


total_loss$��@

error_R�8G?

learning_rate_1�<�7T�JI       6%�	M��E���A�*;


total_losswU�@

error_RH�]?

learning_rate_1�<�784;�I       6%�	�#�E���A�*;


total_losse�@

error_RDsM?

learning_rate_1�<�7bg�{I       6%�	�g�E���A�*;


total_loss���@

error_R�eL?

learning_rate_1�<�7�jȸI       6%�	��E���A�*;


total_loss`TU@

error_Rۉ@?

learning_rate_1�<�7L�o�I       6%�	���E���A�*;


total_loss/o0A

error_R�BD?

learning_rate_1�<�7�`�I       6%�	�1 F���A�*;


total_loss�{�@

error_R̩N?

learning_rate_1�<�7��|I       6%�	r F���A�*;


total_loss��@

error_R�TO?

learning_rate_1�<�7�]I       6%�	�� F���A�*;


total_loss�G�@

error_Rx�T?

learning_rate_1�<�7_�I       6%�	P� F���A�*;


total_loss�A

error_R��O?

learning_rate_1�<�7�Wq�I       6%�	�4F���A�*;


total_loss�W�@

error_R��J?

learning_rate_1�<�7����I       6%�	TyF���A�*;


total_loss^�@

error_R�dP?

learning_rate_1�<�7't/fI       6%�	-�F���A�*;


total_loss�@

error_R�WQ?

learning_rate_1�<�7�_[�I       6%�	��F���A�*;


total_loss��@

error_R��L?

learning_rate_1�<�7����I       6%�	"BF���A�*;


total_loss�O�@

error_R47H?

learning_rate_1�<�78�r�I       6%�	\�F���A�*;


total_loss���@

error_R��A?

learning_rate_1�<�7]`.�I       6%�	��F���A�*;


total_loss��A

error_R�P?

learning_rate_1�<�7�4��I       6%�	CF���A�*;


total_loss&��@

error_RܤT?

learning_rate_1�<�7WhuI       6%�	PF���A�*;


total_lossz�@

error_R��T?

learning_rate_1�<�7���@I       6%�	�F���A�*;


total_loss&�t@

error_R
~I?

learning_rate_1�<�7�OI       6%�	N�F���A�*;


total_loss}E�@

error_R��R?

learning_rate_1�<�7�>�I       6%�	#F���A�*;


total_loss}�@

error_R3�O?

learning_rate_1�<�7R���I       6%�	�QF���A�*;


total_loss�L�@

error_RϏX?

learning_rate_1�<�7f�وI       6%�	��F���A�*;


total_lossX&�@

error_R{zd?

learning_rate_1�<�7��I       6%�	R�F���A�*;


total_loss���@

error_R�5?

learning_rate_1�<�7I@�I       6%�	�F���A�*;


total_lossr�@

error_RR�??

learning_rate_1�<�7��5=I       6%�	�TF���A�*;


total_loss��@

error_R�Q\?

learning_rate_1�<�74�VhI       6%�	�F���A�*;


total_lossk��@

error_R��L?

learning_rate_1�<�7U���I       6%�	r�F���A�*;


total_loss:��@

error_R�K?

learning_rate_1�<�7�C�I       6%�	?F���A�*;


total_lossn��@

error_R��W?

learning_rate_1�<�7�v�I       6%�	FWF���A�*;


total_loss�^�@

error_R8?

learning_rate_1�<�7��jI       6%�	��F���A�*;


total_loss(�@

error_R��9?

learning_rate_1�<�7���I       6%�	��F���A�*;


total_loss�9�@

error_R!T?

learning_rate_1�<�7���!I       6%�	�F���A�*;


total_loss�̩@

error_R�<?

learning_rate_1�<�7'��~I       6%�	�[F���A�*;


total_loss���@

error_R�S?

learning_rate_1�<�7E�IGI       6%�	?�F���A�*;


total_loss%S�@

error_R��M?

learning_rate_1�<�7|��I       6%�	^ F���A�*;


total_loss!�@

error_R�R?

learning_rate_1�<�7��o�I       6%�	�AF���A�*;


total_loss�ج@

error_R}$??

learning_rate_1�<�7��I       6%�	a�F���A�*;


total_lossD{�@

error_R�
b?

learning_rate_1�<�7���wI       6%�	��F���A�*;


total_loss�G�@

error_R�oN?

learning_rate_1�<�7�6s�I       6%�	|	F���A�*;


total_loss�@

error_R�W?

learning_rate_1�<�7�+g�I       6%�	�B	F���A�*;


total_lossSg�@

error_R�MT?

learning_rate_1�<�7��SI       6%�	)�	F���A�*;


total_lossᡮ@

error_R%2_?

learning_rate_1�<�7*�v�I       6%�	��	F���A�*;


total_loss���@

error_R�=?

learning_rate_1�<�7_�I       6%�	�-
F���A�*;


total_lossSH�@

error_R�J?

learning_rate_1�<�7��EmI       6%�	Nn
F���A�*;


total_loss��n@

error_R�c?

learning_rate_1�<�7�`4�I       6%�	�
F���A�*;


total_lossw�@

error_R�0Q?

learning_rate_1�<�7�WHtI       6%�	�$F���A�*;


total_loss�]�@

error_R�S?

learning_rate_1�<�7���I       6%�	�iF���A�*;


total_loss��@

error_R@�Y?

learning_rate_1�<�7W�ˢI       6%�	Q�F���A�*;


total_lossZ�@

error_R~\?

learning_rate_1�<�7�S��I       6%�	�F���A�*;


total_loss�X�@

error_Rl0L?

learning_rate_1�<�7m��#I       6%�	�lF���A�*;


total_loss5�A

error_R�c?

learning_rate_1�<�7�? �I       6%�	��F���A�*;


total_loss��@

error_R,EJ?

learning_rate_1�<�7�ƖPI       6%�	��F���A�*;


total_loss�
�@

error_RjIM?

learning_rate_1�<�7�ϽI       6%�	�bF���A�*;


total_lossĒ�@

error_R�0C?

learning_rate_1�<�7�46�I       6%�	�F���A�*;


total_losseF�@

error_R�2O?

learning_rate_1�<�7_��HI       6%�	��F���A�*;


total_lossȆ@

error_R��H?

learning_rate_1�<�7c	0�I       6%�	�5F���A�*;


total_loss� �@

error_R�LU?

learning_rate_1�<�7u��I       6%�	
vF���A�*;


total_loss�h\@

error_R��J?

learning_rate_1�<�7�0�HI       6%�	�F���A�*;


total_loss�m�@

error_R.`?

learning_rate_1�<�7,8��I       6%�	�F���A�*;


total_lossX�@

error_RG?

learning_rate_1�<�7�5)I       6%�	m\F���A�*;


total_lossh4�@

error_RE?

learning_rate_1�<�7�jI       6%�	؜F���A�*;


total_lossp�@

error_R}AZ?

learning_rate_1�<�7n�O�I       6%�	��F���A�*;


total_loss��@

error_R��R?

learning_rate_1�<�7[1� I       6%�	�F���A�*;


total_lossڔ�@

error_RM�O?

learning_rate_1�<�7�\��I       6%�	,_F���A�*;


total_loss�3�@

error_R��c?

learning_rate_1�<�7���I       6%�	&�F���A�*;


total_loss�|A

error_R�Z?

learning_rate_1�<�7��&�I       6%�	5�F���A�*;


total_lossp��@

error_R��R?

learning_rate_1�<�7�{�I       6%�	&F���A�*;


total_losstD�@

error_R��@?

learning_rate_1�<�7˹��I       6%�	,nF���A�*;


total_loss�@

error_Rju<?

learning_rate_1�<�7L��I       6%�	��F���A�*;


total_loss�M�@

error_R�L?

learning_rate_1�<�7p�I       6%�	�F���A�*;


total_loss��@

error_R��I?

learning_rate_1�<�7R�I       6%�	�:F���A�*;


total_loss�х@

error_R��K?

learning_rate_1�<�74���I       6%�	�F���A�*;


total_lossܽ�@

error_R�cN?

learning_rate_1�<�7�WycI       6%�	S�F���A�*;


total_loss��A

error_R=e??

learning_rate_1�<�7��wI       6%�	�1F���A�*;


total_lossXΧ@

error_R��S?

learning_rate_1�<�7$��I       6%�	tF���A�*;


total_lossJo�@

error_R�&I?

learning_rate_1�<�7�zcI       6%�	�F���A�*;


total_loss̜�@

error_R��N?

learning_rate_1�<�7��=;I       6%�	s�F���A�*;


total_lossHÀ@

error_R8�W?

learning_rate_1�<�7�o�II       6%�	�AF���A�*;


total_losssy�@

error_R�HJ?

learning_rate_1�<�7�[�DI       6%�	�F���A�*;


total_loss=�@

error_R�x=?

learning_rate_1�<�7��˽I       6%�	��F���A�*;


total_losst�@

error_Rq�C?

learning_rate_1�<�7gܾ�I       6%�	/F���A�*;


total_loss^A

error_R�'^?

learning_rate_1�<�7�-*EI       6%�	�UF���A�*;


total_lossC��@

error_R]�Q?

learning_rate_1�<�7!d�~I       6%�	t�F���A�*;


total_loss���@

error_R�uH?

learning_rate_1�<�76�7�I       6%�	<�F���A�*;


total_loss�A

error_R�TA?

learning_rate_1�<�7�#>rI       6%�	�!F���A�*;


total_loss$ߠ@

error_R1�:?

learning_rate_1�<�7N�dI       6%�	0bF���A�*;


total_loss��@

error_R�[?

learning_rate_1�<�7ۺnI       6%�	��F���A�*;


total_loss��@

error_R��G?

learning_rate_1�<�7�]
WI       6%�	��F���A�*;


total_lossvw�@

error_R	�Y?

learning_rate_1�<�7ʼX�I       6%�	$.F���A�*;


total_lossq"�@

error_RMH?

learning_rate_1�<�7d��I       6%�	�rF���A�*;


total_loss(�r@

error_R*a??

learning_rate_1�<�7{���I       6%�	,�F���A�*;


total_loss�"�@

error_R��T?

learning_rate_1�<�76w�I       6%�	#F���A�*;


total_lossf/�@

error_R�Y?

learning_rate_1�<�7�ю?I       6%�	IbF���A�*;


total_loss��@

error_R�7E?

learning_rate_1�<�7�]_I       6%�	��F���A�*;


total_losssþ@

error_R��l?

learning_rate_1�<�7_A��I       6%�	�F���A� *;


total_loss���@

error_R�#E?

learning_rate_1�<�7*w(I       6%�	'F���A� *;


total_loss�%�@

error_Rd�6?

learning_rate_1�<�7�b#I       6%�	eF���A� *;


total_loss.��@

error_R};R?

learning_rate_1�<�7��||I       6%�	:�F���A� *;


total_loss)��@

error_R$�O?

learning_rate_1�<�7]�	I       6%�	��F���A� *;


total_loss!$�@

error_RDLL?

learning_rate_1�<�7u�I�I       6%�	�"F���A� *;


total_loss�1�@

error_R��]?

learning_rate_1�<�7�3 �I       6%�	�aF���A� *;


total_loss���@

error_R1nB?

learning_rate_1�<�7�t	I       6%�	�F���A� *;


total_loss�Y�@

error_R�Q?

learning_rate_1�<�7b8I       6%�	�F���A� *;


total_lossO4�@

error_R��Q?

learning_rate_1�<�7��I       6%�	X#F���A� *;


total_loss�ߟ@

error_Rf"@?

learning_rate_1�<�7S,�I       6%�	�aF���A� *;


total_loss���@

error_R�P?

learning_rate_1�<�7��?I       6%�	��F���A� *;


total_losse�@

error_R�dR?

learning_rate_1�<�75c��I       6%�	A�F���A� *;


total_lossҹ@

error_R�![?

learning_rate_1�<�7��I       6%�	^"F���A� *;


total_lossJ\�@

error_R��P?

learning_rate_1�<�7mf�3I       6%�	(cF���A� *;


total_loss�@

error_R�pH?

learning_rate_1�<�7�s7�I       6%�	�F���A� *;


total_loss�^�@

error_R;;]?

learning_rate_1�<�73��	I       6%�	�F���A� *;


total_loss��@

error_R�N?

learning_rate_1�<�7�ÙmI       6%�	x/F���A� *;


total_loss�ڊ@

error_R�N?

learning_rate_1�<�7���1I       6%�	�wF���A� *;


total_lossa�@

error_R}�K?

learning_rate_1�<�7�:F�I       6%�	;�F���A� *;


total_loss�H�@

error_RH+X?

learning_rate_1�<�7�e�cI       6%�	��F���A� *;


total_loss$-QA

error_R�F?

learning_rate_1�<�7݉�HI       6%�	�=F���A� *;


total_loss;׌@

error_R�`Q?

learning_rate_1�<�7��6I       6%�	�~F���A� *;


total_loss)�l@

error_R��N?

learning_rate_1�<�7قp�I       6%�	�F���A� *;


total_loss��@

error_R��G?

learning_rate_1�<�7}i��I       6%�	dF���A� *;


total_loss��@

error_R�@?

learning_rate_1�<�7��,EI       6%�	�FF���A� *;


total_loss�1�@

error_RZ\4?

learning_rate_1�<�7�L�I       6%�	�F���A� *;


total_loss(M�@

error_R�F?

learning_rate_1�<�7�f)I       6%�	a�F���A� *;


total_lossv_�@

error_R`�6?

learning_rate_1�<�7*C�I       6%�	? F���A� *;


total_lossR?�@

error_Rn�T?

learning_rate_1�<�7�A�VI       6%�	I F���A� *;


total_lossE�@

error_R|<?

learning_rate_1�<�7��{�I       6%�	� F���A� *;


total_lossET�@

error_R��V?

learning_rate_1�<�7wC��I       6%�	�� F���A� *;


total_loss�M�@

error_Rx�>?

learning_rate_1�<�7Q���I       6%�	+!F���A� *;


total_loss@d�@

error_R\�E?

learning_rate_1�<�7�KI"I       6%�	�U!F���A� *;


total_loss���@

error_Rlm[?

learning_rate_1�<�7����I       6%�	�!F���A� *;


total_loss�	�@

error_R��M?

learning_rate_1�<�7s�I       6%�	x�!F���A� *;


total_lossO��@

error_R�V;?

learning_rate_1�<�7!�7OI       6%�	�"F���A� *;


total_loss\�@

error_RS�T?

learning_rate_1�<�7��P�I       6%�	`"F���A� *;


total_loss��@

error_R��[?

learning_rate_1�<�7���I       6%�	p�"F���A� *;


total_loss���@

error_Ra[l?

learning_rate_1�<�7���I       6%�	��"F���A� *;


total_loss���@

error_RذS?

learning_rate_1�<�7�՜I       6%�	�#F���A� *;


total_lossՃ@

error_R�F?

learning_rate_1�<�7�MI       6%�	�e#F���A� *;


total_lossص�@

error_R��U?

learning_rate_1�<�7;N2I       6%�	��#F���A� *;


total_loss�.A

error_R3^Z?

learning_rate_1�<�7�~q�I       6%�	��#F���A� *;


total_losse�@

error_R��L?

learning_rate_1�<�7�䵷I       6%�	:+$F���A� *;


total_loss���@

error_R�mJ?

learning_rate_1�<�7�y�7I       6%�	�j$F���A� *;


total_loss-�@

error_R�2U?

learning_rate_1�<�7E4�oI       6%�	�$F���A� *;


total_loss�S@

error_RO�@?

learning_rate_1�<�7�Y��I       6%�	}�$F���A� *;


total_loss���@

error_R�.]?

learning_rate_1�<�7�R�I       6%�	V'%F���A� *;


total_lossꈼ@

error_R�++?

learning_rate_1�<�7�x=�I       6%�	�g%F���A� *;


total_lossm΂@

error_R��A?

learning_rate_1�<�7����I       6%�	�%F���A� *;


total_loss_l�@

error_Rc�b?

learning_rate_1�<�7F(�VI       6%�	[�%F���A� *;


total_loss���@

error_R-�S?

learning_rate_1�<�7�]�I       6%�	c2&F���A� *;


total_lossgB�@

error_R��O?

learning_rate_1�<�7����I       6%�	Gt&F���A� *;


total_lossw��@

error_R|�C?

learning_rate_1�<�7^0AI       6%�	:�&F���A� *;


total_loss.��@

error_R{O?

learning_rate_1�<�7��b�I       6%�	m�&F���A� *;


total_loss���@

error_R�|C?

learning_rate_1�<�7<}wI       6%�	Z8'F���A� *;


total_loss��@

error_R��\?

learning_rate_1�<�7����I       6%�	�w'F���A� *;


total_loss�T�@

error_R��U?

learning_rate_1�<�7to+I       6%�	��'F���A� *;


total_lossN�^@

error_R��S?

learning_rate_1�<�7�g+MI       6%�	k(F���A� *;


total_loss��@

error_R��O?

learning_rate_1�<�7S�-�I       6%�	xY(F���A� *;


total_loss<j�@

error_R;�N?

learning_rate_1�<�7/��I       6%�	ٚ(F���A� *;


total_loss�P�@

error_R�cP?

learning_rate_1�<�7��I       6%�	��(F���A� *;


total_loss_6�@

error_R��M?

learning_rate_1�<�7��I       6%�	.)F���A� *;


total_loss�e�@

error_RM{[?

learning_rate_1�<�7c���I       6%�	z[)F���A� *;


total_lossA#�@

error_Rx�??

learning_rate_1�<�7mީ�I       6%�	��)F���A� *;


total_loss�y�@

error_R#V?

learning_rate_1�<�7H�hI       6%�	��)F���A� *;


total_loss�}�@

error_R_�^?

learning_rate_1�<�7�b��I       6%�	�*F���A� *;


total_loss�=�@

error_R�M?

learning_rate_1�<�7j��I       6%�	�b*F���A� *;


total_loss�3�@

error_R�3W?

learning_rate_1�<�7�FI       6%�	<�*F���A� *;


total_loss�Q�@

error_R�Z?

learning_rate_1�<�7���I       6%�	��*F���A� *;


total_loss��A

error_RƞW?

learning_rate_1�<�7�s�AI       6%�	k'+F���A� *;


total_loss9�@

error_R�P?

learning_rate_1�<�7��3;I       6%�	 m+F���A� *;


total_loss�a�@

error_R��W?

learning_rate_1�<�7� �I       6%�	N�+F���A� *;


total_loss�Έ@

error_R�G?

learning_rate_1�<�7�M��I       6%�	��+F���A� *;


total_loss׀�@

error_R��H?

learning_rate_1�<�7���I       6%�	�A,F���A� *;


total_lossnQ�@

error_Ri.H?

learning_rate_1�<�7�:=?I       6%�	+�,F���A� *;


total_lossBf�@

error_R_�O?

learning_rate_1�<�7I���I       6%�	��,F���A� *;


total_loss@

error_R��]?

learning_rate_1�<�7���I       6%�	�
-F���A� *;


total_loss�7A

error_R �W?

learning_rate_1�<�7�S�I       6%�	IL-F���A� *;


total_loss��@

error_R��V?

learning_rate_1�<�7㲝 I       6%�	_�-F���A� *;


total_lossU
A

error_R�bV?

learning_rate_1�<�7��}�I       6%�	��-F���A� *;


total_loss;�Q@

error_R�H?

learning_rate_1�<�7!��EI       6%�	^.F���A� *;


total_lossM&�@

error_R\:?

learning_rate_1�<�7)��I       6%�	AV.F���A� *;


total_loss�@

error_RaN?

learning_rate_1�<�7Ɔ�I       6%�	%�.F���A� *;


total_loss���@

error_R�!Z?

learning_rate_1�<�7~�J�I       6%�	A�.F���A� *;


total_lossR0�@

error_R=sb?

learning_rate_1�<�7cliI       6%�	�!/F���A� *;


total_loss��@

error_RqGA?

learning_rate_1�<�7���@I       6%�		b/F���A� *;


total_loss�"�@

error_RԀL?

learning_rate_1�<�7��}I       6%�	��/F���A� *;


total_loss\��@

error_R��S?

learning_rate_1�<�7wΞ`I       6%�	d�/F���A� *;


total_losscB�@

error_R*�R?

learning_rate_1�<�7��I       6%�	� 0F���A� *;


total_loss�ޣ@

error_R@�H?

learning_rate_1�<�7/�	I       6%�	wa0F���A� *;


total_loss�9�@

error_Rv�B?

learning_rate_1�<�7?�h�I       6%�	E�0F���A� *;


total_loss���@

error_R�
N?

learning_rate_1�<�7\�NSI       6%�	��0F���A� *;


total_loss��@

error_Rx5C?

learning_rate_1�<�7��I       6%�	�,1F���A� *;


total_loss�(�@

error_RnT?

learning_rate_1�<�7����I       6%�	Rn1F���A� *;


total_lossw��@

error_R%�^?

learning_rate_1�<�7�$��I       6%�	5�1F���A� *;


total_loss���@

error_RZ�Q?

learning_rate_1�<�7���I       6%�	{�1F���A� *;


total_loss�I�@

error_RZ6K?

learning_rate_1�<�7�m�I       6%�	22F���A� *;


total_loss���@

error_R=�X?

learning_rate_1�<�7IULKI       6%�	�x2F���A� *;


total_lossyх@

error_R�cQ?

learning_rate_1�<�7����I       6%�	R�2F���A� *;


total_lossc��@

error_RƼb?

learning_rate_1�<�7y�JI       6%�	�3F���A� *;


total_lossD&�@

error_R��R?

learning_rate_1�<�7sA�I       6%�	%P3F���A� *;


total_loss�Ҫ@

error_R��H?

learning_rate_1�<�7���I       6%�	�3F���A� *;


total_loss���@

error_R
�7?

learning_rate_1�<�7.���I       6%�	[�3F���A� *;


total_loss���@

error_RDdE?

learning_rate_1�<�7I���I       6%�	�64F���A� *;


total_loss�?u@

error_R��G?

learning_rate_1�<�7Y��$I       6%�	��4F���A� *;


total_loss��@

error_Rx�T?

learning_rate_1�<�7D���I       6%�	��4F���A� *;


total_loss�b�@

error_R�^N?

learning_rate_1�<�7)�wI       6%�	�5F���A� *;


total_loss|�@

error_R�pV?

learning_rate_1�<�7{bI       6%�	�G5F���A� *;


total_loss�õ@

error_RA�I?

learning_rate_1�<�7 h>I       6%�	��5F���A� *;


total_loss�۵@

error_R
�K?

learning_rate_1�<�7�5G�I       6%�	��5F���A� *;


total_lossi�@

error_R��R?

learning_rate_1�<�7�`s�I       6%�	�6F���A� *;


total_lossñd@

error_R@?

learning_rate_1�<�7_@��I       6%�	�]6F���A� *;


total_lossH~A

error_R�DY?

learning_rate_1�<�7~�u�I       6%�	 6F���A� *;


total_loss���@

error_R\�X?

learning_rate_1�<�7�kO�I       6%�	��6F���A� *;


total_lossT��@

error_R�*6?

learning_rate_1�<�7���I       6%�	%7F���A� *;


total_loss���@

error_R%�K?

learning_rate_1�<�7V�I       6%�	�k7F���A� *;


total_loss���@

error_R��M?

learning_rate_1�<�7�MrgI       6%�	^�7F���A� *;


total_lossw�@

error_R7�K?

learning_rate_1�<�7ӜI       6%�	8F���A� *;


total_loss/�@

error_R��V?

learning_rate_1�<�7�W�JI       6%�	,L8F���A� *;


total_loss�P�@

error_R��_?

learning_rate_1�<�7�akI       6%�	��8F���A� *;


total_loss2��@

error_R,T?

learning_rate_1�<�7�� I       6%�	��8F���A� *;


total_loss���@

error_R�tS?

learning_rate_1�<�7V��I       6%�	�9F���A� *;


total_lossԭ�@

error_R�R?

learning_rate_1�<�7d-�II       6%�	N9F���A� *;


total_loss|��@

error_RH�T?

learning_rate_1�<�7��HI       6%�	��9F���A� *;


total_loss���@

error_RN	J?

learning_rate_1�<�7�)II       6%�	�9F���A� *;


total_loss*��@

error_RtS?

learning_rate_1�<�7#�ǲI       6%�	�:F���A� *;


total_lossS��@

error_R=T?

learning_rate_1�<�7�b�NI       6%�	�N:F���A�!*;


total_lossc@

error_Rq�R?

learning_rate_1�<�7�7I       6%�	��:F���A�!*;


total_loss�A�@

error_R2?

learning_rate_1�<�7$�}�I       6%�	*�:F���A�!*;


total_loss%��@

error_R�M?

learning_rate_1�<�7�T�I       6%�	Q;F���A�!*;


total_loss��@

error_RL�F?

learning_rate_1�<�7P��I       6%�	�T;F���A�!*;


total_lossZ��@

error_R@�F?

learning_rate_1�<�7�{�=I       6%�	9�;F���A�!*;


total_loss8�@

error_RԳ^?

learning_rate_1�<�7�O4�I       6%�	��;F���A�!*;


total_loss8��@

error_R�kL?

learning_rate_1�<�7��P�I       6%�	�<F���A�!*;


total_loss��@

error_R��K?

learning_rate_1�<�7�I       6%�	�T<F���A�!*;


total_loss�q�@

error_Re�V?

learning_rate_1�<�7�
X�I       6%�	k�<F���A�!*;


total_loss��@

error_RCP?

learning_rate_1�<�7W&rOI       6%�	l�<F���A�!*;


total_lossa�A

error_R��a?

learning_rate_1�<�7��I       6%�	�=F���A�!*;


total_loss�צ@

error_RN?

learning_rate_1�<�7��k`I       6%�	�]=F���A�!*;


total_loss��@

error_R��T?

learning_rate_1�<�7�f�I       6%�	��=F���A�!*;


total_loss�v�@

error_RM�X?

learning_rate_1�<�7��SI       6%�	0�=F���A�!*;


total_loss�ށ@

error_R��>?

learning_rate_1�<�7Q��I       6%�	�>F���A�!*;


total_lossZ��@

error_R�A?

learning_rate_1�<�7�:�I       6%�	�^>F���A�!*;


total_loss:$�@

error_R\�Y?

learning_rate_1�<�7A�yI       6%�	�>F���A�!*;


total_lossS�@

error_R�OS?

learning_rate_1�<�7�B��I       6%�	�>F���A�!*;


total_loss���@

error_Rf�b?

learning_rate_1�<�7��L�I       6%�	�??F���A�!*;


total_loss���@

error_R�YR?

learning_rate_1�<�7��I       6%�	��?F���A�!*;


total_loss�*�@

error_R��C?

learning_rate_1�<�7�7.I       6%�	�?F���A�!*;


total_loss�ף@

error_R��>?

learning_rate_1�<�7oO �I       6%�	v@F���A�!*;


total_loss���@

error_R�E?

learning_rate_1�<�7;�|kI       6%�	Jd@F���A�!*;


total_loss���@

error_Rii^?

learning_rate_1�<�7��<xI       6%�	S�@F���A�!*;


total_loss�@

error_R��G?

learning_rate_1�<�7i�I       6%�	\�@F���A�!*;


total_loss�;A

error_R�O?

learning_rate_1�<�7��I       6%�	�7AF���A�!*;


total_loss��@

error_R�\I?

learning_rate_1�<�7��fI       6%�	��AF���A�!*;


total_loss���@

error_R�XI?

learning_rate_1�<�7�iI       6%�	��AF���A�!*;


total_loss�@

error_R��X?

learning_rate_1�<�7�$0I       6%�	�BF���A�!*;


total_lossF��@

error_R\�X?

learning_rate_1�<�7Cm�I       6%�	>OBF���A�!*;


total_loss7��@

error_R��U?

learning_rate_1�<�7W�j(I       6%�	X�BF���A�!*;


total_lossJ��@

error_R�T?

learning_rate_1�<�7��I       6%�	h�BF���A�!*;


total_loss��@

error_R�cP?

learning_rate_1�<�7�L��I       6%�	lCF���A�!*;


total_loss���@

error_R��6?

learning_rate_1�<�7�T�uI       6%�	�ZCF���A�!*;


total_loss���@

error_R�qS?

learning_rate_1�<�7ԳpI       6%�	H�CF���A�!*;


total_loss�Ԥ@

error_R��L?

learning_rate_1�<�7��-I       6%�	$�CF���A�!*;


total_loss�c�@

error_R��H?

learning_rate_1�<�7a��I       6%�	�DF���A�!*;


total_lossR��@

error_R�h?

learning_rate_1�<�7e��I       6%�	]DF���A�!*;


total_loss���@

error_R�X?

learning_rate_1�<�7��w�I       6%�	��DF���A�!*;


total_lossF�@

error_R�rZ?

learning_rate_1�<�7N0�hI       6%�	m�DF���A�!*;


total_loss�C�@

error_R��G?

learning_rate_1�<�7��e�I       6%�	�EF���A�!*;


total_loss�RA

error_R�9J?

learning_rate_1�<�7	�KI       6%�	_EF���A�!*;


total_loss,��@

error_R�>?

learning_rate_1�<�7J�I       6%�	��EF���A�!*;


total_loss6��@

error_R��9?

learning_rate_1�<�7-E@I       6%�	S�EF���A�!*;


total_lossm��@

error_RGa?

learning_rate_1�<�7��hOI       6%�	#"FF���A�!*;


total_loss짻@

error_RitT?

learning_rate_1�<�7��t�I       6%�	aFF���A�!*;


total_loss�f�@

error_Rڎf?

learning_rate_1�<�7|� �I       6%�	��FF���A�!*;


total_loss���@

error_Rj�c?

learning_rate_1�<�74�<I       6%�	z�FF���A�!*;


total_loss�a�@

error_R��M?

learning_rate_1�<�7�#ҳI       6%�	� GF���A�!*;


total_loss3��@

error_RkI?

learning_rate_1�<�7X�'�I       6%�	cGF���A�!*;


total_loss��@

error_R�]O?

learning_rate_1�<�7�h�I       6%�	ܯGF���A�!*;


total_loss���@

error_R6�n?

learning_rate_1�<�7d,§I       6%�	�HF���A�!*;


total_losspA

error_R��L?

learning_rate_1�<�7v��I       6%�	�THF���A�!*;


total_loss��@

error_R�vR?

learning_rate_1�<�7�K4�I       6%�	x�HF���A�!*;


total_loss�b�@

error_R�z0?

learning_rate_1�<�7�ʡI       6%�	��HF���A�!*;


total_loss��@

error_R�O?

learning_rate_1�<�7�� |I       6%�	�IF���A�!*;


total_loss<�@

error_RR�U?

learning_rate_1�<�7�dxTI       6%�	�_IF���A�!*;


total_loss%��@

error_Rv�J?

learning_rate_1�<�7�nx�I       6%�	�IF���A�!*;


total_loss�u@

error_R�<<?

learning_rate_1�<�7���I       6%�	��IF���A�!*;


total_loss��@

error_R	�[?

learning_rate_1�<�7ȶPI       6%�	�*JF���A�!*;


total_loss/g�@

error_R|V8?

learning_rate_1�<�7.�WI       6%�	�mJF���A�!*;


total_loss�Y�@

error_R�yJ?

learning_rate_1�<�7qPeI       6%�	F�JF���A�!*;


total_lossQ��@

error_RK7?

learning_rate_1�<�7�+�I       6%�	��JF���A�!*;


total_loss&0�@

error_RW%U?

learning_rate_1�<�7²�oI       6%�	�<KF���A�!*;


total_loss$d�@

error_RN�A?

learning_rate_1�<�7���I       6%�	\�KF���A�!*;


total_loss�*�@

error_R�pV?

learning_rate_1�<�7}��0I       6%�	_�KF���A�!*;


total_lossUa�@

error_R�c?

learning_rate_1�<�7�0a5I       6%�	LLF���A�!*;


total_lossD1�@

error_R�A?

learning_rate_1�<�7��I       6%�	�QLF���A�!*;


total_loss�A

error_R��U?

learning_rate_1�<�7�+��I       6%�	��LF���A�!*;


total_loss	��@

error_R1�_?

learning_rate_1�<�7W*
�I       6%�	��LF���A�!*;


total_loss#(�@

error_R)�Y?

learning_rate_1�<�7g��YI       6%�	~MF���A�!*;


total_lossר�@

error_R;<h?

learning_rate_1�<�7�(zI       6%�	�_MF���A�!*;


total_lossqoA

error_RM~T?

learning_rate_1�<�7�e�I       6%�	�MF���A�!*;


total_loss�x�@

error_R�;T?

learning_rate_1�<�7e�0{I       6%�	z�MF���A�!*;


total_loss�U�@

error_R�pQ?

learning_rate_1�<�7d�e�I       6%�	�&NF���A�!*;


total_losswX%A

error_RDd?

learning_rate_1�<�72���I       6%�	�fNF���A�!*;


total_loss��@

error_R2}[?

learning_rate_1�<�7te;I       6%�	��NF���A�!*;


total_loss%��@

error_R� P?

learning_rate_1�<�7�_wI       6%�	(�NF���A�!*;


total_lossFR�@

error_RiO@?

learning_rate_1�<�7Xo1�I       6%�	2OF���A�!*;


total_loss:p�@

error_R\q^?

learning_rate_1�<�7r��I       6%�	�uOF���A�!*;


total_lossݳ@

error_R�_?

learning_rate_1�<�7��lXI       6%�	z�OF���A�!*;


total_loss\�A

error_R�TH?

learning_rate_1�<�7���mI       6%�	�PF���A�!*;


total_lossz��@

error_R��R?

learning_rate_1�<�7�J%8I       6%�	yDPF���A�!*;


total_loss���@

error_R�^?

learning_rate_1�<�7B3��I       6%�	y�PF���A�!*;


total_lossSA�@

error_R&UW?

learning_rate_1�<�79^�uI       6%�	��PF���A�!*;


total_lossύ�@

error_Re\?

learning_rate_1�<�7�АI       6%�	�QF���A�!*;


total_lossO��@

error_R�R?

learning_rate_1�<�7�g�&I       6%�	`\QF���A�!*;


total_loss� A

error_R��V?

learning_rate_1�<�7Z�tI       6%�	)�QF���A�!*;


total_loss49�@

error_R�WR?

learning_rate_1�<�71]��I       6%�	�QF���A�!*;


total_loss���@

error_R)�J?

learning_rate_1�<�7�&�I       6%�	PRF���A�!*;


total_lossJ��@

error_R�cE?

learning_rate_1�<�7��V�I       6%�	_RF���A�!*;


total_loss0�@

error_R,�_?

learning_rate_1�<�7BiI       6%�	��RF���A�!*;


total_loss2�@

error_Rf�S?

learning_rate_1�<�7�)��I       6%�	&�RF���A�!*;


total_loss�"�@

error_R�$D?

learning_rate_1�<�7@�g"I       6%�	A$SF���A�!*;


total_loss5�@

error_RaO?

learning_rate_1�<�7�-��I       6%�	�fSF���A�!*;


total_loss{�r@

error_R,�N?

learning_rate_1�<�7+h�MI       6%�	��SF���A�!*;


total_loss��@

error_R3�K?

learning_rate_1�<�7UJ�-I       6%�	�TF���A�!*;


total_loss,��@

error_R�K?

learning_rate_1�<�7� ��I       6%�	�LTF���A�!*;


total_loss���@

error_R��a?

learning_rate_1�<�7��Q�I       6%�	&�TF���A�!*;


total_loss��@

error_R�N9?

learning_rate_1�<�7��I       6%�	q�TF���A�!*;


total_lossw��@

error_Ro�P?

learning_rate_1�<�7.߁�I       6%�	UF���A�!*;


total_loss�Q�@

error_R ;P?

learning_rate_1�<�7�D1I       6%�	iSUF���A�!*;


total_loss��@

error_R:H?

learning_rate_1�<�7V*�I       6%�	��UF���A�!*;


total_loss���@

error_ROA?

learning_rate_1�<�75)�I       6%�	��UF���A�!*;


total_loss�+�@

error_R��f?

learning_rate_1�<�7=U�I       6%�	VF���A�!*;


total_loss;��@

error_R�kI?

learning_rate_1�<�7���I       6%�	�\VF���A�!*;


total_lossJ�g@

error_R�DR?

learning_rate_1�<�7�]�I       6%�	��VF���A�!*;


total_loss�A

error_R�^D?

learning_rate_1�<�7q4L�I       6%�	��VF���A�!*;


total_loss4,�@

error_R�N?

learning_rate_1�<�7}�W�I       6%�	�WF���A�!*;


total_lossn��@

error_R�qN?

learning_rate_1�<�7І\�I       6%�	E_WF���A�!*;


total_loss� @

error_R:�[?

learning_rate_1�<�7p$�I       6%�	+�WF���A�!*;


total_loss�s�@

error_Ra�k?

learning_rate_1�<�7�Q�I       6%�	��WF���A�!*;


total_lossش@

error_R�]U?

learning_rate_1�<�7,@5I       6%�	CXF���A�!*;


total_loss�m�@

error_R7D?

learning_rate_1�<�7�QI       6%�	 �XF���A�!*;


total_loss�a�@

error_R��T?

learning_rate_1�<�7ka0�I       6%�	��XF���A�!*;


total_lossɮ�@

error_R�[?

learning_rate_1�<�7���I       6%�	LYF���A�!*;


total_loss��@

error_R�J\?

learning_rate_1�<�7�S{xI       6%�	rQYF���A�!*;


total_loss
�@

error_R�tG?

learning_rate_1�<�7���I       6%�	�YF���A�!*;


total_loss�Z�@

error_R��O?

learning_rate_1�<�7	��]I       6%�	�YF���A�!*;


total_loss�@

error_R85M?

learning_rate_1�<�7�d�I       6%�	ZF���A�!*;


total_loss��@

error_R�^^?

learning_rate_1�<�7�x9I       6%�	QZF���A�!*;


total_loss{��@

error_R�]?

learning_rate_1�<�7�H�I       6%�	(�ZF���A�!*;


total_loss���@

error_R�K?

learning_rate_1�<�7ǃߒI       6%�	��ZF���A�!*;


total_loss`0�@

error_R�jI?

learning_rate_1�<�7l���I       6%�	�[F���A�!*;


total_lossli�@

error_RCN?

learning_rate_1�<�7'���I       6%�	�][F���A�!*;


total_loss���@

error_R̐K?

learning_rate_1�<�7���I       6%�	Ƞ[F���A�!*;


total_loss��@

error_R�Bn?

learning_rate_1�<�78���I       6%�	�[F���A�!*;


total_loss}�@

error_Rtu\?

learning_rate_1�<�7}p��I       6%�	5$\F���A�"*;


total_lossF\�@

error_R��E?

learning_rate_1�<�7b���I       6%�	�b\F���A�"*;


total_lossR��@

error_R�<N?

learning_rate_1�<�7�'�I       6%�	��\F���A�"*;


total_loss���@

error_R�R?

learning_rate_1�<�7��sI       6%�	��\F���A�"*;


total_loss���@

error_R�'d?

learning_rate_1�<�789��I       6%�	�&]F���A�"*;


total_losstɴ@

error_Rh�n?

learning_rate_1�<�70�7I       6%�	�e]F���A�"*;


total_loss�p@

error_ROtM?

learning_rate_1�<�7.�`yI       6%�	��]F���A�"*;


total_lossN�A

error_R��P?

learning_rate_1�<�7���I       6%�	0�]F���A�"*;


total_loss���@

error_Rn�L?

learning_rate_1�<�7�1YzI       6%�	�'^F���A�"*;


total_loss܃@

error_R��A?

learning_rate_1�<�7rAN�I       6%�	&h^F���A�"*;


total_losshm�@

error_R�Z?

learning_rate_1�<�7@��I       6%�	~�^F���A�"*;


total_loss�{�@

error_R�O?

learning_rate_1�<�7�͜]I       6%�	�^F���A�"*;


total_loss�p�@

error_R6�M?

learning_rate_1�<�7�SI       6%�	�)_F���A�"*;


total_loss�$�@

error_R�V?

learning_rate_1�<�7�+nbI       6%�	��_F���A�"*;


total_lossi��@

error_RT�J?

learning_rate_1�<�7,%�I       6%�	*�_F���A�"*;


total_loss߀A

error_R�H?

learning_rate_1�<�7u�I       6%�	�M`F���A�"*;


total_loss/�@

error_R=�[?

learning_rate_1�<�7�&�I       6%�	i�`F���A�"*;


total_loss�D�@

error_R�?S?

learning_rate_1�<�7Z�I       6%�	��`F���A�"*;


total_loss��@

error_R��R?

learning_rate_1�<�7�ȖMI       6%�	�+aF���A�"*;


total_lossoݳ@

error_R�wf?

learning_rate_1�<�7�EI       6%�	oaF���A�"*;


total_loss��@

error_Rט3?

learning_rate_1�<�7�t��I       6%�	i�aF���A�"*;


total_loss䩱@

error_R�ZJ?

learning_rate_1�<�7�a�I       6%�	y�aF���A�"*;


total_lossS��@

error_R�vN?

learning_rate_1�<�7���I       6%�	�7bF���A�"*;


total_loss43�@

error_R$oC?

learning_rate_1�<�7�:�I       6%�	��bF���A�"*;


total_loss�&o@

error_RۨF?

learning_rate_1�<�7�3�I       6%�	x�bF���A�"*;


total_loss��@

error_R��<?

learning_rate_1�<�7���`I       6%�	�<cF���A�"*;


total_lossV8�@

error_R�	X?

learning_rate_1�<�7/���I       6%�	n�cF���A�"*;


total_loss8��@

error_R��Y?

learning_rate_1�<�7�ۼI       6%�	��cF���A�"*;


total_loss:�y@

error_R��R?

learning_rate_1�<�7�@��I       6%�	��cF���A�"*;


total_loss��@

error_R1�=?

learning_rate_1�<�7�(�I       6%�	�AdF���A�"*;


total_lossf�@

error_R1B?

learning_rate_1�<�7Z�8�I       6%�	O�dF���A�"*;


total_loss�1�@

error_R��Q?

learning_rate_1�<�7��I       6%�	��dF���A�"*;


total_loss]H�@

error_R��U?

learning_rate_1�<�7���I       6%�	eF���A�"*;


total_loss��@

error_R��A?

learning_rate_1�<�7���I       6%�	�DeF���A�"*;


total_loss��@

error_RR�E?

learning_rate_1�<�7��I       6%�	�eF���A�"*;


total_loss��@

error_Rr2L?

learning_rate_1�<�7�\)I       6%�	��eF���A�"*;


total_loss�@

error_RēN?

learning_rate_1�<�7NT7mI       6%�	fF���A�"*;


total_lossV�@

error_R�fU?

learning_rate_1�<�7�Ҫ�I       6%�	�JfF���A�"*;


total_lossc��@

error_R�L?

learning_rate_1�<�7!U(MI       6%�	ɈfF���A�"*;


total_loss=�@

error_R S?

learning_rate_1�<�7���]I       6%�	+�fF���A�"*;


total_loss��@

error_RE�@?

learning_rate_1�<�7��+�I       6%�	�gF���A�"*;


total_loss���@

error_R�a?

learning_rate_1�<�7�	��I       6%�	UIgF���A�"*;


total_loss8A

error_R�rV?

learning_rate_1�<�7ulq�I       6%�	&�gF���A�"*;


total_loss~	A

error_R��C?

learning_rate_1�<�7��]�I       6%�	��gF���A�"*;


total_loss.��@

error_R6�M?

learning_rate_1�<�7k��mI       6%�	�<hF���A�"*;


total_loss!$�@

error_R��;?

learning_rate_1�<�7��*	I       6%�	g�hF���A�"*;


total_loss��@

error_Rl!\?

learning_rate_1�<�7`�`�I       6%�	��hF���A�"*;


total_loss��@

error_RʩM?

learning_rate_1�<�7:՘EI       6%�	i
iF���A�"*;


total_lossz�@

error_R�Lc?

learning_rate_1�<�7<�>�I       6%�	mLiF���A�"*;


total_loss�D�@

error_R��U?

learning_rate_1�<�7���KI       6%�	ƒiF���A�"*;


total_lossl��@

error_R��J?

learning_rate_1�<�7�X�I       6%�	��iF���A�"*;


total_lossI�Q@

error_RhA;?

learning_rate_1�<�72��I       6%�	�jF���A�"*;


total_loss��@

error_R�@?

learning_rate_1�<�7�"I       6%�	RjF���A�"*;


total_lossߡ@

error_R��V?

learning_rate_1�<�7U�Z�I       6%�	��jF���A�"*;


total_loss���@

error_R�Pd?

learning_rate_1�<�7G�w"I       6%�	l�jF���A�"*;


total_lossߖ�@

error_R�\?

learning_rate_1�<�7���I       6%�	,kF���A�"*;


total_loss��@

error_RL�_?

learning_rate_1�<�7f_P�I       6%�	�VkF���A�"*;


total_loss���@

error_R(N?

learning_rate_1�<�72ӵI       6%�	��kF���A�"*;


total_loss*��@

error_Rm�N?

learning_rate_1�<�7����I       6%�	�kF���A�"*;


total_loss�X�@

error_R;>?

learning_rate_1�<�7EkR�I       6%�	�lF���A�"*;


total_loss舜@

error_RjQS?

learning_rate_1�<�7DC�I       6%�	Q`lF���A�"*;


total_loss���@

error_Rȭ>?

learning_rate_1�<�7֍5bI       6%�	,�lF���A�"*;


total_loss7aA

error_R�-R?

learning_rate_1�<�7��'I       6%�	mF���A�"*;


total_loss�d�@

error_R4S?

learning_rate_1�<�7���I       6%�	MmF���A�"*;


total_loss)1�@

error_R@?R?

learning_rate_1�<�7H�� I       6%�	��mF���A�"*;


total_loss��@

error_R,�^?

learning_rate_1�<�7��=�I       6%�	7�mF���A�"*;


total_loss�=�@

error_RHhA?

learning_rate_1�<�7�IɧI       6%�	_ nF���A�"*;


total_loss~�@

error_RLG?

learning_rate_1�<�7�J��I       6%�	dnF���A�"*;


total_lossJ,A

error_R��h?

learning_rate_1�<�7T�L�I       6%�	��nF���A�"*;


total_loss���@

error_R�[Q?

learning_rate_1�<�7u{#I       6%�	��nF���A�"*;


total_loss�Ȫ@

error_RD�i?

learning_rate_1�<�7�]iI       6%�	e6oF���A�"*;


total_loss$�L@

error_R��h?

learning_rate_1�<�7J�3I       6%�	]|oF���A�"*;


total_lossqhA

error_R�N?

learning_rate_1�<�7Q�/FI       6%�	��oF���A�"*;


total_loss*j�@

error_R�M?

learning_rate_1�<�7�Q^I       6%�	�pF���A�"*;


total_loss}�@

error_R�Z?

learning_rate_1�<�7`>�;I       6%�	�GpF���A�"*;


total_lossnB�@

error_R��C?

learning_rate_1�<�7j���I       6%�	v�pF���A�"*;


total_loss�آ@

error_R;�R?

learning_rate_1�<�7}	�I       6%�	 �pF���A�"*;


total_loss�@

error_R=�D?

learning_rate_1�<�7����I       6%�	qF���A�"*;


total_lossqo�@

error_R8ec?

learning_rate_1�<�7�גI       6%�	4YqF���A�"*;


total_loss*̏@

error_R�L?

learning_rate_1�<�7���;I       6%�	��qF���A�"*;


total_loss�@

error_R��T?

learning_rate_1�<�7�É�I       6%�	(�qF���A�"*;


total_lossǙ@

error_R{H?

learning_rate_1�<�7��O�I       6%�	l$rF���A�"*;


total_loss1��@

error_R�"D?

learning_rate_1�<�7�ePI       6%�	�grF���A�"*;


total_loss
a�@

error_R�]U?

learning_rate_1�<�7>�f�I       6%�	�rF���A�"*;


total_loss�A�@

error_RF�T?

learning_rate_1�<�7ڮ�rI       6%�	��rF���A�"*;


total_loss���@

error_RmJ?

learning_rate_1�<�7��7CI       6%�	o:sF���A�"*;


total_loss~�@

error_RH_G?

learning_rate_1�<�7�KI       6%�	{sF���A�"*;


total_loss%>�@

error_R�;?

learning_rate_1�<�7���I       6%�	@�sF���A�"*;


total_loss$ͮ@

error_RH}P?

learning_rate_1�<�7���I       6%�	�sF���A�"*;


total_loss[@�@

error_R��H?

learning_rate_1�<�7��]zI       6%�	�<tF���A�"*;


total_loss�k�@

error_R��J?

learning_rate_1�<�76��^I       6%�	l~tF���A�"*;


total_loss���@

error_R��B?

learning_rate_1�<�7��}I       6%�	��tF���A�"*;


total_loss���@

error_R��L?

learning_rate_1�<�7|�o�I       6%�	��tF���A�"*;


total_loss���@

error_R-�P?

learning_rate_1�<�7p�I       6%�	�?uF���A�"*;


total_loss���@

error_R��L?

learning_rate_1�<�7��*;I       6%�		~uF���A�"*;


total_lossS� A

error_R�D?

learning_rate_1�<�7Ю��I       6%�	<�uF���A�"*;


total_loss�w�@

error_R�^V?

learning_rate_1�<�7\Z�wI       6%�	(�uF���A�"*;


total_loss#~�@

error_RN�Q?

learning_rate_1�<�7'r�I       6%�	�=vF���A�"*;


total_lossq��@

error_R�"E?

learning_rate_1�<�7 ��KI       6%�	3}vF���A�"*;


total_loss��@

error_R+C?

learning_rate_1�<�7��:I       6%�	�vF���A�"*;


total_loss?�@

error_R>Q?

learning_rate_1�<�7p��	I       6%�	� wF���A�"*;


total_lossM�A

error_R_=B?

learning_rate_1�<�7k}RWI       6%�	�@wF���A�"*;


total_loss�U�@

error_R@�D?

learning_rate_1�<�7��]5I       6%�	��wF���A�"*;


total_loss���@

error_R�UR?

learning_rate_1�<�7�P�I       6%�	�wF���A�"*;


total_loss�}�@

error_R�=?

learning_rate_1�<�7_��I       6%�	J(xF���A�"*;


total_loss�u�@

error_R�L?

learning_rate_1�<�7�6��I       6%�	<rxF���A�"*;


total_lossCp�@

error_RES?

learning_rate_1�<�7� h�I       6%�	ҸxF���A�"*;


total_lossD7�@

error_R�Y^?

learning_rate_1�<�7uH�I       6%�	VyF���A�"*;


total_loss�bU@

error_R��6?

learning_rate_1�<�7���I       6%�	�CyF���A�"*;


total_lossL��@

error_R�Z?

learning_rate_1�<�7�y��I       6%�	�yF���A�"*;


total_loss؂�@

error_R@�H?

learning_rate_1�<�7%��%I       6%�	��yF���A�"*;


total_loss��@

error_R�3N?

learning_rate_1�<�7�xjqI       6%�	zF���A�"*;


total_loss8��@

error_RyN?

learning_rate_1�<�7_ia I       6%�	�RzF���A�"*;


total_lossM��@

error_R.�J?

learning_rate_1�<�75MI       6%�	&�zF���A�"*;


total_loss�I�@

error_R�	=?

learning_rate_1�<�7���;I       6%�	z�zF���A�"*;


total_loss�U�@

error_R�HY?

learning_rate_1�<�7~:�I       6%�	I{F���A�"*;


total_loss��@

error_R�K?

learning_rate_1�<�7���aI       6%�	�Y{F���A�"*;


total_lossT��@

error_R�B?

learning_rate_1�<�7%I       6%�	��{F���A�"*;


total_loss6��@

error_R�zL?

learning_rate_1�<�7n��I       6%�	��{F���A�"*;


total_loss��o@

error_R��b?

learning_rate_1�<�7�3��I       6%�	�|F���A�"*;


total_lossczA

error_R��V?

learning_rate_1�<�7`m]I       6%�	�Z|F���A�"*;


total_loss�׶@

error_R�@?

learning_rate_1�<�7��L�I       6%�	̙|F���A�"*;


total_lossE��@

error_R[<?

learning_rate_1�<�7���I       6%�	 �|F���A�"*;


total_loss=�@

error_R��Z?

learning_rate_1�<�7}`��I       6%�	t}F���A�"*;


total_loss�-�@

error_R�F?

learning_rate_1�<�71�=}I       6%�	Jc}F���A�"*;


total_loss)��@

error_R:�e?

learning_rate_1�<�7�#�I       6%�	s�}F���A�"*;


total_loss�2A

error_R� 9?

learning_rate_1�<�7��I       6%�	F�}F���A�"*;


total_loss�Ȝ@

error_RH	F?

learning_rate_1�<�7w�-GI       6%�	n8~F���A�"*;


total_lossX��@

error_R| 2?

learning_rate_1�<�7�|��I       6%�	��~F���A�#*;


total_loss�N�@

error_RRHH?

learning_rate_1�<�7b�WI       6%�	��~F���A�#*;


total_lossڒ@

error_RmU?

learning_rate_1�<�7�ϼ�I       6%�	k,F���A�#*;


total_loss�r�@

error_R�l_?

learning_rate_1�<�7o�/�I       6%�	�lF���A�#*;


total_losshr�@

error_R��Q?

learning_rate_1�<�7�n�I       6%�	P�F���A�#*;


total_lossL��@

error_R%�F?

learning_rate_1�<�7��s�I       6%�	��F���A�#*;


total_loss�	�@

error_RA�A?

learning_rate_1�<�72��I       6%�	�8�F���A�#*;


total_loss8 �@

error_Rd_@?

learning_rate_1�<�7NN{I       6%�	 ~�F���A�#*;


total_loss	��@

error_R$�D?

learning_rate_1�<�7]��7I       6%�	���F���A�#*;


total_loss�C�@

error_Rq#M?

learning_rate_1�<�7n��I       6%�	V�F���A�#*;


total_loss���@

error_R��e?

learning_rate_1�<�7#Js*I       6%�	�J�F���A�#*;


total_loss���@

error_R�,S?

learning_rate_1�<�7�RI       6%�	���F���A�#*;


total_lossԀ�@

error_R�$]?

learning_rate_1�<�7�iuI       6%�	�ҁF���A�#*;


total_lossۅ�@

error_Rt�^?

learning_rate_1�<�7&�'I       6%�	��F���A�#*;


total_loss:a�@

error_R�3I?

learning_rate_1�<�7����I       6%�	DW�F���A�#*;


total_lossa� A

error_R� Y?

learning_rate_1�<�7s�בI       6%�	%��F���A�#*;


total_lossi��@

error_R{�K?

learning_rate_1�<�7@<�kI       6%�	��F���A�#*;


total_lossq��@

error_RQW?

learning_rate_1�<�7\F�|I       6%�	=(�F���A�#*;


total_loss.��@

error_R��H?

learning_rate_1�<�7L��I       6%�	�g�F���A�#*;


total_loss��@

error_RŖC?

learning_rate_1�<�7td(gI       6%�	��F���A�#*;


total_loss�\�@

error_RixR?

learning_rate_1�<�7�p��I       6%�	\�F���A�#*;


total_loss@�@

error_R`3W?

learning_rate_1�<�7�t)I       6%�	�)�F���A�#*;


total_loss,c@

error_RͶN?

learning_rate_1�<�7�/g�I       6%�	h�F���A�#*;


total_lossȥ�@

error_R��X?

learning_rate_1�<�7���I       6%�	T��F���A�#*;


total_loss�u�@

error_Ra�F?

learning_rate_1�<�7��ɄI       6%�	*�F���A�#*;


total_lossG��@

error_R1]?

learning_rate_1�<�7A�I       6%�	�)�F���A�#*;


total_lossIh�@

error_R}�U?

learning_rate_1�<�7���I       6%�	Ui�F���A�#*;


total_lossI��@

error_R1Q<?

learning_rate_1�<�7���I       6%�	A��F���A�#*;


total_loss�,�@

error_R
�K?

learning_rate_1�<�7�g�I       6%�	Y�F���A�#*;


total_loss��@

error_RȷK?

learning_rate_1�<�7��:I       6%�	,5�F���A�#*;


total_loss�G�@

error_R��;?

learning_rate_1�<�7� I       6%�	�}�F���A�#*;


total_loss���@

error_R��W?

learning_rate_1�<�7�B�I       6%�	y��F���A�#*;


total_loss=6�@

error_R gH?

learning_rate_1�<�7��O�I       6%�	��F���A�#*;


total_loss�8Y@

error_RH@C?

learning_rate_1�<�7���I       6%�	Qu�F���A�#*;


total_loss(��@

error_R%�C?

learning_rate_1�<�7�-jqI       6%�	^ÇF���A�#*;


total_loss�
y@

error_R}�@?

learning_rate_1�<�7���oI       6%�	R#�F���A�#*;


total_lossE��@

error_R�IE?

learning_rate_1�<�7�I       6%�	�d�F���A�#*;


total_lossH��@

error_RE�S?

learning_rate_1�<�7erI       6%�	ۤ�F���A�#*;


total_loss3�@

error_R6W?

learning_rate_1�<�7%��I       6%�	��F���A�#*;


total_loss�4�@

error_R@RL?

learning_rate_1�<�77Dm@I       6%�	�,�F���A�#*;


total_loss[��@

error_R��P?

learning_rate_1�<�7��v]I       6%�	`�F���A�#*;


total_loss3�@

error_R��9?

learning_rate_1�<�7/��I       6%�	H�F���A�#*;


total_loss�Z�@

error_R��J?

learning_rate_1�<�7�
�I       6%�	�-�F���A�#*;


total_lossV�A

error_R�,`?

learning_rate_1�<�7x�?I       6%�	�p�F���A�#*;


total_loss܇�@

error_RFX?

learning_rate_1�<�7��έI       6%�	���F���A�#*;


total_loss14�@

error_R�HI?

learning_rate_1�<�7բ��I       6%�	�F���A�#*;


total_loss�o�@

error_R�EX?

learning_rate_1�<�7p+��I       6%�	Z6�F���A�#*;


total_loss�}�@

error_RT�C?

learning_rate_1�<�7&%I       6%�	?w�F���A�#*;


total_loss���@

error_Rł[?

learning_rate_1�<�7N)l�I       6%�	m��F���A�#*;


total_lossL��@

error_RxvP?

learning_rate_1�<�7,Ey�I       6%�	���F���A�#*;


total_loss�v�@

error_R��C?

learning_rate_1�<�7o�I       6%�	�A�F���A�#*;


total_loss��@

error_R�=?

learning_rate_1�<�7g���I       6%�	e��F���A�#*;


total_loss$b�@

error_R�L?

learning_rate_1�<�7Q�	I       6%�	WƌF���A�#*;


total_lossƩ@

error_R��L?

learning_rate_1�<�7��JI       6%�	9�F���A�#*;


total_loss�!A

error_R�F?

learning_rate_1�<�7��#�I       6%�	���F���A�#*;


total_loss�э@

error_R��Z?

learning_rate_1�<�7K�KI       6%�	6͍F���A�#*;


total_loss�6�@

error_R��Q?

learning_rate_1�<�7��*�I       6%�	�F���A�#*;


total_lossc[�@

error_R��N?

learning_rate_1�<�7�Y,�I       6%�	�w�F���A�#*;


total_loss���@

error_R6yc?

learning_rate_1�<�7&[4�I       6%�	�ŎF���A�#*;


total_lossqS�@

error_R��P?

learning_rate_1�<�7��I       6%�	��F���A�#*;


total_loss��@

error_RaS?

learning_rate_1�<�7����I       6%�	Z�F���A�#*;


total_loss���@

error_R*MY?

learning_rate_1�<�7�[KdI       6%�	�ÏF���A�#*;


total_loss�Ǳ@

error_RϵB?

learning_rate_1�<�7�J�wI       6%�	��F���A�#*;


total_lossa�@

error_R�4U?

learning_rate_1�<�7�"I       6%�	�X�F���A�#*;


total_losse�@

error_Rćf?

learning_rate_1�<�7C�V�I       6%�	E��F���A�#*;


total_loss�7�@

error_R=�\?

learning_rate_1�<�7@?I       6%�	� �F���A�#*;


total_loss��@

error_R�<H?

learning_rate_1�<�7��[�I       6%�	&J�F���A�#*;


total_loss ��@

error_R8S?

learning_rate_1�<�7�k~}I       6%�	���F���A�#*;


total_loss�W�@

error_RȴZ?

learning_rate_1�<�7�!q2I       6%�	kБF���A�#*;


total_lossp@

error_RnVR?

learning_rate_1�<�7on�I       6%�	�F���A�#*;


total_lossD�@

error_R�DJ?

learning_rate_1�<�7�hX�I       6%�	�b�F���A�#*;


total_loss�@

error_R��N?

learning_rate_1�<�7"r�I       6%�	ĩ�F���A�#*;


total_loss� A

error_R��@?

learning_rate_1�<�7��]�I       6%�	4�F���A�#*;


total_lossv��@

error_R)NG?

learning_rate_1�<�7���`I       6%�	j5�F���A�#*;


total_lossO �@

error_R�W?

learning_rate_1�<�7��I       6%�	֣�F���A�#*;


total_lossX��@

error_R�8M?

learning_rate_1�<�7�#VVI       6%�	��F���A�#*;


total_loss
��@

error_R��O?

learning_rate_1�<�7�@7�I       6%�	1�F���A�#*;


total_loss��{@

error_R� Q?

learning_rate_1�<�7l���I       6%�	�w�F���A�#*;


total_lossLX@

error_R}�^?

learning_rate_1�<�7���I       6%�	I��F���A�#*;


total_loss��@

error_R�>F?

learning_rate_1�<�7#�q�I       6%�	��F���A�#*;


total_loss4��@

error_RҙD?

learning_rate_1�<�7p5�I       6%�	�J�F���A�#*;


total_loss`��@

error_R{
O?

learning_rate_1�<�7	HQ I       6%�	1��F���A�#*;


total_lossRK�@

error_RHNR?

learning_rate_1�<�7����I       6%�	�ߕF���A�#*;


total_loss :�@

error_R��T?

learning_rate_1�<�7�׆I       6%�	}?�F���A�#*;


total_loss��@

error_R�B?

learning_rate_1�<�7�;1I       6%�	!��F���A�#*;


total_lossJ��@

error_RO�L?

learning_rate_1�<�7�VLI       6%�	6ƖF���A�#*;


total_loss
�M@

error_R�9?

learning_rate_1�<�7�@wkI       6%�	P�F���A�#*;


total_loss��@

error_RLZ?

learning_rate_1�<�7��I       6%�	�X�F���A�#*;


total_loss	`�@

error_R�d?

learning_rate_1�<�7l�@kI       6%�	��F���A�#*;


total_loss�@

error_R�^?

learning_rate_1�<�7H�>\I       6%�	��F���A�#*;


total_loss��@

error_R�W4?

learning_rate_1�<�7K16�I       6%�	�R�F���A�#*;


total_loss~9�@

error_R3�;?

learning_rate_1�<�7(��SI       6%�	의F���A�#*;


total_loss���@

error_RҥT?

learning_rate_1�<�7~�I       6%�	4�F���A�#*;


total_loss��@

error_R(Z?

learning_rate_1�<�7E�TI       6%�	�1�F���A�#*;


total_lossCr�@

error_RWK?

learning_rate_1�<�7@|�3I       6%�	�u�F���A�#*;


total_loss��@

error_R
�V?

learning_rate_1�<�7�{�I       6%�	��F���A�#*;


total_loss�B�@

error_R��N?

learning_rate_1�<�7�#��I       6%�	��F���A�#*;


total_loss!T�@

error_RxM?

learning_rate_1�<�7�Ҍ�I       6%�	�D�F���A�#*;


total_lossX��@

error_R�QN?

learning_rate_1�<�7�f�I       6%�	ֆ�F���A�#*;


total_lossX2�@

error_R`e?

learning_rate_1�<�7�b>I       6%�	ɚF���A�#*;


total_loss� A

error_R�]?

learning_rate_1�<�7�"�-I       6%�	�F���A�#*;


total_loss��@

error_R��D?

learning_rate_1�<�7ژ��I       6%�	�M�F���A�#*;


total_lossï�@

error_R�bj?

learning_rate_1�<�7Op��I       6%�	���F���A�#*;


total_loss�ڹ@

error_Rqb5?

learning_rate_1�<�7���I       6%�	�ћF���A�#*;


total_loss��@

error_R;�Y?

learning_rate_1�<�7����I       6%�	>�F���A�#*;


total_loss��@

error_R�]K?

learning_rate_1�<�7|�WkI       6%�	X�F���A�#*;


total_lossS��@

error_R��[?

learning_rate_1�<�7s�Y�I       6%�	J��F���A�#*;


total_lossD��@

error_R?�Z?

learning_rate_1�<�7�b��I       6%�	�ۜF���A�#*;


total_loss-�@

error_R�8?

learning_rate_1�<�7V��I       6%�	)$�F���A�#*;


total_lossD�@

error_R�z<?

learning_rate_1�<�7�;:�I       6%�	�h�F���A�#*;


total_loss��@

error_R��N?

learning_rate_1�<�7�?��I       6%�	,��F���A�#*;


total_lossC~�@

error_R��G?

learning_rate_1�<�7�A��I       6%�	�F���A�#*;


total_loss�
�@

error_Rf�C?

learning_rate_1�<�7�˃I       6%�	�>�F���A�#*;


total_loss��@

error_RXW?

learning_rate_1�<�7�mI       6%�	���F���A�#*;


total_loss���@

error_R� J?

learning_rate_1�<�7&�RI       6%�	��F���A�#*;


total_lossN�@

error_R �D?

learning_rate_1�<�7yj��I       6%�	�(�F���A�#*;


total_loss�3�@

error_Rw0O?

learning_rate_1�<�7CS�+I       6%�	�j�F���A�#*;


total_lossA��@

error_R�S?

learning_rate_1�<�7P��I       6%�	y��F���A�#*;


total_loss���@

error_R=8O?

learning_rate_1�<�7O7I       6%�	��F���A�#*;


total_loss���@

error_R�F?

learning_rate_1�<�7M��5I       6%�	�7�F���A�#*;


total_loss�Y�@

error_RjS?

learning_rate_1�<�7mk�~I       6%�	A~�F���A�#*;


total_loss�D�@

error_R� M?

learning_rate_1�<�7��I       6%�	(��F���A�#*;


total_loss:Y�@

error_R{ K?

learning_rate_1�<�7j�Y�I       6%�	�	�F���A�#*;


total_lossHC�@

error_R�8?

learning_rate_1�<�7�M��I       6%�	�R�F���A�#*;


total_loss�,�@

error_RJ?

learning_rate_1�<�7ޞ��I       6%�	���F���A�#*;


total_loss�)�@

error_R@<R?

learning_rate_1�<�7w츱I       6%�	�ݡF���A�#*;


total_loss2}�@

error_R:�U?

learning_rate_1�<�7����I       6%�	,�F���A�#*;


total_loss���@

error_R{�C?

learning_rate_1�<�7Tj�I       6%�	�v�F���A�#*;


total_loss���@

error_R�RJ?

learning_rate_1�<�79�_�I       6%�	&��F���A�$*;


total_loss��@

error_R�\?

learning_rate_1�<�7R�b/I       6%�	�F���A�$*;


total_lossHif@

error_R�A?

learning_rate_1�<�7z��I       6%�	LC�F���A�$*;


total_lossW޳@

error_R��_?

learning_rate_1�<�7�H��I       6%�	���F���A�$*;


total_loss?�@

error_R�<U?

learning_rate_1�<�7
�)�I       6%�	�ӣF���A�$*;


total_lossMc�@

error_R\S?

learning_rate_1�<�7����I       6%�	�F���A�$*;


total_loss�̾@

error_R�AU?

learning_rate_1�<�7z�L�I       6%�	�T�F���A�$*;


total_loss6N-A

error_R�tT?

learning_rate_1�<�7�wB(I       6%�	ė�F���A�$*;


total_loss��|@

error_R�J?

learning_rate_1�<�7��I       6%�	QۤF���A�$*;


total_loss� �@

error_Rf�a?

learning_rate_1�<�7��I       6%�	�;�F���A�$*;


total_loss_�@

error_RJB?

learning_rate_1�<�7��ZI       6%�	χ�F���A�$*;


total_loss{��@

error_R��M?

learning_rate_1�<�7p[َI       6%�	�̥F���A�$*;


total_loss��@

error_R�D?

learning_rate_1�<�7d��I       6%�	k�F���A�$*;


total_lossn��@

error_R�9D?

learning_rate_1�<�7�jI       6%�	�V�F���A�$*;


total_lossϞ�@

error_R9?

learning_rate_1�<�7|QI       6%�	���F���A�$*;


total_loss�D�@

error_R�4_?

learning_rate_1�<�7�q#uI       6%�	?�F���A�$*;


total_losso>�@

error_R:NN?

learning_rate_1�<�7�H�I       6%�	M0�F���A�$*;


total_loss<�@

error_RmR?

learning_rate_1�<�7�LclI       6%�	}�F���A�$*;


total_loss�
�@

error_RV�C?

learning_rate_1�<�7U���I       6%�	��F���A�$*;


total_loss�י@

error_R�F?

learning_rate_1�<�7|{��I       6%�	_t�F���A�$*;


total_lossr8�@

error_R�W?

learning_rate_1�<�7ʺ��I       6%�	���F���A�$*;


total_lossz+�@

error_RC?

learning_rate_1�<�7�k�I       6%�	��F���A�$*;


total_lossA

error_R�L?

learning_rate_1�<�7�2�1I       6%�	sq�F���A�$*;


total_loss���@

error_R�DB?

learning_rate_1�<�7�͟�I       6%�	���F���A�$*;


total_loss�	�@

error_R��T?

learning_rate_1�<�7}�[I       6%�	� �F���A�$*;


total_loss�\�@

error_R�NP?

learning_rate_1�<�7��X�I       6%�	�_�F���A�$*;


total_loss���@

error_REKR?

learning_rate_1�<�7��*#I       6%�	3��F���A�$*;


total_lossRӮ@

error_R,�Y?

learning_rate_1�<�7 �ӦI       6%�	/�F���A�$*;


total_loss<��@

error_RR�m?

learning_rate_1�<�7�~*I       6%�	�+�F���A�$*;


total_lossw��@

error_R�]B?

learning_rate_1�<�7W�|�I       6%�	lm�F���A�$*;


total_loss,#c@

error_R�MW?

learning_rate_1�<�7aL�I       6%�	���F���A�$*;


total_loss��@

error_R �M?

learning_rate_1�<�7��:(I       6%�	z��F���A�$*;


total_lossi�@

error_R��=?

learning_rate_1�<�7e!��I       6%�	�:�F���A�$*;


total_loss�G�@

error_R��Q?

learning_rate_1�<�7�V�I       6%�	���F���A�$*;


total_loss{��@

error_Ra\G?

learning_rate_1�<�7�;ȚI       6%�	�ìF���A�$*;


total_loss?�A

error_R��N?

learning_rate_1�<�7"U�RI       6%�	�
�F���A�$*;


total_lossn�@

error_R-�Z?

learning_rate_1�<�7ţb�I       6%�	�O�F���A�$*;


total_lossF�~@

error_R��X?

learning_rate_1�<�72|�-I       6%�	^��F���A�$*;


total_loss��@

error_RnM?

learning_rate_1�<�7���I       6%�	@ڭF���A�$*;


total_lossq�g@

error_RJ;\?

learning_rate_1�<�7ZG�I       6%�	�$�F���A�$*;


total_loss�i�@

error_RʷC?

learning_rate_1�<�7�v�'I       6%�	Ak�F���A�$*;


total_loss��@

error_R�L?

learning_rate_1�<�7����I       6%�	���F���A�$*;


total_loss�R�@

error_R&;?

learning_rate_1�<�7�>eI       6%�	���F���A�$*;


total_loss�>�@

error_REva?

learning_rate_1�<�7�ڵjI       6%�	�5�F���A�$*;


total_lossN�@

error_R�BC?

learning_rate_1�<�7��-I       6%�	w�F���A�$*;


total_loss�@

error_Rf�[?

learning_rate_1�<�7�ؔ�I       6%�	���F���A�$*;


total_loss���@

error_R��S?

learning_rate_1�<�7����I       6%�	���F���A�$*;


total_loss��@

error_R�lU?

learning_rate_1�<�7N2��I       6%�	�:�F���A�$*;


total_loss�@

error_R�!;?

learning_rate_1�<�70n"�I       6%�	a�F���A�$*;


total_loss���@

error_Rfa\?

learning_rate_1�<�7��uI       6%�	�İF���A�$*;


total_loss��@

error_Rxc?

learning_rate_1�<�7�]I       6%�	a	�F���A�$*;


total_loss�Γ@

error_Rx�L?

learning_rate_1�<�7Iu��I       6%�	f�F���A�$*;


total_loss7n�@

error_R
�C?

learning_rate_1�<�7����I       6%�	ҩ�F���A�$*;


total_loss}�@

error_R�Yb?

learning_rate_1�<�7t��II       6%�	��F���A�$*;


total_loss$&�@

error_R�R?

learning_rate_1�<�7|�kI       6%�	3�F���A�$*;


total_loss!V�@

error_R19]?

learning_rate_1�<�7f�3�I       6%�	ow�F���A�$*;


total_loss���@

error_R��[?

learning_rate_1�<�7���0I       6%�	绲F���A�$*;


total_loss���@

error_R�$c?

learning_rate_1�<�7?�z�I       6%�	� �F���A�$*;


total_loss��@

error_RU?

learning_rate_1�<�7O�OI       6%�	C�F���A�$*;


total_loss�@

error_R�V?

learning_rate_1�<�7S���I       6%�	r��F���A�$*;


total_loss&D�@

error_R�YB?

learning_rate_1�<�7k�N�I       6%�	ճF���A�$*;


total_loss�6�@

error_R�^P?

learning_rate_1�<�7c��I       6%�	��F���A�$*;


total_loss���@

error_R�Z?

learning_rate_1�<�71�SI       6%�	[�F���A�$*;


total_loss?T�@

error_RJE?

learning_rate_1�<�7��
)I       6%�	i��F���A�$*;


total_lossJ��@

error_R�P?

learning_rate_1�<�7IY#I       6%�		�F���A�$*;


total_loss,A�@

error_REG?

learning_rate_1�<�7��ȐI       6%�	�%�F���A�$*;


total_losss�A

error_R�4C?

learning_rate_1�<�7,I��I       6%�	h�F���A�$*;


total_loss�]�@

error_Ru=?

learning_rate_1�<�7�p4qI       6%�	Ϊ�F���A�$*;


total_loss��@

error_R-�O?

learning_rate_1�<�7���yI       6%�	��F���A�$*;


total_loss�K�@

error_RO�R?

learning_rate_1�<�7Ϧ7�I       6%�	f2�F���A�$*;


total_lossj��@

error_R��T?

learning_rate_1�<�7�1ͮI       6%�	1t�F���A�$*;


total_loss��@

error_R�OQ?

learning_rate_1�<�7;�7I       6%�	��F���A�$*;


total_loss�@

error_RSW<?

learning_rate_1�<�7�նI       6%�	���F���A�$*;


total_loss#o�@

error_R��M?

learning_rate_1�<�7�_͞I       6%�	�@�F���A�$*;


total_loss��@

error_R`R?

learning_rate_1�<�7͆wI       6%�	'��F���A�$*;


total_lossE�@

error_R.�L?

learning_rate_1�<�7v�ۨI       6%�	�ԷF���A�$*;


total_lossD>A

error_RT�P?

learning_rate_1�<�7Oo�XI       6%�	+�F���A�$*;


total_loss��@

error_R_�Q?

learning_rate_1�<�7!�OI       6%�	q�F���A�$*;


total_loss�W+A

error_R��G?

learning_rate_1�<�7.��I       6%�	c��F���A�$*;


total_loss���@

error_R4oa?

learning_rate_1�<�7��I       6%�	?��F���A�$*;


total_loss��JA

error_R)Ec?

learning_rate_1�<�77^4I       6%�	W:�F���A�$*;


total_loss:ӛ@

error_R�ah?

learning_rate_1�<�7��ūI       6%�	n��F���A�$*;


total_loss��@

error_R1�K?

learning_rate_1�<�7Tk)3I       6%�	zŹF���A�$*;


total_loss�A

error_Rs�H?

learning_rate_1�<�7j��I       6%�	�F���A�$*;


total_loss*��@

error_R��V?

learning_rate_1�<�7ܜ�:I       6%�	�b�F���A�$*;


total_loss֠A

error_R�U?

learning_rate_1�<�7l�#I       6%�	��F���A�$*;


total_lossMI�@

error_RT�I?

learning_rate_1�<�7^gTI       6%�	��F���A�$*;


total_loss:��@

error_R�1R?

learning_rate_1�<�7@[BI       6%�	5�F���A�$*;


total_loss-bA

error_R�W?

learning_rate_1�<�7X���I       6%�	�w�F���A�$*;


total_lossD�A

error_R �=?

learning_rate_1�<�7��s�I       6%�	Ѽ�F���A�$*;


total_lossh��@

error_R�J?

learning_rate_1�<�7$p�I       6%�	��F���A�$*;


total_loss ��@

error_REH?

learning_rate_1�<�7}#ҍI       6%�	�G�F���A�$*;


total_lossJ��@

error_R��K?

learning_rate_1�<�7F�oI       6%�	i��F���A�$*;


total_lossF��@

error_R69=?

learning_rate_1�<�7�7�I       6%�	5ҼF���A�$*;


total_loss���@

error_R�S?

learning_rate_1�<�7���I       6%�	k�F���A�$*;


total_lossxѷ@

error_R�Y?

learning_rate_1�<�7��!�I       6%�	�c�F���A�$*;


total_loss��@

error_RT�>?

learning_rate_1�<�7�QD�I       6%�	���F���A�$*;


total_loss�ڴ@

error_R�PK?

learning_rate_1�<�7��dI       6%�	��F���A�$*;


total_lossA�A

error_R�UK?

learning_rate_1�<�7%yK�I       6%�	�;�F���A�$*;


total_loss8�@

error_RH?

learning_rate_1�<�7L)��I       6%�	R��F���A�$*;


total_loss���@

error_R1k\?

learning_rate_1�<�7��2I       6%�	%;F���A�$*;


total_loss!`�@

error_Rs�K?

learning_rate_1�<�7,d�+I       6%�	��F���A�$*;


total_loss���@

error_RH�W?

learning_rate_1�<�7c�I       6%�	�]�F���A�$*;


total_loss,��@

error_R�H?

learning_rate_1�<�7|~=�I       6%�	n��F���A�$*;


total_loss���@

error_RyZ?

learning_rate_1�<�7���I       6%�	��F���A�$*;


total_loss���@

error_R�-D?

learning_rate_1�<�7.B�I       6%�	B7�F���A�$*;


total_loss#g@

error_R&]K?

learning_rate_1�<�7��!I       6%�	&��F���A�$*;


total_loss��@

error_R�b=?

learning_rate_1�<�7��I       6%�	��F���A�$*;


total_loss��@

error_Rd�G?

learning_rate_1�<�7v�2I       6%�	n9�F���A�$*;


total_loss��@

error_R�M?

learning_rate_1�<�7��a�I       6%�	��F���A�$*;


total_loss�z�@

error_R�0W?

learning_rate_1�<�7�,"I       6%�	���F���A�$*;


total_lossn-�@

error_R��H?

learning_rate_1�<�7����I       6%�	�	�F���A�$*;


total_lossM�@

error_RAH?

learning_rate_1�<�79s�I       6%�	�L�F���A�$*;


total_loss(�@

error_R�C?

learning_rate_1�<�7�J�I       6%�	W��F���A�$*;


total_loss���@

error_R��Y?

learning_rate_1�<�7�mwI       6%�	|��F���A�$*;


total_loss(�@

error_R��d?

learning_rate_1�<�7�D:I       6%�	��F���A�$*;


total_loss�p�@

error_RC5J?

learning_rate_1�<�7;�I       6%�	[�F���A�$*;


total_loss�3�@

error_R�/T?

learning_rate_1�<�7%�[�I       6%�	��F���A�$*;


total_lossbA

error_R�2Y?

learning_rate_1�<�76���I       6%�	��F���A�$*;


total_lossnr�@

error_R�"J?

learning_rate_1�<�7�tI       6%�	�#�F���A�$*;


total_loss(,�@

error_R�eN?

learning_rate_1�<�7����I       6%�	Ym�F���A�$*;


total_loss���@

error_R/�S?

learning_rate_1�<�7skABI       6%�	R��F���A�$*;


total_loss.\�@

error_R�I?

learning_rate_1�<�7��I       6%�	��F���A�$*;


total_loss��A

error_R��O?

learning_rate_1�<�7��I       6%�	G>�F���A�$*;


total_lossβ�@

error_R�K?

learning_rate_1�<�7~khI       6%�	e��F���A�$*;


total_loss�6�@

error_R�S?

learning_rate_1�<�7I[��I       6%�	���F���A�$*;


total_loss]��@

error_R��N?

learning_rate_1�<�7����I       6%�	G�F���A�$*;


total_loss��@

error_R�Z?

learning_rate_1�<�7�p�I       6%�	kP�F���A�$*;


total_loss�7�@

error_R�R?

learning_rate_1�<�7͸`�I       6%�	���F���A�%*;


total_loss��@

error_R��N?

learning_rate_1�<�7
 |I       6%�	���F���A�%*;


total_loss;��@

error_R,�R?

learning_rate_1�<�7���I       6%�	��F���A�%*;


total_lossHD�@

error_R�X?

learning_rate_1�<�7<��{I       6%�	�\�F���A�%*;


total_loss���@

error_Ri�J?

learning_rate_1�<�7Dq�I       6%�	D��F���A�%*;


total_loss���@

error_RQSN?

learning_rate_1�<�7\�;�I       6%�	���F���A�%*;


total_lossNs�@

error_RpL?

learning_rate_1�<�7_ ��I       6%�	�B�F���A�%*;


total_lossx��@

error_R��A?

learning_rate_1�<�7��CI       6%�	އ�F���A�%*;


total_lossq@�@

error_R��??

learning_rate_1�<�7� �gI       6%�	���F���A�%*;


total_loss���@

error_R�N`?

learning_rate_1�<�7q��I       6%�	� �F���A�%*;


total_loss�#�@

error_R��I?

learning_rate_1�<�7�;�I       6%�	�f�F���A�%*;


total_loss�&�@

error_Rs�U?

learning_rate_1�<�7X��I       6%�	1��F���A�%*;


total_loss�پ@

error_Rd�@?

learning_rate_1�<�7lLGNI       6%�	I��F���A�%*;


total_lossT��@

error_R��_?

learning_rate_1�<�7��\I       6%�	6<�F���A�%*;


total_loss,��@

error_R-B?

learning_rate_1�<�7*(�FI       6%�	n}�F���A�%*;


total_loss�$�@

error_R�]:?

learning_rate_1�<�7^�I       6%�	���F���A�%*;


total_loss�*�@

error_R�HK?

learning_rate_1�<�7����I       6%�	��F���A�%*;


total_loss2��@

error_R�T?

learning_rate_1�<�7 �CI       6%�	rK�F���A�%*;


total_loss�$A

error_R6�q?

learning_rate_1�<�7O��gI       6%�	T��F���A�%*;


total_loss��@

error_R�H?

learning_rate_1�<�7� �>I       6%�	���F���A�%*;


total_loss��@

error_RXLY?

learning_rate_1�<�7· �I       6%�	��F���A�%*;


total_loss=n�@

error_R$�:?

learning_rate_1�<�7���I       6%�	BZ�F���A�%*;


total_lossƿ�@

error_R�^?

learning_rate_1�<�7V��BI       6%�	��F���A�%*;


total_loss��@

error_R�?K?

learning_rate_1�<�7�$�eI       6%�	���F���A�%*;


total_loss��@

error_R[�J?

learning_rate_1�<�7/�-	I       6%�	�2�F���A�%*;


total_loss�5�@

error_R��C?

learning_rate_1�<�7���JI       6%�	7��F���A�%*;


total_lossWl@

error_R�HE?

learning_rate_1�<�7���I       6%�	��F���A�%*;


total_loss���@

error_R�K?

learning_rate_1�<�7
E�I       6%�	|�F���A�%*;


total_loss���@

error_R��N?

learning_rate_1�<�7�D'I       6%�	�[�F���A�%*;


total_loss��@

error_R��F?

learning_rate_1�<�7�+��I       6%�	ћ�F���A�%*;


total_loss�*�@

error_R�?I?

learning_rate_1�<�7XP8I       6%�	;��F���A�%*;


total_losssY�@

error_R�D?

learning_rate_1�<�7��I       6%�	l!�F���A�%*;


total_loss{I�@

error_R8�N?

learning_rate_1�<�7%Q��I       6%�	dd�F���A�%*;


total_loss=��@

error_R�N?

learning_rate_1�<�7&�iI       6%�	��F���A�%*;


total_loss���@

error_R�<?

learning_rate_1�<�77��aI       6%�	��F���A�%*;


total_lossF��@

error_R��R?

learning_rate_1�<�7�<�yI       6%�	A�F���A�%*;


total_loss��@

error_RT`N?

learning_rate_1�<�7{�m�I       6%�	��F���A�%*;


total_loss�Z�@

error_RdT?

learning_rate_1�<�75�~QI       6%�	��F���A�%*;


total_loss�|�@

error_R��U?

learning_rate_1�<�7�3cI       6%�	��F���A�%*;


total_loss�Y�@

error_R�G?

learning_rate_1�<�7	XI       6%�	�_�F���A�%*;


total_loss�{�@

error_R�M7?

learning_rate_1�<�7�A�I       6%�	"��F���A�%*;


total_lossL�@

error_R
T?

learning_rate_1�<�7��fI       6%�	���F���A�%*;


total_loss�e�@

error_R��K?

learning_rate_1�<�7T��I       6%�	�,�F���A�%*;


total_loss��@

error_R��H?

learning_rate_1�<�7�1�AI       6%�	Aq�F���A�%*;


total_loss�v�@

error_R��@?

learning_rate_1�<�7�0,:I       6%�	c��F���A�%*;


total_loss�9A

error_R� _?

learning_rate_1�<�7#��I       6%�	���F���A�%*;


total_loss��A

error_Rך9?

learning_rate_1�<�7�I       6%�	Z?�F���A�%*;


total_loss3��@

error_R)�^?

learning_rate_1�<�79�I       6%�	n��F���A�%*;


total_loss��@

error_RHE;?

learning_rate_1�<�7�ɵ�I       6%�	'��F���A�%*;


total_loss1�@

error_RfK?

learning_rate_1�<�7Q�<�I       6%�	�	�F���A�%*;


total_loss��@

error_R\�J?

learning_rate_1�<�7��ÐI       6%�	�M�F���A�%*;


total_lossҔ�@

error_Ro�V?

learning_rate_1�<�7iT�cI       6%�	���F���A�%*;


total_loss��@

error_R�	M?

learning_rate_1�<�7� P�I       6%�	���F���A�%*;


total_loss�\�@

error_R�J?

learning_rate_1�<�7r��I       6%�	��F���A�%*;


total_loss3��@

error_R�iJ?

learning_rate_1�<�7��CI       6%�	8a�F���A�%*;


total_loss�w@

error_R��N?

learning_rate_1�<�7,reI       6%�	{��F���A�%*;


total_loss���@

error_R9?

learning_rate_1�<�7,NL�I       6%�	���F���A�%*;


total_loss-d�@

error_R��??

learning_rate_1�<�7d�I       6%�	�,�F���A�%*;


total_loss	��@

error_Rh�]?

learning_rate_1�<�7���I       6%�	7r�F���A�%*;


total_lossTA

error_R��K?

learning_rate_1�<�7�JI       6%�	s��F���A�%*;


total_losst�@

error_RM4L?

learning_rate_1�<�76���I       6%�	i��F���A�%*;


total_loss�*�@

error_R�1Q?

learning_rate_1�<�7"�I       6%�	�7�F���A�%*;


total_loss2)�@

error_R�E?

learning_rate_1�<�7�ݾI       6%�	0~�F���A�%*;


total_lossz�]@

error_R)�S?

learning_rate_1�<�7�Il)I       6%�	���F���A�%*;


total_loss��@

error_Rf�M?

learning_rate_1�<�7(��kI       6%�	�0�F���A�%*;


total_lossRF�@

error_RTf?

learning_rate_1�<�7SXG`I       6%�	�w�F���A�%*;


total_loss-T+A

error_R��[?

learning_rate_1�<�7$��I       6%�	q��F���A�%*;


total_lossaS�@

error_RQjQ?

learning_rate_1�<�7�x�I       6%�	V��F���A�%*;


total_loss�@

error_R&T?

learning_rate_1�<�7��'I       6%�	�B�F���A�%*;


total_loss��@

error_R%�[?

learning_rate_1�<�7?f"�I       6%�	��F���A�%*;


total_lossO�@

error_R13`?

learning_rate_1�<�7'�I       6%�	���F���A�%*;


total_loss���@

error_R{�H?

learning_rate_1�<�7��J�I       6%�	��F���A�%*;


total_lossDD�@

error_R�Uc?

learning_rate_1�<�7k�i�I       6%�	GJ�F���A�%*;


total_loss��@

error_R
�>?

learning_rate_1�<�7'�-�I       6%�	ӌ�F���A�%*;


total_loss4�@

error_RV�I?

learning_rate_1�<�7���I       6%�	���F���A�%*;


total_lossN4HA

error_R�1R?

learning_rate_1�<�7@NxI       6%�	��F���A�%*;


total_lossL�A

error_RjlS?

learning_rate_1�<�7Az�MI       6%�	�Z�F���A�%*;


total_lossH'�@

error_RE�\?

learning_rate_1�<�7sՆI       6%�	T��F���A�%*;


total_loss�}�@

error_R&�T?

learning_rate_1�<�7 ���I       6%�	��F���A�%*;


total_loss�ŋ@

error_R�VH?

learning_rate_1�<�7fY�I       6%�	�.�F���A�%*;


total_loss� �@

error_R�eX?

learning_rate_1�<�7�Ao�I       6%�	ex�F���A�%*;


total_lossQݏ@

error_R�XM?

learning_rate_1�<�7�F�@I       6%�	��F���A�%*;


total_loss��@

error_R$�J?

learning_rate_1�<�7DYPI       6%�	��F���A�%*;


total_loss�v�@

error_R-�=?

learning_rate_1�<�7�dĝI       6%�	mQ�F���A�%*;


total_loss�A

error_R��Z?

learning_rate_1�<�7�	1gI       6%�	���F���A�%*;


total_loss�W�@

error_R&T?

learning_rate_1�<�7�kwI       6%�	~��F���A�%*;


total_loss���@

error_RdvN?

learning_rate_1�<�7Jˠ�I       6%�	q�F���A�%*;


total_loss���@

error_R�PT?

learning_rate_1�<�7]+�I       6%�	�b�F���A�%*;


total_loss���@

error_R�m<?

learning_rate_1�<�7�ǺI       6%�	��F���A�%*;


total_lossH��@

error_R�@?

learning_rate_1�<�7+IA@I       6%�	���F���A�%*;


total_loss���@

error_RKR?

learning_rate_1�<�7��'I       6%�	�,�F���A�%*;


total_loss�@

error_R��;?

learning_rate_1�<�7b���I       6%�	lm�F���A�%*;


total_loss[��@

error_RX`??

learning_rate_1�<�7��QI       6%�	*��F���A�%*;


total_lossI1�@

error_R��J?

learning_rate_1�<�7�`-�I       6%�	���F���A�%*;


total_lossL`�@

error_R��g?

learning_rate_1�<�7oѴI       6%�	�=�F���A�%*;


total_lossE��@

error_RvM?

learning_rate_1�<�7�9�I       6%�	f��F���A�%*;


total_loss#�@

error_R�#c?

learning_rate_1�<�7ݘ'I       6%�	\��F���A�%*;


total_loss<h�@

error_R�oI?

learning_rate_1�<�7��~pI       6%�	�F���A�%*;


total_loss�c�@

error_R6zf?

learning_rate_1�<�7�J=I       6%�	F�F���A�%*;


total_loss�=�@

error_R��N?

learning_rate_1�<�778MI       6%�	��F���A�%*;


total_loss���@

error_R��8?

learning_rate_1�<�7�Ĳ�I       6%�	���F���A�%*;


total_loss|�@

error_R�}Y?

learning_rate_1�<�7(��I       6%�	
�F���A�%*;


total_lossO��@

error_R<:G?

learning_rate_1�<�7<��vI       6%�	k^�F���A�%*;


total_loss���@

error_Rw&U?

learning_rate_1�<�7 �02I       6%�	ɦ�F���A�%*;


total_loss�(�@

error_R{[]?

learning_rate_1�<�7�ŜI       6%�	'��F���A�%*;


total_lossݜ,A

error_R_�K?

learning_rate_1�<�7��I       6%�	n3�F���A�%*;


total_loss!A�@

error_R)�J?

learning_rate_1�<�7�֊cI       6%�	=��F���A�%*;


total_loss���@

error_R�za?

learning_rate_1�<�7�V�`I       6%�	���F���A�%*;


total_lossa�@

error_R�V?

learning_rate_1�<�7�'I       6%�	d�F���A�%*;


total_loss���@

error_R�S?

learning_rate_1�<�7����I       6%�	^�F���A�%*;


total_lossa��@

error_R��Q?

learning_rate_1�<�7�_��I       6%�	���F���A�%*;


total_loss�ܺ@

error_R��V?

learning_rate_1�<�7�BjI       6%�	���F���A�%*;


total_loss��@

error_R�>?

learning_rate_1�<�7�r# I       6%�	�4�F���A�%*;


total_loss�!�@

error_RqgX?

learning_rate_1�<�7O&�I       6%�	ƀ�F���A�%*;


total_loss���@

error_Rt�_?

learning_rate_1�<�7B $I       6%�	\��F���A�%*;


total_loss6��@

error_RםM?

learning_rate_1�<�7=�I       6%�	0�F���A�%*;


total_lossan A

error_R`LV?

learning_rate_1�<�7|�]I       6%�	I�F���A�%*;


total_loss1��@

error_R�L?

learning_rate_1�<�7�Y�@I       6%�	��F���A�%*;


total_loss�[�@

error_R��1?

learning_rate_1�<�7S�HI       6%�	��F���A�%*;


total_loss?Ɯ@

error_Ra�P?

learning_rate_1�<�7<�mI       6%�	��F���A�%*;


total_losssr�@

error_R!�<?

learning_rate_1�<�7�ox�I       6%�	�Z�F���A�%*;


total_loss��@

error_R�aO?

learning_rate_1�<�7�f�:I       6%�	4��F���A�%*;


total_loss��@

error_Rm�X?

learning_rate_1�<�7��~I       6%�	���F���A�%*;


total_loss�Z�@

error_R�JI?

learning_rate_1�<�7O31I       6%�	�D�F���A�%*;


total_loss��z@

error_R@�C?

learning_rate_1�<�7�H�I       6%�	G��F���A�%*;


total_loss�
�@

error_RqvH?

learning_rate_1�<�7Ӈ|I       6%�	���F���A�%*;


total_loss��A

error_Ra�T?

learning_rate_1�<�7ѻ
�I       6%�	��F���A�%*;


total_loss��@

error_R�,7?

learning_rate_1�<�7��<I       6%�	�`�F���A�%*;


total_loss.��@

error_R��I?

learning_rate_1�<�7�D�I       6%�	���F���A�&*;


total_loss�.�@

error_R��M?

learning_rate_1�<�7�A5�I       6%�	���F���A�&*;


total_loss�-�@

error_R[\>?

learning_rate_1�<�7�8�[I       6%�	�.�F���A�&*;


total_lossj[�@

error_RQK?

learning_rate_1�<�78/�I       6%�	r�F���A�&*;


total_loss��~@

error_R�zA?

learning_rate_1�<�7򄢝I       6%�	{��F���A�&*;


total_loss䨊@

error_R�_?

learning_rate_1�<�7�iEI       6%�	���F���A�&*;


total_lossYHA

error_R�YT?

learning_rate_1�<�72��-I       6%�	�D�F���A�&*;


total_lossZ0e@

error_R]�O?

learning_rate_1�<�7��v�I       6%�	ΐ�F���A�&*;


total_loss���@

error_R��Q?

learning_rate_1�<�7H���I       6%�	���F���A�&*;


total_loss}L�@

error_R�xM?

learning_rate_1�<�7�p��I       6%�	�F���A�&*;


total_loss��@

error_RR�N?

learning_rate_1�<�7��wI       6%�	kg�F���A�&*;


total_lossa��@

error_Roj>?

learning_rate_1�<�7� MUI       6%�	<��F���A�&*;


total_loss�ˁ@

error_RQ
I?

learning_rate_1�<�7�|�HI       6%�	���F���A�&*;


total_loss-h�@

error_RM?

learning_rate_1�<�7̭��I       6%�	�>�F���A�&*;


total_loss"��@

error_R�B?

learning_rate_1�<�75���I       6%�	���F���A�&*;


total_lossѧ�@

error_R�4\?

learning_rate_1�<�7�[�_I       6%�	,��F���A�&*;


total_loss���@

error_R��V?

learning_rate_1�<�7�o	hI       6%�	��F���A�&*;


total_loss8��@

error_R��T?

learning_rate_1�<�7���I       6%�	WO�F���A�&*;


total_loss/:�@

error_RE?

learning_rate_1�<�7����I       6%�	n��F���A�&*;


total_lossܒ�@

error_R��R?

learning_rate_1�<�7���I       6%�	���F���A�&*;


total_lossnm�@

error_R�G?

learning_rate_1�<�7�=X0I       6%�	�F���A�&*;


total_loss()�@

error_R��M?

learning_rate_1�<�7�wa�I       6%�	�X�F���A�&*;


total_loss���@

error_R�}Z?

learning_rate_1�<�7���I       6%�	d��F���A�&*;


total_loss8�@

error_RI8d?

learning_rate_1�<�7��O7I       6%�	���F���A�&*;


total_lossK�@

error_R�uO?

learning_rate_1�<�7�z�<I       6%�	��F���A�&*;


total_loss�,�@

error_R��<?

learning_rate_1�<�7�^|I       6%�	�]�F���A�&*;


total_loss�_�@

error_Rr?

learning_rate_1�<�7��6AI       6%�	��F���A�&*;


total_lossʅ�@

error_RW=I?

learning_rate_1�<�7%�N~I       6%�	��F���A�&*;


total_loss�YA

error_RȭV?

learning_rate_1�<�7�[ I       6%�	@&�F���A�&*;


total_loss:4�@

error_R��5?

learning_rate_1�<�7���I       6%�	uh�F���A�&*;


total_loss�L�@

error_R�le?

learning_rate_1�<�7O6I       6%�	V��F���A�&*;


total_loss���@

error_R��W?

learning_rate_1�<�7�N��I       6%�	���F���A�&*;


total_loss�uz@

error_R�
@?

learning_rate_1�<�7�DI       6%�	w-�F���A�&*;


total_loss�R�@

error_R1�K?

learning_rate_1�<�7���I       6%�	�u�F���A�&*;


total_loss���@

error_R�B?

learning_rate_1�<�7��OI       6%�	���F���A�&*;


total_losss��@

error_Ri�I?

learning_rate_1�<�7�t��I       6%�	x�F���A�&*;


total_lossƈk@

error_R�H?

learning_rate_1�<�7'�rI       6%�	�R�F���A�&*;


total_loss
�@

error_RW�S?

learning_rate_1�<�7ݟe�I       6%�	i��F���A�&*;


total_loss��@

error_R�GB?

learning_rate_1�<�7
ǠI       6%�	��F���A�&*;


total_loss���@

error_R!�:?

learning_rate_1�<�7D�NI       6%�	��F���A�&*;


total_lossl�@

error_R��H?

learning_rate_1�<�7�`F�I       6%�	�a�F���A�&*;


total_loss���@

error_R�=?

learning_rate_1�<�76i�UI       6%�	"��F���A�&*;


total_lossv�@

error_R��P?

learning_rate_1�<�7�{&�I       6%�	P��F���A�&*;


total_lossL�@

error_RcZ?

learning_rate_1�<�7��I       6%�	�4�F���A�&*;


total_lossd��@

error_R�5S?

learning_rate_1�<�7�>�I       6%�	�y�F���A�&*;


total_loss�#�@

error_Rx�K?

learning_rate_1�<�7�jw,I       6%�	��F���A�&*;


total_loss��@

error_RJ�??

learning_rate_1�<�7���@I       6%�	-��F���A�&*;


total_loss�A

error_R�7H?

learning_rate_1�<�7�Z0I       6%�	>C�F���A�&*;


total_loss*4�@

error_R��e?

learning_rate_1�<�7�VsI       6%�	���F���A�&*;


total_loss[K�@

error_R�f`?

learning_rate_1�<�7>�m�I       6%�	���F���A�&*;


total_loss�Z�@

error_Rӥe?

learning_rate_1�<�7�4%�I       6%�	��F���A�&*;


total_loss��@

error_R��@?

learning_rate_1�<�7e[I       6%�	�Q�F���A�&*;


total_loss��@

error_Rt/e?

learning_rate_1�<�7�Q1I       6%�	R��F���A�&*;


total_loss4DA

error_R(;?

learning_rate_1�<�7C �bI       6%�	f��F���A�&*;


total_lossn4�@

error_RI)S?

learning_rate_1�<�7�կI       6%�	?�F���A�&*;


total_loss�%�@

error_R��T?

learning_rate_1�<�7��I       6%�	h��F���A�&*;


total_loss���@

error_R�;`?

learning_rate_1�<�7�E0I       6%�	���F���A�&*;


total_loss�Ml@

error_R�N?

learning_rate_1�<�7��OI       6%�	?�F���A�&*;


total_loss��@

error_R�W?

learning_rate_1�<�7�cl�I       6%�	{U�F���A�&*;


total_loss�պ@

error_R-H?

learning_rate_1�<�7��JI       6%�	��F���A�&*;


total_loss��@

error_RQ.L?

learning_rate_1�<�7�!��I       6%�	��F���A�&*;


total_loss@��@

error_R_�@?

learning_rate_1�<�7b[\!I       6%�	[M�F���A�&*;


total_loss���@

error_RO9`?

learning_rate_1�<�7,�PI       6%�	H��F���A�&*;


total_loss�R�@

error_R��Y?

learning_rate_1�<�73�ޗI       6%�	���F���A�&*;


total_loss��@

error_RX�P?

learning_rate_1�<�7SↇI       6%�	B�F���A�&*;


total_lossQ4�@

error_R2"U?

learning_rate_1�<�7ϛ3�I       6%�	�^�F���A�&*;


total_loss���@

error_R�
C?

learning_rate_1�<�7��BI       6%�	���F���A�&*;


total_loss\4�@

error_R�*W?

learning_rate_1�<�7�F�I       6%�	��F���A�&*;


total_loss᧳@

error_R33K?

learning_rate_1�<�7�ٿ�I       6%�	J'�F���A�&*;


total_loss*�@

error_Ro�D?

learning_rate_1�<�7{��I       6%�	�k�F���A�&*;


total_loss���@

error_R&�d?

learning_rate_1�<�7��P�I       6%�	̯�F���A�&*;


total_loss�9�@

error_R�"U?

learning_rate_1�<�7@X#fI       6%�	G��F���A�&*;


total_losss�@

error_R�	??

learning_rate_1�<�7�K�I       6%�	�=�F���A�&*;


total_loss��@

error_Rd�G?

learning_rate_1�<�7�2j}I       6%�	���F���A�&*;


total_loss�r�@

error_R�\?

learning_rate_1�<�7��>I       6%�	g��F���A�&*;


total_losss�A

error_R�AT?

learning_rate_1�<�7���I       6%�	��F���A�&*;


total_lossA��@

error_Rc?

learning_rate_1�<�7z̶�I       6%�	�\�F���A�&*;


total_loss�eo@

error_R��??

learning_rate_1�<�7�]U�I       6%�	���F���A�&*;


total_loss��@

error_R#�>?

learning_rate_1�<�7��wI       6%�	0��F���A�&*;


total_loss���@

error_R��E?

learning_rate_1�<�7����I       6%�	[.�F���A�&*;


total_loss���@

error_R�Q?

learning_rate_1�<�7��.�I       6%�	D}�F���A�&*;


total_loss���@

error_R �A?

learning_rate_1�<�7��=�I       6%�	��F���A�&*;


total_loss���@

error_R��L?

learning_rate_1�<�7e�kI       6%�	� G���A�&*;


total_loss��@

error_R1M?

learning_rate_1�<�75R��I       6%�	�U G���A�&*;


total_loss���@

error_R��Z?

learning_rate_1�<�7.��I       6%�	�$G���A�&*;


total_loss�Ǻ@

error_Rq�>?

learning_rate_1�<�7���I       6%�	w�G���A�&*;


total_lossI�@

error_R�{8?

learning_rate_1�<�7�Ig�I       6%�	4�G���A�&*;


total_loss)��@

error_Ro ??

learning_rate_1�<�7�pQI       6%�	MG���A�&*;


total_loss.�@

error_R�KR?

learning_rate_1�<�7$�~FI       6%�	ޑG���A�&*;


total_loss���@

error_R�VM?

learning_rate_1�<�7C��I       6%�	-�G���A�&*;


total_loss�;�@

error_RM�E?

learning_rate_1�<�7���/I       6%�	�$G���A�&*;


total_loss��@

error_R��Y?

learning_rate_1�<�7GupI       6%�	�kG���A�&*;


total_loss���@

error_R3�O?

learning_rate_1�<�7���BI       6%�	9�G���A�&*;


total_losss��@

error_R��G?

learning_rate_1�<�7ӯ�I       6%�	N�G���A�&*;


total_loss��A

error_R��a?

learning_rate_1�<�7s9&I       6%�	�=G���A�&*;


total_lossR>�@

error_R,�P?

learning_rate_1�<�7��<YI       6%�	��G���A�&*;


total_lossLD�@

error_R@BV?

learning_rate_1�<�7��+I       6%�	�G���A�&*;


total_loss�3�@

error_R�iO?

learning_rate_1�<�7k��sI       6%�	�G���A�&*;


total_loss�1�@

error_R{KM?

learning_rate_1�<�7'G�I       6%�	�TG���A�&*;


total_loss}6A

error_R��J?

learning_rate_1�<�7�z] I       6%�	,�G���A�&*;


total_loss���@

error_R�G?

learning_rate_1�<�7�1� I       6%�	��G���A�&*;


total_lossAO�@

error_RH�X?

learning_rate_1�<�7�s�GI       6%�	�*G���A�&*;


total_loss��@

error_R�1I?

learning_rate_1�<�7���I       6%�	=nG���A�&*;


total_lossF��@

error_R��P?

learning_rate_1�<�7�yI       6%�	��G���A�&*;


total_loss�i�@

error_RR�2?

learning_rate_1�<�7��eI       6%�	,�G���A�&*;


total_loss��@

error_R V?

learning_rate_1�<�7~g�I       6%�	�6G���A�&*;


total_lossE!�@

error_R��;?

learning_rate_1�<�7+�>I       6%�	�zG���A�&*;


total_loss�3�@

error_R$|d?

learning_rate_1�<�7�X
�I       6%�	��G���A�&*;


total_loss�Z�@

error_R�AE?

learning_rate_1�<�7�^TI       6%�	�$G���A�&*;


total_loss�s�@

error_R?R?

learning_rate_1�<�7����I       6%�	�jG���A�&*;


total_loss�YA

error_R��a?

learning_rate_1�<�7���I       6%�	K�G���A�&*;


total_loss�-�@

error_R�FY?

learning_rate_1�<�7��)�I       6%�	��G���A�&*;


total_lossz4�@

error_RL�E?

learning_rate_1�<�7Ô7�I       6%�	�5	G���A�&*;


total_loss�@

error_R�7T?

learning_rate_1�<�7�zE�I       6%�	~	G���A�&*;


total_lossC��@

error_RӍQ?

learning_rate_1�<�7R���I       6%�	�	G���A�&*;


total_loss���@

error_R��S?

learning_rate_1�<�7ᅎ�I       6%�	
G���A�&*;


total_loss�`A

error_R!A?

learning_rate_1�<�7��M,I       6%�	�V
G���A�&*;


total_loss)�@

error_R��T?

learning_rate_1�<�7��ۂI       6%�	=�
G���A�&*;


total_loss�@

error_R {`?

learning_rate_1�<�7W�>I       6%�	W�
G���A�&*;


total_loss	��@

error_R3M?

learning_rate_1�<�7ؠ[:I       6%�	�!G���A�&*;


total_loss��@

error_RƄI?

learning_rate_1�<�7���I       6%�	�dG���A�&*;


total_loss�%�@

error_R�J?

learning_rate_1�<�7CL��I       6%�	�G���A�&*;


total_loss[-�@

error_R WE?

learning_rate_1�<�7̬I       6%�	^�G���A�&*;


total_lossJ4A

error_R��Q?

learning_rate_1�<�7���I       6%�	�,G���A�&*;


total_loss\$�@

error_R�VM?

learning_rate_1�<�7G^R�I       6%�	ilG���A�&*;


total_loss�j�@

error_Rz�G?

learning_rate_1�<�7b��I       6%�	��G���A�&*;


total_loss�\�@

error_RZmT?

learning_rate_1�<�7����I       6%�	8�G���A�&*;


total_loss�/�@

error_RqX.?

learning_rate_1�<�7�}ВI       6%�	;AG���A�&*;


total_loss/0�@

error_R�<V?

learning_rate_1�<�7-�I       6%�	ƂG���A�'*;


total_loss2L�@

error_R�L_?

learning_rate_1�<�7\�>I       6%�	�G���A�'*;


total_loss��@

error_R@:B?

learning_rate_1�<�71ԄNI       6%�	�	G���A�'*;


total_loss�x�@

error_Rn&U?

learning_rate_1�<�7/�CI       6%�	YMG���A�'*;


total_loss�5�@

error_R�	Q?

learning_rate_1�<�7��I       6%�	��G���A�'*;


total_lossN��@

error_R�vK?

learning_rate_1�<�76Z[I       6%�	��G���A�'*;


total_loss$�@

error_R��O?

learning_rate_1�<�7�W�I       6%�	�G���A�'*;


total_lossv��@

error_R�]?

learning_rate_1�<�7 �O	I       6%�	tUG���A�'*;


total_loss���@

error_R�vg?

learning_rate_1�<�7�EoI       6%�	^�G���A�'*;


total_loss�y�@

error_R
AN?

learning_rate_1�<�7��P�I       6%�	�G���A�'*;


total_lossd3�@

error_R �V?

learning_rate_1�<�7�{N-I       6%�	YbG���A�'*;


total_loss�<�@

error_R7wI?

learning_rate_1�<�7�f�I       6%�	�G���A�'*;


total_loss:Ӝ@

error_R܎\?

learning_rate_1�<�7#�AI       6%�	��G���A�'*;


total_lossXƥ@

error_R�WB?

learning_rate_1�<�7aN�I       6%�	�?G���A�'*;


total_loss�	a@

error_R�$B?

learning_rate_1�<�7\�"^I       6%�	��G���A�'*;


total_lossd�@

error_R}$S?

learning_rate_1�<�7���wI       6%�	S�G���A�'*;


total_loss-�@

error_RnE?

learning_rate_1�<�7�5�I       6%�	�7G���A�'*;


total_lossF"	A

error_Rn�E?

learning_rate_1�<�7m/�I       6%�	��G���A�'*;


total_loss\_l@

error_R< P?

learning_rate_1�<�7��!2I       6%�	��G���A�'*;


total_loss�3�@

error_RT?

learning_rate_1�<�7��/I       6%�	� G���A�'*;


total_loss$�@

error_R=�d?

learning_rate_1�<�7b���I       6%�	qbG���A�'*;


total_loss�r�@

error_R#?X?

learning_rate_1�<�7�Z�iI       6%�	r�G���A�'*;


total_loss:A

error_RT�M?

learning_rate_1�<�7@��I       6%�	�G���A�'*;


total_loss	_�@

error_R,cT?

learning_rate_1�<�7�&�I       6%�	�+G���A�'*;


total_lossE{�@

error_R�:J?

learning_rate_1�<�7e�I       6%�	qG���A�'*;


total_loss�[�@

error_R�[G?

learning_rate_1�<�7�8�%I       6%�	��G���A�'*;


total_lossx�@

error_Rs�Y?

learning_rate_1�<�7Ӡy�I       6%�	m%G���A�'*;


total_loss�{�@

error_RE�`?

learning_rate_1�<�7����I       6%�	+kG���A�'*;


total_lossE�@

error_R��P?

learning_rate_1�<�7���XI       6%�	��G���A�'*;


total_loss߶�@

error_RV�E?

learning_rate_1�<�7;B�I       6%�	��G���A�'*;


total_lossx|�@

error_RT�J?

learning_rate_1�<�79D�5I       6%�	�@G���A�'*;


total_loss|Ÿ@

error_REl?

learning_rate_1�<�7���I       6%�	��G���A�'*;


total_lossX4�@

error_R�9H?

learning_rate_1�<�7B�q�I       6%�	;�G���A�'*;


total_lossT��@

error_RHyZ?

learning_rate_1�<�7��ĉI       6%�	�G���A�'*;


total_loss!�@

error_RL�E?

learning_rate_1�<�7Q�I       6%�	�WG���A�'*;


total_loss�`�@

error_R��Q?

learning_rate_1�<�7�_wI       6%�	n�G���A�'*;


total_loss��@

error_R��K?

learning_rate_1�<�7WˡZI       6%�	��G���A�'*;


total_lossf�A@

error_R�SG?

learning_rate_1�<�7�Nb4I       6%�	�G���A�'*;


total_loss��@

error_RT�k?

learning_rate_1�<�7,�.�I       6%�	bG���A�'*;


total_loss�Ӝ@

error_R��Q?

learning_rate_1�<�7r�nI       6%�	�G���A�'*;


total_loss���@

error_R_�G?

learning_rate_1�<�7b1>:I       6%�	]�G���A�'*;


total_loss���@

error_R��O?

learning_rate_1�<�7�C��I       6%�	T<G���A�'*;


total_loss��@

error_R}�Q?

learning_rate_1�<�7�z�uI       6%�	΀G���A�'*;


total_loss6u�@

error_R�Kl?

learning_rate_1�<�7��I       6%�	h�G���A�'*;


total_loss~�@

error_Ra|R?

learning_rate_1�<�7����I       6%�	$G���A�'*;


total_lossX߳@

error_R��C?

learning_rate_1�<�7'7FI       6%�	`XG���A�'*;


total_loss���@

error_R1�I?

learning_rate_1�<�7�T��I       6%�	��G���A�'*;


total_losshA�@

error_RׯF?

learning_rate_1�<�7�y�I       6%�	�G���A�'*;


total_loss~,�@

error_R`�E?

learning_rate_1�<�7δf�I       6%�	sG���A�'*;


total_loss�@

error_R��>?

learning_rate_1�<�7��
I       6%�	��G���A�'*;


total_loss�ˑ@

error_R{VZ?

learning_rate_1�<�7���I       6%�	�LG���A�'*;


total_loss-�@

error_R1[?

learning_rate_1�<�7��c�I       6%�	��G���A�'*;


total_loss}i�@

error_R\oT?

learning_rate_1�<�7���eI       6%�	� G���A�'*;


total_loss���@

error_R7�@?

learning_rate_1�<�7CA�I       6%�	�X G���A�'*;


total_lossq?�@

error_R�7?

learning_rate_1�<�7�/�I       6%�	� G���A�'*;


total_lossq��@

error_R��[?

learning_rate_1�<�7�r��I       6%�	v!G���A�'*;


total_lossf�t@

error_R��e?

learning_rate_1�<�7��X�I       6%�	%p!G���A�'*;


total_lossx��@

error_RF	O?

learning_rate_1�<�7J��I       6%�	w�!G���A�'*;


total_loss�ɰ@

error_R��c?

learning_rate_1�<�7�/��I       6%�	-"G���A�'*;


total_loss���@

error_R}M?

learning_rate_1�<�7I��I       6%�	}w"G���A�'*;


total_loss�$ A

error_R�T?

learning_rate_1�<�7�{5@I       6%�	��"G���A�'*;


total_loss]��@

error_R�N?

learning_rate_1�<�7|m��I       6%�	?,#G���A�'*;


total_loss�FA

error_Ra�B?

learning_rate_1�<�7�K�I       6%�	tz#G���A�'*;


total_loss\*�@

error_R�Kn?

learning_rate_1�<�7���6I       6%�	��#G���A�'*;


total_loss�+�@

error_RA2C?

learning_rate_1�<�7G��PI       6%�	�-$G���A�'*;


total_loss�M�@

error_R:JI?

learning_rate_1�<�7�ޅ I       6%�	P�$G���A�'*;


total_lossѦ@

error_R�L?

learning_rate_1�<�7ݠT�I       6%�	�%G���A�'*;


total_loss�(A

error_R�NC?

learning_rate_1�<�7Ȃ��I       6%�	~^%G���A�'*;


total_losse\�@

error_Rn)B?

learning_rate_1�<�7��I       6%�	�%G���A�'*;


total_loss�x�@

error_R8�@?

learning_rate_1�<�7Xt'I       6%�	{&G���A�'*;


total_loss�ϸ@

error_R�dS?

learning_rate_1�<�7��XSI       6%�	�m&G���A�'*;


total_loss\/�@

error_RJ�Y?

learning_rate_1�<�7)���I       6%�	��&G���A�'*;


total_losse��@

error_R��X?

learning_rate_1�<�7�z�+I       6%�	�'G���A�'*;


total_loss�@

error_R��E?

learning_rate_1�<�7G���I       6%�	�i'G���A�'*;


total_loss�_!A

error_Rq�N?

learning_rate_1�<�7��\�I       6%�	�(G���A�'*;


total_loss֭�@

error_RL�V?

learning_rate_1�<�7�'��I       6%�	�f(G���A�'*;


total_loss���@

error_R{FN?

learning_rate_1�<�7u��I       6%�	��(G���A�'*;


total_loss6C�@

error_R�zc?

learning_rate_1�<�70bI       6%�	�-)G���A�'*;


total_loss"�@

error_R ;R?

learning_rate_1�<�7oM��I       6%�	=�)G���A�'*;


total_loss
q�@

error_RE?

learning_rate_1�<�7 p��I       6%�		�)G���A�'*;


total_loss�!�@

error_R�S?

learning_rate_1�<�7�,��I       6%�	�7*G���A�'*;


total_loss*�R@

error_R�WB?

learning_rate_1�<�7�ܜI       6%�	X�*G���A�'*;


total_loss|��@

error_RlAI?

learning_rate_1�<�7�]�I       6%�	)�*G���A�'*;


total_losss��@

error_R��M?

learning_rate_1�<�7�=�HI       6%�	�6+G���A�'*;


total_loss5��@

error_R��P?

learning_rate_1�<�7��[I       6%�	t�+G���A�'*;


total_loss̦�@

error_R�I?

learning_rate_1�<�7��:�I       6%�	��+G���A�'*;


total_loss�L�@

error_R�G]?

learning_rate_1�<�7��uI       6%�	y,G���A�'*;


total_loss�;�@

error_R��M?

learning_rate_1�<�7T���I       6%�	�l,G���A�'*;


total_loss���@

error_R�E?

learning_rate_1�<�7�'Z I       6%�	��,G���A�'*;


total_loss��@

error_R�6F?

learning_rate_1�<�7�Q�I       6%�	�-G���A�'*;


total_loss,��@

error_RJN5?

learning_rate_1�<�7%�fhI       6%�	}U-G���A�'*;


total_loss��@

error_R�pE?

learning_rate_1�<�7���I       6%�	h�-G���A�'*;


total_loss���@

error_RW�@?

learning_rate_1�<�7'+�I       6%�	G�-G���A�'*;


total_lossZƳ@

error_RkU?

learning_rate_1�<�7Lr��I       6%�	�=.G���A�'*;


total_lossqY�@

error_R�F??

learning_rate_1�<�7-���I       6%�	X�.G���A�'*;


total_loss;Ż@

error_R�L?

learning_rate_1�<�7��I       6%�	��.G���A�'*;


total_loss���@

error_R�i??

learning_rate_1�<�7�6�I       6%�	8/G���A�'*;


total_lossfi�@

error_RoL?

learning_rate_1�<�7>�;�I       6%�	�c/G���A�'*;


total_loss���@

error_R �O?

learning_rate_1�<�7�@�I       6%�	J�/G���A�'*;


total_loss�!�@

error_R�_P?

learning_rate_1�<�7��_�I       6%�	v�/G���A�'*;


total_loss@'�@

error_R�M?

learning_rate_1�<�7M��I       6%�	�70G���A�'*;


total_loss��@

error_Rx�V?

learning_rate_1�<�7� ��I       6%�	E0G���A�'*;


total_loss�r�@

error_R[gT?

learning_rate_1�<�7��z�I       6%�	��0G���A�'*;


total_loss���@

error_R
P?

learning_rate_1�<�7A�gBI       6%�	�1G���A�'*;


total_loss�#�@

error_R��M?

learning_rate_1�<�7�U��I       6%�	�U1G���A�'*;


total_loss���@

error_R��??

learning_rate_1�<�7ߏґI       6%�	-�1G���A�'*;


total_lossܫ�@

error_R�uL?

learning_rate_1�<�7���.I       6%�	��1G���A�'*;


total_loss��|@

error_R�RH?

learning_rate_1�<�7g6�(I       6%�	{=2G���A�'*;


total_loss���@

error_R�ZA?

learning_rate_1�<�73��bI       6%�	)�2G���A�'*;


total_loss*f�@

error_RM�=?

learning_rate_1�<�7M���I       6%�	^�2G���A�'*;


total_loss�ҝ@

error_R=�C?

learning_rate_1�<�7e�6�I       6%�	�3G���A�'*;


total_loss� �@

error_R��P?

learning_rate_1�<�7T�FI       6%�	�W3G���A�'*;


total_loss��@

error_R�W9?

learning_rate_1�<�7I�A�I       6%�	�3G���A�'*;


total_loss\�@

error_Rf?

learning_rate_1�<�7��6I       6%�	[�3G���A�'*;


total_loss���@

error_RQR0?

learning_rate_1�<�7�dI       6%�	�(4G���A�'*;


total_loss���@

error_RR:f?

learning_rate_1�<�7�ӵI       6%�	Gm4G���A�'*;


total_loss�v�@

error_R,�??

learning_rate_1�<�70H0I       6%�	H�4G���A�'*;


total_loss�y[@

error_R;W?

learning_rate_1�<�7��hfI       6%�	�4G���A�'*;


total_loss��@

error_R��C?

learning_rate_1�<�7~�+I       6%�	�45G���A�'*;


total_loss�R�@

error_R��A?

learning_rate_1�<�7���I       6%�	�x5G���A�'*;


total_loss<s�@

error_RZ�^?

learning_rate_1�<�7#�DI       6%�	l�5G���A�'*;


total_loss�d�@

error_RW$[?

learning_rate_1�<�7i�aI       6%�	z�5G���A�'*;


total_loss�%�@

error_R�`S?

learning_rate_1�<�7� �I       6%�	�@6G���A�'*;


total_lossR��@

error_RM?

learning_rate_1�<�7�v�I       6%�	܃6G���A�'*;


total_lossʣ�@

error_R��X?

learning_rate_1�<�7 ��I       6%�	�6G���A�'*;


total_loss��r@

error_R�E?

learning_rate_1�<�7B�I       6%�	�
7G���A�'*;


total_loss�
�@

error_R]�<?

learning_rate_1�<�7=�F\I       6%�	�M7G���A�'*;


total_loss�@

error_R��a?

learning_rate_1�<�7��Z"I       6%�	�7G���A�'*;


total_lossh��@

error_Rn�A?

learning_rate_1�<�7��\�I       6%�	��7G���A�(*;


total_loss�v�@

error_R��b?

learning_rate_1�<�7��L5I       6%�	a-8G���A�(*;


total_loss4�@

error_RùQ?

learning_rate_1�<�7T��I       6%�	ov8G���A�(*;


total_lossJ:�@

error_RfRU?

learning_rate_1�<�7l��I       6%�	U�8G���A�(*;


total_loss�� A

error_R$�\?

learning_rate_1�<�7B���I       6%�	<9G���A�(*;


total_loss�k�@

error_R�sU?

learning_rate_1�<�7$���I       6%�	�Q9G���A�(*;


total_loss�K�@

error_R�DF?

learning_rate_1�<�7��m�I       6%�	Ś9G���A�(*;


total_loss r�@

error_R)\]?

learning_rate_1�<�7��eI       6%�	n�9G���A�(*;


total_loss�x�@

error_RZ~T?

learning_rate_1�<�7}�I       6%�	*:G���A�(*;


total_lossZ��@

error_RI�<?

learning_rate_1�<�7���I       6%�	�s:G���A�(*;


total_lossx�@

error_Rq�N?

learning_rate_1�<�7	Hv'I       6%�	d�:G���A�(*;


total_lossv�@

error_R��I?

learning_rate_1�<�7�A�I       6%�	�;G���A�(*;


total_loss<��@

error_R�[?

learning_rate_1�<�7�{,I       6%�	qM;G���A�(*;


total_loss��@

error_R��T?

learning_rate_1�<�7X��LI       6%�	9�;G���A�(*;


total_loss:ԍ@

error_R�nO?

learning_rate_1�<�7�eu8I       6%�	�;G���A�(*;


total_loss��@

error_R��[?

learning_rate_1�<�7}U�DI       6%�	�.<G���A�(*;


total_loss3�@

error_R{�Z?

learning_rate_1�<�7nI       6%�	sz<G���A�(*;


total_loss ��@

error_R� I?

learning_rate_1�<�7SW@I       6%�	��<G���A�(*;


total_loss��@

error_RI?

learning_rate_1�<�7�C�UI       6%�	�=G���A�(*;


total_loss�°@

error_R8�V?

learning_rate_1�<�7�LpI       6%�	3[=G���A�(*;


total_lossv��@

error_ReMM?

learning_rate_1�<�7��WI       6%�	��=G���A�(*;


total_loss�A

error_RM=S?

learning_rate_1�<�77�V+I       6%�	��=G���A�(*;


total_loss�@

error_R@_?

learning_rate_1�<�7,H�I       6%�	B)>G���A�(*;


total_loss���@

error_R��Q?

learning_rate_1�<�7�˶_I       6%�	�~>G���A�(*;


total_loss@�A

error_R��H?

learning_rate_1�<�7u�b�I       6%�	��>G���A�(*;


total_loss˰@

error_R7TF?

learning_rate_1�<�7-ԞI       6%�	r?G���A�(*;


total_loss솳@

error_RTk7?

learning_rate_1�<�7B��I       6%�	�`?G���A�(*;


total_lossv6�@

error_R�`?

learning_rate_1�<�7vk6<I       6%�	��?G���A�(*;


total_loss�1�@

error_R��??

learning_rate_1�<�7Ll��I       6%�	��?G���A�(*;


total_lossdjd@

error_RF=?

learning_rate_1�<�7�%�I       6%�	�6@G���A�(*;


total_loss�c�@

error_R� 7?

learning_rate_1�<�7K���I       6%�	�z@G���A�(*;


total_loss�ҟ@

error_R�WO?

learning_rate_1�<�7�z߾I       6%�	m�@G���A�(*;


total_lossQO�@

error_R��m?

learning_rate_1�<�7�>��I       6%�	m AG���A�(*;


total_lossr'�@

error_RD?

learning_rate_1�<�7
:ؕI       6%�	�AAG���A�(*;


total_loss��@

error_R�P?

learning_rate_1�<�7�#��I       6%�	مAG���A�(*;


total_lossde�@

error_R�%L?

learning_rate_1�<�7��F,I       6%�	��AG���A�(*;


total_loss:��@

error_RχC?

learning_rate_1�<�7�)I       6%�	BG���A�(*;


total_loss��@

error_RE�O?

learning_rate_1�<�7��RI       6%�	�MBG���A�(*;


total_loss���@

error_R��\?

learning_rate_1�<�7�EvI       6%�	v�BG���A�(*;


total_loss���@

error_Rh8O?

learning_rate_1�<�7�+��I       6%�	��BG���A�(*;


total_loss���@

error_R��M?

learning_rate_1�<�7���I       6%�	1CG���A�(*;


total_loss���@

error_R\?W?

learning_rate_1�<�7Q�O�I       6%�	CYCG���A�(*;


total_loss��@

error_Rd�E?

learning_rate_1�<�7�U�sI       6%�	��CG���A�(*;


total_lossc�@

error_R	�:?

learning_rate_1�<�7�q��I       6%�	��CG���A�(*;


total_lossߔF@

error_R�rg?

learning_rate_1�<�7Մ�@I       6%�	.$DG���A�(*;


total_loss��@

error_Ra�J?

learning_rate_1�<�7��/�I       6%�	�jDG���A�(*;


total_loss���@

error_R,0T?

learning_rate_1�<�7�@o�I       6%�	j�DG���A�(*;


total_lossbڝ@

error_R�M?

learning_rate_1�<�7����I       6%�	��DG���A�(*;


total_loss[�@

error_RE�V?

learning_rate_1�<�7��(=I       6%�	�9EG���A�(*;


total_lossS�@

error_RnV?

learning_rate_1�<�7�!�I       6%�	�~EG���A�(*;


total_loss���@

error_R��[?

learning_rate_1�<�7�P�I       6%�	=�EG���A�(*;


total_loss���@

error_R��P?

learning_rate_1�<�7󠴃I       6%�	�FG���A�(*;


total_loss�gp@

error_R$S?

learning_rate_1�<�7���I       6%�	2DFG���A�(*;


total_lossI�@

error_R��V?

learning_rate_1�<�7�:B(I       6%�	2�FG���A�(*;


total_losso��@

error_R�H?

learning_rate_1�<�7���I       6%�	f�FG���A�(*;


total_lossz�\@

error_RR�V?

learning_rate_1�<�7pu��I       6%�	�GG���A�(*;


total_loss�a�@

error_R��I?

learning_rate_1�<�71�$I       6%�	5NGG���A�(*;


total_loss��@

error_R�L`?

learning_rate_1�<�7v��I       6%�	��GG���A�(*;


total_loss�7�@

error_RJjO?

learning_rate_1�<�7\x޳I       6%�	4�GG���A�(*;


total_loss��@

error_R�zC?

learning_rate_1�<�7�܂I       6%�	'9HG���A�(*;


total_loss�i@

error_R6"J?

learning_rate_1�<�7l&��I       6%�	�HG���A�(*;


total_loss��@

error_RS�??

learning_rate_1�<�7� I       6%�	�HG���A�(*;


total_lossq �@

error_RlL?

learning_rate_1�<�7A�aI       6%�	aIG���A�(*;


total_lossQA�@

error_R��O?

learning_rate_1�<�7]'N�I       6%�	1XIG���A�(*;


total_loss�KA

error_R�qK?

learning_rate_1�<�7�m�I       6%�	њIG���A�(*;


total_lossQ|�@

error_R�7<?

learning_rate_1�<�7R.�jI       6%�	��IG���A�(*;


total_loss���@

error_R�K?

learning_rate_1�<�7����I       6%�	�%JG���A�(*;


total_loss��@

error_R!Y?

learning_rate_1�<�7 �wI       6%�	ViJG���A�(*;


total_loss��_@

error_Rv�V?

learning_rate_1�<�7By�lI       6%�	��JG���A�(*;


total_loss��@

error_R�28?

learning_rate_1�<�7��=<I       6%�	m�JG���A�(*;


total_loss Ӳ@

error_Rj$a?

learning_rate_1�<�7�/eRI       6%�	5KG���A�(*;


total_loss���@

error_R�eb?

learning_rate_1�<�7y[CWI       6%�	�wKG���A�(*;


total_loss�E�@

error_R�`>?

learning_rate_1�<�7��X:I       6%�	��KG���A�(*;


total_lossa�@

error_R��[?

learning_rate_1�<�7(��I       6%�	� LG���A�(*;


total_loss/�u@

error_R��L?

learning_rate_1�<�7&_0�I       6%�	�BLG���A�(*;


total_loss~UA

error_RϠU?

learning_rate_1�<�7�y�I       6%�	�LG���A�(*;


total_loss��@

error_RW	Q?

learning_rate_1�<�7�a�fI       6%�	��LG���A�(*;


total_loss&��@

error_R</R?

learning_rate_1�<�7C�rwI       6%�	�MG���A�(*;


total_loss��@

error_RZ�>?

learning_rate_1�<�7YS��I       6%�	,OMG���A�(*;


total_lossiG�@

error_RXD?

learning_rate_1�<�7'N$"I       6%�	:�MG���A�(*;


total_loss�$�@

error_R� D?

learning_rate_1�<�7c���I       6%�	��MG���A�(*;


total_lossӖ�@

error_R�`?

learning_rate_1�<�7���I       6%�	�NG���A�(*;


total_loss!A

error_RߣZ?

learning_rate_1�<�7M�[VI       6%�	VNG���A�(*;


total_loss�~@

error_RXE?

learning_rate_1�<�7�㯶I       6%�	��NG���A�(*;


total_loss v�@

error_R�nO?

learning_rate_1�<�7&�	I       6%�	��NG���A�(*;


total_loss2��@

error_R�!\?

learning_rate_1�<�7v��I       6%�	�OG���A�(*;


total_loss���@

error_RC�T?

learning_rate_1�<�7���wI       6%�	�XOG���A�(*;


total_loss���@

error_ReT?

learning_rate_1�<�7�^��I       6%�	+�OG���A�(*;


total_loss,��@

error_R�R?

learning_rate_1�<�7��׀I       6%�	��OG���A�(*;


total_lossE�@

error_R$�E?

learning_rate_1�<�7���I       6%�	�PG���A�(*;


total_loss�a�@

error_R8^A?

learning_rate_1�<�7ŵ��I       6%�	._PG���A�(*;


total_lossW2�@

error_RWB?

learning_rate_1�<�7f;mcI       6%�	��PG���A�(*;


total_loss�9A

error_R�*??

learning_rate_1�<�7�-
I       6%�	��PG���A�(*;


total_loss�ՠ@

error_R&Y9?

learning_rate_1�<�7�#[�I       6%�	A"QG���A�(*;


total_loss�A

error_R�kQ?

learning_rate_1�<�7`��I       6%�	�bQG���A�(*;


total_loss�ݮ@

error_R�=L?

learning_rate_1�<�7*u2#I       6%�	i�QG���A�(*;


total_loss:F�@

error_Rf�P?

learning_rate_1�<�7�V�QI       6%�	��QG���A�(*;


total_loss�r�@

error_R\�P?

learning_rate_1�<�7f�I       6%�	�)RG���A�(*;


total_loss��@

error_RC�a?

learning_rate_1�<�72(��I       6%�	�jRG���A�(*;


total_loss#mA

error_R�N?

learning_rate_1�<�7����I       6%�	��RG���A�(*;


total_loss���@

error_R�<P?

learning_rate_1�<�7�C`I       6%�	��RG���A�(*;


total_lossڀ�@

error_R��E?

learning_rate_1�<�7���I       6%�	�,SG���A�(*;


total_loss�E�@

error_R��D?

learning_rate_1�<�7��I       6%�	BnSG���A�(*;


total_loss��@

error_R�<@?

learning_rate_1�<�7Z5�I       6%�	ҶSG���A�(*;


total_loss��@

error_R=!`?

learning_rate_1�<�7��rI       6%�	�TG���A�(*;


total_loss�I�@

error_R\S?

learning_rate_1�<�7�y�I       6%�	�LTG���A�(*;


total_loss�Ҥ@

error_RsZ?

learning_rate_1�<�7[��OI       6%�	�TG���A�(*;


total_loss�N�@

error_Rr�\?

learning_rate_1�<�78/�xI       6%�	��TG���A�(*;


total_loss���@

error_R��<?

learning_rate_1�<�7�C3I       6%�	UG���A�(*;


total_loss��@

error_R��F?

learning_rate_1�<�7��clI       6%�	QXUG���A�(*;


total_lossb'�@

error_R��V?

learning_rate_1�<�7�lB6I       6%�	i�UG���A�(*;


total_lossʹ�@

error_Rqs=?

learning_rate_1�<�7h['�I       6%�	��UG���A�(*;


total_loss G�@

error_R��[?

learning_rate_1�<�76^7�I       6%�	�VG���A�(*;


total_lossT̫@

error_RqXR?

learning_rate_1�<�7�OSI       6%�	�_VG���A�(*;


total_lossSݤ@

error_R�rX?

learning_rate_1�<�7o+հI       6%�	4�VG���A�(*;


total_loss�p�@

error_RZK?

learning_rate_1�<�7��s�I       6%�	2�VG���A�(*;


total_loss��@

error_Rrp>?

learning_rate_1�<�7�{�(I       6%�	�!WG���A�(*;


total_loss?	A

error_R��O?

learning_rate_1�<�7�V4I       6%�	bWG���A�(*;


total_loss�{�@

error_R`X?

learning_rate_1�<�7�O%gI       6%�	��WG���A�(*;


total_lossl��@

error_R
RL?

learning_rate_1�<�7����I       6%�	��WG���A�(*;


total_loss�'�@

error_R��O?

learning_rate_1�<�7H5cI       6%�	�'XG���A�(*;


total_losshٞ@

error_R8)D?

learning_rate_1�<�7w~I       6%�	|XG���A�(*;


total_lossد�@

error_RO_K?

learning_rate_1�<�71��zI       6%�	��XG���A�(*;


total_loss.�@

error_RņY?

learning_rate_1�<�7��\I       6%�	�YG���A�(*;


total_lossZI�@

error_Rf�<?

learning_rate_1�<�7}Q��I       6%�	�@YG���A�(*;


total_loss���@

error_Rh A?

learning_rate_1�<�7M���I       6%�	$�YG���A�(*;


total_lossD��@

error_R�=?

learning_rate_1�<�7��#I       6%�	B�YG���A�(*;


total_loss���@

error_R@u@?

learning_rate_1�<�7)٥�I       6%�	/ZG���A�(*;


total_losszO�@

error_RQ�Z?

learning_rate_1�<�7��x�I       6%�	�JZG���A�)*;


total_loss�A

error_R(�H?

learning_rate_1�<�7�{�I       6%�	+�ZG���A�)*;


total_loss��@

error_R&W?

learning_rate_1�<�7H�^
I       6%�	��ZG���A�)*;


total_losss+�@

error_R7�O?

learning_rate_1�<�7����I       6%�	�"[G���A�)*;


total_loss�@

error_R�0P?

learning_rate_1�<�7�.~I       6%�	sn[G���A�)*;


total_lossˠ@

error_R�SO?

learning_rate_1�<�7�бSI       6%�	ڶ[G���A�)*;


total_loss$�@

error_Rf�D?

learning_rate_1�<�7I:B�I       6%�	��[G���A�)*;


total_loss�t@

error_R�-8?

learning_rate_1�<�7�[_�I       6%�	�B\G���A�)*;


total_loss�E�@

error_R�J\?

learning_rate_1�<�7��x�I       6%�	ǁ\G���A�)*;


total_loss�&�@

error_R�-3?

learning_rate_1�<�7&Q�I       6%�	��\G���A�)*;


total_loss�e�@

error_R�mB?

learning_rate_1�<�7<��I       6%�	3]G���A�)*;


total_lossT�A

error_RʡC?

learning_rate_1�<�7��EI       6%�	�N]G���A�)*;


total_loss�H�@

error_R�@?

learning_rate_1�<�7ZP�I       6%�	ƕ]G���A�)*;


total_loss�s�@

error_R��L?

learning_rate_1�<�7e���I       6%�	��]G���A�)*;


total_loss��w@

error_Rl9F?

learning_rate_1�<�7�x<�I       6%�	3)^G���A�)*;


total_loss
��@

error_R�-H?

learning_rate_1�<�7�)�I       6%�	�k^G���A�)*;


total_loss|��@

error_R�k=?

learning_rate_1�<�7B�BI       6%�	ڭ^G���A�)*;


total_loss���@

error_R{tU?

learning_rate_1�<�7���I       6%�	Y�^G���A�)*;


total_loss]��@

error_RlkW?

learning_rate_1�<�7�&�pI       6%�	0_G���A�)*;


total_loss ��@

error_R�Q?

learning_rate_1�<�7���
I       6%�	�q_G���A�)*;


total_loss��@

error_R�-@?

learning_rate_1�<�7���0I       6%�	��_G���A�)*;


total_lossmԼ@

error_RT�H?

learning_rate_1�<�7N�_�I       6%�	�_G���A�)*;


total_loss��@

error_R��J?

learning_rate_1�<�7�(��I       6%�	�;`G���A�)*;


total_loss��%A

error_R8�V?

learning_rate_1�<�7�L�I       6%�	p`G���A�)*;


total_loss��@

error_R�ha?

learning_rate_1�<�7�]:I       6%�	��`G���A�)*;


total_lossM0�@

error_R�$^?

learning_rate_1�<�7 ��vI       6%�	�aG���A�)*;


total_lossA

error_RVFU?

learning_rate_1�<�7���I       6%�	�FaG���A�)*;


total_loss���@

error_R��F?

learning_rate_1�<�7��yI       6%�	q�aG���A�)*;


total_loss��@

error_RʕI?

learning_rate_1�<�7d+h�I       6%�	��aG���A�)*;


total_lossL��@

error_R�P<?

learning_rate_1�<�7��I       6%�	�bG���A�)*;


total_loss��@

error_RRdF?

learning_rate_1�<�7���I       6%�	�KbG���A�)*;


total_loss$
�@

error_R�jM?

learning_rate_1�<�7Mj�I       6%�	ԌbG���A�)*;


total_loss<��@

error_Rڵ\?

learning_rate_1�<�7�� �I       6%�	�bG���A�)*;


total_losse�@

error_RN�P?

learning_rate_1�<�7�f��I       6%�	�cG���A�)*;


total_lossqj�@

error_R̒\?

learning_rate_1�<�7.�ǹI       6%�	uLcG���A�)*;


total_loss��P@

error_RRH?

learning_rate_1�<�7�yGI       6%�	/�cG���A�)*;


total_loss&M�@

error_REZI?

learning_rate_1�<�7%�N'I       6%�	�cG���A�)*;


total_lossJ��@

error_R�&N?

learning_rate_1�<�7�37>I       6%�	=dG���A�)*;


total_loss�Ǒ@

error_R��O?

learning_rate_1�<�7�I�DI       6%�	�YdG���A�)*;


total_lossO��@

error_R��X?

learning_rate_1�<�7�¯I       6%�	��dG���A�)*;


total_loss���@

error_R��M?

learning_rate_1�<�7��6�I       6%�	��dG���A�)*;


total_loss��@

error_R�gN?

learning_rate_1�<�7��{�I       6%�	�#eG���A�)*;


total_lossA��@

error_R_�S?

learning_rate_1�<�7;�vI       6%�	�feG���A�)*;


total_loss͉�@

error_R�S?

learning_rate_1�<�7���I       6%�	b�eG���A�)*;


total_lossw��@

error_RjX?

learning_rate_1�<�7S>�II       6%�	��eG���A�)*;


total_loss���@

error_Rt�Z?

learning_rate_1�<�7d[Y�I       6%�	u7fG���A�)*;


total_lossNL�@

error_R��L?

learning_rate_1�<�74X9�I       6%�	�|fG���A�)*;


total_loss.�@

error_ROBM?

learning_rate_1�<�7X�jI       6%�	�fG���A�)*;


total_lossh�@

error_R��L?

learning_rate_1�<�7��#tI       6%�	�gG���A�)*;


total_lossԵ�@

error_R�|A?

learning_rate_1�<�7�ԚI       6%�	�PgG���A�)*;


total_loss�?A

error_R��J?

learning_rate_1�<�7 �.I       6%�	��gG���A�)*;


total_loss�ĸ@

error_RZ�M?

learning_rate_1�<�7~%4I       6%�	��gG���A�)*;


total_loss��@

error_R��J?

learning_rate_1�<�7;�]�I       6%�	B"hG���A�)*;


total_loss�^@

error_R��K?

learning_rate_1�<�7KYI       6%�	�uhG���A�)*;


total_loss|ٲ@

error_R�$Q?

learning_rate_1�<�71�gI       6%�	m�hG���A�)*;


total_loss�؈@

error_R`�A?

learning_rate_1�<�7���&I       6%�	iG���A�)*;


total_loss�@�@

error_R�;O?

learning_rate_1�<�7-�XJI       6%�	bSiG���A�)*;


total_loss���@

error_R4�J?

learning_rate_1�<�7_�h�I       6%�	
�iG���A�)*;


total_loss���@

error_R��J?

learning_rate_1�<�7r�PaI       6%�	x�iG���A�)*;


total_loss�RA

error_R�=?

learning_rate_1�<�7����I       6%�	RjG���A�)*;


total_loss��@

error_R��N?

learning_rate_1�<�7M$0I       6%�	�`jG���A�)*;


total_lossD.{@

error_R�N?

learning_rate_1�<�7��)�I       6%�	��jG���A�)*;


total_lossN�@

error_R��B?

learning_rate_1�<�7;���I       6%�	�jG���A�)*;


total_loss��A

error_RI�T?

learning_rate_1�<�7ǕݜI       6%�	w#kG���A�)*;


total_loss���@

error_RaU?

learning_rate_1�<�7�� I       6%�	�ckG���A�)*;


total_lossH�@

error_R��[?

learning_rate_1�<�7�(��I       6%�	��kG���A�)*;


total_loss���@

error_R&�I?

learning_rate_1�<�7�g4rI       6%�	&�kG���A�)*;


total_loss�c�@

error_RŅ`?

learning_rate_1�<�7&�'uI       6%�	�2lG���A�)*;


total_loss���@

error_RC�A?

learning_rate_1�<�7���<I       6%�	�lG���A�)*;


total_loss]zA

error_R)??

learning_rate_1�<�7�Y��I       6%�	��lG���A�)*;


total_loss���@

error_R�1N?

learning_rate_1�<�7,�VI       6%�	mG���A�)*;


total_lossÆ�@

error_R�;?

learning_rate_1�<�7�y*�I       6%�	�ImG���A�)*;


total_loss��@

error_R�Q?

learning_rate_1�<�7fb̝I       6%�	�mG���A�)*;


total_loss���@

error_R;-R?

learning_rate_1�<�7n�7I       6%�	1�mG���A�)*;


total_lossˋ�@

error_RRP>?

learning_rate_1�<�7qK�@I       6%�	=nG���A�)*;


total_loss#F�@

error_R[a?

learning_rate_1�<�7���I       6%�	x_nG���A�)*;


total_lossv�@

error_R�L?

learning_rate_1�<�7Ř�BI       6%�	˨nG���A�)*;


total_loss�T�@

error_RC�E?

learning_rate_1�<�7"I~lI       6%�	��nG���A�)*;


total_lossλ@

error_RarY?

learning_rate_1�<�7���6I       6%�	�:oG���A�)*;


total_loss�r�@

error_R]�4?

learning_rate_1�<�7�+��I       6%�	��oG���A�)*;


total_loss8O�@

error_R�`F?

learning_rate_1�<�7j��qI       6%�	m�oG���A�)*;


total_loss�-�@

error_R��[?

learning_rate_1�<�7�`�I       6%�	�pG���A�)*;


total_lossVƥ@

error_Ra�>?

learning_rate_1�<�7��-I       6%�	�QpG���A�)*;


total_loss_P�@

error_RT�9?

learning_rate_1�<�7���?I       6%�	�pG���A�)*;


total_loss8��@

error_R�!W?

learning_rate_1�<�73\I       6%�	��pG���A�)*;


total_lossJŜ@

error_R�}F?

learning_rate_1�<�7���I       6%�	�qG���A�)*;


total_loss! �@

error_Rl<L?

learning_rate_1�<�7�o�I       6%�	�YqG���A�)*;


total_lossm��@

error_R#C?

learning_rate_1�<�7�?w.I       6%�	Z�qG���A�)*;


total_loss��@

error_R��P?

learning_rate_1�<�7��vFI       6%�	�qG���A�)*;


total_loss���@

error_R*/R?

learning_rate_1�<�7��I       6%�	1rG���A�)*;


total_loss �@

error_R�`?

learning_rate_1�<�7~Y�8I       6%�	�^rG���A�)*;


total_loss��@

error_R_zH?

learning_rate_1�<�7E� �I       6%�	3�rG���A�)*;


total_loss�=�@

error_R 5d?

learning_rate_1�<�7�I       6%�	��rG���A�)*;


total_loss��@

error_R��U?

learning_rate_1�<�7W���I       6%�	�sG���A�)*;


total_lossE@�@

error_RҹC?

learning_rate_1�<�7#Ǎ[I       6%�	�dsG���A�)*;


total_loss#D�@

error_R��O?

learning_rate_1�<�7��(I       6%�	w�sG���A�)*;


total_lossX�@

error_R��W?

learning_rate_1�<�7v���I       6%�	E�sG���A�)*;


total_loss1U�@

error_R\[?

learning_rate_1�<�7uUI       6%�	JAtG���A�)*;


total_loss�΀@

error_R��J?

learning_rate_1�<�7�Sv�I       6%�	ǎtG���A�)*;


total_lossZ!�@

error_R�&M?

learning_rate_1�<�7_�^�I       6%�	��tG���A�)*;


total_lossP�@

error_R͎W?

learning_rate_1�<�78��I       6%�	�uG���A�)*;


total_lossCٻ@

error_R�L?

learning_rate_1�<�7����I       6%�	U]uG���A�)*;


total_loss��@

error_R �R?

learning_rate_1�<�7Z�?-I       6%�	�uG���A�)*;


total_loss��A

error_RߓH?

learning_rate_1�<�7p��nI       6%�	�uG���A�)*;


total_loss�q�@

error_RD�R?

learning_rate_1�<�7��m�I       6%�	�)vG���A�)*;


total_loss��@

error_Rt�^?

learning_rate_1�<�7	���I       6%�	�pvG���A�)*;


total_lossC�@

error_R��E?

learning_rate_1�<�7Gn�I       6%�	d�vG���A�)*;


total_loss�0�@

error_R�PH?

learning_rate_1�<�7rfRJI       6%�	� wG���A�)*;


total_loss���@

error_R :S?

learning_rate_1�<�7�~EI       6%�	�DwG���A�)*;


total_loss�o�@

error_R8�Q?

learning_rate_1�<�7'SjvI       6%�	��wG���A�)*;


total_lossTJ�@

error_R!�R?

learning_rate_1�<�7�SK�I       6%�	��wG���A�)*;


total_loss�1�@

error_RD�Q?

learning_rate_1�<�7ľY�I       6%�	�xG���A�)*;


total_loss3��@

error_R�EU?

learning_rate_1�<�7<6A�I       6%�	�nxG���A�)*;


total_loss��@

error_R�NN?

learning_rate_1�<�7�m$�I       6%�	�xG���A�)*;


total_loss�ތ@

error_R��I?

learning_rate_1�<�7�n#I       6%�	yG���A�)*;


total_loss���@

error_RߊB?

learning_rate_1�<�7)kx�I       6%�	NRyG���A�)*;


total_loss;��@

error_R�
W?

learning_rate_1�<�7jAx�I       6%�	��yG���A�)*;


total_loss,@�@

error_RݢQ?

learning_rate_1�<�7��}I       6%�	��yG���A�)*;


total_loss4�A

error_R��X?

learning_rate_1�<�7�H��I       6%�	�"zG���A�)*;


total_loss�x�@

error_R!�Q?

learning_rate_1�<�7�Ϊ�I       6%�	jzG���A�)*;


total_loss��@

error_R�x??

learning_rate_1�<�7�&�I       6%�	h�zG���A�)*;


total_loss���@

error_RON?

learning_rate_1�<�7l!�yI       6%�	�zG���A�)*;


total_loss3��@

error_R��E?

learning_rate_1�<�7��0I       6%�	�0{G���A�)*;


total_loss?^@

error_R_iA?

learning_rate_1�<�7�n[I       6%�	�q{G���A�)*;


total_loss_�@

error_R��J?

learning_rate_1�<�7��}I       6%�	8�{G���A�)*;


total_loss��@

error_R�G?

learning_rate_1�<�7���xI       6%�	n�{G���A�)*;


total_loss���@

error_R\yS?

learning_rate_1�<�7M&�I       6%�	�:|G���A�)*;


total_lossQ�@

error_R/:P?

learning_rate_1�<�7{�SI       6%�	�}|G���A�)*;


total_loss���@

error_R��S?

learning_rate_1�<�7v�t�I       6%�	f�|G���A�**;


total_lossW�@

error_R�I?

learning_rate_1�<�7���)I       6%�	}G���A�**;


total_loss�)�@

error_R�_??

learning_rate_1�<�7����I       6%�	lL}G���A�**;


total_loss���@

error_R$�F?

learning_rate_1�<�7��7�I       6%�	��}G���A�**;


total_loss�(�@

error_R��>?

learning_rate_1�<�7$��DI       6%�	��}G���A�**;


total_loss�g�@

error_RhNe?

learning_rate_1�<�7���I       6%�	?1~G���A�**;


total_lossqI�@

error_Rs�C?

learning_rate_1�<�7�a?�I       6%�	�q~G���A�**;


total_lossrUr@

error_R�9?

learning_rate_1�<�77QII       6%�	O�~G���A�**;


total_loss�@

error_R�X?

learning_rate_1�<�7g��II       6%�	��~G���A�**;


total_lossHþ@

error_R�{>?

learning_rate_1�<�7J�&I       6%�	�7G���A�**;


total_loss���@

error_R,}??

learning_rate_1�<�7�ŊI       6%�	�wG���A�**;


total_loss�r�@

error_R��E?

learning_rate_1�<�7���I       6%�	l�G���A�**;


total_loss���@

error_RW}<?

learning_rate_1�<�7�ȠI       6%�	V�G���A�**;


total_loss���@

error_R�	]?

learning_rate_1�<�7���I       6%�	B�G���A�**;


total_lossn4A

error_REU?

learning_rate_1�<�7.VdI       6%�	&��G���A�**;


total_loss�և@

error_Rh�C?

learning_rate_1�<�7�TTI       6%�	2ƀG���A�**;


total_loss{��@

error_R35d?

learning_rate_1�<�70l�I       6%�	��G���A�**;


total_loss=�@

error_Rjf?

learning_rate_1�<�7�ΖI       6%�	BJ�G���A�**;


total_loss�C�@

error_RVQ?

learning_rate_1�<�7���;I       6%�	ې�G���A�**;


total_loss���@

error_R��Q?

learning_rate_1�<�7�["�I       6%�		؁G���A�**;


total_lossε�@

error_Rz�G?

learning_rate_1�<�7��XI       6%�	� �G���A�**;


total_loss�F�@

error_R�H?

learning_rate_1�<�7L���I       6%�	8f�G���A�**;


total_loss#<�@

error_R��Y?

learning_rate_1�<�7#�I       6%�	=��G���A�**;


total_loss{��@

error_RxGO?

learning_rate_1�<�7{��UI       6%�	�G���A�**;


total_lossI.�@

error_R��A?

learning_rate_1�<�7!5YI       6%�	�,�G���A�**;


total_lossN�@

error_RM�H?

learning_rate_1�<�7���I       6%�	�n�G���A�**;


total_loss�@

error_R�>T?

learning_rate_1�<�7�ƆI       6%�	A��G���A�**;


total_loss,n�@

error_RԁH?

learning_rate_1�<�7���I       6%�	���G���A�**;


total_loss�;�@

error_R��Z?

learning_rate_1�<�7�qb:I       6%�	(8�G���A�**;


total_loss�Mn@

error_R�G?

learning_rate_1�<�7c(߄I       6%�	Hz�G���A�**;


total_loss�g�@

error_Rq�N?

learning_rate_1�<�7����I       6%�	\��G���A�**;


total_loss�]�@

error_RTkL?

learning_rate_1�<�7L���I       6%�	y��G���A�**;


total_loss�R�@

error_R��L?

learning_rate_1�<�7�@%|I       6%�	�>�G���A�**;


total_loss�'�@

error_R�O?

learning_rate_1�<�7��4TI       6%�	��G���A�**;


total_loss��@

error_R��J?

learning_rate_1�<�7��o�I       6%�	�G���A�**;


total_loss���@

error_R�ER?

learning_rate_1�<�74q7I       6%�	k�G���A�**;


total_lossV:�@

error_R�gN?

learning_rate_1�<�7��i�I       6%�	�F�G���A�**;


total_loss���@

error_R��P?

learning_rate_1�<�7�8��I       6%�	���G���A�**;


total_loss�U�@

error_Rlt@?

learning_rate_1�<�7	TI       6%�	�ɆG���A�**;


total_loss���@

error_RE�O?

learning_rate_1�<�7�?+�I       6%�	�	�G���A�**;


total_loss ��@

error_R�C?

learning_rate_1�<�7쁡MI       6%�	K�G���A�**;


total_loss��@

error_R{�Y?

learning_rate_1�<�7 �$I       6%�	���G���A�**;


total_loss䑚@

error_R�K?

learning_rate_1�<�7�-2�I       6%�	�ӇG���A�**;


total_lossSl�@

error_R�sF?

learning_rate_1�<�7�ltI       6%�	��G���A�**;


total_lossCm�@

error_R�6O?

learning_rate_1�<�7�RrmI       6%�	�^�G���A�**;


total_loss_��@

error_R�!X?

learning_rate_1�<�70��I       6%�	�ĈG���A�**;


total_loss�P�@

error_Rv+A?

learning_rate_1�<�79+.I       6%�	�
�G���A�**;


total_loss��@

error_R�HY?

learning_rate_1�<�7���xI       6%�	�N�G���A�**;


total_lossB�@

error_R�Q?

learning_rate_1�<�7JMhI       6%�	ѕ�G���A�**;


total_loss��@

error_RTQK?

learning_rate_1�<�7��O�I       6%�	��G���A�**;


total_loss���@

error_R�<U?

learning_rate_1�<�7�e�I       6%�	N$�G���A�**;


total_loss�;�@

error_RnA?

learning_rate_1�<�7W�!gI       6%�	9i�G���A�**;


total_loss.��@

error_RhA_?

learning_rate_1�<�7�OXqI       6%�	ӭ�G���A�**;


total_loss&��@

error_R�@?

learning_rate_1�<�7���0I       6%�	��G���A�**;


total_lossq��@

error_R�NJ?

learning_rate_1�<�7��wI       6%�	�2�G���A�**;


total_loss��@

error_R�tO?

learning_rate_1�<�7]P��I       6%�	�x�G���A�**;


total_loss�SA

error_R8L\?

learning_rate_1�<�7����I       6%�	u��G���A�**;


total_loss�z�@

error_RW L?

learning_rate_1�<�7�2�!I       6%�	���G���A�**;


total_lossT��@

error_R�JW?

learning_rate_1�<�7W{��I       6%�	�>�G���A�**;


total_loss�A�@

error_R�U?

learning_rate_1�<�7���xI       6%�	@��G���A�**;


total_loss�Z�@

error_RTW?

learning_rate_1�<�7ܢ�I       6%�	��G���A�**;


total_loss}��@

error_R�PI?

learning_rate_1�<�7�MQDI       6%�	�-�G���A�**;


total_loss�UA

error_R�K?

learning_rate_1�<�7p�q1I       6%�	Kp�G���A�**;


total_loss�[,A

error_R�z@?

learning_rate_1�<�7O�;�I       6%�	{��G���A�**;


total_loss��A

error_R.P?

learning_rate_1�<�7�EsI       6%�	1��G���A�**;


total_loss-��@

error_R��N?

learning_rate_1�<�7��I       6%�	�9�G���A�**;


total_loss���@

error_R,t^?

learning_rate_1�<�7<W��I       6%�	y�G���A�**;


total_loss�7�@

error_R�8W?

learning_rate_1�<�7����I       6%�	��G���A�**;


total_lossj�@

error_RspK?

learning_rate_1�<�7�
��I       6%�	���G���A�**;


total_lossb�@

error_R��C?

learning_rate_1�<�7k���I       6%�	�F�G���A�**;


total_loss�@�@

error_Rn�D?

learning_rate_1�<�79O�I       6%�	Z��G���A�**;


total_lossV��@

error_REES?

learning_rate_1�<�732zI       6%�	G͏G���A�**;


total_loss���@

error_R��U?

learning_rate_1�<�7h�;�I       6%�	��G���A�**;


total_loss�s@

error_R�Q?

learning_rate_1�<�7�b�I       6%�	�R�G���A�**;


total_lossjcA

error_R�R?

learning_rate_1�<�7)4��I       6%�	���G���A�**;


total_loss��@

error_R�tP?

learning_rate_1�<�76/I       6%�	�ِG���A�**;


total_lossTɚ@

error_R$�E?

learning_rate_1�<�7;���I       6%�	��G���A�**;


total_loss2O�@

error_R�L?

learning_rate_1�<�75 �XI       6%�	�\�G���A�**;


total_loss�@

error_Rx�j?

learning_rate_1�<�7�I       6%�	���G���A�**;


total_loss<��@

error_RR_e?

learning_rate_1�<�7��s�I       6%�	���G���A�**;


total_loss��@

error_R�'D?

learning_rate_1�<�7{�PqI       6%�	�"�G���A�**;


total_loss���@

error_Rt�D?

learning_rate_1�<�7T���I       6%�	Sb�G���A�**;


total_lossN:�@

error_R�E<?

learning_rate_1�<�7CY9�I       6%�	���G���A�**;


total_loss��@

error_R=�K?

learning_rate_1�<�7��I       6%�	��G���A�**;


total_lossw�@

error_R�E?

learning_rate_1�<�7�U��I       6%�	�#�G���A�**;


total_loss���@

error_R�F?

learning_rate_1�<�7/�TI       6%�	�d�G���A�**;


total_loss�)�@

error_RT�S?

learning_rate_1�<�7��u�I       6%�	���G���A�**;


total_loss�,y@

error_RC�V?

learning_rate_1�<�7�DTI       6%�	��G���A�**;


total_loss�@

error_R��K?

learning_rate_1�<�7��0I       6%�	1�G���A�**;


total_loss&�@

error_R�H?

learning_rate_1�<�7�2I       6%�	�o�G���A�**;


total_loss���@

error_Rl�F?

learning_rate_1�<�7���I       6%�	���G���A�**;


total_loss��@

error_R��J?

learning_rate_1�<�7`E�]I       6%�	��G���A�**;


total_loss8E�@

error_R��d?

learning_rate_1�<�7�n��I       6%�	$c�G���A�**;


total_loss߳�@

error_R�J?

learning_rate_1�<�7aP��I       6%�	��G���A�**;


total_loss�#�@

error_R��L?

learning_rate_1�<�7|kWSI       6%�	:�G���A�**;


total_loss���@

error_R�]b?

learning_rate_1�<�7ĠTyI       6%�	a$�G���A�**;


total_loss���@

error_R6BH?

learning_rate_1�<�7��*I       6%�	Rf�G���A�**;


total_lossF}�@

error_R\pF?

learning_rate_1�<�7F���I       6%�	���G���A�**;


total_lossq��@

error_Rw;R?

learning_rate_1�<�7�>9�I       6%�	��G���A�**;


total_loss]#A

error_R	�v?

learning_rate_1�<�7xl�I       6%�	�2�G���A�**;


total_loss�ٶ@

error_R�R?

learning_rate_1�<�7���GI       6%�	�t�G���A�**;


total_losssi@

error_RO�H?

learning_rate_1�<�7��=I       6%�	���G���A�**;


total_loss��@

error_RK?

learning_rate_1�<�7�(DI       6%�	���G���A�**;


total_lossl!�@

error_R��\?

learning_rate_1�<�7�S�I       6%�	s>�G���A�**;


total_lossu�	A

error_RX�D?

learning_rate_1�<�78���I       6%�	5��G���A�**;


total_loss��A

error_R�$J?

learning_rate_1�<�7�Aa!I       6%�	4�G���A�**;


total_loss��@

error_R;�Y?

learning_rate_1�<�7��hI       6%�	+$�G���A�**;


total_loss��@

error_R��>?

learning_rate_1�<�7�{=(I       6%�	Uh�G���A�**;


total_loss�w�@

error_R(}G?

learning_rate_1�<�7�/A�I       6%�	���G���A�**;


total_loss�kA

error_Rf�Q?

learning_rate_1�<�7�Ε9I       6%�	:�G���A�**;


total_loss���@

error_R�N?

learning_rate_1�<�7�B]I       6%�	0�G���A�**;


total_lossT˄@

error_Rq0N?

learning_rate_1�<�7����I       6%�	�q�G���A�**;


total_loss�i�@

error_RV�Y?

learning_rate_1�<�7 ��;I       6%�	���G���A�**;


total_loss-�@

error_R��O?

learning_rate_1�<�7��ѴI       6%�	��G���A�**;


total_lossQl�@

error_R��O?

learning_rate_1�<�7)�ǳI       6%�	 >�G���A�**;


total_lossI��@

error_R2]?

learning_rate_1�<�7��H�I       6%�	�~�G���A�**;


total_loss���@

error_RZTN?

learning_rate_1�<�7Ts� I       6%�	���G���A�**;


total_loss�@

error_R�@T?

learning_rate_1�<�7�хI       6%�	��G���A�**;


total_lossʱ�@

error_R}IS?

learning_rate_1�<�7Q�PqI       6%�	Y�G���A�**;


total_loss��A

error_R(�[?

learning_rate_1�<�7�M��I       6%�	ʯ�G���A�**;


total_loss��@

error_R(�J?

learning_rate_1�<�7|�AZI       6%�	��G���A�**;


total_loss3�@

error_R��^?

learning_rate_1�<�7�ʽI       6%�	�8�G���A�**;


total_loss�`�@

error_R�W?

learning_rate_1�<�7�I       6%�	{�G���A�**;


total_lossKW�@

error_R�E?

learning_rate_1�<�7m�I       6%�	�G���A�**;


total_loss=:�@

error_R(�??

learning_rate_1�<�7ꞚoI       6%�	f��G���A�**;


total_loss�̐@

error_R�uH?

learning_rate_1�<�7t$V�I       6%�	=A�G���A�**;


total_loss�@

error_R�tO?

learning_rate_1�<�7D
�nI       6%�	G��G���A�**;


total_loss�	�@

error_R[�W?

learning_rate_1�<�7�d��I       6%�	�ĞG���A�**;


total_lossL�@

error_Rn>?

learning_rate_1�<�7jH�I       6%�	��G���A�+*;


total_loss<n�@

error_R��E?

learning_rate_1�<�7��I       6%�	@l�G���A�+*;


total_loss�G�@

error_R�WT?

learning_rate_1�<�7���o