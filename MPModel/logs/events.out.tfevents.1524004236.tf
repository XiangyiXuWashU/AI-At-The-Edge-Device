       �K"	   c���Abrain.Event:2�u��>K     6�.	��(c���A"��
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
weights/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
,weights/random_normal_1/RandomStandardNormalRandomStandardNormalweights/random_normal_1/shape*
dtype0* 
_output_shapes
:
��*
seed2 *

seed *
T0
�
weights/random_normal_1/mulMul,weights/random_normal_1/RandomStandardNormalweights/random_normal_1/stddev* 
_output_shapes
:
��*
T0
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
biases/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
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
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
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
biases/bias1/readIdentitybiases/bias1*
T0*
_class
loc:@biases/bias1*
_output_shapes	
:�
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

seed *
T0*
dtype0*
_output_shapes	
:�*
seed2 
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
biases_1/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
biases_1/random_normalAddbiases_1/random_normal/mulbiases_1/random_normal/mean*
_output_shapes	
:�*
T0
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
biases_1/bias1/AssignAssignbiases_1/bias1biases_1/random_normal*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
biases_1/random_normal_1/mulMul-biases_1/random_normal_1/RandomStandardNormalbiases_1/random_normal_1/stddev*
_output_shapes	
:�*
T0
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
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
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
biases_1/random_normal_3Addbiases_1/random_normal_3/mulbiases_1/random_normal_3/mean*
_output_shapes
:*
T0
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

result/AddAddresult/MatMulbiases_1/bias_out/read*'
_output_shapes
:���������*
T0
[
subSub
result/Addinput/Label-input*'
_output_shapes
:���������*
T0
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
,learning_rate/ExponentialDecay/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
&learning_rate/ExponentialDecay/truedivRealDiv#learning_rate/ExponentialDecay/Cast%learning_rate/ExponentialDecay/Cast_1*
T0*
_output_shapes
: 
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
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
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
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
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
train/gradients/sub_1_grad/SumSum!train/gradients/Square_grad/mul_10train/gradients/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/result/Add_grad/ReshapeReshape#train/gradients/result/Add_grad/Sum%train/gradients/result/Add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
;train/gradients/layer_3/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/Add_grad/Reshape_12^train/gradients/layer_3/Add_grad/tuple/group_deps*
_output_shapes
:d*
T0*=
_class3
1/loc:@train/gradients/layer_3/Add_grad/Reshape_1
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
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes
:	�d*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
0train/gradients/layer_3/Sigmoid_grad/SigmoidGradSigmoidGradlayer_3/Sigmoid<train/gradients/layer_3/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/Add_grad/ReshapeReshape$train/gradients/layer_2/Add_grad/Sum&train/gradients/layer_2/Add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
(train/gradients/layer_1/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
6train/gradients/layer_1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/Add_grad/Shape(train/gradients/layer_1/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/Add_grad/ReshapeReshape$train/gradients/layer_1/Add_grad/Sum&train/gradients/layer_1/Add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
&train/gradients/layer_1/Add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_1/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
weights_1/weight1/Adam/readIdentityweights_1/weight1/Adam*
T0*$
_class
loc:@weights_1/weight1* 
_output_shapes
:
��
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
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *$
_class
loc:@weights_1/weight1
�
weights_1/weight1/Adam_1/AssignAssignweights_1/weight1/Adam_1*weights_1/weight1/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight1
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
weights_1/weight3/Adam/AssignAssignweights_1/weight3/Adam(weights_1/weight3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
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
weights_1/weight3/Adam_1/AssignAssignweights_1/weight3/Adam_1*weights_1/weight3/Adam_1/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
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
'biases_1/bias1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias1*
valueB�*    
�
biases_1/bias1/Adam_1
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
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
T0*!
_class
loc:@biases_1/bias1*
_output_shapes
: 
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
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBbiases/bias3
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
save/RestoreV2_12/tensor_namesConst*
dtype0*
_output_shapes
:*(
valueBBbiases_1/bias3/Adam
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
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Assign_22Assignweights/weight_outsave/RestoreV2_22*
T0*%
_class
loc:@weights/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
save/RestoreV2_26/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBweights_1/weight2
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Assign_30Assignweights_1/weight3/Adamsave/RestoreV2_30*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
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
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
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
total_loss/tagsConst*
dtype0*
_output_shapes
: *
valueB B
total_loss
X

total_lossScalarSummarytotal_loss/tags	loss/Mean*
_output_shapes
: *
T0
X
Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B : 
k
MeanMeanAbsMean/reduction_indices*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"��݁Ph     6Ԇg	�)c���AJ��
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
weights/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
weights/random_normal_1/mulMul,weights/random_normal_1/RandomStandardNormalweights/random_normal_1/stddev* 
_output_shapes
:
��*
T0
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
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
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
weights/weight3/readIdentityweights/weight3*
T0*"
_class
loc:@weights/weight3*
_output_shapes
:	�d
n
weights/random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      
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
biases/random_normal_1/mulMul+biases/random_normal_1/RandomStandardNormalbiases/random_normal_1/stddev*
_output_shapes	
:�*
T0
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
biases/random_normal_2/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases/random_normal_2/RandomStandardNormalRandomStandardNormalbiases/random_normal_2/shape*
dtype0*
_output_shapes
:d*
seed2 *

seed *
T0
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
biases/random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
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
biases/bias_out/AssignAssignbiases/bias_outbiases/random_normal_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@biases/bias_out
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
 weights_1/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
weights_1/weight2/readIdentityweights_1/weight2* 
_output_shapes
:
��*
T0*$
_class
loc:@weights_1/weight2
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

seed *
T0*
dtype0*
_output_shapes
:	�d*
seed2 
�
weights_1/random_normal_2/mulMul.weights_1/random_normal_2/RandomStandardNormal weights_1/random_normal_2/stddev*
_output_shapes
:	�d*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
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
biases_1/random_normal_2Addbiases_1/random_normal_2/mulbiases_1/random_normal_2/mean*
_output_shapes
:d*
T0
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
 *o�:*
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
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
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
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
%train/gradients/result/Add_grad/ShapeShaperesult/MatMul*
_output_shapes
:*
T0*
out_type0
q
'train/gradients/result/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
5train/gradients/result/Add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/result/Add_grad/Shape'train/gradients/result/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
#train/gradients/result/Add_grad/SumSum3train/gradients/sub_1_grad/tuple/control_dependency5train/gradients/result/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
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
%train/gradients/result/Add_grad/Sum_1Sum3train/gradients/sub_1_grad/tuple/control_dependency7train/gradients/result/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/Add_grad/ReshapeReshape$train/gradients/layer_3/Add_grad/Sum&train/gradients/layer_3/Add_grad/Shape*'
_output_shapes
:���������d*
T0*
Tshape0
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/Add_grad/tuple/control_dependencyweights_1/weight3/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
(train/gradients/layer_2/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
6train/gradients/layer_2/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/Add_grad/Shape(train/gradients/layer_2/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0*
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
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/Add_grad/tuple/control_dependencyweights_1/weight2/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_2/Relu*(
_output_shapes
:����������*
T0
t
&train/gradients/layer_1/Add_grad/ShapeShapelayer_1/MatMul*
_output_shapes
:*
T0*
out_type0
s
(train/gradients/layer_1/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
6train/gradients/layer_1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/Add_grad/Shape(train/gradients/layer_1/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
dtype0*
_output_shapes
: *
shared_name *!
_class
loc:@biases_1/bias1*
	container *
shape: 
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
weights_1/weight3/Adam_1/AssignAssignweights_1/weight3/Adam_1*weights_1/weight3/Adam_1/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight3*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
weights_1/weight3/Adam_1/readIdentityweights_1/weight3/Adam_1*
_output_shapes
:	�d*
T0*$
_class
loc:@weights_1/weight3
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
VariableV2*!
_class
loc:@biases_1/bias3*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
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
-train/Adam/update_weights_1/weight3/ApplyAdam	ApplyAdamweights_1/weight3weights_1/weight3/Adamweights_1/weight3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@weights_1/weight3*
use_nesterov( *
_output_shapes
:	�d*
use_locking( 
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
*train/Adam/update_biases_1/bias2/ApplyAdam	ApplyAdambiases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@biases_1/bias2
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
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:#*�
value�B�#BVariableBbiases/bias1Bbiases/bias2Bbiases/bias3Bbiases/bias_outBbiases_1/bias1Bbiases_1/bias1/AdamBbiases_1/bias1/Adam_1Bbiases_1/bias2Bbiases_1/bias2/AdamBbiases_1/bias2/Adam_1Bbiases_1/bias3Bbiases_1/bias3/AdamBbiases_1/bias3/Adam_1Bbiases_1/bias_outBbiases_1/bias_out/AdamBbiases_1/bias_out/Adam_1Btrain/beta1_powerBtrain/beta2_powerBweights/weight1Bweights/weight2Bweights/weight3Bweights/weight_outBweights_1/weight1Bweights_1/weight1/AdamBweights_1/weight1/Adam_1Bweights_1/weight2Bweights_1/weight2/AdamBweights_1/weight2/Adam_1Bweights_1/weight3Bweights_1/weight3/AdamBweights_1/weight3/Adam_1Bweights_1/weight_outBweights_1/weight_out/AdamBweights_1/weight_out/Adam_1
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
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
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_25/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bweights_1/weight1/Adam_1
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
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Assign_30Assignweights_1/weight3/Adamsave/RestoreV2_30*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*$
_class
loc:@weights_1/weight3
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
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
_output_shapes
:*
dtypes
2
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
MeanMeanAbsMean/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
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
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0��{�F       r5��	o�+c���A*;


total_loss�[�@

error_RF�J?

learning_rate_1��5�9H       ��H�	�1c���A*;


total_loss�,�@

error_R�Z?

learning_rate_1��5�v��H       ��H�	�1c���A*;


total_lossH�@

error_R�x7?

learning_rate_1��5@�c�H       ��H�	�1c���A*;


total_loss���@

error_R	�W?

learning_rate_1��5��_H       ��H�	S2c���A*;


total_loss��A

error_R�N?

learning_rate_1��5��~LH       ��H�	C�2c���A*;


total_loss�S�@

error_R�UN?

learning_rate_1��5���H       ��H�	��2c���A*;


total_loss�ߓ@

error_RH�@?

learning_rate_1��5'�AH       ��H�	�O3c���A*;


total_lossζ�@

error_R�0N?

learning_rate_1��5W�@�H       ��H�	��3c���A*;


total_loss��@

error_R�.[?

learning_rate_1��5�)TH       ��H�	f�3c���A	*;


total_loss7��@

error_Rn7[?

learning_rate_1��5�xN:H       ��H�	�`4c���A
*;


total_loss�z�@

error_RSO?

learning_rate_1��5�� H       ��H�	,�4c���A*;


total_loss!��@

error_RCM?

learning_rate_1��5oGNTH       ��H�	��4c���A*;


total_loss�dA

error_R�0]?

learning_rate_1��5
?IH       ��H�	�<5c���A*;


total_lossq��@

error_R7�Z?

learning_rate_1��5��mtH       ��H�	��5c���A*;


total_loss:�@

error_R�W?

learning_rate_1��5�ԜqH       ��H�	��5c���A*;


total_loss���@

error_Rv1A?

learning_rate_1��5>Y�H       ��H�	�6c���A*;


total_loss�o�@

error_R�EK?

learning_rate_1��5����H       ��H�	^[6c���A*;


total_loss��@

error_R�z>?

learning_rate_1��5n�l�H       ��H�	ˣ6c���A*;


total_loss���@

error_Rs(N?

learning_rate_1��5���vH       ��H�	a�6c���A*;


total_loss�q�@

error_R�rb?

learning_rate_1��5�*!�H       ��H�	G87c���A*;


total_loss�t�@

error_R�J?

learning_rate_1��5���H       ��H�	�~7c���A*;


total_loss�q�@

error_R��[?

learning_rate_1��5N��DH       ��H�	6�7c���A*;


total_loss-F�@

error_R��N?

learning_rate_1��5�k�H       ��H�	8c���A*;


total_lossc�@

error_R(>c?

learning_rate_1��5j��OH       ��H�	Q8c���A*;


total_loss}�@

error_R��S?

learning_rate_1��5�9��H       ��H�	��8c���A*;


total_loss��@

error_R�X?

learning_rate_1��5�ٴ%H       ��H�	\�8c���A*;


total_loss�ʡ@

error_R��D?

learning_rate_1��5GI$�H       ��H�	�9c���A*;


total_loss��@

error_RӠC?

learning_rate_1��5K�NpH       ��H�	�`9c���A*;


total_lossL�@

error_R��V?

learning_rate_1��5oc�#H       ��H�	@�9c���A*;


total_lossZ@�@

error_Rl4K?

learning_rate_1��5u��H       ��H�	H�9c���A*;


total_loss�4�@

error_R!"B?

learning_rate_1��5�FAH       ��H�	�/:c���A*;


total_loss��A

error_R�Z?

learning_rate_1��5���YH       ��H�	mv:c���A *;


total_losso�@

error_R=%M?

learning_rate_1��5�+�H       ��H�	Z�:c���A!*;


total_loss�Dp@

error_R�*P?

learning_rate_1��5�d9�H       ��H�	� ;c���A"*;


total_loss&��@

error_RΥD?

learning_rate_1��5��ІH       ��H�	�F;c���A#*;


total_loss��@

error_R��>?

learning_rate_1��5�_3H       ��H�	�;c���A$*;


total_loss�A�@

error_R�'L?

learning_rate_1��5:0C�H       ��H�	��;c���A%*;


total_loss;��@

error_R3�H?

learning_rate_1��5gu�H       ��H�	5<c���A&*;


total_loss���@

error_RT�=?

learning_rate_1��5e�[�H       ��H�	S<c���A'*;


total_loss�˔@

error_R��2?

learning_rate_1��5d&GH       ��H�	ƕ<c���A(*;


total_loss2�@

error_R(
`?

learning_rate_1��5ߍb�H       ��H�	\�<c���A)*;


total_lossb�@

error_R?�F?

learning_rate_1��5�`C�H       ��H�	F'=c���A**;


total_lossۼ@

error_RB?

learning_rate_1��5�$v�H       ��H�	 �=c���A+*;


total_loss���@

error_R�'Z?

learning_rate_1��5��z H       ��H�	/�=c���A,*;


total_loss��@

error_R�NL?

learning_rate_1��5�ki�H       ��H�	n>c���A-*;


total_loss���@

error_R�gN?

learning_rate_1��5y��H       ��H�	Xf>c���A.*;


total_loss���@

error_R��J?

learning_rate_1��5Pk�H       ��H�	8�>c���A/*;


total_losst|�@

error_R��J?

learning_rate_1��5�Y��H       ��H�	?c���A0*;


total_loss�5�@

error_RW�R?

learning_rate_1��5t~�H       ��H�	0`?c���A1*;


total_loss��@

error_R��L?

learning_rate_1��54��IH       ��H�	F�?c���A2*;


total_loss���@

error_R�cR?

learning_rate_1��5���H       ��H�	T
@c���A3*;


total_loss�N�@

error_R��M?

learning_rate_1��5���XH       ��H�	�T@c���A4*;


total_loss
e�@

error_R�5M?

learning_rate_1��5�N�<H       ��H�	�@c���A5*;


total_loss�@

error_R Ha?

learning_rate_1��5�y}H       ��H�	�@c���A6*;


total_lossw4�@

error_R�zT?

learning_rate_1��5����H       ��H�	�3Ac���A7*;


total_loss��@

error_RXT?

learning_rate_1��5��H       ��H�	ozAc���A8*;


total_loss�MA

error_R�AO?

learning_rate_1��5C�H       ��H�	?�Ac���A9*;


total_lossř�@

error_Rz�S?

learning_rate_1��5�Z6NH       ��H�	\"Bc���A:*;


total_loss��@

error_R�&C?

learning_rate_1��5Js��H       ��H�	�hBc���A;*;


total_losshtA

error_R�I?

learning_rate_1��5+y]H       ��H�	�Bc���A<*;


total_loss-�@

error_RS?

learning_rate_1��5E�C�H       ��H�	{�Bc���A=*;


total_lossYZ@

error_R�YI?

learning_rate_1��5}�mH       ��H�	�2Cc���A>*;


total_lossH��@

error_R�,Y?

learning_rate_1��5�;!�H       ��H�	�vCc���A?*;


total_lossQ��@

error_R�aV?

learning_rate_1��5*�.�H       ��H�	
�Cc���A@*;


total_loss��@

error_R)�_?

learning_rate_1��5u�H       ��H�	�Dc���AA*;


total_loss('�@

error_R��;?

learning_rate_1��5q��H       ��H�	�ODc���AB*;


total_loss�~�@

error_RcHS?

learning_rate_1��5�V�pH       ��H�	��Dc���AC*;


total_loss� �@

error_R/�W?

learning_rate_1��5̾gH       ��H�	��Dc���AD*;


total_lossh��@

error_R��E?

learning_rate_1��5�l�+H       ��H�	/$Ec���AE*;


total_loss{�@

error_R�cV?

learning_rate_1��5D�H       ��H�	�iEc���AF*;


total_loss5�@

error_R�C?

learning_rate_1��5;. H       ��H�	�Ec���AG*;


total_lossT��@

error_R��R?

learning_rate_1��5�&RNH       ��H�	.�Ec���AH*;


total_lossC�~@

error_R8?

learning_rate_1��5v�L)H       ��H�	�5Fc���AI*;


total_lossxۺ@

error_R��??

learning_rate_1��57�׳H       ��H�	/yFc���AJ*;


total_loss�A

error_R�fQ?

learning_rate_1��5�)�GH       ��H�	M�Fc���AK*;


total_loss���@

error_R�Q?

learning_rate_1��5��yH       ��H�	a�Fc���AL*;


total_loss!D�@

error_R��L?

learning_rate_1��5���&H       ��H�	�FGc���AM*;


total_loss/O�@

error_RW�J?

learning_rate_1��5���H       ��H�	`�Gc���AN*;


total_lossc��@

error_R=W?

learning_rate_1��5�H       ��H�	��Gc���AO*;


total_loss�k�@

error_R��H?

learning_rate_1��5G���H       ��H�	_Hc���AP*;


total_lossAW�@

error_R��T?

learning_rate_1��5ͩ{9H       ��H�	-VHc���AQ*;


total_loss��k@

error_R�'L?

learning_rate_1��5�[��H       ��H�	<�Hc���AR*;


total_loss���@

error_R`?

learning_rate_1��5�&LH       ��H�	��Hc���AS*;


total_loss��@

error_RisP?

learning_rate_1��5~0H       ��H�	�'Ic���AT*;


total_loss��@

error_R�Z?

learning_rate_1��5��%HH       ��H�	�lIc���AU*;


total_loss���@

error_RH"Q?

learning_rate_1��5�`3H       ��H�	��Ic���AV*;


total_loss�
A

error_R=�M?

learning_rate_1��5�$qH       ��H�	
�Ic���AW*;


total_loss�A

error_RoQ?

learning_rate_1��5w噁H       ��H�	�7Jc���AX*;


total_loss��@

error_R��R?

learning_rate_1��5Ǔ&�H       ��H�	�{Jc���AY*;


total_loss8ا@

error_RRL??

learning_rate_1��5{;H       ��H�	?�Jc���AZ*;


total_loss��@

error_RU?

learning_rate_1��5���H       ��H�	7Kc���A[*;


total_loss<�@

error_R�`S?

learning_rate_1��5�CTDH       ��H�	_FKc���A\*;


total_loss�@

error_R�}J?

learning_rate_1��5ܔ�H       ��H�	�Kc���A]*;


total_loss�4�@

error_RW,K?

learning_rate_1��5��H       ��H�	v�Kc���A^*;


total_loss�D�@

error_R��@?

learning_rate_1��5ԩ�H       ��H�	VLc���A_*;


total_loss1��@

error_Rq�>?

learning_rate_1��5�]��H       ��H�	�RLc���A`*;


total_loss ~�@

error_RÚE?

learning_rate_1��5=�;�H       ��H�	��Lc���Aa*;


total_loss˞@

error_RSG?

learning_rate_1��5� `H       ��H�	��Lc���Ab*;


total_loss���@

error_R��B?

learning_rate_1��5�B��H       ��H�	w=Mc���Ac*;


total_lossl��@

error_RJ�S?

learning_rate_1��5����H       ��H�	��Mc���Ad*;


total_loss&�@

error_RZuO?

learning_rate_1��5�z�?H       ��H�	��Mc���Ae*;


total_loss��@

error_R��R?

learning_rate_1��5,�tFH       ��H�	2)Nc���Af*;


total_loss
�@

error_Rf[?

learning_rate_1��5\�H       ��H�	qoNc���Ag*;


total_loss`g�@

error_R�??

learning_rate_1��5�F�H       ��H�	�Nc���Ah*;


total_lossG��@

error_R�!C?

learning_rate_1��5&�G H       ��H�	�Nc���Ai*;


total_loss�˩@

error_R�TH?

learning_rate_1��5+�n:H       ��H�	6Oc���Aj*;


total_lossF �@

error_R
�M?

learning_rate_1��5ȋ<�H       ��H�	�zOc���Ak*;


total_loss�I�@

error_R4�S?

learning_rate_1��5��O�H       ��H�	;�Oc���Al*;


total_loss	�@

error_R��^?

learning_rate_1��5$���H       ��H�	?Pc���Am*;


total_loss-��@

error_RښH?

learning_rate_1��5zv�$H       ��H�	XMPc���An*;


total_loss�@

error_RvaC?

learning_rate_1��5�pX�H       ��H�	��Pc���Ao*;


total_loss?��@

error_R��C?

learning_rate_1��5#e}kH       ��H�	Z�Pc���Ap*;


total_loss��@

error_R�9X?

learning_rate_1��5��N�H       ��H�	�%Qc���Aq*;


total_loss�@

error_Ra�Q?

learning_rate_1��5��H       ��H�	�iQc���Ar*;


total_loss>�@

error_R�X?

learning_rate_1��5ǵ�WH       ��H�	��Qc���As*;


total_loss���@

error_R.8F?

learning_rate_1��5���H       ��H�	��Qc���At*;


total_loss%�f@

error_REFJ?

learning_rate_1��5i��H       ��H�	2Rc���Au*;


total_lossoI�@

error_R�Zd?

learning_rate_1��5H�HH       ��H�	�sRc���Av*;


total_loss��@

error_Rj�b?

learning_rate_1��5Y�isH       ��H�	q�Rc���Aw*;


total_loss6�A

error_R �V?

learning_rate_1��5��<�H       ��H�	�Sc���Ax*;


total_loss�ӷ@

error_Rh�V?

learning_rate_1��5ܫJ�H       ��H�	dMSc���Ay*;


total_loss�ܟ@

error_R��d?

learning_rate_1��5"�a/H       ��H�	��Sc���Az*;


total_loss;Q�@

error_Rs�Y?

learning_rate_1��5��H       ��H�	��Sc���A{*;


total_loss=�@

error_R}�K?

learning_rate_1��5��&�H       ��H�	�Tc���A|*;


total_loss�=�@

error_R6�J?

learning_rate_1��5���H       ��H�	b[Tc���A}*;


total_lossmJ�@

error_R��J?

learning_rate_1��5�e��H       ��H�	�Tc���A~*;


total_lossC��@

error_RxU;?

learning_rate_1��5�Q��H       ��H�	��Tc���A*;


total_loss峽@

error_R�NB?

learning_rate_1��5�_�I       6%�	�&Uc���A�*;


total_loss(�@

error_RJ�K?

learning_rate_1��5WzVI       6%�	�jUc���A�*;


total_loss�@

error_R8)B?

learning_rate_1��5AMвI       6%�	��Uc���A�*;


total_loss-��@

error_R<�M?

learning_rate_1��5���I       6%�	2�Uc���A�*;


total_loss$�4A

error_RPK?

learning_rate_1��5�3I       6%�	(1Vc���A�*;


total_loss���@

error_RיT?

learning_rate_1��5Ҥ<I       6%�	*sVc���A�*;


total_lossʧ@

error_R
�L?

learning_rate_1��5�ÖI       6%�	w�Vc���A�*;


total_loss(
�@

error_R�??

learning_rate_1��5�;ƭI       6%�	��Vc���A�*;


total_loss:I�@

error_R=�R?

learning_rate_1��5z~0I       6%�	L?Wc���A�*;


total_loss/��@

error_R�XJ?

learning_rate_1��5��I       6%�	\�Wc���A�*;


total_lossd��@

error_Rߔ[?

learning_rate_1��5��w�I       6%�	E�Wc���A�*;


total_loss�ř@

error_R�`A?

learning_rate_1��5	�Z�I       6%�	nXc���A�*;


total_loss�R�@

error_RmUD?

learning_rate_1��5j&�I       6%�	MXc���A�*;


total_lossi`�@

error_RZT?

learning_rate_1��5�fo:I       6%�	/�Xc���A�*;


total_loss��A

error_R�tN?

learning_rate_1��5u�TI       6%�	y�Xc���A�*;


total_loss2�@

error_R�	S?

learning_rate_1��5]�LI       6%�	
Yc���A�*;


total_loss9�@

error_R��@?

learning_rate_1��5@xVI       6%�	�XYc���A�*;


total_lossF��@

error_RI?

learning_rate_1��5�bI�I       6%�	B�Yc���A�*;


total_loss7w�@

error_R��H?

learning_rate_1��5�|�,I       6%�	@�Yc���A�*;


total_lossi��@

error_R�S?

learning_rate_1��5Tq;I       6%�	''Zc���A�*;


total_loss��@

error_R4D?

learning_rate_1��5�rxI       6%�	�oZc���A�*;


total_loss�Ј@

error_R�TB?

learning_rate_1��5NG0�I       6%�	�Zc���A�*;


total_loss�ȹ@

error_RA�\?

learning_rate_1��5��_�I       6%�	�[c���A�*;


total_loss(<�@

error_RrI?

learning_rate_1��5#���I       6%�	�j[c���A�*;


total_loss8h�@

error_R�a\?

learning_rate_1��5�:�I       6%�	Ӷ[c���A�*;


total_loss];�@

error_RR(X?

learning_rate_1��5�m�I       6%�	��[c���A�*;


total_loss�F�@

error_RzIN?

learning_rate_1��5	+gI       6%�	�L\c���A�*;


total_loss
�@

error_R�]?

learning_rate_1��5��G#I       6%�	�\c���A�*;


total_loss��.A

error_R$�L?

learning_rate_1��5�/I       6%�	y�\c���A�*;


total_lossz��@

error_ROM?

learning_rate_1��5�I       6%�	u&]c���A�*;


total_loss�,�@

error_R��`?

learning_rate_1��52V�!I       6%�	Ȅ]c���A�*;


total_loss��	A

error_RWQ>?

learning_rate_1��5
��I       6%�	��]c���A�*;


total_loss��@

error_R7�N?

learning_rate_1��5nc	�I       6%�	�^c���A�*;


total_loss�p�@

error_R��B?

learning_rate_1��5v�I       6%�	Eg^c���A�*;


total_lossϽ�@

error_RC)X?

learning_rate_1��5��I       6%�	{�^c���A�*;


total_loss�c�@

error_R��G?

learning_rate_1��5 �>�I       6%�	��^c���A�*;


total_loss��@

error_R`I?

learning_rate_1��5K�.�I       6%�	:_c���A�*;


total_loss7L�@

error_R%?F?

learning_rate_1��5!>�8I       6%�	�}_c���A�*;


total_loss��@

error_R�AB?

learning_rate_1��5�hI       6%�	 �_c���A�*;


total_loss��@

error_R��a?

learning_rate_1��5�-�LI       6%�	�`c���A�*;


total_losscG�@

error_R�UN?

learning_rate_1��5ZWt�I       6%�	�J`c���A�*;


total_loss�%A

error_R�K?

learning_rate_1��5O���I       6%�	4�`c���A�*;


total_loss���@

error_R��B?

learning_rate_1��5hT7OI       6%�	.�`c���A�*;


total_loss���@

error_Ri�M?

learning_rate_1��5��EfI       6%�	�ac���A�*;


total_loss�q�@

error_R��B?

learning_rate_1��5��/*I       6%�	�Wac���A�*;


total_loss�q�@

error_R��K?

learning_rate_1��5���I       6%�	t�ac���A�*;


total_loss$ڈ@

error_R =@?

learning_rate_1��5�)��I       6%�	��ac���A�*;


total_loss$X�@

error_R �T?

learning_rate_1��5Y?]�I       6%�	\-bc���A�*;


total_loss]ȶ@

error_R�d?

learning_rate_1��5
v��I       6%�	Jobc���A�*;


total_loss�\�@

error_R. U?

learning_rate_1��5[�C�I       6%�	W�bc���A�*;


total_loss��@

error_RQ&T?

learning_rate_1��5�`;JI       6%�	��bc���A�*;


total_loss��@

error_R�SX?

learning_rate_1��5ոI       6%�	�Acc���A�*;


total_loss?��@

error_R�oF?

learning_rate_1��5�'#I       6%�	�cc���A�*;


total_loss��A

error_R�4R?

learning_rate_1��5\~(I       6%�	3�cc���A�*;


total_loss�3�@

error_R�?G?

learning_rate_1��5Y�I       6%�	�dc���A�*;


total_loss4�@

error_RZ�8?

learning_rate_1��5�q�I       6%�	�^dc���A�*;


total_loss=��@

error_RY?

learning_rate_1��5�lI       6%�	j�dc���A�*;


total_loss#��@

error_R�SI?

learning_rate_1��5��0TI       6%�	f�dc���A�*;


total_loss��@

error_R��U?

learning_rate_1��5`c�I       6%�		&ec���A�*;


total_loss���@

error_R!D?

learning_rate_1��5��
�I       6%�	jec���A�*;


total_loss.�4@

error_R�R?

learning_rate_1��5���I       6%�	��ec���A�*;


total_loss�5LA

error_R��G?

learning_rate_1��5=^tI       6%�	Q�ec���A�*;


total_loss�o�@

error_R��U?

learning_rate_1��5׵U�I       6%�	�0fc���A�*;


total_loss���@

error_R`�l?

learning_rate_1��5l%�MI       6%�	�rfc���A�*;


total_loss��@

error_R4@?

learning_rate_1��5��{I       6%�	��fc���A�*;


total_loss���@

error_RW�G?

learning_rate_1��5+��I       6%�	ugc���A�*;


total_loss���@

error_R.ka?

learning_rate_1��5��m�I       6%�	Igc���A�*;


total_loss���@

error_R1�G?

learning_rate_1��5����I       6%�	֍gc���A�*;


total_lossaX�@

error_R��P?

learning_rate_1��5��&I       6%�	��gc���A�*;


total_loss\
�@

error_RV$L?

learning_rate_1��5�w'�I       6%�	Vhc���A�*;


total_loss�"�@

error_R�"P?

learning_rate_1��5��1I       6%�	ydhc���A�*;


total_loss�%�@

error_RT�X?

learning_rate_1��5jH��I       6%�	��hc���A�*;


total_loss���@

error_R��R?

learning_rate_1��5�G�I       6%�	0�hc���A�*;


total_loss��@

error_R��5?

learning_rate_1��5Oo�rI       6%�	?3ic���A�*;


total_loss,��@

error_R��L?

learning_rate_1��5��>�I       6%�	vic���A�*;


total_loss��@

error_RW'W?

learning_rate_1��5Q���I       6%�	��ic���A�*;


total_lossn��@

error_R:S?

learning_rate_1��5��=�I       6%�	��ic���A�*;


total_loss��@

error_Ra[X?

learning_rate_1��57��I       6%�	�Ijc���A�*;


total_loss
I�@

error_RO�E?

learning_rate_1��5՟��I       6%�	��jc���A�*;


total_loss�2�@

error_R�@J?

learning_rate_1��5n�ǆI       6%�	-�jc���A�*;


total_lossA�@

error_RW�^?

learning_rate_1��5O���I       6%�	])kc���A�*;


total_loss�}�@

error_R�K?

learning_rate_1��5���I       6%�	�mkc���A�*;


total_loss4v�@

error_R��Y?

learning_rate_1��5��
I       6%�	��kc���A�*;


total_loss��@

error_RjR?

learning_rate_1��5�ġI       6%�	 lc���A�*;


total_loss6ޡ@

error_R@9H?

learning_rate_1��5/i�+I       6%�	�llc���A�*;


total_loss`�@

error_Rf\?

learning_rate_1��5",5I       6%�	l�lc���A�*;


total_loss��@

error_RZ;L?

learning_rate_1��5�'bI       6%�	5�lc���A�*;


total_loss8[�@

error_Rd�F?

learning_rate_1��5��~I       6%�	X`mc���A�*;


total_lossJ��@

error_R;�H?

learning_rate_1��5�`�lI       6%�	��mc���A�*;


total_loss��@

error_R��Q?

learning_rate_1��52ЕI       6%�	��mc���A�*;


total_loss�/�@

error_R��T?

learning_rate_1��5�SŊI       6%�	w8nc���A�*;


total_loss�'�@

error_R=]_?

learning_rate_1��5��7�I       6%�	�|nc���A�*;


total_loss��v@

error_RO�P?

learning_rate_1��5e�(UI       6%�	O�nc���A�*;


total_lossTc�@

error_RԀF?

learning_rate_1��5(�I       6%�	>oc���A�*;


total_lossRA�@

error_R��Z?

learning_rate_1��5�KJ�I       6%�	;Koc���A�*;


total_lossڼ�@

error_R/�V?

learning_rate_1��5X#KI       6%�	��oc���A�*;


total_loss&Q�@

error_R�S?

learning_rate_1��5�uCI       6%�	��oc���A�*;


total_lossH�@

error_RE?

learning_rate_1��5��I       6%�	�pc���A�*;


total_loss;;�@

error_RҺM?

learning_rate_1��5�1�BI       6%�	�]pc���A�*;


total_loss�T�@

error_R3L?

learning_rate_1��5!��HI       6%�	��pc���A�*;


total_loss�F�@

error_R&�^?

learning_rate_1��5��I       6%�	��pc���A�*;


total_loss��@

error_Rͱ<?

learning_rate_1��5X�E�I       6%�	)qc���A�*;


total_loss�q�@

error_R��^?

learning_rate_1��5��I       6%�	clqc���A�*;


total_loss@

error_R��K?

learning_rate_1��55(X�I       6%�	
�qc���A�*;


total_lossVN�@

error_R�W?

learning_rate_1��5ѩ�I       6%�	��qc���A�*;


total_loss�d!A

error_R��J?

learning_rate_1��5���I       6%�	�>rc���A�*;


total_loss�@

error_R�KV?

learning_rate_1��5�R��I       6%�	فrc���A�*;


total_loss�H�@

error_Rj�W?

learning_rate_1��5r�I       6%�	��rc���A�*;


total_loss=�A

error_R EB?

learning_rate_1��5��*I       6%�	Xsc���A�*;


total_loss��@

error_R�*@?

learning_rate_1��5
�I       6%�	�Qsc���A�*;


total_lossH��@

error_R��H?

learning_rate_1��5E9&�I       6%�	 �sc���A�*;


total_lossi��@

error_R�&??

learning_rate_1��5T��)I       6%�	]�sc���A�*;


total_lossW>A

error_R&�N?

learning_rate_1��5j���I       6%�	Ttc���A�*;


total_lossX4�@

error_R�BK?

learning_rate_1��5h�R?I       6%�	�^tc���A�*;


total_loss
*�@

error_RR$;?

learning_rate_1��5�*�`I       6%�	��tc���A�*;


total_lossc��@

error_R�pJ?

learning_rate_1��5���I       6%�	��tc���A�*;


total_lossF�@

error_RT4Z?

learning_rate_1��5ߡ��I       6%�	/uc���A�*;


total_lossք@

error_R��f?

learning_rate_1��5�A1oI       6%�	%tuc���A�*;


total_loss�-�@

error_RiM?

learning_rate_1��5U�>I       6%�	*�uc���A�*;


total_loss� �@

error_R�^K?

learning_rate_1��5�P)\I       6%�	�vc���A�*;


total_loss���@

error_RפQ?

learning_rate_1��5?+>I       6%�	UHvc���A�*;


total_loss���@

error_R��L?

learning_rate_1��5���I       6%�	��vc���A�*;


total_loss6��@

error_R��>?

learning_rate_1��5U�<�I       6%�	��vc���A�*;


total_losse~�@

error_RO�H?

learning_rate_1��5q�I       6%�	"wc���A�*;


total_loss5�@

error_R��D?

learning_rate_1��5�@��I       6%�	Zwc���A�*;


total_loss<�@

error_R��X?

learning_rate_1��5��W�I       6%�	w�wc���A�*;


total_loss���@

error_R�1V?

learning_rate_1��5.&��I       6%�	��wc���A�*;


total_lossA�@

error_RO�W?

learning_rate_1��5���I       6%�	�"xc���A�*;


total_lossa��@

error_R��S?

learning_rate_1��5�kΓI       6%�	�ixc���A�*;


total_loss�DA

error_R1�X?

learning_rate_1��5�vWI       6%�	�xc���A�*;


total_lossT~�@

error_R��G?

learning_rate_1��5vto]I       6%�	�xc���A�*;


total_loss�3�@

error_R�FP?

learning_rate_1��5��I       6%�	lDyc���A�*;


total_loss��@

error_R��H?

learning_rate_1��5��ptI       6%�	��yc���A�*;


total_loss)Ҷ@

error_Rl�R?

learning_rate_1��5o]�(I       6%�	D�yc���A�*;


total_loss*�@

error_RN�P?

learning_rate_1��5���I       6%�	ezc���A�*;


total_lossa�@

error_Rt�]?

learning_rate_1��52[��I       6%�	m]zc���A�*;


total_lossfn�@

error_Re�^?

learning_rate_1��5n3�WI       6%�	��zc���A�*;


total_loss�>�@

error_Rf�Q?

learning_rate_1��5�I       6%�	�zc���A�*;


total_lossH��@

error_R��G?

learning_rate_1��5�� �I       6%�	�.{c���A�*;


total_loss�/�@

error_R��P?

learning_rate_1��5��*sI       6%�	zt{c���A�*;


total_loss�A

error_R��M?

learning_rate_1��5�.�rI       6%�	6�{c���A�*;


total_loss_�@

error_R8�W?

learning_rate_1��58M��I       6%�	��{c���A�*;


total_lossM�4@

error_R��R?

learning_rate_1��5��%8I       6%�	�F|c���A�*;


total_loss�e�@

error_R�L?

learning_rate_1��5+)+WI       6%�	g�|c���A�*;


total_loss�-�@

error_R(�D?

learning_rate_1��5����I       6%�	��|c���A�*;


total_loss��@

error_R�H?

learning_rate_1��5_�&�I       6%�	1*}c���A�*;


total_loss2�@

error_R�JT?

learning_rate_1��5xV�I       6%�	;�}c���A�*;


total_loss`�@

error_R�x[?

learning_rate_1��5{�QJI       6%�	��}c���A�*;


total_loss��@

error_R��a?

learning_rate_1��5:f"I       6%�	c#~c���A�*;


total_lossT�@

error_R=R?

learning_rate_1��5Ӻ��I       6%�	4k~c���A�*;


total_loss*Ĵ@

error_R2�V?

learning_rate_1��5����I       6%�	��~c���A�*;


total_lossR�@

error_R�b[?

learning_rate_1��5���WI       6%�	J�~c���A�*;


total_loss\��@

error_R�YH?

learning_rate_1��5!fȔI       6%�	}8c���A�*;


total_loss�
A

error_R/9?

learning_rate_1��5��I       6%�	�}c���A�*;


total_lossdA�@

error_R��;?

learning_rate_1��5��-�I       6%�	��c���A�*;


total_lossM��@

error_RiAI?

learning_rate_1��5WDwI       6%�	p�c���A�*;


total_loss�۾@

error_R��M?

learning_rate_1��5����I       6%�	�O�c���A�*;


total_loss��@

error_R]�??

learning_rate_1��5���I       6%�	���c���A�*;


total_loss�s@

error_R�Y?

learning_rate_1��5.�!I       6%�	5�c���A�*;


total_lossXe�@

error_R��M?

learning_rate_1��5׺NyI       6%�	�.�c���A�*;


total_loss;��@

error_R�L?

learning_rate_1��5f�+QI       6%�	Ä�c���A�*;


total_loss��@

error_RJ$T?

learning_rate_1��5��L�I       6%�	сc���A�*;


total_loss���@

error_R!�H?

learning_rate_1��5�#�1I       6%�	d�c���A�*;


total_loss7s�@

error_R�c?

learning_rate_1��5�isI       6%�	$c�c���A�*;


total_loss��@

error_R�Y?

learning_rate_1��5��| I       6%�	��c���A�*;


total_loss���@

error_RƚT?

learning_rate_1��5�5�XI       6%�	��c���A�*;


total_loss��}@

error_R��E?

learning_rate_1��5���.I       6%�	�M�c���A�*;


total_loss� A

error_RC2K?

learning_rate_1��5��}I       6%�	P��c���A�*;


total_loss���@

error_RNPO?

learning_rate_1��5���I       6%�	�Ӄc���A�*;


total_lossc �@

error_R;eO?

learning_rate_1��5�ScI       6%�	��c���A�*;


total_loss�.�@

error_R�R?

learning_rate_1��5��LI       6%�	�[�c���A�*;


total_loss�@

error_R_�P?

learning_rate_1��5=�uI       6%�	O��c���A�*;


total_lossѤ�@

error_R��B?

learning_rate_1��5i2 I       6%�	5�c���A�*;


total_loss�@

error_RJER?

learning_rate_1��5�m��I       6%�	k3�c���A�*;


total_loss���@

error_R��N?

learning_rate_1��5�AbI       6%�	{�c���A�*;


total_loss���@

error_R�P?

learning_rate_1��5�Ώ�I       6%�	n��c���A�*;


total_loss�e�@

error_R��@?

learning_rate_1��5n@tdI       6%�	4�c���A�*;


total_loss��@

error_R	�S?

learning_rate_1��5���I       6%�	XF�c���A�*;


total_lossa��@

error_RF�W?

learning_rate_1��5hoK�I       6%�	���c���A�*;


total_loss)�@

error_R�jK?

learning_rate_1��5q��'I       6%�	�҆c���A�*;


total_lossΖ@

error_Rn,L?

learning_rate_1��5Z���I       6%�	4�c���A�*;


total_lossR܏@

error_R�O?

learning_rate_1��5}��I       6%�	6]�c���A�*;


total_loss6��@

error_R�MB?

learning_rate_1��5�mT�I       6%�	B��c���A�*;


total_loss���@

error_RE\?

learning_rate_1��5��;I       6%�	/�c���A�*;


total_loss��@

error_RV�B?

learning_rate_1��5q@�I       6%�	�0�c���A�*;


total_lossS��@

error_R��F?

learning_rate_1��5C� I       6%�	6s�c���A�*;


total_loss}��@

error_RUI?

learning_rate_1��5%iI       6%�	洈c���A�*;


total_loss4m�@

error_R3a?

learning_rate_1��5�B�I       6%�	��c���A�*;


total_lossw�@

error_R�?S?

learning_rate_1��5V�,=I       6%�	b8�c���A�*;


total_loss 	�@

error_R��V?

learning_rate_1��5;!AzI       6%�	'|�c���A�*;


total_loss�q�@

error_R��>?

learning_rate_1��5|�I       6%�	ܾ�c���A�*;


total_loss��	A

error_RןG?

learning_rate_1��5j��^I       6%�	H�c���A�*;


total_loss��A

error_Rv[?

learning_rate_1��5�L�I       6%�	�E�c���A�*;


total_lossPHA

error_R�D?

learning_rate_1��5`k�I       6%�	���c���A�*;


total_lossc+�@

error_R�nI?

learning_rate_1��5z�أI       6%�	�֊c���A�*;


total_loss̬�@

error_R�"T?

learning_rate_1��5}��8I       6%�	)�c���A�*;


total_lossQ��@

error_R�IW?

learning_rate_1��5�s[I       6%�	�e�c���A�*;


total_lossFe�@

error_R3�G?

learning_rate_1��5\�z�I       6%�	���c���A�*;


total_loss�T�@

error_Rn2F?

learning_rate_1��5}$�I       6%�	��c���A�*;


total_loss$L�@

error_R)�F?

learning_rate_1��5�2P�I       6%�	�5�c���A�*;


total_lossc��@

error_R��I?

learning_rate_1��5�7fI       6%�	�v�c���A�*;


total_loss&a�@

error_R��B?

learning_rate_1��5+NkI       6%�	���c���A�*;


total_lossMn�@

error_R(�W?

learning_rate_1��5
sf+I       6%�	���c���A�*;


total_lossX� A

error_R�X\?

learning_rate_1��5�	�hI       6%�	d�c���A�*;


total_losss��@

error_R]�A?

learning_rate_1��5��eI       6%�	Լ�c���A�*;


total_loss6Z@

error_R�M?

learning_rate_1��5�R�vI       6%�	��c���A�*;


total_losso��@

error_R�:=?

learning_rate_1��5|^OI       6%�	;C�c���A�*;


total_lossn��@

error_R�EW?

learning_rate_1��5�
ϐI       6%�	H��c���A�*;


total_loss��@

error_R��K?

learning_rate_1��5����I       6%�	�̎c���A�*;


total_loss��@

error_RnU?

learning_rate_1��55A�I       6%�	�c���A�*;


total_loss��A

error_RIE?

learning_rate_1��5S*sbI       6%�	Gl�c���A�*;


total_lossf݊@

error_R�&M?

learning_rate_1��5�ο�I       6%�	��c���A�*;


total_loss�h�@

error_R=�d?

learning_rate_1��5L���I       6%�	��c���A�*;


total_losst�@

error_R��H?

learning_rate_1��5~	�TI       6%�	#^�c���A�*;


total_loss���@

error_RO�O?

learning_rate_1��5�	�/I       6%�	ӯ�c���A�*;


total_loss���@

error_R��M?

learning_rate_1��5p3L�I       6%�	#�c���A�*;


total_loss6��@

error_R��J?

learning_rate_1��5�7��I       6%�	�~�c���A�*;


total_loss�v�@

error_R�bE?

learning_rate_1��5��a�I       6%�	Uˑc���A�*;


total_lossJ�@

error_R�YO?

learning_rate_1��5�蜑I       6%�	��c���A�*;


total_loss��@

error_R?+K?

learning_rate_1��5%��eI       6%�	���c���A�*;


total_lossz�@

error_R��E?

learning_rate_1��5�}f�I       6%�	Oʒc���A�*;


total_loss�@

error_R[�I?

learning_rate_1��56�x�I       6%�	"�c���A�*;


total_loss[�@

error_R��W?

learning_rate_1��5U��UI       6%�	�Y�c���A�*;


total_loss3c�@

error_R��Y?

learning_rate_1��5��թI       6%�	ȓc���A�*;


total_loss�3�@

error_R��U?

learning_rate_1��5:=iPI       6%�	��c���A�*;


total_loss�n�@

error_RH<M?

learning_rate_1��5V]*I       6%�	�T�c���A�*;


total_lossH�@

error_R�`?

learning_rate_1��5j�y]I       6%�		��c���A�*;


total_loss���@

error_RRV?

learning_rate_1��5FS�"I       6%�	 ��c���A�*;


total_loss;�@

error_R�D?

learning_rate_1��5L���I       6%�	2>�c���A�*;


total_loss*��@

error_R;U?

learning_rate_1��5��D�I       6%�	+��c���A�*;


total_loss��@

error_R�L?

learning_rate_1��5Y� �I       6%�	�וc���A�*;


total_loss�@

error_RNO7?

learning_rate_1��5;�r�I       6%�	�.�c���A�*;


total_loss6�@

error_R 	X?

learning_rate_1��5�U�OI       6%�	s�c���A�*;


total_loss.~o@

error_R�,P?

learning_rate_1��5�(I       6%�	ѷ�c���A�*;


total_loss���@

error_R�\?

learning_rate_1��5����I       6%�	?�c���A�*;


total_lossL\�@

error_R�(H?

learning_rate_1��5ө��I       6%�	�b�c���A�*;


total_lossiQ�@

error_R�IP?

learning_rate_1��5��6I       6%�	���c���A�*;


total_loss���@

error_R:�R?

learning_rate_1��5�CbI       6%�	���c���A�*;


total_lossi��@

error_R�">?

learning_rate_1��5����I       6%�	I�c���A�*;


total_loss�t�@

error_R�<?

learning_rate_1��5��rI       6%�	�c���A�*;


total_loss- A

error_R\�P?

learning_rate_1��5��fI       6%�	t�c���A�*;


total_lossޠ�@

error_R�V@?

learning_rate_1��5!�rI       6%�	p-�c���A�*;


total_loss-��@

error_R3^F?

learning_rate_1��5ך�dI       6%�	�z�c���A�*;


total_loss�	�@

error_Rc�;?

learning_rate_1��5h�9I       6%�	�ϙc���A�*;


total_loss{��@

error_R��R?

learning_rate_1��5�SƦI       6%�	��c���A�*;


total_loss2��@

error_RR`Q?

learning_rate_1��52!�I       6%�	&]�c���A�*;


total_lossđ�@

error_R��R?

learning_rate_1��5����I       6%�	t��c���A�*;


total_loss��@

error_R|je?

learning_rate_1��5"���I       6%�	��c���A�*;


total_loss�4�@

error_RxI?

learning_rate_1��5w�j	I       6%�	x7�c���A�*;


total_loss�F{@

error_RR@B?

learning_rate_1��5�Z�I       6%�	�c���A�*;


total_lossL��@

error_R�Q:?

learning_rate_1��5s�65I       6%�		ěc���A�*;


total_loss��@

error_R11A?

learning_rate_1��5�;�I       6%�	�	�c���A�*;


total_loss���@

error_R�YI?

learning_rate_1��52��	I       6%�	�N�c���A�*;


total_loss��@

error_Ra�W?

learning_rate_1��5O���I       6%�	�c���A�*;


total_loss[��@

error_R�eH?

learning_rate_1��5 I��I       6%�	�ޜc���A�*;


total_loss��@

error_R�V?

learning_rate_1��5ث3fI       6%�	>0�c���A�*;


total_loss:A�@

error_R��F?

learning_rate_1��5=5�]I       6%�	o��c���A�*;


total_loss	4�@

error_R�kN?

learning_rate_1��5۳��I       6%�	�՝c���A�*;


total_loss)��@

error_R{�W?

learning_rate_1��5��+�I       6%�	��c���A�*;


total_loss��@

error_R�\?

learning_rate_1��5��I       6%�	�h�c���A�*;


total_lossJ�@

error_R�A?

learning_rate_1��5
S�I       6%�	a��c���A�*;


total_loss�e�@

error_R�H?

learning_rate_1��5e2�I       6%�	��c���A�*;


total_lossyTA

error_RҐP?

learning_rate_1��5a&�DI       6%�	�R�c���A�*;


total_loss���@

error_R��I?

learning_rate_1��5Z2�tI       6%�	���c���A�*;


total_loss���@

error_R�B?

learning_rate_1��58RrI       6%�	d�c���A�*;


total_loss���@

error_R��E?

learning_rate_1��5$��I       6%�	�%�c���A�*;


total_loss���@

error_R�6J?

learning_rate_1��5|Sq�I       6%�	�j�c���A�*;


total_loss���@

error_RN?

learning_rate_1��5���I       6%�	ů�c���A�*;


total_lossv@k@

error_ROYh?

learning_rate_1��5��{ZI       6%�	��c���A�*;


total_loss�s�@

error_R�1H?

learning_rate_1��5�v�:I       6%�	G8�c���A�*;


total_loss�$�@

error_RZ�P?

learning_rate_1��5���tI       6%�	�y�c���A�*;


total_lossY�@

error_R ZD?

learning_rate_1��5�?�I       6%�	r��c���A�*;


total_lossq��@

error_R �B?

learning_rate_1��5&2�I       6%�	��c���A�*;


total_lossؐ�@

error_R�j??

learning_rate_1��5����I       6%�	I�c���A�*;


total_loss
��@

error_R.�J?

learning_rate_1��5#�xzI       6%�	���c���A�*;


total_loss�A

error_R-�<?

learning_rate_1��5*�I       6%�	�Ѣc���A�*;


total_loss?�@

error_Rd�U?

learning_rate_1��5X\BI       6%�	&�c���A�*;


total_loss��@

error_R�L?

learning_rate_1��5X%�I       6%�	�Z�c���A�*;


total_loss�A�@

error_RQ@?

learning_rate_1��5v�FVI       6%�	.��c���A�*;


total_loss�9�@

error_R�P?

learning_rate_1��5��a�I       6%�	��c���A�*;


total_losso �@

error_RvWT?

learning_rate_1��5�_��I       6%�	|"�c���A�*;


total_loss۩�@

error_R��A?

learning_rate_1��5{;�I       6%�	e�c���A�*;


total_loss��@

error_R�cF?

learning_rate_1��5���I       6%�	w��c���A�*;


total_loss���@

error_RĸF?

learning_rate_1��5U�[I       6%�	��c���A�*;


total_lossW��@

error_Ra�Q?

learning_rate_1��5
��I       6%�	y1�c���A�*;


total_loss��@

error_Rd�??

learning_rate_1��5�|�OI       6%�	���c���A�*;


total_loss-��@

error_R\�O?

learning_rate_1��5L/xLI       6%�	vѥc���A�*;


total_loss��@

error_Rפa?

learning_rate_1��5�x\9I       6%�	i�c���A�*;


total_loss��@

error_R
�R?

learning_rate_1��5յA�I       6%�	qW�c���A�*;


total_loss;d�@

error_R)�I?

learning_rate_1��5�l�I       6%�	Y��c���A�*;


total_lossC��@

error_R��L?

learning_rate_1��5��g�I       6%�	�ަc���A�*;


total_loss�@

error_Re�L?

learning_rate_1��5�#I       6%�	!�c���A�*;


total_lossC��@

error_ROO?

learning_rate_1��5s�]I       6%�	�c�c���A�*;


total_loss,ɂ@

error_RR�@?

learning_rate_1��5�?m�I       6%�	ʨ�c���A�*;


total_loss���@

error_R��U?

learning_rate_1��5).t I       6%�	��c���A�*;


total_loss]�@

error_R`�R?

learning_rate_1��5��F-I       6%�	�1�c���A�*;


total_loss{��@

error_RZ9?

learning_rate_1��5�+��I       6%�	�y�c���A�*;


total_loss�=A

error_R=?

learning_rate_1��5�ȣ�I       6%�	��c���A�*;


total_loss=�@

error_R�N?

learning_rate_1��5��|!I       6%�	� �c���A�*;


total_lossU/�@

error_R\A?

learning_rate_1��5:B8I       6%�	C�c���A�*;


total_loss��@

error_Rr�L?

learning_rate_1��5P �I       6%�	���c���A�*;


total_loss�ӭ@

error_R�M?

learning_rate_1��5�Ģ�I       6%�	�˩c���A�*;


total_lossv�@

error_R�K?

learning_rate_1��5e���I       6%�	��c���A�*;


total_loss-O�@

error_R�T?

learning_rate_1��5�,\I       6%�	�Q�c���A�*;


total_loss�5�@

error_R�r^?

learning_rate_1��5�4�I       6%�	��c���A�*;


total_lossW��@

error_R��H?

learning_rate_1��58��;I       6%�	N٪c���A�*;


total_loss��@

error_R1�E?

learning_rate_1��5�ާ�I       6%�	��c���A�*;


total_loss�m�@

error_RC�W?

learning_rate_1��5)?jI       6%�	]�c���A�*;


total_loss�t@

error_R?�Z?

learning_rate_1��5H�DI       6%�	���c���A�*;


total_loss� �@

error_R�^W?

learning_rate_1��5�[�
I       6%�	��c���A�*;


total_loss�~�@

error_R��E?

learning_rate_1��5��I       6%�	�$�c���A�*;


total_loss�Λ@

error_R�D?

learning_rate_1��5��zyI       6%�	ry�c���A�*;


total_lossnG�@

error_R��J?

learning_rate_1��5�f�I       6%�	�¬c���A�*;


total_lossD�@

error_R�"Z?

learning_rate_1��5d�I       6%�	�
�c���A�*;


total_loss��@

error_Rd�Q?

learning_rate_1��5���I       6%�	Cy�c���A�*;


total_loss6�A

error_R:Z[?

learning_rate_1��5^���I       6%�	�­c���A�*;


total_loss,n�@

error_R�G?

learning_rate_1��5]bK�I       6%�	��c���A�*;


total_loss�J�@

error_R
�D?

learning_rate_1��5�I       6%�	IH�c���A�*;


total_loss?�@

error_R�T?

learning_rate_1��5��gPI       6%�	���c���A�*;


total_lossf�@

error_Ra�Z?

learning_rate_1��5$CSI       6%�	;Ϯc���A�*;


total_loss8>�@

error_RM�>?

learning_rate_1��5"Zo�I       6%�	��c���A�*;


total_loss%(�@

error_RgF?

learning_rate_1��5J��(I       6%�	vS�c���A�*;


total_loss.~�@

error_R��X?

learning_rate_1��5/gb�I       6%�	G��c���A�*;


total_loss��@

error_R$�V?

learning_rate_1��5tR�tI       6%�	�ܯc���A�*;


total_loss髪@

error_Ra�X?

learning_rate_1��5-��I       6%�	T�c���A�*;


total_loss���@

error_RO�C?

learning_rate_1��5���I       6%�	f�c���A�*;


total_loss,��@

error_R�~E?

learning_rate_1��53D�I       6%�	���c���A�*;


total_loss۬@

error_R��@?

learning_rate_1��5vq�{I       6%�	��c���A�*;


total_loss���@

error_R>]?

learning_rate_1��5A_/LI       6%�	�0�c���A�*;


total_loss/C�@

error_Rn�:?

learning_rate_1��5���0I       6%�	t�c���A�*;


total_loss�~�@

error_R"R?

learning_rate_1��5YCf�I       6%�	7��c���A�*;


total_lossT��@

error_RL�D?

learning_rate_1��5?M��I       6%�	p��c���A�*;


total_loss1�h@

error_R!b?

learning_rate_1��5��3�I       6%�	�@�c���A�*;


total_loss�2�@

error_R�yH?

learning_rate_1��5��@I       6%�	���c���A�*;


total_loss־4A

error_R8H?

learning_rate_1��5;���I       6%�	ϲc���A�*;


total_lossz�@

error_Rd�O?

learning_rate_1��5���I       6%�	��c���A�*;


total_loss嶛@

error_R�M?

learning_rate_1��5����I       6%�	�Z�c���A�*;


total_loss��@

error_Rx�>?

learning_rate_1��5ͫ�I       6%�	���c���A�*;


total_losss��@

error_R\5O?

learning_rate_1��5�B DI       6%�	3�c���A�*;


total_loss[<�@

error_Rr"F?

learning_rate_1��5�tz�I       6%�	�$�c���A�*;


total_loss�W�@

error_R�oY?

learning_rate_1��5`�5�I       6%�	i�c���A�*;


total_loss���@

error_R��P?

learning_rate_1��59�I       6%�	���c���A�*;


total_lossr�@

error_R��>?

learning_rate_1��5B�{=I       6%�	��c���A�*;


total_loss�A

error_R?E?

learning_rate_1��5�"6I       6%�	�6�c���A�*;


total_loss]p�@

error_R$B?

learning_rate_1��5-o��I       6%�	�x�c���A�*;


total_loss8ز@

error_R�iF?

learning_rate_1��54�2vI       6%�	G��c���A�*;


total_lossb�@

error_R��X?

learning_rate_1��5Y\J]I       6%�	:�c���A�*;


total_loss�@

error_R[?

learning_rate_1��5Vi-�I       6%�	�H�c���A�*;


total_lossџ�@

error_R�L?

learning_rate_1��5E�$�I       6%�	g��c���A�*;


total_loss{�@

error_R��J?

learning_rate_1��54���I       6%�	Z϶c���A�*;


total_lossFo�@

error_R6/O?

learning_rate_1��5�_�.I       6%�	��c���A�*;


total_loss��@

error_R,M?

learning_rate_1��5d�e�I       6%�	�W�c���A�*;


total_loss�v�@

error_R�`?

learning_rate_1��5c��I       6%�	���c���A�*;


total_lossȶ�@

error_R,zK?

learning_rate_1��5��6 I       6%�	��c���A�*;


total_loss��A

error_R�\O?

learning_rate_1��5rWR�I       6%�	�1�c���A�*;


total_loss]�z@

error_R��H?

learning_rate_1��5n77�I       6%�	 u�c���A�*;


total_loss�@

error_RuT?

learning_rate_1��5���I       6%�	9��c���A�*;


total_loss�]A

error_R�"O?

learning_rate_1��5��I       6%�	���c���A�*;


total_lossr�@

error_R�!G?

learning_rate_1��5�&�I       6%�	)[�c���A�*;


total_loss{��@

error_R\V?

learning_rate_1��5eC.ZI       6%�	ѹc���A�*;


total_loss}��@

error_R�T?

learning_rate_1��5���QI       6%�	�c���A�*;


total_lossB�@

error_R�K?

learning_rate_1��5�h�*I       6%�	Q\�c���A�*;


total_losswQ�@

error_Ry>?

learning_rate_1��5x��6I       6%�	
��c���A�*;


total_lossHY@

error_RN?

learning_rate_1��5�3z�I       6%�	��c���A�*;


total_loss&��@

error_R�AW?

learning_rate_1��5h���I       6%�	�.�c���A�*;


total_loss�͸@

error_R@PK?

learning_rate_1��5)8gI       6%�	�z�c���A�*;


total_lossTߝ@

error_RfjF?

learning_rate_1��5�p�I       6%�	>Ļc���A�*;


total_loss��@

error_ReE?

learning_rate_1��51J:�I       6%�	��c���A�*;


total_loss�IA

error_R��R?

learning_rate_1��5�-��I       6%�	�N�c���A�*;


total_loss:��@

error_R��Z?

learning_rate_1��5C���I       6%�	6��c���A�*;


total_loss�U�@

error_R��d?

learning_rate_1��5���I       6%�	}�c���A�*;


total_loss;}�@

error_R�Q?

learning_rate_1��5冯'I       6%�	D/�c���A�*;


total_loss3~�@

error_R��a?

learning_rate_1��5b@8�I       6%�	���c���A�*;


total_loss�T�@

error_R��H?

learning_rate_1��5�=�I       6%�	��c���A�*;


total_loss�oGA

error_R��K?

learning_rate_1��5�o0tI       6%�	�(�c���A�*;


total_lossV3�@

error_R�pM?

learning_rate_1��5�,��I       6%�	�q�c���A�*;


total_loss���@

error_RZfF?

learning_rate_1��5���I       6%�	���c���A�*;


total_loss3�@

error_R�X?

learning_rate_1��5��I       6%�	��c���A�*;


total_lossB��@

error_Rv�K?

learning_rate_1��5�]o�I       6%�	�H�c���A�*;


total_loss{1z@

error_Ra�C?

learning_rate_1��563��I       6%�	Γ�c���A�*;


total_loss�R�@

error_R��F?

learning_rate_1��5)"TTI       6%�	�ڿc���A�*;


total_loss�x@

error_R�#G?

learning_rate_1��5w>�VI       6%�	H�c���A�*;


total_loss��@

error_R��U?

learning_rate_1��5i�I       6%�	�_�c���A�*;


total_lossc�@

error_R�JF?

learning_rate_1��5��TI       6%�	>��c���A�*;


total_loss�A

error_RW�Z?

learning_rate_1��5�x.2I       6%�	p��c���A�*;


total_loss���@

error_R�E?

learning_rate_1��5��*jI       6%�	�5�c���A�*;


total_loss�b�@

error_R�P?

learning_rate_1��5~c�I       6%�	C�c���A�*;


total_loss���@

error_R(Q?

learning_rate_1��59L��I       6%�	A��c���A�*;


total_lossc��@

error_RqF?

learning_rate_1��5���I       6%�	��c���A�*;


total_loss��@

error_Ra�R?

learning_rate_1��5��+I       6%�	�O�c���A�*;


total_losse�A

error_RS?

learning_rate_1��5��:�I       6%�	=��c���A�*;


total_loss���@

error_R�qL?

learning_rate_1��5ƛuI       6%�	S��c���A�*;


total_loss�ۆ@

error_R�nD?

learning_rate_1��58�MI       6%�	��c���A�*;


total_loss@W�@

error_R��??

learning_rate_1��5Jg��I       6%�	�\�c���A�*;


total_loss�W�@

error_R�D?

learning_rate_1��5 p�I       6%�	���c���A�*;


total_loss��@

error_R�EN?

learning_rate_1��5�A��I       6%�	_��c���A�*;


total_loss���@

error_R<s]?

learning_rate_1��5���I       6%�	D3�c���A�*;


total_loss��@

error_R�_L?

learning_rate_1��5k��I       6%�	�y�c���A�*;


total_lossh��@

error_RsAM?

learning_rate_1��5�jemI       6%�	���c���A�*;


total_loss3��@

error_R sA?

learning_rate_1��5۩�I       6%�	���c���A�*;


total_lossR�@

error_RJPB?

learning_rate_1��5^�e;I       6%�	IA�c���A�*;


total_loss�f�@

error_R�mL?

learning_rate_1��5Qh�uI       6%�	3��c���A�*;


total_loss#ē@

error_RP?

learning_rate_1��5��XI       6%�	4��c���A�*;


total_lossɲ�@

error_R\?

learning_rate_1��5���*I       6%�	^�c���A�*;


total_loss�Q�@

error_R�CQ?

learning_rate_1��5�'�QI       6%�	9Y�c���A�*;


total_lossr�@

error_R΅E?

learning_rate_1��5,u#7I       6%�	���c���A�*;


total_loss�h@

error_R�v??

learning_rate_1��5kBI       6%�	���c���A�*;


total_loss\Ŏ@

error_R%�K?

learning_rate_1��5���I       6%�	l,�c���A�*;


total_lossk@

error_RC-Y?

learning_rate_1��5CFI       6%�	�p�c���A�*;


total_loss){A

error_RM?

learning_rate_1��5"���I       6%�	���c���A�*;


total_loss䪅@

error_R��V?

learning_rate_1��5r��I       6%�	3��c���A�*;


total_loss� �@

error_R�}K?

learning_rate_1��5hG��I       6%�	J8�c���A�*;


total_loss���@

error_R�_Z?

learning_rate_1��5�c�[I       6%�	�|�c���A�*;


total_loss���@

error_R�oD?

learning_rate_1��5�.iI       6%�	���c���A�*;


total_loss���@

error_R�NS?

learning_rate_1��5�m�<I       6%�	��c���A�*;


total_lossī�@

error_R(�J?

learning_rate_1��5I#��I       6%�	�O�c���A�*;


total_loss��@

error_R]�L?

learning_rate_1��56�
xI       6%�	���c���A�*;


total_loss-Z�@

error_R��E?

learning_rate_1��5AӃI       6%�	���c���A�*;


total_loss�~@

error_R�5C?

learning_rate_1��5Ye��I       6%�	)�c���A�*;


total_lossƏ�@

error_R�bR?

learning_rate_1��5w�t�I       6%�	m�c���A�*;


total_loss�s�@

error_Rm�Q?

learning_rate_1��5�*@I       6%�	`��c���A�*;


total_loss��@

error_R`J?

learning_rate_1��5�a��I       6%�	���c���A�*;


total_loss-�@

error_R@}U?

learning_rate_1��5�N�I       6%�	�?�c���A�*;


total_loss���@

error_R�yL?

learning_rate_1��5dG�wI       6%�	r��c���A�*;


total_loss�Y�@

error_R44?

learning_rate_1��5
eNI       6%�	���c���A�*;


total_lossW�@

error_Re�L?

learning_rate_1��5���pI       6%�	��c���A�*;


total_loss5��@

error_R�uI?

learning_rate_1��5)0��I       6%�	OZ�c���A�*;


total_loss�bA

error_R�`\?

learning_rate_1��5���-I       6%�	B��c���A�*;


total_loss=B�@

error_RW�D?

learning_rate_1��5���I       6%�	+��c���A�*;


total_loss��@

error_R�5]?

learning_rate_1��5F<<�I       6%�	p�c���A�*;


total_lossz_�@

error_R�/N?

learning_rate_1��5�%vI       6%�	���c���A�*;


total_loss�ݣ@

error_R�yC?

learning_rate_1��5�ڗ�I       6%�	|&�c���A�*;


total_loss��@

error_Ra�Z?

learning_rate_1��5�b�I       6%�	o�c���A�*;


total_loss��@

error_R�UJ?

learning_rate_1��5<��QI       6%�	u��c���A�*;


total_losse�@

error_R��K?

learning_rate_1��5h\n{I       6%�	W��c���A�*;


total_lossBt�@

error_RNO?

learning_rate_1��5��{�I       6%�	0@�c���A�*;


total_loss6<�@

error_R)�S?

learning_rate_1��5��k�I       6%�	d��c���A�*;


total_loss���@

error_R��C?

learning_rate_1��5z�XI       6%�	���c���A�*;


total_lossDښ@

error_R�~O?

learning_rate_1��5���I       6%�	��c���A�*;


total_loss¢@

error_RYB?

learning_rate_1��5�5�I       6%�	}^�c���A�*;


total_loss3W�@

error_R�/f?

learning_rate_1��5���I       6%�	���c���A�*;


total_losse��@

error_R��X?

learning_rate_1��5�v~�I       6%�	���c���A�*;


total_lossa�A

error_R��J?

learning_rate_1��5z�DsI       6%�	�+�c���A�*;


total_lossh��@

error_R]�>?

learning_rate_1��5��G|I       6%�	q�c���A�*;


total_lossq�@

error_RD*Q?

learning_rate_1��5��I       6%�		��c���A�*;


total_loss(٤@

error_R;�J?

learning_rate_1��5�o�I       6%�	f��c���A�*;


total_loss}  A

error_Rl�J?

learning_rate_1��5[�0ZI       6%�	2E�c���A�*;


total_loss���@

error_Rv?;?

learning_rate_1��5Ǳ�5I       6%�	��c���A�*;


total_loss�ck@

error_R�\K?

learning_rate_1��5���I       6%�	���c���A�*;


total_loss�*�@

error_RR0O?

learning_rate_1��5� f�I       6%�	��c���A�*;


total_loss��@

error_R�NI?

learning_rate_1��5<�gI       6%�	V�c���A�*;


total_loss=��@

error_RdlY?

learning_rate_1��5b[��I       6%�	���c���A�*;


total_lossn�@

error_R!�>?

learning_rate_1��5̘ 2I       6%�	Z��c���A�*;


total_loss_ծ@

error_R�C?

learning_rate_1��5I�I       6%�	|%�c���A�*;


total_loss-��@

error_R�2U?

learning_rate_1��5}U�I       6%�	�f�c���A�*;


total_loss;��@

error_R.�R?

learning_rate_1��5qr&�I       6%�	��c���A�*;


total_losstw�@

error_RW:k?

learning_rate_1��5)ˁI       6%�	���c���A�*;


total_lossr/�@

error_R��U?

learning_rate_1��5�%�I       6%�	]0�c���A�*;


total_lossL7�@

error_RL�G?

learning_rate_1��5��8CI       6%�	s�c���A�*;


total_lossZA

error_R	wH?

learning_rate_1��5��I       6%�	ȵ�c���A�*;


total_loss��@

error_RWdR?

learning_rate_1��5>�x�I       6%�	���c���A�*;


total_loss���@

error_R�LL?

learning_rate_1��5���I       6%�	�>�c���A�*;


total_loss���@

error_RϳJ?

learning_rate_1��54>�+I       6%�	���c���A�*;


total_loss+�@

error_R6uA?

learning_rate_1��5�Ɠ�I       6%�	Q��c���A�*;


total_loss�?�@

error_R�@?

learning_rate_1��5gig�I       6%�	r�c���A�*;


total_loss�҅@

error_R�M?

learning_rate_1��5�ʼ
I       6%�	_�c���A�*;


total_loss�ʋ@

error_RlV?

learning_rate_1��5�
�@I       6%�	B��c���A�*;


total_loss���@

error_R~F?

learning_rate_1��5����I       6%�	���c���A�*;


total_loss�g�@

error_RVX?

learning_rate_1��5=��%I       6%�	�5�c���A�*;


total_loss��@

error_R��b?

learning_rate_1��5�j�
I       6%�	^��c���A�*;


total_lossEEA

error_RbQ?

learning_rate_1��5�ԛ�I       6%�	o��c���A�*;


total_loss��@

error_R�%V?

learning_rate_1��5��p*I       6%�	��c���A�*;


total_loss ��@

error_R�37?

learning_rate_1��5|��cI       6%�	�`�c���A�*;


total_loss�`�@

error_R�$`?

learning_rate_1��5��k�I       6%�	���c���A�*;


total_loss�Ȕ@

error_RV�T?

learning_rate_1��5��>I       6%�	���c���A�*;


total_loss7��@

error_R|_?

learning_rate_1��5q�l�I       6%�	�>�c���A�*;


total_loss���@

error_R��V?

learning_rate_1��5�$�xI       6%�	���c���A�*;


total_loss�C�@

error_R��??

learning_rate_1��5�I       6%�	���c���A�*;


total_loss=A�@

error_R�E?

learning_rate_1��5ʜI       6%�	��c���A�*;


total_lossݺ@

error_R�qD?

learning_rate_1��5-B-)I       6%�	0U�c���A�*;


total_lossa��@

error_RC�B?

learning_rate_1��5yʝI       6%�	X��c���A�*;


total_lossS��@

error_R�%K?

learning_rate_1��5��&I       6%�	���c���A�*;


total_losstO�@

error_RzNJ?

learning_rate_1��5nW��I       6%�	��c���A�*;


total_loss���@

error_R�	Y?

learning_rate_1��5�:�I       6%�	b�c���A�*;


total_loss��@

error_R�*G?

learning_rate_1��5�m�,I       6%�	���c���A�*;


total_loss���@

error_R��@?

learning_rate_1��5����I       6%�	���c���A�*;


total_loss��@

error_R��C?

learning_rate_1��5���I       6%�	8�c���A�*;


total_loss���@

error_R}�L?

learning_rate_1��5�U�?I       6%�	���c���A�*;


total_loss��@

error_R��R?

learning_rate_1��5G��I       6%�	���c���A�*;


total_loss�dP@

error_R��C?

learning_rate_1��5���:I       6%�	!+�c���A�*;


total_lossC�@

error_R��L?

learning_rate_1��5��VI       6%�	�s�c���A�*;


total_loss(@�@

error_R�KO?

learning_rate_1��5%�4I       6%�	!��c���A�*;


total_loss���@

error_RM�^?

learning_rate_1��5i�&�I       6%�	@��c���A�*;


total_lossL��@

error_R@V?

learning_rate_1��5�l,I       6%�	E�c���A�*;


total_lossA�@

error_R�h?

learning_rate_1��5l73I       6%�	x��c���A�*;


total_loss��@

error_R�"W?

learning_rate_1��5j���I       6%�	���c���A�*;


total_loss���@

error_R��N?

learning_rate_1��5	&ϺI       6%�	a�c���A�*;


total_loss��@

error_Rq�O?

learning_rate_1��5�nRI       6%�	�^�c���A�*;


total_lossn�f@

error_RRfJ?

learning_rate_1��5z'6�I       6%�	���c���A�*;


total_loss�2�@

error_R8�0?

learning_rate_1��59�ŶI       6%�	s��c���A�*;


total_loss֧�@

error_R�'8?

learning_rate_1��5*1��I       6%�	u4�c���A�*;


total_loss�N�@

error_R��P?

learning_rate_1��5�ض�I       6%�	�~�c���A�*;


total_loss�D�@

error_R߳^?

learning_rate_1��5,\
�I       6%�	���c���A�*;


total_loss-W	A

error_R�K?

learning_rate_1��5JdZ_I       6%�	�c���A�*;


total_loss�gA

error_R�F?

learning_rate_1��56qwI       6%�	�N�c���A�*;


total_loss��A

error_RyZ?

learning_rate_1��5��0�I       6%�	��c���A�*;


total_loss�k�@

error_R.TR?

learning_rate_1��5��5I       6%�	 ��c���A�*;


total_loss�ܦ@

error_R��[?

learning_rate_1��5Y�مI       6%�	K�c���A�*;


total_loss�'�@

error_R��W?

learning_rate_1��5�ŤI       6%�	!a�c���A�*;


total_lossE�@

error_R��K?

learning_rate_1��5G�?I       6%�	Ҥ�c���A�*;


total_loss@�|@

error_R%.O?

learning_rate_1��5�E.�I       6%�	���c���A�*;


total_loss#
A

error_R�9?

learning_rate_1��5U�,�I       6%�	7�c���A�*;


total_loss-PA

error_R�]?

learning_rate_1��5���I       6%�	|{�c���A�*;


total_loss��A

error_R;G?

learning_rate_1��5�,�7I       6%�	���c���A�*;


total_loss��A

error_R��G?

learning_rate_1��5MpՒI       6%�	��c���A�*;


total_loss�<�@

error_R@E?

learning_rate_1��5|��EI       6%�	W�c���A�*;


total_lossCA

error_RnUN?

learning_rate_1��5}`�I       6%�	��c���A�*;


total_loss�v�@

error_Ri�M?

learning_rate_1��5��UNI       6%�	H��c���A�*;


total_loss���@

error_R�G?

learning_rate_1��5c'�FI       6%�	m!�c���A�*;


total_loss}q�@

error_R��@?

learning_rate_1��5+\�WI       6%�	�h�c���A�*;


total_loss;�@

error_R}U??

learning_rate_1��5���yI       6%�	c��c���A�*;


total_loss̮�@

error_Rl�>?

learning_rate_1��5jY]I       6%�	���c���A�*;


total_loss ��@

error_R�]7?

learning_rate_1��5���SI       6%�	5�c���A�*;


total_lossR�A

error_R��D?

learning_rate_1��5px�I       6%�	�w�c���A�*;


total_lossS�A

error_RJ�??

learning_rate_1��5��5lI       6%�	X��c���A�*;


total_loss �b@

error_R;0M?

learning_rate_1��5���I       6%�	��c���A�*;


total_lossxH�@

error_R�?T?

learning_rate_1��5wL�I       6%�	jV�c���A�*;


total_loss�7�@

error_R)MU?

learning_rate_1��5��JI       6%�	ܬ�c���A�*;


total_loss�*w@

error_R��C?

learning_rate_1��5�=ժI       6%�	(��c���A�*;


total_loss�m�@

error_RT�N?

learning_rate_1��5�T�I       6%�	�H�c���A�*;


total_losst��@

error_R�:O?

learning_rate_1��5�q8>I       6%�	D��c���A�*;


total_loss���@

error_R��O?

learning_rate_1��55k�dI       6%�	���c���A�*;


total_lossT��@

error_R�	=?

learning_rate_1��55��I       6%�	>�c���A�*;


total_loss�|@

error_R�J?

learning_rate_1��5_��I       6%�	��c���A�*;


total_losss>�@

error_R�W?

learning_rate_1��5ro�eI       6%�	���c���A�*;


total_loss��@

error_R��O?

learning_rate_1��5���I       6%�	��c���A�*;


total_loss{x�@

error_R�~P?

learning_rate_1��5��_�I       6%�	�U�c���A�*;


total_loss
A

error_R�S?

learning_rate_1��5A��]I       6%�	W��c���A�*;


total_loss,��@

error_R�`^?

learning_rate_1��5r��gI       6%�	[��c���A�*;


total_lossɂ�@

error_R6�9?

learning_rate_1��5���?I       6%�	�)�c���A�*;


total_loss2��@

error_R�$]?

learning_rate_1��5�V�I       6%�	w�c���A�*;


total_lossar�@

error_R�M?

learning_rate_1��5c���I       6%�	��c���A�*;


total_lossJ��@

error_Rv3V?

learning_rate_1��5M̭2I       6%�	��c���A�*;


total_loss��@

error_R�EV?

learning_rate_1��5z9��I       6%�	}b�c���A�*;


total_lossFx�@

error_R:�L?

learning_rate_1��5�c��I       6%�	��c���A�*;


total_loss�+�@

error_R]aR?

learning_rate_1��5c�I       6%�	���c���A�*;


total_loss�"�@

error_RiJ?

learning_rate_1��5;̸vI       6%�	9F�c���A�*;


total_loss6ժ@

error_Ra�6?

learning_rate_1��5`�nbI       6%�	���c���A�*;


total_lossѕ�@

error_R�C?

learning_rate_1��5I��cI       6%�	���c���A�*;


total_lossm�@

error_R�TT?

learning_rate_1��5�"�I       6%�	��c���A�*;


total_loss�م@

error_R�(a?

learning_rate_1��5G6��I       6%�	�U�c���A�*;


total_losss��@

error_R��N?

learning_rate_1��5h;��I       6%�	���c���A�*;


total_loss`@

error_RC�K?

learning_rate_1��5l[.sI       6%�	R��c���A�*;


total_lossZ�@

error_RZ�L?

learning_rate_1��5V��I       6%�	�c���A�*;


total_loss�*�@

error_R$H?

learning_rate_1��5�|?I       6%�	�e�c���A�*;


total_loss*��@

error_R&_?

learning_rate_1��5ǒI       6%�	���c���A�*;


total_loss��@

error_RsyI?

learning_rate_1��5x��vI       6%�	 ��c���A�*;


total_loss4ޢ@

error_R�M?

learning_rate_1��5�� I       6%�	�>�c���A�*;


total_loss���@

error_R&rS?

learning_rate_1��5x���I       6%�	���c���A�*;


total_loss��A

error_R��Y?

learning_rate_1��5!I       6%�	��c���A�*;


total_loss�Ů@

error_Rf7@?

learning_rate_1��5�J�I       6%�	�c���A�*;


total_loss���@

error_RE>F?

learning_rate_1��5�JجI       6%�	�^�c���A�*;


total_lossc6�@

error_R��B?

learning_rate_1��5:��DI       6%�	��c���A�*;


total_loss�.�@

error_R�V?

learning_rate_1��5��k�I       6%�	:��c���A�*;


total_loss���@

error_R�MM?

learning_rate_1��5�O|I       6%�	c/�c���A�*;


total_loss}��@

error_RazZ?

learning_rate_1��5.���I       6%�	Up�c���A�*;


total_loss�^�@

error_R�iD?

learning_rate_1��5�D'I       6%�	|��c���A�*;


total_loss�w�@

error_R�g?

learning_rate_1��5j��rI       6%�	���c���A�*;


total_loss�ǘ@

error_RA�H?

learning_rate_1��5}e�I       6%�	9;�c���A�*;


total_loss���@

error_R��??

learning_rate_1��53��*I       6%�	ł�c���A�*;


total_loss���@

error_R1V?

learning_rate_1��5xA_�I       6%�	O��c���A�*;


total_lossL�@

error_Rȑ@?

learning_rate_1��5�_`�I       6%�	u�c���A�*;


total_loss���@

error_Rz�P?

learning_rate_1��5(��I       6%�	P�c���A�*;


total_loss@

error_R�??

learning_rate_1��5��0I       6%�	ה�c���A�*;


total_loss8��@

error_R�|8?

learning_rate_1��5~�4I       6%�	��c���A�*;


total_loss�D�@

error_R,>T?

learning_rate_1��5EKv�I       6%�	_�c���A�*;


total_loss� �@

error_R��V?

learning_rate_1��5��I       6%�	^�c���A�*;


total_lossҬ�@

error_R�mM?

learning_rate_1��5�N\7I       6%�	���c���A�*;


total_lossl�@

error_RcBK?

learning_rate_1��5�`7�I       6%�	���c���A�*;


total_loss�d�@

error_RؼM?

learning_rate_1��5�%�I       6%�	l:�c���A�*;


total_lossa8�@

error_R�W?

learning_rate_1��5m�r�I       6%�	���c���A�*;


total_loss��@

error_R�Kh?

learning_rate_1��5ECI       6%�	���c���A�*;


total_loss�	A

error_R�M?

learning_rate_1��5�[�
I       6%�	d�c���A�*;


total_loss�.�@

error_R��O?

learning_rate_1��5��)�I       6%�	yS�c���A�*;


total_loss1��@

error_R�X?

learning_rate_1��5Q��I       6%�	%��c���A�*;


total_loss���@

error_R��^?

learning_rate_1��5?>�I       6%�	8��c���A�*;


total_loss.:�@

error_R �^?

learning_rate_1��5�&�I       6%�	�'�c���A�*;


total_loss!w�@

error_R1�Z?

learning_rate_1��5�L�DI       6%�	}n�c���A�*;


total_loss�A�@

error_R��S?

learning_rate_1��55?��I       6%�	��c���A�*;


total_lossQ��@

error_R�T?

learning_rate_1��59�xI       6%�	���c���A�*;


total_lossC��@

error_Ra'Q?

learning_rate_1��5�
kI       6%�	�4�c���A�*;


total_loss�j�@

error_RfM?

learning_rate_1��5��I       6%�	�z�c���A�*;


total_lossm�@

error_R3qZ?

learning_rate_1��5<B�lI       6%�	V��c���A�*;


total_loss�	A

error_R�0N?

learning_rate_1��5��#I       6%�	��c���A�*;


total_loss�'�@

error_R�dN?

learning_rate_1��5���CI       6%�	�K�c���A�*;


total_loss,��@

error_R�B?

learning_rate_1��5F+)II       6%�	���c���A�*;


total_loss���@

error_R�2L?

learning_rate_1��5'{�I       6%�	a��c���A�*;


total_lossm�@

error_R׼^?

learning_rate_1��5�&цI       6%�	#"�c���A�*;


total_loss!��@

error_R_^g?

learning_rate_1��5��	RI       6%�	ke�c���A�*;


total_loss��@

error_R��K?

learning_rate_1��53�OI       6%�	���c���A�*;


total_loss���@

error_R'W?

learning_rate_1��5:�xI       6%�	���c���A�*;


total_lossFg�@

error_R8:?

learning_rate_1��5L�:qI       6%�	�=�c���A�*;


total_loss��@

error_RL�;?

learning_rate_1��5�` �I       6%�	ɝ�c���A�*;


total_lossm�A

error_R4�a?

learning_rate_1��5�1��I       6%�	���c���A�*;


total_loss��@

error_R&gS?

learning_rate_1��5 �-�I       6%�	q&�c���A�*;


total_loss��@

error_RҍC?

learning_rate_1��5U�I       6%�	|n�c���A�*;


total_loss��@

error_R��I?

learning_rate_1��5���I       6%�	���c���A�*;


total_loss|�@

error_R�P?

learning_rate_1��5f%�I       6%�	m��c���A�*;


total_lossYТ@

error_Rr??

learning_rate_1��5)��+I       6%�	J;�c���A�*;


total_loss��@

error_R�xb?

learning_rate_1��5����I       6%�	���c���A�*;


total_loss)V�@

error_R�?C?

learning_rate_1��5���I       6%�	���c���A�*;


total_loss��@

error_R�O?

learning_rate_1��5����I       6%�	� d���A�*;


total_loss�3�@

error_R��J?

learning_rate_1��5��اI       6%�	J d���A�*;


total_loss�^�@

error_Rf�N?

learning_rate_1��5c�ܼI       6%�	� d���A�*;


total_loss���@

error_R�QQ?

learning_rate_1��56Ѥ"I       6%�	�� d���A�*;


total_loss�X�@

error_R_T?

learning_rate_1��5\��I       6%�	d���A�*;


total_lossa��@

error_R��R?

learning_rate_1��5��t3I       6%�	�Zd���A�*;


total_lossӐ�@

error_R�2F?

learning_rate_1��5���`I       6%�	�d���A�*;


total_loss:,�@

error_RvOB?

learning_rate_1��5e�ӴI       6%�	��d���A�*;


total_loss�L�@

error_RW%N?

learning_rate_1��5FE�I       6%�	O-d���A�*;


total_loss�ܫ@

error_RVL?

learning_rate_1��5�Tv�I       6%�	%rd���A�*;


total_loss�s�@

error_R��P?

learning_rate_1��5�\��I       6%�	�d���A�*;


total_loss�[A

error_R�X?

learning_rate_1��5��I       6%�	T�d���A�*;


total_loss'�A

error_R�N?

learning_rate_1��5p�CII       6%�	oGd���A�*;


total_loss�i�@

error_R�W?

learning_rate_1��5yv�I       6%�	�d���A�*;


total_loss�۬@

error_R��C?

learning_rate_1��5F-7I       6%�	��d���A�*;


total_loss\cy@

error_R�)=?

learning_rate_1��50�6hI       6%�	�d���A�*;


total_loss�D�@

error_R1�>?

learning_rate_1��5�9��I       6%�	�Xd���A�*;


total_lossq��@

error_Rx�_?

learning_rate_1��5���I       6%�	�d���A�*;


total_lossr]�@

error_R�X?

learning_rate_1��5<�+<I       6%�	�d���A�*;


total_losssZl@

error_R3�Q?

learning_rate_1��5iOcI       6%�	�%d���A�*;


total_loss���@

error_R�XJ?

learning_rate_1��5K�lpI       6%�	�id���A�*;


total_lossm��@

error_R��\?

learning_rate_1��5���I       6%�	��d���A�*;


total_loss5�@

error_R�$6?

learning_rate_1��5_��I       6%�	�d���A�*;


total_loss�2�@

error_Rc�B?

learning_rate_1��5���I       6%�	�1d���A�*;


total_lossC �@

error_R/09?

learning_rate_1��5�7AI       6%�	^vd���A�*;


total_loss�fA

error_R��Z?

learning_rate_1��5��fI       6%�	8�d���A�*;


total_loss{�@

error_R-gD?

learning_rate_1��5z��XI       6%�	��d���A�*;


total_losss��@

error_R��=?

learning_rate_1��5QdI       6%�	Bd���A�*;


total_loss*��@

error_R�{V?

learning_rate_1��5�fv(I       6%�	��d���A�*;


total_loss
0�@

error_Rnp@?

learning_rate_1��5qq��I       6%�	��d���A�*;


total_loss���@

error_R��X?

learning_rate_1��5�d�I       6%�	d���A�*;


total_loss7[�@

error_R�^J?

learning_rate_1��5�(�HI       6%�	�fd���A�*;


total_lossT��@

error_R�N?

learning_rate_1��5m�>I       6%�	L�d���A�*;


total_loss��@

error_R
�O?

learning_rate_1��5�	I       6%�	j	d���A�*;


total_loss��b@

error_R�N5?

learning_rate_1��5�mp�I       6%�	;L	d���A�*;


total_loss3	A

error_RM�K?

learning_rate_1��562#�I       6%�	�	d���A�*;


total_loss���@

error_R�c?

learning_rate_1��5b;%I       6%�	�	d���A�*;


total_loss$<�@

error_R��B?

learning_rate_1��5@��I       6%�	�>
d���A�*;


total_lossv�"A

error_R�N?

learning_rate_1��5�Q}I       6%�	х
d���A�*;


total_loss��@

error_RԹI?

learning_rate_1��5"��]I       6%�	��
d���A�*;


total_loss���@

error_RJPY?

learning_rate_1��5�(��I       6%�	Nd���A�*;


total_loss���@

error_R!(N?

learning_rate_1��5��CcI       6%�	�Xd���A�*;


total_loss��@

error_R�H?

learning_rate_1��5V��I       6%�	��d���A�*;


total_loss۶@

error_RM5?

learning_rate_1��5�g�I       6%�	!�d���A�*;


total_loss���@

error_R��K?

learning_rate_1��5߻�rI       6%�	�*d���A�*;


total_loss���@

error_R�=P?

learning_rate_1��5�Tx�I       6%�	�ld���A�*;


total_loss�~�@

error_R}5T?

learning_rate_1��5.7aI       6%�	�d���A�*;


total_loss��@

error_R@�Y?

learning_rate_1��54�*I       6%�	�d���A�*;


total_loss��@

error_Rs�;?

learning_rate_1��5Ҏ;�I       6%�	Yhd���A�*;


total_loss''�@

error_RX�K?

learning_rate_1��5�L0I       6%�	;�d���A�*;


total_loss��|@

error_R��O?

learning_rate_1��5ǒe9I       6%�	�d���A�*;


total_lossړ�@

error_R�sV?

learning_rate_1��5��%?I       6%�	�Vd���A�*;


total_loss�Z�@

error_R��P?

learning_rate_1��5��I       6%�	>�d���A�*;


total_loss���@

error_Rw�H?

learning_rate_1��5ɰ�I       6%�	��d���A�*;


total_loss= �@

error_R�P?

learning_rate_1��5���I       6%�	�,d���A�*;


total_loss�:�@

error_R�T?

learning_rate_1��5܊�ZI       6%�	qd���A�*;


total_loss���@

error_R%W?

learning_rate_1��5lM��I       6%�	ηd���A�*;


total_loss��@

error_RvA?

learning_rate_1��5���I       6%�	s�d���A�*;


total_lossܟ�@

error_R�G?

learning_rate_1��5뫫"I       6%�	x<d���A�*;


total_loss�
�@

error_RagU?

learning_rate_1��5�ב�I       6%�	�d���A�*;


total_loss���@

error_R�8?

learning_rate_1��5m��I       6%�	-�d���A�*;


total_lossbeA

error_R��[?

learning_rate_1��5�ԳqI       6%�		d���A�*;


total_lossE�@

error_RfI?

learning_rate_1��5�i|�I       6%�	JMd���A�*;


total_lossC��@

error_R{�f?

learning_rate_1��5I��I       6%�	x�d���A�*;


total_loss�l�@

error_R�T?

learning_rate_1��5�d��I       6%�	F�d���A�*;


total_loss+�@

error_R�3K?

learning_rate_1��5���4I       6%�	�d���A�*;


total_loss��@

error_R��N?

learning_rate_1��5STI       6%�	�^d���A�*;


total_loss���@

error_Rv;X?

learning_rate_1��5]=�I       6%�	ݡd���A�*;


total_loss�#"A

error_Rc�M?

learning_rate_1��5�7u�I       6%�	d�d���A�*;


total_loss�f@

error_RZ8?

learning_rate_1��5�_R�I       6%�	E(d���A�*;


total_loss��@

error_R4.U?

learning_rate_1��5����I       6%�	$jd���A�*;


total_lossMS�@

error_R��I?

learning_rate_1��5�%�I       6%�	��d���A�*;


total_lossF��@

error_R�,M?

learning_rate_1��5M+�=I       6%�	��d���A�*;


total_loss]z�@

error_R��M?

learning_rate_1��5�i�I       6%�	�2d���A�*;


total_loss�<�@

error_RĎf?

learning_rate_1��5/$�I       6%�	�vd���A�*;


total_loss�T�@

error_R@�_?

learning_rate_1��5�D��I       6%�	�d���A�*;


total_loss�A

error_R(�W?

learning_rate_1��5��BI       6%�	��d���A�*;


total_loss(u�@

error_R$�O?

learning_rate_1��5�$�I       6%�	$Ad���A�*;


total_loss�T�@

error_R3�^?

learning_rate_1��5����I       6%�	c�d���A�*;


total_loss%/�@

error_RII?

learning_rate_1��5[k��I       6%�	��d���A�*;


total_lossTt�@

error_R��O?

learning_rate_1��5����I       6%�	d���A�*;


total_lossm�c@

error_R��@?

learning_rate_1��5ᄊI       6%�	@Xd���A�*;


total_loss�R�@

error_R�T?

learning_rate_1��5gKY�I       6%�	s�d���A�*;


total_loss�1�@

error_R�Z?

learning_rate_1��5��I       6%�	a�d���A�*;


total_loss!z@

error_RRC?

learning_rate_1��5ɓE7I       6%�	:$d���A�*;


total_loss�@

error_R�Q=?

learning_rate_1��5_�KI       6%�	�gd���A�*;


total_loss���@

error_RlT?

learning_rate_1��5�t�&I       6%�	+�d���A�*;


total_loss���@

error_R��H?

learning_rate_1��5l$S�I       6%�	�d���A�*;


total_loss�ʍ@

error_R�S?

learning_rate_1��5�ݱtI       6%�	b6d���A�*;


total_loss$�@

error_RAf?

learning_rate_1��5S��I       6%�	�|d���A�*;


total_loss鬴@

error_R��G?

learning_rate_1��5����I       6%�	��d���A�*;


total_lossL`�@

error_R_J?

learning_rate_1��5�P�I       6%�	0d���A�*;


total_lossd�@

error_R��T?

learning_rate_1��5�7nI       6%�	Rd���A�*;


total_loss�)�@

error_R��e?

learning_rate_1��5��;I       6%�	w�d���A�*;


total_lossd��@

error_Rj^R?

learning_rate_1��5٧/II       6%�	7�d���A�*;


total_loss���@

error_RjBQ?

learning_rate_1��5.z1I       6%�	7)d���A�*;


total_lossӨ�@

error_R�S?

learning_rate_1��5�?+zI       6%�	7od���A�*;


total_lossъ@

error_R�H?

learning_rate_1��5���}I       6%�	ϴd���A�*;


total_lossqH�@

error_R-�;?

learning_rate_1��5�_�I       6%�	�d���A�*;


total_lossS�A

error_RaGH?

learning_rate_1��5`f,I       6%�	�=d���A�*;


total_loss� �@

error_R��H?

learning_rate_1��5���I       6%�	b�d���A�*;


total_loss�ɣ@

error_R%&X?

learning_rate_1��5FQ��I       6%�	��d���A�*;


total_lossې�@

error_Rn�`?

learning_rate_1��5��0�I       6%�	!d���A�*;


total_lossTg�@

error_R��W?

learning_rate_1��5r��I       6%�	']d���A�*;


total_loss��@

error_R��Z?

learning_rate_1��5W�|*I       6%�	�d���A�*;


total_lossH�@

error_R�UJ?

learning_rate_1��5Q��DI       6%�	��d���A�*;


total_loss���@

error_R��F?

learning_rate_1��5ݲ��I       6%�	.Nd���A�*;


total_loss��@

error_R�RJ?

learning_rate_1��5�0��I       6%�	o�d���A�*;


total_loss�é@

error_R�Y:?

learning_rate_1��5K�<�I       6%�	��d���A�*;


total_lossNڥ@

error_R|6a?

learning_rate_1��5��mI       6%�	?d���A�*;


total_loss�E�@

error_R.�I?

learning_rate_1��5�!��I       6%�	��d���A�*;


total_loss}ڰ@

error_R�^V?

learning_rate_1��5k ��I       6%�	�d���A�*;


total_loss�^A

error_R��W?

learning_rate_1��5�|�I       6%�	�d���A�*;


total_loss�v�@

error_R*.L?

learning_rate_1��5R]�I       6%�	[d���A�*;


total_lossM��@

error_R�*?

learning_rate_1��5v}�OI       6%�	
�d���A�*;


total_loss��@

error_Re>?

learning_rate_1��5R�I       6%�	^�d���A�*;


total_loss�A�@

error_R�A?

learning_rate_1��5����I       6%�	�$ d���A�*;


total_lossQ�@

error_R2�C?

learning_rate_1��51#rzI       6%�	xi d���A�*;


total_lossl�@

error_R��=?

learning_rate_1��5����I       6%�	�� d���A�*;


total_loss;��@

error_R��M?

learning_rate_1��5\���I       6%�	�� d���A�*;


total_loss֐�@

error_R???

learning_rate_1��5�#*�I       6%�	�3!d���A�*;


total_loss&�l@

error_R��O?

learning_rate_1��5�CWI       6%�	�x!d���A�*;


total_loss	��@

error_R
F?

learning_rate_1��5׼<I       6%�	{�!d���A�*;


total_lossW��@

error_Rv�:?

learning_rate_1��5R��I       6%�	�"d���A�*;


total_loss�"�@

error_R6N?

learning_rate_1��5=P.pI       6%�	qP"d���A�*;


total_loss�=�@

error_R!N?

learning_rate_1��5�޿I       6%�	%�"d���A�*;


total_loss�ɣ@

error_R\FV?

learning_rate_1��5�)��I       6%�	��"d���A�*;


total_loss�_�@

error_Rim>?

learning_rate_1��5�+]I       6%�	J-#d���A�*;


total_lossV˳@

error_RQze?

learning_rate_1��5~{�I       6%�	�u#d���A�*;


total_loss��A

error_R�N?

learning_rate_1��5ok��I       6%�	~�#d���A�*;


total_lossM��@

error_R��Z?

learning_rate_1��5�O�xI       6%�	z$d���A�*;


total_loss�s�@

error_R�&G?

learning_rate_1��5�IJ�I       6%�	�^$d���A�*;


total_lossd��@

error_R�d`?

learning_rate_1��5��(I       6%�	\�$d���A�*;


total_loss���@

error_R��V?

learning_rate_1��5�s@LI       6%�	��$d���A�*;


total_loss)�@

error_R/T?

learning_rate_1��5��_I       6%�	�4%d���A�*;


total_loss�@

error_RߞI?

learning_rate_1��5��M�I       6%�	�%d���A�*;


total_loss�Y�@

error_R�]?

learning_rate_1��5�r�I       6%�	��%d���A�*;


total_loss�k@

error_R $L?

learning_rate_1��5�j	I       6%�	�&d���A�*;


total_loss���@

error_R!�R?

learning_rate_1��5U`VI       6%�	�S&d���A�*;


total_loss_+�@

error_RXR?

learning_rate_1��5�x��I       6%�	��&d���A�*;


total_loss

�@

error_R�/H?

learning_rate_1��5��M�I       6%�	��&d���A�*;


total_lossC�q@

error_R��@?

learning_rate_1��5�gm�I       6%�	�#'d���A�*;


total_loss�F�@

error_RX�F?

learning_rate_1��59���I       6%�	�j'd���A�*;


total_loss�`�@

error_Rl~;?

learning_rate_1��5�4,wI       6%�	̯'d���A�*;


total_loss�9�@

error_RL�Z?

learning_rate_1��5l�%$I       6%�	�'d���A�*;


total_lossɶ�@

error_R�V?

learning_rate_1��5�˔�I       6%�	�=(d���A�*;


total_loss&*�@

error_R=�P?

learning_rate_1��5ڤ�I       6%�	��(d���A�*;


total_loss��@

error_R�lK?

learning_rate_1��5B�#I       6%�	^�(d���A�*;


total_loss��@

error_RX�b?

learning_rate_1��5d/=I       6%�	,%)d���A�*;


total_loss���@

error_R).F?

learning_rate_1��5)zK�I       6%�	Tl)d���A�*;


total_loss��@

error_R�/T?

learning_rate_1��53�;�I       6%�	�)d���A�*;


total_loss?��@

error_R%�=?

learning_rate_1��5.+��I       6%�	X*d���A�*;


total_loss�x�@

error_RR�c?

learning_rate_1��5�;�LI       6%�	L*d���A�*;


total_loss�f�@

error_Rs�J?

learning_rate_1��5�u�I       6%�	[�*d���A�*;


total_loss�O�@

error_R�O?

learning_rate_1��5l�d�I       6%�	�*d���A�*;


total_loss��@

error_R�7V?

learning_rate_1��5"�I       6%�	$'+d���A�*;


total_loss�}A

error_R��k?

learning_rate_1��5D}UI       6%�	4n+d���A�*;


total_loss\�@

error_R��L?

learning_rate_1��5���I       6%�	��+d���A�*;


total_loss_��@

error_RX�H?

learning_rate_1��5)@�I       6%�	,,d���A�*;


total_loss��@

error_R\�V?

learning_rate_1��5�I       6%�	fN,d���A�*;


total_loss<��@

error_R�}P?

learning_rate_1��5���I       6%�	�,d���A�*;


total_loss�>A

error_Rũ4?

learning_rate_1��5�w8RI       6%�	��,d���A�*;


total_lossq��@

error_Ra�_?

learning_rate_1��5[��?I       6%�	;.-d���A�*;


total_loss�L�@

error_Rz�L?

learning_rate_1��5�e��I       6%�	��-d���A�*;


total_loss��@

error_Rd8??

learning_rate_1��5KVI       6%�	��-d���A�*;


total_loss-l�@

error_R!�Z?

learning_rate_1��5����I       6%�	**.d���A�*;


total_loss2��@

error_R��E?

learning_rate_1��5 S��I       6%�	ao.d���A�*;


total_loss,�@

error_R��^?

learning_rate_1��5��-I       6%�	�.d���A�*;


total_lossW;�@

error_R?�Q?

learning_rate_1��5�]�JI       6%�	��.d���A�*;


total_loss���@

error_R��M?

learning_rate_1��5��LI       6%�	�:/d���A�*;


total_loss&;�@

error_R��??

learning_rate_1��5�p�tI       6%�	�}/d���A�*;


total_loss:��@

error_R�I?

learning_rate_1��5�c�I       6%�	J�/d���A�*;


total_loss���@

error_R�{??

learning_rate_1��5��"aI       6%�	D0d���A�*;


total_lossN��@

error_R\>V?

learning_rate_1��5a`�I       6%�	uJ0d���A�*;


total_loss�x�@

error_R�R?

learning_rate_1��51��I       6%�	�0d���A�*;


total_lossw0�@

error_R�L?

learning_rate_1��5M�~I       6%�	��0d���A�*;


total_loss���@

error_R�bF?

learning_rate_1��5����I       6%�	�1d���A�*;


total_lossa8�@

error_R�O?

learning_rate_1��5m�$I       6%�	�g1d���A�*;


total_lossz@

error_R�C?

learning_rate_1��5�7�uI       6%�	Z�1d���A�*;


total_loss���@

error_R�1K?

learning_rate_1��5��xI       6%�	U�1d���A�*;


total_loss�F�@

error_RQ�^?

learning_rate_1��5".�I       6%�	-12d���A�*;


total_loss@�@

error_R&1F?

learning_rate_1��5���I       6%�	�s2d���A�*;


total_loss W�@

error_RX�@?

learning_rate_1��5��yTI       6%�	�2d���A�*;


total_loss7��@

error_ROV?

learning_rate_1��5�~_/I       6%�	@3d���A�*;


total_loss���@

error_R�Lb?

learning_rate_1��5�ĺI       6%�	qO3d���A�*;


total_loss�c�@

error_R�Q?

learning_rate_1��54�	�I       6%�	��3d���A�*;


total_lossO��@

error_R��K?

learning_rate_1��5L���I       6%�	E�3d���A�*;


total_loss,��@

error_R��>?

learning_rate_1��5�JH�I       6%�	�!4d���A�*;


total_lossj�@

error_Rš`?

learning_rate_1��5S=�oI       6%�	i4d���A�*;


total_loss���@

error_R��^?

learning_rate_1��5{���I       6%�	�4d���A�*;


total_loss���@

error_R��Q?

learning_rate_1��5�Z�I       6%�	��4d���A�*;


total_loss=��@

error_R�	@?

learning_rate_1��5�h ZI       6%�	�@5d���A�*;


total_lossͨ�@

error_R_�N?

learning_rate_1��5�%��I       6%�	a�5d���A�*;


total_loss|��@

error_RS?

learning_rate_1��5�G�OI       6%�	�5d���A�*;


total_loss��@

error_R�N?

learning_rate_1��5�`^�I       6%�	e6d���A�*;


total_loss��@

error_R, c?

learning_rate_1��5�ѬI       6%�	Dj6d���A�*;


total_loss�G�@

error_R��Z?

learning_rate_1��5䃵�I       6%�	��6d���A�*;


total_losss��@

error_R��A?

learning_rate_1��5���PI       6%�	��6d���A�*;


total_loss�o@

error_R8Y?

learning_rate_1��5_��NI       6%�	�87d���A�*;


total_loss�'�@

error_Ri.?

learning_rate_1��50>wnI       6%�	�x7d���A�*;


total_loss.��@

error_R��D?

learning_rate_1��57��I       6%�	W�7d���A�*;


total_loss���@

error_R�M?

learning_rate_1��5�=��I       6%�	W�7d���A�*;


total_loss�@

error_Rs.V?

learning_rate_1��5�ũ3I       6%�	WC8d���A�*;


total_lossh/ A

error_R,BA?

learning_rate_1��5��jI       6%�	��8d���A�*;


total_loss��A

error_Rd�Q?

learning_rate_1��5P�z�I       6%�	t�8d���A�*;


total_lossW"_@

error_R��F?

learning_rate_1��5��uI       6%�	$9d���A�*;


total_loss���@

error_R�^P?

learning_rate_1��5�&F�I       6%�	�P9d���A�*;


total_loss%͏@

error_RO?

learning_rate_1��5J�JI       6%�	"�9d���A�*;


total_lossJw�@

error_R��H?

learning_rate_1��5�S�I       6%�	��9d���A�*;


total_loss_h�@

error_R��B?

learning_rate_1��5��~I       6%�	�:d���A�*;


total_lossx��@

error_RK?

learning_rate_1��5v��I       6%�	5b:d���A�*;


total_loss�S�@

error_R�C?

learning_rate_1��5b�7I       6%�	!�:d���A�*;


total_loss�2�@

error_R��a?

learning_rate_1��5� �I       6%�	��:d���A�*;


total_lossN�	A

error_RC�\?

learning_rate_1��5�"�I       6%�	-:;d���A�*;


total_loss!^�@

error_RN�E?

learning_rate_1��5u���I       6%�	��;d���A�*;


total_loss���@

error_R��U?

learning_rate_1��5p�b�I       6%�	5�;d���A�*;


total_loss?��@

error_R��S?

learning_rate_1��5��Q�I       6%�	�<d���A�*;


total_loss�ٍ@

error_R;�L?

learning_rate_1��5Yݎ9I       6%�	�W<d���A�*;


total_loss��@

error_R�M?

learning_rate_1��5|�EOI       6%�	-�<d���A�*;


total_lossh��@

error_RWF?

learning_rate_1��54 "�I       6%�	��<d���A�*;


total_loss���@

error_R� O?

learning_rate_1��5�^S�I       6%�	�5=d���A�*;


total_lossjM�@

error_R�O?

learning_rate_1��5o>�CI       6%�	0�=d���A�*;


total_lossY�A

error_Ra*S?

learning_rate_1��5]�I�I       6%�	��=d���A�*;


total_loss� �@

error_R��N?

learning_rate_1��5���I       6%�	�,>d���A�*;


total_loss(w�@

error_R��J?

learning_rate_1��5vF��I       6%�	�w>d���A�*;


total_loss�	�@

error_Re)\?

learning_rate_1��5�`I       6%�	��>d���A�*;


total_loss��@

error_R��R?

learning_rate_1��5�C�FI       6%�	Z
?d���A�*;


total_loss�y�@

error_R��>?

learning_rate_1��5h�I       6%�	\L?d���A�*;


total_loss�ʤ@

error_R4J@?

learning_rate_1��5+^��I       6%�	��?d���A�*;


total_loss���@

error_R�I?

learning_rate_1��5n�W�I       6%�	��?d���A�*;


total_loss���@

error_R��X?

learning_rate_1��5z�"�I       6%�	@d���A�*;


total_lossO@

error_RjNC?

learning_rate_1��5���dI       6%�	.j@d���A�*;


total_loss��@

error_RcVE?

learning_rate_1��5(i~�I       6%�	=�@d���A�*;


total_loss���@

error_Rx�Y?

learning_rate_1��5$��I       6%�	��@d���A�*;


total_loss:�@

error_RI2Q?

learning_rate_1��5��
OI       6%�	�HAd���A�*;


total_loss��@

error_RR??

learning_rate_1��5O�]vI       6%�	��Ad���A�*;


total_lossF/�@

error_R�%E?

learning_rate_1��5����I       6%�	2�Ad���A�*;


total_loss �@

error_R��H?

learning_rate_1��5�S>I       6%�	*Bd���A�*;


total_loss���@

error_R�U?

learning_rate_1��5F;nlI       6%�	$VBd���A�*;


total_loss	O�@

error_R�$E?

learning_rate_1��5&�O�I       6%�	{�Bd���A�*;


total_lossS-�@

error_R��J?

learning_rate_1��5����I       6%�	��Bd���A�*;


total_loss�ݰ@

error_R�F?

learning_rate_1��5r�zI       6%�	KCd���A�*;


total_lossx5�@

error_R,�F?

learning_rate_1��5.ɚ�I       6%�	cCd���A�*;


total_loss3Ѵ@

error_R:tY?

learning_rate_1��5���I       6%�	��Cd���A�*;


total_losss�@

error_RaL?

learning_rate_1��5�!�jI       6%�	��Cd���A�*;


total_loss���@

error_R��\?

learning_rate_1��5T%��I       6%�	C2Dd���A�*;


total_loss�>�@

error_R��I?

learning_rate_1��5�%yI       6%�	�uDd���A�*;


total_loss�p~@

error_RňR?

learning_rate_1��5�q I       6%�	��Dd���A�*;


total_loss�@

error_RJ3P?

learning_rate_1��5U/�I       6%�	��Dd���A�*;


total_loss;,w@

error_R�U?

learning_rate_1��5�+�{I       6%�	ZAEd���A�*;


total_lossZ�@

error_R�<?

learning_rate_1��5����I       6%�	�Ed���A�*;


total_lossŝ@

error_RbB?

learning_rate_1��5<��I       6%�	��Ed���A�*;


total_loss��@

error_R�7?

learning_rate_1��5�va�I       6%�	�	Fd���A�*;


total_lossj�@

error_R�S?

learning_rate_1��5*��SI       6%�	cOFd���A�*;


total_loss;#�@

error_R8zT?

learning_rate_1��5�cSI       6%�	��Fd���A�*;


total_loss���@

error_R��^?

learning_rate_1��5��2cI       6%�	��Fd���A�*;


total_loss}�x@

error_Rf_N?

learning_rate_1��5��"�I       6%�	� Gd���A�*;


total_loss�S�@

error_R]�@?

learning_rate_1��5Y<�KI       6%�	#gGd���A�*;


total_loss!A

error_R\8P?

learning_rate_1��5�	�I       6%�	��Gd���A�*;


total_loss�>�@

error_R��O?

learning_rate_1��5$e�I       6%�	�Gd���A�*;


total_lossL�@

error_R��L?

learning_rate_1��5���?I       6%�	5Hd���A�*;


total_lossFV�@

error_Ra�C?

learning_rate_1��5�}Z�I       6%�	~Hd���A�*;


total_loss�5�@

error_R��P?

learning_rate_1��5�)��I       6%�	�Kd���A�*;


total_lossߝ�@

error_Ra�[?

learning_rate_1��5r?=I       6%�	q�Kd���A�*;


total_loss��@

error_Rv�S?

learning_rate_1��5�Di;I       6%�	�@Ld���A�*;


total_loss���@

error_R�G\?

learning_rate_1��5<��I       6%�	�Ld���A�*;


total_loss���@

error_R�]E?

learning_rate_1��5�d�I       6%�	v�Ld���A�*;


total_loss�@A

error_RE?

learning_rate_1��5��˵I       6%�	dMd���A�*;


total_lossJm�@

error_RvH?

learning_rate_1��5%� �I       6%�	\�Md���A�*;


total_loss��A

error_R�U?

learning_rate_1��5�
�#I       6%�	q�Md���A�*;


total_loss��@

error_R1e?

learning_rate_1��5��T�I       6%�	�Nd���A�*;


total_loss /�@

error_R!=<?

learning_rate_1��5�|X�I       6%�	_Nd���A�*;


total_lossycA

error_R��P?

learning_rate_1��5���mI       6%�	��Nd���A�*;


total_loss�z�@

error_R[%I?

learning_rate_1��5L��I       6%�	��Nd���A�*;


total_lossq��@

error_RiVF?

learning_rate_1��53��I       6%�	J*Od���A�*;


total_loss��@

error_R�=B?

learning_rate_1��5��6mI       6%�	!nOd���A�*;


total_loss�l�@

error_Rڀb?

learning_rate_1��5b���I       6%�	#�Od���A�*;


total_loss���@

error_R�6W?

learning_rate_1��5-M	I       6%�	}�Od���A�*;


total_lossf�@

error_R{aK?

learning_rate_1��52�5I       6%�	>Pd���A�*;


total_loss���@

error_R��L?

learning_rate_1��5�1�4I       6%�	��Pd���A�*;


total_lossC��@

error_R��E?

learning_rate_1��5�Cw�I       6%�	i�Pd���A�*;


total_lossqʬ@

error_RYG?

learning_rate_1��5If�I       6%�	,Qd���A�*;


total_loss״]@

error_RdK<?

learning_rate_1��5U�OI       6%�	�TQd���A�*;


total_loss���@

error_R{lb?

learning_rate_1��5�VtDI       6%�	�Qd���A�*;


total_loss��@

error_R�,F?

learning_rate_1��5}fE$I       6%�	,�Qd���A�*;


total_loss\��@

error_Rj�L?

learning_rate_1��5]��I       6%�	)Rd���A�*;


total_loss�ȑ@

error_R�#K?

learning_rate_1��5*�i�I       6%�	#rRd���A�*;


total_loss�L�@

error_R_hU?

learning_rate_1��5�/҉I       6%�	��Rd���A�*;


total_loss=I�@

error_R��9?

learning_rate_1��5��9hI       6%�	��Rd���A�*;


total_loss���@

error_Ra�P?

learning_rate_1��5 v$I       6%�	�?Sd���A�*;


total_loss�r�@

error_R��M?

learning_rate_1��5��&7I       6%�	҃Sd���A�*;


total_lossE;�@

error_R�<^?

learning_rate_1��5��v�I       6%�	l�Sd���A�*;


total_loss��@

error_RM!L?

learning_rate_1��5'W�>I       6%�	M
Td���A�*;


total_loss�C�@

error_R_Z?

learning_rate_1��5(W�I       6%�	�OTd���A�*;


total_loss���@

error_R�wN?

learning_rate_1��5��?�I       6%�	ӔTd���A�*;


total_lossAB�@

error_R.;?

learning_rate_1��5ua�I       6%�	�Td���A�*;


total_losscͥ@

error_R�D[?

learning_rate_1��5���I       6%�	�Ud���A�*;


total_loss�Э@

error_R�W?

learning_rate_1��5;��CI       6%�	�gUd���A�*;


total_lossN��@

error_R&�V?

learning_rate_1��5��kI       6%�	îUd���A�*;


total_loss���@

error_RW�f?

learning_rate_1��5�i9�I       6%�	�Ud���A�*;


total_lossRɅ@

error_R�I?

learning_rate_1��5���I       6%�	9Vd���A�*;


total_loss�?�@

error_R]�A?

learning_rate_1��5"��I       6%�	&�Vd���A�*;


total_loss|��@

error_R��G?

learning_rate_1��56�Q_I       6%�	��Vd���A�*;


total_loss���@

error_RV?

learning_rate_1��5Mr�I       6%�	�7Wd���A�*;


total_loss��A

error_R��W?

learning_rate_1��5m�I       6%�	�}Wd���A�*;


total_loss��@

error_R$�c?

learning_rate_1��5� R�I       6%�	�Wd���A�*;


total_loss�8�@

error_Rj�F?

learning_rate_1��5�Z<�I       6%�	TXd���A�*;


total_lossMy�@

error_R�zH?

learning_rate_1��5y�oI       6%�	B�Xd���A�*;


total_loss_K�@

error_R��G?

learning_rate_1��5MD�rI       6%�	��Xd���A�*;


total_loss�g�@

error_R�gk?

learning_rate_1��5W:�	I       6%�	Yd���A�*;


total_loss!�@

error_R@�'?

learning_rate_1��5�.�&I       6%�	�`Yd���A�*;


total_loss=��@

error_R��T?

learning_rate_1��5XA�I       6%�	�Yd���A�*;


total_loss�R�@

error_Rj�L?

learning_rate_1��5}DμI       6%�	]�Yd���A�*;


total_loss���@

error_R�nE?

learning_rate_1��5
?�I       6%�	�:Zd���A�*;


total_loss�W�@

error_R�xP?

learning_rate_1��5�!�_I       6%�	C|Zd���A�*;


total_loss1�@

error_R,|P?

learning_rate_1��5(p��I       6%�	�Zd���A�*;


total_loss$��@

error_R$J?

learning_rate_1��5o�.I       6%�	�[d���A�*;


total_loss�i�@

error_R� J?

learning_rate_1��5{%t-I       6%�	r[[d���A�*;


total_lossvj�@

error_Rf�C?

learning_rate_1��5h>4�I       6%�	��[d���A�*;


total_loss��@

error_RHYG?

learning_rate_1��5�/I       6%�	g\d���A�*;


total_lossH��@

error_R�S?

learning_rate_1��5�[�,I       6%�	L\d���A�*;


total_loss�~�@

error_R�<H?

learning_rate_1��5;OI       6%�	7�\d���A�*;


total_loss�@

error_R!q??

learning_rate_1��5D�s�I       6%�	��\d���A�*;


total_loss�)�@

error_Ra�F?

learning_rate_1��5?4�OI       6%�	V0]d���A�*;


total_loss!��@

error_RZ*T?

learning_rate_1��5"g��I       6%�	Д]d���A�*;


total_loss{H�@

error_R�NO?

learning_rate_1��5��I       6%�	��]d���A�*;


total_loss|�@

error_R�!M?

learning_rate_1��5q!��I       6%�	�!^d���A�*;


total_loss���@

error_R!�T?

learning_rate_1��5�q\I       6%�	�i^d���A�*;


total_loss��@

error_Rn�C?

learning_rate_1��5��9I       6%�	��^d���A�*;


total_lossR�@

error_R��<?

learning_rate_1��5Í!I       6%�	�$_d���A�*;


total_loss�]�@

error_R�L?

learning_rate_1��5tۃlI       6%�	�i_d���A�*;


total_loss�(�@

error_R)$K?

learning_rate_1��5�R��I       6%�	��_d���A�*;


total_lossm�A

error_R  ??

learning_rate_1��5r�#I       6%�	`d���A�*;


total_loss!��@

error_R#�K?

learning_rate_1��5�C�NI       6%�	�a`d���A�*;


total_loss���@

error_R�q>?

learning_rate_1��5q�sI       6%�	��`d���A�*;


total_lossN@�@

error_R0T?

learning_rate_1��5d��;I       6%�	��`d���A�*;


total_loss	^�@

error_Rh�L?

learning_rate_1��5�M��I       6%�	kEad���A�*;


total_lossW�@

error_R-�@?

learning_rate_1��5�&��I       6%�	e�ad���A�*;


total_loss6�@

error_R_K?

learning_rate_1��5ǹ��I       6%�	��ad���A�*;


total_loss-��@

error_R�7D?

learning_rate_1��5�p��I       6%�	kbd���A�*;


total_lossI
�@

error_RxN?

learning_rate_1��5�6��I       6%�	Zbd���A�*;


total_loss�֤@

error_Rd K?

learning_rate_1��5���I       6%�	#�bd���A�*;


total_loss�e�@

error_R�<K?

learning_rate_1��5�$��I       6%�	��bd���A�*;


total_loss�k�@

error_R\�R?

learning_rate_1��5�Gh�I       6%�	�5cd���A�*;


total_loss@)�@

error_REBR?

learning_rate_1��5��I       6%�	�cd���A�*;


total_loss&��@

error_RHA[?

learning_rate_1��5����I       6%�	~�cd���A�*;


total_lossW`�@

error_RZ�P?

learning_rate_1��5��+I       6%�	Xdd���A�*;


total_loss���@

error_R��X?

learning_rate_1��5n��(I       6%�	�Xdd���A�*;


total_lossɯ�@

error_R��)?

learning_rate_1��5T<��I       6%�	/�dd���A�*;


total_loss��@

error_R3>?

learning_rate_1��5eI"�I       6%�	��dd���A�*;


total_loss[��@

error_R�S?

learning_rate_1��5��@I       6%�	.=ed���A�*;


total_lossR��@

error_RI?

learning_rate_1��5��R�I       6%�	!�ed���A�*;


total_loss�N�@

error_R�T?

learning_rate_1��52��I       6%�	��ed���A�*;


total_loss��@

error_R��3?

learning_rate_1��5�lz�I       6%�	hfd���A�*;


total_loss��w@

error_R��S?

learning_rate_1��5[�EI       6%�	�Tfd���A�*;


total_loss�܊@

error_RX�J?

learning_rate_1��5@���I       6%�	�fd���A�*;


total_loss�@

error_R%-L?

learning_rate_1��5��=I       6%�	��fd���A�*;


total_losst��@

error_R��??

learning_rate_1��5ͯv,I       6%�	�gd���A�*;


total_lossV��@

error_R/U?

learning_rate_1��5�,'I       6%�	�agd���A�*;


total_lossqK�@

error_R6cD?

learning_rate_1��5�I�I       6%�	,�gd���A�*;


total_loss��@

error_R��:?

learning_rate_1��5q��I       6%�	)�gd���A�*;


total_lossWЈ@

error_R)�`?

learning_rate_1��5��;�I       6%�	�/hd���A�*;


total_loss��@

error_RͩL?

learning_rate_1��5�I       6%�	�{hd���A�*;


total_loss��>A

error_R]|I?

learning_rate_1��5��I       6%�	=�hd���A�*;


total_loss���@

error_Rl[?

learning_rate_1��5�Ӌ I       6%�	�&id���A�*;


total_loss��@

error_R�w8?

learning_rate_1��5:��I       6%�	oid���A�*;


total_loss�z~@

error_RhqN?

learning_rate_1��5���nI       6%�	T�id���A�*;


total_loss�)�@

error_Rv�N?

learning_rate_1��5����I       6%�	��id���A�*;


total_loss�O�@

error_R$�C?

learning_rate_1��5���I       6%�	�Hjd���A�*;


total_loss�v�@

error_R��4?

learning_rate_1��5�|7GI       6%�	��jd���A�*;


total_lossl��@

error_R�GM?

learning_rate_1��5ļ��I       6%�		�jd���A�*;


total_loss�`�@

error_R\�U?

learning_rate_1��5�2�JI       6%�	!!kd���A�*;


total_loss=}�@

error_R <?

learning_rate_1��5�(C�I       6%�	�gkd���A�*;


total_lossڜ@

error_R�j>?

learning_rate_1��5!�S�I       6%�	�kd���A�*;


total_loss԰�@

error_Rd1T?

learning_rate_1��5ĴJxI       6%�	5�kd���A�*;


total_loss��@

error_R�2K?

learning_rate_1��5筱�I       6%�	�Ald���A�*;


total_loss_��@

error_R�A?

learning_rate_1��5n]�4I       6%�	�ld���A�*;


total_loss&�A

error_Rڕ_?

learning_rate_1��5F��I       6%�	��ld���A�*;


total_loss���@

error_RDH?

learning_rate_1��5�FNRI       6%�	�"md���A�*;


total_lossF=�@

error_R��P?

learning_rate_1��5 ��I       6%�	1�md���A�*;


total_loss���@

error_R�-H?

learning_rate_1��5�n^�I       6%�	��md���A�*;


total_loss��_@

error_R�G?

learning_rate_1��5q�'�I       6%�	�nd���A�*;


total_loss��@

error_Rj[V?

learning_rate_1��5���I       6%�	knd���A�*;


total_loss�ށ@

error_R�>?

learning_rate_1��5��3I       6%�	��nd���A�*;


total_lossT5�@

error_RI�C?

learning_rate_1��5DY�I       6%�	�nd���A�*;


total_loss�V�@

error_RD)X?

learning_rate_1��52I       6%�	ICod���A�*;


total_loss!��@

error_R&�W?

learning_rate_1��5��ѸI       6%�	6�od���A�*;


total_loss^��@

error_R��F?

learning_rate_1��5ˍ�I       6%�	��od���A�*;


total_loss�ߙ@

error_R�%I?

learning_rate_1��5�t��I       6%�	pd���A�*;


total_loss���@

error_RC�E?

learning_rate_1��5x`- I       6%�	IPpd���A�*;


total_loss��@

error_RsPC?

learning_rate_1��5�0�I       6%�	ѓpd���A�*;


total_loss!;z@

error_R��=?

learning_rate_1��5���/I       6%�	��pd���A�*;


total_loss�u�@

error_R�EI?

learning_rate_1��5"0�I       6%�	�"qd���A�*;


total_loss�q@

error_Rc\?

learning_rate_1��59k[I       6%�	_gqd���A�*;


total_loss��A

error_R�IN?

learning_rate_1��5w�SI       6%�	9�qd���A�*;


total_loss��@

error_R�bK?

learning_rate_1��5'8GI       6%�	5�qd���A�*;


total_loss���@

error_R�8?

learning_rate_1��5a��I       6%�	�4rd���A�*;


total_lossfʈ@

error_Rj!G?

learning_rate_1��5�/�wI       6%�	�yrd���A�*;


total_loss��@

error_R��L?

learning_rate_1��5g=t�I       6%�	�rd���A�*;


total_loss�KA

error_R�V?

learning_rate_1��5&��I       6%�	�
sd���A�*;


total_loss�Q@

error_R�9A?

learning_rate_1��5BVPI       6%�	HTsd���A�*;


total_lossm�@

error_R@�I?

learning_rate_1��5���I       6%�	ȝsd���A�*;


total_loss�P�@

error_R��J?

learning_rate_1��5��GI       6%�	a�sd���A�*;


total_lossJ�@

error_R�Q?

learning_rate_1��5��I       6%�	�,td���A�*;


total_lossjr�@

error_Rm6Y?

learning_rate_1��5�8hI       6%�	�ntd���A�*;


total_loss@Y�@

error_R��W?

learning_rate_1��52�V�I       6%�	Y�td���A�*;


total_loss��@

error_R;YV?

learning_rate_1��59�$[I       6%�	��td���A�*;


total_lossf��@

error_R{,L?

learning_rate_1��5��4�I       6%�	4Aud���A�*;


total_loss.4�@

error_R)�E?

learning_rate_1��5IՑI       6%�	��ud���A�*;


total_loss��@

error_R�L?

learning_rate_1��5� �I       6%�	]�ud���A�*;


total_loss�$�@

error_R{Rb?

learning_rate_1��5���pI       6%�	|vd���A�*;


total_loss A

error_RmLG?

learning_rate_1��5_���I       6%�	-\vd���A�*;


total_loss���@

error_R��G?

learning_rate_1��5-U�[I       6%�	}�vd���A�*;


total_loss#!�@

error_R�H?

learning_rate_1��5��,I       6%�	��vd���A�	*;


total_lossB�@

error_R�E?

learning_rate_1��5�{�qI       6%�	�6wd���A�	*;


total_loss�Yi@

error_RX�K?

learning_rate_1��5�$�I       6%�	5zwd���A�	*;


total_loss���@

error_R�gS?

learning_rate_1��5�&��I       6%�	��wd���A�	*;


total_loss�@

error_R��M?

learning_rate_1��5~�nI       6%�	xd���A�	*;


total_loss�A

error_R��X?

learning_rate_1��5\�(I       6%�	Jxd���A�	*;


total_loss�A

error_R�Pj?

learning_rate_1��5kʻ�I       6%�	2�xd���A�	*;


total_loss���@

error_R��T?

learning_rate_1��5�1 �I       6%�	�xd���A�	*;


total_loss�3�@

error_R>L?

learning_rate_1��55���I       6%�	Ryd���A�	*;


total_loss���@

error_R��,?

learning_rate_1��5�ADI       6%�	0�yd���A�	*;


total_loss���@

error_R��A?

learning_rate_1��5�{�yI       6%�	8�yd���A�	*;


total_loss�:A

error_R='Q?

learning_rate_1��5���I       6%�	v<zd���A�	*;


total_loss�x�@

error_Rzc?

learning_rate_1��5���I       6%�	<�zd���A�	*;


total_loss��@

error_R{WY?

learning_rate_1��5���aI       6%�		�zd���A�	*;


total_lossJw@

error_RC�;?

learning_rate_1��5u�II       6%�	{d���A�	*;


total_loss\o�@

error_R�H?

learning_rate_1��5��'I       6%�	Vc{d���A�	*;


total_lossV�@

error_RW�4?

learning_rate_1��5N�I       6%�	Z�{d���A�	*;


total_loss�&�@

error_R�H?

learning_rate_1��5 !�>I       6%�	��{d���A�	*;


total_loss�a�@

error_R}[E?

learning_rate_1��5I��AI       6%�	�0|d���A�	*;


total_loss2L�@

error_R��@?

learning_rate_1��5�kQ�I       6%�	�v|d���A�	*;


total_losshc�@

error_RFD?

learning_rate_1��5���-I       6%�	Ի|d���A�	*;


total_loss)��@

error_R&'??

learning_rate_1��5E^��I       6%�	� }d���A�	*;


total_loss�Q�@

error_R׍;?

learning_rate_1��5�)PI       6%�	�W}d���A�	*;


total_lossW��@

error_RR�\?

learning_rate_1��5��ºI       6%�	�}d���A�	*;


total_loss؟�@

error_R��O?

learning_rate_1��5q�]I       6%�	�}d���A�	*;


total_loss@�@

error_R.�G?

learning_rate_1��5���I       6%�	@~d���A�	*;


total_loss}��@

error_Rq�K?

learning_rate_1��5�dɶI       6%�	֊~d���A�	*;


total_loss�u�@

error_R�^?

learning_rate_1��5���I       6%�	��~d���A�	*;


total_loss�F�@

error_R��7?

learning_rate_1��5�dL�I       6%�	8d���A�	*;


total_loss�&�@

error_R)0S?

learning_rate_1��5�8T�I       6%�	#Zd���A�	*;


total_loss�q�@

error_R W?

learning_rate_1��5�7�I       6%�	f�d���A�	*;


total_loss/@

error_RR�U?

learning_rate_1��5��RI       6%�	~�d���A�	*;


total_loss�Q�@

error_R�AX?

learning_rate_1��5С�I       6%�	�)�d���A�	*;


total_lossa�@

error_R�{I?

learning_rate_1��5��SI       6%�	fq�d���A�	*;


total_loss� �@

error_RJ�G?

learning_rate_1��5ߏhSI       6%�	���d���A�	*;


total_loss�ä@

error_RI?

learning_rate_1��5wl�I       6%�	���d���A�	*;


total_lossف�@

error_R��R?

learning_rate_1��5%U��I       6%�	lA�d���A�	*;


total_lossO$�@

error_R��C?

learning_rate_1��5o��:I       6%�	��d���A�	*;


total_loss	�@

error_R%�P?

learning_rate_1��5�;�I       6%�	qЁd���A�	*;


total_loss^Ȓ@

error_R��M?

learning_rate_1��5 _�I       6%�	��d���A�	*;


total_losshX�@

error_R7�]?

learning_rate_1��5^��:I       6%�	�h�d���A�	*;


total_loss�A

error_R��\?

learning_rate_1��5^}�I       6%�	��d���A�	*;


total_lossW'A

error_R�0O?

learning_rate_1��5�Y"�I       6%�	j�d���A�	*;


total_lossr��@

error_R��S?

learning_rate_1��5mm�cI       6%�	~8�d���A�	*;


total_loss@~A

error_R��7?

learning_rate_1��5�y�I       6%�	Ӆ�d���A�	*;


total_loss�/�@

error_R̲a?

learning_rate_1��5���I       6%�	Q̓d���A�	*;


total_loss�$�@

error_R�vI?

learning_rate_1��5Bt I       6%�	s�d���A�	*;


total_loss��@

error_R�'P?

learning_rate_1��5���,I       6%�	W�d���A�	*;


total_lossce�@

error_R�=F?

learning_rate_1��5���I       6%�	��d���A�	*;


total_loss�B�@

error_R��S?

learning_rate_1��5�mo/I       6%�	��d���A�	*;


total_loss�?�@

error_R�"W?

learning_rate_1��5*okBI       6%�	A/�d���A�	*;


total_lossv��@

error_R�<i?

learning_rate_1��5+�۟I       6%�	Bw�d���A�	*;


total_loss@��@

error_R�>?

learning_rate_1��5�@|,I       6%�	���d���A�	*;


total_loss�@

error_RiX?

learning_rate_1��5?O\]I       6%�	�d���A�	*;


total_loss8ӛ@

error_RN�N?

learning_rate_1��5�+�I       6%�	}L�d���A�	*;


total_lossZ��@

error_R X?

learning_rate_1��5'e�I       6%�	1��d���A�	*;


total_loss,�@

error_R��N?

learning_rate_1��5)��)I       6%�	\ކd���A�	*;


total_lossx��@

error_Rɉ??

learning_rate_1��5�T�OI       6%�	?!�d���A�	*;


total_loss%��@

error_RRFX?

learning_rate_1��5���I       6%�	�g�d���A�	*;


total_loss7u�@

error_R��2?

learning_rate_1��5��X�I       6%�	���d���A�	*;


total_loss���@

error_R�N?

learning_rate_1��5�>��I       6%�	��d���A�	*;


total_loss6�@

error_Rl~:?

learning_rate_1��5E1YlI       6%�	b9�d���A�	*;


total_loss ��@

error_R]�R?

learning_rate_1��5�њbI       6%�	b��d���A�	*;


total_loss�Y�@

error_R�g?

learning_rate_1��5�T��I       6%�	J�d���A�	*;


total_loss�h�@

error_R�P?

learning_rate_1��5�B�I       6%�	`?�d���A�	*;


total_lossjo�@

error_R��I?

learning_rate_1��5)ԩI       6%�	N��d���A�	*;


total_loss���@

error_R��7?

learning_rate_1��5\i-I       6%�	���d���A�	*;


total_loss���@

error_R��E?

learning_rate_1��5b�(I       6%�	�I�d���A�	*;


total_loss"8�@

error_R��W?

learning_rate_1��5\2�I       6%�	ؒ�d���A�	*;


total_loss�ح@

error_R��O?

learning_rate_1��5\9v�I       6%�	�يd���A�	*;


total_loss���@

error_R(�[?

learning_rate_1��5���I       6%�	-�d���A�	*;


total_loss��@

error_RwCZ?

learning_rate_1��5���GI       6%�	�b�d���A�	*;


total_lossl�A

error_R��T?

learning_rate_1��5%G�I       6%�	)��d���A�	*;


total_lossv�@

error_R��S?

learning_rate_1��5㣽I       6%�	��d���A�	*;


total_loss�z�@

error_R�(O?

learning_rate_1��5&v �I       6%�	Q>�d���A�	*;


total_lossi��@

error_R��5?

learning_rate_1��5�x�I       6%�	���d���A�	*;


total_loss-��@

error_R�XN?

learning_rate_1��5~l�I       6%�	�Όd���A�	*;


total_loss�@

error_R�A?

learning_rate_1��5۽ÁI       6%�	��d���A�	*;


total_loss���@

error_Rf~U?

learning_rate_1��5!a�I       6%�	�l�d���A�	*;


total_loss)��@

error_R�I?

learning_rate_1��5m���I       6%�	=��d���A�	*;


total_loss�lA

error_R�E?

learning_rate_1��5-m�I       6%�	"�d���A�	*;


total_loss�v�@

error_R�sM?

learning_rate_1��5a΀�I       6%�	�G�d���A�	*;


total_loss���@

error_R�SU?

learning_rate_1��5'mV�I       6%�	���d���A�	*;


total_lossZ�KA

error_R
�G?

learning_rate_1��5p�xKI       6%�	s͎d���A�	*;


total_loss�w�@

error_R*�`?

learning_rate_1��5UP��I       6%�	�d���A�	*;


total_loss���@

error_R��F?

learning_rate_1��5���I       6%�	�T�d���A�	*;


total_lossٴ@

error_R΍f?

learning_rate_1��5�k�I       6%�	*��d���A�	*;


total_loss2F�@

error_R�]C?

learning_rate_1��5� �,I       6%�	��d���A�	*;


total_lossFق@

error_R�AJ?

learning_rate_1��5�"��I       6%�	,�d���A�	*;


total_lossiu�@

error_R�W?

learning_rate_1��59���I       6%�	6s�d���A�	*;


total_loss�|�@

error_R�O?

learning_rate_1��5c9e�I       6%�	)��d���A�	*;


total_loss.��@

error_Rs�V?

learning_rate_1��5��I       6%�	��d���A�	*;


total_loss�F�@

error_R�[?

learning_rate_1��5&=DI       6%�	D�d���A�	*;


total_loss�	�@

error_R�}K?

learning_rate_1��5KGh�I       6%�	^��d���A�	*;


total_loss�4�@

error_R�1?

learning_rate_1��5c�[I       6%�	h֑d���A�	*;


total_loss�T�@

error_R�cK?

learning_rate_1��5��I       6%�	��d���A�	*;


total_loss�{�@

error_RәA?

learning_rate_1��5�m'ZI       6%�	�f�d���A�	*;


total_lossZ��@

error_Rt7?

learning_rate_1��5#�nI       6%�	���d���A�	*;


total_loss���@

error_R�U?

learning_rate_1��5T�TI       6%�	?�d���A�	*;


total_loss��@

error_R�X?

learning_rate_1��5Eo��I       6%�	�8�d���A�	*;


total_loss9�@

error_R��J?

learning_rate_1��5����I       6%�	�}�d���A�	*;


total_loss&��@

error_R��4?

learning_rate_1��5�[YI       6%�	�Ód���A�	*;


total_lossmU�@

error_RT?

learning_rate_1��5hx��I       6%�	�d���A�	*;


total_loss�$�@

error_R1#T?

learning_rate_1��5�=
~I       6%�	*R�d���A�	*;


total_loss���@

error_Rz�C?

learning_rate_1��5��gI       6%�	ԕ�d���A�	*;


total_lossb�@

error_R�X?

learning_rate_1��5n�aI       6%�	dؔd���A�	*;


total_loss)��@

error_RL�E?

learning_rate_1��5D���I       6%�	J�d���A�	*;


total_lossd0�@

error_R4�L?

learning_rate_1��5Ԝ�I       6%�	c�d���A�	*;


total_loss��@

error_Rn1Y?

learning_rate_1��5��.I       6%�	 ��d���A�	*;


total_loss]�@

error_R�<T?

learning_rate_1��5��oI       6%�	��d���A�	*;


total_lossj:'A

error_RV�a?

learning_rate_1��5�:�I       6%�	j+�d���A�	*;


total_losse �@

error_RLH?

learning_rate_1��5Nׂ�I       6%�	�o�d���A�	*;


total_lossT|�@

error_R�|Q?

learning_rate_1��5�n(�I       6%�	_��d���A�	*;


total_losst�@

error_RO�\?

learning_rate_1��5���}I       6%�	8��d���A�	*;


total_loss��@

error_R��J?

learning_rate_1��5��I       6%�	(8�d���A�	*;


total_lossqq�@

error_RW�N?

learning_rate_1��5Y���I       6%�	^|�d���A�	*;


total_loss%ɗ@

error_R8@?

learning_rate_1��5�ZYI       6%�	4��d���A�	*;


total_loss���@

error_R<�^?

learning_rate_1��5��b�I       6%�	w�d���A�	*;


total_loss�1�@

error_RmDI?

learning_rate_1��5�|6I       6%�	dK�d���A�	*;


total_loss�M�@

error_R�S?

learning_rate_1��5��]I       6%�	g��d���A�	*;


total_loss���@

error_R)AN?

learning_rate_1��5�Á�I       6%�	֘d���A�	*;


total_lossπ�@

error_R�^>?

learning_rate_1��5�f��I       6%�	�d���A�	*;


total_loss�נ@

error_R��T?

learning_rate_1��5dQ�I       6%�	�d�d���A�	*;


total_loss�A

error_R��>?

learning_rate_1��5��I       6%�	��d���A�	*;


total_loss	�@

error_R,�U?

learning_rate_1��5p?�I       6%�	y��d���A�	*;


total_loss��@

error_R�hT?

learning_rate_1��5T�u;I       6%�	�=�d���A�	*;


total_lossL��@

error_R�RQ?

learning_rate_1��5vVhI       6%�	���d���A�	*;


total_lossF�A

error_RaX?

learning_rate_1��500BI       6%�	Q˚d���A�	*;


total_loss (�@

error_R�9?

learning_rate_1��59�;�I       6%�	9�d���A�
*;


total_loss��@

error_RoZ?

learning_rate_1��5Kz�ZI       6%�	�U�d���A�
*;


total_loss���@

error_R��H?

learning_rate_1��5��@I       6%�	��d���A�
*;


total_loss�
�@

error_R��G?

learning_rate_1��5�A�I       6%�	 ޛd���A�
*;


total_loss�;�@

error_RW;?

learning_rate_1��5}�I       6%�	��d���A�
*;


total_loss2��@

error_REi>?

learning_rate_1��5¸��I       6%�	mi�d���A�
*;


total_loss;��@

error_R�8N?

learning_rate_1��5b�$KI       6%�	K��d���A�
*;


total_loss�%�@

error_R�J?

learning_rate_1��58��I       6%�	���d���A�
*;


total_loss6��@

error_R)V?

learning_rate_1��5/�wI       6%�	0K�d���A�
*;


total_loss	��@

error_R�`P?

learning_rate_1��5Gy:3I       6%�	��d���A�
*;


total_losst�@

error_R�Z?

learning_rate_1��5�q��I       6%�	��d���A�
*;


total_loss��@

error_R��E?

learning_rate_1��5��"I       6%�	-M�d���A�
*;


total_lossҀ�@

error_R��X?

learning_rate_1��5f4j]I       6%�	���d���A�
*;


total_loss���@

error_R�MO?

learning_rate_1��5F*21I       6%�	$�d���A�
*;


total_loss�ֺ@

error_R !N?

learning_rate_1��5�]� I       6%�	h1�d���A�
*;


total_losstˤ@

error_R��^?

learning_rate_1��5��uI       6%�	&{�d���A�
*;


total_loss��@

error_R/7?

learning_rate_1��5Y"��I       6%�	徟d���A�
*;


total_loss|�@

error_R�'N?

learning_rate_1��5�!��I       6%�	:�d���A�
*;


total_lossɬ�@

error_R��D?

learning_rate_1��52m��I       6%�	F�d���A�
*;


total_lossWQ	A

error_R��Z?

learning_rate_1��5l5�I       6%�	���d���A�
*;


total_loss�Z�@

error_R�Wh?

learning_rate_1��5;ԕ�I       6%�	�ʠd���A�
*;


total_loss�Đ@

error_R:�L?

learning_rate_1��5Cq�I       6%�	��d���A�
*;


total_loss���@

error_R�C?

learning_rate_1��5>���I       6%�	UP�d���A�
*;


total_loss�e�@

error_R��H?

learning_rate_1��5na��I       6%�	Z��d���A�
*;


total_loss��@

error_R=�F?

learning_rate_1��5`�9I       6%�	�d���A�
*;


total_loss1u�@

error_RsZJ?

learning_rate_1��5L*�\I       6%�	H+�d���A�
*;


total_loss)��@

error_R�U?

learning_rate_1��5A�3I       6%�	�v�d���A�
*;


total_lossY�@

error_Re_b?

learning_rate_1��5�ۣEI       6%�	�բd���A�
*;


total_loss^Ǘ@

error_R �D?

learning_rate_1��5�i�CI       6%�	x%�d���A�
*;


total_loss���@

error_R�V?

learning_rate_1��5���I       6%�	�t�d���A�
*;


total_loss�Ы@

error_R�V?

learning_rate_1��536�8I       6%�	ụd���A�
*;


total_loss�<�@

error_R�Q?

learning_rate_1��5�+UI       6%�	;�d���A�
*;


total_loss_њ@

error_RA�C?

learning_rate_1��5�9U�I       6%�	�^�d���A�
*;


total_loss( �@

error_R��T?

learning_rate_1��5�.�I       6%�	몤d���A�
*;


total_loss��@

error_R��E?

learning_rate_1��5��Y5I       6%�	��d���A�
*;


total_loss�/w@

error_R�C?

learning_rate_1��5�g��I       6%�	48�d���A�
*;


total_loss!^�@

error_R��L?

learning_rate_1��5xG��I       6%�	�|�d���A�
*;


total_loss��@

error_RaME?

learning_rate_1��5�@>�I       6%�	���d���A�
*;


total_lossC��@

error_RjiG?

learning_rate_1��5[w)�I       6%�	i�d���A�
*;


total_lossl��@

error_R��`?

learning_rate_1��5� k)I       6%�	�G�d���A�
*;


total_loss`��@

error_R
?@?

learning_rate_1��5f�I       6%�	��d���A�
*;


total_loss@

error_RC 5?

learning_rate_1��5�W$�I       6%�	�Ӧd���A�
*;


total_loss�p�@

error_R@�R?

learning_rate_1��5�8k�I       6%�	X�d���A�
*;


total_loss	�@

error_RJ9H?

learning_rate_1��5�M�I       6%�	Q`�d���A�
*;


total_loss���@

error_R�R?

learning_rate_1��5����I       6%�	p��d���A�
*;


total_lossM�@

error_R}N?

learning_rate_1��5�^uI       6%�	C�d���A�
*;


total_loss���@

error_R�K?

learning_rate_1��5h�~�I       6%�	�+�d���A�
*;


total_loss��A

error_R��O?

learning_rate_1��5aaI       6%�	5p�d���A�
*;


total_lossOF�@

error_R�S?

learning_rate_1��5�4(�I       6%�	B��d���A�
*;


total_loss�
�@

error_RN!K?

learning_rate_1��5�T^�I       6%�	���d���A�
*;


total_loss �z@

error_R	\?

learning_rate_1��5��k�I       6%�	�;�d���A�
*;


total_lossI�@

error_R�)H?

learning_rate_1��5kr�I       6%�	��d���A�
*;


total_loss��@

error_R�v;?

learning_rate_1��5U~�oI       6%�	�ɩd���A�
*;


total_loss�*
A

error_R3N?

learning_rate_1��5Tl:oI       6%�	��d���A�
*;


total_loss��@

error_R�S?

learning_rate_1��5ie�I       6%�	NY�d���A�
*;


total_loss���@

error_R=�\?

learning_rate_1��5'H�I       6%�	
��d���A�
*;


total_loss_��@

error_R)E?

learning_rate_1��5X��I       6%�	�d���A�
*;


total_losstǻ@

error_RTC@?

learning_rate_1��5��I       6%�	�'�d���A�
*;


total_loss�F�@

error_R��X?

learning_rate_1��5���}I       6%�	dk�d���A�
*;


total_loss]��@

error_R��^?

learning_rate_1��5���I       6%�	b��d���A�
*;


total_loss�`�@

error_R�QX?

learning_rate_1��5�P��I       6%�	��d���A�
*;


total_loss��@

error_R�v:?

learning_rate_1��5���I       6%�	 =�d���A�
*;


total_loss��@

error_RnSI?

learning_rate_1��5��i�I       6%�	P��d���A�
*;


total_loss���@

error_RZI?

learning_rate_1��5
��=I       6%�	jƬd���A�
*;


total_lossy�A

error_R!-F?

learning_rate_1��5�f*I       6%�	#
�d���A�
*;


total_lossʂ�@

error_R�F?

learning_rate_1��5U�I       6%�	oW�d���A�
*;


total_loss�N�@

error_R|M^?

learning_rate_1��5�^LI       6%�	m��d���A�
*;


total_loss��@

error_R��J?

learning_rate_1��5�I
xI       6%�	f��d���A�
*;


total_loss*n�@

error_R[�O?

learning_rate_1��5:�McI       6%�	�>�d���A�
*;


total_loss�W�@

error_R��S?

learning_rate_1��5f��I       6%�	-��d���A�
*;


total_loss�A�@

error_R�8?

learning_rate_1��5K{C�I       6%�	8Ȯd���A�
*;


total_loss1� A

error_R�6B?

learning_rate_1��5�&QlI       6%�	?�d���A�
*;


total_loss�m�@

error_R\�J?

learning_rate_1��5���I       6%�	]�d���A�
*;


total_loss#�@

error_R��b?

learning_rate_1��5o��I       6%�	ġ�d���A�
*;


total_loss��@

error_R�(H?

learning_rate_1��5cmn�I       6%�	.�d���A�
*;


total_loss6ү@

error_R��M?

learning_rate_1��5lq�3I       6%�	�3�d���A�
*;


total_loss\o�@

error_R\�W?

learning_rate_1��5�ƺI       6%�	�y�d���A�
*;


total_loss��@

error_RL�L?

learning_rate_1��5�S�I       6%�	k��d���A�
*;


total_loss!>�@

error_RidL?

learning_rate_1��5$o��I       6%�		�d���A�
*;


total_lossm2�@

error_R�6?

learning_rate_1��5t|�I       6%�	�V�d���A�
*;


total_loss���@

error_R��K?

learning_rate_1��5j�G8I       6%�	(��d���A�
*;


total_lossM�A

error_R`�K?

learning_rate_1��5;޽NI       6%�	I�d���A�
*;


total_loss�3�@

error_R��M?

learning_rate_1��5٠�8I       6%�	D(�d���A�
*;


total_loss��{@

error_R��C?

learning_rate_1��5��4KI       6%�	&l�d���A�
*;


total_loss\��@

error_R;�X?

learning_rate_1��5��-I       6%�	���d���A�
*;


total_lossd��@

error_R�Q?

learning_rate_1��5F��-I       6%�	��d���A�
*;


total_loss�b�@

error_R��N?

learning_rate_1��5S���I       6%�	�3�d���A�
*;


total_loss��@

error_R�bH?

learning_rate_1��5N�N�I       6%�	fx�d���A�
*;


total_loss�"A

error_R�Q?

learning_rate_1��5��)!I       6%�	t��d���A�
*;


total_loss!6�@

error_Rڏ_?

learning_rate_1��5ԓ�9I       6%�	���d���A�
*;


total_loss�@

error_R׏K?

learning_rate_1��54_yI       6%�	UC�d���A�
*;


total_loss.��@

error_R�:O?

learning_rate_1��5U)�I       6%�	@��d���A�
*;


total_loss]�@

error_RMJ?

learning_rate_1��5��U�I       6%�	wдd���A�
*;


total_loss3�@

error_R�&h?

learning_rate_1��5(4I       6%�	��d���A�
*;


total_loss�Z�@

error_RaeI?

learning_rate_1��57�8�I       6%�	r�d���A�
*;


total_loss�-A

error_RfmQ?

learning_rate_1��5Ap_�I       6%�	��d���A�
*;


total_loss�k�@

error_R��H?

learning_rate_1��5��:JI       6%�	���d���A�
*;


total_loss�b�@

error_R[e?

learning_rate_1��5�]{I       6%�	�?�d���A�
*;


total_lossΓ	A

error_R8�F?

learning_rate_1��5��/I       6%�	倶d���A�
*;


total_loss���@

error_Ra�=?

learning_rate_1��5�:!\I       6%�	�Ķd���A�
*;


total_lossҒ@

error_R�C?

learning_rate_1��5��\�I       6%�	v�d���A�
*;


total_loss +A

error_R�`?

learning_rate_1��5��3I       6%�	nQ�d���A�
*;


total_loss��@

error_R�F?

learning_rate_1��5�lI       6%�	���d���A�
*;


total_loss_s�@

error_RAb?

learning_rate_1��5O��I       6%�	fڷd���A�
*;


total_losss�y@

error_R!�L?

learning_rate_1��5@8�)I       6%�	A�d���A�
*;


total_loss<��@

error_RIV?

learning_rate_1��5C[��I       6%�	�d�d���A�
*;


total_loss)S�@

error_Rl�H?

learning_rate_1��5����I       6%�	Ʈ�d���A�
*;


total_loss@L�@

error_R E?

learning_rate_1��5y�%I       6%�	U��d���A�
*;


total_lossQ��@

error_RX�A?

learning_rate_1��5݈(eI       6%�	�A�d���A�
*;


total_loss�­@

error_RE%C?

learning_rate_1��5���.I       6%�	Q��d���A�
*;


total_loss�z�@

error_R&�C?

learning_rate_1��5��4FI       6%�	չd���A�
*;


total_lossx��@

error_R;E?

learning_rate_1��5�jT�I       6%�	��d���A�
*;


total_loss?��@

error_RʫN?

learning_rate_1��5x��*I       6%�	.\�d���A�
*;


total_loss(ӵ@

error_R��W?

learning_rate_1��5���I       6%�	��d���A�
*;


total_lossR��@

error_R��O?

learning_rate_1��5�)@I       6%�	�d���A�
*;


total_loss�u�@

error_R�1]?

learning_rate_1��5X? �I       6%�	�)�d���A�
*;


total_loss�Ӟ@

error_Rs�D?

learning_rate_1��5
;̃I       6%�	�k�d���A�
*;


total_loss�$�@

error_R�*P?

learning_rate_1��5%�¸I       6%�	m��d���A�
*;


total_loss�֙@

error_R�y>?

learning_rate_1��5�:��I       6%�	��d���A�
*;


total_loss��@

error_Rs�S?

learning_rate_1��5��CyI       6%�	k6�d���A�
*;


total_loss�D�@

error_RlJ?

learning_rate_1��5�mI       6%�	�x�d���A�
*;


total_lossm�@

error_R��P?

learning_rate_1��5f�!�I       6%�	���d���A�
*;


total_loss�ݥ@

error_R�d<?

learning_rate_1��5����I       6%�	��d���A�
*;


total_loss�M�@

error_R��a?

learning_rate_1��5Z� �I       6%�	N�d���A�
*;


total_loss�r�@

error_R��R?

learning_rate_1��5k�yI       6%�	=��d���A�
*;


total_loss�.�@

error_Ro�L?

learning_rate_1��5�F�I       6%�	��d���A�
*;


total_lossf�@

error_R
KV?

learning_rate_1��5��OI       6%�	�2�d���A�
*;


total_lossJJ�@

error_R�]b?

learning_rate_1��5��-I       6%�	��d���A�
*;


total_loss"�@

error_R�8L?

learning_rate_1��5;3dPI       6%�	bԾd���A�*;


total_loss=7�@

error_R�N?

learning_rate_1��5��^�I       6%�	 �d���A�*;


total_loss�@

error_R@�E?

learning_rate_1��5�V�I       6%�	9c�d���A�*;


total_loss�@

error_R �W?

learning_rate_1��5A�'�I       6%�	s��d���A�*;


total_loss@&�@

error_R)�W?

learning_rate_1��5�>II       6%�	^�d���A�*;


total_lossX�@

error_Rq�S?

learning_rate_1��5샜�I       6%�	;�d���A�*;


total_loss-�@

error_RC�D?

learning_rate_1��5��q�I       6%�	n��d���A�*;


total_loss�V�@

error_R?O?

learning_rate_1��5K�E�I       6%�	x��d���A�*;


total_lossb�@

error_RW�O?

learning_rate_1��5���I       6%�	�3�d���A�*;


total_lossxpn@

error_R��T?

learning_rate_1��5J�I       6%�	���d���A�*;


total_loss�z�@

error_R��E?

learning_rate_1��5���I       6%�	a��d���A�*;


total_loss�@

error_R��\?

learning_rate_1��5PR��I       6%�	�(�d���A�*;


total_loss��@

error_R�B?

learning_rate_1��5����I       6%�	l�d���A�*;


total_loss2Y�@

error_R#�/?

learning_rate_1��5�	�"I       6%�	R��d���A�*;


total_loss���@

error_R�U?

learning_rate_1��5�L�I       6%�	���d���A�*;


total_loss���@

error_R1�A?

learning_rate_1��5�~�I       6%�	M;�d���A�*;


total_loss E�@

error_R&SM?

learning_rate_1��5ABߟI       6%�	��d���A�*;


total_lossxΞ@

error_R�]?

learning_rate_1��5���:I       6%�	C��d���A�*;


total_loss͑�@

error_R6�C?

learning_rate_1��5�/�I       6%�	��d���A�*;


total_loss���@

error_R}G?

learning_rate_1��5/��2I       6%�	\�d���A�*;


total_loss���@

error_Rs>Y?

learning_rate_1��5���:I       6%�	k��d���A�*;


total_loss�p�@

error_R!�W?

learning_rate_1��5��|I       6%�	���d���A�*;


total_loss�=�@

error_R�C?

learning_rate_1��5ڳ�I       6%�	#*�d���A�*;


total_loss&�@

error_Ra
f?

learning_rate_1��5�ѷ&I       6%�	^m�d���A�*;


total_lossxf�@

error_RO?

learning_rate_1��5Z��CI       6%�	��d���A�*;


total_lossv�@

error_R�\R?

learning_rate_1��5V���I       6%�	���d���A�*;


total_loss���@

error_R��@?

learning_rate_1��5��A�I       6%�	;�d���A�*;


total_lossV��@

error_RHeH?

learning_rate_1��5�L�I       6%�	�{�d���A�*;


total_loss���@

error_Ro�R?

learning_rate_1��5�"�!I       6%�	���d���A�*;


total_loss��@

error_R�N?

learning_rate_1��5�,ZI       6%�	�d���A�*;


total_loss���@

error_R�2E?

learning_rate_1��5���~I       6%�	sG�d���A�*;


total_loss�/�@

error_R�}7?

learning_rate_1��5�ΛnI       6%�	���d���A�*;


total_loss�r�@

error_RRI@?

learning_rate_1��5Ն�I       6%�	���d���A�*;


total_lossR��@

error_R��B?

learning_rate_1��5�9q=I       6%�	9�d���A�*;


total_lossK��@

error_RC�G?

learning_rate_1��5����I       6%�	�]�d���A�*;


total_lossL�@

error_R�?^?

learning_rate_1��5ފV�I       6%�	���d���A�*;


total_losse��@

error_R�S?

learning_rate_1��5`L�II       6%�	N��d���A�*;


total_loss�~�@

error_R$Q?

learning_rate_1��56�dBI       6%�	)'�d���A�*;


total_loss%:�@

error_R�AZ?

learning_rate_1��5�3/FI       6%�	�h�d���A�*;


total_loss-Z�@

error_R��2?

learning_rate_1��58��I       6%�	6��d���A�*;


total_lossLw�@

error_R�K_?

learning_rate_1��5u�x�I       6%�	���d���A�*;


total_loss���@

error_R �J?

learning_rate_1��5��]�I       6%�	�2�d���A�*;


total_loss��A

error_ROM]?

learning_rate_1��5Jt�PI       6%�	�u�d���A�*;


total_loss�$�@

error_R�{U?

learning_rate_1��5S�s�I       6%�	L��d���A�*;


total_loss]o�@

error_R<AZ?

learning_rate_1��5�nI       6%�	1��d���A�*;


total_loss)�|@

error_ReF?

learning_rate_1��5���I       6%�	�B�d���A�*;


total_loss���@

error_R�\M?

learning_rate_1��5�xēI       6%�	��d���A�*;


total_loss��@

error_R_S?

learning_rate_1��5W�3�I       6%�	d��d���A�*;


total_loss8[�@

error_R/�X?

learning_rate_1��5�'��I       6%�	�d���A�*;


total_loss��@

error_R*?

learning_rate_1��5w_4�I       6%�	�Q�d���A�*;


total_loss��@

error_ROEO?

learning_rate_1��5��s�I       6%�	��d���A�*;


total_loss\#�@

error_R��=?

learning_rate_1��552�I       6%�	S��d���A�*;


total_lossf�	A

error_R�,S?

learning_rate_1��5t^N�I       6%�	<'�d���A�*;


total_losso�@

error_RNR?

learning_rate_1��5�K�1I       6%�	Q��d���A�*;


total_loss�@

error_ROia?

learning_rate_1��5J���I       6%�	��d���A�*;


total_loss�ߥ@

error_R)�V?

learning_rate_1��5�I       6%�	x �d���A�*;


total_loss���@

error_R�uN?

learning_rate_1��5#Q2�I       6%�	�h�d���A�*;


total_loss$�@

error_RڢZ?

learning_rate_1��5MO�I       6%�	l��d���A�*;


total_loss���@

error_R�
V?

learning_rate_1��5Rs0DI       6%�	���d���A�*;


total_loss!�@

error_R�\?

learning_rate_1��50 }pI       6%�	�C�d���A�*;


total_lossAs�@

error_R,9V?

learning_rate_1��5��F�I       6%�	���d���A�*;


total_loss��@

error_R��S?

learning_rate_1��5����I       6%�	���d���A�*;


total_lossʻ�@

error_R��X?

learning_rate_1��5��|I       6%�	��d���A�*;


total_loss���@

error_R͖L?

learning_rate_1��5����I       6%�	�_�d���A�*;


total_loss��@

error_R
�G?

learning_rate_1��5�a�I       6%�	���d���A�*;


total_lossa]�@

error_R,�W?

learning_rate_1��5���I       6%�	?��d���A�*;


total_lossW��@

error_RiR?

learning_rate_1��5�g|�I       6%�	`=�d���A�*;


total_loss
\�@

error_R:P?

learning_rate_1��5I���I       6%�	��d���A�*;


total_lossS��@

error_R}�C?

learning_rate_1��5{��AI       6%�	���d���A�*;


total_losssC�@

error_R6'O?

learning_rate_1��5�D�aI       6%�	��d���A�*;


total_loss��@

error_R��O?

learning_rate_1��5�NrI       6%�	?^�d���A�*;


total_loss�� A

error_R�{N?

learning_rate_1��5��6^I       6%�	`��d���A�*;


total_lossɔ@

error_RvK?

learning_rate_1��5y�)�I       6%�	���d���A�*;


total_lossVxA

error_R��U?

learning_rate_1��5�J��I       6%�	�/�d���A�*;


total_loss:��@

error_R��H?

learning_rate_1��5��qI       6%�	cs�d���A�*;


total_loss�H�@

error_R!�:?

learning_rate_1��5"�
zI       6%�	��d���A�*;


total_lossJ�@

error_R��I?

learning_rate_1��5��G�I       6%�	3��d���A�*;


total_lossj�A

error_RaS?

learning_rate_1��5(���I       6%�	�B�d���A�*;


total_lossA�`@

error_R��F?

learning_rate_1��5U�c�I       6%�	��d���A�*;


total_loss�O�@

error_R��O?

learning_rate_1��5��9�I       6%�	%��d���A�*;


total_lossA��@

error_R��Z?

learning_rate_1��5jOkBI       6%�	�d���A�*;


total_loss摗@

error_Rv�F?

learning_rate_1��5��>�I       6%�	�S�d���A�*;


total_loss@�@

error_R��>?

learning_rate_1��5=�C"I       6%�	F��d���A�*;


total_loss���@

error_R��7?

learning_rate_1��5+h�1I       6%�	���d���A�*;


total_loss��@

error_R�ZB?

learning_rate_1��5���I       6%�	�%�d���A�*;


total_loss��A

error_R{�I?

learning_rate_1��5!�ZI       6%�	�o�d���A�*;


total_loss���@

error_R@VQ?

learning_rate_1��5�Z��I       6%�	���d���A�*;


total_loss��@

error_R��??

learning_rate_1��5��QI       6%�	{��d���A�*;


total_loss��@

error_R��V?

learning_rate_1��5��I       6%�	�C�d���A�*;


total_loss5�@

error_R�P?

learning_rate_1��5ڽW�I       6%�	<��d���A�*;


total_loss 1�@

error_R��N?

learning_rate_1��58�7I       6%�	��d���A�*;


total_loss,�@

error_R=�@?

learning_rate_1��5M�7I       6%�	��d���A�*;


total_lossn�@

error_Ru[?

learning_rate_1��5�/�$I       6%�	�W�d���A�*;


total_loss��o@

error_Rn�K?

learning_rate_1��5F�6�I       6%�	0��d���A�*;


total_lossx �@

error_R�`?

learning_rate_1��5*1I       6%�	,��d���A�*;


total_loss���@

error_Rs�T?

learning_rate_1��5��A�I       6%�	(�d���A�*;


total_lossbB�@

error_R��F?

learning_rate_1��5��2�I       6%�	k�d���A�*;


total_lossD;�@

error_R�Z?

learning_rate_1��5ژ�I       6%�	���d���A�*;


total_loss�k�@

error_Rig?

learning_rate_1��5��BI       6%�	��d���A�*;


total_lossM�A

error_RŀB?

learning_rate_1��5����I       6%�	w9�d���A�*;


total_loss�@

error_R��M?

learning_rate_1��5vϝ$I       6%�	�}�d���A�*;


total_loss!o�@

error_R�@?

learning_rate_1��5/g�I       6%�	$��d���A�*;


total_losssp�@

error_R3EB?

learning_rate_1��5`��aI       6%�	~	�d���A�*;


total_loss�5�@

error_Ri�E?

learning_rate_1��5�1%I       6%�	�Q�d���A�*;


total_losslV�@

error_R$b`?

learning_rate_1��5� ��I       6%�	Y��d���A�*;


total_lossq�@

error_R��N?

learning_rate_1��5��t�I       6%�	���d���A�*;


total_loss�|�@

error_RCI?

learning_rate_1��5&`��I       6%�	)�d���A�*;


total_lossJ��@

error_R�gJ?

learning_rate_1��5T��I       6%�	�k�d���A�*;


total_loss?��@

error_R�]E?

learning_rate_1��5��=�I       6%�	��d���A�*;


total_lossEo�@

error_R��=?

learning_rate_1��5�CFI       6%�	
��d���A�*;


total_loss4�@

error_Rs�N?

learning_rate_1��5��I       6%�	�C�d���A�*;


total_loss�T�@

error_R�SV?

learning_rate_1��5e�i?I       6%�	ä�d���A�*;


total_loss!]�@

error_RE�B?

learning_rate_1��5w��I       6%�	���d���A�*;


total_loss��@

error_R]�Q?

learning_rate_1��50ZAVI       6%�	�7�d���A�*;


total_loss@]�@

error_R{�N?

learning_rate_1��5�II       6%�	�}�d���A�*;


total_loss���@

error_R�KQ?

learning_rate_1��5�Ց�I       6%�	,��d���A�*;


total_loss=r�@

error_R߽O?

learning_rate_1��5]q�I       6%�	�d���A�*;


total_loss8m�@

error_R��W?

learning_rate_1��5"�I       6%�	6U�d���A�*;


total_loss�-�@

error_Rv[`?

learning_rate_1��5gWO�I       6%�	��d���A�*;


total_loss�<�@

error_R�R?

learning_rate_1��5����I       6%�	���d���A�*;


total_loss�[�@

error_R�y2?

learning_rate_1��5�}I       6%�	X-�d���A�*;


total_loss@�@

error_R�Q?

learning_rate_1��5�z�I       6%�	�x�d���A�*;


total_loss1˷@

error_R�Cb?

learning_rate_1��5+R�]I       6%�	���d���A�*;


total_lossX��@

error_R[E?

learning_rate_1��5x���I       6%�	"�d���A�*;


total_lossE��@

error_R��A?

learning_rate_1��5��b�I       6%�	j�d���A�*;


total_loss�b�@

error_R�xT?

learning_rate_1��5��I       6%�	H��d���A�*;


total_loss���@

error_RvV?

learning_rate_1��5���I       6%�	k�d���A�*;


total_loss��@

error_R\+^?

learning_rate_1��5�Y4eI       6%�	jR�d���A�*;


total_loss1w@

error_R��U?

learning_rate_1��5\�h�I       6%�	ژ�d���A�*;


total_loss�	A

error_RλR?

learning_rate_1��5�z��I       6%�	���d���A�*;


total_losse4�@

error_R]�I?

learning_rate_1��5L?�I       6%�	��d���A�*;


total_losssI`@

error_RO�F?

learning_rate_1��519�I       6%�	�c�d���A�*;


total_loss{�A

error_RnE?

learning_rate_1��5x��I       6%�	���d���A�*;


total_loss��@

error_R{�R?

learning_rate_1��5]YcI       6%�	���d���A�*;


total_lossE��@

error_Rl�E?

learning_rate_1��5� ��I       6%�	�3�d���A�*;


total_lossэ�@

error_R�1C?

learning_rate_1��5:A�I       6%�	�w�d���A�*;


total_loss��@

error_R�9K?

learning_rate_1��5�G~�I       6%�	��d���A�*;


total_loss��@

error_R?�8?

learning_rate_1��5���tI       6%�	��d���A�*;


total_losszՙ@

error_R��Y?

learning_rate_1��5Ed�aI       6%�	J�d���A�*;


total_loss��s@

error_R�[?

learning_rate_1��5!�`�I       6%�	���d���A�*;


total_lossc��@

error_R(Nh?

learning_rate_1��55˂I       6%�	���d���A�*;


total_loss<z@

error_R�:7?

learning_rate_1��5)�EI       6%�	��d���A�*;


total_loss<+�@

error_RO�N?

learning_rate_1��5^�0I       6%�	�Z�d���A�*;


total_loss�;�@

error_R.;I?

learning_rate_1��5��תI       6%�	���d���A�*;


total_loss��@

error_R��R?

learning_rate_1��5b���I       6%�	3��d���A�*;


total_loss[®@

error_R�YO?

learning_rate_1��5�;I       6%�	_2�d���A�*;


total_loss؆u@

error_R��+?

learning_rate_1��5��.I       6%�	mz�d���A�*;


total_loss��@

error_R}�@?

learning_rate_1��5�n�I       6%�	��d���A�*;


total_loss�[�@

error_R�;I?

learning_rate_1��5]pCyI       6%�	��d���A�*;


total_loss�!�@

error_R�a?

learning_rate_1��5EVJ�I       6%�	IN�d���A�*;


total_loss
��@

error_R�U?

learning_rate_1��5��(@I       6%�	���d���A�*;


total_loss�'A

error_R�`?

learning_rate_1��5�[�wI       6%�	���d���A�*;


total_losso��@

error_R9M?

learning_rate_1��5r��I       6%�	� �d���A�*;


total_loss&I�@

error_R�A?

learning_rate_1��5��"I       6%�	�f�d���A�*;


total_loss�o@

error_R��=?

learning_rate_1��5^�I       6%�	��d���A�*;


total_loss�ag@

error_R��H?

learning_rate_1��5�&�iI       6%�	��d���A�*;


total_loss�(�@

error_R$Qa?

learning_rate_1��5��ώI       6%�	,>�d���A�*;


total_loss�f	A

error_R�jF?

learning_rate_1��5��XI       6%�	��d���A�*;


total_loss���@

error_R�Y?

learning_rate_1��5�4�I       6%�	e��d���A�*;


total_loss�o�@

error_R԰@?

learning_rate_1��5[�CI       6%�	f�d���A�*;


total_loss�_�@

error_R��L?

learning_rate_1��5���dI       6%�	�R�d���A�*;


total_loss�q�@

error_R|�@?

learning_rate_1��5M��I       6%�	��d���A�*;


total_lossH��@

error_R�fG?

learning_rate_1��5|dI       6%�	���d���A�*;


total_loss��j@

error_R�;8?

learning_rate_1��5��PI       6%�	� �d���A�*;


total_loss��@

error_R[}R?

learning_rate_1��5��mI       6%�	Yh�d���A�*;


total_loss��@

error_R=�^?

learning_rate_1��5��VI       6%�	b��d���A�*;


total_loss��@

error_Rx�>?

learning_rate_1��5�Q�nI       6%�	O��d���A�*;


total_loss#��@

error_RsUP?

learning_rate_1��5���^I       6%�	7�d���A�*;


total_loss���@

error_RZ�E?

learning_rate_1��5���0I       6%�	(��d���A�*;


total_loss�\�@

error_R�P?

learning_rate_1��5R5�rI       6%�	���d���A�*;


total_loss3�@

error_R�E?

learning_rate_1��5'!�I       6%�	Q8�d���A�*;


total_loss�͘@

error_RO�T?

learning_rate_1��5[{3�I       6%�	���d���A�*;


total_lossP�@

error_R��]?

learning_rate_1��5=G�QI       6%�	���d���A�*;


total_loss�ț@

error_R�LE?

learning_rate_1��59���I       6%�	?�d���A�*;


total_lossr��@

error_R�*b?

learning_rate_1��56�<hI       6%�	BP�d���A�*;


total_lossv4/A

error_R@�`?

learning_rate_1��57ﮩI       6%�	I��d���A�*;


total_loss���@

error_R�H?

learning_rate_1��5���PI       6%�	O��d���A�*;


total_loss�͢@

error_R�zS?

learning_rate_1��5 ��I       6%�	��d���A�*;


total_loss�y�@

error_R��Y?

learning_rate_1��5� �=I       6%�	�]�d���A�*;


total_loss��y@

error_R�-I?

learning_rate_1��5"�VI       6%�	��d���A�*;


total_loss���@

error_R�S?

learning_rate_1��5E�U�I       6%�	���d���A�*;


total_loss���@

error_R��S?

learning_rate_1��5)���I       6%�	*:�d���A�*;


total_loss'�@

error_R�B?

learning_rate_1��5��!I       6%�	Q}�d���A�*;


total_lossN2�@

error_R�U?

learning_rate_1��5b��	I       6%�	���d���A�*;


total_lossJk�@

error_R��Z?

learning_rate_1��5 +$I       6%�	��d���A�*;


total_loss���@

error_R׋@?

learning_rate_1��5C7�I       6%�	�L�d���A�*;


total_loss���@

error_R�K?

learning_rate_1��5D1I       6%�	��d���A�*;


total_loss!ۤ@

error_Rs�U?

learning_rate_1��5_���I       6%�	z�d���A�*;


total_loss�.�@

error_R�I1?

learning_rate_1��5$X�rI       6%�	?P�d���A�*;


total_losszw�@

error_R8�O?

learning_rate_1��5�-Z6I       6%�	��d���A�*;


total_loss�,�@

error_RQsP?

learning_rate_1��5hG�I       6%�	���d���A�*;


total_lossZ��@

error_R�I?

learning_rate_1��5�j�I       6%�	+�d���A�*;


total_loss�g@

error_R<N?

learning_rate_1��5�/�I       6%�	nx�d���A�*;


total_lossFt�@

error_RJ@I?

learning_rate_1��57���I       6%�	���d���A�*;


total_lossT%�@

error_RHVC?

learning_rate_1��5Y[Y�I       6%�	��d���A�*;


total_loss��@

error_R��S?

learning_rate_1��5�P�I       6%�	�M�d���A�*;


total_loss6�@

error_Rvb?

learning_rate_1��5��qKI       6%�	ϖ�d���A�*;


total_loss���@

error_R�qM?

learning_rate_1��5
���I       6%�	8��d���A�*;


total_lossq<A

error_R�^f?

learning_rate_1��5�I       6%�	�d���A�*;


total_lossR�@

error_R��N?

learning_rate_1��5�`��I       6%�	`a�d���A�*;


total_lossq�@

error_R�R?

learning_rate_1��5�H�5I       6%�	���d���A�*;


total_loss�ͷ@

error_R��??

learning_rate_1��5�!'�I       6%�	���d���A�*;


total_loss��@

error_R�wF?

learning_rate_1��5���I       6%�	�/�d���A�*;


total_lossWVj@

error_Rr:8?

learning_rate_1��5���I       6%�	�x�d���A�*;


total_loss�W�@

error_R��C?

learning_rate_1��5l��I       6%�	��d���A�*;


total_loss
��@

error_Rn�V?

learning_rate_1��5;KΛI       6%�	C�d���A�*;


total_loss!�@

error_R̘K?

learning_rate_1��5���[I       6%�	>P�d���A�*;


total_loss�^�@

error_R��P?

learning_rate_1��5�{��I       6%�	���d���A�*;


total_loss�Ţ@

error_Rx"I?

learning_rate_1��5�n��I       6%�	"��d���A�*;


total_loss=t�@

error_R�V?

learning_rate_1��5F^I       6%�	�!�d���A�*;


total_loss�#�@

error_R�cL?

learning_rate_1��5��I       6%�	�d�d���A�*;


total_loss)�@

error_RZhI?

learning_rate_1��5	y\�I       6%�	���d���A�*;


total_loss���@

error_R�tT?

learning_rate_1��5�&yI       6%�	P��d���A�*;


total_loss��@

error_R�T?

learning_rate_1��5O��}I       6%�	OC�d���A�*;


total_lossfD�@

error_R7�L?

learning_rate_1��5Ղ�kI       6%�	[��d���A�*;


total_loss]�@

error_R��i?

learning_rate_1��5%��I       6%�	���d���A�*;


total_loss�.�@

error_R��M?

learning_rate_1��5N#I       6%�	��d���A�*;


total_loss�ѓ@

error_RæU?

learning_rate_1��5��ɲI       6%�	SV�d���A�*;


total_loss;��@

error_Rs�9?

learning_rate_1��5�?.FI       6%�	���d���A�*;


total_loss���@

error_R�iX?

learning_rate_1��5}i�wI       6%�	���d���A�*;


total_loss�1�@

error_R7TV?

learning_rate_1��5^��I       6%�	�,�d���A�*;


total_lossī�@

error_R�?B?

learning_rate_1��5�� �I       6%�	�s�d���A�*;


total_lossW1�@

error_R�W?

learning_rate_1��5H{�I       6%�	ϵ�d���A�*;


total_loss�n�@

error_R�T?

learning_rate_1��5(x-BI       6%�	���d���A�*;


total_loss���@

error_R&}X?

learning_rate_1��5��I       6%�	�=�d���A�*;


total_loss+�@

error_R�-L?

learning_rate_1��5��1LI       6%�	\��d���A�*;


total_loss��@

error_RC0C?

learning_rate_1��5�=��I       6%�	Z��d���A�*;


total_loss�w�@

error_RM�@?

learning_rate_1��5�^�8I       6%�	�/�d���A�*;


total_loss��@

error_R�I?

learning_rate_1��5��&�I       6%�	qx�d���A�*;


total_loss���@

error_RE�P?

learning_rate_1��5:�GI       6%�		��d���A�*;


total_loss�լ@

error_RE�S?

learning_rate_1��5�*��I       6%�	;�d���A�*;


total_loss2mv@

error_RZ@U?

learning_rate_1��5�3�I       6%�	gG�d���A�*;


total_loss�0u@

error_R1:=?

learning_rate_1��5@�2�I       6%�	���d���A�*;


total_loss�4A

error_R�R?

learning_rate_1��5���I       6%�	���d���A�*;


total_loss���@

error_R	�S?

learning_rate_1��5�k�I       6%�	[% e���A�*;


total_loss5�#A

error_R�.N?

learning_rate_1��5��e`I       6%�	�t e���A�*;


total_lossm'A

error_R��i?

learning_rate_1��5�I       6%�	� e���A�*;


total_lossҌ�@

error_R;:M?

learning_rate_1��5P�PBI       6%�	�e���A�*;


total_loss<��@

error_Rl}W?

learning_rate_1��5�@��I       6%�	|de���A�*;


total_loss7^�@

error_RsaT?

learning_rate_1��5��|HI       6%�	��e���A�*;


total_loss	��@

error_Rs[G?

learning_rate_1��5��h�I       6%�	G
e���A�*;


total_lossq�@

error_R�qR?

learning_rate_1��5�b�SI       6%�	<Re���A�*;


total_loss�b�@

error_R��B?

learning_rate_1��5m'I       6%�	��e���A�*;


total_loss��@

error_R�I?

learning_rate_1��56��=I       6%�	 �e���A�*;


total_loss;޵@

error_R&�O?

learning_rate_1��5���I       6%�	�+e���A�*;


total_loss���@

error_R�$L?

learning_rate_1��5g��I       6%�	�ve���A�*;


total_loss�s�@

error_R�Y?

learning_rate_1��5`V�I       6%�	��e���A�*;


total_loss/�@

error_Rn6H?

learning_rate_1��5�gx;I       6%�	�e���A�*;


total_lossV�@

error_R��I?

learning_rate_1��57#~�I       6%�	�He���A�*;


total_loss���@

error_R�V?

learning_rate_1��5��-�I       6%�	m�e���A�*;


total_loss�֦@

error_R�V?

learning_rate_1��5�s�I       6%�	��e���A�*;


total_loss���@

error_RE�R?

learning_rate_1��5��,pI       6%�	@e���A�*;


total_loss,�@

error_RW#C?

learning_rate_1��5�^I       6%�	Ye���A�*;


total_loss��@

error_R��\?

learning_rate_1��5��VI       6%�	�e���A�*;


total_loss���@

error_RxOL?

learning_rate_1��5���wI       6%�	��e���A�*;


total_loss:S�@

error_R�sC?

learning_rate_1��5g��I       6%�	W,e���A�*;


total_lossE��@

error_R�Ca?

learning_rate_1��5��b�I       6%�	�qe���A�*;


total_loss���@

error_R��D?

learning_rate_1��5���I       6%�	z�e���A�*;


total_loss#��@

error_R��Y?

learning_rate_1��5��a�I       6%�	�e���A�*;


total_loss�$�@

error_R��S?

learning_rate_1��5~%v�I       6%�	�=e���A�*;


total_loss\��@

error_RC?

learning_rate_1��59��yI       6%�	��e���A�*;


total_loss��p@

error_RD�T?

learning_rate_1��5�JI       6%�	f�e���A�*;


total_loss��@

error_R��T?

learning_rate_1��5@�I       6%�	�
e���A�*;


total_loss���@

error_R�`Z?

learning_rate_1��5t?��I       6%�	�Le���A�*;


total_loss]��@

error_R��??

learning_rate_1��5.��I       6%�	0�e���A�*;


total_loss4V�@

error_RT�X?

learning_rate_1��5�!�I       6%�	��e���A�*;


total_loss�_�@

error_Ra�G?

learning_rate_1��5��-I       6%�	�	e���A�*;


total_loss$�@

error_R=\?

learning_rate_1��5��I       6%�	�_	e���A�*;


total_loss
��@

error_R��J?

learning_rate_1��5ݯ;�I       6%�	x�	e���A�*;


total_loss$�@

error_R��G?

learning_rate_1��5p\ksI       6%�	#�	e���A�*;


total_loss��@

error_R�I?

learning_rate_1��5�!��I       6%�	�9
e���A�*;


total_loss���@

error_R&;D?

learning_rate_1��56h�I       6%�	L}
e���A�*;


total_loss�I�@

error_R��S?

learning_rate_1��5��|I       6%�	^�
e���A�*;


total_lossd�U@

error_R�N?

learning_rate_1��5U��LI       6%�	,e���A�*;


total_loss�;�@

error_R�*7?

learning_rate_1��5�E=�I       6%�	QHe���A�*;


total_lossa}�@

error_R�=?

learning_rate_1��5V=�I       6%�	}�e���A�*;


total_loss�5�@

error_R�VB?

learning_rate_1��5:UO^I       6%�	s�e���A�*;


total_loss�-�@

error_R�_E?

learning_rate_1��5hz�I       6%�	_e���A�*;


total_loss[HK@

error_R��<?

learning_rate_1��5n?~I       6%�	�de���A�*;


total_loss��y@

error_RwH?

learning_rate_1��5��yI       6%�	�e���A�*;


total_loss�u�@

error_R�.O?

learning_rate_1��5���,I       6%�	��e���A�*;


total_loss�R�@

error_R�M?

learning_rate_1��5�`�I       6%�	�;e���A�*;


total_lossH�@

error_R��M?

learning_rate_1��5�l�I       6%�	.�e���A�*;


total_loss���@

error_R��W?

learning_rate_1��5��E:I       6%�	��e���A�*;


total_loss_<�@

error_R�Q?

learning_rate_1��5ǋ�2I       6%�	�5e���A�*;


total_loss_f@

error_R�QH?

learning_rate_1��5Sx2aI       6%�	�{e���A�*;


total_loss>Y�@

error_R&�D?

learning_rate_1��5���vI       6%�	ھe���A�*;


total_loss��@

error_R��M?

learning_rate_1��5P�kI       6%�	te���A�*;


total_loss���@

error_R�,;?

learning_rate_1��5�
�,I       6%�	UJe���A�*;


total_loss�.A

error_Rs�O?

learning_rate_1��5#S8I       6%�	��e���A�*;


total_loss��@

error_R��Q?

learning_rate_1��5F��I       6%�	��e���A�*;


total_loss�t�@

error_Rc�G?

learning_rate_1��5|�%I       6%�	�e���A�*;


total_loss@h�@

error_R�J?

learning_rate_1��5����I       6%�	FUe���A�*;


total_loss]�o@

error_R��[?

learning_rate_1��5��I       6%�	3�e���A�*;


total_loss.�@

error_R�*Q?

learning_rate_1��5��I       6%�	��e���A�*;


total_loss�;�@

error_R��M?

learning_rate_1��5	{jkI       6%�	@%e���A�*;


total_loss�K�@

error_R�mC?

learning_rate_1��5D<ZI       6%�	�ie���A�*;


total_lossn��@

error_R�B?

learning_rate_1��5���I       6%�	l�e���A�*;


total_loss��@

error_R�UY?

learning_rate_1��5J�I       6%�	"�e���A�*;


total_loss���@

error_R��H?

learning_rate_1��5@6��I       6%�	�Ie���A�*;


total_lossAF�@

error_R��D?

learning_rate_1��5���I       6%�	6�e���A�*;


total_loss���@

error_R��A?

learning_rate_1��5Jz��I       6%�	q�e���A�*;


total_loss��@

error_R��L?

learning_rate_1��5�{I       6%�	Se���A�*;


total_lossE-z@

error_R8�M?

learning_rate_1��5�ؒ�I       6%�	�^e���A�*;


total_loss�K�@

error_R��J?

learning_rate_1��5�׻I       6%�	�e���A�*;


total_loss��@

error_R�'[?

learning_rate_1��5�^/�I       6%�	(�e���A�*;


total_loss���@

error_R�F?

learning_rate_1��5#H�I       6%�	�*e���A�*;


total_lossG�@

error_R�O?

learning_rate_1��5�7UI       6%�	�ne���A�*;


total_loss#��@

error_R�X?

learning_rate_1��5}�lUI       6%�	y�e���A�*;


total_loss�|�@

error_R�F<?

learning_rate_1��5Q���I       6%�	��e���A�*;


total_loss-ʣ@

error_R�C8?

learning_rate_1��5�n�I       6%�	�3e���A�*;


total_loss�@

error_R7(M?

learning_rate_1��5-KܩI       6%�		xe���A�*;


total_loss�1�@

error_R#PV?

learning_rate_1��5J`�`I       6%�	9�e���A�*;


total_loss��@

error_R��F?

learning_rate_1��5���I       6%�	� e���A�*;


total_loss@

error_R�=?

learning_rate_1��5���/I       6%�	De���A�*;


total_loss��@

error_Ra�U?

learning_rate_1��55/�I       6%�	�e���A�*;


total_loss���@

error_R+Q?

learning_rate_1��59(��I       6%�	�e���A�*;


total_loss�^�@

error_RM�a?

learning_rate_1��5�cn]I       6%�	�e���A�*;


total_loss��@

error_RZM?

learning_rate_1��5کII       6%�	�^e���A�*;


total_loss=\�@

error_R/�K?

learning_rate_1��5���xI       6%�	צe���A�*;


total_loss.��@

error_RJO;?

learning_rate_1��5�ۭEI       6%�	��e���A�*;


total_loss]_�@

error_RQ??

learning_rate_1��5����I       6%�	�5e���A�*;


total_loss�W�@

error_R��D?

learning_rate_1��5+zBI       6%�	�{e���A�*;


total_lossR_�@

error_R
*H?

learning_rate_1��5����I       6%�	$�e���A�*;


total_loss�ܾ@

error_RH5M?

learning_rate_1��5���MI       6%�	Xe���A�*;


total_lossID�@

error_R��W?

learning_rate_1��5��r�I       6%�	�Oe���A�*;


total_loss�]w@

error_R��I?

learning_rate_1��59��SI       6%�	��e���A�*;


total_loss��@

error_R�Z??

learning_rate_1��5�ТI       6%�	��e���A�*;


total_loss���@

error_R�-M?

learning_rate_1��5��I       6%�	�&e���A�*;


total_lossn�@

error_R1w8?

learning_rate_1��5"jًI       6%�	|ke���A�*;


total_loss<�@

error_R��??

learning_rate_1��5T!�fI       6%�	y�e���A�*;


total_lossA

error_R��@?

learning_rate_1��5%�/rI       6%�	 �e���A�*;


total_loss�a@

error_RH�D?

learning_rate_1��5��o�I       6%�	Ge���A�*;


total_loss�"�@

error_R��K?

learning_rate_1��5�Z�I       6%�	]�e���A�*;


total_loss���@

error_R��C?

learning_rate_1��5�;��I       6%�	��e���A�*;


total_loss�vk@

error_R�C?

learning_rate_1��5D��SI       6%�	x"e���A�*;


total_loss���@

error_R��S?

learning_rate_1��5q_s6I       6%�	$ie���A�*;


total_loss��A

error_R��V?

learning_rate_1��5dŀ�I       6%�	��e���A�*;


total_lossz��@

error_R�wE?

learning_rate_1��5�VJ�I       6%�	� e���A�*;


total_loss�K�@

error_R|b?

learning_rate_1��5Rv��I       6%�	�`e���A�*;


total_loss/�@

error_R{�P?

learning_rate_1��5K��I       6%�	��e���A�*;


total_loss�0�@

error_R�2T?

learning_rate_1��5c��CI       6%�	�e���A�*;


total_lossH�@

error_R��B?

learning_rate_1��5=��uI       6%�	�Re���A�*;


total_loss6W�@

error_R��;?

learning_rate_1��5��<�I       6%�	��e���A�*;


total_loss��@

error_R�_?

learning_rate_1��56���I       6%�	f�e���A�*;


total_loss���@

error_RErT?

learning_rate_1��5��f�I       6%�	�e���A�*;


total_lossE^@

error_R��P?

learning_rate_1��5�駾I       6%�	oae���A�*;


total_loss��@

error_R�}B?

learning_rate_1��5R�%I       6%�	�e���A�*;


total_loss���@

error_RLLB?

learning_rate_1��5��I       6%�	.�e���A�*;


total_lossI��@

error_R�WN?

learning_rate_1��5tg�.I       6%�	�) e���A�*;


total_loss7�@

error_R6V<?

learning_rate_1��5nSI       6%�	p e���A�*;


total_loss��@

error_R�VX?

learning_rate_1��5�l}�I       6%�	W� e���A�*;


total_lossoՅ@

error_Rr�M?

learning_rate_1��5óxI       6%�	�!e���A�*;


total_loss<}�@

error_R�K?

learning_rate_1��5[�I       6%�	�c!e���A�*;


total_lossJe�@

error_R��G?

learning_rate_1��5ҪI       6%�	_�!e���A�*;


total_losstޖ@

error_R�L?

learning_rate_1��5�uu3I       6%�	�
"e���A�*;


total_loss�_�@

error_Rn`@?

learning_rate_1��5&8�I       6%�	kT"e���A�*;


total_loss�Ƞ@

error_R��O?

learning_rate_1��5��I       6%�	��"e���A�*;


total_loss���@

error_R9[?

learning_rate_1��5��nI       6%�	I�"e���A�*;


total_loss#��@

error_Ra�O?

learning_rate_1��5m�g�I       6%�	[7#e���A�*;


total_loss"��@

error_R�-\?

learning_rate_1��5��k�I       6%�	�}#e���A�*;


total_loss)�l@

error_R��Z?

learning_rate_1��5��eAI       6%�	�#e���A�*;


total_loss���@

error_R��F?

learning_rate_1��5^���I       6%�		$e���A�*;


total_loss*Ê@

error_R�R?

learning_rate_1��5�xNrI       6%�	U$e���A�*;


total_lossp�@

error_R?mX?

learning_rate_1��5!��I       6%�	�$e���A�*;


total_loss�|�@

error_RS�S?

learning_rate_1��5߭w�I       6%�	��$e���A�*;


total_loss���@

error_R �]?

learning_rate_1��5��*�I       6%�	�&%e���A�*;


total_loss���@

error_R�L?

learning_rate_1��5�5_I       6%�	qr%e���A�*;


total_loss���@

error_Ro}U?

learning_rate_1��5<�w�I       6%�	��%e���A�*;


total_loss���@

error_R�RV?

learning_rate_1��5)u!HI       6%�	�&e���A�*;


total_loss��@

error_R��7?

learning_rate_1��5ͧ`*I       6%�	�K&e���A�*;


total_loss�ux@

error_R_G?

learning_rate_1��5G�a�I       6%�	��&e���A�*;


total_loss�>�@

error_R�]V?

learning_rate_1��5a���I       6%�	��&e���A�*;


total_loss���@

error_R��c?

learning_rate_1��5s��I       6%�	�'e���A�*;


total_loss���@

error_R&�E?

learning_rate_1��5
��gI       6%�	�e'e���A�*;


total_loss��@

error_Rtb]?

learning_rate_1��5z�w�I       6%�	�'e���A�*;


total_lossrL�@

error_Rf$L?

learning_rate_1��5?-O�I       6%�	��'e���A�*;


total_loss���@

error_RVhX?

learning_rate_1��5�tEEI       6%�	":(e���A�*;


total_loss�I�@

error_R4�Y?

learning_rate_1��5X�3>I       6%�		�(e���A�*;


total_loss���@

error_R<�L?

learning_rate_1��5��%�I       6%�	�(e���A�*;


total_loss4��@

error_Rd�??

learning_rate_1��5���|I       6%�	)e���A�*;


total_lossf��@

error_R*P?

learning_rate_1��5a]�)I       6%�	�Z)e���A�*;


total_loss�3�@

error_RQ?

learning_rate_1��5�z4'I       6%�	_�)e���A�*;


total_loss��@

error_RJ!X?

learning_rate_1��5�̩`I       6%�	�)e���A�*;


total_loss��@

error_R�]D?

learning_rate_1��5���I       6%�	�(*e���A�*;


total_loss�:�@

error_R�M?

learning_rate_1��5�I       6%�	�k*e���A�*;


total_loss���@

error_R��c?

learning_rate_1��5���I       6%�	y�*e���A�*;


total_loss���@

error_R,�I?

learning_rate_1��5J1`II       6%�	��*e���A�*;


total_lossH�@

error_R�l??

learning_rate_1��5�-��I       6%�	6+e���A�*;


total_loss�|�@

error_R	G?

learning_rate_1��5jˆ�I       6%�	y+e���A�*;


total_loss:��@

error_RF?

learning_rate_1��5�BI       6%�	��+e���A�*;


total_loss��@

error_R��U?

learning_rate_1��5��4�I       6%�	,e���A�*;


total_lossD��@

error_R�I?

learning_rate_1��5�@�+I       6%�	0G,e���A�*;


total_loss��@

error_R�GI?

learning_rate_1��5�Q�I       6%�	��,e���A�*;


total_loss���@

error_R_CH?

learning_rate_1��5��doI       6%�	j�,e���A�*;


total_loss ^�@

error_R8PU?

learning_rate_1��5�G��I       6%�	�-e���A�*;


total_loss�_�@

error_R��G?

learning_rate_1��5���I       6%�	�[-e���A�*;


total_loss�p�@

error_R<�B?

learning_rate_1��5]��I       6%�	��-e���A�*;


total_loss�f�@

error_R��K?

learning_rate_1��5[��{I       6%�	�.e���A�*;


total_loss���@

error_R@�]?

learning_rate_1��5�GI       6%�	&O.e���A�*;


total_loss׉�@

error_R�*H?

learning_rate_1��5���I       6%�	��.e���A�*;


total_loss���@

error_R!�:?

learning_rate_1��5��-I       6%�	a�.e���A�*;


total_loss�G�@

error_RP?

learning_rate_1��5?l*CI       6%�	>%/e���A�*;


total_loss�?CA

error_RMI?

learning_rate_1��5���I       6%�	�o/e���A�*;


total_loss�a�@

error_RYO?

learning_rate_1��5���I       6%�	}�/e���A�*;


total_loss�L�@

error_RnO]?

learning_rate_1��5mBII       6%�	w�/e���A�*;


total_loss��@

error_RH0N?

learning_rate_1��5?b�I       6%�	�D0e���A�*;


total_loss�-A

error_R�CP?

learning_rate_1��5��L)I       6%�	�0e���A�*;


total_lossN��@

error_Rj�c?

learning_rate_1��5R�) I       6%�	��0e���A�*;


total_lossf�@

error_R��E?

learning_rate_1��5­N�I       6%�	61e���A�*;


total_loss�&�@

error_R��3?

learning_rate_1��5��N�I       6%�	j\1e���A�*;


total_loss�A

error_R��k?

learning_rate_1��5Ԇ��I       6%�	��1e���A�*;


total_loss�ҽ@

error_R�=J?

learning_rate_1��5���I       6%�	��1e���A�*;


total_loss;� A

error_R�)N?

learning_rate_1��5jn�OI       6%�	�=2e���A�*;


total_loss�A

error_R��N?

learning_rate_1��5˭��I       6%�	L�2e���A�*;


total_loss�[�@

error_R�M?

learning_rate_1��5L��I       6%�	��2e���A�*;


total_lossi5A

error_R�??

learning_rate_1��5eI       6%�	3e���A�*;


total_lossŌ@

error_RR�N?

learning_rate_1��5�'FI       6%�	f3e���A�*;


total_loss���@

error_R�Qh?

learning_rate_1��53�LI       6%�	�3e���A�*;


total_loss�J�@

error_R�PR?

learning_rate_1��5!5�rI       6%�	��3e���A�*;


total_lossD��@

error_R�E?

learning_rate_1��51��I       6%�	=14e���A�*;


total_lossn��@

error_R�P?

learning_rate_1��5��I       6%�	�u4e���A�*;


total_loss8�@

error_R)�T?

learning_rate_1��5��I       6%�	�4e���A�*;


total_loss���@

error_R�s[?

learning_rate_1��5�O��I       6%�	�5e���A�*;


total_loss�@

error_R��o?

learning_rate_1��5q��BI       6%�	&R5e���A�*;


total_loss@�@

error_R3�H?

learning_rate_1��5�_�I       6%�	��5e���A�*;


total_loss\��@

error_R��O?

learning_rate_1��5KR}�I       6%�	�5e���A�*;


total_loss��-A

error_R��L?

learning_rate_1��5=v9�I       6%�	�6e���A�*;


total_loss���@

error_R{5\?

learning_rate_1��5�IVI       6%�	_c6e���A�*;


total_loss&nR@

error_R1�R?

learning_rate_1��5V�٥I       6%�	;�6e���A�*;


total_loss���@

error_R.�V?

learning_rate_1��5{0�I       6%�	'�6e���A�*;


total_lossF�@

error_Rl/O?

learning_rate_1��5ܚ��I       6%�	�/7e���A�*;


total_loss�P�@

error_R�J?

learning_rate_1��5D�X&I       6%�	�s7e���A�*;


total_loss��@

error_R�&R?

learning_rate_1��5�};I       6%�	ɻ7e���A�*;


total_loss�~�@

error_R;OO?

learning_rate_1��5�t�yI       6%�	�8e���A�*;


total_loss̺�@

error_RwNJ?

learning_rate_1��5	�I       6%�	�I8e���A�*;


total_loss�X�@

error_R�P?

learning_rate_1��5�33�I       6%�	W�8e���A�*;


total_loss��@

error_R�\?

learning_rate_1��5�HӻI       6%�	��8e���A�*;


total_loss���@

error_R=�G?

learning_rate_1��5����I       6%�	\%9e���A�*;


total_loss�y�@

error_R�uK?

learning_rate_1��5��DI       6%�	ƙ9e���A�*;


total_loss�c�@

error_R�W?

learning_rate_1��5Z��I       6%�	��9e���A�*;


total_loss�@

error_RMA?

learning_rate_1��5*�VI       6%�	>G:e���A�*;


total_loss�[�@

error_RDV?

learning_rate_1��5M}�I       6%�	/�:e���A�*;


total_loss#��@

error_R�dX?

learning_rate_1��5p�I       6%�	N�:e���A�*;


total_loss2�@

error_R�@G?

learning_rate_1��5�#�I       6%�	�%;e���A�*;


total_loss���@

error_R@�7?

learning_rate_1��5����I       6%�	�n;e���A�*;


total_loss�ٴ@

error_RHG?

learning_rate_1��5[!X�I       6%�	��;e���A�*;


total_loss|`�@

error_R\vR?

learning_rate_1��5[Z��I       6%�	�;e���A�*;


total_loss��g@

error_R�eR?

learning_rate_1��5=�*tI       6%�	DB<e���A�*;


total_losso�@

error_R�]?

learning_rate_1��5��,<I       6%�	�<e���A�*;


total_loss;	�@

error_R�I?

learning_rate_1��5�C��I       6%�	��<e���A�*;


total_lossҔ�@

error_R,�]?

learning_rate_1��5�rwsI       6%�	�&=e���A�*;


total_loss�j�@

error_RvCC?

learning_rate_1��5��I       6%�	�=e���A�*;


total_loss �}@

error_R�#g?

learning_rate_1��5�+%I       6%�	�=e���A�*;


total_loss�Ë@

error_R�V?

learning_rate_1��5Od�I       6%�	p>e���A�*;


total_lossF��@

error_R�>V?

learning_rate_1��5؂��I       6%�	�d>e���A�*;


total_loss���@

error_R��L?

learning_rate_1��5['�UI       6%�	��>e���A�*;


total_lossE��@

error_RXCS?

learning_rate_1��5���I       6%�	�>e���A�*;


total_lossZ�@

error_R�X?

learning_rate_1��5���I       6%�	�:?e���A�*;


total_loss���@

error_R�:d?

learning_rate_1��5գ+I       6%�	z�?e���A�*;


total_lossCu@

error_Rx==?

learning_rate_1��52��I       6%�	7�?e���A�*;


total_loss�Z�@

error_R��^?

learning_rate_1��5=��XI       6%�	[@e���A�*;


total_lossT`�@

error_R�A?

learning_rate_1��5���I       6%�	�b@e���A�*;


total_loss�3�@

error_R�J?

learning_rate_1��5���TI       6%�	��@e���A�*;


total_loss<��@

error_R�Y?

learning_rate_1��5�S}yI       6%�	�"Ae���A�*;


total_loss�Q�@

error_R��R?

learning_rate_1��56X7\I       6%�	rmAe���A�*;


total_lossQN�@

error_R��R?

learning_rate_1��5Ŝr�I       6%�	��Ae���A�*;


total_loss���@

error_R*jc?

learning_rate_1��5J[�I       6%�	=Be���A�*;


total_loss��@

error_R�-L?

learning_rate_1��5��=I       6%�	��Be���A�*;


total_loss\U�@

error_R��C?

learning_rate_1��5�N WI       6%�	`�Be���A�*;


total_loss�Y�@

error_R&G?

learning_rate_1��5H�!I       6%�	�Ce���A�*;


total_loss��@

error_RJE]?

learning_rate_1��5�'�I       6%�	.QCe���A�*;


total_loss���@

error_R\�R?

learning_rate_1��5��'�I       6%�	��Ce���A�*;


total_lossQJ�@

error_R�rV?

learning_rate_1��5!��I       6%�	��Ce���A�*;


total_loss}E�@

error_R��[?

learning_rate_1��5t
�I       6%�	i&De���A�*;


total_lossZ�@

error_R��G?

learning_rate_1��5+�pI       6%�	kkDe���A�*;


total_lossܩ�@

error_R�QD?

learning_rate_1��5K0�I       6%�	d�De���A�*;


total_loss�J�@

error_R��J?

learning_rate_1��5o͡iI       6%�	�De���A�*;


total_loss���@

error_RmL?

learning_rate_1��5�/76I       6%�	B5Ee���A�*;


total_loss���@

error_R�E?

learning_rate_1��5��FI       6%�	�{Ee���A�*;


total_losstoo@

error_R&�X?

learning_rate_1��5�p�I       6%�	��Ee���A�*;


total_loss��@

error_R�zG?

learning_rate_1��5 �� I       6%�	Fe���A�*;


total_loss�Z�@

error_R��:?

learning_rate_1��5M���I       6%�	zSFe���A�*;


total_loss|z�@

error_R��C?

learning_rate_1��5=��I       6%�	L�Fe���A�*;


total_loss��@

error_RW<?

learning_rate_1��5�uQI       6%�	��Fe���A�*;


total_loss�f�@

error_R�SA?

learning_rate_1��5
	�I       6%�	�"Ge���A�*;


total_lossq��@

error_RO�[?

learning_rate_1��5�v�I       6%�	xeGe���A�*;


total_loss��@

error_R��@?

learning_rate_1��5�y��I       6%�	6�Ge���A�*;


total_lossAܣ@

error_R��I?

learning_rate_1��5|M�<I       6%�	�Ge���A�*;


total_loss�׽@

error_R��U?

learning_rate_1��5WP��I       6%�	�0He���A�*;


total_loss6��@

error_Rr\?

learning_rate_1��5NMYAI       6%�	�tHe���A�*;


total_loss�v�@

error_R��W?

learning_rate_1��5w*�I       6%�	q�He���A�*;


total_loss#+�@

error_R(�C?

learning_rate_1��5�˷I       6%�	$�He���A�*;


total_loss��@

error_R$RV?

learning_rate_1��5|�5{I       6%�	BIe���A�*;


total_loss}M�@

error_R;�A?

learning_rate_1��5�e��I       6%�	��Ie���A�*;


total_loss$��@

error_RseF?

learning_rate_1��5)7�I       6%�	��Ie���A�*;


total_loss�:�@

error_R�sG?

learning_rate_1��5�%�I       6%�	�Je���A�*;


total_loss/#A

error_Rq�[?

learning_rate_1��5]��I       6%�	6VJe���A�*;


total_loss��@

error_RJ _?

learning_rate_1��5�Y1KI       6%�	��Je���A�*;


total_loss>k�@

error_R��@?

learning_rate_1��5*��I       6%�	��Je���A�*;


total_lossI��@

error_RaW?

learning_rate_1��5��l�I       6%�	!9Ke���A�*;


total_lossH֎@

error_Ra�V?

learning_rate_1��5���I       6%�	r�Ke���A�*;


total_loss]�@

error_R�IF?

learning_rate_1��5�q�ZI       6%�	��Ke���A�*;


total_loss�@

error_RH�6?

learning_rate_1��5���,I       6%�	Le���A�*;


total_loss���@

error_R�D?

learning_rate_1��5��I       6%�	WLe���A�*;


total_loss;��@

error_R�3O?

learning_rate_1��5,tf�I       6%�	|�Le���A�*;


total_loss�6�@

error_R̛F?

learning_rate_1��54̓I       6%�	��Le���A�*;


total_lossAi�@

error_RO�P?

learning_rate_1��5��I       6%�	�0Me���A�*;


total_lossLOA

error_R�dN?

learning_rate_1��5�v�I       6%�	�Me���A�*;


total_loss2��@

error_R�n?

learning_rate_1��5y��I       6%�	��Me���A�*;


total_lossF�@

error_R�??

learning_rate_1��5H���I       6%�	�&Ne���A�*;


total_loss�|�@

error_R��7?

learning_rate_1��5���I       6%�	[nNe���A�*;


total_loss�c�@

error_R�>T?

learning_rate_1��5��VI       6%�	�Ne���A�*;


total_loss���@

error_R;@?

learning_rate_1��5��|I       6%�	�Ne���A�*;


total_loss�l�@

error_R��G?

learning_rate_1��5�2�I       6%�	�7Oe���A�*;


total_loss���@

error_R��F?

learning_rate_1��502�<I       6%�	P�Oe���A�*;


total_loss���@

error_RR�=?

learning_rate_1��5 if�I       6%�	��Oe���A�*;


total_loss��@

error_R��Q?

learning_rate_1��5c��)I       6%�	Pe���A�*;


total_loss�/�@

error_R�a\?

learning_rate_1��5ΊC�I       6%�	aPPe���A�*;


total_loss4A

error_R��[?

learning_rate_1��5#P!I       6%�	ݔPe���A�*;


total_loss�]�@

error_R��J?

learning_rate_1��5�TI       6%�	��Pe���A�*;


total_loss��p@

error_R�9F?

learning_rate_1��5�Z�I       6%�	-Qe���A�*;


total_loss�!5A

error_R	PH?

learning_rate_1��55I       6%�	~_Qe���A�*;


total_loss�ǵ@

error_R�9`?

learning_rate_1��5�;�I       6%�	@�Qe���A�*;


total_loss,�@

error_R�^H?

learning_rate_1��5dϷI       6%�	I�Qe���A�*;


total_lossz�@

error_RܓV?

learning_rate_1��5��g�I       6%�	T*Re���A�*;


total_lossl�@

error_RךR?

learning_rate_1��5`�TI       6%�	�mRe���A�*;


total_loss)�@

error_RcHV?

learning_rate_1��5����I       6%�	��Re���A�*;


total_lossi�|@

error_R)�S?

learning_rate_1��50OtI       6%�	i�Re���A�*;


total_loss��@

error_R�IQ?

learning_rate_1��5͝��I       6%�	�BSe���A�*;


total_loss�,�@

error_R��W?

learning_rate_1��5&�I       6%�	��Se���A�*;


total_lossT��@

error_R6�a?

learning_rate_1��5�'GEI       6%�	R�Se���A�*;


total_loss���@

error_R�K?

learning_rate_1��5�=%�I       6%�	�Te���A�*;


total_loss���@

error_R�3Q?

learning_rate_1��5�I       6%�	�YTe���A�*;


total_lossd��@

error_Rz�]?

learning_rate_1��5ؓ"I       6%�	��Te���A�*;


total_losssݗ@

error_Rd�L?

learning_rate_1��5�d�I       6%�	�Te���A�*;


total_lossƸ�@

error_R1�c?

learning_rate_1��5[//�I       6%�	>+Ue���A�*;


total_loss��@

error_RZ�X?

learning_rate_1��5����I       6%�	�pUe���A�*;


total_loss��A

error_R��S?

learning_rate_1��5����I       6%�	��Ue���A�*;


total_loss0�@

error_R4�`?

learning_rate_1��51�]�I       6%�	j�Ue���A�*;


total_loss�@

error_R��H?

learning_rate_1��5��=I       6%�	9Ve���A�*;


total_loss,u�@

error_RɃ^?

learning_rate_1��5�жMI       6%�	�zVe���A�*;


total_lossW��@

error_Rl�N?

learning_rate_1��5���!I       6%�	��Ve���A�*;


total_loss7�@

error_R)�G?

learning_rate_1��5�Q�I       6%�	� We���A�*;


total_loss��@

error_RS#R?

learning_rate_1��5[��{I       6%�	�CWe���A�*;


total_lossdV�@

error_RJ�K?

learning_rate_1��5� �I       6%�	��We���A�*;


total_loss�=�@

error_R$X?

learning_rate_1��5�e=I       6%�	��We���A�*;


total_loss L�@

error_R*@`?

learning_rate_1��5���HI       6%�	�Xe���A�*;


total_loss�-�@

error_R[�L?

learning_rate_1��5�=n�I       6%�	�]Xe���A�*;


total_loss��@

error_R=�U?

learning_rate_1��5ծ�WI       6%�	\�Xe���A�*;


total_loss���@

error_R�
A?

learning_rate_1��5�V6&I       6%�	t�Xe���A�*;


total_lossZ��@

error_R��[?

learning_rate_1��5i���I       6%�	58Ye���A�*;


total_lossH��@

error_RJDN?

learning_rate_1��5��HsI       6%�	�{Ye���A�*;


total_loss���@

error_R��L?

learning_rate_1��5�r�I       6%�	�Ye���A�*;


total_loss8P�@

error_R�wf?

learning_rate_1��5��:I       6%�	�Ze���A�*;


total_loss.0�@

error_R�U?

learning_rate_1��5T��I       6%�	tLZe���A�*;


total_lossH�@

error_R�X?

learning_rate_1��5�v�<I       6%�	t�Ze���A�*;


total_loss�;�@

error_R�B?

learning_rate_1��5���2I       6%�	��Ze���A�*;


total_loss=[�@

error_R�[?

learning_rate_1��5��޲I       6%�	[e���A�*;


total_lossd��@

error_R�Z?

learning_rate_1��5�fZEI       6%�	c[e���A�*;


total_lossC��@

error_R��P?

learning_rate_1��5�{��I       6%�	��[e���A�*;


total_lossL��@

error_R@EZ?

learning_rate_1��5ߋ��I       6%�	o�[e���A�*;


total_loss�Ne@

error_R��;?

learning_rate_1��5{gϜI       6%�	 8\e���A�*;


total_lossNU�@

error_RC�N?

learning_rate_1��5���pI       6%�	��\e���A�*;


total_loss&n�@

error_R��=?

learning_rate_1��5$eYI       6%�	��\e���A�*;


total_loss�C�@

error_R}�N?

learning_rate_1��5fKLI       6%�	�]e���A�*;


total_loss!�A

error_R�>V?

learning_rate_1��5kx�I       6%�	Ep]e���A�*;


total_lossWڼ@

error_Ri�A?

learning_rate_1��5V���I       6%�	�]e���A�*;


total_loss*��@

error_R-`W?

learning_rate_1��5:� dI       6%�	%&^e���A�*;


total_loss��@

error_RO\C?

learning_rate_1��5���nI       6%�	?z^e���A�*;


total_loss�VA

error_R��T?

learning_rate_1��5�ٰI       6%�	μ^e���A�*;


total_lossX��@

error_R�M?

learning_rate_1��58��sI       6%�	�_e���A�*;


total_loss�SA

error_RcU?

learning_rate_1��5|�I       6%�	JO_e���A�*;


total_lossV��@

error_R�dO?

learning_rate_1��5Y֨�I       6%�	9�_e���A�*;


total_loss���@

error_RmWH?

learning_rate_1��5�2#;I       6%�	��_e���A�*;


total_loss�%�@

error_R�U?

learning_rate_1��5��{I       6%�	�,`e���A�*;


total_loss�b�@

error_R�YM?

learning_rate_1��5Fc:YI       6%�	,r`e���A�*;


total_loss/˞@

error_R��X?

learning_rate_1��5��e�I       6%�	r�`e���A�*;


total_loss�A

error_R�G?

learning_rate_1��53���I       6%�	�ae���A�*;


total_loss�@

error_R< d?

learning_rate_1��5I�%0I       6%�	;[ae���A�*;


total_loss�C�@

error_RnE?

learning_rate_1��5���I       6%�	G�ae���A�*;


total_loss���@

error_Rs�R?

learning_rate_1��5}I[�I       6%�	��ae���A�*;


total_lossm�}@

error_RZG?

learning_rate_1��55X�I       6%�	�Abe���A�*;


total_lossq��@

error_Rj~/?

learning_rate_1��5�d��I       6%�	L�be���A�*;


total_loss��@

error_R�~K?

learning_rate_1��5�![I       6%�	j�be���A�*;


total_loss���@

error_R�3[?

learning_rate_1��5H��LI       6%�	<ce���A�*;


total_loss��@

error_R��E?

learning_rate_1��51
u�I       6%�	�Wce���A�*;


total_loss(`�@

error_Re�H?

learning_rate_1��5����I       6%�	��ce���A�*;


total_loss�ڣ@

error_R�?T?

learning_rate_1��5/jI       6%�	��ce���A�*;


total_losskq�@

error_R��U?

learning_rate_1��5�+jI       6%�	?8de���A�*;


total_lossz��@

error_R�<W?

learning_rate_1��5�PxI       6%�	�~de���A�*;


total_loss�_�@

error_R�cQ?

learning_rate_1��5��K�I       6%�	��de���A�*;


total_loss�0�@

error_R}�P?

learning_rate_1��5�>]I       6%�	Wee���A�*;


total_loss�@

error_RJJM?

learning_rate_1��5M)I       6%�	�he���A�*;


total_loss���@

error_RM�E?

learning_rate_1��5���{I       6%�	�khe���A�*;


total_loss��@

error_Ra�O?

learning_rate_1��5�ӷiI       6%�	��he���A�*;


total_loss}��@

error_RF-F?

learning_rate_1��5��vI       6%�	�ie���A�*;


total_loss���@

error_R LL?

learning_rate_1��5M~�I       6%�	Jie���A�*;


total_lossLF�@

error_R��A?

learning_rate_1��5���I       6%�	'�ie���A�*;


total_loss�e�@

error_R�Db?

learning_rate_1��5���SI       6%�	�ie���A�*;


total_loss��A

error_R�tV?

learning_rate_1��5��[�I       6%�	?je���A�*;


total_loss�Q�@

error_R�D?

learning_rate_1��5�,�I       6%�	�`je���A�*;


total_lossAF�@

error_Rf�W?

learning_rate_1��5ٷ��I       6%�	æje���A�*;


total_loss熑@

error_Rɠ=?

learning_rate_1��5��H�I       6%�	��je���A�*;


total_loss#��@

error_RvsS?

learning_rate_1��5)��I       6%�	�1ke���A�*;


total_loss���@

error_R_zT?

learning_rate_1��5�Y��I       6%�	:vke���A�*;


total_loss�b�@

error_R��H?

learning_rate_1��5T�+II       6%�	�ke���A�*;


total_loss?�@

error_R;?

learning_rate_1��5�!+�I       6%�	�ke���A�*;


total_lossT6�@

error_RF�P?

learning_rate_1��5��OtI       6%�	�Cle���A�*;


total_loss��@

error_R��G?

learning_rate_1��5���I       6%�	��le���A�*;


total_lossZӥ@

error_R��K?

learning_rate_1��5?ZI       6%�	��le���A�*;


total_loss�!A

error_R��_?

learning_rate_1��5I�ߤI       6%�	�me���A�*;


total_lossmq�@

error_Ri,P?

learning_rate_1��5c�,�I       6%�	Wme���A�*;


total_lossd/�@

error_R}J?

learning_rate_1��5W-�I       6%�	,�me���A�*;


total_loss�@

error_R��K?

learning_rate_1��5�㓤I       6%�	}�me���A�*;


total_lossڻ@

error_R;H?

learning_rate_1��5jZ��I       6%�	r>ne���A�*;


total_loss%�@

error_R��O?

learning_rate_1��5s�I       6%�	,�ne���A�*;


total_loss� A

error_R�Y?

learning_rate_1��5[̪I       6%�	�ne���A�*;


total_loss��@

error_Rm�S?

learning_rate_1��5rV�I       6%�	roe���A�*;


total_loss���@

error_R\�R?

learning_rate_1��50�I       6%�	3\oe���A�*;


total_loss��A

error_R�(U?

learning_rate_1��50&Q�I       6%�	M�oe���A�*;


total_lossV�@

error_R�vI?

learning_rate_1��5����I       6%�	��oe���A�*;


total_lossP�@

error_R�zL?

learning_rate_1��5}�euI       6%�	�2pe���A�*;


total_loss�@

error_R�_O?

learning_rate_1��5�݀I       6%�	�upe���A�*;


total_loss��@

error_Rd�N?

learning_rate_1��52���I       6%�	��pe���A�*;


total_loss��@

error_RH*N?

learning_rate_1��5%g�4I       6%�	�qe���A�*;


total_loss���@

error_R�m?

learning_rate_1��5�:E�I       6%�	�Kqe���A�*;


total_lossc��@

error_R�:[?

learning_rate_1��5{V�I       6%�	q�qe���A�*;


total_loss���@

error_R7�V?

learning_rate_1��5�RiRI       6%�	R�qe���A�*;


total_loss5�@

error_R�T?

learning_rate_1��58xkI       6%�	@)re���A�*;


total_loss
�A

error_R_>H?

learning_rate_1��5F���I       6%�	sre���A�*;


total_loss���@

error_RN�T?

learning_rate_1��5R�Q�I       6%�	��re���A�*;


total_loss{��@

error_RD.E?

learning_rate_1��5l��<I       6%�	2se���A�*;


total_loss�/�@

error_RR@?

learning_rate_1��5`D��I       6%�	��se���A�*;


total_lossO��@

error_R�N?

learning_rate_1��5�ip9I       6%�	�se���A�*;


total_loss��@

error_R�'G?

learning_rate_1��5��NBI       6%�	�te���A�*;


total_lossV��@

error_RϛJ?

learning_rate_1��5#p�I       6%�	1hte���A�*;


total_loss:�@

error_R�R?

learning_rate_1��5��hI       6%�	��te���A�*;


total_losss3�@

error_R��I?

learning_rate_1��5��I       6%�	Sue���A�*;


total_loss:��@

error_R,va?

learning_rate_1��5�³I       6%�	u`ue���A�*;


total_loss���@

error_Ri\L?

learning_rate_1��5��ЊI       6%�	ίue���A�*;


total_loss�e�@

error_R�H?

learning_rate_1��5;[F4I       6%�	�ve���A�*;


total_lossXM�@

error_R͸E?

learning_rate_1��56�I       6%�	�Sve���A�*;


total_loss�A

error_R)�Q?

learning_rate_1��5aC=]I       6%�	A�ve���A�*;


total_loss��@

error_R�H?

learning_rate_1��5�TnI       6%�	��ve���A�*;


total_lossoY�@

error_R|tQ?

learning_rate_1��5�U��I       6%�	�!we���A�*;


total_loss���@

error_R,�;?

learning_rate_1��5�Ί�I       6%�	ggwe���A�*;


total_loss ��@

error_R{�\?

learning_rate_1��5r��I       6%�	n�we���A�*;


total_loss[>�@

error_R�XG?

learning_rate_1��5o�C�I       6%�	J�we���A�*;


total_loss��A

error_R��F?

learning_rate_1��5�8^}I       6%�	Mxe���A�*;


total_loss��@

error_Rm�D?

learning_rate_1��5U (I       6%�	�xe���A�*;


total_loss��A

error_R�(S?

learning_rate_1��5�I"I       6%�	�xe���A�*;


total_lossaA

error_R��H?

learning_rate_1��5
,X�I       6%�	jFye���A�*;


total_loss[��@

error_R��\?

learning_rate_1��5��=�I       6%�	p�ye���A�*;


total_lossN��@

error_R&�T?

learning_rate_1��5�#=I       6%�	��ye���A�*;


total_loss���@

error_R�#W?

learning_rate_1��5�k;>I       6%�	ze���A�*;


total_loss�o�@

error_R�#B?

learning_rate_1��5��I       6%�	�Xze���A�*;


total_loss��@

error_R��Y?

learning_rate_1��5��I       6%�	a�ze���A�*;


total_loss�L�@

error_Rx�R?

learning_rate_1��5�
H#I       6%�	��ze���A�*;


total_loss;�@

error_R��N?

learning_rate_1��5`�N|I       6%�	�2{e���A�*;


total_loss�M�@

error_R)�Q?

learning_rate_1��5��G�I       6%�	cv{e���A�*;


total_lossxq�@

error_R�WG?

learning_rate_1��5^�HI       6%�	��{e���A�*;


total_loss��@

error_RXKO?

learning_rate_1��5X��I       6%�	|e���A�*;


total_loss��@

error_R��L?

learning_rate_1��5�Z�/I       6%�	�M|e���A�*;


total_lossE�@

error_R[?

learning_rate_1��5����I       6%�	��|e���A�*;


total_loss��@

error_RAJF?

learning_rate_1��5���I       6%�	��|e���A�*;


total_loss��@

error_R�9B?

learning_rate_1��5'�RI       6%�	�}e���A�*;


total_lossﰳ@

error_R�X?

learning_rate_1��5�jMgI       6%�	j}e���A�*;


total_loss4ӗ@

error_R_Q?

learning_rate_1��5�Y�.I       6%�	4�}e���A�*;


total_lossܸ�@

error_R`ae?

learning_rate_1��5�i��I       6%�	~e���A�*;


total_loss��.A

error_R�`?

learning_rate_1��5Wa��I       6%�	=S~e���A�*;


total_loss�bA

error_Rn*?

learning_rate_1��5N%�cI       6%�	�~e���A�*;


total_loss���@

error_RƑZ?

learning_rate_1��5
ЮSI       6%�	�!e���A�*;


total_loss]�@

error_RR?

learning_rate_1��5y��nI       6%�	�ie���A�*;


total_loss�sA

error_R�lR?

learning_rate_1��5"?�tI       6%�	�e���A�*;


total_lossڥ�@

error_R��h?

learning_rate_1��5�L��I       6%�	e-�e���A�*;


total_loss�A

error_R�gR?

learning_rate_1��5u��I       6%�	y|�e���A�*;


total_loss��@

error_R �T?

learning_rate_1��5NOn�I       6%�	mǀe���A�*;


total_loss���@

error_Ri�[?

learning_rate_1��57ȫ@I       6%�	K3�e���A�*;


total_loss�k�@

error_R�UT?

learning_rate_1��5#�-�I       6%�	�y�e���A�*;


total_loss��@

error_R$L?

learning_rate_1��5��4yI       6%�	r��e���A�*;


total_lossLu�@

error_R��^?

learning_rate_1��5�o�JI       6%�	��e���A�*;


total_loss�6�@

error_R=;b?

learning_rate_1��5	���I       6%�	�Q�e���A�*;


total_loss���@

error_RHEE?

learning_rate_1��5��I       6%�	��e���A�*;


total_lossT_�@

error_R �D?

learning_rate_1��5�,90I       6%�	�ނe���A�*;


total_loss���@

error_Ri^??

learning_rate_1��5�G_�I       6%�	�$�e���A�*;


total_lossz"�@

error_R��K?

learning_rate_1��5_?�I       6%�	�h�e���A�*;


total_loss�?�@

error_R�@I?

learning_rate_1��5�s�I       6%�	w��e���A�*;


total_loss"�A

error_RW�D?

learning_rate_1��5�oQ�I       6%�	�e���A�*;


total_loss��@

error_Rxs??

learning_rate_1��5�7��I       6%�	�4�e���A�*;


total_lossEV�@

error_RO?

learning_rate_1��5�.��I       6%�	�~�e���A�*;


total_loss��@

error_R�/J?

learning_rate_1��5�w�&I       6%�	�Ǆe���A�*;


total_loss��A

error_R?TG?

learning_rate_1��5�-I       6%�	}�e���A�*;


total_lossS�@

error_R_�F?

learning_rate_1��5�OpI       6%�	�W�e���A�*;


total_lossr�@

error_R�OZ?

learning_rate_1��5.`2I       6%�	��e���A�*;


total_loss�~�@

error_R>B?

learning_rate_1��5����I       6%�	��e���A�*;


total_loss��@

error_R֘T?

learning_rate_1��5���I       6%�	d2�e���A�*;


total_loss��@

error_R!�D?

learning_rate_1��5����I       6%�	5v�e���A�*;


total_lossd&�@

error_RfS?

learning_rate_1��5�[��I       6%�	蹆e���A�*;


total_loss��@

error_R��Y?

learning_rate_1��5Iܡ�I       6%�	���e���A�*;


total_loss丸@

error_R�@?

learning_rate_1��5��x�I       6%�	�B�e���A�*;


total_lossۛ�@

error_R�'S?

learning_rate_1��5�<�8I       6%�	Έ�e���A�*;


total_loss�l�@

error_R��<?

learning_rate_1��5b~I       6%�	t·e���A�*;


total_loss%k�@

error_R8�T?

learning_rate_1��5�p��I       6%�	��e���A�*;


total_loss%��@

error_R�w6?

learning_rate_1��55V�!I       6%�	WZ�e���A�*;


total_lossO��@

error_R�<d?

learning_rate_1��5�LiI       6%�	t��e���A�*;


total_lossWƤ@

error_R4S?

learning_rate_1��5
��RI       6%�	��e���A�*;


total_loss��@

error_RR$H?

learning_rate_1��5�@��I       6%�	�)�e���A�*;


total_loss�'�@

error_R�Y?

learning_rate_1��5zVsI       6%�	�n�e���A�*;


total_loss�6�@

error_RC-Q?

learning_rate_1��5�DfI       6%�	��e���A�*;


total_loss�|�@

error_Rs�Y?

learning_rate_1��5B5VI       6%�	0��e���A�*;


total_loss�f�@

error_R��F?

learning_rate_1��5sQ<�I       6%�	�>�e���A�*;


total_loss��@

error_R;�;?

learning_rate_1��5Y5�I       6%�	|��e���A�*;


total_losst7�@

error_Ri:L?

learning_rate_1��5�I       6%�	|Ɋe���A�*;


total_lossjg�@

error_R�CT?

learning_rate_1��5�,�HI       6%�	��e���A�*;


total_lossj֝@

error_R�K?

learning_rate_1��5��GI       6%�	U�e���A�*;


total_lossf��@

error_Rq�A?

learning_rate_1��5��K�I       6%�	���e���A�*;


total_loss�X�@

error_R�1M?

learning_rate_1��5Igv�I       6%�	o�e���A�*;


total_loss� �@

error_R�!:?

learning_rate_1��5f,EI       6%�	�3�e���A�*;


total_loss7��@

error_R��Z?

learning_rate_1��5��q�I       6%�	@|�e���A�*;


total_loss]I�@

error_R��U?

learning_rate_1��51��I       6%�	3��e���A�*;


total_loss�;c@

error_R_�@?

learning_rate_1��5�j?I       6%�	O �e���A�*;


total_loss���@

error_R�=H?

learning_rate_1��5A�2I       6%�	mF�e���A�*;


total_lossɮ�@

error_R��S?

learning_rate_1��5<8�I       6%�	s��e���A�*;


total_loss ��@

error_R�^D?

learning_rate_1��5�Gj�I       6%�	F�e���A�*;


total_loss��@

error_R�>?

learning_rate_1��5~��I       6%�	2�e���A�*;


total_lossڇd@

error_Rj�;?

learning_rate_1��5���I       6%�	j~�e���A�*;


total_loss� �@

error_R.)V?

learning_rate_1��5CvjI       6%�	�Îe���A�*;


total_loss堕@

error_R��q?

learning_rate_1��5�{5�I       6%�	<�e���A�*;


total_loss��@

error_R|�S?

learning_rate_1��5/���I       6%�	aW�e���A�*;


total_loss�߲@

error_R�ET?

learning_rate_1��5˺``I       6%�	m��e���A�*;


total_loss��@

error_R�A?

learning_rate_1��5'˟�I       6%�	܏e���A�*;


total_loss�A

error_R�N?

learning_rate_1��5���I       6%�	��e���A�*;


total_loss<͜@

error_R�L?

learning_rate_1��5�qwI       6%�	�j�e���A�*;


total_lossݶ@

error_R�4Q?

learning_rate_1��5���I       6%�	β�e���A�*;


total_loss���@

error_R��P?

learning_rate_1��5%\�I       6%�	D��e���A�*;


total_losse�A

error_R��a?

learning_rate_1��5C�;`I       6%�	�F�e���A�*;


total_loss�¡@

error_R��l?

learning_rate_1��5$�@�I       6%�	Ќ�e���A�*;


total_lossd��@

error_R�VZ?

learning_rate_1��5����I       6%�	�Ցe���A�*;


total_loss	"�@

error_R$�J?

learning_rate_1��5h׮�I       6%�	8�e���A�*;


total_loss���@

error_R��A?

learning_rate_1��55�&�I       6%�	�i�e���A�*;


total_loss���@

error_R3ul?

learning_rate_1��5
"�I       6%�	���e���A�*;


total_loss�ؒ@

error_R�U?

learning_rate_1��5�� �I       6%�	��e���A�*;


total_lossiR�@

error_R a?

learning_rate_1��5�j�I       6%�	>5�e���A�*;


total_loss�&�@

error_R�L?

learning_rate_1��5^�?�I       6%�	�z�e���A�*;


total_lossqλ@

error_R[5Z?

learning_rate_1��5���I       6%�	��e���A�*;


total_loss���@

error_R��H?

learning_rate_1��5���I       6%�	t�e���A�*;


total_lossl�@

error_R:�l?

learning_rate_1��5�^�*I       6%�	�F�e���A�*;


total_loss���@

error_Rh�]?

learning_rate_1��5\��I       6%�	;��e���A�*;


total_loss���@

error_RRL?

learning_rate_1��5��QiI       6%�	fڔe���A�*;


total_loss%j�@

error_R��D?

learning_rate_1��5�~��I       6%�	� �e���A�*;


total_loss�`�@

error_R,�7?

learning_rate_1��5�*]�I       6%�	)m�e���A�*;


total_loss� �@

error_R�4H?

learning_rate_1��59_��I       6%�	��e���A�*;


total_loss�'�@

error_R�e?

learning_rate_1��5H%��I       6%�	�e���A�*;


total_loss���@

error_R>E?

learning_rate_1��5[Q+I       6%�	�F�e���A�*;


total_lossk��@

error_RvB6?

learning_rate_1��5(��I       6%�	I��e���A�*;


total_loss:�@

error_R�\?

learning_rate_1��5�?I       6%�	Ζe���A�*;


total_loss=(�@

error_R�*S?

learning_rate_1��5�4�"I       6%�	�e���A�*;


total_loss��@

error_R��U?

learning_rate_1��5�dKI       6%�	�U�e���A�*;


total_loss��@

error_R��N?

learning_rate_1��5Hk�I       6%�	L��e���A�*;


total_loss{�@

error_R��a?

learning_rate_1��5�தI       6%�	ߗe���A�*;


total_lossDk A

error_Rp4?

learning_rate_1��5���zI       6%�	�'�e���A�*;


total_lossDo�@

error_R�WT?

learning_rate_1��5��Y�I       6%�	r�e���A�*;


total_loss�RA

error_Rf-S?

learning_rate_1��5�[��I       6%�	缘e���A�*;


total_loss��@

error_RԨQ?

learning_rate_1��5�PW�I       6%�	�e���A�*;


total_lossT^�@

error_R!QS?

learning_rate_1��5M�I       6%�	�Q�e���A�*;


total_lossp�@

error_Rs,H?

learning_rate_1��5�T�I       6%�	.��e���A�*;


total_loss��I@

error_RڨM?

learning_rate_1��5v�^�I       6%�	��e���A�*;


total_loss��@

error_R�/W?

learning_rate_1��5ԁuI       6%�	8+�e���A�*;


total_loss\��@

error_R��@?

learning_rate_1��5�SI       6%�	�p�e���A�*;


total_loss��x@

error_R��G?

learning_rate_1��5&���I       6%�	ܳ�e���A�*;


total_loss���@

error_R}�A?

learning_rate_1��5���I       6%�	���e���A�*;


total_loss��@

error_R'G?

learning_rate_1��5"^I       6%�	D;�e���A�*;


total_loss���@

error_R@C?

learning_rate_1��5L��GI       6%�	߂�e���A�*;


total_lossh��@

error_Rq�M?

learning_rate_1��5����I       6%�	}Ûe���A�*;


total_loss���@

error_R�F=?

learning_rate_1��5�48�I       6%�	��e���A�*;


total_loss\K�@

error_R�RI?

learning_rate_1��5ԫ}�I       6%�	0O�e���A�*;


total_loss@

error_R#R?

learning_rate_1��5��Y:I       6%�	"��e���A�*;


total_loss̿�@

error_Rh�N?

learning_rate_1��5�Y��I       6%�	�ڜe���A�*;


total_loss�M�@

error_R8L?

learning_rate_1��5�bI       6%�	w�e���A�*;


total_loss�/A

error_R�wS?

learning_rate_1��5F.�/I       6%�	Fj�e���A�*;


total_loss�A

error_R8N?

learning_rate_1��5�ƎfI       6%�	�ʝe���A�*;


total_loss,�@

error_RFZU?

learning_rate_1��5�n�:I       6%�	��e���A�*;


total_loss���@

error_R�+D?

learning_rate_1��5c��I       6%�	X�e���A�*;


total_loss��@

error_RO�N?

learning_rate_1��5����I       6%�	��e���A�*;


total_loss�ya@

error_R,8??

learning_rate_1��5�qK�I       6%�	��e���A�*;


total_loss.�@

error_Rz�:?

learning_rate_1��5���TI       6%�	�-�e���A�*;


total_loss���@

error_R�'\?

learning_rate_1��5b��I       6%�	�v�e���A�*;


total_loss��~@

error_Rz,W?

learning_rate_1��53��I       6%�	m��e���A�*;


total_loss_i�@

error_R�T?

learning_rate_1��5���gI       6%�	��e���A�*;


total_loss�x�@

error_Rh�@?

learning_rate_1��5
�I       6%�	�S�e���A�*;


total_loss�c�@

error_R)�=?

learning_rate_1��5'�I       6%�	&��e���A�*;


total_loss
j@

error_R8�M?

learning_rate_1��5�fI       6%�	�e���A�*;


total_loss���@

error_R�_Q?

learning_rate_1��5�7 iI       6%�	 V�e���A�*;


total_loss��@

error_R(�S?

learning_rate_1��5��M�I       6%�	ߜ�e���A�*;


total_lossr��@

error_RW�6?

learning_rate_1��5ɑO�I       6%�	���e���A�*;


total_lossTR�@

error_R��S?

learning_rate_1��5Hmd�I       6%�	D�e���A�*;


total_loss�n�@

error_R�SB?

learning_rate_1��5���I       6%�	���e���A�*;


total_loss$z�@

error_R�^?

learning_rate_1��5�_׈I       6%�	U̢e���A�*;


total_lossȦ�@

error_R��N?

learning_rate_1��5 ��I       6%�	��e���A�*;


total_loss�#�@

error_RseF?

learning_rate_1��5�U��I       6%�	�]�e���A�*;


total_lossT<|@

error_R�zV?

learning_rate_1��5���CI       6%�	��e���A�*;


total_loss[��@

error_R��N?

learning_rate_1��5F{�I       6%�	���e���A�*;


total_loss�ů@

error_R
�M?

learning_rate_1��5ⓛ�I       6%�	�9�e���A�*;


total_losssl�@

error_R�'?

learning_rate_1��59?�I       6%�	��e���A�*;


total_loss���@

error_R R?

learning_rate_1��5��%I       6%�	�Ƥe���A�*;


total_loss��@

error_R�XN?

learning_rate_1��5#���I       6%�	g�e���A�*;


total_lossr�x@

error_R��G?

learning_rate_1��5�עPI       6%�	�Z�e���A�*;


total_loss?Ī@

error_R�X?

learning_rate_1��5�<�I       6%�	ؤ�e���A�*;


total_lossJ�A

error_R:�K?

learning_rate_1��5�%�*I       6%�	��e���A�*;


total_loss#�@

error_R�>?

learning_rate_1��5-�I       6%�	=�e���A�*;


total_loss&�@

error_R�N?

learning_rate_1��5r9m�I       6%�	���e���A�*;


total_loss���@

error_R�~P?

learning_rate_1��5j*�DI       6%�	�ͦe���A�*;


total_lossd#�@

error_R�[?

learning_rate_1��5�V�I       6%�	��e���A�*;


total_loss|�A

error_R�|L?

learning_rate_1��5JJ3�I       6%�	<Z�e���A�*;


total_loss@h�@

error_R)�I?

learning_rate_1��5:��~I       6%�	��e���A�*;


total_loss�@

error_ReFO?

learning_rate_1��51�CbI       6%�	ߧe���A�*;


total_loss�!�@

error_R�??

learning_rate_1��5��p�I       6%�	� �e���A�*;


total_lossQ�@

error_RR�\?

learning_rate_1��5���I       6%�	�c�e���A�*;


total_loss-�@

error_Rd�N?

learning_rate_1��5�,_I       6%�	a��e���A�*;


total_loss�9�@

error_R1�H?

learning_rate_1��5#��I       6%�	��e���A�*;


total_loss���@

error_R��D?

learning_rate_1��5���I       6%�	`2�e���A�*;


total_lossFN�@

error_RN�N?

learning_rate_1��5��(fI       6%�	Gu�e���A�*;


total_lossZ��@

error_RM*E?

learning_rate_1��5�#�I       6%�	��e���A�*;


total_losshˆ@

error_R�Q?

learning_rate_1��5�#,\I       6%�	���e���A�*;


total_lossT2A

error_RM�W?

learning_rate_1��5�X�I       6%�	�C�e���A�*;


total_loss��@

error_R�K?

learning_rate_1��5N;I       6%�	���e���A�*;


total_loss�ν@

error_R��Z?

learning_rate_1��5���I       6%�	'ͪe���A�*;


total_loss��@

error_R��R?

learning_rate_1��5|I       6%�	��e���A�*;


total_lossl�A

error_R�RT?

learning_rate_1��5��I       6%�	pS�e���A�*;


total_loss3��@

error_RϜ`?

learning_rate_1��5{���I       6%�	���e���A�*;


total_loss�'�@

error_Ri"Z?

learning_rate_1��5�1)�I       6%�	8�e���A�*;


total_lossø�@

error_RV�G?

learning_rate_1��5,��I       6%�	i'�e���A�*;


total_loss�Q�@

error_R�C?

learning_rate_1��5OX�I       6%�	#q�e���A�*;


total_loss��@

error_R�E?

learning_rate_1��5�{!@I       6%�	���e���A�*;


total_loss1H�@

error_REWb?

learning_rate_1��5����I       6%�	P��e���A�*;


total_lossm��@

error_R�k?

learning_rate_1��5���SI       6%�	�<�e���A�*;


total_loss��@

error_R��N?

learning_rate_1��5�;-I       6%�	T��e���A�*;


total_loss�q~@

error_R��Z?

learning_rate_1��5E@�OI       6%�	(�e���A�*;


total_loss��@

error_RH/H?

learning_rate_1��5И��I       6%�	�@�e���A�*;


total_loss��@

error_R�RL?

learning_rate_1��5=~(I       6%�	��e���A�*;


total_loss$�X@

error_R3�S?

learning_rate_1��5��@I       6%�	.Ԯe���A�*;


total_lossn��@

error_R�tX?

learning_rate_1��53c�I       6%�	��e���A�*;


total_loss���@

error_RIc?

learning_rate_1��5���I       6%�	�`�e���A�*;


total_loss�|�@

error_R��O?

learning_rate_1��5(���I       6%�	0��e���A�*;


total_loss{��@

error_R�J?

learning_rate_1��5��GI       6%�	�e���A�*;


total_loss�l�@

error_R��L?

learning_rate_1��5�7�II       6%�	�)�e���A�*;


total_loss��@

error_R��=?

learning_rate_1��5q���I       6%�	�k�e���A�*;


total_loss��@

error_R`�_?

learning_rate_1��5�h�YI       6%�	Q��e���A�*;


total_loss�O�@

error_R`�O?

learning_rate_1��5%�BI       6%�	`�e���A�*;


total_lossL��@

error_R�j?

learning_rate_1��5>f�I       6%�	L0�e���A�*;


total_loss��@

error_R�Q?

learning_rate_1��5�ݚ�I       6%�	�s�e���A�*;


total_lossA#�@

error_R�KL?

learning_rate_1��5M�]�I       6%�	s��e���A�*;


total_lossm��@

error_R4�J?

learning_rate_1��5-�:I       6%�	���e���A�*;


total_loss)W�@

error_R��g?

learning_rate_1��5�#G3I       6%�	�B�e���A�*;


total_lossO.�@

error_R�5?

learning_rate_1��5\,cEI       6%�	���e���A�*;


total_loss��@

error_RdW[?

learning_rate_1��5�
��I       6%�	Xʲe���A�*;


total_lossi�A

error_R�zN?

learning_rate_1��5¦NI       6%�	��e���A�*;


total_loss �@

error_R�I?

learning_rate_1��5���I       6%�	SZ�e���A�*;


total_loss�@

error_R�P?

learning_rate_1��5�B�qI       6%�	X��e���A�*;


total_lossLQ�@

error_RCa?

learning_rate_1��5�I       6%�	�e���A�*;


total_losst��@

error_RcU?

learning_rate_1��5�29&I       6%�	�2�e���A�*;


total_loss���@

error_R�k@?

learning_rate_1��5����I       6%�	�x�e���A�*;


total_loss|�@

error_Rn�M?

learning_rate_1��5t�]I       6%�	���e���A�*;


total_loss3�@

error_RTEF?

learning_rate_1��5��qI       6%�	��e���A�*;


total_loss�}�@

error_R�8G?

learning_rate_1��5��'pI       6%�	7N�e���A�*;


total_loss�t�@

error_RM�a?

learning_rate_1��5)��EI       6%�	8��e���A�*;


total_lossI��@

error_R$aF?

learning_rate_1��5�u;�I       6%�	Sֵe���A�*;


total_loss;2�@

error_R�\_?

learning_rate_1��5���I       6%�	��e���A�*;


total_loss�@

error_R�nJ?

learning_rate_1��5��`�I       6%�	�d�e���A�*;


total_lossR��@

error_R�7?

learning_rate_1��5��	MI       6%�	b��e���A�*;


total_loss)�@

error_R�%S?

learning_rate_1��5UU9I       6%�	S�e���A�*;


total_lossQ)�@

error_RW�S?

learning_rate_1��5R.�KI       6%�	*�e���A�*;


total_loss�%�@

error_R�N?

learning_rate_1��59G�I       6%�	q�e���A�*;


total_lossJ�@

error_R��C?

learning_rate_1��5|f�I       6%�	*��e���A�*;


total_loss7�A

error_R�$U?

learning_rate_1��5*���I       6%�	\�e���A�*;


total_loss +�@

error_R��K?

learning_rate_1��5�[P�I       6%�	�M�e���A�*;


total_loss���@

error_RH!J?

learning_rate_1��5P�kI       6%�	���e���A�*;


total_loss�A

error_R&�M?

learning_rate_1��5����I       6%�	��e���A�*;


total_loss��@

error_Rd�F?

learning_rate_1��5P���I       6%�	%6�e���A�*;


total_lossV��@

error_R�O?

learning_rate_1��5�:�I       6%�	w}�e���A�*;


total_loss�ۆ@

error_R6�J?

learning_rate_1��5H&yI       6%�	ùe���A�*;


total_loss�G�@

error_R��M?

learning_rate_1��52�w�I       6%�	>�e���A�*;


total_loss��@

error_R&[a?

learning_rate_1��5�e1I       6%�	�M�e���A�*;


total_loss6��@

error_Rf�L?

learning_rate_1��5��BI       6%�	���e���A�*;


total_loss��@

error_Rz;I?

learning_rate_1��5닇I       6%�	CԺe���A�*;


total_loss7��@

error_R��L?

learning_rate_1��5;�+I       6%�	$�e���A�*;


total_loss�ʣ@

error_REB?

learning_rate_1��5w.�I       6%�	%Y�e���A�*;


total_loss���@

error_RSaT?

learning_rate_1��55�
3I       6%�	-��e���A�*;


total_lossM��@

error_R]�P?

learning_rate_1��5ƊɲI       6%�	�߻e���A�*;


total_loss��@

error_R��K?

learning_rate_1��5�A��I       6%�	i"�e���A�*;


total_loss�"�@

error_R	W[?

learning_rate_1��5'��>I       6%�	�e�e���A�*;


total_loss���@

error_R��S?

learning_rate_1��5s��I       6%�	1��e���A�*;


total_loss�x�@

error_R�:E?

learning_rate_1��5���I       6%�	���e���A�*;


total_loss�ǽ@

error_R�+I?

learning_rate_1��5��|I       6%�	�H�e���A�*;


total_losseM�@

error_R�9:?

learning_rate_1��5���I       6%�	֮�e���A�*;


total_lossʅ�@

error_R�m\?

learning_rate_1��5�9k�I       6%�	���e���A�*;


total_lossm�A

error_R6??

learning_rate_1��5���I       6%�	J?�e���A�*;


total_loss�Ì@

error_RiTJ?

learning_rate_1��5�d��I       6%�	
��e���A�*;


total_loss)A

error_RMKL?

learning_rate_1��5���I       6%�	�޾e���A�*;


total_loss`��@

error_R�H?

learning_rate_1��5\��'I       6%�	�)�e���A�*;


total_loss��@

error_R��X?

learning_rate_1��54z�&I       6%�	�p�e���A�*;


total_loss-eA

error_RÑH?

learning_rate_1��5�b�I       6%�	E��e���A�*;


total_loss�F�@

error_R�V[?

learning_rate_1��5L.��I       6%�	���e���A�*;


total_loss�H�@

error_R�G?

learning_rate_1��5��3�I       6%�	�=�e���A�*;


total_loss%�@

error_R	�O?

learning_rate_1��5 ���I       6%�	���e���A�*;


total_lossz�@

error_R*R?

learning_rate_1��5Kt��I       6%�	���e���A�*;


total_loss���@

error_R��[?

learning_rate_1��5«�bI       6%�	�$�e���A�*;


total_loss�
A

error_R�^[?

learning_rate_1��5� }VI       6%�	k�e���A�*;


total_loss�L�@

error_R�PX?

learning_rate_1��5PK3^I       6%�	��e���A�*;


total_loss�t�@

error_R�(T?

learning_rate_1��5A�s�I       6%�	��e���A�*;


total_loss���@

error_R�Q?

learning_rate_1��5�K=I       6%�	�X�e���A�*;


total_loss�@

error_R%
O?

learning_rate_1��5����I       6%�	���e���A�*;


total_loss���@

error_R��[?

learning_rate_1��5�*�iI       6%�	-��e���A�*;


total_lossx��@

error_R��H?

learning_rate_1��5*H�jI       6%�	�B�e���A�*;


total_loss��@

error_RH?

learning_rate_1��5&�
�I       6%�	j��e���A�*;


total_loss�4�@

error_R�M?

learning_rate_1��5���I       6%�	#��e���A�*;


total_losssӬ@

error_RȧS?

learning_rate_1��5n��I       6%�	�%�e���A�*;


total_loss���@

error_R��N?

learning_rate_1��5���|I       6%�	5s�e���A�*;


total_loss�g�@

error_R]�@?

learning_rate_1��5'�18I       6%�	���e���A�*;


total_lossɧ�@

error_R,�R?

learning_rate_1��5��U�I       6%�	��e���A�*;


total_lossd�@

error_RW&Z?

learning_rate_1��5���I       6%�	g_�e���A�*;


total_loss\��@

error_R1�T?

learning_rate_1��5�i�I       6%�	���e���A�*;


total_lossÍ�@

error_R� V?

learning_rate_1��5բ�I       6%�	5�e���A�*;


total_loss�m�@

error_Rd�S?

learning_rate_1��5W�U�I       6%�	W[�e���A�*;


total_loss�B�@

error_R�=X?

learning_rate_1��5�<[I       6%�	���e���A�*;


total_loss�l�@

error_RR�S?

learning_rate_1��5+R�I       6%�	���e���A�*;


total_loss��@

error_R��D?

learning_rate_1��5Qi*�I       6%�	�t�e���A�*;


total_loss���@

error_R�G?

learning_rate_1��5���@I       6%�	`��e���A�*;


total_loss�S�@

error_RI�N?

learning_rate_1��5��	�I       6%�	��e���A�*;


total_loss���@

error_R�1:?

learning_rate_1��5��	I       6%�	�S�e���A�*;


total_loss���@

error_RQE?

learning_rate_1��5v���I       6%�	���e���A�*;


total_loss���@

error_R�J?

learning_rate_1��5�7`I       6%�	���e���A�*;


total_loss��@

error_RZ�>?

learning_rate_1��5���I       6%�	�$�e���A�*;


total_loss���@

error_R)mE?

learning_rate_1��5�Lb�I       6%�	i�e���A�*;


total_loss�>A

error_R�&I?

learning_rate_1��5}_6&I       6%�	��e���A�*;


total_loss�M�@

error_R@.M?

learning_rate_1��5�ֲ�I       6%�	���e���A�*;


total_loss�m�@

error_R�K?

learning_rate_1��58��WI       6%�	L>�e���A�*;


total_loss��JA

error_R�jQ?

learning_rate_1��5���I       6%�	���e���A�*;


total_loss<q�@

error_R�I?

learning_rate_1��5�1
6I       6%�	:��e���A�*;


total_loss�	A

error_R��>?

learning_rate_1��5��
I       6%�	:�e���A�*;


total_lossH߷@

error_R�3I?

learning_rate_1��5L$I       6%�	zZ�e���A�*;


total_loss�A�@

error_R)_?

learning_rate_1��5!��I       6%�	���e���A�*;


total_loss๯@

error_R�7H?

learning_rate_1��5����I       6%�	n��e���A�*;


total_lossEh�@

error_R��U?

learning_rate_1��5���I       6%�	�'�e���A�*;


total_loss(��@

error_R�K?

learning_rate_1��5 �Z�I       6%�	+o�e���A�*;


total_lossI̡@

error_R8�7?

learning_rate_1��5��˔I       6%�	���e���A�*;


total_loss�@

error_R!&L?

learning_rate_1��5����I       6%�	/	�e���A�*;


total_loss�x�@

error_R.�X?

learning_rate_1��5�� �I       6%�	:V�e���A�*;


total_loss}N A

error_R�SU?

learning_rate_1��5{H��I       6%�	��e���A�*;


total_losszS�@

error_R��K?

learning_rate_1��5�XI       6%�	�	�e���A�*;


total_lossԏ�@

error_R�D?

learning_rate_1��5z ��I       6%�	,M�e���A�*;


total_loss���@

error_RMB?

learning_rate_1��5~×I       6%�	ܒ�e���A�*;


total_loss.��@

error_R�SG?

learning_rate_1��5v�Y�I       6%�	L��e���A�*;


total_loss�@

error_R�kA?

learning_rate_1��5���I       6%�	N�e���A�*;


total_loss�T�@

error_R��:?

learning_rate_1��5�4��I       6%�	ac�e���A�*;


total_loss휥@

error_ROWG?

learning_rate_1��5�'Z�I       6%�	ئ�e���A�*;


total_loss18�@

error_R��:?

learning_rate_1��5�Y�I       6%�	���e���A�*;


total_loss��@

error_RJ�\?

learning_rate_1��5pc��I       6%�	�/�e���A�*;


total_loss8�@

error_R`�;?

learning_rate_1��5�Q�I       6%�	�}�e���A�*;


total_loss�N�@

error_R\-R?

learning_rate_1��5���I       6%�	G��e���A�*;


total_loss��@

error_R&�M?

learning_rate_1��5G�]I       6%�	}�e���A�*;


total_lossEr�@

error_R_�N?

learning_rate_1��59��I       6%�	m`�e���A�*;


total_losst%�@

error_R KT?

learning_rate_1��5]{��I       6%�	ת�e���A�*;


total_loss�#�@

error_R�rE?

learning_rate_1��5�P�I       6%�	u��e���A�*;


total_loss���@

error_R;�??

learning_rate_1��5y�scI       6%�	�?�e���A�*;


total_loss���@

error_R&�A?

learning_rate_1��5�d-�I       6%�	u��e���A�*;


total_loss���@

error_Rq�=?

learning_rate_1��5�3>
I       6%�		��e���A�*;


total_loss�:�@

error_R{�B?

learning_rate_1��5O �9I       6%�	A�e���A�*;


total_lossQ�@

error_R��T?

learning_rate_1��5�e�hI       6%�	W�e���A�*;


total_loss-��@

error_R� i?

learning_rate_1��5k}@I       6%�	��e���A�*;


total_loss.Ϫ@

error_R�R?

learning_rate_1��5�PL\I       6%�	`��e���A�*;


total_lossm��@

error_R�2V?

learning_rate_1��59j�AI       6%�	�+�e���A�*;


total_lossc�@

error_R@O?

learning_rate_1��5ISjI       6%�	T{�e���A�*;


total_loss$�@

error_R�XB?

learning_rate_1��5ܢ�I       6%�	��e���A�*;


total_loss�A

error_R��L?

learning_rate_1��5���qI       6%�	��e���A�*;


total_loss3\ A

error_R�gN?

learning_rate_1��5!���I       6%�	�\�e���A�*;


total_losst�@

error_R�N?

learning_rate_1��5k�3�I       6%�	[��e���A�*;


total_loss���@

error_R�WL?

learning_rate_1��5�k}�I       6%�	���e���A�*;


total_lossc�@

error_R6�H?

learning_rate_1��5CW�eI       6%�	9�e���A�*;


total_loss/R�@

error_RJR?

learning_rate_1��5�UI       6%�	A}�e���A�*;


total_loss�=�@

error_RɇY?

learning_rate_1��5��7�I       6%�	D��e���A�*;


total_loss�"�@

error_R6UU?

learning_rate_1��5]Ql~I       6%�	��e���A�*;


total_lossÝ�@

error_RtHQ?

learning_rate_1��5;�I       6%�	�B�e���A�*;


total_loss�X�@

error_R��N?

learning_rate_1��5U�EI       6%�	'��e���A�*;


total_loss~�@

error_R �O?

learning_rate_1��5^dI       6%�	���e���A�*;


total_loss�~�@

error_R�=?

learning_rate_1��5�+VI       6%�	��e���A�*;


total_loss�@

error_RM�J?

learning_rate_1��5] I       6%�	%S�e���A�*;


total_loss*۸@

error_R�XR?

learning_rate_1��5K���I       6%�	S��e���A�*;


total_loss� �@

error_RM(V?

learning_rate_1��5<��I       6%�	���e���A�*;


total_loss}�@

error_R��L?

learning_rate_1��5oхzI       6%�	-!�e���A�*;


total_lossx$�@

error_R3-N?

learning_rate_1��5^�I       6%�	�c�e���A�*;


total_lossvT�@

error_Ra
H?

learning_rate_1��5:4��I       6%�	o��e���A�*;


total_lossz��@

error_R@S?

learning_rate_1��5s�U�I       6%�	���e���A�*;


total_loss�@

error_RM?

learning_rate_1��5/���I       6%�	.�e���A�*;


total_lossho@

error_R�E?

learning_rate_1��5��W�I       6%�	�q�e���A�*;


total_loss>Ї@

error_R.\@?

learning_rate_1��5�S�I       6%�	i��e���A�*;


total_lossD��@

error_RɪN?

learning_rate_1��5���I       6%�	��e���A�*;


total_lossF}�@

error_RͪL?

learning_rate_1��5�ZKI       6%�	#<�e���A�*;


total_loss<�@

error_R�)Y?

learning_rate_1��5��^rI       6%�	e~�e���A�*;


total_loss
߻@

error_RrbM?

learning_rate_1��5`��lI       6%�	 ��e���A�*;


total_loss�=�@

error_R7�K?

learning_rate_1��5`'�I       6%�	o�e���A�*;


total_loss}A

error_R4�D?

learning_rate_1��5HV�cI       6%�	`M�e���A�*;


total_lossæ�@

error_R��N?

learning_rate_1��5��;I       6%�	C��e���A�*;


total_loss��@

error_R�<K?

learning_rate_1��5e6rI       6%�	��e���A�*;


total_loss�!�@

error_R&�H?

learning_rate_1��5�G�&I       6%�	�Q�e���A�*;


total_loss<H�@

error_R�DO?

learning_rate_1��5ض�SI       6%�	���e���A�*;


total_loss~�@

error_RA�R?

learning_rate_1��5)d��I       6%�	7�e���A�*;


total_loss�.�@

error_R��L?

learning_rate_1��5�G�I       6%�	�[�e���A�*;


total_loss��@

error_R��S?

learning_rate_1��5�Fp^I       6%�	X��e���A�*;


total_loss���@

error_R�IW?

learning_rate_1��5�h�rI       6%�	���e���A�*;


total_loss(H�@

error_R�=J?

learning_rate_1��5��~I       6%�	q2�e���A�*;


total_loss;>�@

error_Rv�S?

learning_rate_1��5�TB�I       6%�	�x�e���A�*;


total_loss'�@

error_R�J?

learning_rate_1��5�i<�I       6%�	c��e���A�*;


total_loss��@

error_R�Q?

learning_rate_1��5Vu�`I       6%�	y��e���A�*;


total_loss<��@

error_R
S?

learning_rate_1��5�
W>I       6%�	2=�e���A�*;


total_loss抴@

error_Rv.M?

learning_rate_1��5s�I�I       6%�	��e���A�*;


total_lossI�@

error_Rq�N?

learning_rate_1��5� �I       6%�	$��e���A�*;


total_loss�a�@

error_RQO?

learning_rate_1��5ÊFI       6%�	}�e���A�*;


total_loss��@

error_Rf�P?

learning_rate_1��52��I       6%�	���e���A�*;


total_lossؑ�@

error_R�L?

learning_rate_1��5<'�.I       6%�	���e���A�*;


total_losso�@

error_R�I?

learning_rate_1��5�	}�I       6%�	��e���A�*;


total_losszɞ@

error_R)kQ?

learning_rate_1��5P�FI       6%�	%h�e���A�*;


total_loss;&A

error_R��X?

learning_rate_1��5L�=�I       6%�	}��e���A�*;


total_loss���@

error_RZqS?

learning_rate_1��5���I       6%�	���e���A�*;


total_loss��@

error_R�GI?

learning_rate_1��5���I       6%�	�9�e���A�*;


total_loss��@

error_RW�^?

learning_rate_1��5 ��I       6%�	���e���A�*;


total_loss�p�@

error_RO:K?

learning_rate_1��5�-4�I       6%�	]��e���A�*;


total_loss��@

error_R��M?

learning_rate_1��5O�OnI       6%�	��e���A�*;


total_loss�%�@

error_R�,?

learning_rate_1��5�^I       6%�	r_�e���A�*;


total_lossl��@

error_RC�??

learning_rate_1��5��7rI       6%�	��e���A�*;


total_loss
�@

error_R.O?

learning_rate_1��5L�;I       6%�	���e���A�*;


total_lossڅ@

error_RVk6?

learning_rate_1��5H��QI       6%�	)�e���A�*;


total_loss��@

error_R�CQ?

learning_rate_1��5�\;�I       6%�	�k�e���A�*;


total_lossA

error_RqE?

learning_rate_1��5�p6�I       6%�	��e���A�*;


total_loss_c�@

error_R�TS?

learning_rate_1��5^p(I       6%�	q��e���A�*;


total_lossƣ�@

error_Rv6?

learning_rate_1��5�8��I       6%�	i5�e���A�*;


total_loss��@

error_R��K?

learning_rate_1��5�f`I       6%�	�|�e���A�*;


total_loss���@

error_R1J?

learning_rate_1��5�3cI       6%�	��e���A�*;


total_loss���@

error_Rik_?

learning_rate_1��5m�!I       6%�	l�e���A�*;


total_loss�@

error_R�Q?

learning_rate_1��5E���I       6%�	R�e���A�*;


total_loss!i�@

error_R��I?

learning_rate_1��5cM�%I       6%�	j��e���A�*;


total_loss���@

error_R��I?

learning_rate_1��5�~�I       6%�	��e���A�*;


total_loss@ַ@

error_R@bS?

learning_rate_1��5���bI       6%�	�"�e���A�*;


total_loss��@

error_R$�d?

learning_rate_1��5����I       6%�	�d�e���A�*;


total_loss&@�@

error_R��X?

learning_rate_1��5\�h�I       6%�	S��e���A�*;


total_loss<�@

error_R3�W?

learning_rate_1��5?:�I       6%�	O��e���A�*;


total_loss$`�@

error_RrQ?

learning_rate_1��5
Ep I       6%�	M0�e���A�*;


total_loss�0�@

error_R�q:?

learning_rate_1��5�-��I       6%�	�r�e���A�*;


total_lossl�@

error_R��@?

learning_rate_1��5�[5 I       6%�	��e���A�*;


total_loss&��@

error_R��R?

learning_rate_1��5�'�I       6%�	j��e���A�*;


total_loss��@

error_RtBK?

learning_rate_1��5�V�nI       6%�	s=�e���A�*;


total_lossW��@

error_R%�L?

learning_rate_1��5�ƺ�I       6%�	��e���A�*;


total_loss�{@

error_Rv�H?

learning_rate_1��5F�P�I       6%�	���e���A�*;


total_loss��Y@

error_R6`@?

learning_rate_1��5�dOI       6%�	d�e���A�*;


total_loss���@

error_RT?

learning_rate_1��5�`�I       6%�	�_�e���A�*;


total_lossc�@

error_RE�U?

learning_rate_1��5%1��I       6%�	ި�e���A�*;


total_loss�E�@

error_R��V?

learning_rate_1��5g���I       6%�		��e���A�*;


total_loss7#�@

error_R�k=?

learning_rate_1��5&��-I       6%�	B0�e���A�*;


total_lossD��@

error_R�zG?

learning_rate_1��5ҏ&�I       6%�	�y�e���A�*;


total_loss:?�@

error_R�P?

learning_rate_1��5��I       6%�	���e���A�*;


total_lossP�@

error_RF�F?

learning_rate_1��5&��I       6%�	��e���A�*;


total_loss���@

error_R��U?

learning_rate_1��5��I       6%�	PH�e���A�*;


total_lossJ��@

error_RY?

learning_rate_1��5J���I       6%�	ݧ�e���A�*;


total_loss3��@

error_R�.?

learning_rate_1��5��8UI       6%�	���e���A�*;


total_lossOR�@

error_RK?

learning_rate_1��5(`=�I       6%�	�F�e���A�*;


total_loss���@

error_R�GU?

learning_rate_1��5{2�I       6%�	���e���A�*;


total_lossm֧@

error_R�G?

learning_rate_1��5��>ZI       6%�	���e���A�*;


total_lossIj�@

error_RS�=?

learning_rate_1��5]��I       6%�	��e���A�*;


total_loss�ť@

error_RT�I?

learning_rate_1��5����I       6%�	X�e���A�*;


total_lossv��@

error_R1�Y?

learning_rate_1��5hd��I       6%�	���e���A�*;


total_loss3��@

error_R�K?

learning_rate_1��5��;I       6%�	���e���A�*;


total_loss-a�@

error_RݰJ?

learning_rate_1��5g�R�I       6%�	Z*�e���A�*;


total_loss{��@

error_R�	J?

learning_rate_1��53g�I       6%�	�r�e���A�*;


total_lossN�@

error_Rf�Z?

learning_rate_1��5��~lI       6%�	ҷ�e���A�*;


total_loss��@

error_R�wB?

learning_rate_1��5%+�I       6%�	���e���A�*;


total_loss���@

error_R�T?

learning_rate_1��50{��I       6%�	�?�e���A�*;


total_loss�_�@

error_R��E?

learning_rate_1��5��щI       6%�	u��e���A�*;


total_lossZ��@

error_R-K?

learning_rate_1��5y(I       6%�	I��e���A�*;


total_lossI��@

error_R3rN?

learning_rate_1��5�alOI       6%�	��e���A�*;


total_loss��@

error_R��R?

learning_rate_1��5�MrI       6%�	)W�e���A�*;


total_loss���@

error_R�	I?

learning_rate_1��5��sI       6%�	(��e���A�*;


total_loss��b@

error_R�I?

learning_rate_1��5NA}�I       6%�	d��e���A�*;


total_loss�)�@

error_R��P?

learning_rate_1��5�v¯I       6%�	�+�e���A�*;


total_loss���@

error_R�,K?

learning_rate_1��5�yI       6%�	]p�e���A�*;


total_loss�ҽ@

error_R�RI?

learning_rate_1��5+��I       6%�	,��e���A�*;


total_loss�l@

error_RmY?

learning_rate_1��5[���I       6%�	���e���A�*;


total_loss�3A

error_RX�P?

learning_rate_1��5;��}I       6%�	�@�e���A�*;


total_loss��@

error_R_�H?

learning_rate_1��5ѭקI       6%�	���e���A�*;


total_lossJ: A

error_R|SG?

learning_rate_1��5,1I       6%�	���e���A�*;


total_lossR��@

error_R8A?

learning_rate_1��5+�I       6%�	�e���A�*;


total_lossvj�@

error_R
�f?

learning_rate_1��5���YI       6%�	�T�e���A�*;


total_loss�@

error_RyO?

learning_rate_1��5��&�I       6%�	���e���A�*;


total_loss���@

error_Rrp=?

learning_rate_1��5����I       6%�	���e���A�*;


total_loss.�@

error_R��^?

learning_rate_1��5ey�ZI       6%�	��e���A�*;


total_lossA�@

error_R��G?

learning_rate_1��5��\tI       6%�	(j�e���A�*;


total_loss?X�@

error_R�T?

learning_rate_1��5��I       6%�	]��e���A�*;


total_loss�@

error_R�>G?

learning_rate_1��5F^�I       6%�	k��e���A�*;


total_loss�Q�@

error_RJFN?

learning_rate_1��51�I       6%�	Q2�e���A�*;


total_loss8͠@

error_R�g?

learning_rate_1��5��]3I       6%�	�y�e���A�*;


total_loss1:�@

error_R֯W?

learning_rate_1��53OޯI       6%�	|��e���A�*;


total_loss�QA

error_R�j?

learning_rate_1��5[<�I       6%�	  �e���A�*;


total_loss���@

error_R_Sh?

learning_rate_1��5G��I       6%�	�E�e���A�*;


total_lossg��@

error_R��Q?

learning_rate_1��5YM�yI       6%�	���e���A�*;


total_loss�1	A

error_RJ�H?

learning_rate_1��5܆$�I       6%�	���e���A�*;


total_lossi�@

error_R�^=?

learning_rate_1��5�N�I       6%�	S�e���A�*;


total_loss���@

error_R�CU?

learning_rate_1��5���I       6%�	�v�e���A�*;


total_loss�{�@

error_Rn�U?

learning_rate_1��5|�t�I       6%�	���e���A�*;


total_loss���@

error_RO-O?

learning_rate_1��5Z8I       6%�	�(�e���A�*;


total_loss�^�@

error_R�/8?

learning_rate_1��5�[3I       6%�	!p�e���A�*;


total_loss]��@

error_RaWB?

learning_rate_1��5w�[`I       6%�	i��e���A�*;


total_loss��@

error_R,E?

learning_rate_1��5���8I       6%�	k��e���A�*;


total_loss�T�@

error_Ra�I?

learning_rate_1��5.���I       6%�	9B�e���A�*;


total_loss�"�@

error_R�IA?

learning_rate_1��5n;I       6%�	��e���A�*;


total_loss�"�@

error_RVR?

learning_rate_1��5	j$I       6%�	���e���A�*;


total_loss;a�@

error_R��T?

learning_rate_1��5��b�I       6%�	6�e���A�*;


total_loss`_�@

error_R��A?

learning_rate_1��5��FI       6%�	Hz�e���A�*;


total_lossv��@

error_R�aA?

learning_rate_1��5^i��I       6%�	���e���A�*;


total_loss���@

error_R(L?

learning_rate_1��5%I       6%�	��e���A�*;


total_loss���@

error_R��N?

learning_rate_1��5W�|�I       6%�	�I�e���A�*;


total_lossn5A

error_R�eA?

learning_rate_1��5Z$|�I       6%�	���e���A�*;


total_loss�d@

error_RmE?

learning_rate_1��5 u�jI       6%�	���e���A�*;


total_lossÿ�@

error_R$/b?

learning_rate_1��5�޹I       6%�	�B�e���A�*;


total_loss3��@

error_R @L?

learning_rate_1��5�;��I       6%�	���e���A�*;


total_loss��@

error_R�bC?

learning_rate_1��5/B�=I       6%�	���e���A�*;


total_loss�8�@

error_R�A?

learning_rate_1��5^=�,I       6%�	��e���A�*;


total_loss��@

error_R�V?

learning_rate_1��5SZ�I       6%�	�`�e���A�*;


total_loss��@

error_R{�R?

learning_rate_1��5�[vI       6%�		��e���A�*;


total_loss*��@

error_R1�S?

learning_rate_1��5巫tI       6%�	_��e���A�*;


total_loss���@

error_R��B?

learning_rate_1��5S
�4I       6%�	L7 f���A�*;


total_loss���@

error_R$�D?

learning_rate_1��5t�6@I       6%�	�{ f���A�*;


total_loss㺱@

error_R��[?

learning_rate_1��5�#B�I       6%�	U� f���A�*;


total_loss�\�@

error_R/�>?

learning_rate_1��5���@I       6%�	=f���A�*;


total_lossZ�g@

error_R��C?

learning_rate_1��5e%�I       6%�	�Ff���A�*;


total_lossOb�@

error_R�)H?

learning_rate_1��5��r�I       6%�	S�f���A�*;


total_loss���@

error_R}+L?

learning_rate_1��5]��)I       6%�	{�f���A�*;


total_loss\:�@

error_Rc�P?

learning_rate_1��5Li��I       6%�	�f���A�*;


total_loss���@

error_R�`A?

learning_rate_1��5����I       6%�	�Tf���A�*;


total_loss?��@

error_R�0R?

learning_rate_1��5�XvFI       6%�	P�f���A�*;


total_lossy�@

error_RR�E?

learning_rate_1��5�"m�I       6%�	��f���A�*;


total_loss���@

error_R�e4?

learning_rate_1��5����I       6%�	�$f���A�*;


total_loss���@

error_Rߎ8?

learning_rate_1��5no�AI       6%�	~if���A�*;


total_loss�6�@

error_R�`?

learning_rate_1��5L$�8I       6%�	��f���A�*;


total_lossZe�@

error_R[ZK?

learning_rate_1��5�s4sI       6%�	e�f���A�*;


total_loss�� A

error_R��??

learning_rate_1��5�6SiI       6%�	�;f���A�*;


total_lossL�@

error_Rm�E?

learning_rate_1��5���I       6%�	�~f���A�*;


total_lossQ��@

error_R-ST?

learning_rate_1��5�d��I       6%�	X�f���A�*;


total_lossl�@

error_R�M?

learning_rate_1��5�G�I       6%�	�f���A�*;


total_loss
��@

error_R�iB?

learning_rate_1��50�r�I       6%�	�ff���A�*;


total_loss���@

error_R�L?

learning_rate_1��5�%�I       6%�	�f���A�*;


total_loss{��@

error_R��9?

learning_rate_1��5���6I       6%�	f���A�*;


total_lossد�@

error_R�U?

learning_rate_1��5��meI       6%�	Rf���A�*;


total_loss�n�@

error_R�H?

learning_rate_1��5D�idI       6%�	�f���A�*;


total_loss�[�@

error_R=�\?

learning_rate_1��5�+E�I       6%�	��f���A�*;


total_loss�'�@

error_R��E?

learning_rate_1��5\^�I       6%�	�f���A�*;


total_loss�}�@

error_R�@?

learning_rate_1��5|���I       6%�	df���A�*;


total_loss��@

error_R��J?

learning_rate_1��5�5�I       6%�	Ψf���A�*;


total_loss:��@

error_R��E?

learning_rate_1��5ܒ:I       6%�	��f���A�*;


total_loss}�@

error_RE�J?

learning_rate_1��5M���I       6%�	k2f���A�*;


total_loss�
�@

error_R��7?

learning_rate_1��55�w�I       6%�	�~f���A�*;


total_lossR�@

error_RA?

learning_rate_1��5/ I       6%�	S�f���A�*;


total_lossV1�@

error_R�Zf?

learning_rate_1��5�cI       6%�	P	f���A�*;


total_loss�X�@

error_R�K?

learning_rate_1��5���lI       6%�	Nb	f���A�*;


total_loss%�@

error_RH8\?

learning_rate_1��5m#%I       6%�	n�	f���A�*;


total_lossx<�@

error_R�DR?

learning_rate_1��5h�*�I       6%�	N�	f���A�*;


total_loss�I�@

error_R3*L?

learning_rate_1��5����I       6%�	?
f���A�*;


total_loss��@

error_R��P?

learning_rate_1��5���I       6%�	ӄ
f���A�*;


total_loss�q�@

error_R�bP?

learning_rate_1��54��YI       6%�	��
f���A�*;


total_loss��@

error_R�;J?

learning_rate_1��5��rI       6%�	�f���A�*;


total_loss�@

error_R�R?

learning_rate_1��5Ԓ��I       6%�	Yf���A�*;


total_lossυ�@

error_R� ]?

learning_rate_1��5J��*I       6%�	 �f���A�*;


total_loss��@

error_R?hK?

learning_rate_1��53�I       6%�	�f���A�*;


total_lossO �@

error_R�V?

learning_rate_1��5��ۉI       6%�	B6f���A�*;


total_loss�a�@

error_Rx�D?

learning_rate_1��5�ǋI       6%�	`wf���A�*;


total_loss6*�@

error_Rl�=?

learning_rate_1��5Q6�I       6%�	��f���A�*;


total_loss���@

error_R LP?

learning_rate_1��5���I       6%�	|f���A�*;


total_loss���@

error_RO�S?

learning_rate_1��5w��I       6%�	�Gf���A�*;


total_loss!ؠ@

error_RyG?

learning_rate_1��5��MI       6%�	��f���A�*;


total_lossl7�@

error_Ri*I?

learning_rate_1��52kII       6%�	7�f���A�*;


total_loss2Ѡ@

error_R)�M?

learning_rate_1��5f/h�I       6%�	�6f���A�*;


total_loss���@

error_R��[?

learning_rate_1��58��I       6%�	�zf���A�*;


total_loss�D�@

error_R*�M?

learning_rate_1��52b��I       6%�	m�f���A�*;


total_loss��A

error_RE?

learning_rate_1��5��	�I       6%�	_f���A�*;


total_loss�;�@

error_R�(=?

learning_rate_1��5�d{I       6%�	�Yf���A�*;


total_loss��@

error_R8gR?

learning_rate_1��5����I       6%�	W�f���A�*;


total_loss��@

error_RϪJ?

learning_rate_1��5�'��I       6%�	��f���A�*;


total_lossT�@

error_Rf�Z?

learning_rate_1��5���vI       6%�	�,f���A�*;


total_lossi��@

error_RH/]?

learning_rate_1��5��z�I       6%�	�qf���A�*;


total_loss��@

error_R	 R?

learning_rate_1��5�+�I       6%�	��f���A�*;


total_loss��@

error_RdO@?

learning_rate_1��5�X�I       6%�	�f���A�*;


total_loss��@

error_R&�O?

learning_rate_1��50%hI       6%�	$>f���A�*;


total_loss�{�@

error_R͈N?

learning_rate_1��5�ցI       6%�	�f���A�*;


total_lossD��@

error_R��V?

learning_rate_1��5��AI       6%�	��f���A�*;


total_loss��@

error_RO-?

learning_rate_1��5�&�RI       6%�	�f���A�*;


total_loss�@

error_R�pV?

learning_rate_1��5�i��I       6%�	�]f���A�*;


total_loss*��@

error_R�g8?

learning_rate_1��5�&2I       6%�	��f���A�*;


total_loss�=�@

error_R{OP?

learning_rate_1��5 ��I       6%�	��f���A�*;


total_lossβ@

error_R)wE?

learning_rate_1��5�I       6%�	}&f���A�*;


total_loss��@

error_R�<?

learning_rate_1��5��4�I       6%�	�jf���A�*;


total_loss&f�@

error_R�U?

learning_rate_1��5�x�%I       6%�	W�f���A�*;


total_loss� �@

error_R�8Z?

learning_rate_1��5�hRI       6%�	 �f���A�*;


total_loss 1�@

error_R	�M?

learning_rate_1��5a���I       6%�	�0f���A�*;


total_loss�@

error_R�4J?

learning_rate_1��5��.�I       6%�	�tf���A�*;


total_lossYd�@

error_R��N?

learning_rate_1��5���I       6%�	Ϲf���A�*;


total_lossh��@

error_R<)P?

learning_rate_1��5��.I       6%�	Bf���A�*;


total_loss��@

error_R��N?

learning_rate_1��5�U �I       6%�	�Nf���A�*;


total_loss�ߒ@

error_R;�R?

learning_rate_1��5�p6I       6%�	$�f���A�*;


total_lossw�@

error_R6%O?

learning_rate_1��5�6w/I       6%�	��f���A�*;


total_loss��@

error_R BL?

learning_rate_1��5!�,-I       6%�	~(f���A�*;


total_loss_H~@

error_RGW?

learning_rate_1��5���I       6%�	bqf���A�*;


total_loss�@

error_R�L?

learning_rate_1��5p�AI       6%�	��f���A�*;


total_loss��@

error_RiU?

learning_rate_1��5�D�I       6%�	��f���A�*;


total_lossJ�@

error_R�^[?

learning_rate_1��5,e�OI       6%�	vAf���A�*;


total_lossn��@

error_R23\?

learning_rate_1��5�=�YI       6%�	'�f���A�*;


total_lossE" A

error_R��F?

learning_rate_1��5)�5�I       6%�	��f���A�*;


total_lossf\�@

error_R�I[?

learning_rate_1��5\ء*I       6%�	Tf���A�*;


total_loss�ˢ@

error_R�S?

learning_rate_1��5���GI       6%�	nNf���A�*;


total_lossx��@

error_R��P?

learning_rate_1��5�V�'I       6%�	�f���A�*;


total_loss(��@

error_R�H?

learning_rate_1��5�fVqI       6%�	��f���A�*;


total_lossP��@

error_RWS?

learning_rate_1��5��*I       6%�	2f���A�*;


total_loss��@

error_R�I?

learning_rate_1��5�#P�I       6%�	�`f���A�*;


total_lossLE�@

error_Rߋ@?

learning_rate_1��5�|�cI       6%�	9�f���A�*;


total_loss%�@

error_R�:b?

learning_rate_1��5�*��I       6%�	j�f���A�*;


total_loss_r�@

error_R��W?

learning_rate_1��5�y�lI       6%�	<f���A�*;


total_loss�֟@

error_R&�N?

learning_rate_1��5�aI       6%�	сf���A�*;


total_loss>��@

error_R&�>?

learning_rate_1��5,�E�I       6%�	��f���A�*;


total_loss���@

error_R�F?

learning_rate_1��5�L\�I       6%�	Yf���A�*;


total_loss��@

error_R��C?

learning_rate_1��5���I       6%�	EIf���A�*;


total_loss���@

error_RE(P?

learning_rate_1��5�`�I       6%�	ɏf���A�*;


total_loss���@

error_R�rL?

learning_rate_1��5���yI       6%�	��f���A�*;


total_lossε�@

error_Rba?

learning_rate_1��5YTY!I       6%�	V%f���A�*;


total_losst��@

error_R�	G?

learning_rate_1��5��I       6%�	�if���A�*;


total_losssp-A

error_R�]??

learning_rate_1��5�u�}I       6%�	*�f���A�*;


total_loss/�A

error_R�R?

learning_rate_1��5�^��I       6%�	��f���A�*;


total_loss.��@

error_R�RX?

learning_rate_1��5}�+PI       6%�	�-f���A�*;


total_loss�~A

error_R3�X?

learning_rate_1��5URw�I       6%�	Y�f���A�*;


total_losstȧ@

error_R�K?

learning_rate_1��5*�^jI       6%�	��f���A�*;


total_loss���@

error_Rj�N?

learning_rate_1��5f�-�I       6%�	1f���A�*;


total_lossN��@

error_RlZ=?

learning_rate_1��5=
��I       6%�	�xf���A�*;


total_loss���@

error_R��A?

learning_rate_1��5���I       6%�		�f���A�*;


total_loss�L�@

error_R�N?

learning_rate_1��5��cI       6%�	�	f���A�*;


total_loss ��@

error_RR?

learning_rate_1��5�ws�I       6%�	�Qf���A�*;


total_loss�)w@

error_R��A?

learning_rate_1��5���[I       6%�	H�f���A�*;


total_loss)�@

error_R��I?

learning_rate_1��5�O3�I       6%�	��f���A�*;


total_loss�أ@

error_R��U?

learning_rate_1��5���zI       6%�	i f���A�*;


total_loss�@

error_RzbL?

learning_rate_1��5�(?I       6%�	Mc f���A�*;


total_loss2"�@

error_R$7?

learning_rate_1��5J�x�I       6%�	B� f���A�*;


total_lossâ�@

error_R!U?

learning_rate_1��5c��I       6%�	o� f���A�*;


total_lossΆ�@

error_R��R?

learning_rate_1��5�}��I       6%�	�K!f���A�*;


total_loss�E�@

error_R�UT?

learning_rate_1��5�4I       6%�	�!f���A�*;


total_lossH��@

error_R�jL?

learning_rate_1��5��kI       6%�	�!f���A�*;


total_lossd0�@

error_Ra�I?

learning_rate_1��5ѯ_&I       6%�	�&"f���A�*;


total_loss.Ms@

error_RL>?

learning_rate_1��5f�(�I       6%�	Ƅ"f���A�*;


total_loss�J�@

error_R�%9?

learning_rate_1��5qB=@I       6%�	��"f���A�*;


total_loss���@

error_R�R?

learning_rate_1��5�4کI       6%�	p#f���A�*;


total_loss�1�@

error_R�Q?

learning_rate_1��5'�;�I       6%�	�W#f���A�*;


total_loss���@

error_R��R?

learning_rate_1��52��]I       6%�	Z�#f���A�*;


total_loss,&�@

error_R�X;?

learning_rate_1��5vN�I       6%�	��#f���A�*;


total_loss±�@

error_R�#8?

learning_rate_1��5��1�I       6%�	C$$f���A�*;


total_lossDG�@

error_R3iK?

learning_rate_1��5���II       6%�	�i$f���A�*;


total_lossF��@

error_R�BK?

learning_rate_1��5�x"iI       6%�	ȯ$f���A�*;


total_lossAu�@

error_R�OU?

learning_rate_1��5b�I       6%�	��$f���A�*;


total_loss$�@

error_R��L?

learning_rate_1��5�.xI       6%�	:%f���A�*;


total_loss@�@

error_R��V?

learning_rate_1��5<e�YI       6%�	�~%f���A�*;


total_loss���@

error_RI�Q?

learning_rate_1��5A��I       6%�	W�%f���A�*;


total_loss�@

error_R�p@?

learning_rate_1��5�Rt�I       6%�	�
&f���A�*;


total_lossI��@

error_R�H?

learning_rate_1��5v`�I       6%�	N&f���A�*;


total_loss3�@

error_R��R?

learning_rate_1��5TQV�I       6%�	��&f���A�*;


total_loss�;�@

error_R��A?

learning_rate_1��5���~I       6%�	��&f���A�*;


total_loss4��@

error_R�5:?

learning_rate_1��5*�I       6%�	'f���A�*;


total_loss	��@

error_R}#Y?

learning_rate_1��5]HI       6%�	rb'f���A�*;


total_loss���@

error_R�%T?

learning_rate_1��5����I       6%�	6�'f���A�*;


total_loss!��@

error_R�fP?

learning_rate_1��5u |�I       6%�	<�'f���A�*;


total_loss�!�@

error_RsIU?

learning_rate_1��5Ⱦ�}I       6%�	./(f���A�*;


total_loss��@

error_RI9G?

learning_rate_1��5	��I       6%�	�s(f���A�*;


total_loss�B�@

error_Ri�]?

learning_rate_1��5ݰ��I       6%�	Ӷ(f���A�*;


total_loss�V�@

error_R��L?

learning_rate_1��5i�%I       6%�	�(f���A�*;


total_loss��@

error_R[:b?

learning_rate_1��5s���I       6%�	B)f���A�*;


total_loss��@

error_R	_^?

learning_rate_1��5���I       6%�	��)f���A�*;


total_lossƺ@

error_RW`X?

learning_rate_1��5�^i�I       6%�	�)f���A�*;


total_loss�D�@

error_R�L?

learning_rate_1��5���I       6%�	�*f���A�*;


total_loss<VA

error_Rx�P?

learning_rate_1��53�W�I       6%�	vb*f���A�*;


total_lossx�^@

error_RTC?

learning_rate_1��5�(�I       6%�	c�*f���A�*;


total_loss�G�@

error_R�E?

learning_rate_1��5̔ZnI       6%�	8�*f���A�*;


total_loss��@

error_R�#Z?

learning_rate_1��5����I       6%�	k3+f���A�*;


total_loss��@

error_R$�C?

learning_rate_1��5.�@I       6%�	�w+f���A�*;


total_loss֏@

error_RfXL?

learning_rate_1��5� ��I       6%�	�+f���A�*;


total_loss�.�@

error_RsK?

learning_rate_1��5�/�I       6%�	T�+f���A�*;


total_loss���@

error_R*�U?

learning_rate_1��5h-jLI       6%�	�I,f���A�*;


total_lossd��@

error_RlH?

learning_rate_1��5(��eI       6%�	��,f���A�*;


total_loss^o�@

error_R�EP?

learning_rate_1��5K%�NI       6%�	��,f���A�*;


total_loss;A

error_RnG?

learning_rate_1��5�!�I       6%�	+!-f���A�*;


total_loss�v�@

error_RL�Q?

learning_rate_1��5+VI       6%�	[h-f���A�*;


total_loss���@

error_R=�B?

learning_rate_1��5{h��I       6%�	*�-f���A�*;


total_lossh8�@

error_R�R?

learning_rate_1��5iIAXI       6%�	.f���A�*;


total_loss�@

error_R�F?

learning_rate_1��5�X �I       6%�	�\.f���A�*;


total_loss4@

error_R-oM?

learning_rate_1��5O왞I       6%�	Ơ.f���A�*;


total_loss׹�@

error_R9C?

learning_rate_1��5��tI       6%�	Z�.f���A�*;


total_loss!U�@

error_R�g?

learning_rate_1��5s�rI       6%�	�+/f���A�*;


total_loss�?�@

error_R#R?

learning_rate_1��5��C�I       6%�	v/f���A�*;


total_loss8׽@

error_R�4J?

learning_rate_1��5n�NI       6%�	�/f���A�*;


total_loss�}�@

error_Rx/M?

learning_rate_1��5��b�I       6%�	
0f���A�*;


total_lossS��@

error_Rv�U?

learning_rate_1��5a��I       6%�	O0f���A�*;


total_loss̉�@

error_R��U?

learning_rate_1��50Q�I       6%�	��0f���A�*;


total_lossdb�@

error_R��R?

learning_rate_1��5�oX�I       6%�	i�0f���A�*;


total_lossF_A

error_R6F?

learning_rate_1��5�3eJI       6%�	�1f���A�*;


total_loss�v�@

error_R3s[?

learning_rate_1��5��&I       6%�	c1f���A�*;


total_loss�ȧ@

error_R�^?

learning_rate_1��5Bu�I       6%�	C�1f���A�*;


total_loss���@

error_RKX?

learning_rate_1��5�8�I       6%�	S�1f���A�*;


total_lossl�@

error_R�O?

learning_rate_1��5�!5�I       6%�	�)2f���A�*;


total_lossA��@

error_R\8?

learning_rate_1��5;�,I       6%�	Gn2f���A�*;


total_lossR�@

error_R�OA?

learning_rate_1��5)F�JI       6%�	Z�2f���A�*;


total_loss��@

error_Rl=T?

learning_rate_1��5���I       6%�	b�2f���A�*;


total_lossN'�@

error_R��F?

learning_rate_1��5�j�I       6%�	n93f���A�*;


total_loss�@

error_R)�E?

learning_rate_1��5v�nI       6%�	�~3f���A�*;


total_loss��@

error_R
�R?

learning_rate_1��5��C�I       6%�	Q�3f���A�*;


total_loss֐�@

error_RO�[?

learning_rate_1��5�2�I       6%�	�4f���A�*;


total_lossR��@

error_R@�J?

learning_rate_1��5aSI       6%�	
U4f���A�*;


total_lossq:�@

error_R>H?

learning_rate_1��5<
�I       6%�	��4f���A�*;


total_lossn�@

error_Rs�M?

learning_rate_1��5���I       6%�	b�4f���A�*;


total_lossm��@

error_R3\?

learning_rate_1��5�=I       6%�	�5f���A�*;


total_loss;�g@

error_RܔJ?

learning_rate_1��5b�[�I       6%�		c5f���A�*;


total_loss���@

error_R�
T?

learning_rate_1��5��I       6%�	 �5f���A�*;


total_loss��@

error_R`�U?

learning_rate_1��5ϗ�I       6%�	��5f���A�*;


total_loss���@

error_R��J?

learning_rate_1��5�=��I       6%�	�,6f���A�*;


total_loss�C�@

error_RE9?

learning_rate_1��5�`k
I       6%�	v6f���A�*;


total_loss���@

error_Rx�O?

learning_rate_1��5D�I       6%�	��6f���A�*;


total_loss��@

error_RSNJ?

learning_rate_1��5�!��I       6%�	� 7f���A�*;


total_loss-�@

error_R��H?

learning_rate_1��5��@�I       6%�	uE7f���A�*;


total_losso�x@

error_RkB?

learning_rate_1��5��HI       6%�	!�7f���A�*;


total_loss��~@

error_R��@?

learning_rate_1��5I�� I       6%�	&�7f���A�*;


total_loss|=�@

error_R�u9?

learning_rate_1��5���I       6%�	�8f���A�*;


total_loss�xA

error_R�Y?

learning_rate_1��5�c�I       6%�	>[8f���A�*;


total_lossnU�@

error_RRy@?

learning_rate_1��5!���I       6%�	X�8f���A�*;


total_loss�3�@

error_R��N?

learning_rate_1��5��
I       6%�	��8f���A�*;


total_loss���@

error_R�<D?

learning_rate_1��5�!�I       6%�	�19f���A�*;


total_loss�G�@

error_R$�B?

learning_rate_1��5��<I       6%�	w9f���A�*;


total_loss#�@

error_R4�U?

learning_rate_1��5�>�I       6%�	E�9f���A�*;


total_loss�x�@

error_Rq�L?

learning_rate_1��5��!TI       6%�	5�9f���A�*;


total_lossFX�@

error_R��D?

learning_rate_1��5:&�I       6%�	�@:f���A�*;


total_loss���@

error_R�C?

learning_rate_1��5�[��I       6%�	�:f���A�*;


total_loss���@

error_R��G?

learning_rate_1��56�X�I       6%�	!�:f���A�*;


total_loss���@

error_R�H?

learning_rate_1��5�.vI       6%�	�;f���A�*;


total_loss�@

error_RɼH?

learning_rate_1��5��kI       6%�	�T;f���A�*;


total_loss�z�@

error_R�:N?

learning_rate_1��5�/��I       6%�	��;f���A�*;


total_loss-d�@

error_Rf??

learning_rate_1��53�<�I       6%�	��;f���A�*;


total_losss��@

error_R�C?

learning_rate_1��5��VI       6%�	� <f���A�*;


total_loss�A�@

error_RGL?

learning_rate_1��5���I       6%�	 d<f���A�*;


total_lossO�@

error_R�``?

learning_rate_1��5�?��I       6%�	��<f���A�*;


total_loss��@

error_R�^J?

learning_rate_1��5��:I       6%�	f�<f���A�*;


total_loss4�@

error_RNeW?

learning_rate_1��5��+I       6%�	�2=f���A�*;


total_loss���@

error_R\�L?

learning_rate_1��5�P�I       6%�	f=f���A�*;


total_loss_�@

error_Rl#=?

learning_rate_1��5V�+�I       6%�	$�=f���A�*;


total_loss��@

error_R�=r?

learning_rate_1��5��I       6%�	!>f���A�*;


total_loss)zx@

error_R��O?

learning_rate_1��5�[BI       6%�	g>f���A�*;


total_losswÎ@

error_R��??

learning_rate_1��5�Ձ$I       6%�	ԫ>f���A�*;


total_loss���@

error_R:�A?

learning_rate_1��5�:��I       6%�	�>f���A�*;


total_loss;�j@

error_R�^J?

learning_rate_1��5�y��I       6%�	2?f���A�*;


total_loss,��@

error_RDnL?

learning_rate_1��5��cI       6%�	7{?f���A�*;


total_lossi�@

error_Rs�;?

learning_rate_1��5|��I       6%�	�?f���A�*;


total_lossZa�@

error_RW�P?

learning_rate_1��5A��I       6%�	�@f���A�*;


total_loss¤@

error_R��M?

learning_rate_1��5�V2�I       6%�	�Y@f���A�*;


total_loss�bA

error_R{L[?

learning_rate_1��5���wI       6%�	��@f���A�*;


total_lossX�@

error_R)?L?

learning_rate_1��5��g�I       6%�	A�@f���A�*;


total_loss)��@

error_RN�J?

learning_rate_1��5�E3KI       6%�	�8Af���A�*;


total_lossL��@

error_R
pT?

learning_rate_1��5do�mI       6%�	�Af���A�*;


total_loss|k�@

error_R�^I?

learning_rate_1��5�T�I       6%�	��Af���A�*;


total_loss��@

error_Rc�9?

learning_rate_1��5�/ݏI       6%�	�Bf���A�*;


total_loss�*�@

error_R��>?

learning_rate_1��5��-�I       6%�	�[Bf���A�*;


total_loss?��@

error_REI?

learning_rate_1��5 �f�I       6%�	'�Bf���A�*;


total_loss���@

error_R��T?

learning_rate_1��5��3I       6%�	��Bf���A�*;


total_loss4֚@

error_R��]?

learning_rate_1��5o1
I       6%�	8Cf���A�*;


total_loss�i�@

error_R��O?

learning_rate_1��5�NTI       6%�	��Cf���A�*;


total_loss�ы@

error_R�)V?

learning_rate_1��5s!"
I       6%�	X�Cf���A�*;


total_loss
@�@

error_R�'M?

learning_rate_1��5Z�=�I       6%�	}	Df���A�*;


total_loss:�@

error_RO[?

learning_rate_1��5��yRI       6%�	MDf���A�*;


total_loss<1�@

error_R�GH?

learning_rate_1��5��l�I       6%�	i�Df���A�*;


total_losse�@

error_R�)W?

learning_rate_1��5zjāI       6%�	�Df���A�*;


total_loss��@

error_R:|<?

learning_rate_1��5�g6�I       6%�	�"Ef���A�*;


total_lossI^v@

error_RJ�Q?

learning_rate_1��5{�I       6%�	�jEf���A�*;


total_lossTЅ@

error_R��I?

learning_rate_1��5�Y|I       6%�	�Ef���A�*;


total_loss��@

error_R�Q?

learning_rate_1��5����I       6%�	i�Ef���A�*;


total_loss�r�@

error_R-�E?

learning_rate_1��5��weI       6%�	17Ff���A�*;


total_loss�GA

error_R��^?

learning_rate_1��5OO�
I       6%�	zyFf���A�*;


total_loss��@

error_R�L?

learning_rate_1��5o'a�I       6%�	��Ff���A�*;


total_loss-��@

error_R�c?

learning_rate_1��5AD�iI       6%�	9Gf���A�*;


total_losslWA

error_R�P?

learning_rate_1��5i�"�I       6%�	�VGf���A�*;


total_loss�A

error_RZO?

learning_rate_1��5J��QI       6%�	��Gf���A�*;


total_loss��@

error_RT$W?

learning_rate_1��5Ba�BI       6%�	
�Gf���A�*;


total_lossmM�@

error_R,�R?

learning_rate_1��5H�<I       6%�	+&Hf���A�*;


total_loss��@

error_R�U?

learning_rate_1��5��MI       6%�	�gHf���A�*;


total_losso��@

error_R��N?

learning_rate_1��5�s$>I       6%�	ߪHf���A�*;


total_loss�ҝ@

error_Rr??

learning_rate_1��5z2�gI       6%�	�Hf���A�*;


total_loss�'�@

error_RMa?

learning_rate_1��52��I       6%�	�4If���A�*;


total_loss���@

error_R��??

learning_rate_1��5�~
NI       6%�	�xIf���A�*;


total_loss=+�@

error_R�[?

learning_rate_1��5�\ �I       6%�	��If���A�*;


total_lossȳ�@

error_R��I?

learning_rate_1��5G�� I       6%�	^Jf���A�*;


total_lossa�A

error_REI?

learning_rate_1��5�%��I       6%�	0FJf���A�*;


total_loss���@

error_RzR?

learning_rate_1��5m+�4I       6%�	�Jf���A�*;


total_lossmM�@

error_R��T?

learning_rate_1��5��]�I       6%�	��Jf���A�*;


total_loss��@

error_RE?

learning_rate_1��5�T�I       6%�	GKf���A�*;


total_loss3z�@

error_R�B?

learning_rate_1��5��I       6%�	�bKf���A�*;


total_loss�k�@

error_R�B?

learning_rate_1��5��\TI       6%�	��Kf���A�*;


total_lossH��@

error_R�J?

learning_rate_1��5�]��I       6%�	G�Kf���A�*;


total_loss_�A

error_R�JY?

learning_rate_1��5��0�I       6%�	1Lf���A�*;


total_loss�y�@

error_R/d?

learning_rate_1��5��[I       6%�	 xLf���A�*;


total_loss:��@

error_RZE?

learning_rate_1��5�##I       6%�	��Lf���A�*;


total_loss�@

error_R��>?

learning_rate_1��5!�N�I       6%�	�Mf���A�*;


total_loss�^�@

error_R<MW?

learning_rate_1��5��8�I       6%�	�pMf���A�*;


total_lossfE�@

error_R�L?

learning_rate_1��5��e�I       6%�	]�Mf���A�*;


total_lossm~�@

error_R�g??

learning_rate_1��5%��'I       6%�	QBNf���A�*;


total_lossm��@

error_RWFp?

learning_rate_1��5��)I       6%�	v�Nf���A�*;


total_loss�@

error_R&OF?

learning_rate_1��5���I       6%�	=�Nf���A�*;


total_lossī@

error_RC�E?

learning_rate_1��5�D4�I       6%�	KOf���A�*;


total_loss���@

error_R!�Y?

learning_rate_1��5u)�XI       6%�	�[Of���A�*;


total_lossd� A

error_R��4?

learning_rate_1��5a�޲I       6%�	�Of���A�*;


total_lossֱ�@

error_R�
k?

learning_rate_1��5��bI       6%�	R�Of���A�*;


total_loss���@

error_R� H?

learning_rate_1��5�+y8I       6%�	�-Pf���A�*;


total_loss��@

error_R�I?

learning_rate_1��5*&I       6%�	�qPf���A�*;


total_loss��@

error_RG?

learning_rate_1��5��y0I       6%�	��Pf���A�*;


total_loss׵z@

error_RwBA?

learning_rate_1��5��I       6%�	B�Pf���A�*;


total_loss�'�@

error_R��[?

learning_rate_1��5��q�I       6%�	�GQf���A�*;


total_loss�K�@

error_R<x>?

learning_rate_1��5�[qI       6%�	0�Qf���A�*;


total_lossF"r@

error_Rn:?

learning_rate_1��5�L��I       6%�	V�Qf���A�*;


total_loss��>A

error_R�SX?

learning_rate_1��5۷-�I       6%�	�Rf���A�*;


total_loss�~�@

error_RI%O?

learning_rate_1��5�ПI       6%�	/cRf���A�*;


total_loss�P�@

error_R�SR?

learning_rate_1��5�XE#I       6%�	p�Rf���A�*;


total_loss"�A

error_Rn�H?

learning_rate_1��5��H"I       6%�	��Rf���A�*;


total_loss,�@

error_RI�U?

learning_rate_1��5+Á�I       6%�	�0Sf���A�*;


total_loss ٤@

error_R��c?

learning_rate_1��5yg��I       6%�	sSf���A�*;


total_loss` �@

error_R6�Y?

learning_rate_1��57�NI       6%�	˴Sf���A�*;


total_loss{�@

error_R;�N?

learning_rate_1��5��
�I       6%�	��Sf���A�*;


total_lossȗ_@

error_R��D?

learning_rate_1��5sަUI       6%�	�=Tf���A�*;


total_loss(]�@

error_R�rN?

learning_rate_1��5�F�I       6%�	r�Tf���A�*;


total_loss��@

error_R%`?

learning_rate_1��5�&��I       6%�	��Tf���A�*;


total_loss� h@

error_R\A?

learning_rate_1��5.��aI       6%�	�Uf���A�*;


total_loss1,�@

error_R��;?

learning_rate_1��5�9��I       6%�	#PUf���A�*;


total_loss���@

error_R�!R?

learning_rate_1��5�[I       6%�	��Uf���A�*;


total_loss<ߜ@

error_R�D?

learning_rate_1��5�N40I       6%�	��Uf���A�*;


total_loss8+z@

error_R=Ed?

learning_rate_1��5H��I       6%�	� Vf���A�*;


total_loss��@

error_RqTC?

learning_rate_1��5�=3�I       6%�	�cVf���A�*;


total_loss�@

error_RED\?

learning_rate_1��5F�#FI       6%�	��Vf���A�*;


total_loss*�@

error_R�W?

learning_rate_1��5Ir^�I       6%�	�Vf���A�*;


total_loss1��@

error_R L?

learning_rate_1��5���I       6%�	�1Wf���A�*;


total_loss ,s@

error_R�F?

learning_rate_1��5Tg�)I       6%�	}uWf���A�*;


total_lossi8�@

error_R sJ?

learning_rate_1��5kE��I       6%�	
�Wf���A�*;


total_loss��@

error_R��>?

learning_rate_1��5��I       6%�	!�Wf���A�*;


total_loss�i�@

error_R��H?

learning_rate_1��5��/XI       6%�	�>Xf���A�*;


total_lossՁ@

error_RzTK?

learning_rate_1��5�q�I       6%�	�Xf���A�*;


total_losssA�@

error_R�o9?

learning_rate_1��5�;n�I       6%�	.�Xf���A�*;


total_loss$]�@

error_R�wW?

learning_rate_1��5��hI       6%�	q	Yf���A�*;


total_lossZ8�@

error_RC�T?

learning_rate_1��5zQVI       6%�	MYf���A�*;


total_loss$�@

error_R�e?

learning_rate_1��5�f;I       6%�	��Yf���A�*;


total_loss�N�@

error_R��R?

learning_rate_1��5�#�RI       6%�	��Yf���A�*;


total_loss��@

error_Rx�H?

learning_rate_1��5�8�I       6%�	�Zf���A�*;


total_loss�՗@

error_R��D?

learning_rate_1��5ǵ�I       6%�	�`Zf���A�*;


total_lossX6�@

error_R�rM?

learning_rate_1��5����I       6%�	I�Zf���A�*;


total_loss��
A

error_RJP?

learning_rate_1��5�CJ�I       6%�	�Zf���A�*;


total_losse��@

error_R��R?

learning_rate_1��5Ʉ��I       6%�	 5[f���A�*;


total_loss�ʩ@

error_Rr	M?

learning_rate_1��5�k�I       6%�	�z[f���A�*;


total_loss�3�@

error_R�M?

learning_rate_1��5��I       6%�	��[f���A�*;


total_loss�]�@

error_R�#Y?

learning_rate_1��5��k�I       6%�	S\f���A�*;


total_loss2´@

error_REJU?

learning_rate_1��5:���I       6%�	(M\f���A�*;


total_loss�FA

error_R��K?

learning_rate_1��5���(I       6%�	��\f���A�*;


total_lossw��@

error_R�O?

learning_rate_1��5�2I       6%�	��\f���A�*;


total_loss3��@

error_R�~M?

learning_rate_1��5�1v�I       6%�	�]f���A�*;


total_loss�@

error_R;[?

learning_rate_1��55���I       6%�	+e]f���A�*;


total_lossl��@

error_RÑj?

learning_rate_1��5��_�I       6%�	[�]f���A�*;


total_lossض@

error_R�IP?

learning_rate_1��5�\��I       6%�	^f���A�*;


total_loss�ת@

error_R�9W?

learning_rate_1��58��I       6%�	�g^f���A�*;


total_loss�Ή@

error_R;�V?

learning_rate_1��5o�K0I       6%�	h�^f���A�*;


total_loss���@

error_R�M:?

learning_rate_1��5�t�I       6%�	��^f���A�*;


total_lossd(�@

error_R�d@?

learning_rate_1��5���;I       6%�	�8_f���A�*;


total_loss�H�@

error_Rm�H?

learning_rate_1��5ǡ�OI       6%�	I}_f���A�*;


total_loss �@

error_R�BO?

learning_rate_1��5y�w�I       6%�	y�_f���A�*;


total_loss���@

error_R��X?

learning_rate_1��5P���I       6%�	`f���A�*;


total_loss#ˏ@

error_R]=?

learning_rate_1��5�T6�I       6%�	�G`f���A�*;


total_lossL��@

error_R��U?

learning_rate_1��5a���I       6%�	f�`f���A�*;


total_loss��e@

error_R��J?

learning_rate_1��5O�P]I       6%�	��`f���A�*;


total_loss1Ԟ@

error_R�@?

learning_rate_1��5����I       6%�	!af���A�*;


total_loss,��@

error_R�{]?

learning_rate_1��5)XI       6%�	Ndaf���A�*;


total_loss�Η@

error_R�8?

learning_rate_1��5xS�^I       6%�	�af���A�*;


total_loss�ۜ@

error_RE�A?

learning_rate_1��58>e�I       6%�	��af���A�*;


total_loss�6�@

error_R{`G?

learning_rate_1��5�i��I       6%�	�Ebf���A�*;


total_loss��_@

error_R��D?

learning_rate_1��5�t�wI       6%�	B�bf���A�*;


total_lossA�@

error_R��U?

learning_rate_1��5�c��I       6%�	��bf���A�*;


total_loss��@

error_RփM?

learning_rate_1��5�j�I       6%�	!cf���A�*;


total_lossZH�@

error_R�A?

learning_rate_1��5��Y#I       6%�	_tcf���A�*;


total_loss�>�@

error_R�/h?

learning_rate_1��5�r��I       6%�	��cf���A�*;


total_losse�~@

error_R�sZ?

learning_rate_1��5��(�I       6%�	�	df���A�*;


total_loss|s�@

error_R3>?

learning_rate_1��5̌ I       6%�	�Ndf���A�*;


total_loss6}A

error_R�?K?

learning_rate_1��54ʃ�I       6%�	M�df���A�*;


total_loss��@

error_R\�E?

learning_rate_1��5���
I       6%�	��df���A�*;


total_loss]A�@

error_R�R?

learning_rate_1��5���I       6%�	�ef���A�*;


total_lossK�
A

error_R%\?

learning_rate_1��50��I       6%�	�\ef���A�*;


total_loss���@

error_Rw6H?

learning_rate_1��5$I       6%�	%�ef���A�*;


total_lossj@A

error_R��L?

learning_rate_1��5�ʣ=I       6%�	g�ef���A�*;


total_loss���@

error_R��A?

learning_rate_1��5��zI       6%�	,ff���A�*;


total_lossI��@

error_R�E?

learning_rate_1��5.yݬI       6%�	 nff���A�*;


total_loss� �@

error_RMBM?

learning_rate_1��5�<��I       6%�	!�ff���A�*;


total_loss�H�@

error_R��Q?

learning_rate_1��5\�I�I       6%�	��ff���A�*;


total_loss�}�@

error_R
�F?

learning_rate_1��5��I       6%�	&:gf���A�*;


total_loss3 �@

error_R	�G?

learning_rate_1��5��,I       6%�	}}gf���A�*;


total_loss�8�@

error_R��K?

learning_rate_1��5�){I       6%�	��gf���A�*;


total_loss���@

error_R84[?

learning_rate_1��5G���I       6%�	Thf���A�*;


total_loss]&�@

error_R(�8?

learning_rate_1��5�qI       6%�	_hf���A�*;


total_loss�j�@

error_R
�A?

learning_rate_1��5:^�I       6%�	"�hf���A�*;


total_loss�m�@

error_RE,:?

learning_rate_1��5'�=0I       6%�	��hf���A�*;


total_loss}]�@

error_R��L?

learning_rate_1��5xejI       6%�	c8if���A�*;


total_lossv�A

error_RS�P?

learning_rate_1��5A -I       6%�	�}if���A�*;


total_loss��@

error_R��G?

learning_rate_1��5��G+I       6%�	��if���A�*;


total_loss!~�@

error_R)1D?

learning_rate_1��5?�4I       6%�	�jf���A�*;


total_loss�4�@

error_R��B?

learning_rate_1��5s�f#I       6%�	�Hjf���A�*;


total_loss�f�@

error_R�=V?

learning_rate_1��5�i@I       6%�	f�jf���A�*;


total_loss�@

error_R�P?

learning_rate_1��5ĸ�I       6%�	��jf���A�*;


total_lossvp�@

error_RW�E?

learning_rate_1��5}���I       6%�	�kf���A�*;


total_lossOyA

error_Ra]W?

learning_rate_1��51n�I       6%�	rkf���A�*;


total_loss��@

error_RߤA?

learning_rate_1��5I�F<I       6%�	�kf���A�*;


total_lossi��@

error_R��Q?

learning_rate_1��5�t��I       6%�	�kf���A�*;


total_lossi��@

error_R�b?

learning_rate_1��5v59I       6%�	|Elf���A�*;


total_loss��@

error_R)ID?

learning_rate_1��5�dA8I       6%�	�lf���A�*;


total_loss���@

error_R�gl?

learning_rate_1��5��I       6%�	4�lf���A�*;


total_loss2y�@

error_R��P?

learning_rate_1��5�1_�I       6%�	�=mf���A�*;


total_loss`�@

error_R�dM?

learning_rate_1��5~��I       6%�	~�mf���A�*;


total_losst�@

error_R�nV?

learning_rate_1��5��Q�I       6%�	��mf���A�*;


total_loss�J�@

error_Rx�@?

learning_rate_1��5_���I       6%�	+5nf���A�*;


total_loss�A�@

error_Rq{H?

learning_rate_1��5��t:I       6%�	}nf���A�*;


total_lossM��@

error_R�wK?

learning_rate_1��5�n
I       6%�	y�nf���A�*;


total_loss}�@

error_Rntc?

learning_rate_1��5�I       6%�	�of���A�*;


total_loss�d�@

error_R�a?

learning_rate_1��5cb�I       6%�	�Qof���A�*;


total_loss�m�@

error_R��Q?

learning_rate_1��5J���I       6%�	'�of���A�*;


total_lossѠ�@

error_R��U?

learning_rate_1��5�N7�I       6%�	��of���A�*;


total_loss��@

error_RTkO?

learning_rate_1��5�|�WI       6%�	-'pf���A�*;


total_loss�!�@

error_R�VQ?

learning_rate_1��5!��I       6%�	kpf���A�*;


total_lossrq�@

error_R�=?

learning_rate_1��5��I       6%�	��pf���A�*;


total_lossA��@

error_R�Q?

learning_rate_1��5��6:I       6%�	-�pf���A�*;


total_lossh��@

error_R]�;?

learning_rate_1��5$�SI       6%�	8Rqf���A�*;


total_loss,�@

error_R��N?

learning_rate_1��5�!Z�I       6%�	��qf���A�*;


total_lossR�@

error_R�{@?

learning_rate_1��5f��3I       6%�	0�qf���A�*;


total_loss���@

error_RE�I?

learning_rate_1��5��.�I       6%�	�rf���A�*;


total_loss�H�@

error_R,>?

learning_rate_1��5+LI       6%�	�`rf���A�*;


total_loss��@

error_R	V?

learning_rate_1��5��I       6%�	̣rf���A�*;


total_loss�j�@

error_Rc�P?

learning_rate_1��5!�I       6%�	��rf���A�*;


total_loss\��@

error_R�F?

learning_rate_1��5��6-I       6%�	+)sf���A�*;


total_lossjF�@

error_R��F?

learning_rate_1��5j���I       6%�	?msf���A�*;


total_loss���@

error_R��J?

learning_rate_1��5FnI       6%�	z�sf���A�*;


total_loss�:�@

error_R�"U?

learning_rate_1��5�T6�I       6%�	`�sf���A�*;


total_loss�O�@

error_R �O?

learning_rate_1��5���UI       6%�	9tf���A�*;


total_lossx�@

error_R�MU?

learning_rate_1��5�9^I       6%�	�|tf���A�*;


total_loss��@

error_R��b?

learning_rate_1��5�Hk�I       6%�	��tf���A�*;


total_loss���@

error_R�	G?

learning_rate_1��5��j�I       6%�	uf���A�*;


total_lossK̜@

error_R��C?

learning_rate_1��5�-+I       6%�	�Huf���A�*;


total_loss�H�@

error_R�O?

learning_rate_1��5�z�I       6%�	
�uf���A�*;


total_loss薿@

error_R��[?

learning_rate_1��5N��SI       6%�	B�uf���A�*;


total_loss��@

error_R!&K?

learning_rate_1��5�U�I       6%�	ivf���A�*;


total_loss��n@

error_R�E?

learning_rate_1��5Z*��I       6%�	[vf���A�*;


total_loss���@

error_R�a@?

learning_rate_1��5�;d�I       6%�	+�vf���A�*;


total_loss*��@

error_R��L?

learning_rate_1��5��+�I       6%�	J�vf���A�*;


total_lossS�A

error_R�Q?

learning_rate_1��5�aHbI       6%�	6'wf���A�*;


total_lossY��@

error_RS�[?

learning_rate_1��5R��I       6%�	�lwf���A�*;


total_loss V�@

error_RrSW?

learning_rate_1��5z(I       6%�	�wf���A�*;


total_lossDf�@

error_RTwB?

learning_rate_1��5��I       6%�	��wf���A�*;


total_loss@

error_R��D?

learning_rate_1��5�U��I       6%�	�=xf���A�*;


total_loss௑@

error_Rvr^?

learning_rate_1��5܎~�I       6%�	a�xf���A�*;


total_lossﺮ@

error_R39V?

learning_rate_1��5��I       6%�	��xf���A�*;


total_losso?�@

error_R�_?

learning_rate_1��5�"��I       6%�	vyf���A�*;


total_loss�ZZ@

error_RO�H?

learning_rate_1��5�iI       6%�	�Iyf���A�*;


total_loss�3A

error_R�]A?

learning_rate_1��5P��HI       6%�	�yf���A�*;


total_lossԨ�@

error_RB<?

learning_rate_1��5�xA�I       6%�	.�yf���A�*;


total_loss�JA

error_R�Q?

learning_rate_1��5H*0I       6%�	�zf���A�*;


total_loss	]�@

error_R�X^?

learning_rate_1��5��I       6%�	Xzf���A�*;


total_lossߒ@

error_RO?

learning_rate_1��5N�6I       6%�	��zf���A�*;


total_loss�+�@

error_R�rC?

learning_rate_1��5U5XI       6%�	`�zf���A�*;


total_loss��l@

error_R8�Q?

learning_rate_1��5
�?�I       6%�	�8{f���A�*;


total_loss�d�@

error_R�GG?

learning_rate_1��5z���I       6%�	��{f���A�*;


total_lossd��@

error_RO?

learning_rate_1��5-V��I       6%�	��{f���A�*;


total_losse�@

error_R�@?

learning_rate_1��5��lNI       6%�	�|f���A�*;


total_loss:�@

error_Rs�=?

learning_rate_1��5�h��I       6%�	QT|f���A�*;


total_loss�9�@

error_R�V?

learning_rate_1��5:��QI       6%�	R�|f���A�*;


total_lossM��@

error_R��V?

learning_rate_1��5֍�yI       6%�	�|f���A�*;


total_loss]"�@

error_R�&C?

learning_rate_1��5��˴I       6%�	[}f���A�*;


total_loss���@

error_RxM?

learning_rate_1��5S���I       6%�	bd}f���A�*;


total_lossF�@

error_R�	H?

learning_rate_1��5�G�I       6%�	��}f���A�*;


total_lossqģ@

error_RZM?

learning_rate_1��5���I       6%�	� ~f���A�*;


total_loss�A�@

error_R@??

learning_rate_1��5�0��I       6%�	o~f���A�*;


total_loss�Y�@

error_RʛF?

learning_rate_1��5B�JI       6%�	��~f���A�*;


total_lossѿ�@

error_RќS?

learning_rate_1��5�2:XI       6%�	 f���A�*;


total_loss,ȸ@

error_R�*\?

learning_rate_1��5* fNI       6%�	Ef���A�*;


total_lossf�A

error_R�9K?

learning_rate_1��5yc>I       6%�	ކf���A�*;


total_lossf٨@

error_R߯F?

learning_rate_1��5����I       6%�	��f���A�*;


total_loss�
�@

error_R�LA?

learning_rate_1��5o{* I       6%�	�
�f���A�*;


total_lossX�r@

error_R��J?

learning_rate_1��5g:2I       6%�	�L�f���A�*;


total_loss�#�@

error_Rf�N?

learning_rate_1��5	R��I       6%�	b��f���A�*;


total_loss&uA

error_R��T?

learning_rate_1��5��V6I       6%�	g��f���A�*;


total_loss�3�@

error_Rl�K?

learning_rate_1��5%��I       6%�	�ڃf���A�*;


total_loss A

error_R1	D?

learning_rate_1��5�+I       6%�	L&�f���A�*;


total_lossD�A

error_R��Q?

learning_rate_1��5`�s�I       6%�	(n�f���A�*;


total_loss���@

error_R�3C?

learning_rate_1��5��~�I       6%�	���f���A�*;


total_loss�G�@

error_R
?V?

learning_rate_1��55c0	I       6%�	h��f���A�*;


total_loss�@

error_R,hG?

learning_rate_1��5���I       6%�	�=�f���A�*;


total_lossҠ�@

error_R�O?

learning_rate_1��5d^I       6%�	��f���A�*;


total_loss_g�@

error_RC?

learning_rate_1��5�S�I       6%�	Åf���A�*;


total_loss��@

error_Rf�T?

learning_rate_1��5�,m�I       6%�	��f���A�*;


total_lossL�@

error_R#�V?

learning_rate_1��5K���I       6%�	3H�f���A�*;


total_loss$�@

error_R��R?

learning_rate_1��5��`�I       6%�	銆f���A�*;


total_loss��@

error_R�W?

learning_rate_1��5�r��I       6%�	MІf���A�*;


total_lossv��@

error_R$B?

learning_rate_1��5�t��I       6%�	��f���A�*;


total_loss֝�@

error_R�U?

learning_rate_1��5�`eaI       6%�	$a�f���A�*;


total_loss\c�@

error_R;�H?

learning_rate_1��52��fI       6%�	���f���A�*;


total_loss3|�@

error_R)�@?

learning_rate_1��5�
�I       6%�	:�f���A�*;


total_lossL
�@

error_R`G?

learning_rate_1��5�{6I       6%�	60�f���A�*;


total_lossxB�@

error_R��T?

learning_rate_1��5����I       6%�	r�f���A�*;


total_lossև�@

error_R�NQ?

learning_rate_1��5L�F�I       6%�	��f���A�*;


total_losse�A

error_R�2a?

learning_rate_1��5� ��I       6%�	���f���A�*;


total_loss���@

error_Ra=S?

learning_rate_1��5`�xsI       6%�	gL�f���A�*;


total_loss\��@

error_R�[V?

learning_rate_1��5B��-I       6%�	���f���A�*;


total_lossWk�@

error_R�K?

learning_rate_1��5�W�=I       6%�	I܉f���A�*;


total_loss.�@

error_RG?

learning_rate_1��5���I       6%�	\%�f���A�*;


total_loss��@

error_R�!X?

learning_rate_1��5����I       6%�	�h�f���A�*;


total_loss��	A

error_RejF?

learning_rate_1��55Ix,I       6%�	���f���A�*;


total_loss�#�@

error_R��L?

learning_rate_1��5�k2I       6%�	O��f���A�*;


total_loss,Σ@

error_R��8?

learning_rate_1��5X*'�I       6%�	3Q�f���A�*;


total_loss
�u@

error_R� Y?

learning_rate_1��5�P�I       6%�	���f���A�*;


total_lossJ�@

error_RI�F?

learning_rate_1��5Q�XI       6%�	%�f���A�*;


total_loss�w�@

error_R]�M?

learning_rate_1��57�4I       6%�	d6�f���A�*;


total_lossRVt@

error_RvHI?

learning_rate_1��53e��I       6%�	���f���A�*;


total_loss诇@

error_RO�=?

learning_rate_1��5�wI       6%�	i�f���A�*;


total_loss���@

error_R�E?

learning_rate_1��5�ZI       6%�	�3�f���A�*;


total_loss��@

error_R�qD?

learning_rate_1��5�SI       6%�	���f���A�*;


total_loss1*�@

error_R�RG?

learning_rate_1��5לI       6%�	6�f���A�*;


total_loss��@

error_R�EX?

learning_rate_1��5���cI       6%�	�,�f���A�*;


total_loss7U�@

error_R1P?

learning_rate_1��5� u�I       6%�	w�f���A�*;


total_loss�w�@

error_R&�Z?

learning_rate_1��5����I       6%�	Y�f���A�*;


total_lossX��@

error_Rd�R?

learning_rate_1��5���xI       6%�	�3�f���A�*;


total_loss ��@

error_RT�N?

learning_rate_1��5���I       6%�	�f���A�*;


total_lossq�@

error_RH�d?

learning_rate_1��5U��I       6%�	�Ώf���A�*;


total_lossᒭ@

error_R�C?

learning_rate_1��5l�I       6%�	�0�f���A�*;


total_loss�i�@

error_R.bI?

learning_rate_1��5i��DI       6%�	�w�f���A�*;


total_loss<�@

error_R%�T?

learning_rate_1��5�T[�I       6%�	���f���A�*;


total_loss�@

error_R8EI?

learning_rate_1��5���I       6%�	��f���A�*;


total_loss.J�@

error_Rt�H?

learning_rate_1��5]"�I       6%�	pr�f���A�*;


total_loss$��@

error_R��d?

learning_rate_1��5��nI       6%�	�Ǒf���A�*;


total_lossױ@

error_R��>?

learning_rate_1��5��SI       6%�	�f���A�*;


total_loss�&�@

error_R`�G?

learning_rate_1��547�I       6%�	�_�f���A�*;


total_lossϔ�@

error_R�A?

learning_rate_1��5juzBI       6%�	ӥ�f���A�*;


total_lossp�@

error_RZD@?

learning_rate_1��5��t4I       6%�	��f���A�*;


total_loss&B�@

error_R@�D?

learning_rate_1��5�(��I       6%�	13�f���A�*;


total_loss�d�@

error_RJ�H?

learning_rate_1��5�n��I       6%�	���f���A�*;


total_lossjU�@

error_RW�T?

learning_rate_1��5�o��I       6%�	Lޓf���A�*;


total_loss�.�@

error_R�a?

learning_rate_1��5ޭ��I       6%�	�@�f���A�*;


total_loss\dv@

error_RֶV?

learning_rate_1��5���_I       6%�	G��f���A�*;


total_loss̓@

error_R�<V?

learning_rate_1��5W�"�I       6%�	�ϔf���A�*;


total_loss�B�@

error_Rz"O?

learning_rate_1��5�J9/I       6%�	*�f���A�*;


total_loss�@

error_R�(`?

learning_rate_1��5a��9I       6%�	�]�f���A�*;


total_loss.��@

error_R�[?

learning_rate_1��5�HI       6%�	���f���A�*;


total_loss��@

error_R�Cc?

learning_rate_1��59�.I       6%�	&�f���A�*;


total_losscU(A

error_Rq=Z?

learning_rate_1��5"�hI       6%�		-�f���A�*;


total_loss�5}@

error_R�hN?

learning_rate_1��5�1��I       6%�	Dr�f���A�*;


total_loss�+�@

error_RXK?

learning_rate_1��5�*��I       6%�	���f���A�*;


total_loss�-�@

error_R#�X?

learning_rate_1��5Ҹ��I       6%�	b��f���A�*;


total_loss���@

error_RRT?

learning_rate_1��5ʴs	I       6%�	�<�f���A�*;


total_loss��@

error_RH�U?

learning_rate_1��5��ޘI       6%�	��f���A�*;


total_loss7#�@

error_R�]?

learning_rate_1��5���NI       6%�	 Ǘf���A�*;


total_loss(U�@

error_R��K?

learning_rate_1��5N,'I       6%�	a�f���A�*;


total_loss3��@

error_R�|U?

learning_rate_1��5B| �I       6%�	�P�f���A�*;


total_lossA�@

error_R�_X?

learning_rate_1��5���I       6%�	l��f���A�*;


total_lossej�@

error_R�L?

learning_rate_1��5�%(I       6%�	Kטf���A�*;


total_loss6�@

error_R}�R?

learning_rate_1��5�J�I       6%�	��f���A�*;


total_loss��@

error_R�BC?

learning_rate_1��5$W'�I       6%�	:]�f���A�*;


total_loss��@

error_R&QI?

learning_rate_1��5v@P I       6%�	V��f���A�*;


total_loss�Ѯ@

error_R�=?

learning_rate_1��5:Q��I       6%�	��f���A�*;


total_loss%�@

error_R �B?

learning_rate_1��5T�_I       6%�	�(�f���A�*;


total_loss$�@

error_R�Y?

learning_rate_1��5PX�KI       6%�	�m�f���A�*;


total_lossoh�@

error_RET?

learning_rate_1��5�œI       6%�	u��f���A�*;


total_loss�g�@

error_R��F?

learning_rate_1��5"fOiI       6%�	���f���A�*;


total_loss�f}@

error_R��f?

learning_rate_1��5�k��I       6%�	�6�f���A�*;


total_loss�֗@

error_R#�R?

learning_rate_1��5�	�I       6%�	({�f���A�*;


total_loss�Y�@

error_RvH?

learning_rate_1��5@x�I       6%�	ž�f���A�*;


total_loss֛�@

error_R�_D?

learning_rate_1��57�T/I       6%�	~�f���A�*;


total_lossMD�@

error_Rr�]?

learning_rate_1��5���I       6%�	�L�f���A�*;


total_loss!�@

error_R��4?

learning_rate_1��5)NI       6%�	���f���A�*;


total_loss��@

error_R
eO?

learning_rate_1��5��v�I       6%�	uߜf���A�*;


total_losss��@

error_R�@?

learning_rate_1��5���LI       6%�	�$�f���A�*;


total_lossL�@

error_R�rb?

learning_rate_1��5R���I       6%�	�g�f���A�*;


total_loss�F�@

error_R�yD?

learning_rate_1��5�Z�xI       6%�	eȝf���A�*;


total_loss,��@

error_Rre?

learning_rate_1��5�A�tI       6%�	��f���A�*;


total_lossWi�@

error_RHk<?

learning_rate_1��5�ZI       6%�	7T�f���A�*;


total_loss��@

error_R�OD?

learning_rate_1��5���I       6%�	���f���A�*;


total_loss�y�@

error_R�I?

learning_rate_1��5�{7�I       6%�	Fޞf���A�*;


total_loss�է@

error_R�;?

learning_rate_1��5h���I       6%�	�*�f���A�*;


total_loss+�@

error_R@eL?

learning_rate_1��5��k{I       6%�	~p�f���A�*;


total_loss��@

error_RS?

learning_rate_1��5��uI       6%�	_��f���A�*;


total_losst.�@

error_RmdS?

learning_rate_1��5Ei�UI       6%�	���f���A�*;


total_loss��@

error_RX�H?

learning_rate_1��5ݍ�I       6%�	�?�f���A�*;


total_lossfX�@

error_R��\?

learning_rate_1��5�VE�I       6%�	���f���A�*;


total_loss$�@

error_R��G?

learning_rate_1��5���I       6%�	�̠f���A�*;


total_lossT�@

error_RM�]?

learning_rate_1��5?}��I       6%�	��f���A�*;


total_loss���@

error_R�!M?

learning_rate_1��5�n�>I       6%�	�X�f���A�*;


total_lossȪ�@

error_RR�2?

learning_rate_1��5�FItI       6%�	���f���A�*;


total_loss<�@

error_R!]?

learning_rate_1��5���{I       6%�	��f���A�*;


total_lossڞA

error_RR�Z?

learning_rate_1��5��HI       6%�	�,�f���A�*;


total_loss���@

error_R�]b?

learning_rate_1��5"��AI       6%�	y�f���A�*;


total_loss�^�@

error_R�=?

learning_rate_1��5���I       6%�	���f���A�*;


total_lossa��@

error_R��Y?

learning_rate_1��57(�aI       6%�	��f���A�*;


total_lossa�n@

error_R3�D?

learning_rate_1��5�pI       6%�	�F�f���A�*;


total_losscП@

error_R�I?

learning_rate_1��5�m�I       6%�	���f���A�*;


total_loss��@

error_Rn�K?

learning_rate_1��5��/gI       6%�	Oңf���A�*;


total_lossՋ@

error_R��@?

learning_rate_1��5�h��I       6%�	m�f���A�*;


total_loss���@

error_R3�S?

learning_rate_1��5-D�I       6%�	C^�f���A�*;


total_loss��@

error_Rd�T?

learning_rate_1��5f�hCI       6%�	饤f���A�*;


total_lossLA�@

error_RL%H?

learning_rate_1��5�|a�I       6%�	���f���A�*;


total_loss��@

error_RR�A?

learning_rate_1��5*�>I       6%�	�3�f���A�*;


total_lossҡ�@

error_RT?

learning_rate_1��5ˌ�I       6%�	�}�f���A�*;


total_loss�A�@

error_R�O?

learning_rate_1��5���I       6%�	eåf���A�*;


total_loss��@

error_RxW?

learning_rate_1��5o̾�I       6%�	a�f���A�*;


total_loss�מ@

error_R=Y?

learning_rate_1��5[���I       6%�	J�f���A�*;


total_loss��@

error_R�2c?

learning_rate_1��5��4I       6%�	��f���A�*;


total_loss�G�@

error_R�D?

learning_rate_1��5�f�I       6%�	�ۦf���A�*;


total_loss�
�@

error_RȑC?

learning_rate_1��5xtI       6%�	e�f���A�*;


total_loss���@

error_R�Q?

learning_rate_1��5Q�(�I       6%�	j�f���A�*;


total_losszC�@

error_R��R?

learning_rate_1��5]�ϚI       6%�	��f���A�*;


total_loss��@

error_R%�]?

learning_rate_1��5�KI       6%�	���f���A�*;


total_loss�9�@

error_R��]?

learning_rate_1��5n{#I       6%�	�=�f���A�*;


total_loss1�A

error_R�-N?

learning_rate_1��5���I       6%�	���f���A�*;


total_losszKy@

error_R3N?

learning_rate_1��5���I       6%�	ʨf���A�*;


total_loss_��@

error_Rx1P?

learning_rate_1��5Z� �I       6%�	p�f���A�*;


total_loss���@

error_RCU?

learning_rate_1��5��'(I       6%�	:T�f���A�*;


total_loss�W�@

error_R��P?

learning_rate_1��5_q��I       6%�	*��f���A�*;


total_loss��@

error_RdpM?

learning_rate_1��5�ЌI       6%�	٩f���A�*;


total_loss���@

error_R�A@?

learning_rate_1��5��=�I       6%�	�f���A�*;


total_lossSS
A

error_R�EP?

learning_rate_1��5B�?�I       6%�	Db�f���A�*;


total_loss�El@

error_R��Z?

learning_rate_1��5�֓�I       6%�	o��f���A�*;


total_lossQ��@

error_R�oB?

learning_rate_1��56&bI       6%�	:�f���A�*;


total_loss��@

error_R�A?

learning_rate_1��5�BI       6%�	�B�f���A�*;


total_loss_�@

error_R_-^?

learning_rate_1��5T�I       6%�	@��f���A�*;


total_loss�@

error_R��V?

learning_rate_1��5-��OI       6%�	 ��f���A�*;


total_loss`�@

error_Rq%I?

learning_rate_1��5ZI"I       6%�	)9�f���A�*;


total_lossJ��@

error_R��J?

learning_rate_1��5�Dx�I       6%�	۠�f���A�*;


total_loss�m�@

error_RVTL?

learning_rate_1��5��-I       6%�	+�f���A�*;


total_loss��@

error_R��C?

learning_rate_1��5{-I       6%�	0�f���A�*;


total_loss4v�@

error_R�ZM?

learning_rate_1��5A,��I       6%�	w�f���A�*;


total_lossm	A

error_R�SK?

learning_rate_1��5��$�I       6%�	@�f���A�*;


total_loss1�z@

error_R �@?

learning_rate_1��5�Q�MI       6%�	
9�f���A�*;


total_loss/�@

error_R��=?

learning_rate_1��5��C6I       6%�	_��f���A�*;


total_loss]��@

error_R��Q?

learning_rate_1��5���mI       6%�	FϮf���A�*;


total_loss@��@

error_R��O?

learning_rate_1��5�w�dI       6%�	��f���A�*;


total_loss��@

error_R��]?

learning_rate_1��5{��I       6%�	[e�f���A�*;


total_loss!�@

error_R��E?

learning_rate_1��5�ށPI       6%�	Z��f���A�*;


total_loss	�@

error_R-�M?

learning_rate_1��5&_��I       6%�	D��f���A�*;


total_loss��@

error_R�T?

learning_rate_1��5�=��I       6%�	�;�f���A�*;


total_loss��@

error_R�pH?

learning_rate_1��5c[�I       6%�	|��f���A�*;


total_loss�r�@

error_Rx�I?

learning_rate_1��5�E<I       6%�	�Ȱf���A�*;


total_loss$S�@

error_RMr<?

learning_rate_1��5�I8NI       6%�	>�f���A�*;


total_loss㰣@

error_R2
M?

learning_rate_1��5Y�:I       6%�	�R�f���A�*;


total_loss�$�@

error_RXlT?

learning_rate_1��5Am�mI       6%�	���f���A�*;


total_loss���@

error_R�BM?

learning_rate_1��5c>��I       6%�	�۱f���A�*;


total_loss�.�@

error_R�|H?

learning_rate_1��5�II       6%�	Q �f���A�*;


total_loss8-�@

error_Rl�B?

learning_rate_1��5��9I       6%�	
b�f���A�*;


total_loss�2A

error_R�S?

learning_rate_1��5��f�I       6%�	E��f���A�*;


total_loss��@

error_R��X?

learning_rate_1��5$�|�I       6%�	=�f���A�*;


total_loss��@

error_R��Z?

learning_rate_1��57��I       6%�	�9�f���A�*;


total_loss��@

error_R��P?

learning_rate_1��5�W�I       6%�	1�f���A�*;


total_loss��{@

error_R�LN?

learning_rate_1��5����I       6%�	Hĳf���A�*;


total_loss�̫@

error_R��V?

learning_rate_1��5�<�I       6%�	�	�f���A�*;


total_lossR��@

error_R�*X?

learning_rate_1��5�h�I       6%�	�M�f���A�*;


total_loss�~@

error_RT�N?

learning_rate_1��56rt�I       6%�	0��f���A�*;


total_loss��@

error_R�L?

learning_rate_1��5��I       6%�	�ִf���A�*;


total_loss{-�@

error_R=
C?

learning_rate_1��5���DI       6%�	��f���A�*;


total_loss;��@

error_R��L?

learning_rate_1��5�~_3I       6%�	\k�f���A�*;


total_lossΙ�@

error_R�V?

learning_rate_1��5��*I       6%�	���f���A�*;


total_loss���@

error_R�cV?

learning_rate_1��5��W<I       6%�	���f���A�*;


total_loss�H�@

error_RCy>?

learning_rate_1��5�v�I       6%�	EA�f���A�*;


total_loss�I"A

error_R��O?

learning_rate_1��5�2�{I       6%�	6��f���A�*;


total_loss���@

error_R�N^?

learning_rate_1��5@��I       6%�	'˶f���A�*;


total_loss���@

error_R�l]?

learning_rate_1��5?��I       6%�	c�f���A�*;


total_loss�"�@

error_Rc�B?

learning_rate_1��5��gI       6%�	dV�f���A�*;


total_loss�ܝ@

error_RƓK?

learning_rate_1��5��I       6%�	���f���A�*;


total_loss��@

error_R��W?

learning_rate_1��5��#I       6%�	�f���A�*;


total_lossZ��@

error_RT�;?

learning_rate_1��5{�y�I       6%�	�0�f���A�*;


total_loss�+�@

error_R{�V?

learning_rate_1��5��sI       6%�	xv�f���A�*;


total_loss7��@

error_R��4?

learning_rate_1��5z1�I       6%�	���f���A�*;


total_loss\��@

error_Rl\@?

learning_rate_1��5���I       6%�	E �f���A�*;


total_loss���@

error_Rf�??

learning_rate_1��5��nII       6%�	�X�f���A�*;


total_loss�	�@

error_R{�R?

learning_rate_1��57�+�I       6%�	�ٹf���A�*;


total_loss�N%A

error_R�X??

learning_rate_1��5��h�I       6%�	�&�f���A�*;


total_loss�`�@

error_RcLJ?

learning_rate_1��5���tI       6%�	�w�f���A�*;


total_loss�Z�@

error_RqLK?

learning_rate_1��5�
�_I       6%�	���f���A�*;


total_loss���@

error_Rs�N?

learning_rate_1��5�S�I       6%�	;
�f���A�*;


total_loss�@

error_R��N?

learning_rate_1��5�<ZI       6%�	2T�f���A�*;


total_lossv�@

error_R�Y??

learning_rate_1��5g	��I       6%�	���f���A�*;


total_loss��@

error_R�T?

learning_rate_1��5���I       6%�	e׻f���A�*;


total_loss��@

error_RS�M?

learning_rate_1��5�$w�I       6%�	��f���A�*;


total_loss�g�@

error_R��??

learning_rate_1��5���\I       6%�	[g�f���A�*;


total_loss�n�@

error_R<�??

learning_rate_1��5\@�+I       6%�	h��f���A�*;


total_loss7t�@

error_R|�7?

learning_rate_1��5N� ^I       6%�	�f���A�*;


total_loss�%�@

error_RTZL?

learning_rate_1��5H�`�I       6%�	:�f���A�*;


total_loss��@

error_R#C?

learning_rate_1��5A ��I       6%�	���f���A�*;


total_lossTS�@

error_R��R?

learning_rate_1��5 �%�I       6%�	O�f���A�*;


total_lossd�v@

error_R��J?

learning_rate_1��5Xő�I       6%�	�,�f���A�*;


total_loss��@

error_R�:?

learning_rate_1��50�e+I       6%�	�y�f���A�*;


total_lossh��@

error_R|W?

learning_rate_1��5%	hI       6%�	�žf���A�*;


total_loss-v�@

error_RWlH?

learning_rate_1��5̥8KI       6%�	��f���A�*;


total_lossOȉ@

error_R��M?

learning_rate_1��5FqN=I       6%�	�T�f���A�*;


total_lossOp�@

error_R�V?

learning_rate_1��5��%I       6%�	J��f���A�*;


total_lossw<�@

error_RL~N?

learning_rate_1��5���I       6%�	#߿f���A�*;


total_lossW��@

error_R�?6?

learning_rate_1��5�ez�I       6%�	�$�f���A�*;


total_loss���@

error_Ri�E?

learning_rate_1��5>/*I       6%�	�l�f���A�*;


total_loss_�@

error_R;�\?

learning_rate_1��5��~�I       6%�	���f���A�*;


total_loss�t�@

error_R��O?

learning_rate_1��5�j^I       6%�	���f���A�*;


total_loss|�@

error_R-�J?

learning_rate_1��5����I       6%�	�6�f���A�*;


total_loss���@

error_R�J?

learning_rate_1��5,mfdI       6%�	Yy�f���A�*;


total_loss��@

error_R;�L?

learning_rate_1��5��z�I       6%�	O��f���A�*;


total_loss3��@

error_R,K?

learning_rate_1��5�@I       6%�	��f���A�*;


total_loss �@

error_R=/_?

learning_rate_1��5�aI       6%�	�F�f���A�*;


total_lossz��@

error_R��W?

learning_rate_1��5�Z7sI       6%�	��f���A�*;


total_loss��@

error_R{'A?

learning_rate_1��5�͹�I       6%�	���f���A�*;


total_loss�o�@

error_R�W?

learning_rate_1��5u	�5I       6%�	�f���A�*;


total_lossOK�@

error_R�S?

learning_rate_1��5�׃�I       6%�	_�f���A�*;


total_lossI�@

error_RO?

learning_rate_1��5�Р�I       6%�	M��f���A�*;


total_lossi��@

error_RI�S?

learning_rate_1��5ZG)�I       6%�	E��f���A�*;


total_lossJ��@

error_RӁN?

learning_rate_1��5)��I       6%�	�2�f���A�*;


total_loss�2�@

error_R��X?

learning_rate_1��5S]�gI       6%�	�w�f���A�*;


total_loss��@

error_R��R?

learning_rate_1��5��oTI       6%�	���f���A�*;


total_loss4��@

error_RO�N?

learning_rate_1��5���I       6%�	��f���A�*;


total_loss���@

error_R(L?

learning_rate_1��5�ZI       6%�	I�f���A�*;


total_lossQ�@

error_R��I?

learning_rate_1��53�.II       6%�	(��f���A�*;


total_loss���@

error_R��K?

learning_rate_1��5��#�I       6%�	>��f���A�*;


total_loss�A

error_R1qJ?

learning_rate_1��5���I       6%�	)�f���A�*;


total_loss���@

error_Ro�D?

learning_rate_1��5@�#I       6%�	TT�f���A�*;


total_loss���@

error_R,6?

learning_rate_1��5��YI       6%�	P��f���A�*;


total_loss.�q@

error_R\�J?

learning_rate_1��5#WeI       6%�	y��f���A�*;


total_lossE��@

error_R,~\?

learning_rate_1��5@��I       6%�	�5�f���A�*;


total_loss��@

error_R/fL?

learning_rate_1��5Ds�NI       6%�	���f���A�*;


total_lossN��@

error_RkN?

learning_rate_1��5�x�/I       6%�	���f���A�*;


total_loss*%�@

error_R8K?

learning_rate_1��5g<�I       6%�	��f���A�*;


total_lossM��@

error_R�	H?

learning_rate_1��53��I       6%�	*^�f���A�*;


total_loss8��@

error_R�Y?

learning_rate_1��5	;�I       6%�		��f���A�*;


total_loss�/�@

error_Rj'[?

learning_rate_1��5_���I       6%�	k��f���A�*;


total_loss���@

error_RשP?

learning_rate_1��5�f�>I       6%�	(�f���A�*;


total_loss<r�@

error_RnP?

learning_rate_1��5�;I       6%�	�l�f���A�*;


total_loss�}�@

error_R� U?

learning_rate_1��5'��I       6%�	��f���A�*;


total_lossְ�@

error_R'Y?

learning_rate_1��5ɨ�4I       6%�	���f���A�*;


total_loss�@

error_R�fM?

learning_rate_1��5��&I       6%�	�5�f���A�*;


total_loss��@

error_R�GU?

learning_rate_1��5�?T�I       6%�	�z�f���A�*;


total_lossʙt@

error_R��<?

learning_rate_1��5'V�qI       6%�	���f���A�*;


total_loss�L�@

error_R�:F?

learning_rate_1��5��e:I       6%�	�f���A�*;


total_lossz��@

error_Rx�L?

learning_rate_1��5���I       6%�	Km�f���A�*;


total_lossJ}@

error_R;�f?

learning_rate_1��5f�ܿI       6%�	`��f���A�*;


total_loss�I�@

error_R��C?

learning_rate_1��5��pI       6%�	��f���A�*;


total_loss��A

error_RD�H?

learning_rate_1��5н��I       6%�	BN�f���A�*;


total_loss��@

error_R��c?

learning_rate_1��5���I       6%�	ڸ�f���A�*;


total_loss��@

error_R��P?

learning_rate_1��5#�EI       6%�	D��f���A�*;


total_loss�n�@

error_R8=?

learning_rate_1��5����I       6%�	�D�f���A�*;


total_loss��@

error_R�[?

learning_rate_1��5V΍kI       6%�	c��f���A�*;


total_lossx��@

error_R<8G?

learning_rate_1��5��I       6%�	���f���A�*;


total_lossh��@

error_RW�L?

learning_rate_1��5�ZI       6%�	�?�f���A�*;


total_loss,u�@

error_R��I?

learning_rate_1��5Cu�I       6%�	���f���A�*;


total_loss�ݩ@

error_R��V?

learning_rate_1��5
_�ZI       6%�	���f���A�*;


total_losszÅ@

error_R�%J?

learning_rate_1��5��1�I       6%�	@�f���A�*;


total_loss[x�@

error_R�cU?

learning_rate_1��5z�BI       6%�	�S�f���A�*;


total_lossnp�@

error_R3�I?

learning_rate_1��5J�xbI       6%�	���f���A�*;


total_loss��@

error_R��U?

learning_rate_1��5�qĨI       6%�	���f���A�*;


total_loss햹@

error_R��O?

learning_rate_1��5+��I       6%�	�"�f���A�*;


total_loss�@

error_R� -?

learning_rate_1��5�#	�I       6%�	�p�f���A�*;


total_loss�@

error_R�jN?

learning_rate_1��5wԃoI       6%�	`��f���A�*;


total_loss�p�@

error_Rs�L?

learning_rate_1��5D7�jI       6%�	���f���A�*;


total_loss
Ӫ@

error_RԢO?

learning_rate_1��5褉I       6%�	%H�f���A�*;


total_loss��@

error_R�M?

learning_rate_1��5�K��I       6%�	ō�f���A�*;


total_loss�r�@

error_R�7?

learning_rate_1��5p5(�I       6%�	[��f���A�*;


total_loss;$�@

error_R��O?

learning_rate_1��5��zPI       6%�	��f���A�*;


total_loss4*�@

error_R�)\?

learning_rate_1��5�I       6%�	Z�f���A�*;


total_loss�{�@

error_R!V?

learning_rate_1��5����I       6%�	���f���A�*;


total_lossV�@

error_Rnh?

learning_rate_1��5lwI       6%�	��f���A�*;


total_lossVܘ@

error_R�TI?

learning_rate_1��5F �I       6%�	�&�f���A�*;


total_loss��@

error_R��R?

learning_rate_1��5Ł�%I       6%�	Sl�f���A�*;


total_loss�9�@

error_R|�1?

learning_rate_1��5s��4I       6%�	_��f���A�*;


total_lossE�@

error_R��D?

learning_rate_1��5�c(I       6%�	}��f���A�*;


total_loss���@

error_RVAP?

learning_rate_1��5���I       6%�	)9�f���A�*;


total_lossa��@

error_R�If?

learning_rate_1��5X��I       6%�	M|�f���A�*;


total_loss��@

error_R�aH?

learning_rate_1��5`,��I       6%�	��f���A�*;


total_lossX��@

error_R�CL?

learning_rate_1��5�i�dI       6%�	�f���A�*;


total_loss_��@

error_R�M?

learning_rate_1��5�BRI       6%�	^I�f���A�*;


total_loss�v�@

error_RT�A?

learning_rate_1��5��I       6%�	��f���A�*;


total_loss���@

error_R��H?

learning_rate_1��5���WI       6%�	���f���A�*;


total_loss���@

error_R!�>?

learning_rate_1��5"/�I       6%�	��f���A�*;


total_loss�%�@

error_R�<?

learning_rate_1��5ח[�I       6%�	�b�f���A�*;


total_loss�t�@

error_Ro�Q?

learning_rate_1��5.^�I       6%�	=��f���A�*;


total_loss��@

error_R�qE?

learning_rate_1��5y&s0I       6%�	���f���A�*;


total_lossH��@

error_REU?

learning_rate_1��5�	�1I       6%�	z;�f���A�*;


total_loss� �@

error_R{�a?

learning_rate_1��5 D��I       6%�	C��f���A�*;


total_loss�u�@

error_R��U?

learning_rate_1��5�Z�I       6%�	��f���A�*;


total_loss���@

error_Rx�>?

learning_rate_1��5I�=cI       6%�	��f���A�*;


total_lossD��@

error_R͹K?

learning_rate_1��5��hI       6%�	sS�f���A�*;


total_loss�5�@

error_RI9?

learning_rate_1��5�߆�I       6%�	;��f���A�*;


total_loss�U�@

error_R�F?

learning_rate_1��5:v�I       6%�	���f���A�*;


total_loss���@

error_R�JY?

learning_rate_1��5!��I       6%�	#�f���A�*;


total_loss�@

error_R�T?

learning_rate_1��5��5I       6%�	�i�f���A�*;


total_loss�ݔ@

error_Rs!Z?

learning_rate_1��5��UI       6%�	Ϯ�f���A�*;


total_loss2��@

error_R1[?

learning_rate_1��5����I       6%�	���f���A�*;


total_loss�e�@

error_R�[?

learning_rate_1��5���I       6%�	c8�f���A�*;


total_loss!�@

error_RlF?

learning_rate_1��5���>I       6%�	�|�f���A�*;


total_loss8��@

error_R�S?

learning_rate_1��5���I       6%�	���f���A�*;


total_lossŚ�@

error_RJ[P?

learning_rate_1��5�yX�I       6%�	�f���A�*;


total_loss��@

error_R��C?

learning_rate_1��5P�FI       6%�	-P�f���A�*;


total_lossa�@

error_RR�??

learning_rate_1��5e��SI       6%�	���f���A�*;


total_loss!�A

error_R�jQ?

learning_rate_1��5�>�I       6%�	���f���A�*;


total_lossR��@

error_RvI?

learning_rate_1��5�;�*I       6%�	{+�f���A�*;


total_loss�c�@

error_R8LW?

learning_rate_1��50�'FI       6%�	�t�f���A�*;


total_loss� �@

error_R��W?

learning_rate_1��5�$�@I       6%�	���f���A�*;


total_loss��@

error_R�H?

learning_rate_1��5k<oI       6%�	��f���A�*;


total_loss��@

error_R��O?

learning_rate_1��5�B&I       6%�	�M�f���A�*;


total_lossAV�@

error_R�sM?

learning_rate_1��5��I       6%�	ѧ�f���A�*;


total_loss
��@

error_R!�I?

learning_rate_1��5͟:I       6%�	��f���A�*;


total_loss`��@

error_R��??

learning_rate_1��5|Cq_I       6%�	�K�f���A�*;


total_loss�F�@

error_R�:`?

learning_rate_1��5L���I       6%�	]��f���A�*;


total_lossH�@

error_R�$N?

learning_rate_1��5��{nI       6%�	2��f���A�*;


total_loss��@

error_R\�R?

learning_rate_1��5����I       6%�	n�f���A�*;


total_loss�V�@

error_R%�C?

learning_rate_1��5W�+I       6%�	d�f���A�*;


total_loss��@

error_R#lQ?

learning_rate_1��5r(Y�I       6%�	���f���A�*;


total_lossoƖ@

error_R�oV?

learning_rate_1��5��LI       6%�	���f���A�*;


total_loss��s@

error_Rl�F?

learning_rate_1��5���oI       6%�	�;�f���A�*;


total_loss�'�@

error_R�{`?

learning_rate_1��5V��YI       6%�	��f���A�*;


total_loss�Q�@

error_R@bH?

learning_rate_1��5��u�I       6%�	M��f���A�*;


total_loss�ښ@

error_R�>?

learning_rate_1��5Q��I       6%�		�f���A�*;


total_loss$�@

error_RDO?

learning_rate_1��5���JI       6%�	�S�f���A�*;


total_loss���@

error_R2�L?

learning_rate_1��5��JI       6%�	l��f���A�*;


total_lossq�y@

error_R!�J?

learning_rate_1��5���I       6%�	/��f���A�*;


total_loss��@

error_R�{o?

learning_rate_1��5�=I       6%�	�-�f���A�*;


total_losss�@

error_R��P?

learning_rate_1��5�dɆI       6%�	�z�f���A�*;


total_loss"��@

error_Rt�]?

learning_rate_1��5���yI       6%�	���f���A�*;


total_lossDǡ@

error_R�A?

learning_rate_1��5��<�I       6%�	 %�f���A�*;


total_loss���@

error_R��U?

learning_rate_1��5���I       6%�	�x�f���A�*;


total_loss1P�@

error_R�^b?

learning_rate_1��5����I       6%�	���f���A�*;


total_lossC��@

error_R3�H?

learning_rate_1��5q�/�I       6%�	��f���A�*;


total_loss��@

error_R	6^?

learning_rate_1��5��yI       6%�	5d�f���A�*;


total_loss���@

error_R�x,?

learning_rate_1��5�xsI       6%�	��f���A�*;


total_loss(�@

error_R�YI?

learning_rate_1��5��H�I       6%�	@��f���A�*;


total_losscA

error_R�$\?

learning_rate_1��5ъ�TI       6%�	~:�f���A�*;


total_lossl��@

error_R�AP?

learning_rate_1��5h6�I       6%�	��f���A�*;


total_loss$�@

error_R;�E?

learning_rate_1��5��I       6%�	���f���A�*;


total_loss���@

error_RH>B?

learning_rate_1��5�5��I       6%�	3�f���A�*;


total_loss<ո@

error_RïQ?

learning_rate_1��5%��I       6%�	+Q�f���A�*;


total_loss�@

error_R�+Q?

learning_rate_1��5rf�ZI       6%�	\��f���A�*;


total_lossXl@

error_Rs�Y?

learning_rate_1��5�C��I       6%�	���f���A�*;


total_lossw�@

error_Rl�V?

learning_rate_1��5��++I       6%�	� �f���A�*;


total_loss�x�@

error_Ri�??

learning_rate_1��5����I       6%�	�d�f���A�*;


total_loss�&�@

error_Rj�T?

learning_rate_1��5� �dI       6%�	j��f���A�*;


total_loss�@

error_R�F?

learning_rate_1��5���I       6%�	��f���A�*;


total_lossf��@

error_RQ�H?

learning_rate_1��5�׸I       6%�	lA�f���A�*;


total_loss���@

error_R 3I?

learning_rate_1��5�N�I       6%�	���f���A�*;


total_loss	��@

error_Rr�T?

learning_rate_1��5����I       6%�	(��f���A�*;


total_lossIq�@

error_R�S?

learning_rate_1��5$�<PI       6%�	X�f���A�*;


total_lossU�@

error_Rv�C?

learning_rate_1��5���I       6%�	�i�f���A�*;


total_lossλ�@

error_R��O?

learning_rate_1��5iƿI       6%�	��f���A�*;


total_losst�@

error_R��A?

learning_rate_1��5��H�I       6%�	���f���A�*;


total_lossA��@

error_RL-5?

learning_rate_1��55�CI       6%�	�?�f���A�*;


total_loss=��@

error_RThc?

learning_rate_1��5}��I       6%�	���f���A�*;


total_loss?P�@

error_R�?X?

learning_rate_1��5�pI       6%�	���f���A�*;


total_loss���@

error_R׈C?

learning_rate_1��5��fI       6%�	=�f���A�*;


total_loss�@

error_R�j?

learning_rate_1��5=;{�I       6%�	)c�f���A�*;


total_lossC��@

error_R$&Z?

learning_rate_1��5M0(I       6%�	��f���A�*;


total_loss���@

error_R vM?

learning_rate_1��5y�7�I       6%�	��f���A�*;


total_loss�_�@

error_R�3@?

learning_rate_1��5��f>I       6%�	n>�f���A�*;


total_loss���@

error_R�W?

learning_rate_1��57-c{I       6%�	��f���A�*;


total_loss��@

error_R� V?

learning_rate_1��5�sn3I       6%�	C��f���A�*;


total_loss�Gc@

error_R.I?

learning_rate_1��5sw�^I       6%�	��f���A�*;


total_loss���@

error_Rv�N?

learning_rate_1��5\�q~I       6%�	S]�f���A�*;


total_loss͛ A

error_Ri�X?

learning_rate_1��5����I       6%�	;��f���A�*;


total_loss���@

error_R�+M?

learning_rate_1��5rB�I       6%�	%�f���A�*;


total_lossz��@

error_R�W?

learning_rate_1��5$���I       6%�	�U�f���A�*;


total_loss<j�@

error_R]BC?

learning_rate_1��5��I       6%�	���f���A�*;


total_loss�f�@

error_R� M?

learning_rate_1��5R��I       6%�	���f���A�*;


total_loss�H�@

error_R]�L?

learning_rate_1��5��YI       6%�	.-�f���A�*;


total_loss��@

error_R�m;?

learning_rate_1��5�k�I       6%�	t�f���A�*;


total_loss��@

error_R��L?

learning_rate_1��5�_BI       6%�	Ҽ�f���A�*;


total_loss��@

error_R%�Z?

learning_rate_1��5J��I       6%�		�f���A�*;


total_loss3@

error_Rin7?

learning_rate_1��5�/?�I       6%�	�R�f���A�*;


total_lossAv�@

error_R�,J?

learning_rate_1��5�A/�I       6%�	F��f���A�*;


total_lossVF�@

error_RAEQ?

learning_rate_1��5���I       6%�	F��f���A�*;


total_loss���@

error_R�V?

learning_rate_1��5�GwI       6%�	T!�f���A�*;


total_loss�%l@

error_Rs^Q?

learning_rate_1��5P>ӋI       6%�	tf�f���A�*;


total_lossQ]�@

error_R;$L?

learning_rate_1��5����I       6%�	}��f���A�*;


total_lossl!�@

error_RO�Z?

learning_rate_1��5�<%3I       6%�	���f���A�*;


total_loss��@

error_R��[?

learning_rate_1��5���I       6%�	�4�f���A�*;


total_loss��@

error_RfP?

learning_rate_1��5��V\I       6%�	kz�f���A�*;


total_loss�ۮ@

error_R�N?

learning_rate_1��52_��I       6%�	���f���A�*;


total_lossȔ@

error_RT�[?

learning_rate_1��5^q\I       6%�	��f���A�*;


total_losss$�@

error_R�}^?

learning_rate_1��5z�5�I       6%�	EW�f���A�*;


total_loss��@

error_R��g?

learning_rate_1��5�Y��I       6%�	=��f���A�*;


total_loss_G�@

error_R��I?

learning_rate_1��5)��iI       6%�	���f���A�*;


total_loss,ή@

error_R��D?

learning_rate_1��5qCm�I       6%�	V0�f���A�*;


total_loss=��@

error_RڶJ?

learning_rate_1��58�a;I       6%�	�z�f���A�*;


total_loss�C�@

error_R�Hi?

learning_rate_1��5_1�I       6%�	ؿ�f���A�*;


total_loss�@

error_R?VL?

learning_rate_1��5��dI       6%�	��f���A�*;


total_loss��@

error_R�r?

learning_rate_1��5�3�I       6%�	G�f���A�*;


total_loss�ح@

error_R��`?

learning_rate_1��5#cqpI       6%�	��f���A�*;


total_lossU�@

error_RqE?

learning_rate_1��5(���I       6%�	��f���A�*;


total_loss�ژ@

error_RqjE?

learning_rate_1��5��j�I       6%�	��f���A�*;


total_loss_q�@

error_R�{P?

learning_rate_1��5��'uI       6%�	�f�f���A�*;


total_lossl��@

error_R��_?

learning_rate_1��5,�^I       6%�	���f���A�*;


total_loss�@�@

error_R�xK?

learning_rate_1��5ٚRI       6%�	��f���A�*;


total_lossɶ�@

error_R�ZM?

learning_rate_1��5���dI       6%�	ZC�f���A�*;


total_loss���@

error_RsWI?

learning_rate_1��5M��I       6%�	��f���A�*;


total_loss��l@

error_R�7M?

learning_rate_1��5._{:I       6%�	��f���A�*;


total_loss���@

error_R�>?

learning_rate_1��5�Tu�I       6%�	[�f���A�*;


total_loss�i�@

error_R��@?

learning_rate_1��5��\�I       6%�	{j�f���A�*;


total_loss��@

error_R�fE?

learning_rate_1��5]��)I       6%�	���f���A�*;


total_loss�N�@

error_R�Q?

learning_rate_1��5�Z��I       6%�	���f���A�*;


total_loss�,�@

error_R��J?

learning_rate_1��5�e��I       6%�	F�f���A�*;


total_loss�ے@

error_R��D?

learning_rate_1��55�/�I       6%�	9��f���A�*;


total_loss���@

error_R�xQ?

learning_rate_1��5���uI       6%�	���f���A�*;


total_loss���@

error_R�R\?

learning_rate_1��5n�DI       6%�	�f���A�*;


total_lossc�@

error_R��I?

learning_rate_1��5&�I       6%�	�W�f���A�*;


total_loss`�@

error_R��Q?

learning_rate_1��5���9I       6%�	ě�f���A�*;


total_lossL
A

error_RCQ?

learning_rate_1��5�}�I       6%�	���f���A�*;


total_loss��@

error_R�}M?

learning_rate_1��5���I       6%�	//�f���A�*;


total_losst+�@

error_R��T?

learning_rate_1��5-*��I       6%�	��f���A�*;


total_loss��A

error_R�R?

learning_rate_1��5�S��I       6%�	!��f���A�*;


total_loss{&{@

error_R	K?

learning_rate_1��5�/nI       6%�	h�f���A�*;


total_loss���@

error_R4Y@?

learning_rate_1��5�d�I       6%�	�X�f���A�*;


total_lossIո@

error_Rl�R?

learning_rate_1��5q9�I       6%�	��f���A�*;


total_lossN��@

error_RqNb?

learning_rate_1��5	�I       6%�	���f���A�*;


total_lossnC�@

error_Rs]_?

learning_rate_1��5�1��I       6%�	D�f���A�*;


total_loss:te@

error_Rܢ=?

learning_rate_1��5�MːI       6%�	���f���A�*;


total_loss��@

error_R�GX?

learning_rate_1��5���I       6%�	���f���A�*;


total_lossֶ�@

error_RT�N?

learning_rate_1��5�~��I       6%�	jB�f���A�*;


total_loss2 A

error_R�G?

learning_rate_1��5ڞi�I       6%�	���f���A�*;


total_loss�8�@

error_R�_?

learning_rate_1��54�n�I       6%�	���f���A�*;


total_lossf�@

error_RYP?

learning_rate_1��5��єI       6%�	q%�f���A�*;


total_loss��@

error_R��U?

learning_rate_1��5�Q�I       6%�	r�f���A�*;


total_loss/'�@

error_R��D?

learning_rate_1��5��,�I       6%�	��f���A�*;


total_lossf��@

error_R��\?

learning_rate_1��5ژ�UI       6%�	 g���A�*;


total_loss�y�@

error_Rn�U?

learning_rate_1��5S���I       6%�	�G g���A�*;


total_lossn��@

error_R�IO?

learning_rate_1��5���I       6%�	� g���A�*;


total_loss�@

error_Ri�C?

learning_rate_1��5���xI       6%�	j� g���A�*;


total_loss��y@

error_R��P?

learning_rate_1��5���DI       6%�	�5g���A�*;


total_loss/֭@

error_R�BF?

learning_rate_1��5줷�I       6%�	ozg���A�*;


total_lossJH�@

error_R�qC?

learning_rate_1��5=�I       6%�	)�g���A�*;


total_loss��@

error_RHB?

learning_rate_1��5��szI       6%�	�g���A�*;


total_lossss�@

error_R�:M?

learning_rate_1��5���uI       6%�	Kg���A�*;


total_lossLr�@

error_R\�C?

learning_rate_1��5t8.�I       6%�	^�g���A�*;


total_lossS!�@

error_Rh�>?

learning_rate_1��5i�r�I       6%�	q�g���A�*;


total_loss�c@

error_RRl@?

learning_rate_1��5��AI       6%�	�g���A�*;


total_lossA��@

error_R��_?

learning_rate_1��5� �<I       6%�	�]g���A�*;


total_loss���@

error_R �V?

learning_rate_1��5�\?nI       6%�	ئg���A�*;


total_loss��
A

error_R�7>?

learning_rate_1��5�Y�I       6%�	�g���A�*;


total_loss��@

error_R�`?

learning_rate_1��5ɿ�I       6%�	:g���A�*;


total_loss�-w@

error_R�T?

learning_rate_1��5.N4MI       6%�	e}g���A�*;


total_loss�n�@

error_RWn]?

learning_rate_1��5��tI       6%�	��g���A�*;


total_loss�m�@

error_R�7W?

learning_rate_1��5�}��I       6%�	g���A�*;


total_loss�#�@

error_R��U?

learning_rate_1��58Y��I       6%�	�Ig���A�*;


total_loss6A�@

error_R��^?

learning_rate_1��5{+F�I       6%�	��g���A�*;


total_lossA�A

error_R�J?

learning_rate_1��5����I       6%�	��g���A�*;


total_loss���@

error_Rf�S?

learning_rate_1��5ϴ�I       6%�	�g���A�*;


total_loss>��@

error_R
J=?

learning_rate_1��5/�{I       6%�	Yg���A�*;


total_loss#	�@

error_R|??

learning_rate_1��5@>M�I       6%�	|�g���A�*;


total_loss���@

error_R��L?

learning_rate_1��5�0��I       6%�	j�g���A�*;


total_loss��@

error_R�Kc?

learning_rate_1��5_H�I       6%�	�&g���A�*;


total_loss�S�@

error_R��<?

learning_rate_1��5�ĬI       6%�	�ig���A�*;


total_lossG��@

error_R&�G?

learning_rate_1��5�]3I       6%�	Ǭg���A�*;


total_lossrl�@

error_R�JM?

learning_rate_1��5	3�I       6%�	��g���A�*;


total_loss�W�@

error_R��;?

learning_rate_1��5�{�
I       6%�	�>g���A�*;


total_loss2�.A

error_R�`?

learning_rate_1��5*2yI       6%�	��g���A�*;


total_loss`-�@

error_RJ�T?

learning_rate_1��5t>L�I       6%�	��g���A�*;


total_lossm	�@

error_R͘E?

learning_rate_1��5	s��I       6%�	Q	g���A�*;


total_lossc,�@

error_R3�A?

learning_rate_1��5X-�1I       6%�	rX	g���A�*;


total_loss�&A

error_R�M?

learning_rate_1��5���I       6%�	F�	g���A�*;


total_loss*-�@

error_RD<P?

learning_rate_1��5��g�I       6%�	�	g���A�*;


total_loss�� A

error_RHTN?

learning_rate_1��5��	�I       6%�	+7
g���A�*;


total_lossTc�@

error_R �A?

learning_rate_1��5�4/�I       6%�	��
g���A�*;


total_loss$�@

error_R ^O?

learning_rate_1��5{��8I       6%�	��
g���A�*;


total_lossC7�@

error_R��J?

learning_rate_1��5}�:�I       6%�	�g���A�*;


total_loss���@

error_Rr�8?

learning_rate_1��58"�I       6%�	GPg���A�*;


total_loss|ٸ@

error_R��U?

learning_rate_1��5�S��I       6%�	Õg���A�*;


total_lossA��@

error_R-�6?

learning_rate_1��5M��1I       6%�	X�g���A�*;


total_loss�V�@

error_R�T?

learning_rate_1��5�N�I       6%�	�$g���A�*;


total_loss��@

error_R�V?

learning_rate_1��5�o��I       6%�	1og���A�*;


total_loss�@

error_R�s??

learning_rate_1��5�'��I       6%�	^�g���A�*;


total_loss��@

error_R�TF?

learning_rate_1��5&ɛ�I       6%�	Ug���A�*;


total_loss�k�@

error_R��F?

learning_rate_1��5h��I       6%�	�Ig���A�*;


total_loss�I�@

error_R��J?

learning_rate_1��5L��I       6%�	��g���A�*;


total_loss���@

error_R�tR?

learning_rate_1��5�@��I       6%�	�g���A�*;


total_loss��@

error_RW?

learning_rate_1��5�?9I       6%�	�9g���A�*;


total_losscw>A

error_R�X?

learning_rate_1��5����I       6%�	�g���A�*;


total_loss9w@

error_RC�;?

learning_rate_1��5Nny&I       6%�	�g���A�*;


total_loss�q@

error_ROI?

learning_rate_1��5̌I       6%�	Z	g���A�*;


total_loss(~�@

error_RdM?

learning_rate_1��54m�I       6%�	2Ng���A�*;


total_lossmd�@

error_R2E?

learning_rate_1��5XFʫI       6%�	��g���A�*;


total_loss,	�@

error_R�oU?

learning_rate_1��5����I       6%�	4�g���A�*;


total_loss�gv@

error_R�2S?

learning_rate_1��5�8�I       6%�	Lg���A�*;


total_loss�̈@

error_RT9?

learning_rate_1��5e���I       6%�	Sfg���A�*;


total_lossza�@

error_R�N?

learning_rate_1��50���I       6%�	@�g���A�*;


total_lossqk�@

error_R�8a?

learning_rate_1��5�U9
I       6%�	��g���A�*;


total_loss��@

error_R�[?

learning_rate_1��5N�?�I       6%�	�>g���A�*;


total_loss�x�@

error_R�.=?

learning_rate_1��5>��I       6%�	��g���A�*;


total_loss���@

error_R�RK?

learning_rate_1��5��YI       6%�	��g���A�*;


total_losswX�@

error_R lZ?

learning_rate_1��5��KPI       6%�	k	g���A�*;


total_loss�Ú@

error_R�+N?

learning_rate_1��5��I       6%�	�Lg���A�*;


total_loss�̓@

error_R(O?

learning_rate_1��5wޡI       6%�	�g���A�*;


total_loss$9~@

error_R�H?

learning_rate_1��5�H��I       6%�	Q�g���A�*;


total_loss��A

error_R��I?

learning_rate_1��5�d��I       6%�	�g���A�*;


total_lossM�@

error_R�G?

learning_rate_1��5�6��I       6%�	�gg���A�*;


total_loss���@

error_RE�X?

learning_rate_1��5f��I       6%�	|�g���A�*;


total_loss���@

error_R�K?

learning_rate_1��5��I       6%�	��g���A�*;


total_loss�0�@

error_R X?

learning_rate_1��5p��I       6%�	:;g���A�*;


total_loss6׻@

error_R�zA?

learning_rate_1��5�+:I       6%�	�g���A�*;


total_lossuv�@

error_R��J?

learning_rate_1��5�]�I       6%�	#�g���A�*;


total_loss�@

error_R=:P?

learning_rate_1��5�I       6%�	g���A�*;


total_lossx�@

error_R/TK?

learning_rate_1��5ͫ��I       6%�	Qg���A�*;


total_loss�A�@

error_R8�I?

learning_rate_1��5Ј?QI       6%�	)�g���A�*;


total_loss��@

error_R�CM?

learning_rate_1��5�liI       6%�	2�g���A�*;


total_loss@Ƽ@

error_R�mP?

learning_rate_1��5v|�tI       6%�	,g���A�*;


total_loss���@

error_R��I?

learning_rate_1��5�4z$I       6%�	�wg���A�*;


total_lossĺ�@

error_R�G?

learning_rate_1��5`�|�I       6%�	пg���A�*;


total_loss�B�@

error_Riv9?

learning_rate_1��5]��I       6%�	"g���A�*;


total_loss;ڵ@

error_RNba?

learning_rate_1��5^j�*I       6%�	Ig���A�*;


total_loss��@

error_R�XR?

learning_rate_1��5a1�<I       6%�	�g���A�*;


total_lossh�@

error_R��S?

learning_rate_1��5�n�I       6%�	H�g���A�*;


total_loss��@

error_R��b?

learning_rate_1��5�I=OI       6%�	�"g���A�*;


total_loss2!�@

error_RvS?

learning_rate_1��5��I       6%�	�ng���A�*;


total_loss��@

error_R��D?

learning_rate_1��5N�âI       6%�	��g���A�*;


total_loss6	l@

error_R�bQ?

learning_rate_1��5
4�eI       6%�	g���A�*;


total_loss�@

error_R��E?

learning_rate_1��5g��I       6%�	�Hg���A�*;


total_loss`�@

error_R��P?

learning_rate_1��5��PAI       6%�	d�g���A�*;


total_loss�_�@

error_RJ�8?

learning_rate_1��5����I       6%�	Q�g���A�*;


total_loss�3�@

error_R$�G?

learning_rate_1��5h@�I       6%�	;g���A�*;


total_loss��@

error_R�aK?

learning_rate_1��5����I       6%�	�^g���A�*;


total_loss3
�@

error_R��U?

learning_rate_1��5Ξ[�I       6%�	��g���A�*;


total_lossj��@

error_Ra�M?

learning_rate_1��5?wAvI       6%�	N�g���A�*;


total_loss�#�@

error_R3oE?

learning_rate_1��5�ґI       6%�	~Bg���A�*;


total_losscP�@

error_R��Q?

learning_rate_1��5�h�I       6%�	o�g���A�*;


total_loss7�@

error_R.�>?

learning_rate_1��5�B��I       6%�	P�g���A�*;


total_lossу�@

error_R,mA?

learning_rate_1��5h���I       6%�	�g���A�*;


total_lossf��@

error_R2�G?

learning_rate_1��5I2["I       6%�	>ag���A�*;


total_losscA

error_R�k_?

learning_rate_1��5A�alI       6%�	��g���A�*;


total_loss�A

error_RJ�T?

learning_rate_1��5{Z�nI       6%�	��g���A�*;


total_lossJ�@

error_R&�V?

learning_rate_1��5ާ�I       6%�	�:g���A�*;


total_lossE"�@

error_Rx�E?

learning_rate_1��56��I       6%�	7�g���A�*;


total_lossFq	A

error_R�Q?

learning_rate_1��5Z�I       6%�	��g���A�*;


total_loss��@

error_R �8?

learning_rate_1��5퓷%I       6%�	�.g���A�*;


total_loss���@

error_Rc�W?

learning_rate_1��55D�fI       6%�	�sg���A�*;


total_loss)P�@

error_R;�M?

learning_rate_1��5�g)�I       6%�	t�g���A�*;


total_lossL�@

error_RC�T?

learning_rate_1��5�c��I       6%�	� g���A�*;


total_loss���@

error_R�G?

learning_rate_1��5��b?I       6%�	�Fg���A�*;


total_loss��A

error_Rj�Y?

learning_rate_1��55xsHI       6%�	�g���A�*;


total_loss�I�@

error_R8�K?

learning_rate_1��5�FWI       6%�	��g���A�*;


total_loss���@

error_R��G?

learning_rate_1��5�T�I       6%�	h g���A�*;


total_lossz�@

error_R��^?

learning_rate_1��5Z�W[I       6%�	_ g���A�*;


total_loss�u�@

error_R@�F?

learning_rate_1��5���I       6%�	{� g���A�*;


total_lossu�@

error_Rx�H?

learning_rate_1��5Ч�I       6%�	Z� g���A�*;


total_loss�F�@

error_R��P?

learning_rate_1��5L��I       6%�	�2!g���A�*;


total_loss.��@

error_R�Ch?

learning_rate_1��5�C�PI       6%�	v!g���A�*;


total_loss1�@

error_RK?

learning_rate_1��5:�`I       6%�	��!g���A�*;


total_loss!��@

error_R#�>?

learning_rate_1��5�Pt[I       6%�	�"g���A�*;


total_loss�@

error_R�xL?

learning_rate_1��5w�5I       6%�	�L"g���A�*;


total_loss]�n@

error_R��O?

learning_rate_1��5	��sI       6%�	�"g���A�*;


total_lossvO�@

error_R^Q?

learning_rate_1��5GZI       6%�	k�"g���A�*;


total_loss�Y�@

error_Rn�O?

learning_rate_1��5?�5�I       6%�	##g���A�*;


total_loss���@

error_R��Q?

learning_rate_1��5�^\6I       6%�	�m#g���A�*;


total_loss��@

error_R�n@?

learning_rate_1��5�sxI       6%�	��#g���A�*;


total_lossM�@

error_RCR?

learning_rate_1��5����I       6%�	$�#g���A�*;


total_loss��@

error_R��P?

learning_rate_1��5�U�I       6%�	�=$g���A�*;


total_loss� �@

error_R�
J?

learning_rate_1��5�|�I       6%�	�$g���A�*;


total_loss��@

error_Rʏ<?

learning_rate_1��56U�I       6%�	6�$g���A�*;


total_loss�|�@

error_RCp=?

learning_rate_1��5%�I       6%�	�%g���A�*;


total_loss��@

error_R&\R?

learning_rate_1��5�g�RI       6%�	�d%g���A�*;


total_loss�A

error_Rc�R?

learning_rate_1��5YsI       6%�	��%g���A�*;


total_loss�ę@

error_Rhe^?

learning_rate_1��5;C��I       6%�	D�%g���A�*;


total_loss��@

error_RR�F?

learning_rate_1��5�\F�I       6%�	�D&g���A�*;


total_loss���@

error_R&^?

learning_rate_1��5�+�I       6%�	��&g���A�*;


total_loss=�@

error_R��E?

learning_rate_1��5��4�I       6%�	��&g���A�*;


total_loss�Ы@

error_R�$V?

learning_rate_1��5G�s�I       6%�	�'g���A�*;


total_loss���@

error_R�RE?

learning_rate_1��5�<��I       6%�	�e'g���A�*;


total_loss�8A

error_R�R?

learning_rate_1��59�g�I       6%�	©'g���A�*;


total_loss�ڻ@

error_Rm�S?

learning_rate_1��5����I       6%�	��'g���A�*;


total_loss���@

error_R.�A?

learning_rate_1��5��?tI       6%�	�2(g���A�*;


total_loss{�@

error_Ri�T?

learning_rate_1��5��I       6%�	y(g���A�*;


total_loss�A

error_R\M?

learning_rate_1��5!<I       6%�	�(g���A�*;


total_lossr��@

error_R>F?

learning_rate_1��5<X�I       6%�	D)g���A�*;


total_lossn��@

error_R��V?

learning_rate_1��5��VI       6%�	5I)g���A�*;


total_loss���@

error_Rf~N?

learning_rate_1��5�4s`I       6%�	��)g���A�*;


total_loss�A

error_RHU?

learning_rate_1��5�wEWI       6%�	U�)g���A�*;


total_loss���@

error_R�W?

learning_rate_1��5f�=I       6%�	�)*g���A�*;


total_loss4�@

error_RJ]T?

learning_rate_1��5�&$pI       6%�	6p*g���A�*;


total_lossL��@

error_Rve?

learning_rate_1��5mO� I       6%�	=�*g���A�*;


total_lossO��@

error_R��Y?

learning_rate_1��51��I       6%�	��*g���A�*;


total_loss)F�@

error_R�I?

learning_rate_1��5��^�I       6%�	�=+g���A�*;


total_loss� d@

error_RPE?

learning_rate_1��5M���I       6%�	Ʌ+g���A�*;


total_lossm�jA

error_R�C?

learning_rate_1��5O�+I       6%�	��+g���A�*;


total_loss#Z�@

error_R�"O?

learning_rate_1��5�~%�I       6%�	^,g���A�*;


total_loss���@

error_RJNU?

learning_rate_1��5C*�uI       6%�	�f,g���A�*;


total_lossrǵ@

error_ROWJ?

learning_rate_1��5��cI       6%�	ׯ,g���A�*;


total_loss���@

error_Ro�S?

learning_rate_1��5+�GI       6%�	��,g���A�*;


total_loss���@

error_R
�D?

learning_rate_1��5�\�MI       6%�	]=-g���A�*;


total_lossa�@

error_R?[R?

learning_rate_1��5)��I       6%�	��-g���A�*;


total_loss}�@

error_R��F?

learning_rate_1��5�Z�0I       6%�	
�-g���A�*;


total_loss�JA

error_RPS?

learning_rate_1��53}U^I       6%�	O1.g���A�*;


total_loss+,A

error_RL�F?

learning_rate_1��5���I       6%�	3v.g���A�*;


total_loss�u�@

error_R�u;?

learning_rate_1��5��-_I       6%�	��.g���A�*;


total_loss���@

error_R�S?

learning_rate_1��5��S�I       6%�	E�.g���A�*;


total_lossϠ�@

error_R�QI?

learning_rate_1��5|4MmI       6%�	�>/g���A�*;


total_loss�gt@

error_R�>>?

learning_rate_1��5==^I       6%�	��/g���A�*;


total_loss<�@

error_R��V?

learning_rate_1��5@�&I       6%�	��/g���A�*;


total_lossEj�@

error_RlB=?

learning_rate_1��5��I       6%�	�0g���A�*;


total_loss��@

error_R�F?

learning_rate_1��5�8 I       6%�	�W0g���A�*;


total_lossH��@

error_R܍O?

learning_rate_1��5�/��I       6%�	��0g���A�*;


total_lossE��@

error_R��H?

learning_rate_1��5���I       6%�	J�0g���A�*;


total_loss�@

error_R�rQ?

learning_rate_1��5�pbPI       6%�	�/1g���A�*;


total_lossw��@

error_R��a?

learning_rate_1��5<#z�I       6%�	s1g���A�*;


total_lossR8�@

error_R��f?

learning_rate_1��5�V�vI       6%�	��1g���A�*;


total_loss�ܠ@

error_R��K?

learning_rate_1��5�48nI       6%�	9�1g���A�*;


total_loss���@

error_R)84?

learning_rate_1��5���uI       6%�	6B2g���A�*;


total_loss��@

error_R��Y?

learning_rate_1��5�	_�I       6%�	��2g���A�*;


total_lossf�@

error_R"L?

learning_rate_1��5C�رI       6%�	�2g���A�*;


total_lossF�@

error_R�L?

learning_rate_1��5R�XI       6%�	�3g���A�*;


total_lossʾ�@

error_R��H?

learning_rate_1��50��I       6%�	\3g���A�*;


total_loss̰�@

error_R�f?

learning_rate_1��5ck�-I       6%�	�3g���A�*;


total_loss���@

error_R&�I?

learning_rate_1��5���OI       6%�	>�3g���A�*;


total_loss̊�@

error_R��\?

learning_rate_1��5�K�I       6%�	�*4g���A�*;


total_lossX��@

error_R�Y?

learning_rate_1��5y��I       6%�	�r4g���A�*;


total_lossZ��@

error_R	�S?

learning_rate_1��5ޠ?�I       6%�	�4g���A�*;


total_loss�K�@

error_Rs%b?

learning_rate_1��5�}HI       6%�	l�4g���A�*;


total_loss���@

error_R�F?

learning_rate_1��5$ie,I       6%�	K5g���A�*;


total_loss��A

error_R�2[?

learning_rate_1��5�v�^I       6%�	̑5g���A�*;


total_loss��@

error_R��A?

learning_rate_1��5���I       6%�	�5g���A�*;


total_loss�@

error_R�@?

learning_rate_1��5m&5I       6%�	�#6g���A�*;


total_loss�2�@

error_R�AD?

learning_rate_1��5�H�1I       6%�	�l6g���A�*;


total_loss���@

error_R��S?

learning_rate_1��5��j�I       6%�	:�6g���A�*;


total_loss6�A

error_R�f\?

learning_rate_1��5�A�I       6%�	��6g���A�*;


total_loss�A

error_Ri]H?

learning_rate_1��5J�t�I       6%�	A7g���A�*;


total_loss��@

error_R�dU?

learning_rate_1��5���I       6%�	�7g���A�*;


total_loss7��@

error_RZ�T?

learning_rate_1��5���)I       6%�	�7g���A�*;


total_lossƎ�@

error_R��R?

learning_rate_1��5ܰ�I       6%�	58g���A�*;


total_loss��@

error_R��O?

learning_rate_1��5��7/I       6%�	 b8g���A�*;


total_lossM�@

error_R�E?

learning_rate_1��5�`�eI       6%�	��8g���A�*;


total_loss��@

error_R� T?

learning_rate_1��5�1tI       6%�	L�8g���A�*;


total_lossW�y@

error_RM$N?

learning_rate_1��5�M@I       6%�	!49g���A�*;


total_loss��@

error_R�NL?

learning_rate_1��5�@"�I       6%�	y9g���A�*;


total_lossF3�@

error_RR =?

learning_rate_1��5���_I       6%�	�9g���A�*;


total_loss��@

error_R�	5?

learning_rate_1��5Dj�I       6%�		:g���A�*;


total_loss�=�@

error_Rs�\?

learning_rate_1��5ܴ#�I       6%�	�N:g���A�*;


total_lossƔ�@

error_R6MN?

learning_rate_1��5��kI       6%�	C�:g���A�*;


total_lossF�@

error_R��V?

learning_rate_1��5u\ I       6%�	 �:g���A�*;


total_lossM�v@

error_Ri�J?

learning_rate_1��5?�9I       6%�	d#;g���A�*;


total_lossԐ�@

error_R�N?

learning_rate_1��5��t�I       6%�	lg;g���A�*;


total_loss ʤ@

error_Rs�E?

learning_rate_1��5sV�~I       6%�	k�;g���A�*;


total_loss�r.A

error_R��A?

learning_rate_1��5���I       6%�	��;g���A�*;


total_loss�v�@

error_R}�E?

learning_rate_1��5���I       6%�	i1<g���A�*;


total_lossd'�@

error_R��a?

learning_rate_1��5�S��I       6%�	rt<g���A�*;


total_loss���@

error_R@�C?

learning_rate_1��5���{I       6%�	X�<g���A�*;


total_loss��A

error_R�P?

learning_rate_1��5Za�eI       6%�	��<g���A�*;


total_loss�n�@

error_Rd�B?

learning_rate_1��5��JI       6%�	�<=g���A�*;


total_loss�V�@

error_R�C?

learning_rate_1��5OJ9�I       6%�	�=g���A�*;


total_loss�\�@

error_R�2^?

learning_rate_1��5ױdwI       6%�	��=g���A�*;


total_loss�@

error_Ra�P?

learning_rate_1��5z��bI       6%�	/3>g���A�*;


total_loss�p�@

error_R`�C?

learning_rate_1��5 }��I       6%�	�v>g���A�*;


total_loss��@

error_RX�A?

learning_rate_1��5�a`I       6%�	��>g���A�*;


total_lossC�l@

error_RvhZ?

learning_rate_1��5*�qI       6%�	�?g���A�*;


total_loss( �@

error_R�zD?

learning_rate_1��5B.�3I       6%�	�G?g���A�*;


total_losswT�@

error_Rv<E?

learning_rate_1��5#�AmI       6%�	�?g���A�*;


total_loss?r�@

error_R��\?

learning_rate_1��5б�mI       6%�	��?g���A�*;


total_loss��@

error_R�>?

learning_rate_1��56�G�I       6%�	b@g���A�*;


total_loss�rA

error_R��L?

learning_rate_1��5����I       6%�	_@g���A�*;


total_lossB��@

error_RC�a?

learning_rate_1��5T�z9I       6%�	��@g���A�*;


total_lossɫ@

error_R3C?

learning_rate_1��5�rI       6%�	��@g���A�*;


total_loss3��@

error_R��P?

learning_rate_1��5�$�I       6%�	�3Ag���A�*;


total_loss@B�@

error_R�9T?

learning_rate_1��53:��I       6%�	l{Ag���A�*;


total_loss �@

error_R�R?

learning_rate_1��5�1bI       6%�	��Ag���A�*;


total_loss$=�@

error_RM*U?

learning_rate_1��5:;��I       6%�	�Bg���A�*;


total_loss���@

error_R�d?

learning_rate_1��5���I       6%�	"IBg���A�*;


total_loss���@

error_RE�`?

learning_rate_1��5�nrI       6%�	�Bg���A�*;


total_loss���@

error_R�1F?

learning_rate_1��5��:vI       6%�	��Bg���A�*;


total_loss���@

error_R\�G?

learning_rate_1��5�1=I       6%�	*Cg���A�*;


total_loss j�@

error_Ra|T?

learning_rate_1��52��I       6%�	�jCg���A�*;


total_lossxq�@

error_Rlw4?

learning_rate_1��5���I       6%�	߰Cg���A�*;


total_loss]��@

error_R�	A?

learning_rate_1��5W��II       6%�	�Dg���A�*;


total_loss�A

error_RSgP?

learning_rate_1��5(�'I       6%�	�RDg���A�*;


total_loss�Ƕ@

error_R@KC?

learning_rate_1��50I       6%�	�Dg���A�*;


total_lossiB�@

error_R�M?

learning_rate_1��5��	I       6%�	��Dg���A�*;


total_loss
�@

error_RߜK?

learning_rate_1��5mI       6%�	�5Eg���A�*;


total_loss�@

error_R/\?

learning_rate_1��5�^ϧI       6%�	ǃEg���A�*;


total_lossT�@

error_Rr�I?

learning_rate_1��5"ӹ�I       6%�	��Eg���A�*;


total_loss ø@

error_R��L?

learning_rate_1��5YJ��I       6%�	�Fg���A�*;


total_loss�^�@

error_R��K?

learning_rate_1��5\"%I       6%�	;^Fg���A�*;


total_loss���@

error_R�#H?

learning_rate_1��5\Xb�I       6%�	[�Fg���A�*;


total_loss,�@

error_R��F?

learning_rate_1��5��JI       6%�	K�Fg���A�*;


total_loss��@

error_Rt�9?

learning_rate_1��5����I       6%�	=(Gg���A�*;


total_loss��@

error_R7?N?

learning_rate_1��5šz�I       6%�	�kGg���A�*;


total_loss2�@

error_RC�Q?

learning_rate_1��5ckq�I       6%�	��Gg���A�*;


total_loss�R�@

error_R߇I?

learning_rate_1��5��=6I       6%�	@�Gg���A�*;


total_loss|R�@

error_R�K?

learning_rate_1��5�6I       6%�	�5Hg���A�*;


total_lossZ��@

error_R��N?

learning_rate_1��5X_
KI       6%�	�{Hg���A�*;


total_lossX]�@

error_R�D?

learning_rate_1��5$�I       6%�	��Hg���A�*;


total_loss���@

error_RvfQ?

learning_rate_1��5xN(MI       6%�	�Ig���A�*;


total_losszA�@

error_RZ'[?

learning_rate_1��5���I       6%�	,KIg���A�*;


total_loss�v�@

error_RψA?

learning_rate_1��5����I       6%�	;�Ig���A�*;


total_lossϔ�@

error_R.J?

learning_rate_1��5���I       6%�	b�Ig���A�*;


total_loss�<�@

error_R,VY?

learning_rate_1��5�t��I       6%�	�Jg���A�*;


total_lossU�@

error_R�0Q?

learning_rate_1��5� ��I       6%�	�^Jg���A�*;


total_lossN�@

error_R�a?

learning_rate_1��5���I       6%�	��Jg���A�*;


total_loss8Vs@

error_RT�N?

learning_rate_1��5�>I       6%�	�Jg���A�*;


total_lossJG�@

error_RCPV?

learning_rate_1��5;l�&I       6%�	�/Kg���A�*;


total_lossd��@

error_R��9?

learning_rate_1��5sa��I       6%�	�rKg���A�*;


total_loss<��@

error_R&mS?

learning_rate_1��5�T7I       6%�	u�Kg���A�*;


total_lossA��@

error_R��X?

learning_rate_1��5����I       6%�	��Kg���A�*;


total_loss��@

error_R�;T?

learning_rate_1��5��I       6%�	�=Lg���A�*;


total_loss��@

error_R}R?

learning_rate_1��5'�X�I       6%�	&�Lg���A�*;


total_loss&p�@

error_RѢa?

learning_rate_1��5]"M�I       6%�	��Lg���A�*;


total_loss,��@

error_R��R?

learning_rate_1��5��I       6%�	]Mg���A�*;


total_loss�ͦ@

error_R�Z?

learning_rate_1��5��>I       6%�	�UMg���A�*;


total_loss�=A

error_R�M_?

learning_rate_1��5��y$I       6%�	��Mg���A�*;


total_lossD��@

error_RM�S?

learning_rate_1��5�W��I       6%�	�Ng���A�*;


total_lossf�@

error_R4�H?

learning_rate_1��5���4I       6%�	�YNg���A�*;


total_loss�Ϫ@

error_R�b?

learning_rate_1��5�#�xI       6%�	��Ng���A�*;


total_loss@��@

error_R�G?

learning_rate_1��5(�q�I       6%�	��Ng���A�*;


total_loss"�@

error_R��b?

learning_rate_1��5�"3"I       6%�	>.Og���A�*;


total_loss�@

error_RfhK?

learning_rate_1��5t`�I       6%�	}sOg���A�*;


total_loss�a�@

error_R!�E?

learning_rate_1��5�ˣI       6%�	o�Og���A�*;


total_lossO��@

error_R�^?

learning_rate_1��5��2�I       6%�	��Og���A�*;


total_loss���@

error_RԈS?

learning_rate_1��5��=+I       6%�	�APg���A�*;


total_lossz�@

error_R��W?

learning_rate_1��5�#�
I       6%�	�Pg���A�*;


total_loss1�@

error_R%M?

learning_rate_1��5�|�!I       6%�	��Pg���A�*;


total_loss ܗ@

error_RZ�K?

learning_rate_1��5��s�I       6%�	�Qg���A�*;


total_loss�H�@

error_R*wP?

learning_rate_1��5�RXI       6%�	.^Qg���A�*;


total_losshv@

error_R��C?

learning_rate_1��5�+�VI       6%�	(�Qg���A�*;


total_loss��@

error_Rn6T?

learning_rate_1��5����I       6%�	N�Qg���A�*;


total_loss;u�@

error_R�PM?

learning_rate_1��5��'�I       6%�	K8Rg���A�*;


total_loss��@

error_R.�L?

learning_rate_1��5x��I       6%�	|Rg���A�*;


total_loss���@

error_R��V?

learning_rate_1��5�f2�I       6%�	[�Rg���A�*;


total_loss{�A

error_R��;?

learning_rate_1��5�%�I       6%�	GSg���A�*;


total_loss���@

error_RJmA?

learning_rate_1��5�1�I       6%�	dQSg���A�*;


total_loss�A

error_RAL?

learning_rate_1��5V�d�I       6%�	��Sg���A�*;


total_losstD�@

error_R��D?

learning_rate_1��5I-C�I       6%�	�Sg���A�*;


total_loss���@

error_R�Af?

learning_rate_1��5|{ �I       6%�	� Tg���A�*;


total_loss7��@

error_R;
c?

learning_rate_1��5���I       6%�	�cTg���A�*;


total_loss.|�@

error_R��S?

learning_rate_1��5¤��I       6%�	|�Tg���A�*;


total_lossi��@

error_RMH?

learning_rate_1��5�p:�I       6%�	��Tg���A�*;


total_loss�ޝ@

error_R�i?

learning_rate_1��5��I       6%�	F;Ug���A�*;


total_loss�;�@

error_RZ4G?

learning_rate_1��5�duI       6%�	�Ug���A�*;


total_loss���@

error_R�uC?

learning_rate_1��5ɒl�I       6%�	��Ug���A�*;


total_loss!ͽ@

error_R�}W?

learning_rate_1��5���I       6%�	�Vg���A�*;


total_loss�߃@

error_R�zW?

learning_rate_1��5��Z�I       6%�	�NVg���A�*;


total_loss�;�@

error_RD�f?

learning_rate_1��5����I       6%�	��Vg���A�*;


total_loss���@

error_Rj�E?

learning_rate_1��5�к�I       6%�	2�Vg���A�*;


total_lossQ˪@

error_R�ZH?

learning_rate_1��5�V�I       6%�	&Wg���A�*;


total_loss�D�@

error_RZ�]?

learning_rate_1��5�?I       6%�	G_Wg���A�*;


total_loss[�@

error_R��6?

learning_rate_1��5V�I       6%�	4�Wg���A�*;


total_loss���@

error_R�sW?

learning_rate_1��5�>cI       6%�	��Wg���A�*;


total_loss��@

error_R�%O?

learning_rate_1��5����I       6%�	�3Xg���A�*;


total_lossWă@

error_R��B?

learning_rate_1��5z�^�I       6%�	AwXg���A�*;


total_loss(�@

error_R�YH?

learning_rate_1��5����I       6%�	��Xg���A�*;


total_loss�=�@

error_R�F?

learning_rate_1��5˖ �I       6%�	�Yg���A�*;


total_lossϜ�@

error_R�_?

learning_rate_1��5_��(I       6%�	U[Yg���A�*;


total_loss>�@

error_R@]?

learning_rate_1��5���I       6%�	`�Yg���A�*;


total_loss��@

error_R��N?

learning_rate_1��5C�A>I       6%�	o�Yg���A�*;


total_lossJ�0A

error_R|o\?

learning_rate_1��5j���I       6%�	(0Zg���A�*;


total_lossH�A

error_R�*]?

learning_rate_1��5�g�I       6%�	�pZg���A�*;


total_loss!�@

error_R}�G?

learning_rate_1��5yQYI       6%�	)�Zg���A�*;


total_loss���@

error_R�[?

learning_rate_1��5�v4WI       6%�	+ [g���A�*;


total_loss���@

error_R{�Q?

learning_rate_1��5�"cpI       6%�	�B[g���A�*;


total_losss��@

error_R?�E?

learning_rate_1��5"�m�I       6%�	܊[g���A�*;


total_loss���@

error_Rs�9?

learning_rate_1��5P��I       6%�	�[g���A�*;


total_loss�b�@

error_R\�E?

learning_rate_1��5���I       6%�	�\g���A�*;


total_loss�qw@

error_R/^B?

learning_rate_1��5���I       6%�	�T\g���A�*;


total_loss�ޗ@

error_R�b?

learning_rate_1��5�b�I       6%�	�\g���A�*;


total_lossō�@

error_R�R?

learning_rate_1��5�qFI       6%�	~�\g���A�*;


total_lossr��@

error_R��N?

learning_rate_1��5��~I       6%�	� ]g���A�*;


total_lossl�@

error_R8�Q?

learning_rate_1��5d�SI       6%�	+e]g���A�*;


total_lossF�@

error_R�V?

learning_rate_1��5GD�I       6%�	�]g���A�*;


total_loss���@

error_RȻQ?

learning_rate_1��5S���I       6%�	�^g���A�*;


total_loss�6�@

error_R��J?

learning_rate_1��5d��gI       6%�	�\^g���A�*;


total_lossS��@

error_R��H?

learning_rate_1��5���I       6%�	��^g���A�*;


total_lossD�#A

error_R��O?

learning_rate_1��5�p�I       6%�	"�^g���A�*;


total_lossW�@

error_R�P?

learning_rate_1��5����I       6%�	�9_g���A�*;


total_loss%*�@

error_R7Z?

learning_rate_1��5����I       6%�	8�_g���A�*;


total_loss6��@

error_R$�V?

learning_rate_1��5�hN�I       6%�	��_g���A�*;


total_loss��@

error_R�G?

learning_rate_1��5*0��I       6%�	%`g���A�*;


total_loss�q�@

error_R�`?

learning_rate_1��5���I       6%�	�b`g���A�*;


total_loss�@

error_R��R?

learning_rate_1��5�ʁ�I       6%�	"�`g���A�*;


total_loss=�@

error_R6YO?

learning_rate_1��5$�fI       6%�	��`g���A�*;


total_loss�)�@

error_R3�B?

learning_rate_1��5��,JI       6%�	�=ag���A�*;


total_loss���@

error_R,DM?

learning_rate_1��5�$�I       6%�	t�ag���A�*;


total_loss!��@

error_R��R?

learning_rate_1��5�s"�I       6%�	h�ag���A�*;


total_loss_[�@

error_RK?

learning_rate_1��5}�c�I       6%�	�bg���A�*;


total_lossT`�@

error_R�L?

learning_rate_1��5���VI       6%�		Wbg���A�*;


total_loss�ږ@

error_Rj�=?

learning_rate_1��5pj��I       6%�	Ҟbg���A�*;


total_loss(4�@

error_R�57?

learning_rate_1��5�?I       6%�	��bg���A�*;


total_lossx�@

error_RM-V?

learning_rate_1��5�D�I       6%�	�,cg���A�*;


total_lossz��@

error_R��N?

learning_rate_1��5�i�I       6%�	�tcg���A�*;


total_loss���@

error_R�T?

learning_rate_1��59�?I       6%�	%�cg���A�*;


total_lossT�f@

error_R�iN?

learning_rate_1��5�ApI       6%�	Zdg���A�*;


total_lossE��@

error_RO�F?

learning_rate_1��5�pK�I       6%�	�Gdg���A�*;


total_loss�$�@

error_R�H_?

learning_rate_1��5�O]I       6%�	s�dg���A�*;


total_loss
H�@

error_R�mU?

learning_rate_1��5�kv�I       6%�	��dg���A�*;


total_loss�n�@

error_R-oZ?

learning_rate_1��5{��I       6%�	�eg���A�*;


total_loss�W�@

error_R��Q?

learning_rate_1��5�'��I       6%�	�Yeg���A�*;


total_loss�%�@

error_R8T?

learning_rate_1��5�l�I       6%�	�eg���A�*;


total_loss���@

error_Re�M?

learning_rate_1��5�1�I       6%�	R�eg���A�*;


total_loss�Xc@

error_R��G?

learning_rate_1��5���I       6%�	-4fg���A�*;


total_lossċ A

error_R��N?

learning_rate_1��5k�;�I       6%�	�xfg���A�*;


total_loss��@

error_R�'c?

learning_rate_1��5�ȼ�I       6%�	5�fg���A�*;


total_loss/�@

error_R�L?

learning_rate_1��53I       6%�	��fg���A�*;


total_lossm(�@

error_R�vR?

learning_rate_1��5��myI       6%�	�Cgg���A�*;


total_loss#g�@

error_Rj9?

learning_rate_1��5�Xa�I       6%�	��gg���A�*;


total_loss���@

error_R*�E?

learning_rate_1��5CYL,I       6%�	��gg���A�*;


total_lossA��@

error_R�I?

learning_rate_1��5[�r%I       6%�	}hg���A�*;


total_loss
��@

error_R��T?

learning_rate_1��5�z�!I       6%�	qUhg���A�*;


total_loss?wA

error_R��O?

learning_rate_1��5�KWI       6%�	��hg���A�*;


total_loss���@

error_RsE?

learning_rate_1��5�^��I       6%�	��hg���A�*;


total_lossh��@

error_R�C?

learning_rate_1��5��zI       6%�	G;ig���A�*;


total_loss� �@

error_R�4N?

learning_rate_1��5K8�I       6%�	��ig���A�*;


total_loss��@

error_R�	@?

learning_rate_1��5�?�HI       6%�	��ig���A�*;


total_lossVK�@

error_Rq�9?

learning_rate_1��5�,0I       6%�	kjg���A�*;


total_loss�.�@

error_R{`I?

learning_rate_1��5�Xt>I       6%�	BZjg���A�*;


total_loss{0�@

error_R��J?

learning_rate_1��5[@�dI       6%�	��jg���A�*;


total_loss�@

error_R�R?

learning_rate_1��5���I       6%�	��jg���A�*;


total_loss�B�@

error_R]pP?

learning_rate_1��5����I       6%�	.+kg���A�*;


total_loss���@

error_R_�C?

learning_rate_1��5=&'GI       6%�	�pkg���A�*;


total_loss�,�@

error_R�Y?

learning_rate_1��50$+XI       6%�	�kg���A�*;


total_loss�#�@

error_R1LX?

learning_rate_1��5��G$I       6%�	F�kg���A�*;


total_loss���@

error_R�F?

learning_rate_1��5�y��I       6%�	?lg���A�*;


total_loss3�@

error_R�Q?

learning_rate_1��5���jI       6%�	X�lg���A�*;


total_loss���@

error_Ra�@?

learning_rate_1��5���I       6%�	��lg���A�*;


total_lossD-�@

error_R��E?

learning_rate_1��5@T��I       6%�	�mg���A�*;


total_loss���@

error_R&�O?

learning_rate_1��5`$�LI       6%�	~Qmg���A�*;


total_loss
�@

error_RZ!F?

learning_rate_1��5�=1I       6%�	Šmg���A�*;


total_loss�!�@

error_R/�O?

learning_rate_1��5�
a�I       6%�	��mg���A�*;


total_loss	��@

error_R�W?

learning_rate_1��5�"ZI       6%�	Fng���A�*;


total_loss\�@

error_RT�F?

learning_rate_1��5���I       6%�	��ng���A�*;


total_lossRy�@

error_Rj�b?

learning_rate_1��5�e�oI       6%�	�ng���A�*;


total_lossx��@

error_R�J?

learning_rate_1��5��ܪI       6%�	�og���A�*;


total_loss�Ϭ@

error_R&�S?

learning_rate_1��5ZZ��I       6%�	�^og���A�*;


total_loss`:w@

error_R�nS?

learning_rate_1��5�蠸I       6%�	U�og���A�*;


total_loss6͸@

error_R�"B?

learning_rate_1��5�!O(I       6%�	��og���A�*;


total_lossW��@

error_RJ:?

learning_rate_1��5ư��I       6%�	�7pg���A�*;


total_lossc|�@

error_RdnI?

learning_rate_1��5�a�I       6%�	�}pg���A�*;


total_loss�^�@

error_R�`?

learning_rate_1��5�/�I       6%�	N�pg���A�*;


total_loss�M�@

error_R�
Q?

learning_rate_1��5�2�I       6%�	�qg���A�*;


total_lossV�@

error_R��H?

learning_rate_1��5�/��I       6%�	�Jqg���A�*;


total_loss\t�@

error_R�yI?

learning_rate_1��5�/fI       6%�	�qg���A�*;


total_loss%S�@

error_R#B?

learning_rate_1��5�=�I       6%�	8�qg���A�*;


total_loss?�@

error_R|:>?

learning_rate_1��5�p��I       6%�	�rg���A�*;


total_loss��@

error_R�N?

learning_rate_1��5�S|�I       6%�	`rg���A�*;


total_loss���@

error_R�K?

learning_rate_1��5��	�I       6%�	��rg���A�*;


total_loss|�@

error_R��H?

learning_rate_1��5(��fI       6%�	)�rg���A�*;


total_loss�@

error_RE�b?

learning_rate_1��5��1I       6%�	.sg���A�*;


total_loss��@

error_R#PR?

learning_rate_1��5_�~$I       6%�	�psg���A�*;


total_loss�ͥ@

error_REl@?

learning_rate_1��5#���I       6%�	s�sg���A�*;


total_loss��@

error_R��@?

learning_rate_1��5FtI0I       6%�	��sg���A�*;


total_loss��@

error_R�5?

learning_rate_1��5z�K2I       6%�	X<tg���A�*;


total_lossV3�@

error_R��L?

learning_rate_1��5V�
�I       6%�	>�tg���A�*;


total_lossH��@

error_R�"W?

learning_rate_1��5=H�I       6%�	��tg���A�*;


total_loss�R�@

error_RS?

learning_rate_1��5��&I       6%�	�,ug���A�*;


total_loss�9�@

error_R
#P?

learning_rate_1��5���I       6%�	��ug���A�*;


total_loss@��@

error_RF�F?

learning_rate_1��5��?I       6%�	I�ug���A�*;


total_losso�@

error_R�&_?

learning_rate_1��5/���I       6%�	�-vg���A�*;


total_loss���@

error_RH�`?

learning_rate_1��5�:�I       6%�	�yvg���A�*;


total_loss�R�@

error_RO@?

learning_rate_1��5���I       6%�	��vg���A�*;


total_lossr�@

error_RW?

learning_rate_1��5�G�I       6%�	�wg���A�*;


total_lossO�@

error_R��O?

learning_rate_1��5j��`I       6%�	0Kwg���A�*;


total_loss?ħ@

error_R��M?

learning_rate_1��5�D
I       6%�	��wg���A�*;


total_lossZ�@

error_R�I?

learning_rate_1��5*��I       6%�	��wg���A�*;


total_lossI�@

error_R8�I?

learning_rate_1��5�4e|I       6%�	�xg���A�*;


total_lossW��@

error_R� Z?

learning_rate_1��5`�+'I       6%�	�gxg���A�*;


total_loss�Ļ@

error_R6�D?

learning_rate_1��5�r�
I       6%�	=�xg���A�*;


total_loss[��@

error_RQ)T?

learning_rate_1��5)��lI       6%�	��xg���A�*;


total_loss�e�@

error_RO@?

learning_rate_1��50!-I       6%�	/:yg���A�*;


total_lossJz�@

error_RD�D?

learning_rate_1��5�γ�I       6%�	��yg���A�*;


total_loss�A

error_RWL?

learning_rate_1��5����I       6%�	Czg���A�*;


total_loss<g�@

error_R��^?

learning_rate_1��5ק��I       6%�	�jzg���A�*;


total_loss�t�@

error_R)�T?

learning_rate_1��5���I       6%�	��zg���A�*;


total_loss��@

error_R��^?

learning_rate_1��5[���I       6%�	�{g���A�*;


total_loss���@

error_R��Z?

learning_rate_1��5���I       6%�	�P{g���A�*;


total_loss���@

error_R�XX?

learning_rate_1��5�8-�I       6%�	c�{g���A�*;


total_loss�+z@

error_R�X?

learning_rate_1��5�m��I       6%�	^�{g���A�*;


total_lossϧ�@

error_R�T?

learning_rate_1��5�C�mI       6%�	�+|g���A�*;


total_lossƨ@

error_RpD?

learning_rate_1��5e��I       6%�	�m|g���A�*;


total_loss�{�@

error_R�{2?

learning_rate_1��5���I       6%�	s�|g���A�*;


total_loss�G�@

error_RJ�P?

learning_rate_1��5J��tI       6%�	<�|g���A�*;


total_lossQ1A

error_RS=W?

learning_rate_1��5��z�I       6%�	Z4}g���A�*;


total_loss�؂@

error_R�W?

learning_rate_1��5�I       6%�	Cz}g���A�*;


total_loss��@

error_R�sS?

learning_rate_1��5�r�\I       6%�	�}g���A�*;


total_lossD��@

error_R��;?

learning_rate_1��5��jI       6%�	�9~g���A�*;


total_loss���@

error_R�R?

learning_rate_1��5�8��I       6%�	֋~g���A�*;


total_loss�?�@

error_R�S?

learning_rate_1��5���I       6%�	��~g���A�*;


total_loss��@

error_R$�S?

learning_rate_1��5��
\I       6%�	�)g���A�*;


total_loss�v�@

error_R �M?

learning_rate_1��5��4�I       6%�	�mg���A�*;


total_loss���@

error_R�E<?

learning_rate_1��5@�lI       6%�	n�g���A�*;


total_loss�g�@

error_R��L?

learning_rate_1��5�/I       6%�	��g���A�*;


total_loss!��@

error_R�>H?

learning_rate_1��5���HI       6%�	�;�g���A�*;


total_lossA¼@

error_R�eQ?

learning_rate_1��5�#��I       6%�	�~�g���A�*;


total_loss���@

error_R�hb?

learning_rate_1��5��I       6%�	=Āg���A�*;


total_loss]��@

error_R��b?

learning_rate_1��5�8�I       6%�	��g���A�*;


total_loss|S�@

error_R
�E?

learning_rate_1��5(CX.I       6%�	�Q�g���A�*;


total_lossr��@

error_R��g?

learning_rate_1��5K>��I       6%�	g���A�*;


total_loss�Ŗ@

error_RIi;?

learning_rate_1��5�2TI       6%�	�߁g���A�*;


total_loss;2�@

error_ReUa?

learning_rate_1��5�cCGI       6%�	�'�g���A�*;


total_lossoS�@

error_R�K?

learning_rate_1��52��I       6%�	�l�g���A�*;


total_loss�g	A

error_RnN?

learning_rate_1��5�tY�I       6%�	Ю�g���A�*;


total_loss��@

error_R�rK?

learning_rate_1��5��(I       6%�	�g���A�*;


total_loss)��@

error_RϜI?

learning_rate_1��5!�6I       6%�	�:�g���A�*;


total_loss�?�@

error_R��I?

learning_rate_1��5��#+I       6%�	��g���A�*;


total_loss�@

error_RÓA?

learning_rate_1��5�Y&I       6%�	oǃg���A�*;


total_loss�J�@

error_R��X?

learning_rate_1��5��I       6%�	��g���A�*;


total_loss��A

error_R�gI?

learning_rate_1��5ɵI       6%�	"T�g���A�*;


total_loss�s�@

error_Ra�@?

learning_rate_1��56�w�I       6%�	���g���A�*;


total_loss`?�@

error_R�.?

learning_rate_1��5L�I       6%�	+�g���A�*;


total_lossDʤ@

error_R16T?

learning_rate_1��5	�nI       6%�	�%�g���A�*;


total_loss_��@

error_Rv�D?

learning_rate_1��5&\�I       6%�	�k�g���A�*;


total_loss�j�@

error_R=K?

learning_rate_1��5���I       6%�	���g���A�*;


total_loss2�@

error_RvuD?

learning_rate_1��5��FI       6%�	���g���A�*;


total_loss��@

error_RtP?

learning_rate_1��5K�HbI       6%�	�A�g���A�*;


total_loss`��@

error_R��B?

learning_rate_1��5���0I       6%�	���g���A�*;


total_lossF�KA

error_R��a?

learning_rate_1��5+���I       6%�	CɆg���A�*;


total_loss�^�@

error_R�AW?

learning_rate_1��5ʁ(�I       6%�	��g���A�*;


total_loss�/�@

error_Rf0Q?

learning_rate_1��5��6mI       6%�	T�g���A�*;


total_lossM�@

error_R,Y?

learning_rate_1��5^#9I       6%�	���g���A�*;


total_loss�@

error_R\qM?

learning_rate_1��5�RI       6%�	g݇g���A�*;


total_lossFڝ@

error_RJSK?

learning_rate_1��5����I       6%�	�"�g���A�*;


total_loss*�@

error_R{Z?

learning_rate_1��59��#I       6%�	Ni�g���A�*;


total_loss��@

error_Rt%G?

learning_rate_1��5���*I       6%�	1��g���A�*;


total_loss�2�@

error_Rv�J?

learning_rate_1��5D�I       6%�	��g���A�*;


total_loss���@

error_R� P?

learning_rate_1��56��I       6%�	�3�g���A�*;


total_loss���@

error_R��T?

learning_rate_1��5�Wp.I       6%�	z�g���A�*;


total_loss$��@

error_R��C?

learning_rate_1��5�W��I       6%�	Yg���A�*;


total_loss��@

error_R�4c?

learning_rate_1��5�IsI       6%�	��g���A�*;


total_loss�%�@

error_R/�??

learning_rate_1��5�Y��I       6%�	�R�g���A�*;


total_lossۣ�@

error_R��L?

learning_rate_1��5F��I       6%�	���g���A�*;


total_lossD�	A

error_R�Q?

learning_rate_1��5���I       6%�	i�g���A�*;


total_lossL`�@

error_R{�H?

learning_rate_1��5�
~�I       6%�	2�g���A�*;


total_lossF�@

error_R��E?

learning_rate_1��5��ĢI       6%�	��g���A�*;


total_lossە�@

error_R hP?

learning_rate_1��5%�DI       6%�	�ɋg���A�*;


total_loss�,�@

error_R)�K?

learning_rate_1��5���I       6%�	�g���A�*;


total_loss=uA

error_RH?

learning_rate_1��5��I       6%�	�T�g���A�*;


total_lossc[�@

error_R�3E?

learning_rate_1��57S&�I       6%�	n��g���A�*;


total_lossVX�@

error_RE�c?

learning_rate_1��5  �NI       6%�	�ڌg���A�*;


total_loss�Ϗ@

error_RI`?

learning_rate_1��5aq�I       6%�	d �g���A�*;


total_loss��@

error_R.O?

learning_rate_1��5��vI       6%�	Se�g���A�*;


total_loss}��@

error_R#�P?

learning_rate_1��5�3}`I       6%�	Ͷ�g���A�*;


total_loss6*�@

error_R�wO?

learning_rate_1��5�f_I       6%�	��g���A�*;


total_lossaS�@

error_R?5?

learning_rate_1��5L �pI       6%�	�Z�g���A�*;


total_loss�#�@

error_R@�G?

learning_rate_1��5x�9TI       6%�	g���A�*;


total_lossf��@

error_R� R?

learning_rate_1��5 �8I       6%�	��g���A�*;


total_loss���@

error_R�O<?

learning_rate_1��5�ڹ{I       6%�	7'�g���A�*;


total_loss
x�@

error_R�qO?

learning_rate_1��5�L�I       6%�	qk�g���A�*;


total_loss��@

error_R�C?

learning_rate_1��5�c�I       6%�	u��g���A�*;


total_loss�޿@

error_R1!V?

learning_rate_1��5�!I       6%�	��g���A�*;


total_lossݺ�@

error_RI�??

learning_rate_1��5�4�I       6%�	�K�g���A�*;


total_lossf��@

error_R�_?

learning_rate_1��5��XI       6%�	���g���A�*;


total_loss���@

error_R	�W?

learning_rate_1��5�Ү�I       6%�	�֐g���A�*;


total_loss*��@

error_R�gU?

learning_rate_1��5�K�I       6%�	m!�g���A�*;


total_lossm�@

error_R$V?

learning_rate_1��5B��I       6%�	Uh�g���A�*;


total_loss�΀@

error_R_gD?

learning_rate_1��5_$(I       6%�	୑g���A�*;


total_loss/��@

error_Ro�\?

learning_rate_1��5[֒(I       6%�	��g���A�*;


total_loss��@

error_R�F?

learning_rate_1��5.�L�I       6%�	)<�g���A�*;


total_lossĝ�@

error_R�r?

learning_rate_1��5�Mw4I       6%�	���g���A�*;


total_loss�m�@

error_R��]?

learning_rate_1��5�)��I       6%�	�̒g���A�*;


total_loss��@

error_R��U?

learning_rate_1��5��4�I       6%�	��g���A�*;


total_loss�@

error_RR�Y?

learning_rate_1��5�P�I       6%�	XT�g���A�*;


total_lossT��@

error_R��N?

learning_rate_1��5�h�SI       6%�	���g���A�*;


total_loss5��@

error_R�hM?

learning_rate_1��5��DI       6%�	�ޓg���A�*;


total_lossE�A

error_R�;L?

learning_rate_1��5�M,I       6%�	m"�g���A�*;


total_lossS/�@

error_R.V?

learning_rate_1��5kG؁I       6%�	2j�g���A�*;


total_loss���@

error_R��V?

learning_rate_1��5�.�)I       6%�	PДg���A�*;


total_loss�b}@

error_RR�f?

learning_rate_1��5V%jI       6%�	!�g���A�*;


total_lossj��@

error_R��R?

learning_rate_1��5�6+I       6%�	^�g���A�*;


total_loss!��@

error_RۖK?

learning_rate_1��5��I       6%�	͕g���A�*;


total_loss?u�@

error_R�zH?

learning_rate_1��5k��I       6%�	��g���A�*;


total_loss��@

error_R!gZ?

learning_rate_1��5yψI       6%�	 Z�g���A�*;


total_loss[Y�@

error_R|�k?

learning_rate_1��5c��aI       6%�	���g���A�*;


total_loss��u@

error_R��[?

learning_rate_1��5��TI       6%�	��g���A�*;


total_loss���@

error_R�KM?

learning_rate_1��5,��I       6%�	V3�g���A�*;


total_loss�}�@

error_R�E?

learning_rate_1��5`���I       6%�	#��g���A�*;


total_lossM�@

error_RV>?

learning_rate_1��5�k,�I       6%�	6ȗg���A�*;


total_lossN��@

error_R�]?

learning_rate_1��5�/�KI       6%�	��g���A�*;


total_loss�3�@

error_R8�S?

learning_rate_1��52�MI       6%�	=U�g���A�*;


total_loss
��@

error_R�2]?

learning_rate_1��5X&�I       6%�	���g���A�*;


total_loss��@

error_RظC?

learning_rate_1��5N�D$I       6%�	���g���A�*;


total_loss�U�@

error_R��C?

learning_rate_1��5K�gI       6%�	&�g���A�*;


total_lossX��@

error_RZ�U?

learning_rate_1��5�u}I       6%�	�i�g���A�*;


total_loss ܓ@

error_R��N?

learning_rate_1��5B�
�I       6%�	���g���A�*;


total_loss�R�@

error_R;@?

learning_rate_1��5���(I       6%�	��g���A�*;


total_loss�w�@

error_R�7?

learning_rate_1��5�T��I       6%�	X6�g���A�*;


total_loss��@

error_RToW?

learning_rate_1��5�ZmI       6%�	�x�g���A�*;


total_lossR��@

error_R�IR?

learning_rate_1��5�Q1sI       6%�	D��g���A�*;


total_lossOj�@

error_R_�W?

learning_rate_1��5�G1�I       6%�	�g���A�*;


total_lossWQ�@

error_R3�^?

learning_rate_1��5q�U@I       6%�	�L�g���A�*;


total_lossᦴ@

error_R��T?

learning_rate_1��576~I       6%�	���g���A�*;


total_loss�,�@

error_R�}S?

learning_rate_1��5J_܅I       6%�	�כg���A�*;


total_loss-/�@

error_R�uE?

learning_rate_1��5Z�I       6%�	��g���A�*;


total_lossw\ A

error_RN�??

learning_rate_1��5ؠ�&I       6%�	Xg�g���A�*;


total_lossS�@

error_R�bS?

learning_rate_1��5�KI       6%�	h��g���A�*;


total_loss�G4@

error_R;rG?

learning_rate_1��5�U�QI       6%�	3��g���A�*;


total_loss�H�@

error_Re�;?

learning_rate_1��5� �I       6%�	".�g���A�*;


total_loss�$�@

error_Rf�@?

learning_rate_1��5�o8I       6%�	|�g���A�*;


total_loss�v�@

error_R-�I?

learning_rate_1��5k$�I       6%�	�Ǡg���A�*;


total_loss���@

error_R��J?

learning_rate_1��5UK�I       6%�	��g���A�*;


total_loss���@

error_R�R?

learning_rate_1��5�.��I       6%�	�T�g���A�*;


total_loss��@

error_Rl[?

learning_rate_1��5��[I       6%�	���g���A�*;


total_loss:4�@

error_R�L?

learning_rate_1��5+�[�I       6%�	$ޡg���A�*;


total_loss���@

error_RC�I?

learning_rate_1��5^_�I       6%�	�"�g���A�*;


total_lossV��@

error_RF�X?

learning_rate_1��55OY�I       6%�	ah�g���A�*;


total_lossڿ�@

error_R�G?

learning_rate_1��5B$�@I       6%�	.��g���A�*;


total_loss���@

error_R�Q?

learning_rate_1��5��[�I       6%�	+��g���A�*;


total_lossF,A

error_RE�D?

learning_rate_1��5���!I       6%�	(8�g���A�*;


total_lossdT�@

error_R�VJ?

learning_rate_1��5~Q�xI       6%�	A}�g���A�*;


total_loss��@

error_R��L?

learning_rate_1��5��rdI       6%�	i��g���A�*;


total_loss�3@

error_R�zR?

learning_rate_1��5{�I�I       6%�	��g���A�*;


total_lossC�@

error_RWX?

learning_rate_1��5@���I       6%�	�E�g���A�*;


total_loss���@

error_R��[?

learning_rate_1��5W55BI       6%�	x��g���A�*;


total_loss݄�@

error_R�U?

learning_rate_1��5:vrI       6%�	�Τg���A�*;


total_lossQ��@

error_R�mO?

learning_rate_1��5Nώ�I       6%�	'�g���A�*;


total_loss6��@

error_R߷J?

learning_rate_1��5���I       6%�	Hc�g���A�*;


total_loss,��@

error_R}�G?

learning_rate_1��5C#��I       6%�	���g���A�*;


total_lossq�@

error_R�X?

learning_rate_1��5�I       6%�	��g���A�*;


total_loss@�-A

error_R�L?

learning_rate_1��5��HI       6%�	�/�g���A�*;


total_lossg�A

error_R�tS?

learning_rate_1��5�'�I       6%�	v�g���A�*;


total_loss�͵@

error_R�qO?

learning_rate_1��5Pf��I       6%�	/��g���A�*;


total_lossSj�@

error_R�P?

learning_rate_1��5�xn2I       6%�	��g���A�*;


total_loss� �@

error_R�H?

learning_rate_1��5+�>�I       6%�	�G�g���A�*;


total_loss��U@

error_R�C9?

learning_rate_1��5ʅ~aI       6%�	X��g���A�*;


total_lossI9�@

error_R��_?

learning_rate_1��5Ǯ4I       6%�	}ҧg���A�*;


total_loss��@

error_RhL?

learning_rate_1��5R��>I       6%�	'�g���A�*;


total_loss	��@

error_R��I?

learning_rate_1��5!-�I       6%�	�Y�g���A�*;


total_loss�:�@

error_RI?

learning_rate_1��5	H�I       6%�	��g���A�*;


total_loss��@

error_R��F?

learning_rate_1��53���I       6%�	��g���A�*;


total_loss��@

error_R}VF?

learning_rate_1��5�G�I       6%�	G3�g���A�*;


total_loss=|�@

error_RzWM?

learning_rate_1��5���I       6%�	6{�g���A�*;


total_loss�U�@

error_R\DL?

learning_rate_1��5]P��I       6%�	N��g���A�*;


total_loss�@

error_R�oF?

learning_rate_1��5��^�I       6%�	Q	�g���A�*;


total_lossH��@

error_R)�j?

learning_rate_1��5�@9�I       6%�	�L�g���A�*;


total_lossү�@

error_R� D?

learning_rate_1��5"���I       6%�	���g���A�*;


total_loss:(�@

error_RnP?

learning_rate_1��5���I       6%�	A��g���A�*;


total_loss,�{@

error_Rn�V?

learning_rate_1��5�-�I       6%�	�V�g���A�*;


total_lossa�@

error_R7�O?

learning_rate_1��5��I       6%�	��g���A�*;


total_loss�ı@

error_R<�I?

learning_rate_1��5�cF�I       6%�	"��g���A�*;


total_loss瑞@

error_R�T??

learning_rate_1��5���I       6%�	�q�g���A�*;


total_loss�f�@

error_R��W?

learning_rate_1��5���BI       6%�	u��g���A�*;


total_loss�<	A

error_R�S?

learning_rate_1��5F:�QI       6%�	� �g���A�*;


total_lossl��@

error_R�J?

learning_rate_1��5P���I       6%�	:{�g���A�*;


total_loss6��@

error_R��M?

learning_rate_1��5Z:"PI       6%�	��g���A�*;


total_lossӚ�@

error_R�uN?

learning_rate_1��5$��I       6%�	�]�g���A�*;


total_loss�W�@

error_R`�A?

learning_rate_1��5=*�0I       6%�	t��g���A�*;


total_loss���@

error_R
�[?

learning_rate_1��5p9I       6%�	��g���A�*;


total_lossV-�@

error_R�(Q?

learning_rate_1��5+�coI       6%�	B;�g���A�*;


total_loss�0�@

error_RZ�@?

learning_rate_1��5��'/I       6%�	���g���A�*;


total_loss2į@

error_R�E>?

learning_rate_1��5�j@OI       6%�	�Ưg���A�*;


total_loss�C�@

error_R=�C?

learning_rate_1��5�
}I       6%�	��g���A�*;


total_loss���@

error_Rq!P?

learning_rate_1��5jy��I       6%�	�y�g���A�*;


total_loss0��@

error_R1T?

learning_rate_1��5s�L?I       6%�	�ưg���A�*;


total_loss�ל@

error_RS�\?

learning_rate_1��5|T�I       6%�	��g���A�*;


total_loss�@

error_R�dS?

learning_rate_1��5R���I       6%�	�\�g���A�*;


total_loss.=z@

error_R.�D?

learning_rate_1��5��I       6%�	���g���A�*;


total_losstz@

error_R��8?

learning_rate_1��5��bMI       6%�	%�g���A�*;


total_loss6pt@

error_R.&]?

learning_rate_1��5��}I       6%�	/-�g���A�*;


total_loss '�@

error_R�IR?

learning_rate_1��5�)�#I       6%�	�r�g���A�*;


total_lossf�@

error_R�V?

learning_rate_1��5�C��I       6%�	<��g���A�*;


total_loss���@

error_R�T`?

learning_rate_1��5�pS�I       6%�	�g���A�*;


total_loss���@

error_R;#M?

learning_rate_1��5��I       6%�	M�g���A�*;


total_loss�>�@

error_R,g?

learning_rate_1��5����I       6%�	��g���A�*;


total_loss���@

error_R�/M?

learning_rate_1��5ᒬI       6%�	�g���A�*;


total_loss���@

error_R3�I?

learning_rate_1��5"�sOI       6%�	e-�g���A�*;


total_loss�D�@

error_R6wZ?

learning_rate_1��5�]YHI       6%�	s|�g���A�*;


total_loss��A

error_R�eE?

learning_rate_1��5P+sI       6%�	oݴg���A�*;


total_loss(t�@

error_R:�f?

learning_rate_1��5c��I       6%�	C(�g���A�*;


total_loss(��@

error_R��;?

learning_rate_1��5�d�6I       6%�	;j�g���A�*;


total_loss�C�@

error_R��C?

learning_rate_1��5J�ݺI       6%�	毵g���A�*;


total_loss?�@

error_R?S?

learning_rate_1��5-�|�I       6%�	y�g���A�*;


total_loss�:�@

error_R�M?

learning_rate_1��5���I       6%�	�8�g���A�*;


total_loss�*1A

error_R�\?

learning_rate_1��5s���I       6%�	}}�g���A�*;


total_loss_��@

error_RT�K?

learning_rate_1��5�@�I       6%�	V��g���A�*;


total_loss�Uw@

error_RD?

learning_rate_1��5�3��I       6%�	-�g���A�*;


total_loss(a�@

error_R��Y?

learning_rate_1��5��7:I       6%�	>H�g���A�*;


total_loss>,�@

error_Rf{T?

learning_rate_1��5�U��I       6%�	���g���A�*;


total_loss%'�@

error_Rd�M?

learning_rate_1��5�e3I       6%�	eҷg���A�*;


total_loss酨@

error_R�|@?

learning_rate_1��5��I       6%�	��g���A�*;


total_loss{�@

error_R�ZC?

learning_rate_1��5�!��I       6%�	{U�g���A�*;


total_loss��@

error_R�fH?

learning_rate_1��5�;��I       6%�	��g���A�*;


total_loss�#�@

error_RO�Q?

learning_rate_1��5���kI       6%�	�۸g���A�*;


total_loss��@

error_R�>Z?

learning_rate_1��5 0YLI       6%�	Z�g���A�*;


total_loss�\�@

error_R|<?

learning_rate_1��5k���I       6%�	'h�g���A�*;


total_loss3�@

error_R{�I?

learning_rate_1��5\K�I       6%�	x��g���A�*;


total_loss�@

error_RZ�>?

learning_rate_1��537I       6%�	��g���A�*;


total_lossr�.A

error_Rt�`?

learning_rate_1��5ݖ�I       6%�	�8�g���A�*;


total_loss���@

error_R1;]?

learning_rate_1��56�rI       6%�	x~�g���A�*;


total_loss���@

error_R�XA?

learning_rate_1��5�7[I       6%�	úg���A�*;


total_lossJ��@

error_R�_H?

learning_rate_1��5��m�I       6%�	C�g���A�*;


total_loss���@

error_R�aG?

learning_rate_1��5�f�bI       6%�	3N�g���A�*;


total_loss���@

error_R��H?

learning_rate_1��5��ifI       6%�	m��g���A� *;


total_lossd�@

error_RUS?

learning_rate_1��5׃(^I       6%�	�g���A� *;


total_loss2p�@

error_R?V?

learning_rate_1��5��I       6%�	�'�g���A� *;


total_loss�A

error_R3�\?

learning_rate_1��5��|�I       6%�	�l�g���A� *;


total_loss4�@

error_R��I?

learning_rate_1��5����I       6%�	���g���A� *;


total_loss��@

error_R�RI?

learning_rate_1��5 �E�I       6%�	��g���A� *;


total_loss�V�@

error_R�F?

learning_rate_1��5�55fI       6%�	�8�g���A� *;


total_loss;�0A

error_R[OW?

learning_rate_1��5���I       6%�	�z�g���A� *;


total_loss���@

error_Rx�H?

learning_rate_1��52��I       6%�	�۽g���A� *;


total_lossڬ�@

error_R8�H?

learning_rate_1��5z��I       6%�	�,�g���A� *;


total_lossغ@

error_R� ??

learning_rate_1��5����I       6%�	�p�g���A� *;


total_loss�;�@

error_R��4?

learning_rate_1��5��P�I       6%�	O��g���A� *;


total_loss:��@

error_R�C?

learning_rate_1��5�G�I       6%�	�g���A� *;


total_loss}E�@

error_R�R?

learning_rate_1��5����I       6%�	>F�g���A� *;


total_losshe�@

error_Rm�I?

learning_rate_1��5��mI       6%�	;��g���A� *;


total_loss͘�@

error_RfCL?

learning_rate_1��5�=�I       6%�	�˿g���A� *;


total_loss�g�@

error_RE�F?

learning_rate_1��5��(`I       6%�	��g���A� *;


total_loss=��@

error_R,(h?

learning_rate_1��5�9�I       6%�	�S�g���A� *;


total_loss���@

error_R�^E?

learning_rate_1��5?s_�I       6%�	���g���A� *;


total_loss�V�@

error_R��S?

learning_rate_1��5y���I       6%�	���g���A� *;


total_loss���@

error_RK9?

learning_rate_1��5�*�)I       6%�	k#�g���A� *;


total_loss���@

error_R/�b?

learning_rate_1��5��I       6%�	"j�g���A� *;


total_loss��@

error_Rq =?

learning_rate_1��5q�I       6%�	��g���A� *;


total_loss���@

error_Rd(Y?

learning_rate_1��5�P|(I       6%�	���g���A� *;


total_loss�S�@

error_R�e[?

learning_rate_1��5ٯ��I       6%�	I�g���A� *;


total_loss�(�@

error_RE�b?

learning_rate_1��5�}	I       6%�	̌�g���A� *;


total_loss���@

error_RH?

learning_rate_1��5k:6�I       6%�	���g���A� *;


total_lossZ�w@

error_RXN?

learning_rate_1��5�Y�I       6%�	��g���A� *;


total_loss��@

error_R�S?

learning_rate_1��5�7�=I       6%�	�Y�g���A� *;


total_lossx��@

error_R��V?

learning_rate_1��5o#��I       6%�	ǝ�g���A� *;


total_loss���@

error_R�ab?

learning_rate_1��5c0�WI       6%�	7��g���A� *;


total_loss�B�@

error_R�R?

learning_rate_1��5C�*qI       6%�	2�g���A� *;


total_loss��@

error_R�aL?

learning_rate_1��5d�FI       6%�	1z�g���A� *;


total_loss�5�@

error_R�_E?

learning_rate_1��5��-�I       6%�	���g���A� *;


total_loss�߁@

error_R
|D?

learning_rate_1��5���I       6%�	��g���A� *;


total_loss��@

error_Rt�U?

learning_rate_1��5�r�*I       6%�	]M�g���A� *;


total_loss���@

error_R �K?

learning_rate_1��5DBw`I       6%�	~��g���A� *;


total_lossO��@

error_Rt�L?

learning_rate_1��5�a�I       6%�	+��g���A� *;


total_loss̆�@

error_R �Q?

learning_rate_1��5~��I       6%�	 �g���A� *;


total_lossa��@

error_R��\?

learning_rate_1��5��gI       6%�	�i�g���A� *;


total_loss�6�@

error_R��A?

learning_rate_1��5i���I       6%�	N��g���A� *;


total_lossi�@

error_R�T?

learning_rate_1��5o���I       6%�	���g���A� *;


total_loss섺@

error_R�zg?

learning_rate_1��52`d�I       6%�	�B�g���A� *;


total_loss��@

error_Rd�W?

learning_rate_1��5�5��I       6%�	��g���A� *;


total_loss�0�@

error_R�??

learning_rate_1��5��RRI       6%�	���g���A� *;


total_loss�}@

error_RרA?

learning_rate_1��5���I       6%�	��g���A� *;


total_loss3Ox@

error_RD?

learning_rate_1��53��I       6%�	�a�g���A� *;


total_loss���@

error_Rz�H?

learning_rate_1��5�vK�I       6%�	x��g���A� *;


total_loss�͗@

error_R�PI?

learning_rate_1��5[�]_I       6%�	���g���A� *;


total_lossU<A

error_RN�`?

learning_rate_1��5S��cI       6%�	�)�g���A� *;


total_loss�@�@

error_Ra\U?

learning_rate_1��5�T�'I       6%�	o�g���A� *;


total_loss�!A

error_R�64?

learning_rate_1��5�aI       6%�	��g���A� *;


total_loss�2�@

error_R�hE?

learning_rate_1��5�q2I       6%�	���g���A� *;


total_lossQ��@

error_R_?

learning_rate_1��5[+ `I       6%�	�9�g���A� *;


total_lossᑦ@

error_R1�'?

learning_rate_1��5m�OI       6%�	��g���A� *;


total_lossA�@

error_R�g<?

learning_rate_1��5��$I       6%�	���g���A� *;


total_loss��@

error_R_5@?

learning_rate_1��5�� I       6%�	��g���A� *;


total_loss��v@

error_R�#K?

learning_rate_1��5��	I       6%�	Y�g���A� *;


total_loss�@

error_R}:E?

learning_rate_1��5KP�I       6%�	��g���A� *;


total_lossԩ�@

error_Rf�_?

learning_rate_1��5?s,jI       6%�	c��g���A� *;


total_loss���@

error_R�zS?

learning_rate_1��5Ԉ�}I       6%�	0;�g���A� *;


total_loss���@

error_R)�\?

learning_rate_1��5:\@I       6%�	D��g���A� *;


total_loss<��@

error_R�v@?

learning_rate_1��5B��MI       6%�	r��g���A� *;


total_loss���@

error_R��C?

learning_rate_1��5��5I       6%�	$�g���A� *;


total_loss�Y@

error_R?V?

learning_rate_1��5	���I       6%�	$U�g���A� *;


total_lossᬺ@

error_R�U?

learning_rate_1��5�-I       6%�	���g���A� *;


total_loss|�@

error_R��J?

learning_rate_1��5;x�I       6%�	o��g���A� *;


total_lossD+�@

error_R��U?

learning_rate_1��5�vI       6%�	@B�g���A� *;


total_loss��@

error_RZ�R?

learning_rate_1��5�u��I       6%�	d��g���A� *;


total_loss�v�@

error_ROl;?

learning_rate_1��5J��I       6%�	I��g���A� *;


total_loss��A

error_R6U?

learning_rate_1��5p�II       6%�	��g���A� *;


total_lossҲ�@

error_RM�a?

learning_rate_1��5<3,I       6%�	Z�g���A� *;


total_lossې@

error_R7�[?

learning_rate_1��5�eWI       6%�	��g���A� *;


total_loss���@

error_R�B?

learning_rate_1��5���I       6%�	���g���A� *;


total_lossѩ�@

error_R@-;?

learning_rate_1��5Y���I       6%�	4!�g���A� *;


total_loss̘�@

error_R�=?

learning_rate_1��5G掘I       6%�	Le�g���A� *;


total_loss���@

error_R��U?

learning_rate_1��5Tf��I       6%�	���g���A� *;


total_loss��@

error_R�HC?

learning_rate_1��5*���I       6%�	w��g���A� *;


total_loss�`�@

error_R�C?

learning_rate_1��5���I       6%�	
@�g���A� *;


total_lossC��@

error_R<�P?

learning_rate_1��5u�I       6%�	��g���A� *;


total_loss�ϻ@

error_RŘQ?

learning_rate_1��5ab�tI       6%�	R��g���A� *;


total_loss���@

error_R��R?

learning_rate_1��5��,I       6%�	�g���A� *;


total_loss䒏@

error_R�9?

learning_rate_1��5�3zI       6%�	�P�g���A� *;


total_loss?A

error_R
uR?

learning_rate_1��5T��I       6%�	Ж�g���A� *;


total_loss��@

error_RC6>?

learning_rate_1��5x+.I       6%�	}��g���A� *;


total_loss�ؤ@

error_R�^W?

learning_rate_1��5�� �I       6%�	V �g���A� *;


total_loss�{�@

error_R� Y?

learning_rate_1��5o�I       6%�	�e�g���A� *;


total_losse
	A

error_R�@7?

learning_rate_1��5�Z>fI       6%�	��g���A� *;


total_lossEK�@

error_R��E?

learning_rate_1��5��&�I       6%�	���g���A� *;


total_loss�׾@

error_ReW?

learning_rate_1��5?��cI       6%�	�1�g���A� *;


total_loss�'q@

error_R�J?

learning_rate_1��5x I       6%�	V{�g���A� *;


total_loss�5�@

error_R
�Y?

learning_rate_1��5>=�(I       6%�	q��g���A� *;


total_loss{��@

error_R�yH?

learning_rate_1��5�G�I       6%�	�/�g���A� *;


total_loss߈�@

error_R%�R?

learning_rate_1��5��aI       6%�	�z�g���A� *;


total_lossA

error_RO-U?

learning_rate_1��5���I       6%�	p��g���A� *;


total_loss�8�@

error_RS;Z?

learning_rate_1��5�4A�I       6%�	�$�g���A� *;


total_loss@i�@

error_Rl�>?

learning_rate_1��5�RQ�I       6%�	+j�g���A� *;


total_loss@��@

error_R!�<?

learning_rate_1��5.��I       6%�	���g���A� *;


total_loss�@

error_R��]?

learning_rate_1��5!ȃI       6%�	���g���A� *;


total_loss`�@

error_RhJ?

learning_rate_1��5���I       6%�	�>�g���A� *;


total_lossd�@

error_Rf�F?

learning_rate_1��5����I       6%�	��g���A� *;


total_loss��@

error_R_�M?

learning_rate_1��5��7I       6%�	Q��g���A� *;


total_loss�>�@

error_R��S?

learning_rate_1��5&��vI       6%�	w�g���A� *;


total_loss!
�@

error_R�"??

learning_rate_1��5�j_I       6%�	�W�g���A� *;


total_loss7]�@

error_R�^?

learning_rate_1��5	�hI       6%�	���g���A� *;


total_loss�WA

error_R�U?

learning_rate_1��5�$eI       6%�	���g���A� *;


total_lossE[�@

error_R��K?

learning_rate_1��5�/D�I       6%�	�%�g���A� *;


total_loss���@

error_RF�F?

learning_rate_1��5��I       6%�	�r�g���A� *;


total_loss�Ӛ@

error_R�N?

learning_rate_1��5*��I       6%�	ܻ�g���A� *;


total_loss�0�@

error_R��:?

learning_rate_1��5U;ҮI       6%�	��g���A� *;


total_loss!�$A

error_RsL?

learning_rate_1��5q�k�I       6%�	W�g���A� *;


total_loss�3�@

error_R��H?

learning_rate_1��5��I       6%�	ʠ�g���A� *;


total_loss%�@

error_R3�S?

learning_rate_1��5�I       6%�	���g���A� *;


total_loss��@

error_RsEB?

learning_rate_1��5y�8�I       6%�	(�g���A� *;


total_loss�+�@

error_R�N?

learning_rate_1��5�/�I       6%�	�l�g���A� *;


total_loss�H�@

error_R!�R?

learning_rate_1��5<R�I       6%�	V��g���A� *;


total_loss�&�@

error_RmKZ?

learning_rate_1��5�
;I       6%�	��g���A� *;


total_loss���@

error_R܌F?

learning_rate_1��5��8�I       6%�	<�g���A� *;


total_loss�t�@

error_R!�:?

learning_rate_1��5��I       6%�	�g���A� *;


total_loss�D�@

error_R�aF?

learning_rate_1��5�Ք�I       6%�	���g���A� *;


total_lossl��@

error_R�U?

learning_rate_1��5��ZzI       6%�		�g���A� *;


total_loss�.�@

error_R[H?

learning_rate_1��5P��I       6%�	GO�g���A� *;


total_loss͏�@

error_R��T?

learning_rate_1��50�a�I       6%�	]��g���A� *;


total_loss}X�@

error_R�<?

learning_rate_1��5����I       6%�	���g���A� *;


total_loss?��@

error_R7C?

learning_rate_1��5�_�^I       6%�	;�g���A� *;


total_loss�ڛ@

error_R{�@?

learning_rate_1��5>��<I       6%�	(~�g���A� *;


total_loss���@

error_RN�=?

learning_rate_1��5˘�I       6%�	���g���A� *;


total_lossj��@

error_R�}H?

learning_rate_1��5	�"I       6%�	-�g���A� *;


total_loss�@

error_R�F?

learning_rate_1��5�eҰI       6%�	J�g���A�!*;


total_losse՝@

error_R�J?

learning_rate_1��5�i�AI       6%�	���g���A�!*;


total_loss-J�@

error_R��l?

learning_rate_1��5P�UI       6%�	���g���A�!*;


total_lossz̜@

error_R��<?

learning_rate_1��5*�niI       6%�	��g���A�!*;


total_losse��@

error_RN@I?

learning_rate_1��5���.I       6%�	�]�g���A�!*;


total_lossX
�@

error_R�9G?

learning_rate_1��5)J�I       6%�	u��g���A�!*;


total_loss��@

error_Rc�b?

learning_rate_1��5A�LI       6%�	��g���A�!*;


total_loss��@

error_R��X?

learning_rate_1��5�^I       6%�	�0�g���A�!*;


total_loss���@

error_R�I?

learning_rate_1��5D���I       6%�	�|�g���A�!*;


total_loss�A�@

error_RHh?

learning_rate_1��5�@I       6%�	���g���A�!*;


total_loss�?�@

error_RR>?

learning_rate_1��5�>>�I       6%�	��g���A�!*;


total_lossn!�@

error_R,�P?

learning_rate_1��5�k�I       6%�	�N�g���A�!*;


total_loss�5�@

error_Ra~;?

learning_rate_1��5ȐRjI       6%�	I��g���A�!*;


total_loss���@

error_R�_?

learning_rate_1��5Z��oI       6%�	J��g���A�!*;


total_lossdS�@

error_R�CH?

learning_rate_1��5sz�FI       6%�	��g���A�!*;


total_loss��@

error_R�pR?

learning_rate_1��5�0�I       6%�	^�g���A�!*;


total_loss�ĕ@

error_RZIX?

learning_rate_1��5�;��I       6%�	¤�g���A�!*;


total_lossLͽ@

error_Rѡ_?

learning_rate_1��5�*�I       6%�	���g���A�!*;


total_loss�ߪ@

error_R6�C?

learning_rate_1��5���I       6%�	�3�g���A�!*;


total_lossw��@

error_RX�P?

learning_rate_1��5��I       6%�	�w�g���A�!*;


total_loss���@

error_R�J?

learning_rate_1��5�x�I       6%�	˻�g���A�!*;


total_loss��A

error_RI�W?

learning_rate_1��5b��I       6%�	� �g���A�!*;


total_loss|��@

error_Rx~<?

learning_rate_1��5ꐾII       6%�	6M�g���A�!*;


total_lossn��@

error_RWvH?

learning_rate_1��5�k/&I       6%�	���g���A�!*;


total_loss�2�@

error_R*L?

learning_rate_1��5ٴu�I       6%�	.��g���A�!*;


total_loss��@

error_R�M?

learning_rate_1��5���I       6%�	�%�g���A�!*;


total_loss�|�@

error_Rv�G?

learning_rate_1��5Dt��I       6%�	mk�g���A�!*;


total_loss���@

error_R� O?

learning_rate_1��5�V��I       6%�	ݳ�g���A�!*;


total_loss��@

error_RW�b?

learning_rate_1��5ܦ�}I       6%�	���g���A�!*;


total_loss���@

error_R�-V?

learning_rate_1��5f�X�I       6%�	�E�g���A�!*;


total_loss[4~@

error_R8�=?

learning_rate_1��5�uIpI       6%�	���g���A�!*;


total_loss��@

error_R�c?

learning_rate_1��5��XI       6%�	��g���A�!*;


total_loss�"�@

error_RH�>?

learning_rate_1��5髄�I       6%�	"�g���A�!*;


total_loss���@

error_R�^L?

learning_rate_1��5�xI       6%�	Ig�g���A�!*;


total_lossN�@

error_R�mF?

learning_rate_1��5��4I       6%�	+��g���A�!*;


total_lossNv�@

error_Rv�e?

learning_rate_1��5�p5�I       6%�	��g���A�!*;


total_lossl�@

error_R�:?

learning_rate_1��5]���I       6%�	.3�g���A�!*;


total_loss���@

error_RNlY?

learning_rate_1��5g�E�I       6%�	{�g���A�!*;


total_lossx��@

error_R�,R?

learning_rate_1��5@h�I       6%�	���g���A�!*;


total_lossuH�@

error_R��M?

learning_rate_1��5�LRI       6%�	��g���A�!*;


total_loss�w�@

error_R/yS?

learning_rate_1��5WvI�I       6%�	K�g���A�!*;


total_loss�p�@

error_R��F?

learning_rate_1��5��`�I       6%�	���g���A�!*;


total_lossd��@

error_RhM?

learning_rate_1��5F7I       6%�	���g���A�!*;


total_loss3,�@

error_R�a?

learning_rate_1��5Ź�GI       6%�	v�g���A�!*;


total_lossC�@

error_R�Q?

learning_rate_1��5��I       6%�	�`�g���A�!*;


total_loss)з@

error_RssR?

learning_rate_1��54���I       6%�	��g���A�!*;


total_loss;��@

error_RV$^?

learning_rate_1��5{��[I       6%�	���g���A�!*;


total_lossv��@

error_R�U?

learning_rate_1��5�	��I       6%�	�<�g���A�!*;


total_loss�@

error_R�{T?

learning_rate_1��5�EM�I       6%�	ԋ�g���A�!*;


total_loss��@

error_R8�F?

learning_rate_1��5�r9I       6%�	R��g���A�!*;


total_lossc��@

error_R�BB?

learning_rate_1��5����I       6%�	��g���A�!*;


total_loss�T�@

error_R�<P?

learning_rate_1��5���%I       6%�	De�g���A�!*;


total_loss�ͮ@

error_R��M?

learning_rate_1��5|e+�I       6%�	���g���A�!*;


total_lossM9�@

error_R��S?

learning_rate_1��5[lXBI       6%�	z�g���A�!*;


total_loss���@

error_RE8Y?

learning_rate_1��5�c�I       6%�	>R�g���A�!*;


total_lossaͽ@

error_R�!O?

learning_rate_1��5ܺ&�I       6%�	Ӗ�g���A�!*;


total_loss��@

error_R{�C?

learning_rate_1��5���_I       6%�	���g���A�!*;


total_loss���@

error_R߬S?

learning_rate_1��5z�I       6%�	f �g���A�!*;


total_loss�%�@

error_R{�Z?

learning_rate_1��5G6�I       6%�	f�g���A�!*;


total_loss;�@

error_Rv�G?

learning_rate_1��5��1LI       6%�	x��g���A�!*;


total_lossT��@

error_Rߢ\?

learning_rate_1��5�%I       6%�	���g���A�!*;


total_loss�y�@

error_R)B;?

learning_rate_1��5�`
\I       6%�	4�g���A�!*;


total_lossSL�@

error_R%TO?

learning_rate_1��5�Ō�I       6%�	Pw�g���A�!*;


total_loss��@

error_R:*T?

learning_rate_1��5a�XI       6%�	���g���A�!*;


total_lossȽ@

error_RqgE?

learning_rate_1��5O'W�I       6%�	��g���A�!*;


total_loss@��@

error_R�QY?

learning_rate_1��5����I       6%�	F�g���A�!*;


total_loss���@

error_R:c?

learning_rate_1��5Y�I       6%�		��g���A�!*;


total_lossCt�@

error_R�&:?

learning_rate_1��5����I       6%�	���g���A�!*;


total_loss��@

error_R-T?

learning_rate_1��5GR�I       6%�	�!�g���A�!*;


total_lossD��@

error_RX�??

learning_rate_1��5ڀj�I       6%�	�e�g���A�!*;


total_loss�2�@

error_RF�R?

learning_rate_1��5��%I       6%�	��g���A�!*;


total_loss���@

error_Ri�X?

learning_rate_1��5 "�I       6%�	y��g���A�!*;


total_loss�U�@

error_R�F?

learning_rate_1��5Ӫ�I       6%�	�:�g���A�!*;


total_loss��3@

error_R�iG?

learning_rate_1��5\>��I       6%�	B��g���A�!*;


total_loss@V�@

error_R��=?

learning_rate_1��5D�SI       6%�	���g���A�!*;


total_lossc)�@

error_R\X?

learning_rate_1��5HqMFI       6%�	�g���A�!*;


total_loss�z@

error_R�B?

learning_rate_1��5P��I       6%�	�^�g���A�!*;


total_loss�Ր@

error_R�i[?

learning_rate_1��5v��YI       6%�	ü�g���A�!*;


total_lossnڟ@

error_R&Gb?

learning_rate_1��5�TNiI       6%�	Q�g���A�!*;


total_lossZ!�@

error_R��I?

learning_rate_1��5�S��I       6%�	RO�g���A�!*;


total_losss$�@

error_R3�^?

learning_rate_1��5���I       6%�	&��g���A�!*;


total_loss�p�@

error_R��@?

learning_rate_1��5;�93I       6%�	���g���A�!*;


total_loss��@

error_R�pU?

learning_rate_1��5E�I       6%�	�8�g���A�!*;


total_loss�t�@

error_R��q?

learning_rate_1��5ϱ��I       6%�	��g���A�!*;


total_lossr�@

error_Ri�K?

learning_rate_1��5F�I       6%�	+��g���A�!*;


total_loss1|@

error_R%�V?

learning_rate_1��5|*�DI       6%�	B�g���A�!*;


total_lossf�@

error_R�N?

learning_rate_1��5}<'�I       6%�	fZ�g���A�!*;


total_loss�x�@

error_R3�W?

learning_rate_1��5V�{RI       6%�	6��g���A�!*;


total_loss�,�@

error_R�(B?

learning_rate_1��5�Z7~I       6%�	��g���A�!*;


total_loss�@

error_R�:?

learning_rate_1��5�Y�I       6%�	�0�g���A�!*;


total_loss�y�@

error_R�>7?

learning_rate_1��5�FGI       6%�	v�g���A�!*;


total_lossA��@

error_Rz�F?

learning_rate_1��5DI       6%�	���g���A�!*;


total_loss� A

error_R�L?

learning_rate_1��5J�_�I       6%�	H �g���A�!*;


total_lossͦ�@

error_R�K?

learning_rate_1��5RE3I       6%�	�C�g���A�!*;


total_loss�o�@

error_R��`?

learning_rate_1��5���@I       6%�	��g���A�!*;


total_lossZ;�@

error_R|�;?

learning_rate_1��5���I       6%�	_��g���A�!*;


total_loss�p�@

error_R�H?

learning_rate_1��5�PI       6%�	x�g���A�!*;


total_loss���@

error_R�QI?

learning_rate_1��5��TnI       6%�	�b�g���A�!*;


total_lossv��@

error_R��P?

learning_rate_1��56c��I       6%�	���g���A�!*;


total_loss�h�@

error_R.xU?

learning_rate_1��5$�U�I       6%�	~��g���A�!*;


total_loss�@

error_RnTB?

learning_rate_1��5����I       6%�	I7�g���A�!*;


total_lossry�@

error_R�-Q?

learning_rate_1��5��=I       6%�	4y�g���A�!*;


total_loss("�@

error_RaaX?

learning_rate_1��5;�KI       6%�	Һ�g���A�!*;


total_loss��@

error_R�O?

learning_rate_1��5�(�I       6%�	��g���A�!*;


total_loss���@

error_R#P?

learning_rate_1��5��/�I       6%�	D�g���A�!*;


total_loss�l�@

error_R׿J?

learning_rate_1��5�&�I       6%�	_��g���A�!*;


total_loss�Ô@

error_R7C?

learning_rate_1��5�@�I       6%�	���g���A�!*;


total_lossQx�@

error_R�PR?

learning_rate_1��5�	0�I       6%�	��g���A�!*;


total_loss�d@

error_R�+H?

learning_rate_1��5����I       6%�	�N�g���A�!*;


total_loss��@

error_R@mC?

learning_rate_1��5����I       6%�	O��g���A�!*;


total_loss�;�@

error_R�@?

learning_rate_1��5^f�dI       6%�	y��g���A�!*;


total_lossa��@

error_Rd�H?

learning_rate_1��5{7�6I       6%�	?�g���A�!*;


total_loss48�@

error_RZ�W?

learning_rate_1��5�|�I       6%�	C��g���A�!*;


total_loss;�@

error_RߵR?

learning_rate_1��5A�e�I       6%�	f��g���A�!*;


total_loss�E�@

error_RߋI?

learning_rate_1��5vo�I       6%�	s�g���A�!*;


total_lossã�@

error_R�_?

learning_rate_1��5��Q�I       6%�	VX�g���A�!*;


total_loss@

error_R�UA?

learning_rate_1��5�V2I       6%�	��g���A�!*;


total_loss_�@

error_R��_?

learning_rate_1��5B�@�I       6%�	���g���A�!*;


total_loss�
�@

error_RV�V?

learning_rate_1��59f3I       6%�	�$ h���A�!*;


total_loss�%�@

error_R�S?

learning_rate_1��56$m�I       6%�	o h���A�!*;


total_loss�T�@

error_RwLX?

learning_rate_1��5�EOI       6%�	�� h���A�!*;


total_lossۘ�@

error_R:�C?

learning_rate_1��5��I       6%�	�� h���A�!*;


total_loss�LA

error_R��O?

learning_rate_1��5�+�UI       6%�	VIh���A�!*;


total_loss�{�@

error_R�A`?

learning_rate_1��5��I       6%�	�h���A�!*;


total_lossl�@

error_R��O?

learning_rate_1��5�,D.I       6%�	^�h���A�!*;


total_loss���@

error_R�P,?

learning_rate_1��5���I       6%�	�!h���A�!*;


total_lossV�@

error_R&�=?

learning_rate_1��5�m�I       6%�	�fh���A�!*;


total_lossX��@

error_R�V?

learning_rate_1��5f(ɾI       6%�	��h���A�!*;


total_lossZI�@

error_R�U?

learning_rate_1��5K��I       6%�	��h���A�"*;


total_lossXR�@

error_RyP?

learning_rate_1��5���1I       6%�	E4h���A�"*;


total_lossS3�@

error_R��V?

learning_rate_1��5��I       6%�	�vh���A�"*;


total_lossꟌ@

error_RİR?

learning_rate_1��5����I       6%�	��h���A�"*;


total_loss�@

error_R�C?

learning_rate_1��5q�3I       6%�	�h���A�"*;


total_loss��@

error_R8	E?

learning_rate_1��5p��I       6%�	AAh���A�"*;


total_loss�s�@

error_R�W?

learning_rate_1��5�[.$I       6%�	P�h���A�"*;


total_loss,�@

error_RTZ?

learning_rate_1��5���-I       6%�	��h���A�"*;


total_loss�>�@

error_RR�g?

learning_rate_1��5��zI       6%�	�
h���A�"*;


total_loss�g�@

error_R.ib?

learning_rate_1��5�t�I       6%�		Lh���A�"*;


total_lossq��@

error_R�M?

learning_rate_1��5g��I       6%�	Αh���A�"*;


total_loss}x�@

error_R��R?

learning_rate_1��5Ķ�#I       6%�	�h���A�"*;


total_loss�k@

error_RL�M?

learning_rate_1��5��9I       6%�	�h���A�"*;


total_loss�s@

error_R&�9?

learning_rate_1��5��ܮI       6%�	1_h���A�"*;


total_loss�@

error_RzBQ?

learning_rate_1��5+bMI       6%�	��h���A�"*;


total_loss�{�@

error_R�6p?

learning_rate_1��5���I       6%�	Y�h���A�"*;


total_loss��@

error_R�tK?

learning_rate_1��5f��lI       6%�	;*h���A�"*;


total_loss���@

error_RE5Q?

learning_rate_1��5�C8�I       6%�	�qh���A�"*;


total_loss_�@

error_R}�V?

learning_rate_1��5�(�I       6%�	8�h���A�"*;


total_loss�@

error_R�IK?

learning_rate_1��5�G1I       6%�	�h���A�"*;


total_loss?c�@

error_R�J?

learning_rate_1��5{��}I       6%�	qHh���A�"*;


total_loss=S�@

error_Rz�V?

learning_rate_1��5��W�I       6%�	��h���A�"*;


total_loss�|�@

error_R�TL?

learning_rate_1��5���mI       6%�	~�h���A�"*;


total_loss84�@

error_R��N?

learning_rate_1��5O GI       6%�	"(	h���A�"*;


total_lossV��@

error_R��L?

learning_rate_1��52��I       6%�	]j	h���A�"*;


total_lossa�@

error_R��6?

learning_rate_1��5��X�I       6%�	]�	h���A�"*;


total_loss(ˡ@

error_R�7?

learning_rate_1��5�E2�I       6%�	�	h���A�"*;


total_loss�{�@

error_RW�C?

learning_rate_1��58��I       6%�	l=
h���A�"*;


total_loss.��@

error_RAR?

learning_rate_1��5	?3I       6%�	��
h���A�"*;


total_lossn��@

error_R@�N?

learning_rate_1��5����I       6%�	��
h���A�"*;


total_loss�*�@

error_R�@K?

learning_rate_1��52��dI       6%�	h���A�"*;


total_loss��.A

error_R�a?

learning_rate_1��5�O2I       6%�	�gh���A�"*;


total_loss�4�@

error_R1�^?

learning_rate_1��5���I       6%�	�h���A�"*;


total_lossH�A

error_R��H?

learning_rate_1��5^c��I       6%�	t�h���A�"*;


total_lossW��@

error_RfV?

learning_rate_1��5L�m5I       6%�	u<h���A�"*;


total_losse��@

error_R��;?

learning_rate_1��5���QI       6%�	`�h���A�"*;


total_loss�%z@

error_R��K?

learning_rate_1��5/XnI       6%�	��h���A�"*;


total_losse\�@

error_R;xA?

learning_rate_1��5FsOI       6%�	�	h���A�"*;


total_loss]ۊ@

error_R[�I?

learning_rate_1��5�DqzI       6%�	�Lh���A�"*;


total_lossÌ@

error_R�sI?

learning_rate_1��5�F�I       6%�	
�h���A�"*;


total_loss�A

error_R*K?

learning_rate_1��5�-�&I       6%�	��h���A�"*;


total_loss�E�@

error_R1�H?

learning_rate_1��5��qI       6%�	�Bh���A�"*;


total_loss�|�@

error_R��W?

learning_rate_1��5]��I       6%�	)�h���A�"*;


total_loss�u�@

error_R��I?

learning_rate_1��5v��I       6%�	��h���A�"*;


total_loss��@

error_R��X?

learning_rate_1��5՛�TI       6%�	�h���A�"*;


total_loss-��@

error_R�hK?

learning_rate_1��5�۰I       6%�	�Yh���A�"*;


total_loss�d
A

error_RI�N?

learning_rate_1��5M6�fI       6%�	%�h���A�"*;


total_lossn��@

error_R�ML?

learning_rate_1��5��TBI       6%�	�h���A�"*;


total_loss�@

error_R�W?

learning_rate_1��5�c�uI       6%�	-1h���A�"*;


total_loss�ӊ@

error_RH�:?

learning_rate_1��5�F�)I       6%�	�uh���A�"*;


total_lossa�A

error_Rw�O?

learning_rate_1��5��ƯI       6%�	�h���A�"*;


total_loss���@

error_R��S?

learning_rate_1��5�#
�I       6%�	�h���A�"*;


total_loss:��@

error_R&�C?

learning_rate_1��5�ŧI       6%�	|Nh���A�"*;


total_lossX��@

error_R�WQ?

learning_rate_1��5�O�<I       6%�	p�h���A�"*;


total_loss��@

error_R��V?

learning_rate_1��5U��I       6%�	��h���A�"*;


total_loss,K�@

error_R�1E?

learning_rate_1��5�]��I       6%�	�)h���A�"*;


total_loss�t�@

error_R��E?

learning_rate_1��5��>I       6%�	�lh���A�"*;


total_loss�z�@

error_R�BH?

learning_rate_1��5�;zI       6%�	_�h���A�"*;


total_loss\^�@

error_R42`?

learning_rate_1��5ρI       6%�	>�h���A�"*;


total_lossI:�@

error_RH'O?

learning_rate_1��5R���I       6%�	�3h���A�"*;


total_loss�v�@

error_R��C?

learning_rate_1��5y#�?I       6%�	oxh���A�"*;


total_loss|X�@

error_R2�>?

learning_rate_1��5�@�dI       6%�	\�h���A�"*;


total_loss@��@

error_R��Q?

learning_rate_1��5�ՙI       6%�	��h���A�"*;


total_loss�1�@

error_R��E?

learning_rate_1��5�n)NI       6%�	GEh���A�"*;


total_lossX��@

error_R�Q?

learning_rate_1��5����I       6%�	��h���A�"*;


total_loss��A

error_R�UQ?

learning_rate_1��5O��I       6%�	q�h���A�"*;


total_loss��@

error_R}O_?

learning_rate_1��589l�I       6%�	�6h���A�"*;


total_lossg�@

error_R�#B?

learning_rate_1��5p��I       6%�	J�h���A�"*;


total_loss��@

error_R�lV?

learning_rate_1��5�>��I       6%�	C�h���A�"*;


total_loss�*�@

error_RM"Q?

learning_rate_1��5N�-�I       6%�	z*h���A�"*;


total_loss�@

error_Rf�=?

learning_rate_1��5w�ѭI       6%�	�qh���A�"*;


total_loss�Ģ@

error_R6PH?

learning_rate_1��5@��TI       6%�	S�h���A�"*;


total_loss�z�@

error_RσA?

learning_rate_1��5�(vI       6%�	�h���A�"*;


total_loss�f�@

error_R�bM?

learning_rate_1��5,�I       6%�	�Dh���A�"*;


total_loss��@

error_R�6?

learning_rate_1��5��yI       6%�	�h���A�"*;


total_loss�A

error_R3O?

learning_rate_1��5�>�cI       6%�	�h���A�"*;


total_loss]��@

error_RX�8?

learning_rate_1��5�<�eI       6%�	.
h���A�"*;


total_lossU�@

error_R��N?

learning_rate_1��5��F!I       6%�	�Mh���A�"*;


total_loss�;`@

error_R��K?

learning_rate_1��5'n!)I       6%�	L�h���A�"*;


total_loss@.�@

error_R&\?

learning_rate_1��5��wI       6%�	d�h���A�"*;


total_loss�6A

error_R2J?

learning_rate_1��5̢o�I       6%�	�(h���A�"*;


total_loss��@

error_R�]?

learning_rate_1��5��I       6%�	vh���A�"*;


total_loss���@

error_R�O?

learning_rate_1��5P�A�I       6%�	�h���A�"*;


total_loss/��@

error_R�xW?

learning_rate_1��5���#I       6%�	�h���A�"*;


total_loss�i�@

error_R�W?

learning_rate_1��5΃aPI       6%�	�Hh���A�"*;


total_losse�@

error_R(UJ?

learning_rate_1��5�5��I       6%�	��h���A�"*;


total_loss���@

error_R�HR?

learning_rate_1��5)5;KI       6%�	�h���A�"*;


total_loss4�Z@

error_R3#N?

learning_rate_1��5V�I       6%�	}h���A�"*;


total_loss/3A

error_R��Z?

learning_rate_1��57�n�I       6%�	L]h���A�"*;


total_loss�Ӂ@

error_R�Y?

learning_rate_1��5�oI       6%�	��h���A�"*;


total_lossC�@

error_R�N?

learning_rate_1��5o��tI       6%�	��h���A�"*;


total_loss�-A

error_R��L?

learning_rate_1��5O �I       6%�	�*h���A�"*;


total_lossz z@

error_R}�J?

learning_rate_1��5� �I       6%�	�lh���A�"*;


total_loss\}�@

error_R\�Q?

learning_rate_1��5�	�I       6%�	�h���A�"*;


total_loss�b�@

error_Rr�K?

learning_rate_1��5$<��I       6%�	M�h���A�"*;


total_loss捷@

error_Rm3W?

learning_rate_1��5e(�2I       6%�	�8h���A�"*;


total_loss7A

error_R_�R?

learning_rate_1��5i0Y�I       6%�	|h���A�"*;


total_lossZ�@

error_R. 9?

learning_rate_1��5��8�I       6%�	��h���A�"*;


total_loss��@

error_R�O?

learning_rate_1��5 �5I       6%�	�)h���A�"*;


total_loss:�@

error_Re�@?

learning_rate_1��5��I?I       6%�	�oh���A�"*;


total_lossh��@

error_R�>S?

learning_rate_1��5�R�BI       6%�	ɷh���A�"*;


total_lossi=�@

error_R��F?

learning_rate_1��5L�+1I       6%�	��h���A�"*;


total_lossgK�@

error_R�mX?

learning_rate_1��5K��-I       6%�	9Eh���A�"*;


total_loss,(�@

error_R�8R?

learning_rate_1��5o�/�I       6%�	S�h���A�"*;


total_lossS��@

error_R�?I?

learning_rate_1��5��w�I       6%�	��h���A�"*;


total_loss=��@

error_R� W?

learning_rate_1��5'y�I       6%�	/
 h���A�"*;


total_lossE��@

error_R
�H?

learning_rate_1��5�[�.I       6%�	LM h���A�"*;


total_loss���@

error_R�2?

learning_rate_1��5�h��I       6%�	đ h���A�"*;


total_loss#K�@

error_RֳQ?

learning_rate_1��5��C9I       6%�	�� h���A�"*;


total_loss_$�@

error_R��D?

learning_rate_1��5�D��I       6%�	!h���A�"*;


total_loss���@

error_Ri�O?

learning_rate_1��5�ћuI       6%�	�]!h���A�"*;


total_loss�ޤ@

error_R=BN?

learning_rate_1��5Q1��I       6%�	6�!h���A�"*;


total_loss��o@

error_R�a\?

learning_rate_1��5��(�I       6%�	L�!h���A�"*;


total_loss���@

error_R]b?

learning_rate_1��5rII       6%�	z,"h���A�"*;


total_loss7Ǖ@

error_Rn H?

learning_rate_1��5ѹ/"I       6%�	�o"h���A�"*;


total_lossq��@

error_RhGb?

learning_rate_1��5��I�I       6%�	߱"h���A�"*;


total_loss�X�@

error_RZV?

learning_rate_1��5�4mtI       6%�	4�"h���A�"*;


total_loss&��@

error_R�S?

learning_rate_1��5|�I       6%�	�E#h���A�"*;


total_loss�ҹ@

error_R��R?

learning_rate_1��5AֆvI       6%�	~�#h���A�"*;


total_loss:p@

error_RCT?

learning_rate_1��5�_d&I       6%�	w�#h���A�"*;


total_loss���@

error_R��L?

learning_rate_1��5���%I       6%�	$h���A�"*;


total_loss8�@

error_R�l8?

learning_rate_1��5���I       6%�	�f$h���A�"*;


total_loss�X�@

error_R��O?

learning_rate_1��5Ւ�I       6%�	��$h���A�"*;


total_loss�o\@

error_RAV<?

learning_rate_1��5c�tI       6%�	��$h���A�"*;


total_lossR��@

error_RXD_?

learning_rate_1��5�*͒I       6%�	X<%h���A�"*;


total_loss���@

error_R�\?

learning_rate_1��5��jI       6%�	"�%h���A�"*;


total_losssO�@

error_R�<N?

learning_rate_1��5�M&MI       6%�	��%h���A�"*;


total_loss���@

error_R�g??

learning_rate_1��5�n�I       6%�	z&h���A�"*;


total_loss6��@

error_R��N?

learning_rate_1��5s.�I       6%�	W&h���A�#*;


total_loss)�@

error_RGE?

learning_rate_1��5�U�nI       6%�	=�&h���A�#*;


total_losst��@

error_RfUA?

learning_rate_1��5`��I       6%�	g�&h���A�#*;


total_loss�X�@

error_RT�J?

learning_rate_1��5���I       6%�	�)'h���A�#*;


total_loss�yA

error_R,�_?

learning_rate_1��5��4�I       6%�	Os'h���A�#*;


total_loss) �@

error_R_!P?

learning_rate_1��5�y�I       6%�	 �'h���A�#*;


total_loss���@

error_RH7Q?

learning_rate_1��5Mn�_I       6%�	I(h���A�#*;


total_loss;�@

error_R׍I?

learning_rate_1��5p͇'I       6%�	�I(h���A�#*;


total_loss�z�@

error_R��X?

learning_rate_1��5�Y�I       6%�	j�(h���A�#*;


total_loss4E�@

error_RAL?

learning_rate_1��5���cI       6%�	I�(h���A�#*;


total_loss��@

error_RH�P?

learning_rate_1��5 S��I       6%�	)h���A�#*;


total_lossq��@

error_RZ�X?

learning_rate_1��5a��I       6%�	|Y)h���A�#*;


total_loss���@

error_RDA??

learning_rate_1��5N���I       6%�	 �)h���A�#*;


total_lossE7�@

error_R��J?

learning_rate_1��5+N�1I       6%�	��)h���A�#*;


total_loss���@

error_R�[?

learning_rate_1��5��a�I       6%�	�"*h���A�#*;


total_loss:��@

error_RA?

learning_rate_1��5ѱd~I       6%�	mn*h���A�#*;


total_loss=Л@

error_R�U?

learning_rate_1��5b���I       6%�	��*h���A�#*;


total_loss'�@

error_R�M?

learning_rate_1��5J*��I       6%�	~�*h���A�#*;


total_loss3��@

error_RưI?

learning_rate_1��5m%>I       6%�	�B+h���A�#*;


total_loss��@

error_RO�U?

learning_rate_1��5rpFwI       6%�	
�+h���A�#*;


total_loss_��@

error_R�G?

learning_rate_1��5>'�I       6%�	��+h���A�#*;


total_loss{d�@

error_R_�;?

learning_rate_1��5�ԊnI       6%�	),h���A�#*;


total_lossz��@

error_R}�U?

learning_rate_1��5`�eI       6%�	0T,h���A�#*;


total_lossH�@

error_Rl5M?

learning_rate_1��5�A�mI       6%�	��,h���A�#*;


total_lossV��@

error_R�:F?

learning_rate_1��59pI       6%�	��,h���A�#*;


total_lossC\�@

error_R�[??

learning_rate_1��5�+jI       6%�	?&-h���A�#*;


total_loss��[@

error_R�S<?

learning_rate_1��5���I       6%�	Xq-h���A�#*;


total_loss���@

error_R��7?

learning_rate_1��5}�I       6%�	i�-h���A�#*;


total_lossd�@

error_RA�K?

learning_rate_1��51�G�I       6%�	�).h���A�#*;


total_loss`5�@

error_Rdc?

learning_rate_1��5\�yI       6%�	6u.h���A�#*;


total_loss�̙@

error_R��G?

learning_rate_1��5=o�I       6%�	�.h���A�#*;


total_loss��A

error_RDAK?

learning_rate_1��5�DoI       6%�	�/h���A�#*;


total_loss��@

error_R��W?

learning_rate_1��5�r�gI       6%�	0H/h���A�#*;


total_loss�j�@

error_RZ�X?

learning_rate_1��5݌��I       6%�	��/h���A�#*;


total_lossCJ�@

error_R��>?

learning_rate_1��5k�ӳI       6%�	k�/h���A�#*;


total_lossZ��@

error_R{V?

learning_rate_1��5��JI       6%�	�0h���A�#*;


total_loss;��@

error_R vG?

learning_rate_1��5V|�I       6%�	�e0h���A�#*;


total_loss`��@

error_RôJ?

learning_rate_1��5�9bI       6%�	��0h���A�#*;


total_loss�0�@

error_R�(T?

learning_rate_1��5c�qI       6%�	X�0h���A�#*;


total_lossʞ�@

error_RHBG?

learning_rate_1��5�XHI       6%�	$+1h���A�#*;


total_lossJ��@

error_Rw�F?

learning_rate_1��5�L��I       6%�	Dp1h���A�#*;


total_lossC�@

error_RC�O?

learning_rate_1��5��ǍI       6%�	f�1h���A�#*;


total_lossn� A

error_R�FN?

learning_rate_1��5y���I       6%�	_�1h���A�#*;


total_loss[*�@

error_R��=?

learning_rate_1��5���I       6%�	�>2h���A�#*;


total_loss2E�@

error_R��I?

learning_rate_1��5]@��I       6%�	ք2h���A�#*;


total_loss�z�@

error_R��/?

learning_rate_1��5(V[�I       6%�	�2h���A�#*;


total_loss=�@

error_R
�@?

learning_rate_1��5�|�(I       6%�	=3h���A�#*;


total_lossJ�@

error_Rf�E?

learning_rate_1��5v�Q/I       6%�	5S3h���A�#*;


total_loss���@

error_RRAB?

learning_rate_1��5w��I       6%�	�3h���A�#*;


total_loss&
�@

error_RCiF?

learning_rate_1��5��`I       6%�	@�3h���A�#*;


total_lossF��@

error_R�RP?

learning_rate_1��5��XI       6%�	H4h���A�#*;


total_lossr��@

error_R��F?

learning_rate_1��5E.�I       6%�	�_4h���A�#*;


total_loss�=�@

error_R��@?

learning_rate_1��5�&^"I       6%�	��4h���A�#*;


total_loss��x@

error_R[�C?

learning_rate_1��5=�!|I       6%�	f�4h���A�#*;


total_lossd�@

error_R�h9?

learning_rate_1��5� �I       6%�	rL5h���A�#*;


total_lossH^�@

error_R�>?

learning_rate_1��5�Z�I       6%�	ܘ5h���A�#*;


total_lossOI�@

error_RgF?

learning_rate_1��5[��I       6%�	�5h���A�#*;


total_loss���@

error_R�*L?

learning_rate_1��5���I       6%�	�E6h���A�#*;


total_loss_�S@

error_R,F?

learning_rate_1��5�SnYI       6%�	��6h���A�#*;


total_loss�>A

error_RiI?

learning_rate_1��5n��I       6%�	L�6h���A�#*;


total_lossѤ�@

error_RWtR?

learning_rate_1��5Rڣ/I       6%�	r!7h���A�#*;


total_loss=gw@

error_Rd?

learning_rate_1��5���I       6%�	!j7h���A�#*;


total_loss���@

error_RO=?

learning_rate_1��5\9�~I       6%�	��7h���A�#*;


total_loss]+�@

error_R�"G?

learning_rate_1��5�uvfI       6%�	��7h���A�#*;


total_lossͫ�@

error_R�^?

learning_rate_1��5�I       6%�	h>8h���A�#*;


total_loss���@

error_R:Z?

learning_rate_1��5Ef&I       6%�	Q�8h���A�#*;


total_loss3��@

error_R)eY?

learning_rate_1��5��n�I       6%�	�8h���A�#*;


total_loss��@

error_R��Q?

learning_rate_1��5TA�?I       6%�	�9h���A�#*;


total_loss䐧@

error_R�=U?

learning_rate_1��5���I       6%�	�}9h���A�#*;


total_loss�E A

error_R�1S?

learning_rate_1��5��R�I       6%�	��9h���A�#*;


total_lossiz�@

error_Ri|B?

learning_rate_1��5q���I       6%�	J2:h���A�#*;


total_loss̋s@

error_R�F?

learning_rate_1��5��I       6%�	Ox:h���A�#*;


total_loss�ҧ@

error_RlN^?

learning_rate_1��5m��I       6%�	�:h���A�#*;


total_loss��@

error_Rs�O?

learning_rate_1��5�0\I       6%�	;h���A�#*;


total_loss�ʻ@

error_R��j?

learning_rate_1��5��F�I       6%�	�E;h���A�#*;


total_loss��@

error_R��U?

learning_rate_1��5����I       6%�	e�;h���A�#*;


total_loss�6�@

error_Rd�L?

learning_rate_1��5Ϭ��I       6%�	��;h���A�#*;


total_loss]u�@

error_R6�g?

learning_rate_1��5��l�I       6%�	�<h���A�#*;


total_lossn�u@

error_RC�:?

learning_rate_1��5�Zb�I       6%�	�X<h���A�#*;


total_loss�>�@

error_RrfH?

learning_rate_1��5F�r4I       6%�	�<h���A�#*;


total_loss;��@

error_RX�U?

learning_rate_1��5ɘ!I       6%�	��<h���A�#*;


total_loss61�@

error_R�M?

learning_rate_1��5�L�I       6%�	�&=h���A�#*;


total_lossh�@

error_R��P?

learning_rate_1��5���I       6%�	j=h���A�#*;


total_lossᯭ@

error_R,@[?

learning_rate_1��5{�T�I       6%�	��=h���A�#*;


total_loss�ݳ@

error_R��`?

learning_rate_1��5� �0I       6%�	�>h���A�#*;


total_loss ��@

error_R�*D?

learning_rate_1��5�c6�I       6%�	�Y>h���A�#*;


total_loss�@

error_Rlzd?

learning_rate_1��5 �I       6%�	��>h���A�#*;


total_loss���@

error_RZ"Q?

learning_rate_1��5���I       6%�	�>h���A�#*;


total_loss3�@

error_R�O?

learning_rate_1��5���I       6%�	�2?h���A�#*;


total_loss��A

error_R�N?

learning_rate_1��5Y�I       6%�	&x?h���A�#*;


total_lossP
A

error_R�xZ?

learning_rate_1��5s��I       6%�	�?h���A�#*;


total_lossHA

error_R��J?

learning_rate_1��5��>jI       6%�	-@h���A�#*;


total_loss�Q�@

error_R@h?

learning_rate_1��5�^I       6%�	WI@h���A�#*;


total_lossܶ�@

error_R�L?

learning_rate_1��5ؐI       6%�	}�@h���A�#*;


total_lossx�@

error_R��??

learning_rate_1��5�3FI       6%�	��@h���A�#*;


total_loss�l�@

error_R��L?

learning_rate_1��5][}�I       6%�	?Ah���A�#*;


total_lossm��@

error_R�T?

learning_rate_1��5)eB�I       6%�	�`Ah���A�#*;


total_loss��@

error_Rn�d?

learning_rate_1��5���>I       6%�	�Ah���A�#*;


total_lossx��@

error_R�;R?

learning_rate_1��5�bI       6%�	�Ah���A�#*;


total_loss	0�@

error_R�S?

learning_rate_1��5�&�I       6%�	�0Bh���A�#*;


total_lossͶ�@

error_R�R?

learning_rate_1��5���I       6%�	~tBh���A�#*;


total_loss'G�@

error_R��@?

learning_rate_1��5��`�I       6%�	ոBh���A�#*;


total_loss\մ@

error_R�UF?

learning_rate_1��5�DI       6%�	L�Bh���A�#*;


total_lossWq�@

error_R�/X?

learning_rate_1��5�!�I       6%�	�PCh���A�#*;


total_lossos�@

error_RڏW?

learning_rate_1��5P�ZLI       6%�	M�Ch���A�#*;


total_loss1׬@

error_RCV?

learning_rate_1��5�BJI       6%�	��Ch���A�#*;


total_loss��@

error_R��J?

learning_rate_1��5$���I       6%�	H Dh���A�#*;


total_loss���@

error_R��D?

learning_rate_1��5�ƾI       6%�	ShDh���A�#*;


total_loss,��@

error_R�V?

learning_rate_1��5����I       6%�	d�Dh���A�#*;


total_loss���@

error_RM�[?

learning_rate_1��5�d�1I       6%�	��Dh���A�#*;


total_losssA�@

error_R�lP?

learning_rate_1��5%���I       6%�	\?Eh���A�#*;


total_lossڛ�@

error_R�}O?

learning_rate_1��53c�I       6%�	��Eh���A�#*;


total_loss�9�@

error_R8!W?

learning_rate_1��5���I       6%�	��Eh���A�#*;


total_loss���@

error_RҴO?

learning_rate_1��5y^j�I       6%�	�Fh���A�#*;


total_loss�ɪ@

error_RF�W?

learning_rate_1��5sI       6%�	ghFh���A�#*;


total_lossD�@

error_R�wC?

learning_rate_1��5"�RUI       6%�	��Fh���A�#*;


total_loss��A

error_R�M?

learning_rate_1��5?5�I       6%�	v�Fh���A�#*;


total_loss�R�@

error_R�C?

learning_rate_1��5�3;I       6%�	:Gh���A�#*;


total_loss#��@

error_R�KY?

learning_rate_1��5��2*I       6%�	�~Gh���A�#*;


total_losss�@

error_R$�@?

learning_rate_1��5�愄I       6%�	�Gh���A�#*;


total_loss��@

error_R�
R?

learning_rate_1��5����I       6%�	?Hh���A�#*;


total_loss���@

error_ReZQ?

learning_rate_1��5����I       6%�	�WHh���A�#*;


total_loss+gA

error_R��V?

learning_rate_1��5V�AI       6%�	�Hh���A�#*;


total_loss[�*A

error_R� V?

learning_rate_1��5�3��I       6%�	��Hh���A�#*;


total_losss��@

error_R8�Z?

learning_rate_1��5K��I       6%�	1Ih���A�#*;


total_loss<�@

error_R��M?

learning_rate_1��5���I       6%�	fyIh���A�#*;


total_loss�S�@

error_Rf9??

learning_rate_1��5lٿ�I       6%�	U�Ih���A�#*;


total_loss<7�@

error_R�R?

learning_rate_1��5uP�PI       6%�	�Jh���A�#*;


total_loss���@

error_RC+Y?

learning_rate_1��52ԍI       6%�	�LJh���A�$*;


total_loss�@

error_R��L?

learning_rate_1��50a�I       6%�	��Jh���A�$*;


total_loss%��@

error_Re|k?

learning_rate_1��5͓��I       6%�	��Jh���A�$*;


total_lossI�@

error_Rn_A?

learning_rate_1��5��{I       6%�	�Kh���A�$*;


total_loss}�~@

error_Rn�E?

learning_rate_1��5�m�I       6%�	N\Kh���A�$*;


total_loss:�@

error_R�tT?

learning_rate_1��5&�p:I       6%�	��Kh���A�$*;


total_loss�A

error_RkG?

learning_rate_1��5k�y�I       6%�	��Kh���A�$*;


total_loss涣@

error_R�[O?

learning_rate_1��5��#�I       6%�	�*Lh���A�$*;


total_loss(X4A

error_R� I?

learning_rate_1��5[�H/I       6%�	nLh���A�$*;


total_loss=XA

error_R_eI?

learning_rate_1��5�.P I       6%�	3�Lh���A�$*;


total_loss�Z�@

error_RZ�=?

learning_rate_1��5Jw�fI       6%�	o	Mh���A�$*;


total_lossi��@

error_R��I?

learning_rate_1��5��ҽI       6%�	�QMh���A�$*;


total_loss���@

error_R�X?

learning_rate_1��5Kb��I       6%�	}�Mh���A�$*;


total_loss츕@

error_R �G?

learning_rate_1��5��P�I       6%�	��Mh���A�$*;


total_loss�O�@

error_RW�Q?

learning_rate_1��5��\�I       6%�	�DNh���A�$*;


total_loss�_@

error_R!�G?

learning_rate_1��5�D�`I       6%�	��Nh���A�$*;


total_loss��@

error_R�P?

learning_rate_1��5�~I       6%�	��Nh���A�$*;


total_loss*�@

error_R�J?

learning_rate_1��5t�k|I       6%�	�Oh���A�$*;


total_loss�S�@

error_R�5?

learning_rate_1��5s�I       6%�	bOh���A�$*;


total_loss��$A

error_R8nL?

learning_rate_1��5�`I       6%�	Y�Oh���A�$*;


total_loss��@

error_R�Z?

learning_rate_1��5�2m�I       6%�	�Oh���A�$*;


total_loss�z�@

error_R��H?

learning_rate_1��5��q
I       6%�	5Ph���A�$*;


total_lossC�@

error_R$r?

learning_rate_1��5���I       6%�	 �Ph���A�$*;


total_loss�f�@

error_R;�K?

learning_rate_1��5���I       6%�	��Ph���A�$*;


total_loss�!�@

error_R��V?

learning_rate_1��5���EI       6%�	�Qh���A�$*;


total_loss���@

error_R�HD?

learning_rate_1��5�"�zI       6%�	�XQh���A�$*;


total_loss�g�@

error_R��Q?

learning_rate_1��5�3�I       6%�	��Qh���A�$*;


total_loss���@

error_Rx�X?

learning_rate_1��56�*I       6%�	U�Qh���A�$*;


total_loss��@

error_Rj�W?

learning_rate_1��5Q�P]I       6%�	##Rh���A�$*;


total_lossAѮ@

error_R�Q?

learning_rate_1��5��f�I       6%�	�fRh���A�$*;


total_loss;�@

error_R��J?

learning_rate_1��5O{�I       6%�	��Rh���A�$*;


total_loss��A

error_ROY?

learning_rate_1��5_f�I       6%�	F�Rh���A�$*;


total_loss�z�@

error_R�(P?

learning_rate_1��5��MI       6%�	�/Sh���A�$*;


total_loss�u�@

error_R}_[?

learning_rate_1��5�yOI       6%�	�rSh���A�$*;


total_loss��@

error_Rݩ9?

learning_rate_1��5��BdI       6%�	#�Sh���A�$*;


total_lossC�@

error_RC�I?

learning_rate_1��5B.&�I       6%�	��Sh���A�$*;


total_loss�Z�@

error_R��,?

learning_rate_1��5�y��I       6%�	pHTh���A�$*;


total_lossv��@

error_R4�G?

learning_rate_1��5C�I       6%�	�Th���A�$*;


total_loss���@

error_RXN?

learning_rate_1��5��1�I       6%�	_�Th���A�$*;


total_loss?��@

error_R=�C?

learning_rate_1��5f$A�I       6%�	9GUh���A�$*;


total_loss���@

error_RF]L?

learning_rate_1��5P�I       6%�	�Uh���A�$*;


total_loss���@

error_RlV?

learning_rate_1��5zs�xI       6%�	~Vh���A�$*;


total_losshFs@

error_R�Dq?

learning_rate_1��5g�I       6%�	qNVh���A�$*;


total_loss�d@

error_RCIE?

learning_rate_1��5�h��I       6%�	��Vh���A�$*;


total_loss ��@

error_R�L?

learning_rate_1��5)���I       6%�	U�Vh���A�$*;


total_lossD��@

error_R��L?

learning_rate_1��5iI       6%�	:Wh���A�$*;


total_loss
��@

error_R�fa?

learning_rate_1��5cq\I       6%�	�bWh���A�$*;


total_loss�@

error_Ro??

learning_rate_1��5� 6�I       6%�	�Wh���A�$*;


total_loss3ܟ@

error_R��I?

learning_rate_1��5�ìEI       6%�	f�Wh���A�$*;


total_losslȠ@

error_R��S?

learning_rate_1��5��E�I       6%�	~3Xh���A�$*;


total_lossN��@

error_R��S?

learning_rate_1��5J�I       6%�	PxXh���A�$*;


total_loss���@

error_R��W?

learning_rate_1��5��I       6%�	H�Xh���A�$*;


total_loss!n�@

error_R�\U?

learning_rate_1��5�̻KI       6%�		Yh���A�$*;


total_loss���@

error_R$DQ?

learning_rate_1��5�<yI       6%�	�JYh���A�$*;


total_loss
M�@

error_R�X?

learning_rate_1��5 /�I       6%�	�Yh���A�$*;


total_loss�_�@

error_R�Q?

learning_rate_1��5����I       6%�	�Yh���A�$*;


total_loss���@

error_RR�T?

learning_rate_1��5@�_I       6%�	�Zh���A�$*;


total_loss�)�@

error_R�d?

learning_rate_1��5%��@I       6%�	`Zh���A�$*;


total_loss.�@

error_R��;?

learning_rate_1��5�u�I       6%�	H�Zh���A�$*;


total_loss���@

error_R�N?

learning_rate_1��5e,�I       6%�	9�Zh���A�$*;


total_loss���@

error_R,R@?

learning_rate_1��5-���I       6%�	m,[h���A�$*;


total_lossaj�@

error_R��8?

learning_rate_1��5@s�{I       6%�	oo[h���A�$*;


total_loss�Y�@

error_R�Z?

learning_rate_1��5e�ܿI       6%�	�[h���A�$*;


total_loss�A�@

error_R�5?

learning_rate_1��5^0I       6%�	��[h���A�$*;


total_loss1Ҷ@

error_RO?

learning_rate_1��5�>��I       6%�	�9\h���A�$*;


total_loss�8}@

error_R��f?

learning_rate_1��5l�]I       6%�	a{\h���A�$*;


total_loss�4�@

error_R��X?

learning_rate_1��5�2��I       6%�	�\h���A�$*;


total_loss;.�@

error_Rl[B?

learning_rate_1��5%���I       6%�	T]h���A�$*;


total_loss�@

error_R��c?

learning_rate_1��5�zz�I       6%�	_I]h���A�$*;


total_lossA�@

error_RZX?

learning_rate_1��5e�/�I       6%�	�]h���A�$*;


total_loss�@

error_R�7?

learning_rate_1��5Ӵ4�I       6%�	��]h���A�$*;


total_lossxA�@

error_RUT?

learning_rate_1��5I�{I       6%�	�C^h���A�$*;


total_loss��@

error_R�vK?

learning_rate_1��5�N
BI       6%�	�^h���A�$*;


total_loss��@

error_R�Q?

learning_rate_1��5kS{I       6%�	��^h���A�$*;


total_loss}V�@

error_R�UX?

learning_rate_1��5q���I       6%�	� _h���A�$*;


total_loss�@

error_R]�B?

learning_rate_1��5��v�I       6%�	Qd_h���A�$*;


total_loss��@

error_R��[?

learning_rate_1��5�z��I       6%�	�_h���A�$*;


total_loss�H@

error_R��G?

learning_rate_1��5J��I       6%�	:�_h���A�$*;


total_loss?�@

error_R!�[?

learning_rate_1��5�(I       6%�	D/`h���A�$*;


total_loss\L�@

error_R=W>?

learning_rate_1��5b^�I       6%�	�z`h���A�$*;


total_loss4"�@

error_R%4Q?

learning_rate_1��5�y4�I       6%�	$�`h���A�$*;


total_loss���@

error_R�VU?

learning_rate_1��5�>I       6%�	�	ah���A�$*;


total_loss���@

error_RC�:?

learning_rate_1��5;<��I       6%�	GQah���A�$*;


total_loss|1A

error_R�nW?

learning_rate_1��5�r�I       6%�	E�ah���A�$*;


total_loss`%�@

error_R̱Y?

learning_rate_1��5��9I       6%�	e�ah���A�$*;


total_loss���@

error_R6�D?

learning_rate_1��5Z"I       6%�	*(bh���A�$*;


total_loss��@

error_Rs�W?

learning_rate_1��5�</I       6%�	�pbh���A�$*;


total_loss�"�@

error_R��[?

learning_rate_1��5{M�TI       6%�	P�bh���A�$*;


total_lossc�@

error_R�vU?

learning_rate_1��5�j��I       6%�	�ch���A�$*;


total_loss�a�@

error_R��B?

learning_rate_1��5�#I       6%�	Kch���A�$*;


total_loss�Q�@

error_R�=?

learning_rate_1��5���I       6%�	�ch���A�$*;


total_lossǲ@

error_R&�[?

learning_rate_1��5U�/I       6%�	�ch���A�$*;


total_lossl�@

error_RZE?

learning_rate_1��57��I       6%�	n dh���A�$*;


total_lossj^�@

error_R]C?

learning_rate_1��5;���I       6%�	ddh���A�$*;


total_loss
�@

error_R�P?

learning_rate_1��5V>��I       6%�	��dh���A�$*;


total_lossZ�@

error_R�4@?

learning_rate_1��52;�I       6%�	,�dh���A�$*;


total_lossH��@

error_R�rH?

learning_rate_1��5��k�I       6%�	�3eh���A�$*;


total_loss7|�@

error_RLi?

learning_rate_1��5a5"�I       6%�	�veh���A�$*;


total_loss�ֈ@

error_R?�O?

learning_rate_1��5��I       6%�	�eh���A�$*;


total_lossEr�@

error_R-F@?

learning_rate_1��5Z�PjI       6%�	qfh���A�$*;


total_loss�
�@

error_R�<L?

learning_rate_1��5w�I       6%�	RPfh���A�$*;


total_lossFK�@

error_R��;?

learning_rate_1��5n�z�I       6%�	&�fh���A�$*;


total_loss�3�@

error_RZ|W?

learning_rate_1��5v���I       6%�	��fh���A�$*;


total_loss��@

error_R��=?

learning_rate_1��5��l�I       6%�	�gh���A�$*;


total_loss���@

error_R��A?

learning_rate_1��5d�c:I       6%�	kYgh���A�$*;


total_loss��@

error_RŻ@?

learning_rate_1��5l��wI       6%�	l�gh���A�$*;


total_loss���@

error_RF�W?

learning_rate_1��5�)Y�I       6%�	��gh���A�$*;


total_lossS�@

error_R"H?

learning_rate_1��52=;I       6%�	~!hh���A�$*;


total_lossH��@

error_R5^?

learning_rate_1��5���I       6%�	kehh���A�$*;


total_lossT<�@

error_RM�Z?

learning_rate_1��5 ߁�I       6%�	®hh���A�$*;


total_loss���@

error_R�t=?

learning_rate_1��5�~�I       6%�	B�hh���A�$*;


total_loss���@

error_Re�O?

learning_rate_1��5�
mI       6%�	q@ih���A�$*;


total_lossѻ�@

error_R.J?

learning_rate_1��5o?��I       6%�	Q�ih���A�$*;


total_lossq�@

error_R_�O?

learning_rate_1��5�R�I       6%�	?�ih���A�$*;


total_loss},�@

error_R�/7?

learning_rate_1��5YU�I       6%�	x)jh���A�$*;


total_lossT?�@

error_R��[?

learning_rate_1��5L���I       6%�	�sjh���A�$*;


total_lossT��@

error_Rw P?

learning_rate_1��5&�I       6%�	e�jh���A�$*;


total_lossv1�@

error_RȭT?

learning_rate_1��5%0A}I       6%�	�kh���A�$*;


total_lossw�A

error_R�S?

learning_rate_1��5wT�I       6%�	�Hkh���A�$*;


total_loss��@

error_R�PW?

learning_rate_1��5kq$�I       6%�	'�kh���A�$*;


total_lossH�@

error_R8	N?

learning_rate_1��5b+�I       6%�	��kh���A�$*;


total_loss�}�@

error_R�8P?

learning_rate_1��5����I       6%�	�lh���A�$*;


total_loss_\�@

error_R��@?

learning_rate_1��5>���I       6%�	�alh���A�$*;


total_lossn$�@

error_R�A?

learning_rate_1��5� 1I       6%�	��lh���A�$*;


total_lossC��@

error_Rϒa?

learning_rate_1��5@�A�I       6%�	o�lh���A�$*;


total_loss�.�@

error_R�kH?

learning_rate_1��5v��MI       6%�	i1mh���A�$*;


total_loss%~�@

error_Rv�L?

learning_rate_1��5bC$I       6%�	�tmh���A�$*;


total_loss\�@

error_RɉZ?

learning_rate_1��5W��I       6%�	A�mh���A�$*;


total_loss�u@

error_R��@?

learning_rate_1��5cƗ�I       6%�	�5nh���A�%*;


total_loss��@

error_Rh�J?

learning_rate_1��5� FwI       6%�	��nh���A�%*;


total_loss��A

error_R_�O?

learning_rate_1��5��#[I       6%�	h�nh���A�%*;


total_lossxܟ@

error_R��O?

learning_rate_1��5vqAI       6%�	;	oh���A�%*;


total_loss= �@

error_RʛT?

learning_rate_1��5��tpI       6%�	�Koh���A�%*;


total_loss*&�@

error_R��I?

learning_rate_1��5��I       6%�	��oh���A�%*;


total_loss�O�@

error_R��>?

learning_rate_1��5����I       6%�	�oh���A�%*;


total_loss��@

error_R}�K?

learning_rate_1��5�Q�I       6%�	�ph���A�%*;


total_loss�+�@

error_RϝD?

learning_rate_1��5b��I       6%�	�cph���A�%*;


total_loss3��@

error_RkI?

learning_rate_1��5�
CI       6%�	y�ph���A�%*;


total_loss�;�@

error_Rq#[?

learning_rate_1��5="I       6%�	S�ph���A�%*;


total_loss�5�@

error_R��O?

learning_rate_1��5�xkI       6%�	�/qh���A�%*;


total_loss鄗@

error_R�~H?

learning_rate_1��5i�DI       6%�	Usqh���A�%*;


total_lossW�@

error_R4�7?

learning_rate_1��5��I       6%�	��qh���A�%*;


total_loss��@

error_Ri�W?

learning_rate_1��5�\1I       6%�	�rh���A�%*;


total_loss-��@

error_R�=K?

learning_rate_1��5��*[I       6%�	Mrh���A�%*;


total_loss�H�@

error_R_L?

learning_rate_1��5��u+I       6%�	��rh���A�%*;


total_loss���@

error_Rq�f?

learning_rate_1��5�A�I       6%�	��rh���A�%*;


total_loss���@

error_R��S?

learning_rate_1��5;��I       6%�	� sh���A�%*;


total_loss�ԫ@

error_R,$A?

learning_rate_1��5Y<K�I       6%�	hsh���A�%*;


total_loss��z@

error_RQU?

learning_rate_1��5zM�I       6%�	��sh���A�%*;


total_lossf��@

error_R�jR?

learning_rate_1��5bއwI       6%�	��sh���A�%*;


total_loss�$A

error_R�>L?

learning_rate_1��5U�C�I       6%�	c1th���A�%*;


total_loss�[�@

error_R�\?

learning_rate_1��5'��I       6%�	!sth���A�%*;


total_lossb�@

error_R�S?

learning_rate_1��5���I       6%�	��th���A�%*;


total_lossW[�@

error_RO�E?

learning_rate_1��5M�&<I       6%�	�uh���A�%*;


total_loss���@

error_R=T?

learning_rate_1��5)� I       6%�	�[uh���A�%*;


total_lossQA@

error_R��@?

learning_rate_1��5�w�"I       6%�	�uh���A�%*;


total_loss{��@

error_R1�=?

learning_rate_1��5���I       6%�	 vh���A�%*;


total_loss�/�@

error_Ri�A?

learning_rate_1��5g9I       6%�	�Jvh���A�%*;


total_loss�z�@

error_R��J?

learning_rate_1��5���EI       6%�	��vh���A�%*;


total_loss�^�@

error_R��<?

learning_rate_1��5L�s�I       6%�	$�vh���A�%*;


total_losss��@

error_R��>?

learning_rate_1��5����I       6%�	0wh���A�%*;


total_lossq�@

error_R�"H?

learning_rate_1��5yeI       6%�	;awh���A�%*;


total_lossMV�@

error_R�@?

learning_rate_1��5ۚ�I       6%�	��wh���A�%*;


total_loss
�@

error_R�T?

learning_rate_1��5�;#}I       6%�	��wh���A�%*;


total_loss�ý@

error_RΆ??

learning_rate_1��5�I       6%�	�2xh���A�%*;


total_loss�G�@

error_RO�J?

learning_rate_1��53��I       6%�	�uxh���A�%*;


total_loss�/�@

error_R�E?

learning_rate_1��5&J1@I       6%�	v�xh���A�%*;


total_loss��@

error_R�AK?

learning_rate_1��5JEI       6%�	�yh���A�%*;


total_loss�f�@

error_R��G?

learning_rate_1��5.���I       6%�	lJyh���A�%*;


total_loss�8�@

error_R�^?

learning_rate_1��5φy�I       6%�	?�yh���A�%*;


total_loss���@

error_R�D?

learning_rate_1��5����I       6%�	��yh���A�%*;


total_loss���@

error_Re]6?

learning_rate_1��5��wI       6%�	zh���A�%*;


total_loss�%�@

error_RlQ?

learning_rate_1��5G�O1I       6%�	�azh���A�%*;


total_lossQ��@

error_Ra�_?

learning_rate_1��5��PI       6%�	-�zh���A�%*;


total_lossi��@

error_R��M?

learning_rate_1��5PS��I       6%�	��zh���A�%*;


total_loss���@

error_RO�J?

learning_rate_1��5�~��I       6%�	W/{h���A�%*;


total_loss艗@

error_R��C?

learning_rate_1��5�~�eI       6%�	�r{h���A�%*;


total_lossy$�@

error_R��N?

learning_rate_1��5՜�@I       6%�	��{h���A�%*;


total_loss�@

error_R�\a?

learning_rate_1��5k��I       6%�	��{h���A�%*;


total_lossl�@

error_R��O?

learning_rate_1��5Ӹ��I       6%�	�A|h���A�%*;


total_loss*�@

error_R�>?

learning_rate_1��5.o�*I       6%�	�|h���A�%*;


total_loss��z@

error_RkE?

learning_rate_1��5�*I       6%�	~�|h���A�%*;


total_loss%��@

error_RȘQ?

learning_rate_1��5'��I       6%�	�}h���A�%*;


total_loss���@

error_Rj??

learning_rate_1��5"��I       6%�	�`}h���A�%*;


total_loss2ֶ@

error_RW�G?

learning_rate_1��5�͍�I       6%�	��}h���A�%*;


total_loss���@

error_R��;?

learning_rate_1��56�a�I       6%�	
~h���A�%*;


total_loss�@

error_R��[?

learning_rate_1��5<
�I       6%�	[S~h���A�%*;


total_lossq!�@

error_R�aT?

learning_rate_1��5����I       6%�	��~h���A�%*;


total_loss+~A

error_R��B?

learning_rate_1��5��PI       6%�	��~h���A�%*;


total_loss�ZA

error_R\�R?

learning_rate_1��5�@}�I       6%�	&#h���A�%*;


total_loss��@

error_Rl>?

learning_rate_1��5a��CI       6%�	�ih���A�%*;


total_loss ,�@

error_RQ�T?

learning_rate_1��5W9�I       6%�	��h���A�%*;


total_loss�Y�@

error_R�?2?

learning_rate_1��5\�[I       6%�	��h���A�%*;


total_lossM�x@

error_RҖX?

learning_rate_1��5�nQ"I       6%�	�:�h���A�%*;


total_loss���@

error_R��J?

learning_rate_1��5�QD�I       6%�	@��h���A�%*;


total_lossZ[�@

error_R��B?

learning_rate_1��5��E�I       6%�	�̀h���A�%*;


total_loss��@

error_RCE?

learning_rate_1��5f�yI       6%�	s�h���A�%*;


total_lossE��@

error_R��Z?

learning_rate_1��5t���I       6%�	IU�h���A�%*;


total_loss�=�@

error_R(G?

learning_rate_1��5� |I       6%�	K��h���A�%*;


total_loss�G�@

error_R:Y?

learning_rate_1��5�i��I       6%�	�܁h���A�%*;


total_loss �@

error_R�5L?

learning_rate_1��5_Qf�I       6%�	�h���A�%*;


total_loss�r�@

error_R;�:?

learning_rate_1��5f��I       6%�	�c�h���A�%*;


total_loss���@

error_R��:?

learning_rate_1��5T��I       6%�	"��h���A�%*;


total_loss��@

error_R64>?

learning_rate_1��5��o�I       6%�	��h���A�%*;


total_loss��@

error_R�Q?

learning_rate_1��5UG=�I       6%�	0�h���A�%*;


total_loss��{@

error_Rf�O?

learning_rate_1��57g7I       6%�	kt�h���A�%*;


total_loss���@

error_R߄T?

learning_rate_1��5�C��I       6%�	���h���A�%*;


total_loss;v�@

error_R��L?

learning_rate_1��5Є((I       6%�	���h���A�%*;


total_loss,�@

error_R�C?

learning_rate_1��5jk�I       6%�	kH�h���A�%*;


total_loss�D�@

error_RR7Z?

learning_rate_1��5���I       6%�	"��h���A�%*;


total_lossH~�@

error_RO�K?

learning_rate_1��5|/ҩI       6%�	�̈́h���A�%*;


total_loss��@

error_R�:?

learning_rate_1��5JAr<I       6%�	X�h���A�%*;


total_loss�!�@

error_Rq�_?

learning_rate_1��51]��I       6%�	�U�h���A�%*;


total_loss���@

error_RÎc?

learning_rate_1��5��/�I       6%�	��h���A�%*;


total_loss�t�@

error_RZ�L?

learning_rate_1��5��4�I       6%�	F݅h���A�%*;


total_loss�A

error_R�>R?

learning_rate_1��5VY�I       6%�	�#�h���A�%*;


total_loss���@

error_R
�^?

learning_rate_1��5�F^JI       6%�	�l�h���A�%*;


total_lossF֒@

error_R��I?

learning_rate_1��5��X7I       6%�	���h���A�%*;


total_loss�u�@

error_R��:?

learning_rate_1��5 3 �I       6%�	��h���A�%*;


total_lossD;A

error_RN?

learning_rate_1��5ˆ=�I       6%�	�:�h���A�%*;


total_lossLf�@

error_R�GT?

learning_rate_1��5h�t�I       6%�	y��h���A�%*;


total_loss�4�@

error_Rn�N?

learning_rate_1��5in'�I       6%�	�Ƈh���A�%*;


total_loss�Ԍ@

error_R��I?

learning_rate_1��5iѡFI       6%�	��h���A�%*;


total_loss,��@

error_R�E?

learning_rate_1��5w�gmI       6%�	X�h���A�%*;


total_loss,l�@

error_RſF?

learning_rate_1��5#��_I       6%�	���h���A�%*;


total_lossq��@

error_R�R?

learning_rate_1��5*0+I       6%�	c�h���A�%*;


total_loss��A

error_R�Dj?

learning_rate_1��5�
I       6%�	�'�h���A�%*;


total_loss��@

error_R.�@?

learning_rate_1��5_�q�I       6%�	;n�h���A�%*;


total_lossJF�@

error_Rw�I?

learning_rate_1��5�(�I       6%�	Z��h���A�%*;


total_loss���@

error_R�W?

learning_rate_1��5�ɚ�I       6%�	�h���A�%*;


total_loss��@

error_RI_M?

learning_rate_1��5En�HI       6%�	8N�h���A�%*;


total_losscЅ@

error_R�>?

learning_rate_1��5�21aI       6%�	��h���A�%*;


total_lossM�@

error_R�<?

learning_rate_1��5r�m�I       6%�	m��h���A�%*;


total_loss �@

error_R�o3?

learning_rate_1��5����I       6%�	�(�h���A�%*;


total_loss��@

error_R�xK?

learning_rate_1��5��]:I       6%�	�h�h���A�%*;


total_loss_��@

error_R�8?

learning_rate_1��5x+�|I       6%�	��h���A�%*;


total_loss�T�@

error_R��4?

learning_rate_1��5���I       6%�	N�h���A�%*;


total_loss��@

error_R�K?

learning_rate_1��5ې�I       6%�	5�h���A�%*;


total_loss���@

error_RT�O?

learning_rate_1��5����I       6%�	H�h���A�%*;


total_loss�|�@

error_R�M?

learning_rate_1��5�i�I       6%�	DÌh���A�%*;


total_loss�3�@

error_R��N?

learning_rate_1��5���QI       6%�	�	�h���A�%*;


total_loss�b�@

error_R�Q?

learning_rate_1��5+g�4I       6%�	HL�h���A�%*;


total_lossF�@

error_RN^?

learning_rate_1��5r��5I       6%�	c��h���A�%*;


total_loss���@

error_R�#=?

learning_rate_1��5bI       6%�	���h���A�%*;


total_loss�ӏ@

error_R�	P?

learning_rate_1��5��nI       6%�	�H�h���A�%*;


total_loss��A

error_R��E?

learning_rate_1��5V�`I       6%�	���h���A�%*;


total_lossq2�@

error_R�CG?

learning_rate_1��5��
I       6%�	�͎h���A�%*;


total_loss���@

error_R�K?

learning_rate_1��5�L^�I       6%�	��h���A�%*;


total_loss�o�@

error_R�0[?

learning_rate_1��5�H��I       6%�	U�h���A�%*;


total_loss|Z�@

error_R.�N?

learning_rate_1��5��~LI       6%�	���h���A�%*;


total_loss�@�@

error_R�H?

learning_rate_1��5}�6I       6%�	}ڏh���A�%*;


total_loss<��@

error_Rd�D?

learning_rate_1��5�o�YI       6%�	��h���A�%*;


total_lossi�@

error_RWkW?

learning_rate_1��5�kw�I       6%�	�`�h���A�%*;


total_loss7'�@

error_RŞS?

learning_rate_1��5.J	�I       6%�	���h���A�%*;


total_loss�A

error_R�M?

learning_rate_1��5M�;�I       6%�	d�h���A�%*;


total_loss>�@

error_R;�N?

learning_rate_1��5�o �I       6%�	a(�h���A�%*;


total_loss�^j@

error_R��K?

learning_rate_1��5`��I       6%�	l�h���A�&*;


total_loss���@

error_Re�H?

learning_rate_1��5J���I       6%�	q��h���A�&*;


total_loss4\�@

error_R�X?

learning_rate_1��5���I       6%�	��h���A�&*;


total_loss;�@

error_R
:?

learning_rate_1��53>vEI       6%�	�5�h���A�&*;


total_loss�1p@

error_R��i?

learning_rate_1��5.��3I       6%�	�y�h���A�&*;


total_loss-�@

error_R�^?

learning_rate_1��5v-_�I       6%�	a��h���A�&*;


total_loss���@

error_R��^?

learning_rate_1��5�sTI       6%�	��h���A�&*;


total_lossb��@

error_RTf8?

learning_rate_1��5���I       6%�	N�h���A�&*;


total_loss�֠@

error_R�1A?

learning_rate_1��5OC �I       6%�	2��h���A�&*;


total_loss&�@

error_R�C?

learning_rate_1��5m��	I       6%�	wדh���A�&*;


total_loss��
A

error_R�R?

learning_rate_1��5��oI       6%�	��h���A�&*;


total_lossw�@

error_R�aJ?

learning_rate_1��5���I       6%�	�]�h���A�&*;


total_loss��@

error_R�??

learning_rate_1��5���AI       6%�	��h���A�&*;


total_loss��@

error_R6eS?

learning_rate_1��5X��I       6%�	#��h���A�&*;


total_loss�,�@

error_RMO?

learning_rate_1��5���uI       6%�	�J�h���A�&*;


total_loss_��@

error_R3�>?

learning_rate_1��5WT��I       6%�	���h���A�&*;


total_loss��@

error_RdEi?

learning_rate_1��5��JrI       6%�	~ڕh���A�&*;


total_loss#��@

error_RӘO?

learning_rate_1��5ZIOI       6%�	�5�h���A�&*;


total_loss��@

error_RR�R?

learning_rate_1��5�w�I       6%�	�|�h���A�&*;


total_loss_�@

error_R<�S?

learning_rate_1��5AVMI       6%�	�ǖh���A�&*;


total_lossM��@

error_R�/Y?

learning_rate_1��5��j�I       6%�	�
�h���A�&*;


total_loss���@

error_R�@?

learning_rate_1��5Lz��I       6%�	pO�h���A�&*;


total_loss!v�@

error_RR=Q?

learning_rate_1��5?WB`I       6%�	~��h���A�&*;


total_loss���@

error_RiJ/?

learning_rate_1��57�4�I       6%�	q�h���A�&*;


total_lossD��@

error_R�He?

learning_rate_1��5��VI       6%�	{4�h���A�&*;


total_loss�c�@

error_R�^[?

learning_rate_1��5x��,I       6%�	�{�h���A�&*;


total_loss���@

error_R3�R?

learning_rate_1��5d_8�I       6%�	�Øh���A�&*;


total_loss���@

error_RE�P?

learning_rate_1��5���I       6%�	��h���A�&*;


total_loss��@

error_Rs??

learning_rate_1��5�}rI       6%�	XN�h���A�&*;


total_loss���@

error_R]IK?

learning_rate_1��5x�(=I       6%�	W��h���A�&*;


total_loss��@

error_R=�D?

learning_rate_1��5�ߩ�I       6%�	�ҙh���A�&*;


total_lossE�@

error_R;�:?

learning_rate_1��5f��I       6%�	��h���A�&*;


total_loss��@

error_R��P?

learning_rate_1��5-!+�I       6%�	&X�h���A�&*;


total_loss?�h@

error_R3<i?

learning_rate_1��5&�S�I       6%�	l��h���A�&*;


total_losscl�@

error_RιN?

learning_rate_1��5Q�BI       6%�	��h���A�&*;


total_lossQd�@

error_R=NS?

learning_rate_1��5�D,�I       6%�	�%�h���A�&*;


total_loss�RA

error_RTJ?

learning_rate_1��5?��CI       6%�	�j�h���A�&*;


total_loss�4�@

error_R6�N?

learning_rate_1��5�'.JI       6%�	Y��h���A�&*;


total_lossA�{@

error_R[K]?

learning_rate_1��5�3J�I       6%�	���h���A�&*;


total_loss�X�@

error_R�BK?

learning_rate_1��5�uusI       6%�	�8�h���A�&*;


total_loss�i�@

error_R��N?

learning_rate_1��5n(MI       6%�	ŀ�h���A�&*;


total_loss�*�@

error_R L?

learning_rate_1��5�s��I       6%�	Zǜh���A�&*;


total_loss
��@

error_R��G?

learning_rate_1��5zۖ�I       6%�	��h���A�&*;


total_lossC��@

error_R�Jg?

learning_rate_1��5XS�I       6%�	�P�h���A�&*;


total_lossT�@

error_R_�G?

learning_rate_1��5z�<I       6%�	Ӓ�h���A�&*;


total_loss��A

error_Rn�N?

learning_rate_1��5ñ�I       6%�	-�h���A�&*;


total_loss�Ȥ@

error_RŦG?

learning_rate_1��5�n!I       6%�	x>�h���A�&*;


total_loss�ڸ@

error_R)vO?

learning_rate_1��5A�qmI       6%�	w��h���A�&*;


total_loss�b�@

error_R��_?

learning_rate_1��5�A@I       6%�	�ўh���A�&*;


total_loss���@

error_R�U?

learning_rate_1��5CUU3I       6%�	�h���A�&*;


total_loss��A

error_R�O?

learning_rate_1��5t�LI       6%�	>b�h���A�&*;


total_loss�D�@

error_RT$H?

learning_rate_1��5O�AI       6%�	~��h���A�&*;


total_lossc��@

error_RWVQ?

learning_rate_1��5n�l�I       6%�	��h���A�&*;


total_loss�¶@

error_R��@?

learning_rate_1��5�vI       6%�	�1�h���A�&*;


total_loss���@

error_R��@?

learning_rate_1��5��6I       6%�	�s�h���A�&*;


total_loss�ԅ@

error_R��>?

learning_rate_1��5P�%�I       6%�	��h���A�&*;


total_loss(�@

error_Rx�I?

learning_rate_1��5��گI       6%�	{��h���A�&*;


total_loss��@

error_R��V?

learning_rate_1��53ZxI       6%�	;A�h���A�&*;


total_loss}ku@

error_R�z[?

learning_rate_1��5�'�I       6%�	₡h���A�&*;


total_loss�cy@

error_R�mC?

learning_rate_1��5(9(eI       6%�	ơh���A�&*;


total_loss�h�@

error_Rr"A?

learning_rate_1��5��~I       6%�	W
�h���A�&*;


total_loss�C�@

error_RT�R?

learning_rate_1��5��
�I       6%�	�N�h���A�&*;


total_loss� �@

error_R�I?

learning_rate_1��5�e�ZI       6%�	ؓ�h���A�&*;


total_lossX@�@

error_RI�Y?

learning_rate_1��5:8��I       6%�		עh���A�&*;


total_loss�^�@

error_R�,M?

learning_rate_1��5f8VI       6%�	H �h���A�&*;


total_loss���@

error_R�R?

learning_rate_1��5�] �I       6%�	fk�h���A�&*;


total_loss�ݨ@

error_R��E?

learning_rate_1��5&��I       6%�	˵�h���A�&*;


total_loss$�@

error_R��P?

learning_rate_1��5�
^�I       6%�	���h���A�&*;


total_loss�U�@

error_R[Q?

learning_rate_1��5�;�|I       6%�	 @�h���A�&*;


total_loss��@

error_RC@f?

learning_rate_1��5ĵ�fI       6%�	F��h���A�&*;


total_loss*P�@

error_R�J?

learning_rate_1��5�8.PI       6%�	!ɤh���A�&*;


total_lossN�@

error_R��<?

learning_rate_1��5�z*I       6%�	��h���A�&*;


total_loss���@

error_RW�U?

learning_rate_1��5�	�lI       6%�	P�h���A�&*;


total_loss_��@

error_RE*G?

learning_rate_1��5���I       6%�	5��h���A�&*;


total_losswˣ@

error_R�mN?

learning_rate_1��5J�I       6%�	`�h���A�&*;


total_losso4@

error_Ri~R?

learning_rate_1��5\�$�I       6%�	+�h���A�&*;


total_loss���@

error_RH�c?

learning_rate_1��5ϥ0�I       6%�	y�h���A�&*;


total_loss=�A

error_RZ>b?

learning_rate_1��5r�yI       6%�	��h���A�&*;


total_loss�$�@

error_R��o?

learning_rate_1��5�΍�I       6%�	��h���A�&*;


total_lossZ��@

error_R�^G?

learning_rate_1��5
�m�I       6%�	�N�h���A�&*;


total_loss��@

error_RO@L?

learning_rate_1��5�'I       6%�	ɖ�h���A�&*;


total_loss�ޣ@

error_RC�^?

learning_rate_1��5?�I       6%�	�ۧh���A�&*;


total_lossh͞@

error_R�HI?

learning_rate_1��5�4��I       6%�	f"�h���A�&*;


total_loss_��@

error_RH?

learning_rate_1��5�Z�fI       6%�	Bj�h���A�&*;


total_lossl��@

error_RrTF?

learning_rate_1��5X�YI       6%�	��h���A�&*;


total_loss���@

error_Ra�D?

learning_rate_1��5&z�:I       6%�	��h���A�&*;


total_loss\�@

error_RM@Q?

learning_rate_1��5�aE�I       6%�	P8�h���A�&*;


total_loss^Q�@

error_R��F?

learning_rate_1��53:�|I       6%�	���h���A�&*;


total_loss�@

error_R�Q?

learning_rate_1��5�jH�I       6%�	�ȩh���A�&*;


total_loss38�@

error_R*�[?

learning_rate_1��5*-TJI       6%�	��h���A�&*;


total_lossvw�@

error_Rw�??

learning_rate_1��5b>6�I       6%�	�S�h���A�&*;


total_loss;*�@

error_R��E?

learning_rate_1��5�&AI       6%�	��h���A�&*;


total_loss
- A

error_R��]?

learning_rate_1��5ʡbI       6%�	�ܪh���A�&*;


total_loss_t�@

error_R�K?

learning_rate_1��54ȥ�I       6%�	:#�h���A�&*;


total_loss��@

error_R�CI?

learning_rate_1��5�!�cI       6%�	{g�h���A�&*;


total_loss8�@

error_R�K?

learning_rate_1��5ڙ.oI       6%�	��h���A�&*;


total_loss���@

error_R@�V?

learning_rate_1��5��mI       6%�	���h���A�&*;


total_loss��y@

error_R��J?

learning_rate_1��5I�WI       6%�	�<�h���A�&*;


total_loss�	�@

error_Rx�R?

learning_rate_1��5�1�I       6%�	 ��h���A�&*;


total_loss���@

error_R6�J?

learning_rate_1��5=��I       6%�	�Ԭh���A�&*;


total_lossd��@

error_R$I?

learning_rate_1��5	ٲI       6%�	C�h���A�&*;


total_lossR�@

error_RnV?

learning_rate_1��5l�K7I       6%�	�\�h���A�&*;


total_loss�6�@

error_Raa\?

learning_rate_1��5�}��I       6%�	e��h���A�&*;


total_loss�n�@

error_R�??

learning_rate_1��5#h��I       6%�	w�h���A�&*;


total_losso�@

error_R�E?

learning_rate_1��5�۩I       6%�	�T�h���A�&*;


total_loss�E�@

error_R��??

learning_rate_1��5	���I       6%�	㚮h���A�&*;


total_loss�X�@

error_R�JN?

learning_rate_1��5���I       6%�	O�h���A�&*;


total_loss8J�@

error_R��J?

learning_rate_1��5�f�I       6%�	�)�h���A�&*;


total_loss�ɫ@

error_R�T?

learning_rate_1��5k�X�I       6%�	Zp�h���A�&*;


total_loss�@

error_R��D?

learning_rate_1��5�H`�I       6%�	4��h���A�&*;


total_lossLs�@

error_R��L?

learning_rate_1��5r��.I       6%�	��h���A�&*;


total_loss[��@

error_R��I?

learning_rate_1��5�PC5I       6%�	TO�h���A�&*;


total_lossƴ@

error_R��M?

learning_rate_1��5j[`I       6%�	B��h���A�&*;


total_loss���@

error_R��Y?

learning_rate_1��5�"I       6%�	�ݰh���A�&*;


total_loss"%A

error_R�`L?

learning_rate_1��5�b�I       6%�	A)�h���A�&*;


total_lossd�@

error_R.SI?

learning_rate_1��5�3:�I       6%�	�s�h���A�&*;


total_loss#Sy@

error_RXJ?

learning_rate_1��5���7I       6%�	~��h���A�&*;


total_loss�NA

error_R�W?

learning_rate_1��5�`I       6%�	h�h���A�&*;


total_lossJ��@

error_RJ�A?

learning_rate_1��5�ݏI       6%�	�K�h���A�&*;


total_loss!��@

error_R�2Q?

learning_rate_1��5HE��I       6%�	b��h���A�&*;


total_loss� �@

error_R�\?

learning_rate_1��5�A�>I       6%�	_ֲh���A�&*;


total_loss�}�@

error_R�Ta?

learning_rate_1��5��c�I       6%�	!�h���A�&*;


total_loss��@

error_R�tK?

learning_rate_1��5���I       6%�	�m�h���A�&*;


total_loss���@

error_R֯E?

learning_rate_1��5|�I       6%�	;��h���A�&*;


total_loss��@

error_R3�:?

learning_rate_1��5�R#�I       6%�	���h���A�&*;


total_lossz��@

error_R��N?

learning_rate_1��5��N�I       6%�	�=�h���A�&*;


total_loss]	�@

error_R�iL?

learning_rate_1��5̬�I       6%�	-�h���A�&*;


total_loss��@

error_R�B?

learning_rate_1��5jq��I       6%�	�˴h���A�&*;


total_loss�U�@

error_RM|F?

learning_rate_1��5��I       6%�	�#�h���A�'*;


total_loss���@

error_R��E?

learning_rate_1��51_I       6%�	j�h���A�'*;


total_loss��@

error_Rt�Q?

learning_rate_1��5��aI       6%�	���h���A�'*;


total_losswZ�@

error_R��=?

learning_rate_1��5��9I       6%�	��h���A�'*;


total_loss�Q�@

error_R��[?

learning_rate_1��5�dwI       6%�	�M�h���A�'*;


total_loss��@

error_R�s_?

learning_rate_1��5���I       6%�	_��h���A�'*;


total_loss���@

error_R�nL?

learning_rate_1��5�y��I       6%�	�׶h���A�'*;


total_lossBA

error_RR?

learning_rate_1��5�ɞ�I       6%�	q�h���A�'*;


total_loss���@

error_R�xR?

learning_rate_1��5����I       6%�	�b�h���A�'*;


total_loss�Œ@

error_R�J?

learning_rate_1��5	G�	I       6%�	�}�h���A�'*;


total_lossF�@

error_R�`C?

learning_rate_1��5�;v�I       6%�	ɺh���A�'*;


total_loss$
�@

error_R�L?

learning_rate_1��5zh�%I       6%�	��h���A�'*;


total_loss�՞@

error_Rh�C?

learning_rate_1��5f�I       6%�	�U�h���A�'*;


total_lossT�@

error_Rq�G?

learning_rate_1��5X�YI       6%�	���h���A�'*;


total_loss��@

error_RzC?

learning_rate_1��5����I       6%�	6�h���A�'*;


total_loss�ũ@

error_RZeO?

learning_rate_1��5���I       6%�	$*�h���A�'*;


total_loss)2�@

error_R�oJ?

learning_rate_1��5e�@I       6%�	p�h���A�'*;


total_loss�A

error_R;�H?

learning_rate_1��5w��I       6%�	&��h���A�'*;


total_loss:��@

error_R�]@?

learning_rate_1��5�uG�I       6%�	z��h���A�'*;


total_loss��A

error_R��W?

learning_rate_1��5�f�I       6%�	�9�h���A�'*;


total_loss��@

error_Rn-Y?

learning_rate_1��5�Ɯ.I       6%�	~�h���A�'*;


total_loss��@

error_R �_?

learning_rate_1��55&�I       6%�	�ʽh���A�'*;


total_lossi�@

error_R4X_?

learning_rate_1��5"UP�I       6%�	l'�h���A�'*;


total_loss�Ț@

error_R�Sc?

learning_rate_1��5�
g�I       6%�	�n�h���A�'*;


total_loss�Ԍ@

error_R�kG?

learning_rate_1��5��]I       6%�	괾h���A�'*;


total_loss��@

error_R
�P?

learning_rate_1��5 걇I       6%�	���h���A�'*;


total_loss4ڝ@

error_R�yF?

learning_rate_1��5~%��I       6%�	�>�h���A�'*;


total_loss�+�@

error_R�I?

learning_rate_1��5V�~�I       6%�	H��h���A�'*;


total_losss��@

error_R�E?

learning_rate_1��5R]��I       6%�	EĿh���A�'*;


total_loss�@

error_R�|F?

learning_rate_1��5XBHmI       6%�	��h���A�'*;


total_lossѿ@

error_RO??

learning_rate_1��5���I       6%�	K�h���A�'*;


total_loss�n�@

error_R��N?

learning_rate_1��5��I       6%�	O��h���A�'*;


total_loss��@

error_RE9Q?

learning_rate_1��5�J�I       6%�	��h���A�'*;


total_loss�|Y@

error_R��M?

learning_rate_1��5Cqr�I       6%�	��h���A�'*;


total_loss[��@

error_R�5B?

learning_rate_1��5r�n�I       6%�	�]�h���A�'*;


total_lossj�@

error_R}�W?

learning_rate_1��5��I       6%�	W��h���A�'*;


total_loss��@

error_R�XF?

learning_rate_1��5��:I       6%�	���h���A�'*;


total_loss�Շ@

error_R��D?

learning_rate_1��5��I       6%�	`*�h���A�'*;


total_loss��@

error_RnB?

learning_rate_1��5�X�I       6%�	�p�h���A�'*;


total_loss��@

error_R�TL?

learning_rate_1��5��I       6%�	ӳ�h���A�'*;


total_lossLǖ@

error_R$�E?

learning_rate_1��5�y��I       6%�	1��h���A�'*;


total_loss��@

error_R�1Z?

learning_rate_1��5"o��I       6%�		>�h���A�'*;


total_loss���@

error_R��N?

learning_rate_1��5�j�I       6%�	т�h���A�'*;


total_loss���@

error_R��V?

learning_rate_1��5/�I       6%�	��h���A�'*;


total_loss�#�@

error_Rf�H?

learning_rate_1��5�' �I       6%�	^�h���A�'*;


total_loss)��@

error_RiG?

learning_rate_1��5%L��I       6%�	�N�h���A�'*;


total_loss���@

error_R{�>?

learning_rate_1��5�CB�I       6%�	[��h���A�'*;


total_loss�d�@

error_R�X?

learning_rate_1��5���I       6%�	���h���A�'*;


total_loss�D�@

error_R��O?

learning_rate_1��5Y�t�I       6%�	�%�h���A�'*;


total_loss.��@

error_R�:L?

learning_rate_1��5���I       6%�	 v�h���A�'*;


total_lossn�@

error_RɢS?

learning_rate_1��5n�}VI       6%�	���h���A�'*;


total_loss�`�@

error_R
�3?

learning_rate_1��5��xI       6%�	��h���A�'*;


total_loss��@

error_R�+I?

learning_rate_1��5�F�\I       6%�	�^�h���A�'*;


total_loss��@

error_R�B?

learning_rate_1��5�h&�I       6%�	���h���A�'*;


total_loss{�@

error_R
S8?

learning_rate_1��5p���I       6%�	��h���A�'*;


total_loss���@

error_R?(I?

learning_rate_1��5r�!�I       6%�	fd�h���A�'*;


total_loss~�@

error_R�GR?

learning_rate_1��5�f�I       6%�	���h���A�'*;


total_loss��@

error_RڮL?

learning_rate_1��5�IrI       6%�	��h���A�'*;


total_loss�+�@

error_R��L?

learning_rate_1��5L
�I       6%�	NX�h���A�'*;


total_loss�@

error_R!)J?

learning_rate_1��5��(I       6%�	���h���A�'*;


total_lossHx�@

error_RqRV?

learning_rate_1��5��I       6%�	��h���A�'*;


total_loss�@

error_RE�V?

learning_rate_1��5�2�-I       6%�	�Y�h���A�'*;


total_loss�ׂ@

error_R�v7?

learning_rate_1��5�X�I       6%�	���h���A�'*;


total_loss�)�@

error_R�}S?

learning_rate_1��5��)I       6%�	C��h���A�'*;


total_loss�|�@

error_R��K?

learning_rate_1��5����I       6%�	3�h���A�'*;


total_loss��@

error_R��M?

learning_rate_1��5��fhI       6%�	�y�h���A�'*;


total_loss3Ŵ@

error_R_C?

learning_rate_1��5ov��I       6%�	���h���A�'*;


total_loss�tA

error_R\$X?

learning_rate_1��5���%I       6%�	�6�h���A�'*;


total_loss�Ƒ@

error_RZ�I?

learning_rate_1��5� �?I       6%�	�y�h���A�'*;


total_loss���@

error_R��f?

learning_rate_1��5P9�I       6%�	���h���A�'*;


total_loss���@

error_RT+L?

learning_rate_1��5IE!bI       6%�	u�h���A�'*;


total_loss`ߺ@

error_R�SA?

learning_rate_1��5�Ξ�I       6%�	�m�h���A�'*;


total_loss�С@

error_R��N?

learning_rate_1��5v`�|I       6%�	i��h���A�'*;


total_loss筜@

error_R,_N?

learning_rate_1��5IҢdI       6%�	���h���A�'*;


total_loss���@

error_R��^?

learning_rate_1��5E��I       6%�	^A�h���A�'*;


total_loss�%�@

error_R��C?

learning_rate_1��5c�P�I       6%�	x��h���A�'*;


total_loss�u�@

error_R�0`?

learning_rate_1��5v��=I       6%�	 ��h���A�'*;


total_lossR�@

error_R�|b?

learning_rate_1��5��J�I       6%�	P:�h���A�'*;


total_loss`�@

error_RO?

learning_rate_1��5��G�I       6%�	'��h���A�'*;


total_loss�;�@

error_R�J?

learning_rate_1��5��n�I       6%�	���h���A�'*;


total_loss��A

error_R.nT?

learning_rate_1��5�0�I       6%�	Q�h���A�'*;


total_loss�>�@

error_R6�X?

learning_rate_1��5R7N�I       6%�	�V�h���A�'*;


total_lossE��@

error_RN�V?

learning_rate_1��5�Z�wI       6%�	0��h���A�'*;


total_loss�r�@

error_RL�P?

learning_rate_1��5��aI       6%�	1��h���A�'*;


total_loss�X�@

error_R�YI?

learning_rate_1��5�nPI       6%�	��h���A�'*;


total_lossM�@

error_RtS?

learning_rate_1��5D�g�I       6%�	�f�h���A�'*;


total_lossЬ�@

error_R3(K?

learning_rate_1��5:~9NI       6%�	2��h���A�'*;


total_loss�$�@

error_R1zK?

learning_rate_1��5��%�I       6%�	��h���A�'*;


total_lossz�@

error_R��H?

learning_rate_1��5t�I       6%�	�6�h���A�'*;


total_loss6�@

error_R�A?

learning_rate_1��5��A�I       6%�	�}�h���A�'*;


total_loss�y@

error_R�C?

learning_rate_1��5��5LI       6%�	+��h���A�'*;


total_loss�r@

error_R��M?

learning_rate_1��5�V��I       6%�	j�h���A�'*;


total_loss���@

error_R3�H?

learning_rate_1��5t�ȺI       6%�	O�h���A�'*;


total_loss�t�@

error_RJ�V?

learning_rate_1��5��cI       6%�	��h���A�'*;


total_losssq�@

error_R�;?

learning_rate_1��5��RtI       6%�	S��h���A�'*;


total_loss���@

error_R�<?

learning_rate_1��5G�otI       6%�	[�h���A�'*;


total_loss���@

error_R��O?

learning_rate_1��5$W�I       6%�	[�h���A�'*;


total_loss1��@

error_R��Q?

learning_rate_1��5����I       6%�	��h���A�'*;


total_lossL A

error_R��R?

learning_rate_1��5��*I       6%�	m��h���A�'*;


total_loss˾A

error_R��H?

learning_rate_1��59�eI       6%�	%�h���A�'*;


total_loss�2�@

error_R=,P?

learning_rate_1��5���I       6%�	4g�h���A�'*;


total_loss2A�@

error_RVD?

learning_rate_1��58O�I       6%�	)��h���A�'*;


total_loss���@

error_R�J?

learning_rate_1��5���I       6%�	��h���A�'*;


total_lossl��@

error_RL�I?

learning_rate_1��5k��-I       6%�	�Z�h���A�'*;


total_loss?�@

error_RF�I?

learning_rate_1��5X�I       6%�	���h���A�'*;


total_loss�}�@

error_RF�\?

learning_rate_1��5c�_vI       6%�	���h���A�'*;


total_loss�1�@

error_R,zC?

learning_rate_1��5�(K�I       6%�	�.�h���A�'*;


total_loss�R�@

error_R8|=?

learning_rate_1��5�e�I       6%�	bv�h���A�'*;


total_loss�%�@

error_R#�;?

learning_rate_1��5�`ZI       6%�	���h���A�'*;


total_loss�@

error_R=?

learning_rate_1��5��e�I       6%�	��h���A�'*;


total_loss��@

error_R_Wc?

learning_rate_1��5�K�I       6%�	vP�h���A�'*;


total_loss��%A

error_RM�a?

learning_rate_1��5��ňI       6%�	���h���A�'*;


total_loss7G�@

error_RXYL?

learning_rate_1��5D�x�I       6%�	&��h���A�'*;


total_loss���@

error_R?�P?

learning_rate_1��5 ��I       6%�	-(�h���A�'*;


total_loss%��@

error_RW�g?

learning_rate_1��5��6�I       6%�	u�h���A�'*;


total_loss�/�@

error_R��K?

learning_rate_1��5����I       6%�	���h���A�'*;


total_loss�W�@

error_RuQ?

learning_rate_1��51 t�I       6%�	�	�h���A�'*;


total_loss6"�@

error_R�C@?

learning_rate_1��5�܈TI       6%�	M�h���A�'*;


total_lossj��@

error_RmJ?

learning_rate_1��5�X�I       6%�	
��h���A�'*;


total_loss��@

error_R�B?

learning_rate_1��5=#�I       6%�	���h���A�'*;


total_lossJ��@

error_R?)P?

learning_rate_1��5%G��I       6%�	�)�h���A�'*;


total_loss
L�@

error_R�2Q?

learning_rate_1��5D���I       6%�	�l�h���A�'*;


total_loss_	�@

error_R�bP?

learning_rate_1��5�1@�I       6%�	y��h���A�'*;


total_loss<�@

error_R S_?

learning_rate_1��5[�N�I       6%�	k��h���A�'*;


total_loss,;�@

error_R2�L?

learning_rate_1��5���I       6%�	@�h���A�'*;


total_loss�lA

error_R�N?

learning_rate_1��52q��I       6%�	ō�h���A�'*;


total_lossQ~�@

error_R�T?

learning_rate_1��5��	�I       6%�	���h���A�'*;


total_lossf8�@

error_REU?

learning_rate_1��5�M�I       6%�	��h���A�'*;


total_loss��@

error_R�:R?

learning_rate_1��5D�zI       6%�	�Y�h���A�(*;


total_loss�� A

error_RnVI?

learning_rate_1��5�w;�I       6%�	[��h���A�(*;


total_loss�@

error_R�T?

learning_rate_1��5�yI       6%�	C��h���A�(*;


total_loss�A

error_R��X?

learning_rate_1��5Q��I       6%�	Y,�h���A�(*;


total_lossW�@

error_R4>?

learning_rate_1��5���I       6%�	�n�h���A�(*;


total_losse`�@

error_R\UJ?

learning_rate_1��5B*k�I       6%�	���h���A�(*;


total_lossű�@

error_R�K?

learning_rate_1��5�]�I       6%�	=�h���A�(*;


total_lossNy�@

error_R�N?

learning_rate_1��5U�wI       6%�	HU�h���A�(*;


total_lossծ�@

error_R�yB?

learning_rate_1��5Lo�I       6%�	���h���A�(*;


total_loss���@

error_R��W?

learning_rate_1��5�D�I       6%�	���h���A�(*;


total_loss�r�@

error_Rϻd?

learning_rate_1��5��)�I       6%�	Z0�h���A�(*;


total_loss;ԫ@

error_R�G?

learning_rate_1��5.���I       6%�	�s�h���A�(*;


total_loss���@

error_R6�C?

learning_rate_1��5U)v�I       6%�	W��h���A�(*;


total_lossr-�@

error_R\y6?

learning_rate_1��5�>��I       6%�	���h���A�(*;


total_loss��@

error_Rx�S?

learning_rate_1��5�k?�I       6%�	c>�h���A�(*;


total_losss �@

error_RmeD?

learning_rate_1��5V(l	I       6%�	���h���A�(*;


total_loss�6�@

error_R�rF?

learning_rate_1��5��cI       6%�	���h���A�(*;


total_loss��@

error_R�F?

learning_rate_1��5S�qI       6%�	��h���A�(*;


total_loss�'�@

error_R��P?

learning_rate_1��5Fy%I       6%�	QT�h���A�(*;


total_loss��@

error_R�DL?

learning_rate_1��5��gI       6%�	���h���A�(*;


total_loss�?�@

error_R��S?

learning_rate_1��5>&�I       6%�	���h���A�(*;


total_loss�,�@

error_R�L?

learning_rate_1��5��I       6%�	��h���A�(*;


total_losszR�@

error_R�O?

learning_rate_1��5����I       6%�	c�h���A�(*;


total_loss��A

error_R��B?

learning_rate_1��5�y
�I       6%�	��h���A�(*;


total_loss!<�@

error_R�lb?

learning_rate_1��5��WI       6%�	���h���A�(*;


total_loss�1�@

error_Ri O?

learning_rate_1��5ca��I       6%�	�0�h���A�(*;


total_loss:b�@

error_R� M?

learning_rate_1��5�֫I       6%�	�r�h���A�(*;


total_loss���@

error_R�)5?

learning_rate_1��5��I       6%�	<��h���A�(*;


total_loss	X�@

error_R��F?

learning_rate_1��5
��I       6%�	���h���A�(*;


total_loss-�@

error_R��;?

learning_rate_1��5��^HI       6%�	�F�h���A�(*;


total_loss?)�@

error_R��8?

learning_rate_1��5�b�^I       6%�	Y��h���A�(*;


total_loss�T�@

error_R�Q@?

learning_rate_1��5R@�6I       6%�	{��h���A�(*;


total_lossƻ@

error_R��M?

learning_rate_1��5J��;I       6%�	u�h���A�(*;


total_loss�h�@

error_RM�F?

learning_rate_1��5;�M1I       6%�	pb�h���A�(*;


total_lossׂ�@

error_R�D?

learning_rate_1��5����I       6%�	���h���A�(*;


total_loss�5�@

error_R�B?

learning_rate_1��5��N;I       6%�	���h���A�(*;


total_loss�֮@

error_R��Q?

learning_rate_1��5	c �I       6%�	�A�h���A�(*;


total_lossōA

error_RJlE?

learning_rate_1��5sII       6%�	=��h���A�(*;


total_loss�B�@

error_RtuB?

learning_rate_1��5�/��I       6%�	a��h���A�(*;


total_loss-`�@

error_R��U?

learning_rate_1��5"�C�I       6%�	��h���A�(*;


total_lossȋ@

error_R��J?

learning_rate_1��5��&�I       6%�	@X�h���A�(*;


total_loss���@

error_RWl;?

learning_rate_1��5����I       6%�	қ�h���A�(*;


total_losso��@

error_R��X?

learning_rate_1��5~y�@I       6%�	���h���A�(*;


total_loss�9�@

error_R�=?

learning_rate_1��5����I       6%�	? �h���A�(*;


total_loss��@

error_R�*Q?

learning_rate_1��5�ҋI       6%�	�g�h���A�(*;


total_lossD9�@

error_RJ6Y?

learning_rate_1��5�0�I       6%�	8��h���A�(*;


total_loss���@

error_R@�G?

learning_rate_1��5>8��I       6%�	
��h���A�(*;


total_lossA�@

error_R:�F?

learning_rate_1��5��uI       6%�	qF�h���A�(*;


total_loss�5�@

error_R��G?

learning_rate_1��5=G�I       6%�	4��h���A�(*;


total_loss�\�@

error_Ri�V?

learning_rate_1��5Z�4�I       6%�	w��h���A�(*;


total_loss�x�@

error_R�M?

learning_rate_1��5TZT�I       6%�	��h���A�(*;


total_loss��@

error_R�YF?

learning_rate_1��5��ҝI       6%�	"_�h���A�(*;


total_loss)�@

error_R�jV?

learning_rate_1��5���8I       6%�	V��h���A�(*;


total_loss�|�@

error_R!IK?

learning_rate_1��5�ߘ�I       6%�	��h���A�(*;


total_loss-�@

error_R3[V?

learning_rate_1��5�W��I       6%�	�0�h���A�(*;


total_loss�,�@

error_R�J?

learning_rate_1��5䇶lI       6%�	v�h���A�(*;


total_lossA��@

error_R��H?

learning_rate_1��5p�,I       6%�	���h���A�(*;


total_loss�ٸ@

error_R�	Y?

learning_rate_1��5Tm;VI       6%�	� �h���A�(*;


total_loss�X�@

error_R	`?

learning_rate_1��5��SI       6%�	�I�h���A�(*;


total_lossD�@

error_R�L?

learning_rate_1��5*�t^I       6%�	��h���A�(*;


total_loss���@

error_R3$C?

learning_rate_1��5�2U�I       6%�	���h���A�(*;


total_lossl��@

error_R@\?

learning_rate_1��5�P��I       6%�	0�h���A�(*;


total_lossϧ@

error_R��D?

learning_rate_1��5G
�I       6%�	�`�h���A�(*;


total_loss�"G@

error_R:2S?

learning_rate_1��5p��@I       6%�	k��h���A�(*;


total_loss��b@

error_R�E3?

learning_rate_1��5d��I       6%�	�h���A�(*;


total_lossqF�@

error_RFD?

learning_rate_1��5���I       6%�	�O�h���A�(*;


total_lossG&	A

error_R�5G?

learning_rate_1��5�?ߺI       6%�	ޔ�h���A�(*;


total_lossf�@

error_R�w>?

learning_rate_1��5Ů�I       6%�	���h���A�(*;


total_lossa��@

error_R��b?

learning_rate_1��5��5 I       6%�	T�h���A�(*;


total_loss�Ҿ@

error_R{�d?

learning_rate_1��5B���I       6%�	>h�h���A�(*;


total_loss�r�@

error_R��\?

learning_rate_1��5*�"�I       6%�	���h���A�(*;


total_loss/֯@

error_R��D?

learning_rate_1��5R`�MI       6%�	���h���A�(*;


total_loss{��@

error_Rj�C?

learning_rate_1��5��4�I       6%�	08�h���A�(*;


total_loss�.�@

error_R&I?

learning_rate_1��5X�I       6%�	�z�h���A�(*;


total_lossΪ@

error_R;H?

learning_rate_1��5��=AI       6%�	ܾ�h���A�(*;


total_lossX0�@

error_RZ�T?

learning_rate_1��5�2��I       6%�	%�h���A�(*;


total_lossu�@

error_R�V?

learning_rate_1��5E�(I       6%�	>E�h���A�(*;


total_loss��@

error_R9H?

learning_rate_1��5%_�I       6%�	%��h���A�(*;


total_loss%i�@

error_R�FK?

learning_rate_1��52��I       6%�	���h���A�(*;


total_loss��@

error_R�M?

learning_rate_1��5��hI       6%�	R'�h���A�(*;


total_loss���@

error_R@�T?

learning_rate_1��58J�I       6%�	5n�h���A�(*;


total_loss.ހ@

error_R}�^?

learning_rate_1��5;�YI       6%�	G��h���A�(*;


total_loss�L�@

error_RnH?

learning_rate_1��5�ZDI       6%�	��h���A�(*;


total_loss���@

error_R�@?

learning_rate_1��5i��(I       6%�	P�h���A�(*;


total_loss�ټ@

error_R�hD?

learning_rate_1��5�]F>I       6%�	L��h���A�(*;


total_lossO�@

error_R)LC?

learning_rate_1��5�$I       6%�	���h���A�(*;


total_loss��A

error_RW�D?

learning_rate_1��5֜H�I       6%�	D�h���A�(*;


total_loss��x@

error_R�X?

learning_rate_1��5�O�tI       6%�	�_�h���A�(*;


total_loss�^�@

error_R?=H?

learning_rate_1��5����I       6%�	���h���A�(*;


total_loss�c�@

error_R�DE?

learning_rate_1��5}}I       6%�	(�h���A�(*;


total_loss�@

error_R�dP?

learning_rate_1��5܄Q�I       6%�	@O�h���A�(*;


total_loss���@

error_R�IC?

learning_rate_1��5VO�I       6%�	؛�h���A�(*;


total_lossE2�@

error_Rx�N?

learning_rate_1��5�<f�I       6%�	Q��h���A�(*;


total_lossq��@

error_R�L?

learning_rate_1��5�\x�I       6%�	BJ�h���A�(*;


total_loss7)�@

error_R�L?

learning_rate_1��5�OI       6%�	>��h���A�(*;


total_loss5â@

error_R=c?

learning_rate_1��5	y5�I       6%�	���h���A�(*;


total_loss��@

error_R�PL?

learning_rate_1��5��I�I       6%�	E+�h���A�(*;


total_loss�p�@

error_R�G?

learning_rate_1��5�6I       6%�	)r�h���A�(*;


total_loss\�@

error_R�K?

learning_rate_1��5<S�I       6%�	���h���A�(*;


total_loss$�A

error_R��K?

learning_rate_1��5�I       6%�		��h���A�(*;


total_loss�d�@

error_Ri>M?

learning_rate_1��5�G�I       6%�	;<�h���A�(*;


total_loss�9�@

error_R� S?

learning_rate_1��5���I       6%�	}�h���A�(*;


total_loss���@

error_R��R?

learning_rate_1��5΄�I       6%�	���h���A�(*;


total_lossW8�@

error_R!�N?

learning_rate_1��5�
�I       6%�	�	�h���A�(*;


total_lossi��@

error_R�Q?

learning_rate_1��5��Q�I       6%�	�~�h���A�(*;


total_loss��@

error_R8dS?

learning_rate_1��5=ŗsI       6%�	���h���A�(*;


total_loss$�@

error_R�V?

learning_rate_1��5���I       6%�	�1�h���A�(*;


total_loss�|�@

error_RgE?

learning_rate_1��5�9��I       6%�	v�h���A�(*;


total_loss�)�@

error_R[O?

learning_rate_1��5ra"�I       6%�	���h���A�(*;


total_loss��@

error_R2�E?

learning_rate_1��5��8uI       6%�	s��h���A�(*;


total_loss)��@

error_R��S?

learning_rate_1��5��NyI       6%�	�A�h���A�(*;


total_loss>�@

error_R�pN?

learning_rate_1��5 �_I       6%�	��h���A�(*;


total_loss1Az@

error_Rh�P?

learning_rate_1��5N�7I       6%�	x��h���A�(*;


total_loss�S�@

error_RcZM?

learning_rate_1��5DvȮI       6%�	\�h���A�(*;


total_loss���@

error_R1h9?

learning_rate_1��5JI��I       6%�	.[�h���A�(*;


total_loss��@

error_R�_U?

learning_rate_1��5��؏I       6%�	*��h���A�(*;


total_loss1q�@

error_RZ?G?

learning_rate_1��5ksI       6%�	���h���A�(*;


total_lossl��@

error_R��=?

learning_rate_1��5��g�I       6%�	n4�h���A�(*;


total_loss�O�@

error_R�v??

learning_rate_1��5z�BI       6%�	ay�h���A�(*;


total_loss ��@

error_R
]?

learning_rate_1��5�gI       6%�	���h���A�(*;


total_loss���@

error_R�f?

learning_rate_1��5�9�2I       6%�	]0�h���A�(*;


total_loss�4A

error_R	�H?

learning_rate_1��5�Du�I       6%�	�y�h���A�(*;


total_loss�^@

error_R�<C?

learning_rate_1��5�!�I       6%�	���h���A�(*;


total_loss��@

error_R:�C?

learning_rate_1��5��� I       6%�	�h���A�(*;


total_loss��@

error_RW�@?

learning_rate_1��5����I       6%�	�M�h���A�(*;


total_lossR��@

error_R�F?

learning_rate_1��5���I       6%�	���h���A�(*;


total_loss��@

error_R{ld?

learning_rate_1��5�,{�I       6%�	���h���A�(*;


total_lossr�@

error_R�lb?

learning_rate_1��5`�I       6%�	t% i���A�(*;


total_loss���@

error_R��E?

learning_rate_1��51@�ZI       6%�	�f i���A�)*;


total_loss�!�@

error_Ro�^?

learning_rate_1��5U�Y�I       6%�	�� i���A�)*;


total_loss,��@

error_RE�C?

learning_rate_1��5;ݻ�I       6%�	�� i���A�)*;


total_loss姘@

error_R$P?

learning_rate_1��5���I       6%�	�3i���A�)*;


total_lossb�@

error_R6R?

learning_rate_1��5��PI       6%�	�~i���A�)*;


total_loss���@

error_R�/L?

learning_rate_1��5!acI       6%�	�i���A�)*;


total_loss��@

error_R �O?

learning_rate_1��5]���I       6%�	�	i���A�)*;


total_loss���@

error_R��K?

learning_rate_1��5[g�iI       6%�	�Mi���A�)*;


total_loss�l�@

error_RW�I?

learning_rate_1��5�;�I       6%�	��i���A�)*;


total_loss�%�@

error_R�fZ?

learning_rate_1��5�1��I       6%�	��i���A�)*;


total_loss��@

error_RS8?

learning_rate_1��5(0v�I       6%�	Ki���A�)*;


total_loss��@

error_R��C?

learning_rate_1��5�L�I       6%�	�ci���A�)*;


total_lossi�A

error_R<Cj?

learning_rate_1��5>���I       6%�	��i���A�)*;


total_lossN`�@

error_R1�7?

learning_rate_1��5Z�@&I       6%�	�i���A�)*;


total_loss�.�@

error_R��??

learning_rate_1��5��?<I       6%�	P;i���A�)*;


total_loss'�@

error_R͋O?

learning_rate_1��5H���I       6%�	��i���A�)*;


total_loss��@

error_R��Y?

learning_rate_1��5.�Z*I       6%�	��i���A�)*;


total_loss��@

error_R��;?

learning_rate_1��5D�I       6%�	�
i���A�)*;


total_loss}B�@

error_RN[?

learning_rate_1��5Y
�`I       6%�	�Pi���A�)*;


total_loss6�@

error_R
�O?

learning_rate_1��5�^�I       6%�	n�i���A�)*;


total_loss�^�@

error_R��>?

learning_rate_1��5Qi�KI       6%�	w�i���A�)*;


total_loss�ʰ@

error_R��t?

learning_rate_1��5�#��I       6%�	�>i���A�)*;


total_loss��@

error_R�[?

learning_rate_1��5��ݺI       6%�	N�i���A�)*;


total_loss��A

error_R��L?

learning_rate_1��5����I       6%�	��i���A�)*;


total_lossT�@

error_R1f?

learning_rate_1��5�Tl�I       6%�	3i���A�)*;


total_lossX�J@

error_Rϐ;?

learning_rate_1��5?���I       6%�	;wi���A�)*;


total_loss&z�@

error_R�G?

learning_rate_1��5�\2�I       6%�	׻i���A�)*;


total_loss��@

error_R:?

learning_rate_1��5HWI       6%�	D i���A�)*;


total_loss�ʨ@

error_RMi<?

learning_rate_1��5E�;wI       6%�	�Di���A�)*;


total_lossa&A

error_R�X?

learning_rate_1��5:�/�I       6%�	@�i���A�)*;


total_loss\mA

error_R8�M?

learning_rate_1��5K���I       6%�	��i���A�)*;


total_loss��@

error_R\�M?

learning_rate_1��5�BQ�I       6%�	�	i���A�)*;


total_losslV�@

error_R=�M?

learning_rate_1��5L�Y�I       6%�	�Z	i���A�)*;


total_loss��@

error_R�N?

learning_rate_1��5c�/aI       6%�	��	i���A�)*;


total_loss�ȹ@

error_RC^W?

learning_rate_1��5�t(I       6%�	��	i���A�)*;


total_loss�_�@

error_R��G?

learning_rate_1��5�rI       6%�	�3
i���A�)*;


total_lossHߜ@

error_RҡD?

learning_rate_1��5u8�>I       6%�	�
i���A�)*;


total_lossP�@

error_RV?

learning_rate_1��5):��I       6%�	%�
i���A�)*;


total_loss��@

error_R��A?

learning_rate_1��5��ݗI       6%�	X	i���A�)*;


total_loss�@

error_R�D?

learning_rate_1��5���I       6%�	}Qi���A�)*;


total_lossE�@

error_R� f?

learning_rate_1��55m�I       6%�	H�i���A�)*;


total_loss��k@

error_R�K?

learning_rate_1��5���;I       6%�	{�i���A�)*;


total_lossq^�@

error_R(+K?

learning_rate_1��5p��ZI       6%�	�3i���A�)*;


total_loss$�@

error_R�G?

learning_rate_1��5�T�zI       6%�	Kyi���A�)*;


total_loss���@

error_R��J?

learning_rate_1��5�C�I       6%�	иi���A�)*;


total_lossd �@

error_R8HN?

learning_rate_1��5<��I       6%�	�i���A�)*;


total_loss��@

error_Rf&X?

learning_rate_1��5�2��I       6%�	O>i���A�)*;


total_loss��@

error_R�	P?

learning_rate_1��5�rd�I       6%�	��i���A�)*;


total_loss�F�@

error_R�T?

learning_rate_1��5��UI       6%�	#�i���A�)*;


total_loss���@

error_RH�N?

learning_rate_1��5��hI       6%�	 i���A�)*;


total_loss��@

error_R	�\?

learning_rate_1��5�I       6%�	fi���A�)*;


total_loss�/�@

error_R�7H?

learning_rate_1��5՟}�I       6%�		�i���A�)*;


total_lossEq�@

error_RyX?

learning_rate_1��5u�1�I       6%�	c�i���A�)*;


total_loss��@

error_R!�U?

learning_rate_1��5~�@�I       6%�	e4i���A�)*;


total_lossAG�@

error_R6�Y?

learning_rate_1��5�x�I       6%�	!wi���A�)*;


total_loss62�@

error_R�;?

learning_rate_1��5gAS�I       6%�	R�i���A�)*;


total_loss�T�@

error_R;4?

learning_rate_1��57���I       6%�	��i���A�)*;


total_loss�R�@

error_RnKW?

learning_rate_1��5n�NI       6%�	�>i���A�)*;


total_lossd��@

error_R}�^?

learning_rate_1��5*g�I       6%�	��i���A�)*;


total_loss��@

error_R��=?

learning_rate_1��5��?�I       6%�	��i���A�)*;


total_lossv�@

error_R6�V?

learning_rate_1��5l6vI       6%�	�i���A�)*;


total_lossa�@

error_R a?

learning_rate_1��5���I       6%�	*Ni���A�)*;


total_loss잼@

error_R�bA?

learning_rate_1��5�{I       6%�	��i���A�)*;


total_lossx��@

error_R�6S?

learning_rate_1��5�l�I       6%�	=�i���A�)*;


total_loss�@

error_R��F?

learning_rate_1��5�̡�I       6%�	ui���A�)*;


total_loss8�@

error_R�M?

learning_rate_1��5�T�I       6%�	�Zi���A�)*;


total_loss�5�@

error_Re}K?

learning_rate_1��5	�I       6%�	A�i���A�)*;


total_loss���@

error_R�S?

learning_rate_1��5�I       6%�	��i���A�)*;


total_losst��@

error_R�y\?

learning_rate_1��5q�I       6%�	1"i���A�)*;


total_loss(�@

error_R�L?

learning_rate_1��5��EI       6%�	�ei���A�)*;


total_loss/��@

error_R�\?

learning_rate_1��5�^ΝI       6%�	��i���A�)*;


total_loss��@

error_RR9?

learning_rate_1��5�/��I       6%�	(�i���A�)*;


total_loss���@

error_R�!T?

learning_rate_1��5W3HI       6%�	X=i���A�)*;


total_lossH�@

error_R1lI?

learning_rate_1��5���I       6%�	*�i���A�)*;


total_losswy�@

error_R?kH?

learning_rate_1��5ϔAqI       6%�	s�i���A�)*;


total_loss��@

error_R��\?

learning_rate_1��5.jA�I       6%�	�!i���A�)*;


total_loss�ƫ@

error_R�I?

learning_rate_1��5;4�I       6%�	�ji���A�)*;


total_loss��@

error_R�O\?

learning_rate_1��5 ��I       6%�	�i���A�)*;


total_lossʷ�@

error_R�]N?

learning_rate_1��5&��I       6%�	� i���A�)*;


total_losso�A

error_R�CM?

learning_rate_1��5��N�I       6%�	�Zi���A�)*;


total_loss7�@

error_R�^?

learning_rate_1��5#��I       6%�	Ξi���A�)*;


total_loss�*�@

error_R^?

learning_rate_1��5
��I       6%�	�i���A�)*;


total_lossMPA

error_R<W?

learning_rate_1��5�χ�I       6%�	/i���A�)*;


total_loss=�@

error_R��4?

learning_rate_1��5!�riI       6%�	)zi���A�)*;


total_loss�f�@

error_R�~P?

learning_rate_1��5\W��I       6%�	��i���A�)*;


total_loss���@

error_R��J?

learning_rate_1��5`Ԗ�I       6%�	�
i���A�)*;


total_loss�֣@

error_R�Kg?

learning_rate_1��5�0:,I       6%�	Ui���A�)*;


total_loss�I�@

error_R��??

learning_rate_1��5k̷I       6%�	��i���A�)*;


total_lossX�n@

error_R;�\?

learning_rate_1��5�e�^I       6%�	a�i���A�)*;


total_loss�hn@

error_R�QI?

learning_rate_1��5:��I       6%�	�&i���A�)*;


total_loss��@

error_R�C?

learning_rate_1��5&�<"I       6%�	�li���A�)*;


total_lossaI�@

error_R�iE?

learning_rate_1��5�ŹI       6%�	�i���A�)*;


total_lossa��@

error_R�Z?

learning_rate_1��5�*	BI       6%�	��i���A�)*;


total_lossn��@

error_R$�L?

learning_rate_1��5J���I       6%�	�>i���A�)*;


total_loss�X�@

error_R��[?

learning_rate_1��5C�aI       6%�	z�i���A�)*;


total_lossd��@

error_R��E?

learning_rate_1��5|��I       6%�	c�i���A�)*;


total_lossf��@

error_R��Z?

learning_rate_1��5��f�I       6%�	�i���A�)*;


total_loss�Ǘ@

error_R#_Q?

learning_rate_1��5a�lI       6%�	bVi���A�)*;


total_loss�@

error_Ri�Q?

learning_rate_1��5peI       6%�	ۗi���A�)*;


total_lossL��@

error_RQ�Q?

learning_rate_1��5�'h�I       6%�	��i���A�)*;


total_loss�#�@

error_R�P?

learning_rate_1��5%�@!I       6%�	Pi���A�)*;


total_lossc��@

error_R��S?

learning_rate_1��5�ܥ�I       6%�	v]i���A�)*;


total_lossp�@

error_R�)I?

learning_rate_1��5�_�I       6%�	a�i���A�)*;


total_losss�Y@

error_Rm�W?

learning_rate_1��5P;j�I       6%�	7�i���A�)*;


total_loss�.�@

error_R�4?

learning_rate_1��5���vI       6%�	R'i���A�)*;


total_loss�@

error_R��N?

learning_rate_1��5��I       6%�	�hi���A�)*;


total_loss��@

error_R�L?

learning_rate_1��5-���I       6%�	g�i���A�)*;


total_lossWY�@

error_RXM?

learning_rate_1��5k��I       6%�	ri���A�)*;


total_loss49A

error_R~C?

learning_rate_1��5jۊI       6%�	^\i���A�)*;


total_loss�4�@

error_R�L?

learning_rate_1��5����I       6%�	�i���A�)*;


total_loss>�@

error_RE�B?

learning_rate_1��5"���I       6%�	��i���A�)*;


total_loss䗤@

error_R۳R?

learning_rate_1��5�~s4I       6%�	+i���A�)*;


total_loss��@

error_Ra�S?

learning_rate_1��5�ږ�I       6%�	.ni���A�)*;


total_loss�z�@

error_R�{Y?

learning_rate_1��5eq�I       6%�	,�i���A�)*;


total_loss�޶@

error_R��Y?

learning_rate_1��5ůwI       6%�	��i���A�)*;


total_loss=��@

error_R�j`?

learning_rate_1��5A&yI       6%�	�A i���A�)*;


total_loss��@

error_R��C?

learning_rate_1��5~�3�I       6%�	� i���A�)*;


total_losse��@

error_Rj�U?

learning_rate_1��5|c�zI       6%�	�� i���A�)*;


total_loss���@

error_R�6S?

learning_rate_1��5��pI       6%�	h$!i���A�)*;


total_lossI�@

error_RM?

learning_rate_1��5G�/I       6%�	�m!i���A�)*;


total_lossW�@

error_R�*?

learning_rate_1��5M��I       6%�	��!i���A�)*;


total_loss�*�@

error_Rq�L?

learning_rate_1��5���I       6%�	��!i���A�)*;


total_losse��@

error_R{N?

learning_rate_1��5�ν�I       6%�	Z6"i���A�)*;


total_loss\��@

error_R.P?

learning_rate_1��5V��4I       6%�	z"i���A�)*;


total_loss���@

error_R��Y?

learning_rate_1��5��I       6%�	��"i���A�)*;


total_lossv��@

error_R��F?

learning_rate_1��5��_I       6%�	|�"i���A�)*;


total_loss�q�@

error_R@4?

learning_rate_1��5 ��I       6%�	�A#i���A�)*;


total_lossrjA

error_R��\?

learning_rate_1��5�� �I       6%�	��#i���A�)*;


total_loss!�@

error_R�pI?

learning_rate_1��5PӣhI       6%�	$�#i���A�**;


total_loss�Ǟ@

error_R8�D?

learning_rate_1��5�ՏI       6%�	�$i���A�**;


total_loss	~�@

error_Rs�U?

learning_rate_1��5	cAI       6%�	�J$i���A�**;


total_lossL��@

error_R�u[?

learning_rate_1��5,Y�8I       6%�	�$i���A�**;


total_loss\��@

error_R�gV?

learning_rate_1��5@X�dI       6%�	]�$i���A�**;


total_lossܯ�@

error_RtC??

learning_rate_1��59��I       6%�	i#%i���A�**;


total_lossx��@

error_R�DQ?

learning_rate_1��52�ZkI       6%�	�h%i���A�**;


total_loss.��@

error_R5I?

learning_rate_1��5��t�I       6%�	�%i���A�**;


total_loss���@

error_R�^S?

learning_rate_1��51S�[I       6%�	7�%i���A�**;


total_loss#��@

error_R��L?

learning_rate_1��5�� I       6%�	i:&i���A�**;


total_loss��@

error_RI�Y?

learning_rate_1��5NӼ�I       6%�	��&i���A�**;


total_loss	v@

error_RE�N?

learning_rate_1��5��� I       6%�	K�&i���A�**;


total_loss���@

error_R`�D?

learning_rate_1��5A�Q�I       6%�	�'i���A�**;


total_loss��@

error_R1�[?

learning_rate_1��5���I       6%�	�e'i���A�**;


total_loss2� A

error_RfL4?

learning_rate_1��5�_��I       6%�	Q�'i���A�**;


total_loss�P�@

error_R#�V?

learning_rate_1��5���I       6%�	��'i���A�**;


total_losss��@

error_RJ�A?

learning_rate_1��5��bI       6%�	W2(i���A�**;


total_loss��@

error_R ?H?

learning_rate_1��5"ޜI       6%�	�s(i���A�**;


total_lossT2�@

error_R�^H?

learning_rate_1��5��K�I       6%�	��(i���A�**;


total_loss|�@

error_RC�D?

learning_rate_1��5)>?�I       6%�	��(i���A�**;


total_loss[�@

error_R;_U?

learning_rate_1��5��AI       6%�	D)i���A�**;


total_loss��@

error_R$�<?

learning_rate_1��5�[m�I       6%�	��)i���A�**;


total_loss駾@

error_R��S?

learning_rate_1��5,$��I       6%�	��)i���A�**;


total_loss�ˏ@

error_R�gT?

learning_rate_1��5�PmI       6%�	/*i���A�**;


total_loss��@

error_RH�W?

learning_rate_1��5��"I       6%�	[[*i���A�**;


total_loss��A

error_RIuQ?

learning_rate_1��5g%«I       6%�	��*i���A�**;


total_lossM��@

error_R�WP?

learning_rate_1��5>G.�I       6%�	��*i���A�**;


total_loss2}�@

error_RV�A?

learning_rate_1��5�p��I       6%�	�&+i���A�**;


total_lossW��@

error_R�xZ?

learning_rate_1��5@騜I       6%�	�j+i���A�**;


total_loss�MA

error_R}�[?

learning_rate_1��5y�>.I       6%�	��+i���A�**;


total_loss&�@

error_R!�L?

learning_rate_1��5����I       6%�	o�+i���A�**;


total_lossm�@

error_R��[?

learning_rate_1��5��MtI       6%�	�0,i���A�**;


total_loss1��@

error_R� [?

learning_rate_1��5����I       6%�	�w,i���A�**;


total_loss��@

error_R�d?

learning_rate_1��5�O:~I       6%�	��,i���A�**;


total_loss)ٽ@

error_RM??

learning_rate_1��5y+_I       6%�	��,i���A�**;


total_loss|�p@

error_R�Z?

learning_rate_1��55W1,I       6%�	�C-i���A�**;


total_loss�y�@

error_R+X?

learning_rate_1��5tP�I       6%�	[�-i���A�**;


total_loss�[�@

error_Ro?:?

learning_rate_1��5ڌ�.I       6%�	��-i���A�**;


total_loss��@

error_R�q/?

learning_rate_1��5�+a<I       6%�	�3.i���A�**;


total_loss��A

error_R��X?

learning_rate_1��5ڧz9I       6%�	�w.i���A�**;


total_loss�۫@

error_R[V?

learning_rate_1��5E�١I       6%�	��.i���A�**;


total_loss�b�@

error_R]@?

learning_rate_1��5A��I       6%�	1/i���A�**;


total_loss�	�@

error_R��K?

learning_rate_1��5'��ZI       6%�	�V/i���A�**;


total_loss�_�@

error_Ra@?

learning_rate_1��5� ��I       6%�	ę/i���A�**;


total_loss���@

error_RCoE?

learning_rate_1��5�A�I       6%�	��/i���A�**;


total_loss�I�@

error_R�I?

learning_rate_1��5ځ�I       6%�	$!0i���A�**;


total_loss�:�@

error_R�HN?

learning_rate_1��5��NI       6%�	�d0i���A�**;


total_loss�A�@

error_R-gP?

learning_rate_1��5���I       6%�	��0i���A�**;


total_loss4��@

error_RH?

learning_rate_1��5FzI       6%�	��0i���A�**;


total_losst�@

error_R�U_?

learning_rate_1��5�T�I       6%�	�<1i���A�**;


total_loss#�@

error_RȠR?

learning_rate_1��5�?��I       6%�	&�1i���A�**;


total_lossT��@

error_R�I?

learning_rate_1��5��Y�I       6%�	=�1i���A�**;


total_lossiO�@

error_R�GO?

learning_rate_1��5u�,0I       6%�	2i���A�**;


total_loss)A�@

error_R�)J?

learning_rate_1��5��$I       6%�	MW2i���A�**;


total_loss�ߙ@

error_R3%^?

learning_rate_1��5�N��I       6%�	��2i���A�**;


total_loss��@

error_RWa?

learning_rate_1��5՞�,I       6%�	��2i���A�**;


total_lossO��@

error_R@?

learning_rate_1��5�!�]I       6%�	�'3i���A�**;


total_loss���@

error_R��M?

learning_rate_1��5綨�I       6%�	Mn3i���A�**;


total_loss�[�@

error_R�I?

learning_rate_1��5pcDI       6%�	$�3i���A�**;


total_loss��
A

error_R=�P?

learning_rate_1��5H_e�I       6%�	�4i���A�**;


total_loss�M�@

error_Rj�I?

learning_rate_1��5^,�$I       6%�	�E4i���A�**;


total_loss�}�@

error_Rx�T?

learning_rate_1��5*P��I       6%�	(�4i���A�**;


total_loss$��@

error_R�sF?

learning_rate_1��5Z �I       6%�	\�4i���A�**;


total_loss�c	A

error_R$C?

learning_rate_1��5��"I       6%�	n25i���A�**;


total_loss#A

error_Rt�<?

learning_rate_1��5�ЌI       6%�	5z5i���A�**;


total_lossɝA

error_R;_?

learning_rate_1��5p�&I       6%�	)�5i���A�**;


total_loss�z�@

error_RV|S?

learning_rate_1��5����I       6%�	�6i���A�**;


total_loss�0�@

error_R\�R?

learning_rate_1��5`pښI       6%�	2c6i���A�**;


total_loss��@

error_R�K`?

learning_rate_1��5��xI       6%�	a�6i���A�**;


total_loss���@

error_R�UI?

learning_rate_1��5(�u�I       6%�	b�6i���A�**;


total_lossc8�@

error_R�N?

learning_rate_1��5�­�I       6%�	j/7i���A�**;


total_loss���@

error_R�)W?

learning_rate_1��5�[��I       6%�	v7i���A�**;


total_loss���@

error_R��R?

learning_rate_1��5FM��I       6%�	��7i���A�**;


total_loss�Ќ@

error_R��I?

learning_rate_1��5UײI       6%�	�8i���A�**;


total_loss�O�@

error_R�hU?

learning_rate_1��5� ��I       6%�	/V8i���A�**;


total_losso'�@

error_R�-D?

learning_rate_1��5�W(I       6%�	��8i���A�**;


total_loss��@

error_R��K?

learning_rate_1��5��jAI       6%�	f�8i���A�**;


total_loss+m	A

error_R�*S?

learning_rate_1��5�]8�I       6%�	J#9i���A�**;


total_loss��@

error_R�H[?

learning_rate_1��5��+I       6%�	�h9i���A�**;


total_lossS�@

error_R��W?

learning_rate_1��5���I       6%�	ڭ9i���A�**;


total_lossȘ�@

error_R`Q@?

learning_rate_1��5�*n�I       6%�	��9i���A�**;


total_loss��@

error_RSO?

learning_rate_1��5�$P�I       6%�	,5:i���A�**;


total_loss�Q�@

error_R�vZ?

learning_rate_1��5(�I       6%�	5x:i���A�**;


total_loss���@

error_R�F?

learning_rate_1��5�7��I       6%�	��:i���A�**;


total_loss�'�@

error_R�L?

learning_rate_1��5?@?�I       6%�	�;i���A�**;


total_loss#%�@

error_RJ�L?

learning_rate_1��5,7/3I       6%�	3G;i���A�**;


total_loss���@

error_RqO?

learning_rate_1��5⥶)I       6%�	;i���A�**;


total_lossS��@

error_RU?

learning_rate_1��5R�|VI       6%�	T�;i���A�**;


total_loss�
�@

error_R\U?

learning_rate_1��56b�TI       6%�	�<i���A�**;


total_loss�gA

error_R�%??

learning_rate_1��5��!�I       6%�	FR<i���A�**;


total_loss@��@

error_R�D?

learning_rate_1��5��H}I       6%�	|�<i���A�**;


total_loss ��@

error_R^7?

learning_rate_1��5�\II       6%�	n�<i���A�**;


total_lossOɞ@

error_R/mZ?

learning_rate_1��5R�t�I       6%�	�"=i���A�**;


total_lossxs�@

error_R�K?

learning_rate_1��5?�\�I       6%�	fe=i���A�**;


total_loss���@

error_RwV?

learning_rate_1��5��I       6%�	�=i���A�**;


total_loss�&�@

error_R�kF?

learning_rate_1��5�덴I       6%�	�>i���A�**;


total_loss�'�@

error_R��J?

learning_rate_1��5�%�JI       6%�	a>i���A�**;


total_loss�5�@

error_R@�R?

learning_rate_1��5oqI       6%�	ۭ>i���A�**;


total_loss�l�@

error_R��Z?

learning_rate_1��5�_fI       6%�	<�>i���A�**;


total_loss#��@

error_Rק.?

learning_rate_1��5O%�I       6%�	t;?i���A�**;


total_loss_��@

error_R�b?

learning_rate_1��5�P/�I       6%�	��?i���A�**;


total_loss��@

error_R1Y<?

learning_rate_1��5���I       6%�	�?i���A�**;


total_loss��
A

error_R�F?

learning_rate_1��5s��,I       6%�	3@i���A�**;


total_lossa��@

error_R�iV?

learning_rate_1��5�(2�I       6%�	�S@i���A�**;


total_lossJ��@

error_R�3^?

learning_rate_1��5u ��I       6%�	d�@i���A�**;


total_loss%��@

error_R�V?

learning_rate_1��5���I       6%�	��@i���A�**;


total_lossŮ�@

error_R�R?

learning_rate_1��5�t�I       6%�	�Ai���A�**;


total_loss��@

error_R��C?

learning_rate_1��5�FjI       6%�	dAi���A�**;


total_loss\ّ@

error_R
�H?

learning_rate_1��5[:f�I       6%�	�Ai���A�**;


total_loss�
�@

error_R�J?

learning_rate_1��5ֱ]nI       6%�	e�Ai���A�**;


total_loss�Kf@

error_RQv=?

learning_rate_1��5���RI       6%�	W:Bi���A�**;


total_lossg�@

error_R�6S?

learning_rate_1��5�� �I       6%�	ւBi���A�**;


total_loss�s�@

error_Rd�>?

learning_rate_1��5S�1�I       6%�	�Bi���A�**;


total_loss���@

error_RA�B?

learning_rate_1��5�s�PI       6%�	�Ci���A�**;


total_loss82�@

error_R�(Y?

learning_rate_1��5"}C?I       6%�	iJCi���A�**;


total_lossm5�@

error_R�.?

learning_rate_1��5���2I       6%�	5�Ci���A�**;


total_lossM�@

error_RW�D?

learning_rate_1��5�-2�I       6%�	��Ci���A�**;


total_loss�f�@

error_R�5Z?

learning_rate_1��5�D�I       6%�	�Di���A�**;


total_loss\C�@

error_R}}K?

learning_rate_1��5�<��I       6%�	�\Di���A�**;


total_loss���@

error_R�LK?

learning_rate_1��5��4I       6%�	i�Di���A�**;


total_loss4�@

error_R��C?

learning_rate_1��5�{%I       6%�	7�Di���A�**;


total_loss3��@

error_R�AS?

learning_rate_1��5 �VFI       6%�	P(Ei���A�**;


total_loss%�@

error_R\sT?

learning_rate_1��5A�jtI       6%�	�lEi���A�**;


total_lossA��@

error_R�b?

learning_rate_1��5���I       6%�	C�Ei���A�**;


total_loss�\�@

error_Rs}Z?

learning_rate_1��55]��I       6%�	o�Ei���A�**;


total_loss���@

error_R��Y?

learning_rate_1��5:�TI       6%�	�4Fi���A�**;


total_loss��@

error_R?'L?

learning_rate_1��5�I�I       6%�	�Fi���A�**;


total_loss��@

error_ROXN?

learning_rate_1��5c	�uI       6%�	$�Fi���A�**;


total_loss)w�@

error_R(�I?

learning_rate_1��5
��I       6%�	�Gi���A�+*;


total_lossa��@

error_R=�H?

learning_rate_1��5�TRI       6%�	�cGi���A�+*;


total_loss#0�@

error_R�QE?

learning_rate_1��57��I       6%�	�Gi���A�+*;


total_loss{"�@

error_R�K?

learning_rate_1��5��e8I       6%�	a�Gi���A�+*;


total_loss�?�@

error_R�@?

learning_rate_1��5���BI       6%�	�AHi���A�+*;


total_loss О@

error_R47?

learning_rate_1��5g�j�I       6%�	��Hi���A�+*;


total_loss[I�@

error_Rϛb?

learning_rate_1��5�{y~I       6%�	u�Hi���A�+*;


total_loss���@

error_R\|\?

learning_rate_1��5���I       6%�	NIi���A�+*;


total_loss��x@

error_RH�B?

learning_rate_1��5~>��I       6%�	4^Ii���A�+*;


total_loss���@

error_R �G?

learning_rate_1��5]�I       6%�	.�Ii���A�+*;


total_loss.P�@

error_R7jJ?

learning_rate_1��5y���I       6%�	��Ii���A�+*;


total_loss�T�@

error_R8�O?

learning_rate_1��5So/GI       6%�	�:Ji���A�+*;


total_lossɱ�@

error_R��F?

learning_rate_1��5��aI       6%�	+~Ji���A�+*;


total_loss��@

error_R��Q?

learning_rate_1��5
��I       6%�	��Ji���A�+*;


total_loss��@

error_Rn�F?

learning_rate_1��5L3�oI       6%�	�Ki���A�+*;


total_loss��E@

error_R��H?

learning_rate_1��5 Ub�I       6%�	{IKi���A�+*;


total_loss���@

error_R\t>?

learning_rate_1��5��@I       6%�	��Ki���A�+*;


total_loss�w�@

error_Rf�J?

learning_rate_1��5�|,I       6%�	�Ki���A�+*;


total_loss�h�@

error_RѰN?

learning_rate_1��5͙V?I       6%�	b(Li���A�+*;


total_lossW(�@

error_R��K?

learning_rate_1��5�q�I       6%�	�nLi���A�+*;


total_loss�L�@

error_R]yA?

learning_rate_1��5���I       6%�	��Li���A�+*;


total_loss��@

error_R��Z?

learning_rate_1��5�,GI       6%�	G�Li���A�+*;


total_loss���@

error_R��P?

learning_rate_1��56o�I       6%�	N8Mi���A�+*;


total_loss�2�@

error_Ri&P?

learning_rate_1��5��=~I       6%�	�zMi���A�+*;


total_lossv��@

error_R4�D?

learning_rate_1��5ll)6I       6%�	i�Mi���A�+*;


total_lossߚ�@

error_R�#D?

learning_rate_1��5<_b9I       6%�	dNi���A�+*;


total_loss�}�@

error_R�V?

learning_rate_1��5�]I       6%�	�`Ni���A�+*;


total_lossWY�@

error_R�[\?

learning_rate_1��5��	�I       6%�	~�Ni���A�+*;


total_loss�NA

error_R��G?

learning_rate_1��5�[0rI       6%�	!�Ni���A�+*;


total_lossv�@

error_R��N?

learning_rate_1��5|��I       6%�	�;Oi���A�+*;


total_loss�!A

error_R�fJ?

learning_rate_1��5�X<aI       6%�	�~Oi���A�+*;


total_loss�u@

error_R�>?

learning_rate_1��5 4oI       6%�	��Oi���A�+*;


total_loss��@

error_R��8?

learning_rate_1��5�IeI       6%�	bPi���A�+*;


total_loss��@

error_R��[?

learning_rate_1��5®�0I       6%�	�PPi���A�+*;


total_loss���@

error_RϘ\?

learning_rate_1��5���I       6%�	
�Pi���A�+*;


total_loss֝�@

error_R��I?

learning_rate_1��5�݌RI       6%�	r�Pi���A�+*;


total_loss ��@

error_RqjT?

learning_rate_1��5�+�I       6%�	�!Qi���A�+*;


total_lossnͼ@

error_R��[?

learning_rate_1��5m��I       6%�	�cQi���A�+*;


total_loss���@

error_R�6>?

learning_rate_1��5*�M�I       6%�	l�Qi���A�+*;


total_loss���@

error_RNL?

learning_rate_1��5��GI       6%�	��Qi���A�+*;


total_loss�A

error_R�N?

learning_rate_1��5^�\iI       6%�	1-Ri���A�+*;


total_loss��A

error_R$
??

learning_rate_1��5�.7�I       6%�	�uRi���A�+*;


total_lossM�@

error_R��K?

learning_rate_1��5�-�I       6%�	��Ri���A�+*;


total_loss{��@

error_RbN?

learning_rate_1��5��I       6%�	��Ri���A�+*;


total_loss�܍@

error_RA�X?

learning_rate_1��5G#�bI       6%�	@@Si���A�+*;


total_lossA/�@

error_R��F?

learning_rate_1��5�h.I       6%�	��Si���A�+*;


total_loss$G�@

error_RD�Q?

learning_rate_1��5����I       6%�	��Si���A�+*;


total_lossſA

error_R��T?

learning_rate_1��5��I       6%�	�Ti���A�+*;


total_loss�1�@

error_R�[S?

learning_rate_1��5Ç�I       6%�	�\Ti���A�+*;


total_loss�\�@

error_R@H?

learning_rate_1��5��MI       6%�	'�Ti���A�+*;


total_loss�A�@

error_Ri�R?

learning_rate_1��5��uI       6%�	��Ti���A�+*;


total_lossHL�@

error_RZ�E?

learning_rate_1��5���4I       6%�	�IUi���A�+*;


total_loss���@

error_R G??

learning_rate_1��5��_�I       6%�	��Ui���A�+*;


total_loss���@

error_R��Q?

learning_rate_1��5n\��I       6%�	�Ui���A�+*;


total_lossaO�@

error_R�\?

learning_rate_1��5K�Z�I       6%�	�;Vi���A�+*;


total_lossꤶ@

error_Rd<?

learning_rate_1��5��\�I       6%�	3�Vi���A�+*;


total_lossi��@

error_R�2N?

learning_rate_1��5��L�I       6%�	��Vi���A�+*;


total_loss)@�@

error_R3gH?

learning_rate_1��5(�I       6%�	Wi���A�+*;


total_loss���@

error_R\;M?

learning_rate_1��5�G��I       6%�	C^Wi���A�+*;


total_loss(#�@

error_R.TK?

learning_rate_1��5)T%"I       6%�	ШWi���A�+*;


total_lossTl�@

error_R.�Q?

learning_rate_1��5��E�I       6%�	��Wi���A�+*;


total_lossI��@

error_R��V?

learning_rate_1��5���9I       6%�	;8Xi���A�+*;


total_lossu�@

error_R�@?

learning_rate_1��5�-�^I       6%�	r�Xi���A�+*;


total_lossgNA

error_Rl.0?

learning_rate_1��5���I       6%�	^�Xi���A�+*;


total_lossf��@

error_R.C?

learning_rate_1��5K��I       6%�	�Yi���A�+*;


total_loss�G�@

error_RRW?

learning_rate_1��5-GvI       6%�	MWYi���A�+*;


total_loss���@

error_RW;M?

learning_rate_1��5���I       6%�	ɛYi���A�+*;


total_loss��@

error_R�>?

learning_rate_1��5V-�I       6%�	�Yi���A�+*;


total_loss�@�@

error_R�[?

learning_rate_1��5��lI       6%�	P!Zi���A�+*;


total_loss�u�@

error_R� D?

learning_rate_1��5�:p�I       6%�	FfZi���A�+*;


total_loss���@

error_R��d?

learning_rate_1��5@��I       6%�	Q�Zi���A�+*;


total_lossԆ�@

error_R{oW?

learning_rate_1��5JzYNI       6%�	X�Zi���A�+*;


total_loss��@

error_R�>?

learning_rate_1��58�I       6%�	/[i���A�+*;


total_loss�P�@

error_R��??

learning_rate_1��5w[��I       6%�	eq[i���A�+*;


total_losst��@

error_R��O?

learning_rate_1��5��c�I       6%�	�[i���A�+*;


total_loss��A

error_RÎ^?

learning_rate_1��5�2#�I       6%�	��[i���A�+*;


total_loss
�@

error_Rl$c?

learning_rate_1��5c
�fI       6%�	�<\i���A�+*;


total_loss��@

error_R�AS?

learning_rate_1��5F�I       6%�	�\i���A�+*;


total_loss���@

error_R�??

learning_rate_1��5�]$UI       6%�	c�\i���A�+*;


total_loss��@

error_R�;?

learning_rate_1��5�1G�I       6%�	�]i���A�+*;


total_loss��@

error_RKC?

learning_rate_1��5(�xI       6%�	�Q]i���A�+*;


total_loss([A

error_R�%O?

learning_rate_1��5�X��I       6%�	}�]i���A�+*;


total_loss�@

error_R�I?

learning_rate_1��5�Y�,I       6%�	4�]i���A�+*;


total_loss��@

error_R�_?

learning_rate_1��5@�?)I       6%�	�@^i���A�+*;


total_lossd��@

error_Ra�N?

learning_rate_1��5�āI       6%�	��^i���A�+*;


total_loss�8�@

error_R3Q?

learning_rate_1��5�zI       6%�	��^i���A�+*;


total_lossaA

error_R&nU?

learning_rate_1��5<�m�I       6%�	�_i���A�+*;


total_loss�&A

error_Rx�l?

learning_rate_1��5ӱ�HI       6%�	~W_i���A�+*;


total_loss!��@

error_R�@<?

learning_rate_1��5a��I       6%�	u�_i���A�+*;


total_loss���@

error_RM^L?

learning_rate_1��5��!I       6%�	��_i���A�+*;


total_loss��@

error_RT X?

learning_rate_1��5XkO�I       6%�	!`i���A�+*;


total_loss.��@

error_R�B7?

learning_rate_1��5_�+�I       6%�	�e`i���A�+*;


total_loss��{@

error_R��R?

learning_rate_1��5D�׶I       6%�	A�`i���A�+*;


total_loss㍺@

error_RT�V?

learning_rate_1��5�B�
I       6%�	�`i���A�+*;


total_lossr��@

error_RRpM?

learning_rate_1��5��J�I       6%�	3ai���A�+*;


total_loss��@

error_RY?

learning_rate_1��5Tg��I       6%�	Atai���A�+*;


total_loss�۾@

error_R�\?

learning_rate_1��5 ͤ@I       6%�	��ai���A�+*;


total_loss��_@

error_R6�O?

learning_rate_1��5���I       6%�	R�ai���A�+*;


total_loss���@

error_R;�H?

learning_rate_1��5���qI       6%�	�Abi���A�+*;


total_loss���@

error_R��g?

learning_rate_1��5ٓ�HI       6%�	A�bi���A�+*;


total_loss�U�@

error_R#mB?

learning_rate_1��5I$AI       6%�	��bi���A�+*;


total_loss�Tw@

error_R
�T?

learning_rate_1��5(�1�I       6%�	�ci���A�+*;


total_loss�/A

error_R��Q?

learning_rate_1��5*kQI       6%�	COci���A�+*;


total_loss��@

error_R�18?

learning_rate_1��5(��I       6%�	"�ci���A�+*;


total_loss	�@

error_R*&G?

learning_rate_1��5x��I       6%�	�ci���A�+*;


total_loss��A

error_R�W?

learning_rate_1��5eA8I       6%�	.di���A�+*;


total_lossO��@

error_R�0C?

learning_rate_1��53�I       6%�	{adi���A�+*;


total_loss��@

error_R.wI?

learning_rate_1��5�f?`I       6%�	�di���A�+*;


total_losss$�@

error_R�IO?

learning_rate_1��5]8�pI       6%�	��di���A�+*;


total_loss�Z�@

error_R1R?

learning_rate_1��5�K��I       6%�	�3ei���A�+*;


total_loss:��@

error_Ra�H?

learning_rate_1��5-��(I       6%�	�wei���A�+*;


total_loss H�@

error_RJ_??

learning_rate_1��5�_ܵI       6%�	׹ei���A�+*;


total_loss]0A

error_R�u\?

learning_rate_1��5(��I       6%�	�fi���A�+*;


total_loss��@

error_Rf�K?

learning_rate_1��5�U��I       6%�	RJfi���A�+*;


total_lossꏓ@

error_RM�M?

learning_rate_1��5g�<I       6%�	َfi���A�+*;


total_loss��@

error_R]B?

learning_rate_1��5
	�I       6%�	#�fi���A�+*;


total_loss_�@

error_RH�;?

learning_rate_1��5D�I       6%�	Mgi���A�+*;


total_loss��@

error_R�Q?

learning_rate_1��5VʤI       6%�	�cgi���A�+*;


total_lossi
A

error_R�E?

learning_rate_1��5�NI       6%�	��gi���A�+*;


total_loss!Mz@

error_R�~D?

learning_rate_1��5�[X~I       6%�	H�gi���A�+*;


total_loss�C�@

error_R �J?

learning_rate_1��5`�YKI       6%�	�1hi���A�+*;


total_loss�f�@

error_R.iS?

learning_rate_1��5-�PI       6%�	�uhi���A�+*;


total_lossR��@

error_R?�T?

learning_rate_1��5-�I       6%�	��hi���A�+*;


total_loss:��@

error_R�[?

learning_rate_1��5;�a�I       6%�	Mii���A�+*;


total_loss��@

error_R�U?

learning_rate_1��5�7�oI       6%�	eOii���A�+*;


total_loss}��@

error_R�V?

learning_rate_1��5b���I       6%�	��ii���A�+*;


total_lossg�A

error_RTbV?

learning_rate_1��5���I       6%�	=�ii���A�+*;


total_lossJ�@

error_R�[A?

learning_rate_1��5�(R�I       6%�	�6ji���A�+*;


total_loss�_	A

error_R,Q?

learning_rate_1��5�c��I       6%�	yji���A�,*;


total_loss=�@

error_R�=?

learning_rate_1��5}��I       6%�	��ji���A�,*;


total_loss[?�@

error_R��U?

learning_rate_1��5�?�QI       6%�	�ki���A�,*;


total_lossC��@

error_R��S?

learning_rate_1��5���tI       6%�	PIki���A�,*;


total_loss�@

error_R�8?

learning_rate_1��5���I       6%�	s�ki���A�,*;


total_loss%��@

error_R)�I?

learning_rate_1��5r�I       6%�	�ki���A�,*;


total_loss�
�@

error_RlI?

learning_rate_1��5z��I       6%�	\li���A�,*;


total_loss!��@

error_R�I?

learning_rate_1��5�P�I       6%�	Yali���A�,*;


total_lossڔ�@

error_RV�S?

learning_rate_1��5�o��I       6%�	��li���A�,*;


total_loss�(�@

error_Rw�J?

learning_rate_1��5����I       6%�	H�li���A�,*;


total_loss]��@

error_R��M?

learning_rate_1��50��UI       6%�	N0mi���A�,*;


total_loss��@

error_R�UG?

learning_rate_1��5�ϣ�I       6%�	2smi���A�,*;


total_loss�@

error_Rw8B?

learning_rate_1��5Z�I       6%�	+�mi���A�,*;


total_loss�;A

error_R�NN?

learning_rate_1��5X(q�I       6%�	.ni���A�,*;


total_loss��@

error_R,�S?

learning_rate_1��5�3��I       6%�	Fdni���A�,*;


total_loss-�@

error_R
T?

learning_rate_1��5*�5�I       6%�	��ni���A�,*;


total_lossRM�@

error_Rԥ\?

learning_rate_1��5H�@�I       6%�	��ni���A�,*;


total_loss�:�@

error_R=B\?

learning_rate_1��5�&V�I       6%�	�6oi���A�,*;


total_loss���@

error_R6DJ?

learning_rate_1��5[���I       6%�	}oi���A�,*;


total_loss�j�@

error_R(�X?

learning_rate_1��5�TI       6%�	?�oi���A�,*;


total_loss�K�@

error_R�&P?

learning_rate_1��5c���I       6%�	�pi���A�,*;


total_lossM��@

error_R�HP?

learning_rate_1��5�b��I       6%�	�Mpi���A�,*;


total_loss%��@

error_Rl�F?

learning_rate_1��5��VI       6%�	_�pi���A�,*;


total_loss��@

error_R;�^?

learning_rate_1��5'3I       6%�	v�pi���A�,*;


total_lossW��@

error_R�I?

learning_rate_1��5|/$I       6%�	Wqi���A�,*;


total_loss�;�@

error_R�gE?

learning_rate_1��5��,I       6%�	H^qi���A�,*;


total_loss�ǟ@

error_RѡC?

learning_rate_1��5.Ԗ�I       6%�	=�qi���A�,*;


total_loss�Z�@

error_R�X?

learning_rate_1��5f��I       6%�	��qi���A�,*;


total_loss�%�@

error_R��S?

learning_rate_1��5\�I       6%�	�'ri���A�,*;


total_loss)��@

error_RhSJ?

learning_rate_1��5M���I       6%�	ori���A�,*;


total_loss ��@

error_R(a?

learning_rate_1��5O��I       6%�	��ri���A�,*;


total_loss>�@

error_R�DA?

learning_rate_1��59��I       6%�	��ri���A�,*;


total_loss$by@

error_RH�P?

learning_rate_1��5�j�I       6%�	g8si���A�,*;


total_loss�f�@

error_R��U?

learning_rate_1��5>�j�I       6%�	�ysi���A�,*;


total_loss��@

error_R;�C?

learning_rate_1��51!�I       6%�	ľsi���A�,*;


total_loss��@

error_R��Z?

learning_rate_1��5�`)�I       6%�	�ti���A�,*;


total_loss�ϻ@

error_RMM?

learning_rate_1��5Z�זI       6%�	Kti���A�,*;


total_loss�.�@

error_RS�@?

learning_rate_1��5bj�:I       6%�	��ti���A�,*;


total_loss�@

error_RO1J?

learning_rate_1��5�Y�eI       6%�	��ti���A�,*;


total_loss�ذ@

error_R�R?

learning_rate_1��5�%I       6%�	.ui���A�,*;


total_loss&`�@

error_R 7g?

learning_rate_1��5��I       6%�	Ktui���A�,*;


total_loss��@

error_RL.Y?

learning_rate_1��5[�C�I       6%�	��ui���A�,*;


total_loss���@

error_R�HH?

learning_rate_1��5��I       6%�	�vi���A�,*;


total_loss��@

error_R6?

learning_rate_1��5�B��I       6%�	f^vi���A�,*;


total_loss�z�@

error_RB[?

learning_rate_1��5�+��I       6%�	��vi���A�,*;


total_loss�ʵ@

error_R��]?

learning_rate_1��5BG�I       6%�	��vi���A�,*;


total_loss\�@

error_R�rK?

learning_rate_1��5IV��I       6%�	�-wi���A�,*;


total_loss��@

error_Ra�O?

learning_rate_1��5
[�I       6%�	�{wi���A�,*;


total_loss���@

error_R�R?

learning_rate_1��5i�$I       6%�	{�wi���A�,*;


total_loss�X@

error_R�U?

learning_rate_1��5Q�4�I       6%�	f	xi���A�,*;


total_loss$l�@

error_R�KD?

learning_rate_1��5��lI       6%�	�Kxi���A�,*;


total_loss���@

error_R�SB?

learning_rate_1��5����I       6%�	?�xi���A�,*;


total_lossԨ�@

error_Rq�W?

learning_rate_1��5�[YI       6%�	u�xi���A�,*;


total_loss(J�@

error_R&�>?

learning_rate_1��5媔�I       6%�	�%yi���A�,*;


total_lossO�A

error_Rz\I?

learning_rate_1��5�o �I       6%�	�iyi���A�,*;


total_loss'��@

error_Rj�??

learning_rate_1��5mh}�I       6%�	D�yi���A�,*;


total_loss��@

error_R�=?

learning_rate_1��5ן�YI       6%�	X�yi���A�,*;


total_loss�V�@

error_R1eE?

learning_rate_1��57i�)I       6%�	�<zi���A�,*;


total_loss�q�@

error_RX�??

learning_rate_1��5��I       6%�	҆zi���A�,*;


total_loss���@

error_R�(a?

learning_rate_1��5dSKI       6%�	;�zi���A�,*;


total_loss�+A

error_R�&Y?

learning_rate_1��5-��I       6%�	0{i���A�,*;


total_loss��@

error_R��J?

learning_rate_1��5�Bx�I       6%�	�O{i���A�,*;


total_loss5D�@

error_R3!D?

learning_rate_1��5�09�I       6%�	Ȓ{i���A�,*;


total_loss��@

error_RI�S?

learning_rate_1��5�iZI       6%�	��{i���A�,*;


total_loss���@

error_R��V?

learning_rate_1��5;�.�I       6%�	�|i���A�,*;


total_loss*��@

error_RʫC?

learning_rate_1��5W1�I       6%�	�Z|i���A�,*;


total_loss�ʟ@

error_Ra�T?

learning_rate_1��5a�I       6%�	�|i���A�,*;


total_loss��@

error_R��S?

learning_rate_1��5�(I       6%�	�|i���A�,*;


total_lossz��@

error_R�kT?

learning_rate_1��5�Pt�I       6%�	-}i���A�,*;


total_loss���@

error_R��G?

learning_rate_1��5=�{I       6%�	�t}i���A�,*;


total_loss6�
A

error_R�P?

learning_rate_1��5�ݩ�I       6%�	��}i���A�,*;


total_loss֦A

error_R�
I?

learning_rate_1��5r�#I       6%�	#~i���A�,*;


total_loss)��@

error_RORQ?

learning_rate_1��5����I       6%�	�h~i���A�,*;


total_loss܅�@

error_R��F?

learning_rate_1��5��PI       6%�	�~i���A�,*;


total_lossS3A

error_R�|N?

learning_rate_1��5�wI       6%�	!�~i���A�,*;


total_loss�*�@

error_R��M?

learning_rate_1��5�<�I       6%�	M3i���A�,*;


total_loss���@

error_R*iK?

learning_rate_1��5��3I       6%�	�wi���A�,*;


total_loss,w�@

error_RJ�U?

learning_rate_1��5p��I       6%�	��i���A�,*;


total_loss�ƥ@

error_R� F?

learning_rate_1��5�M�6I       6%�	��i���A�,*;


total_loss�A

error_R�S?

learning_rate_1��5��I       6%�	�E�i���A�,*;


total_lossԱ�@

error_R�=I?

learning_rate_1��5`	�zI       6%�	�i���A�,*;


total_loss�PA

error_R;Q?

learning_rate_1��5�g<I       6%�	�ڀi���A�,*;


total_loss�)�@

error_R�P?

learning_rate_1��52t�LI       6%�	�"�i���A�,*;


total_loss�}�@

error_R�ON?

learning_rate_1��5����I       6%�	�l�i���A�,*;


total_loss�~QA

error_R�zL?

learning_rate_1��5���NI       6%�	���i���A�,*;


total_loss�@

error_R�\S?

learning_rate_1��5?��&I       6%�	��i���A�,*;


total_lossQ�@

error_R��W?

learning_rate_1��5bI       6%�	�9�i���A�,*;


total_loss,��@

error_R�;O?

learning_rate_1��5W\3�I       6%�	�}�i���A�,*;


total_loss��@

error_R�g^?

learning_rate_1��5#�N�I       6%�	�i���A�,*;


total_lossi¹@

error_R�3d?

learning_rate_1��5rd�hI       6%�	I�i���A�,*;


total_loss���@

error_R8fC?

learning_rate_1��5�2';I       6%�	O�i���A�,*;


total_loss��@

error_R %E?

learning_rate_1��5�I       6%�	O��i���A�,*;


total_loss��q@

error_RT�E?

learning_rate_1��5uQ��I       6%�	`؃i���A�,*;


total_loss��@

error_R�7[?

learning_rate_1��5���I       6%�	��i���A�,*;


total_lossS�@

error_R��R?

learning_rate_1��5k�I       6%�	^`�i���A�,*;


total_loss�p�@

error_R?P?

learning_rate_1��5�w��I       6%�	
��i���A�,*;


total_loss��@

error_R[LK?

learning_rate_1��5�	�aI       6%�	��i���A�,*;


total_losscp�@

error_R��M?

learning_rate_1��5 �I       6%�	)�i���A�,*;


total_losszo�@

error_RZZV?

learning_rate_1��5�C>I       6%�	�m�i���A�,*;


total_lossCj�@

error_R��[?

learning_rate_1��5P>�kI       6%�	���i���A�,*;


total_loss���@

error_R;�O?

learning_rate_1��5�,#�I       6%�	���i���A�,*;


total_loss�ڪ@

error_R1�I?

learning_rate_1��5�˞qI       6%�	c:�i���A�,*;


total_loss1�@

error_R��U?

learning_rate_1��5�j�7I       6%�	�~�i���A�,*;


total_lossa��@

error_R��K?

learning_rate_1��5*%�I       6%�	a��i���A�,*;


total_loss���@

error_R�T?

learning_rate_1��5���I       6%�	��i���A�,*;


total_loss��@

error_R.D?

learning_rate_1��5��εI       6%�	hG�i���A�,*;


total_lossѣ�@

error_RTO?

learning_rate_1��5��xI       6%�	��i���A�,*;


total_losseP�@

error_R$_M?

learning_rate_1��5m�sI       6%�	�͇i���A�,*;


total_loss���@

error_Rq`W?

learning_rate_1��5r:�1I       6%�	��i���A�,*;


total_lossV��@

error_R��U?

learning_rate_1��5��QrI       6%�	W�i���A�,*;


total_lossߏ�@

error_Ri�>?

learning_rate_1��5+��I       6%�	Z��i���A�,*;


total_loss�A

error_R`�N?

learning_rate_1��5q�|I       6%�	N��i���A�,*;


total_loss�
�@

error_R�)N?

learning_rate_1��5)f�/I       6%�	�:�i���A�,*;


total_loss
A

error_R@�D?

learning_rate_1��5�wlI       6%�	���i���A�,*;


total_loss���@

error_R��U?

learning_rate_1��5 (�I       6%�	n͉i���A�,*;


total_loss���@

error_Rm�B?

learning_rate_1��5�@�I       6%�	_�i���A�,*;


total_lossM�@

error_R�]?

learning_rate_1��5C���I       6%�	Zd�i���A�,*;


total_loss��@

error_R��I?

learning_rate_1��5ĩ�BI       6%�	̥�i���A�,*;


total_loss�h�@

error_R��N?

learning_rate_1��5:��I       6%�	��i���A�,*;


total_loss��A

error_R�6H?

learning_rate_1��5��a=I       6%�	�*�i���A�,*;


total_loss��@

error_R�\X?

learning_rate_1��5��кI       6%�	�n�i���A�,*;


total_lossC��@

error_R�3_?

learning_rate_1��5p�X�I       6%�	���i���A�,*;


total_lossLx�@

error_R��`?

learning_rate_1��5M4ƉI       6%�	���i���A�,*;


total_lossʸ�@

error_R_�>?

learning_rate_1��5ũ6�I       6%�	gF�i���A�,*;


total_loss���@

error_R!-A?

learning_rate_1��5�e0I       6%�	Ɉ�i���A�,*;


total_loss�A

error_ROE?

learning_rate_1��5J�cI       6%�		͌i���A�,*;


total_loss�B�@

error_RZ�_?

learning_rate_1��5���I       6%�	��i���A�,*;


total_loss6��@

error_R�I?

learning_rate_1��5U�HI       6%�	�W�i���A�,*;


total_loss�q�@

error_R�{B?

learning_rate_1��5u�i�I       6%�	���i���A�-*;


total_lossX@

error_R6V?

learning_rate_1��5X���I       6%�	��i���A�-*;


total_loss�I�@

error_R��8?

learning_rate_1��5 %c�I       6%�	�A�i���A�-*;


total_lossD�@

error_Rc�M?

learning_rate_1��5��y�I       6%�	j��i���A�-*;


total_loss\՘@

error_R�;?

learning_rate_1��5CNE�I       6%�	�̎i���A�-*;


total_lossZ&�@

error_R$�U?

learning_rate_1��5�Y�HI       6%�	6�i���A�-*;


total_loss��@

error_R$�L?

learning_rate_1��5��PI       6%�	]�i���A�-*;


total_loss���@

error_R
�b?

learning_rate_1��5m�RI       6%�	���i���A�-*;


total_loss���@

error_R4�R?

learning_rate_1��5�|-�I       6%�	a�i���A�-*;


total_loss�g�@

error_R�L?

learning_rate_1��5�V��I       6%�	�+�i���A�-*;


total_lossj �@

error_R6�??

learning_rate_1��5���I       6%�	t�i���A�-*;


total_loss���@

error_RׄV?

learning_rate_1��5&�g�I       6%�	x��i���A�-*;


total_lossfD�@

error_Rl�P?

learning_rate_1��5��I       6%�	���i���A�-*;


total_loss3�@

error_R�.M?

learning_rate_1��5��I       6%�	�>�i���A�-*;


total_loss��@

error_R�aB?

learning_rate_1��5�E�I       6%�	���i���A�-*;


total_loss6�s@

error_R��F?

learning_rate_1��5urp:I       6%�	�ȑi���A�-*;


total_loss�z@

error_R��8?

learning_rate_1��5�[I       6%�	�i���A�-*;


total_loss��
A

error_R$m[?

learning_rate_1��5 �gI       6%�	�\�i���A�-*;


total_loss��@

error_R{�C?

learning_rate_1��5�"�!I       6%�	)��i���A�-*;


total_loss��@

error_RBP?

learning_rate_1��5��CI       6%�	��i���A�-*;


total_loss�oC@

error_R��V?

learning_rate_1��5���I       6%�	~6�i���A�-*;


total_loss+�@

error_R�VA?

learning_rate_1��5�& �I       6%�	w{�i���A�-*;


total_lossn�@

error_R �W?

learning_rate_1��5��QI       6%�	�i���A�-*;


total_loss6��@

error_R`�@?

learning_rate_1��5�R#�I       6%�	��i���A�-*;


total_loss*��@

error_R�i9?

learning_rate_1��5l��I       6%�	W�i���A�-*;


total_lossz&�@

error_R��V?

learning_rate_1��5yJ��I       6%�	i���A�-*;


total_loss`W�@

error_R$yX?

learning_rate_1��5��I       6%�	��i���A�-*;


total_lossd��@

error_R��C?

learning_rate_1��5��I       6%�	�<�i���A�-*;


total_lossq��@

error_Ra�G?

learning_rate_1��5���I       6%�	偕i���A�-*;


total_loss�uA

error_R,Qa?

learning_rate_1��5[���I       6%�	@ĕi���A�-*;


total_lossɖ�@

error_R��_?

learning_rate_1��5ѹ�ZI       6%�	�.�i���A�-*;


total_lossͿ�@

error_R��Z?

learning_rate_1��5����I       6%�	bu�i���A�-*;


total_lossS�@

error_RԡQ?

learning_rate_1��5�%�.I       6%�	���i���A�-*;


total_loss��(A

error_R�Z?

learning_rate_1��5�y�I       6%�	2��i���A�-*;


total_lossJ��@

error_R��^?

learning_rate_1��5��T�I       6%�	YD�i���A�-*;


total_loss]�@

error_R4�]?

learning_rate_1��5���MI       6%�	ً�i���A�-*;


total_loss �@

error_R:�Q?

learning_rate_1��5����I       6%�	)՗i���A�-*;


total_loss��@

error_R��_?

learning_rate_1��5ݰJI       6%�	��i���A�-*;


total_loss��=A

error_R��J?

learning_rate_1��5:�(I       6%�	:f�i���A�-*;


total_loss<��@

error_R�HW?

learning_rate_1��5�o|I       6%�	稘i���A�-*;


total_loss�9�@

error_RDM?

learning_rate_1��5J��cI       6%�	��i���A�-*;


total_lossG�@

error_R��>?

learning_rate_1��5ę�I       6%�	n4�i���A�-*;


total_lossب�@

error_R�d?

learning_rate_1��58��I       6%�	7y�i���A�-*;


total_loss}�@

error_R�J?

learning_rate_1��5����I       6%�	)��i���A�-*;


total_losscu�@

error_R	�;?

learning_rate_1��5��I       6%�	��i���A�-*;


total_lossq��@

error_R��P?

learning_rate_1��5d"��I       6%�	.H�i���A�-*;


total_loss_��@

error_R�5g?

learning_rate_1��5a ��I       6%�	슚i���A�-*;


total_loss@A

error_RV�O?

learning_rate_1��5}ԧ}I       6%�	TӚi���A�-*;


total_loss��@

error_R�N?

learning_rate_1��5�1I       6%�	?�i���A�-*;


total_loss��g@

error_RT�J?

learning_rate_1��5�8�I       6%�	|\�i���A�-*;


total_loss��@

error_Ri�V?

learning_rate_1��52��I       6%�	9��i���A�-*;


total_loss;�@

error_R��@?

learning_rate_1��50[I       6%�	(�i���A�-*;


total_loss ��@

error_Rx@?

learning_rate_1��5�3I       6%�	D1�i���A�-*;


total_lossa��@

error_R�%Y?

learning_rate_1��5]�BI       6%�	8|�i���A�-*;


total_loss�/�@

error_R�Q?

learning_rate_1��5�0��I       6%�	���i���A�-*;


total_loss�פ@

error_R�0C?

learning_rate_1��5���I       6%�	I�i���A�-*;


total_lossL�@

error_Rh�M?

learning_rate_1��5� K�I       6%�	�J�i���A�-*;


total_lossB��@

error_R�lC?

learning_rate_1��5$��wI       6%�	.��i���A�-*;


total_lossn��@

error_R4@M?

learning_rate_1��5*�u�I       6%�	�ܝi���A�-*;


total_lossWs�@

error_R) M?

learning_rate_1��532PLI       6%�	oB�i���A�-*;


total_loss۝�@

error_R�AU?

learning_rate_1��5���dI       6%�	N��i���A�-*;


total_loss
ū@

error_R�VS?

learning_rate_1��5.���I       6%�	�Оi���A�-*;


total_loss�A�@

error_R^I?

learning_rate_1��5i��@I       6%�	��i���A�-*;


total_loss�ܚ@

error_R{�B?

learning_rate_1��5^\'OI       6%�	�`�i���A�-*;


total_lossw!�@

error_R�`^?

learning_rate_1��5R��)I       6%�	q��i���A�-*;


total_loss@.�@

error_R@�Q?

learning_rate_1��5F|rzI       6%�	H�i���A�-*;


total_loss�v�@

error_R�@?

learning_rate_1��5|���I       6%�	�9�i���A�-*;


total_loss#��@

error_R��G?

learning_rate_1��5���I       6%�	���i���A�-*;


total_loss��@

error_R��C?

learning_rate_1��5ɂ�PI       6%�	�ɠi���A�-*;


total_loss;�@

error_R�J?

learning_rate_1��5��
I       6%�	��i���A�-*;


total_lossq(�@

error_R�f_?

learning_rate_1��57��'I       6%�	�P�i���A�-*;


total_loss$,�@

error_R�\?

learning_rate_1��5��|I       6%�	S��i���A�-*;


total_loss��_@

error_R�K?

learning_rate_1��5�B�I       6%�	]�i���A�-*;


total_loss���@

error_R,�S?

learning_rate_1��5�TtYI       6%�		(�i���A�-*;


total_loss�op@

error_R*pT?

learning_rate_1��5W�I       6%�	�k�i���A�-*;


total_lossO:�@

error_RjFG?

learning_rate_1��5��
�I       6%�	���i���A�-*;


total_loss�f�@

error_RT�L?

learning_rate_1��5{әI       6%�	� �i���A�-*;


total_loss=�@

error_R�v[?

learning_rate_1��5B�wPI       6%�	�E�i���A�-*;


total_loss�ŵ@

error_R�q^?

learning_rate_1��5�c�@I       6%�	�i���A�-*;


total_loss�'�@

error_R�j?

learning_rate_1��5R�I       6%�	ԣi���A�-*;


total_loss ��@

error_R�P?

learning_rate_1��5�	�I       6%�	��i���A�-*;


total_loss�@

error_R1�O?

learning_rate_1��5�N�I       6%�	Na�i���A�-*;


total_loss��@

error_R4�C?

learning_rate_1��5}�\I       6%�	���i���A�-*;


total_loss���@

error_R��S?

learning_rate_1��5�|�I       6%�	��i���A�-*;


total_loss'�@

error_R�U?

learning_rate_1��5>�I       6%�	�*�i���A�-*;


total_loss�Ջ@

error_R_�I?

learning_rate_1��5��KI       6%�	�o�i���A�-*;


total_lossx�@

error_R��C?

learning_rate_1��5�۲I       6%�	г�i���A�-*;


total_loss|p�@

error_Rz�S?

learning_rate_1��53�FQI       6%�	���i���A�-*;


total_loss��@

error_Ra_C?

learning_rate_1��57��I       6%�	�=�i���A�-*;


total_loss�h@

error_RєT?

learning_rate_1��5!���I       6%�	c�i���A�-*;


total_loss-�@

error_R�8:?

learning_rate_1��5���I       6%�	�Ʀi���A�-*;


total_lossE�v@

error_RC?

learning_rate_1��5��fTI       6%�	�
�i���A�-*;


total_loss��@

error_R�tI?

learning_rate_1��5�Z�I       6%�	hO�i���A�-*;


total_loss��	A

error_R�2A?

learning_rate_1��5X��I       6%�	n��i���A�-*;


total_loss��@

error_R��K?

learning_rate_1��5��I       6%�	ԧi���A�-*;


total_loss[�@

error_R�Q?

learning_rate_1��5UY��I       6%�	(�i���A�-*;


total_loss�w�@

error_R�J?

learning_rate_1��5!Ey�I       6%�	�[�i���A�-*;


total_loss'z@

error_R�QQ?

learning_rate_1��5�3l]I       6%�	s��i���A�-*;


total_loss{Ө@

error_RlcS?

learning_rate_1��5�o�I       6%�	��i���A�-*;


total_loss�@

error_R�wL?

learning_rate_1��5yTI       6%�	�-�i���A�-*;


total_loss�\�@

error_R3)Q?

learning_rate_1��5|��yI       6%�	p�i���A�-*;


total_loss��@

error_RJ�L?

learning_rate_1��5'~��I       6%�	충i���A�-*;


total_loss`��@

error_R_tD?

learning_rate_1��5w��I       6%�	���i���A�-*;


total_loss7�@

error_R��A?

learning_rate_1��5�UzoI       6%�	y?�i���A�-*;


total_loss��@

error_R�6?

learning_rate_1��5k�I       6%�	Ǆ�i���A�-*;


total_loss��@

error_R)�V?

learning_rate_1��5i�m�I       6%�	Ȫi���A�-*;


total_loss|>�@

error_R8�B?

learning_rate_1��55�OyI       6%�	��i���A�-*;


total_loss}��@

error_R�jC?

learning_rate_1��5���'I       6%�	�Y�i���A�-*;


total_loss�2�@

error_R�	W?

learning_rate_1��5�:3�I       6%�	ܢ�i���A�-*;


total_lossm�	A

error_R��F?

learning_rate_1��5jd��I       6%�	��i���A�-*;


total_loss@��@

error_R��L?

learning_rate_1��5D��I       6%�	�*�i���A�-*;


total_lossd.�@

error_R	YR?

learning_rate_1��5��d�I       6%�	Ny�i���A�-*;


total_loss�2�@

error_R�GT?

learning_rate_1��5E���I       6%�	ž�i���A�-*;


total_loss!�j@

error_Rl37?

learning_rate_1��5��nEI       6%�		�i���A�-*;


total_loss���@

error_RÎQ?

learning_rate_1��5�ΣI       6%�	�T�i���A�-*;


total_lossݑA

error_R!)K?

learning_rate_1��5M���I       6%�	���i���A�-*;


total_loss$A

error_RcGI?

learning_rate_1��5Zm�I       6%�	�i���A�-*;


total_loss$v$A

error_RFN?

learning_rate_1��51g�I       6%�	�M�i���A�-*;


total_loss�5�@

error_RJ@5?

learning_rate_1��5�¿I       6%�	���i���A�-*;


total_lossȔ�@

error_R��^?

learning_rate_1��5�n	�I       6%�	mޮi���A�-*;


total_loss7�@

error_R��C?

learning_rate_1��5
isVI       6%�	4&�i���A�-*;


total_loss���@

error_Ra�Q?

learning_rate_1��5*=�DI       6%�	�k�i���A�-*;


total_loss�ݸ@

error_R=|R?

learning_rate_1��5o=��I       6%�	��i���A�-*;


total_loss�EX@

error_R��U?

learning_rate_1��5w�UI       6%�	|�i���A�-*;


total_loss{��@

error_R�B_?

learning_rate_1��5f�I       6%�	�>�i���A�-*;


total_lossHU�@

error_R��[?

learning_rate_1��5��MI       6%�	Ã�i���A�-*;


total_loss%�@

error_R��:?

learning_rate_1��5��#�I       6%�	�İi���A�-*;


total_loss6��@

error_R�2?

learning_rate_1��5ڽ��I       6%�	�i���A�-*;


total_loss�Z�@

error_R�o]?

learning_rate_1��5X_s�I       6%�	�J�i���A�.*;


total_lossS_A

error_R�:4?

learning_rate_1��5]��oI       6%�	���i���A�.*;


total_loss���@

error_R��D?

learning_rate_1��5�I       6%�	ұi���A�.*;


total_lossYN�@

error_RO�<?

learning_rate_1��5�^�I       6%�	�i���A�.*;


total_loss��@

error_Rh?H?

learning_rate_1��5ͷfjI       6%�	�X�i���A�.*;


total_loss�k�@

error_Rv�F?

learning_rate_1��5츢:I       6%�	���i���A�.*;


total_loss#�@

error_R��W?

learning_rate_1��5��i<I       6%�	��i���A�.*;


total_loss�E�@

error_R�3I?

learning_rate_1��5�#HI       6%�	5,�i���A�.*;


total_loss��@

error_R��H?

learning_rate_1��5j�|<I       6%�	oo�i���A�.*;


total_loss��@

error_R�Q?

learning_rate_1��5�<I       6%�	w��i���A�.*;


total_loss/:�@

error_R�UV?

learning_rate_1��5*ʵ~I       6%�	��i���A�.*;


total_loss�ۓ@

error_R�rA?

learning_rate_1��5�P�DI       6%�	�L�i���A�.*;


total_loss��@

error_R�@?

learning_rate_1��5�p%�I       6%�	 ��i���A�.*;


total_loss�O�@

error_R��_?

learning_rate_1��5EU�I       6%�	�۴i���A�.*;


total_loss��A

error_Rd�N?

learning_rate_1��5ZQ��I       6%�	D6�i���A�.*;


total_loss�d�@

error_R�E?

learning_rate_1��5X���I       6%�	���i���A�.*;


total_lossα�@

error_R�O;?

learning_rate_1��5�4"I       6%�	C̵i���A�.*;


total_lossJ7�@

error_R��[?

learning_rate_1��5�HM�I       6%�	��i���A�.*;


total_loss(2�@

error_R�5?

learning_rate_1��5O���I       6%�	r�i���A�.*;


total_loss{��@

error_R-�??

learning_rate_1��5��,�I       6%�	X��i���A�.*;


total_losst{�@

error_R��L?

learning_rate_1��5C�YI       6%�	���i���A�.*;


total_loss@

error_R�V?

learning_rate_1��5ċ�xI       6%�	RG�i���A�.*;


total_loss;k�@

error_R�DX?

learning_rate_1��5��t�I       6%�	���i���A�.*;


total_lossr6�@

error_R)�N?

learning_rate_1��5���6I       6%�	~ӷi���A�.*;


total_loss#~�@

error_R�*G?

learning_rate_1��5��N/I       6%�	��i���A�.*;


total_loss��@

error_R�=?

learning_rate_1��5߄��I       6%�	�Y�i���A�.*;


total_loss��A

error_R�l?

learning_rate_1��5�ޑI       6%�	B��i���A�.*;


total_loss�P�@

error_Rq�I?

learning_rate_1��5��@�I       6%�	-�i���A�.*;


total_lossv �@

error_R�zJ?

learning_rate_1��5-MiI       6%�	+/�i���A�.*;


total_loss�-�@

error_R�a?

learning_rate_1��517�I       6%�	,��i���A�.*;


total_losscO�@

error_RL?

learning_rate_1��5	VcVI       6%�	L�i���A�.*;


total_loss<��@

error_RC8J?

learning_rate_1��5DUI       6%�	�M�i���A�.*;


total_lossl3�@

error_R��H?

learning_rate_1��5s3�I       6%�	���i���A�.*;


total_lossݧ�@

error_R�WL?

learning_rate_1��5��I       6%�	�պi���A�.*;


total_loss
�@

error_R�U?

learning_rate_1��5B���I       6%�	^�i���A�.*;


total_loss�K�@

error_R�}\?

learning_rate_1��5�	I       6%�	�`�i���A�.*;


total_loss���@

error_Rf&X?

learning_rate_1��5,���I       6%�	���i���A�.*;


total_loss?Y�@

error_R;�G?

learning_rate_1��5���I       6%�	�i���A�.*;


total_loss�m�@

error_R6KW?

learning_rate_1��5LG�sI       6%�	d1�i���A�.*;


total_loss� �@

error_RH�[?

learning_rate_1��5�I       6%�	�~�i���A�.*;


total_loss�~�@

error_RM�N?

learning_rate_1��5�G�I       6%�	cμi���A�.*;


total_loss�~�@

error_RW~??

learning_rate_1��51��I       6%�		�i���A�.*;


total_loss�n�@

error_Rq2L?

learning_rate_1��5�2�I       6%�	�Z�i���A�.*;


total_lossT��@

error_R1�E?

learning_rate_1��5�9d�I       6%�	r��i���A�.*;


total_loss�{�@

error_R$J?

learning_rate_1��5	�E�I       6%�	$�i���A�.*;


total_loss�̓@

error_RzM>?

learning_rate_1��5�� I       6%�	�K�i���A�.*;


total_loss_�@

error_R��N?

learning_rate_1��51iI       6%�	]��i���A�.*;


total_loss.[�@

error_RH�N?

learning_rate_1��5Cj�2I       6%�	�ܾi���A�.*;


total_loss���@

error_R%�[?

learning_rate_1��56�(OI       6%�	� �i���A�.*;


total_lossB�@

error_R��H?

learning_rate_1��5�y�JI       6%�	�d�i���A�.*;


total_lossr$�@

error_R�@?

learning_rate_1��5>�S�I       6%�	 ��i���A�.*;


total_lossx�@

error_Rʸg?

learning_rate_1��5��I       6%�	n�i���A�.*;


total_loss���@

error_RD^?

learning_rate_1��5��ǲI       6%�	�7�i���A�.*;


total_lossT�k@

error_R�[Z?

learning_rate_1��5ȄąI       6%�	i{�i���A�.*;


total_loss��A

error_R�0N?

learning_rate_1��5�NL�I       6%�	��i���A�.*;


total_lossQ�@

error_R;�D?

learning_rate_1��5"�I       6%�	���i���A�.*;


total_loss�p�@

error_R$kP?

learning_rate_1��5^�aI       6%�	tD�i���A�.*;


total_loss�Ѱ@

error_REMK?

learning_rate_1��5���KI       6%�	E��i���A�.*;


total_loss��A

error_R�C?

learning_rate_1��5,U�I       6%�	���i���A�.*;


total_loss��@

error_R7�>?

learning_rate_1��5��M�I       6%�	3�i���A�.*;


total_loss���@

error_R.Z?

learning_rate_1��5�A�I       6%�	�M�i���A�.*;


total_loss�m�@

error_RzT?

learning_rate_1��5͚~I       6%�	��i���A�.*;


total_loss#��@

error_R<�B?

learning_rate_1��5~3 yI       6%�	���i���A�.*;


total_loss�-�@

error_R�K?

learning_rate_1��5�<T�I       6%�	��i���A�.*;


total_losse��@

error_R6�P?

learning_rate_1��5:I       6%�	#]�i���A�.*;


total_loss�1�@

error_R�rg?

learning_rate_1��5�T��I       6%�	���i���A�.*;


total_lossb�@

error_R��J?

learning_rate_1��5��|@I       6%�	$��i���A�.*;


total_loss?��@

error_R&�V?

learning_rate_1��5���I       6%�	�)�i���A�.*;


total_loss��@

error_R;�A?

learning_rate_1��5��I       6%�	�u�i���A�.*;


total_loss�0�@

error_R�EA?

learning_rate_1��5��_I       6%�	���i���A�.*;


total_lossDK�@

error_R.rW?

learning_rate_1��5��0DI       6%�	�i���A�.*;


total_loss��@

error_RZ�F?

learning_rate_1��5INI       6%�	�K�i���A�.*;


total_loss�?�@

error_R�vT?

learning_rate_1��5_z(.I       6%�	L��i���A�.*;


total_loss��@

error_R�AO?

learning_rate_1��5��I       6%�	{��i���A�.*;


total_loss:B�@

error_R�NL?

learning_rate_1��5*�$�I       6%�	G"�i���A�.*;


total_lossg�@

error_RԸN?

learning_rate_1��5'���I       6%�	f�i���A�.*;


total_loss�eX@

error_R��U?

learning_rate_1��5� �I       6%�	��i���A�.*;


total_loss�@

error_R�pF?

learning_rate_1��5l%�I       6%�	l��i���A�.*;


total_loss���@

error_R�rF?

learning_rate_1��5��`3I       6%�	4�i���A�.*;


total_losss}�@

error_R��J?

learning_rate_1��5t	I       6%�	w�i���A�.*;


total_loss� �@

error_R_�R?

learning_rate_1��5l�LI       6%�	)��i���A�.*;


total_loss��@

error_R6xQ?

learning_rate_1��5x�4I       6%�	2��i���A�.*;


total_loss�´@

error_R��U?

learning_rate_1��5@OӜI       6%�	zB�i���A�.*;


total_loss)��@

error_R.I?

learning_rate_1��5)�RI       6%�	,��i���A�.*;


total_loss�@

error_RҚ[?

learning_rate_1��5��t�I       6%�	i��i���A�.*;


total_lossO��@

error_R�b?

learning_rate_1��5��I       6%�	��i���A�.*;


total_loss�ˡ@

error_R��R?

learning_rate_1��5;�9�I       6%�	�Z�i���A�.*;


total_loss*u�@

error_R��U?

learning_rate_1��5'��I       6%�	���i���A�.*;


total_loss�yA

error_RڑP?

learning_rate_1��5c��FI       6%�	��i���A�.*;


total_lossi:�@

error_RW�J?

learning_rate_1��5{��PI       6%�	�6�i���A�.*;


total_loss�)�@

error_RO9R?

learning_rate_1��5Ah��I       6%�	�}�i���A�.*;


total_loss���@

error_R]jA?

learning_rate_1��5X��I       6%�	6��i���A�.*;


total_loss�C�@

error_R2�E?

learning_rate_1��5d?��I       6%�	��i���A�.*;


total_lossF��@

error_R��G?

learning_rate_1��5��lI       6%�	zI�i���A�.*;


total_loss��c@

error_R_�D?

learning_rate_1��5��VQI       6%�	@��i���A�.*;


total_loss�A

error_R;4Y?

learning_rate_1��5b�VI       6%�	���i���A�.*;


total_loss��@

error_R��6?

learning_rate_1��5V��5I       6%�	A�i���A�.*;


total_loss�g�@

error_Rt@?

learning_rate_1��5�\�JI       6%�	U�i���A�.*;


total_loss
w@

error_R8RK?

learning_rate_1��5�zR�I       6%�	��i���A�.*;


total_loss�P�@

error_R�;S?

learning_rate_1��5 ��I       6%�	��i���A�.*;


total_loss�l�@

error_Rm`Z?

learning_rate_1��5�-�I       6%�	�$�i���A�.*;


total_loss�@

error_RMrW?

learning_rate_1��5l7эI       6%�	�m�i���A�.*;


total_loss�i�@

error_Re�I?

learning_rate_1��5��hI       6%�	���i���A�.*;


total_lossz�@

error_RsOJ?

learning_rate_1��5p�GI       6%�		�i���A�.*;


total_loss���@

error_R�5K?

learning_rate_1��5�h�I       6%�	s�i���A�.*;


total_loss�QA

error_R�K?

learning_rate_1��5Ӹ%I       6%�	[��i���A�.*;


total_lossӬ�@

error_R�N?

learning_rate_1��5���cI       6%�	���i���A�.*;


total_loss���@

error_R��D?

learning_rate_1��5��{�I       6%�	�H�i���A�.*;


total_loss��@

error_RM9D?

learning_rate_1��5B~��I       6%�	J��i���A�.*;


total_loss�6A

error_R�UU?

learning_rate_1��5�-�XI       6%�	(��i���A�.*;


total_lossl�@

error_ReG?

learning_rate_1��5lYGrI       6%�	^�i���A�.*;


total_loss��@

error_Re~@?

learning_rate_1��5i��I       6%�	e\�i���A�.*;


total_loss�|�@

error_RO8P?

learning_rate_1��5�\ NI       6%�	T��i���A�.*;


total_loss�X�@

error_R�E?

learning_rate_1��5��qI       6%�	��i���A�.*;


total_loss?��@

error_R��B?

learning_rate_1��5F)I       6%�	���i���A�.*;


total_loss�1�@

error_R
nT?

learning_rate_1��5x2,kI       6%�	�F�i���A�.*;


total_losssO�@

error_R eO?

learning_rate_1��5'+zI       6%�	]��i���A�.*;


total_loss���@

error_R�=X?

learning_rate_1��5�;8�I       6%�	S��i���A�.*;


total_loss��@

error_R��B?

learning_rate_1��5���aI       6%�	W�i���A�.*;


total_loss/^�@

error_RLS?

learning_rate_1��5�2nI       6%�	t�i���A�.*;


total_loss�k�@

error_RCB??

learning_rate_1��5M�;�I       6%�	Լ�i���A�.*;


total_lossX��@

error_R�X?

learning_rate_1��5��<�I       6%�	j �i���A�.*;


total_loss�th@

error_R��P?

learning_rate_1��5�)�I       6%�	qt�i���A�.*;


total_loss1±@

error_RR�R?

learning_rate_1��5��I       6%�	��i���A�.*;


total_loss���@

error_R&)Y?

learning_rate_1��5C�a�I       6%�	F��i���A�.*;


total_loss��@

error_RE�Z?

learning_rate_1��5�lI       6%�	�@�i���A�.*;


total_loss<6A

error_R��V?

learning_rate_1��5fa�I       6%�	:��i���A�.*;


total_loss_��@

error_R��R?

learning_rate_1��5�v�I       6%�	:��i���A�.*;


total_loss'� A

error_R4N?

learning_rate_1��5ݵ��I       6%�	e�i���A�/*;


total_loss�H�@

error_R��C?

learning_rate_1��5����I       6%�	BY�i���A�/*;


total_loss!��@

error_R@�P?

learning_rate_1��5ܖ�I       6%�	?��i���A�/*;


total_lossv[�@

error_R);?

learning_rate_1��5Q+� I       6%�	���i���A�/*;


total_lossO��@

error_R�K?

learning_rate_1��5�/��I       6%�	�&�i���A�/*;


total_loss�!�@

error_R�3B?

learning_rate_1��5XY �I       6%�	h�i���A�/*;


total_loss�y@

error_R��H?

learning_rate_1��51U��I       6%�	��i���A�/*;


total_loss��@

error_R�Q?

learning_rate_1��5n	��I       6%�	h��i���A�/*;


total_loss6!�@

error_R�U?

learning_rate_1��5M��I       6%�	T3�i���A�/*;


total_losst��@

error_Rw7`?

learning_rate_1��5Mc��I       6%�	�u�i���A�/*;


total_loss$��@

error_R�bW?

learning_rate_1��5BI       6%�	 ��i���A�/*;


total_loss�(�@

error_R��L?

learning_rate_1��5�\:�I       6%�	"��i���A�/*;


total_loss��@

error_R�:S?

learning_rate_1��5S��I       6%�	t?�i���A�/*;


total_loss��@

error_R=�W?

learning_rate_1��5�k�I       6%�	
��i���A�/*;


total_loss�1�@

error_R�bd?

learning_rate_1��5Lq�I       6%�	���i���A�/*;


total_loss8�@

error_R��[?

learning_rate_1��5�*�I       6%�	��i���A�/*;


total_loss�̜@

error_R�L?

learning_rate_1��5wO˺I       6%�	�I�i���A�/*;


total_lossjg�@

error_R�N?

learning_rate_1��5�3a�I       6%�	���i���A�/*;


total_loss��@

error_R.�??

learning_rate_1��5���8I       6%�	���i���A�/*;


total_losse}�@

error_R��I?

learning_rate_1��55�I       6%�	�i���A�/*;


total_loss�Ļ@

error_R'V?

learning_rate_1��5"/BI       6%�	QX�i���A�/*;


total_lossH��@

error_R�'J?

learning_rate_1��5VuI       6%�	���i���A�/*;


total_loss�r�@

error_R͙7?

learning_rate_1��5M���I       6%�	���i���A�/*;


total_loss-j�@

error_R��Y?

learning_rate_1��5 S�.I       6%�	5d�i���A�/*;


total_loss0�@

error_R��J?

learning_rate_1��5�_y�I       6%�	)��i���A�/*;


total_losss	�@

error_R]D?

learning_rate_1��5#��wI       6%�	��i���A�/*;


total_loss >�@

error_R)yQ?

learning_rate_1��5�B{I       6%�	qY�i���A�/*;


total_loss=q�@

error_R��K?

learning_rate_1��5���I       6%�	���i���A�/*;


total_loss�0�@

error_R~L?

learning_rate_1��5����I       6%�	~ �i���A�/*;


total_loss$]A

error_R��X?

learning_rate_1��5� Q�I       6%�	�e�i���A�/*;


total_loss���@

error_R9N?

learning_rate_1��5|�iI       6%�	��i���A�/*;


total_loss�r�@

error_RaEG?

learning_rate_1��5])J�I       6%�	e'�i���A�/*;


total_lossZRA

error_R�PX?

learning_rate_1��5��B�I       6%�	^s�i���A�/*;


total_loss�r�@

error_RW�^?

learning_rate_1��5�~3aI       6%�	p��i���A�/*;


total_loss��b@

error_R��G?

learning_rate_1��5��I       6%�	�8�i���A�/*;


total_loss��Z@

error_R��H?

learning_rate_1��5�;{I       6%�	��i���A�/*;


total_loss_[x@

error_R��M?

learning_rate_1��5�M6�I       6%�	e��i���A�/*;


total_loss&��@

error_Rȏb?

learning_rate_1��5�&�_I       6%�	��i���A�/*;


total_loss�m�@

error_R��Q?

learning_rate_1��5��I       6%�	R�i���A�/*;


total_lossTR�@

error_Rl@?

learning_rate_1��5��z�I       6%�	���i���A�/*;


total_loss� �@

error_R��Y?

learning_rate_1��59lηI       6%�	��i���A�/*;


total_loss�(�@

error_R��K?

learning_rate_1��5%ZI       6%�	�V�i���A�/*;


total_lossN��@

error_RϸN?

learning_rate_1��5�:��I       6%�	���i���A�/*;


total_lossc��@

error_R��5?

learning_rate_1��5�3kI       6%�	���i���A�/*;


total_loss�A

error_R8"D?

learning_rate_1��5w�I       6%�	�3�i���A�/*;


total_losst��@

error_R�E?

learning_rate_1��5�|4I       6%�	�i���A�/*;


total_loss��@

error_RJ�W?

learning_rate_1��5`� �I       6%�	D��i���A�/*;


total_lossL9�@

error_R��L?

learning_rate_1��5Z>6I       6%�	p�i���A�/*;


total_lossp�@

error_R�W?

learning_rate_1��5�a�I       6%�	=X�i���A�/*;


total_lossIC�@

error_R�VZ?

learning_rate_1��5
�-&I       6%�	E��i���A�/*;


total_lossC'�@

error_R��I?

learning_rate_1��5�T�*I       6%�	���i���A�/*;


total_lossr�@

error_R��I?

learning_rate_1��5�褐I       6%�	v=�i���A�/*;


total_loss؝�@

error_R��L?

learning_rate_1��5"���I       6%�	g��i���A�/*;


total_loss[��@

error_R��@?

learning_rate_1��5��>2I       6%�	u��i���A�/*;


total_loss���@

error_R�B?

learning_rate_1��5�+I       6%�	��i���A�/*;


total_lossX+A

error_R�R?

learning_rate_1��5�RI       6%�	x^�i���A�/*;


total_loss���@

error_R�=?

learning_rate_1��5œ}�I       6%�	���i���A�/*;


total_loss�
�@

error_RN?

learning_rate_1��5qh�bI       6%�	h��i���A�/*;


total_loss�g@

error_R�A?

learning_rate_1��5��U�I       6%�	?1�i���A�/*;


total_loss���@

error_RV(L?

learning_rate_1��5&ú�I       6%�	q�i���A�/*;


total_loss�q�@

error_R. >?

learning_rate_1��5��I       6%�	w��i���A�/*;


total_loss�S�@

error_R\@^?

learning_rate_1��5n��I       6%�	+��i���A�/*;


total_loss��@

error_R�E?

learning_rate_1��5y�-�I       6%�	�:�i���A�/*;


total_loss�Ps@

error_Rveq?

learning_rate_1��5>���I       6%�	��i���A�/*;


total_loss�A

error_R�B?

learning_rate_1��5e��I       6%�	���i���A�/*;


total_loss�ɾ@

error_R%H?

learning_rate_1��5���I       6%�	��i���A�/*;


total_loss��@

error_R�L?

learning_rate_1��5Z�3I       6%�	�e�i���A�/*;


total_loss��@

error_RߟN?

learning_rate_1��5��#�I       6%�	��i���A�/*;


total_loss�ˤ@

error_R�oH?

learning_rate_1��5κk�I       6%�	7G�i���A�/*;


total_loss��@

error_Ri�Q?

learning_rate_1��5��'�I       6%�	*��i���A�/*;


total_loss.ĕ@

error_R�pU?

learning_rate_1��5S��I       6%�	K��i���A�/*;


total_losst`�@

error_R�yN?

learning_rate_1��5����I       6%�	!�i���A�/*;


total_lossT�@

error_R�
N?

learning_rate_1��5�uI       6%�	\o�i���A�/*;


total_loss�p�@

error_R��Y?

learning_rate_1��5�
bI       6%�	��i���A�/*;


total_loss��@

error_R,�L?

learning_rate_1��5T��I       6%�	��i���A�/*;


total_loss���@

error_RcHT?

learning_rate_1��5`�I       6%�	�f�i���A�/*;


total_loss��@

error_Rs3S?

learning_rate_1��5`��I       6%�	��i���A�/*;


total_loss�$�@

error_R��A?

learning_rate_1��5ߠV�I       6%�	`�i���A�/*;


total_losst�@

error_R��L?

learning_rate_1��5u[�I       6%�	I�i���A�/*;


total_lossH?�@

error_Rd�K?

learning_rate_1��5L�YeI       6%�	$��i���A�/*;


total_lossU�@

error_R��[?

learning_rate_1��5R�F'I       6%�	���i���A�/*;


total_loss8k�@

error_R �U?

learning_rate_1��5��GI       6%�	�i���A�/*;


total_loss
�@

error_R�DF?

learning_rate_1��5�|�I       6%�	�c�i���A�/*;


total_loss��@

error_RL)T?

learning_rate_1��5��B�I       6%�	���i���A�/*;


total_loss��~@

error_RHS?

learning_rate_1��5�tI       6%�	���i���A�/*;


total_loss���@

error_R!�[?

learning_rate_1��5�%߫I       6%�	�6�i���A�/*;


total_loss:��@

error_R�}W?

learning_rate_1��5ͦ:�I       6%�	�|�i���A�/*;


total_loss&ٺ@

error_R.D<?

learning_rate_1��5"�$I       6%�	���i���A�/*;


total_loss ΍@

error_R��P?

learning_rate_1��5'\I       6%�	��i���A�/*;


total_loss���@

error_R�YY?

learning_rate_1��5ygI       6%�	M�i���A�/*;


total_lossOؖ@

error_R%�E?

learning_rate_1��5oR�I       6%�	M��i���A�/*;


total_loss�y�@

error_R&F?

learning_rate_1��5�zy�I       6%�	���i���A�/*;


total_lossq�@

error_R)�J?

learning_rate_1��5�\�!I       6%�	��i���A�/*;


total_loss�1�@

error_R��<?

learning_rate_1��5a��I       6%�	i�i���A�/*;


total_loss�>�@

error_R�aQ?

learning_rate_1��5��'I       6%�	0��i���A�/*;


total_loss�%�@

error_R E;?

learning_rate_1��5$f��I       6%�	���i���A�/*;


total_lossé"A

error_R=�V?

learning_rate_1��5#@�I       6%�	�0�i���A�/*;


total_loss�@

error_R8�A?

learning_rate_1��5��YI       6%�	Nu�i���A�/*;


total_loss��@

error_RԠL?

learning_rate_1��5<:�I       6%�	<��i���A�/*;


total_loss��@

error_RVME?

learning_rate_1��5`ʤzI       6%�	�i���A�/*;


total_loss'ݒ@

error_R�AZ?

learning_rate_1��5�^j�I       6%�	[�i���A�/*;


total_loss%�b@

error_R$�G?

learning_rate_1��5w��I       6%�	©�i���A�/*;


total_loss}��@

error_R{�E?

learning_rate_1��5�m�`I       6%�	q��i���A�/*;


total_lossq�@

error_RJ�=?

learning_rate_1��5�O?�I       6%�	/�i���A�/*;


total_lossӧ�@

error_R��8?

learning_rate_1��5:�SqI       6%�	�s�i���A�/*;


total_loss�շ@

error_Rn(B?

learning_rate_1��5�)�bI       6%�	���i���A�/*;


total_losss�
A

error_R�>G?

learning_rate_1��5�ݶ8I       6%�	��i���A�/*;


total_loss��@

error_R,�Y?

learning_rate_1��5��I       6%�	�=�i���A�/*;


total_loss-ɧ@

error_R�bS?

learning_rate_1��5���I       6%�	X��i���A�/*;


total_loss8�X@

error_R��S?

learning_rate_1��5̜?�I       6%�	���i���A�/*;


total_loss*�@

error_R=3C?

learning_rate_1��5(��}I       6%�	]�i���A�/*;


total_lossE0�@

error_Rs�T?

learning_rate_1��5�xgYI       6%�	�N�i���A�/*;


total_loss�3�@

error_R8�T?

learning_rate_1��5��v�I       6%�	��i���A�/*;


total_lossą�@

error_R�A?

learning_rate_1��5�ɴ�I       6%�	c��i���A�/*;


total_loss��@

error_R��S?

learning_rate_1��5,��I       6%�	%�i���A�/*;


total_lossh�@

error_R�dQ?

learning_rate_1��5=EA�I       6%�	�n�i���A�/*;


total_loss�V�@

error_R�]?

learning_rate_1��5N��I       6%�	{��i���A�/*;


total_loss �v@

error_R�JV?

learning_rate_1��5;R1�I       6%�	��i���A�/*;


total_loss�2�@

error_R�zF?

learning_rate_1��5�UHiI       6%�	tJ�i���A�/*;


total_lossN4�@

error_RTBF?

learning_rate_1��5���I       6%�	���i���A�/*;


total_loss���@

error_R- ??

learning_rate_1��54�I       6%�	��i���A�/*;


total_loss��@

error_R �K?

learning_rate_1��5��\�I       6%�	K�i���A�/*;


total_loss���@

error_Rv)8?

learning_rate_1��5���lI       6%�	R�i���A�/*;


total_loss�L0A

error_R)y\?

learning_rate_1��5���I       6%�	���i���A�/*;


total_lossjŚ@

error_R�'L?

learning_rate_1��5x�?9I       6%�	���i���A�/*;


total_loss�2�@

error_R�FX?

learning_rate_1��5Ak�qI       6%�	��i���A�/*;


total_lossdC�@

error_R�D?

learning_rate_1��5�)$�I       6%�	[�i���A�/*;


total_loss���@

error_R(D?

learning_rate_1��5���I       6%�	��i���A�/*;


total_loss���@

error_RfYG?

learning_rate_1��5�{I       6%�	���i���A�0*;


total_loss���@

error_RʹV?

learning_rate_1��5��,�I       6%�	�+�i���A�0*;


total_loss�4�@

error_R&ZL?

learning_rate_1��5ۤ�I       6%�	�o�i���A�0*;


total_loss$u�@

error_R��R?

learning_rate_1��5���I       6%�	���i���A�0*;


total_loss�2�@

error_R�%^?

learning_rate_1��5>`��I       6%�	q�i���A�0*;


total_loss�6�@

error_R=Q?

learning_rate_1��5��I       6%�	?e�i���A�0*;


total_loss�{�@

error_RTY?

learning_rate_1��5��I       6%�	���i���A�0*;


total_lossCw�@

error_RpN?

learning_rate_1��5��I       6%�	k��i���A�0*;


total_lossQ�@

error_RO?

learning_rate_1��5��*I       6%�	�:�i���A�0*;


total_lossp�@

error_R $Q?

learning_rate_1��5�Wn�I       6%�	���i���A�0*;


total_loss�~�@

error_R�8J?

learning_rate_1��5�h�EI       6%�	f��i���A�0*;


total_loss7��@

error_R��;?

learning_rate_1��5i��I       6%�	K	 j���A�0*;


total_loss���@

error_R��R?

learning_rate_1��5�'}�I       6%�	�K j���A�0*;


total_loss���@

error_R7N?

learning_rate_1��5�"�I       6%�	a� j���A�0*;


total_lossnX@

error_RC�U?

learning_rate_1��5�	hlI       6%�	)� j���A�0*;


total_loss�N�@

error_R_�Z?

learning_rate_1��5��b�I       6%�	-j���A�0*;


total_loss�@

error_R��j?

learning_rate_1��5��[I       6%�	�]j���A�0*;


total_loss���@

error_R�bN?

learning_rate_1��5��FI       6%�	��j���A�0*;


total_lossV�@

error_R E?

learning_rate_1��5�gSI       6%�	��j���A�0*;


total_loss�?A

error_R�db?

learning_rate_1��54�_�I       6%�	R/j���A�0*;


total_loss&��@

error_Rx�P?

learning_rate_1��5�.F�I       6%�	~pj���A�0*;


total_loss�[�@

error_R�<?

learning_rate_1��5�5I       6%�	òj���A�0*;


total_loss��@

error_R�@?

learning_rate_1��5���I       6%�	S�j���A�0*;


total_loss�?�@

error_R��W?

learning_rate_1��5_�|CI       6%�	\<j���A�0*;


total_loss-;@

error_R
�P?

learning_rate_1��5�GI       6%�	�}j���A�0*;


total_loss
�@

error_R�E?

learning_rate_1��5�ݐI       6%�	&�j���A�0*;


total_loss}�@

error_R��K?

learning_rate_1��5�i>I       6%�	.j���A�0*;


total_loss? �@

error_RX�S?

learning_rate_1��5�(�3I       6%�	Gj���A�0*;


total_lossX��@

error_R}H?

learning_rate_1��5��I       6%�	��j���A�0*;


total_lossW��@

error_R�hP?

learning_rate_1��5����I       6%�	V�j���A�0*;


total_loss7	�@

error_R�I^?

learning_rate_1��5A�� I       6%�	j���A�0*;


total_loss��@

error_R ~L?

learning_rate_1��5;[j?I       6%�	#\j���A�0*;


total_loss��@

error_R\{i?

learning_rate_1��57���I       6%�	ˢj���A�0*;


total_loss2	�@

error_R	R?

learning_rate_1��5l��5I       6%�	��j���A�0*;


total_losst8�@

error_R\zU?

learning_rate_1��5���I       6%�	82j���A�0*;


total_lossS\�@

error_R �D?

learning_rate_1��5;��HI       6%�	�uj���A�0*;


total_loss!�@

error_R�j\?

learning_rate_1��5˹�I       6%�	�j���A�0*;


total_loss�0�@

error_RC�8?

learning_rate_1��5�1NI       6%�	��j���A�0*;


total_loss}��@

error_Rt�H?

learning_rate_1��5i�>mI       6%�	WHj���A�0*;


total_loss+A

error_RNQ?

learning_rate_1��5�A�I       6%�	ۏj���A�0*;


total_lossC��@

error_R4(O?

learning_rate_1��5�R{�I       6%�	�j���A�0*;


total_loss��k@

error_R��T?

learning_rate_1��55*W'I       6%�	"j���A�0*;


total_lossIN�@

error_R��>?

learning_rate_1��5\*aI       6%�	�Xj���A�0*;


total_losse!�@

error_RԔS?

learning_rate_1��5�O�#I       6%�	O�j���A�0*;


total_loss�$�@

error_R�'B?

learning_rate_1��5���-I       6%�	��j���A�0*;


total_loss2�U@

error_R!O?

learning_rate_1��55{��I       6%�	I&	j���A�0*;


total_loss�ˑ@

error_R�/[?

learning_rate_1��5���+I       6%�	�k	j���A�0*;


total_loss_{�@

error_Rײh?

learning_rate_1��5o��I       6%�	�	j���A�0*;


total_loss��@

error_R�:?

learning_rate_1��5g�1�I       6%�	4�	j���A�0*;


total_loss10�@

error_RI�W?

learning_rate_1��5.���I       6%�	�;
j���A�0*;


total_loss�|�@

error_R��L?

learning_rate_1��5�%�I       6%�	��
j���A�0*;


total_loss��@

error_R��_?

learning_rate_1��5\��I       6%�	��
j���A�0*;


total_lossR:�@

error_R�@?

learning_rate_1��5d/�I       6%�	&j���A�0*;


total_loss:q_@

error_RRO?

learning_rate_1��5��vI       6%�	mYj���A�0*;


total_loss)wg@

error_R�A?

learning_rate_1��5y�I       6%�	 �j���A�0*;


total_loss���@

error_R$Q?

learning_rate_1��5��aaI       6%�	�j���A�0*;


total_loss�}�@

error_RRyG?

learning_rate_1��5���I       6%�	�$j���A�0*;


total_loss �@

error_R�F?

learning_rate_1��5^�}�I       6%�	fj���A�0*;


total_loss:�@

error_R�\Q?

learning_rate_1��5�U�I       6%�	s�j���A�0*;


total_loss���@

error_R&J?

learning_rate_1��5��نI       6%�	K�j���A�0*;


total_lossXs�@

error_R&:I?

learning_rate_1��5��&�I       6%�	G6j���A�0*;


total_loss��A

error_Rx�K?

learning_rate_1��5a�vQI       6%�	{zj���A�0*;


total_lossC_�@

error_R��I?

learning_rate_1��5\�I       6%�	S�j���A�0*;


total_loss�/�@

error_R�Ah?

learning_rate_1��5����I       6%�	@j���A�0*;


total_loss�A

error_R=PE?

learning_rate_1��5�e5QI       6%�	�bj���A�0*;


total_lossĭ@

error_R$�C?

learning_rate_1��5�8$I       6%�	c�j���A�0*;


total_loss�@�@

error_R!N?

learning_rate_1��5���mI       6%�	c�j���A�0*;


total_loss���@

error_R�9?

learning_rate_1��53M�I       6%�	�5j���A�0*;


total_lossq��@

error_R��J?

learning_rate_1��5ުȎI       6%�	�{j���A�0*;


total_lossD4�@

error_R�A?

learning_rate_1��5�II�I       6%�	e�j���A�0*;


total_lossCm�@

error_R�Q?

learning_rate_1��5ϧeQI       6%�	-
j���A�0*;


total_loss�A�@

error_R�j?

learning_rate_1��5޶�I       6%�	�Lj���A�0*;


total_lossM�@

error_R@eI?

learning_rate_1��5XaOI       6%�	[�j���A�0*;


total_lossƉ�@

error_R!I?

learning_rate_1��5vw8�I       6%�	D�j���A�0*;


total_loss���@

error_Rm�L?

learning_rate_1��5���NI       6%�	�j���A�0*;


total_lossࡳ@

error_RSI?

learning_rate_1��56�A~I       6%�	�bj���A�0*;


total_lossQ��@

error_R;�J?

learning_rate_1��5�#�PI       6%�	֥j���A�0*;


total_loss���@

error_RrgM?

learning_rate_1��5�!dMI       6%�	��j���A�0*;


total_loss���@

error_R��I?

learning_rate_1��5��4rI       6%�	&0j���A�0*;


total_loss�9�@

error_Re+D?

learning_rate_1��5^I�6I       6%�	�sj���A�0*;


total_loss7R�@

error_Rf�G?

learning_rate_1��5���I       6%�	x�j���A�0*;


total_loss&4�@

error_R�Y?

learning_rate_1��5�CqdI       6%�	��j���A�0*;


total_loss�n�@

error_R�N[?

learning_rate_1��5��rI       6%�	Cj���A�0*;


total_lossz$�@

error_RdG?

learning_rate_1��5���mI       6%�	��j���A�0*;


total_loss���@

error_RS[?

learning_rate_1��5�6M�I       6%�	�j���A�0*;


total_loss�F�@

error_RZ�J?

learning_rate_1��5'C2PI       6%�	 j���A�0*;


total_lossr¬@

error_R2�I?

learning_rate_1��5/��I       6%�	�aj���A�0*;


total_lossݏ�@

error_R��X?

learning_rate_1��5���I       6%�	��j���A�0*;


total_loss�X�@

error_R)�P?

learning_rate_1��5IV�RI       6%�	v�j���A�0*;


total_loss�1�@

error_RԭH?

learning_rate_1��5�OHI       6%�	n-j���A�0*;


total_loss�[,A

error_RS5=?

learning_rate_1��5S!�I       6%�	Њj���A�0*;


total_loss��@

error_RZ�A?

learning_rate_1��5FÌI       6%�	��j���A�0*;


total_loss	��@

error_R��V?

learning_rate_1��5haa<I       6%�	� j���A�0*;


total_loss�!�@

error_R��J?

learning_rate_1��5㐩AI       6%�	Z�j���A�0*;


total_loss (�@

error_R6�=?

learning_rate_1��5w�U�I       6%�	��j���A�0*;


total_lossns�@

error_R�t@?

learning_rate_1��5腋�I       6%�	�j���A�0*;


total_lossX��@

error_R/L?

learning_rate_1��5���I       6%�	�Tj���A�0*;


total_lossV�@

error_R��U?

learning_rate_1��5���LI       6%�	Ǘj���A�0*;


total_loss��@

error_R��P?

learning_rate_1��5�s`�I       6%�	��j���A�0*;


total_lossX�@

error_R,�N?

learning_rate_1��5�{�LI       6%�	�$j���A�0*;


total_lossI��@

error_R7�K?

learning_rate_1��5)�.I       6%�	�fj���A�0*;


total_loss�d�@

error_R� N?

learning_rate_1��5���I       6%�	��j���A�0*;


total_loss}щ@

error_R��G?

learning_rate_1��5k;��I       6%�	��j���A�0*;


total_loss��@

error_Rګg?

learning_rate_1��5b��I       6%�	|0j���A�0*;


total_loss� �@

error_R
*??

learning_rate_1��5��nZI       6%�	sj���A�0*;


total_lossn�@

error_RQ�J?

learning_rate_1��56�I       6%�	"�j���A�0*;


total_loss��@

error_R��T?

learning_rate_1��5��	I       6%�	��j���A�0*;


total_loss�&�@

error_R	�W?

learning_rate_1��5H#~{I       6%�	�Gj���A�0*;


total_loss=��@

error_R�iC?

learning_rate_1��5qn�I       6%�	�j���A�0*;


total_loss1l�@

error_R}�J?

learning_rate_1��5Y�,I       6%�	]�j���A�0*;


total_lossi�@

error_RԼF?

learning_rate_1��5Va�MI       6%�	j���A�0*;


total_lossW�@

error_R[|D?

learning_rate_1��5ء�cI       6%�	_j���A�0*;


total_loss|O�@

error_R NW?

learning_rate_1��5/y�I       6%�	�j���A�0*;


total_lossܠ�@

error_R1R?

learning_rate_1��5D�=uI       6%�	5�j���A�0*;


total_lossXņ@

error_RQ[?

learning_rate_1��5r��I       6%�	�7j���A�0*;


total_loss�x�@

error_Rc2:?

learning_rate_1��5E��dI       6%�	�j���A�0*;


total_loss)r�@

error_R�>?

learning_rate_1��5�a��I       6%�	��j���A�0*;


total_lossDH�@

error_RaK?

learning_rate_1��5����I       6%�	j���A�0*;


total_lossS��@

error_R�R?

learning_rate_1��5���I       6%�	Mj���A�0*;


total_loss��@

error_R��K?

learning_rate_1��5���I       6%�	�j���A�0*;


total_loss7@

error_R�PK?

learning_rate_1��5��I       6%�	�j���A�0*;


total_loss���@

error_R��N?

learning_rate_1��5���I       6%�	�>j���A�0*;


total_loss�Ϯ@

error_R�O?

learning_rate_1��5�m��I       6%�	�j���A�0*;


total_loss�@

error_R��T?

learning_rate_1��58>��I       6%�	7�j���A�0*;


total_loss9�@

error_R:PH?

learning_rate_1��5�s+AI       6%�	�j���A�0*;


total_loss���@

error_R.
B?

learning_rate_1��5���>I       6%�	=Vj���A�0*;


total_loss�@

error_R�T?

learning_rate_1��5�[�I       6%�	��j���A�0*;


total_loss��A

error_R`�M?

learning_rate_1��5̥�WI       6%�	��j���A�0*;


total_loss��@

error_R��6?

learning_rate_1��5C��tI       6%�	/! j���A�1*;


total_loss�8�@

error_R��5?

learning_rate_1��5t\I       6%�	�e j���A�1*;


total_loss�@

error_RM�M?

learning_rate_1��5�9�I       6%�	�� j���A�1*;


total_loss7� A

error_R�me?

learning_rate_1��5d[��I       6%�	�� j���A�1*;


total_loss �@

error_R!z(?

learning_rate_1��5����I       6%�	�/!j���A�1*;


total_loss���@

error_R\3;?

learning_rate_1��5Uڦ�I       6%�	�q!j���A�1*;


total_lossC��@

error_R�H?

learning_rate_1��5�!YI       6%�	��!j���A�1*;


total_lossvУ@

error_RfDD?

learning_rate_1��5���I       6%�	��!j���A�1*;


total_loss,��@

error_RjKF?

learning_rate_1��5��D�I       6%�	="j���A�1*;


total_loss���@

error_R_;D?

learning_rate_1��5&�<I       6%�	�"j���A�1*;


total_loss�$�@

error_R�H?

learning_rate_1��5����I       6%�	k�"j���A�1*;


total_loss��@

error_R�bU?

learning_rate_1��5�F�I       6%�	�#j���A�1*;


total_loss���@

error_R��G?

learning_rate_1��5��C�I       6%�	�J#j���A�1*;


total_loss���@

error_R��O?

learning_rate_1��5"��I       6%�		�#j���A�1*;


total_loss��@

error_R�P?

learning_rate_1��5/�8I       6%�	�#j���A�1*;


total_losssÊ@

error_Ria?

learning_rate_1��5»��I       6%�	�$j���A�1*;


total_loss'�@

error_R�UR?

learning_rate_1��5g��I       6%�	c]$j���A�1*;


total_loss
g�@

error_Rw;?

learning_rate_1��5xQRKI       6%�	0�$j���A�1*;


total_loss)�@

error_R)�m?

learning_rate_1��5|��I       6%�	I�$j���A�1*;


total_lossH�@

error_RH?

learning_rate_1��5R�
MI       6%�	A(%j���A�1*;


total_loss���@

error_R �>?

learning_rate_1��5B�9�I       6%�	�o%j���A�1*;


total_loss�h�@

error_R.�M?

learning_rate_1��5�_�I       6%�	��%j���A�1*;


total_losswݫ@

error_R��V?

learning_rate_1��5s��I       6%�	�&j���A�1*;


total_loss���@

error_R��e?

learning_rate_1��5JP��I       6%�	�H&j���A�1*;


total_loss��@

error_R(
`?

learning_rate_1��5RFI       6%�	S�&j���A�1*;


total_lossz�@

error_R�eO?

learning_rate_1��5=؁II       6%�	P�&j���A�1*;


total_lossC��@

error_R�#E?

learning_rate_1��5{�ѧI       6%�	!'j���A�1*;


total_lossTb@

error_Rf5?

learning_rate_1��5�-I       6%�	�U'j���A�1*;


total_loss3�k@

error_RCZ?

learning_rate_1��5�l8�I       6%�	�'j���A�1*;


total_loss1�JA

error_RZ�Q?

learning_rate_1��5���I       6%�	^�'j���A�1*;


total_loss��@

error_R�tT?

learning_rate_1��5>��zI       6%�	; (j���A�1*;


total_loss���@

error_R�^@?

learning_rate_1��5hnE�I       6%�	(a(j���A�1*;


total_loss@��@

error_RV�R?

learning_rate_1��5�G�3I       6%�	��(j���A�1*;


total_loss��@

error_RxFA?

learning_rate_1��5��q�I       6%�	r�(j���A�1*;


total_lossz��@

error_R#�>?

learning_rate_1��5��ZI       6%�	�')j���A�1*;


total_loss��A

error_R��P?

learning_rate_1��5��CI       6%�	Tk)j���A�1*;


total_lossF��@

error_RC&^?

learning_rate_1��57�k�I       6%�	��)j���A�1*;


total_loss���@

error_RAG?

learning_rate_1��5f�I       6%�	��)j���A�1*;


total_lossLA

error_RW�f?

learning_rate_1��5b�i�I       6%�	�0*j���A�1*;


total_loss�g�@

error_R@�F?

learning_rate_1��5�.(CI       6%�	�v*j���A�1*;


total_loss���@

error_R��G?

learning_rate_1��5�z�I       6%�	7�*j���A�1*;


total_loss��@

error_R�fN?

learning_rate_1��5���I       6%�	P�*j���A�1*;


total_loss��@

error_RHJ?

learning_rate_1��5+22�I       6%�	o>+j���A�1*;


total_loss�	�@

error_R3�S?

learning_rate_1��5
o4I       6%�	&�+j���A�1*;


total_loss�=�@

error_R�5K?

learning_rate_1��5y�I       6%�	��+j���A�1*;


total_loss��@

error_R�(O?

learning_rate_1��5�CxI       6%�	�,j���A�1*;


total_loss-��@

error_R)�R?

learning_rate_1��5k�I       6%�	�Y,j���A�1*;


total_loss���@

error_R�QM?

learning_rate_1��5��8sI       6%�	e�,j���A�1*;


total_loss��@

error_R ;W?

learning_rate_1��5����I       6%�	3�,j���A�1*;


total_lossaà@

error_R*L??

learning_rate_1��5;D��I       6%�	�<-j���A�1*;


total_loss�d@

error_R&�D?

learning_rate_1��5��b�I       6%�	��-j���A�1*;


total_loss`��@

error_RQ>=?

learning_rate_1��5���I       6%�	g�-j���A�1*;


total_loss�@

error_R��a?

learning_rate_1��5����I       6%�	&'.j���A�1*;


total_loss`��@

error_R�SO?

learning_rate_1��5��*�I       6%�	�y.j���A�1*;


total_loss�v�@

error_Rq�R?

learning_rate_1��5
o�rI       6%�	X�.j���A�1*;


total_lossDg�@

error_R��T?

learning_rate_1��5���NI       6%�	8	/j���A�1*;


total_loss@q�@

error_R|�N?

learning_rate_1��5zr3I       6%�	�Q/j���A�1*;


total_loss1ʩ@

error_R�S?

learning_rate_1��5����I       6%�	�/j���A�1*;


total_loss�W�@

error_RTM?

learning_rate_1��5�JI       6%�	'�/j���A�1*;


total_lossQ�@

error_R|Xp?

learning_rate_1��5���I       6%�	B0j���A�1*;


total_lossT� A

error_R�EJ?

learning_rate_1��5�W�I       6%�	R^0j���A�1*;


total_loss�"�@

error_R��=?

learning_rate_1��5�t�nI       6%�	<�0j���A�1*;


total_lossN=�@

error_R�I?

learning_rate_1��5��B>I       6%�	��0j���A�1*;


total_loss��@

error_RikS?

learning_rate_1��5RI       6%�	�'1j���A�1*;


total_lossw�j@

error_R�c??

learning_rate_1��5��ǢI       6%�	?j1j���A�1*;


total_lossc|�@

error_R�GL?

learning_rate_1��5���I       6%�	��1j���A�1*;


total_loss]U�@

error_Rs�g?

learning_rate_1��5(��6I       6%�	��1j���A�1*;


total_loss���@

error_RM;N?

learning_rate_1��5���VI       6%�	�22j���A�1*;


total_lossnn�@

error_R8�Q?

learning_rate_1��5#A7�I       6%�	mt2j���A�1*;


total_loss�`�@

error_R��O?

learning_rate_1��5z�lI       6%�	��2j���A�1*;


total_loss�8�@

error_R�V?

learning_rate_1��5r_��I       6%�	��2j���A�1*;


total_lossJ<�@

error_RŁ:?

learning_rate_1��5�-"\I       6%�	7A3j���A�1*;


total_loss��@

error_R]uQ?

learning_rate_1��5��I       6%�	ߋ3j���A�1*;


total_loss�@

error_R�4J?

learning_rate_1��5(�ɥI       6%�	��3j���A�1*;


total_loss���@

error_Rq�P?

learning_rate_1��59`WI       6%�	N4j���A�1*;


total_lossV�@

error_R�=O?

learning_rate_1��5��I       6%�	2b4j���A�1*;


total_loss���@

error_R��D?

learning_rate_1��5U��I       6%�	5�4j���A�1*;


total_lossI �@

error_R��c?

learning_rate_1��5xSm6I       6%�	~�4j���A�1*;


total_losseͬ@

error_R�1?

learning_rate_1��5��m�I       6%�	+05j���A�1*;


total_loss�|�@

error_R��P?

learning_rate_1��5ɜdI       6%�	��5j���A�1*;


total_loss�r�@

error_R�qC?

learning_rate_1��5U�a�I       6%�	�5j���A�1*;


total_loss׫�@

error_R.e8?

learning_rate_1��5	.��I       6%�	�6j���A�1*;


total_loss�wq@

error_R�_N?

learning_rate_1��5ɵK�I       6%�	Il6j���A�1*;


total_lossVS�@

error_RS�W?

learning_rate_1��5��I       6%�	s�6j���A�1*;


total_loss�d�@

error_R�&W?

learning_rate_1��5�C�I       6%�	�7j���A�1*;


total_loss㬉@

error_R*'_?

learning_rate_1��5�l�sI       6%�	Z7j���A�1*;


total_loss��@

error_R�[?

learning_rate_1��5 �/I       6%�	F�7j���A�1*;


total_loss��@

error_R��C?

learning_rate_1��5�(�<I       6%�	M�7j���A�1*;


total_loss`�@

error_Rv�<?

learning_rate_1��5t�$=I       6%�	�(8j���A�1*;


total_loss���@

error_R�eT?

learning_rate_1��5����I       6%�	�l8j���A�1*;


total_loss�n�@

error_R�jO?

learning_rate_1��59�+]I       6%�	��8j���A�1*;


total_loss<�@

error_R2<?

learning_rate_1��5��sI       6%�	��8j���A�1*;


total_loss?��@

error_R�G?

learning_rate_1��5:�bI       6%�	1K9j���A�1*;


total_lossN͊@

error_R�cC?

learning_rate_1��5cT\3I       6%�	ߙ9j���A�1*;


total_lossi�@

error_R��Q?

learning_rate_1��5_�byI       6%�	6�9j���A�1*;


total_loss�B�@

error_R�Lr?

learning_rate_1��5��I       6%�	"#:j���A�1*;


total_loss�8a@

error_R�oO?

learning_rate_1��5�0��I       6%�	�e:j���A�1*;


total_lossi��@

error_R��a?

learning_rate_1��5Y��I       6%�	��:j���A�1*;


total_lossCk	A

error_R�BV?

learning_rate_1��5o�I       6%�	U�:j���A�1*;


total_loss�˭@

error_RZ�J?

learning_rate_1��5��M�I       6%�	�5;j���A�1*;


total_lossQ��@

error_R)A?

learning_rate_1��5.3��I       6%�	X�;j���A�1*;


total_lossonA

error_RZJ\?

learning_rate_1��5��?I       6%�	-�;j���A�1*;


total_loss2]@

error_REcQ?

learning_rate_1��5�JzI       6%�	=<j���A�1*;


total_loss��@

error_RC�\?

learning_rate_1��5�=I       6%�	�U<j���A�1*;


total_loss�)�@

error_Rv&c?

learning_rate_1��5�>�I       6%�	�<j���A�1*;


total_loss��A

error_RQDX?

learning_rate_1��5�Y.5I       6%�	V�<j���A�1*;


total_loss���@

error_R�He?

learning_rate_1��5��I       6%�	�1=j���A�1*;


total_loss���@

error_R��R?

learning_rate_1��5BT�I       6%�	�x=j���A�1*;


total_loss��@

error_R��K?

learning_rate_1��5���pI       6%�	g�=j���A�1*;


total_loss̙�@

error_RO=M?

learning_rate_1��5��m�I       6%�	8>j���A�1*;


total_loss�|�@

error_Ru`?

learning_rate_1��5[���I       6%�	o>j���A�1*;


total_lossa�@

error_RiB?

learning_rate_1��5�E7I       6%�	��>j���A�1*;


total_loss.G�@

error_RR�T?

learning_rate_1��5��I       6%�	u�>j���A�1*;


total_loss���@

error_R�,G?

learning_rate_1��5k�A�I       6%�	=;?j���A�1*;


total_lossHBl@

error_Rn�C?

learning_rate_1��5*}�^I       6%�	�{?j���A�1*;


total_loss��@

error_R�nS?

learning_rate_1��5K0'I       6%�	�?j���A�1*;


total_loss��@

error_R�M?

learning_rate_1��5��PxI       6%�	�@j���A�1*;


total_loss&
�@

error_RF?

learning_rate_1��50�I       6%�	�D@j���A�1*;


total_loss�r�@

error_R��6?

learning_rate_1��5X�pI       6%�	އ@j���A�1*;


total_lossv��@

error_Ri�P?

learning_rate_1��5�;�I       6%�	v�@j���A�1*;


total_loss��@

error_R��B?

learning_rate_1��5��}�I       6%�	lAj���A�1*;


total_loss�@

error_Ri8`?

learning_rate_1��5��GI       6%�	�PAj���A�1*;


total_loss]��@

error_R�^?

learning_rate_1��5\�1�I       6%�	ߔAj���A�1*;


total_loss�e�@

error_RrgD?

learning_rate_1��5}�^QI       6%�	{�Aj���A�1*;


total_loss|^�@

error_R�<?

learning_rate_1��5y���I       6%�	�!Bj���A�1*;


total_loss�3�@

error_R�YQ?

learning_rate_1��5�3�DI       6%�	�cBj���A�1*;


total_loss�a�@

error_R|�G?

learning_rate_1��5���I       6%�	��Bj���A�1*;


total_loss�
�@

error_R�b?

learning_rate_1��5�ϗ�I       6%�	s�Bj���A�1*;


total_loss�~�@

error_R��O?

learning_rate_1��5��I       6%�	�ACj���A�2*;


total_loss	ơ@

error_R��k?

learning_rate_1��5�ڑUI       6%�	ЉCj���A�2*;


total_lossR�@

error_R=i7?

learning_rate_1��5��WI       6%�	��Cj���A�2*;


total_lossr��@

error_R��U?

learning_rate_1��5؂*6I       6%�	�Dj���A�2*;


total_loss���@

error_R�RY?

learning_rate_1��5WG�I       6%�	�VDj���A�2*;


total_loss�!�@

error_Rf�I?

learning_rate_1��5҈UqI       6%�	��Dj���A�2*;


total_loss}�@

error_R��R?

learning_rate_1��5j�AnI       6%�	��Dj���A�2*;


total_loss�b�@

error_R�y^?

learning_rate_1��54��OI       6%�	�(Ej���A�2*;


total_loss�4�@

error_R��C?

learning_rate_1��5��I       6%�	�kEj���A�2*;


total_loss�4�@

error_R�_T?

learning_rate_1��5ycI       6%�	h�Ej���A�2*;


total_lossu�@

error_R�{U?

learning_rate_1��5��1�I       6%�	��Ej���A�2*;


total_lossR�@

error_R)M<?

learning_rate_1��5�;I       6%�	tEFj���A�2*;


total_loss���@

error_Rl=S?

learning_rate_1��57�NI       6%�	a�Fj���A�2*;


total_lossbÕ@

error_R1�P?

learning_rate_1��5b7HI       6%�	<�Fj���A�2*;


total_lossnϟ@

error_RXc?

learning_rate_1��5����I       6%�	�Gj���A�2*;


total_loss6�@

error_R��H?

learning_rate_1��5�9�I       6%�	�]Gj���A�2*;


total_loss��@

error_Rw
]?

learning_rate_1��5t9eyI       6%�	y�Gj���A�2*;


total_loss�V�@

error_R�wJ?

learning_rate_1��5�V�?I       6%�	y�Gj���A�2*;


total_loss���@

error_R�V?

learning_rate_1��5奚WI       6%�	F(Hj���A�2*;


total_lossʢ�@

error_R;�E?

learning_rate_1��5'ߴI       6%�	ooHj���A�2*;


total_loss�
�@

error_R=�P?

learning_rate_1��5_�	�I       6%�	��Hj���A�2*;


total_loss���@

error_R��V?

learning_rate_1��5dQ��I       6%�		�Hj���A�2*;


total_lossԼ�@

error_R�fD?

learning_rate_1��5A4F)I       6%�	�;Ij���A�2*;


total_lossČ�@

error_Rw�??

learning_rate_1��5�-0tI       6%�	Ij���A�2*;


total_loss��@

error_RJHD?

learning_rate_1��5?&"�I       6%�	�Ij���A�2*;


total_loss�_�@

error_R�4??

learning_rate_1��5�)�.I       6%�	�Jj���A�2*;


total_loss{ӟ@

error_R �D?

learning_rate_1��5�g3I       6%�	 VJj���A�2*;


total_lossJ�u@

error_R��A?

learning_rate_1��5��'�I       6%�	ҚJj���A�2*;


total_loss� �@

error_R��Q?

learning_rate_1��5E�'I       6%�	�Jj���A�2*;


total_loss��@

error_R8�;?

learning_rate_1��5�q4I       6%�	6Kj���A�2*;


total_lossa7�@

error_R��K?

learning_rate_1��5�x�I       6%�	`cKj���A�2*;


total_loss ��@

error_R�pX?

learning_rate_1��5��/�I       6%�	W�Kj���A�2*;


total_loss��@

error_R�GB?

learning_rate_1��5��6I       6%�	��Kj���A�2*;


total_loss#ĳ@

error_R�P?

learning_rate_1��5���mI       6%�	%>Lj���A�2*;


total_loss3�@

error_R�I?

learning_rate_1��5����I       6%�	J�Lj���A�2*;


total_loss\*�@

error_R��Z?

learning_rate_1��5�әI       6%�	x�Lj���A�2*;


total_loss���@

error_R�7D?

learning_rate_1��5��O�I       6%�	�Mj���A�2*;


total_loss�:�@

error_R�S?

learning_rate_1��5K� I       6%�	�UMj���A�2*;


total_loss��@

error_RA�X?

learning_rate_1��5��{9I       6%�	6�Mj���A�2*;


total_loss�>�@

error_RM[?

learning_rate_1��5�Q�I       6%�	��Mj���A�2*;


total_loss�|�@

error_R��A?

learning_rate_1��5���I       6%�	�TNj���A�2*;


total_loss���@

error_R�gE?

learning_rate_1��5�i�QI       6%�	��Nj���A�2*;


total_loss�-�@

error_R�rP?

learning_rate_1��5����I       6%�	��Nj���A�2*;


total_loss��@

error_R�Z??

learning_rate_1��5�/³I       6%�	$Oj���A�2*;


total_loss	j�@

error_Rv�7?

learning_rate_1��5�aa�I       6%�	�iOj���A�2*;


total_lossd	�@

error_R��X?

learning_rate_1��5Wzk�I       6%�	ϭOj���A�2*;


total_loss�[�@

error_ROV?

learning_rate_1��5�w;@I       6%�	��Oj���A�2*;


total_loss}�@

error_R@R?

learning_rate_1��5�@b�I       6%�	�9Pj���A�2*;


total_loss���@

error_RO�^?

learning_rate_1��5NЉ�I       6%�	ހPj���A�2*;


total_loss�!�@

error_R@8T?

learning_rate_1��5��&I       6%�	>�Pj���A�2*;


total_lossC�@

error_RY?

learning_rate_1��5&��I       6%�	�
Qj���A�2*;


total_loss�E�@

error_R��>?

learning_rate_1��5]�}�I       6%�	�PQj���A�2*;


total_loss�-A

error_R�+*?

learning_rate_1��5�u6�I       6%�	��Qj���A�2*;


total_loss���@

error_R�zL?

learning_rate_1��5O�u<I       6%�	I�Qj���A�2*;


total_loss/ĥ@

error_RIHS?

learning_rate_1��52�;PI       6%�	�0Rj���A�2*;


total_lossӆ�@

error_R`�A?

learning_rate_1��5����I       6%�	�|Rj���A�2*;


total_loss�ý@

error_R�`??

learning_rate_1��5��J�I       6%�	��Rj���A�2*;


total_lossm��@

error_R�L?

learning_rate_1��5\��I       6%�	CSj���A�2*;


total_loss$��@

error_R��6?

learning_rate_1��5ǂ%QI       6%�	�[Sj���A�2*;


total_loss<�@

error_R�C?

learning_rate_1��5�K,@I       6%�	W�Sj���A�2*;


total_lossЈ@

error_RN-I?

learning_rate_1��5��(I       6%�	��Sj���A�2*;


total_losss��@

error_RvB?

learning_rate_1��5�Nm�I       6%�	�)Tj���A�2*;


total_loss��@

error_RȲM?

learning_rate_1��5���I       6%�	mTj���A�2*;


total_lossI-�@

error_R�Z?

learning_rate_1��5��[I       6%�	�Tj���A�2*;


total_loss-p�@

error_R�"R?

learning_rate_1��5ex^�I       6%�	��Tj���A�2*;


total_loss�q�@

error_R�A?

learning_rate_1��5��ZQI       6%�	D;Uj���A�2*;


total_loss#��@

error_R��S?

learning_rate_1��5��DI       6%�	�Uj���A�2*;


total_loss��@

error_R�W?

learning_rate_1��5�?N�I       6%�	:�Uj���A�2*;


total_loss��@

error_R6GF?

learning_rate_1��5����I       6%�	$Vj���A�2*;


total_loss���@

error_RۆD?

learning_rate_1��5�ޭnI       6%�	_kVj���A�2*;


total_loss ��@

error_R@hF?

learning_rate_1��5�&iJI       6%�	R�Vj���A�2*;


total_lossf�@

error_R�;?

learning_rate_1��5&��aI       6%�	�Wj���A�2*;


total_loss�v�@

error_RM�B?

learning_rate_1��5�RI       6%�	�LWj���A�2*;


total_loss�ږ@

error_R��S?

learning_rate_1��5���I       6%�	�Wj���A�2*;


total_loss&��@

error_Rq�U?

learning_rate_1��5ҋ�I       6%�	*�Wj���A�2*;


total_loss�&�@

error_R�P?

learning_rate_1��5t�
pI       6%�	WXj���A�2*;


total_lossOc@

error_R.:E?

learning_rate_1��5�,	�I       6%�	@[Xj���A�2*;


total_loss��@

error_R�G>?

learning_rate_1��5(��I       6%�	��Xj���A�2*;


total_loss�f�@

error_R�1D?

learning_rate_1��52�LI       6%�	p�Xj���A�2*;


total_loss���@

error_RUY?

learning_rate_1��5��I       6%�	'Yj���A�2*;


total_lossL��@

error_R�_A?

learning_rate_1��5��lI       6%�	�hYj���A�2*;


total_loss���@

error_R:%S?

learning_rate_1��5��EI       6%�	ҮYj���A�2*;


total_loss��@

error_R�!S?

learning_rate_1��5��@I       6%�	n�Yj���A�2*;


total_loss�@

error_R,/<?

learning_rate_1��5�f	I       6%�	?Zj���A�2*;


total_loss�¬@

error_R8_?

learning_rate_1��5��1�I       6%�	%�Zj���A�2*;


total_loss�9�@

error_R��9?

learning_rate_1��5��o�I       6%�	i�Zj���A�2*;


total_loss��@

error_R�<F?

learning_rate_1��5����I       6%�	�[j���A�2*;


total_loss�p�@

error_R-QG?

learning_rate_1��5�9�I       6%�	�P[j���A�2*;


total_loss|�A

error_R� *?

learning_rate_1��5���ZI       6%�	��[j���A�2*;


total_loss�A

error_R��R?

learning_rate_1��5S^Q�I       6%�	*�[j���A�2*;


total_loss[	�@

error_Rs�E?

learning_rate_1��5׆I       6%�	�\j���A�2*;


total_loss��v@

error_R�
]?

learning_rate_1��5�JI       6%�	�`\j���A�2*;


total_loss�>�@

error_R�;A?

learning_rate_1��5�0�5I       6%�	£\j���A�2*;


total_loss���@

error_R��M?

learning_rate_1��5zךtI       6%�	V�\j���A�2*;


total_lossZ>�@

error_R��J?

learning_rate_1��5��dmI       6%�	.]j���A�2*;


total_loss�A�@

error_R��d?

learning_rate_1��5�x{I       6%�	+v]j���A�2*;


total_loss]�@

error_RxjL?

learning_rate_1��5+���I       6%�	{�]j���A�2*;


total_loss-3�@

error_REGH?

learning_rate_1��5�~fI       6%�	^j���A�2*;


total_loss{��@

error_R�~R?

learning_rate_1��5gƮ,I       6%�	&r^j���A�2*;


total_loss���@

error_R�J?

learning_rate_1��50��I       6%�	��^j���A�2*;


total_loss��@

error_R�RY?

learning_rate_1��5v&�I       6%�	_j���A�2*;


total_loss���@

error_RߎF?

learning_rate_1��5�$�I       6%�	0I_j���A�2*;


total_loss�e�@

error_R
.G?

learning_rate_1��5����I       6%�	\�_j���A�2*;


total_loss�ܠ@

error_RhFB?

learning_rate_1��5���I       6%�	T�_j���A�2*;


total_lossR'�@

error_RLJ?

learning_rate_1��5֦�I       6%�	b`j���A�2*;


total_lossJ��@

error_Rl�L?

learning_rate_1��5H���I       6%�	4O`j���A�2*;


total_loss㼙@

error_R�J?

learning_rate_1��5�Y�$I       6%�	U�`j���A�2*;


total_loss�-�@

error_R3Q?

learning_rate_1��5�6 I       6%�	��`j���A�2*;


total_loss��@

error_RD�H?

learning_rate_1��5�~-I       6%�	�aj���A�2*;


total_loss~;�@

error_RW�=?

learning_rate_1��5�Δ�I       6%�	�\aj���A�2*;


total_loss�j�@

error_R�p_?

learning_rate_1��5Nm�I       6%�	8�aj���A�2*;


total_lossH��@

error_R�A?

learning_rate_1��5��I       6%�	��aj���A�2*;


total_loss.��@

error_Rq�Z?

learning_rate_1��5,��I       6%�	`+bj���A�2*;


total_loss&��@

error_R��[?

learning_rate_1��5��uI       6%�	Snbj���A�2*;


total_loss�&�@

error_RqW?

learning_rate_1��5L[`�I       6%�	`�bj���A�2*;


total_lossI�0A

error_R��H?

learning_rate_1��5��-I       6%�	�bj���A�2*;


total_lossa�@

error_R�6\?

learning_rate_1��5�P�pI       6%�	�?cj���A�2*;


total_lossE�@

error_R��Z?

learning_rate_1��5���AI       6%�	҃cj���A�2*;


total_loss��A

error_R�qD?

learning_rate_1��5���6I       6%�	��cj���A�2*;


total_lossӥ@

error_R��N?

learning_rate_1��5�e|I       6%�	dj���A�2*;


total_lossC��@

error_R$S?

learning_rate_1��5]*�bI       6%�	�Qdj���A�2*;


total_loss��@

error_R gP?

learning_rate_1��5�y�"I       6%�	��dj���A�2*;


total_loss��r@

error_R�`?

learning_rate_1��5��LI       6%�	��dj���A�2*;


total_loss��@

error_R��=?

learning_rate_1��5�2�I       6%�	$ej���A�2*;


total_lossA( A

error_R�E?

learning_rate_1��5%
�dI       6%�	7hej���A�2*;


total_loss���@

error_R��@?

learning_rate_1��5F��I       6%�	?�ej���A�2*;


total_lossLϤ@

error_R�2?

learning_rate_1��5�>b�I       6%�	��ej���A�2*;


total_loss/=�@

error_R�;?

learning_rate_1��5�k}I       6%�	Y;fj���A�2*;


total_loss��A

error_R��J?

learning_rate_1��5��4fI       6%�	3�fj���A�3*;


total_loss���@

error_R�W?

learning_rate_1��5�m�I       6%�	4�fj���A�3*;


total_loss���@

error_RH
D?

learning_rate_1��5�&�I       6%�	&gj���A�3*;


total_loss�d�@

error_RM`T?

learning_rate_1��5I�dI       6%�	Tagj���A�3*;


total_loss�Ϩ@

error_R��L?

learning_rate_1��5)�>I       6%�	�gj���A�3*;


total_lossO�@

error_R�B?

learning_rate_1��5�N�I       6%�	��gj���A�3*;


total_loss��A

error_R��P?

learning_rate_1��5ˈ��I       6%�	3hj���A�3*;


total_loss� �@

error_R��<?

learning_rate_1��5!D^�I       6%�	Bthj���A�3*;


total_loss;�@

error_R�[G?

learning_rate_1��5�)}	I       6%�	v�hj���A�3*;


total_lossRe�@

error_R�A?

learning_rate_1��5f��I       6%�	V�hj���A�3*;


total_loss���@

error_R�Q?

learning_rate_1��5wZ<I       6%�	7=ij���A�3*;


total_lossdݯ@

error_R��A?

learning_rate_1��5L�<�I       6%�	��ij���A�3*;


total_loss�w�@

error_R��d?

learning_rate_1��5����I       6%�	��ij���A�3*;


total_loss�=�@

error_R-�N?

learning_rate_1��5Y��GI       6%�	=jj���A�3*;


total_loss���@

error_R#m??

learning_rate_1��5tQz�I       6%�	<cjj���A�3*;


total_lossŎ�@

error_RbI?

learning_rate_1��5��@qI       6%�	3�jj���A�3*;


total_loss��@

error_R
�B?

learning_rate_1��5E�ucI       6%�	 �jj���A�3*;


total_lossɚ�@

error_R�@L?

learning_rate_1��5���_I       6%�	�+kj���A�3*;


total_loss`.�@

error_R�H?

learning_rate_1��5���`I       6%�	�pkj���A�3*;


total_loss=��@

error_RוT?

learning_rate_1��5\tI       6%�	�kj���A�3*;


total_loss��@

error_R�R?

learning_rate_1��5���I       6%�	��kj���A�3*;


total_loss�w!A

error_R��K?

learning_rate_1��5@9�I       6%�	I:lj���A�3*;


total_lossh��@

error_R�_?

learning_rate_1��5���I       6%�	��lj���A�3*;


total_lossW��@

error_R
�B?

learning_rate_1��5Yd�I       6%�	H�lj���A�3*;


total_losszԫ@

error_R�]?

learning_rate_1��5|��=I       6%�	Z
mj���A�3*;


total_loss�7�@

error_R�4N?

learning_rate_1��5��3iI       6%�	�Mmj���A�3*;


total_loss��@

error_R?2j?

learning_rate_1��5!ܨI       6%�	r�mj���A�3*;


total_loss��@

error_R}�[?

learning_rate_1��5���cI       6%�	l�mj���A�3*;


total_loss8��@

error_R��S?

learning_rate_1��51">$I       6%�	9nj���A�3*;


total_lossN��@

error_R�B?

learning_rate_1��5���|I       6%�	��nj���A�3*;


total_loss�g A

error_R�,N?

learning_rate_1��5�Ʌ�I       6%�	��nj���A�3*;


total_lossf��@

error_RM8D?

learning_rate_1��54��I       6%�	Aoj���A�3*;


total_lossXxA

error_Rm0V?

learning_rate_1��5��*I       6%�	8Zoj���A�3*;


total_loss���@

error_Rt�H?

learning_rate_1��5���I       6%�	��oj���A�3*;


total_loss��r@

error_R/�[?

learning_rate_1��5�̸I       6%�	��oj���A�3*;


total_lossV9�@

error_R�a?

learning_rate_1��5M~��I       6%�	p'pj���A�3*;


total_lossDJ�@

error_R��Q?

learning_rate_1��5��$XI       6%�	��pj���A�3*;


total_loss@0�@

error_R�1??

learning_rate_1��5�MCI       6%�	��pj���A�3*;


total_loss��@

error_R�;H?

learning_rate_1��5�t|�I       6%�	�)qj���A�3*;


total_lossO��@

error_R�ma?

learning_rate_1��5f|kI       6%�	�rqj���A�3*;


total_loss��@

error_R�tD?

learning_rate_1��5��^I       6%�	�qj���A�3*;


total_loss;��@

error_R��Q?

learning_rate_1��5�8�VI       6%�	rj���A�3*;


total_loss��A

error_R�C?

learning_rate_1��5�B��I       6%�	[rj���A�3*;


total_loss4W�@

error_RBM?

learning_rate_1��5G��I       6%�	��rj���A�3*;


total_loss���@

error_RqV?

learning_rate_1��5��AwI       6%�	�rj���A�3*;


total_loss}2�@

error_Rȅ.?

learning_rate_1��5&�[ I       6%�	>!sj���A�3*;


total_loss��l@

error_R_Z?

learning_rate_1��5��A�I       6%�	�dsj���A�3*;


total_loss��@

error_Rl�9?

learning_rate_1��5�ᬖI       6%�	��sj���A�3*;


total_lossh�@

error_RJ�>?

learning_rate_1��5��oI       6%�	=�sj���A�3*;


total_loss@�@

error_R��N?

learning_rate_1��5E��I       6%�	L4tj���A�3*;


total_lossd��@

error_RZ�B?

learning_rate_1��5I�iKI       6%�	̀tj���A�3*;


total_loss�@

error_R7M?

learning_rate_1��5�dvI       6%�	l�tj���A�3*;


total_loss��}@

error_RبL?

learning_rate_1��5"�w�I       6%�	uj���A�3*;


total_loss�9�@

error_R�*[?

learning_rate_1��5���@I       6%�	$_uj���A�3*;


total_loss�A

error_R�{K?

learning_rate_1��5��~�I       6%�	i�uj���A�3*;


total_loss�H�@

error_R�T?

learning_rate_1��5�j]�I       6%�	*�uj���A�3*;


total_lossn�}@

error_R;5;?

learning_rate_1��5n���I       6%�	�Nvj���A�3*;


total_losso9@

error_RngO?

learning_rate_1��5Us�I       6%�	��vj���A�3*;


total_lossf��@

error_RO!O?

learning_rate_1��55:�I       6%�	k�vj���A�3*;


total_loss:e�@

error_R}�W?

learning_rate_1��5�bI�I       6%�	�%wj���A�3*;


total_loss�$�@

error_RC?

learning_rate_1��5�;�	I       6%�	frwj���A�3*;


total_loss�q�@

error_RR?

learning_rate_1��5�ڬ�I       6%�	=�wj���A�3*;


total_lossڬl@

error_Rd�Y?

learning_rate_1��5�ZЖI       6%�	��wj���A�3*;


total_loss��@

error_R�B?

learning_rate_1��5��BI       6%�	OBxj���A�3*;


total_lossd��@

error_R�a?

learning_rate_1��5��<�I       6%�	��xj���A�3*;


total_lossCr@

error_R&/\?

learning_rate_1��5�[��I       6%�	T�xj���A�3*;


total_loss/E�@

error_R�Q?

learning_rate_1��5��L�I       6%�	�yj���A�3*;


total_lossA�@

error_R��]?

learning_rate_1��5_���I       6%�	�Xyj���A�3*;


total_lossI�@

error_R
�U?

learning_rate_1��54��I       6%�	��yj���A�3*;


total_loss{��@

error_R�/M?

learning_rate_1��5��(�I       6%�	�7zj���A�3*;


total_loss�@

error_RCQ_?

learning_rate_1��5V�?1I       6%�	�}zj���A�3*;


total_lossz]�@

error_R�^A?

learning_rate_1��5��K<I       6%�	��zj���A�3*;


total_loss��@

error_R_�f?

learning_rate_1��5*�iI       6%�	�{j���A�3*;


total_loss��@

error_R�[B?

learning_rate_1��57��I       6%�	�]{j���A�3*;


total_loss� �@

error_Ri�R?

learning_rate_1��5+i�I       6%�	n�{j���A�3*;


total_loss�q@

error_R��`?

learning_rate_1��5!�I       6%�	K�{j���A�3*;


total_loss�]�@

error_R�^H?

learning_rate_1��5
�HJI       6%�	"<|j���A�3*;


total_lossA

error_R��P?

learning_rate_1��5���I       6%�	��|j���A�3*;


total_loss�
�@

error_R�F?

learning_rate_1��5g_1�I       6%�	w�|j���A�3*;


total_lossP�@

error_R�I?

learning_rate_1��5���I       6%�	�}j���A�3*;


total_loss(�@

error_RmI?

learning_rate_1��5}Z��I       6%�	W}j���A�3*;


total_loss|��@

error_R 6[?

learning_rate_1��5VC��I       6%�	��}j���A�3*;


total_lossiA�@

error_RiIK?

learning_rate_1��5��p~I       6%�	c�}j���A�3*;


total_lossg�@

error_R�lB?

learning_rate_1��5�d�I       6%�	/�~j���A�3*;


total_loss	�@

error_R��L?

learning_rate_1��5�C�I       6%�	9%j���A�3*;


total_loss��@

error_R�LN?

learning_rate_1��5��I       6%�	�zj���A�3*;


total_loss-`A

error_RI?

learning_rate_1��5~���I       6%�	'�j���A�3*;


total_lossj��@

error_R��Z?

learning_rate_1��5�W*PI       6%�	U �j���A�3*;


total_loss}�@

error_R�4O?

learning_rate_1��5�ήI       6%�	�q�j���A�3*;


total_loss�J�@

error_R;J?

learning_rate_1��5���xI       6%�	W��j���A�3*;


total_loss�L�@

error_R
&[?

learning_rate_1��5	5�I       6%�	U��j���A�3*;


total_loss�|�@

error_R�~o?

learning_rate_1��5C�4vI       6%�	B�j���A�3*;


total_lossㅯ@

error_R8�A?

learning_rate_1��5�&VGI       6%�	���j���A�3*;


total_loss�@

error_RH?

learning_rate_1��57�.NI       6%�	�ˁj���A�3*;


total_loss���@

error_R�~W?

learning_rate_1��5��BI       6%�	`�j���A�3*;


total_loss��@

error_R��R?

learning_rate_1��5��ZnI       6%�	U�j���A�3*;


total_loss�	�@

error_R�Y]?

learning_rate_1��5�HG�I       6%�	T��j���A�3*;


total_losst�@

error_RW?

learning_rate_1��5���I       6%�	�ނj���A�3*;


total_lossN��@

error_R$@H?

learning_rate_1��5E]�
I       6%�	!*�j���A�3*;


total_lossL��@

error_R1)[?

learning_rate_1��52$Z�I       6%�	Ur�j���A�3*;


total_loss�r�@

error_R�o=?

learning_rate_1��5�p�I       6%�	�ăj���A�3*;


total_lossE�@

error_RZ�C?

learning_rate_1��5	��I       6%�	(%�j���A�3*;


total_loss��@

error_R yI?

learning_rate_1��5�<I�I       6%�	F~�j���A�3*;


total_loss�W�@

error_R�5V?

learning_rate_1��5%�"�I       6%�	��j���A�3*;


total_lossZ֊@

error_R��M?

learning_rate_1��5W'I       6%�	�S�j���A�3*;


total_loss��@

error_R�xF?

learning_rate_1��5L�$I       6%�	���j���A�3*;


total_loss�#�@

error_R��B?

learning_rate_1��5XI*�I       6%�	��j���A�3*;


total_loss���@

error_R�IL?

learning_rate_1��5J���I       6%�	#b�j���A�3*;


total_loss?��@

error_R��C?

learning_rate_1��5P}�CI       6%�	��j���A�3*;


total_loss?g�@

error_R�9?

learning_rate_1��5�L��I       6%�	�
�j���A�3*;


total_loss���@

error_R{�i?

learning_rate_1��5�l��I       6%�	�q�j���A�3*;


total_loss�h�@

error_R�IK?

learning_rate_1��5m��I       6%�	#�j���A�3*;


total_lossS��@

error_R��4?

learning_rate_1��5|*��I       6%�	�4�j���A�3*;


total_lossT]�@

error_R�L?

learning_rate_1��5>���I       6%�	8��j���A�3*;


total_lossI��@

error_RGJ?

learning_rate_1��5��HI       6%�	�׈j���A�3*;


total_loss4:�@

error_REQM?

learning_rate_1��5�ٿ�I       6%�	�7�j���A�3*;


total_losse=�@

error_R�H?

learning_rate_1��5��$I       6%�	��j���A�3*;


total_loss��A

error_R߸X?

learning_rate_1��5(��I       6%�	�+�j���A�3*;


total_loss���@

error_R��L?

learning_rate_1��5BgټI       6%�	@��j���A�3*;


total_loss�#�@

error_R*P?

learning_rate_1��5��5I       6%�	��j���A�3*;


total_loss��@

error_R��??

learning_rate_1��5S��I       6%�	U�j���A�3*;


total_lossQ�@

error_R�5K?

learning_rate_1��5�EeI       6%�	���j���A�3*;


total_loss�_�@

error_R,H?

learning_rate_1��5��<I       6%�	�j���A�3*;


total_loss��A

error_R�>W?

learning_rate_1��5s�I       6%�	�S�j���A�3*;


total_loss}��@

error_RVdM?

learning_rate_1��5��x�I       6%�	b��j���A�3*;


total_lossz�@

error_R$�L?

learning_rate_1��5���gI       6%�	���j���A�3*;


total_loss��@

error_R��C?

learning_rate_1��5�l��I       6%�	�C�j���A�3*;


total_loss�y@

error_R�~J?

learning_rate_1��5 r�KI       6%�	���j���A�3*;


total_loss�C�@

error_R�1F?

learning_rate_1��5(!k�I       6%�	@؍j���A�4*;


total_loss���@

error_RF�T?

learning_rate_1��5� �I       6%�	�^�j���A�4*;


total_lossM��@

error_RԹQ?

learning_rate_1��5��^&I       6%�	���j���A�4*;


total_loss��@

error_R�V?

learning_rate_1��5�&nI       6%�	7�j���A�4*;


total_loss�@�@

error_RdV?

learning_rate_1��5��>�I       6%�	�?�j���A�4*;


total_loss�U�@

error_RdDL?

learning_rate_1��5Y?��I       6%�	���j���A�4*;


total_loss��@

error_R��`?

learning_rate_1��5��$�I       6%�	��j���A�4*;


total_loss���@

error_R3bX?

learning_rate_1��5���CI       6%�	l9�j���A�4*;


total_loss��@

error_R�C?

learning_rate_1��5`��WI       6%�	�~�j���A�4*;


total_loss��@

error_R�R?

learning_rate_1��5\VӛI       6%�	�Ðj���A�4*;


total_lossVȆ@

error_R�.]?

learning_rate_1��5f��I       6%�	��j���A�4*;


total_loss A

error_R�P?

learning_rate_1��5-�V�I       6%�	�K�j���A�4*;


total_lossXG�@

error_R_[?

learning_rate_1��5��xI       6%�	+��j���A�4*;


total_loss�y�@

error_R�J?

learning_rate_1��5OL�I       6%�	 ��j���A�4*;


total_lossE��@

error_R��I?

learning_rate_1��5L��I       6%�	�@�j���A�4*;


total_loss$ �@

error_R VL?

learning_rate_1��5�Q��I       6%�	���j���A�4*;


total_lossC�@

error_R��Y?

learning_rate_1��5�#lpI       6%�	�̒j���A�4*;


total_loss��@

error_RL�O?

learning_rate_1��5�W7�I       6%�	��j���A�4*;


total_lossq��@

error_R�BH?

learning_rate_1��5޾�sI       6%�	�T�j���A�4*;


total_loss��@

error_R�j9?

learning_rate_1��5�:I       6%�	���j���A�4*;


total_lossF��@

error_R?�H?

learning_rate_1��5�5I       6%�	Rߓj���A�4*;


total_loss��@

error_R��K?

learning_rate_1��5��.�I       6%�	!�j���A�4*;


total_loss���@

error_R͓i?

learning_rate_1��5-6��I       6%�	�c�j���A�4*;


total_losst��@

error_RE�^?

learning_rate_1��5]���I       6%�	��j���A�4*;


total_loss�
�@

error_R��E?

learning_rate_1��5�	�3I       6%�	��j���A�4*;


total_loss���@

error_R$�O?

learning_rate_1��5�v#I       6%�	�.�j���A�4*;


total_loss�X�@

error_R�E?

learning_rate_1��5�A*�I       6%�	u�j���A�4*;


total_loss��@

error_R�I?

learning_rate_1��5���uI       6%�	2��j���A�4*;


total_loss��@

error_RĐP?

learning_rate_1��5A�I       6%�	D��j���A�4*;


total_lossS_�@

error_R�!0?

learning_rate_1��5l�pOI       6%�	@�j���A�4*;


total_loss&�@

error_RW�D?

learning_rate_1��5�9�I       6%�	B��j���A�4*;


total_loss	B�@

error_R�Z?

learning_rate_1��5?��I       6%�	�Ɩj���A�4*;


total_loss�k�@

error_RH?

learning_rate_1��5CQ�UI       6%�	�
�j���A�4*;


total_loss�t�@

error_RHP?

learning_rate_1��5o�"�I       6%�	qN�j���A�4*;


total_loss_�@

error_RX�B?

learning_rate_1��5��I       6%�	ސ�j���A�4*;


total_loss7�A

error_R8�8?

learning_rate_1��5i�EI       6%�	�ӗj���A�4*;


total_loss��@

error_Ra�L?

learning_rate_1��5,�b`I       6%�	� �j���A�4*;


total_loss���@

error_RaB?

learning_rate_1��5��s?I       6%�	�h�j���A�4*;


total_loss�ȱ@

error_RfL?

learning_rate_1��5�C�I       6%�	���j���A�4*;


total_loss�L�@

error_R �B?

learning_rate_1��5���VI       6%�	(��j���A�4*;


total_lossc0�@

error_R8�Z?

learning_rate_1��5tu�I       6%�	wC�j���A�4*;


total_loss�F�@

error_R2�Z?

learning_rate_1��5�FXvI       6%�	���j���A�4*;


total_lossf��@

error_R�P?

learning_rate_1��5qF��I       6%�	�ԙj���A�4*;


total_loss�d�@

error_R�}P?

learning_rate_1��5�}I       6%�	��j���A�4*;


total_loss�ߕ@

error_R	�D?

learning_rate_1��5�moI       6%�	�b�j���A�4*;


total_loss��@

error_R��C?

learning_rate_1��5�,�6I       6%�	���j���A�4*;


total_loss�@�@

error_R��:?

learning_rate_1��5т�I       6%�	�j���A�4*;


total_loss���@

error_R��Y?

learning_rate_1��5��vsI       6%�	5�j���A�4*;


total_loss�N�@

error_R;\@?

learning_rate_1��5P1jI       6%�	�{�j���A�4*;


total_loss`-�@

error_R�P?

learning_rate_1��5�y�WI       6%�	��j���A�4*;


total_loss�;�@

error_R�>P?

learning_rate_1��5
F�I       6%�	�
�j���A�4*;


total_loss-�&A

error_RY?

learning_rate_1��5y��I       6%�	CY�j���A�4*;


total_loss��@

error_Rm�C?

learning_rate_1��5�)I       6%�	ɩ�j���A�4*;


total_losse��@

error_R�~J?

learning_rate_1��5US. I       6%�	s�j���A�4*;


total_loss��@

error_R�<C?

learning_rate_1��5���wI       6%�	�K�j���A�4*;


total_loss�zA

error_RZ�H?

learning_rate_1��5�	�I       6%�	퍝j���A�4*;


total_loss �@

error_R7>J?

learning_rate_1��56�vDI       6%�	��j���A�4*;


total_lossC�A

error_R��J?

learning_rate_1��5�/>I       6%�	"I�j���A�4*;


total_loss�T�@

error_RsZ?

learning_rate_1��5(	 �I       6%�	���j���A�4*;


total_loss��@

error_Rn�L?

learning_rate_1��5�>�&I       6%�	�ٞj���A�4*;


total_lossXr�@

error_Rr�E?

learning_rate_1��5�G�mI       6%�	��j���A�4*;


total_loss��@

error_R�(P?

learning_rate_1��5����I       6%�	
`�j���A�4*;


total_loss�,�@

error_R�}H?

learning_rate_1��5�1vI       6%�	)��j���A�4*;


total_lossD2�@

error_R��A?

learning_rate_1��5n�)I       6%�	T�j���A�4*;


total_lossh�@

error_ROT?

learning_rate_1��5�=�7I       6%�	m+�j���A�4*;


total_loss �e@

error_R$JR?

learning_rate_1��5��DI       6%�	Oo�j���A�4*;


total_loss�@

error_R�Q?

learning_rate_1��5<x�I       6%�	���j���A�4*;


total_lossP�@

error_R`X?

learning_rate_1��5)MSI       6%�	K��j���A�4*;


total_loss��@

error_R �`?

learning_rate_1��5��I       6%�	�:�j���A�4*;


total_lossg)A

error_RZ=W?

learning_rate_1��5�<�6I       6%�	��j���A�4*;


total_lossZ�@

error_R��N?

learning_rate_1��5+��I       6%�	�¡j���A�4*;


total_lossr�@

error_R�}S?

learning_rate_1��5'���I       6%�	��j���A�4*;


total_loss9�@

error_R��R?

learning_rate_1��5�t�I       6%�	�G�j���A�4*;


total_loss;+�@

error_R��G?

learning_rate_1��5��<~I       6%�	k��j���A�4*;


total_loss��@

error_RDjM?

learning_rate_1��5���I       6%�	�Тj���A�4*;


total_lossy�@

error_R��A?

learning_rate_1��5Gϧ_I       6%�	G*�j���A�4*;


total_lossZ2�@

error_RȗN?

learning_rate_1��5�g�I       6%�	\s�j���A�4*;


total_loss<��@

error_R�[L?

learning_rate_1��5|9��I       6%�	(��j���A�4*;


total_loss�ά@

error_R�)T?

learning_rate_1��5��LWI       6%�	���j���A�4*;


total_loss��A

error_R��[?

learning_rate_1��5��fI       6%�	$@�j���A�4*;


total_loss��@

error_R;�W?

learning_rate_1��5'g�cI       6%�	r��j���A�4*;


total_loss��@

error_R�"T?

learning_rate_1��5�҃I       6%�	aɤj���A�4*;


total_loss��@

error_R�KQ?

learning_rate_1��5�;LrI       6%�	��j���A�4*;


total_loss�l@

error_RR2Q?

learning_rate_1��5����I       6%�	�N�j���A�4*;


total_loss��@

error_R�B?

learning_rate_1��5����I       6%�	;��j���A�4*;


total_loss�v�@

error_R��T?

learning_rate_1��5��9�I       6%�	�ԥj���A�4*;


total_lossA��@

error_R�O?

learning_rate_1��5�
�I       6%�	=�j���A�4*;


total_loss\z�@

error_R7�I?

learning_rate_1��5jT�I       6%�	2[�j���A�4*;


total_loss���@

error_R�/O?

learning_rate_1��5���I       6%�	��j���A�4*;


total_loss%�@

error_R�]Q?

learning_rate_1��5Ԁ��I       6%�	�ݦj���A�4*;


total_loss(�@

error_R��Y?

learning_rate_1��5�~N�I       6%�	��j���A�4*;


total_loss��@

error_R�B?

learning_rate_1��5HOS�I       6%�	mc�j���A�4*;


total_loss`p�@

error_RݸP?

learning_rate_1��5 �w5I       6%�	���j���A�4*;


total_loss�U�@

error_R�fM?

learning_rate_1��5���I       6%�	F�j���A�4*;


total_loss3+�@

error_R��f?

learning_rate_1��5��I       6%�	0�j���A�4*;


total_loss��@

error_Rf2??

learning_rate_1��5Jq�I       6%�	Eq�j���A�4*;


total_loss�O�@

error_R�UF?

learning_rate_1��5h��"I       6%�	볨j���A�4*;


total_lossM��@

error_R��J?

learning_rate_1��5_βWI       6%�	��j���A�4*;


total_loss�?�@

error_R�mI?

learning_rate_1��5
��~I       6%�	�:�j���A�4*;


total_loss|-�@

error_R8�N?

learning_rate_1��5�){I       6%�	+|�j���A�4*;


total_loss�$�@

error_RZF?

learning_rate_1��5@�3�I       6%�	c��j���A�4*;


total_loss�/�@

error_R�1?

learning_rate_1��5�uI       6%�	� �j���A�4*;


total_loss��X@

error_R��S?

learning_rate_1��5�\LI       6%�	.E�j���A�4*;


total_loss-b�@

error_Re�E?

learning_rate_1��5�&�7I       6%�	%��j���A�4*;


total_lossV��@

error_R��U?

learning_rate_1��5�L,�I       6%�	1ʪj���A�4*;


total_loss l�@

error_R�[?

learning_rate_1��5�Q�I       6%�	5�j���A�4*;


total_loss�Q�@

error_RO�??

learning_rate_1��5�LMI       6%�	_M�j���A�4*;


total_losse� A

error_R,X?

learning_rate_1��5\I       6%�	u��j���A�4*;


total_loss�ԗ@

error_R!�O?

learning_rate_1��5�$~I       6%�	�ѫj���A�4*;


total_loss��@

error_R�W?

learning_rate_1��5SV"�I       6%�	v�j���A�4*;


total_loss,=�@

error_R!|I?

learning_rate_1��5BЏUI       6%�	�[�j���A�4*;


total_loss
��@

error_R�U?

learning_rate_1��5��I       6%�	���j���A�4*;


total_lossZ-�@

error_R�mF?

learning_rate_1��56��I       6%�	�j���A�4*;


total_loss/��@

error_RMB?

learning_rate_1��5�hBI       6%�	m'�j���A�4*;


total_loss�O�@

error_R#bQ?

learning_rate_1��5��I       6%�	_i�j���A�4*;


total_lossn�	A

error_R$R?

learning_rate_1��5�6��I       6%�	8��j���A�4*;


total_losss՛@

error_REY?

learning_rate_1��5�!KI       6%�	��j���A�4*;


total_loss@ͮ@

error_RFuM?

learning_rate_1��5}:u�I       6%�	�M�j���A�4*;


total_loss���@

error_R�EE?

learning_rate_1��5�(��I       6%�	���j���A�4*;


total_loss���@

error_R�]?

learning_rate_1��57�OI       6%�	tԮj���A�4*;


total_loss�٘@

error_R^H?

learning_rate_1��5�J�vI       6%�	��j���A�4*;


total_loss��A

error_R�V?

learning_rate_1��5w��I       6%�	Z]�j���A�4*;


total_loss�'A

error_R8�N?

learning_rate_1��5��97I       6%�	���j���A�4*;


total_loss���@

error_Ro�U?

learning_rate_1��5���I       6%�	��j���A�4*;


total_loss�\�@

error_R��M?

learning_rate_1��5�}nI       6%�	�&�j���A�4*;


total_loss���@

error_R�F?

learning_rate_1��5�ZN�I       6%�	k�j���A�4*;


total_loss�DA

error_R�H?

learning_rate_1��5h�s�I       6%�	���j���A�4*;


total_lossZi@

error_R�fi?

learning_rate_1��5�^��I       6%�	8��j���A�4*;


total_loss��@

error_R��C?

learning_rate_1��5�
?�I       6%�	o/�j���A�5*;


total_lossᶿ@

error_R3iT?

learning_rate_1��57��eI       6%�	�q�j���A�5*;


total_lossh�@

error_R�\?

learning_rate_1��5<��I       6%�	ֳ�j���A�5*;


total_loss��@

error_R�kM?

learning_rate_1��5l&�I       6%�	��j���A�5*;


total_loss��@

error_R��K?

learning_rate_1��5�j�#I       6%�	�8�j���A�5*;


total_loss`�@

error_R��>?

learning_rate_1��5��.cI       6%�	|�j���A�5*;


total_loss�f�@

error_R�.1?

learning_rate_1��5��t�I       6%�	'��j���A�5*;


total_loss-"�@

error_R�>?

learning_rate_1��5H��!I       6%�	2�j���A�5*;


total_lossQxA

error_R��G?

learning_rate_1��5�N��I       6%�	�G�j���A�5*;


total_loss֋�@

error_R�[j?

learning_rate_1��5#)5I       6%�	!��j���A�5*;


total_loss��@

error_R8�a?

learning_rate_1��5GY�I       6%�	�γj���A�5*;


total_loss���@

error_R�M?

learning_rate_1��5�JI       6%�	T�j���A�5*;


total_loss�6�@

error_R,fV?

learning_rate_1��5���I       6%�	�R�j���A�5*;


total_loss�~�@

error_R[�8?

learning_rate_1��5�eZI       6%�	~��j���A�5*;


total_loss�%�@

error_R�L?

learning_rate_1��5R�rI       6%�	qٴj���A�5*;


total_loss-"�@

error_R��^?

learning_rate_1��5n��aI       6%�	I�j���A�5*;


total_loss�@�@

error_R!�Q?

learning_rate_1��5���I       6%�	Y[�j���A�5*;


total_losswǳ@

error_Ri�Q?

learning_rate_1��5�=��I       6%�	_��j���A�5*;


total_lossO��@

error_R�EC?

learning_rate_1��5<��I       6%�	eߵj���A�5*;


total_lossx{@

error_RJJB?

learning_rate_1��5$��rI       6%�	�j���A�5*;


total_loss+�@

error_R�[N?

learning_rate_1��5�NiI       6%�	�]�j���A�5*;


total_loss<��@

error_R�O:?

learning_rate_1��5���}I       6%�	�j���A�5*;


total_loss=ѫ@

error_RW�<?

learning_rate_1��5���I       6%�	$߶j���A�5*;


total_lossD��@

error_RjfB?

learning_rate_1��5�I       6%�	| �j���A�5*;


total_loss#��@

error_RIfS?

learning_rate_1��5=�<�I       6%�	�c�j���A�5*;


total_loss�,�@

error_R\�F?

learning_rate_1��5,�2�I       6%�	䢷j���A�5*;


total_lossN��@

error_R�G?

learning_rate_1��5��sI       6%�	��j���A�5*;


total_loss�+�@

error_R̢Q?

learning_rate_1��5pL�~I       6%�	O#�j���A�5*;


total_lossFF�@

error_R�T?

learning_rate_1��5@J��I       6%�	�d�j���A�5*;


total_loss��@

error_R�9V?

learning_rate_1��5���I       6%�	4��j���A�5*;


total_lossܵ	A

error_R�@S?

learning_rate_1��5��I       6%�	��j���A�5*;


total_lossE4|@

error_R8�8?

learning_rate_1��5�̞I       6%�	#�j���A�5*;


total_loss ��@

error_R��Z?

learning_rate_1��5:��I       6%�	d�j���A�5*;


total_loss7ɳ@

error_R#�>?

learning_rate_1��5���I       6%�	d��j���A�5*;


total_loss�Tv@

error_R�6D?

learning_rate_1��5����I       6%�	��j���A�5*;


total_loss�Ԭ@

error_R��M?

learning_rate_1��5����I       6%�	�&�j���A�5*;


total_loss��@

error_R�B?

learning_rate_1��5��q�I       6%�	�j�j���A�5*;


total_lossҫ�@

error_R?�3?

learning_rate_1��5Nr��I       6%�	K��j���A�5*;


total_loss��@

error_RV5W?

learning_rate_1��55�tLI       6%�	��j���A�5*;


total_loss�8�@

error_R��<?

learning_rate_1��5?ĵI       6%�	 0�j���A�5*;


total_loss��@

error_R�:?

learning_rate_1��5%���I       6%�	}v�j���A�5*;


total_loss_��@

error_R�Z@?

learning_rate_1��5����I       6%�	���j���A�5*;


total_loss���@

error_R��[?

learning_rate_1��5~��I       6%�	��j���A�5*;


total_loss�v@

error_R1�S?

learning_rate_1��5��I       6%�	�H�j���A�5*;


total_lossN�A

error_RE�Y?

learning_rate_1��5`�iI       6%�	��j���A�5*;


total_lossv*�@

error_Rd�R?

learning_rate_1��5�^(~I       6%�	Xȼj���A�5*;


total_loss6ı@

error_R!H?

learning_rate_1��5s-�I       6%�	�j���A�5*;


total_lossN�*A

error_R�V?

learning_rate_1��5�+xgI       6%�	�H�j���A�5*;


total_lossE��@

error_R�iT?

learning_rate_1��5򖤶I       6%�	d��j���A�5*;


total_loss��@

error_RJeL?

learning_rate_1��5��j{I       6%�	,ǽj���A�5*;


total_loss���@

error_Rz?J?

learning_rate_1��57�I       6%�	�j���A�5*;


total_loss#�@

error_R�eZ?

learning_rate_1��5��5%I       6%�	�r�j���A�5*;


total_loss=�@

error_RN�H?

learning_rate_1��5�"YlI       6%�	���j���A�5*;


total_losso��@

error_R6�N?

learning_rate_1��5#��I       6%�	���j���A�5*;


total_loss�@

error_RO1?

learning_rate_1��5����I       6%�	};�j���A�5*;


total_lossհ@

error_R�iL?

learning_rate_1��5�'JI       6%�	�|�j���A�5*;


total_loss�a�@

error_R[�H?

learning_rate_1��5!f VI       6%�	��j���A�5*;


total_lossjZ�@

error_RQ0K?

learning_rate_1��5?���I       6%�	���j���A�5*;


total_lossI�b@

error_R��G?

learning_rate_1��5��{I       6%�	�?�j���A�5*;


total_loss۟�@

error_R�P?

learning_rate_1��5���I       6%�	��j���A�5*;


total_loss�6�@

error_R��R?

learning_rate_1��5\"I       6%�	r��j���A�5*;


total_loss�:�@

error_Rn�>?

learning_rate_1��5���#I       6%�	���j���A�5*;


total_loss�ݬ@

error_R�&T?

learning_rate_1��5���uI       6%�	m=�j���A�5*;


total_loss�a�@

error_R ??

learning_rate_1��5�V^�I       6%�	r�j���A�5*;


total_loss��@

error_R��e?

learning_rate_1��54�	EI       6%�	:��j���A�5*;


total_loss�z@

error_R��[?

learning_rate_1��5ȉ>I       6%�	�j���A�5*;


total_loss�@

error_R�F?

learning_rate_1��5���I       6%�	�H�j���A�5*;


total_lossTJu@

error_R�A?

learning_rate_1��5xɗ1I       6%�	P��j���A�5*;


total_loss9BA

error_R��M?

learning_rate_1��5'*��I       6%�	���j���A�5*;


total_loss��@

error_R�O?

learning_rate_1��5��4!I       6%�	%�j���A�5*;


total_loss[1�@

error_RahQ?

learning_rate_1��5X��I       6%�	�R�j���A�5*;


total_loss׾�@

error_R��J?

learning_rate_1��5©k@I       6%�	���j���A�5*;


total_loss���@

error_R�[?

learning_rate_1��5j$ԳI       6%�	9��j���A�5*;


total_loss@

error_RD�O?

learning_rate_1��5`�I       6%�	X�j���A�5*;


total_loss��@

error_R�FH?

learning_rate_1��5�&.�I       6%�	qZ�j���A�5*;


total_losso��@

error_R�kN?

learning_rate_1��5���I       6%�	X��j���A�5*;


total_lossD��@

error_R��V?

learning_rate_1��5KP	*I       6%�	%��j���A�5*;


total_loss#m_@

error_RE�D?

learning_rate_1��5k,��I       6%�	�j���A�5*;


total_loss�W|@

error_RZ�K?

learning_rate_1��5� ��I       6%�	�W�j���A�5*;


total_loss��@

error_R#O?

learning_rate_1��5��wI       6%�	З�j���A�5*;


total_loss�� A

error_R8F?

learning_rate_1��5�-K[I       6%�	E��j���A�5*;


total_loss���@

error_R��M?

learning_rate_1��5a��I       6%�	��j���A�5*;


total_loss��A

error_RSU?

learning_rate_1��5��)I       6%�	�^�j���A�5*;


total_loss�J A

error_R�C?

learning_rate_1��5��ۚI       6%�	��j���A�5*;


total_loss�4�@

error_R��L?

learning_rate_1��5�9�OI       6%�	���j���A�5*;


total_loss�ڔ@

error_R�P?

learning_rate_1��5�Dj�I       6%�	�'�j���A�5*;


total_loss�ݎ@

error_R@�G?

learning_rate_1��5�r��I       6%�	�j�j���A�5*;


total_loss��@

error_R��V?

learning_rate_1��5v���I       6%�	��j���A�5*;


total_loss���@

error_R
�X?

learning_rate_1��5�I       6%�	��j���A�5*;


total_lossd{A

error_R�RY?

learning_rate_1��5F��I       6%�	;�j���A�5*;


total_lossT��@

error_R�IQ?

learning_rate_1��5�2d�I       6%�	b}�j���A�5*;


total_loss�h�@

error_RtD>?

learning_rate_1��5�~��I       6%�	Ծ�j���A�5*;


total_loss�z�@

error_RßL?

learning_rate_1��5��x.I       6%�	���j���A�5*;


total_loss��@

error_R��V?

learning_rate_1��5�[�(I       6%�	G@�j���A�5*;


total_lossm��@

error_RJ�X?

learning_rate_1��5��-I       6%�	�j���A�5*;


total_lossQ�@

error_R��N?

learning_rate_1��5w�m�I       6%�	f��j���A�5*;


total_loss�_�@

error_R6�P?

learning_rate_1��5���I       6%�	3��j���A�5*;


total_loss�"�@

error_R�}B?

learning_rate_1��5q���I       6%�	�A�j���A�5*;


total_loss�܎@

error_R}GE?

learning_rate_1��5���I       6%�	��j���A�5*;


total_loss��@

error_R89S?

learning_rate_1��5�L��I       6%�	��j���A�5*;


total_loss;��@

error_RI?

learning_rate_1��5����I       6%�	x�j���A�5*;


total_loss�ϔ@

error_R��H?

learning_rate_1��5q�ǝI       6%�	MI�j���A�5*;


total_loss�i�@

error_R��I?

learning_rate_1��5W�II       6%�	q��j���A�5*;


total_loss�~�@

error_R�5?

learning_rate_1��5eߍEI       6%�	���j���A�5*;


total_loss]�@

error_R��R?

learning_rate_1��5�o�XI       6%�	��j���A�5*;


total_loss��@

error_RAF?

learning_rate_1��5�& I       6%�	xS�j���A�5*;


total_loss�d�@

error_R�._?

learning_rate_1��5HsI       6%�	M��j���A�5*;


total_loss���@

error_RI%A?

learning_rate_1��5��aOI       6%�	(��j���A�5*;


total_loss�׌@

error_R��U?

learning_rate_1��5�PʍI       6%�	(;�j���A�5*;


total_loss@��@

error_R�[:?

learning_rate_1��5��\I       6%�	{��j���A�5*;


total_lossA

error_R�cG?

learning_rate_1��5Im@�I       6%�	��j���A�5*;


total_loss�ߣ@

error_R�-K?

learning_rate_1��5 ��I       6%�	�%�j���A�5*;


total_lossFà@

error_RʅP?

learning_rate_1��5 �^I       6%�	��j���A�5*;


total_loss���@

error_Rɀ9?

learning_rate_1��5g)��I       6%�	���j���A�5*;


total_loss*y�@

error_R�P?

learning_rate_1��5�J$I       6%�	��j���A�5*;


total_lossA

error_R�M?

learning_rate_1��5�U��I       6%�	Q�j���A�5*;


total_lossV��@

error_RV�R?

learning_rate_1��5��/I       6%�	��j���A�5*;


total_loss�A

error_R8�R?

learning_rate_1��5�X�I       6%�	��j���A�5*;


total_loss/��@

error_R�UR?

learning_rate_1��5'O I       6%�	�'�j���A�5*;


total_loss�dR@

error_R��R?

learning_rate_1��5'�}I       6%�	�i�j���A�5*;


total_loss�]�@

error_RNaM?

learning_rate_1��5�8��I       6%�	T��j���A�5*;


total_loss6ǭ@

error_RaWP?

learning_rate_1��5�;��I       6%�	���j���A�5*;


total_loss���@

error_Rw�Z?

learning_rate_1��5c;�0I       6%�	o0�j���A�5*;


total_loss�ϊ@

error_Rd�S?

learning_rate_1��5�N�I       6%�	�q�j���A�5*;


total_loss�[�@

error_R<�D?

learning_rate_1��5��1I       6%�	���j���A�5*;


total_losszW�@

error_Rf�S?

learning_rate_1��5���#I       6%�	#��j���A�5*;


total_loss��@

error_R�QT?

learning_rate_1��5oӊ�I       6%�	�H�j���A�5*;


total_loss��@

error_R��R?

learning_rate_1��50C2I       6%�	���j���A�5*;


total_lossw1�@

error_R_�:?

learning_rate_1��5�Ҟ�I       6%�	���j���A�6*;


total_loss-J�@

error_R��L?

learning_rate_1��5���$I       6%�	M�j���A�6*;


total_loss[�@

error_R$I?

learning_rate_1��5���I       6%�	X�j���A�6*;


total_loss�O�@

error_R{$P?

learning_rate_1��5ҩ�I       6%�	���j���A�6*;


total_loss�ɴ@

error_Rze?

learning_rate_1��58��I       6%�	��j���A�6*;


total_loss�L�@

error_Rr�R?

learning_rate_1��53 CI       6%�	�0�j���A�6*;


total_loss|w�@

error_R{�B?

learning_rate_1��5��O"I       6%�	&y�j���A�6*;


total_loss�;�@

error_RH&X?

learning_rate_1��5���I       6%�	���j���A�6*;


total_loss�@�@

error_R��N?

learning_rate_1��5�?�GI       6%�	��j���A�6*;


total_loss_��@

error_Rצ=?

learning_rate_1��5��Z�I       6%�	M�j���A�6*;


total_loss_W�@

error_RtU?

learning_rate_1��5���I       6%�	R��j���A�6*;


total_loss���@

error_R��;?

learning_rate_1��5< 3�I       6%�	j��j���A�6*;


total_loss�W�@

error_R�sf?

learning_rate_1��5�)�I       6%�	��j���A�6*;


total_lossf��@

error_R��M?

learning_rate_1��5ۯe�I       6%�	�P�j���A�6*;


total_loss���@

error_R��J?

learning_rate_1��5JH�,I       6%�	_��j���A�6*;


total_loss[��@

error_R�W?

learning_rate_1��5l)��I       6%�	���j���A�6*;


total_loss�@

error_R�fT?

learning_rate_1��5*l�I       6%�	m�j���A�6*;


total_loss�"�@

error_R[�U?

learning_rate_1��5G�I       6%�	sX�j���A�6*;


total_lossQٮ@

error_R
]O?

learning_rate_1��5x=��I       6%�	���j���A�6*;


total_losso�@A

error_R�F?

learning_rate_1��5cv�I       6%�	D��j���A�6*;


total_loss��@

error_R]�V?

learning_rate_1��5 D��I       6%�	�&�j���A�6*;


total_loss曢@

error_R��J?

learning_rate_1��55�RI       6%�	ii�j���A�6*;


total_loss�܆@

error_R�-D?

learning_rate_1��5�0h�I       6%�	���j���A�6*;


total_loss3#�@

error_R�uX?

learning_rate_1��5�y�I       6%�	���j���A�6*;


total_loss��@

error_R��L?

learning_rate_1��5����I       6%�	),�j���A�6*;


total_loss��@

error_R��O?

learning_rate_1��5����I       6%�	�r�j���A�6*;


total_lossr�@

error_R��;?

learning_rate_1��5[%͗I       6%�	��j���A�6*;


total_loss6ľ@

error_R!eO?

learning_rate_1��5��H8I       6%�	���j���A�6*;


total_loss��@

error_R��D?

learning_rate_1��5��I       6%�	�B�j���A�6*;


total_loss8��@

error_R�GL?

learning_rate_1��5��zI       6%�	���j���A�6*;


total_loss-ˀ@

error_R�N?

learning_rate_1��5��I       6%�	7��j���A�6*;


total_loss��@

error_R�!Z?

learning_rate_1��5-�xMI       6%�		�j���A�6*;


total_loss�S�@

error_R�N?

learning_rate_1��5c�I       6%�	Q�j���A�6*;


total_lossh�k@

error_R��O?

learning_rate_1��5���I       6%�	>��j���A�6*;


total_lossӤd@

error_R��E?

learning_rate_1��5��OgI       6%�	x��j���A�6*;


total_loss@��@

error_R_E?

learning_rate_1��5�i�!I       6%�	�9�j���A�6*;


total_loss�b�@

error_R��S?

learning_rate_1��5����I       6%�	�z�j���A�6*;


total_loss;�@

error_R��W?

learning_rate_1��5���TI       6%�	���j���A�6*;


total_loss��@

error_R�E?

learning_rate_1��5��ؙI       6%�	��j���A�6*;


total_loss׾�@

error_R�f4?

learning_rate_1��5ӹ� I       6%�	,H�j���A�6*;


total_loss�ƿ@

error_RڞB?

learning_rate_1��5�8� I       6%�	���j���A�6*;


total_loss�T�@

error_R �O?

learning_rate_1��5��`�I       6%�	4��j���A�6*;


total_lossOx�@

error_R�[?

learning_rate_1��5��d�I       6%�	WS�j���A�6*;


total_loss	>�@

error_R��R?

learning_rate_1��5�U�I       6%�	B��j���A�6*;


total_loss;��@

error_R�EL?

learning_rate_1��5���I       6%�	���j���A�6*;


total_lossC��@

error_R ;?

learning_rate_1��5~�fI       6%�	�7�j���A�6*;


total_lossH��@

error_R��??

learning_rate_1��5{{�I       6%�	C��j���A�6*;


total_loss���@

error_R�R?

learning_rate_1��50��I       6%�	'��j���A�6*;


total_loss���@

error_R��L?

learning_rate_1��5��ߒI       6%�	�*�j���A�6*;


total_lossR��@

error_R�rA?

learning_rate_1��5��&I       6%�	U��j���A�6*;


total_losse�@

error_RȟV?

learning_rate_1��5͠�:I       6%�	���j���A�6*;


total_losse��@

error_R�X?

learning_rate_1��5M�fI       6%�	�'�j���A�6*;


total_loss8��@

error_R�D?

learning_rate_1��5>��I       6%�	_��j���A�6*;


total_loss���@

error_R�Y?

learning_rate_1��5�# �I       6%�	��j���A�6*;


total_loss��@

error_Ri�S?

learning_rate_1��5 SFQI       6%�	
"�j���A�6*;


total_lossσA

error_R=/X?

learning_rate_1��5Pu��I       6%�	�o�j���A�6*;


total_loss�֯@

error_R.�F?

learning_rate_1��5uV|�I       6%�	h��j���A�6*;


total_loss�R�@

error_R�?V?

learning_rate_1��5M`��I       6%�	M �j���A�6*;


total_loss���@

error_R,W^?

learning_rate_1��5q%� I       6%�	�f�j���A�6*;


total_loss�}@

error_RLf?

learning_rate_1��5V��I       6%�	���j���A�6*;


total_loss�YA

error_RʄQ?

learning_rate_1��5�9�I       6%�	��j���A�6*;


total_loss���@

error_RIxJ?

learning_rate_1��5��QmI       6%�	�a�j���A�6*;


total_loss��@

error_RW�F?

learning_rate_1��5|)mI       6%�	���j���A�6*;


total_loss�L�@

error_RES?

learning_rate_1��5mʼI       6%�	��j���A�6*;


total_loss��@

error_R�`F?

learning_rate_1��5.��DI       6%�	�S�j���A�6*;


total_lossꐞ@

error_R��Q?

learning_rate_1��5ُ��I       6%�	 ��j���A�6*;


total_loss���@

error_Riwa?

learning_rate_1��56�idI       6%�	E��j���A�6*;


total_loss�7�@

error_Rs0??

learning_rate_1��5���tI       6%�	A �j���A�6*;


total_lossz�@

error_RV�I?

learning_rate_1��5�b��I       6%�	�`�j���A�6*;


total_loss� �@

error_RK?

learning_rate_1��5l�o�I       6%�	���j���A�6*;


total_lossƇ�@

error_R|wW?

learning_rate_1��53�<�I       6%�	Y��j���A�6*;


total_loss6��@

error_R$]?

learning_rate_1��5�ȚlI       6%�	x0�j���A�6*;


total_loss�^�@

error_R�gO?

learning_rate_1��5��p5I       6%�	
{�j���A�6*;


total_loss;��@

error_RM�[?

learning_rate_1��5ێ�8I       6%�	@��j���A�6*;


total_loss/��@

error_R��G?

learning_rate_1��5tPb�I       6%�	�>�j���A�6*;


total_loss�A

error_RM4a?

learning_rate_1��5�EI       6%�	��j���A�6*;


total_loss��@

error_Rl�J?

learning_rate_1��5���`I       6%�	���j���A�6*;


total_loss�?�@

error_R7LG?

learning_rate_1��5��I       6%�	�&�j���A�6*;


total_loss���@

error_R�ga?

learning_rate_1��5J�B�I       6%�	ni�j���A�6*;


total_loss �@

error_R�R?

learning_rate_1��5P9cI       6%�	#��j���A�6*;


total_loss�C�@

error_Ri�T?

learning_rate_1��5��I       6%�	W��j���A�6*;


total_loss�|�@

error_R�M?

learning_rate_1�h�5��EI       6%�	�:�j���A�6*;


total_loss��@

error_R�Mc?

learning_rate_1�h�5��I       6%�	�j���A�6*;


total_loss�S�@

error_RuS?

learning_rate_1�h�57�؈I       6%�	��j���A�6*;


total_loss\�@

error_Ro�I?

learning_rate_1�h�5ه�I       6%�	c	�j���A�6*;


total_lossw�~@

error_R϶S?

learning_rate_1�h�545�oI       6%�	�R�j���A�6*;


total_loss�W}@

error_R��T?

learning_rate_1�h�5OC~I       6%�	��j���A�6*;


total_loss;�@

error_R�=8?

learning_rate_1�h�5?�P�I       6%�	���j���A�6*;


total_loss���@

error_R��V?

learning_rate_1�h�5&�YI       6%�	�)�j���A�6*;


total_loss���@

error_R��O?

learning_rate_1�h�5��s3I       6%�	�x�j���A�6*;


total_lossv��@

error_R�d<?

learning_rate_1�h�5��~�I       6%�	���j���A�6*;


total_loss��@

error_R!QJ?

learning_rate_1�h�5/J0I       6%�	�=�j���A�6*;


total_loss,��@

error_R��S?

learning_rate_1�h�5���I       6%�	��j���A�6*;


total_loss}Ѥ@

error_R�:C?

learning_rate_1�h�5A��OI       6%�	���j���A�6*;


total_loss���@

error_RJ�J?

learning_rate_1�h�5a�Z
I       6%�	��j���A�6*;


total_lossR��@

error_R��l?

learning_rate_1�h�5?7�#I       6%�	�d�j���A�6*;


total_loss�{�@

error_R��M?

learning_rate_1�h�5�JI       6%�	k��j���A�6*;


total_loss���@

error_R��N?

learning_rate_1�h�5HLMI       6%�	���j���A�6*;


total_lossk��@

error_R�MZ?

learning_rate_1�h�5��e�I       6%�	X>�j���A�6*;


total_loss��@

error_R\/?

learning_rate_1�h�5��I       6%�	`��j���A�6*;


total_lossa��@

error_R1�`?

learning_rate_1�h�5_�5I       6%�	��j���A�6*;


total_loss��@

error_R��L?

learning_rate_1�h�5�V�YI       6%�	�&�j���A�6*;


total_losszu�@

error_R�M?

learning_rate_1�h�5���I       6%�	(��j���A�6*;


total_lossR�@

error_RQ�X?

learning_rate_1�h�5~FΪI       6%�	|��j���A�6*;


total_loss��@

error_RפU?

learning_rate_1�h�5��`�I       6%�	�)�j���A�6*;


total_loss[N�@

error_RT	_?

learning_rate_1�h�5�Ys�I       6%�	ș�j���A�6*;


total_loss��A

error_R�N?

learning_rate_1�h�5�X��I       6%�	���j���A�6*;


total_lossz��@

error_R3E?

learning_rate_1�h�5�� 5I       6%�	b0�j���A�6*;


total_lossvh�@

error_Ri+V?

learning_rate_1�h�5w���I       6%�	~}�j���A�6*;


total_loss��@

error_R�OG?

learning_rate_1�h�5��"`I       6%�	E��j���A�6*;


total_loss���@

error_Rx�L?

learning_rate_1�h�5���=I       6%�	E�j���A�6*;


total_loss� A

error_R��<?

learning_rate_1�h�5"��I       6%�	�Z�j���A�6*;


total_lossX�@

error_R�V?

learning_rate_1�h�5O�dI       6%�	5��j���A�6*;


total_loss6њ@

error_R�-V?

learning_rate_1�h�5UC/I       6%�	'��j���A�6*;


total_loss���@

error_R�2T?

learning_rate_1�h�5�2��I       6%�	�B�j���A�6*;


total_loss��@

error_R�:P?

learning_rate_1�h�5����I       6%�	���j���A�6*;


total_loss��
A

error_R[]C?

learning_rate_1�h�5n��I       6%�	��j���A�6*;


total_loss�A

error_R=[Q?

learning_rate_1�h�50֧bI       6%�	\�j���A�6*;


total_loss��@

error_R}�A?

learning_rate_1�h�5٥ŜI       6%�	�a�j���A�6*;


total_loss�.�@

error_R�0K?

learning_rate_1�h�5uUFI       6%�	���j���A�6*;


total_loss�)�@

error_R*M?

learning_rate_1�h�5$���I       6%�	���j���A�6*;


total_loss��@

error_R�PW?

learning_rate_1�h�5�PI       6%�	�:�j���A�6*;


total_loss@��@

error_R�K?

learning_rate_1�h�5G�ɅI       6%�	��j���A�6*;


total_loss:O�@

error_R�"V?

learning_rate_1�h�5KF�sI       6%�	v��j���A�6*;


total_loss�bA

error_R��D?

learning_rate_1�h�5�T�I       6%�	�j���A�6*;


total_lossL��@

error_R�uY?

learning_rate_1�h�5E�x�I       6%�	���j���A�6*;


total_lossz��@

error_R�=e?

learning_rate_1�h�5�6L�I       6%�	<��j���A�6*;


total_loss�[�@

error_RҌI?

learning_rate_1�h�5�P6I       6%�	-�j���A�6*;


total_loss��@

error_R=T?

learning_rate_1�h�5�K�I       6%�	���j���A�7*;


total_loss���@

error_RT�X?

learning_rate_1�h�5g&xI       6%�	<��j���A�7*;


total_loss�}_@

error_RfdO?

learning_rate_1�h�5v��)I       6%�	�A�j���A�7*;


total_loss���@

error_R�H\?

learning_rate_1�h�5OƸI       6%�	���j���A�7*;


total_loss�'�@

error_R��F?

learning_rate_1�h�5�`(II       6%�	R��j���A�7*;


total_loss)ߝ@

error_R,�\?

learning_rate_1�h�5 �7�I       6%�	76�j���A�7*;


total_loss͒@

error_R<�B?

learning_rate_1�h�5�H�XI       6%�	$|�j���A�7*;


total_loss���@

error_RaG7?

learning_rate_1�h�5�тRI       6%�	���j���A�7*;


total_loss� A

error_R�G?

learning_rate_1�h�5%#�I       6%�	�D�j���A�7*;


total_loss]i@

error_R��>?

learning_rate_1�h�5#��I       6%�	`��j���A�7*;


total_lossRS�@

error_R�2?

learning_rate_1�h�5���hI       6%�	�*�j���A�7*;


total_loss�k�@

error_R�O?

learning_rate_1�h�5}NEKI       6%�	Ѐ�j���A�7*;


total_loss�ٷ@

error_RZB?

learning_rate_1�h�5���I       6%�	���j���A�7*;


total_loss�~@

error_R��S?

learning_rate_1�h�5���I       6%�	-2 k���A�7*;


total_loss���@

error_R�/Y?

learning_rate_1�h�5��tI       6%�	| k���A�7*;


total_lossƾ�@

error_R��H?

learning_rate_1�h�5��I       6%�	� k���A�7*;


total_loss��@

error_R�H?

learning_rate_1�h�5���+I       6%�	Ck���A�7*;


total_loss4��@

error_R��J?

learning_rate_1�h�5�V��I       6%�	Ekk���A�7*;


total_loss=�@

error_RD!F?

learning_rate_1�h�5f� �I       6%�	�k���A�7*;


total_loss6ۙ@

error_Ra3K?

learning_rate_1�h�5'D�PI       6%�	5�k���A�7*;


total_lossx�\@

error_R��[?

learning_rate_1�h�5;_�|I       6%�	�?k���A�7*;


total_loss��@

error_R��5?

learning_rate_1�h�5ĀI       6%�	��k���A�7*;


total_loss|�@

error_Ri�T?

learning_rate_1�h�5���I       6%�	
�k���A�7*;


total_losski�@

error_R�AZ?

learning_rate_1�h�5;�FI       6%�	�
k���A�7*;


total_lossJ<�@

error_RE�]?

learning_rate_1�h�5{���I       6%�	uNk���A�7*;


total_loss$��@

error_Rd�R?

learning_rate_1�h�5��Z�I       6%�	t�k���A�7*;


total_loss�)�@

error_Ri�=?

learning_rate_1�h�5 O��I       6%�	%�k���A�7*;


total_loss1�@

error_R�0E?

learning_rate_1�h�5���+I       6%�	�(k���A�7*;


total_lossW�A

error_R_�Q?

learning_rate_1�h�5$��I       6%�	�rk���A�7*;


total_loss���@

error_Rq�X?

learning_rate_1�h�5r��I       6%�	�k���A�7*;


total_loss�@

error_R%+M?

learning_rate_1�h�5�X�I       6%�	<�k���A�7*;


total_loss��@

error_R�4Y?

learning_rate_1�h�5��GI       6%�	�@k���A�7*;


total_loss]k�@

error_RL[A?

learning_rate_1�h�5I��DI       6%�	Ƀk���A�7*;


total_loss��	A

error_R��7?

learning_rate_1�h�5��GI       6%�	d�k���A�7*;


total_loss�Ŋ@

error_R�#c?

learning_rate_1�h�5�D�I       6%�	}k���A�7*;


total_loss���@

error_R�*Y?

learning_rate_1�h�5�$d�I       6%�	o^k���A�7*;


total_loss���@

error_R�vR?

learning_rate_1�h�5m��I       6%�	A�k���A�7*;


total_loss��@

error_R��>?

learning_rate_1�h�5�h7�I       6%�	y�k���A�7*;


total_loss��@

error_R�G?

learning_rate_1�h�5�h�I       6%�	�2k���A�7*;


total_loss=�j@

error_R D?

learning_rate_1�h�5|g�I       6%�	{k���A�7*;


total_lossvV�@

error_RH(P?

learning_rate_1�h�5���I       6%�	a�k���A�7*;


total_loss�,�@

error_R�kT?

learning_rate_1�h�5T��I       6%�	1	k���A�7*;


total_loss���@

error_RL�:?

learning_rate_1�h�5J���I       6%�	�Kk���A�7*;


total_lossͨ�@

error_R��@?

learning_rate_1�h�5>�R�I       6%�	�k���A�7*;


total_loss(WA

error_RI�C?

learning_rate_1�h�5����I       6%�	j�k���A�7*;


total_lossmn�@

error_R.�L?

learning_rate_1�h�5�UI       6%�	Q	k���A�7*;


total_loss�f�@

error_R�Z?

learning_rate_1�h�5r6nI       6%�	�b	k���A�7*;


total_lossY�@

error_R�ED?

learning_rate_1�h�5�7�DI       6%�	2�	k���A�7*;


total_lossڿ�@

error_R�P?

learning_rate_1�h�5i��(I       6%�	��	k���A�7*;


total_loss�5�@

error_RO=I?

learning_rate_1�h�5�КwI       6%�	 +
k���A�7*;


total_loss���@

error_R�$M?

learning_rate_1�h�5+Q�I       6%�	gq
k���A�7*;


total_lossI!�@

error_R(�E?

learning_rate_1�h�5~��I       6%�	��
k���A�7*;


total_loss��@

error_R.�T?

learning_rate_1�h�5��*'I       6%�	��
k���A�7*;


total_loss���@

error_R��P?

learning_rate_1�h�5��e�I       6%�	.?k���A�7*;


total_lossX�@

error_R�0A?

learning_rate_1�h�5����I       6%�	��k���A�7*;


total_loss�c�@

error_R��W?

learning_rate_1�h�5<��}I       6%�	��k���A�7*;


total_loss<Q�@

error_R��Y?

learning_rate_1�h�5B� �I       6%�	Dk���A�7*;


total_loss�Ӌ@

error_R��F?

learning_rate_1�h�5���I       6%�	>fk���A�7*;


total_loss�C�@

error_R�.P?

learning_rate_1�h�5�M�I       6%�	E�k���A�7*;


total_loss�Κ@

error_RJUC?

learning_rate_1�h�5��I       6%�	�k���A�7*;


total_loss�{�@

error_R)�^?

learning_rate_1�h�5��aI       6%�	�@k���A�7*;


total_lossu�@

error_R\�Y?

learning_rate_1�h�5V;��I       6%�	�k���A�7*;


total_loss�(�@

error_R��X?

learning_rate_1�h�5N��I       6%�	��k���A�7*;


total_loss�1�@

error_R\�Q?

learning_rate_1�h�5�w�I       6%�	�k���A�7*;


total_lossj�@

error_R�M?

learning_rate_1�h�5��OI       6%�	�zk���A�7*;


total_loss�8�@

error_R�F?

learning_rate_1�h�5�pMlI       6%�	q�k���A�7*;


total_loss��@

error_R�fO?

learning_rate_1�h�5�eN/I       6%�		k���A�7*;


total_loss�U�@

error_R� H?

learning_rate_1�h�5a%ʃI       6%�	�Lk���A�7*;


total_lossv�@

error_R��N?

learning_rate_1�h�5����I       6%�	�k���A�7*;


total_loss���@

error_R! `?

learning_rate_1�h�5��ZQI       6%�	r�k���A�7*;


total_loss�m�@

error_R�bI?

learning_rate_1�h�5��) I       6%�	v*k���A�7*;


total_loss�x�@

error_R��>?

learning_rate_1�h�5"��I       6%�	�vk���A�7*;


total_loss���@

error_R�`A?

learning_rate_1�h�5� ��I       6%�	�k���A�7*;


total_lossA�@

error_R��U?

learning_rate_1�h�5�'��I       6%�	�k���A�7*;


total_loss/�@

error_R�^?

learning_rate_1�h�5Zt�nI       6%�	�Pk���A�7*;


total_loss�E�@

error_Rw�Z?

learning_rate_1�h�5��ڋI       6%�	�k���A�7*;


total_lossT�@

error_R#"T?

learning_rate_1�h�5���I       6%�	(�k���A�7*;


total_lossݙ�@

error_RvcI?

learning_rate_1�h�5��I       6%�	$k���A�7*;


total_lossFu�@

error_Rh�R?

learning_rate_1�h�5�}_�I       6%�	Kmk���A�7*;


total_loss�3�@

error_R|Y?

learning_rate_1�h�5}`I       6%�	�k���A�7*;


total_loss<A

error_R��h?

learning_rate_1�h�5d�_KI       6%�	��k���A�7*;


total_lossc��@

error_R��W?

learning_rate_1�h�5-ʀwI       6%�	�>k���A�7*;


total_losss��@

error_RT?

learning_rate_1�h�5j�`I       6%�	I�k���A�7*;


total_loss*j�@

error_Ra�I?

learning_rate_1�h�5K](`I       6%�	,�k���A�7*;


total_lossr��@

error_RϮO?

learning_rate_1�h�5۠�I       6%�	�k���A�7*;


total_lossOVA

error_R?kR?

learning_rate_1�h�5�_OI       6%�	�Sk���A�7*;


total_loss�%�@

error_R��G?

learning_rate_1�h�5M�8I       6%�	��k���A�7*;


total_lossmo�@

error_R�%]?

learning_rate_1�h�5`��I       6%�	g�k���A�7*;


total_loss���@

error_RlL?

learning_rate_1�h�5��UCI       6%�	�+k���A�7*;


total_lossl�k@

error_RίK?

learning_rate_1�h�5�G{=I       6%�	trk���A�7*;


total_lossX{�@

error_R�$Q?

learning_rate_1�h�5g6oaI       6%�	{�k���A�7*;


total_loss9��@

error_R��Q?

learning_rate_1�h�5 �KI       6%�	T�k���A�7*;


total_lossD'�@

error_RO�K?

learning_rate_1�h�5ʻTnI       6%�	PAk���A�7*;


total_loss/�@

error_RQI?

learning_rate_1�h�5gy~I       6%�	��k���A�7*;


total_lossA�@

error_R2�N?

learning_rate_1�h�5 ��I       6%�	��k���A�7*;


total_loss��@

error_R�sZ?

learning_rate_1�h�5��?I       6%�	�k���A�7*;


total_loss�]�@

error_R��E?

learning_rate_1�h�5�-��I       6%�	�Sk���A�7*;


total_lossβ@

error_R,�B?

learning_rate_1�h�5V�hI       6%�	��k���A�7*;


total_loss�>�@

error_R�9?

learning_rate_1�h�59�(I       6%�	��k���A�7*;


total_loss�8�@

error_R�T?

learning_rate_1�h�5i|�I       6%�	 "k���A�7*;


total_loss
 x@

error_RwwA?

learning_rate_1�h�5<.�ZI       6%�	ifk���A�7*;


total_loss�y�@

error_R��R?

learning_rate_1�h�5:0�pI       6%�	i�k���A�7*;


total_loss}bA

error_R�	??

learning_rate_1�h�5�
�;I       6%�	��k���A�7*;


total_loss���@

error_R�XT?

learning_rate_1�h�5Gm��I       6%�	9k���A�7*;


total_loss���@

error_R�/b?

learning_rate_1�h�5��_�I       6%�	�}k���A�7*;


total_loss��A

error_R
I?

learning_rate_1�h�5)�7pI       6%�	��k���A�7*;


total_loss�5A

error_R	�_?

learning_rate_1�h�5-1wI       6%�	�k���A�7*;


total_loss��@

error_R��L?

learning_rate_1�h�5cw�NI       6%�	�Vk���A�7*;


total_loss��%A

error_RE:b?

learning_rate_1�h�5�{�mI       6%�	|�k���A�7*;


total_lossR��@

error_R}�N?

learning_rate_1�h�5Ym�%I       6%�	�k���A�7*;


total_loss���@

error_R��O?

learning_rate_1�h�5�E�I       6%�	'#k���A�7*;


total_loss�	�@

error_R-XV?

learning_rate_1�h�5��%XI       6%�	\rk���A�7*;


total_loss�"�@

error_RֈG?

learning_rate_1�h�5m�;0I       6%�	��k���A�7*;


total_loss�'�@

error_R�Q?

learning_rate_1�h�5f' I       6%�	�k���A�7*;


total_lossä�@

error_R�hF?

learning_rate_1�h�5N{d�I       6%�	xLk���A�7*;


total_loss�2�@

error_R�]?

learning_rate_1�h�5�SLI       6%�	Йk���A�7*;


total_loss�z�@

error_Rq1B?

learning_rate_1�h�5}�b;I       6%�	��k���A�7*;


total_loss5��@

error_R��V?

learning_rate_1�h�5��I       6%�	T1k���A�7*;


total_loss���@

error_RL�I?

learning_rate_1�h�5uѣ�I       6%�	F{k���A�7*;


total_lossC�@

error_R�B?

learning_rate_1�h�5��U�I       6%�	��k���A�7*;


total_loss���@

error_R��K?

learning_rate_1�h�5��I I       6%�	�k���A�7*;


total_loss�T�@

error_R��Y?

learning_rate_1�h�5D2��I       6%�	Hrk���A�7*;


total_loss;7�@

error_RÿW?

learning_rate_1�h�5��I       6%�	p�k���A�7*;


total_loss�O�@

error_R�D?

learning_rate_1�h�5Zv~�I       6%�	�#k���A�7*;


total_loss�t�@

error_R��-?

learning_rate_1�h�5��+I       6%�	�ik���A�7*;


total_loss���@

error_Rq1C?

learning_rate_1�h�5�S5I       6%�	*�k���A�7*;


total_loss�@

error_R$�L?

learning_rate_1�h�5��WI       6%�	[�k���A�7*;


total_loss�h�@

error_R,�N?

learning_rate_1�h�5�K nI       6%�	2 k���A�7*;


total_loss��A

error_R�T?

learning_rate_1�h�5Pd0�I       6%�	�x k���A�8*;


total_loss��@

error_R�pR?

learning_rate_1�h�5����I       6%�	� k���A�8*;


total_loss!��@

error_R��O?

learning_rate_1�h�5d�^	I       6%�	�!k���A�8*;


total_loss�h�@

error_RM�Z?

learning_rate_1�h�5�1I       6%�	&P!k���A�8*;


total_loss�A

error_R��>?

learning_rate_1�h�5^��I       6%�	��!k���A�8*;


total_loss(��@

error_R�	N?

learning_rate_1�h�5w��wI       6%�	Z�!k���A�8*;


total_loss�8�@

error_RԈE?

learning_rate_1�h�5�;��I       6%�	A"k���A�8*;


total_lossĴ�@

error_R��I?

learning_rate_1�h�5���I       6%�	��"k���A�8*;


total_loss\��@

error_R��G?

learning_rate_1�h�5y��$I       6%�	x�"k���A�8*;


total_loss��@

error_R6�F?

learning_rate_1�h�5��g^I       6%�	O6#k���A�8*;


total_lossv��@

error_R��d?

learning_rate_1�h�5��K=I       6%�	*�#k���A�8*;


total_loss��@

error_R�??

learning_rate_1�h�5��.�I       6%�	��#k���A�8*;


total_lossĻ�@

error_RHb??

learning_rate_1�h�5��BI       6%�	q$k���A�8*;


total_loss��A

error_R�aQ?

learning_rate_1�h�5���I       6%�	�e$k���A�8*;


total_loss�)�@

error_R�%U?

learning_rate_1�h�5���	I       6%�	-�$k���A�8*;


total_loss���@

error_Ri�T?

learning_rate_1�h�5��ctI       6%�	��$k���A�8*;


total_loss�<�@

error_R��R?

learning_rate_1�h�5!҅�I       6%�	?F%k���A�8*;


total_lossA�@

error_R#JC?

learning_rate_1�h�5��mI       6%�	8�%k���A�8*;


total_lossR��@

error_R�[?

learning_rate_1�h�5�dI       6%�	S�%k���A�8*;


total_loss�u�@

error_R��W?

learning_rate_1�h�5a)��I       6%�	�'&k���A�8*;


total_lossD��@

error_R�r]?

learning_rate_1�h�5@�I       6%�	Gp&k���A�8*;


total_loss
X�@

error_R�??

learning_rate_1�h�5EZb�I       6%�	Ź&k���A�8*;


total_lossꛗ@

error_R]�R?

learning_rate_1�h�5���I       6%�	(�&k���A�8*;


total_lossԏ�@

error_R:�Q?

learning_rate_1�h�5&V�I       6%�	IF'k���A�8*;


total_loss��@

error_R�p]?

learning_rate_1�h�5�rI       6%�	�'k���A�8*;


total_loss�{�@

error_R�jC?

learning_rate_1�h�5�c}�I       6%�	�'k���A�8*;


total_lossŎ@

error_R�O?

learning_rate_1�h�5}r��I       6%�	�(k���A�8*;


total_loss�f@

error_R�mN?

learning_rate_1�h�5lP�^I       6%�	q_(k���A�8*;


total_loss��@

error_R�%B?

learning_rate_1�h�5�!8I       6%�	h�(k���A�8*;


total_lossͦ�@

error_R��=?

learning_rate_1�h�5�^/+I       6%�	��(k���A�8*;


total_loss�@

error_R�FT?

learning_rate_1�h�5n��I       6%�	�,)k���A�8*;


total_loss��@

error_R��V?

learning_rate_1�h�5��I       6%�	Cr)k���A�8*;


total_loss�@

error_R.cE?

learning_rate_1�h�5ntٍI       6%�	-�)k���A�8*;


total_loss���@

error_R��U?

learning_rate_1�h�5)�fI       6%�	B�)k���A�8*;


total_loss�"`@

error_RӯG?

learning_rate_1�h�5��p'I       6%�	�=*k���A�8*;


total_loss��@

error_R�W?

learning_rate_1�h�5��mI       6%�	�*k���A�8*;


total_loss�]�@

error_R�>L?

learning_rate_1�h�5��j�I       6%�	Y�*k���A�8*;


total_loss�@

error_R�6?

learning_rate_1�h�5&�5NI       6%�	�	+k���A�8*;


total_loss���@

error_R�\9?

learning_rate_1�h�5�bXI       6%�	�M+k���A�8*;


total_lossl��@

error_R�
F?

learning_rate_1�h�5&��LI       6%�	l�+k���A�8*;


total_loss.��@

error_R\Y?

learning_rate_1�h�5��6(I       6%�	��+k���A�8*;


total_lossw�@

error_RLB7?

learning_rate_1�h�5\8�I       6%�	�,k���A�8*;


total_loss�m@

error_R� Z?

learning_rate_1�h�5m�W*I       6%�	c,k���A�8*;


total_loss1م@

error_RҤU?

learning_rate_1�h�5�+�!I       6%�	�,k���A�8*;


total_loss*��@

error_RtOK?

learning_rate_1�h�5-|��I       6%�	<�,k���A�8*;


total_losst�@

error_Rj�J?

learning_rate_1�h�5�?ƬI       6%�	t.-k���A�8*;


total_lossf��@

error_R�G?

learning_rate_1�h�5�^�I       6%�	�p-k���A�8*;


total_loss�@

error_R�W?

learning_rate_1�h�5�C��I       6%�	��-k���A�8*;


total_loss�P�@

error_R,�9?

learning_rate_1�h�5!bYI       6%�	�-k���A�8*;


total_loss���@

error_RD�N?

learning_rate_1�h�5��>I       6%�	q^.k���A�8*;


total_loss���@

error_RZZ?

learning_rate_1�h�5�GeI       6%�	�.k���A�8*;


total_loss	��@

error_R�\R?

learning_rate_1�h�5m�aI       6%�	*�.k���A�8*;


total_loss� �@

error_R_L?

learning_rate_1�h�5Q�/I       6%�	�=/k���A�8*;


total_lossO��@

error_R�K?

learning_rate_1�h�5�R�(I       6%�	��/k���A�8*;


total_loss*i�@

error_R�)U?

learning_rate_1�h�5{�I       6%�	��/k���A�8*;


total_loss���@

error_R�P?

learning_rate_1�h�59g6�I       6%�	� 0k���A�8*;


total_lossa��@

error_R�Q?

learning_rate_1�h�5�E5TI       6%�	�j0k���A�8*;


total_lossƁ�@

error_R��4?

learning_rate_1�h�5�7�[I       6%�	\�0k���A�8*;


total_lossE��@

error_R��U?

learning_rate_1�h�5��*I       6%�	��0k���A�8*;


total_lossߥ�@

error_Rn|7?

learning_rate_1�h�5��=I       6%�	q71k���A�8*;


total_lossF�@

error_RvP?

learning_rate_1�h�5y$��I       6%�	B1k���A�8*;


total_loss�r�@

error_R�(X?

learning_rate_1�h�5\� I       6%�	J�1k���A�8*;


total_lossb�@

error_R�D?

learning_rate_1�h�5�y=I       6%�	�2k���A�8*;


total_loss���@

error_R�C?

learning_rate_1�h�5�NTNI       6%�	 a2k���A�8*;


total_lossE��@

error_R�]?

learning_rate_1�h�5M��9I       6%�	�2k���A�8*;


total_loss}��@

error_R6s??

learning_rate_1�h�5:�.NI       6%�	�2k���A�8*;


total_loss3	�@

error_Rz�C?

learning_rate_1�h�5�G�I       6%�	�<3k���A�8*;


total_lossE��@

error_R��9?

learning_rate_1�h�5�j�I       6%�	̃3k���A�8*;


total_loss���@

error_R�\S?

learning_rate_1�h�5�\�I       6%�	�3k���A�8*;


total_loss|��@

error_R�@?

learning_rate_1�h�5e�`�I       6%�	S4k���A�8*;


total_loss3M�@

error_R(yQ?

learning_rate_1�h�5�\�zI       6%�	�S4k���A�8*;


total_loss���@

error_R�uT?

learning_rate_1�h�5��2�I       6%�	ۖ4k���A�8*;


total_loss���@

error_R��R?

learning_rate_1�h�5[աI       6%�	�4k���A�8*;


total_loss�,�@

error_R� 6?

learning_rate_1�h�5�H}�I       6%�	�5k���A�8*;


total_loss�l�@

error_R�kI?

learning_rate_1�h�5<�;VI       6%�	�d5k���A�8*;


total_loss,�@

error_R�>?

learning_rate_1�h�5_�<�I       6%�	��5k���A�8*;


total_loss3��@

error_R(*R?

learning_rate_1�h�5���I       6%�	J�5k���A�8*;


total_losswN�@

error_RO�Z?

learning_rate_1�h�5~zTI       6%�	�86k���A�8*;


total_losse��@

error_R�BQ?

learning_rate_1�h�5ҭ�I       6%�	{|6k���A�8*;


total_lossMʞ@

error_RhbZ?

learning_rate_1�h�5֏SnI       6%�	��6k���A�8*;


total_loss&q�@

error_R��b?

learning_rate_1�h�5��M�I       6%�	� 7k���A�8*;


total_lossF��@

error_RW
^?

learning_rate_1�h�5S�%I       6%�	�A7k���A�8*;


total_lossTB�@

error_Rn�K?

learning_rate_1�h�5�wdLI       6%�	��7k���A�8*;


total_loss���@

error_R3 S?

learning_rate_1�h�5�8��I       6%�	��7k���A�8*;


total_loss$�@

error_RͰN?

learning_rate_1�h�55�/I       6%�	
8k���A�8*;


total_loss��s@

error_R@�[?

learning_rate_1�h�54<�I       6%�	iO8k���A�8*;


total_loss�Y�@

error_R|TW?

learning_rate_1�h�5�,��I       6%�	}�8k���A�8*;


total_loss�|�@

error_R,�J?

learning_rate_1�h�5K�I       6%�	J�8k���A�8*;


total_loss���@

error_R��D?

learning_rate_1�h�5�ؘ�I       6%�	A19k���A�8*;


total_loss:��@

error_Rn�K?

learning_rate_1�h�5��e�I       6%�	�9k���A�8*;


total_lossn��@

error_R@e=?

learning_rate_1�h�5���I       6%�	p:k���A�8*;


total_loss�*�@

error_R�(R?

learning_rate_1�h�5�HJ�I       6%�	)M:k���A�8*;


total_loss���@

error_R�.L?

learning_rate_1�h�5.���I       6%�	�:k���A�8*;


total_lossԍ�@

error_R܍??

learning_rate_1�h�5X@�I       6%�	j�:k���A�8*;


total_loss���@

error_R��:?

learning_rate_1�h�5b?Y1I       6%�	�1;k���A�8*;


total_loss�Y�@

error_R]nZ?

learning_rate_1�h�5I��fI       6%�	�|;k���A�8*;


total_loss�ֹ@

error_R�Q?

learning_rate_1�h�5H$��I       6%�	��;k���A�8*;


total_lossCP�@

error_R;I?

learning_rate_1�h�5���WI       6%�	�<k���A�8*;


total_loss��@

error_RQ�P?

learning_rate_1�h�56!s�I       6%�	$\<k���A�8*;


total_loss*%A

error_R��J?

learning_rate_1�h�5byO�I       6%�	
�<k���A�8*;


total_losstچ@

error_RɤQ?

learning_rate_1�h�5:�.I       6%�	��<k���A�8*;


total_loss���@

error_R�#T?

learning_rate_1�h�5wm��I       6%�	/=k���A�8*;


total_loss~J�@

error_R�S?

learning_rate_1�h�5��I       6%�	 v=k���A�8*;


total_loss#��@

error_R�XW?

learning_rate_1�h�5���xI       6%�	v�=k���A�8*;


total_loss�ٰ@

error_R*�U?

learning_rate_1�h�53�$$I       6%�	��=k���A�8*;


total_loss�,�@

error_R��\?

learning_rate_1�h�5�I       6%�	J`>k���A�8*;


total_loss�9�@

error_RvbJ?

learning_rate_1�h�5���I       6%�	��>k���A�8*;


total_lossNu�@

error_R}�Z?

learning_rate_1�h�5�"dI       6%�	s�>k���A�8*;


total_loss���@

error_R��U?

learning_rate_1�h�5�=��I       6%�	�@?k���A�8*;


total_loss���@

error_R%K?

learning_rate_1�h�5��vI       6%�	=�?k���A�8*;


total_loss�)�@

error_R�KR?

learning_rate_1�h�5���
I       6%�	��?k���A�8*;


total_lossa�@

error_R�<M?

learning_rate_1�h�5��"NI       6%�	�@k���A�8*;


total_loss���@

error_R��Y?

learning_rate_1�h�5(d�I       6%�	_@k���A�8*;


total_loss؈@

error_RJ�P?

learning_rate_1�h�5�rDI       6%�	k�@k���A�8*;


total_lossI��@

error_Rq�S?

learning_rate_1�h�5
��I       6%�	w�@k���A�8*;


total_loss}�@

error_R��X?

learning_rate_1�h�5�$��I       6%�	%9Ak���A�8*;


total_loss��@

error_R��^?

learning_rate_1�h�5b�>I       6%�	�{Ak���A�8*;


total_loss=ݪ@

error_R�A?

learning_rate_1�h�5Q��NI       6%�	��Ak���A�8*;


total_loss�q�@

error_R��O?

learning_rate_1�h�5�D�-I       6%�	9Bk���A�8*;


total_lossC,�@

error_RZ�a?

learning_rate_1�h�5U�[I       6%�	bFBk���A�8*;


total_loss��@

error_ReU?

learning_rate_1�h�5��i�I       6%�	ƓBk���A�8*;


total_loss��@

error_RI?

learning_rate_1�h�5�VL�I       6%�	`Ck���A�8*;


total_lossL[A

error_RWE?

learning_rate_1�h�5���/I       6%�	)HCk���A�8*;


total_lossTv�@

error_R)�`?

learning_rate_1�h�5��I       6%�	��Ck���A�8*;


total_loss ��@

error_R�??

learning_rate_1�h�5W��XI       6%�	��Ck���A�8*;


total_lossv��@

error_R��N?

learning_rate_1�h�5-V�I       6%�	�:Dk���A�8*;


total_loss��@

error_R�K?

learning_rate_1�h�5�`<I       6%�	ȀDk���A�8*;


total_loss_�@

error_R�T7?

learning_rate_1�h�5��m�I       6%�	N�Dk���A�8*;


total_loss��@

error_R;S?

learning_rate_1�h�5�0ˢI       6%�	�Ek���A�9*;


total_loss9�@

error_R2hI?

learning_rate_1�h�5��v�I       6%�	�VEk���A�9*;


total_loss�@

error_R�pX?

learning_rate_1�h�5u��I       6%�	b�Ek���A�9*;


total_loss���@

error_R��C?

learning_rate_1�h�5TF��I       6%�	��Ek���A�9*;


total_loss�sr@

error_RL�q?

learning_rate_1�h�5��QMI       6%�	�&Fk���A�9*;


total_loss���@

error_R��W?

learning_rate_1�h�5aI       6%�	�lFk���A�9*;


total_loss���@

error_RMfS?

learning_rate_1�h�5 �2II       6%�	��Fk���A�9*;


total_loss}� A

error_R4qX?

learning_rate_1�h�5�)��I       6%�	��Fk���A�9*;


total_loss��@

error_R1�U?

learning_rate_1�h�5��O�I       6%�	�DGk���A�9*;


total_loss�}�@

error_R3�M?

learning_rate_1�h�5��0I       6%�	9�Gk���A�9*;


total_lossc�@

error_R78M?

learning_rate_1�h�5Kx`%I       6%�	��Gk���A�9*;


total_loss���@

error_R��X?

learning_rate_1�h�5�PuHI       6%�	"Hk���A�9*;


total_loss�|�@

error_Rn�U?

learning_rate_1�h�5g��I       6%�	jHk���A�9*;


total_loss���@

error_R��X?

learning_rate_1�h�5����I       6%�	��Hk���A�9*;


total_loss���@

error_R�}U?

learning_rate_1�h�5��dI       6%�	��Hk���A�9*;


total_loss�[�@

error_R�cM?

learning_rate_1�h�5|��I       6%�	L9Ik���A�9*;


total_loss��@

error_R3�o?

learning_rate_1�h�5� �I       6%�	��Ik���A�9*;


total_loss���@

error_R�QR?

learning_rate_1�h�5����I       6%�	�Ik���A�9*;


total_loss� �@

error_R��??

learning_rate_1�h�5��HKI       6%�	%Jk���A�9*;


total_loss=�@

error_R�C?

learning_rate_1�h�5�$wI       6%�	�WJk���A�9*;


total_loss���@

error_R{+M?

learning_rate_1�h�5("�ZI       6%�	p�Jk���A�9*;


total_loss|��@

error_RJ�R?

learning_rate_1�h�5H�޲I       6%�	�Jk���A�9*;


total_loss䏑@

error_R �J?

learning_rate_1�h�5��I       6%�	$7Kk���A�9*;


total_loss O�@

error_R}G??

learning_rate_1�h�5/��I       6%�	g|Kk���A�9*;


total_lossq7�@

error_R�1A?

learning_rate_1�h�5Q7I       6%�	7�Kk���A�9*;


total_loss�3�@

error_R�I?

learning_rate_1�h�5;��eI       6%�	7Lk���A�9*;


total_loss;a@

error_R�B?

learning_rate_1�h�5�F�;I       6%�	�ZLk���A�9*;


total_loss|�@

error_R��??

learning_rate_1�h�5�Kc�I       6%�	>�Lk���A�9*;


total_lossf��@

error_R�5E?

learning_rate_1�h�5�!B7I       6%�	��Lk���A�9*;


total_lossK��@

error_Ri�U?

learning_rate_1�h�5mSc5I       6%�	�'Mk���A�9*;


total_loss��@

error_R�Y?

learning_rate_1�h�5���OI       6%�	mMk���A�9*;


total_loss���@

error_ReN?

learning_rate_1�h�5b^�I       6%�	��Mk���A�9*;


total_loss��@

error_R�pM?

learning_rate_1�h�5�O�I       6%�	��Mk���A�9*;


total_loss��@

error_Rh�U?

learning_rate_1�h�5���I       6%�	�^Nk���A�9*;


total_loss���@

error_R�~^?

learning_rate_1�h�5���I       6%�	x�Nk���A�9*;


total_loss8K@

error_R@�<?

learning_rate_1�h�5־_=I       6%�	Q�Nk���A�9*;


total_loss�͠@

error_R�J?

learning_rate_1�h�5G�D�I       6%�	 1Ok���A�9*;


total_loss�W�@

error_R0N?

learning_rate_1�h�5�G*4I       6%�	pyOk���A�9*;


total_loss3A

error_R�[G?

learning_rate_1�h�5��I       6%�	ѿOk���A�9*;


total_loss�h�@

error_R:[U?

learning_rate_1�h�5�wK�I       6%�	>Pk���A�9*;


total_loss�X�@

error_R�\?

learning_rate_1�h�5��N�I       6%�	�OPk���A�9*;


total_loss�&�@

error_R��^?

learning_rate_1�h�5e�w=I       6%�	$�Pk���A�9*;


total_loss9�@

error_R@�X?

learning_rate_1�h�5�˲�I       6%�	b�Pk���A�9*;


total_loss��@

error_R�vY?

learning_rate_1�h�5X�ִI       6%�	 Qk���A�9*;


total_loss���@

error_R��B?

learning_rate_1�h�5����I       6%�	�cQk���A�9*;


total_lossb A

error_R��P?

learning_rate_1�h�5�Bz�I       6%�	m�Qk���A�9*;


total_lossf��@

error_R��S?

learning_rate_1�h�5lܔ�I       6%�	8�Qk���A�9*;


total_loss�3-A

error_R�??

learning_rate_1�h�5ķ]I       6%�	�2Rk���A�9*;


total_loss��@

error_R�%9?

learning_rate_1�h�5�=�I       6%�	�wRk���A�9*;


total_loss��A

error_RN�E?

learning_rate_1�h�5(֭I       6%�	_�Rk���A�9*;


total_loss[�@

error_Rd9H?

learning_rate_1�h�5�e1sI       6%�	_Sk���A�9*;


total_lossi]�@

error_R	�;?

learning_rate_1�h�5#	/�I       6%�	�YSk���A�9*;


total_loss3\�@

error_R.aB?

learning_rate_1�h�5�^6I       6%�	�Sk���A�9*;


total_loss֙�@

error_R��I?

learning_rate_1�h�5]pB-I       6%�	W�Sk���A�9*;


total_loss�ܢ@

error_R�|K?

learning_rate_1�h�5�츀I       6%�	EKTk���A�9*;


total_loss齵@

error_RW�O?

learning_rate_1�h�5�yZI       6%�	~�Tk���A�9*;


total_loss��@

error_R�.<?

learning_rate_1�h�5��I       6%�	�Uk���A�9*;


total_loss�Yu@

error_R�N?

learning_rate_1�h�5��x�I       6%�	s�Uk���A�9*;


total_loss��@

error_R��??

learning_rate_1�h�5���I       6%�	#�Uk���A�9*;


total_loss#T�@

error_R�f?

learning_rate_1�h�5A��I       6%�	;Vk���A�9*;


total_loss�D�@

error_R�G?

learning_rate_1�h�5��I       6%�	��Vk���A�9*;


total_lossT�@

error_R-�]?

learning_rate_1�h�5~}�I       6%�	 �Vk���A�9*;


total_loss1A

error_R�qE?

learning_rate_1�h�5s�V�I       6%�	"Wk���A�9*;


total_lossϻ�@

error_R��F?

learning_rate_1�h�5��[I       6%�	�wWk���A�9*;


total_loss/�@

error_Rv�M?

learning_rate_1�h�5�3��I       6%�	��Wk���A�9*;


total_lossI��@

error_RE�R?

learning_rate_1�h�5^ۣ%I       6%�	�9Xk���A�9*;


total_loss���@

error_R��]?

learning_rate_1�h�5�x</I       6%�	ЃXk���A�9*;


total_loss���@

error_R}�Z?

learning_rate_1�h�5���NI       6%�	o�Xk���A�9*;


total_loss���@

error_R3?U?

learning_rate_1�h�5R W�I       6%�	�(Yk���A�9*;


total_lossWo�@

error_RVg?

learning_rate_1�h�5e�Z�I       6%�	GoYk���A�9*;


total_loss���@

error_R-[?

learning_rate_1�h�5����I       6%�	ϹYk���A�9*;


total_loss���@

error_R�lH?

learning_rate_1�h�5����I       6%�	� Zk���A�9*;


total_loss�I^@

error_RcV7?

learning_rate_1�h�5��3�I       6%�	�fZk���A�9*;


total_loss��@

error_R<V?

learning_rate_1�h�5~�C4I       6%�	�Zk���A�9*;


total_loss?�@

error_R@�d?

learning_rate_1�h�5��zI       6%�	��Zk���A�9*;


total_loss���@

error_Rw<Y?

learning_rate_1�h�5�}>�I       6%�	GO[k���A�9*;


total_loss�8�@

error_R��;?

learning_rate_1�h�5�>yI       6%�	�[k���A�9*;


total_loss}J�@

error_R�'[?

learning_rate_1�h�5*'ѤI       6%�	��[k���A�9*;


total_loss���@

error_RR�U?

learning_rate_1�h�5_I       6%�	�,\k���A�9*;


total_loss�)�@

error_R��??

learning_rate_1�h�5j��I       6%�	�\k���A�9*;


total_loss)��@

error_R�D?

learning_rate_1�h�5Ǭ�I       6%�	"�\k���A�9*;


total_loss�n�@

error_R�G?

learning_rate_1�h�5�,CI       6%�	w]k���A�9*;


total_loss\E�@

error_R��K?

learning_rate_1�h�5��I       6%�	�d]k���A�9*;


total_lossa7�@

error_R�1N?

learning_rate_1�h�5��6'I       6%�	��]k���A�9*;


total_loss��@

error_Rj|W?

learning_rate_1�h�5\�Q�I       6%�	t^k���A�9*;


total_loss<i�@

error_R�JP?

learning_rate_1�h�5P��I       6%�	π^k���A�9*;


total_loss�+�@

error_R��P?

learning_rate_1�h�5�'['I       6%�	��^k���A�9*;


total_loss��@

error_R�I?

learning_rate_1�h�5��s�I       6%�	�_k���A�9*;


total_loss�~@

error_R�yS?

learning_rate_1�h�5<"�I       6%�	Y|_k���A�9*;


total_loss��@

error_RrG?

learning_rate_1�h�5�NtI       6%�	��_k���A�9*;


total_lossZ4�@

error_R�sE?

learning_rate_1�h�5��=:I       6%�		`k���A�9*;


total_loss��@

error_R_�=?

learning_rate_1�h�5�te:I       6%�	�U`k���A�9*;


total_loss���@

error_R.]_?

learning_rate_1�h�5����I       6%�	��`k���A�9*;


total_loss��@

error_R�l8?

learning_rate_1�h�5"*B&I       6%�	��`k���A�9*;


total_loss�@

error_R�
I?

learning_rate_1�h�5@��I       6%�	&+ak���A�9*;


total_lossF��@

error_R��M?

learning_rate_1�h�5bs�I       6%�	�mak���A�9*;


total_loss���@

error_R�
Y?

learning_rate_1�h�5W��I       6%�	J�ak���A�9*;


total_loss���@

error_R��U?

learning_rate_1�h�5���I       6%�	��ak���A�9*;


total_loss��y@

error_R��O?

learning_rate_1�h�5�JjI       6%�	�;bk���A�9*;


total_losst<�@

error_R=�K?

learning_rate_1�h�5���<I       6%�	�bk���A�9*;


total_loss&�@

error_R��N?

learning_rate_1�h�5�N�I       6%�	��bk���A�9*;


total_lossOX}@

error_R�V]?

learning_rate_1�h�5>1�I       6%�	�	ck���A�9*;


total_loss���@

error_R1^K?

learning_rate_1�h�5մ�hI       6%�	qKck���A�9*;


total_lossq3�@

error_Rn_?

learning_rate_1�h�5ܼ�I       6%�	b�ck���A�9*;


total_loss<C�@

error_R]L?

learning_rate_1�h�5F�I       6%�	H�ck���A�9*;


total_lossd��@

error_R11J?

learning_rate_1�h�5-l�I       6%�	ydk���A�9*;


total_losscR�@

error_RT�E?

learning_rate_1�h�5��B�I       6%�	�[dk���A�9*;


total_loss#��@

error_R�P?

learning_rate_1�h�5x�$�I       6%�	��dk���A�9*;


total_lossNl�@

error_R�J?

learning_rate_1�h�5�r��I       6%�	��dk���A�9*;


total_lossEŗ@

error_R�5H?

learning_rate_1�h�5�EI       6%�	b%ek���A�9*;


total_lossS	A

error_R׳F?

learning_rate_1�h�5&��I       6%�	�gek���A�9*;


total_loss<��@

error_RRcA?

learning_rate_1�h�5╡�I       6%�	u�ek���A�9*;


total_lossA

error_Rd-B?

learning_rate_1�h�5/�hMI       6%�	e�ek���A�9*;


total_loss�C�@

error_Rָ;?

learning_rate_1�h�5�;�BI       6%�	�2fk���A�9*;


total_loss{��@

error_R��M?

learning_rate_1�h�5��C�I       6%�	�vfk���A�9*;


total_loss�X�@

error_R|�Y?

learning_rate_1�h�5�:��I       6%�	:�fk���A�9*;


total_loss�y�@

error_R��@?

learning_rate_1�h�5ϱ�I       6%�	�fk���A�9*;


total_loss��@

error_R�+[?

learning_rate_1�h�5Qa^I       6%�	�Dgk���A�9*;


total_loss�a�@

error_R��^?

learning_rate_1�h�5r�5QI       6%�	L�gk���A�9*;


total_lossH#t@

error_Ra�N?

learning_rate_1�h�52�`8I       6%�	��gk���A�9*;


total_loss�@

error_R�lG?

learning_rate_1�h�51�^~I       6%�	ehk���A�9*;


total_loss���@

error_R��`?

learning_rate_1�h�5 C�RI       6%�	*Yhk���A�9*;


total_loss.��@

error_R�7?

learning_rate_1�h�5r!��I       6%�	��hk���A�9*;


total_loss�_�@

error_R�S?

learning_rate_1�h�54�I       6%�	��hk���A�9*;


total_loss�&�@

error_Rn;\?

learning_rate_1�h�5���I       6%�	�,ik���A�9*;


total_loss�@A

error_ROCN?

learning_rate_1�h�5@�I       6%�	_xik���A�9*;


total_loss���@

error_RW�V?

learning_rate_1�h�5�KI       6%�	T�ik���A�9*;


total_loss���@

error_R\}k?

learning_rate_1�h�5��"�I       6%�	Mjk���A�9*;


total_loss���@

error_R�W?

learning_rate_1�h�5�J�I       6%�	�mjk���A�:*;


total_loss9�@

error_R��=?

learning_rate_1�h�5u���I       6%�	�jk���A�:*;


total_lossi[u@

error_R��<?

learning_rate_1�h�5�R��I       6%�	��jk���A�:*;


total_loss)��@

error_R��S?

learning_rate_1�h�5za�I       6%�	�Dkk���A�:*;


total_loss�(�@

error_R�MF?

learning_rate_1�h�5�/��I       6%�	��kk���A�:*;


total_loss%Y	A

error_R�F?

learning_rate_1�h�5��(6I       6%�	��kk���A�:*;


total_lossH��@

error_R�c\?

learning_rate_1�h�5�e�I       6%�	�lk���A�:*;


total_lossL�@

error_RmD?

learning_rate_1�h�5X�YI       6%�	�^lk���A�:*;


total_lossZS�@

error_R�mE?

learning_rate_1�h�59��I       6%�	��lk���A�:*;


total_loss'�@

error_R_m;?

learning_rate_1�h�5��I       6%�	e�lk���A�:*;


total_lossN��@

error_R_�C?

learning_rate_1�h�5���I       6%�	5*mk���A�:*;


total_lossҞ�@

error_R�p[?

learning_rate_1�h�5��[I       6%�		pmk���A�:*;


total_losslz�@

error_R�S?

learning_rate_1�h�5ɑ��I       6%�	q�mk���A�:*;


total_loss��@

error_R�WL?

learning_rate_1�h�5v[I       6%�	��mk���A�:*;


total_loss��A

error_R��M?

learning_rate_1�h�5}]:�I       6%�	idnk���A�:*;


total_loss�ɥ@

error_RZ1K?

learning_rate_1�h�5�.�BI       6%�	C�nk���A�:*;


total_loss���@

error_R��I?

learning_rate_1�h�5fC�I       6%�	�nk���A�:*;


total_loss���@

error_R]�N?

learning_rate_1�h�5�z�I       6%�	+6ok���A�:*;


total_lossa��@

error_Rc�V?

learning_rate_1�h�5�F��I       6%�	�yok���A�:*;


total_loss`˲@

error_Rn�J?

learning_rate_1�h�5�Bg�I       6%�	?�ok���A�:*;


total_loss[j(A

error_R��Y?

learning_rate_1�h�5ǣ�1I       6%�	pk���A�:*;


total_lossS/�@

error_Rv�L?

learning_rate_1�h�5#��I       6%�	�Dpk���A�:*;


total_loss���@

error_R�a?

learning_rate_1�h�5��I       6%�	X�pk���A�:*;


total_loss2��@

error_R�ZJ?

learning_rate_1�h�5���%I       6%�	��pk���A�:*;


total_loss�Ū@

error_R|�J?

learning_rate_1�h�5��I       6%�	wqk���A�:*;


total_losstҍ@

error_R�JR?

learning_rate_1�h�5A��dI       6%�	�\qk���A�:*;


total_lossJ�w@

error_R�TB?

learning_rate_1�h�5��??I       6%�	��qk���A�:*;


total_loss)��@

error_R��L?

learning_rate_1�h�5��*I       6%�	��qk���A�:*;


total_loss$��@

error_R�pW?

learning_rate_1�h�5)��hI       6%�	�+rk���A�:*;


total_loss���@

error_R�r@?

learning_rate_1�h�5�xHI       6%�	�trk���A�:*;


total_loss3�@

error_R�I?

learning_rate_1�h�5�N�*I       6%�	��rk���A�:*;


total_loss]8A

error_RG?

learning_rate_1�h�5 O�I       6%�	sk���A�:*;


total_loss/�A

error_R�U?

learning_rate_1�h�5t�>�I       6%�	LOsk���A�:*;


total_loss�	�@

error_Ro=?

learning_rate_1�h�5;��I       6%�	v�sk���A�:*;


total_loss���@

error_R$�Q?

learning_rate_1�h�5�$A�I       6%�	��sk���A�:*;


total_loss.�@

error_R�`?

learning_rate_1�h�5�)P�I       6%�	� tk���A�:*;


total_loss\�@

error_R��Y?

learning_rate_1�h�5e��xI       6%�	^atk���A�:*;


total_lossj��@

error_R��Q?

learning_rate_1�h�5�
�I       6%�	��tk���A�:*;


total_loss���@

error_RC�O?

learning_rate_1�h�5�z��I       6%�	��tk���A�:*;


total_loss��@

error_R=[?

learning_rate_1�h�5�~�I       6%�	�)uk���A�:*;


total_loss쓊@

error_R��P?

learning_rate_1�h�5RsLMI       6%�	-ouk���A�:*;


total_loss!�@

error_R�GM?

learning_rate_1�h�5�;�I       6%�	��uk���A�:*;


total_loss���@

error_R�mL?

learning_rate_1�h�5�`d'I       6%�	~�uk���A�:*;


total_loss%�@

error_RS�T?

learning_rate_1�h�5?�!�I       6%�	A:vk���A�:*;


total_lossJ1�@

error_R�Z?

learning_rate_1�h�5R�9�I       6%�	a}vk���A�:*;


total_loss<�@

error_Rl4?

learning_rate_1�h�53@�I       6%�	x�vk���A�:*;


total_lossY��@

error_R��Q?

learning_rate_1�h�5۩�zI       6%�	�wk���A�:*;


total_loss]��@

error_R��N?

learning_rate_1�h�5]�D�I       6%�	Gwk���A�:*;


total_loss�w@

error_R��J?

learning_rate_1�h�5�nv�I       6%�	��wk���A�:*;


total_lossCc�@

error_R)�L?

learning_rate_1�h�5�AI       6%�	��wk���A�:*;


total_loss�5�@

error_RJ�^?

learning_rate_1�h�5���bI       6%�	
xk���A�:*;


total_loss՟@

error_R��I?

learning_rate_1�h�5�I�*I       6%�	�]xk���A�:*;


total_loss�p�@

error_R�k??

learning_rate_1�h�5�G�I       6%�	��xk���A�:*;


total_loss��@

error_R�:P?

learning_rate_1�h�5g�jI       6%�	o�xk���A�:*;


total_losswzg@

error_R,!A?

learning_rate_1�h�5�	I       6%�	$.yk���A�:*;


total_loss���@

error_R�S?

learning_rate_1�h�5���I       6%�	=uyk���A�:*;


total_loss�:�@

error_RxQ?

learning_rate_1�h�5k��I       6%�	�yk���A�:*;


total_loss�ۘ@

error_R�oD?

learning_rate_1�h�5���I       6%�	Q�yk���A�:*;


total_losso A

error_R�RY?

learning_rate_1�h�5J�wI       6%�	TCzk���A�:*;


total_loss?��@

error_R�ZL?

learning_rate_1�h�5�/\�I       6%�	��zk���A�:*;


total_loss�~@

error_R�nH?

learning_rate_1�h�5X)O&I       6%�	��zk���A�:*;


total_loss�ޝ@

error_RO,_?

learning_rate_1�h�5���mI       6%�	�{k���A�:*;


total_loss���@

error_R@�T?

learning_rate_1�h�56��I       6%�	8T{k���A�:*;


total_losse�@

error_RChS?

learning_rate_1�h�5b2�I       6%�	��{k���A�:*;


total_loss ޼@

error_R��>?

learning_rate_1�h�5�.^�I       6%�	��{k���A�:*;


total_loss(��@

error_R�-5?

learning_rate_1�h�5�7��I       6%�	�|k���A�:*;


total_loss̐@

error_RL�J?

learning_rate_1�h�5�8.�I       6%�	�b|k���A�:*;


total_loss+�@

error_R�Z?

learning_rate_1�h�5�9)�I       6%�	��|k���A�:*;


total_lossf�@

error_R�0e?

learning_rate_1�h�5S���I       6%�	v�|k���A�:*;


total_loss�?�@

error_Rn�L?

learning_rate_1�h�5>��jI       6%�	�N}k���A�:*;


total_lossS}�@

error_R�SJ?

learning_rate_1�h�53̸I       6%�	��}k���A�:*;


total_lossJT�@

error_R��G?

learning_rate_1�h�5�2�I       6%�	��}k���A�:*;


total_loss.��@

error_R�G?

learning_rate_1�h�5���I       6%�	�)~k���A�:*;


total_loss*�@

error_R�mH?

learning_rate_1�h�5l���I       6%�	o�~k���A�:*;


total_loss8r!A

error_R��K?

learning_rate_1�h�5���I       6%�	�~k���A�:*;


total_loss�A

error_R�Z?

learning_rate_1�h�5���CI       6%�	&k���A�:*;


total_loss��@

error_Rf�Z?

learning_rate_1�h�5��cI       6%�	�mk���A�:*;


total_loss�@

error_Rq<G?

learning_rate_1�h�5~��I       6%�	�k���A�:*;


total_loss���@

error_R.@>?

learning_rate_1�h�5�3�I       6%�	[�k���A�:*;


total_lossCH�@

error_R{F?

learning_rate_1�h�5N�fI       6%�	J?�k���A�:*;


total_loss$ܗ@

error_RJ�K?

learning_rate_1�h�5�F+I       6%�	Ԇ�k���A�:*;


total_loss���@

error_R8S?

learning_rate_1�h�5.M�I       6%�	fπk���A�:*;


total_loss7��@

error_R��L?

learning_rate_1�h�5h�I       6%�	��k���A�:*;


total_lossH��@

error_R��N?

learning_rate_1�h�5X�mI       6%�	�a�k���A�:*;


total_loss��@

error_R��K?

learning_rate_1�h�5o;ZI       6%�	O��k���A�:*;


total_lossOa�@

error_R�1H?

learning_rate_1�h�5.��I       6%�	}��k���A�:*;


total_loss��@

error_R�kR?

learning_rate_1�h�5z��<I       6%�	�=�k���A�:*;


total_loss��@

error_R8�W?

learning_rate_1�h�560�xI       6%�	3��k���A�:*;


total_loss?)�@

error_R��>?

learning_rate_1�h�5��u+I       6%�	�̂k���A�:*;


total_loss,'A

error_R�??

learning_rate_1�h�5��O�I       6%�	I�k���A�:*;


total_loss� {@

error_R=�G?

learning_rate_1�h�5ڧ4�I       6%�	aT�k���A�:*;


total_loss���@

error_RͭH?

learning_rate_1�h�5kv�I       6%�	���k���A�:*;


total_loss�A

error_R�xD?

learning_rate_1�h�5����I       6%�	��k���A�:*;


total_loss��@

error_R�	P?

learning_rate_1�h�5R%UI       6%�	�&�k���A�:*;


total_loss=�w@

error_R{�J?

learning_rate_1�h�5���I       6%�	Cm�k���A�:*;


total_loss�h�@

error_R�zP?

learning_rate_1�h�5L�iI       6%�	���k���A�:*;


total_loss�*�@

error_R��O?

learning_rate_1�h�5z�vlI       6%�	��k���A�:*;


total_loss���@

error_R�Z>?

learning_rate_1�h�5�41�I       6%�	�F�k���A�:*;


total_loss�y@

error_R��M?

learning_rate_1�h�5�M,2I       6%�	E��k���A�:*;


total_loss���@

error_R%tL?

learning_rate_1�h�5w�`I       6%�	�ޅk���A�:*;


total_loss�fA

error_R��S?

learning_rate_1�h�5���I       6%�	�#�k���A�:*;


total_loss"�@

error_R��Y?

learning_rate_1�h�5�:��I       6%�	�e�k���A�:*;


total_loss�A

error_R�uR?

learning_rate_1�h�5���LI       6%�	Ʃ�k���A�:*;


total_loss=_�@

error_R��E?

learning_rate_1�h�5�s	I       6%�	��k���A�:*;


total_loss=�@

error_RJ�A?

learning_rate_1�h�5x�{I       6%�	V0�k���A�:*;


total_loss�#A

error_R�C?

learning_rate_1�h�5���I       6%�	�u�k���A�:*;


total_loss��@

error_R}�Q?

learning_rate_1�h�5�'72I       6%�	���k���A�:*;


total_lossE�@

error_R�XL?

learning_rate_1�h�5O��jI       6%�	z �k���A�:*;


total_loss��k@

error_R�oO?

learning_rate_1�h�5<�I       6%�	�C�k���A�:*;


total_loss|�@

error_R�U?

learning_rate_1�h�5�kb%I       6%�	���k���A�:*;


total_loss�\�@

error_R;BY?

learning_rate_1�h�5�D�I       6%�	�ˈk���A�:*;


total_loss���@

error_R�9@?

learning_rate_1�h�5q+I       6%�	�k���A�:*;


total_loss��@

error_R�yG?

learning_rate_1�h�5����I       6%�	R�k���A�:*;


total_loss%��@

error_R�fR?

learning_rate_1�h�5�T:I       6%�	є�k���A�:*;


total_loss-��@

error_R�W?

learning_rate_1�h�5hI�!I       6%�	 ىk���A�:*;


total_loss�R�@

error_R��[?

learning_rate_1�h�5��r�I       6%�	��k���A�:*;


total_lossi��@

error_R�&I?

learning_rate_1�h�5�x�+I       6%�	�b�k���A�:*;


total_lossλ�@

error_RsuN?

learning_rate_1�h�5<",I       6%�	���k���A�:*;


total_lossE��@

error_R�tN?

learning_rate_1�h�5��aI       6%�	i�k���A�:*;


total_loss(ך@

error_R��D?

learning_rate_1�h�5Ɏ�fI       6%�	�5�k���A�:*;


total_lossX�@

error_R|Q?

learning_rate_1�h�5*j)�I       6%�	_{�k���A�:*;


total_lossQϬ@

error_R�4N?

learning_rate_1�h�5i��I       6%�	��k���A�:*;


total_loss�~�@

error_R��^?

learning_rate_1�h�55�;�I       6%�	��k���A�:*;


total_loss���@

error_RܼN?

learning_rate_1�h�5u�EI       6%�	�J�k���A�:*;


total_loss��@

error_R��K?

learning_rate_1�h�5�I       6%�	%��k���A�:*;


total_loss%8�@

error_RHD?

learning_rate_1�h�5�� �I       6%�	}܌k���A�:*;


total_lossM�A

error_R=hQ?

learning_rate_1�h�5�O+�I       6%�	�(�k���A�:*;


total_loss���@

error_RF�E?

learning_rate_1�h�5U �I       6%�	-u�k���A�:*;


total_loss=��@

error_Rs/Q?

learning_rate_1�h�5Z���I       6%�	u��k���A�;*;


total_loss�Q�@

error_R�GN?

learning_rate_1�h�5�JuI       6%�	��k���A�;*;


total_lossB�@

error_Rl�B?

learning_rate_1�h�5L�LRI       6%�	�x�k���A�;*;


total_loss	+�@

error_R �T?

learning_rate_1�h�5��cI       6%�	�k���A�;*;


total_loss;��@

error_R�gV?

learning_rate_1�h�5���I       6%�	��k���A�;*;


total_loss`�@

error_RT1D?

learning_rate_1�h�5@�D�I       6%�	�U�k���A�;*;


total_loss<�@

error_RcO?

learning_rate_1�h�5ʊ�2I       6%�	r��k���A�;*;


total_loss:I�@

error_R״R?

learning_rate_1�h�5~���I       6%�	9�k���A�;*;


total_loss �_@

error_R�O?

learning_rate_1�h�5�OZ4I       6%�	�+�k���A�;*;


total_loss�Т@

error_R��B?

learning_rate_1�h�5੣I       6%�	�p�k���A�;*;


total_lossS`�@

error_RfuH?

learning_rate_1�h�5���I       6%�	0��k���A�;*;


total_loss �.A

error_Ra5a?

learning_rate_1�h�5Oi��I       6%�	���k���A�;*;


total_loss���@

error_REO?

learning_rate_1�h�5��I       6%�	+?�k���A�;*;


total_loss�pA

error_R�P?

learning_rate_1�h�5��a�I       6%�	���k���A�;*;


total_loss��@

error_R��\?

learning_rate_1�h�5���^I       6%�	�ɑk���A�;*;


total_loss�@

error_RvoN?

learning_rate_1�h�5#��I       6%�	~�k���A�;*;


total_loss��@

error_R\pQ?

learning_rate_1�h�5g�!BI       6%�	�U�k���A�;*;


total_loss��@

error_R�F?

learning_rate_1�h�5�'��I       6%�	ӛ�k���A�;*;


total_loss��@

error_RbR?

learning_rate_1�h�5[D��I       6%�	�k���A�;*;


total_loss��@

error_RhS?

learning_rate_1�h�5=�
HI       6%�	&�k���A�;*;


total_loss���@

error_R��L?

learning_rate_1�h�5&?iI       6%�	'm�k���A�;*;


total_loss�G�@

error_Rq:X?

learning_rate_1�h�5u<I       6%�	W��k���A�;*;


total_loss̪�@

error_R`]@?

learning_rate_1�h�5�*nI       6%�	y�k���A�;*;


total_loss���@

error_RF�H?

learning_rate_1�h�5��ZrI       6%�	�:�k���A�;*;


total_losso��@

error_R��E?

learning_rate_1�h�5*�=[I       6%�	���k���A�;*;


total_lossa�A

error_R�X?

learning_rate_1�h�5:�I       6%�	Kʔk���A�;*;


total_loss�^@

error_R.C?

learning_rate_1�h�5��I       6%�	9�k���A�;*;


total_loss=��@

error_R6AN?

learning_rate_1�h�59�ͨI       6%�	�h�k���A�;*;


total_loss�:�@

error_R�Y?

learning_rate_1�h�5p�I       6%�	���k���A�;*;


total_loss(�A

error_R��P?

learning_rate_1�h�5c�^�I       6%�	��k���A�;*;


total_loss�ܮ@

error_R{�t?

learning_rate_1�h�5�!��I       6%�	�5�k���A�;*;


total_loss� �@

error_Rz�G?

learning_rate_1�h�5+?��I       6%�	�y�k���A�;*;


total_loss�]�@

error_RRET?

learning_rate_1�h�5w��I       6%�	��k���A�;*;


total_lossH �@

error_R��[?

learning_rate_1�h�5�G6�I       6%�	#�k���A�;*;


total_loss�@�@

error_R3�W?

learning_rate_1�h�5�x��I       6%�	:M�k���A�;*;


total_loss#&�@

error_R�G?

learning_rate_1�h�5N�T�I       6%�	���k���A�;*;


total_lossl��@

error_REG?

learning_rate_1�h�5���I       6%�	�ߗk���A�;*;


total_loss*0A

error_R�Ta?

learning_rate_1�h�5.DՌI       6%�	�&�k���A�;*;


total_loss��/A

error_R��\?

learning_rate_1�h�5!'
I       6%�	xi�k���A�;*;


total_loss��@

error_R�PE?

learning_rate_1�h�5���I       6%�	���k���A�;*;


total_lossz��@

error_R�U?

learning_rate_1�h�5�R��I       6%�	Q��k���A�;*;


total_lossOo�@

error_R�VE?

learning_rate_1�h�5G\�9I       6%�	N>�k���A�;*;


total_loss���@

error_R(�J?

learning_rate_1�h�5ϒ>I       6%�	���k���A�;*;


total_loss�jA

error_Rx�J?

learning_rate_1�h�5�%I       6%�	�Ιk���A�;*;


total_loss-A�@

error_RF�N?

learning_rate_1�h�5�I       6%�	l�k���A�;*;


total_loss%�@

error_R��<?

learning_rate_1�h�5 /I       6%�	^�k���A�;*;


total_loss���@

error_Rt�]?

learning_rate_1�h�58v3�I       6%�	ס�k���A�;*;


total_loss�Z�@

error_Rn�G?

learning_rate_1�h�52I�!I       6%�	�k���A�;*;


total_lossH�@

error_R��H?

learning_rate_1�h�5�Vc5I       6%�	_0�k���A�;*;


total_loss[{�@

error_R��I?

learning_rate_1�h�5 ��I       6%�	Gs�k���A�;*;


total_loss���@

error_R2�c?

learning_rate_1�h�5)�6OI       6%�	���k���A�;*;


total_loss�[�@

error_R��<?

learning_rate_1�h�5�2�I       6%�	&�k���A�;*;


total_loss?�@

error_R_�H?

learning_rate_1�h�5MAkI       6%�	gL�k���A�;*;


total_loss��@

error_R1@8?

learning_rate_1�h�5J~�hI       6%�	p��k���A�;*;


total_lossH�@

error_RtH?

learning_rate_1�h�5��I       6%�	ۜk���A�;*;


total_loss�r�@

error_R?S?

learning_rate_1�h�5|<I       6%�	�"�k���A�;*;


total_lossά~@

error_R�7?

learning_rate_1�h�5�� �I       6%�	�l�k���A�;*;


total_loss��@

error_RyV?

learning_rate_1�h�5OF�I       6%�	ⷝk���A�;*;


total_loss1��@

error_RJ�a?

learning_rate_1�h�5r�.�I       6%�	Y��k���A�;*;


total_loss|Y�@

error_R�6W?

learning_rate_1�h�5Q3�I       6%�	iZ�k���A�;*;


total_loss��@

error_R`3]?

learning_rate_1�h�5��רI       6%�	k��k���A�;*;


total_loss3��@

error_R�H?

learning_rate_1�h�5H`I�I       6%�	%�k���A�;*;


total_loss|�@

error_R�nP?

learning_rate_1�h�5>��%I       6%�	W7�k���A�;*;


total_loss�`�@

error_R1RO?

learning_rate_1�h�5v�p�I       6%�	g�k���A�;*;


total_loss�H�@

error_R�WL?

learning_rate_1�h�5W��I       6%�	 ȟk���A�;*;


total_lossd'A

error_RHSE?

learning_rate_1�h�5�"I       6%�	��k���A�;*;


total_lossꪧ@

error_R_�`?

learning_rate_1�h�5$1�I       6%�	�[�k���A�;*;


total_loss1:�@

error_Rx2I?

learning_rate_1�h�5��OvI       6%�	���k���A�;*;


total_loss�I�@

error_R�G?

learning_rate_1�h�5a��I       6%�	��k���A�;*;


total_loss�Ǳ@

error_R"V?

learning_rate_1�h�5��XI       6%�	K'�k���A�;*;


total_loss�@

error_Rx�M?

learning_rate_1�h�5�^�yI       6%�	�i�k���A�;*;


total_loss�:�@

error_R��^?

learning_rate_1�h�5�XI       6%�	[��k���A�;*;


total_loss�T�@

error_R�R?

learning_rate_1�h�5-�xI       6%�	!�k���A�;*;


total_loss���@

error_R��8?

learning_rate_1�h�5�*��I       6%�	�6�k���A�;*;


total_loss��@

error_R�eH?

learning_rate_1�h�5�>�I       6%�	#{�k���A�;*;


total_lossvq�@

error_RiuK?

learning_rate_1�h�5B��I       6%�	#��k���A�;*;


total_lossH�@

error_R
ML?

learning_rate_1�h�5`y,_I       6%�	`��k���A�;*;


total_lossT��@

error_R7�C?

learning_rate_1�h�5�9�I       6%�	B�k���A�;*;


total_lossC�@

error_R�<a?

learning_rate_1�h�5說�I       6%�	Մ�k���A�;*;


total_loss���@

error_RV]??

learning_rate_1�h�5���VI       6%�	Oȣk���A�;*;


total_loss���@

error_R��P?

learning_rate_1�h�5��w�I       6%�	M
�k���A�;*;


total_loss��@

error_R&\?

learning_rate_1�h�5h�<,I       6%�	�R�k���A�;*;


total_loss�I�@

error_R��H?

learning_rate_1�h�5  �I       6%�	И�k���A�;*;


total_loss[�{@

error_RO K?

learning_rate_1�h�5?2�/I       6%�	��k���A�;*;


total_loss$��@

error_R�!K?

learning_rate_1�h�5��}�I       6%�	�-�k���A�;*;


total_loss���@

error_RQF?

learning_rate_1�h�5OC�2I       6%�	Fx�k���A�;*;


total_loss�T�@

error_R�SN?

learning_rate_1�h�5�aH'I       6%�	�åk���A�;*;


total_loss��@

error_R��V?

learning_rate_1�h�5�Qq�I       6%�	T�k���A�;*;


total_loss_L�@

error_R��C?

learning_rate_1�h�5�'R9I       6%�	�J�k���A�;*;


total_loss�@

error_R V?

learning_rate_1�h�5r"dI       6%�	���k���A�;*;


total_loss�
�@

error_R��H?

learning_rate_1�h�5�sK�I       6%�	�Ӧk���A�;*;


total_loss��@

error_R̧K?

learning_rate_1�h�5OmVI       6%�	��k���A�;*;


total_loss�`A

error_RYM?

learning_rate_1�h�5ovVI       6%�	�[�k���A�;*;


total_loss���@

error_R�{H?

learning_rate_1�h�5�� I       6%�	-��k���A�;*;


total_loss��A

error_R�%I?

learning_rate_1�h�5Dޟ�I       6%�	��k���A�;*;


total_loss+=�@

error_R�kS?

learning_rate_1�h�5��g�I       6%�	�$�k���A�;*;


total_loss���@

error_Rx�Z?

learning_rate_1�h�5<v��I       6%�	�h�k���A�;*;


total_loss��@

error_R�ZF?

learning_rate_1�h�5���I       6%�	���k���A�;*;


total_loss;3�@

error_R��J?

learning_rate_1�h�5�XY�I       6%�	s�k���A�;*;


total_loss2�R@

error_R��E?

learning_rate_1�h�5�6bI       6%�	n0�k���A�;*;


total_loss[��@

error_R
%P?

learning_rate_1�h�5�~e�I       6%�	�w�k���A�;*;


total_loss��@

error_R.�J?

learning_rate_1�h�5�1� I       6%�	+��k���A�;*;


total_loss[q�@

error_R֣Y?

learning_rate_1�h�5!��I       6%�	�k���A�;*;


total_loss���@

error_R8�_?

learning_rate_1�h�5=2y,I       6%�	NN�k���A�;*;


total_loss8��@

error_R�JH?

learning_rate_1�h�5}<�I       6%�	���k���A�;*;


total_loss#M�@

error_R��d?

learning_rate_1�h�5��RiI       6%�	bڪk���A�;*;


total_loss堇@

error_RژN?

learning_rate_1�h�5"��I       6%�	 �k���A�;*;


total_loss���@

error_R�=O?

learning_rate_1�h�5C�j�I       6%�	�d�k���A�;*;


total_lossC��@

error_RSVZ?

learning_rate_1�h�5!�I       6%�	ץ�k���A�;*;


total_loss��@

error_RC-Y?

learning_rate_1�h�5���I       6%�	��k���A�;*;


total_loss�+�@

error_R�7V?

learning_rate_1�h�5H��bI       6%�	f-�k���A�;*;


total_loss�s�@

error_R�`6?

learning_rate_1�h�5�@nmI       6%�	vr�k���A�;*;


total_loss/��@

error_R��L?

learning_rate_1�h�5m�#I       6%�	z��k���A�;*;


total_loss�@

error_R�J@?

learning_rate_1�h�5y)��I       6%�	���k���A�;*;


total_loss}l�@

error_R�M?

learning_rate_1�h�5w=�$I       6%�	�;�k���A�;*;


total_lossmj�@

error_R�cB?

learning_rate_1�h�5]H(I       6%�	f��k���A�;*;


total_loss~�
A

error_R�=G?

learning_rate_1�h�5L���I       6%�	�ĭk���A�;*;


total_loss��@

error_R/Q?

learning_rate_1�h�5�X��I       6%�	6	�k���A�;*;


total_loss@

error_R�gI?

learning_rate_1�h�54��I       6%�	�l�k���A�;*;


total_loss���@

error_R�U?

learning_rate_1�h�5X=��I       6%�	���k���A�;*;


total_loss�@�@

error_R�/=?

learning_rate_1�h�5��MI       6%�	���k���A�;*;


total_loss	�@

error_R�M?

learning_rate_1�h�5S�$�I       6%�	JG�k���A�;*;


total_lossT��@

error_R��D?

learning_rate_1�h�5�M�:I       6%�	���k���A�;*;


total_loss�۞@

error_R�]?

learning_rate_1�h�5%��AI       6%�	ӯk���A�;*;


total_lossRS�@

error_RHtK?

learning_rate_1�h�5À��I       6%�	D�k���A�;*;


total_loss�$�@

error_R�~N?

learning_rate_1�h�5G���I       6%�	LX�k���A�;*;


total_loss���@

error_R`H?

learning_rate_1�h�5I���I       6%�	���k���A�;*;


total_loss/�A

error_Rנ;?

learning_rate_1�h�5X��I       6%�	�ܰk���A�;*;


total_loss�X�@

error_RfU?

learning_rate_1�h�5�K��I       6%�	l!�k���A�<*;


total_loss���@

error_R�T?

learning_rate_1�h�5�D�I       6%�	�i�k���A�<*;


total_loss�y�@

error_R�V@?

learning_rate_1�h�5)�gI       6%�	���k���A�<*;


total_loss�ؽ@

error_R�>O?

learning_rate_1�h�5�J$I       6%�	��k���A�<*;


total_loss[J�@

error_R.�R?

learning_rate_1�h�5�J�I       6%�	�:�k���A�<*;


total_loss�Z�@

error_R
�T?

learning_rate_1�h�5���aI       6%�	녲k���A�<*;


total_loss�N�@

error_R�\X?

learning_rate_1�h�5���JI       6%�	�޲k���A�<*;


total_loss��@

error_R_h?

learning_rate_1�h�5*X-I       6%�	�&�k���A�<*;


total_lossN��@

error_RW;?

learning_rate_1�h�5Tm�I       6%�	�o�k���A�<*;


total_loss�H�@

error_R�k?

learning_rate_1�h�5�ãaI       6%�	 ˳k���A�<*;


total_loss���@

error_R��G?

learning_rate_1�h�5��m�I       6%�	A�k���A�<*;


total_loss-��@

error_R}vL?

learning_rate_1�h�5���I       6%�	�`�k���A�<*;


total_loss��@

error_R]eI?

learning_rate_1�h�5,n�'I       6%�	4��k���A�<*;


total_loss��@

error_R2Z?

learning_rate_1�h�5���I       6%�	��k���A�<*;


total_loss���@

error_RL??

learning_rate_1�h�5����I       6%�	4-�k���A�<*;


total_loss��@

error_Rn�Q?

learning_rate_1�h�5�Rv�I       6%�	ct�k���A�<*;


total_lossEs�@

error_R6HI?

learning_rate_1�h�5�dI       6%�	r��k���A�<*;


total_loss��@

error_R��U?

learning_rate_1�h�5�&��I       6%�	���k���A�<*;


total_loss�vi@

error_R �K?

learning_rate_1�h�5˔өI       6%�	k=�k���A�<*;


total_loss��@

error_RRW?

learning_rate_1�h�54�'I       6%�	O��k���A�<*;


total_lossc�@

error_R�T?

learning_rate_1�h�5(�,�I       6%�	�ȶk���A�<*;


total_loss狑@

error_R��_?

learning_rate_1�h�5\��I       6%�	��k���A�<*;


total_loss$�@

error_R{�9?

learning_rate_1�h�5��'mI       6%�	[�k���A�<*;


total_lossdR�@

error_R��;?

learning_rate_1�h�5G�7GI       6%�	�k���A�<*;


total_loss�H�@

error_R��R?

learning_rate_1�h�5�0I       6%�	�k���A�<*;


total_lossu��@

error_R&G?

learning_rate_1�h�5k0�XI       6%�	1�k���A�<*;


total_loss�@

error_Rɜ@?

learning_rate_1�h�5��I       6%�	j��k���A�<*;


total_loss�5�@

error_R�sC?

learning_rate_1�h�5���CI       6%�	PŸk���A�<*;


total_loss�.�@

error_R\rQ?

learning_rate_1�h�5v��GI       6%�	|�k���A�<*;


total_lossN+�@

error_Rq%H?

learning_rate_1�h�5wfĿI       6%�	YU�k���A�<*;


total_loss��@

error_R�FZ?

learning_rate_1�h�5��7�I       6%�	ޚ�k���A�<*;


total_lossz@

error_R��[?

learning_rate_1�h�5g��I       6%�	��k���A�<*;


total_loss*á@

error_R��Y?

learning_rate_1�h�5�V��I       6%�	,�k���A�<*;


total_lossDޝ@

error_R�,K?

learning_rate_1�h�5R�OI       6%�	�q�k���A�<*;


total_losse�P@

error_Rq~>?

learning_rate_1�h�5�ge�I       6%�	̷�k���A�<*;


total_loss�3�@

error_R�K?

learning_rate_1�h�5"���I       6%�	A��k���A�<*;


total_loss�yA

error_R�9I?

learning_rate_1�h�59Cd�I       6%�	�B�k���A�<*;


total_lossط�@

error_R8!a?

learning_rate_1�h�5�R��I       6%�	r��k���A�<*;


total_loss�<�@

error_R�fL?

learning_rate_1�h�5���I       6%�	dϻk���A�<*;


total_loss�d�@

error_R8�_?

learning_rate_1�h�5�>I       6%�	��k���A�<*;


total_loss��@

error_R�K?

learning_rate_1�h�5:_��I       6%�	V�k���A�<*;


total_loss=x�@

error_R��D?

learning_rate_1�h�5Vϟ�I       6%�	隼k���A�<*;


total_loss&�@

error_R��U?

learning_rate_1�h�5�&�I       6%�	xݼk���A�<*;


total_loss�t�@

error_R=V?

learning_rate_1�h�5��vdI       6%�	&#�k���A�<*;


total_loss�{�@

error_R�&N?

learning_rate_1�h�5��I       6%�	qi�k���A�<*;


total_loss���@

error_RW�I?

learning_rate_1�h�5NkI       6%�	m��k���A�<*;


total_loss=��@

error_RPQ?

learning_rate_1�h�5�w�I       6%�	��k���A�<*;


total_loss��@

error_Rܦ^?

learning_rate_1�h�5.3�I       6%�	�I�k���A�<*;


total_lossM��@

error_RaY?

learning_rate_1�h�5���I       6%�	���k���A�<*;


total_loss�/�@

error_R��Q?

learning_rate_1�h�57X�YI       6%�	��k���A�<*;


total_lossSG�@

error_R��V?

learning_rate_1�h�5)@�6I       6%�	�1�k���A�<*;


total_loss�v�@

error_Rc�P?

learning_rate_1�h�5�\Y I       6%�	*x�k���A�<*;


total_loss���@

error_R��T?

learning_rate_1�h�5ẘ%I       6%�	��k���A�<*;


total_loss���@

error_RM�K?

learning_rate_1�h�5��I       6%�	� �k���A�<*;


total_lossZG�@

error_R�\J?

learning_rate_1�h�5�BR)I       6%�	B�k���A�<*;


total_lossx_�@

error_R�M?

learning_rate_1�h�5�Rc�I       6%�	���k���A�<*;


total_loss�P�@

error_R�KH?

learning_rate_1�h�5�}I^I       6%�	��k���A�<*;


total_loss�ܰ@

error_R��M?

learning_rate_1�h�5��pI       6%�	��k���A�<*;


total_loss�5�@

error_R��F?

learning_rate_1�h�5�`%�I       6%�	�W�k���A�<*;


total_loss$�@

error_R#Y?

learning_rate_1�h�5��"�I       6%�	\��k���A�<*;


total_loss��@

error_R �??

learning_rate_1�h�5�,a�I       6%�	f��k���A�<*;


total_lossۙ@

error_R��O?

learning_rate_1�h�5�� �I       6%�	:'�k���A�<*;


total_loss��@

error_R�/N?

learning_rate_1�h�5�wONI       6%�	�j�k���A�<*;


total_loss��@

error_R�^?

learning_rate_1�h�5O:�I       6%�	P��k���A�<*;


total_lossݎ�@

error_R1Fa?

learning_rate_1�h�5tK�I       6%�	G��k���A�<*;


total_loss� �@

error_R�i4?

learning_rate_1�h�5��� I       6%�	8�k���A�<*;


total_loss�Ґ@

error_R�[J?

learning_rate_1�h�5Db��I       6%�	{�k���A�<*;


total_loss��@

error_RG??

learning_rate_1�h�5�@I       6%�	���k���A�<*;


total_lossD��@

error_R�qX?

learning_rate_1�h�5��o�I       6%�	4�k���A�<*;


total_loss!��@

error_R��T?

learning_rate_1�h�5Dj$(I       6%�	CK�k���A�<*;


total_loss�@

error_R)�D?

learning_rate_1�h�5�|��I       6%�	���k���A�<*;


total_loss��A

error_R$qL?

learning_rate_1�h�5���I       6%�	���k���A�<*;


total_loss���@

error_R��L?

learning_rate_1�h�5�7��I       6%�	�k���A�<*;


total_loss*A�@

error_R=9?

learning_rate_1�h�5��I>I       6%�	6_�k���A�<*;


total_loss̑@

error_R&�R?

learning_rate_1�h�5�0kI       6%�	F��k���A�<*;


total_loss���@

error_R�#M?

learning_rate_1�h�5q�gI       6%�	p��k���A�<*;


total_loss��@

error_R)<L?

learning_rate_1�h�5|"�I       6%�	�9�k���A�<*;


total_loss}kA

error_R�1L?

learning_rate_1�h�5��I       6%�	�{�k���A�<*;


total_loss__@

error_R�mX?

learning_rate_1�h�5�GSEI       6%�	���k���A�<*;


total_lossn��@

error_R��X?

learning_rate_1�h�5qv1aI       6%�	��k���A�<*;


total_loss�"o@

error_RX�E?

learning_rate_1�h�5����I       6%�	ZI�k���A�<*;


total_lossȕ�@

error_R�qC?

learning_rate_1�h�5p��I       6%�	\��k���A�<*;


total_loss��z@

error_RM�F?

learning_rate_1�h�5p�i�I       6%�	���k���A�<*;


total_loss�a�@

error_R$DI?

learning_rate_1�h�5��ۥI       6%�	� �k���A�<*;


total_loss��@

error_RJ�M?

learning_rate_1�h�5�M�I       6%�	�a�k���A�<*;


total_lossd�@

error_Ri�P?

learning_rate_1�h�5,�*~I       6%�	"��k���A�<*;


total_loss���@

error_R��Q?

learning_rate_1�h�5|�kI       6%�	���k���A�<*;


total_lossl��@

error_R�RL?

learning_rate_1�h�5��}�I       6%�		=�k���A�<*;


total_loss��@

error_R8�R?

learning_rate_1�h�5�s �I       6%�	O��k���A�<*;


total_loss)ܼ@

error_RT�;?

learning_rate_1�h�5�O�.I       6%�	���k���A�<*;


total_loss4��@

error_R��L?

learning_rate_1�h�5�ϻVI       6%�	
�k���A�<*;


total_loss��@

error_RCH?

learning_rate_1�h�5�I       6%�	�Y�k���A�<*;


total_loss�y�@

error_R�/E?

learning_rate_1�h�5�\�I       6%�	���k���A�<*;


total_loss7�@

error_R�	X?

learning_rate_1�h�5��n.I       6%�	��k���A�<*;


total_loss!i�@

error_R�b<?

learning_rate_1�h�5&Z�XI       6%�	�.�k���A�<*;


total_loss�3�@

error_R�J?

learning_rate_1�h�5Vt8I       6%�	�{�k���A�<*;


total_loss-S�@

error_R�T?

learning_rate_1�h�5��I       6%�	���k���A�<*;


total_loss���@

error_R�Q?

learning_rate_1�h�5�s	�I       6%�	�k���A�<*;


total_loss?��@

error_R\�J?

learning_rate_1�h�5�RY�I       6%�	�R�k���A�<*;


total_lossHA

error_R�AG?

learning_rate_1�h�5�I       6%�	'��k���A�<*;


total_loss��@

error_R��U?

learning_rate_1�h�5���TI       6%�	���k���A�<*;


total_loss�]�@

error_R*�??

learning_rate_1�h�5�9iaI       6%�	�2�k���A�<*;


total_loss�%\@

error_R��[?

learning_rate_1�h�5gk�)I       6%�	x�k���A�<*;


total_loss�!�@

error_Rf�K?

learning_rate_1�h�5mS�I       6%�	��k���A�<*;


total_loss���@

error_R�FQ?

learning_rate_1�h�5?ڥI       6%�	�k���A�<*;


total_loss(Q�@

error_RJL]?

learning_rate_1�h�5��RPI       6%�	Ea�k���A�<*;


total_loss\�Y@

error_R��<?

learning_rate_1�h�5��0I       6%�	p��k���A�<*;


total_loss��@

error_RW�A?

learning_rate_1�h�5���I       6%�	���k���A�<*;


total_lossH�@

error_R�|E?

learning_rate_1�h�5��#�I       6%�	�3�k���A�<*;


total_losss"�@

error_RɗV?

learning_rate_1�h�5	T�I       6%�	@y�k���A�<*;


total_loss�
�@

error_R�UD?

learning_rate_1�h�5d�<!I       6%�	��k���A�<*;


total_loss�4�@

error_R��c?

learning_rate_1�h�5��I       6%�	��k���A�<*;


total_loss�+�@

error_R;.r?

learning_rate_1�h�5x�ϜI       6%�	S�k���A�<*;


total_loss/�}@

error_R�U?

learning_rate_1�h�5�ОI       6%�	S��k���A�<*;


total_loss[ys@

error_R��=?

learning_rate_1�h�5;SCI       6%�	<��k���A�<*;


total_lossS�@

error_R7y\?

learning_rate_1�h�5Y�~GI       6%�	�(�k���A�<*;


total_loss�r'A

error_R�a?

learning_rate_1�h�5��AI       6%�	�r�k���A�<*;


total_lossi��@

error_R��K?

learning_rate_1�h�5�#I       6%�	m��k���A�<*;


total_lossZ�@

error_R.�;?

learning_rate_1�h�5��/�I       6%�	&�k���A�<*;


total_loss���@

error_R�H?

learning_rate_1�h�5�fiXI       6%�	�O�k���A�<*;


total_loss<f�@

error_RO�H?

learning_rate_1�h�5�@*�I       6%�	;��k���A�<*;


total_lossF��@

error_R�(N?

learning_rate_1�h�5O�o�I       6%�	G��k���A�<*;


total_loss�'�@

error_R<�G?

learning_rate_1�h�5B�k>I       6%�	V8�k���A�<*;


total_lossZ#�@

error_R)�M?

learning_rate_1�h�5$�sI       6%�	�|�k���A�<*;


total_loss�@

error_R��Y?

learning_rate_1�h�5�N�I       6%�	��k���A�<*;


total_loss���@

error_Ri�N?

learning_rate_1�h�5���LI       6%�	Q �k���A�<*;


total_losscg@

error_R�ZC?

learning_rate_1�h�5t�zI       6%�	1f�k���A�<*;


total_lossX�@

error_R�vT?

learning_rate_1�h�5f�48I       6%�	O��k���A�<*;


total_loss,��@

error_ROfK?

learning_rate_1�h�5���OI       6%�	���k���A�=*;


total_loss�+�@

error_R��V?

learning_rate_1�h�5�B�I       6%�	�<�k���A�=*;


total_loss@/�@

error_R;Xe?

learning_rate_1�h�5j�HI       6%�	p��k���A�=*;


total_loss��@

error_RfkJ?

learning_rate_1�h�5��U�I       6%�	N��k���A�=*;


total_loss��@

error_R��I?

learning_rate_1�h�5���rI       6%�	O�k���A�=*;


total_loss7\�@

error_R�x@?

learning_rate_1�h�5qyTlI       6%�	�Q�k���A�=*;


total_loss�O�@

error_R�n;?

learning_rate_1�h�53��$I       6%�	���k���A�=*;


total_loss��@

error_RWMF?

learning_rate_1�h�5��KlI       6%�	���k���A�=*;


total_lossI��@

error_R}�<?

learning_rate_1�h�5/`�I       6%�	��k���A�=*;


total_lossH�@

error_R�`?

learning_rate_1�h�5t��I       6%�	|f�k���A�=*;


total_loss7�@

error_R,�i?

learning_rate_1�h�5����I       6%�	���k���A�=*;


total_loss�:�@

error_R�M?

learning_rate_1�h�5����I       6%�	*��k���A�=*;


total_loss�=�@

error_R��I?

learning_rate_1�h�5���`I       6%�	M6�k���A�=*;


total_loss���@

error_RdZ?

learning_rate_1�h�5K���I       6%�	@��k���A�=*;


total_loss�A�@

error_Rq�??

learning_rate_1�h�5���I       6%�	���k���A�=*;


total_loss�z�@

error_RD�B?

learning_rate_1�h�5���I       6%�	6�k���A�=*;


total_loss���@

error_R{�P?

learning_rate_1�h�5���XI       6%�	1P�k���A�=*;


total_lossc1A

error_Rl�[?

learning_rate_1�h�5C���I       6%�	.��k���A�=*;


total_loss��A

error_R��S?

learning_rate_1�h�5�ǖ�I       6%�	���k���A�=*;


total_loss&��@

error_R��L?

learning_rate_1�h�5w&�pI       6%�	�+�k���A�=*;


total_loss6�@

error_R�2E?

learning_rate_1�h�5���pI       6%�	�p�k���A�=*;


total_lossX��@

error_R!�@?

learning_rate_1�h�5��4I       6%�	;��k���A�=*;


total_loss	�@

error_Ri�N?

learning_rate_1�h�5��^I       6%�	��k���A�=*;


total_loss���@

error_R�N?

learning_rate_1�h�5�~�I       6%�	f>�k���A�=*;


total_loss\��@

error_R 8?

learning_rate_1�h�5��وI       6%�	���k���A�=*;


total_loss�@

error_R\�Q?

learning_rate_1�h�5"�rI       6%�	���k���A�=*;


total_loss� A

error_R�mL?

learning_rate_1�h�5�{��I       6%�	%�k���A�=*;


total_lossV�@

error_R8�A?

learning_rate_1�h�5���_I       6%�	�`�k���A�=*;


total_loss�c�@

error_R�3?

learning_rate_1�h�54�bI       6%�	Ԧ�k���A�=*;


total_lossj��@

error_RO�E?

learning_rate_1�h�5'ʅ�I       6%�	���k���A�=*;


total_loss��@

error_R��Z?

learning_rate_1�h�5����I       6%�	+�k���A�=*;


total_loss��@

error_RO?

learning_rate_1�h�5Զi�I       6%�	Cl�k���A�=*;


total_loss�t�@

error_R�3P?

learning_rate_1�h�5��I�I       6%�	f��k���A�=*;


total_lossQE�@

error_R�8O?

learning_rate_1�h�5�];�I       6%�	��k���A�=*;


total_loss�<�@

error_R��:?

learning_rate_1�h�5�$�I       6%�	�C�k���A�=*;


total_loss$A�@

error_R	�O?

learning_rate_1�h�5���I       6%�	M��k���A�=*;


total_loss�9�@

error_R
U?

learning_rate_1�h�5��I       6%�	<��k���A�=*;


total_loss?�@

error_R��@?

learning_rate_1�h�5j��OI       6%�	�%�k���A�=*;


total_loss�d�@

error_RR�Q?

learning_rate_1�h�5*�u7I       6%�	o�k���A�=*;


total_loss�S�@

error_R��5?

learning_rate_1�h�5Iq/�I       6%�	��k���A�=*;


total_loss�ݮ@

error_R{)`?

learning_rate_1�h�5V���I       6%�	� �k���A�=*;


total_loss�r�@

error_R�Z?

learning_rate_1�h�5OL�I       6%�	*H�k���A�=*;


total_loss��o@

error_R��>?

learning_rate_1�h�5}���I       6%�	-��k���A�=*;


total_lossk�@

error_RƙJ?

learning_rate_1�h�5���}I       6%�	��k���A�=*;


total_lossAٳ@

error_R}�@?

learning_rate_1�h�5���jI       6%�	�$�k���A�=*;


total_loss�w@

error_R�KB?

learning_rate_1�h�5�<"I       6%�	ck�k���A�=*;


total_loss�e"A

error_R?PP?

learning_rate_1�h�5)��I       6%�	Ӷ�k���A�=*;


total_lossدA

error_R$*X?

learning_rate_1�h�5�ڬ^I       6%�	���k���A�=*;


total_lossz��@

error_R_�^?

learning_rate_1�h�5��q�I       6%�	z?�k���A�=*;


total_loss�٥@

error_R2aL?

learning_rate_1�h�5kI^�I       6%�	���k���A�=*;


total_lossZ��@

error_R ?Y?

learning_rate_1�h�5��9I       6%�	���k���A�=*;


total_loss���@

error_R�N?

learning_rate_1�h�5��nI       6%�	��k���A�=*;


total_loss���@

error_R�
P?

learning_rate_1�h�5;�I       6%�	�P�k���A�=*;


total_lossLB�@

error_R\HG?

learning_rate_1�h�5oD�WI       6%�	y��k���A�=*;


total_loss���@

error_R��J?

learning_rate_1�h�5��c7I       6%�	=��k���A�=*;


total_loss�C A

error_R�mX?

learning_rate_1�h�5�3~uI       6%�	��k���A�=*;


total_loss���@

error_R
�J?

learning_rate_1�h�5��I       6%�	�^�k���A�=*;


total_loss�Y�@

error_R�O?

learning_rate_1�h�5(��rI       6%�	w��k���A�=*;


total_loss�O�@

error_RJL?

learning_rate_1�h�5��_I       6%�	/��k���A�=*;


total_loss�v@

error_R�K?

learning_rate_1�h�5 kI       6%�	�0�k���A�=*;


total_loss[�A

error_RRO[?

learning_rate_1�h�5�V@I       6%�	�}�k���A�=*;


total_loss�w�@

error_R�LB?

learning_rate_1�h�5i�I       6%�	"��k���A�=*;


total_lossj�@

error_R6�M?

learning_rate_1�h�5�w�FI       6%�	J�k���A�=*;


total_loss��A

error_R�2B?

learning_rate_1�h�5�DN`I       6%�	�U�k���A�=*;


total_loss��@

error_R{�P?

learning_rate_1�h�57��EI       6%�	j��k���A�=*;


total_loss�-�@

error_R��@?

learning_rate_1�h�5�t�pI       6%�	4��k���A�=*;


total_loss�@

error_R�O?

learning_rate_1�h�5�Z�I       6%�	�(�k���A�=*;


total_loss�C�@

error_RYC?

learning_rate_1�h�5ȇ��I       6%�	�l�k���A�=*;


total_loss{��@

error_Ra"G?

learning_rate_1�h�5Du/�I       6%�	��k���A�=*;


total_loss���@

error_R�[4?

learning_rate_1�h�5����I       6%�	E��k���A�=*;


total_loss!�@

error_RtJ?

learning_rate_1�h�5g5�aI       6%�	�:�k���A�=*;


total_loss��@

error_RaL`?

learning_rate_1�h�5A\I       6%�	=��k���A�=*;


total_lossԝ�@

error_RZ@N?

learning_rate_1�h�5�2�WI       6%�	���k���A�=*;


total_loss��@

error_Rh_O?

learning_rate_1�h�5Av��I       6%�	��k���A�=*;


total_loss2��@

error_R36H?

learning_rate_1�h�5�nRI       6%�	�O�k���A�=*;


total_loss���@

error_R�	K?

learning_rate_1�h�5�ppI       6%�	��k���A�=*;


total_loss��@

error_R;iQ?

learning_rate_1�h�5\f|kI       6%�	���k���A�=*;


total_loss���@

error_R��K?

learning_rate_1�h�5���iI       6%�	_�k���A�=*;


total_lossM�r@

error_R@�=?

learning_rate_1�h�5�]I       6%�	/^�k���A�=*;


total_loss�/�@

error_R�J?

learning_rate_1�h�5���I       6%�	���k���A�=*;


total_lossC��@

error_RvWC?

learning_rate_1�h�5W�dI       6%�	P��k���A�=*;


total_loss��@

error_R!�W?

learning_rate_1�h�5U�BI       6%�	�$�k���A�=*;


total_lossE��@

error_R�iC?

learning_rate_1�h�53�A�I       6%�	;i�k���A�=*;


total_loss;˖@

error_R��S?

learning_rate_1�h�5B
�I       6%�	3��k���A�=*;


total_lossÓ@

error_R|�^?

learning_rate_1�h�5��=�I       6%�	8��k���A�=*;


total_lossp�@

error_R�{I?

learning_rate_1�h�5h��I       6%�	:�k���A�=*;


total_loss�B�@

error_Rd�K?

learning_rate_1�h�5�Ʉ�I       6%�	��k���A�=*;


total_lossC�@

error_R�O_?

learning_rate_1�h�5v��I       6%�	]��k���A�=*;


total_loss4z�@

error_R�e7?

learning_rate_1�h�5zSheI       6%�	{	�k���A�=*;


total_loss�ӭ@

error_R�|I?

learning_rate_1�h�5��;�I       6%�	-M�k���A�=*;


total_loss�@

error_Rj@Z?

learning_rate_1�h�5T��I       6%�	7��k���A�=*;


total_lossi�@

error_RH�K?

learning_rate_1�h�5�u�I       6%�	���k���A�=*;


total_loss_��@

error_R�tL?

learning_rate_1�h�5K3I       6%�	}�k���A�=*;


total_loss=�s@

error_R��[?

learning_rate_1�h�55�=-I       6%�	���k���A�=*;


total_loss�w�@

error_R�ZU?

learning_rate_1�h�5���FI       6%�	.��k���A�=*;


total_lossib�@

error_R?�X?

learning_rate_1�h�5�d�I       6%�	?�k���A�=*;


total_loss��@

error_R�Y?

learning_rate_1�h�5��;�I       6%�	lW�k���A�=*;


total_loss���@

error_R)lT?

learning_rate_1�h�5�d6kI       6%�	Ś�k���A�=*;


total_lossr�n@

error_R�aI?

learning_rate_1�h�5�G>I       6%�	���k���A�=*;


total_loss]ʢ@

error_R��K?

learning_rate_1�h�5C��}I       6%�	�$�k���A�=*;


total_loss_��@

error_R�H`?

learning_rate_1�h�5�R-�I       6%�	�g�k���A�=*;


total_lossK�@

error_R�uR?

learning_rate_1�h�5-��WI       6%�	���k���A�=*;


total_loss(�A

error_RST?

learning_rate_1�h�5����I       6%�	K��k���A�=*;


total_loss#��@

error_R�wC?

learning_rate_1�h�5�U�I       6%�	�1�k���A�=*;


total_lossD��@

error_R�Q?

learning_rate_1�h�5��I       6%�	{�k���A�=*;


total_loss�,�@

error_RjJf?

learning_rate_1�h�5��sI       6%�	��k���A�=*;


total_loss�ί@

error_RH�E?

learning_rate_1�h�5�1�eI       6%�	��k���A�=*;


total_loss�6�@

error_R��<?

learning_rate_1�h�5�9�7I       6%�	eN�k���A�=*;


total_loss��@

error_R.g[?

learning_rate_1�h�5��8aI       6%�	5��k���A�=*;


total_loss�A�@

error_R��U?

learning_rate_1�h�5Z��I       6%�	���k���A�=*;


total_lossV�t@

error_R��\?

learning_rate_1�h�5�
�.I       6%�	�7�k���A�=*;


total_lossJ�@

error_Rz\a?

learning_rate_1�h�5��BSI       6%�	*}�k���A�=*;


total_loss�ՙ@

error_R�O?

learning_rate_1�h�5�KM�I       6%�	���k���A�=*;


total_loss:�@

error_RH�=?

learning_rate_1�h�5�i5�I       6%�	��k���A�=*;


total_lossS�@

error_RO�Q?

learning_rate_1�h�5��d�I       6%�	r`�k���A�=*;


total_loss��@

error_R�vX?

learning_rate_1�h�5fO�I       6%�	���k���A�=*;


total_loss��@

error_R�,V?

learning_rate_1�h�5^gI       6%�	���k���A�=*;


total_lossfc�@

error_R�S?

learning_rate_1�h�5�X$�I       6%�	�@�k���A�=*;


total_loss��@

error_R�D?

learning_rate_1�h�5�YldI       6%�	#��k���A�=*;


total_loss���@

error_Rn�B?

learning_rate_1�h�5���I       6%�	��k���A�=*;


total_loss��@

error_R��:?

learning_rate_1�h�5��#�I       6%�	m�k���A�=*;


total_loss\9�@

error_R��W?

learning_rate_1�h�5�^I       6%�	!V�k���A�=*;


total_loss�'�@

error_R��G?

learning_rate_1�h�5���aI       6%�	��k���A�=*;


total_lossG��@

error_Rd�E?

learning_rate_1�h�5_o3�I       6%�	���k���A�=*;


total_loss�Ĭ@

error_R�[]?

learning_rate_1�h�5��\I       6%�	�%�k���A�=*;


total_lossO��@

error_R��J?

learning_rate_1�h�5p��_I       6%�	|k�k���A�=*;


total_loss���@

error_R�EO?

learning_rate_1�h�5z�I       6%�	+��k���A�=*;


total_loss��@

error_R�Q^?

learning_rate_1�h�5�kI       6%�	\��k���A�=*;


total_loss�.�@

error_R}�U?

learning_rate_1�h�5�'[�I       6%�	�;�k���A�>*;


total_loss��v@

error_R�cF?

learning_rate_1�h�5\ѥ�I       6%�	���k���A�>*;


total_lossJ��@

error_R� D?

learning_rate_1�h�5#[o�I       6%�	���k���A�>*;


total_lossښ�@

error_REe?

learning_rate_1�h�5K)�I       6%�	��k���A�>*;


total_lossc�@

error_R3�N?

learning_rate_1�h�5�C*�I       6%�	W�k���A�>*;


total_loss���@

error_R_nS?

learning_rate_1�h�5GɱDI       6%�	m��k���A�>*;


total_loss���@

error_R
?C?

learning_rate_1�h�5�]�SI       6%�	�9�k���A�>*;


total_loss�	A

error_Rt�E?

learning_rate_1�h�5�߿I       6%�	��k���A�>*;


total_lossM��@

error_R�bN?

learning_rate_1�h�5�Y�I       6%�	���k���A�>*;


total_loss�o�@

error_RTnD?

learning_rate_1�h�5���VI       6%�	��k���A�>*;


total_lossw��@

error_R1!O?

learning_rate_1�h�5�0�I       6%�	�J�k���A�>*;


total_lossm�@

error_R��G?

learning_rate_1�h�5ɏIBI       6%�	���k���A�>*;


total_loss�~�@

error_R��T?

learning_rate_1�h�5p��I       6%�	!��k���A�>*;


total_loss�K�@

error_Rv`I?

learning_rate_1�h�5o�I       6%�	��k���A�>*;


total_loss���@

error_R$�G?

learning_rate_1�h�5D:gI       6%�	Z�k���A�>*;


total_loss/�@

error_R�.B?

learning_rate_1�h�5�`σI       6%�	P��k���A�>*;


total_loss�҉@

error_R��M?

learning_rate_1�h�5Rw��I       6%�	6��k���A�>*;


total_loss	��@

error_RJ�C?

learning_rate_1�h�5��ʌI       6%�	+�k���A�>*;


total_lossF��@

error_R?�L?

learning_rate_1�h�5v���I       6%�	Cn�k���A�>*;


total_loss�G�@

error_R�8?

learning_rate_1�h�5f�t�I       6%�	[��k���A�>*;


total_loss��@

error_RV�]?

learning_rate_1�h�5�x�I       6%�	���k���A�>*;


total_loss��@

error_R��[?

learning_rate_1�h�5�*I       6%�	�I�k���A�>*;


total_loss��@

error_R��M?

learning_rate_1�h�5W�-�I       6%�	��k���A�>*;


total_loss���@

error_R�vS?

learning_rate_1�h�5k�6I       6%�	T��k���A�>*;


total_lossϕ A

error_R�
H?

learning_rate_1�h�5:�� I       6%�	�+�k���A�>*;


total_loss��|@

error_R�U?

learning_rate_1�h�5/�7HI       6%�	.s�k���A�>*;


total_loss�W�@

error_R�HS?

learning_rate_1�h�5�r&tI       6%�	:��k���A�>*;


total_loss�Ѷ@

error_R�W?

learning_rate_1�h�5f�8I       6%�	F	 l���A�>*;


total_loss�1�@

error_R�N?

learning_rate_1�h�5��I�I       6%�	�O l���A�>*;


total_loss�ϧ@

error_RTlE?

learning_rate_1�h�55WIwI       6%�	�� l���A�>*;


total_loss�@

error_REb@?

learning_rate_1�h�5�)d�I       6%�	�� l���A�>*;


total_loss�A

error_R�	Y?

learning_rate_1�h�5A�9�I       6%�	&l���A�>*;


total_lossڋ�@

error_R��O?

learning_rate_1�h�59з�I       6%�	2al���A�>*;


total_loss�p�@

error_R\�`?

learning_rate_1�h�5�B!&I       6%�	�l���A�>*;


total_loss���@

error_R�P?

learning_rate_1�h�5���I       6%�	��l���A�>*;


total_loss<�A

error_R�T?

learning_rate_1�h�5AW�NI       6%�	_/l���A�>*;


total_lossr&A

error_R��O?

learning_rate_1�h�5�P8*I       6%�	�rl���A�>*;


total_loss�A

error_R�M?

learning_rate_1�h�50�-:I       6%�	׹l���A�>*;


total_loss�a�@

error_RRrW?

learning_rate_1�h�5#-�I       6%�	Q�l���A�>*;


total_loss4��@

error_Rl W?

learning_rate_1�h�5���I       6%�	�Al���A�>*;


total_loss�h�@

error_R{�[?

learning_rate_1�h�5"���I       6%�	K�l���A�>*;


total_loss�m�@

error_R� I?

learning_rate_1�h�5=(&I       6%�	��l���A�>*;


total_loss�A

error_Rl�K?

learning_rate_1�h�5�u�I       6%�	�l���A�>*;


total_loss�T�@

error_R�O@?

learning_rate_1�h�5#R�I       6%�	(Kl���A�>*;


total_loss�̛@

error_R�&e?

learning_rate_1�h�5s�Q\I       6%�	�l���A�>*;


total_lossO8�@

error_R)VW?

learning_rate_1�h�5�>WrI       6%�	��l���A�>*;


total_loss�@

error_R��S?

learning_rate_1�h�5'�I       6%�	Sl���A�>*;


total_loss䙌@

error_R`pK?

learning_rate_1�h�5IV:KI       6%�	Zl���A�>*;


total_loss�w�@

error_RT�D?

learning_rate_1�h�5b��?I       6%�	��l���A�>*;


total_loss��@

error_R��Y?

learning_rate_1�h�5kOI       6%�	F�l���A�>*;


total_losss��@

error_R;�I?

learning_rate_1�h�5�w�I       6%�	5l���A�>*;


total_loss͠�@

error_R2!S?

learning_rate_1�h�5�%bWI       6%�	΃l���A�>*;


total_loss�>�@

error_RF�D?

learning_rate_1�h�5�ɞ�I       6%�	0�l���A�>*;


total_loss�_�@

error_R@|Q?

learning_rate_1�h�5{�AI       6%�	�l���A�>*;


total_lossI��@

error_RF_]?

learning_rate_1�h�5K4|�I       6%�	5_l���A�>*;


total_lossO+�@

error_R��P?

learning_rate_1�h�5�`.�I       6%�	Ŧl���A�>*;


total_loss7��@

error_RW�@?

learning_rate_1�h�5��!�I       6%�	�l���A�>*;


total_loss���@

error_R��Z?

learning_rate_1�h�5�!~'I       6%�	�9l���A�>*;


total_loss6�@

error_R8�6?

learning_rate_1�h�5�V�~I       6%�	q{l���A�>*;


total_loss���@

error_R<�;?

learning_rate_1�h�5evm�I       6%�	Z�l���A�>*;


total_lossF��@

error_R)�U?

learning_rate_1�h�5�:GGI       6%�	��l���A�>*;


total_loss���@

error_R�2N?

learning_rate_1�h�5�`��I       6%�	�A	l���A�>*;


total_loss:�@

error_R�+X?

learning_rate_1�h�5�ωUI       6%�	}�	l���A�>*;


total_lossT��@

error_RUK?

learning_rate_1�h�5��YI       6%�	��	l���A�>*;


total_loss8��@

error_R`*V?

learning_rate_1�h�5�y��I       6%�	~
l���A�>*;


total_loss��@

error_R��C?

learning_rate_1�h�5��
I       6%�	��l���A�>*;


total_lossr�@

error_R��C?

learning_rate_1�h�5Q��sI       6%�	~+l���A�>*;


total_lossN��@

error_R��F?

learning_rate_1�h�5�q��I       6%�	)zl���A�>*;


total_loss�@

error_RҧX?

learning_rate_1�h�5O.&I       6%�	Ӿl���A�>*;


total_loss�ލ@

error_R�\/?

learning_rate_1�h�5�iI       6%�	�l���A�>*;


total_loss�ɰ@

error_R�@?

learning_rate_1�h�5��zI       6%�	NPl���A�>*;


total_loss&ػ@

error_Rq�O?

learning_rate_1�h�5&��hI       6%�	��l���A�>*;


total_loss�~�@

error_R�\Y?

learning_rate_1�h�5�4I       6%�	��l���A�>*;


total_loss��@

error_R�lN?

learning_rate_1�h�5%��I       6%�	{"l���A�>*;


total_loss�5�@

error_RxuE?

learning_rate_1�h�5��&�I       6%�	ffl���A�>*;


total_loss��@

error_R�sI?

learning_rate_1�h�5�,��I       6%�	W�l���A�>*;


total_loss�j�@

error_R�H?

learning_rate_1�h�5��\�I       6%�	.�l���A�>*;


total_loss)/�@

error_Rc�T?

learning_rate_1�h�5/�>AI       6%�	z2l���A�>*;


total_loss�@

error_R�Z?

learning_rate_1�h�5��xQI       6%�	pvl���A�>*;


total_loss� �@

error_RcS?

learning_rate_1�h�5����I       6%�	��l���A�>*;


total_loss���@

error_RO[O?

learning_rate_1�h�5�Z�7I       6%�	F&l���A�>*;


total_loss=A

error_R��C?

learning_rate_1�h�5�ʚI       6%�	�ll���A�>*;


total_loss�Z�@

error_Rڙ??

learning_rate_1�h�5���I       6%�	]�l���A�>*;


total_losstg�@

error_R��E?

learning_rate_1�h�5�uQWI       6%�	�l���A�>*;


total_lossݸ�@

error_R&�W?

learning_rate_1�h�5��I       6%�	]l���A�>*;


total_loss���@

error_R��J?

learning_rate_1�h�5ܤ]_I       6%�	��l���A�>*;


total_loss�
A

error_R�y[?

learning_rate_1�h�5�xuI       6%�	V�l���A�>*;


total_loss��@

error_R�^X?

learning_rate_1�h�5�]�I       6%�	�'l���A�>*;


total_loss���@

error_R��Z?

learning_rate_1�h�5v�AvI       6%�	�kl���A�>*;


total_loss�:�@

error_R�UM?

learning_rate_1�h�5��e�I       6%�	|�l���A�>*;


total_loss�5A

error_R��S?

learning_rate_1�h�5�o��I       6%�	v�l���A�>*;


total_loss���@

error_R��E?

learning_rate_1�h�5g�G9I       6%�	;l���A�>*;


total_loss���@

error_Rf�=?

learning_rate_1�h�5�FfI       6%�	\�l���A�>*;


total_lossc�@

error_Rq�R?

learning_rate_1�h�5�Λ�I       6%�	��l���A�>*;


total_loss���@

error_R�;?

learning_rate_1�h�5o⾤I       6%�	Ml���A�>*;


total_lossʄA

error_R�'e?

learning_rate_1�h�5���I       6%�	Tl���A�>*;


total_losst�@

error_R�(_?

learning_rate_1�h�5�iI       6%�	�l���A�>*;


total_loss_t�@

error_R�4C?

learning_rate_1�h�5P mI       6%�	z�l���A�>*;


total_loss�n{@

error_RJ�S?

learning_rate_1�h�5ό��I       6%�	OGl���A�>*;


total_lossv��@

error_R��_?

learning_rate_1�h�5�])�I       6%�	��l���A�>*;


total_loss��@

error_R��S?

learning_rate_1�h�5��d�I       6%�	��l���A�>*;


total_lossR��@

error_R��V?

learning_rate_1�h�5�j�I       6%�	0"l���A�>*;


total_lossx��@

error_R=jN?

learning_rate_1�h�5
�I       6%�	�l���A�>*;


total_loss���@

error_RC�A?

learning_rate_1�h�5��4I       6%�	�l���A�>*;


total_loss���@

error_R׍??

learning_rate_1�h�5�L�qI       6%�	e
l���A�>*;


total_loss�Դ@

error_R��V?

learning_rate_1�h�5�
uI       6%�	�Yl���A�>*;


total_loss��@

error_RjY?

learning_rate_1�h�5���~I       6%�	��l���A�>*;


total_loss� �@

error_R��O?

learning_rate_1�h�5`��I       6%�	�l���A�>*;


total_loss���@

error_R�yF?

learning_rate_1�h�50�VI       6%�	�Wl���A�>*;


total_loss�0�@

error_R�L?

learning_rate_1�h�5��_I       6%�	;�l���A�>*;


total_lossl�A

error_R�Z?

learning_rate_1�h�5�Y%�I       6%�	l���A�>*;


total_loss�e�@

error_R�=G?

learning_rate_1�h�5`�_I       6%�	�Sl���A�>*;


total_loss@�w@

error_R�F;?

learning_rate_1�h�54���I       6%�	B�l���A�>*;


total_loss<{�@

error_R
9U?

learning_rate_1�h�5����I       6%�	9�l���A�>*;


total_lossEN�@

error_R��Q?

learning_rate_1�h�5���I       6%�	�.l���A�>*;


total_loss%�@

error_RhKN?

learning_rate_1�h�5�Wu�I       6%�	��l���A�>*;


total_loss��@

error_RqB?

learning_rate_1�h�5�j�I       6%�	��l���A�>*;


total_loss"��@

error_R�(L?

learning_rate_1�h�5��N	I       6%�	�l���A�>*;


total_loss�*�@

error_R\aM?

learning_rate_1�h�5���jI       6%�	��l���A�>*;


total_loss���@

error_Rj6M?

learning_rate_1�h�5�ma�I       6%�	r�l���A�>*;


total_loss��@

error_R��N?

learning_rate_1�h�5zqI       6%�	Al���A�>*;


total_lossw�i@

error_R�]R?

learning_rate_1�h�5Ì��I       6%�	ȕl���A�>*;


total_lossQE�@

error_R�@?

learning_rate_1�h�5�%��I       6%�	H�l���A�>*;


total_loss��@

error_R��4?

learning_rate_1�h�5���I       6%�	�* l���A�>*;


total_lossz@

error_R�[7?

learning_rate_1�h�5'�-I       6%�	�� l���A�>*;


total_loss�(�@

error_R{�I?

learning_rate_1�h�5��I       6%�	� l���A�>*;


total_loss�}�@

error_R	�J?

learning_rate_1�h�5�H��I       6%�	!l���A�>*;


total_losss��@

error_RSE?

learning_rate_1�h�5����I       6%�	3Y!l���A�>*;


total_loss�A

error_R)�P?

learning_rate_1�h�5�+̬I       6%�	�!l���A�?*;


total_loss2ϫ@

error_R�Z?

learning_rate_1�h�5\BPI       6%�	��!l���A�?*;


total_loss1�d@

error_R�J?

learning_rate_1�h�55�#I       6%�	�)"l���A�?*;


total_loss���@

error_R�<[?

learning_rate_1�h�5���'I       6%�	|q"l���A�?*;


total_loss�s�@

error_R��B?

learning_rate_1�h�5(pr�I       6%�	�"l���A�?*;


total_loss��@

error_R�0U?

learning_rate_1�h�5bfEI       6%�	 #l���A�?*;


total_loss�pA

error_R��`?

learning_rate_1�h�5ݳ�I       6%�	�E#l���A�?*;


total_loss� 	A

error_R�6?

learning_rate_1�h�5����I       6%�	��#l���A�?*;


total_loss�i�@

error_RiL?

learning_rate_1�h�5	�I       6%�	��#l���A�?*;


total_loss40�@

error_R�>?

learning_rate_1�h�5�Ek�I       6%�	0$l���A�?*;


total_lossĖ�@

error_R��9?

learning_rate_1�h�5��K�I       6%�	F[$l���A�?*;


total_loss��@

error_R��S?

learning_rate_1�h�5d��`I       6%�	�$l���A�?*;


total_loss���@

error_R�hG?

learning_rate_1�h�53���I       6%�	��$l���A�?*;


total_loss�P�@

error_R��O?

learning_rate_1�h�5ɸ�I       6%�	�#%l���A�?*;


total_loss��@

error_RZZ?

learning_rate_1�h�5��^�I       6%�	zq%l���A�?*;


total_loss#h�@

error_R�Z?

learning_rate_1�h�5=R2�I       6%�	��%l���A�?*;


total_loss_��@

error_RoJR?

learning_rate_1�h�5v�)qI       6%�	g�%l���A�?*;


total_loss���@

error_R�|F?

learning_rate_1�h�5��)lI       6%�	nD&l���A�?*;


total_loss���@

error_R��T?

learning_rate_1�h�5��bI       6%�	��&l���A�?*;


total_loss�|�@

error_Rx/\?

learning_rate_1�h�5s���I       6%�	��&l���A�?*;


total_loss��@

error_R��<?

learning_rate_1�h�5U�qI       6%�	l'l���A�?*;


total_lossz�@

error_R|TQ?

learning_rate_1�h�5zE<JI       6%�	�T'l���A�?*;


total_loss@�@

error_R״V?

learning_rate_1�h�5����I       6%�	�'l���A�?*;


total_loss�}@

error_R1K?

learning_rate_1�h�5e(^�I       6%�	f�'l���A�?*;


total_loss�@

error_R�5H?

learning_rate_1�h�5��e�I       6%�	(l���A�?*;


total_lossֆ�@

error_R�H?

learning_rate_1�h�5l|I       6%�	Ha(l���A�?*;


total_loss�E�@

error_R��T?

learning_rate_1�h�5���I       6%�	�(l���A�?*;


total_loss��@

error_R�E?

learning_rate_1�h�5�II       6%�	��(l���A�?*;


total_loss��@

error_R�]=?

learning_rate_1�h�5ɢ�I       6%�	�))l���A�?*;


total_loss�`�@

error_R�@T?

learning_rate_1�h�5�:�I       6%�	�n)l���A�?*;


total_loss�_�@

error_R�J?

learning_rate_1�h�5����I       6%�	x�)l���A�?*;


total_lossjת@

error_R(nV?

learning_rate_1�h�5֞B$I       6%�	*�)l���A�?*;


total_loss�ف@

error_R��A?

learning_rate_1�h�5x�`�I       6%�	
<*l���A�?*;


total_loss��@

error_R�"E?

learning_rate_1�h�5��z�I       6%�	��*l���A�?*;


total_loss���@

error_RxD?

learning_rate_1�h�5�+r�I       6%�	��*l���A�?*;


total_loss
�@

error_R�bE?

learning_rate_1�h�5�o>I       6%�	�+l���A�?*;


total_loss�V�@

error_R�Y?

learning_rate_1�h�5
_��I       6%�	~W+l���A�?*;


total_loss�b�@

error_R�HD?

learning_rate_1�h�5 � I       6%�	�+l���A�?*;


total_loss/<�@

error_Rx�J?

learning_rate_1�h�5��I       6%�	1�+l���A�?*;


total_loss?[�@

error_R�TI?

learning_rate_1�h�5�XY[I       6%�	�',l���A�?*;


total_loss}wr@

error_R�cK?

learning_rate_1�h�5M�I       6%�	l,l���A�?*;


total_loss��@

error_R�3X?

learning_rate_1�h�5A#I       6%�	��,l���A�?*;


total_loss���@

error_R�_I?

learning_rate_1�h�5�$	I       6%�	�,l���A�?*;


total_lossG�@

error_R�wM?

learning_rate_1�h�5̠LI       6%�	E-l���A�?*;


total_loss�Y�@

error_R�rI?

learning_rate_1�h�5�J�I       6%�	��-l���A�?*;


total_loss�
A

error_Ru>?

learning_rate_1�h�59ԱI       6%�	 �-l���A�?*;


total_loss/�@

error_R��L?

learning_rate_1�h�5��rI       6%�	.l���A�?*;


total_loss؏@

error_RsL?

learning_rate_1�h�5��;I       6%�	xx.l���A�?*;


total_loss*8�@

error_RS�@?

learning_rate_1�h�5���I       6%�	�.l���A�?*;


total_loss��@

error_R�O?

learning_rate_1�h�5�@�.I       6%�	�/l���A�?*;


total_lossE�@

error_R��P?

learning_rate_1�h�5��I       6%�	V[/l���A�?*;


total_loss.�@

error_R�J?

learning_rate_1�h�5{~ǜI       6%�	A�/l���A�?*;


total_lossF�@

error_R;-Y?

learning_rate_1�h�5ǽ��I       6%�	%�/l���A�?*;


total_lossN��@

error_R_,C?

learning_rate_1�h�5d�I       6%�	U'0l���A�?*;


total_loss�@

error_R&R?

learning_rate_1�h�5N��WI       6%�	k0l���A�?*;


total_loss���@

error_R1�8?

learning_rate_1�h�5��w�I       6%�	��0l���A�?*;


total_loss��A

error_R}�[?

learning_rate_1�h�5ʷS�I       6%�	�0l���A�?*;


total_loss٤@

error_R1�@?

learning_rate_1�h�5B��I       6%�	�31l���A�?*;


total_loss��@

error_Rd�L?

learning_rate_1�h�5qTfI       6%�		x1l���A�?*;


total_loss�T�@

error_R�pe?

learning_rate_1�h�5���I       6%�	�1l���A�?*;


total_loss�;�@

error_RD�L?

learning_rate_1�h�5H�7wI       6%�	� 2l���A�?*;


total_lossx�@

error_R�6L?

learning_rate_1�h�5C]BI       6%�	F2l���A�?*;


total_loss�G�@

error_RZ�J?

learning_rate_1�h�5�i�+I       6%�	��2l���A�?*;


total_loss~�@

error_R.Qg?

learning_rate_1�h�5�H�NI       6%�	�2l���A�?*;


total_lossO�@

error_Rs�Q?

learning_rate_1�h�5����I       6%�	�23l���A�?*;


total_loss�z�@

error_R|C?

learning_rate_1�h�5���I       6%�	�}3l���A�?*;


total_loss���@

error_Ra�\?

learning_rate_1�h�5����I       6%�	w�3l���A�?*;


total_lossf\�@

error_R�Y?

learning_rate_1�h�5 `�I       6%�	�4l���A�?*;


total_loss�:�@

error_RZ5?

learning_rate_1�h�5�c�I       6%�	BR4l���A�?*;


total_loss�!�@

error_R�]>?

learning_rate_1�h�5e��CI       6%�	��4l���A�?*;


total_lossjx�@

error_R��L?

learning_rate_1�h�5�?	�I       6%�	��4l���A�?*;


total_loss��@

error_RlP?

learning_rate_1�h�5����I       6%�	�5l���A�?*;


total_lossC�A

error_R6�P?

learning_rate_1�h�5��ޖI       6%�	�_5l���A�?*;


total_loss���@

error_RF�`?

learning_rate_1�h�5[���I       6%�	��5l���A�?*;


total_lossCA

error_Rnl>?

learning_rate_1�h�5��*/I       6%�	��5l���A�?*;


total_loss-d A

error_R!�A?

learning_rate_1�h�5u�UI       6%�	7.6l���A�?*;


total_loss;��@

error_RبS?

learning_rate_1�h�5�iTbI       6%�	�r6l���A�?*;


total_lossZ��@

error_R��O?

learning_rate_1�h�5�gj�I       6%�	H�6l���A�?*;


total_loss��@

error_R�<?

learning_rate_1�h�5��/�I       6%�	c7l���A�?*;


total_loss8��@

error_R�i?

learning_rate_1�h�5����I       6%�	�E7l���A�?*;


total_loss=�@

error_R�gS?

learning_rate_1�h�5����I       6%�	��7l���A�?*;


total_loss�@

error_R/�U?

learning_rate_1�h�5E5�I       6%�	6�7l���A�?*;


total_loss��@

error_RV�V?

learning_rate_1�h�5���I       6%�	8l���A�?*;


total_lossK�A

error_R;aT?

learning_rate_1�h�5@��7I       6%�	8V8l���A�?*;


total_lossOx A

error_Rj�S?

learning_rate_1�h�5{<B#I       6%�	��8l���A�?*;


total_loss���@

error_R�&G?

learning_rate_1�h�5jw��I       6%�	��8l���A�?*;


total_lossc��@

error_R�b9?

learning_rate_1�h�5��[�I       6%�	�29l���A�?*;


total_loss��s@

error_R�%`?

learning_rate_1�h�5�acI       6%�	�w9l���A�?*;


total_loss��@

error_R�a0?

learning_rate_1�h�5-�Q�I       6%�	��9l���A�?*;


total_lossXA�@

error_Rx�K?

learning_rate_1�h�5Mj�I       6%�	�
:l���A�?*;


total_loss��A

error_R�7[?

learning_rate_1�h�5"S� I       6%�	W:l���A�?*;


total_loss��@

error_R��^?

learning_rate_1�h�5G�bI       6%�	�:l���A�?*;


total_loss���@

error_Rc�U?

learning_rate_1�h�5���WI       6%�	��:l���A�?*;


total_loss(��@

error_R@wM?

learning_rate_1�h�5�zE�I       6%�	%;l���A�?*;


total_loss�ĕ@

error_Rl[?

learning_rate_1�h�5�aˠI       6%�	o;l���A�?*;


total_loss�P�@

error_R=kC?

learning_rate_1�h�5vF#I       6%�	�;l���A�?*;


total_loss��@

error_R)5Z?

learning_rate_1�h�5����I       6%�	��;l���A�?*;


total_loss
�@

error_R�4O?

learning_rate_1�h�5���I       6%�	EF<l���A�?*;


total_loss��@

error_RݣT?

learning_rate_1�h�5�?��I       6%�	B�<l���A�?*;


total_loss���@

error_R&QU?

learning_rate_1�h�5��}HI       6%�	��<l���A�?*;


total_loss@%�@

error_R~??

learning_rate_1�h�5��I       6%�	�=l���A�?*;


total_loss1��@

error_Rg?

learning_rate_1�h�5���I       6%�	d_=l���A�?*;


total_loss��@

error_Rj�Y?

learning_rate_1�h�5钙�I       6%�	I�=l���A�?*;


total_loss�@

error_R��K?

learning_rate_1�h�5�u��I       6%�	��=l���A�?*;


total_loss���@

error_R��8?

learning_rate_1�h�5C�I       6%�	Z/>l���A�?*;


total_losspcA

error_R�-D?

learning_rate_1�h�5�H�sI       6%�	W�>l���A�?*;


total_loss7b�@

error_R�O?

learning_rate_1�h�5Z��JI       6%�	��>l���A�?*;


total_loss�5�@

error_R�>?

learning_rate_1�h�5�`I       6%�	�.?l���A�?*;


total_loss���@

error_R��G?

learning_rate_1�h�5n�iI       6%�	�q?l���A�?*;


total_loss�i�@

error_R�N?

learning_rate_1�h�5���I       6%�	��?l���A�?*;


total_loss-̶@

error_R�X?

learning_rate_1�h�5C�I       6%�	��?l���A�?*;


total_loss��A

error_R	�P?

learning_rate_1�h�5�V�	I       6%�	#A@l���A�?*;


total_loss���@

error_R�)]?

learning_rate_1�h�5xy��I       6%�	v�@l���A�?*;


total_lossO�@

error_R�K?

learning_rate_1�h�5rc&I       6%�	��@l���A�?*;


total_loss�I�@

error_R��H?

learning_rate_1�h�5J��}I       6%�	�Al���A�?*;


total_loss~]@

error_R�Q?

learning_rate_1�h�53�I       6%�	-eAl���A�?*;


total_loss�~�@

error_R�4H?

learning_rate_1�h�5X�dzI       6%�	s�Al���A�?*;


total_loss���@

error_RM�h?

learning_rate_1�h�5M�شI       6%�	��Al���A�?*;


total_lossEy�@

error_R=3\?

learning_rate_1�h�5�rb1I       6%�	=Bl���A�?*;


total_loss6	A

error_RfGU?

learning_rate_1�h�5�<�I       6%�	a�Bl���A�?*;


total_loss���@

error_RtO?

learning_rate_1�h�5�^OI       6%�	b�Bl���A�?*;


total_loss*װ@

error_RV?

learning_rate_1�h�5hs�^I       6%�	�Cl���A�?*;


total_loss�5}@

error_R�P?

learning_rate_1�h�5[H�I       6%�	�TCl���A�?*;


total_lossJ۷@

error_R.]`?

learning_rate_1�h�5����I       6%�	ɗCl���A�?*;


total_loss���@

error_Rn�R?

learning_rate_1�h�5��W[I       6%�	N�Cl���A�?*;


total_loss�2�@

error_RO?

learning_rate_1�h�5��
I       6%�	�!Dl���A�?*;


total_loss
a�@

error_R��S?

learning_rate_1�h�5�7CRI       6%�	XfDl���A�?*;


total_loss��@

error_R��Y?

learning_rate_1�h�5͝��I       6%�	ުDl���A�?*;


total_lossW��@

error_Roj=?

learning_rate_1�h�5!e�8I       6%�	!�Dl���A�@*;


total_loss�A�@

error_R�L?

learning_rate_1�h�5�4�}I       6%�	]4El���A�@*;


total_lossX�%A

error_R=cO?

learning_rate_1�h�5��)I       6%�	IxEl���A�@*;


total_loss���@

error_RAM?

learning_rate_1�h�5����I       6%�	P�El���A�@*;


total_loss�o�@

error_R��P?

learning_rate_1�h�5�/��I       6%�	�Fl���A�@*;


total_loss��@

error_R`*I?

learning_rate_1�h�5�#	I       6%�	�LFl���A�@*;


total_loss�p�@

error_R��>?

learning_rate_1�h�5��SI       6%�	{�Fl���A�@*;


total_loss�y�@

error_R7�j?

learning_rate_1�h�5�QRI       6%�	R�Fl���A�@*;


total_loss���@

error_RDVQ?

learning_rate_1�h�5��	I       6%�	CGl���A�@*;


total_loss�
�@

error_R(�Q?

learning_rate_1�h�5�&�6I       6%�	@]Gl���A�@*;


total_loss@

error_R!�D?

learning_rate_1�h�5���fI       6%�	�Gl���A�@*;


total_loss0�@

error_R��X?

learning_rate_1�h�5�\��I       6%�	3�Gl���A�@*;


total_lossO>�@

error_R=\?

learning_rate_1�h�5��m�I       6%�	'Hl���A�@*;


total_loss���@

error_R�I?

learning_rate_1�h�5��
�I       6%�	,iHl���A�@*;


total_loss7�@

error_R �5?

learning_rate_1�h�5�l�I       6%�	ůHl���A�@*;


total_lossC�@

error_R
�P?

learning_rate_1�h�55bI       6%�	�Hl���A�@*;


total_loss_A

error_R)KX?

learning_rate_1�h�5�_�I       6%�	jDIl���A�@*;


total_loss�d�@

error_R�WC?

learning_rate_1�h�5���I       6%�	h�Il���A�@*;


total_lossK��@

error_R�5?

learning_rate_1�h�5u�I       6%�	K�Il���A�@*;


total_loss�2�@

error_R.�L?

learning_rate_1�h�5�+YhI       6%�	,&Jl���A�@*;


total_loss��@

error_Ri S?

learning_rate_1�h�5��2I       6%�	��Jl���A�@*;


total_loss�Ԏ@

error_R׋??

learning_rate_1�h�5A?fI       6%�	U�Jl���A�@*;


total_losshY�@

error_R��B?

learning_rate_1�h�5֋I       6%�	�,Kl���A�@*;


total_lossjX�@

error_R.�P?

learning_rate_1�h�59��qI       6%�	�Kl���A�@*;


total_loss�P�@

error_R�mX?

learning_rate_1�h�5D̜I       6%�	v�Kl���A�@*;


total_loss:�@

error_R��P?

learning_rate_1�h�57��eI       6%�	Ll���A�@*;


total_lossU'�@

error_R��6?

learning_rate_1�h�5�=@I       6%�	oqLl���A�@*;


total_loss:ë@

error_Rc�P?

learning_rate_1�h�5e�I       6%�	��Ll���A�@*;


total_loss�r�@

error_Ra�P?

learning_rate_1�h�5_���I       6%�	NMl���A�@*;


total_lossmF�@

error_R�J?

learning_rate_1�h�5�Y�PI       6%�	IMl���A�@*;


total_loss�P�@

error_RhH?

learning_rate_1�h�5�0�I       6%�	��Ml���A�@*;


total_loss���@

error_RJ�A?

learning_rate_1�h�5����I       6%�	��Ml���A�@*;


total_loss��@

error_R��S?

learning_rate_1�h�5�s�4I       6%�	;Nl���A�@*;


total_loss�e�@

error_Rn?

learning_rate_1�h�5�"�I       6%�	ǈNl���A�@*;


total_loss6�@

error_R��E?

learning_rate_1�h�5I��I       6%�	'�Nl���A�@*;


total_loss\8�@

error_R��??

learning_rate_1�h�5��VI       6%�	�KOl���A�@*;


total_lossi��@

error_Rx�I?

learning_rate_1�h�5$70�I       6%�	 �Ol���A�@*;


total_lossE!�@

error_RTJ?

learning_rate_1�h�59FlI       6%�	�Ol���A�@*;


total_loss ��@

error_R!�J?

learning_rate_1�h�5��xI       6%�	'Pl���A�@*;


total_loss��@

error_R&�e?

learning_rate_1�h�5�ĸ�I       6%�	�rPl���A�@*;


total_loss��@

error_R��`?

learning_rate_1�h�5�qK�I       6%�	*�Pl���A�@*;


total_loss���@

error_R8D?

learning_rate_1�h�58�I       6%�	Ql���A�@*;


total_loss�j�@

error_R_�C?

learning_rate_1�h�5W�NI       6%�	�GQl���A�@*;


total_loss�_�@

error_R�M?

learning_rate_1�h�5/���I       6%�	?�Ql���A�@*;


total_lossJ��@

error_R�TU?

learning_rate_1�h�5�
�I       6%�	��Ql���A�@*;


total_lossm+�@

error_R�?J?

learning_rate_1�h�5�5?