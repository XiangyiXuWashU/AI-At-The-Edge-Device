       �K"	  �L���Abrain.Event:2A�43>K     6�.	��L���A"��
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
weights/random_normal/mulMul*weights/random_normal/RandomStandardNormalweights/random_normal/stddev* 
_output_shapes
:
��*
T0
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
dtype0*
_output_shapes
:*
valueB"�   d   
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
weights/weight_out/readIdentityweights/weight_out*
_output_shapes

:d*
T0*%
_class
loc:@weights/weight_out
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
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

seed *
T0*
dtype0*
_output_shapes	
:�*
seed2 
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
biases_1/bias3/AssignAssignbiases_1/bias3biases_1/random_normal_2*
T0*!
_class
loc:@biases_1/bias3*
validate_shape(*
_output_shapes
:d*
use_locking(
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
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
 *���=*
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
$learning_rate/ExponentialDecay/FloorFloor&learning_rate/ExponentialDecay/truediv*
_output_shapes
: *
T0
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
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
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
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
&train/gradients/layer_3/Add_grad/Sum_1Sum)train/gradients/result/Relu_grad/ReluGrad8train/gradients/layer_3/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
+weights_1/weight_out/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:d*'
_class
loc:@weights_1/weight_out*
valueBd*    
�
weights_1/weight_out/Adam
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
biases_1/bias1/Adam/AssignAssignbiases_1/bias1/Adam%biases_1/bias1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@biases_1/bias1
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
biases_1/bias2/Adam_1/AssignAssignbiases_1/bias2/Adam_1'biases_1/bias2/Adam_1/Initializer/zeros*
T0*!
_class
loc:@biases_1/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@biases_1/bias_out*
	container 
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
*train/Adam/update_biases_1/bias1/ApplyAdam	ApplyAdambiases_1/bias1biases_1/bias1/Adambiases_1/bias1/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@biases_1/bias1
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
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:#*�
value�B�#BVariableBbiases/bias1Bbiases/bias2Bbiases/bias3Bbiases/bias_outBbiases_1/bias1Bbiases_1/bias1/AdamBbiases_1/bias1/Adam_1Bbiases_1/bias2Bbiases_1/bias2/AdamBbiases_1/bias2/Adam_1Bbiases_1/bias3Bbiases_1/bias3/AdamBbiases_1/bias3/Adam_1Bbiases_1/bias_outBbiases_1/bias_out/AdamBbiases_1/bias_out/Adam_1Btrain/beta1_powerBtrain/beta2_powerBweights/weight1Bweights/weight2Bweights/weight3Bweights/weight_outBweights_1/weight1Bweights_1/weight1/AdamBweights_1/weight1/Adam_1Bweights_1/weight2Bweights_1/weight2/AdamBweights_1/weight2/Adam_1Bweights_1/weight3Bweights_1/weight3/AdamBweights_1/weight3/Adam_1Bweights_1/weight_outBweights_1/weight_out/AdamBweights_1/weight_out/Adam_1
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:**
value!BBbiases_1/bias1/Adam_1
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
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Assign_12Assignbiases_1/bias3/Adamsave/RestoreV2_12*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
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
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_14/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBbiases_1/bias_out
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
save/Assign_18Assigntrain/beta2_powersave/RestoreV2_18*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: *
use_locking(
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
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27Assignweights_1/weight2/Adamsave/RestoreV2_27*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
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
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign"�.�Ph     6Ԇg	���L���AJ��
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
weights/random_normal/mulMul*weights/random_normal/RandomStandardNormalweights/random_normal/stddev* 
_output_shapes
:
��*
T0
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
weights/random_normal_2/mulMul,weights/random_normal_2/RandomStandardNormalweights/random_normal_2/stddev*
_output_shapes
:	�d*
T0
�
weights/random_normal_2Addweights/random_normal_2/mulweights/random_normal_2/mean*
_output_shapes
:	�d*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
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
biases/bias2/readIdentitybiases/bias2*
_output_shapes	
:�*
T0*
_class
loc:@biases/bias2
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
,weights_1/random_normal/RandomStandardNormalRandomStandardNormalweights_1/random_normal/shape*

seed *
T0*
dtype0* 
_output_shapes
:
��*
seed2 
�
weights_1/random_normal/mulMul,weights_1/random_normal/RandomStandardNormalweights_1/random_normal/stddev* 
_output_shapes
:
��*
T0
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
weights_1/random_normal_3Addweights_1/random_normal_3/mulweights_1/random_normal_3/mean*
_output_shapes

:d*
T0
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
biases_1/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+biases_1/random_normal/RandomStandardNormalRandomStandardNormalbiases_1/random_normal/shape*

seed *
T0*
dtype0*
_output_shapes	
:�*
seed2 
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
-biases_1/random_normal_3/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_3/shape*

seed *
T0*
dtype0*
_output_shapes
:*
seed2 
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
layer_2/AddAddlayer_2/MatMulbiases_1/bias2/read*(
_output_shapes
:����������*
T0
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
result/ReluRelulayer_3/Add*'
_output_shapes
:���������d*
T0
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
 *���=*
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
0train/gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs train/gradients/sub_1_grad/Shape"train/gradients/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
)train/gradients/result/Add_grad/Reshape_1Reshape%train/gradients/result/Add_grad/Sum_1'train/gradients/result/Add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
&train/gradients/layer_3/Add_grad/Sum_1Sum)train/gradients/result/Relu_grad/ReluGrad8train/gradients/layer_3/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_3/Sigmoid9train/gradients/layer_3/Add_grad/tuple/control_dependency*
_output_shapes
:	�d*
transpose_a(*
transpose_b( *
T0
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
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
(train/gradients/layer_1/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
6train/gradients/layer_1/Add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/Add_grad/Shape(train/gradients/layer_1/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
weights_1/weight1/Adam/AssignAssignweights_1/weight1/Adam(weights_1/weight1/Adam/Initializer/zeros*
T0*$
_class
loc:@weights_1/weight1*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
+weights_1/weight_out/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:d*'
_class
loc:@weights_1/weight_out*
valueBd*    
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
%biases_1/bias1/Adam/Initializer/zerosConst*!
_class
loc:@biases_1/bias1*
valueB�*    *
dtype0*
_output_shapes	
:�
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
%biases_1/bias2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias2*
valueB�*    
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
*train/Adam/update_biases_1/bias3/ApplyAdam	ApplyAdambiases_1/bias3biases_1/bias3/Adambiases_1/bias3/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@biases_1/bias3*
use_nesterov( *
_output_shapes
:d
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
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Assign_13Assignbiases_1/bias3/Adam_1save/RestoreV2_13*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
x
save/RestoreV2_14/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBbiases_1/bias_out
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
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_27Assignweights_1/weight2/Adamsave/RestoreV2_27*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*$
_class
loc:@weights_1/weight2
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
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign""
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
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0"�
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
learning_rate_1:0*֑F       r5��	4�L���A*;


total_lossCj�@

error_R�ZF?

learning_rate_1�9{��H       ��H�	b:�L���A*;


total_lossmg�@

error_R|�Z?

learning_rate_1�9�d�xH       ��H�	���L���A*;


total_loss	*�@

error_RE1P?

learning_rate_1�9�>rH       ��H�	j��L���A*;


total_lossI��@

error_R�;?

learning_rate_1�9T���H       ��H�	��L���A*;


total_loss6��@

error_R&E?

learning_rate_1�9q��#H       ��H�	#a�L���A*;


total_loss#��@

error_R��F?

learning_rate_1�9eݖ.H       ��H�	���L���A*;


total_losso�
A

error_RlbD?

learning_rate_1�9��pH       ��H�	���L���A*;


total_loss�T�@

error_RR�G?

learning_rate_1�9��H       ��H�	�\�L���A*;


total_lossʌ�@

error_R�nZ?

learning_rate_1�9���H       ��H�	G��L���A	*;


total_lossۆ�@

error_R�wP?

learning_rate_1�9�@�H       ��H�	���L���A
*;


total_lossqv�@

error_RT?

learning_rate_1�9�A\�H       ��H�	�4�L���A*;


total_loss���@

error_R��O?

learning_rate_1�9�Uu�H       ��H�	�}�L���A*;


total_loss�<A

error_R��J?

learning_rate_1�9quӮH       ��H�	���L���A*;


total_lossᜡ@

error_REo=?

learning_rate_1�9N��H       ��H�	[�L���A*;


total_lossX��@

error_R7OH?

learning_rate_1�9Y,�H       ��H�	AP�L���A*;


total_loss�T�@

error_R@,\?

learning_rate_1�9b|a�H       ��H�	���L���A*;


total_loss���@

error_R�C?

learning_rate_1�9ʕ@"H       ��H�	x��L���A*;


total_loss�|�@

error_R�H?

learning_rate_1�9��6H       ��H�	,�L���A*;


total_loss��@

error_R�oG?

learning_rate_1�9?�7$H       ��H�	�t�L���A*;


total_loss�8�@

error_R�(5?

learning_rate_1�9��35H       ��H�	���L���A*;


total_lossO��@

error_Rq�Y?

learning_rate_1�9/�wH       ��H�	]1�L���A*;


total_loss
H�@

error_R-=N?

learning_rate_1�9kSH�H       ��H�	�y�L���A*;


total_loss1Y�@

error_R�>?

learning_rate_1�9Om��H       ��H�	���L���A*;


total_loss,´@

error_R��G?

learning_rate_1�9-���H       ��H�	��L���A*;


total_loss_��@

error_R�6M?

learning_rate_1�9o�@H       ��H�	�L�L���A*;


total_loss8?A

error_Rv�D?

learning_rate_1�9�	YH       ��H�	��L���A*;


total_loss���@

error_RIcE?

learning_rate_1�9�TmH       ��H�	���L���A*;


total_loss���@

error_R:l\?

learning_rate_1�9���H       ��H�	2�L���A*;


total_loss'A

error_Rt�G?

learning_rate_1�9���H       ��H�	P^�L���A*;


total_loss<�@

error_R�7H?

learning_rate_1�9N0�!H       ��H�	���L���A*;


total_loss_�@

error_R&O?

learning_rate_1�9B��@H       ��H�	S��L���A*;


total_lossE��@

error_R��Q?

learning_rate_1�9�ߎAH       ��H�	5�L���A *;


total_lossO�x@

error_R �L?

learning_rate_1�9� �2H       ��H�	��L���A!*;


total_lossρ�@

error_R�rN?

learning_rate_1�9��g'H       ��H�	p��L���A"*;


total_lossN~�@

error_R�N?

learning_rate_1�9�~q�H       ��H�	��L���A#*;


total_loss�/�@

error_R��X?

learning_rate_1�9����H       ��H�	�U�L���A$*;


total_loss%��@

error_R%�G?

learning_rate_1�9F�.�H       ��H�	���L���A%*;


total_loss��@

error_Rd�I?

learning_rate_1�9�H�H       ��H�	���L���A&*;


total_loss�[�@

error_RI�A?

learning_rate_1�9��H       ��H�	
%�L���A'*;


total_loss���@

error_RT?

learning_rate_1�9gt=�H       ��H�	Wi�L���A(*;


total_loss4��@

error_R,�@?

learning_rate_1�9Z�H       ��H�	���L���A)*;


total_loss1��@

error_R�mT?

learning_rate_1�9`���H       ��H�	���L���A**;


total_loss��@

error_ROxK?

learning_rate_1�9�+,�H       ��H�	�7�L���A+*;


total_loss�>�@

error_R��F?

learning_rate_1�9�Y� H       ��H�	S{�L���A,*;


total_loss�V�@

error_R��b?

learning_rate_1�9����H       ��H�	g��L���A-*;


total_lossϵ�@

error_Rr??

learning_rate_1�9)Q�H       ��H�	
�L���A.*;


total_loss3ľ@

error_R��G?

learning_rate_1�9�}�NH       ��H�	3T�L���A/*;


total_loss�N�@

error_R��L?

learning_rate_1�9S��MH       ��H�	<��L���A0*;


total_loss���@

error_R��_?

learning_rate_1�92ɍ�H       ��H�	v��L���A1*;


total_loss
��@

error_R��J?

learning_rate_1�9	�"�H       ��H�	�%�L���A2*;


total_loss�,�@

error_R��[?

learning_rate_1�94��)H       ��H�	�n�L���A3*;


total_loss��@

error_RXbC?

learning_rate_1�9��H       ��H�	��L���A4*;


total_loss�ּ@

error_R;F?

learning_rate_1�9�x�H       ��H�	1��L���A5*;


total_losscq�@

error_R�V?

learning_rate_1�9����H       ��H�	LH�L���A6*;


total_lossF��@

error_R��D?

learning_rate_1�91m
�H       ��H�	h��L���A7*;


total_loss}6�@

error_RO�N?

learning_rate_1�9�٢�H       ��H�	���L���A8*;


total_loss
Ú@

error_R�[\?

learning_rate_1�9�\ H       ��H�	p�L���A9*;


total_loss��@

error_RcW?

learning_rate_1�9�|8�H       ��H�	�b�L���A:*;


total_loss���@

error_R�`J?

learning_rate_1�9���H       ��H�	)��L���A;*;


total_lossr��@

error_R1�6?

learning_rate_1�9S�ԊH       ��H�	R��L���A<*;


total_loss���@

error_R��X?

learning_rate_1�93�}H       ��H�	�<�L���A=*;


total_lossS�@

error_RM ??

learning_rate_1�9AW�H       ��H�	���L���A>*;


total_lossE]�@

error_R�'H?

learning_rate_1�9����H       ��H�	I��L���A?*;


total_loss�^A

error_R6R?

learning_rate_1�9!�NH       ��H�	�4�L���A@*;


total_loss$_�@

error_Rd�F?

learning_rate_1�9��~�H       ��H�	���L���AA*;


total_loss+|	A

error_Rv}e?

learning_rate_1�94X�H       ��H�	��L���AB*;


total_lossE��@

error_R�\?

learning_rate_1�9Q$��H       ��H�	� �L���AC*;


total_loss�%�@

error_R3XH?

learning_rate_1�9-3}H       ��H�	�d�L���AD*;


total_loss-�A

error_R�XS?

learning_rate_1�9��z-H       ��H�	A��L���AE*;


total_loss���@

error_R�Z?

learning_rate_1�9�X�H       ��H�	,��L���AF*;


total_lossJ��@

error_Rw�L?

learning_rate_1�9��>�H       ��H�	�A�L���AG*;


total_loss�@

error_Rf�8?

learning_rate_1�9��؎H       ��H�	y��L���AH*;


total_loss��@

error_RsF?

learning_rate_1�9���H       ��H�	���L���AI*;


total_loss��@

error_R�F@?

learning_rate_1�9�-tH       ��H�	-�L���AJ*;


total_loss ��@

error_R�Y?

learning_rate_1�9��H       ��H�	�W�L���AK*;


total_loss<8�@

error_R�dA?

learning_rate_1�9��H       ��H�	���L���AL*;


total_loss$�@

error_RT�T?

learning_rate_1�9F.��H       ��H�	��L���AM*;


total_loss�+�@

error_R��H?

learning_rate_1�9�x�hH       ��H�	%�L���AN*;


total_loss��@

error_RÚS?

learning_rate_1�9 �yH       ��H�	Xq�L���AO*;


total_loss��@

error_RCk?

learning_rate_1�90.��H       ��H�	.��L���AP*;


total_loss�P%A

error_R��E?

learning_rate_1�9�|?:H       ��H�	
��L���AQ*;


total_loss�f�@

error_Rs�B?

learning_rate_1�9����H       ��H�	yI�L���AR*;


total_loss���@

error_RER?

learning_rate_1�9�$9GH       ��H�	r��L���AS*;


total_loss�-�@

error_R��\?

learning_rate_1�9�jH       ��H�	���L���AT*;


total_lossT�@

error_R��G?

learning_rate_1�9Yh�-H       ��H�	�'�L���AU*;


total_losss��@

error_R,9?

learning_rate_1�9�$��H       ��H�	�n�L���AV*;


total_lossƵ�@

error_R��U?

learning_rate_1�9n��H       ��H�	Q��L���AW*;


total_loss�A

error_R�\?

learning_rate_1�9��:H       ��H�	���L���AX*;


total_loss��@

error_R�{U?

learning_rate_1�95��qH       ��H�	�>�L���AY*;


total_lossx��@

error_R��S?

learning_rate_1�9��@xH       ��H�	��L���AZ*;


total_loss�bA

error_RiH?

learning_rate_1�9S��"H       ��H�	���L���A[*;


total_loss���@

error_R[�P?

learning_rate_1�95ac�H       ��H�	P�L���A\*;


total_loss�]@

error_R�R[?

learning_rate_1�9W�gH       ��H�	WW�L���A]*;


total_loss(�@

error_Rd�P?

learning_rate_1�9א��H       ��H�	���L���A^*;


total_loss���@

error_R?M?

learning_rate_1�9��d/H       ��H�	H��L���A_*;


total_loss�J�@

error_R�L?

learning_rate_1�9���H       ��H�	I.�L���A`*;


total_lossh7�@

error_R�_N?

learning_rate_1�9�-�H       ��H�	�t�L���Aa*;


total_loss�� A

error_R!.U?

learning_rate_1�9G۬H       ��H�	i��L���Ab*;


total_lossy#A

error_R�S?

learning_rate_1�9Z�vBH       ��H�	D��L���Ac*;


total_loss#n�@

error_R;�K?

learning_rate_1�9'^]�H       ��H�	pC�L���Ad*;


total_loss�s�@

error_R}$T?

learning_rate_1�9vH;	H       ��H�	���L���Ae*;


total_loss���@

error_RU?

learning_rate_1�9m�H       ��H�	[��L���Af*;


total_lossM�@

error_R�N?

learning_rate_1�9N|�CH       ��H�	��L���Ag*;


total_lossη�@

error_R�Z?

learning_rate_1�9�*�H       ��H�	q\�L���Ah*;


total_loss���@

error_Rv�\?

learning_rate_1�9��H       ��H�	J��L���Ai*;


total_loss�Z�@

error_R.�P?

learning_rate_1�9G�J�H       ��H�	m��L���Aj*;


total_lossD5A

error_RQB?

learning_rate_1�9�ˇ�H       ��H�	i!�L���Ak*;


total_loss!o�@

error_RT�S?

learning_rate_1�9-#Y)H       ��H�	bf�L���Al*;


total_loss�sA

error_R7�f?

learning_rate_1�9�}vH       ��H�	.��L���Am*;


total_lossQ��@

error_R�\^?

learning_rate_1�9Ӵ_�H       ��H�	k��L���An*;


total_lossDE�@

error_R�PQ?

learning_rate_1�9}F��H       ��H�	D1�L���Ao*;


total_loss;��@

error_R�_C?

learning_rate_1�9���H       ��H�	�~�L���Ap*;


total_loss���@

error_R�:[?

learning_rate_1�94��H       ��H�	���L���Aq*;


total_loss��A

error_RnTs?

learning_rate_1�9�,'�H       ��H�	s.�L���Ar*;


total_lossK]�@

error_RR]G?

learning_rate_1�9����H       ��H�	Py�L���As*;


total_lossG�@

error_R&�V?

learning_rate_1�9w�XBH       ��H�	���L���At*;


total_loss�z�@

error_R�K?

learning_rate_1�9e,��H       ��H�	��L���Au*;


total_lossȒ�@

error_R�xO?

learning_rate_1�9�.��H       ��H�	�Y�L���Av*;


total_lossn��@

error_R�(`?

learning_rate_1�9�_PH       ��H�	ߣ�L���Aw*;


total_lossqx�@

error_R/P?

learning_rate_1�9��1�H       ��H�	P��L���Ax*;


total_loss��@

error_R��Z?

learning_rate_1�99�H`H       ��H�	�J�L���Ay*;


total_loss�j�@

error_R��L?

learning_rate_1�9�q��H       ��H�	p��L���Az*;


total_loss��@

error_R�Q?

learning_rate_1�9��?WH       ��H�	���L���A{*;


total_loss$?A

error_R�J?

learning_rate_1�9�KPH       ��H�	�'�L���A|*;


total_lossh��@

error_R�K?

learning_rate_1�9�>�H       ��H�	�l�L���A}*;


total_loss�2A

error_R�U?

learning_rate_1�9��=H       ��H�	���L���A~*;


total_loss�:�@

error_R��L?

learning_rate_1�9٢��H       ��H�	���L���A*;


total_loss���@

error_RQ�@?

learning_rate_1�9KZ�I       6%�	M0�L���A�*;


total_lossc�@

error_Rc�N?

learning_rate_1�9�f�I       6%�	Tr�L���A�*;


total_loss��@

error_R��V?

learning_rate_1�9+slI       6%�	 ��L���A�*;


total_loss��g@

error_R�S?

learning_rate_1�9b���I       6%�	��L���A�*;


total_loss���@

error_R�>R?

learning_rate_1�9�ٝ�I       6%�	nL�L���A�*;


total_loss��@

error_R�(T?

learning_rate_1�9䔴I       6%�	��L���A�*;


total_lossΨ~@

error_Rd�Y?

learning_rate_1�9��J	I       6%�	���L���A�*;


total_loss`��@

error_R�MK?

learning_rate_1�9ѥ��I       6%�	��L���A�*;


total_lossE}�@

error_R!�V?

learning_rate_1�9�7�-I       6%�	)a�L���A�*;


total_loss�S�@

error_RT O?

learning_rate_1�9�N	,I       6%�	֣�L���A�*;


total_loss�DA

error_RT-B?

learning_rate_1�9i�I       6%�	1��L���A�*;


total_loss���@

error_R�3Q?

learning_rate_1�9U8�3I       6%�	 *�L���A�*;


total_lossb�A

error_R�U?

learning_rate_1�9�3�I       6%�	�l�L���A�*;


total_lossVG�@

error_R�j9?

learning_rate_1�9B�X�I       6%�	���L���A�*;


total_loss���@

error_R	�k?

learning_rate_1�9L��I       6%�	z��L���A�*;


total_loss���@

error_R<�T?

learning_rate_1�9�\�I       6%�	,F�L���A�*;


total_loss|�@

error_R��G?

learning_rate_1�96��I       6%�	D��L���A�*;


total_lossJ��@

error_R̶M?

learning_rate_1�9��K2I       6%�	���L���A�*;


total_loss��A

error_R�Y?

learning_rate_1�9'�2I       6%�	
�L���A�*;


total_loss�A

error_RڥV?

learning_rate_1�9��? I       6%�	c�L���A�*;


total_lossS$�@

error_RXVN?

learning_rate_1�9@���I       6%�	U��L���A�*;


total_loss�6�@

error_R�J?

learning_rate_1�9^�EI       6%�	���L���A�*;


total_loss<��@

error_R��M?

learning_rate_1�9TT\vI       6%�	37�L���A�*;


total_loss���@

error_R��O?

learning_rate_1�9���I       6%�	���L���A�*;


total_loss[�@

error_R�9P?

learning_rate_1�9�D�GI       6%�	���L���A�*;


total_lossH�@

error_R[�\?

learning_rate_1�9&��I       6%�	U�L���A�*;


total_loss4�A

error_R�_X?

learning_rate_1�9h�\�I       6%�	{b�L���A�*;


total_loss��@

error_R.�W?

learning_rate_1�99��I       6%�	߰�L���A�*;


total_lossoԝ@

error_R��Q?

learning_rate_1�9|ߌI       6%�	���L���A�*;


total_loss���@

error_R�_?

learning_rate_1�9�M]OI       6%�	�D�L���A�*;


total_loss�@

error_R{\?

learning_rate_1�9:��I       6%�	���L���A�*;


total_loss삢@

error_R��M?

learning_rate_1�97�^�I       6%�	���L���A�*;


total_loss��@

error_R��T?

learning_rate_1�9��I       6%�	r�L���A�*;


total_loss(�A

error_R�.U?

learning_rate_1�9�W{OI       6%�	�`�L���A�*;


total_losst�@

error_Rl�X?

learning_rate_1�9��#}I       6%�	���L���A�*;


total_loss���@

error_R�8E?

learning_rate_1�9>�dtI       6%�	M��L���A�*;


total_loss�,�@

error_Rv�T?

learning_rate_1�9�/[�I       6%�	�)�L���A�*;


total_loss�U�@

error_R1�P?

learning_rate_1�9[�zI       6%�	�l�L���A�*;


total_loss�<�@

error_R�<?

learning_rate_1�9b�I       6%�	{��L���A�*;


total_loss�?A

error_R�Q?

learning_rate_1�9��pI       6%�	v��L���A�*;


total_loss���@

error_Rx^?

learning_rate_1�95~��I       6%�	8�L���A�*;


total_loss�@

error_R�3m?

learning_rate_1�9���5I       6%�	���L���A�*;


total_loss
��@

error_Rc�K?

learning_rate_1�9.+`�I       6%�	`��L���A�*;


total_lossd�A

error_RQhd?

learning_rate_1�9�O	 I       6%�	��L���A�*;


total_lossy	A

error_RZ�W?

learning_rate_1�9y���I       6%�	\T�L���A�*;


total_lossE��@

error_R(]?

learning_rate_1�9Qö�I       6%�	ӛ�L���A�*;


total_loss��@

error_R@�I?

learning_rate_1�9/�ҖI       6%�	���L���A�*;


total_lossWԙ@

error_R�B?

learning_rate_1�9$`�I       6%�	�2�L���A�*;


total_lossp�@

error_R��X?

learning_rate_1�9���sI       6%�	ex�L���A�*;


total_loss;�A

error_R�[:?

learning_rate_1�9�T�_I       6%�	���L���A�*;


total_loss�{�@

error_R�tL?

learning_rate_1�9��ُI       6%�	��L���A�*;


total_loss%f@

error_R��I?

learning_rate_1�9LE	�I       6%�	gb�L���A�*;


total_loss���@

error_RX?

learning_rate_1�9u?.I       6%�	���L���A�*;


total_loss�DA

error_RN�C?

learning_rate_1�9�!�qI       6%�	x��L���A�*;


total_loss��@

error_R�G?

learning_rate_1�9��4{I       6%�	�:�L���A�*;


total_loss"�@

error_R�G?

learning_rate_1�9u��I       6%�	��L���A�*;


total_lossf��@

error_R=FS?

learning_rate_1�9͕��I       6%�	R��L���A�*;


total_loss��@

error_R�N?

learning_rate_1�9�c�LI       6%�		�L���A�*;


total_loss`��@

error_R��K?

learning_rate_1�9yp`�I       6%�	�Z�L���A�*;


total_loss�mA

error_R�7N?

learning_rate_1�9^��HI       6%�	R��L���A�*;


total_loss&��@

error_R
aS?

learning_rate_1�9�BCI       6%�	���L���A�*;


total_loss,��@

error_Rdw_?

learning_rate_1�9��z�I       6%�	A�L���A�*;


total_loss;D�@

error_Ra�d?

learning_rate_1�9`��I       6%�	i��L���A�*;


total_loss ��@

error_RF�J?

learning_rate_1�9J�#	I       6%�	#��L���A�*;


total_lossk�@

error_R.�F?

learning_rate_1�9E̰I       6%�	o$�L���A�*;


total_loss\!�@

error_Rm�B?

learning_rate_1�9&^+�I       6%�		m�L���A�*;


total_loss���@

error_R�9g?

learning_rate_1�9PA(�I       6%�	���L���A�*;


total_loss.�@

error_R��e?

learning_rate_1�9����I       6%�	��L���A�*;


total_loss�ލ@

error_R��C?

learning_rate_1�9�}$I       6%�	O�L���A�*;


total_loss[��@

error_R#�b?

learning_rate_1�9c��.I       6%�	Қ�L���A�*;


total_loss*�@

error_RL�O?

learning_rate_1�9�xI       6%�	���L���A�*;


total_loss�/�@

error_R�8?

learning_rate_1�9ߴ�tI       6%�	�#�L���A�*;


total_lossE�@

error_R��V?

learning_rate_1�9���I       6%�	�e�L���A�*;


total_loss�@

error_Re�P?

learning_rate_1�9.G�I       6%�	թ�L���A�*;


total_loss��@

error_R?e^?

learning_rate_1�9x8�I       6%�	���L���A�*;


total_lossvo�@

error_R�qW?

learning_rate_1�9Ӏ��I       6%�	3 M���A�*;


total_loss���@

error_R�%Q?

learning_rate_1�9�f�I       6%�	�z M���A�*;


total_lossڹA

error_R�*[?

learning_rate_1�9"yH-I       6%�	� M���A�*;


total_lossRj�@

error_R�\L?

learning_rate_1�9��V�I       6%�	<M���A�*;


total_lossm�@

error_R{L_?

learning_rate_1�9NsI       6%�	�TM���A�*;


total_losssϾ@

error_R�)L?

learning_rate_1�9���I       6%�	��M���A�*;


total_loss��@

error_Rs~B?

learning_rate_1�9�
eI       6%�	c�M���A�*;


total_loss��@

error_R��]?

learning_rate_1�9��I       6%�	�:M���A�*;


total_loss[��@

error_R8uM?

learning_rate_1�9nA{�I       6%�	M�M���A�*;


total_loss��@

error_R��Q?

learning_rate_1�9�BlOI       6%�	��M���A�*;


total_loss���@

error_R�L?

learning_rate_1�9vɓ�I       6%�	DM���A�*;


total_loss
��@

error_R��X?

learning_rate_1�9
��I       6%�	/]M���A�*;


total_loss�Ƴ@

error_R�_?

learning_rate_1�9���RI       6%�	�M���A�*;


total_lossHQ�@

error_R�H?

learning_rate_1�9����I       6%�	��M���A�*;


total_loss�ؓ@

error_R�:N?

learning_rate_1�9���jI       6%�	-3M���A�*;


total_loss�D{@

error_R��W?

learning_rate_1�9�Lu�I       6%�	�xM���A�*;


total_loss��@

error_R �b?

learning_rate_1�9j&�I       6%�	�M���A�*;


total_lossӸ�@

error_R)�@?

learning_rate_1�9\�4EI       6%�	�M���A�*;


total_loss���@

error_R$"=?

learning_rate_1�9��oZI       6%�	.LM���A�*;


total_loss��@

error_R
�F?

learning_rate_1�9���I       6%�	�M���A�*;


total_loss���@

error_R��T?

learning_rate_1�9����I       6%�	`�M���A�*;


total_loss��@

error_R�I?

learning_rate_1�9�<��I       6%�	*M���A�*;


total_loss��1A

error_R@�B?

learning_rate_1�9'�2lI       6%�	.ZM���A�*;


total_lossr��@

error_R�M?

learning_rate_1�9��I       6%�	�M���A�*;


total_loss���@

error_RS�H?

learning_rate_1�9��2I       6%�	��M���A�*;


total_loss��q@

error_RR/S?

learning_rate_1�9��I       6%�	~.M���A�*;


total_loss*
�@

error_R�\?

learning_rate_1�9�κII       6%�	�pM���A�*;


total_loss R�@

error_R�T?

learning_rate_1�94�0cI       6%�	��M���A�*;


total_lossà@

error_R�SB?

learning_rate_1�9H�=I       6%�	��M���A�*;


total_loss��A

error_R��P?

learning_rate_1�92��$I       6%�	�:M���A�*;


total_loss�Έ@

error_RRgJ?

learning_rate_1�9��I       6%�	�~M���A�*;


total_loss�|�@

error_R��J?

learning_rate_1�9���}I       6%�	�M���A�*;


total_lossӴ@

error_RE�Q?

learning_rate_1�9�/|�I       6%�	�	M���A�*;


total_lossɁ�@

error_R�K?

learning_rate_1�9��l�I       6%�	vm	M���A�*;


total_loss���@

error_R�&^?

learning_rate_1�9�#BI       6%�	��	M���A�*;


total_lossF��@

error_Rv�8?

learning_rate_1�9LZ��I       6%�	4�	M���A�*;


total_loss�H�@

error_R�S[?

learning_rate_1�9l�@I       6%�	�?
M���A�*;


total_loss�@

error_R�a?

learning_rate_1�95�]wI       6%�	�
M���A�*;


total_losso�@

error_R�'I?

learning_rate_1�9Ϝ��I       6%�	F�
M���A�*;


total_lossq	�@

error_R��Q?

learning_rate_1�9/���I       6%�	�M���A�*;


total_losslk�@

error_Rn�U?

learning_rate_1�9)쐢I       6%�	
bM���A�*;


total_loss��@

error_R
DT?

learning_rate_1�9�ԥ�I       6%�	c�M���A�*;


total_loss�a�@

error_RM�c?

learning_rate_1�9�bAZI       6%�	��M���A�*;


total_loss�R�@

error_RRT?

learning_rate_1�9-�A�I       6%�	�6M���A�*;


total_loss���@

error_R�rJ?

learning_rate_1�9�O�:I       6%�	�zM���A�*;


total_loss8�@

error_R
N?

learning_rate_1�9;��I       6%�	,�M���A�*;


total_loss$d�@

error_Rvh?

learning_rate_1�9���I       6%�	�M���A�*;


total_lossJ4�@

error_R�:Q?

learning_rate_1�9=3q�I       6%�	�\M���A�*;


total_loss|��@

error_R�^R?

learning_rate_1�9FX՘I       6%�	?�M���A�*;


total_loss���@

error_Rf�)?

learning_rate_1�9��@I       6%�	��M���A�*;


total_loss1�@

error_R1C?

learning_rate_1�9#��I       6%�	�.M���A�*;


total_loss�{~@

error_R k`?

learning_rate_1�9���I       6%�	$xM���A�*;


total_lossN)�@

error_R�EP?

learning_rate_1�9$!��I       6%�	�M���A�*;


total_loss��A

error_R�WI?

learning_rate_1�9�@I       6%�	|M���A�*;


total_loss�6�@

error_R2�O?

learning_rate_1�9%h�{I       6%�	�TM���A�*;


total_loss�� A

error_R�7X?

learning_rate_1�9Mf�^I       6%�	T�M���A�*;


total_lossa]�@

error_R�Y?

learning_rate_1�9�߾�I       6%�	'�M���A�*;


total_loss<8�@

error_R��=?

learning_rate_1�9�v�FI       6%�	�1M���A�*;


total_loss
e�@

error_R�N?

learning_rate_1�9���I       6%�	�M���A�*;


total_loss��@

error_R�r9?

learning_rate_1�9��0XI       6%�	��M���A�*;


total_lossm��@

error_R)OI?

learning_rate_1�9b�I       6%�	�M���A�*;


total_loss���@

error_RqgT?

learning_rate_1�9|b�I       6%�	\M���A�*;


total_loss]�@

error_R�RU?

learning_rate_1�9��X�I       6%�	��M���A�*;


total_loss���@

error_R�(_?

learning_rate_1�9���;I       6%�	��M���A�*;


total_loss���@

error_Rc-W?

learning_rate_1�9EI       6%�	b'M���A�*;


total_loss �A

error_R.cN?

learning_rate_1�9{\��I       6%�	DjM���A�*;


total_loss{�@

error_R��U?

learning_rate_1�9vc��I       6%�	��M���A�*;


total_loss��@

error_RJ�e?

learning_rate_1�9�O�]I       6%�	��M���A�*;


total_loss��@

error_R�]?

learning_rate_1�9�'�sI       6%�	�6M���A�*;


total_lossא�@

error_R��M?

learning_rate_1�9t��
I       6%�	{M���A�*;


total_loss��@

error_RX_?

learning_rate_1�9���7I       6%�	��M���A�*;


total_loss|O�@

error_Rl�X?

learning_rate_1�9��ڛI       6%�	�M���A�*;


total_loss��@

error_RE.W?

learning_rate_1�9"�7I       6%�	�XM���A�*;


total_loss_��@

error_R�Y?

learning_rate_1�9v���I       6%�	��M���A�*;


total_loss� �@

error_R@X?

learning_rate_1�96���I       6%�	M�M���A�*;


total_loss�n�@

error_R.�H?

learning_rate_1�9TZ��I       6%�	<'M���A�*;


total_loss��@

error_R�AR?

learning_rate_1�9?f�>I       6%�	�kM���A�*;


total_loss�[�@

error_R��>?

learning_rate_1�9w*{I       6%�	��M���A�*;


total_loss���@

error_R��Z?

learning_rate_1�9�9rI       6%�	1�M���A�*;


total_loss� A

error_R�Gb?

learning_rate_1�9z+7I       6%�	8;M���A�*;


total_loss!��@

error_R�O?

learning_rate_1�9|9�I       6%�	M�M���A�*;


total_lossП#A

error_R��D?

learning_rate_1�9c��SI       6%�	��M���A�*;


total_loss�@

error_R��A?

learning_rate_1�9{'B�I       6%�	M���A�*;


total_loss�V�@

error_R��R?

learning_rate_1�9�)"kI       6%�	�YM���A�*;


total_lossZA

error_R�8K?

learning_rate_1�91�
qI       6%�	U�M���A�*;


total_loss��@

error_R��Q?

learning_rate_1�9����I       6%�	�M���A�*;


total_loss���@

error_R�@P?

learning_rate_1�9L`��I       6%�	#3M���A�*;


total_loss*�@

error_R_�d?

learning_rate_1�9��bCI       6%�	�{M���A�*;


total_loss�p�@

error_RRTV?

learning_rate_1�9}�I       6%�	`�M���A�*;


total_loss�Z�@

error_R�F=?

learning_rate_1�9q�;I       6%�	M���A�*;


total_lossX��@

error_R�xL?

learning_rate_1�9%�#I       6%�	�sM���A�*;


total_lossW��@

error_R��Z?

learning_rate_1�9�vn�I       6%�	�M���A�*;


total_loss���@

error_R��L?

learning_rate_1�9y�,"I       6%�	&M���A�*;


total_loss86A

error_RIJ?

learning_rate_1�9�'� I       6%�	ZFM���A�*;


total_loss���@

error_R}�I?

learning_rate_1�9�d�I       6%�	�M���A�*;


total_loss��@

error_R�b?

learning_rate_1�9'o��I       6%�	��M���A�*;


total_loss���@

error_R��@?

learning_rate_1�9t�P�I       6%�	�!M���A�*;


total_loss�e�@

error_R߱??

learning_rate_1�9��hI       6%�	�iM���A�*;


total_loss#��@

error_RJj=?

learning_rate_1�9�G,I       6%�	��M���A�*;


total_loss-��@

error_R�K?

learning_rate_1�9拊�I       6%�	��M���A�*;


total_loss�y�@

error_RD6^?

learning_rate_1�9�޺I       6%�	�4M���A�*;


total_loss�@

error_R�@?

learning_rate_1�9�!U�I       6%�	�xM���A�*;


total_loss.��@

error_R��O?

learning_rate_1�9���"I       6%�	Y�M���A�*;


total_loss6�@

error_R1 =?

learning_rate_1�9_�^ZI       6%�	��M���A�*;


total_loss�ӆ@

error_R�R\?

learning_rate_1�9��I       6%�	:EM���A�*;


total_loss3��@

error_R�CN?

learning_rate_1�9���|I       6%�	 �M���A�*;


total_loss .�@

error_R�FQ?

learning_rate_1�9ɡ9�I       6%�	��M���A�*;


total_lossR�@

error_R�E?

learning_rate_1�9�!7�I       6%�	h"M���A�*;


total_loss�&�@

error_R�yL?

learning_rate_1�9ˌ�I       6%�	�eM���A�*;


total_loss�آ@

error_R��8?

learning_rate_1�93nW�I       6%�	��M���A�*;


total_lossM�A

error_R��X?

learning_rate_1�9�*�I       6%�	��M���A�*;


total_loss�F�@

error_R��P?

learning_rate_1�9�P�I       6%�	�/M���A�*;


total_loss�T�@

error_R��R?

learning_rate_1�9�ӤI       6%�	�qM���A�*;


total_loss���@

error_RME]?

learning_rate_1�9K�I       6%�	߲M���A�*;


total_lossfЧ@

error_R8�J?

learning_rate_1�9ҁ�7I       6%�	�M���A�*;


total_lossm��@

error_RES?

learning_rate_1�9��htI       6%�	�: M���A�*;


total_loss���@

error_R�[?

learning_rate_1�9j���I       6%�	� M���A�*;


total_lossl��@

error_R��Q?

learning_rate_1�9\9�fI       6%�	�� M���A�*;


total_loss�۫@

error_R��W?

learning_rate_1�9��
tI       6%�	�!M���A�*;


total_loss�6�@

error_R��U?

learning_rate_1�9�xI       6%�	�M!M���A�*;


total_loss�=�@

error_R;�B?

learning_rate_1�9�ՌyI       6%�	�!M���A�*;


total_loss[��@

error_R�*A?

learning_rate_1�9K�y�I       6%�	��!M���A�*;


total_loss�25A

error_R7P?

learning_rate_1�9(�}<I       6%�	�"M���A�*;


total_loss���@

error_R� ]?

learning_rate_1�9QxȫI       6%�	�a"M���A�*;


total_loss�%�@

error_R6\?

learning_rate_1�9��̤I       6%�	��"M���A�*;


total_loss*`�@

error_RT�a?

learning_rate_1�9�P�I       6%�	^�"M���A�*;


total_loss�[A

error_R�OE?

learning_rate_1�9�C3�I       6%�	�5#M���A�*;


total_loss��@

error_R#�G?

learning_rate_1�9blL�I       6%�	�x#M���A�*;


total_lossI��@

error_R3`F?

learning_rate_1�9~PI       6%�	:�#M���A�*;


total_lossT�@

error_R�DG?

learning_rate_1�9�8�I       6%�	�$M���A�*;


total_lossH��@

error_Rm�L?

learning_rate_1�9�d�I       6%�	;N$M���A�*;


total_loss���@

error_R�nK?

learning_rate_1�9 ��I       6%�	x�$M���A�*;


total_loss�\A

error_R�E?

learning_rate_1�9��9I       6%�	�$M���A�*;


total_lossa2�@

error_Rse:?

learning_rate_1�9R�II       6%�	1%M���A�*;


total_loss��@

error_R)�M?

learning_rate_1�9��A�I       6%�	a%M���A�*;


total_loss��@

error_R�JJ?

learning_rate_1�9�C�I       6%�	��%M���A�*;


total_loss�r�@

error_R}bB?

learning_rate_1�9ʩp�I       6%�	8�%M���A�*;


total_loss��@

error_R��Z?

learning_rate_1�9��{�I       6%�	7+&M���A�*;


total_loss��@

error_R�Z?

learning_rate_1�9Ө-dI       6%�	�r&M���A�*;


total_loss.Y�@

error_R��J?

learning_rate_1�9�J+�I       6%�	?�&M���A�*;


total_loss�h�@

error_R��T?

learning_rate_1�9P'�I       6%�	2 'M���A�*;


total_loss]H�@

error_R-�P?

learning_rate_1�9��+TI       6%�	|G'M���A�*;


total_lossR��@

error_R�TF?

learning_rate_1�98��cI       6%�	��'M���A�*;


total_loss�!�@

error_R8M?

learning_rate_1�9v��I       6%�	�'M���A�*;


total_lossӱ@

error_R�LS?

learning_rate_1�9��"rI       6%�	�"(M���A�*;


total_loss��@

error_Rf�S?

learning_rate_1�9��=I       6%�	|(M���A�*;


total_loss���@

error_R&wU?

learning_rate_1�9/S��I       6%�	��(M���A�*;


total_loss��@

error_R�JO?

learning_rate_1�9P�=I       6%�	e+)M���A�*;


total_loss�{�@

error_RLR?

learning_rate_1�9�U�I       6%�	�)M���A�*;


total_lossR<�@

error_R�nO?

learning_rate_1�9����I       6%�	��)M���A�*;


total_loss�V�@

error_R�V?

learning_rate_1�9ДXI       6%�	S?*M���A�*;


total_loss���@

error_RC�Y?

learning_rate_1�9 ��xI       6%�	*�*M���A�*;


total_loss��A

error_R�\a?

learning_rate_1�9'Us�I       6%�	��*M���A�*;


total_loss��@

error_R�L?

learning_rate_1�9�R�cI       6%�	�+M���A�*;


total_loss:Ư@

error_R��??

learning_rate_1�9��+I       6%�	�U+M���A�*;


total_loss%d�@

error_R�N?

learning_rate_1�9���>I       6%�	y�+M���A�*;


total_loss�j�@

error_R�T?

learning_rate_1�9�N
I       6%�	��+M���A�*;


total_lossJ��@

error_R�t?

learning_rate_1�9�O�I       6%�	�$,M���A�*;


total_loss��@

error_R�F?

learning_rate_1�9�-^I       6%�	n,M���A�*;


total_loss=>�@

error_R*�P?

learning_rate_1�9��dI       6%�	��,M���A�*;


total_loss�R�@

error_RZL?

learning_rate_1�9L�`I       6%�	��,M���A�*;


total_loss��@

error_Rq�;?

learning_rate_1�9{23I       6%�	�<-M���A�*;


total_loss�#�@

error_R=�b?

learning_rate_1�92�6�I       6%�	�}-M���A�*;


total_lossq��@

error_R��\?

learning_rate_1�9��2I       6%�	��-M���A�*;


total_loss!M�@

error_RC[?

learning_rate_1�9���I       6%�	�.M���A�*;


total_loss�mA

error_RV[g?

learning_rate_1�9&�ܦI       6%�	�G.M���A�*;


total_loss{�@

error_R�lm?

learning_rate_1�9�,I       6%�	��.M���A�*;


total_loss~n@

error_R�qL?

learning_rate_1�9U2�uI       6%�	��.M���A�*;


total_lossHA�@

error_R�DY?

learning_rate_1�9�6q�I       6%�	�/M���A�*;


total_loss!(�@

error_R�5?

learning_rate_1�9G�3I       6%�	@X/M���A�*;


total_loss�A

error_RHH?

learning_rate_1�9�W�I       6%�	�/M���A�*;


total_loss/g�@

error_R T?

learning_rate_1�9��_�I       6%�	��/M���A�*;


total_loss<��@

error_R_�^?

learning_rate_1�9�ƭI       6%�	+0M���A�*;


total_loss=g�@

error_R_�S?

learning_rate_1�9m�=�I       6%�	\p0M���A�*;


total_loss�/�@

error_R��??

learning_rate_1�9�F�\I       6%�	}�0M���A�*;


total_loss��@

error_R�yN?

learning_rate_1�9j�.I       6%�	�1M���A�*;


total_loss�g�@

error_R��L?

learning_rate_1�9!�I       6%�	iL1M���A�*;


total_losso��@

error_R��C?

learning_rate_1�9�OHwI       6%�	~�1M���A�*;


total_loss�ѹ@

error_R,�e?

learning_rate_1�9�G��I       6%�	��1M���A�*;


total_loss3dA

error_R	�>?

learning_rate_1�9��JUI       6%�	�(2M���A�*;


total_loss�'�@

error_R�!M?

learning_rate_1�9k���I       6%�	bn2M���A�*;


total_loss7��@

error_R��I?

learning_rate_1�9�If�I       6%�	��2M���A�*;


total_lossr��@

error_RX/L?

learning_rate_1�9��I       6%�	�2M���A�*;


total_loss*�@

error_R�UI?

learning_rate_1�9v��
I       6%�	>93M���A�*;


total_lossv�@

error_RR?

learning_rate_1�9���KI       6%�	Q|3M���A�*;


total_loss���@

error_R3�W?

learning_rate_1�9SI       6%�	¾3M���A�*;


total_lossͭ�@

error_R��P?

learning_rate_1�9�m�bI       6%�	�4M���A�*;


total_loss��@

error_R_8C?

learning_rate_1�9�KGI       6%�	;G4M���A�*;


total_loss��@

error_R��E?

learning_rate_1�9���I       6%�	�4M���A�*;


total_losshL�@

error_Rd�Q?

learning_rate_1�9�]�I       6%�	0�4M���A�*;


total_lossթ@

error_R�.b?

learning_rate_1�9���NI       6%�	�5M���A�*;


total_loss$�@

error_R�U?

learning_rate_1�9pm�~I       6%�	,`5M���A�*;


total_lossq��@

error_R�M?

learning_rate_1�9P<�I       6%�	V�5M���A�*;


total_loss|x�@

error_R3SF?

learning_rate_1�9�ILI       6%�	q�5M���A�*;


total_lossmF�@

error_R;<?

learning_rate_1�9d�}�I       6%�	�16M���A�*;


total_loss��@

error_R�C?

learning_rate_1�9!���I       6%�	�v6M���A�*;


total_lossc��@

error_R�CM?

learning_rate_1�9A���I       6%�	��6M���A�*;


total_loss���@

error_R��[?

learning_rate_1�9���I       6%�	?�6M���A�*;


total_lossƕ@

error_R\K?

learning_rate_1�9�S6I       6%�	B?7M���A�*;


total_loss�c�@

error_RF�T?

learning_rate_1�9ژL�I       6%�	\�7M���A�*;


total_loss���@

error_ROQ?

learning_rate_1�9s!�I       6%�	z�7M���A�*;


total_lossF_�@

error_RC�F?

learning_rate_1�9n}I       6%�	98M���A�*;


total_loss�Ʃ@

error_RRKQ?

learning_rate_1�9�(I       6%�	JK8M���A�*;


total_loss,I�@

error_R�N?

learning_rate_1�9�~�I       6%�	7�8M���A�*;


total_loss8�@

error_R
�U?

learning_rate_1�9^��I       6%�	��8M���A�*;


total_loss�ֿ@

error_RV�??

learning_rate_1�9�8j�I       6%�	h,9M���A�*;


total_lossnP�@

error_R��D?

learning_rate_1�9q�i�I       6%�	�9M���A�*;


total_lossM{A

error_Rn�W?

learning_rate_1�9���vI       6%�	��9M���A�*;


total_loss�A

error_R��H?

learning_rate_1�9poWI       6%�	z:M���A�*;


total_lossc��@

error_R�G?

learning_rate_1�9z��+I       6%�	.d:M���A�*;


total_losssI�@

error_R��X?

learning_rate_1�9 z�I       6%�	U�:M���A�*;


total_loss4��@

error_R� e?

learning_rate_1�9��I       6%�	��:M���A�*;


total_loss�z�@

error_R$<?

learning_rate_1�9WxgI       6%�	>/;M���A�*;


total_loss�A�@

error_RwU?

learning_rate_1�9�&�I       6%�	��;M���A�*;


total_loss��A

error_R�T?

learning_rate_1�9o�nI       6%�	h<M���A�*;


total_lossx�@

error_R��`?

learning_rate_1�9=�`�I       6%�	uK<M���A�*;


total_loss$��@

error_R��K?

learning_rate_1�9�^ >I       6%�	��<M���A�*;


total_lossj��@

error_R�9=?

learning_rate_1�9���LI       6%�	H�<M���A�*;


total_lossi��@

error_R.O?

learning_rate_1�9�̨�I       6%�	I=M���A�*;


total_lossDI�@

error_RhG?

learning_rate_1�9{R�|I       6%�	a=M���A�*;


total_lossm��@

error_R\~L?

learning_rate_1�9cK��I       6%�	X�=M���A�*;


total_loss�J�@

error_Rf�_?

learning_rate_1�9��I       6%�	��=M���A�*;


total_lossN�@

error_R)PP?

learning_rate_1�9��)I       6%�	�9>M���A�*;


total_loss���@

error_R6�C?

learning_rate_1�9���I       6%�	.�>M���A�*;


total_lossF�@

error_R�
^?

learning_rate_1�9��I       6%�	s�>M���A�*;


total_lossO��@

error_R}�b?

learning_rate_1�9�2��I       6%�	K?M���A�*;


total_loss���@

error_R�E?

learning_rate_1�9�]I       6%�	G]?M���A�*;


total_loss��@

error_RP?

learning_rate_1�9!�YI       6%�	��?M���A�*;


total_loss<ŭ@

error_Rԛ@?

learning_rate_1�9Ί��I       6%�	7�?M���A�*;


total_loss;ө@

error_R��J?

learning_rate_1�9��4sI       6%�	�6@M���A�*;


total_loss./i@

error_R��{?

learning_rate_1�9?�	�I       6%�	�z@M���A�*;


total_loss1͹@

error_R�bO?

learning_rate_1�9����I       6%�	:�@M���A�*;


total_loss�j�@

error_R9K?

learning_rate_1�9WLR=I       6%�	_�@M���A�*;


total_loss��@

error_R�S?

learning_rate_1�9S�PI       6%�	q@AM���A�*;


total_loss�Ė@

error_Rz�R?

learning_rate_1�9���I       6%�	ڂAM���A�*;


total_lossf�@

error_Rx�S?

learning_rate_1�9A�X�I       6%�	S�AM���A�*;


total_lossIR�@

error_R�l`?

learning_rate_1�9p���I       6%�	-BM���A�*;


total_loss{,�@

error_R��[?

learning_rate_1�9����I       6%�	�QBM���A�*;


total_loss�o�@

error_R$(X?

learning_rate_1�9�ui�I       6%�	v�BM���A�*;


total_loss���@

error_R}Y?

learning_rate_1�9�GI       6%�	��BM���A�*;


total_loss���@

error_RD�T?

learning_rate_1�9Pk�I       6%�	�CM���A�*;


total_lossܶ�@

error_R�U?

learning_rate_1�9�qu�I       6%�	VdCM���A�*;


total_loss���@

error_R�G?

learning_rate_1�9�)]WI       6%�	��CM���A�*;


total_loss���@

error_R�+F?

learning_rate_1�9[�srI       6%�	a�CM���A�*;


total_loss�o�@

error_R)�Q?

learning_rate_1�9�COI       6%�	�0DM���A�*;


total_lossN�@

error_R3L?

learning_rate_1�9�I�#I       6%�	�qDM���A�*;


total_loss���@

error_R�*N?

learning_rate_1�9*}w�I       6%�	G�DM���A�*;


total_lossҬ@

error_R��R?

learning_rate_1�9M���I       6%�	�DM���A�*;


total_lossh�y@

error_RҩN?

learning_rate_1�9;���I       6%�	>EM���A�*;


total_lossj�@

error_R��3?

learning_rate_1�9�IlI       6%�	��EM���A�*;


total_loss]*�@

error_Rda?

learning_rate_1�9d���I       6%�	�EM���A�*;


total_loss�{�@

error_R1�V?

learning_rate_1�9f��I       6%�	PFM���A�*;


total_loss�8�@

error_RI�e?

learning_rate_1�9E�I       6%�	�HFM���A�*;


total_loss��@

error_RSC?

learning_rate_1�9���II       6%�	6�FM���A�*;


total_loss�̸@

error_R=�L?

learning_rate_1�9q���I       6%�	%�FM���A�*;


total_loss���@

error_RS=D?

learning_rate_1�9�;icI       6%�	�GM���A�*;


total_loss�4�@

error_R�X?

learning_rate_1�9?�u�I       6%�	�[GM���A�*;


total_loss,�@

error_R�cB?

learning_rate_1�9��/�I       6%�	�GM���A�*;


total_lossژ@

error_R�7R?

learning_rate_1�9����I       6%�	Y�GM���A�*;


total_loss��@

error_RԽ]?

learning_rate_1�9'�cI       6%�	'.HM���A�*;


total_loss��@

error_R��q?

learning_rate_1�9��d�I       6%�	�HM���A�*;


total_lossn��@

error_R��Y?

learning_rate_1�9�G��I       6%�	�HM���A�*;


total_lossI!�@

error_R�P?

learning_rate_1�9@,��I       6%�	k%IM���A�*;


total_loss��@

error_R��T?

learning_rate_1�9� ��I       6%�	��IM���A�*;


total_loss{A@

error_R1�S?

learning_rate_1�9
T�I       6%�	��IM���A�*;


total_loss3A

error_R��]?

learning_rate_1�9դQ�I       6%�	�3JM���A�*;


total_lossl|�@

error_R�/9?

learning_rate_1�9n
��I       6%�	}JM���A�*;


total_lossm�@

error_R�V`?

learning_rate_1�9C��nI       6%�	i�JM���A�*;


total_lossڬ�@

error_R(DP?

learning_rate_1�9C�[I       6%�	�KM���A�*;


total_loss�B�@

error_RC{I?

learning_rate_1�9�8�sI       6%�	hYKM���A�*;


total_lossӶ�@

error_R;�^?

learning_rate_1�9Zs�I       6%�	�KM���A�*;


total_loss��@

error_R�J?

learning_rate_1�9ig��I       6%�	4�KM���A�*;


total_loss���@

error_R��M?

learning_rate_1�9\�0NI       6%�	u,LM���A�*;


total_loss/*�@

error_R�N?

learning_rate_1�9�i��I       6%�	xLM���A�*;


total_loss_�@

error_RHV?

learning_rate_1�9lDxI       6%�	b�LM���A�*;


total_lossDi�@

error_RfJ?

learning_rate_1�9s�I       6%�	�MM���A�*;


total_lossMu�@

error_R��\?

learning_rate_1�9��uI       6%�	QJMM���A�*;


total_loss[��@

error_RT�J?

learning_rate_1�9|j��I       6%�	C�MM���A�*;


total_loss�̵@

error_R��M?

learning_rate_1�9�1�I       6%�	��MM���A�*;


total_lossC��@

error_R!%Q?

learning_rate_1�9wh��I       6%�	�NM���A�*;


total_loss8�@

error_R��J?

learning_rate_1�9�p��I       6%�	�aNM���A�*;


total_lossI=�@

error_Rn�U?

learning_rate_1�9���}I       6%�	ۣNM���A�*;


total_lossFކ@

error_R1G?

learning_rate_1�9��DHI       6%�	��NM���A�*;


total_loss`��@

error_R�G?

learning_rate_1�9-BעI       6%�	�.OM���A�*;


total_loss��@

error_R�T?

learning_rate_1�9���NI       6%�	tOM���A�*;


total_loss���@

error_R�HN?

learning_rate_1�9�so�I       6%�	y�OM���A�*;


total_loss$h@

error_R��a?

learning_rate_1�9��I       6%�	��OM���A�*;


total_loss$�@

error_R
f:?

learning_rate_1�9��4]I       6%�	?PM���A�*;


total_loss j�@

error_R:�6?

learning_rate_1�9��I       6%�	�PM���A�*;


total_loss؁�@

error_R�(`?

learning_rate_1�9D�D0I       6%�	f�PM���A�*;


total_loss���@

error_R�{P?

learning_rate_1�9g���I       6%�	�
QM���A�*;


total_loss�fA

error_R�1U?

learning_rate_1�9��?�I       6%�	MQM���A�*;


total_lossQ�@

error_RsS?

learning_rate_1�9�3�I       6%�	
�QM���A�*;


total_loss��@

error_R�N?

learning_rate_1�9Ý��I       6%�	��QM���A�*;


total_loss�G�@

error_R8qO?

learning_rate_1�9&���I       6%�	�RM���A�*;


total_loss_�@

error_R�BG?

learning_rate_1�9�.�>I       6%�	^RM���A�*;


total_loss�*�@

error_RI]H?

learning_rate_1�9h� �I       6%�	ڠRM���A�*;


total_loss ��@

error_R'S?

learning_rate_1�9���I       6%�	��RM���A�*;


total_lossv�@

error_RV?

learning_rate_1�9� ��I       6%�	�%SM���A�*;


total_loss�@

error_RH?

learning_rate_1�9j�CBI       6%�	�gSM���A�*;


total_loss�?v@

error_Rv�C?

learning_rate_1�9ㄦ�I       6%�	e�SM���A�*;


total_losso�@

error_R�K?

learning_rate_1�9c�PI       6%�	�SM���A�*;


total_loss<��@

error_R��7?

learning_rate_1�9O�dI       6%�	�7TM���A�*;


total_loss��@

error_R.7I?

learning_rate_1�9��Q�I       6%�	]}TM���A�*;


total_loss�@�@

error_R_�X?

learning_rate_1�9���I       6%�	=�TM���A�*;


total_loss��@

error_R�]?

learning_rate_1�9��i9I       6%�	'	UM���A�*;


total_lossM�@

error_R�$T?

learning_rate_1�9�'8"I       6%�	AQUM���A�*;


total_loss���@

error_R|�`?

learning_rate_1�9v;qI       6%�	t�UM���A�*;


total_loss��@

error_R�7]?

learning_rate_1�9L��MI       6%�	G�UM���A�*;


total_loss��@

error_R�?I?

learning_rate_1�9�v��I       6%�	A'VM���A�*;


total_loss��@

error_R�qH?

learning_rate_1�9�jBI       6%�	�oVM���A�*;


total_loss ��@

error_Rl�G?

learning_rate_1�9���fI       6%�	�VM���A�*;


total_loss�)�@

error_R��N?

learning_rate_1�9�%�~I       6%�	 WM���A�*;


total_loss�F�@

error_R!�i?

learning_rate_1�9�emI       6%�	LWM���A�*;


total_loss�1�@

error_R�
A?

learning_rate_1�9�c��I       6%�	��WM���A�*;


total_loss��@

error_R��^?

learning_rate_1�9��)(I       6%�	�WM���A�*;


total_loss-N�@

error_R�H?

learning_rate_1�9۪(;I       6%�	XM���A�*;


total_loss�`A

error_R6�b?

learning_rate_1�9�e9I       6%�	�[XM���A�*;


total_lossD�@

error_R�3F?

learning_rate_1�9K?�oI       6%�	��XM���A�*;


total_lossq}�@

error_R�i?

learning_rate_1�9	
I       6%�	�XM���A�*;


total_loss�چ@

error_R��Q?

learning_rate_1�9�>4�I       6%�	C3YM���A�*;


total_loss�Kx@

error_R�iu?

learning_rate_1�9��?I       6%�	0�YM���A�*;


total_loss�P�@

error_R5:?

learning_rate_1�9�pI       6%�	��YM���A�*;


total_lossI��@

error_R�#[?

learning_rate_1�9/�`�I       6%�	VZM���A�*;


total_loss���@

error_R�S?

learning_rate_1�9g$9�I       6%�	�_ZM���A�*;


total_loss���@

error_R�UV?

learning_rate_1�9���I       6%�	٧ZM���A�*;


total_lossJ�@

error_R;l??

learning_rate_1�9F)U�I       6%�	��ZM���A�*;


total_lossq[�@

error_R��K?

learning_rate_1�9��<�I       6%�	74[M���A�*;


total_lossf&�@

error_R�\?

learning_rate_1�9w�KaI       6%�	j~[M���A�*;


total_lossA��@

error_RֆL?

learning_rate_1�9��I       6%�	��[M���A�*;


total_loss�Ӷ@

error_R�!M?

learning_rate_1�9����I       6%�	/	\M���A�*;


total_lossѫ�@

error_RŢ\?

learning_rate_1�9&Oi�I       6%�	O\M���A�*;


total_loss�d�@

error_RJ
^?

learning_rate_1�9W�!�I       6%�	��\M���A�*;


total_loss���@

error_RC<N?

learning_rate_1�9$}�KI       6%�	i�\M���A�*;


total_loss]	�@

error_R=]H?

learning_rate_1�9Z\=uI       6%�	�"]M���A�*;


total_loss���@

error_R�UR?

learning_rate_1�9#�I       6%�	�w]M���A�*;


total_loss�@

error_R�QX?

learning_rate_1�9,%�HI       6%�	d�]M���A�*;


total_lossD�"A

error_R��>?

learning_rate_1�9�x��I       6%�	� ^M���A�*;


total_loss@"A

error_Rt�@?

learning_rate_1�9���I       6%�	�G^M���A�*;


total_loss�~|@

error_R͔U?

learning_rate_1�9k%4�I       6%�	�^M���A�*;


total_lossc-�@

error_R��T?

learning_rate_1�9c��I       6%�	]�^M���A�*;


total_lossg0A

error_R4�M?

learning_rate_1�9)Y�?I       6%�	�_M���A�*;


total_loss���@

error_R�CL?

learning_rate_1�9MERI       6%�	�j_M���A�*;


total_losslh@

error_R	�R?

learning_rate_1�9\�L�I       6%�	0�_M���A�*;


total_loss6!�@

error_R �B?

learning_rate_1�9�<8I       6%�	��_M���A�*;


total_loss�@

error_R��??

learning_rate_1�9�@KI       6%�	a2`M���A�*;


total_loss;�@

error_RI�c?

learning_rate_1�9���{I       6%�	�w`M���A�*;


total_loss���@

error_R�Q?

learning_rate_1�9��QI       6%�	/�`M���A�*;


total_loss-�A

error_Rt3L?

learning_rate_1�9�9��I       6%�	��`M���A�*;


total_loss�V�@

error_R��J?

learning_rate_1�9��)�I       6%�	�EaM���A�*;


total_lossM��@

error_R�Pb?

learning_rate_1�9�m(+I       6%�	3�aM���A�*;


total_losst��@

error_R��[?

learning_rate_1�9(�I       6%�	��aM���A�*;


total_loss4>�@

error_R�`H?

learning_rate_1�9��I       6%�	w"bM���A�*;


total_loss���@

error_R�QS?

learning_rate_1�9^��`I       6%�	�pbM���A�*;


total_loss��A

error_R�_?

learning_rate_1�9Ǽ�\I       6%�	q�bM���A�*;


total_loss�Po@

error_Rn�`?

learning_rate_1�9�ɲ�I       6%�	�cM���A�*;


total_loss3�A

error_R�L?

learning_rate_1�9q�(�I       6%�	EEcM���A�*;


total_loss�.�@

error_RH-L?

learning_rate_1�9���]I       6%�	��cM���A�*;


total_loss���@

error_R��F?

learning_rate_1�9_5*hI       6%�	�cM���A�*;


total_loss���@

error_Ra?

learning_rate_1�9���I       6%�	dM���A�*;


total_lossV.A

error_R�p]?

learning_rate_1�9oЎI       6%�	�XdM���A�*;


total_lossӕ�@

error_RׄG?

learning_rate_1�9_�[I       6%�	�dM���A�*;


total_lossou�@

error_R�na?

learning_rate_1�9�2` I       6%�	��dM���A�*;


total_loss0�A

error_R�*O?

learning_rate_1�9X��I       6%�	N/eM���A�*;


total_loss[n�@

error_R��I?

learning_rate_1�9����I       6%�	�ueM���A�*;


total_lossc��@

error_R�FR?

learning_rate_1�9�M��I       6%�	�eM���A�*;


total_loss�YA

error_RR�`?

learning_rate_1�9UоI       6%�	$ fM���A�*;


total_lossZ6�@

error_R�I?

learning_rate_1�9�^)	I       6%�	�CfM���A�*;


total_loss�V�@

error_R�Q?

learning_rate_1�9)O�'I       6%�	ʆfM���A�*;


total_loss��A

error_R_�Y?

learning_rate_1�9�7��I       6%�	 �fM���A�*;


total_loss��A

error_R�lJ?

learning_rate_1�9	��I       6%�	�gM���A�*;


total_loss��@

error_R74?

learning_rate_1�9��SI       6%�	YPgM���A�*;


total_loss���@

error_R�V?

learning_rate_1�9k��'I       6%�	{�gM���A�*;


total_loss�ΰ@

error_R�FK?

learning_rate_1�9��B�I       6%�	)�gM���A�*;


total_loss��A

error_R��]?

learning_rate_1�95CGI       6%�	� hM���A�*;


total_loss@��@

error_RQ?

learning_rate_1�9:�[�I       6%�	�whM���A�*;


total_lossI�@

error_R,�D?

learning_rate_1�9Q�{I       6%�	��hM���A�*;


total_loss�ܬ@

error_Rv�J?

learning_rate_1�9_zI       6%�	�iM���A�*;


total_loss�n�@

error_R0Q?

learning_rate_1�9�椌I       6%�	��iM���A�*;


total_loss�Ͼ@

error_R��Z?

learning_rate_1�9���iI       6%�	��iM���A�*;


total_loss�@

error_R�0A?

learning_rate_1�9
�Z�I       6%�	�jM���A�*;


total_loss�W�@

error_R�S?

learning_rate_1�9�Q��I       6%�	hcjM���A�*;


total_loss��@

error_RacT?

learning_rate_1�9��iI       6%�	��jM���A�*;


total_loss�e�@

error_R�iJ?

learning_rate_1�9*ج#I       6%�	��jM���A�*;


total_loss��@

error_Rf�L?

learning_rate_1�9G�58I       6%�	1kM���A�*;


total_loss"��@

error_R�E?

learning_rate_1�9��|I       6%�	�ukM���A�*;


total_loss�[	A

error_RfH?

learning_rate_1�9=
2�I       6%�	иkM���A�*;


total_lossy�@

error_RȘP?

learning_rate_1�9\�V�I       6%�	_�kM���A�*;


total_loss�ݭ@

error_RҔN?

learning_rate_1�9U��I       6%�	fClM���A�*;


total_lossX��@

error_R�C?

learning_rate_1�9��M{I       6%�	 �lM���A�*;


total_loss���@

error_R�	]?

learning_rate_1�9 ֊�I       6%�	��lM���A�*;


total_loss���@

error_R�J?

learning_rate_1�9�1)eI       6%�	�mM���A�*;


total_lossA

error_R��_?

learning_rate_1�9P���I       6%�	�[mM���A�*;


total_loss@

error_R�9B?

learning_rate_1�9�9s�I       6%�	��mM���A�*;


total_loss=��@

error_R�F?

learning_rate_1�9��I       6%�	��mM���A�*;


total_lossGA

error_RM-W?

learning_rate_1�9�ɨ�I       6%�	�-nM���A�*;


total_loss��k@

error_Rd�J?

learning_rate_1�9��8I       6%�	\rnM���A�*;


total_loss]|�@

error_R�:i?

learning_rate_1�9&��I       6%�	��nM���A�*;


total_loss��@

error_R_Y?

learning_rate_1�9;esDI       6%�	d�nM���A�*;


total_lossYJ	A

error_R@c^?

learning_rate_1�9ZtL�I       6%�	aCoM���A�*;


total_loss4R�@

error_R��V?

learning_rate_1�9܂I       6%�	�oM���A�*;


total_loss�=�@

error_R��K?

learning_rate_1�9�Z��I       6%�	��oM���A�*;


total_loss���@

error_RV2Z?

learning_rate_1�9��F�I       6%�	�pM���A�*;


total_loss��@

error_R��B?

learning_rate_1�90�I       6%�	�dpM���A�*;


total_loss?S�@

error_R�`?

learning_rate_1�9R�A�I       6%�	�pM���A�*;


total_loss�A

error_R@�N?

learning_rate_1�9W��%I       6%�	��pM���A�*;


total_loss��@

error_R��E?

learning_rate_1�9�b�$I       6%�	�9qM���A�*;


total_lossղ�@

error_R�_?

learning_rate_1�9�K��I       6%�	҂qM���A�*;


total_loss�8z@

error_R;U?

learning_rate_1�9(=�\I       6%�	�qM���A�*;


total_loss��@

error_Rs�W?

learning_rate_1�9�}�I       6%�	rM���A�*;


total_loss6�m@

error_R��F?

learning_rate_1�9��/I       6%�	@OrM���A�*;


total_loss���@

error_Rl�T?

learning_rate_1�9�7�!I       6%�	G�rM���A�*;


total_lossht�@

error_R�YN?

learning_rate_1�9I�eI       6%�	��rM���A�*;


total_loss�Z�@

error_R��K?

learning_rate_1�9��{�I       6%�	sM���A�*;


total_loss���@

error_R�vC?

learning_rate_1�9QNC�I       6%�	]sM���A�*;


total_loss,��@

error_RH�S?

learning_rate_1�9z�W�I       6%�	�sM���A�*;


total_loss/}�@

error_R�T_?

learning_rate_1�9qK�I       6%�	��sM���A�*;


total_loss���@

error_R��I?

learning_rate_1�9'P^`I       6%�	%1tM���A�*;


total_loss..�@

error_R�G?

learning_rate_1�9�q��I       6%�	W{tM���A�*;


total_loss��@

error_RCHK?

learning_rate_1�99S�~I       6%�	r�tM���A�*;


total_loss�d�@

error_R�4?

learning_rate_1�9bE%I       6%�	DuM���A�*;


total_loss2lA

error_Ri	a?

learning_rate_1�9YsoI       6%�	�SuM���A�*;


total_loss�R�@

error_Rx�K?

learning_rate_1�9�*�{I       6%�	�uM���A�*;


total_lossE�@

error_R�Jb?

learning_rate_1�9�EnI       6%�	�uM���A�*;


total_lossj~A

error_R5O?

learning_rate_1�9���DI       6%�	�!vM���A�*;


total_loss���@

error_R34/?

learning_rate_1�9����I       6%�	�evM���A�*;


total_loss�1�@

error_R��L?

learning_rate_1�9_k)�I       6%�	��vM���A�*;


total_lossl��@

error_R�0Q?

learning_rate_1�9���?I       6%�	�vM���A�*;


total_loss.�@

error_RH�M?

learning_rate_1�9�E��I       6%�	�;wM���A�*;


total_loss���@

error_R�sH?

learning_rate_1�9ak`�I       6%�	#wM���A�*;


total_loss�׭@

error_R�.P?

learning_rate_1�9T1(SI       6%�	y�wM���A�*;


total_loss�o�@

error_R�K?

learning_rate_1�9_�@aI       6%�	�xM���A�*;


total_lossC͞@

error_Rf�T?

learning_rate_1�9āI       6%�	�PxM���A�*;


total_loss�L�@

error_RV�J?

learning_rate_1�9�T�lI       6%�	z�xM���A�*;


total_loss��@

error_RL�G?

learning_rate_1�94&�I       6%�	��xM���A�*;


total_loss��h@

error_R=�9?

learning_rate_1�9=�^�I       6%�	�-yM���A�*;


total_loss�1�@

error_R
�U?

learning_rate_1�9r��I       6%�	��yM���A�*;


total_lossVo�@

error_RI>G?

learning_rate_1�9I       6%�	~�yM���A�*;


total_loss�F�@

error_R�Q?

learning_rate_1�9���hI       6%�	zM���A�*;


total_loss�j�@

error_R|X?

learning_rate_1�9(�!I       6%�	�UzM���A�*;


total_loss\�@

error_R6z8?

learning_rate_1�9;�@�I       6%�	W�zM���A�*;


total_loss=;�@

error_RlWN?

learning_rate_1�9�E�I       6%�	_�zM���A�*;


total_loss���@

error_Ra`K?

learning_rate_1�9mǿ~I       6%�	t,{M���A�*;


total_loss��A

error_R(uI?

learning_rate_1�9` �UI       6%�	p{M���A�*;


total_loss銡@

error_R�[?

learning_rate_1�9�#¿I       6%�	��{M���A�*;


total_loss�_�@

error_R�rR?

learning_rate_1�9�ٝI       6%�	�{M���A�*;


total_loss�KA

error_R��E?

learning_rate_1�95M4I       6%�	$?|M���A�*;


total_lossOv�@

error_RѢU?

learning_rate_1�9O��SI       6%�	Y�|M���A�*;


total_losss�A

error_Rwi?

learning_rate_1�9DuNI       6%�	��|M���A�*;


total_lossͫ�@

error_R��N?

learning_rate_1�9M���I       6%�	}M���A�*;


total_loss�B�@

error_R�%P?

learning_rate_1�9��-3I       6%�	oU}M���A�*;


total_losss��@

error_R_.Y?

learning_rate_1�9Ug�UI       6%�	9�}M���A�*;


total_loss�;�@

error_R*\?

learning_rate_1�9^��I       6%�	��}M���A�*;


total_loss���@

error_R3�O?

learning_rate_1�9�c�I       6%�	�2~M���A�*;


total_loss��@

error_R�L?

learning_rate_1�9_��wI       6%�	�w~M���A�*;


total_loss �t@

error_R6Da?

learning_rate_1�9��}�I       6%�	0�~M���A�*;


total_loss�f�@

error_R�LT?

learning_rate_1�9Vw׭I       6%�	�M���A�*;


total_loss��@

error_R��L?

learning_rate_1�9 �#�I       6%�	[JM���A�*;


total_loss ��@

error_RT�O?

learning_rate_1�9�]�I       6%�	D�M���A�*;


total_loss)�@

error_R��`?

learning_rate_1�9����I       6%�	��M���A�*;


total_loss���@

error_RF?

learning_rate_1�9�\�I       6%�	#$�M���A�*;


total_loss�x�@

error_RR�W?

learning_rate_1�9���RI       6%�	i�M���A�*;


total_loss�&�@

error_R6:T?

learning_rate_1�9�u
I       6%�	㭀M���A�*;


total_loss�Y�@

error_R��R?

learning_rate_1�9��I       6%�	e�M���A�*;


total_lossM��@

error_R@�<?

learning_rate_1�9 i�I       6%�	i;�M���A�*;


total_loss7�@

error_R��T?

learning_rate_1�9q��9I       6%�	���M���A�*;


total_lossq/�@

error_R�O?

learning_rate_1�9���DI       6%�	"́M���A�*;


total_loss��@

error_R�Q?

learning_rate_1�9��X�I       6%�	��M���A�*;


total_lossO��@

error_R,J?

learning_rate_1�9E�4�I       6%�	MV�M���A�*;


total_loss�UA

error_R��R?

learning_rate_1�9�tmwI       6%�	b��M���A�*;


total_loss�,�@

error_R7>M?

learning_rate_1�9)�Z�I       6%�	�قM���A�*;


total_loss<�@

error_R)�O?

learning_rate_1�9�ꙕI       6%�	��M���A�*;


total_loss�0�@

error_R��Z?

learning_rate_1�9fRHqI       6%�	�a�M���A�*;


total_loss��@

error_RZNK?

learning_rate_1�9�B��I       6%�	��M���A�*;


total_loss�@�@

error_R�K?

learning_rate_1�9�CPI       6%�	��M���A�*;


total_loss|�A

error_R��U?

learning_rate_1�9 AqPI       6%�	|1�M���A�*;


total_lossm��@

error_R��??

learning_rate_1�9Q�t�I       6%�	�y�M���A�*;


total_loss�W�@

error_R�0?

learning_rate_1�9c��I       6%�	+ƄM���A�*;


total_losss�@

error_R)R?

learning_rate_1�9y�TI       6%�	8�M���A�*;


total_loss1C�@

error_R��T?

learning_rate_1�9Fg�I       6%�	Q�M���A�*;


total_loss;�9A

error_R�xC?

learning_rate_1�9�11�I       6%�	���M���A�*;


total_loss,�A

error_R�M?

learning_rate_1�9�V�I       6%�	jօM���A�*;


total_loss�GA

error_RMH?

learning_rate_1�9�;/ZI       6%�	e�M���A�*;


total_loss�j�@

error_R{�Z?

learning_rate_1�9R�uI       6%�	._�M���A�*;


total_loss�/A

error_R �W?

learning_rate_1�9�s�I       6%�	n��M���A�*;


total_loss���@

error_R%�N?

learning_rate_1�9�/I       6%�	X�M���A�*;


total_loss�A

error_R]�O?

learning_rate_1�9�S-I       6%�	3�M���A�*;


total_lossE��@

error_RE�P?

learning_rate_1�9��I       6%�	�z�M���A�*;


total_loss�B�@

error_RCX?

learning_rate_1�9T��I       6%�	���M���A�*;


total_lossV�A

error_Ri@>?

learning_rate_1�9x�I       6%�	c�M���A�*;


total_loss��A

error_R�M?

learning_rate_1�9n�<	I       6%�	�\�M���A�*;


total_loss�m@

error_R:�@?

learning_rate_1�9�x�$I       6%�	*��M���A�*;


total_loss]n�@

error_RZ�T?

learning_rate_1�9ԹJ�I       6%�	��M���A�*;


total_loss���@

error_R��R?

learning_rate_1�9��4ZI       6%�	=�M���A�*;


total_loss]��@

error_R�\?

learning_rate_1�9��^�I       6%�	f��M���A�*;


total_loss.p�@

error_R�N?

learning_rate_1�9�4�I       6%�	l�M���A�*;


total_loss�@

error_R�gh?

learning_rate_1�9&�8%I       6%�	�7�M���A�*;


total_loss�.�@

error_R�6X?

learning_rate_1�9��S�I       6%�	�M���A�*;


total_loss
�@

error_R�?M?

learning_rate_1�9O]�I       6%�	0ˊM���A�*;


total_loss9a�@

error_R��P?

learning_rate_1�9��m5I       6%�	w�M���A�*;


total_loss�2�@

error_RMC?

learning_rate_1�9�}�I       6%�	K`�M���A�*;


total_loss1�@

error_RJ�Z?

learning_rate_1�9���BI       6%�	���M���A�*;


total_lossS��@

error_RxQ:?

learning_rate_1�9	�~�I       6%�	C�M���A�*;


total_loss��@

error_REC2?

learning_rate_1�9��LI       6%�	�8�M���A�*;


total_loss�,�@

error_R8T?

learning_rate_1�9�W�KI       6%�	}��M���A�*;


total_loss..x@

error_R��[?

learning_rate_1�9�R��I       6%�	�ǌM���A�*;


total_loss�H�@

error_ReV>?

learning_rate_1�9�iE�I       6%�	��M���A�*;


total_loss
A�@

error_R�_K?

learning_rate_1�9st	
I       6%�	$V�M���A�*;


total_loss�H�@

error_RqJ?

learning_rate_1�9�%�I       6%�	ڞ�M���A�*;


total_loss칐@

error_R?�O?

learning_rate_1�9g+9I       6%�	l�M���A�*;


total_losse��@

error_R��Y?

learning_rate_1�9Q03GI       6%�	�4�M���A�*;


total_loss���@

error_R�aW?

learning_rate_1�9��fuI       6%�	%{�M���A�*;


total_loss��)A

error_R,�O?

learning_rate_1�9�8�I       6%�	;��M���A�*;


total_loss�"�@

error_R��j?

learning_rate_1�9��h+I       6%�	 �M���A�*;


total_loss�s�@

error_R3_S?

learning_rate_1�9B\
!I       6%�	MD�M���A�*;


total_loss1aA

error_R��c?

learning_rate_1�9@�I�I       6%�	ێ�M���A�*;


total_lossf�@

error_R�W?

learning_rate_1�9ڪb�I       6%�	�׏M���A�*;


total_loss&�@

error_R{PW?

learning_rate_1�9��{�I       6%�	��M���A�*;


total_lossr%�@

error_R_�V?

learning_rate_1�9}���I       6%�	7h�M���A�*;


total_loss A

error_R �F?

learning_rate_1�9Q��?I       6%�	u��M���A�*;


total_loss���@

error_R,OY?

learning_rate_1�9�X;VI       6%�	2�M���A�*;


total_loss3��@

error_R�E?

learning_rate_1�9�P�I       6%�	�;�M���A�*;


total_loss�)�@

error_RS.K?

learning_rate_1�9� A~I       6%�	$��M���A�*;


total_loss�'�@

error_R�S?

learning_rate_1�9���`I       6%�	�ˑM���A�*;


total_loss��@

error_RHoV?

learning_rate_1�9�:�I       6%�	��M���A�*;


total_loss�0�@

error_R�L?

learning_rate_1�9�I       6%�	�W�M���A�*;


total_loss)�A

error_RחC?

learning_rate_1�9�Fy�I       6%�	���M���A�*;


total_loss ��@

error_R_iX?

learning_rate_1�9�q��I       6%�	#�M���A�*;


total_loss$y�@

error_R}qT?

learning_rate_1�9Q���I       6%�	v,�M���A�*;


total_loss*��@

error_R�5M?

learning_rate_1�9q�<I       6%�	o�M���A�*;


total_lossA;A

error_R��S?

learning_rate_1�9��k�I       6%�	���M���A�*;


total_lossA`�@

error_R�O?

learning_rate_1�9pD�mI       6%�	��M���A�*;


total_loss)J9A

error_R�^S?

learning_rate_1�9	��II       6%�	x?�M���A�*;


total_loss���@

error_RD�G?

learning_rate_1�9��I       6%�	���M���A�*;


total_loss���@

error_RZfX?

learning_rate_1�9dn�jI       6%�	oǔM���A�*;


total_loss͂�@

error_R��R?

learning_rate_1�9
V�I       6%�	�
�M���A�*;


total_lossTw�@

error_RڕR?

learning_rate_1�9�f��I       6%�	�O�M���A�*;


total_loss�6�@

error_R��V?

learning_rate_1�9��I       6%�	���M���A�*;


total_loss���@

error_R�,O?

learning_rate_1�9��kI       6%�	FߕM���A�*;


total_lossӽ�@

error_RJ?

learning_rate_1�9�7�I       6%�	�!�M���A�*;


total_loss=�@

error_R�L?

learning_rate_1�9��H7I       6%�	�h�M���A�*;


total_lossò�@

error_R�E_?

learning_rate_1�9�|zI       6%�	���M���A�*;


total_lossa'�@

error_R.V?

learning_rate_1�9�Ĥ�I       6%�	8��M���A�*;


total_loss3r�@

error_R�yU?

learning_rate_1�9���I       6%�	�C�M���A�*;


total_loss҂�@

error_R.AA?

learning_rate_1�9:e�I       6%�	3��M���A�*;


total_loss���@

error_R��P?

learning_rate_1�9��5$I       6%�	�ЗM���A�*;


total_loss�<�@

error_R��Q?

learning_rate_1�9���rI       6%�	2�M���A�*;


total_loss�I�@

error_R��Y?

learning_rate_1�9�8OkI       6%�	9[�M���A�*;


total_lossY@

error_ReE?

learning_rate_1�9��KI       6%�	&��M���A�*;


total_loss�D�@

error_R�];?

learning_rate_1�9���I       6%�	m�M���A�*;


total_loss���@

error_R^U?

learning_rate_1�9L�mUI       6%�	a9�M���A�*;


total_loss���@

error_R�4?

learning_rate_1�9h�r�I       6%�	N��M���A�*;


total_loss��@

error_R'P?

learning_rate_1�9�=�+I       6%�	���M���A�*;


total_loss)1�@

error_R�Ua?

learning_rate_1�9I�I       6%�	�)�M���A�*;


total_lossm��@

error_R�H?

learning_rate_1�9�)ÁI       6%�	�o�M���A�*;


total_loss��@

error_R N?

learning_rate_1�9���I       6%�	I��M���A�*;


total_lossܺ�@

error_RhM9?

learning_rate_1�9�]�I       6%�	���M���A�*;


total_lossA�@

error_R��N?

learning_rate_1�9�R��I       6%�	�?�M���A�*;


total_lossC�@

error_R�0h?

learning_rate_1�9`���I       6%�	���M���A�*;


total_loss���@

error_R�jW?

learning_rate_1�9gW�I       6%�	�śM���A�*;


total_loss(�@

error_Rq�5?

learning_rate_1�9���I       6%�	��M���A�*;


total_loss��@

error_R��R?

learning_rate_1�9j�I       6%�	DJ�M���A�*;


total_loss�Y�@

error_Rv�f?

learning_rate_1�9<��CI       6%�	���M���A�*;


total_loss��@

error_R�3`?

learning_rate_1�9
��I       6%�	F�M���A�*;


total_loss���@

error_R{�8?

learning_rate_1�9@
�I       6%�	�(�M���A�*;


total_lossxe�@

error_RdW?

learning_rate_1�9�_q�I       6%�	q�M���A�*;


total_loss2=�@

error_R?N?

learning_rate_1�9��
�I       6%�	>��M���A�*;


total_loss��@

error_R�C?

learning_rate_1�9�� I       6%�	���M���A�*;


total_loss�+�@

error_REhK?

learning_rate_1�9��ơI       6%�	�B�M���A�*;


total_loss	F�@

error_RRDH?

learning_rate_1�9b9�I       6%�	���M���A�*;


total_loss#�@

error_R�<?

learning_rate_1�9���I       6%�	מM���A�*;


total_loss�z�@

error_Rn�O?

learning_rate_1�9̬�0I       6%�	o�M���A�*;


total_loss=��@

error_R�`J?

learning_rate_1�9�HRI       6%�	�]�M���A�*;


total_lossV��@

error_R�rM?

learning_rate_1�9|�,I       6%�	E��M���A�*;


total_loss���@

error_RO|D?

learning_rate_1�9d���I       6%�	=�M���A�*;


total_loss�ş@

error_R�hF?

learning_rate_1�9��I       6%�	p+�M���A�*;


total_lossK�@

error_Rx�Z?

learning_rate_1�9��$�I       6%�	�p�M���A�*;


total_loss-�@

error_R�]C?

learning_rate_1�9gL-�I       6%�	��M���A�*;


total_loss=��@

error_R�yN?

learning_rate_1�9\�i^I       6%�	j��M���A�*;


total_loss̉�@

error_R�pI?

learning_rate_1�9mlb�I       6%�	�:�M���A�*;


total_loss�R�@

error_RN�]?

learning_rate_1�9C I       6%�	@|�M���A�*;


total_lossO��@

error_R2�H?

learning_rate_1�9�E+I       6%�	��M���A�*;


total_loss|��@

error_R��G?

learning_rate_1�9e�cI       6%�	
��M���A�*;


total_loss%�@

error_R@�J?

learning_rate_1�9~�+RI       6%�	A�M���A�*;


total_lossQ��@

error_R!WP?

learning_rate_1�90�X�I       6%�	t��M���A�*;


total_loss��@

error_R��]?

learning_rate_1�9��)I       6%�	�ʢM���A�*;


total_loss{��@

error_R6�^?

learning_rate_1�9+Y�I       6%�	k�M���A�*;


total_loss���@

error_R�[?

learning_rate_1�9�˻I       6%�	�X�M���A�*;


total_loss���@

error_R;"U?

learning_rate_1�9}χ�I       6%�	���M���A�*;


total_lossm��@

error_R�U?

learning_rate_1�9�ۀ�I       6%�	'�M���A�*;


total_loss�6A

error_R�H?

learning_rate_1�9�+5aI       6%�	�(�M���A�*;


total_loss��@

error_R!1@?

learning_rate_1�9��]I       6%�	�s�M���A�*;


total_lossT�y@

error_RA�R?

learning_rate_1�9
��I       6%�	¤M���A�*;


total_loss�BA

error_R��Z?

learning_rate_1�9�q�I       6%�	��M���A�*;


total_lossV�A

error_RæM?

learning_rate_1�9'i<�I       6%�	xU�M���A�*;


total_lossC��@

error_R��L?

learning_rate_1�9ՙ�I       6%�	r��M���A�*;


total_loss(��@

error_Rf_C?

learning_rate_1�9��r�I       6%�	�ڥM���A�*;


total_loss$��@

error_RZdV?

learning_rate_1�9g̀�I       6%�	��M���A�*;


total_loss�!�@

error_Rq$d?

learning_rate_1�9[ �I       6%�	�c�M���A�*;


total_loss�n�@

error_RJ�H?

learning_rate_1�9i˼�I       6%�	@��M���A�*;


total_lossT�@

error_R�Q?

learning_rate_1�9W��fI       6%�	�M���A�*;


total_loss/X�@

error_Rf(r?

learning_rate_1�9+�įI       6%�	&1�M���A�*;


total_loss��@

error_R�e9?

learning_rate_1�9�)VhI       6%�	�r�M���A�*;


total_loss��@

error_R�l?

learning_rate_1�9 ��I       6%�	h��M���A�*;


total_lossh�@

error_R��[?

learning_rate_1�9�++{I       6%�	3��M���A�*;


total_loss���@

error_RajO?

learning_rate_1�9��I       6%�	�@�M���A�*;


total_loss,�@

error_R�X?

learning_rate_1�9�LdI       6%�	�M���A�*;


total_loss��@

error_R��G?

learning_rate_1�97i�<I       6%�	}�M���A�*;


total_loss(��@

error_Rl�??

learning_rate_1�9�G��I       6%�	T+�M���A�*;


total_lossR�A

error_R�M[?

learning_rate_1�9�]ڲI       6%�	M���A�*;


total_loss��@

error_R1�N?

learning_rate_1�9l�wI       6%�	��M���A�*;


total_loss��@

error_R�U?

learning_rate_1�9 �X�I       6%�	C1�M���A�*;


total_loss�Ґ@

error_R�	Q?

learning_rate_1�9��mI       6%�	�w�M���A�*;


total_loss �@

error_R*JP?

learning_rate_1�9��@I       6%�	E��M���A�*;


total_loss���@

error_RF�S?

learning_rate_1�9�CI       6%�	��M���A�*;


total_losssD�@

error_R&Y?

learning_rate_1�9�C�zI       6%�	�L�M���A�*;


total_loss<�@

error_R
�Q?

learning_rate_1�98i�qI       6%�	T��M���A�*;


total_loss���@

error_RԃY?

learning_rate_1�9쏎%I       6%�	GԫM���A�*;


total_lossl��@

error_R&PM?

learning_rate_1�9Y%�I       6%�	�M���A�*;


total_lossu	A

error_RE?

learning_rate_1�9��QI       6%�	�a�M���A�*;


total_loss�ͩ@

error_R4�_?

learning_rate_1�9Y��I       6%�	m��M���A�*;


total_lossn��@

error_R JN?

learning_rate_1�9xC-�I       6%�	�M���A�*;


total_loss&I�@

error_R��g?

learning_rate_1�9��.I       6%�	=�M���A�*;


total_loss��@

error_R}O?

learning_rate_1�9ŬI       6%�	���M���A�*;


total_lossʎ@

error_R6�F?

learning_rate_1�98���I       6%�	�ЭM���A�*;


total_loss��@

error_RY=?

learning_rate_1�9��b"I       6%�	{�M���A�*;


total_loss]Q�@

error_R W9?

learning_rate_1�9cB��I       6%�	-\�M���A�*;


total_loss �A

error_R��Q?

learning_rate_1�9ݍI       6%�	���M���A�*;


total_lossO�A

error_RipA?

learning_rate_1�9���I       6%�	d�M���A�*;


total_loss�HA

error_R|�S?

learning_rate_1�9�!�I       6%�	%(�M���A�*;


total_loss-6�@

error_R1kY?

learning_rate_1�9���I       6%�	/s�M���A�*;


total_loss���@

error_RX2]?

learning_rate_1�9;8�I       6%�	���M���A�*;


total_loss�t�@

error_R��S?

learning_rate_1�9#ƅI       6%�	��M���A�*;


total_loss��@

error_RѧQ?

learning_rate_1�9q�UWI       6%�	�[�M���A�*;


total_loss�%�@

error_R��P?

learning_rate_1�9��_�I       6%�	"��M���A�*;


total_loss��@

error_R��<?

learning_rate_1�9�"��I       6%�	��M���A�*;


total_lossI.x@

error_RA`?

learning_rate_1�9��p�I       6%�	�8�M���A�*;


total_loss�c�@

error_R��A?

learning_rate_1�9v};I       6%�	�y�M���A�*;


total_lossʰ�@

error_RtCQ?

learning_rate_1�9��^nI       6%�	���M���A�*;


total_loss�@

error_RTO\?

learning_rate_1�9{��I       6%�	W�M���A�*;


total_loss�T�@

error_R��Q?

learning_rate_1�9?wd�I       6%�	!H�M���A�*;


total_loss���@

error_R��P?

learning_rate_1�9
�P8I       6%�	܌�M���A�*;


total_loss�&�@

error_R�I?

learning_rate_1�9I���I       6%�	nϲM���A�*;


total_loss�F�@

error_R��N?

learning_rate_1�9�گ}I       6%�	��M���A�*;


total_loss�a@

error_R=�K?

learning_rate_1�9V��I       6%�	)h�M���A�*;


total_loss؀~@

error_R}V>?

learning_rate_1�9�nH�I       6%�	���M���A�*;


total_loss��t@

error_R��<?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss���@

error_R��Q?

learning_rate_1�9׏��I       6%�	�;�M���A�*;


total_lossT	�@

error_RF�K?

learning_rate_1�9*ѪsI       6%�	�}�M���A�*;


total_loss�{�@

error_RWQ?

learning_rate_1�9���I       6%�	KŴM���A�*;


total_lossJ�@

error_R�cQ?

learning_rate_1�9RL0�I       6%�	��M���A�*;


total_loss��@

error_Rx�8?

learning_rate_1�9�b:�I       6%�	fM�M���A�*;


total_loss[��@

error_Rm�^?

learning_rate_1�9_6��I       6%�	���M���A�*;


total_loss�0�@

error_R�R^?

learning_rate_1�9 ���I       6%�	�صM���A�*;


total_loss�r�@

error_Re�H?

learning_rate_1�9���YI       6%�	! �M���A�*;


total_loss��@

error_RVE?

learning_rate_1�9�+l�I       6%�	d�M���A�*;


total_loss1C�@

error_RHT>?

learning_rate_1�9�:]GI       6%�	o��M���A�*;


total_loss��@

error_R�E@?

learning_rate_1�9��3I       6%�	��M���A�*;


total_losszwA

error_Rv�D?

learning_rate_1�9TL�xI       6%�	x,�M���A�*;


total_lossC1h@

error_RөB?

learning_rate_1�98�I       6%�	qn�M���A�*;


total_loss�A�@

error_R6�U?

learning_rate_1�9��I       6%�	���M���A�*;


total_loss���@

error_R��D?

learning_rate_1�9�UlI       6%�	���M���A�*;


total_loss��@

error_RZ?

learning_rate_1�9����I       6%�	K�M���A�*;


total_loss�"�@

error_R�/L?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss�Ŗ@

error_R�L??

learning_rate_1�9o��MI       6%�	dӸM���A�*;


total_loss�)�@

error_R6�B?

learning_rate_1�9?u�I       6%�	R�M���A�*;


total_loss��@

error_R�:T?

learning_rate_1�9:��I       6%�	w�M���A�*;


total_loss=�@

error_R�C?

learning_rate_1�9�M�FI       6%�	1��M���A�*;


total_loss[�@

error_R��A?

learning_rate_1�9�z�yI       6%�	�M���A�*;


total_loss�;�@

error_RxA?

learning_rate_1�9�7��I       6%�	CD�M���A�*;


total_lossO��@

error_R�dZ?

learning_rate_1�9*>�cI       6%�	��M���A�*;


total_lossIB�@

error_R��^?

learning_rate_1�9���I       6%�	IкM���A�*;


total_loss@

error_R=JP?

learning_rate_1�9<�g�I       6%�	>�M���A�*;


total_loss�Z�@

error_R/NP?

learning_rate_1�9��״I       6%�	Ee�M���A�*;


total_loss /�@

error_RiXJ?

learning_rate_1�9��k	I       6%�	��M���A�*;


total_loss�W�@

error_R��_?

learning_rate_1�9��XjI       6%�	��M���A�*;


total_loss@�@

error_Rlg^?

learning_rate_1�9��:UI       6%�	�-�M���A�*;


total_loss.�@

error_R�H?

learning_rate_1�9�n�I       6%�	@{�M���A�*;


total_loss�<�@

error_R��A?

learning_rate_1�9/�CI       6%�	%ǼM���A�*;


total_lossQ��@

error_R��A?

learning_rate_1�9:ǰ�I       6%�	#�M���A�*;


total_loss���@

error_R��<?

learning_rate_1�9�j$�I       6%�	DV�M���A�*;


total_lossS��@

error_R2�C?

learning_rate_1�9�zL�I       6%�	���M���A�*;


total_loss���@

error_RJ9]?

learning_rate_1�9���
I       6%�	�ܽM���A�*;


total_lossh|�@

error_R�G?

learning_rate_1�9}i	�I       6%�	K �M���A�*;


total_lossŏA

error_R�qZ?

learning_rate_1�9��I       6%�	�f�M���A�*;


total_loss*��@

error_R]xC?

learning_rate_1�9}�1I       6%�	6��M���A�*;


total_lossn��@

error_R�W?

learning_rate_1�9L[%�I       6%�	c�M���A�*;


total_loss6��@

error_R�I?

learning_rate_1�9n�ɰI       6%�	�-�M���A�*;


total_lossx��@

error_R��D?

learning_rate_1�9���I       6%�	�r�M���A�*;


total_loss{@�@

error_R�T?

learning_rate_1�9�j�I       6%�	���M���A�*;


total_loss�>�@

error_R�V?

learning_rate_1�9���I       6%�	���M���A�*;


total_loss�k�@

error_R��K?

learning_rate_1�9US�I       6%�	A=�M���A�*;


total_loss�s�@

error_R �[?

learning_rate_1�9�9ٳI       6%�	��M���A�*;


total_loss��}@

error_R��Q?

learning_rate_1�9��4�I       6%�	���M���A�*;


total_loss���@

error_RF1S?

learning_rate_1�9s���I       6%�	L�M���A�*;


total_loss���@

error_R�,Z?

learning_rate_1�9�A�HI       6%�	�f�M���A�*;


total_loss�[�@

error_R\�Q?

learning_rate_1�9Ɂ�I       6%�	K��M���A�*;


total_loss�b�@

error_R��J?

learning_rate_1�9��>I       6%�	���M���A�*;


total_loss(0�@

error_R�8d?

learning_rate_1�9D��I       6%�	4E�M���A�*;


total_loss�X�@

error_R�R?

learning_rate_1�9D~I       6%�	}��M���A�*;


total_loss���@

error_R�[?

learning_rate_1�9a���I       6%�	.��M���A�*;


total_loss��@

error_R�bS?

learning_rate_1�9y��I       6%�	e�M���A�*;


total_loss�x�@

error_R�N?

learning_rate_1�98W)�I       6%�	-e�M���A�*;


total_lossS�@

error_R�b?

learning_rate_1�9�'�I       6%�	.��M���A�*;


total_loss�l�@

error_R��J?

learning_rate_1�9�[cI       6%�	���M���A�*;


total_loss[��@

error_R\?

learning_rate_1�9���I       6%�	�,�M���A�*;


total_loss���@

error_R�H?

learning_rate_1�9 ��I       6%�	z�M���A�*;


total_lossx$A

error_RԋE?

learning_rate_1�9�Ei�I       6%�	���M���A�*;


total_lossd��@

error_R6[g?

learning_rate_1�9�W�I       6%�	��M���A�*;


total_loss���@

error_R��1?

learning_rate_1�9��I       6%�	�[�M���A�*;


total_lossߗ�@

error_R��H?

learning_rate_1�97S�NI       6%�	K��M���A�*;


total_lossP�@

error_RW�:?

learning_rate_1�9p9y�I       6%�	'��M���A�*;


total_loss*��@

error_RX�D?

learning_rate_1�9�զI       6%�	�-�M���A�*;


total_loss�Ш@

error_R��Y?

learning_rate_1�9e��`I       6%�	8u�M���A�*;


total_lossz��@

error_R 9X?

learning_rate_1�9|�ͰI       6%�	Y��M���A�*;


total_loss���@

error_R�R?

learning_rate_1�9Od��I       6%�	��M���A�*;


total_loss�}�@

error_R_;N?

learning_rate_1�9T�I       6%�	PR�M���A�*;


total_losss�@

error_R?�Q?

learning_rate_1�9�0��I       6%�	i��M���A�*;


total_loss�>�@

error_R��D?

learning_rate_1�9���I       6%�	���M���A�*;


total_loss�@

error_R�2J?

learning_rate_1�9����I       6%�	D�M���A�*;


total_loss�Z�@

error_R->^?

learning_rate_1�9DS��I       6%�	�y�M���A�*;


total_lossSg�@

error_RC�U?

learning_rate_1�9e�;�I       6%�	U��M���A�*;


total_loss���@

error_R[�`?

learning_rate_1�9Q4D�I       6%�	��M���A�*;


total_loss�{�@

error_R�QM?

learning_rate_1�93X**I       6%�	�t�M���A�*;


total_lossI�@

error_R�Oc?

learning_rate_1�9T(�\I       6%�	W��M���A�*;


total_loss���@

error_R�OS?

learning_rate_1�9k�I       6%�	#�M���A�*;


total_loss}��@

error_Ra�L?

learning_rate_1�9�52;I       6%�	h�M���A�*;


total_loss��@

error_R�"U?

learning_rate_1�9?ލ�I       6%�	���M���A�*;


total_loss�@

error_RmNN?

learning_rate_1�9=S1I       6%�	���M���A�*;


total_loss�x)A

error_R��\?

learning_rate_1�9F�yI       6%�	�4�M���A�*;


total_lossW��@

error_Rf0O?

learning_rate_1�9�W�+I       6%�	x�M���A�*;


total_loss��@

error_RϾ[?

learning_rate_1�9��N�I       6%�	4��M���A�*;


total_loss\��@

error_R��G?

learning_rate_1�9�>	8I       6%�	� �M���A�*;


total_loss1(�@

error_Rh�H?

learning_rate_1�9�м�I       6%�	�C�M���A�*;


total_lossn��@

error_R��D?

learning_rate_1�9��Q�I       6%�	���M���A�*;


total_lossO��@

error_R�aN?

learning_rate_1�9\ҶI       6%�	��M���A�*;


total_loss1Sp@

error_R�)L?

learning_rate_1�9�٩�I       6%�	�M���A�*;


total_lossB% A

error_R��??

learning_rate_1�9��,�I       6%�	�M�M���A�*;


total_loss��@

error_R��B?

learning_rate_1�9O�I       6%�	��M���A�*;


total_loss��@

error_RE[?

learning_rate_1�90�O�I       6%�	���M���A�*;


total_loss���@

error_RJ\O?

learning_rate_1�9y�qdI       6%�	��M���A�*;


total_loss��@A

error_R_F?

learning_rate_1�9���ZI       6%�	oc�M���A�*;


total_loss���@

error_R��P?

learning_rate_1�9P怴I       6%�	Ы�M���A�*;


total_loss�H�@

error_R�qL?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss���@

error_R�G?

learning_rate_1�9� sFI       6%�	�:�M���A�*;


total_loss�@

error_R�R?

learning_rate_1�9�
AI       6%�	���M���A�*;


total_loss�޺@

error_Rz�N?

learning_rate_1�9�)��I       6%�	q��M���A�*;


total_lossCG�@

error_R4�`?

learning_rate_1�9�9%I       6%�	@�M���A�*;


total_loss�eA

error_R�HP?

learning_rate_1�9����I       6%�	�M�M���A�*;


total_loss��@

error_R�T?

learning_rate_1�9!3DI       6%�	���M���A�*;


total_loss&��@

error_R��I?

learning_rate_1�9�E�cI       6%�	)��M���A�*;


total_losss�@

error_R��T?

learning_rate_1�9�xBI       6%�	<�M���A�*;


total_lossj�A

error_R�4J?

learning_rate_1�9�ߒ�I       6%�	�_�M���A�*;


total_loss��@

error_RR?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss�A

error_RQ�<?

learning_rate_1�9��NI       6%�	���M���A�*;


total_loss��@

error_R�L?

learning_rate_1�9Y�pnI       6%�	(�M���A�*;


total_loss���@

error_R�N?

learning_rate_1�9����I       6%�	Ul�M���A�*;


total_loss`3�@

error_R��B?

learning_rate_1�9X�I       6%�	���M���A�*;


total_loss���@

error_R3�d?

learning_rate_1�9��ǥI       6%�	|��M���A�*;


total_lossR:�@

error_R�k?

learning_rate_1�9Y�I       6%�	6�M���A�*;


total_loss���@

error_R��E?

learning_rate_1�9���SI       6%�	�y�M���A�*;


total_loss@A

error_R��_?

learning_rate_1�9�7�I       6%�	Լ�M���A�*;


total_loss���@

error_RjX<?

learning_rate_1�9g|�4I       6%�	w��M���A�*;


total_loss ��@

error_RM2X?

learning_rate_1�9,��I       6%�	�?�M���A�*;


total_loss�@

error_R��_?

learning_rate_1�9&��I       6%�	X��M���A�*;


total_loss�A

error_R��k?

learning_rate_1�9���I       6%�	~��M���A�*;


total_lossYX�@

error_Rs�p?

learning_rate_1�9o$ԠI       6%�	��M���A�*;


total_loss� A

error_RC'>?

learning_rate_1�9��stI       6%�	?Q�M���A�*;


total_loss_�@

error_RܢS?

learning_rate_1�9W�DI       6%�	��M���A�*;


total_loss�
A

error_R�oM?

learning_rate_1�9F�/I       6%�	���M���A�*;


total_loss�+�@

error_Rs�U?

learning_rate_1�9�`|=I       6%�	��M���A�*;


total_loss�܏@

error_R *Y?

learning_rate_1�9�])'I       6%�	�b�M���A�*;


total_loss��@

error_R��I?

learning_rate_1�9 �QI       6%�	���M���A�*;


total_loss�W�@

error_R��R?

learning_rate_1�9��*�I       6%�	���M���A�*;


total_loss���@

error_R�D?

learning_rate_1�9�Ϩ#I       6%�	�2�M���A�*;


total_lossO� A

error_R]?

learning_rate_1�9�5'I       6%�	�x�M���A�*;


total_loss�$�@

error_R�Q?

learning_rate_1�94�LI       6%�	h��M���A�*;


total_loss�7�@

error_R?A?

learning_rate_1�9��I       6%�	���M���A�*;


total_loss���@

error_R�|U?

learning_rate_1�9�p�-I       6%�	�@�M���A�*;


total_loss���@

error_R\�F?

learning_rate_1�98F�I       6%�	���M���A�*;


total_lossT�@

error_RJ�_?

learning_rate_1�9��́I       6%�	d��M���A�*;


total_loss�N�@

error_RnRP?

learning_rate_1�98`�I       6%�	��M���A�*;


total_loss�A

error_R�6X?

learning_rate_1�9��*:I       6%�	�^�M���A�*;


total_loss#�@

error_R��D?

learning_rate_1�9P���I       6%�	F��M���A�*;


total_loss���@

error_R��i?

learning_rate_1�9�I       6%�	��M���A�*;


total_loss22�@

error_R� U?

learning_rate_1�9���I       6%�	=�M���A�*;


total_loss]iA

error_R�DT?

learning_rate_1�9�3n�I       6%�	��M���A�*;


total_loss ��@

error_R\HU?

learning_rate_1�9�H��I       6%�	���M���A�*;


total_loss=��@

error_R�Z?

learning_rate_1�9��&�I       6%�	Y�M���A�*;


total_lossXTw@

error_R?^?

learning_rate_1�9����I       6%�	�U�M���A�*;


total_lossŧ�@

error_R��Y?

learning_rate_1�9$C+I       6%�	͖�M���A�*;


total_loss*��@

error_R�M:?

learning_rate_1�9a0UI       6%�	���M���A�*;


total_loss�Ӯ@

error_R�iG?

learning_rate_1�9�~>I       6%�	9%�M���A�*;


total_loss��@

error_R�%e?

learning_rate_1�9����I       6%�	Vl�M���A�*;


total_losso��@

error_Rx{??

learning_rate_1�9���I       6%�	.��M���A�*;


total_loss��@

error_R��Y?

learning_rate_1�9�dI       6%�	��M���A�*;


total_loss�%�@

error_R�P?

learning_rate_1�9��I       6%�	7K�M���A�*;


total_loss;�@

error_R"`?

learning_rate_1�9ÿ��I       6%�	���M���A�*;


total_loss���@

error_R�-D?

learning_rate_1�9�x�I       6%�	�u�M���A�*;


total_lossr�@

error_R��R?

learning_rate_1�9Z&I       6%�	Q��M���A�*;


total_lossX\�@

error_R��d?

learning_rate_1�9ʾZI       6%�	��M���A�*;


total_loss�"�@

error_R��Q?

learning_rate_1�9�㣕I       6%�	�R�M���A�*;


total_lossDT�@

error_R�3O?

learning_rate_1�9�~�>I       6%�	?��M���A�*;


total_loss� �@

error_R&�M?

learning_rate_1�9���gI       6%�	���M���A�*;


total_lossQ�@

error_R{�Y?

learning_rate_1�9�<C#I       6%�	�"�M���A�*;


total_loss۲m@

error_R�!H?

learning_rate_1�9�z�I       6%�	�j�M���A�*;


total_loss�z@

error_R�wR?

learning_rate_1�9���I       6%�	i��M���A�*;


total_lossR�@

error_R/�<?

learning_rate_1�9�L�I       6%�	>��M���A�*;


total_loss���@

error_R�a?

learning_rate_1�9d\��I       6%�	�<�M���A�*;


total_lossQ[�@

error_R��N?

learning_rate_1�9㌢KI       6%�	���M���A�*;


total_loss���@

error_R��P?

learning_rate_1�9S}��I       6%�	���M���A�*;


total_lossn�@

error_Rt+V?

learning_rate_1�9yx/6I       6%�	
�M���A�*;


total_loss���@

error_R=�i?

learning_rate_1�9�b*�I       6%�	�M�M���A�*;


total_loss���@

error_R�CB?

learning_rate_1�9>��I       6%�	��M���A�*;


total_losszJ�@

error_R��Y?

learning_rate_1�9)�I       6%�	��M���A�*;


total_loss�V�@

error_R�GU?

learning_rate_1�9(n!RI       6%�	
�M���A�*;


total_loss�V�@

error_R8�I?

learning_rate_1�9��5�I       6%�	\�M���A�*;


total_loss��@

error_RmQ\?

learning_rate_1�9�`I       6%�	���M���A�*;


total_lossEz@

error_R LT?

learning_rate_1�9��1I       6%�	���M���A�*;


total_loss刿@

error_R�L?

learning_rate_1�9��SI       6%�	*�M���A�*;


total_loss�A

error_Rv�U?

learning_rate_1�9��KI       6%�	 n�M���A�*;


total_loss���@

error_R�U@?

learning_rate_1�9�}�I       6%�	_��M���A�*;


total_loss�t�@

error_R�Y?

learning_rate_1�9"�"JI       6%�	b��M���A�*;


total_loss��@

error_R��@?

learning_rate_1�9X'] I       6%�	�3�M���A�*;


total_lossN�@

error_R��K?

learning_rate_1�9A�	I       6%�	�y�M���A�*;


total_loss_�@

error_RT�O?

learning_rate_1�9�;yI       6%�	���M���A�*;


total_loss�1�@

error_Re�L?

learning_rate_1�9(j�I       6%�	� �M���A�*;


total_loss�I�@

error_R8�P?

learning_rate_1�9��G�I       6%�	@I�M���A�*;


total_loss�j�@

error_RöT?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss�c�@

error_R� W?

learning_rate_1�9a0^I       6%�	%��M���A�*;


total_loss�ĥ@

error_RX�e?

learning_rate_1�9�zRI       6%�	�M�M���A�*;


total_loss}^�@

error_R��B?

learning_rate_1�9ql��I       6%�	���M���A�*;


total_loss�ƪ@

error_R�.P?

learning_rate_1�9�Tv�I       6%�	B'�M���A�*;


total_loss �U@

error_R��U?

learning_rate_1�9�[�II       6%�	�t�M���A�*;


total_loss���@

error_RV�H?

learning_rate_1�9'�>I       6%�	���M���A�*;


total_loss�&�@

error_R��5?

learning_rate_1�9>w�KI       6%�	R�M���A�*;


total_loss��@

error_R
�V?

learning_rate_1�9��I       6%�	9O�M���A�*;


total_lossd�@

error_R�4_?

learning_rate_1�9���pI       6%�	!��M���A�*;


total_loss�L�@

error_R,�T?

learning_rate_1�9�k�iI       6%�	a�M���A�*;


total_lossؗ@

error_R@.M?

learning_rate_1�9�UF�I       6%�	�N�M���A�*;


total_loss-��@

error_R_#X?

learning_rate_1�9R�VI       6%�	Z��M���A�*;


total_loss21�@

error_R��L?

learning_rate_1�9`�wI       6%�	,��M���A�*;


total_lossqޢ@

error_R�H?

learning_rate_1�9	?@I       6%�	�+�M���A�*;


total_loss@��@

error_R�MG?

learning_rate_1�9lMy�I       6%�	��M���A�*;


total_loss��@

error_R�M?

learning_rate_1�9�0�LI       6%�	���M���A�*;


total_loss���@

error_R�+W?

learning_rate_1�9@��I       6%�	�M���A�*;


total_loss���@

error_Rf|W?

learning_rate_1�9Uk(I       6%�	^�M���A�*;


total_loss���@

error_R�kX?

learning_rate_1�9[%I       6%�	ݢ�M���A�*;


total_lossŋ�@

error_R��]?

learning_rate_1�9]���I       6%�	��M���A�*;


total_lossnN�@

error_R�\?

learning_rate_1�9�
�I       6%�	�,�M���A�*;


total_lossC��@

error_RaN?

learning_rate_1�9h���I       6%�	wp�M���A�*;


total_loss:�@

error_R��T?

learning_rate_1�9h�NuI       6%�	0��M���A�*;


total_loss�]�@

error_R`�Q?

learning_rate_1�9�">LI       6%�	(��M���A�*;


total_loss���@

error_R��L?

learning_rate_1�9=��I       6%�		G�M���A�*;


total_loss*�A

error_R<xR?

learning_rate_1�9ͱs�I       6%�	��M���A�*;


total_lossϑ@

error_R.�E?

learning_rate_1�9�RcSI       6%�	���M���A�*;


total_loss���@

error_RdgS?

learning_rate_1�9dYwiI       6%�	^4�M���A�*;


total_loss��@

error_R�.F?

learning_rate_1�9�7p�I       6%�	�y�M���A�*;


total_loss��@

error_RsD?

learning_rate_1�9F[4I       6%�	���M���A�*;


total_loss���@

error_Rn�B?

learning_rate_1�9n�@I       6%�	�M���A�*;


total_loss�}�@

error_R1�H?

learning_rate_1�9���I       6%�	�M�M���A�*;


total_loss=v6A

error_RpR?

learning_rate_1�9h�v�I       6%�	���M���A�*;


total_loss���@

error_R�OQ?

learning_rate_1�9�H�I       6%�	���M���A�*;


total_loss�`�@

error_RM�X?

learning_rate_1�9�r�CI       6%�	=�M���A�*;


total_loss(��@

error_R�IU?

learning_rate_1�9GC �I       6%�	�^�M���A�*;


total_losse�`@

error_R�uS?

learning_rate_1�9ݙ�8I       6%�	C��M���A�*;


total_loss��@

error_RQ??

learning_rate_1�9c��I       6%�	{��M���A�*;


total_lossZ6�@

error_R��C?

learning_rate_1�9ϮKI       6%�	�)�M���A�*;


total_loss�C�@

error_R�Ql?

learning_rate_1�9%�_%I       6%�	�l�M���A�*;


total_loss8��@

error_R@�C?

learning_rate_1�9�P�I       6%�	��M���A�*;


total_loss	�@

error_R,�T?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss�>�@

error_R�J?

learning_rate_1�9i��I       6%�	�=�M���A�*;


total_loss�"�@

error_R��I?

learning_rate_1�9ٰ
wI       6%�	W��M���A�*;


total_loss�^�@

error_R��F?

learning_rate_1�9���I       6%�	���M���A�*;


total_loss��@

error_R}*P?

learning_rate_1�9)�LI       6%�	_�M���A�*;


total_lossv=�@

error_R�@N?

learning_rate_1�9I�I       6%�	kT�M���A�*;


total_loss��@

error_R��W?

learning_rate_1�9��DI       6%�	���M���A�*;


total_loss��@

error_R��P?

learning_rate_1�9�Y��I       6%�	���M���A�*;


total_lossҋ�@

error_RW�K?

learning_rate_1�9j�^I       6%�	"�M���A�*;


total_loss�a�@

error_R��X?

learning_rate_1�9����I       6%�	f�M���A�*;


total_loss/��@

error_R$�B?

learning_rate_1�9i�X�I       6%�	ȫ�M���A�*;


total_loss-"�@

error_R�_?

learning_rate_1�9l�ZI       6%�	���M���A�*;


total_lossX��@

error_R=XN?

learning_rate_1�9�U�bI       6%�	�/�M���A�*;


total_loss̽�@

error_R��e?

learning_rate_1�93�-I       6%�	 s�M���A�*;


total_lossr��@

error_R�[?

learning_rate_1�9Zv�I       6%�	��M���A�*;


total_loss�(�@

error_R �`?

learning_rate_1�9�=	�I       6%�	���M���A�*;


total_loss`�A

error_R,*K?

learning_rate_1�9�+��I       6%�	�E�M���A�*;


total_loss�t�@

error_R�O]?

learning_rate_1�9�m��I       6%�	���M���A�*;


total_loss�YA

error_R�S?

learning_rate_1�9��II       6%�	*��M���A�*;


total_loss���@

error_R@�e?

learning_rate_1�9;a\hI       6%�	�7�M���A�*;


total_loss/ӑ@

error_R&$G?

learning_rate_1�98�h�I       6%�	Vz�M���A�*;


total_loss�8�@

error_R${X?

learning_rate_1�9����I       6%�	���M���A�*;


total_loss���@

error_RC�W?

learning_rate_1�9R'�9I       6%�	�M���A�*;


total_loss���@

error_R�,J?

learning_rate_1�9�
�I       6%�	]R�M���A�*;


total_lossC�@

error_R�u@?

learning_rate_1�9B+��I       6%�	Y��M���A�*;


total_loss��@

error_RH�I?

learning_rate_1�9,Bw�I       6%�	���M���A�*;


total_loss�1�@

error_R�>i?

learning_rate_1�9H��@I       6%�	��M���A�*;


total_lossW,�@

error_RW�I?

learning_rate_1�9�ơI       6%�	_�M���A�*;


total_lossD#�@

error_R(.K?

learning_rate_1�99�eI       6%�	=��M���A�*;


total_loss�ߥ@

error_R#�@?

learning_rate_1�9;�liI       6%�	���M���A�*;


total_loss$�@

error_R}�T?

learning_rate_1�9�{n�I       6%�	�1�M���A�*;


total_loss���@

error_R}1`?

learning_rate_1�9*��$I       6%�	�}�M���A�*;


total_lossQ;�@

error_R��W?

learning_rate_1�9�P�=I       6%�	���M���A�*;


total_loss68�@

error_R;qF?

learning_rate_1�9�8��I       6%�	��M���A�*;


total_loss+q@

error_R��X?

learning_rate_1�9j�3I       6%�	lY�M���A�*;


total_loss���@

error_Rq�G?

learning_rate_1�9S]��I       6%�	��M���A�*;


total_loss�ڋ@

error_Rd9E?

learning_rate_1�9�&�\I       6%�	���M���A�*;


total_loss���@

error_R��R?

learning_rate_1�9	�>MI       6%�	�(�M���A�*;


total_loss䐭@

error_R=�]?

learning_rate_1�9͋�I       6%�	�l�M���A�*;


total_loss}O�@

error_R �N?

learning_rate_1�9�nFWI       6%�	w��M���A�*;


total_loss�ݰ@

error_Rl9I?

learning_rate_1�9ZF��I       6%�	���M���A�*;


total_loss3��@

error_R�LN?

learning_rate_1�9�� )I       6%�	D4 N���A�*;


total_loss���@

error_Rz�G?

learning_rate_1�9%=��I       6%�	�x N���A�*;


total_loss�L�@

error_R��]?

learning_rate_1�9dC_I       6%�	�� N���A�*;


total_loss_��@

error_R]?

learning_rate_1�9����I       6%�	�� N���A�*;


total_lossA

error_R��V?

learning_rate_1�9�lI       6%�	�BN���A�*;


total_loss�W�@

error_R��S?

learning_rate_1�9����I       6%�	��N���A�*;


total_loss���@

error_R�}T?

learning_rate_1�9���I       6%�	��N���A�*;


total_lossj�@

error_R�S?

learning_rate_1�9\P�dI       6%�	\N���A�*;


total_loss��h@

error_R��N?

learning_rate_1�9�)�I       6%�	�VN���A�*;


total_loss���@

error_R�T?

learning_rate_1�9ӓB�I       6%�	/�N���A�*;


total_loss.TA

error_Rv�F?

learning_rate_1�9�z�I       6%�	w�N���A�*;


total_loss�I�@

error_R��`?

learning_rate_1�9�$(	I       6%�	]N���A�*;


total_loss�K�@

error_R�cZ?

learning_rate_1�9��0CI       6%�	dbN���A�*;


total_loss���@

error_R�O?

learning_rate_1�9�/��I       6%�	��N���A�*;


total_loss�h�@

error_R��Y?

learning_rate_1�9���tI       6%�	x�N���A�*;


total_loss-{�@

error_R�)\?

learning_rate_1�9;�e�I       6%�	�4N���A�*;


total_loss��@

error_R��c?

learning_rate_1�96��I       6%�	Q~N���A�*;


total_loss�Z�@

error_R� O?

learning_rate_1�9r8cPI       6%�	T�N���A�*;


total_lossTڍ@

error_R�K?

learning_rate_1�9�M�I       6%�	�N���A�*;


total_lossG��@

error_R�GI?

learning_rate_1�9��RTI       6%�	wPN���A�*;


total_loss3!�@

error_R�9H?

learning_rate_1�9����I       6%�	��N���A�*;


total_lossh A

error_RH"X?

learning_rate_1�93T��I       6%�	��N���A�*;


total_loss��A

error_R��L?

learning_rate_1�9�J|&I       6%�	 )N���A�*;


total_loss�1�@

error_RO�:?

learning_rate_1�9;��I       6%�	qN���A�*;


total_lossh[�@

error_RWNB?

learning_rate_1�9�@��I       6%�	��N���A�*;


total_loss|&�@

error_RZI?

learning_rate_1�9���I       6%�	�N���A�*;


total_loss3|�@

error_R9B?

learning_rate_1�9��EI       6%�	UN���A�*;


total_loss�ͬ@

error_RFG?

learning_rate_1�9��:I       6%�	j�N���A�*;


total_lossy�@

error_RȞ\?

learning_rate_1�9z�5�I       6%�	��N���A�*;


total_lossi��@

error_R��T?

learning_rate_1�9�@ОI       6%�	"N���A�*;


total_loss(��@

error_RŀY?

learning_rate_1�9u dI       6%�	�xN���A�*;


total_loss���@

error_R�S?

learning_rate_1�9��ٝI       6%�	��N���A�*;


total_loss��@

error_R	�M?

learning_rate_1�9���I       6%�	�	N���A�*;


total_lossJC�@

error_Rn�Q?

learning_rate_1�9H�I       6%�	&�	N���A�*;


total_loss�)�@

error_R��[?

learning_rate_1�9i�ϺI       6%�	#�	N���A�*;


total_loss��t@

error_R_�I?

learning_rate_1�9@�1NI       6%�	�=
N���A�*;


total_loss�8�@

error_R��T?

learning_rate_1�9���FI       6%�	��
N���A�*;


total_loss#$�@

error_R�O?

learning_rate_1�9��q�I       6%�	��
N���A�*;


total_loss$��@

error_R�<[?

learning_rate_1�9�r��I       6%�	jN���A�	*;


total_loss�(�@

error_RH?

learning_rate_1�9�@�I       6%�	�_N���A�	*;


total_lossc�@

error_R)�N?

learning_rate_1�9���I       6%�	�N���A�	*;


total_loss ��@

error_R�,D?

learning_rate_1�9VFI       6%�	��N���A�	*;


total_lossA�A

error_R!<U?

learning_rate_1�9���I       6%�	�=N���A�	*;


total_loss��@

error_R�"G?

learning_rate_1�9GTe�I       6%�	:�N���A�	*;


total_loss�W�@

error_R�??

learning_rate_1�9r!�I       6%�	��N���A�	*;


total_lossPڛ@

error_R�#>?

learning_rate_1�9LX�OI       6%�	�N���A�	*;


total_loss?A

error_R�g`?

learning_rate_1�9�	��I       6%�	�\N���A�	*;


total_loss{�@

error_R�WJ?

learning_rate_1�9�n�I       6%�	n�N���A�	*;


total_loss�P�@

error_R<CB?

learning_rate_1�9e:�I       6%�	g�N���A�	*;


total_lossj��@

error_R$Y^?

learning_rate_1�9��9wI       6%�	H.N���A�	*;


total_lossf(A

error_R�fM?

learning_rate_1�9�`�I       6%�	}sN���A�	*;


total_losshW+A

error_R$	T?

learning_rate_1�9�Ԃ^I       6%�	�N���A�	*;


total_loss��@

error_RRU?

learning_rate_1�98|I       6%�	N���A�	*;


total_lossr��@

error_R��]?

learning_rate_1�9��$I       6%�	}KN���A�	*;


total_loss`�g@

error_R=d\?

learning_rate_1�9J�PzI       6%�	�N���A�	*;


total_loss�˰@

error_R_�O?

learning_rate_1�9��I       6%�	��N���A�	*;


total_loss��@

error_R�if?

learning_rate_1�9��I       6%�	kN���A�	*;


total_lossif0A

error_R{�I?

learning_rate_1�9*�3uI       6%�	aN���A�	*;


total_loss�`�@

error_R�7K?

learning_rate_1�9Y1��I       6%�	ܨN���A�	*;


total_loss�w*A

error_R@L?

learning_rate_1�9��;dI       6%�	`�N���A�	*;


total_loss��@

error_R�CK?

learning_rate_1�9mP�I       6%�	�6N���A�	*;


total_loss���@

error_R.V?

learning_rate_1�9�b��I       6%�	�{N���A�	*;


total_loss�G�@

error_Ri�E?

learning_rate_1�9�}�I       6%�	��N���A�	*;


total_loss�9�@

error_RqK?

learning_rate_1�9@�ZsI       6%�	�N���A�	*;


total_loss���@

error_R!PI?

learning_rate_1�9��' I       6%�	�NN���A�	*;


total_loss	��@

error_R�D?

learning_rate_1�9U"oiI       6%�	<�N���A�	*;


total_losscj�@

error_R}_]?

learning_rate_1�9�|U�I       6%�	z�N���A�	*;


total_loss`��@

error_R?�U?

learning_rate_1�9���I       6%�	N���A�	*;


total_loss�ڧ@

error_R.8?

learning_rate_1�93*�I       6%�	�ZN���A�	*;


total_loss��@

error_R�Q?

learning_rate_1�9$G�I       6%�	)�N���A�	*;


total_loss4��@

error_R�FL?

learning_rate_1�9&a�I       6%�	��N���A�	*;


total_loss���@

error_R�bE?

learning_rate_1�9���I       6%�	-$N���A�	*;


total_loss�'�@

error_R�Y?

learning_rate_1�9O=ѴI       6%�	�iN���A�	*;


total_loss߱A

error_R��c?

learning_rate_1�9om5{I       6%�	جN���A�	*;


total_loss\�@

error_R�M?

learning_rate_1�9Α�I       6%�	p�N���A�	*;


total_lossö�@

error_R�=?

learning_rate_1�9u�I       6%�	�3N���A�	*;


total_loss�a�@

error_RdVF?

learning_rate_1�9�邺I       6%�	�zN���A�	*;


total_loss�@

error_Rd�I?

learning_rate_1�9Y�{I       6%�	��N���A�	*;


total_loss�3�@

error_RaVL?

learning_rate_1�9et@�I       6%�	qN���A�	*;


total_lossc�@

error_R.�=?

learning_rate_1�9�#��I       6%�	5[N���A�	*;


total_loss��@

error_R'V?

learning_rate_1�9ve�_I       6%�	�N���A�	*;


total_loss���@

error_RZ�F?

learning_rate_1�9�:��I       6%�	��N���A�	*;


total_lossY\�@

error_R��X?

learning_rate_1�9��bI       6%�	13N���A�	*;


total_lossO��@

error_R��D?

learning_rate_1�9v�I       6%�	p~N���A�	*;


total_loss���@

error_R�O?

learning_rate_1�9�ڶjI       6%�	H�N���A�	*;


total_loss���@

error_R�R?

learning_rate_1�9;�*sI       6%�	9N���A�	*;


total_loss�#�@

error_R��B?

learning_rate_1�9��I       6%�	+SN���A�	*;


total_loss��@

error_Rn�??

learning_rate_1�9&�I       6%�	˝N���A�	*;


total_loss\�@

error_R?�_?

learning_rate_1�9NH�RI       6%�	��N���A�	*;


total_lossGKA

error_Rv�t?

learning_rate_1�9!�(I       6%�	i)N���A�	*;


total_loss�r�@

error_R�nG?

learning_rate_1�9X�ohI       6%�	��N���A�	*;


total_lossa�@

error_R:�=?

learning_rate_1�9+�J5I       6%�	��N���A�	*;


total_loss��A

error_R�?S?

learning_rate_1�9��� I       6%�	�0N���A�	*;


total_loss�,�@

error_R.�R?

learning_rate_1�9C���I       6%�	uN���A�	*;


total_loss���@

error_R�9S?

learning_rate_1�9��I       6%�	u�N���A�	*;


total_loss/D�@

error_R�$]?

learning_rate_1�9Y �yI       6%�	�N���A�	*;


total_loss��@

error_R�N?

learning_rate_1�9wc�cI       6%�	�UN���A�	*;


total_lossL�@

error_R�bN?

learning_rate_1�9��+I       6%�	ޚN���A�	*;


total_loss�
�@

error_R�@?

learning_rate_1�9M,TsI       6%�	o�N���A�	*;


total_loss�@

error_R�hH?

learning_rate_1�9k^ٿI       6%�	�-N���A�	*;


total_loss�#�@

error_R�gU?

learning_rate_1�9���I       6%�	SqN���A�	*;


total_loss�8�@

error_R��Q?

learning_rate_1�9��#�I       6%�	�N���A�	*;


total_loss���@

error_R�aA?

learning_rate_1�9J��I       6%�	l�N���A�	*;


total_loss�A

error_RsS?

learning_rate_1�9]��I       6%�	�=N���A�	*;


total_lossf��@

error_R��H?

learning_rate_1�9���I       6%�	��N���A�	*;


total_loss���@

error_R�I?

learning_rate_1�9�G)^I       6%�	S�N���A�	*;


total_loss��@

error_R��S?

learning_rate_1�9�I       6%�	�N���A�	*;


total_loss6�@

error_R�=>?

learning_rate_1�9�v��I       6%�	]N���A�	*;


total_loss�@

error_R��N?

learning_rate_1�9��I       6%�	C�N���A�	*;


total_lossNT�@

error_Rd�Y?

learning_rate_1�9�xI       6%�	��N���A�	*;


total_loss���@

error_RDbG?

learning_rate_1�9�#uI       6%�	�0N���A�	*;


total_loss�
 A

error_R�\?

learning_rate_1�9�=�I       6%�	�tN���A�	*;


total_loss^�@

error_R{M?

learning_rate_1�9���iI       6%�	��N���A�	*;


total_loss%�1A

error_ROK?

learning_rate_1�9�2��I       6%�	N�N���A�	*;


total_loss���@

error_R._X?

learning_rate_1�9�Ph�I       6%�	> N���A�	*;


total_loss�E�@

error_RZJA?

learning_rate_1�9��I       6%�	�� N���A�	*;


total_loss�\�@

error_R�G?

learning_rate_1�9M���I       6%�	� N���A�	*;


total_loss��@

error_Rf�R?

learning_rate_1�9�%JI       6%�	G!N���A�	*;


total_loss��@

error_R�b?

learning_rate_1�9Ը]CI       6%�	�Z!N���A�	*;


total_loss+�A

error_R�P?

learning_rate_1�9�I       6%�	��!N���A�	*;


total_loss:�@

error_R��I?

learning_rate_1�9GG�I       6%�	�!N���A�	*;


total_loss�>�@

error_R�D?

learning_rate_1�9YßI       6%�	�1"N���A�	*;


total_loss��@

error_R�V1?

learning_rate_1�9��V�I       6%�	dv"N���A�	*;


total_lossߠ@

error_R)PO?

learning_rate_1�9WQ��I       6%�	��"N���A�	*;


total_loss�r�@

error_R��E?

learning_rate_1�9��I       6%�	n�"N���A�	*;


total_loss�j�@

error_R�a?

learning_rate_1�9�i��I       6%�	�E#N���A�	*;


total_loss2��@

error_R]d]?

learning_rate_1�9Üe�I       6%�	T�#N���A�	*;


total_loss��@

error_Rv<]?

learning_rate_1�9jd��I       6%�	��#N���A�	*;


total_loss�+�@

error_R��J?

learning_rate_1�9����I       6%�	�$N���A�	*;


total_loss!�@

error_R�NB?

learning_rate_1�9�e�5I       6%�	[$N���A�	*;


total_loss���@

error_R�<M?

learning_rate_1�9��+�I       6%�	��$N���A�	*;


total_losst�@

error_R��O?

learning_rate_1�9�+I       6%�	��$N���A�	*;


total_lossϭ�@

error_R�zJ?

learning_rate_1�9�ݹ�I       6%�	,%N���A�	*;


total_loss���@

error_R�O?

learning_rate_1�9�M��I       6%�	�r%N���A�	*;


total_loss�G�@

error_RW]R?

learning_rate_1�9~�9I       6%�	��%N���A�	*;


total_loss[>�@

error_R�H?

learning_rate_1�9���I       6%�	T	&N���A�	*;


total_loss6��@

error_R��K?

learning_rate_1�9����I       6%�	�S&N���A�	*;


total_loss"��@

error_R
�R?

learning_rate_1�97P�I       6%�	a�&N���A�	*;


total_loss���@

error_R��K?

learning_rate_1�9����I       6%�	��&N���A�	*;


total_lossx��@

error_R#�F?

learning_rate_1�9H?2�I       6%�	,'N���A�	*;


total_loss�؍@

error_R�	G?

learning_rate_1�9��~I       6%�	}n'N���A�	*;


total_loss��@

error_R��U?

learning_rate_1�9����I       6%�	X�'N���A�	*;


total_loss���@

error_R�Q?

learning_rate_1�9�m�OI       6%�	��'N���A�	*;


total_lossܵ�@

error_R�lF?

learning_rate_1�98��I       6%�	@(N���A�	*;


total_loss_Z�@

error_R�X?

learning_rate_1�9]�ѫI       6%�	#�(N���A�	*;


total_loss$eA

error_R%I>?

learning_rate_1�9$��`I       6%�	I�(N���A�	*;


total_lossts�@

error_R�??

learning_rate_1�9�7,I       6%�	g:)N���A�	*;


total_loss�ar@

error_R6�M?

learning_rate_1�9���I       6%�	w�)N���A�	*;


total_lossw3A

error_RsL?

learning_rate_1�9\��I       6%�	k*N���A�	*;


total_loss�B�@

error_R\�K?

learning_rate_1�9�wm�I       6%�	�]*N���A�	*;


total_loss�MX@

error_R��;?

learning_rate_1�9%M�LI       6%�	ڧ*N���A�	*;


total_loss���@

error_Rs�4?

learning_rate_1�9sKj�I       6%�	�*N���A�	*;


total_loss�2�@

error_R�aQ?

learning_rate_1�9�&��I       6%�	�;+N���A�	*;


total_loss�2�@

error_Re�8?

learning_rate_1�9o���I       6%�	�+N���A�	*;


total_loss�jA

error_RE�Z?

learning_rate_1�9c��I       6%�	��+N���A�	*;


total_loss���@

error_Rq"\?

learning_rate_1�9.��I       6%�	�,N���A�	*;


total_lossշ@

error_RibS?

learning_rate_1�9��;I       6%�	PV,N���A�	*;


total_lossԽ�@

error_R�??

learning_rate_1�9	O�I       6%�	�,N���A�	*;


total_loss���@

error_R�FA?

learning_rate_1�9��zI       6%�	��,N���A�	*;


total_loss{��@

error_R;�`?

learning_rate_1�99���I       6%�	0-N���A�	*;


total_lossӮ@

error_R!�O?

learning_rate_1�9���TI       6%�	�w-N���A�	*;


total_loss���@

error_R�jN?

learning_rate_1�9�X5�I       6%�	�-N���A�	*;


total_lossY�@

error_Rq�E?

learning_rate_1�9�U1I       6%�	h
.N���A�	*;


total_lossŮ�@

error_R-�2?

learning_rate_1�9.�XI       6%�	�O.N���A�	*;


total_losszO�@

error_R�@J?

learning_rate_1�9.���I       6%�	֖.N���A�	*;


total_loss��	A

error_RmG?

learning_rate_1�9-��LI       6%�	��.N���A�	*;


total_loss��@

error_R�c?

learning_rate_1�9<9��I       6%�	�)/N���A�
*;


total_loss�2�@

error_R�f?

learning_rate_1�9m�ctI       6%�	wy/N���A�
*;


total_loss?p�@

error_R �i?

learning_rate_1�94�!�I       6%�	��/N���A�
*;


total_loss��@

error_R)]`?

learning_rate_1�9�<RI       6%�	�0N���A�
*;


total_loss]��@

error_R�zF?

learning_rate_1�9���I       6%�	fR0N���A�
*;


total_loss��@

error_RM%H?

learning_rate_1�9���I       6%�	E�0N���A�
*;


total_lossIv�@

error_R)�Q?

learning_rate_1�9��M�I       6%�	x�0N���A�
*;


total_loss���@

error_R��F?

learning_rate_1�9."I       6%�	�1N���A�
*;


total_loss7�@

error_R.�A?

learning_rate_1�9�dI       6%�	ib1N���A�
*;


total_loss*i�@

error_RX?

learning_rate_1�95�e�I       6%�	%�1N���A�
*;


total_loss�ı@

error_RQiX?

learning_rate_1�9�k�fI       6%�	/�1N���A�
*;


total_loss�j�@

error_R��g?

learning_rate_1�9�"{I       6%�	,2N���A�
*;


total_lossq�@

error_R��I?

learning_rate_1�9�j�I       6%�	_p2N���A�
*;


total_lossm�@

error_RM�D?

learning_rate_1�9U���I       6%�	�2N���A�
*;


total_loss�$�@

error_R�I>?

learning_rate_1�9W�:�I       6%�	|�2N���A�
*;


total_loss4��@

error_R3#@?

learning_rate_1�9j�F�I       6%�	�@3N���A�
*;


total_loss�ֆ@

error_R�'S?

learning_rate_1�9�/� I       6%�	ˈ3N���A�
*;


total_loss�#�@

error_R~>?

learning_rate_1�9�,w`I       6%�	{�3N���A�
*;


total_loss1Z�@

error_R�(W?

learning_rate_1�9���I       6%�	�4N���A�
*;


total_loss8�@

error_RÓH?

learning_rate_1�9A��I       6%�	�W4N���A�
*;


total_loss��@

error_Rq�]?

learning_rate_1�9�C%�I       6%�	��4N���A�
*;


total_loss^��@

error_R)6Q?

learning_rate_1�9�h
�I       6%�	A�4N���A�
*;


total_loss�F�@

error_RR[J?

learning_rate_1�9�D�I       6%�	�#5N���A�
*;


total_lossW�@

error_R7�X?

learning_rate_1�9W��I       6%�	�e5N���A�
*;


total_loss�A

error_RZ�K?

learning_rate_1�9p���I       6%�	O�5N���A�
*;


total_loss?�A

error_RH�J?

learning_rate_1�9T;tQI       6%�	q�5N���A�
*;


total_loss���@

error_R��H?

learning_rate_1�9�>^fI       6%�	�16N���A�
*;


total_loss�E�@

error_RMO?

learning_rate_1�9YX!~I       6%�	�{6N���A�
*;


total_lossi��@

error_R�s>?

learning_rate_1�9�X=�I       6%�	��6N���A�
*;


total_loss��@

error_R2�P?

learning_rate_1�9akYI       6%�	g7N���A�
*;


total_loss߿@

error_R�CU?

learning_rate_1�9�!�I       6%�	kI7N���A�
*;


total_loss��@

error_RmJ?

learning_rate_1�9���I       6%�	Ҍ7N���A�
*;


total_loss)V�@

error_Rq 2?

learning_rate_1�9�ȡ I       6%�	h�7N���A�
*;


total_lossƫ�@

error_R�F?

learning_rate_1�9bX8�I       6%�	F8N���A�
*;


total_loss]��@

error_R��S?

learning_rate_1�9c�)I       6%�	(U8N���A�
*;


total_loss���@

error_R10J?

learning_rate_1�9�}�"I       6%�	��8N���A�
*;


total_loss���@

error_R�H?

learning_rate_1�9-�QI       6%�	R�8N���A�
*;


total_lossC�@

error_R!�`?

learning_rate_1�9�`:�I       6%�	�9N���A�
*;


total_loss(��@

error_R��C?

learning_rate_1�9����I       6%�	�~9N���A�
*;


total_loss�o�@

error_R��U?

learning_rate_1�9�dL�I       6%�	��9N���A�
*;


total_lossߏ�@

error_R��R?

learning_rate_1�9�/�I       6%�	n:N���A�
*;


total_loss��@

error_R��L?

learning_rate_1�9��U�I       6%�	�Y:N���A�
*;


total_lossEM�@

error_R��B?

learning_rate_1�9f!��I       6%�	��:N���A�
*;


total_loss���@

error_RNaV?

learning_rate_1�95=	�I       6%�	�;N���A�
*;


total_loss���@

error_R.�Q?

learning_rate_1�9qI       6%�	�H;N���A�
*;


total_loss�v�@

error_R��M?

learning_rate_1�9���$I       6%�	A�;N���A�
*;


total_loss.`A

error_R�w?

learning_rate_1�9|�ѤI       6%�	��;N���A�
*;


total_loss 6�@

error_Ra�Q?

learning_rate_1�9�W�0I       6%�	/<N���A�
*;


total_loss�!�@

error_R��B?

learning_rate_1�9��GI       6%�	�c<N���A�
*;


total_loss(��@

error_R/J@?

learning_rate_1�9eA�vI       6%�	ߪ<N���A�
*;


total_loss.^�@

error_R h[?

learning_rate_1�9�lEI       6%�	O�<N���A�
*;


total_loss��@

error_R�U?

learning_rate_1�9S$��I       6%�	�==N���A�
*;


total_lossP�A

error_R�jP?

learning_rate_1�97��JI       6%�	��=N���A�
*;


total_lossZ`�@

error_R��L?

learning_rate_1�9���I       6%�	�=N���A�
*;


total_loss�P�@

error_R�nS?

learning_rate_1�9���I       6%�	�;>N���A�
*;


total_loss��@

error_R��R?

learning_rate_1�9�ƕ]I       6%�	��>N���A�
*;


total_loss�)CA

error_R��K?

learning_rate_1�9�r�I       6%�	Q�>N���A�
*;


total_loss�j�@

error_RC�f?

learning_rate_1�9�B6\I       6%�	?N���A�
*;


total_lossƛ�@

error_R�_?

learning_rate_1�9�y��I       6%�	�a?N���A�
*;


total_loss�TFA

error_RE*[?

learning_rate_1�97�<�I       6%�	֧?N���A�
*;


total_losst��@

error_R�R?

learning_rate_1�9�9�I       6%�	��?N���A�
*;


total_loss���@

error_R�C?

learning_rate_1�9AjAI       6%�	0@N���A�
*;


total_lossL�@

error_RMQ?

learning_rate_1�9�B�I       6%�	xr@N���A�
*;


total_lossq}i@

error_R�>J?

learning_rate_1�9Ic�XI       6%�	�@N���A�
*;


total_loss#c�@

error_R@IR?

learning_rate_1�9���I       6%�	�AN���A�
*;


total_loss�|�@

error_R4M?

learning_rate_1�9n�gWI       6%�	SAN���A�
*;


total_lossrË@

error_R�
K?

learning_rate_1�9�,�I       6%�	��AN���A�
*;


total_loss��@

error_R
�P?

learning_rate_1�9�I       6%�	��AN���A�
*;


total_loss7m�@

error_R�Q?

learning_rate_1�9��FI       6%�	-BN���A�
*;


total_loss�Y/A

error_RS�[?

learning_rate_1�9�[�VI       6%�	`~BN���A�
*;


total_loss���@

error_R�[N?

learning_rate_1�9~�dtI       6%�	�BN���A�
*;


total_lossM�JA

error_RG?

learning_rate_1�9�E��I       6%�	�CN���A�
*;


total_loss��@

error_RTP\?

learning_rate_1�9���5I       6%�	'eCN���A�
*;


total_loss���@

error_Rn�H?

learning_rate_1�9�Ӏ+I       6%�	��CN���A�
*;


total_loss��@

error_RAG?

learning_rate_1�9��c�I       6%�	��CN���A�
*;


total_lossd�@

error_R��P?

learning_rate_1�9��گI       6%�	P;DN���A�
*;


total_loss��@

error_R�K?

learning_rate_1�9��I       6%�	��DN���A�
*;


total_loss1��@

error_R�C^?

learning_rate_1�9�|&�I       6%�	-�DN���A�
*;


total_loss�i�@

error_R�6P?

learning_rate_1�9?]�I       6%�	|EN���A�
*;


total_lossC��@

error_Rn [?

learning_rate_1�9��M�I       6%�	HaEN���A�
*;


total_loss��V@

error_R�U?

learning_rate_1�9���#I       6%�	��EN���A�
*;


total_loss
�@

error_Ro�V?

learning_rate_1�9 �v�I       6%�	#�EN���A�
*;


total_lossJ��@

error_R�I?

learning_rate_1�94]AJI       6%�	�.FN���A�
*;


total_loss�6 A

error_RIEd?

learning_rate_1�9!�I       6%�	erFN���A�
*;


total_lossRݗ@

error_RCdY?

learning_rate_1�9�rʒI       6%�	�FN���A�
*;


total_loss���@

error_R�+a?

learning_rate_1�9��[I       6%�	��FN���A�
*;


total_loss8��@

error_R1UG?

learning_rate_1�9�ߙ�I       6%�	�<GN���A�
*;


total_loss�@�@

error_R�~T?

learning_rate_1�9y��I       6%�	:�GN���A�
*;


total_losst��@

error_R�<?

learning_rate_1�9���I       6%�	s�GN���A�
*;


total_loss��@

error_R�Q?

learning_rate_1�9�~�I       6%�	�HN���A�
*;


total_loss ϧ@

error_R�K?

learning_rate_1�9b��8I       6%�	?cHN���A�
*;


total_loss�9�@

error_R6J?

learning_rate_1�9��I       6%�	��HN���A�
*;


total_loss���@

error_R2O?

learning_rate_1�9�-hI       6%�	)�HN���A�
*;


total_loss�#�@

error_R!lF?

learning_rate_1�9|^=�I       6%�	�IIN���A�
*;


total_loss��@

error_R�/;?

learning_rate_1�9Kq�gI       6%�	��IN���A�
*;


total_lossm�@

error_RL�9?

learning_rate_1�9h���I       6%�	�IN���A�
*;


total_loss�]�@

error_R}�P?

learning_rate_1�9��I       6%�	9@JN���A�
*;


total_loss���@

error_R�:?

learning_rate_1�9����I       6%�	�JN���A�
*;


total_lossU��@

error_R\6?

learning_rate_1�9qҿI       6%�	P�JN���A�
*;


total_loss&��@

error_Rs5W?

learning_rate_1�9�_r�I       6%�	�KN���A�
*;


total_loss	��@

error_R.b?

learning_rate_1�9���I       6%�	<dKN���A�
*;


total_loss!X�@

error_R�a?

learning_rate_1�9�[��I       6%�	��KN���A�
*;


total_loss�n�@

error_R��Q?

learning_rate_1�9�A�I       6%�	p�KN���A�
*;


total_loss��@

error_R��i?

learning_rate_1�9N=�
I       6%�	mFLN���A�
*;


total_loss#�@

error_R}�Q?

learning_rate_1�9��WI       6%�	��LN���A�
*;


total_loss���@

error_R\|R?

learning_rate_1�9=$I       6%�	=�LN���A�
*;


total_lossFê@

error_R��H?

learning_rate_1�9���OI       6%�	l,MN���A�
*;


total_loss�h�@

error_R�hQ?

learning_rate_1�9ǚ_I       6%�	VrMN���A�
*;


total_loss��@

error_R4O?

learning_rate_1�9I)�I       6%�	
�MN���A�
*;


total_lossS��@

error_R_�P?

learning_rate_1�9�FąI       6%�	i�MN���A�
*;


total_loss�s�@

error_RM�c?

learning_rate_1�9,���I       6%�	hBNN���A�
*;


total_loss�Y�@

error_R�hJ?

learning_rate_1�91�s�I       6%�	0�NN���A�
*;


total_loss�o�@

error_R��\?

learning_rate_1�94��I       6%�	M�NN���A�
*;


total_loss�<^@

error_R�F?

learning_rate_1�9{wL�I       6%�	�ON���A�
*;


total_loss�ֺ@

error_R\�L?

learning_rate_1�9N�&�I       6%�	GZON���A�
*;


total_loss:�@

error_R&\?

learning_rate_1�9q���I       6%�	E�ON���A�
*;


total_loss}~�@

error_R�V?

learning_rate_1�9z��I       6%�	h�ON���A�
*;


total_lossnG�@

error_R1S?

learning_rate_1�9��G=I       6%�	"2PN���A�
*;


total_loss�!�@

error_RN^L?

learning_rate_1�9�ÚI       6%�	�yPN���A�
*;


total_loss	��@

error_R=�B?

learning_rate_1�9��aI       6%�	p�PN���A�
*;


total_loss���@

error_R�j`?

learning_rate_1�9�B`�I       6%�	�QN���A�
*;


total_loss9�@

error_R��Q?

learning_rate_1�9��I       6%�	~VQN���A�
*;


total_loss��	A

error_R�Z?

learning_rate_1�9�&�I       6%�	��QN���A�
*;


total_loss�q�@

error_RaY?

learning_rate_1�9'G��I       6%�	 �QN���A�
*;


total_loss�s�@

error_R�"X?

learning_rate_1�9�
{ I       6%�	x/RN���A�
*;


total_loss���@

error_Rz�J?

learning_rate_1�9�wg+I       6%�	fvRN���A�
*;


total_loss
�@

error_R�/W?

learning_rate_1�9�I       6%�	��RN���A�
*;


total_loss��@

error_R,??

learning_rate_1�9XńI       6%�	'SN���A�
*;


total_loss��@

error_R,2a?

learning_rate_1�9��3�I       6%�	�XSN���A�*;


total_loss�2�@

error_Rb@?

learning_rate_1�9�1��I       6%�	7�SN���A�*;


total_loss��@

error_R��T?

learning_rate_1�9s�I       6%�	�SN���A�*;


total_loss�R�@

error_RZI?

learning_rate_1�9�t��I       6%�	Y-TN���A�*;


total_loss4;�@

error_R�;R?

learning_rate_1�9�6�^I       6%�	@qTN���A�*;


total_loss�.�@

error_R��A?

learning_rate_1�9E��I       6%�	��TN���A�*;


total_loss��A

error_Rw|T?

learning_rate_1�9�;Z�I       6%�	�UN���A�*;


total_loss[ʞ@

error_R�WL?

learning_rate_1�9be�AI       6%�	�\UN���A�*;


total_loss���@

error_RH�K?

learning_rate_1�9t��I       6%�	c�UN���A�*;


total_loss�h�@

error_Rx�K?

learning_rate_1�9S��I       6%�	w�UN���A�*;


total_lossg�@

error_R�N?

learning_rate_1�9��7rI       6%�	�AVN���A�*;


total_lossIj�@

error_Rv�Y?

learning_rate_1�9U�D�I       6%�	 �VN���A�*;


total_loss���@

error_R-0J?

learning_rate_1�9��W�I       6%�	J�VN���A�*;


total_loss���@

error_R:M?

learning_rate_1�9�f�I       6%�	!WN���A�*;


total_loss[��@

error_R�PQ?

learning_rate_1�94OϙI       6%�	=_WN���A�*;


total_lossE�@

error_R|C?

learning_rate_1�9Y4�YI       6%�	'�WN���A�*;


total_loss��@

error_Rq]C?

learning_rate_1�9�ATnI       6%�	J�WN���A�*;


total_loss�`�@

error_R�/_?

learning_rate_1�9��{=I       6%�	=XN���A�*;


total_loss;ݰ@

error_R�5K?

learning_rate_1�9��&�I       6%�	��XN���A�*;


total_loss=��@

error_R�)D?

learning_rate_1�95Dl�I       6%�	��XN���A�*;


total_lossSd�@

error_R{$\?

learning_rate_1�9���eI       6%�	mYN���A�*;


total_loss&�@

error_R��S?

learning_rate_1�95�mI       6%�	ewYN���A�*;


total_loss�+�@

error_R�\?

learning_rate_1�9��W[I       6%�	��YN���A�*;


total_loss̼�@

error_R�??

learning_rate_1�9�T��I       6%�	fZN���A�*;


total_loss�;@

error_R8UI?

learning_rate_1�9�,9I       6%�	bdZN���A�*;


total_lossX9�@

error_ROM?

learning_rate_1�9W±�I       6%�	�ZN���A�*;


total_loss��A

error_R�O?

learning_rate_1�9=�$�I       6%�	^�ZN���A�*;


total_lossv�@

error_Rt�J?

learning_rate_1�9o#dI       6%�	B[N���A�*;


total_loss��A

error_RlvW?

learning_rate_1�9{	��I       6%�	��[N���A�*;


total_lossQ��@

error_R;rA?

learning_rate_1�90���I       6%�	��[N���A�*;


total_loss�Г@

error_RҔU?

learning_rate_1�9I�I       6%�	�\N���A�*;


total_lossq��@

error_R�8`?

learning_rate_1�9haI       6%�	�W\N���A�*;


total_loss&Oe@

error_R�G?

learning_rate_1�9��j\I       6%�	�\N���A�*;


total_loss���@

error_R�N?

learning_rate_1�9�ܕpI       6%�	^�\N���A�*;


total_loss �@

error_R�4D?

learning_rate_1�94={�I       6%�	("]N���A�*;


total_lossͣ�@

error_R3�`?

learning_rate_1�9#�G\I       6%�	�l]N���A�*;


total_loss\A

error_R�HP?

learning_rate_1�9<��I       6%�	�]N���A�*;


total_loss_��@

error_R��S?

learning_rate_1�9V�LI       6%�	&^N���A�*;


total_loss���@

error_Rl�k?

learning_rate_1�9n'%]I       6%�	�Y^N���A�*;


total_loss�/�@

error_RCNS?

learning_rate_1�9m��OI       6%�	!�^N���A�*;


total_lossF�@

error_R$G?

learning_rate_1�9�7N}I       6%�	��^N���A�*;


total_loss]@�@

error_R��9?

learning_rate_1�9���{I       6%�	�C_N���A�*;


total_loss�8�@

error_R&lQ?

learning_rate_1�9�
"�I       6%�	Y�_N���A�*;


total_loss�Ş@

error_R
M?

learning_rate_1�9�Y�tI       6%�	Q�_N���A�*;


total_lossX��@

error_R`�M?

learning_rate_1�9���I       6%�	T%`N���A�*;


total_loss��@

error_R�P?

learning_rate_1�9����I       6%�	9k`N���A�*;


total_loss�H�@

error_R�NA?

learning_rate_1�9msiI       6%�	~�`N���A�*;


total_loss�x�@

error_R��a?

learning_rate_1�9|��I       6%�	*�`N���A�*;


total_loss�=�@

error_R�T?

learning_rate_1�9��g&I       6%�	6AaN���A�*;


total_loss��@

error_R�@?

learning_rate_1�9�rf�I       6%�	]�aN���A�*;


total_loss���@

error_R4�M?

learning_rate_1�9�fϧI       6%�	9�aN���A�*;


total_loss��|@

error_R�}S?

learning_rate_1�9�T��I       6%�	bbN���A�*;


total_loss؏�@

error_RW�J?

learning_rate_1�9;��I       6%�	.abN���A�*;


total_lossx��@

error_R�]J?

learning_rate_1�90U�I       6%�	s�bN���A�*;


total_loss?ӛ@

error_RfAK?

learning_rate_1�9׿��I       6%�	O�bN���A�*;


total_lossC�@

error_R mX?

learning_rate_1�9왽sI       6%�	�2cN���A�*;


total_loss��@

error_R�K?

learning_rate_1�9�<�I       6%�	��cN���A�*;


total_loss䟠@

error_R%�Y?

learning_rate_1�9��B�I       6%�	��cN���A�*;


total_lossb��@

error_R6N?

learning_rate_1�9�cb�I       6%�	�dN���A�*;


total_loss��@

error_R(�D?

learning_rate_1�9��hI       6%�	O[dN���A�*;


total_lossۊ�@

error_Rj�K?

learning_rate_1�9���I       6%�	��dN���A�*;


total_lossݢ�@

error_R2f?

learning_rate_1�9�+I       6%�	��dN���A�*;


total_loss��@

error_R�M?

learning_rate_1�9Y/��I       6%�	A:eN���A�*;


total_loss8��@

error_R��\?

learning_rate_1�9�1�GI       6%�	�eN���A�*;


total_loss�K�@

error_R��S?

learning_rate_1�9o�.BI       6%�	��eN���A�*;


total_lossbH�@

error_R�sR?

learning_rate_1�9:mFPI       6%�	fN���A�*;


total_loss�+�@

error_R��R?

learning_rate_1�9!f�I       6%�	UUfN���A�*;


total_loss��@

error_RZ�B?

learning_rate_1�9���/I       6%�	��fN���A�*;


total_loss�.v@

error_R��Q?

learning_rate_1�9��<�I       6%�	��fN���A�*;


total_losslt�@

error_R�"N?

learning_rate_1�9�¼�I       6%�	�gN���A�*;


total_lossn&�@

error_R�sT?

learning_rate_1�9�xx'I       6%�	�agN���A�*;


total_loss[�@

error_R�Q?

learning_rate_1�9��l�I       6%�	f�gN���A�*;


total_loss|�@

error_R^?

learning_rate_1�9�؀�I       6%�	��gN���A�*;


total_loss�@

error_R�/^?

learning_rate_1�9?�#I       6%�	�0hN���A�*;


total_loss� �@

error_R��H?

learning_rate_1�9��I       6%�	�xhN���A�*;


total_loss��@

error_R�M?

learning_rate_1�9�p{�I       6%�	��hN���A�*;


total_lossAdA

error_Rz�Q?

learning_rate_1�9'��I       6%�	�iN���A�*;


total_loss�N�@

error_R�BH?

learning_rate_1�9���I       6%�	�UiN���A�*;


total_loss~��@

error_R��@?

learning_rate_1�9�v�I       6%�	ذiN���A�*;


total_loss�0A

error_R�G?

learning_rate_1�9���I       6%�	��iN���A�*;


total_loss6��@

error_R��i?

learning_rate_1�9�֙�I       6%�	�8jN���A�*;


total_loss&v�@

error_R(�]?

learning_rate_1�9v�,JI       6%�	j�jN���A�*;


total_loss;Z�@

error_R:�??

learning_rate_1�9���I       6%�	�jN���A�*;


total_loss�A

error_R��K?

learning_rate_1�9�� I       6%�	kN���A�*;


total_loss���@

error_R�,B?

learning_rate_1�9p�׮I       6%�	�OkN���A�*;


total_lossa�@

error_R�N?

learning_rate_1�9J@j�I       6%�	7�kN���A�*;


total_loss֠@

error_R�'i?

learning_rate_1�9�oarI       6%�	t�kN���A�*;


total_losss��@

error_R��K?

learning_rate_1�9��H�I       6%�	�)lN���A�*;


total_lossUA

error_Rd�R?

learning_rate_1�9(�v�I       6%�	�ylN���A�*;


total_lossJ8�@

error_RHj;?

learning_rate_1�9��.�I       6%�	��lN���A�*;


total_loss�s�@

error_R}�^?

learning_rate_1�9\��I       6%�	mN���A�*;


total_lossZXA

error_R$�E?

learning_rate_1�9�D�I       6%�	�VmN���A�*;


total_loss���@

error_R�pN?

learning_rate_1�9[�<I       6%�	h�mN���A�*;


total_loss!s�@

error_R��R?

learning_rate_1�9��^/I       6%�	��mN���A�*;


total_loss��A

error_R��T?

learning_rate_1�9�4I       6%�	�(nN���A�*;


total_loss�,�@

error_R�bP?

learning_rate_1�9��<+I       6%�	�rnN���A�*;


total_loss���@

error_R�V?

learning_rate_1�9��pI       6%�	�nN���A�*;


total_loss�@

error_RRS?

learning_rate_1�9ѯ'�I       6%�	��nN���A�*;


total_loss��@

error_R._`?

learning_rate_1�9�#��I       6%�	�IoN���A�*;


total_lossHf�@

error_R��L?

learning_rate_1�9�`��I       6%�	��oN���A�*;


total_loss�t�@

error_R��W?

learning_rate_1�9彇I       6%�	\�oN���A�*;


total_loss7f�@

error_R[IH?

learning_rate_1�9�_�<I       6%�	pN���A�*;


total_loss6�A

error_RC�<?

learning_rate_1�9LA�I       6%�	�]pN���A�*;


total_loss��@

error_R�(N?

learning_rate_1�9R]�I       6%�	ѨpN���A�*;


total_loss���@

error_R�wL?

learning_rate_1�9X�zI       6%�	��pN���A�*;


total_loss~��@

error_R��Q?

learning_rate_1�9Q~�I       6%�	�EqN���A�*;


total_lossW��@

error_R)ZA?

learning_rate_1�92�I       6%�	=�qN���A�*;


total_lossJT�@

error_R|�e?

learning_rate_1�9%��]I       6%�	��qN���A�*;


total_loss�}�@

error_R�ZI?

learning_rate_1�9�X�I       6%�	�rN���A�*;


total_loss��@

error_R��U?

learning_rate_1�9�(I       6%�	�_rN���A�*;


total_loss��r@

error_R�@?

learning_rate_1�9��V6I       6%�	}�rN���A�*;


total_lossjy�@

error_R�A?

learning_rate_1�9J��9I       6%�	��rN���A�*;


total_loss�@

error_R�X?

learning_rate_1�9޸GnI       6%�	6sN���A�*;


total_lossEw�@

error_RT�M?

learning_rate_1�9
#��I       6%�	�~sN���A�*;


total_loss��@

error_RJ�^?

learning_rate_1�9�h�nI       6%�	��sN���A�*;


total_loss|I�@

error_R)E?

learning_rate_1�9%�]RI       6%�	/tN���A�*;


total_loss��@

error_RϤB?

learning_rate_1�9�m��I       6%�	�^tN���A�*;


total_loss��@

error_R`�T?

learning_rate_1�96*�TI       6%�	��tN���A�*;


total_loss���@

error_R�+`?

learning_rate_1�9f�t�I       6%�	�tN���A�*;


total_losss0�@

error_R�K?

learning_rate_1�9!��I       6%�	
.uN���A�*;


total_lossZ��@

error_R
=M?

learning_rate_1�9�K4RI       6%�	i�uN���A�*;


total_loss=��@

error_R�L?

learning_rate_1�9��I       6%�	L�uN���A�*;


total_loss�d@

error_R��>?

learning_rate_1�9�բ�I       6%�	�vN���A�*;


total_lossD��@

error_RwK\?

learning_rate_1�9!���I       6%�	�avN���A�*;


total_loss[u�@

error_R_�X?

learning_rate_1�9*�aI       6%�	j�vN���A�*;


total_loss�Z�@

error_RQV?

learning_rate_1�9��OI       6%�	#�vN���A�*;


total_loss6ϥ@

error_R�`J?

learning_rate_1�96v�)I       6%�	�1wN���A�*;


total_loss���@

error_RO�Q?

learning_rate_1�9���I       6%�	�zwN���A�*;


total_loss��@

error_R�A?

learning_rate_1�9��EI       6%�	��wN���A�*;


total_losslH�@

error_R��M?

learning_rate_1�9���I       6%�	�xN���A�*;


total_losseY�@

error_R�N?

learning_rate_1�9���
I       6%�	x]xN���A�*;


total_loss��@

error_R��A?

learning_rate_1�9�#T�I       6%�	��xN���A�*;


total_loss��@

error_RGR?

learning_rate_1�9"&I       6%�	��xN���A�*;


total_lossW��@

error_R�|O?

learning_rate_1�9���I       6%�	�OyN���A�*;


total_loss���@

error_RíT?

learning_rate_1�90�WI       6%�	P�yN���A�*;


total_lossf�@

error_R?L?

learning_rate_1�9���I       6%�	�yN���A�*;


total_loss��@

error_R��Z?

learning_rate_1�9�_I       6%�	CzN���A�*;


total_loss�؋@

error_R��U?

learning_rate_1�9K|9I       6%�	~�zN���A�*;


total_loss�3�@

error_R�R?

learning_rate_1�9Z�W�I       6%�	B�zN���A�*;


total_lossQ�@

error_R6�J?

learning_rate_1�9�c�I       6%�	p{N���A�*;


total_loss�ש@

error_R��L?

learning_rate_1�9�b�I       6%�	�X{N���A�*;


total_lossd��@

error_Rd�X?

learning_rate_1�96#
�I       6%�	��{N���A�*;


total_lossc��@

error_R!tU?

learning_rate_1�9XP
�I       6%�	��{N���A�*;


total_loss<�@

error_Rl�W?

learning_rate_1�9p�<�I       6%�	�2|N���A�*;


total_losst��@

error_R�e?

learning_rate_1�9��}RI       6%�	�x|N���A�*;


total_loss���@

error_R\?

learning_rate_1�9>��FI       6%�	��|N���A�*;


total_loss,$�@

error_R_�X?

learning_rate_1�9�_��I       6%�	��|N���A�*;


total_loss6�@

error_RH�J?

learning_rate_1�9����I       6%�	�E}N���A�*;


total_loss���@

error_R�8P?

learning_rate_1�9폾&I       6%�	��}N���A�*;


total_loss8��@

error_R��W?

learning_rate_1�9���NI       6%�	��}N���A�*;


total_loss�
�@

error_RZB?

learning_rate_1�90c��I       6%�	=~N���A�*;


total_loss䯺@

error_R�P>?

learning_rate_1�9�II       6%�	�s~N���A�*;


total_loss�i�@

error_R�VF?

learning_rate_1�9��!�I       6%�	o�~N���A�*;


total_loss>�@

error_R=�L?

learning_rate_1�9��DGI       6%�	;N���A�*;


total_lossxo�@

error_R�S?

learning_rate_1�9�5_�I       6%�	�FN���A�*;


total_lossw_�@

error_R�M?

learning_rate_1�9̇��I       6%�	��N���A�*;


total_lossId�@

error_R�S?

learning_rate_1�9U�� I       6%�	��N���A�*;


total_loss�V�@

error_R#�D?

learning_rate_1�9��I       6%�	!�N���A�*;


total_loss�ў@

error_R��G?

learning_rate_1�9�0�I       6%�	:n�N���A�*;


total_loss֓�@

error_R�_N?

learning_rate_1�9,�	GI       6%�	5��N���A�*;


total_lossqԬ@

error_R$Q?

learning_rate_1�9Q�*]I       6%�	_��N���A�*;


total_loss�3A

error_R�/q?

learning_rate_1�9ry�I       6%�	�?�N���A�*;


total_loss>�@

error_RM�Z?

learning_rate_1�9x�I       6%�	���N���A�*;


total_loss�A

error_R�T?

learning_rate_1�9��KzI       6%�	;ɁN���A�*;


total_loss3��@

error_R�<?

learning_rate_1�9B�/I       6%�	��N���A�*;


total_loss֪�@

error_R�c^?

learning_rate_1�9�}��I       6%�	�]�N���A�*;


total_loss.y�@

error_RW$Z?

learning_rate_1�9�!zI       6%�	���N���A�*;


total_loss� A

error_R�{S?

learning_rate_1�9Cz:UI       6%�	8�N���A�*;


total_loss@/�@

error_R�pZ?

learning_rate_1�9�Z�,I       6%�	1�N���A�*;


total_loss<l@

error_RC?

learning_rate_1�9W-I       6%�	x�N���A�*;


total_lossW�@

error_R}P?

learning_rate_1�9"o�I       6%�	k��N���A�*;


total_loss}�@

error_Ra|M?

learning_rate_1�9xӉ�I       6%�	���N���A�*;


total_loss�] A

error_RZtM?

learning_rate_1�9���I       6%�	�A�N���A�*;


total_loss��@

error_R�`?

learning_rate_1�9��zoI       6%�	7��N���A�*;


total_loss}m�@

error_R��Q?

learning_rate_1�9ʜI       6%�	%ӄN���A�*;


total_loss�/�@

error_R��3?

learning_rate_1�9l{n�I       6%�	��N���A�*;


total_loss��@

error_R�rM?

learning_rate_1�9��IEI       6%�	�\�N���A�*;


total_lossA	�@

error_R��H?

learning_rate_1�9���I       6%�	���N���A�*;


total_loss��@

error_R�[?

learning_rate_1�9�ԯ4I       6%�	��N���A�*;


total_loss��@

error_RJIY?

learning_rate_1�9�DG#I       6%�	�1�N���A�*;


total_loss�B�@

error_Rd�`?

learning_rate_1�9+N��I       6%�	�y�N���A�*;


total_loss}�A

error_R&�O?

learning_rate_1�9�!�`I       6%�	�N���A�*;


total_loss�z�@

error_R��U?

learning_rate_1�90�2PI       6%�	��N���A�*;


total_loss��A

error_RMNB?

learning_rate_1�9�/I       6%�	�H�N���A�*;


total_loss`��@

error_R�oC?

learning_rate_1�9�>g�I       6%�	ԍ�N���A�*;


total_loss1��@

error_RF�C?

learning_rate_1�9x���I       6%�	�҇N���A�*;


total_loss6U�@

error_Ra�R?

learning_rate_1�9Q���I       6%�	��N���A�*;


total_lossW1�@

error_RRM?

learning_rate_1�9(I       6%�	�^�N���A�*;


total_lossR��@

error_R�U]?

learning_rate_1�9x�جI       6%�	࢈N���A�*;


total_lossMP�@

error_RM�S?

learning_rate_1�9�9<oI       6%�	U�N���A�*;


total_lossjٶ@

error_R�A??

learning_rate_1�90ՋI       6%�	r9�N���A�*;


total_loss�r�@

error_R@I?

learning_rate_1�9,�|I       6%�	���N���A�*;


total_loss�i�@

error_R�yI?

learning_rate_1�9n��I       6%�	.�N���A�*;


total_loss$M�@

error_R@�W?

learning_rate_1�9� � I       6%�	�%�N���A�*;


total_loss�U�@

error_R�eN?

learning_rate_1�9		I       6%�	�f�N���A�*;


total_loss��@

error_RD+X?

learning_rate_1�9':!I       6%�	��N���A�*;


total_lossZY�@

error_RD�R?

learning_rate_1�9�!įI       6%�	,��N���A�*;


total_loss
ݪ@

error_Rl.I?

learning_rate_1�9,�l�I       6%�	�=�N���A�*;


total_loss�&�@

error_R,
X?

learning_rate_1�9eւ�I       6%�	=��N���A�*;


total_loss���@

error_R�*_?

learning_rate_1�9�oI       6%�	SċN���A�*;


total_loss���@

error_R\MC?

learning_rate_1�9����I       6%�	E�N���A�*;


total_loss?��@

error_R��<?

learning_rate_1�91b��I       6%�	@P�N���A�*;


total_loss���@

error_Ri�m?

learning_rate_1�9���I       6%�	���N���A�*;


total_lossl8�@

error_R�$G?

learning_rate_1�9 
�zI       6%�	X݌N���A�*;


total_loss�2�@

error_R��9?

learning_rate_1�96�SrI       6%�	� �N���A�*;


total_loss�0�@

error_R�I?

learning_rate_1�9��fI       6%�	f�N���A�*;


total_loss�/�@

error_RV�M?

learning_rate_1�92�dI       6%�	z��N���A�*;


total_loss%CA

error_R��H?

learning_rate_1�9�F��I       6%�	��N���A�*;


total_loss\�A

error_R��Q?

learning_rate_1�9E1�I       6%�	�1�N���A�*;


total_lossA�@

error_R�`?

learning_rate_1�9��ןI       6%�	t�N���A�*;


total_loss��@

error_R|�h?

learning_rate_1�9~�EI       6%�	淎N���A�*;


total_loss?G�@

error_R�>R?

learning_rate_1�9���I       6%�	���N���A�*;


total_loss��@

error_R�/L?

learning_rate_1�96%��I       6%�	�>�N���A�*;


total_loss� �@

error_R=�S?

learning_rate_1�9��WI       6%�	&��N���A�*;


total_loss
�@

error_RJ�G?

learning_rate_1�9��ÄI       6%�	�ɏN���A�*;


total_loss��@

error_R ?S?

learning_rate_1�9<yV�I       6%�	��N���A�*;


total_loss�tA

error_Rs�T?

learning_rate_1�9&n<I       6%�	�_�N���A�*;


total_lossЧ@

error_R��S?

learning_rate_1�9؋IlI       6%�	A��N���A�*;


total_lossj�@

error_R�P?

learning_rate_1�9s9}aI       6%�	C�N���A�*;


total_lossʝ@

error_R�[K?

learning_rate_1�9o��0I       6%�	�3�N���A�*;


total_loss�i�@

error_R�+L?

learning_rate_1�9[�";I       6%�	�y�N���A�*;


total_loss@

error_Rc�I?

learning_rate_1�9�@��I       6%�	yN���A�*;


total_lossR�@

error_R�{D?

learning_rate_1�9��>I       6%�	(�N���A�*;


total_loss�̱@

error_R.QN?

learning_rate_1�9�R# I       6%�	GX�N���A�*;


total_loss<�@

error_R��[?

learning_rate_1�9�*I       6%�	�N���A�*;


total_loss�yr@

error_R-	\?

learning_rate_1�9d��LI       6%�	t�N���A�*;


total_losska�@

error_Rr"K?

learning_rate_1�9�{W�I       6%�	z5�N���A�*;


total_loss�9�@

error_Rl�_?

learning_rate_1�9r��I       6%�		��N���A�*;


total_lossS�@

error_R�HA?

learning_rate_1�9~d �I       6%�	�˓N���A�*;


total_loss��@

error_R�Q?

learning_rate_1�9��~�I       6%�	�N���A�*;


total_lossM��@

error_RCf?

learning_rate_1�9���I       6%�	{i�N���A�*;


total_loss�@

error_R�R?

learning_rate_1�9�8Q I       6%�	`��N���A�*;


total_loss��@

error_Rg[?

learning_rate_1�9H��:I       6%�	���N���A�*;


total_lossJ�@

error_R1YH?

learning_rate_1�9����I       6%�	H�N���A�*;


total_loss���@

error_R߂Y?

learning_rate_1�9C˰�I       6%�	k��N���A�*;


total_loss`��@

error_Rf�J?

learning_rate_1�93]�I       6%�	~�N���A�*;


total_loss��@

error_R�E?

learning_rate_1�9� ]
I       6%�	N,�N���A�*;


total_loss7��@

error_R��Y?

learning_rate_1�9M�c�I       6%�	Js�N���A�*;


total_loss�+�@

error_R��E?

learning_rate_1�9�@-�I       6%�		��N���A�*;


total_loss�)A

error_R�3a?

learning_rate_1�9��5I       6%�	��N���A�*;


total_lossz�c@

error_R��B?

learning_rate_1�9]��/I       6%�	,V�N���A�*;


total_loss���@

error_R�	Q?

learning_rate_1�9��Q2I       6%�	ڤ�N���A�*;


total_lossι�@

error_R&:F?

learning_rate_1�9;~~jI       6%�	p�N���A�*;


total_loss\��@

error_Rj�M?

learning_rate_1�9�#I       6%�	M9�N���A�*;


total_loss�!�@

error_R M?

learning_rate_1�9�]_�I       6%�	8|�N���A�*;


total_loss��@

error_R��F?

learning_rate_1�9���I       6%�	�N���A�*;


total_loss�+�@

error_R�oR?

learning_rate_1�9-{�I       6%�	��N���A�*;


total_loss��@

error_R,�??

learning_rate_1�9�T*]I       6%�	q�N���A�*;


total_loss�Щ@

error_R��4?

learning_rate_1�9!(&I       6%�	�ЙN���A�*;


total_lossX	A

error_R%M?

learning_rate_1�9]�PtI       6%�	��N���A�*;


total_lossFr�@

error_RJ�O?

learning_rate_1�9�ː�I       6%�	fV�N���A�*;


total_loss�ج@

error_R8_;?

learning_rate_1�9�.�}I       6%�	s��N���A�*;


total_lossc4@

error_R�LE?

learning_rate_1�9����I       6%�	ߚN���A�*;


total_loss���@

error_Rџ_?

learning_rate_1�9#=�I       6%�	$�N���A�*;


total_loss���@

error_R�G?

learning_rate_1�9�Z��I       6%�	}g�N���A�*;


total_lossoX�@

error_RX�I?

learning_rate_1�9�l�.I       6%�	���N���A�*;


total_loss�+A

error_R�VE?

learning_rate_1�9�(�I       6%�	��N���A�*;


total_loss��@

error_RfRP?

learning_rate_1�9#�I       6%�	�4�N���A�*;


total_lossx�@

error_RHN?

learning_rate_1�9�YCEI       6%�	�N���A�*;


total_loss��@

error_R��U?

learning_rate_1�9�G`TI       6%�	TĜN���A�*;


total_lossz�@

error_R,�Y?

learning_rate_1�9�A0�I       6%�	7
�N���A�*;


total_loss��@

error_R�c??

learning_rate_1�9#=I       6%�	5O�N���A�*;


total_loss���@

error_R�U?

learning_rate_1�9���I       6%�	P��N���A�*;


total_lossC��@

error_R8DM?

learning_rate_1�9��I       6%�	D�N���A�*;


total_lossf�@

error_R��N?

learning_rate_1�9�:,I       6%�	�&�N���A�*;


total_lossP�@

error_R�R?

learning_rate_1�9u�/�I       6%�	�n�N���A�*;


total_loss�d�@

error_RC�[?

learning_rate_1�9`�I       6%�	�ϞN���A�*;


total_lossWM�@

error_RD�L?

learning_rate_1�9�V�I       6%�	r�N���A�*;


total_loss}�@

error_R�"L?

learning_rate_1�9���I       6%�	u`�N���A�*;


total_lossܤ@

error_R�BT?

learning_rate_1�9P��I       6%�	��N���A�*;


total_loss�
�@

error_R�>P?

learning_rate_1�9xU�eI       6%�	��N���A�*;


total_loss�,�@

error_R�yY?

learning_rate_1�9����I       6%�	`�N���A�*;


total_loss7��@

error_R{pU?

learning_rate_1�9�o�vI       6%�		��N���A�*;


total_lossHm�@

error_R,NL?

learning_rate_1�9.�O�I       6%�	m�N���A�*;


total_loss8��@

error_R��D?

learning_rate_1�9 �I       6%�	4�N���A�*;


total_lossJ¡@

error_R!�D?

learning_rate_1�9��e>I       6%�	!z�N���A�*;


total_loss�ڸ@

error_Rv�Y?

learning_rate_1�9�.�I       6%�	wȡN���A�*;


total_loss�ڛ@

error_R��A?

learning_rate_1�9_��I       6%�	�N���A�*;


total_lossH�Z@

error_R��L?

learning_rate_1�9�K~�I       6%�	j_�N���A�*;


total_loss���@

error_R{:a?

learning_rate_1�9
PI       6%�	���N���A�*;


total_lossm��@

error_Ri]?

learning_rate_1�9��x�I       6%�	���N���A�*;


total_loss@��@

error_R��N?

learning_rate_1�9�fȑI       6%�	�A�N���A�*;


total_loss�j�@

error_R��C?

learning_rate_1�9����I       6%�	��N���A�*;


total_loss��@

error_Rc�G?

learning_rate_1�9tC�>I       6%�	�֣N���A�*;


total_lossw��@

error_R�&E?

learning_rate_1�9��I       6%�	��N���A�*;


total_lossW_�@

error_R�B?

learning_rate_1�9*�6�I       6%�	�e�N���A�*;


total_losseq�@

error_R8^I?

learning_rate_1�9,#r�I       6%�	���N���A�*;


total_loss$�@

error_R��D?

learning_rate_1�9��1I       6%�	��N���A�*;


total_loss��@

error_Rv�R?

learning_rate_1�9
"`[I       6%�	�?�N���A�*;


total_lossh�@

error_R�@c?

learning_rate_1�9%ߗUI       6%�	{��N���A�*;


total_lossM�@

error_R֠K?

learning_rate_1�9�e�I       6%�	B̥N���A�*;


total_lossf�@

error_RWQ?

learning_rate_1�9��UI       6%�	��N���A�*;


total_loss�,�@

error_RW[?

learning_rate_1�9%��6I       6%�	�S�N���A�*;


total_lossv��@

error_R��Y?

learning_rate_1�9��.nI       6%�	Y��N���A�*;


total_loss-��@

error_R�tF?

learning_rate_1�9
9'I       6%�	7ݦN���A�*;


total_loss�"�@

error_R��L?

learning_rate_1�9 �%I       6%�	�$�N���A�*;


total_lossŽ�@

error_R��O?

learning_rate_1�9��rhI       6%�	�k�N���A�*;


total_loss��@

error_R�79?

learning_rate_1�9�I       6%�	���N���A�*;


total_loss���@

error_Rv�X?

learning_rate_1�9q�yYI       6%�	�N���A�*;


total_loss�@

error_R�U?

learning_rate_1�9�_8I       6%�	�O�N���A�*;


total_loss-n�@

error_Rs�\?

learning_rate_1�9��pI       6%�	���N���A�*;


total_lossz��@

error_R�:P?

learning_rate_1�95y��I       6%�	��N���A�*;


total_loss�x�@

error_RI�V?

learning_rate_1�9Z���I       6%�	�%�N���A�*;


total_loss(Ƌ@

error_R @P?

learning_rate_1�9��VlI       6%�	L��N���A�*;


total_loss�@

error_Rl5[?

learning_rate_1�9M'�I       6%�	rөN���A�*;


total_loss�+�@

error_R�M?

learning_rate_1�9URD�I       6%�	��N���A�*;


total_loss�A

error_R�]e?

learning_rate_1�9�yaI       6%�	�\�N���A�*;


total_loss��@

error_R�+G?

learning_rate_1�9�(��I       6%�	k��N���A�*;


total_lossO�@

error_R)�A?

learning_rate_1�9a���I       6%�	9�N���A�*;


total_loss���@

error_R�V?

learning_rate_1�9���XI       6%�	�'�N���A�*;


total_loss<_�@

error_R_R?

learning_rate_1�9nq~�I       6%�	u�N���A�*;


total_lossm�@

error_R�af?

learning_rate_1�9�J�I       6%�	���N���A�*;


total_loss�9�@

error_R��X?

learning_rate_1�9�8WI       6%�	~�N���A�*;


total_loss|P�@

error_REWR?

learning_rate_1�9*���I       6%�	&L�N���A�*;


total_loss,�@

error_R��N?

learning_rate_1�9�
]I       6%�	���N���A�*;


total_loss�o�@

error_R�5S?

learning_rate_1�9��bI       6%�	�լN���A�*;


total_lossћ�@

error_R��g?

learning_rate_1�9��l�I       6%�	��N���A�*;


total_loss'&�@

error_R۷J?

learning_rate_1�9!��I       6%�	�\�N���A�*;


total_loss;�@

error_R$3Q?

learning_rate_1�9 x�I       6%�	x��N���A�*;


total_loss���@

error_REV?

learning_rate_1�9<�/I       6%�	 �N���A�*;


total_loss�@

error_R�?m?

learning_rate_1�9/׽I       6%�	v'�N���A�*;


total_loss��@

error_R.,Y?

learning_rate_1�9�"|�I       6%�	
l�N���A�*;


total_losss�A

error_RcXT?

learning_rate_1�9A��I       6%�	���N���A�*;


total_loss5�@

error_R��X?

learning_rate_1�9dS'BI       6%�	��N���A�*;


total_loss�@

error_R�A?

learning_rate_1�9��<?I       6%�	'6�N���A�*;


total_loss=�@

error_R_�G?

learning_rate_1�9 I�I       6%�	�z�N���A�*;


total_loss]9�@

error_R��S?

learning_rate_1�9U�M�I       6%�	���N���A�*;


total_loss��@

error_RF?

learning_rate_1�9oC�#I       6%�	��N���A�*;


total_loss�U�@

error_R�G?

learning_rate_1�9a��I       6%�	�E�N���A�*;


total_loss�D�@

error_R��L?

learning_rate_1�9����I       6%�	���N���A�*;


total_lossu�@

error_Rv/c?

learning_rate_1�9q�F�I       6%�	԰N���A�*;


total_loss�f�@

error_R�U?

learning_rate_1�9�IMAI       6%�	��N���A�*;


total_lossZ��@

error_Rx�L?

learning_rate_1�99��I       6%�	c\�N���A�*;


total_loss�C�@

error_R��O?

learning_rate_1�9z�%I       6%�	r��N���A�*;


total_losst�A

error_R|�Z?

learning_rate_1�9���I       6%�	�N���A�*;


total_loss�Ԫ@

error_R��M?

learning_rate_1�9֪�I       6%�	�9�N���A�*;


total_loss(��@

error_R��W?

learning_rate_1�9{d"}I       6%�	��N���A�*;


total_loss��$A

error_RE?

learning_rate_1�9�C
I       6%�	&ѲN���A�*;


total_loss�x�@

error_RT?

learning_rate_1�9���PI       6%�	A�N���A�*;


total_loss͡@

error_R�XK?

learning_rate_1�9o�7�I       6%�	�b�N���A�*;


total_loss"�@

error_R��R?

learning_rate_1�9���I       6%�	H��N���A�*;


total_lossrҙ@

error_R�Y?

learning_rate_1�9���#I       6%�	^�N���A�*;


total_lossMҟ@

error_RN3R?

learning_rate_1�9�z�/I       6%�	5�N���A�*;


total_loss��}@

error_R��Q?

learning_rate_1�9�z��I       6%�	y�N���A�*;


total_loss�R�@

error_R��H?

learning_rate_1�9��`�I       6%�	ôN���A�*;


total_loss��@

error_R%�;?

learning_rate_1�9�� �I       6%�	��N���A�*;


total_loss�ե@

error_R�@f?

learning_rate_1�9l��I       6%�	�Z�N���A�*;


total_loss@�A

error_R\�R?

learning_rate_1�9҂�lI       6%�	�N���A�*;


total_loss��@

error_R��V?

learning_rate_1�9�V�HI       6%�	7�N���A�*;


total_loss�A�@

error_R��U?

learning_rate_1�9B���I       6%�	1�N���A�*;


total_loss���@

error_R��P?

learning_rate_1�9���I       6%�	5x�N���A�*;


total_loss)�@

error_R�*O?

learning_rate_1�94�R�I       6%�	��N���A�*;


total_loss!��@

error_R��L?

learning_rate_1�9υpI       6%�	��N���A�*;


total_lossԓ�@

error_RWQ?

learning_rate_1�98l��I       6%�	jF�N���A�*;


total_loss|h�@

error_RT�U?

learning_rate_1�9����I       6%�	ě�N���A�*;


total_loss6�A

error_R��\?

learning_rate_1�99_K~I       6%�	;�N���A�*;


total_loss6�@

error_R��N?

learning_rate_1�9�L��I       6%�	)�N���A�*;


total_loss�m�@

error_R�G?

learning_rate_1�9i���I       6%�	�r�N���A�*;


total_loss��A

error_R��[?

learning_rate_1�9�ҧI       6%�	ѻ�N���A�*;


total_loss/�jA

error_R݋H?

learning_rate_1�9D&�jI       6%�	�N���A�*;


total_loss�W�@

error_R��W?

learning_rate_1�9T�I       6%�	^U�N���A�*;


total_lossj��@

error_R8R?

learning_rate_1�9�c��I       6%�	蹹N���A�*;


total_loss���@

error_R�_N?

learning_rate_1�9@�~I       6%�	��N���A�*;


total_lossj}A

error_Rf�P?

learning_rate_1�9�ZQ�I       6%�	QL�N���A�*;


total_loss�O�@

error_R�]?

learning_rate_1�9T���I       6%�	��N���A�*;


total_lossz�A

error_Ro]_?

learning_rate_1�9(��I       6%�	�ۺN���A�*;


total_lossC��@

error_R�Z?

learning_rate_1�9��h,I       6%�	�+�N���A�*;


total_loss⭢@

error_R�*O?

learning_rate_1�9�
��I       6%�	Yy�N���A�*;


total_lossx��@

error_R\�??

learning_rate_1�9h��MI       6%�	l��N���A�*;


total_loss3�@

error_R8N?

learning_rate_1�9ܝt5I       6%�	u�N���A�*;


total_loss���@

error_R3�^?

learning_rate_1�9�	QhI       6%�	PF�N���A�*;


total_loss$S�@

error_R�A?

learning_rate_1�9��I       6%�	���N���A�*;


total_loss�D�@

error_R9I?

learning_rate_1�9��CI       6%�	̼N���A�*;


total_lossDj�@

error_R'C?

learning_rate_1�9��RI       6%�	A�N���A�*;


total_lossA�@

error_RQGH?

learning_rate_1�9γ�I       6%�	�Q�N���A�*;


total_loss�c�@

error_R�K?

learning_rate_1�9��I�I       6%�	l��N���A�*;


total_loss<?�@

error_R��X?

learning_rate_1�97�,�I       6%�	L۽N���A�*;


total_loss*�@

error_R�<?

learning_rate_1�9���I       6%�	 �N���A�*;


total_loss`�@

error_R��J?

learning_rate_1�9����I       6%�	�b�N���A�*;


total_loss��@

error_R��F?

learning_rate_1�9!��tI       6%�	���N���A�*;


total_lossv#A

error_Rq�T?

learning_rate_1�9�n �I       6%�	�N���A�*;


total_loss���@

error_R��Z?

learning_rate_1�9��?I       6%�	a/�N���A�*;


total_loss؅�@

error_R3�O?

learning_rate_1�98W�I       6%�	+y�N���A�*;


total_loss7��@

error_R��=?

learning_rate_1�9AKbI       6%�	�ÿN���A�*;


total_loss
�A

error_RN�H?

learning_rate_1�9��NI       6%�	��N���A�*;


total_loss*�@

error_R҇N?

learning_rate_1�9�^�~I       6%�	�T�N���A�*;


total_loss���@

error_Rx|@?

learning_rate_1�9�)�I       6%�	���N���A�*;


total_loss@��@

error_RC�D?

learning_rate_1�9\�
VI       6%�	^��N���A�*;


total_loss�j�@

error_R��N?

learning_rate_1�9�t�I       6%�	�2�N���A�*;


total_lossV�@

error_RZM?

learning_rate_1�9|yޢI       6%�	Y��N���A�*;


total_loss$�k@

error_Rm+X?

learning_rate_1�9킦iI       6%�	���N���A�*;


total_lossN��@

error_RW�S?

learning_rate_1�9�b*�I       6%�	��N���A�*;


total_lossۤ�@

error_R�wP?

learning_rate_1�9�m$�I       6%�	�c�N���A�*;


total_loss�h�@

error_R%J?

learning_rate_1�9K%+mI       6%�	���N���A�*;


total_loss06�@

error_R;%_?

learning_rate_1�9�c��I       6%�	��N���A�*;


total_loss8u�@

error_R�#a?

learning_rate_1�9��f�I       6%�	g@�N���A�*;


total_lossm4�@

error_R]@?

learning_rate_1�9롗I       6%�	<��N���A�*;


total_loss�Ŧ@

error_R3�;?

learning_rate_1�9�W�CI       6%�	���N���A�*;


total_loss�ԛ@

error_Rv}^?

learning_rate_1�9�b��I       6%�	��N���A�*;


total_loss}MA

error_R�S?

learning_rate_1�9x=~�I       6%�	^V�N���A�*;


total_loss��@

error_RW�@?

learning_rate_1�9����I       6%�	���N���A�*;


total_lossu�@

error_R�\?

learning_rate_1�92��I       6%�	���N���A�*;


total_loss]/�@

error_R��X?

learning_rate_1�9e.U�I       6%�	�8�N���A�*;


total_lossZA

error_R}�M?

learning_rate_1�9�� �I       6%�	:��N���A�*;


total_lossU�!A

error_RO7a?

learning_rate_1�9�1Z�I       6%�	��N���A�*;


total_loss���@

error_R��M?

learning_rate_1�9t��#I       6%�	a�N���A�*;


total_loss�dHA

error_R�QO?

learning_rate_1�9)���I       6%�	VO�N���A�*;


total_loss�^A

error_Ra�D?

learning_rate_1�9'.nI       6%�	s��N���A�*;


total_loss���@

error_RI�[?

learning_rate_1�9���I       6%�	?��N���A�*;


total_loss��@

error_Rn�D?

learning_rate_1�9T�� I       6%�	1�N���A�*;


total_lossA

error_RH�I?

learning_rate_1�9:pI       6%�	�\�N���A�*;


total_loss���@

error_R�;?

learning_rate_1�9-�$I       6%�	ӡ�N���A�*;


total_loss��@

error_R
/M?

learning_rate_1�9��^ I       6%�	���N���A�*;


total_losszQ�@

error_R��G?

learning_rate_1�9?�fCI       6%�	7.�N���A�*;


total_lossSU�@

error_R�P?

learning_rate_1�9�q~I       6%�	t�N���A�*;


total_lossc��@

error_R�	R?

learning_rate_1�9O٢&I       6%�	���N���A�*;


total_loss���@

error_R4�Y?

learning_rate_1�99�l�I       6%�	��N���A�*;


total_lossSP�@

error_RnVE?

learning_rate_1�9��I       6%�	�P�N���A�*;


total_loss~��@

error_Rח??

learning_rate_1�9���I       6%�	R��N���A�*;


total_lossXE�@

error_R��U?

learning_rate_1�9l��I       6%�	F��N���A�*;


total_loss*��@

error_R$�B?

learning_rate_1�95��I       6%�	�5�N���A�*;


total_loss[(�@

error_R*�G?

learning_rate_1�9�sI       6%�	�x�N���A�*;


total_loss�/�@

error_Rr&X?

learning_rate_1�9��I       6%�	#��N���A�*;


total_lossh�@

error_Rl+Q?

learning_rate_1�9+�
�I       6%�	&�N���A�*;


total_loss� A

error_RvT?

learning_rate_1�9�?��I       6%�	oO�N���A�*;


total_loss�Ӵ@

error_R&�F?

learning_rate_1�9�]hI       6%�	|��N���A�*;


total_lossvP�@

error_Rd�J?

learning_rate_1�9)7�
I       6%�	���N���A�*;


total_lossׅ�@

error_R&�[?

learning_rate_1�9lA��I       6%�	,%�N���A�*;


total_loss�x�@

error_R�JG?

learning_rate_1�9A���I       6%�	Vj�N���A�*;


total_lossfب@

error_R��I?

learning_rate_1�91K[I       6%�	���N���A�*;


total_loss��@

error_R��Z?

learning_rate_1�9��*2I       6%�	���N���A�*;


total_loss�W�@

error_R�dO?

learning_rate_1�9��}eI       6%�	�<�N���A�*;


total_loss�wA

error_R\�A?

learning_rate_1�9��ɨI       6%�	n~�N���A�*;


total_loss�/A

error_R�tK?

learning_rate_1�9~�f>I       6%�	$��N���A�*;


total_loss� �@

error_R�Q?

learning_rate_1�9�4
HI       6%�	�	�N���A�*;


total_loss� �@

error_R��X?

learning_rate_1�9��I       6%�	�M�N���A�*;


total_lossv*�@

error_R��T?

learning_rate_1�9[6Z�I       6%�	���N���A�*;


total_loss�R�@

error_R�E?

learning_rate_1�99ݤ�I       6%�	���N���A�*;


total_loss|�@

error_R�(Z?

learning_rate_1�9 R�I       6%�	n�N���A�*;


total_loss��@

error_RR�P?

learning_rate_1�9�dI       6%�	Z�N���A�*;


total_lossz�@

error_Rl&P?

learning_rate_1�9I�pUI       6%�	<��N���A�*;


total_lossIޡ@

error_R(�I?

learning_rate_1�9	f3I       6%�		��N���A�*;


total_loss@��@

error_R�N?

learning_rate_1�9ЊI       6%�	�!�N���A�*;


total_losso^A

error_R�%[?

learning_rate_1�9ؠ�.I       6%�	�d�N���A�*;


total_lossf��@

error_R�
F?

learning_rate_1�9�n�~I       6%�	s��N���A�*;


total_loss��@

error_RH�;?

learning_rate_1�9t龵I       6%�	���N���A�*;


total_lossa��@

error_R�R?

learning_rate_1�9���]I       6%�	�2�N���A�*;


total_loss6��@

error_RجH?

learning_rate_1�9�I�I       6%�	�}�N���A�*;


total_loss�8�@

error_RA�^?

learning_rate_1�9�^��I       6%�	���N���A�*;


total_loss���@

error_R4Sb?

learning_rate_1�95djKI       6%�	�N���A�*;


total_losso��@

error_R�R?

learning_rate_1�9yK7�I       6%�	5F�N���A�*;


total_lossJ��@

error_R�Q?

learning_rate_1�9¶b/I       6%�	/��N���A�*;


total_lossV7�@

error_R��E?

learning_rate_1�9�M�AI       6%�	���N���A�*;


total_loss��@

error_RCEE?

learning_rate_1�9�6&�I       6%�	��N���A�*;


total_lossE�@

error_R�9S?

learning_rate_1�9Ϊa�I       6%�	�b�N���A�*;


total_loss4P A

error_R�SU?

learning_rate_1�9�{ǌI       6%�	���N���A�*;


total_loss�,�@

error_R1XP?

learning_rate_1�9B}Q�I       6%�	L��N���A�*;


total_loss�@

error_RW�Z?

learning_rate_1�9fb�"I       6%�	�1�N���A�*;


total_lossn��@

error_R�gS?

learning_rate_1�9�]֕I       6%�	`v�N���A�*;


total_loss�V�@

error_R�[P?

learning_rate_1�9���?I       6%�	��N���A�*;


total_loss�,�@

error_RO,K?

learning_rate_1�9�o_I       6%�	��N���A�*;


total_loss���@

error_R�iT?

learning_rate_1�9�\BI       6%�	9X�N���A�*;


total_losss�@

error_R1tJ?

learning_rate_1�9[�HI       6%�	���N���A�*;


total_loss�A@

error_RDJ??

learning_rate_1�9�}1I       6%�	���N���A�*;


total_loss���@

error_R�\U?

learning_rate_1�9�C(I       6%�	z+�N���A�*;


total_loss.�@

error_R�4Z?

learning_rate_1�9���>I       6%�	�p�N���A�*;


total_lossA��@

error_R/�X?

learning_rate_1�9���I       6%�	
��N���A�*;


total_loss�e�@

error_R�VX?

learning_rate_1�9��XI       6%�	c��N���A�*;


total_loss�/�@

error_Rv�X?

learning_rate_1�9R��I       6%�	�>�N���A�*;


total_losso�c@

error_RMmX?

learning_rate_1�9��zOI       6%�	��N���A�*;


total_loss�AA

error_R&�F?

learning_rate_1�9=]VI       6%�	s��N���A�*;


total_lossMu�@

error_R�R?

learning_rate_1�9��i�I       6%�	��N���A�*;


total_loss7I�@

error_R��T?

learning_rate_1�9�94<I       6%�	�[�N���A�*;


total_loss�@

error_R�|A?

learning_rate_1�9���4I       6%�	��N���A�*;


total_loss'A

error_Ro�R?

learning_rate_1�9T��LI       6%�	���N���A�*;


total_loss 
�@

error_R�BI?

learning_rate_1�9QC�cI       6%�	�-�N���A�*;


total_lossJN�@

error_R�8T?

learning_rate_1�9��LiI       6%�	ے�N���A�*;


total_loss��@

error_R
�K?

learning_rate_1�9foI       6%�	���N���A�*;


total_loss�ˀ@

error_R�M?

learning_rate_1�9�ڂI       6%�	*/�N���A�*;


total_loss��@

error_R�tT?

learning_rate_1�9:a/SI       6%�	
{�N���A�*;


total_lossR�a@

error_R�~L?

learning_rate_1�9Rp�I       6%�	:��N���A�*;


total_loss�S�@

error_R)?R?

learning_rate_1�9�8,�I       6%�	��N���A�*;


total_loss�A

error_RCI?

learning_rate_1�9juZ�I       6%�	�Z�N���A�*;


total_loss�t�@

error_R�=Q?

learning_rate_1�9*d�I       6%�	d��N���A�*;


total_loss֐�@

error_ROQ?

learning_rate_1�9�p;I       6%�	]��N���A�*;


total_lossl�x@

error_R}�X?

learning_rate_1�95�I       6%�	�H�N���A�*;


total_loss�PA

error_R
�P?

learning_rate_1�9Y^S�I       6%�	H��N���A�*;


total_loss�r~@

error_Rj�]?

learning_rate_1�9y�I|I       6%�	'��N���A�*;


total_loss!�LA

error_RܯY?

learning_rate_1�9>+�I       6%�	��N���A�*;


total_loss��@

error_RsR?

learning_rate_1�9"�2�I       6%�	�w�N���A�*;


total_loss�p A

error_R��a?

learning_rate_1�9ѩڮI       6%�	���N���A�*;


total_loss�
�@

error_R�S?

learning_rate_1�9p�0�I       6%�	�	�N���A�*;


total_loss�f�@

error_R�K?

learning_rate_1�9m}��I       6%�	7N�N���A�*;


total_loss�@

error_R��M?

learning_rate_1�9�ZI       6%�	���N���A�*;


total_lossѦ�@

error_R��G?

learning_rate_1�9�6,I       6%�	t��N���A�*;


total_loss;��@

error_R1�M?

learning_rate_1�9_�^I       6%�	e'�N���A�*;


total_loss�G�@

error_R��1?

learning_rate_1�9
I��I       6%�	�m�N���A�*;


total_loss� �@

error_RCFQ?

learning_rate_1�9�E*I       6%�	��N���A�*;


total_loss���@

error_Rj�U?

learning_rate_1�9f�NI       6%�	� �N���A�*;


total_lossѝ�@

error_R�nV?

learning_rate_1�9���MI       6%�	�K�N���A�*;


total_lossn��@

error_RHWJ?

learning_rate_1�9��� I       6%�	���N���A�*;


total_loss4UA

error_RZ�U?

learning_rate_1�9R�]I       6%�	���N���A�*;


total_loss,��@

error_R�ha?

learning_rate_1�9nza�I       6%�	]+�N���A�*;


total_loss~�@

error_R��Y?

learning_rate_1�9����I       6%�	Uq�N���A�*;


total_loss1a�@

error_RC9Q?

learning_rate_1�9�)dI       6%�	���N���A�*;


total_loss�d�@

error_R�"N?

learning_rate_1�9�B�>I       6%�	#��N���A�*;


total_loss�@

error_R�?a?

learning_rate_1�9r��SI       6%�	�@�N���A�*;


total_loss؁�@

error_R�MH?

learning_rate_1�9hƐI       6%�	n��N���A�*;


total_lossfQ�@

error_R�#H?

learning_rate_1�9T^��I       6%�	��N���A�*;


total_lossŦ�@

error_R
Q?

learning_rate_1�9���I       6%�	��N���A�*;


total_lossVB�@

error_R@�c?

learning_rate_1�9ĭV�I       6%�	^�N���A�*;


total_loss��@

error_Rw�P?

learning_rate_1�9�kL�I       6%�	��N���A�*;


total_lossz�~@

error_RW�I?

learning_rate_1�9ٹ��I       6%�	��N���A�*;


total_loss���@

error_R,tD?

learning_rate_1�9�#�I       6%�	{0�N���A�*;


total_loss�b�@

error_R��W?

learning_rate_1�9���I       6%�	�w�N���A�*;


total_loss{H�@

error_R�H7?

learning_rate_1�9���sI       6%�	"��N���A�*;


total_lossf�A

error_R��J?

learning_rate_1�9ؑ8mI       6%�	w�N���A�*;


total_lossՈ@

error_R2cM?

learning_rate_1�9����I       6%�	2J�N���A�*;


total_lossD�@

error_R�M?

learning_rate_1�9�r�qI       6%�	��N���A�*;


total_loss&A

error_R�df?

learning_rate_1�9���I       6%�	���N���A�*;


total_lossj )A

error_RL0b?

learning_rate_1�9�-ɖI       6%�	� �N���A�*;


total_lossf�@

error_R�]?

learning_rate_1�9���uI       6%�	�e�N���A�*;


total_lossy��@

error_R��>?

learning_rate_1�9[:��I       6%�	���N���A�*;


total_loss��@

error_R��K?

learning_rate_1�9U�uI       6%�	D��N���A�*;


total_loss��
A

error_RJ�??

learning_rate_1�9Y'��I       6%�	|0�N���A�*;


total_loss&��@

error_R�Q?

learning_rate_1�9�F}WI       6%�	qr�N���A�*;


total_loss@S�@

error_RL&O?

learning_rate_1�9���I       6%�	���N���A�*;


total_loss��@

error_R3MC?

learning_rate_1�9���oI       6%�	���N���A�*;


total_lossP�A

error_R��B?

learning_rate_1�9jXI       6%�	z@�N���A�*;


total_loss� �@

error_R�
R?

learning_rate_1�9 �I       6%�	,��N���A�*;


total_lossO��@

error_R��=?

learning_rate_1�9�^u�I       6%�	B��N���A�*;


total_lossL��@

error_R
mC?

learning_rate_1�9�	�I       6%�	;�N���A�*;


total_lossТ�@

error_R�A?

learning_rate_1�9�� I       6%�	8]�N���A�*;


total_loss��@

error_R�t>?

learning_rate_1�9�m�I       6%�	V��N���A�*;


total_loss/v@

error_R	�C?

learning_rate_1�9P��I       6%�	Z�N���A�*;


total_lossEA�@

error_R iW?

learning_rate_1�9�D�SI       6%�	N\�N���A�*;


total_lossQ�@

error_RN?

learning_rate_1�9|�;�I       6%�	��N���A�*;


total_loss3k�@

error_Rv�I?

learning_rate_1�9���I       6%�	���N���A�*;


total_loss���@

error_RO�i?

learning_rate_1�9��I       6%�	�.�N���A�*;


total_loss���@

error_R�pS?

learning_rate_1�9�}��I       6%�	�w�N���A�*;


total_loss�A

error_R��>?

learning_rate_1�9_M��I       6%�	��N���A�*;


total_loss�q�@

error_R\�N?

learning_rate_1�9�x��I       6%�	��N���A�*;


total_loss��@

error_RJfQ?

learning_rate_1�97(|VI       6%�	�O�N���A�*;


total_lossn��@

error_Rv�S?

learning_rate_1�9yF�|I       6%�	���N���A�*;


total_loss���@

error_R\[R?

learning_rate_1�9A�~I       6%�	!��N���A�*;


total_loss� A

error_RE�R?

learning_rate_1�9A2PI       6%�	�)�N���A�*;


total_lossj�*A

error_R�DH?

learning_rate_1�9�l� I       6%�	�o�N���A�*;


total_losslT�@

error_RߕQ?

learning_rate_1�9��I       6%�	y��N���A�*;


total_loss�
A

error_R��>?

learning_rate_1�9��|�I       6%�	���N���A�*;


total_loss](�@

error_R��C?

learning_rate_1�9&�=�I       6%�	�A�N���A�*;


total_loss|@�@

error_R��A?

learning_rate_1�9'|`�I       6%�	,��N���A�*;


total_loss�;�@

error_R�O?

learning_rate_1�9��I       6%�	?��N���A�*;


total_loss�@�@

error_R��S?

learning_rate_1�9z��I       6%�	@�N���A�*;


total_loss�I�@

error_RM�X?

learning_rate_1�9{N��I       6%�	 X�N���A�*;


total_lossB�@

error_R��T?

learning_rate_1�9�hII       6%�	���N���A�*;


total_loss�5�@

error_R}R?

learning_rate_1�9?m��I       6%�	x��N���A�*;


total_loss�ʣ@

error_R%pT?

learning_rate_1�9ku�>I       6%�	#&�N���A�*;


total_loss
��@

error_R��Q?

learning_rate_1�9x�.�I       6%�	�j�N���A�*;


total_lossd�@

error_R=AP?

learning_rate_1�92|7I       6%�	v��N���A�*;


total_loss���@

error_R��O?

learning_rate_1�9�QI       6%�	X��N���A�*;


total_loss��@

error_R�&I?

learning_rate_1�9U�(I       6%�	@�N���A�*;


total_loss}��@

error_RO�[?

learning_rate_1�9��I       6%�	H��N���A�*;


total_loss�Æ@

error_R	�h?

learning_rate_1�9
�EI       6%�	E��N���A�*;


total_loss�@a@

error_R}�@?

learning_rate_1�9�ۋ�I       6%�	��N���A�*;


total_loss�d�@

error_RsrJ?

learning_rate_1�9:�]I       6%�	nU�N���A�*;


total_loss�*�@

error_R3(H?

learning_rate_1�9�F�zI       6%�	��N���A�*;


total_loss���@

error_RaV?

learning_rate_1�9*���I       6%�	���N���A�*;


total_loss�#�@

error_R��_?

learning_rate_1�9�Q�I       6%�	�/�N���A�*;


total_loss�ǡ@

error_R$S?

learning_rate_1�9Vd��I       6%�	�t�N���A�*;


total_loss���@

error_R��W?

learning_rate_1�9K9o�I       6%�	0��N���A�*;


total_loss���@

error_R�H?

learning_rate_1�9�h�.I       6%�	t��N���A�*;


total_loss�$�@

error_RN�R?

learning_rate_1�9���I       6%�	D�N���A�*;


total_lossR�@

error_Rv�_?

learning_rate_1�9"�iI       6%�	��N���A�*;


total_loss:4�@

error_RV??

learning_rate_1�9� ��I       6%�	y��N���A�*;


total_loss�.�@

error_R��J?

learning_rate_1�9�PI       6%�	*�N���A�*;


total_loss.I�@

error_Ri�Y?

learning_rate_1�9i�QI       6%�	�S�N���A�*;


total_loss�Tx@

error_R��J?

learning_rate_1�9�:<I       6%�	��N���A�*;


total_lossnu�@

error_R�Y?

learning_rate_1�9�baI       6%�	���N���A�*;


total_loss�Ô@

error_R�_\?

learning_rate_1�9�2I       6%�	�6�N���A�*;


total_lossV�v@

error_R��\?

learning_rate_1�9^��+I       6%�	�~�N���A�*;


total_loss���@

error_R��J?

learning_rate_1�9�E�xI       6%�	���N���A�*;


total_loss3r�@

error_R� O?

learning_rate_1�9��.&I       6%�	{	�N���A�*;


total_loss��@

error_R�%a?

learning_rate_1�99K1VI       6%�	�O�N���A�*;


total_loss1g�@

error_R��L?

learning_rate_1�9z�OI       6%�	Ö�N���A�*;


total_loss��@

error_R
\?

learning_rate_1�9��1I       6%�	��N���A�*;


total_losst��@

error_ReR2?

learning_rate_1�9�l�I       6%�	�)�N���A�*;


total_loss� A

error_R�Z?

learning_rate_1�9ƂK I       6%�	u�N���A�*;


total_loss�d�@

error_R6�N?

learning_rate_1�9�0�NI       6%�	��N���A�*;


total_loss�@

error_R� V?

learning_rate_1�9���I       6%�	w��N���A�*;


total_loss�٣@

error_R�J?

learning_rate_1�9��~�I       6%�	�?�N���A�*;


total_loss\G�@

error_R�SU?

learning_rate_1�9���I       6%�	f��N���A�*;


total_loss�AA

error_Rw�S?

learning_rate_1�9�j�I       6%�	~��N���A�*;


total_loss�p�@

error_R-
L?

learning_rate_1�94��I       6%�	�7�N���A�*;


total_loss�N�@

error_R�U?

learning_rate_1�9K)KI       6%�	f^�N���A�*;


total_losse��@

error_R�4V?

learning_rate_1�9h��I       6%�	 ��N���A�*;


total_loss�A�@

error_R�b;?

learning_rate_1�9���VI       6%�	 �N���A�*;


total_loss�i�@

error_R�H?

learning_rate_1�9W��I       6%�	yL�N���A�*;


total_lossi}�@

error_R3�K?

learning_rate_1�9��I       6%�	���N���A�*;


total_loss69�@

error_R�gQ?

learning_rate_1�9�e I       6%�	L��N���A�*;


total_loss��@

error_R �7?

learning_rate_1�9yFϙI       6%�	>�N���A�*;


total_loss��m@

error_R�'8?

learning_rate_1�9���I       6%�	Wb�N���A�*;


total_loss��@

error_R�W?

learning_rate_1�9�#!�I       6%�	Ҧ�N���A�*;


total_lossO��@

error_R��G?

learning_rate_1�9	�ǀI       6%�	���N���A�*;


total_loss|= A

error_RM`?

learning_rate_1�9��T1I       6%�	)+ O���A�*;


total_loss���@

error_R��h?

learning_rate_1�9�>�5I       6%�	on O���A�*;


total_loss� �@

error_RZ�C?

learning_rate_1�9��eI       6%�	`� O���A�*;


total_loss��@

error_R�M?

learning_rate_1�9s�vGI       6%�	�� O���A�*;


total_loss��@

error_R��C?

learning_rate_1�99�G`I       6%�	�BO���A�*;


total_lossF�@

error_R8�K?

learning_rate_1�9���I       6%�	��O���A�*;


total_loss7�@

error_R��H?

learning_rate_1�9�l`[I       6%�	h�O���A�*;


total_loss.�@

error_R��F?

learning_rate_1�9��bkI       6%�	�O���A�*;


total_lossI<�@

error_R��L?

learning_rate_1�9�bעI       6%�	�ZO���A�*;


total_loss��b@

error_R��>?

learning_rate_1�9��+I       6%�	��O���A�*;


total_loss=�@

error_R��d?

learning_rate_1�9[T�qI       6%�	�O���A�*;


total_lossr�@

error_Rl�U?

learning_rate_1�9��I       6%�	C4O���A�*;


total_loss?H�@

error_RԸ`?

learning_rate_1�9��I       6%�	C}O���A�*;


total_loss�N�@

error_RX0F?

learning_rate_1�9��� I       6%�	��O���A�*;


total_loss�:�@

error_R��K?

learning_rate_1�9F\"I       6%�	�O���A�*;


total_loss(�A

error_R�DL?

learning_rate_1�9����I       6%�	�HO���A�*;


total_lossQ]�@

error_R��e?

learning_rate_1�9x��5I       6%�	��O���A�*;


total_loss���@

error_R�yO?

learning_rate_1�9t�:�I       6%�	��O���A�*;


total_loss�h@

error_R��J?

learning_rate_1�9�N|I       6%�	�O���A�*;


total_loss8�@

error_R��A?

learning_rate_1�9��'I       6%�	�ZO���A�*;


total_loss���@

error_R�C?

learning_rate_1�9����I       6%�	�O���A�*;


total_loss�V�@

error_R��g?

learning_rate_1�9�AI       6%�	B�O���A�*;


total_losst`�@

error_R|�O?

learning_rate_1�9?��I       6%�	�5O���A�*;


total_loss�^�@

error_R!�H?

learning_rate_1�9��r�I       6%�	KyO���A�*;


total_lossOa�@

error_R�Rg?

learning_rate_1�9VW;I       6%�	�O���A�*;


total_lossE��@

error_R��J?

learning_rate_1�9���I       6%�	��O���A�*;


total_loss�w�@

error_R�:O?

learning_rate_1�9�E&�I       6%�	@O���A�*;


total_loss�@

error_R�@?

learning_rate_1�9~�vI       6%�	-�O���A�*;


total_loss`��@

error_RE&B?

learning_rate_1�9Ma�I       6%�	P�O���A�*;


total_losslO�@

error_R�yH?

learning_rate_1�9��PcI       6%�	�&O���A�*;


total_loss�e�@

error_R�P?

learning_rate_1�9�
,I       6%�	��O���A�*;


total_loss��@

error_R\�K?

learning_rate_1�9�.��I       6%�	��O���A�*;


total_loss}e�@

error_R�T?

learning_rate_1�9�E̙I       6%�		O���A�*;


total_lossEG�@

error_R7W?

learning_rate_1�9�}�I       6%�	$m	O���A�*;


total_lossz�A

error_R��o?

learning_rate_1�9���I       6%�	� 
O���A�*;


total_losst�A

error_R�Y?

learning_rate_1�9c��I       6%�	jJ
O���A�*;


total_loss�+A

error_RYN?

learning_rate_1�90^YI       6%�	U�
O���A�*;


total_loss���@

error_R�<M?

learning_rate_1�9�r�lI       6%�	�O���A�*;


total_loss��@

error_RnoC?

learning_rate_1�9����I       6%�	9PO���A�*;


total_lossʓ�@

error_R��e?

learning_rate_1�9��^I       6%�	��O���A�*;


total_lossM^�@

error_R <b?

learning_rate_1�9`@�I       6%�	O���A�*;


total_lossl�@

error_RʐA?

learning_rate_1�9�X�I       6%�	~UO���A�*;


total_lossK�@

error_R/yN?

learning_rate_1�9��9I       6%�	�O���A�*;


total_loss��@

error_R�nS?

learning_rate_1�9���I       6%�	d�O���A�*;


total_loss输@

error_R1�E?

learning_rate_1�9��D*I       6%�	:#O���A�*;


total_loss�` A

error_RS�T?

learning_rate_1�9�0�/I       6%�	}O���A�*;


total_loss_Ԛ@

error_RH�E?

learning_rate_1�9����I       6%�	��O���A�*;


total_loss�l�@

error_R?g?

learning_rate_1�9��
I       6%�	S)O���A�*;


total_loss��@

error_Rf�B?

learning_rate_1�9���/I       6%�	xO���A�*;


total_loss�u@

error_RiF?

learning_rate_1�9��NI       6%�	��O���A�*;


total_loss�@

error_R��K?

learning_rate_1�9Q�1�I       6%�	�
O���A�*;


total_loss��}@

error_RE/O?

learning_rate_1�9^	N�I       6%�	�PO���A�*;


total_losss�@

error_R)yD?

learning_rate_1�9c��I       6%�	�O���A�*;


total_loss�ƈ@

error_RO(L?

learning_rate_1�9r|�I       6%�	_�O���A�*;


total_loss���@

error_Ri�[?

learning_rate_1�9T?I       6%�	�5O���A�*;


total_lossq��@

error_R��X?

learning_rate_1�9�@kI       6%�	�O���A�*;


total_loss���@

error_R�N?

learning_rate_1�9`rHI       6%�	��O���A�*;


total_losst(�@

error_R�#G?

learning_rate_1�9Ue��I       6%�	%O���A�*;


total_loss��A

error_R��;?

learning_rate_1�9���I       6%�	5fO���A�*;


total_loss��b@

error_Rt�O?

learning_rate_1�9I���I       6%�	c�O���A�*;


total_loss=�@

error_R��]?

learning_rate_1�9�m�I       6%�	��O���A�*;


total_lossa�@

error_R%!K?

learning_rate_1�9BbI       6%�	�CO���A�*;


total_lossa��@

error_Rr�\?

learning_rate_1�9M��I       6%�	��O���A�*;


total_loss��W@

error_R�H?

learning_rate_1�9���I       6%�	�O���A�*;


total_lossn��@

error_RĕX?

learning_rate_1�9�G�JI       6%�	�O���A�*;


total_loss1��@

error_R�=Y?

learning_rate_1�9'���I       6%�	�YO���A�*;


total_loss��&A

error_R,�\?

learning_rate_1�9��}I       6%�	ͤO���A�*;


total_loss@��@

error_R�c?

learning_rate_1�9<��YI       6%�	I�O���A�*;


total_loss�/�@

error_R�]?

learning_rate_1�9���I       6%�	�2O���A�*;


total_loss`�@

error_RS?

learning_rate_1�9�EI       6%�	{O���A�*;


total_losst�@

error_RCME?

learning_rate_1�9Z�6fI       6%�	'�O���A�*;


total_loss�Q�@

error_R�oJ?

learning_rate_1�9�\�I       6%�	aO���A�*;


total_lossO��@

error_R3�O?

learning_rate_1�9�&�SI       6%�	WO���A�*;


total_loss�@

error_Rd�X?

learning_rate_1�9��9I       6%�	��O���A�*;


total_lossg��@

error_RdC?

learning_rate_1�9�d��I       6%�	�O���A�*;


total_loss堸@

error_R�J?

learning_rate_1�9�}I       6%�	�"O���A�*;


total_lossA

error_RrG?

learning_rate_1�9�aq�I       6%�	pgO���A�*;


total_loss��@

error_R�o5?

learning_rate_1�9�Ǻ�I       6%�	A�O���A�*;


total_loss:��@

error_R�G?

learning_rate_1�92	+�I       6%�	?�O���A�*;


total_loss��@

error_R��M?

learning_rate_1�9��I       6%�	�0O���A�*;


total_loss��@

error_R�rh?

learning_rate_1�9��֬I       6%�	"xO���A�*;


total_loss�;�@

error_R�jB?

learning_rate_1�9���I       6%�	��O���A�*;


total_loss%}$A

error_R�[?

learning_rate_1�94o�I       6%�	rO���A�*;


total_lossw�@

error_R��R?

learning_rate_1�9A���I       6%�	�OO���A�*;


total_lossm5�@

error_R�6?

learning_rate_1�9��D�I       6%�	ٝO���A�*;


total_lossf��@

error_R�RR?

learning_rate_1�9�f�I       6%�	�O���A�*;


total_loss&�@

error_R@�C?

learning_rate_1�9�WI       6%�	�/O���A�*;


total_loss�:�@

error_R8/M?

learning_rate_1�9�k��I       6%�	(�O���A�*;


total_loss��3A

error_RϦM?

learning_rate_1�99,�I       6%�	��O���A�*;


total_loss)ƪ@

error_R�o\?

learning_rate_1�9�~f�I       6%�	�O���A�*;


total_lossI��@

error_R��E?

learning_rate_1�9�?/�I       6%�	�iO���A�*;


total_losslʾ@

error_RBL?

learning_rate_1�9s_NI       6%�	R�O���A�*;


total_loss�=�@

error_R�s=?

learning_rate_1�9�J��I       6%�	��O���A�*;


total_loss���@

error_R��:?

learning_rate_1�9m�I       6%�	�AO���A�*;


total_loss=�A

error_R�:\?

learning_rate_1�9.�R1I       6%�	�O���A�*;


total_loss&o�@

error_R�\?

learning_rate_1�9K��	I       6%�	��O���A�*;


total_loss��p@

error_RIC?

learning_rate_1�9LL/�I       6%�	\O���A�*;


total_loss��A

error_Rl�@?

learning_rate_1�9JW�NI       6%�	\VO���A�*;


total_lossO��@

error_R��O?

learning_rate_1�9I       6%�	+�O���A�*;


total_loss(��@

error_R;�>?

learning_rate_1�9R�I       6%�	�O���A�*;


total_loss���@

error_RE�J?

learning_rate_1�9%�=�I       6%�	�)O���A�*;


total_loss]��@

error_RؘH?

learning_rate_1�9�w��I       6%�	sO���A�*;


total_loss���@

error_R��^?

learning_rate_1�9��eAI       6%�	M�O���A�*;


total_loss�C�@

error_Rd�@?

learning_rate_1�9��JI       6%�	�O���A�*;


total_loss�4!A

error_Re�=?

learning_rate_1�9�$��I       6%�	�BO���A�*;


total_loss��@

error_R��K?

learning_rate_1�9�F�$I       6%�	߈O���A�*;


total_lossX��@

error_R:�I?

learning_rate_1�9���xI       6%�	�O���A�*;


total_loss���@

error_R_�B?

learning_rate_1�9d�kI       6%�	�O���A�*;


total_lossd.s@

error_R��C?

learning_rate_1�9N�I       6%�	�dO���A�*;


total_loss���@

error_R��]?

learning_rate_1�9��b�I       6%�	"�O���A�*;


total_loss�w@

error_RvF?

learning_rate_1�9ed�I       6%�	��O���A�*;


total_loss�A

error_R�QY?

learning_rate_1�9��;{I       6%�	_1 O���A�*;


total_loss�!A

error_R��N?

learning_rate_1�9us��I       6%�	�~ O���A�*;


total_loss�T�@

error_RS[P?

learning_rate_1�9����I       6%�	�� O���A�*;


total_loss䠻@

error_R=J\?

learning_rate_1�9�s�I       6%�	�!O���A�*;


total_loss��@

error_RNAD?

learning_rate_1�9���RI       6%�	�N!O���A�*;


total_loss�_�@

error_RI�A?

learning_rate_1�9yªI       6%�		�!O���A�*;


total_loss���@

error_R;�X?

learning_rate_1�9K?�DI       6%�	�!O���A�*;


total_loss �@

error_R�Y?

learning_rate_1�9G+�I       6%�	N#"O���A�*;


total_loss��@

error_R�}P?

learning_rate_1�9�9eI       6%�	�i"O���A�*;


total_loss���@

error_Rj�W?

learning_rate_1�9?�EI       6%�	��"O���A�*;


total_loss<$�@

error_R�E?

learning_rate_1�9zZ2HI       6%�	e�"O���A�*;


total_loss)N�@

error_R��G?

learning_rate_1�9�VݕI       6%�	Z4#O���A�*;


total_lossL��@

error_R�F?

learning_rate_1�9��
FI       6%�	�u#O���A�*;


total_loss�ڷ@

error_R%Lc?

learning_rate_1�9�(�UI       6%�	#�#O���A�*;


total_loss$�@

error_RtM?

learning_rate_1�9�+;�I       6%�	� $O���A�*;


total_loss@

error_R�K?

learning_rate_1�9Y�I       6%�	+D$O���A�*;


total_lossaW�@

error_Rq�V?

learning_rate_1�9!��I       6%�	0�$O���A�*;


total_loss���@

error_R�3K?

learning_rate_1�9
b�I       6%�	v�$O���A�*;


total_loss/�@

error_R�_P?

learning_rate_1�9[7�I       6%�	[%O���A�*;


total_loss�x�@

error_Rq�M?

learning_rate_1�9�O�I       6%�	4Q%O���A�*;


total_lossc��@

error_RlH?

learning_rate_1�9��{I       6%�	_�%O���A�*;


total_loss
�A

error_R��d?

learning_rate_1�9���I       6%�	3�%O���A�*;


total_loss�C�@

error_R�wO?

learning_rate_1�9�f8dI       6%�	&O���A�*;


total_loss�G�@

error_Rnb?

learning_rate_1�9&A0�I       6%�	�_&O���A�*;


total_loss,�@

error_RM�B?

learning_rate_1�9�N�:I       6%�	��&O���A�*;


total_lossd�@

error_R�IE?

learning_rate_1�9(�&,I       6%�	��&O���A�*;


total_lossI�@

error_RR�E?

learning_rate_1�9��'I       6%�	�.'O���A�*;


total_lossʯ�@

error_R�\?

learning_rate_1�9?T�mI       6%�	r'O���A�*;


total_loss���@

error_R؊M?

learning_rate_1�9����I       6%�	��'O���A�*;


total_lossc��@

error_R��2?

learning_rate_1�9f�)I       6%�	��'O���A�*;


total_loss_��@

error_R86J?

learning_rate_1�9��HI       6%�	�F(O���A�*;


total_loss��|@

error_R�SV?

learning_rate_1�9�S��I       6%�	[�(O���A�*;


total_loss��@

error_Rl�G?

learning_rate_1�9^ BI       6%�	��(O���A�*;


total_loss�u�@

error_R4�R?

learning_rate_1�98�z,I       6%�	q)O���A�*;


total_loss�A

error_Rc�V?

learning_rate_1�9a� �I       6%�	Rq)O���A�*;


total_loss���@

error_R�Z?

learning_rate_1�9K-ڞI       6%�	��)O���A�*;


total_loss��@

error_R��>?

learning_rate_1�9 =�I       6%�	�*O���A�*;


total_loss��@

error_R��[?

learning_rate_1�9�2I       6%�	BV*O���A�*;


total_loss砊@

error_R \E?

learning_rate_1�9o�I       6%�	�*O���A�*;


total_loss�x�@

error_R_Z?

learning_rate_1�9_F�HI       6%�	b�*O���A�*;


total_lossEI�@

error_RMGW?

learning_rate_1�9�/��I       6%�	G++O���A�*;


total_lossq=�@

error_R;�W?

learning_rate_1�9��.�I       6%�	+O���A�*;


total_loss�ї@

error_R3eL?

learning_rate_1�9R�5�I       6%�	��+O���A�*;


total_loss�ú@

error_RYY?

learning_rate_1�9�H�jI       6%�	,O���A�*;


total_loss�ׯ@

error_R_\b?

learning_rate_1�9� �I       6%�	),O���A�*;


total_loss���@

error_R��j?

learning_rate_1�9���I       6%�	,�,O���A�*;


total_loss�R�@

error_R,�??

learning_rate_1�9�*�_I       6%�	�-O���A�*;


total_loss]��@

error_R�:R?

learning_rate_1�9>��I       6%�	�W-O���A�*;


total_loss��z@

error_R�QZ?

learning_rate_1�9Aa�I       6%�	��-O���A�*;


total_loss�ؘ@

error_RM�J?

learning_rate_1�9�b��I       6%�	s�-O���A�*;


total_loss��A

error_RFZ?

learning_rate_1�9�X0�I       6%�	G%.O���A�*;


total_loss#��@

error_R=�G?

learning_rate_1�9���I       6%�	n.O���A�*;


total_loss�HA

error_RVY?

learning_rate_1�9Wa6I       6%�	z�.O���A�*;


total_loss�ט@

error_R.�_?

learning_rate_1�9�x��I       6%�	��.O���A�*;


total_loss���@

error_R��c?

learning_rate_1�9�hI       6%�	&:/O���A�*;


total_loss2�@

error_R�SV?

learning_rate_1�9xX��I       6%�	�/O���A�*;


total_lossWۯ@

error_R�\?

learning_rate_1�9w�6I       6%�	,�/O���A�*;


total_loss���@

error_R�X_?

learning_rate_1�9�>[�I       6%�	9	0O���A�*;


total_loss�3�@

error_R��Y?

learning_rate_1�9�/�YI       6%�	ON0O���A�*;


total_loss�g�@

error_R��N?

learning_rate_1�9�=\I       6%�	/�0O���A�*;


total_loss��@

error_R�dM?

learning_rate_1�9�RA	I       6%�	{�0O���A�*;


total_loss׏A

error_R��M?

learning_rate_1�9��g3I       6%�	�1O���A�*;


total_lossv�A

error_R��??

learning_rate_1�94���I       6%�	|Z1O���A�*;


total_lossf��@

error_R1[j?

learning_rate_1�9��+I       6%�	�1O���A�*;


total_loss���@

error_R,<[?

learning_rate_1�9�5��I       6%�	��1O���A�*;


total_loss1�	A

error_R�oB?

learning_rate_1�9�qX�I       6%�	J'2O���A�*;


total_lossrt�@

error_R��R?

learning_rate_1�9��-I       6%�	}j2O���A�*;


total_lossd�@

error_RTHQ?

learning_rate_1�9R:JI       6%�	q�2O���A�*;


total_loss���@

error_Rne?

learning_rate_1�9�GI       6%�	X�2O���A�*;


total_loss���@

error_R#T?

learning_rate_1�9��I       6%�	�33O���A�*;


total_loss��@

error_R�S?

learning_rate_1�9�lu_I       6%�	�y3O���A�*;


total_loss�!�@

error_R T?

learning_rate_1�9n<jI       6%�	u�3O���A�*;


total_loss�b�@

error_R��b?

learning_rate_1�9:�!YI       6%�	�
4O���A�*;


total_loss\�@

error_R��P?

learning_rate_1�9�q�wI       6%�	�L4O���A�*;


total_loss�:}@

error_R2,H?

learning_rate_1�9�7K�I       6%�	��4O���A�*;


total_loss�A

error_R��d?

learning_rate_1�9��"I       6%�	j�4O���A�*;


total_loss���@

error_R��R?

learning_rate_1�9x�
�I       6%�	�5O���A�*;


total_loss]�@

error_R�?P?

learning_rate_1�9;�6I       6%�	r]5O���A�*;


total_loss�m�@

error_R��Q?

learning_rate_1�9����I       6%�	�5O���A�*;


total_loss�K
A

error_R�~M?

learning_rate_1�9��I       6%�	�5O���A�*;


total_loss�{%A

error_R��`?

learning_rate_1�9m�D�I       6%�	�76O���A�*;


total_loss��@

error_R��5?

learning_rate_1�9ml�I       6%�	�|6O���A�*;


total_loss�&�@

error_R*mF?

learning_rate_1�9��+I       6%�	��6O���A�*;


total_loss骧@

error_RI�T?

learning_rate_1�9�x��I       6%�	7O���A�*;


total_loss�@

error_R��K?

learning_rate_1�9I�I       6%�	JM7O���A�*;


total_loss�1A

error_RO?

learning_rate_1�9���
I       6%�	o�7O���A�*;


total_loss�Б@

error_R��T?

learning_rate_1�9�9�I       6%�	��7O���A�*;


total_lossl��@

error_R�S?

learning_rate_1�9x��I       6%�	�8O���A�*;


total_lossH7�@

error_RJ�M?

learning_rate_1�9�l,�I       6%�	�[8O���A�*;


total_loss�e�@

error_R)�m?

learning_rate_1�97Ԧ�I       6%�	s�8O���A�*;


total_loss�-�@

error_R��S?

learning_rate_1�9�I       6%�	��8O���A�*;


total_loss�.�@

error_R*�o?

learning_rate_1�9AӴWI       6%�	D'9O���A�*;


total_loss)��@

error_R�rI?

learning_rate_1�9�pI       6%�	J}9O���A�*;


total_loss��@

error_Rəd?

learning_rate_1�9�U�NI       6%�	w�9O���A�*;


total_loss�,�@

error_R�]?

learning_rate_1�9�I��I       6%�	�:O���A�*;


total_lossD�%A

error_R�;J?

learning_rate_1�9 Jh�I       6%�	�`:O���A�*;


total_lossŎ�@

error_R��>?

learning_rate_1�9(��I       6%�	��:O���A�*;


total_loss�u�@

error_Rʈ^?

learning_rate_1�9؟��I       6%�	��:O���A�*;


total_loss�K A

error_R\�C?

learning_rate_1�9ד�I       6%�	�6;O���A�*;


total_loss���@

error_R_^?

learning_rate_1�9�|بI       6%�	�{;O���A�*;


total_lossDi�@

error_R:�F?

learning_rate_1�9���I       6%�	Z�;O���A�*;


total_loss�CA

error_R��L?

learning_rate_1�9�QY�I       6%�	�<O���A�*;


total_loss�7A

error_R�5P?

learning_rate_1�9���NI       6%�	�H<O���A�*;


total_lossx�@

error_R8�C?

learning_rate_1�9�њ�I       6%�	�<O���A�*;


total_loss�e�@

error_R/;?

learning_rate_1�9ړ��I       6%�	��<O���A�*;


total_loss��@

error_Rq�Z?

learning_rate_1�9�5%�I       6%�	.=O���A�*;


total_loss�Y�@

error_R6�2?

learning_rate_1�9,W�PI       6%�	o\=O���A�*;


total_loss��@

error_R��A?

learning_rate_1�94�u6I       6%�	��=O���A�*;


total_loss���@

error_R�O?

learning_rate_1�90��=I       6%�	}�=O���A�*;


total_loss9��@

error_RR�W?

learning_rate_1�9H$��I       6%�	u*>O���A�*;


total_loss�>�@

error_Rh6Y?

learning_rate_1�9���II       6%�	�m>O���A�*;


total_loss�k�@

error_R��`?

learning_rate_1�9A��I       6%�	��>O���A�*;


total_lossئ@

error_RjdB?

learning_rate_1�9Y��MI       6%�	��>O���A�*;


total_lossae�@

error_R��G?

learning_rate_1�9�ytI       6%�	�9?O���A�*;


total_lossϓA

error_RW�4?

learning_rate_1�9}��I       6%�	�|?O���A�*;


total_lossd�@

error_R��K?

learning_rate_1�91�I       6%�	�?O���A�*;


total_loss�R�@

error_R�wL?

learning_rate_1�9�@�?I       6%�	@O���A�*;


total_loss�F�@

error_R�YZ?

learning_rate_1�9:��I       6%�	%D@O���A�*;


total_lossC=t@

error_R�L?

learning_rate_1�9H;�WI       6%�	��@O���A�*;


total_lossX�@

error_Rh|d?

learning_rate_1�9��SI       6%�	��@O���A�*;


total_lossOB�@

error_R�U?

learning_rate_1�9r�%I       6%�	�AO���A�*;


total_loss���@

error_R`�U?

learning_rate_1�9
�uI       6%�	�`AO���A�*;


total_loss��@

error_R�\?

learning_rate_1�9B	_�I       6%�	��AO���A�*;


total_lossr�@

error_R��X?

learning_rate_1�9I��I       6%�	&�AO���A�*;


total_loss�x�@

error_R�W?

learning_rate_1�9��$�I       6%�	 'BO���A�*;


total_loss]�A

error_R&�\?

learning_rate_1�9�?eI       6%�	�iBO���A�*;


total_loss��@

error_R��J?

learning_rate_1�9��+�I       6%�	��BO���A�*;


total_loss��@

error_R�O?

learning_rate_1�9h��I       6%�	��BO���A�*;


total_loss���@

error_RZ�I?

learning_rate_1�9���I       6%�	X=CO���A�*;


total_loss��@

error_RتH?

learning_rate_1�9ܭ�I       6%�	��CO���A�*;


total_loss�>�@

error_RL|M?

learning_rate_1�9o��I       6%�	�CO���A�*;


total_loss��@

error_R�P?

learning_rate_1�9�!xI       6%�	�DO���A�*;


total_loss*Ts@

error_R��M?

learning_rate_1�9�7PI       6%�	�gDO���A�*;


total_lossi��@

error_R�-J?

learning_rate_1�9�H0I       6%�	��DO���A�*;


total_loss�J�@

error_Rw�N?

learning_rate_1�9[$dI       6%�	��DO���A�*;


total_loss6`�@

error_R8�S?

learning_rate_1�9�gWI       6%�	�=EO���A�*;


total_loss�_�@

error_R@tX?

learning_rate_1�9u�{�I       6%�	��EO���A�*;


total_loss�	�@

error_R	�M?

learning_rate_1�9x�
�I       6%�	f�EO���A�*;


total_loss��@

error_RiX?

learning_rate_1�9���RI       6%�	 FO���A�*;


total_lossF�@

error_R_*U?

learning_rate_1�9$��I       6%�	)fFO���A�*;


total_loss�(A

error_R�i?

learning_rate_1�9.`�I       6%�	ҭFO���A�*;


total_loss\M�@

error_R�/O?

learning_rate_1�9��wI       6%�	�FO���A�*;


total_loss���@

error_R�V\?

learning_rate_1�9�M�I       6%�	�=GO���A�*;


total_lossVo�@

error_R��O?

learning_rate_1�9yI̊I       6%�	��GO���A�*;


total_loss[b�@

error_R]xe?

learning_rate_1�9.�ǪI       6%�	{�GO���A�*;


total_loss���@

error_RScF?

learning_rate_1�9b�'�I       6%�	HO���A�*;


total_loss�~o@

error_R��V?

learning_rate_1�9�j�
I       6%�	�UHO���A�*;


total_loss�{�@

error_R\�R?

learning_rate_1�9,&ÜI       6%�	+�HO���A�*;


total_lossZ7�@

error_R��P?

learning_rate_1�9��?I       6%�	x�HO���A�*;


total_loss���@

error_R
�Z?

learning_rate_1�9�н�I       6%�	?+IO���A�*;


total_loss�|'A

error_Rm^A?

learning_rate_1�94��I       6%�	��IO���A�*;


total_losssq�@

error_R&>?

learning_rate_1�9�s�fI       6%�	��IO���A�*;


total_loss^��@

error_R�{R?

learning_rate_1�9�cAI       6%�	�$JO���A�*;


total_loss���@

error_RaT?

learning_rate_1�9���I       6%�	�nJO���A�*;


total_loss�+�@

error_R��X?

learning_rate_1�9��D�I       6%�	�JO���A�*;


total_loss�`�@

error_R��H?

learning_rate_1�9���I       6%�	KO���A�*;


total_loss1�@

error_R1�R?

learning_rate_1�9y5��I       6%�	�eKO���A�*;


total_lossv��@

error_R��_?

learning_rate_1�9�;g-I       6%�	�KO���A�*;


total_loss;�@

error_R2LY?

learning_rate_1�9��hI       6%�	�KO���A�*;


total_loss���@

error_RW�Y?

learning_rate_1�9�l�hI       6%�	�ALO���A�*;


total_loss~��@

error_RLM?

learning_rate_1�9F��I       6%�	��LO���A�*;


total_loss)��@

error_R
�J?

learning_rate_1�9�گI       6%�	��LO���A�*;


total_loss�AA

error_RC�D?

learning_rate_1�9�j�I       6%�	�%MO���A�*;


total_loss3~�@

error_R�<?

learning_rate_1�9�?�I       6%�	nlMO���A�*;


total_loss]�@

error_R��M?

learning_rate_1�9�g�iI       6%�	�MO���A�*;


total_loss���@

error_R%dZ?

learning_rate_1�9���pI       6%�	��MO���A�*;


total_loss�އ@

error_RH#B?

learning_rate_1�9҄�@I       6%�	Q?NO���A�*;


total_lossu�A

error_R�	D?

learning_rate_1�9�v��I       6%�	u�NO���A�*;


total_loss��;A

error_R�IQ?

learning_rate_1�9�ΏI       6%�	��NO���A�*;


total_loss!fA

error_RPG?

learning_rate_1�9Y'��I       6%�	VOO���A�*;


total_loss���@

error_Rl�Z?

learning_rate_1�9E��I       6%�	LTOO���A�*;


total_lossŘ�@

error_R��K?

learning_rate_1�9��CXI       6%�	|�OO���A�*;


total_lossĐ�@

error_RC�U?

learning_rate_1�9��ڒI       6%�	�OO���A�*;


total_loss��@

error_R��N?

learning_rate_1�9=�kI       6%�	�$PO���A�*;


total_loss�@

error_RQ�\?

learning_rate_1�9M�0 I       6%�	�hPO���A�*;


total_loss���@

error_R�r=?

learning_rate_1�9�]�I       6%�	��PO���A�*;


total_loss��A

error_RȉP?

learning_rate_1�9f��I       6%�	�PO���A�*;


total_loss�?�@

error_Rf�8?

learning_rate_1�9�OoI       6%�	U<QO���A�*;


total_loss�r�@

error_R�cX?

learning_rate_1�9eӝI       6%�	��QO���A�*;


total_lossa��@

error_R�G?

learning_rate_1�9:
��I       6%�	��QO���A�*;


total_loss5�@

error_R�Q?

learning_rate_1�9]�I       6%�	�RO���A�*;


total_loss�D�@

error_R��S?

learning_rate_1�9���oI       6%�	�SRO���A�*;


total_lossM�@

error_R=�;?

learning_rate_1�9���I       6%�	|�RO���A�*;


total_lossO��@

error_R�d?

learning_rate_1�9��XI       6%�	��RO���A�*;


total_loss��@

error_R_�d?

learning_rate_1�9�R��I       6%�	3SO���A�*;


total_loss"�@

error_R�K?

learning_rate_1�9�&a�I       6%�	wSO���A�*;


total_loss�p�@

error_R@,X?

learning_rate_1�9�_��I       6%�	#�SO���A�*;


total_loss��A

error_R!�Q?

learning_rate_1�9�Hg`I       6%�	��SO���A�*;


total_lossSV�@

error_RHY?

learning_rate_1�9,�]I       6%�	BTO���A�*;


total_loss���@

error_R\\?

learning_rate_1�9hU^�I       6%�	�TO���A�*;


total_loss��@

error_RÎX?

learning_rate_1�9�@�]I       6%�	��TO���A�*;


total_lossh9�@

error_R�pZ?

learning_rate_1�9~o׽I       6%�	zUO���A�*;


total_loss,Ҭ@

error_RiV?

learning_rate_1�9���I       6%�	�RUO���A�*;


total_loss���@

error_R$L?

learning_rate_1�9�U�I       6%�	Y�UO���A�*;


total_loss��@

error_RԇU?

learning_rate_1�9�b��I       6%�	�UO���A�*;


total_loss���@

error_R�\]?

learning_rate_1�9�{��I       6%�	VO���A�*;


total_loss���@

error_RΓ7?

learning_rate_1�9�k�5I       6%�	�cVO���A�*;


total_loss���@

error_R\`]?

learning_rate_1�9s}�I       6%�	w�VO���A�*;


total_loss���@

error_R��g?

learning_rate_1�9GjțI       6%�	��VO���A�*;


total_loss���@

error_R)P?

learning_rate_1�9*i�oI       6%�	7WO���A�*;


total_lossʿ�@

error_R
�??

learning_rate_1�9�x��I       6%�	�yWO���A�*;


total_lossi��@

error_R#O??

learning_rate_1�9Z`��I       6%�	��WO���A�*;


total_loss�H1A

error_R�"K?

learning_rate_1�9�;�I       6%�	��WO���A�*;


total_loss���@

error_R1�T?

learning_rate_1�9%���I       6%�	�DXO���A�*;


total_loss�.A

error_R�'N?

learning_rate_1�9JT0I       6%�	�XO���A�*;


total_loss�޹@

error_R��a?

learning_rate_1�9�_��I       6%�	~�XO���A�*;


total_loss���@

error_RW�H?

learning_rate_1�9{���I       6%�	GYO���A�*;


total_loss�u�@

error_R�P?

learning_rate_1�9Z�skI       6%�	pYO���A�*;


total_loss3s�@

error_R�SA?

learning_rate_1�9��E�I       6%�	v�YO���A�*;


total_loss��@

error_R@�S?

learning_rate_1�9=� I       6%�	�ZO���A�*;


total_loss.6�@

error_R�
]?

learning_rate_1�9?	s�I       6%�	�]ZO���A�*;


total_loss���@

error_R��M?

learning_rate_1�9n$�I       6%�	3�ZO���A�*;


total_loss��@

error_R?\?

learning_rate_1�9FzI       6%�	�ZO���A�*;


total_loss���@

error_R��=?

learning_rate_1�9����I       6%�	n0[O���A�*;


total_lossb�@

error_Re�Q?

learning_rate_1�9ź�aI       6%�	�s[O���A�*;


total_loss]��@

error_Ro�L?

learning_rate_1�9ǎRyI       6%�	��[O���A�*;


total_loss]dx@

error_R�]?

learning_rate_1�9���I       6%�	�=\O���A�*;


total_lossf��@

error_RAUF?

learning_rate_1�9���@I       6%�	P�\O���A�*;


total_loss�=h@

error_R`J?

learning_rate_1�9<L�I       6%�	�\O���A�*;


total_loss�H�@

error_R�UF?

learning_rate_1�97��KI       6%�	b#]O���A�*;


total_loss��@

error_R}�E?

learning_rate_1�9��_I       6%�	�m]O���A�*;


total_loss���@

error_R�0c?

learning_rate_1�9�6�I       6%�	Y�]O���A�*;


total_lossSo�@

error_R�5T?

learning_rate_1�9ꓲ�I       6%�	��]O���A�*;


total_loss���@

error_RkD?

learning_rate_1�9㦽DI       6%�	}@^O���A�*;


total_loss�-A

error_R׭Z?

learning_rate_1�9���I       6%�	0�^O���A�*;


total_loss�>�@

error_R�W?

learning_rate_1�9݌�I       6%�	��^O���A�*;


total_loss���@

error_R1bF?

learning_rate_1�9c���I       6%�	�_O���A�*;


total_loss�@

error_RZ�=?

learning_rate_1�9�gQ*I       6%�	�^_O���A�*;


total_loss�v�@

error_R�@?

learning_rate_1�9Տ7I       6%�	4�_O���A�*;


total_loss���@

error_RʠN?

learning_rate_1�9����I       6%�	��_O���A�*;


total_loss�A

error_R�sA?

learning_rate_1�9�9�I       6%�	�'`O���A�*;


total_lossL׎@

error_R[�i?

learning_rate_1�9�9�I       6%�	�k`O���A�*;


total_loss���@

error_R6<P?

learning_rate_1�9XS��I       6%�	x�`O���A�*;


total_loss�O�@

error_R�<J?

learning_rate_1�9<II       6%�	
�`O���A�*;


total_lossò@

error_R��\?

learning_rate_1�9UP�I       6%�	�3aO���A�*;


total_loss���@

error_R�$Y?

learning_rate_1�9"\�I       6%�	D|aO���A�*;


total_loss-#�@

error_Ri�Y?

learning_rate_1�9��3I       6%�	��aO���A�*;


total_loss`�@

error_R(#Y?

learning_rate_1�9��I       6%�	1	bO���A�*;


total_loss|�@

error_RcI?

learning_rate_1�9|��I       6%�	�RbO���A�*;


total_lossF��@

error_RcT?

learning_rate_1�9F1��I       6%�	םbO���A�*;


total_loss�	�@

error_R�8U?

learning_rate_1�9�HEI       6%�	��bO���A�*;


total_losse�@

error_R�V?

learning_rate_1�9�nnI       6%�	v/cO���A�*;


total_loss�h�@

error_Rs�i?

learning_rate_1�9��G3I       6%�	�zcO���A�*;


total_loss��@

error_Ri|g?

learning_rate_1�9���I       6%�	V�cO���A�*;


total_loss��@

error_R�wT?

learning_rate_1�9���I       6%�	7dO���A�*;


total_loss7q�@

error_R�Q?

learning_rate_1�9��I�I       6%�	VdO���A�*;


total_loss\v�@

error_R=7J?

learning_rate_1�9XT��I       6%�	��dO���A�*;


total_loss�%�@

error_R �>?

learning_rate_1�9ے��I       6%�	�dO���A�*;


total_lossʷ�@

error_RZ{Z?

learning_rate_1�9 ��CI       6%�	&'eO���A�*;


total_loss!�@

error_R�R?

learning_rate_1�9% I       6%�	�neO���A�*;


total_loss���@

error_RM8?

learning_rate_1�9�<�I       6%�	 �eO���A�*;


total_loss@�@

error_Rd�^?

learning_rate_1�9���I       6%�	��eO���A�*;


total_loss�G�@

error_R�YX?

learning_rate_1�9u�U�I       6%�	�HfO���A�*;


total_loss��@

error_R{�^?

learning_rate_1�9��uI       6%�	C�fO���A�*;


total_loss�ؿ@

error_R�g?

learning_rate_1�9s&g=I       6%�	��fO���A�*;


total_loss�'�@

error_R_ua?

learning_rate_1�9+7��I       6%�	w#gO���A�*;


total_lossO�@

error_Rx�Z?

learning_rate_1�9$�<�I       6%�	;egO���A�*;


total_lossd�@

error_R��U?

learning_rate_1�9T��I       6%�	��gO���A�*;


total_loss,��@

error_RD?

learning_rate_1�9ݹ~I       6%�	��gO���A�*;


total_loss�V�@

error_R]Z\?

learning_rate_1�9e_��I       6%�	�5hO���A�*;


total_loss��(A

error_R�sQ?

learning_rate_1�9�|�I       6%�	�hO���A�*;


total_lossT3�@

error_Rq�Q?

learning_rate_1�9�"�I       6%�	�hO���A�*;


total_loss�g�@

error_R��E?

learning_rate_1�9�q�I       6%�	x	iO���A�*;


total_lossD��@

error_R�mS?

learning_rate_1�9]y}�I       6%�	�NiO���A�*;


total_loss�n�@

error_R�R?

learning_rate_1�9�0*�I       6%�	۲iO���A�*;


total_loss��@

error_RSA?

learning_rate_1�9ӓl�I       6%�	��iO���A�*;


total_loss���@

error_R��V?

learning_rate_1�9�?�I       6%�	&DjO���A�*;


total_loss��A

error_Rܴ]?

learning_rate_1�9@$LI       6%�	�jO���A�*;


total_lossJ��@

error_R��=?

learning_rate_1�9lH��I       6%�	��jO���A�*;


total_loss#e�@

error_R�W?

learning_rate_1�9�4�I       6%�	zkO���A�*;


total_lossh�A

error_ROC?

learning_rate_1�9�"wI       6%�	X}kO���A�*;


total_loss�O�@

error_R	�G?

learning_rate_1�97VI       6%�	��kO���A�*;


total_loss�X A

error_R8W?

learning_rate_1�9���I       6%�	lO���A�*;


total_loss�`�@

error_Rq�N?

learning_rate_1�9�j�EI       6%�	�YlO���A�*;


total_loss8R�@

error_RL�I?

learning_rate_1�9�⁽I       6%�	�lO���A�*;


total_loss�@

error_R��M?

learning_rate_1�9!��I       6%�	��lO���A�*;


total_lossw��@

error_R�'`?

learning_rate_1�9i���I       6%�	oAmO���A�*;


total_loss\��@

error_Rs�X?

learning_rate_1�9	D�I       6%�	��mO���A�*;


total_loss�˓@

error_R�sI?

learning_rate_1�9��-�I       6%�	�mO���A�*;


total_loss�H�@

error_R��??

learning_rate_1�9�`��I       6%�	`nO���A�*;


total_loss���@

error_RvgJ?

learning_rate_1�9[�;I       6%�	�ZnO���A�*;


total_lossq��@

error_R��U?

learning_rate_1�9�)�1I       6%�	_�nO���A�*;


total_loss�@

error_R #A?

learning_rate_1�9��KI       6%�	�nO���A�*;


total_loss.Ⱥ@

error_R,MD?

learning_rate_1�9�q�}I       6%�	�8oO���A�*;


total_loss/��@

error_R�WF?

learning_rate_1�9,�w�I       6%�	<�oO���A�*;


total_loss#�1A

error_R%hD?

learning_rate_1�9�t)�I       6%�	�oO���A�*;


total_loss��@

error_R��[?

learning_rate_1�9/�-I       6%�	pO���A�*;


total_loss|x�@

error_R�I`?

learning_rate_1�9���vI       6%�	�NpO���A�*;


total_loss��@

error_R�@?

learning_rate_1�9~zdI       6%�	��pO���A�*;


total_loss��@

error_R��_?

learning_rate_1�9�z��I       6%�	v�pO���A�*;


total_losst'N@

error_R�oA?

learning_rate_1�9Or�}I       6%�	gqO���A�*;


total_loss���@

error_RamJ?

learning_rate_1�9H�M I       6%�	�_qO���A�*;


total_loss?w�@

error_RsOI?

learning_rate_1�9��%�I       6%�	 �qO���A�*;


total_loss?P�@

error_RL�L?

learning_rate_1�9:ǭ|I       6%�	��qO���A�*;


total_loss�ƕ@

error_R�K?

learning_rate_1�9G I       6%�	j0rO���A�*;


total_lossf��@

error_R��H?

learning_rate_1�9��FI       6%�	JvrO���A�*;


total_loss6��@

error_R�j?

learning_rate_1�9S���I       6%�	?�rO���A�*;


total_lossaL�@

error_R�V?

learning_rate_1�9��85I       6%�	&�rO���A�*;


total_loss#��@

error_R�=Q?

learning_rate_1�9)���I       6%�	>sO���A�*;


total_loss< A

error_R�X?

learning_rate_1�9 ���I       6%�	M�sO���A�*;


total_lossOU�@

error_R��B?

learning_rate_1�9�}D&I       6%�	��sO���A�*;


total_loss���@

error_Rw�F?

learning_rate_1�9&	��I       6%�	�tO���A�*;


total_loss�س@

error_R�fK?

learning_rate_1�9r���I       6%�	zUtO���A�*;


total_loss��@

error_R�Y?

learning_rate_1�9W�}I       6%�	B�tO���A�*;


total_loss��@

error_RlHY?

learning_rate_1�9�U�.I       6%�	��tO���A�*;


total_lossd�@

error_R��Z?

learning_rate_1�9��X�I       6%�	� uO���A�*;


total_loss8�(A

error_R[]Z?

learning_rate_1�9�]�I       6%�	�fuO���A�*;


total_lossC�@

error_R��L?

learning_rate_1�9���I       6%�	թuO���A�*;


total_lossjz@

error_RV�h?

learning_rate_1�9���=I       6%�	��uO���A�*;


total_lossP��@

error_RMiT?

learning_rate_1�9CD��I       6%�	\/vO���A�*;


total_loss)�@

error_R8�P?

learning_rate_1�9�&I       6%�	�qvO���A�*;


total_loss\A

error_Rm�R?

learning_rate_1�9���yI       6%�	-�vO���A�*;


total_loss�R�@

error_R��N?

learning_rate_1�9E]�I       6%�	��vO���A�*;


total_lossz��@

error_Rv�I?

learning_rate_1�9''/�I       6%�	�?wO���A�*;


total_loss	:�@

error_R��V?

learning_rate_1�9>�	I       6%�	�wO���A�*;


total_loss�ʗ@

error_R��B?

learning_rate_1�9�>6I       6%�	��wO���A�*;


total_loss��(A

error_Rii]?

learning_rate_1�9��r�I       6%�	9xO���A�*;


total_loss�A�@

error_R��U?

learning_rate_1�9���aI       6%�	�`xO���A�*;


total_loss��t@

error_R��S?

learning_rate_1�9��I       6%�	��xO���A�*;


total_loss,չ@

error_R�N?

learning_rate_1�98��I       6%�	��xO���A�*;


total_loss�@

error_R��Y?

learning_rate_1�9�`hI       6%�	7yO���A�*;


total_loss]A

error_RJ�L?

learning_rate_1�9��3!I       6%�	��yO���A�*;


total_loss��@

error_R��K?

learning_rate_1�9ؘ!I       6%�	��yO���A�*;


total_loss�@

error_R�MV?

learning_rate_1�9=|��I       6%�	/zO���A�*;


total_loss��A

error_RJ�O?

learning_rate_1�9���I       6%�	�uzO���A�*;


total_loss1*�@

error_R}>?

learning_rate_1�9����I       6%�	�zO���A�*;


total_loss8 A

error_R�OX?

learning_rate_1�9		tRI       6%�	5{O���A�*;


total_loss/[�@

error_R�V?

learning_rate_1�9���6I       6%�	�J{O���A�*;


total_lossHҬ@

error_R��A?

learning_rate_1�9T�*cI       6%�	��{O���A�*;


total_losstȡ@

error_R��H?

learning_rate_1�9D���I       6%�	"�{O���A�*;


total_loss�<�@

error_R�1T?

learning_rate_1�9UTI       6%�	�|O���A�*;


total_loss���@

error_R�N?

learning_rate_1�9��FI       6%�	�^|O���A�*;


total_loss���@

error_R�U?

learning_rate_1�9OU8I       6%�	0�|O���A�*;


total_losscA

error_R��H?

learning_rate_1�9ؚ��I       6%�	M�|O���A�*;


total_lossE��@

error_R�wX?

learning_rate_1�9Y���I       6%�	�.}O���A�*;


total_loss�Ϻ@

error_R{�??

learning_rate_1�9͜N
I       6%�		u}O���A�*;


total_loss�.�@

error_ROv]?

learning_rate_1�9�g�vI       6%�	G�}O���A�*;


total_loss�[�@

error_R$�=?

learning_rate_1�9�\I       6%�	�~O���A�*;


total_lossߦp@

error_R:J?

learning_rate_1�9�HgTI       6%�	�G~O���A�*;


total_loss�B�@

error_R�K?

learning_rate_1�9>�I       6%�	�~O���A�*;


total_loss���@

error_R�w<?

learning_rate_1�9�f�I       6%�	��~O���A�*;


total_loss�E�@

error_R�>Q?

learning_rate_1�9g1�I       6%�	�!O���A�*;


total_loss�+�@

error_RoJ?

learning_rate_1�9���I       6%�	kO���A�*;


total_loss��@

error_RxO?

learning_rate_1�9Ć�I       6%�	��O���A�*;


total_loss@.A

error_R�*H?

learning_rate_1�9�N��I       6%�	��O���A�*;


total_loss2ʰ@

error_RxvA?

learning_rate_1�9���I       6%�	z6�O���A�*;


total_lossI��@

error_RngV?

learning_rate_1�92ҫI       6%�	$z�O���A�*;


total_loss�a A

error_RdD?

learning_rate_1�9��GI       6%�	n��O���A�*;


total_loss��=A

error_R�M?

learning_rate_1�9�5�I       6%�	t�O���A�*;


total_loss#�@

error_R�:R?

learning_rate_1�9�;a�I       6%�	fP�O���A�*;


total_loss{��@

error_R�\\?

learning_rate_1�9�M�I       6%�	̕�O���A�*;


total_loss��@

error_R5T?

learning_rate_1�9j��I       6%�	�ځO���A�*;


total_loss��@

error_R�~7?

learning_rate_1�9W�}!I       6%�	��O���A�*;


total_loss/��@

error_RL�Z?

learning_rate_1�96N��I       6%�	[d�O���A�*;


total_lossF��@

error_R|'?

learning_rate_1�9p8��I       6%�	Ԯ�O���A�*;


total_loss8�@

error_R8�N?

learning_rate_1�9 �l!I       6%�	c�O���A�*;


total_lossm��@

error_R�#U?

learning_rate_1�9[��I       6%�	�?�O���A�*;


total_lossWt�@

error_R1;?

learning_rate_1�9K�QI       6%�	��O���A�*;


total_lossI�@

error_R�Va?

learning_rate_1�9���I       6%�	rуO���A�*;


total_loss��1A

error_R�PZ?

learning_rate_1�9�Ӑ!I       6%�	�O���A�*;


total_loss�Θ@

error_R��I?

learning_rate_1�97��I       6%�	�a�O���A�*;


total_lossg-�@

error_R��T?

learning_rate_1�9>�	I       6%�	3��O���A�*;


total_loss��@

error_R��V?

learning_rate_1�9'��I       6%�	��O���A�*;


total_lossd��@

error_R�pX?

learning_rate_1�90���I       6%�	%6�O���A�*;


total_lossؽ�@

error_R��I?

learning_rate_1�9��LI       6%�	p��O���A�*;


total_loss�֜@

error_R�tT?

learning_rate_1�9�J�-I       6%�	�ЅO���A�*;


total_loss_��@

error_R�T?

learning_rate_1�9�~#I       6%�	J�O���A�*;


total_loss�h�@

error_R�L?

learning_rate_1�9X�I       6%�	�d�O���A�*;


total_loss�:�@

error_RqXK?

learning_rate_1�9M�II       6%�	쮆O���A�*;


total_loss#;�@

error_RJsE?

learning_rate_1�9�每I       6%�	��O���A�*;


total_loss2A

error_R#L?

learning_rate_1�9˺ޕI       6%�	E�O���A�*;


total_loss�d�@

error_RcZY?

learning_rate_1�9.�vI       6%�	���O���A�*;


total_loss�
b@

error_R}9O?

learning_rate_1�9u�&I       6%�	(·O���A�*;


total_loss���@

error_R3�`?

learning_rate_1�9�V�I       6%�	&�O���A�*;


total_loss慪@

error_RA9[?

learning_rate_1�9w�_�I       6%�	�T�O���A�*;


total_losso��@

error_RW D?

learning_rate_1�9�?��I       6%�	��O���A�*;


total_lossC
�@

error_R
3Q?

learning_rate_1�9��)�I       6%�	�݈O���A�*;


total_loss�W�@

error_RѮW?

learning_rate_1�9c�cI       6%�	I!�O���A�*;


total_loss�s�@

error_R�KE?

learning_rate_1�9󞾌I       6%�	iu�O���A�*;


total_loss)R�@

error_R3`k?

learning_rate_1�9���I       6%�	�ΉO���A�*;


total_loss	ȿ@

error_R��M?

learning_rate_1�9E�tI       6%�	~�O���A�*;


total_lossO�@

error_R`SA?

learning_rate_1�9����I       6%�	*`�O���A�*;


total_loss��r@

error_R�8H?

learning_rate_1�9=�HHI       6%�	��O���A�*;


total_loss��@

error_R��K?

learning_rate_1�9"�I       6%�	;�O���A�*;


total_losss"�@

error_R]�n?

learning_rate_1�9]��I       6%�	�1�O���A�*;


total_loss��@

error_R-:_?

learning_rate_1�9=��I       6%�	w��O���A�*;


total_loss��@

error_R=AM?

learning_rate_1�9.�/nI       6%�	�ыO���A�*;


total_loss��A

error_R��Q?

learning_rate_1�9���I       6%�	z�O���A�*;


total_lossC��@

error_R1�I?

learning_rate_1�9���I       6%�	�d�O���A�*;


total_lossSE�@

error_RM�L?

learning_rate_1�9�q̺I       6%�	�O���A�*;


total_loss`̪@

error_R `K?

learning_rate_1�9stSI       6%�	l�O���A�*;


total_lossj�@

error_R�JN?

learning_rate_1�9^ �8I       6%�	Y�O���A�*;


total_lossH.�@

error_RvG?

learning_rate_1�9g���I       6%�	G��O���A�*;


total_loss�0�@

error_R��J?

learning_rate_1�9æ��I       6%�	J�O���A�*;


total_lossr��@

error_R4�[?

learning_rate_1�9��7I       6%�	�+�O���A�*;


total_loss
N�@

error_Rv�W?

learning_rate_1�9�5>WI       6%�	0q�O���A�*;


total_loss�x�@

error_R~[?

learning_rate_1�9����I       6%�	,��O���A�*;


total_loss�_�@

error_R��K?

learning_rate_1�9�t�I       6%�	���O���A�*;


total_loss Q�@

error_R%J?

learning_rate_1�9Y߫|I       6%�	?�O���A�*;


total_loss�A

error_R�CR?

learning_rate_1�9ϸ��I       6%�	~��O���A�*;


total_loss_��@

error_R�7K?

learning_rate_1�9��QI       6%�	0ɏO���A�*;


total_loss���@

error_RW�@?

learning_rate_1�9
�v�I       6%�	��O���A�*;


total_loss�a�@

error_RD�P?

learning_rate_1�9���I       6%�	�R�O���A�*;


total_lossf��@

error_R(N?

learning_rate_1�9���VI       6%�	��O���A�*;


total_loss��A

error_R�{D?

learning_rate_1�9�|�I       6%�	hڐO���A�*;


total_lossCS�@

error_Rv.B?

learning_rate_1�9B�5 I       6%�	�&�O���A�*;


total_loss�z�@

error_R��S?

learning_rate_1�9�i
�I       6%�	�o�O���A�*;


total_lossM�A

error_Ri�]?

learning_rate_1�9ϓI       6%�	���O���A�*;


total_loss���@

error_R�"I?

learning_rate_1�9�$xI       6%�	V��O���A�*;


total_loss�
�@

error_Rn�P?

learning_rate_1�9lݔ[I       6%�	GC�O���A�*;


total_loss�A�@

error_R��S?

learning_rate_1�9v�<I       6%�	��O���A�*;


total_lossE)�@

error_RT}W?

learning_rate_1�9��I       6%�	ȒO���A�*;


total_lossѶ�@

error_RD[?

learning_rate_1�9 J�I       6%�	��O���A�*;


total_loss�S�@

error_RE�M?

learning_rate_1�9�3I       6%�	T�O���A�*;


total_lossZ|�@

error_R��W?

learning_rate_1�9MT�I       6%�	W��O���A�*;


total_lossx(A

error_R�D?

learning_rate_1�9��I       6%�	��O���A�*;


total_lossW��@

error_R��M?

learning_rate_1�9&�/8I       6%�	�3�O���A�*;


total_loss!t�@

error_R�^?

learning_rate_1�9�_\|I       6%�	�~�O���A�*;


total_loss#�@

error_R=c?

learning_rate_1�9�.TI       6%�	=ĔO���A�*;


total_loss���@

error_R�):?

learning_rate_1�9�b�I       6%�	��O���A�*;


total_loss��@

error_Rr�>?

learning_rate_1�9U���I       6%�	�K�O���A�*;


total_lossMn�@

error_R�jT?

learning_rate_1�9�/Q�I       6%�	Ҏ�O���A�*;


total_loss��@

error_R3H7?

learning_rate_1�9��I       6%�	cҕO���A�*;


total_loss�L�@

error_R�'Q?

learning_rate_1�9�GY(I       6%�	��O���A�*;


total_loss�@

error_R�	N?

learning_rate_1�9���I       6%�	�_�O���A�*;


total_loss��@

error_RZ�]?

learning_rate_1�9�,GI       6%�	���O���A�*;


total_loss�׿@

error_R�G?

learning_rate_1�9���I       6%�	&��O���A�*;


total_loss۞A

error_R3#R?

learning_rate_1�90��I       6%�	�A�O���A�*;


total_lossm��@

error_R�+A?

learning_rate_1�9�!�-I       6%�	��O���A�*;


total_loss� �@

error_R�TO?

learning_rate_1�9�ԭCI       6%�	�ϗO���A�*;


total_loss�!�@

error_R��U?

learning_rate_1�9��afI       6%�	S�O���A�*;


total_lossoi�@

error_R�O?

learning_rate_1�9�BLI       6%�	�]�O���A�*;


total_loss��@

error_R.mE?

learning_rate_1�9���I       6%�	*��O���A�*;


total_loss��@

error_R,PT?

learning_rate_1�9�ӘI       6%�	��O���A�*;


total_loss��	A

error_R��Y?

learning_rate_1�9�}�I       6%�	�(�O���A�*;


total_loss�yA

error_RZ7]?

learning_rate_1�9$�+I       6%�	���O���A�*;


total_loss�~�@

error_R�FM?

learning_rate_1�9D�I       6%�	)��O���A�*;


total_losstN�@

error_RhM?

learning_rate_1�9�4�I       6%�	�:�O���A�*;


total_lossS�@

error_R��I?

learning_rate_1�9A���I       6%�	c�O���A�*;


total_loss萾@

error_R�+C?

learning_rate_1�9��4~I       6%�	�ĚO���A�*;


total_loss̤�@

error_R�UV?

learning_rate_1�9n�a8I       6%�	S�O���A�*;


total_loss��@

error_RD?

learning_rate_1�9V-�(I       6%�	Q�O���A�*;


total_loss���@

error_RM6\?

learning_rate_1�9��BI       6%�	-��O���A�*;


total_lossm1LA

error_R�U?

learning_rate_1�9f���I       6%�	��O���A�*;


total_lossl��@

error_R��O?

learning_rate_1�9dA�I       6%�	o.�O���A�*;


total_loss���@

error_R0D?

learning_rate_1�9X���I       6%�	�s�O���A�*;


total_loss�i�@

error_RͰG?

learning_rate_1�9) 8I       6%�	���O���A�*;


total_loss;A

error_RsLK?

learning_rate_1�9�`��I       6%�	��O���A�*;


total_loss��@

error_R��K?

learning_rate_1�9X��HI       6%�	3J�O���A�*;


total_lossMQ�@

error_R$�H?

learning_rate_1�9�ѢI       6%�	 ��O���A�*;


total_loss���@

error_R��K?

learning_rate_1�9���I       6%�	iܝO���A�*;


total_loss�]�@

error_R�O?

learning_rate_1�9��5�I       6%�	�"�O���A�*;


total_lossҌ�@

error_RW�Y?

learning_rate_1�9�WZ#I       6%�	�h�O���A�*;


total_loss#!�@

error_R�O?

learning_rate_1�9�=�eI       6%�	��O���A�*;


total_loss�f�@

error_R̋_?

learning_rate_1�9�7�~I       6%�	�O���A�*;


total_loss�3�@

error_R8�O?

learning_rate_1�9�d4sI       6%�	�0�O���A�*;


total_loss��@

error_Rd=E?

learning_rate_1�9Էg�I       6%�	�u�O���A�*;


total_loss��@

error_R�c??

learning_rate_1�9��f�I       6%�	���O���A�*;


total_loss�T�@

error_R�.W?

learning_rate_1�9xԣ�I       6%�	-��O���A�*;


total_lossXE�@

error_R܁P?

learning_rate_1�9pMI       6%�	�>�O���A�*;


total_lossXC�@

error_R�R?

learning_rate_1�9?p�I       6%�	���O���A�*;


total_loss���@

error_R�rA?

learning_rate_1�9�a�I       6%�	�ĠO���A�*;


total_loss{��@

error_RT�Y?

learning_rate_1�9Rx6�I       6%�	��O���A�*;


total_loss���@

error_R��Y?

learning_rate_1�9ȗ�OI       6%�	�M�O���A�*;


total_loss�U�@

error_R�gJ?

learning_rate_1�9�+��I       6%�	��O���A�*;


total_loss�ݡ@

error_R(�V?

learning_rate_1�9�X�~I       6%�	�աO���A�*;


total_loss�g�@

error_R�JP?

learning_rate_1�9�V��I       6%�	��O���A�*;


total_loss�D�@

error_R��V?

learning_rate_1�9fۙ�I       6%�	b�O���A�*;


total_loss��@

error_R�\?

learning_rate_1�9�q�8I       6%�	��O���A�*;


total_loss�
�@

error_RnjV?

learning_rate_1�9��yI       6%�	W�O���A�*;


total_loss2O�@

error_Rx i?

learning_rate_1�9{-I       6%�	)-�O���A�*;


total_loss���@

error_R�AP?

learning_rate_1�9ǂ�I       6%�	0o�O���A�*;


total_loss��@

error_R��Q?

learning_rate_1�93τI       6%�	W��O���A�*;


total_loss�ɳ@

error_R�p?

learning_rate_1�9cG�I       6%�	���O���A�*;


total_loss��@

error_R�@:?

learning_rate_1�9��I       6%�	�E�O���A�*;


total_loss�?�@

error_R$�K?

learning_rate_1�9N0klI       6%�	,��O���A�*;


total_lossS�@

error_R3jV?

learning_rate_1�9�8�I       6%�	�פO���A�*;


total_loss�e@

error_RmS?

learning_rate_1�9ĀosI       6%�	��O���A�*;


total_loss�^�@

error_R�lL?

learning_rate_1�9b��	I       6%�	Ye�O���A�*;


total_lossO�@

error_R�=I?

learning_rate_1�9"�	I       6%�	%��O���A�*;


total_lossIf�@

error_RQ�V?

learning_rate_1�9Us�PI       6%�	���O���A�*;


total_loss�D�@

error_R�M?

learning_rate_1�9�H�I       6%�	wE�O���A�*;


total_loss&P�@

error_R&�J?

learning_rate_1�9kH�I       6%�	���O���A�*;


total_loss
�@

error_R=]?

learning_rate_1�9���I       6%�	ΦO���A�*;


total_lossH/�@

error_R��T?

learning_rate_1�9�@�I       6%�	��O���A�*;


total_loss�BA

error_Re�T?

learning_rate_1�9NR�I       6%�	bT�O���A�*;


total_loss��@

error_R�Q:?

learning_rate_1�9�s]�I       6%�	ě�O���A�*;


total_losso�@

error_R�Y?

learning_rate_1�9��4I       6%�	p�O���A�*;


total_lossqv�@

error_R��_?

learning_rate_1�93�I       6%�	0�O���A�*;


total_loss_��@

error_R�hE?

learning_rate_1�9�;�I       6%�	Ms�O���A�*;


total_loss�A

error_R�w?

learning_rate_1�9k+C�I       6%�	䳨O���A�*;


total_loss3`�@

error_R��E?

learning_rate_1�9�^XI       6%�	���O���A�*;


total_lossI��@

error_R��S?

learning_rate_1�9��!�I       6%�	�<�O���A�*;


total_loss1�@

error_RV�O?

learning_rate_1�9�2�I       6%�	9��O���A�*;


total_loss�P�@

error_RɜZ?

learning_rate_1�9��z!I       6%�	e�O���A�*;


total_loss��@

error_R�G?

learning_rate_1�9�AHI       6%�	�6�O���A�*;


total_loss�#�@

error_R�W?

learning_rate_1�9M�arI       6%�	�x�O���A�*;


total_loss0�@

error_R�T?

learning_rate_1�9��#I       6%�	+��O���A�*;


total_loss&-�@

error_R�N?

learning_rate_1�9I]�eI       6%�	n��O���A�*;


total_loss�L�@

error_R�eF?

learning_rate_1�9�j��I       6%�	�C�O���A�*;


total_loss��@

error_R
�O?

learning_rate_1�9���I       6%�	��O���A�*;


total_loss���@

error_RnTH?

learning_rate_1�9fzII       6%�	��O���A�*;


total_loss��@

error_RCjU?

learning_rate_1�9zP}I       6%�	#�O���A�*;


total_loss6lA

error_R|V?

learning_rate_1�9�GKI       6%�	�l�O���A�*;


total_lossz��@

error_RmKE?

learning_rate_1�9����I       6%�	RĬO���A�*;


total_lossQ��@

error_R��c?

learning_rate_1�9���I       6%�	^�O���A�*;


total_loss�ԭ@

error_Rl�L?

learning_rate_1�9��߼I       6%�	tS�O���A�*;


total_losse$�@

error_R<X?

learning_rate_1�9S��I       6%�	k��O���A�*;


total_lossAl(A

error_R�SA?

learning_rate_1�9�M�I       6%�	9ݭO���A�*;


total_loss�vA

error_R�U?

learning_rate_1�9ӡDI       6%�	Y�O���A�*;


total_loss,��@

error_R�v??

learning_rate_1�93RCNI       6%�	�b�O���A�*;


total_lossHv�@

error_R��Y?

learning_rate_1�9 L��I       6%�	ƨ�O���A�*;


total_loss�\�@

error_RČL?

learning_rate_1�9��0�I       6%�	&�O���A�*;


total_loss<��@

error_Rͫe?

learning_rate_1�9�d��I       6%�	�8�O���A�*;


total_loss��@

error_R�WD?

learning_rate_1�9�Z�I       6%�	�z�O���A�*;


total_loss��@

error_R�<O?

learning_rate_1�9?�UAI       6%�	D��O���A�*;


total_loss
r�@

error_RZ,e?

learning_rate_1�9� }�I       6%�	��O���A�*;


total_loss�u�@

error_R}�F?

learning_rate_1�9}�jyI       6%�	�E�O���A�*;


total_loss=O�@

error_R��R?

learning_rate_1�9��I       6%�	��O���A�*;


total_loss<�@

error_R��Q?

learning_rate_1�9h|mWI       6%�	7�O���A�*;


total_loss	բ@

error_R�F?

learning_rate_1�9ǅ92I       6%�	�*�O���A�*;


total_loss?�@

error_R=�G?

learning_rate_1�9L��I       6%�	�p�O���A�*;


total_loss���@

error_Rr3a?

learning_rate_1�9��DGI       6%�	���O���A�*;


total_lossA�@

error_R&tF?

learning_rate_1�9�)yEI       6%�	���O���A�*;


total_loss\7�@

error_R�lK?

learning_rate_1�9p�I       6%�	KC�O���A�*;


total_lossL�@

error_R\aL?

learning_rate_1�9�4:7I       6%�	ō�O���A�*;


total_loss<��@

error_Rf�S?

learning_rate_1�9Dt�I       6%�	�ҲO���A�*;


total_loss�4	A

error_R�Wd?

learning_rate_1�9l3@�I       6%�	��O���A�*;


total_lossĿ�@

error_R�oT?

learning_rate_1�9��lI       6%�	.Z�O���A�*;


total_loss�'�@

error_R@H_?

learning_rate_1�9����I       6%�	暳O���A�*;


total_loss�i�@

error_R�zc?

learning_rate_1�9����I       6%�	�߳O���A�*;


total_loss㪨@

error_R�Q?

learning_rate_1�9�D�TI       6%�	5&�O���A�*;


total_loss��@

error_R��g?

learning_rate_1�9���3I       6%�	�h�O���A�*;


total_loss�ٖ@

error_RĚK?

learning_rate_1�9�U��I       6%�	���O���A�*;


total_loss���@

error_R�0f?

learning_rate_1�9S_ʹI       6%�	[��O���A�*;


total_loss��@

error_RL�`?

learning_rate_1�9=�غI       6%�	�9�O���A�*;


total_loss�%�@

error_R��f?

learning_rate_1�9Wv$�I       6%�	�{�O���A�*;


total_loss!��@

error_R�N?

learning_rate_1�9����I       6%�	HŵO���A�*;


total_loss���@

error_RI�>?

learning_rate_1�9T�NI       6%�	`�O���A�*;


total_loss!��@

error_Rc�J?

learning_rate_1�9K�)TI       6%�	}Q�O���A�*;


total_loss���@

error_RӸM?

learning_rate_1�9��u�I       6%�	�O���A�*;


total_loss==�@

error_Re�a?

learning_rate_1�9�Un�I       6%�	�ٶO���A�*;


total_lossl�@

error_R��s?

learning_rate_1�9�n�I       6%�	�O���A�*;


total_loss�خ@

error_RY_?

learning_rate_1�9��I       6%�	D_�O���A�*;


total_loss��@

error_R��X?

learning_rate_1�9vm��I       6%�	��O���A�*;


total_lossRƹ@

error_R�2V?

learning_rate_1�9WMZ�I       6%�	��O���A�*;


total_lossu �@

error_R_ZQ?

learning_rate_1�9hԢBI       6%�	�-�O���A�*;


total_loss�Ű@

error_R��S?

learning_rate_1�9�;fI       6%�	q�O���A�*;


total_loss�$�@

error_Rn�D?

learning_rate_1�9�ɗI       6%�	���O���A�*;


total_loss���@

error_R��I?

learning_rate_1�9�<GI       6%�	l��O���A�*;


total_loss���@

error_R�1[?

learning_rate_1�9���I       6%�	�;�O���A�*;


total_loss�2�@

error_R�I?

learning_rate_1�9��BI       6%�	،�O���A�*;


total_lossh1A

error_R�Z?

learning_rate_1�9��CI       6%�	��O���A�*;


total_loss�ӆ@

error_R��G?

learning_rate_1�9�?�&I       6%�	;�O���A�*;


total_loss���@

error_R�H4?

learning_rate_1�9 v�`I       6%�	��O���A�*;


total_loss��@

error_R�B9?

learning_rate_1�9JdT�I       6%�	κO���A�*;


total_loss�A

error_RDQ?

learning_rate_1�9Z`I       6%�	W�O���A�*;


total_loss�e�@

error_R�-L?

learning_rate_1�9�+�I       6%�	Bd�O���A�*;


total_loss�{�@

error_R�hU?

learning_rate_1�9�K�LI       6%�	���O���A�*;


total_loss�s�@

error_R�^?

learning_rate_1�9$I       6%�	�O���A�*;


total_loss�p@

error_R�?d?

learning_rate_1�9��y2I       6%�	?5�O���A�*;


total_loss���@

error_R��@?

learning_rate_1�9�1gI       6%�	Hx�O���A�*;


total_lossZk�@

error_R F?

learning_rate_1�9*TaI       6%�	i��O���A�*;


total_loss	�@

error_R��M?

learning_rate_1�9�%�I       6%�	 ��O���A�*;


total_loss\�@

error_R�;?

learning_rate_1�9Eg�I       6%�	D@�O���A�*;


total_lossTX�@

error_R��Y?

learning_rate_1�9���WI       6%�	��O���A�*;


total_loss�f�@

error_RRHC?

learning_rate_1�9ɓ�I       6%�	[̽O���A�*;


total_loss�+�@

error_R�C?

learning_rate_1�93��I       6%�	��O���A�*;


total_losss`�@

error_Rq�Z?

learning_rate_1�9�b{I       6%�	OT�O���A�*;


total_loss,W	A

error_R�L?

learning_rate_1�9^�eI       6%�	蘾O���A�*;


total_loss#��@

error_R��_?

learning_rate_1�94ѣ#I       6%�	eݾO���A�*;


total_loss�X@

error_RʤK?

learning_rate_1�9�/$I       6%�	�&�O���A�*;


total_loss5�@

error_Rm�Y?

learning_rate_1�9+u)/I       6%�	j�O���A�*;


total_loss��A

error_R�NU?

learning_rate_1�9W���I       6%�	
��O���A�*;


total_loss�D�@

error_R�S?

learning_rate_1�9���\I       6%�	h��O���A�*;


total_loss[��@

error_RNT?

learning_rate_1�9A'GI       6%�	�;�O���A�*;


total_lossi�@

error_R6�X?

learning_rate_1�9���I       6%�	���O���A�*;


total_loss�x�@

error_R\�Z?

learning_rate_1�9�lW�I       6%�	��O���A�*;


total_loss�!�@

error_R=�U?

learning_rate_1�9���I       6%�	B	�O���A�*;


total_loss��@

error_R�M?

learning_rate_1�9S���I       6%�		P�O���A�*;


total_loss��l@

error_R{�U?

learning_rate_1�9�]~!I       6%�	N��O���A�*;


total_loss8��@

error_R^?

learning_rate_1�9: �fI       6%�	���O���A�*;


total_losslV�@

error_R �q?

learning_rate_1�9�,�I       6%�	o7�O���A�*;


total_loss�k�@

error_R;]?

learning_rate_1�9e�#�I       6%�	��O���A�*;


total_loss�#�@

error_R��N?

learning_rate_1�9旃�I       6%�	���O���A�*;


total_loss�r�@

error_R�hS?

learning_rate_1�9���iI       6%�	;�O���A�*;


total_loss��@

error_R��V?

learning_rate_1�9�Z�	I       6%�	�J�O���A�*;


total_loss���@

error_R��a?

learning_rate_1�9��-I       6%�	W��O���A�*;


total_loss`T�@

error_Rz�X?

learning_rate_1�9�zAI       6%�	?��O���A�*;


total_loss��A

error_RzF?

learning_rate_1�9\�/}I       6%�	%�O���A�*;


total_loss��@

error_R��J?

learning_rate_1�9�h�=I       6%�	�s�O���A�*;


total_loss5�@

error_Rw(Y?

learning_rate_1�9SOiI       6%�	]��O���A�*;


total_losse�@

error_R��U?

learning_rate_1�9��I       6%�	��O���A�*;


total_lossO߱@

error_RZ�C?

learning_rate_1�9���I       6%�	�M�O���A�*;


total_loss!�@

error_RT�T?

learning_rate_1�9��QI       6%�	���O���A�*;


total_loss�-A

error_R�ZU?

learning_rate_1�9�*I       6%�	s��O���A�*;


total_loss80aA

error_RDO?

learning_rate_1�9�*�I       6%�	��O���A�*;


total_loss��@

error_R�/T?

learning_rate_1�9�k�qI       6%�	�]�O���A�*;


total_loss�ȶ@

error_R$�J?

learning_rate_1�9J��I       6%�	W��O���A�*;


total_loss��@

error_R��P?

learning_rate_1�98��:I       6%�	��O���A�*;


total_loss��@

error_RߥF?

learning_rate_1�9��gI       6%�	�,�O���A�*;


total_loss��@

error_R89G?

learning_rate_1�9m��[I       6%�	�n�O���A�*;


total_loss��J@

error_ROK?

learning_rate_1�9K
�I       6%�	���O���A�*;


total_loss�@

error_R$;T?

learning_rate_1�9W, �I       6%�	?��O���A�*;


total_loss��@

error_R��X?

learning_rate_1�9��ZI       6%�	�F�O���A�*;


total_lossn�9A

error_R�>U?

learning_rate_1�9�X7�I       6%�	���O���A�*;


total_lossb�@

error_Rd�Z?

learning_rate_1�9�i��I       6%�	��O���A�*;


total_loss��@

error_R.U?

learning_rate_1�9����I       6%�	h�O���A�*;


total_loss��:A

error_R�WU?

learning_rate_1�9��%I       6%�	.S�O���A�*;


total_loss�Q�@

error_R.�e?

learning_rate_1�9�+KI       6%�	���O���A�*;


total_loss���@

error_R�8?

learning_rate_1�9Qz2�I       6%�	��O���A�*;


total_loss���@

error_R�XA?

learning_rate_1�9|�U�I       6%�	QB�O���A�*;


total_lossq��@

error_R�)K?

learning_rate_1�9*F_I       6%�	���O���A�*;


total_lossqB�@

error_RM?

learning_rate_1�9,\�I       6%�	/��O���A�*;


total_loss휗@

error_R�Y?

learning_rate_1�9is��I       6%�	 �O���A�*;


total_lossx�@

error_Rs�D?

learning_rate_1�9���I       6%�	1R�O���A�*;


total_loss�+�@

error_R�[?

learning_rate_1�9��?�I       6%�	]��O���A�*;


total_loss��m@

error_R��P?

learning_rate_1�9��|�I       6%�	-��O���A�*;


total_lossrX�@

error_R�`?

learning_rate_1�96�� I       6%�	�:�O���A�*;


total_losse�@

error_RX/Q?

learning_rate_1�90�5I       6%�	b�O���A�*;


total_loss؃p@

error_R.�;?

learning_rate_1�9a՗&I       6%�	���O���A�*;


total_loss�	�@

error_Ra>P?

learning_rate_1�9G�I       6%�	�)�O���A�*;


total_loss�%A

error_R�#K?

learning_rate_1�9���I       6%�	�q�O���A�*;


total_loss㪶@

error_RAU?

learning_rate_1�9d�I       6%�	Q��O���A�*;


total_loss���@

error_R��P?

learning_rate_1�9�pb�I       6%�	���O���A�*;


total_loss���@

error_Ryh?

learning_rate_1�9�R�{I       6%�	@�O���A�*;


total_loss̤@

error_R�VI?

learning_rate_1�9-$G�I       6%�	{��O���A�*;


total_loss<)�@

error_R��P?

learning_rate_1�9�$�I       6%�	��O���A�*;


total_losss�p@

error_R��L?

learning_rate_1�9�F�DI       6%�	�
�O���A�*;


total_lossl>�@

error_RưT?

learning_rate_1�9%��I       6%�	�Q�O���A�*;


total_lossf[A

error_Rܾ]?

learning_rate_1�9��r�I       6%�	Ĕ�O���A�*;


total_loss��@

error_R��W?

learning_rate_1�9b�]jI       6%�	���O���A�*;


total_loss���@

error_R�KV?

learning_rate_1�9
NQI       6%�	?�O���A�*;


total_loss�A�@

error_R�vi?

learning_rate_1�9��I       6%�	�]�O���A�*;


total_loss3١@

error_RaTN?

learning_rate_1�9NU<�I       6%�	���O���A�*;


total_lossL��@

error_Rf�U?

learning_rate_1�9#��,I       6%�	���O���A�*;


total_loss���@

error_R�K?

learning_rate_1�9
zr5I       6%�	�-�O���A�*;


total_loss�M�@

error_R� Y?

learning_rate_1�9g���I       6%�	Fr�O���A�*;


total_loss�;�@

error_R�)Y?

learning_rate_1�9m�-�I       6%�	���O���A�*;


total_losso��@

error_R=�O?

learning_rate_1�9���I       6%�	{��O���A�*;


total_losse�@

error_Ra#K?

learning_rate_1�9�'&=I       6%�	 <�O���A�*;


total_lossܲ�@

error_Rq�M?

learning_rate_1�9�&�]I       6%�	�~�O���A�*;


total_loss���@

error_R�,I?

learning_rate_1�9ܓ�I       6%�	&��O���A�*;


total_loss��@

error_RڵL?

learning_rate_1�9S� I       6%�	�O���A�*;


total_lossOl�@

error_R�O?

learning_rate_1�9��EI       6%�	�O�O���A�*;


total_loss���@

error_R /d?

learning_rate_1�93��I       6%�	ʗ�O���A�*;


total_lossVjA

error_R�R?

learning_rate_1�9�﷈I       6%�	���O���A�*;


total_loss��@

error_R�[?

learning_rate_1�9��&�I       6%�	�$�O���A�*;


total_loss���@

error_R�eS?

learning_rate_1�9=*�I       6%�	:m�O���A�*;


total_loss�\�@

error_Ri"`?

learning_rate_1�9�/m"I       6%�	���O���A�*;


total_loss,<�@

error_R��K?

learning_rate_1�9XP��I       6%�	q��O���A�*;


total_loss���@

error_R�[H?

learning_rate_1�9�YfI       6%�	-6�O���A�*;


total_loss=��@

error_R=�9?

learning_rate_1�9��0�I       6%�	�w�O���A�*;


total_loss��@

error_R�uH?

learning_rate_1�9�5
I       6%�	���O���A�*;


total_loss���@

error_R�Y`?

learning_rate_1�9��	�I       6%�	|��O���A�*;


total_loss���@

error_R��>?

learning_rate_1�9d���I       6%�	�A�O���A�*;


total_loss��@

error_R��G?

learning_rate_1�9*?�I       6%�	��O���A�*;


total_loss�:�@

error_R��U?

learning_rate_1�9�s"�I       6%�	���O���A�*;


total_lossi+�@

error_RX?

learning_rate_1�9�J�II       6%�	�O���A�*;


total_loss�A

error_RԚX?

learning_rate_1�9*$�I       6%�	=Z�O���A�*;


total_loss���@

error_R��c?

learning_rate_1�9z�iI       6%�	���O���A�*;


total_lossŁ@

error_R��R?

learning_rate_1�9r�[2I       6%�	���O���A�*;


total_loss���@

error_R{�A?

learning_rate_1�9��nI       6%�	3�O���A�*;


total_loss�а@

error_R�g^?

learning_rate_1�9����I       6%�	Y~�O���A�*;


total_lossq�@

error_R��e?

learning_rate_1�9�TVI       6%�	ݿ�O���A�*;


total_lossw�@

error_R��M?

learning_rate_1�9�%I       6%�	��O���A�*;


total_loss��@

error_R��Z?

learning_rate_1�9�6TyI       6%�	?D�O���A�*;


total_loss�H�@

error_R�X?

learning_rate_1�9��
I       6%�	4��O���A�*;


total_loss�ڳ@

error_RDU?

learning_rate_1�9r�/QI       6%�	!��O���A�*;


total_lossl�@

error_R�oH?

learning_rate_1�9Q���I       6%�	74�O���A�*;


total_loss���@

error_RqG?

learning_rate_1�9B<GI       6%�	�y�O���A�*;


total_loss���@

error_R�G?

learning_rate_1�9]I       6%�	���O���A�*;


total_loss���@

error_R�@L?

learning_rate_1�9X�r�I       6%�	���O���A�*;


total_lossFK�@

error_R�T?

learning_rate_1�9� �I       6%�	�D�O���A�*;


total_loss��@

error_R�vP?

learning_rate_1�9ݮ�I       6%�	j��O���A�*;


total_loss�@

error_Rn�S?

learning_rate_1�9y�xI       6%�	���O���A�*;


total_loss���@

error_R��G?

learning_rate_1�9�G+%I       6%�	T%�O���A�*;


total_loss��@

error_R`O?

learning_rate_1�9��I       6%�	�g�O���A�*;


total_lossC_�@

error_RP\?

learning_rate_1�9%�=�I       6%�	H��O���A�*;


total_lossD/�@

error_R�BU?

learning_rate_1�96�MXI       6%�	���O���A�*;


total_loss�W�@

error_R��P?

learning_rate_1�9Y�~�I       6%�	V5�O���A�*;


total_loss�*�@

error_R�!A?

learning_rate_1�92	@I       6%�	��O���A�*;


total_loss۹�@

error_R��N?

learning_rate_1�9�~vI       6%�	���O���A�*;


total_loss���@

error_Rw�K?

learning_rate_1�9����I       6%�	
�O���A�*;


total_loss�|�@

error_R��L?

learning_rate_1�9!�._I       6%�	�Q�O���A�*;


total_loss4�@

error_R��k?

learning_rate_1�9��LoI       6%�	���O���A�*;


total_loss&A

error_R�m?

learning_rate_1�91n��I       6%�	���O���A�*;


total_loss|��@

error_R
�S?

learning_rate_1�9t���I       6%�	��O���A�*;


total_loss��@

error_RTsN?

learning_rate_1�9�J2;I       6%�	Sf�O���A�*;


total_loss,~�@

error_R�2M?

learning_rate_1�9͵kI       6%�	(��O���A�*;


total_loss=0�@

error_R�\?

learning_rate_1�9�5�I       6%�	Z��O���A�*;


total_loss��@

error_R�hU?

learning_rate_1�9�r��I       6%�	0�O���A�*;


total_loss|��@

error_R)�Q?

learning_rate_1�9����I       6%�	Ju�O���A�*;


total_loss*Ѿ@

error_Rq/F?

learning_rate_1�9�D�I       6%�	͸�O���A�*;


total_loss��@

error_R�>X?

learning_rate_1�9�u~I       6%�	@��O���A�*;


total_loss��@

error_RX�O?

learning_rate_1�9�/NI       6%�	H>�O���A�*;


total_loss���@

error_R}%W?

learning_rate_1�9Y�WI       6%�	��O���A�*;


total_loss���@

error_R��`?

learning_rate_1�9�a�$I       6%�	��O���A�*;


total_loss�*�@

error_R;Y?

learning_rate_1�9���RI       6%�	��O���A�*;


total_lossK�@

error_Rv,L?

learning_rate_1�9�e�I       6%�	�I�O���A�*;


total_loss �@

error_RH�\?

learning_rate_1�9�s��I       6%�	I��O���A�*;


total_lossE��@

error_REQ?

learning_rate_1�9&@��I       6%�	���O���A�*;


total_losss�@

error_RWtg?

learning_rate_1�9��$�I       6%�	��O���A�*;


total_loss�A

error_R�C?

learning_rate_1�9��r+I       6%�	�R�O���A�*;


total_loss��@

error_R$pJ?

learning_rate_1�9�3��I       6%�	ޗ�O���A�*;


total_loss\��@

error_R��K?

learning_rate_1�9�l�DI       6%�	���O���A�*;


total_loss���@

error_R*�M?

learning_rate_1�9+cy�I       6%�	�#�O���A�*;


total_lossa��@

error_R/�>?

learning_rate_1�9�g}I       6%�	�h�O���A�*;


total_loss���@

error_R[MJ?

learning_rate_1�9�"��I       6%�	)��O���A�*;


total_lossd��@

error_R� <?

learning_rate_1�9D'%}I       6%�	"��O���A�*;


total_lossÐ�@

error_RJ_R?

learning_rate_1�9o��rI       6%�	
4�O���A�*;


total_loss3��@

error_R,:B?

learning_rate_1�9vo�I       6%�	�w�O���A�*;


total_loss] �@

error_R��V?

learning_rate_1�9f�SI       6%�	���O���A�*;


total_loss� �@

error_RQ?

learning_rate_1�92jg�I       6%�	� �O���A�*;


total_lossj(�@

error_RZ?

learning_rate_1�9��I       6%�	C�O���A�*;


total_loss0��@

error_Ri�0?

learning_rate_1�9x��I       6%�	t��O���A�*;


total_loss8�@

error_R;�I?

learning_rate_1�9W1��I       6%�	"��O���A�*;


total_loss�n�@

error_Rd�J?

learning_rate_1�9�2�I       6%�	c�O���A�*;


total_loss}��@

error_R7K?

learning_rate_1�9��LI       6%�	YS�O���A�*;


total_loss��A

error_R��W?

learning_rate_1�9��"fI       6%�	���O���A�*;


total_loss@��@

error_R!�^?

learning_rate_1�9V7��I       6%�	��O���A�*;


total_lossÄ�@

error_R��P?

learning_rate_1�9G��I       6%�	"*�O���A�*;


total_lossΤ�@

error_RZ�W?

learning_rate_1�9�?�BI       6%�	�r�O���A�*;


total_loss�ӻ@

error_R��V?

learning_rate_1�9�/I       6%�	0��O���A�*;


total_loss���@

error_RO�P?

learning_rate_1�9�6�I       6%�	�O���A�*;


total_loss���@

error_R
�S?

learning_rate_1�9�OI       6%�	�I�O���A�*;


total_lossW��@

error_R�sW?

learning_rate_1�9j3�TI       6%�	E��O���A�*;


total_loss$�@

error_R��M?

learning_rate_1�9��diI       6%�	��O���A�*;


total_loss���@

error_R��H?

learning_rate_1�9��I       6%�	�;�O���A�*;


total_loss�J�@

error_R�td?

learning_rate_1�9�ӻ�I       6%�	_}�O���A�*;


total_loss��@

error_R�I?

learning_rate_1�91�0I       6%�	-��O���A�*;


total_loss�У@

error_R�M?

learning_rate_1�9�)TI       6%�	��O���A�*;


total_loss<�@

error_RFI?

learning_rate_1�9�ܳ I       6%�	ZP�O���A�*;


total_loss�A

error_RsEO?

learning_rate_1�9�c|[I       6%�	̩�O���A�*;


total_loss���@

error_R��S?

learning_rate_1�9��qI       6%�	���O���A�*;


total_lossl[�@

error_R
k^?

learning_rate_1�9yI       6%�	?�O���A�*;


total_lossn��@

error_Rv�Q?

learning_rate_1�9����I       6%�	C��O���A�*;


total_loss�D A

error_R�pa?

learning_rate_1�9��g�I       6%�	���O���A�*;


total_loss�u�@

error_R��Y?

learning_rate_1�9�7��I       6%�	�'�O���A�*;


total_loss���@

error_RFW?

learning_rate_1�9uhvjI       6%�	k�O���A�*;


total_loss��@

error_RH%L?

learning_rate_1�9:��5I       6%�	���O���A�*;


total_loss(��@

error_R�6;?

learning_rate_1�9���.I       6%�	��O���A�*;


total_loss�A

error_R��;?

learning_rate_1�9�8�HI       6%�	�>�O���A�*;


total_loss���@

error_RC�`?

learning_rate_1�92q��I       6%�	���O���A�*;


total_loss��@

error_RcU]?

learning_rate_1�9�e!oI       6%�	���O���A�*;


total_loss��@

error_RI�@?

learning_rate_1�9�щ�I       6%�	�O���A�*;


total_lossS��@

error_R�E?

learning_rate_1�9�~,�I       6%�	�S�O���A�*;


total_lossDl�@

error_R`�b?

learning_rate_1�9 y{I       6%�	?��O���A�*;


total_loss�d@

error_RU?

learning_rate_1�9���I       6%�	"��O���A�*;


total_loss[?�@

error_R�C?

learning_rate_1�9�~V�I       6%�	\*�O���A�*;


total_loss�M�@

error_Ri�`?

learning_rate_1�9��RI       6%�	�s�O���A�*;


total_lossV(�@

error_RH�J?

learning_rate_1�9o�|VI       6%�	���O���A�*;


total_loss���@

error_RR?

learning_rate_1�9*��|I       6%�	���O���A�*;


total_loss<m	A

error_R1�P?

learning_rate_1�9��d�I       6%�	�G�O���A�*;


total_loss<Ѫ@

error_R&h?

learning_rate_1�9�ؑDI       6%�	��O���A�*;


total_lossW��@

error_R�I?

learning_rate_1�9�-��I       6%�	���O���A�*;


total_loss
�A

error_R,M?

learning_rate_1�9����I       6%�	!�O���A�*;


total_loss���@

error_R�L?

learning_rate_1�9��M�I       6%�	�c�O���A�*;


total_loss�٧@

error_R~A?

learning_rate_1�9�0I       6%�	ͧ�O���A�*;


total_loss�p�@

error_R��J?

learning_rate_1�9����I       6%�	��O���A�*;


total_loss�!�@

error_R�M?

learning_rate_1�9lz�DI       6%�	�+�O���A�*;


total_lossl0�@

error_Rq�k?

learning_rate_1�9�=ߥI       6%�	o�O���A�*;


total_lossE�@

error_R}�N?

learning_rate_1�9aLLI       6%�	���O���A�*;


total_loss���@

error_R�|a?

learning_rate_1�9_+'�I       6%�	z��O���A�*;


total_loss���@

error_Rf@U?

learning_rate_1�9�	�I       6%�	�>�O���A�*;


total_lossH��@

error_R�=?

learning_rate_1�9Gf�I       6%�	k��O���A�*;


total_loss�1�@

error_R��M?

learning_rate_1�9e���I       6%�	���O���A�*;


total_loss�@

error_R]2P?

learning_rate_1�9��(%I       6%�	�O���A�*;


total_loss��k@

error_R1�G?

learning_rate_1�90EjnI       6%�	-`�O���A�*;


total_loss���@

error_R��H?

learning_rate_1�9L��I       6%�	~��O���A�*;


total_loss*1A

error_R�lS?

learning_rate_1�9�d�I       6%�	���O���A�*;


total_lossƓ@

error_R��E?

learning_rate_1�9���MI       6%�	U2�O���A�*;


total_loss��@

error_R߽g?

learning_rate_1�9�.�I       6%�	u�O���A�*;


total_loss�k�@

error_R�N?

learning_rate_1�90�D�I       6%�	-��O���A�*;


total_lossRV@

error_R`�;?

learning_rate_1�94�FI       6%�	��O���A�*;


total_loss�ק@

error_R��f?

learning_rate_1�9�`�OI       6%�	�@�O���A�*;


total_loss�r�@

error_R�RL?

learning_rate_1�9~��^I       6%�	���O���A�*;


total_loss<�@

error_R��P?

learning_rate_1�9�̶yI       6%�	5��O���A�*;


total_loss��@

error_RΤf?

learning_rate_1�95=eKI       6%�	8�O���A�*;


total_losse��@

error_R�KW?

learning_rate_1�9QE I       6%�	^N�O���A�*;


total_loss�@�@

error_R@�F?

learning_rate_1�9��JI       6%�	��O���A�*;


total_losso`�@

error_R�l@?

learning_rate_1�9��b�I       6%�	*��O���A�*;


total_loss��@

error_Rhc?

learning_rate_1�9C5�I       6%�	��O���A�*;


total_loss���@

error_R`OJ?

learning_rate_1�9�_I       6%�	\e�O���A�*;


total_lossc|�@

error_R��H?

learning_rate_1�9A�I       6%�	���O���A�*;


total_loss��@

error_R�.\?

learning_rate_1�9��
QI       6%�	��O���A�*;


total_loss<i�@

error_Rv�F?

learning_rate_1�99xS�I       6%�	V[�O���A�*;


total_loss�2�@

error_R�R?

learning_rate_1�9N�@.I       6%�	���O���A�*;


total_loss��@

error_R�>Q?

learning_rate_1�9�B8I       6%�	=��O���A�*;


total_loss\��@

error_Ra�I?

learning_rate_1�9���I       6%�	%9�O���A�*;


total_loss={�@

error_R=l?

learning_rate_1�9y��aI       6%�	Մ�O���A�*;


total_lossvq�@

error_R,WS?

learning_rate_1�9��L�I       6%�	��O���A�*;


total_lossMG�@

error_RBE?

learning_rate_1�9Y:�I       6%�	��O���A�*;


total_loss�<�@

error_R��7?

learning_rate_1�9��$iI       6%�	md�O���A�*;


total_lossc�@

error_R6P?

learning_rate_1�9���I       6%�	���O���A�*;


total_lossv��@

error_R6_?

learning_rate_1�9R�cI       6%�	���O���A�*;


total_lossF�A

error_RV�J?

learning_rate_1�95��zI       6%�	Y:�O���A�*;


total_lossZM�@

error_R��Y?

learning_rate_1�9�XI       6%�	��O���A�*;


total_lossR��@

error_R��;?

learning_rate_1�9�-�II       6%�	���O���A�*;


total_lossEsb@

error_R�B?

learning_rate_1�9��:FI       6%�	��O���A�*;


total_lossa��@

error_R�g?

learning_rate_1�9l|gI       6%�	2g�O���A�*;


total_loss�0A

error_R��E?

learning_rate_1�9OX��I       6%�	x��O���A�*;


total_loss�)�@

error_R��A?

learning_rate_1�9{��I       6%�	���O���A�*;


total_loss,��@

error_R/�K?

learning_rate_1�9��-I       6%�	?J�O���A�*;


total_loss$%�@

error_R�d?

learning_rate_1�9�T1�I       6%�	;��O���A�*;


total_lossó@

error_R �Q?

learning_rate_1�9�W<�I       6%�	���O���A�*;


total_loss�z�@

error_RdS?

learning_rate_1�9ye�I       6%�	� P���A�*;


total_loss��@

error_RHgV?

learning_rate_1�9L�5I       6%�	�` P���A�*;


total_loss��@

error_R�.U?

learning_rate_1�9�?��I       6%�	u� P���A�*;


total_lossX� A

error_RJ�]?

learning_rate_1�9Z�mI       6%�	O� P���A�*;


total_loss��@

error_R��C?

learning_rate_1�9ZI       6%�	�)P���A�*;


total_loss1�@

error_RO?

learning_rate_1�9��I       6%�	�nP���A�*;


total_loss ��@

error_R��H?

learning_rate_1�97��I       6%�	�P���A�*;


total_losss��@

error_R�Q?

learning_rate_1�9Dj|I       6%�	��P���A�*;


total_lossd��@

error_R�/Q?

learning_rate_1�9��%TI       6%�	�=P���A�*;


total_loss`�@

error_R�_L?

learning_rate_1�9I�
�I       6%�	s�P���A�*;


total_loss��@

error_R�[?

learning_rate_1�9+���I       6%�	��P���A�*;


total_loss�y�@

error_RT?

learning_rate_1�9��oI       6%�	,P���A�*;


total_loss���@

error_Rc�G?

learning_rate_1�9��I       6%�	LP���A�*;


total_lossIj�@

error_R�5U?

learning_rate_1�98��I       6%�	!�P���A�*;


total_loss��r@

error_R��J?

learning_rate_1�9�W�I       6%�	.�P���A�*;


total_lossS��@

error_R�WI?

learning_rate_1�9�p��I       6%�	�"P���A�*;


total_loss���@

error_R��^?

learning_rate_1�9��ؓI       6%�	nP���A�*;


total_loss���@

error_R��P?

learning_rate_1�9���XI       6%�	~�P���A�*;


total_loss{��@

error_R.P?

learning_rate_1�9���XI       6%�	�P���A�*;


total_losse��@

error_R	P?

learning_rate_1�9�]�I       6%�	IP���A�*;


total_loss���@

error_R�J?

learning_rate_1�9}�ܷI       6%�	��P���A�*;


total_loss:s�@

error_R��M?

learning_rate_1�9�	?I       6%�	��P���A�*;


total_loss��@

error_RqZ?

learning_rate_1�9PkI       6%�	P���A�*;


total_loss�E�@

error_RE�R?

learning_rate_1�9!��I       6%�	mXP���A�*;


total_loss.��@

error_R�V?

learning_rate_1�9���I       6%�	��P���A�*;


total_loss�/A

error_R�,a?

learning_rate_1�9^��I       6%�	�P���A�*;


total_loss� A

error_R�J??

learning_rate_1�9��� I       6%�	�'P���A�*;


total_loss�L�@

error_R��T?

learning_rate_1�9YVZI       6%�	�lP���A�*;


total_loss�Ȁ@

error_R�X?

learning_rate_1�98C�I       6%�	 �P���A�*;


total_loss�k�@

error_Rx�[?

learning_rate_1�9o
cKI       6%�	��P���A�*;


total_loss�ɲ@

error_R��N?

learning_rate_1�9?C4�I       6%�	|5P���A�*;


total_loss�Px@

error_R��H?

learning_rate_1�9�i��I       6%�	�wP���A�*;


total_loss�K�@

error_R#�E?

learning_rate_1�9bQ I       6%�	��P���A�*;


total_lossh�@

error_R"X?

learning_rate_1�9̿!�I       6%�	|	P���A�*;


total_loss���@

error_R2>S?

learning_rate_1�9��(|I       6%�	jI	P���A�*;


total_loss���@

error_R��U?

learning_rate_1�9��OI       6%�	�	P���A�*;


total_loss|v�@

error_R��P?

learning_rate_1�93��I       6%�	|
P���A�*;


total_loss��m@

error_R}Ix?

learning_rate_1�9��m+I       6%�	&F
P���A�*;


total_loss*��@

error_RO?

learning_rate_1�9�\�.I       6%�	v�
P���A�*;


total_loss�%�@

error_R��U?

learning_rate_1�9&y�yI       6%�	��
P���A�*;


total_loss6�@

error_R�*P?

learning_rate_1�9>�I       6%�	�P���A�*;


total_loss��@

error_R�M?

learning_rate_1�9�T I       6%�	ZcP���A�*;


total_loss���@

error_R��c?

learning_rate_1�95Z=rI       6%�	'�P���A�*;


total_loss}��@

error_R��M?

learning_rate_1�9U�Q�I       6%�	� P���A�*;


total_loss=\�@

error_Ra�F?

learning_rate_1�9'�\�I       6%�	2GP���A�*;


total_loss$*�@

error_R[fF?

learning_rate_1�9���I       6%�	ɓP���A�*;


total_loss�q�@

error_RJ]?

learning_rate_1�9�RS�I       6%�	~�P���A�*;


total_loss�o�@

error_R��L?

learning_rate_1�9_V3I       6%�	�6P���A�*;


total_loss��@

error_R=�[?

learning_rate_1�9w7t[I       6%�	�{P���A�*;


total_loss+W�@

error_R� @?

learning_rate_1�9���I       6%�	
�P���A�*;


total_lossne�@

error_RL?

learning_rate_1�9.O�eI       6%�	DP���A�*;


total_loss���@

error_R=�c?

learning_rate_1�9�}��I       6%�	KP���A�*;


total_loss�@

error_Ri�;?

learning_rate_1�97p3�I       6%�	ƔP���A�*;


total_loss���@

error_R�8K?

learning_rate_1�9�̧YI       6%�	4�P���A�*;


total_loss�0�@

error_R�6K?

learning_rate_1�9a�I       6%�	"%P���A�*;


total_loss��@

error_R�9?

learning_rate_1�9g1a�I       6%�	�iP���A�*;


total_lossߒ@

error_R.%Q?

learning_rate_1�9Q��JI       6%�	%�P���A�*;


total_loss��LA

error_R�69?

learning_rate_1�9v�I       6%�	�P���A�*;


total_loss���@

error_R�KG?

learning_rate_1�9BQ>aI       6%�	P?P���A�*;


total_loss�7�@

error_R�F?

learning_rate_1�9dg�I       6%�	��P���A�*;


total_lossZ�@

error_R]@W?

learning_rate_1�9�Qw�I       6%�	��P���A�*;


total_loss=�A

error_R��d?

learning_rate_1�9�W�gI       6%�	�P���A�*;


total_lossW?�@

error_R;_K?

learning_rate_1�9i��I       6%�	�dP���A�*;


total_loss��A

error_Rq>?

learning_rate_1�9�6�[I       6%�	�P���A�*;


total_loss./�@

error_R\hR?

learning_rate_1�9���I       6%�	o�P���A�*;


total_loss WA

error_R3CT?

learning_rate_1�9f��tI       6%�	�/P���A�*;


total_lossCܬ@

error_RsV?

learning_rate_1�9m�a7I       6%�	LuP���A�*;


total_lossA�@

error_R�zK?

learning_rate_1�9VouI       6%�	m�P���A�*;


total_loss,��@

error_R�G?

learning_rate_1�9Б<CI       6%�	�P���A�*;


total_loss@�h@

error_R�U?

learning_rate_1�9�}wI       6%�	�?P���A�*;


total_loss/c�@

error_Rn�J?

learning_rate_1�9��t�I       6%�	��P���A�*;


total_loss���@

error_R�.R?

learning_rate_1�9l�^I       6%�	��P���A�*;


total_lossW@�@

error_R��O?

learning_rate_1�9�-�I       6%�	+P���A�*;


total_loss$��@

error_Rn0L?

learning_rate_1�9L2\�I       6%�	VP���A�*;


total_loss}3�@

error_R��D?

learning_rate_1�9�Ң�I       6%�	�dP���A�*;


total_loss��@

error_R(gB?

learning_rate_1�9�=^I       6%�	t�P���A�*;


total_loss{�A

error_R؝H?

learning_rate_1�9�j+I       6%�	�P���A�*;


total_loss��@

error_R��H?

learning_rate_1�9���uI       6%�	�OP���A�*;


total_loss��@

error_Ra�Z?

learning_rate_1�9����I       6%�	��P���A�*;


total_lossr�@

error_R�T?

learning_rate_1�9`�SI       6%�	��P���A�*;


total_lossE��@

error_R�Z?

learning_rate_1�9֮ĵI       6%�	'P���A�*;


total_loss���@

error_R��K?

learning_rate_1�9j��I       6%�	lP���A�*;


total_loss<�v@

error_R�Lk?

learning_rate_1�9MթI       6%�	8�P���A�*;


total_loss�A

error_R�3b?

learning_rate_1�9֙b�I       6%�	"P���A�*;


total_loss�G�@

error_Rڙp?

learning_rate_1�9�4\zI       6%�	�dP���A�*;


total_loss�f�@

error_RIF?

learning_rate_1�9R��QI       6%�	=�P���A�*;


total_lossZ�@

error_R3�Y?

learning_rate_1�9)�y5I       6%�	e�P���A�*;


total_lossu՟@

error_R&L?

learning_rate_1�9��J�I       6%�	�4P���A�*;


total_lossPA

error_R L?

learning_rate_1�9t�@I       6%�	�{P���A�*;


total_loss�*�@

error_R
N?

learning_rate_1�9�{$I       6%�	��P���A�*;


total_loss_�b@

error_RO*S?

learning_rate_1�9�>�VI       6%�	�
P���A�*;


total_lossI�@

error_R��Y?

learning_rate_1�9�α�I       6%�	YP���A�*;


total_loss�v�@

error_Rd�N?

learning_rate_1�9F��'I       6%�		�P���A�*;


total_lossKu�@

error_R�cH?

learning_rate_1�9څ�I       6%�	�P���A�*;


total_lossU�@

error_R�uE?

learning_rate_1�96bI       6%�	+)P���A�*;


total_loss(Z�@

error_R%Z?

learning_rate_1�9����I       6%�	FpP���A�*;


total_loss�l�@

error_RE�_?

learning_rate_1�9��n�I       6%�	��P���A�*;


total_loss�g�@

error_R5\?

learning_rate_1�9�<o�I       6%�	+�P���A�*;


total_loss�@

error_R�:V?

learning_rate_1�9�V�6I       6%�	�CP���A�*;


total_loss6�A

error_RY?

learning_rate_1�9ނu[I       6%�	)�P���A�*;


total_loss1��@

error_R�A?

learning_rate_1�9��ǱI       6%�	��P���A�*;


total_loss<�@

error_R�IV?

learning_rate_1�9��nnI       6%�	�P���A�*;


total_loss��@

error_R��E?

learning_rate_1�9��JI       6%�	`P���A�*;


total_loss���@

error_R8c`?

learning_rate_1�9��ȊI       6%�	��P���A�*;


total_loss��z@

error_Rv�L?

learning_rate_1�9��gVI       6%�	��P���A�*;


total_loss���@

error_R��\?

learning_rate_1�9��lI       6%�	�) P���A�*;


total_lossz�@

error_R�R?

learning_rate_1�9�U�3I       6%�	Op P���A�*;


total_loss"S�@

error_R�eS?

learning_rate_1�9p��YI       6%�	E� P���A�*;


total_loss�t�@

error_R	�2?

learning_rate_1�9�;�/I       6%�	v� P���A�*;


total_loss���@

error_R9L?

learning_rate_1�9~��I       6%�	{B!P���A�*;


total_loss�?�@

error_R �@?

learning_rate_1�9�KX!I       6%�	�!P���A�*;


total_loss���@

error_R� K?

learning_rate_1�9��"�I       6%�	3�!P���A�*;


total_loss��A

error_RC|Q?

learning_rate_1�9����I       6%�	2?"P���A�*;


total_loss)��@

error_Ri�E?

learning_rate_1�9����I       6%�	��"P���A�*;


total_loss���@

error_R�qP?

learning_rate_1�9hBI       6%�	��"P���A�*;


total_loss�.�@

error_RDwP?

learning_rate_1�9PV�1I       6%�	=#P���A�*;


total_lossW��@

error_ROlR?

learning_rate_1�9<�1�I       6%�	҈#P���A�*;


total_loss�p�@

error_R`A?

learning_rate_1�9��5I       6%�	9�#P���A�*;


total_loss��@

error_R�Ji?

learning_rate_1�9�	"�I       6%�	�;$P���A�*;


total_loss��@

error_R�Q^?

learning_rate_1�91�kYI       6%�	J�$P���A�*;


total_loss���@

error_R!LW?

learning_rate_1�9�Q	I       6%�	k�$P���A�*;


total_losso��@

error_R��T?

learning_rate_1�9��~�I       6%�	IG%P���A�*;


total_loss\�A

error_R�B?

learning_rate_1�9�ve7I       6%�	��%P���A�*;


total_loss���@

error_Rs�R?

learning_rate_1�9��vI       6%�	8�%P���A�*;


total_loss�y�@

error_R_�U?

learning_rate_1�9z��&I       6%�	G)&P���A�*;


total_loss�]�@

error_R_�V?

learning_rate_1�9��\I       6%�	Nt&P���A�*;


total_loss��@

error_RߑH?

learning_rate_1�9	D�sI       6%�	��&P���A�*;


total_loss�,�@

error_R��G?

learning_rate_1�9(�I       6%�	B'P���A�*;


total_loss��@

error_RdBA?

learning_rate_1�9��+I       6%�	�q'P���A�*;


total_loss�e@

error_R��f?

learning_rate_1�9T7��I       6%�	C�'P���A�*;


total_lossV	�@

error_R�5N?

learning_rate_1�9��I       6%�	�(P���A�*;


total_lossq�@

error_Rϻ<?

learning_rate_1�9To=�I       6%�	}H(P���A�*;


total_loss�׻@

error_R�A?

learning_rate_1�9�Y$I       6%�	��(P���A�*;


total_loss��@

error_R�TD?

learning_rate_1�95րI       6%�	y�(P���A�*;


total_loss�w�@

error_R��S?

learning_rate_1�9�5�SI       6%�	x)P���A�*;


total_lossn�@

error_R��O?

learning_rate_1�9U���I       6%�	�[)P���A�*;


total_loss�s�@

error_R�O?

learning_rate_1�9����I       6%�	�)P���A�*;


total_lossʘ@

error_R��Z?

learning_rate_1�9%�CuI       6%�	�
*P���A�*;


total_loss$st@

error_R{cR?

learning_rate_1�9I29�I       6%�	�Q*P���A�*;


total_loss���@

error_R�$L?

learning_rate_1�9b@ �I       6%�	^�*P���A�*;


total_loss�,�@

error_R�J?

learning_rate_1�9B��I       6%�	T�*P���A�*;


total_loss4
�@

error_RҮM?

learning_rate_1�9_�m�I       6%�	�4+P���A�*;


total_loss�}A

error_R)�C?

learning_rate_1�9y-��I       6%�	��+P���A�*;


total_loss#��@

error_RIg\?

learning_rate_1�9�8��I       6%�	��+P���A�*;


total_loss:JA

error_RJ�E?

learning_rate_1�9JB��I       6%�	lO,P���A�*;


total_lossMQA

error_R�_?

learning_rate_1�9�0x�I       6%�	C�,P���A�*;


total_loss��@

error_RO>?

learning_rate_1�9��ւI       6%�	H�,P���A�*;


total_loss��@

error_R6�??

learning_rate_1�9�}I       6%�	�--P���A�*;


total_loss���@

error_R\1X?

learning_rate_1�9���I       6%�	=v-P���A�*;


total_loss��@

error_R��]?

learning_rate_1�9�D�3I       6%�	��-P���A�*;


total_loss�@

error_RvR?

learning_rate_1�9%��I       6%�	�.P���A�*;


total_loss$A�@

error_R�I6?

learning_rate_1�9w�|I       6%�	M.P���A�*;


total_loss���@

error_RA�O?

learning_rate_1�91��I       6%�	�.P���A�*;


total_loss<B�@

error_R�:Q?

learning_rate_1�9]��I       6%�	��.P���A�*;


total_loss�n�@

error_Rf�B?

learning_rate_1�9q��)I       6%�	�/P���A�*;


total_loss��A

error_R7)L?

learning_rate_1�9�HԟI       6%�	�g/P���A�*;


total_loss,��@

error_R\8?

learning_rate_1�9�kk�I       6%�		�/P���A�*;


total_loss-]�@

error_RU?

learning_rate_1�9�.�I       6%�	I0P���A�*;


total_lossTN�@

error_R�5L?

learning_rate_1�9#��OI       6%�	�N0P���A�*;


total_loss��@

error_R�P?

learning_rate_1�9�,�I       6%�	�0P���A�*;


total_lossȠA

error_Rd4S?

learning_rate_1�92�a�I       6%�	�0P���A�*;


total_loss��A

error_RO�T?

learning_rate_1�9�.�SI       6%�	� 1P���A�*;


total_loss~�@

error_R)�F?

learning_rate_1�9���I       6%�	h1P���A�*;


total_loss7;�@

error_R��V?

learning_rate_1�94CB�I       6%�	�1P���A�*;


total_loss���@

error_R,�L?

learning_rate_1�9�b��I       6%�		�1P���A�*;


total_loss9�@

error_R�%G?

learning_rate_1�9����I       6%�	12P���A�*;


total_loss�^�@

error_R�`d?

learning_rate_1�9|袇I       6%�	Ls2P���A�*;


total_lossa�@

error_Rx[?

learning_rate_1�9UzH�I       6%�	/�2P���A�*;


total_loss�M�@

error_R�>W?

learning_rate_1�9�7��I       6%�	�2P���A�*;


total_loss�;�@

error_R��X?

learning_rate_1�9��1!I       6%�	�>3P���A�*;


total_loss�M�@

error_R��W?

learning_rate_1�9�(�?I       6%�	��3P���A�*;


total_loss	��@

error_R��O?

learning_rate_1�9�m��I       6%�	[�3P���A�*;


total_loss���@

error_Re]O?

learning_rate_1�9�E�TI       6%�	z4P���A�*;


total_loss��@

error_R��D?

learning_rate_1�9y��qI       6%�	;\4P���A�*;


total_losss6�@

error_R��e?

learning_rate_1�9B�1I       6%�	ȡ4P���A�*;


total_loss�s�@

error_RfBK?

learning_rate_1�95��I       6%�	W�4P���A�*;


total_loss2��@

error_R�TI?

learning_rate_1�9T�AI       6%�	9+5P���A�*;


total_loss�O�@

error_R��D?

learning_rate_1�9���I       6%�	2o5P���A�*;


total_loss�(A

error_R��W?

learning_rate_1�9��{I       6%�	��5P���A�*;


total_lossRD�@

error_R
M?

learning_rate_1�9�cI       6%�	��5P���A�*;


total_loss��A

error_R��=?

learning_rate_1�9���{I       6%�	X:6P���A�*;


total_loss��@

error_R�H?

learning_rate_1�9�m��I       6%�	�~6P���A�*;


total_loss��@

error_Ra�T?

learning_rate_1�9���lI       6%�	�6P���A�*;


total_lossAg�@

error_RZoO?

learning_rate_1�9�7DZI       6%�	7P���A�*;


total_lossQ��@

error_R��I?

learning_rate_1�9�ENyI       6%�	xI7P���A�*;


total_loss2��@

error_R�hO?

learning_rate_1�9Q���I       6%�	E�7P���A�*;


total_loss�Wl@

error_R(b?

learning_rate_1�9y��I       6%�	G�7P���A�*;


total_loss�+�@

error_R�wV?

learning_rate_1�9h���I       6%�	�8P���A�*;


total_loss�y�@

error_R��G?

learning_rate_1�9���-I       6%�	d`8P���A�*;


total_loss߯�@

error_R��A?

learning_rate_1�9���jI       6%�	٪8P���A�*;


total_loss��q@

error_R�L?

learning_rate_1�9Z��I       6%�	6�8P���A�*;


total_loss&q�@

error_R�ZJ?

learning_rate_1�9ĩM	I       6%�	849P���A�*;


total_loss�B�@

error_R&M?

learning_rate_1�9�&<OI       6%�	|9P���A�*;


total_loss�3�@

error_R��]?

learning_rate_1�9WJm1I       6%�	I�9P���A�*;


total_loss�e�@

error_Rf�V?

learning_rate_1�9�[�I       6%�	W>:P���A�*;


total_lossSA

error_R�\?

learning_rate_1�9�r�I       6%�	؅:P���A�*;


total_loss�@

error_R�2F?

learning_rate_1�9�r_�I       6%�	}�:P���A�*;


total_loss<j�@

error_R-�P?

learning_rate_1�9�c�TI       6%�	X;P���A�*;


total_lossA��@

error_R�oS?

learning_rate_1�9�U�6I       6%�	:[;P���A�*;


total_loss��A

error_R�^?

learning_rate_1�9�;��I       6%�	'�;P���A�*;


total_loss	r�@

error_R�gH?

learning_rate_1�9L"��I       6%�	��;P���A�*;


total_loss�'�@

error_R��_?

learning_rate_1�9�T(�I       6%�	�2<P���A�*;


total_loss?3�@

error_R��M?

learning_rate_1�9G�bI       6%�	�w<P���A�*;


total_loss���@

error_R�$M?

learning_rate_1�9y�I       6%�	8�<P���A�*;


total_lossͼ�@

error_R�PF?

learning_rate_1�9Q[`)I       6%�	��<P���A�*;


total_loss1��@

error_R6`?

learning_rate_1�9���I       6%�		G=P���A�*;


total_losss&�@

error_R��T?

learning_rate_1�9�E�HI       6%�	e�=P���A�*;


total_loss��@

error_R�6K?

learning_rate_1�9��>I       6%�	W�=P���A�*;


total_loss�Ʌ@

error_R�KR?

learning_rate_1�9{�I       6%�	+>P���A�*;


total_loss;��@

error_R�:Z?

learning_rate_1�9V���I       6%�	�a>P���A�*;


total_loss
�@

error_R��R?

learning_rate_1�9{�JUI       6%�	��>P���A�*;


total_loss�x�@

error_R��F?

learning_rate_1�9�ܐI       6%�	��>P���A�*;


total_loss�l�@

error_R�;N?

learning_rate_1�9J��I       6%�	�5?P���A�*;


total_loss�l�@

error_R�\?

learning_rate_1�9H��zI       6%�	�}?P���A�*;


total_lossjdA

error_Rs�H?

learning_rate_1�9��I       6%�	1�?P���A�*;


total_loss8�@

error_R(sM?

learning_rate_1�9Pր�I       6%�	M@P���A�*;


total_loss�w�@

error_R�X?

learning_rate_1�9S�ϞI       6%�	�Y@P���A�*;


total_loss.D�@

error_R[#Z?

learning_rate_1�9"Ph�I       6%�	X�@P���A�*;


total_loss��@

error_R�?K?

learning_rate_1�9e��I       6%�	��@P���A�*;


total_loss�}�@

error_R).K?

learning_rate_1�9F�I       6%�	�/AP���A�*;


total_loss���@

error_RRwF?

learning_rate_1�9�޷�I       6%�	�rAP���A�*;


total_loss@

error_RJJ?

learning_rate_1�9�9cI       6%�	��AP���A�*;


total_loss�h�@

error_R��O?

learning_rate_1�9��B�I       6%�	��AP���A�*;


total_lossT�@

error_R#W?

learning_rate_1�9	6��I       6%�	�>BP���A�*;


total_lossv��@

error_RWQ?

learning_rate_1�9��2I       6%�	�BP���A�*;


total_lossF	�@

error_R�MW?

learning_rate_1�9�
�I       6%�	��BP���A�*;


total_loss���@

error_RwQ?

learning_rate_1�9��3+I       6%�	vCP���A�*;


total_loss��@

error_Rq�\?

learning_rate_1�9�p�I       6%�	+QCP���A�*;


total_loss8M�@

error_R!X?

learning_rate_1�90�I       6%�	�CP���A�*;


total_lossx;�@

error_R|2Q?

learning_rate_1�9}�I       6%�	I�CP���A�*;


total_loss��	A

error_RO??

learning_rate_1�9�?��I       6%�	X(DP���A�*;


total_loss��)A

error_R�wM?

learning_rate_1�9	�SI       6%�	?tDP���A�*;


total_loss3�|@

error_R�S?

learning_rate_1�9F�ĕI       6%�	��DP���A�*;


total_loss���@

error_R��P?

learning_rate_1�9�y�bI       6%�	�EP���A�*;


total_loss`�@

error_Rl�J?

learning_rate_1�9j�v�I       6%�	�SEP���A�*;


total_loss=x�@

error_R�I?

learning_rate_1�9�ڴ4I       6%�	5�EP���A�*;


total_loss.��@

error_R��E?

learning_rate_1�9��zI       6%�	 �EP���A�*;


total_loss֘�@

error_R�G?

learning_rate_1�9`�5�I       6%�	�#FP���A�*;


total_loss��@

error_R[�W?

learning_rate_1�9�-�I       6%�	�eFP���A�*;


total_loss�~�@

error_R�2F?

learning_rate_1�9��3�I       6%�	�FP���A�*;


total_loss�HA

error_Rv�V?

learning_rate_1�9��|)I       6%�	H�FP���A�*;


total_lossѷ�@

error_R�I?

learning_rate_1�9�R;>I       6%�	h4GP���A�*;


total_lossf&�@

error_R��Y?

learning_rate_1�9�w��I       6%�	
|GP���A�*;


total_loss���@

error_R��J?

learning_rate_1�9��I       6%�	)�GP���A�*;


total_loss���@

error_R�i?

learning_rate_1�9�1��I       6%�	�HP���A�*;


total_lossJ�*A

error_R�G?

learning_rate_1�9KNp�I       6%�	GHP���A�*;


total_loss�[�@

error_R�2O?

learning_rate_1�9�m�#I       6%�	A�HP���A�*;


total_lossqI@

error_R�T?

learning_rate_1�9���I       6%�	�HP���A�*;


total_loss���@

error_RR�V?

learning_rate_1�9bL�8I       6%�	�IP���A�*;


total_loss0�A

error_R��G?

learning_rate_1�9��P/I       6%�	�VIP���A�*;


total_loss���@

error_R�Z?

learning_rate_1�9�/��I       6%�	�IP���A�*;


total_loss:�@

error_R��M?

learning_rate_1�9�p��I       6%�	�JP���A�*;


total_loss���@

error_R1�K?

learning_rate_1�9f�ҏI       6%�	�QJP���A�*;


total_lossS2�@

error_R�f<?

learning_rate_1�9A&,I       6%�	�JP���A�*;


total_loss�w�@

error_R�0U?

learning_rate_1�9u�߉I       6%�	@�JP���A�*;


total_lossW��@

error_R�%R?

learning_rate_1�9|ֽ�I       6%�	�5KP���A�*;


total_loss׭�@

error_R:�J?

learning_rate_1�9_W�3I       6%�	ՁKP���A�*;


total_loss��@

error_R� `?

learning_rate_1�91a�;I       6%�	v�KP���A�*;


total_lossT"A

error_R�T?

learning_rate_1�9�iFI       6%�	!,LP���A�*;


total_loss�K�@

error_R�LK?

learning_rate_1�9���I       6%�	�qLP���A�*;


total_loss�@

error_R�S?

learning_rate_1�9�|�I       6%�	+�LP���A�*;


total_loss���@

error_RlQ?

learning_rate_1�9����I       6%�	�MP���A�*;


total_loss]�A

error_R�ig?

learning_rate_1�91�|	I       6%�	"[MP���A�*;


total_loss��@

error_R� 6?

learning_rate_1�9�O
tI       6%�	9�MP���A�*;


total_loss���@

error_R�/Q?

learning_rate_1�9Q��
I       6%�	X�MP���A�*;


total_loss��}@

error_R�@?

learning_rate_1�9'v��I       6%�	/NP���A�*;


total_loss���@

error_R�.b?

learning_rate_1�9I��I       6%�	�uNP���A�*;


total_loss7��@

error_RsxI?

learning_rate_1�9�~�
I       6%�	��NP���A�*;


total_loss��~@

error_R*l`?

learning_rate_1�9��aI       6%�	fOP���A�*;


total_losslA

error_R1fV?

learning_rate_1�9�y�I       6%�	�KOP���A�*;


total_loss鱗@

error_R��J?

learning_rate_1�9�-��I       6%�	��OP���A�*;


total_lossT9�@

error_R�~@?

learning_rate_1�9��{I       6%�	��OP���A�*;


total_loss�һ@

error_R6�[?

learning_rate_1�9\�mI       6%�	�!PP���A�*;


total_loss���@

error_R�~S?

learning_rate_1�9tw��I       6%�	�iPP���A�*;


total_lossMUA

error_Rm,T?

learning_rate_1�9�#I       6%�	ƳPP���A�*;


total_loss���@

error_R��:?

learning_rate_1�99�I       6%�	��PP���A�*;


total_loss��@

error_R�V?

learning_rate_1�9EA�7I       6%�	e@QP���A�*;


total_lossN`�@

error_RE�Y?

learning_rate_1�9���I       6%�	��QP���A�*;


total_loss���@

error_R8<K?

learning_rate_1�9�� �I       6%�	��QP���A�*;


total_loss.�#A

error_R�DK?

learning_rate_1�9���I       6%�	MRP���A�*;


total_loss,&w@

error_R/�o?

learning_rate_1�9]
/I       6%�	(_RP���A�*;


total_loss���@

error_R UI?

learning_rate_1�9���I       6%�	��RP���A�*;


total_loss�κ@

error_R`�U?

learning_rate_1�9�4kNI       6%�	�RP���A�*;


total_loss��@

error_R��Z?

learning_rate_1�9�I       6%�	�:SP���A�*;


total_loss�i�@

error_R�_?

learning_rate_1�9 ��I       6%�	�SP���A�*;


total_lossi_@

error_RIA?

learning_rate_1�9fQ��I       6%�	��SP���A�*;


total_loss!q�@

error_RR�F?

learning_rate_1�9�.f�I       6%�	�TP���A�*;


total_lossR߿@

error_R�E?

learning_rate_1�9�.VI       6%�	?LTP���A�*;


total_loss���@

error_R}iS?

learning_rate_1�9�1N�I       6%�	��TP���A�*;


total_loss�z@

error_R��I?

learning_rate_1�9�Q�I       6%�	��TP���A�*;


total_loss;��@

error_R�<S?

learning_rate_1�9�I       6%�	$UP���A�*;


total_loss3�@

error_R��L?

learning_rate_1�9(���I       6%�	�lUP���A�*;


total_loss�*�@

error_R��U?

learning_rate_1�9��1I       6%�	��UP���A�*;


total_lossqٴ@

error_R�lY?

learning_rate_1�9�n��I       6%�	��UP���A�*;


total_loss?��@

error_R��L?

learning_rate_1�9�2�oI       6%�	5=VP���A�*;


total_lossqx�@

error_R��<?

learning_rate_1�9��rI       6%�	f�VP���A�*;


total_loss�D�@

error_R<�Y?

learning_rate_1�9t�ŊI       6%�	-�VP���A�*;


total_lossҜA

error_R�mP?

learning_rate_1�9`�I       6%�	WP���A�*;


total_loss��A

error_Rj�W?

learning_rate_1�9(�$�I       6%�	CVWP���A�*;


total_loss�@

error_R87P?

learning_rate_1�9-��I       6%�	�WP���A�*;


total_lossi=A

error_R[bX?

learning_rate_1�9�8�KI       6%�	S�WP���A�*;


total_loss��A

error_R�0U?

learning_rate_1�9��F�I       6%�	�0XP���A�*;


total_loss[@

error_Rl�f?

learning_rate_1�9Rh��I       6%�	ctXP���A�*;


total_loss���@

error_R��>?

learning_rate_1�9���I       6%�	��XP���A�*;


total_loss�˨@

error_R��O?

learning_rate_1�9��lI       6%�	/�XP���A�*;


total_lossz��@

error_RH�G?

learning_rate_1�9Dc�	I       6%�	�>YP���A�*;


total_loss��@

error_R�rY?

learning_rate_1�9��dzI       6%�	$�YP���A�*;


total_loss��@

error_R6�Q?

learning_rate_1�9pVKI       6%�	��YP���A�*;


total_loss�x�@

error_R��W?

learning_rate_1�9�#�I       6%�	�=ZP���A�*;


total_loss�<�@

error_R��H?

learning_rate_1�9�hfeI       6%�	D�ZP���A�*;


total_loss���@

error_R�[?

learning_rate_1�9�!�I       6%�	��ZP���A�*;


total_lossjF	A

error_R��Y?

learning_rate_1�9�I�I       6%�	�[P���A�*;


total_loss��@

error_R�F?

learning_rate_1�9P7>lI       6%�	�U[P���A�*;


total_loss�5�@

error_Rsa?

learning_rate_1�9�b�I       6%�	��[P���A�*;


total_loss��A

error_R_�M?

learning_rate_1�9�6>�I       6%�	d�[P���A�*;


total_loss�ڣ@

error_R!en?

learning_rate_1�9����I       6%�	J+\P���A�*;


total_loss� c@

error_R��9?

learning_rate_1�9;"��I       6%�	u\P���A�*;


total_loss�9�@

error_R��:?

learning_rate_1�9b�I       6%�	��\P���A�*;


total_lossʊ A

error_RM�S?

learning_rate_1�9�6:�I       6%�	�]P���A�*;


total_loss֑A

error_R�F?

learning_rate_1�9u]OI       6%�	xN]P���A�*;


total_loss\H�@

error_R��N?

learning_rate_1�9�@[-I       6%�	�]P���A�*;


total_loss �@

error_RȨ\?

learning_rate_1�92ۄ�I       6%�	g�]P���A�*;


total_loss�U�@

error_R�xS?

learning_rate_1�9��I       6%�	#4^P���A�*;


total_loss���@

error_R�S?

learning_rate_1�9�t�|I       6%�	�y^P���A�*;


total_loss	]�@

error_R�eZ?

learning_rate_1�9���I       6%�	a�^P���A�*;


total_loss?0�@

error_RNrR?

learning_rate_1�9ۓ#|I       6%�	o_P���A�*;


total_lossNA

error_R�	M?

learning_rate_1�9\Q�I       6%�	�Q_P���A�*;


total_loss&C�@

error_R@cM?

learning_rate_1�9�\�qI       6%�	�_P���A�*;


total_lossoR�@

error_RdL?

learning_rate_1�9�bh3I       6%�	��_P���A�*;


total_loss�Ȝ@

error_R=�G?

learning_rate_1�9��EI       6%�	.`P���A�*;


total_loss	��@

error_RC�e?

learning_rate_1�9��C�I       6%�	jt`P���A�*;


total_loss
�@

error_R^M?

learning_rate_1�9���I       6%�	��`P���A�*;


total_loss��@

error_R�;Y?

learning_rate_1�9���TI       6%�	�aP���A�*;


total_loss7��@

error_R
PY?

learning_rate_1�9���I       6%�	�MaP���A�*;


total_loss*lA

error_R�P?

learning_rate_1�9�Y�NI       6%�	>�aP���A�*;


total_loss��i@

error_R6?

learning_rate_1�9���I       6%�	s�aP���A�*;


total_loss��@

error_Rq�]?

learning_rate_1�9r�XI       6%�	@bP���A�*;


total_lossd��@

error_R)xO?

learning_rate_1�9�5��I       6%�	ebP���A�*;


total_loss�M�@

error_R��@?

learning_rate_1�9����I       6%�	۬bP���A�*;


total_loss	ʳ@

error_R1�Q?

learning_rate_1�9Q�II       6%�	��bP���A�*;


total_loss�&�@

error_R&�G?

learning_rate_1�9=~
I       6%�	�7cP���A�*;


total_loss�K�@

error_R�1P?

learning_rate_1�9�
�I       6%�	�|cP���A�*;


total_lossE��@

error_R�|=?

learning_rate_1�92	zI       6%�	%�cP���A�*;


total_loss�h�@

error_R��Q?

learning_rate_1�9� ��I       6%�	bdP���A�*;


total_loss͒�@

error_R�N?

learning_rate_1�9|o�I       6%�	�DdP���A�*;


total_loss���@

error_R�,F?

learning_rate_1�9�	X�I       6%�	��dP���A�*;


total_lossV�@

error_R[�Z?

learning_rate_1�9{�/�I       6%�	x�dP���A�*;


total_lossႋ@

error_RUB?

learning_rate_1�9���DI       6%�	�eP���A�*;


total_lossqU�@

error_R�B?

learning_rate_1�9 E�I       6%�	_eP���A�*;


total_loss2>�@

error_R�A?

learning_rate_1�9'
�I       6%�	:�eP���A�*;


total_loss��@

error_R�?+?

learning_rate_1�9���yI       6%�	a�eP���A�*;


total_lossH��@

error_R@S?

learning_rate_1�9�R�I       6%�	1fP���A�*;


total_lossa@�@

error_R�:]?

learning_rate_1�9X I       6%�	tfP���A�*;


total_loss���@

error_R�4O?

learning_rate_1�9[u�WI       6%�	��fP���A�*;


total_loss�yA

error_R@]6?

learning_rate_1�9B��}I       6%�	$gP���A�*;


total_lossh��@

error_Rr_P?

learning_rate_1�9��D�I       6%�	RgP���A�*;


total_loss/v�@

error_R��H?

learning_rate_1�9�;iI       6%�	�gP���A�*;


total_loss�A

error_R:�[?

learning_rate_1�9T�I       6%�	'�gP���A�*;


total_losscń@

error_RT@?

learning_rate_1�9V��~I       6%�	/&hP���A�*;


total_loss�A

error_Rn`?

learning_rate_1�9茗I       6%�	�khP���A�*;


total_loss�A

error_R�o^?

learning_rate_1�9=m�I       6%�	��hP���A�*;


total_loss�۷@

error_R�'N?

learning_rate_1�9�v7:I       6%�	��hP���A�*;


total_loss,��@

error_R�FP?

learning_rate_1�9?7^%I       6%�	�5iP���A�*;


total_losse�A

error_R��N?

learning_rate_1�9��1fI       6%�		�iP���A�*;


total_loss��@

error_R�?P?

learning_rate_1�9� "�I       6%�	&�iP���A�*;


total_lossF��@

error_R.N?

learning_rate_1�9'w�<I       6%�	!6jP���A�*;


total_loss��z@

error_R�7?

learning_rate_1�9G���I       6%�	�~jP���A�*;


total_loss(��@

error_R��]?

learning_rate_1�9Id	�I       6%�	��jP���A�*;


total_lossn0�@

error_R��^?

learning_rate_1�9}ҿ]I       6%�	kP���A�*;


total_lossNN�@

error_R�T?

learning_rate_1�9�I       6%�	1ikP���A�*;


total_loss��@

error_R�xK?

learning_rate_1�9g`�kI       6%�	��kP���A�*;


total_lossf��@

error_R�fg?

learning_rate_1�9�L�TI       6%�	�lP���A�*;


total_loss��@

error_R�Z[?

learning_rate_1�9��*�I       6%�	�TlP���A�*;


total_loss�#�@

error_R
�c?

learning_rate_1�9��I       6%�	7�lP���A�*;


total_loss{�@

error_Rv`N?

learning_rate_1�9rw�I       6%�	 mP���A�*;


total_lossIZ�@

error_R��^?

learning_rate_1�9ˏI       6%�	�OmP���A�*;


total_lossƨ�@

error_R{�X?

learning_rate_1�9���I       6%�	�mP���A�*;


total_loss���@

error_R�jJ?

learning_rate_1�96X�*I       6%�	��mP���A�*;


total_loss��A

error_RE�_?

learning_rate_1�9�y`JI       6%�	�7nP���A�*;


total_loss(<�@

error_R��E?

learning_rate_1�9���/I       6%�	��nP���A�*;


total_loss��@

error_ROiV?

learning_rate_1�9� �I       6%�	��nP���A�*;


total_loss�@

error_R<cH?

learning_rate_1�9a2h�I       6%�	oP���A�*;


total_lossE�@

error_RM�B?

learning_rate_1�9��[I       6%�	�]oP���A�*;


total_loss�S�@

error_R��R?

learning_rate_1�9���I       6%�	b�oP���A�*;


total_loss���@

error_R?�e?

learning_rate_1�9�`I       6%�	>�oP���A�*;


total_lossڧ�@

error_R��<?

learning_rate_1�91��`I       6%�	]ApP���A�*;


total_loss��@

error_R�zM?

learning_rate_1�9�q�I       6%�	��pP���A�*;


total_loss�4�@

error_R�d?

learning_rate_1�9���I       6%�	��pP���A�*;


total_loss�ͼ@

error_R��O?

learning_rate_1�9�=�I       6%�	}qP���A�*;


total_loss�K�@

error_RA�X?

learning_rate_1�9|qoI       6%�	SYqP���A�*;


total_lossk�@

error_R��N?

learning_rate_1�9���I       6%�	��qP���A�*;


total_loss�7�@

error_R{ b?

learning_rate_1�9�nd;I       6%�	�qP���A�*;


total_loss���@

error_R��B?

learning_rate_1�9
�I       6%�	�4rP���A�*;


total_loss�BA

error_R��]?

learning_rate_1�9�#�I       6%�	�rP���A�*;


total_loss�8�@

error_R�U?

learning_rate_1�9�;7_I       6%�	��rP���A�*;


total_loss�@

error_R��]?

learning_rate_1�9�УI       6%�	�sP���A�*;


total_loss<��@

error_R��K?

learning_rate_1�9��7-I       6%�	�VsP���A�*;


total_loss��-A

error_Rq�V?

learning_rate_1�9[PvI       6%�	d�sP���A�*;


total_loss���@

error_R14f?

learning_rate_1�9��
XI       6%�	*�sP���A�*;


total_loss�@

error_RdG?

learning_rate_1�9�d��I       6%�	�/tP���A�*;


total_loss���@

error_R�nR?

learning_rate_1�9Ǧ"�I       6%�	U|tP���A�*;


total_lossW�@

error_RM�R?

learning_rate_1�9�^>I       6%�	D�tP���A�*;


total_loss�D�@

error_R.�B?

learning_rate_1�9��9dI       6%�	�uP���A�*;


total_loss�՚@

error_R8�E?

learning_rate_1�9��FI       6%�	4HuP���A�*;


total_lossqx�@

error_Ro_?

learning_rate_1�9�U�I       6%�	�uP���A�*;


total_loss˄�@

error_R��M?

learning_rate_1�9��M�I       6%�	a�uP���A�*;


total_lossh8@

error_R�w<?

learning_rate_1�9Ύ"I       6%�	xvP���A�*;


total_losso�@

error_R�b?

learning_rate_1�9m��oI       6%�	VvP���A�*;


total_loss��@

error_R.�G?

learning_rate_1�9�.��I       6%�	ٛvP���A�*;


total_loss��@

error_R.�I?

learning_rate_1�9ʴI       6%�	a�vP���A�*;


total_lossdA

error_R��R?

learning_rate_1�9�2ڣI       6%�	s%wP���A�*;


total_loss\��@

error_R�gL?

learning_rate_1�9Ax�SI       6%�	{hwP���A�*;


total_loss���@

error_Rn�J?

learning_rate_1�9z I       6%�	�wP���A�*;


total_loss���@

error_R*ub?

learning_rate_1�9ps��I       6%�	M�wP���A�*;


total_losss��@

error_R�JI?

learning_rate_1�9�j�I       6%�	�5xP���A�*;


total_loss}�@

error_RJI?

learning_rate_1�9ҭ�I       6%�	yxP���A�*;


total_loss��@

error_R�GL?

learning_rate_1�9g �3I       6%�	��xP���A�*;


total_loss;�@

error_R��D?

learning_rate_1�9��XI       6%�	�yP���A�*;


total_loss�[�@

error_R[�R?

learning_rate_1�9�� I       6%�	�GyP���A�*;


total_loss2iA

error_RJ�D?

learning_rate_1�9.X�XI       6%�	�yP���A�*;


total_lossN-z@

error_Rn�N?

learning_rate_1�9���9I       6%�	��yP���A�*;


total_loss4�@

error_R��Q?

learning_rate_1�9	IZlI       6%�	g7zP���A�*;


total_lossA

error_R�OF?

learning_rate_1�9�ڜI       6%�	�~zP���A�*;


total_lossR��@

error_R�K?

learning_rate_1�9�:�I       6%�	7�zP���A�*;


total_lossv#A

error_R�P?

learning_rate_1�9n��I       6%�	�{P���A�*;


total_loss�t�@

error_RT7M?

learning_rate_1�9�g5I       6%�	*N{P���A�*;


total_lossr�d@

error_RM?

learning_rate_1�9F�,fI       6%�	��{P���A�*;


total_loss��f@

error_R�<?

learning_rate_1�9��cJI       6%�	��{P���A�*;


total_loss��@

error_R��W?

learning_rate_1�9�O�/I       6%�	|P���A�*;


total_lossܑ@

error_RZpV?

learning_rate_1�9����I       6%�	�^|P���A�*;


total_loss4�@

error_R$�Y?

learning_rate_1�9�G�I       6%�	�|P���A�*;


total_loss�ܧ@

error_RT�Z?

learning_rate_1�9��ϝI       6%�	��|P���A�*;


total_loss��@

error_R�bV?

learning_rate_1�9}�I       6%�	�1}P���A�*;


total_loss�
�@

error_RJF?

learning_rate_1�9!��I       6%�	c~}P���A�*;


total_loss3�@

error_R��J?

learning_rate_1�9w�T�I       6%�	��}P���A�*;


total_lossj��@

error_R�`;?

learning_rate_1�9r��I       6%�	�~P���A�*;


total_loss (�@

error_R�nQ?

learning_rate_1�9�kuI       6%�	vY~P���A�*;


total_loss)�@

error_R6 V?

learning_rate_1�9WSP�I       6%�	��~P���A�*;


total_loss'tA

error_R,�D?

learning_rate_1�9�
�OI       6%�	3�~P���A�*;


total_lossŹ�@

error_R&4G?

learning_rate_1�9�uuI       6%�	�'P���A�*;


total_loss�ܝ@

error_R��]?

learning_rate_1�9�_I       6%�	iP���A�*;


total_losst��@

error_R�B?

learning_rate_1�9Ͼ�I       6%�	v�P���A�*;


total_loss_4�@

error_R��J?

learning_rate_1�9���I       6%�	��P���A�*;


total_loss��@

error_RE�b?

learning_rate_1�9,��I       6%�	�;�P���A�*;


total_loss��A

error_R�B[?

learning_rate_1�9��cI       6%�	ׇ�P���A�*;


total_lossR��@

error_R��G?

learning_rate_1�9��iI       6%�	$̀P���A�*;


total_losstL�@

error_Rx�J?

learning_rate_1�9����I       6%�	{�P���A�*;


total_loss�	�@

error_R��O?

learning_rate_1�9�(;I       6%�	�^�P���A�*;


total_lossD`�@

error_R�R?

learning_rate_1�9(H�I       6%�	���P���A�*;


total_loss�?�@

error_RifR?

learning_rate_1�9�֜�I       6%�	��P���A�*;


total_loss��}@

error_R�\?

learning_rate_1�9\��I       6%�	�.�P���A�*;


total_lossF��@

error_R�]a?

learning_rate_1�9��3I       6%�	�o�P���A�*;


total_loss=�A

error_R��P?

learning_rate_1�9Ԝ�I       6%�	���P���A�*;


total_loss���@

error_R��O?

learning_rate_1�9�:�I       6%�	���P���A�*;


total_loss	��@

error_R�X?

learning_rate_1�9%*kcI       6%�	�;�P���A�*;


total_loss;��@

error_R�S?

learning_rate_1�9d���I       6%�	�}�P���A�*;


total_loss��@

error_R(�F?

learning_rate_1�9�=��I       6%�	.ÃP���A�*;


total_loss�
�@

error_R��_?

learning_rate_1�9��A'I       6%�	=�P���A�*;


total_lossڌ�@

error_Rt�W?

learning_rate_1�9pԚI       6%�	�L�P���A�*;


total_lossH�@

error_R�R?

learning_rate_1�9�3R�I       6%�	���P���A�*;


total_loss��@

error_RͼY?

learning_rate_1�9�%�I       6%�	KӄP���A�*;


total_loss��@

error_R*�;?

learning_rate_1�9�a�I       6%�	k�P���A�*;


total_loss�&�@

error_R�U=?

learning_rate_1�9�}{�I       6%�	�\�P���A�*;


total_lossl��@

error_Rw�@?

learning_rate_1�9_2g�I       6%�	N��P���A�*;


total_loss��@

error_ReZX?

learning_rate_1�9�1�UI       6%�	��P���A�*;


total_loss��i@

error_R�B?

learning_rate_1�9n�.�I       6%�	!0�P���A�*;


total_lossD�@

error_RRgX?

learning_rate_1�9,fPHI       6%�	�y�P���A�*;


total_loss�R�@

error_R
�G?

learning_rate_1�93���I       6%�	J��P���A�*;


total_loss@f�@

error_RO�@?

learning_rate_1�9�D.I       6%�	��P���A�*;


total_losss��@

error_R6�\?

learning_rate_1�9���I       6%�	BT�P���A�*;


total_lossE,�@

error_RM�K?

learning_rate_1�9嘷&I       6%�	럇P���A�*;


total_loss�,�@

error_R�W?

learning_rate_1�9Y-�qI       6%�	E�P���A�*;


total_loss.��@

error_RC�^?

learning_rate_1�9
�
I       6%�	�:�P���A�*;


total_loss<-�@

error_REMe?

learning_rate_1�9/�qNI       6%�	[��P���A�*;


total_loss}j�@

error_RWT?

learning_rate_1�9��fwI       6%�	�ʈP���A�*;


total_loss7{�@

error_R$CK?

learning_rate_1�9O��I       6%�	9�P���A�*;


total_loss��A

error_R

J?

learning_rate_1�9���:I       6%�	�^�P���A�*;


total_loss��@

error_R�L?

learning_rate_1�9����I       6%�	eɉP���A�*;


total_loss���@

error_R�?R?

learning_rate_1�9�z@I       6%�	��P���A�*;


total_lossl��@

error_RZ�b?

learning_rate_1�9;��I       6%�	�]�P���A�*;


total_loss���@

error_R�wQ?

learning_rate_1�9iC'I       6%�	Ҧ�P���A�*;


total_lossR'�@

error_R��A?

learning_rate_1�9C���I       6%�	��P���A�*;


total_loss4�t@

error_R�R?

learning_rate_1�9�hC#I       6%�	�3�P���A�*;


total_lossW>A

error_R��N?

learning_rate_1�9��I       6%�	j��P���A�*;


total_lossH�@

error_R�$Z?

learning_rate_1�9,esoI       6%�	��P���A�*;


total_loss-��@

error_Rv�\?

learning_rate_1�9k�=I       6%�	t)�P���A�*;


total_loss&H A

error_R�4E?

learning_rate_1�9�-�cI       6%�	�m�P���A�*;


total_loss4�@

error_RZS?

learning_rate_1�9e��dI       6%�	�ŌP���A�*;


total_loss�`�@

error_R�vH?

learning_rate_1�9R;,�I       6%�	G�P���A�*;


total_loss���@

error_R��??

learning_rate_1�9��5xI       6%�	{Y�P���A�*;


total_lossj`�@

error_Ri�3?

learning_rate_1�9�J�CI       6%�	蟍P���A�*;


total_loss�F�@

error_R�O?

learning_rate_1�9��.%I       6%�	/�P���A�*;


total_loss�+�@

error_R~a?

learning_rate_1�9/չI       6%�	�2�P���A�*;


total_loss8f�@

error_R�0E?

learning_rate_1�9���cI       6%�	�x�P���A�*;


total_loss7�@

error_R;�A?

learning_rate_1�9��EI       6%�	-��P���A�*;


total_loss
��@

error_R��M?

learning_rate_1�9�S#xI       6%�	��P���A�*;


total_loss�*�@

error_R�]?

learning_rate_1�9jnC�I       6%�	M�P���A�*;


total_loss�r�@

error_Rx%Y?

learning_rate_1�9)�UhI       6%�	Q��P���A�*;


total_loss���@

error_R�`K?

learning_rate_1�9��I       6%�	�֏P���A�*;


total_loss�@

error_R�O?

learning_rate_1�9�e��I       6%�	��P���A�*;


total_lossn��@

error_R�@?

learning_rate_1�9$��I       6%�	f�P���A�*;


total_loss%�@

error_R&a?

learning_rate_1�9K4SI       6%�	Ы�P���A�*;


total_loss��@

error_R2eD?

learning_rate_1�9�T��I       6%�	��P���A�*;


total_lossp�@

error_R��<?

learning_rate_1�9hb� I       6%�	84�P���A�*;


total_loss�t�@

error_RNN?

learning_rate_1�9Ȼ�I       6%�	Lz�P���A�*;


total_loss��@

error_R�I?

learning_rate_1�9�q�I       6%�	���P���A�*;


total_lossS+�@

error_R��W?

learning_rate_1�9 v�7I       6%�	��P���A�*;


total_loss��@

error_R�F?

learning_rate_1�9�n
�I       6%�	K�P���A�*;


total_loss� A

error_R��O?

learning_rate_1�9�2�I       6%�	���P���A�*;


total_loss�a2A

error_R�4b?

learning_rate_1�9}��pI       6%�	*ВP���A�*;


total_loss�p�@

error_R��V?

learning_rate_1�9�֑�I       6%�	��P���A�*;


total_lossVz�@

error_R�hU?

learning_rate_1�9^��4I       6%�	?Z�P���A�*;


total_loss�e�@

error_R�G?

learning_rate_1�9Y�iVI       6%�	d��P���A�*;


total_loss=��@

error_Ra�E?

learning_rate_1�9j�o�I       6%�	��P���A�*;


total_lossĐ�@

error_R�yB?

learning_rate_1�9��I       6%�	C0�P���A�*;


total_loss_��@

error_R�!F?

learning_rate_1�9gj�I       6%�	x�P���A�*;


total_lossu��@

error_R��U?

learning_rate_1�9�?�FI       6%�	��P���A�*;


total_loss��@

error_R�B?

learning_rate_1�9���]I       6%�	��P���A�*;


total_loss�K�@

error_R��M?

learning_rate_1�9Cw:}I       6%�	M�P���A�*;


total_loss���@

error_R��A?

learning_rate_1�9�`��I       6%�	<��P���A�*;


total_loss�P�@

error_R��L?

learning_rate_1�9�;kI       6%�	��P���A�*;


total_loss���@

error_R,�e?

learning_rate_1�9O�mI       6%�	*�P���A�*;


total_loss��A

error_R��e?

learning_rate_1�9 �'�I       6%�	=n�P���A�*;


total_lossM�h@

error_R��V?

learning_rate_1�9xAI       6%�	
��P���A�*;


total_lossxx�@

error_RHTF?

learning_rate_1�91�KI       6%�	��P���A�*;


total_loss�=�@

error_R��Z?

learning_rate_1�9���0I       6%�	M;�P���A�*;


total_lossDc�@

error_RE2a?

learning_rate_1�9�SK�I       6%�	,�P���A�*;


total_lossqc�@

error_R� T?

learning_rate_1�9�%� I       6%�	^��P���A�*;


total_loss3�@

error_RO�U?

learning_rate_1�9��I       6%�		�P���A�*;


total_loss��@

error_R�d?

learning_rate_1�9h	�I       6%�	�L�P���A�*;


total_loss��@

error_R�DX?

learning_rate_1�9�y�-I       6%�	��P���A�*;


total_loss3�3A

error_R8CL?

learning_rate_1�9s;��I       6%�	�՘P���A�*;


total_loss���@

error_R��??

learning_rate_1�9��L!I       6%�	� �P���A�*;


total_loss���@

error_Rܺb?

learning_rate_1�9; ��I       6%�	@d�P���A�*;


total_loss���@

error_R�e\?

learning_rate_1�9sX��I       6%�	�ÙP���A�*;


total_loss%S�@

error_R��S?

learning_rate_1�9j�LI       6%�	
�P���A�*;


total_loss�۩@

error_R��X?

learning_rate_1�9�9�SI       6%�	�e�P���A�*;


total_lossh�@

error_R�J?

learning_rate_1�9����I       6%�	嫚P���A�*;


total_loss1L�@

error_R��n?

learning_rate_1�9��q�I       6%�	��P���A�*;


total_losst,�@

error_Rd�D?

learning_rate_1�9���XI       6%�	�6�P���A�*;


total_losso�@

error_R�M?

learning_rate_1�9��#I       6%�	]}�P���A�*;


total_loss2�@

error_R�D?

learning_rate_1�9��?I       6%�	JɛP���A�*;


total_loss�0�@

error_RZ�W?

learning_rate_1�9	4I       6%�	a�P���A�*;


total_loss��@

error_R��C?

learning_rate_1�9e��wI       6%�	�V�P���A�*;


total_loss���@

error_RJ'Q?

learning_rate_1�9�I       6%�	᚜P���A�*;


total_lossO�@

error_R��U?

learning_rate_1�9_#�ZI       6%�	hߜP���A�*;


total_loss� �@

error_RÁ[?

learning_rate_1�9�[I       6%�	�"�P���A�*;


total_loss���@

error_R��I?

learning_rate_1�9�h�I       6%�	i�P���A�*;


total_loss���@

error_R��L?

learning_rate_1�9�]�I       6%�	J��P���A�*;


total_lossH�@

error_RK?

learning_rate_1�9�l��I       6%�	��P���A�*;


total_loss���@

error_R2BL?

learning_rate_1�9&�MI       6%�	2I�P���A�*;


total_lossݩ@

error_R�OS?

learning_rate_1�9}j�I       6%�	���P���A�*;


total_loss�*�@

error_R��E?

learning_rate_1�9�{�I       6%�	OҞP���A�*;


total_loss�u�@

error_R}J?

learning_rate_1�9~�I       6%�	v�P���A�*;


total_lossi��@

error_R�uC?

learning_rate_1�9�	�I       6%�	�[�P���A�*;


total_loss>6�@

error_R��L?

learning_rate_1�9l�WI       6%�	ѡ�P���A�*;


total_loss�ލ@

error_RNA?

learning_rate_1�9�~I       6%�	��P���A�*;


total_loss�c�@

error_Rn�E?

learning_rate_1�9�ſI       6%�	�/�P���A�*;


total_loss���@

error_RnFN?

learning_rate_1�9��ZUI       6%�	�u�P���A�*;


total_loss���@

error_R._T?

learning_rate_1�9���I       6%�	���P���A�*;


total_loss��A

error_ROY?

learning_rate_1�9�ߎ6I       6%�	@��P���A�*;


total_loss>�@

error_R؞U?

learning_rate_1�9U�I       6%�	�@�P���A�*;


total_loss��@

error_R��5?

learning_rate_1�9OĆXI       6%�	r��P���A�*;


total_lossJhA

error_R��W?

learning_rate_1�9�y&�I       6%�	{ɡP���A�*;


total_loss�W�@

error_R_L?

learning_rate_1�9��X�I       6%�	;�P���A�*;


total_loss\��@

error_R_�T?

learning_rate_1�9	���I       6%�	R�P���A�*;


total_loss{g�@

error_R�B?

learning_rate_1�9z��KI       6%�	t��P���A�*;


total_loss���@

error_R�PH?

learning_rate_1�9ͪ�[I       6%�	(آP���A�*;


total_lossw��@

error_R�|^?

learning_rate_1�9���I       6%�	4�P���A�*;


total_loss�l�@

error_R$T?

learning_rate_1�9�5�I       6%�	�b�P���A�*;


total_loss	!�@

error_RSQD?

learning_rate_1�9���HI       6%�	���P���A�*;


total_loss�-�@

error_Rh�Q?

learning_rate_1�9�LvI       6%�	��P���A�*;


total_loss,VA

error_R}F?

learning_rate_1�9?oI       6%�	�-�P���A�*;


total_loss��@

error_R�c?

learning_rate_1�9Y��I       6%�	+u�P���A�*;


total_loss��@

error_R3K?

learning_rate_1�9�h�xI       6%�	��P���A�*;


total_loss��@

error_R$/K?

learning_rate_1�9���PI       6%�	�
�P���A�*;


total_lossA��@

error_RÈU?

learning_rate_1�9���I       6%�	T�P���A�*;


total_loss�Mj@

error_R�pC?

learning_rate_1�9�I� I       6%�	�P���A�*;


total_loss��@

error_R��\?

learning_rate_1�9P�j�I       6%�	��P���A�*;


total_loss���@

error_R*�L?

learning_rate_1�9_i�I       6%�	�*�P���A�*;


total_lossɑj@

error_Rn�E?

learning_rate_1�9̕?xI       6%�	Do�P���A�*;


total_lossZ�@

error_R�W?

learning_rate_1�9O���I       6%�	%��P���A�*;


total_loss�@

error_R$H?

learning_rate_1�9�]I       6%�	���P���A�*;


total_lossv �@

error_R+E?

learning_rate_1�9����I       6%�	XD�P���A�*;


total_lossz̘@

error_R��X?

learning_rate_1�9-��I       6%�	q��P���A�*;


total_loss{��@

error_R��i?

learning_rate_1�9�Xm�I       6%�	�ӧP���A�*;


total_lossǜ@

error_RJG7?

learning_rate_1�9��=�I       6%�	.�P���A�*;


total_lossᆭ@

error_Rr�H?

learning_rate_1�9� �I       6%�	Kb�P���A�*;


total_loss�I@

error_Rf p?

learning_rate_1�9!2I       6%�	���P���A�*;


total_loss�@

error_RT8N?

learning_rate_1�9�ySI       6%�	�P���A�*;


total_lossc�A

error_R��A?

learning_rate_1�9��0I       6%�	3�P���A�*;


total_loss�x�@

error_Rv�T?

learning_rate_1�9�|kI       6%�	�x�P���A�*;


total_lossd��@

error_RHX?

learning_rate_1�94|I       6%�	-۩P���A�*;


total_loss��@

error_RL4?

learning_rate_1�9ĂGI       6%�	$&�P���A�*;


total_loss=��@

error_R��S?

learning_rate_1�9�ޑ I       6%�	�o�P���A�*;


total_loss0�@

error_R��W?

learning_rate_1�9��MI       6%�	"��P���A�*;


total_loss8F�@

error_R��U?

learning_rate_1�9�	�I       6%�	G��P���A�*;


total_lossL�@

error_R
�I?

learning_rate_1�9��VI       6%�	�<�P���A�*;


total_loss΂�@

error_R�wJ?

learning_rate_1�9�Ĵ�I       6%�	�P���A�*;


total_loss@��@

error_RJO?

learning_rate_1�90�E�I       6%�	�ޫP���A�*;


total_loss��@

error_R]T@?

learning_rate_1�91u;�I       6%�	\)�P���A�*;


total_loss�I�@

error_R)�J?

learning_rate_1�9aCI       6%�	Dp�P���A�*;


total_loss���@

error_RsS^?

learning_rate_1�9�HI       6%�	���P���A�*;


total_loss�@

error_R�2K?

learning_rate_1�9�W��I       6%�	=�P���A�*;


total_lossj�@

error_R��T?

learning_rate_1�9r�tSI       6%�	�a�P���A�*;


total_loss��@

error_R�\9?

learning_rate_1�9�r�7I       6%�	���P���A�*;


total_lossOd�@

error_R�]J?

learning_rate_1�9��YI       6%�	@�P���A�*;


total_lossM��@

error_R T?

learning_rate_1�9�=L|I       6%�	�4�P���A�*;


total_lossb�@

error_R�RR?

learning_rate_1�9�C2I       6%�	{�P���A�*;


total_lossZ�@

error_R&�_?

learning_rate_1�9'�[�I       6%�	�®P���A�*;


total_loss�/A

error_R3�\?

learning_rate_1�96��wI       6%�	#�P���A�*;


total_loss@P�@

error_Rv%K?

learning_rate_1�9g�7I       6%�	*N�P���A�*;


total_loss��@

error_Rn�Y?

learning_rate_1�9ݿ��I       6%�	ۚ�P���A�*;


total_loss���@

error_R��R?

learning_rate_1�95�:�I       6%�	��P���A�*;


total_loss�ѧ@

error_R�O?

learning_rate_1�9/�-I       6%�	D+�P���A�*;


total_loss��@

error_R�^?

learning_rate_1�9��	I       6%�	�o�P���A�*;


total_lossKf�@

error_R;�N?

learning_rate_1�9�;��I       6%�	��P���A�*;


total_loss{'�@

error_RZI?

learning_rate_1�95,�|I       6%�	���P���A�*;


total_loss��@

error_R�k`?

learning_rate_1�9�,�I       6%�	'?�P���A�*;


total_loss��@

error_R��H?

learning_rate_1�9/�I+I       6%�	��P���A�*;


total_loss�7�@

error_R�X?

learning_rate_1�9@��I       6%�	>ƱP���A�*;


total_lossϰ�@

error_R͓E?

learning_rate_1�9�N��I       6%�	Y
�P���A�*;


total_loss�@

error_R�mS?

learning_rate_1�9��OwI       6%�	�Q�P���A�*;


total_loss�6�@

error_R8L?

learning_rate_1�9��7�I       6%�	��P���A�*;


total_loss�f�@

error_R�pO?

learning_rate_1�9B�1I       6%�	[ڲP���A�*;


total_loss�e@

error_R�KP?

learning_rate_1�9���"I       6%�	V�P���A�*;


total_loss���@

error_R��L?

learning_rate_1�98��!I       6%�	�c�P���A�*;


total_lossѺ�@

error_RfA`?

learning_rate_1�9 }��I       6%�	Ϧ�P���A�*;


total_loss�8�@

error_R�7U?

learning_rate_1�9֚}I       6%�	��P���A�*;


total_loss�8�@

error_R��O?

learning_rate_1�9C-I       6%�	?*�P���A�*;


total_loss��@

error_R�I?

learning_rate_1�9�H��I       6%�	�l�P���A�*;


total_loss�s�@

error_R�Q?

learning_rate_1�9P]:�I       6%�	D��P���A�*;


total_loss;��@

error_R��??

learning_rate_1�9��fI       6%�	.�P���A�*;


total_lossƫ�@

error_R��[?

learning_rate_1�9��A I       6%�	�=�P���A�*;


total_lossJ��@

error_R,+U?

learning_rate_1�9�{��I       6%�	��P���A�*;


total_loss��@

error_R�|L?

learning_rate_1�9�(I       6%�	�ӵP���A�*;


total_loss��@

error_RN�G?

learning_rate_1�9H���I       6%�	 �P���A�*;


total_loss�V�@

error_RX�Z?

learning_rate_1�9Y��9I       6%�	h�P���A�*;


total_loss���@

error_RO�X?

learning_rate_1�9�?��I       6%�	(��P���A�*;


total_loss�u�@

error_Rx�P?

learning_rate_1�9�`��I       6%�	���P���A�*;


total_loss�ƴ@

error_R�V?

learning_rate_1�9h�.QI       6%�	HE�P���A�*;


total_loss��@

error_R��`?

learning_rate_1�9*x%�I       6%�	׋�P���A�*;


total_loss�tA

error_R�oR?

learning_rate_1�9Ձ��I       6%�	�ԷP���A�*;


total_loss�I�@

error_R��K?

learning_rate_1�9�xF�I       6%�	��P���A�*;


total_loss�ʒ@

error_R)rN?

learning_rate_1�9���hI       6%�	B_�P���A�*;


total_loss��A

error_R�L?

learning_rate_1�9�"�I       6%�	㦸P���A�*;


total_loss��@

error_Ri�H?

learning_rate_1�9~m�]I       6%�	m�P���A�*;


total_loss�l�@

error_R�H?

learning_rate_1�9l9��I       6%�	�8�P���A�*;


total_lossRE�@

error_R�xM?

learning_rate_1�9y��OI       6%�	���P���A�*;


total_lossI�@

error_R��P?

learning_rate_1�9��
�I       6%�	]��P���A�*;


total_loss,h�@

error_R:X?

learning_rate_1�9�dTI       6%�	5G�P���A�*;


total_loss�z�@

error_R�^?

learning_rate_1�9��b�I       6%�	��P���A�*;


total_lossA�@

error_R�6T?

learning_rate_1�9���_I       6%�	�ٺP���A�*;


total_loss���@

error_R#�F?

learning_rate_1�9�w��I       6%�	W�P���A�*;


total_loss]��@

error_R.�@?

learning_rate_1�9#�YI       6%�	�c�P���A�*;


total_loss-m�@

error_RR�3?

learning_rate_1�9�ڋ�I       6%�	W��P���A�*;


total_loss�@

error_R��K?

learning_rate_1�9:C�I       6%�	?��P���A�*;


total_lossZ�@

error_RR�k?

learning_rate_1�9���MI       6%�	O9�P���A�*;


total_loss�{@

error_R�
[?

learning_rate_1�9��<>I       6%�	؃�P���A�*;


total_lossӸ�@

error_R�Z?

learning_rate_1�9�&^,I       6%�	�ɼP���A�*;


total_loss���@

error_R$'[?

learning_rate_1�9'�I       6%�	��P���A�*;


total_lossF�@

error_R�@S?

learning_rate_1�9�'�I       6%�	CN�P���A�*;


total_loss���@

error_R�']?

learning_rate_1�9��I       6%�	���P���A�*;


total_lossSt�@

error_R�6?

learning_rate_1�9/�fTI       6%�	�ܽP���A�*;


total_lossN�y@

error_R�I]?

learning_rate_1�9��MWI       6%�	�"�P���A�*;


total_loss�+�@

error_R��X?

learning_rate_1�9�dzoI       6%�	�i�P���A�*;


total_loss��@

error_R3�H?

learning_rate_1�9���I       6%�	��P���A�*;


total_lossꭠ@

error_R�P?

learning_rate_1�9��SI       6%�	���P���A�*;


total_loss�_�@

error_R�!Q?

learning_rate_1�9���>I       6%�	bC�P���A�*;


total_loss�E�@

error_R�x]?

learning_rate_1�9��I       6%�	Ǚ�P���A�*;


total_loss�7~@

error_R�i?

learning_rate_1�9+ GDI       6%�	1�P���A�*;


total_loss�h�@

error_RfCD?

learning_rate_1�9~Й)I       6%�	I*�P���A�*;


total_lossU�@

error_R��X?

learning_rate_1�9msZI       6%�	�~�P���A�*;


total_loss	��@

error_R��R?

learning_rate_1�9��8�I       6%�	8��P���A�*;


total_loss�`�@

error_RE�I?

learning_rate_1�9�.��I       6%�		�P���A�*;


total_lossD��@

error_REKH?

learning_rate_1�9x5�BI       6%�	�W�P���A�*;


total_lossa��@

error_Rv{V?

learning_rate_1�9�E�I       6%�	g��P���A�*;


total_loss�v�@

error_RM�e?

learning_rate_1�9�Z �I       6%�	O��P���A�*;


total_loss�?�@

error_R�I?

learning_rate_1�9�9QI       6%�	�0�P���A�*;


total_loss漰@

error_RDi?

learning_rate_1�94ǖ�I       6%�	�|�P���A�*;


total_loss�І@

error_R~N?

learning_rate_1�9�YI       6%�	���P���A�*;


total_loss��@

error_RN�@?

learning_rate_1�9e�I       6%�	��P���A�*;


total_loss���@

error_R�xJ?

learning_rate_1�9����I       6%�	�R�P���A�*;


total_lossxI�@

error_R�R?

learning_rate_1�9��d>I       6%�	t��P���A�*;


total_loss�<�@

error_Rs�??

learning_rate_1�9`�I       6%�	���P���A�*;


total_lossZ��@

error_R��O?

learning_rate_1�9?2kyI       6%�	#)�P���A�*;


total_lossd��@

error_R�nC?

learning_rate_1�9����I       6%�	�o�P���A�*;


total_loss!��@

error_R��I?

learning_rate_1�9�*mI       6%�	��P���A�*;


total_loss;��@

error_RTV?

learning_rate_1�9����I       6%�	9��P���A�*;


total_loss�ƚ@

error_RM�.?

learning_rate_1�9�=I       6%�	�;�P���A�*;


total_loss|q�@

error_R�>g?

learning_rate_1�9�9f�I       6%�	��P���A�*;


total_loss6�@

error_RW�C?

learning_rate_1�9ŀ-I       6%�	���P���A�*;


total_loss@��@

error_R��>?

learning_rate_1�9�'�I       6%�	�
�P���A�*;


total_loss���@

error_RWM;?

learning_rate_1�9J�8�I       6%�	EX�P���A�*;


total_lossC�A

error_R�K?

learning_rate_1�9c^@�I       6%�	���P���A�*;


total_lossH�@

error_Rj`?

learning_rate_1�9i���I       6%�	?��P���A�*;


total_loss؃�@

error_RT6M?

learning_rate_1�9���FI       6%�	/3�P���A�*;


total_lossԠ�@

error_RJJR?

learning_rate_1�9�K��I       6%�	�z�P���A�*;


total_loss��@

error_Rz�Z?

learning_rate_1�9�ʵI       6%�	���P���A�*;


total_loss��@

error_R��R?

learning_rate_1�9��R,I       6%�	�P���A�*;


total_loss)��@

error_R��L?

learning_rate_1�9�,�I       6%�	FN�P���A�*;


total_loss��@

error_RO7R?

learning_rate_1�9�R�I       6%�	��P���A�*;


total_loss4_�@

error_RWR??

learning_rate_1�9�	mI       6%�	 ��P���A�*;


total_loss?7�@

error_R��,?

learning_rate_1�9T:I       6%�	�P���A�*;


total_loss;�@

error_R��D?

learning_rate_1�94	��I       6%�	�`�P���A�*;


total_loss[W�@

error_R��H?

learning_rate_1�9�#I       6%�	��P���A�*;


total_loss��A

error_R�`Q?

learning_rate_1�9�ήqI       6%�	��P���A�*;


total_loss�q�@

error_R)sF?

learning_rate_1�9h��3I       6%�	�M�P���A�*;


total_lossEP�@

error_RW�@?

learning_rate_1�9�dg�I       6%�	��P���A�*;


total_loss4��@

error_RWO9?

learning_rate_1�9�J#�I       6%�	r��P���A�*;


total_loss֠�@

error_RC�P?

learning_rate_1�9��# I       6%�	,"�P���A�*;


total_loss��@

error_R&KO?

learning_rate_1�9�$+�I       6%�	g�P���A�*;


total_loss1+�@

error_R�Q9?

learning_rate_1�9]��vI       6%�	���P���A�*;


total_lossl�@

error_R�I?

learning_rate_1�9�"KI       6%�	���P���A�*;


total_loss�@

error_ROlD?

learning_rate_1�9��z�I       6%�	;;�P���A�*;


total_loss��@

error_R�"T?

learning_rate_1�9G=�I       6%�	L��P���A�*;


total_loss	��@

error_Ri}Q?

learning_rate_1�9�?�I       6%�	:��P���A�*;


total_lossP��@

error_RX�O?

learning_rate_1�9u�8I       6%�	y�P���A�*;


total_lossi��@

error_R��B?

learning_rate_1�9��MI       6%�	<W�P���A�*;


total_lossHܕ@

error_RW?D?

learning_rate_1�9�"PI       6%�	��P���A�*;


total_loss4�@

error_R� O?

learning_rate_1�9u�utI       6%�	���P���A�*;


total_lossɠ�@

error_R��R?

learning_rate_1�9 ot0I       6%�	�5�P���A�*;


total_loss�@�@

error_R�EO?

learning_rate_1�9���I       6%�	�w�P���A�*;


total_loss�;�@

error_R��K?

learning_rate_1�9!�>�I       6%�	���P���A�*;


total_loss��@

error_RdNB?

learning_rate_1�9�9EwI       6%�	��P���A�*;


total_loss���@

error_R�^?

learning_rate_1�9?�v&I       6%�	�F�P���A�*;


total_loss�&�@

error_Ri�@?

learning_rate_1�9�3uI       6%�	��P���A�*;


total_loss��@

error_RL�P?

learning_rate_1�9����I       6%�	R��P���A�*;


total_loss���@

error_R=pO?

learning_rate_1�9�{I       6%�	��P���A�*;


total_loss�@

error_R�SQ?

learning_rate_1�9�Q/I       6%�	j�P���A�*;


total_lossEÇ@

error_RC:?

learning_rate_1�9F�ԶI       6%�	ٰ�P���A�*;


total_loss=�@

error_R�M?

learning_rate_1�9e��I       6%�	c��P���A�*;


total_loss���@

error_R��M?

learning_rate_1�9����I       6%�	c<�P���A�*;


total_loss�5�@

error_R�eQ?

learning_rate_1�9J��^I       6%�	���P���A�*;


total_lossv��@

error_Rl�E?

learning_rate_1�9���I       6%�	���P���A�*;


total_loss��A

error_R,-S?

learning_rate_1�9{I       6%�	��P���A�*;


total_loss+��@

error_R�`?

learning_rate_1�9f�B�I       6%�	�O�P���A�*;


total_loss��A

error_R��l?

learning_rate_1�9��I       6%�	]��P���A�*;


total_loss��@

error_R}_T?

learning_rate_1�9�ݴ�I       6%�	q��P���A�*;


total_lossGvA

error_R�0M?

learning_rate_1�9%�T�I       6%�	(#�P���A�*;


total_lossl�@

error_R��N?

learning_rate_1�9�#�I       6%�	wk�P���A�*;


total_loss�̉@

error_RO�B?

learning_rate_1�9.�XI       6%�	R��P���A�*;


total_loss1ɘ@

error_R��]?

learning_rate_1�9�*��I       6%�	e��P���A�*;


total_loss��?@

error_R�R?

learning_rate_1�9�du�I       6%�	�J�P���A�*;


total_loss]��@

error_RZ�P?

learning_rate_1�9�.72I       6%�	_��P���A�*;


total_loss_��@

error_RÆT?

learning_rate_1�9[4�I       6%�	���P���A�*;


total_loss:�@

error_R�/;?

learning_rate_1�9%�e�I       6%�	i�P���A�*;


total_loss��A

error_R�m`?

learning_rate_1�9r���I       6%�	�^�P���A�*;


total_loss��@

error_R��L?

learning_rate_1�9���BI       6%�	C��P���A�*;


total_loss���@

error_RfvX?

learning_rate_1�9.��I       6%�	���P���A�*;


total_loss䂹@

error_R�W?

learning_rate_1�9!B��I       6%�	�B�P���A�*;


total_loss_V@

error_RGH?

learning_rate_1�9¼��I       6%�	]��P���A�*;


total_loss
,�@

error_Re6K?

learning_rate_1�99n�8I       6%�	N��P���A�*;


total_loss1J�@

error_R�3N?

learning_rate_1�9��͎I       6%�	��P���A�*;


total_loss	�@

error_R=VP?

learning_rate_1�9� QI       6%�	pd�P���A�*;


total_loss\��@

error_R-�I?

learning_rate_1�9���oI       6%�	���P���A�*;


total_loss,��@

error_R2�C?

learning_rate_1�9��cI       6%�	9��P���A�*;


total_loss�@

error_R�*N?

learning_rate_1�9��zmI       6%�	55�P���A�*;


total_loss]��@

error_R}�@?

learning_rate_1�9����I       6%�	��P���A�*;


total_loss�A

error_RluF?

learning_rate_1�9bOiI       6%�	3��P���A�*;


total_loss&��@

error_R�[?

learning_rate_1�9����I       6%�	��P���A�*;


total_lossϙ�@

error_R hI?

learning_rate_1�9@��NI       6%�	�[�P���A�*;


total_loss��@

error_R��a?

learning_rate_1�9�=a�I       6%�	���P���A�*;


total_lossFJ�@

error_Rra?

learning_rate_1�9rľ�I       6%�	m�P���A�*;


total_loss���@

error_R=bF?

learning_rate_1�9�I       6%�	�Q�P���A�*;


total_loss���@

error_R)�K?

learning_rate_1�9��8I       6%�	���P���A�*;


total_loss��@

error_RhC_?

learning_rate_1�9�q/�I       6%�	���P���A�*;


total_loss�
A

error_R��>?

learning_rate_1�9r�EI       6%�	$�P���A�*;


total_loss6�@

error_R��I?

learning_rate_1�9G�g�I       6%�	(e�P���A�*;


total_lossCŷ@

error_RM�C?

learning_rate_1�9 �;�I       6%�	��P���A�*;


total_lossl_�@

error_R��b?

learning_rate_1�955��I       6%�	-��P���A�*;


total_loss�#�@

error_R�qK?

learning_rate_1�9�&�bI       6%�	�=�P���A�*;


total_lossXZ�@

error_R��T?

learning_rate_1�9�1�pI       6%�	��P���A�*;


total_loss�β@

error_RA!,?

learning_rate_1�9R�I�I       6%�	���P���A�*;


total_loss�#�@

error_Rt�S?

learning_rate_1�9U3��I       6%�	t�P���A�*;


total_loss�6�@

error_R�YN?

learning_rate_1�9���CI       6%�	�c�P���A�*;


total_loss��@

error_R��E?

learning_rate_1�9��I       6%�	��P���A�*;


total_lossz��@

error_R�[?

learning_rate_1�9!�eI       6%�	,��P���A�*;


total_loss���@

error_R�M?

learning_rate_1�9TW6oI       6%�	�4�P���A�*;


total_loss׾@

error_R�nQ?

learning_rate_1�9�؄I       6%�	{�P���A�*;


total_loss)�@

error_R=�G?

learning_rate_1�9���I       6%�	��P���A�*;


total_loss��@

error_R��Q?

learning_rate_1�9�X�I       6%�	��P���A�*;


total_lossQ��@

error_R8	P?

learning_rate_1�9��"I       6%�	d]�P���A�*;


total_loss�fA

error_R8�f?

learning_rate_1�9Z�]I       6%�	v��P���A�*;


total_loss`��@

error_RjEN?

learning_rate_1�9��1�I       6%�	���P���A�*;


total_lossV�A

error_Rq�`?

learning_rate_1�9���,I       6%�	V.�P���A�*;


total_losse��@

error_R�QX?

learning_rate_1�9�=T�I       6%�	�p�P���A�*;


total_loss2��@

error_R�`X?

learning_rate_1�9�|2I       6%�	��P���A�*;


total_lossl�~@

error_RX?

learning_rate_1�9p��I       6%�	,��P���A�*;


total_loss�A

error_R*�J?

learning_rate_1�9�I       6%�	�7�P���A�*;


total_loss��@

error_RHO?

learning_rate_1�9p�KI       6%�	�z�P���A�*;


total_loss@c�@

error_R�T?

learning_rate_1�9�-��I       6%�	N��P���A�*;


total_lossѪA

error_R�A?

learning_rate_1�9Ȇs-I       6%�	��P���A�*;


total_loss@�@

error_R�_?

learning_rate_1�9�,I       6%�	�J�P���A�*;


total_loss���@

error_RD�S?

learning_rate_1�9x*I       6%�	<��P���A�*;


total_lossLߤ@

error_Rl�L?

learning_rate_1�9O��.I       6%�	���P���A�*;


total_loss(5�@

error_R��M?

learning_rate_1�9>3��I       6%�	��P���A�*;


total_loss҇�@

error_R��Q?

learning_rate_1�9E�39I       6%�	%b�P���A�*;


total_loss��@

error_RҸL?

learning_rate_1�9�kpUI       6%�	��P���A�*;


total_loss�~|@

error_RO�Q?

learning_rate_1�9���DI       6%�	���P���A�*;


total_lossU�@

error_RfB?

learning_rate_1�9Ԉ�NI       6%�	G�P���A�*;


total_loss��@

error_R�k?

learning_rate_1�9�pFI       6%�	_��P���A�*;


total_loss�b�@

error_R{�I?

learning_rate_1�9�w��I       6%�	���P���A�*;


total_loss��@

error_R%�`?

learning_rate_1�9��I       6%�	��P���A�*;


total_lossVQ�@

error_R��Y?

learning_rate_1�9V�ʜI       6%�	c�P���A�*;


total_loss�@�@

error_R�KW?

learning_rate_1�9��ŅI       6%�	U��P���A�*;


total_lossV��@

error_R+M?

learning_rate_1�9�G�I       6%�	+��P���A�*;


total_loss>�@

error_R�B?

learning_rate_1�9OU6I       6%�	f,�P���A�*;


total_lossO��@

error_R�/L?

learning_rate_1�9|���I       6%�	p�P���A�*;


total_loss���@

error_RE8V?

learning_rate_1�9���I       6%�	G��P���A�*;


total_loss��@

error_R��\?

learning_rate_1�9�IyI       6%�	���P���A�*;


total_loss��@

error_R �K?

learning_rate_1�9>�V�I       6%�	�;�P���A�*;


total_loss]v�@

error_R=�D?

learning_rate_1�9�zGMI       6%�	���P���A�*;


total_loss��@

error_RwT;?

learning_rate_1�9�W�I       6%�	L��P���A�*;


total_loss=�A

error_R2�U?

learning_rate_1�9���I       6%�	��P���A�*;


total_lossv�@

error_R�wI?

learning_rate_1�9�j�I       6%�	V�P���A�*;


total_loss���@

error_R.jO?

learning_rate_1�9� �I       6%�	ϟ�P���A�*;


total_loss��@

error_R��:?

learning_rate_1�9��I       6%�	���P���A�*;


total_loss�A

error_Rd�E?

learning_rate_1�93��I       6%�	�5�P���A�*;


total_loss�@

error_R�uZ?

learning_rate_1�9��ibI       6%�	��P���A�*;


total_loss�@

error_R$`?

learning_rate_1�9���I       6%�	���P���A�*;


total_loss�H�@

error_R�'\?

learning_rate_1�9[���I       6%�	�1�P���A�*;


total_loss\@�@

error_RbH?

learning_rate_1�98�kI       6%�	�t�P���A�*;


total_lossr�@

error_R;�[?

learning_rate_1�9}O��I       6%�	K��P���A�*;


total_loss8��@

error_R�G?

learning_rate_1�9��I       6%�	h �P���A�*;


total_loss�O1A

error_Rȴ[?

learning_rate_1�9���HI       6%�	cD�P���A�*;


total_loss��@

error_R��l?

learning_rate_1�9�j9I       6%�	R��P���A�*;


total_loss�fA

error_Rx�I?

learning_rate_1�9�@cI       6%�	���P���A�*;


total_lossL��@

error_R1�S?

learning_rate_1�9�r?hI       6%�	:�P���A�*;


total_loss)��@

error_R6<Z?

learning_rate_1�9r�E�I       6%�	W�P���A�*;


total_loss/�a@

error_R��P?

learning_rate_1�9Gh,I       6%�	���P���A�*;


total_loss���@

error_R�!b?

learning_rate_1�9CE
KI       6%�	���P���A�*;


total_loss�@

error_R!�b?

learning_rate_1�9���I       6%�	^"�P���A�*;


total_loss�?�@

error_R�T?

learning_rate_1�97�ȉI       6%�	�l�P���A�*;


total_loss�7�@

error_R��??

learning_rate_1�9�H�I       6%�	w��P���A�*;


total_loss�h�@

error_R�S?

learning_rate_1�9�_��I       6%�	���P���A�*;


total_loss���@

error_R�v<?

learning_rate_1�9_=tI       6%�	�D�P���A�*;


total_lossMG�@

error_R�~Y?

learning_rate_1�9@�LI       6%�	o��P���A�*;


total_loss*��@

error_R@K?

learning_rate_1�9�D]�I       6%�	��P���A�*;


total_loss�@A

error_R2�T?

learning_rate_1�9:R.�I       6%�	��P���A�*;


total_loss�A

error_Rc�T?

learning_rate_1�9�]�II       6%�	'W�P���A�*;


total_loss���@

error_R4{A?

learning_rate_1�9|��eI       6%�	'��P���A�*;


total_loss#�@

error_Rv�I?

learning_rate_1�9@l^,I       6%�	 ��P���A�*;


total_lossȽ�@

error_R��]?

learning_rate_1�9_+s/I       6%�	� �P���A�*;


total_loss�U�@

error_R �??

learning_rate_1�9�r�I       6%�	f�P���A�*;


total_loss�γ@

error_R�AW?

learning_rate_1�9�KrI       6%�	[��P���A�*;


total_loss=E�@

error_R{uV?

learning_rate_1�9�z�I       6%�	���P���A�*;


total_losst�A

error_R6qG?

learning_rate_1�9�1��I       6%�	1C�P���A�*;


total_loss`T�@

error_RR�F?

learning_rate_1�9"�uAI       6%�	
��P���A�*;


total_lossQ�@

error_RV�V?

learning_rate_1�9q�^�I       6%�	���P���A�*;


total_lossf��@

error_R��<?

learning_rate_1�9d�
I       6%�	O+�P���A�*;


total_lossE�@

error_R�	]?

learning_rate_1�9��9�I       6%�	z�P���A�*;


total_lossz4�@

error_R�9R?

learning_rate_1�9���XI       6%�	y��P���A�*;


total_loss�~�@

error_R&_?

learning_rate_1�9�jI       6%�	i�P���A�*;


total_loss<��@

error_R�"P?

learning_rate_1�9��5I       6%�	�V�P���A�*;


total_loss�Q�@

error_R�L?

learning_rate_1�9���1I       6%�	���P���A�*;


total_lossm��@

error_RZC?

learning_rate_1�9$��*I       6%�	���P���A�*;


total_lossT��@

error_R)VX?

learning_rate_1�9����I       6%�	�&�P���A�*;


total_loss�=�@

error_R*�<?

learning_rate_1�93��I       6%�	�j�P���A�*;


total_lossj��@

error_R�X?

learning_rate_1�9���I       6%�	!��P���A�*;


total_loss��@

error_R�%d?

learning_rate_1�9�Y��I       6%�	���P���A�*;


total_loss�O�@

error_R$E?

learning_rate_1�9�0 �I       6%�	�?�P���A�*;


total_loss%X�@

error_R��\?

learning_rate_1�9d\��I       6%�	���P���A�*;


total_lossd��@

error_R��.?

learning_rate_1�9d�:I       6%�	���P���A�*;


total_loss�ޖ@

error_R3YV?

learning_rate_1�9k^y�I       6%�	W�P���A�*;


total_losso�@

error_R�V?

learning_rate_1�9����I       6%�	�U�P���A�*;


total_loss�K�@

error_R)EU?

learning_rate_1�9�fI       6%�	���P���A�*;


total_lossv�@

error_R�{@?

learning_rate_1�9.`�I       6%�	���P���A�*;


total_loss��@

error_R:�f?

learning_rate_1�9�	I       6%�	{%�P���A�*;


total_loss=e�@

error_RxLB?

learning_rate_1�9����I       6%�	�j�P���A�*;


total_loss���@

error_R��J?

learning_rate_1�9i�bI       6%�	0��P���A�*;


total_loss��@

error_R��c?

learning_rate_1�9�L:�I       6%�	���P���A�*;


total_loss��@

error_R��Q?

learning_rate_1�9��I       6%�	�7�P���A�*;


total_lossb�A

error_R��c?

learning_rate_1�9�UI       6%�	��P���A�*;


total_loss�f�@

error_R8�E?

learning_rate_1�9aŤ�I       6%�	���P���A�*;


total_lossO��@

error_Rv�b?

learning_rate_1�9x��`I       6%�	r�P���A�*;


total_loss�@

error_R�0B?

learning_rate_1�96���I       6%�	�I�P���A�*;


total_loss���@

error_R.aJ?

learning_rate_1�9 f��I       6%�	��P���A�*;


total_loss��@

error_R�pF?

learning_rate_1�9�ZtEI       6%�	���P���A�*;


total_loss��@

error_RT�>?

learning_rate_1�9d;�I       6%�	�9�P���A�*;


total_loss���@

error_R(X[?

learning_rate_1�9�27�I       6%�	_~�P���A�*;


total_loss���@

error_R�\?

learning_rate_1�9�Z��I       6%�	���P���A�*;


total_loss�ߚ@

error_Rűe?

learning_rate_1�9�t�CI       6%�	q�P���A�*;


total_loss��@

error_R�W?

learning_rate_1�9Jw�'I       6%�	JQ�P���A�*;


total_loss�p�@

error_R�ML?

learning_rate_1�9LUmI       6%�	x��P���A�*;


total_lossK�@

error_R;�C?

learning_rate_1�9U��I       6%�	`��P���A�*;


total_lossf�@

error_R��Z?

learning_rate_1�9o�48I       6%�	�*�P���A�*;


total_losszg�@

error_RߕM?

learning_rate_1�9Օ��I       6%�	Nt�P���A�*;


total_lossn��@

error_R��F?

learning_rate_1�99���I       6%�	��P���A�*;


total_loss �r@

error_RRU?

learning_rate_1�9��I       6%�	D�P���A�*;


total_loss)��@

error_R7oD?

learning_rate_1�9Cw��I       6%�	KN�P���A�*;


total_loss6��@

error_R͜P?

learning_rate_1�9��c0I       6%�	u��P���A�*;


total_loss��@

error_R��V?

learning_rate_1�9n39QI       6%�	2��P���A�*;


total_lossȗ�@

error_R��:?

learning_rate_1�9u��I       6%�	 2�P���A�*;


total_losss�	A

error_R�G?

learning_rate_1�9=�\I       6%�	,}�P���A�*;


total_loss���@

error_R��A?

learning_rate_1�9�c�I       6%�	%��P���A�*;


total_loss��,A

error_R�TC?

learning_rate_1�9/$3rI       6%�	u�P���A�*;


total_loss���@

error_R��Q?

learning_rate_1�9��'�I       6%�	#Q�P���A�*;


total_loss{��@

error_R�hN?

learning_rate_1�9.Z�UI       6%�	���P���A�*;


total_lossI��@

error_R-
R?

learning_rate_1�9c[�SI       6%�	��P���A�*;


total_loss���@

error_R�PB?

learning_rate_1�9�Un�I       6%�	O Q���A�*;


total_lossAW�@

error_R��L?

learning_rate_1�9��$�I       6%�	6i Q���A�*;


total_loss<��@

error_R_�N?

learning_rate_1�9\CQ?I       6%�	j� Q���A�*;


total_loss��@

error_R!R?

learning_rate_1�9x�sI       6%�	� Q���A�*;


total_loss1ߔ@

error_Rx�J?

learning_rate_1�9Cv� I       6%�	9Q���A�*;


total_loss 3�@

error_R6$[?

learning_rate_1�9�D��I       6%�	�{Q���A�*;


total_lossrz�@

error_RcPW?

learning_rate_1�9�0M�I       6%�	��Q���A�*;


total_loss�z�@

error_R��B?

learning_rate_1�9�<-�I       6%�	SQ���A�*;


total_lossm� A

error_R<]?

learning_rate_1�9�&�YI       6%�	�TQ���A�*;


total_lossZ��@

error_R
D>?

learning_rate_1�9I=�I       6%�	�Q���A�*;


total_loss��@

error_R��:?

learning_rate_1�9XG9�I       6%�	��Q���A�*;


total_loss\
�@

error_RrS?

learning_rate_1�9����I       6%�	�Q���A�*;


total_loss_Z�@

error_R{[?

learning_rate_1�9���QI       6%�	�iQ���A�*;


total_loss��@

error_R��`?

learning_rate_1�9��o�I       6%�	i�Q���A�*;


total_loss���@

error_R�BA?

learning_rate_1�9�k�QI       6%�	��Q���A�*;


total_loss��A

error_R��V?

learning_rate_1�9�"�*I       6%�	�GQ���A�*;


total_loss6��@

error_R@�N?

learning_rate_1�9�=��I       6%�	��Q���A�*;


total_loss���@

error_R��e?

learning_rate_1�9���I       6%�	��Q���A�*;


total_lossR��@

error_R)�R?

learning_rate_1�9�-��I       6%�	�Q���A�*;


total_lossH(�@

error_Rq�K?

learning_rate_1�9{Z�GI       6%�	LnQ���A�*;


total_loss��@

error_R�P?

learning_rate_1�9��- I       6%�	y�Q���A�*;


total_loss���@

error_RʛP?

learning_rate_1�9�x�I       6%�	<Q���A�*;


total_lossq��@

error_R=�H?

learning_rate_1�9QGujI       6%�	GQ���A�*;


total_loss2��@

error_R!sG?

learning_rate_1�9L�#)I       6%�	��Q���A�*;


total_loss���@

error_R�??

learning_rate_1�9_@uI       6%�	��Q���A�*;


total_loss��@

error_R�AM?

learning_rate_1�9��i�I       6%�	�Q���A�*;


total_loss���@

error_R�ZZ?

learning_rate_1�9X�� I       6%�	#WQ���A�*;


total_lossrr�@

error_R�HJ?

learning_rate_1�9����I       6%�	;�Q���A�*;


total_loss
c�@

error_Rm>?

learning_rate_1�9�^�kI       6%�	4�Q���A�*;


total_loss���@

error_R];=?

learning_rate_1�9ٳxI       6%�	�-Q���A�*;


total_lossh�A

error_R�I?

learning_rate_1�9f��I       6%�	�pQ���A�*;


total_loss�@

error_R.�_?

learning_rate_1�9D�r�I       6%�	t�Q���A�*;


total_loss)ӟ@

error_R��\?

learning_rate_1�9^_II       6%�	g�Q���A�*;


total_loss�X�@

error_RR�>?

learning_rate_1�9��g�I       6%�	�?	Q���A�*;


total_loss��@

error_R�@?

learning_rate_1�9�kG1I       6%�	>�	Q���A�*;


total_lossߪ�@

error_R=�d?

learning_rate_1�9C�$I       6%�	��	Q���A�*;


total_loss;��@

error_R,)J?

learning_rate_1�9
�I       6%�	�1
Q���A�*;


total_losss��@

error_R/6?

learning_rate_1�9�^��I       6%�	^t
Q���A�*;


total_loss䏝@

error_R(xI?

learning_rate_1�9)b�I       6%�	/�
Q���A�*;


total_loss���@

error_R�N;?

learning_rate_1�9/$��I       6%�	��
Q���A�*;


total_lossa;�@

error_R1�Y?

learning_rate_1�9n��I       6%�	KBQ���A�*;


total_loss�@

error_R}�L?

learning_rate_1�9[�7
I       6%�	R�Q���A�*;


total_loss�@�@

error_R�U?

learning_rate_1�9�]�I       6%�	)�Q���A�*;


total_loss}Z�@

error_Ri�K?

learning_rate_1�9���I       6%�	�Q���A�*;


total_lossљ�@

error_R��H?

learning_rate_1�9�mmI       6%�	�_Q���A�*;


total_loss�4A

error_R�F?

learning_rate_1�9�4iI       6%�	��Q���A�*;


total_loss ��@

error_R�M?

learning_rate_1�9��OI       6%�	;�Q���A�*;


total_loss��@

error_R:{P?

learning_rate_1�9����I       6%�	�=Q���A�*;


total_loss̋A

error_RH}V?

learning_rate_1�90~I       6%�	��Q���A�*;


total_loss���@

error_R��H?

learning_rate_1�9@];�I       6%�	r�Q���A�*;


total_loss�<�@

error_R=hZ?

learning_rate_1�9��(I       6%�	YQ���A�*;


total_loss�Y�@

error_R�i?

learning_rate_1�9�ڝ�I       6%�	�OQ���A�*;


total_lossx��@

error_R�&C?

learning_rate_1�9]]��I       6%�	�Q���A�*;


total_loss��@

error_Rn�W?

learning_rate_1�9*��I       6%�	��Q���A�*;


total_lossV��@

error_R��S?

learning_rate_1�9g��I       6%�	�+Q���A�*;


total_loss�a�@

error_R��O?

learning_rate_1�936
GI       6%�	�xQ���A�*;


total_loss�t�@

error_R��P?

learning_rate_1�9n�cI       6%�	��Q���A�*;


total_loss��@

error_R{�S?

learning_rate_1�9���I       6%�	�Q���A�*;


total_loss,�@

error_R??

learning_rate_1�9(ũI       6%�	�\Q���A�*;


total_loss=&A

error_R�D?

learning_rate_1�9�7"]I       6%�	ϥQ���A�*;


total_loss�@

error_R�E?

learning_rate_1�9�EC�I       6%�	��Q���A�*;


total_loss\�A

error_Rf�Y?

learning_rate_1�9I:{I       6%�	>Q���A�*;


total_loss��@

error_R��[?

learning_rate_1�9i}��I       6%�	��Q���A�*;


total_losso��@

error_RR�T?

learning_rate_1�9��UQI       6%�	�Q���A�*;


total_loss1
�@

error_Rl�[?

learning_rate_1�9��{I       6%�	%"Q���A�*;


total_loss���@

error_R@/Y?

learning_rate_1�9��`�I       6%�	�iQ���A�*;


total_loss��@

error_R�D?

learning_rate_1�9��m�I       6%�	��Q���A�*;


total_loss= A

error_RMYK?

learning_rate_1�9+�uI       6%�	&�Q���A�*;


total_loss�T@

error_R�#h?

learning_rate_1�93�}�I       6%�	�VQ���A�*;


total_loss�9�@

error_R�D?

learning_rate_1�9�JM]I       6%�	]�Q���A�*;


total_loss�'�@

error_R��J?

learning_rate_1�9��tI       6%�	��Q���A�*;


total_lossM��@

error_Rf�D?

learning_rate_1�9�6O9I       6%�	%$Q���A�*;


total_loss�xA

error_RnC?

learning_rate_1�97V�I       6%�	(kQ���A�*;


total_loss�A

error_R��G?

learning_rate_1�9�a�\I       6%�	M�Q���A�*;


total_loss��@

error_R�\?

learning_rate_1�9eq�I       6%�	��Q���A�*;


total_loss��@

error_R�CS?

learning_rate_1�9(��I       6%�	?Q���A�*;


total_loss{[�@

error_RW�R?

learning_rate_1�9�.sI       6%�	φQ���A�*;


total_losse�@

error_R��T?

learning_rate_1�9���I       6%�	^�Q���A�*;


total_loss���@

error_RҸR?

learning_rate_1�9 a�KI       6%�	�Q���A�*;


total_lossSY�@

error_R_�\?

learning_rate_1�9�E��I       6%�	�_Q���A�*;


total_lossC��@

error_R��V?

learning_rate_1�9` LI       6%�	u�Q���A�*;


total_lossiܺ@

error_R��X?

learning_rate_1�9��UlI       6%�	��Q���A�*;


total_loss��@

error_R��B?

learning_rate_1�9�I       6%�	68Q���A�*;


total_lossF5�@

error_R�M:?

learning_rate_1�9���I       6%�	+yQ���A�*;


total_lossDe�@

error_R*�\?

learning_rate_1�9�AI       6%�	-�Q���A�*;


total_lossoe�@

error_RI�h?

learning_rate_1�9ǘ�nI       6%�	vQ���A�*;


total_loss�,v@

error_R,qE?

learning_rate_1�9I�:
I       6%�	IQ���A�*;


total_loss�¸@

error_R&H?

learning_rate_1�9�|`I       6%�	�Q���A�*;


total_lossiQ�@

error_REL?

learning_rate_1�9<�\XI       6%�	��Q���A�*;


total_loss��@

error_R,U?

learning_rate_1�9�Z��I       6%�	Q���A�*;


total_loss��@

error_R�5g?

learning_rate_1�9�6�I       6%�	UQ���A�*;


total_loss;�@

error_R]!K?

learning_rate_1�96FI       6%�	�Q���A�*;


total_lossx��@

error_R
E?

learning_rate_1�9s}VI       6%�	sQ���A�*;


total_loss�F�@

error_RMrN?

learning_rate_1�9��ߡI       6%�	kQQ���A�*;


total_loss�t@

error_R@L?

learning_rate_1�9i+�eI       6%�	˔Q���A�*;


total_loss���@

error_R��W?

learning_rate_1�9����I       6%�	��Q���A�*;


total_loss�0�@

error_R�|V?

learning_rate_1�9���lI       6%�	�Q���A�*;


total_loss:�@

error_R��D?

learning_rate_1�9���$I       6%�	�hQ���A�*;


total_loss� �@

error_R!^I?

learning_rate_1�9��m�I       6%�	_�Q���A�*;


total_lossF��@

error_R
�A?

learning_rate_1�9���TI       6%�	Y�Q���A�*;


total_losss��@

error_R7a?

learning_rate_1�99}1�I       6%�	X4Q���A�*;


total_lossC��@

error_R2�X?

learning_rate_1�9�[I       6%�	�xQ���A�*;


total_loss�-�@

error_RR*N?

learning_rate_1�9��I       6%�	_�Q���A�*;


total_loss�<�@

error_RI�Q?

learning_rate_1�9��I       6%�	tQ���A�*;


total_loss��@

error_R�#h?

learning_rate_1�9WOI       6%�	�QQ���A�*;


total_loss`-�@

error_R
s??

learning_rate_1�9?1��I       6%�	Q�Q���A�*;


total_loss��A

error_R�5Y?

learning_rate_1�9l�h I       6%�	��Q���A�*;


total_loss�\�@

error_R�I?

learning_rate_1�9{m�I       6%�	�:Q���A�*;


total_loss���@

error_R�*M?

learning_rate_1�9���I       6%�	��Q���A�*;


total_loss��A

error_R,xC?

learning_rate_1�9w�%I       6%�	m�Q���A�*;


total_loss6��@

error_R�jN?

learning_rate_1�9[L�I       6%�	AQ���A�*;


total_lossF�@

error_R͡E?

learning_rate_1�9kS!�I       6%�	<TQ���A�*;


total_loss��A

error_R�{J?

learning_rate_1�9�ŲyI       6%�	��Q���A�*;


total_loss�}@

error_R��F?

learning_rate_1�95���I       6%�	��Q���A�*;


total_loss{�@

error_R��J?

learning_rate_1�9wN$_I       6%�	�' Q���A�*;


total_loss�ݤ@

error_RxwD?

learning_rate_1�9�8:I       6%�	gi Q���A�*;


total_loss�@

error_R!�O?

learning_rate_1�9�f��I       6%�	G� Q���A�*;


total_lossV� A

error_RLI\?

learning_rate_1�9Ժl�I       6%�		� Q���A�*;


total_losstk�@

error_R3 T?

learning_rate_1�9��OI       6%�	�4!Q���A�*;


total_loss��@

error_R&�E?

learning_rate_1�9���I       6%�	gx!Q���A�*;


total_lossݫ�@

error_R�(D?

learning_rate_1�9���nI       6%�	�!Q���A�*;


total_loss`��@

error_RCK?

learning_rate_1�9uJ��I       6%�	��!Q���A�*;


total_loss��@

error_Rd�Z?

learning_rate_1�9��&�I       6%�	�@"Q���A�*;


total_loss/�@

error_RZ�L?

learning_rate_1�9i4ކI       6%�	�"Q���A�*;


total_loss�j�@

error_RD/1?

learning_rate_1�9�q�>I       6%�	��"Q���A�*;


total_loss�,�@

error_RQIR?

learning_rate_1�9�l/mI       6%�	
#Q���A�*;


total_loss��@

error_RX]E?

learning_rate_1�9t��I       6%�	�L#Q���A�*;


total_loss��@

error_R��@?

learning_rate_1�9+�zI       6%�	�#Q���A�*;


total_loss�	A

error_R��T?

learning_rate_1�9�k��I       6%�	��#Q���A�*;


total_loss�~�@

error_R��@?

learning_rate_1�9�l:FI       6%�	�$Q���A�*;


total_loss�͵@

error_Rl
L?

learning_rate_1�9M �)I       6%�	�`$Q���A�*;


total_loss|��@

error_RFZ?

learning_rate_1�9˭zI       6%�	/�$Q���A�*;


total_loss��@

error_R�NO?

learning_rate_1�90]��I       6%�	z�$Q���A�*;


total_loss�qi@

error_Rq�F?

learning_rate_1�9��W�I       6%�	�-%Q���A�*;


total_loss]~@

error_R�QS?

learning_rate_1�9��I       6%�	�u%Q���A�*;


total_loss�z�@

error_RR8U?

learning_rate_1�9��I4I       6%�	g�%Q���A�*;


total_loss��@

error_RtJY?

learning_rate_1�9ɨpI       6%�	 &Q���A�*;


total_loss@�@

error_R}:<?

learning_rate_1�9�B42I       6%�	�J&Q���A�*;


total_lossF	,A

error_R݉Q?

learning_rate_1�9�ڎI       6%�	l�&Q���A�*;


total_loss(~�@

error_R�,O?

learning_rate_1�9���;I       6%�	F�&Q���A�*;


total_loss��@

error_R�P?

learning_rate_1�9����I       6%�	�$'Q���A�*;


total_lossXC�@

error_RS_R?

learning_rate_1�9HT�lI       6%�	�i'Q���A�*;


total_lossZ�@

error_R�mX?

learning_rate_1�9m~�,I       6%�	H�'Q���A�*;


total_lossa�@

error_RE[?

learning_rate_1�9�,<bI       6%�	��'Q���A�*;


total_loss�1�@

error_R��N?

learning_rate_1�9�,I       6%�	B=(Q���A�*;


total_lossj��@

error_R��[?

learning_rate_1�9���I       6%�	6�(Q���A�*;


total_loss��@

error_R�T?

learning_rate_1�9�	<I       6%�	��(Q���A�*;


total_loss]��@

error_RbH?

learning_rate_1�9>.=�I       6%�	�)Q���A�*;


total_loss$)�@

error_R)�R?

learning_rate_1�9,+JI       6%�	oi)Q���A�*;


total_lossEh�@

error_R7Q?

learning_rate_1�9R9fI       6%�	��)Q���A�*;


total_loss�/A

error_R�Ya?

learning_rate_1�9)��{I       6%�	�*Q���A�*;


total_loss*��@

error_R�]M?

learning_rate_1�9�!�oI       6%�	i`*Q���A�*;


total_loss���@

error_Rs�P?

learning_rate_1�9�jƼI       6%�	v�*Q���A�*;


total_loss8��@

error_R �B?

learning_rate_1�9QazI       6%�	��*Q���A�*;


total_loss��@

error_R�O?

learning_rate_1�9��E�I       6%�	�7+Q���A�*;


total_loss��@

error_R��F?

learning_rate_1�9�7�I       6%�	�~+Q���A�*;


total_loss؆�@

error_R �F?

learning_rate_1�9<�I'I       6%�	u�+Q���A�*;


total_losso�@

error_R�cJ?

learning_rate_1�9ǡ��I       6%�	�	,Q���A�*;


total_loss��@

error_R��J?

learning_rate_1�9��TI       6%�	�U,Q���A�*;


total_loss}��@

error_R�5S?

learning_rate_1�9vAUI       6%�	��,Q���A�*;


total_lossl-�@

error_R��`?

learning_rate_1�9y��bI       6%�	9�,Q���A�*;


total_loss{ʹ@

error_RSOF?

learning_rate_1�9�2^�I       6%�	�*-Q���A�*;


total_loss/-�@

error_R��G?

learning_rate_1�99���I       6%�	�n-Q���A�*;


total_lossR�@

error_Ro�=?

learning_rate_1�9(�шI       6%�	�-Q���A�*;


total_loss�C�@

error_RW�C?

learning_rate_1�9(G��I       6%�	�-Q���A�*;


total_lossŝZ@

error_R�Q?

learning_rate_1�9U��I       6%�	n<.Q���A�*;


total_loss{��@

error_R8�Y?

learning_rate_1�9*�pI       6%�	�~.Q���A�*;


total_loss6�A

error_R�H?

learning_rate_1�9|�>I       6%�	��.Q���A�*;


total_loss_�@

error_R�]?

learning_rate_1��8���I       6%�	L/Q���A�*;


total_loss���@

error_R�3S?

learning_rate_1��8�'�I       6%�	7I/Q���A�*;


total_loss|7�@

error_R�V?

learning_rate_1��8�	�I       6%�	��/Q���A�*;


total_lossh�@

error_R6�>?

learning_rate_1��8�ZH*I       6%�	��/Q���A�*;


total_loss:X�@

error_Rwb?

learning_rate_1��8�J��I       6%�	�0Q���A�*;


total_lossv��@

error_RVQ?

learning_rate_1��8g�zeI       6%�	:S3Q���A�*;


total_loss���@

error_R8P?

learning_rate_1��8dF�dI       6%�	��3Q���A�*;


total_loss��@

error_Rf-S?

learning_rate_1��8�1�I       6%�	��3Q���A�*;


total_lossѥ@

error_RåK?

learning_rate_1��8��pI       6%�	b84Q���A�*;


total_loss�@

error_R��@?

learning_rate_1��8V>��I       6%�	ʁ4Q���A�*;


total_loss�֠@

error_R&YP?

learning_rate_1��8�I       6%�	��4Q���A�*;


total_loss��@

error_R#PQ?

learning_rate_1��8�Ft�I       6%�	5Q���A�*;


total_loss�mw@

error_R�"M?

learning_rate_1��8gd�1I       6%�	�V5Q���A�*;


total_loss@r�@

error_R.kB?

learning_rate_1��8LB�I       6%�	I�5Q���A�*;


total_loss�ў@

error_R�b[?

learning_rate_1��8�v�I       6%�	��5Q���A�*;


total_loss ��@

error_R�A?

learning_rate_1��8��4I       6%�	�76Q���A�*;


total_loss�B�@

error_R)@`?

learning_rate_1��8o�u�I       6%�	��6Q���A�*;


total_lossJ/�@

error_RV?

learning_rate_1��8/�7�I       6%�	�6Q���A�*;


total_loss�=�@

error_R�P?

learning_rate_1��8}<"I       6%�	r7Q���A�*;


total_loss���@

error_Rê[?

learning_rate_1��8�{�I       6%�	K7Q���A�*;


total_loss���@

error_R��T?

learning_rate_1��88�XDI       6%�	c�7Q���A�*;


total_loss�V�@

error_R1�T?

learning_rate_1��8�'D�I       6%�	`�7Q���A�*;


total_loss���@

error_Ri�U?

learning_rate_1��8-p��I       6%�	�8Q���A�*;


total_lossQ�@A

error_R��@?

learning_rate_1��8\0a�I       6%�	X8Q���A�*;


total_loss�R@

error_R��C?

learning_rate_1��8�T�I       6%�	ɛ8Q���A�*;


total_loss��@

error_R.Q?

learning_rate_1��8�E[I       6%�	�8Q���A�*;


total_loss��
A

error_R�\?

learning_rate_1��84T�dI       6%�	["9Q���A�*;


total_losss5�@

error_Ro�M?

learning_rate_1��86N�%I       6%�	�d9Q���A�*;


total_loss<��@

error_RVwS?

learning_rate_1��8��=I       6%�	f�9Q���A�*;


total_loss�@

error_R��O?

learning_rate_1��8� �I       6%�	�:Q���A�*;


total_loss��@

error_RcB?

learning_rate_1��8䴏�I       6%�	�V:Q���A�*;


total_loss��@

error_RؖL?

learning_rate_1��8�0�I       6%�	�:Q���A�*;


total_lossZ�q@

error_R]D?

learning_rate_1��8�<I       6%�	��:Q���A�*;


total_loss��@

error_R�FV?

learning_rate_1��8���I       6%�	o+;Q���A�*;


total_loss��@

error_RWE?

learning_rate_1��8���<I       6%�	�q;Q���A�*;


total_loss�	A

error_RHE?

learning_rate_1��8��CI       6%�	ִ;Q���A�*;


total_lossJ�@

error_R�dI?

learning_rate_1��8�M&�I       6%�	��;Q���A�*;


total_loss1(�@

error_Ra�[?

learning_rate_1��8�M�I       6%�	`><Q���A�*;


total_lossf`�@

error_R�`?

learning_rate_1��8U��I       6%�	K�<Q���A�*;


total_loss�o�@

error_RJG?

learning_rate_1��8z*�I       6%�	��<Q���A�*;


total_lossܘ1A

error_R�X?

learning_rate_1��8��$�I       6%�	�=Q���A�*;


total_loss���@

error_R�(N?

learning_rate_1��8����I       6%�	�Q=Q���A�*;


total_lossEe�@

error_R�O?

learning_rate_1��8���|I       6%�	�=Q���A�*;


total_lossTy�@

error_R�U;?

learning_rate_1��8bn(I       6%�	8�=Q���A�*;


total_lossq��@

error_R�J?

learning_rate_1��8'b�I       6%�	�D>Q���A�*;


total_lossy�@

error_R��R?

learning_rate_1��8���I       6%�	}�>Q���A�*;


total_loss���@

error_R�^X?

learning_rate_1��8�|/hI       6%�	��>Q���A�*;


total_loss� �@

error_R��D?

learning_rate_1��8��� I       6%�	-?Q���A�*;


total_loss� A

error_R�H?

learning_rate_1��8gq�I       6%�	�?Q���A�*;


total_loss�,�@

error_R:�;?

learning_rate_1��83�&�I       6%�	�?Q���A�*;


total_loss�>�@

error_R �M?

learning_rate_1��8@�RI       6%�	�@Q���A�*;


total_lossx:�@

error_R�1_?

learning_rate_1��8�Q�I       6%�	։@Q���A�*;


total_loss��@

error_R�?O?

learning_rate_1��8�%�I       6%�	��@Q���A�*;


total_loss1��@

error_R�Q?

learning_rate_1��8�'^�I       6%�	h&AQ���A�*;


total_lossiĦ@

error_RYW?

learning_rate_1��8.zEI       6%�	ǍAQ���A�*;


total_loss�Q�@

error_Ro�I?

learning_rate_1��8�U�I       6%�	��AQ���A�*;


total_loss(�@

error_R{W?

learning_rate_1��8����I       6%�	!BQ���A�*;


total_loss	˦@

error_R�G?

learning_rate_1��8�Џ;I       6%�	�fBQ���A�*;


total_lossSA

error_RlSQ?

learning_rate_1��8<yTI       6%�	��BQ���A�*;


total_loss%�;@

error_R��G?

learning_rate_1��8	f[�I       6%�	�BQ���A�*;


total_lossS�@

error_R�0G?

learning_rate_1��8`�AI       6%�	zKCQ���A�*;


total_loss!�@

error_Rq�F?

learning_rate_1��8ub��I       6%�	J�CQ���A�*;


total_loss�@

error_R7�[?

learning_rate_1��8mF��I       6%�	��CQ���A�*;


total_loss���@

error_Rx�@?

learning_rate_1��8g��I       6%�	EDQ���A�*;


total_loss:��@

error_RJ�M?

learning_rate_1��8X78�I       6%�	��DQ���A�*;


total_loss��@

error_RݥJ?

learning_rate_1��8��'I       6%�	��DQ���A�*;


total_loss� A

error_R�?F?

learning_rate_1��88l��I       6%�	3EQ���A�*;


total_loss��@

error_RljI?

learning_rate_1��8���I       6%�	.UEQ���A�*;


total_loss3�@

error_R8�D?

learning_rate_1��8�db4I       6%�	R�EQ���A�*;


total_loss*s@

error_R�lV?

learning_rate_1��8�@��I       6%�	�EQ���A�*;


total_loss��@

error_R��M?

learning_rate_1��8�Y�I       6%�	�1FQ���A�*;


total_loss��@

error_R�S?

learning_rate_1��8�#I       6%�	G{FQ���A�*;


total_loss���@

error_R�J?

learning_rate_1��8���I       6%�	��FQ���A�*;


total_lossu��@

error_R2�C?

learning_rate_1��8Q���I       6%�	�
GQ���A�*;


total_loss�_r@

error_R�M?

learning_rate_1��8�MI       6%�	PGQ���A�*;


total_lossxs�@

error_RR�R?

learning_rate_1��8��U�I       6%�	��GQ���A�*;


total_loss�`�@

error_RʹZ?

learning_rate_1��8ۑI       6%�	n�GQ���A�*;


total_loss�.�@

error_R�GP?

learning_rate_1��8u�I       6%�	�HQ���A�*;


total_loss?~�@

error_R��Z?

learning_rate_1��8.��I       6%�	I\HQ���A�*;


total_loss<��@

error_Rd/K?

learning_rate_1��8��:�I       6%�	8�HQ���A�*;


total_loss�3�@

error_RO�e?

learning_rate_1��8���I       6%�	�HQ���A�*;


total_loss_��@

error_R��N?

learning_rate_1��8�(�I       6%�	�/IQ���A�*;


total_loss<�@

error_R?{P?

learning_rate_1��8kZ�I       6%�	�qIQ���A�*;


total_lossj��@

error_R�{P?

learning_rate_1��8�sd�I       6%�	��IQ���A�*;


total_loss���@

error_Ri`?

learning_rate_1��8��_?I       6%�	�JQ���A�*;


total_loss���@

error_R��X?

learning_rate_1��8�:܀I       6%�	`JQ���A�*;


total_loss���@

error_R&�U?

learning_rate_1��8x~��I       6%�	�JQ���A�*;


total_loss��@

error_R�_?

learning_rate_1��8.��I       6%�	�JQ���A�*;


total_loss���@

error_RNMJ?

learning_rate_1��8��BI       6%�	�2KQ���A�*;


total_lossD��@

error_Rt�E?

learning_rate_1��8��"�I       6%�	<uKQ���A�*;


total_loss�$�@

error_R��D?

learning_rate_1��8���I       6%�	��KQ���A�*;


total_loss즈@

error_RC�f?

learning_rate_1��8�!��I       6%�	�LQ���A�*;


total_loss���@

error_RO�=?

learning_rate_1��8���2I       6%�	`ILQ���A�*;


total_loss_�@

error_R�U?

learning_rate_1��8[)hbI       6%�	:�LQ���A�*;


total_loss�Ա@

error_Rf:H?

learning_rate_1��8V[@I       6%�	~�LQ���A�*;


total_loss]�@

error_R�=?

learning_rate_1��8�g��I       6%�	)MQ���A�*;


total_loss��@

error_R3�Y?

learning_rate_1��8�]4HI       6%�	[MQ���A�*;


total_lossZIu@

error_R(H?

learning_rate_1��8�}�I       6%�	ΠMQ���A�*;


total_loss�XA

error_R /Z?

learning_rate_1��8Wa��I       6%�	�MQ���A�*;


total_losst�@

error_R�rN?

learning_rate_1��8�(��I       6%�	�7NQ���A�*;


total_loss�a�@

error_R��T?

learning_rate_1��8�N��I       6%�	�}NQ���A� *;


total_lossr�@

error_R��L?

learning_rate_1��8��oI       6%�	��NQ���A� *;


total_lossJKA

error_R��I?

learning_rate_1��8�)rI       6%�	hOQ���A� *;


total_loss(Y�@

error_RvN?

learning_rate_1��8K���I       6%�	�JOQ���A� *;


total_loss��@

error_R��>?

learning_rate_1��8��{I       6%�	^�OQ���A� *;


total_lossŉ�@

error_RE
I?

learning_rate_1��8ન�I       6%�	��OQ���A� *;


total_loss4~�@

error_R�kL?

learning_rate_1��8��%lI       6%�	�PQ���A� *;


total_loss�Yt@

error_R��C?

learning_rate_1��8
�=I       6%�	kYPQ���A� *;


total_loss/X�@

error_R;�@?

learning_rate_1��8��K	I       6%�	I�PQ���A� *;


total_loss@�@

error_R��G?

learning_rate_1��8��PI       6%�	Q�PQ���A� *;


total_loss(�t@

error_R��X?

learning_rate_1��8}�֤I       6%�	/!QQ���A� *;


total_lossԦ�@

error_R��R?

learning_rate_1��8�tBI       6%�	�cQQ���A� *;


total_loss$��@

error_R��`?

learning_rate_1��8�m�cI       6%�	�QQ���A� *;


total_lossT�@

error_R�T?

learning_rate_1��8d�%I       6%�	��QQ���A� *;


total_lossZ��@

error_R7M?

learning_rate_1��8�Yl5I       6%�	�JRQ���A� *;


total_lossʔA

error_R8C\?

learning_rate_1��8��q�I       6%�	��RQ���A� *;


total_loss3�@

error_R��S?

learning_rate_1��8	l��I       6%�	��RQ���A� *;


total_lossIҰ@

error_R��K?

learning_rate_1��8	��I       6%�	�SQ���A� *;


total_lossLQ|@

error_R۽E?

learning_rate_1��8�7�I       6%�	ugSQ���A� *;


total_lossH��@

error_R�V]?

learning_rate_1��8t��I       6%�	e�SQ���A� *;


total_loss���@

error_R7�c?

learning_rate_1��8�B��I       6%�		�SQ���A� *;


total_loss
?�@

error_R��H?

learning_rate_1��8U�bkI       6%�	>TQ���A� *;


total_loss�Q}@

error_R�!K?

learning_rate_1��8���I       6%�	��TQ���A� *;


total_lossD�@

error_R�<B?

learning_rate_1��8v#9�I       6%�	�TQ���A� *;


total_loss���@

error_R �B?

learning_rate_1��8�z�6I       6%�	mUQ���A� *;


total_loss�`�@

error_R�M?

learning_rate_1��8�!LII       6%�	2YUQ���A� *;


total_loss��@

error_R�iO?

learning_rate_1��8���I       6%�	ƥUQ���A� *;


total_loss���@

error_R�/T?

learning_rate_1��8�;�LI       6%�	��UQ���A� *;


total_lossqp�@

error_R�O??

learning_rate_1��8Q��xI       6%�	�0VQ���A� *;


total_loss�}A

error_R�_b?

learning_rate_1��8����I       6%�	rVQ���A� *;


total_loss?��@

error_RC�G?

learning_rate_1��8�y�I       6%�	߲VQ���A� *;


total_loss̇�@

error_R��@?

learning_rate_1��8���AI       6%�	��VQ���A� *;


total_loss�V�@

error_RݰM?

learning_rate_1��8���UI       6%�	)8WQ���A� *;


total_loss!�@

error_RgK?

learning_rate_1��8�z]�I       6%�	'zWQ���A� *;


total_lossjQ�@

error_R��Q?

learning_rate_1��8j7o�I       6%�	��WQ���A� *;


total_loss̉�@

error_RL@3?

learning_rate_1��8��)I       6%�	QXQ���A� *;


total_lossrթ@

error_R�S?

learning_rate_1��8�kI       6%�	�NXQ���A� *;


total_loss�ʔ@

error_Rf�H?

learning_rate_1��8m>>CI       6%�	e�XQ���A� *;


total_loss���@

error_R�[?

learning_rate_1��8����I       6%�	��XQ���A� *;


total_loss�[�@

error_RW1G?

learning_rate_1��8zk��I       6%�	k#YQ���A� *;


total_loss\L�@

error_R�Z?

learning_rate_1��8o�(I       6%�	ieYQ���A� *;


total_lossWf�@

error_R�"??

learning_rate_1��8���I       6%�	��YQ���A� *;


total_lossJ�%A

error_R�l`?

learning_rate_1��8N-�I       6%�	�ZQ���A� *;


total_lossD��@

error_R͌\?

learning_rate_1��8���I       6%�	�PZQ���A� *;


total_loss���@

error_R��A?

learning_rate_1��8�=��I       6%�	�ZQ���A� *;


total_loss���@

error_R�A?

learning_rate_1��8�t��I       6%�	��ZQ���A� *;


total_lossX�@

error_R�gJ?

learning_rate_1��8��h>I       6%�	H[Q���A� *;


total_lossѬ�@

error_RZ%D?

learning_rate_1��8�b�2I       6%�	�][Q���A� *;


total_loss��@

error_R�7U?

learning_rate_1��8���I       6%�	ȟ[Q���A� *;


total_lossv��@

error_R�nR?

learning_rate_1��8L ��I       6%�	��[Q���A� *;


total_loss$1�@

error_R��J?

learning_rate_1��8��{GI       6%�	�$\Q���A� *;


total_lossV��@

error_R��a?

learning_rate_1��8��T�I       6%�	�e\Q���A� *;


total_loss���@

error_R�GP?

learning_rate_1��8	I       6%�	c�\Q���A� *;


total_loss���@

error_R�GI?

learning_rate_1��8��j�I       6%�	0�\Q���A� *;


total_loss�r�@

error_R��]?

learning_rate_1��8]'j�I       6%�	�:]Q���A� *;


total_lossm�A

error_RWT?

learning_rate_1��8��6II       6%�	ׂ]Q���A� *;


total_loss�{�@

error_R(zV?

learning_rate_1��8e\��I       6%�	��]Q���A� *;


total_loss6Y�@

error_R��V?

learning_rate_1��8�vI       6%�	�^Q���A� *;


total_lossQ��@

error_R(�Q?

learning_rate_1��8@k�I       6%�	�Z^Q���A� *;


total_lossE��@

error_R
D?

learning_rate_1��8�lŐI       6%�	��^Q���A� *;


total_loss��@

error_Rv�N?

learning_rate_1��8E2K=I       6%�	��^Q���A� *;


total_lossv4�@

error_R� O?

learning_rate_1��8����I       6%�	v$_Q���A� *;


total_loss[<�@

error_R��C?

learning_rate_1��8���I       6%�	�e_Q���A� *;


total_loss���@

error_R��`?

learning_rate_1��8�~�I       6%�	��_Q���A� *;


total_loss
J�@

error_Rxr^?

learning_rate_1��8�&��I       6%�	��_Q���A� *;


total_loss��@

error_R�oZ?

learning_rate_1��8�K�I       6%�	y,`Q���A� *;


total_loss/0�@

error_RN�R?

learning_rate_1��8�1KbI       6%�	o`Q���A� *;


total_loss�@

error_R߶]?

learning_rate_1��8���I       6%�	0�`Q���A� *;


total_loss���@

error_R� K?

learning_rate_1��8IB��I       6%�	��`Q���A� *;


total_loss2A

error_RJ�S?

learning_rate_1��8�^
?I       6%�	22aQ���A� *;


total_lossHA

error_R@�Y?

learning_rate_1��8!REI       6%�	�uaQ���A� *;


total_lossV��@

error_R�J?

learning_rate_1��822I       6%�	ݸaQ���A� *;


total_loss���@

error_Rs�@?

learning_rate_1��8�4��I       6%�	/�aQ���A� *;


total_lossֽ@

error_R,>B?

learning_rate_1��8y]xI       6%�	>FbQ���A� *;


total_lossȏ<A

error_R?E?

learning_rate_1��8$X��I       6%�	�bQ���A� *;


total_losstɏ@

error_R9M?

learning_rate_1��8y�W�I       6%�	��bQ���A� *;


total_loss;A

error_R�V?

learning_rate_1��8��o�I       6%�	�	cQ���A� *;


total_loss�;�@

error_R��U?

learning_rate_1��8W��I       6%�	OcQ���A� *;


total_loss�ڒ@

error_Rr�\?

learning_rate_1��8vh��I       6%�	ӕcQ���A� *;


total_loss�#�@

error_R�M??

learning_rate_1��8F��[I       6%�	a�cQ���A� *;


total_loss�i�@

error_RÚH?

learning_rate_1��8`��pI       6%�	MdQ���A� *;


total_loss�@

error_R7�L?

learning_rate_1��8W�!I       6%�	�^dQ���A� *;


total_loss�_�@

error_Rn�=?

learning_rate_1��8���I       6%�	t�dQ���A� *;


total_loss�,�@

error_R&}F?

learning_rate_1��8�c��I       6%�	&�dQ���A� *;


total_lossrC�@

error_R�I?

learning_rate_1��8X�K?I       6%�	h&eQ���A� *;


total_lossJ��@

error_R��\?

learning_rate_1��8=R�'I       6%�	�ieQ���A� *;


total_loss\9�@

error_Rs�[?

learning_rate_1��8���I       6%�	��eQ���A� *;


total_loss:Ϭ@

error_R�K?

learning_rate_1��8�s�<I       6%�	��eQ���A� *;


total_loss�U�@

error_RH`Y?

learning_rate_1��8��0I       6%�	:fQ���A� *;


total_loss��@

error_RE�Y?

learning_rate_1��8�[-I       6%�	��fQ���A� *;


total_loss�#�@

error_RV?

learning_rate_1��8��k�I       6%�	��fQ���A� *;


total_loss��@

error_R��_?

learning_rate_1��8��VI       6%�	�gQ���A� *;


total_losso�@

error_RJ;C?

learning_rate_1��8ol�I       6%�	5]gQ���A� *;


total_lossSP�@

error_R\�P?

learning_rate_1��8���I       6%�	z�gQ���A� *;


total_loss��@

error_R��[?

learning_rate_1��8�!��I       6%�	�gQ���A� *;


total_lossL�A

error_R,�N?

learning_rate_1��8���NI       6%�	N1hQ���A� *;


total_loss���@

error_R��H?

learning_rate_1��8ݰ�I       6%�	�rhQ���A� *;


total_loss�P�@

error_R�$\?

learning_rate_1��86-{0I       6%�	�hQ���A� *;


total_loss[s@

error_R��@?

learning_rate_1��8ъu�I       6%�	m�hQ���A� *;


total_loss@��@

error_RR�N?

learning_rate_1��82Q�I       6%�	�;iQ���A� *;


total_loss�1�@

error_R=3F?

learning_rate_1��8Yuz�I       6%�	�iQ���A� *;


total_loss�:�@

error_R��<?

learning_rate_1��8x���I       6%�	��iQ���A� *;


total_lossAf�@

error_R[?

learning_rate_1��8���HI       6%�	l.jQ���A� *;


total_loss�
�@

error_R��Q?

learning_rate_1��8 ��I       6%�	tjQ���A� *;


total_loss���@

error_R�N?

learning_rate_1��8j��_I       6%�	�jQ���A� *;


total_loss6A

error_R�lP?

learning_rate_1��8��T�I       6%�	��jQ���A� *;


total_loss9T�@

error_R�S?

learning_rate_1��8�4�I       6%�	�FkQ���A� *;


total_loss��@

error_R�;?

learning_rate_1��8Mr�I       6%�	ˉkQ���A� *;


total_lossi��@

error_RH�E?

learning_rate_1��8>�!I       6%�	��kQ���A� *;


total_loss�A

error_R��^?

learning_rate_1��8�+o8I       6%�	KlQ���A� *;


total_loss�A�@

error_R�5M?

learning_rate_1��81h�I       6%�	�alQ���A� *;


total_loss��^@

error_RVkM?

learning_rate_1��8�T"II       6%�	ˣlQ���A� *;


total_loss<�A

error_R��Q?

learning_rate_1��8���|I       6%�	�lQ���A� *;


total_lossq~�@

error_R��P?

learning_rate_1��8L�I       6%�	3mQ���A� *;


total_loss���@

error_R�~a?

learning_rate_1��8��V�I       6%�	$�mQ���A� *;


total_loss���@

error_Rq@?

learning_rate_1��8ԋ�-I       6%�	"�mQ���A� *;


total_loss��G@

error_R K?

learning_rate_1��8�4I       6%�	�nQ���A� *;


total_lossFۊ@

error_RTjL?

learning_rate_1��8G⠄I       6%�	jTnQ���A� *;


total_loss�A

error_RZQ?

learning_rate_1��8��%I       6%�	��nQ���A� *;


total_lossVh�@

error_R(D?

learning_rate_1��8�Z�=I       6%�	
�nQ���A� *;


total_lossE��@

error_R�<[?

learning_rate_1��8�Sx�I       6%�	?'oQ���A� *;


total_loss��x@

error_R�_?

learning_rate_1��8��I       6%�	�roQ���A� *;


total_lossO��@

error_R}<g?

learning_rate_1��8җ@I       6%�	��oQ���A� *;


total_loss���@

error_R��U?

learning_rate_1��87 )�I       6%�	��oQ���A� *;


total_lossF��@

error_R6[^?

learning_rate_1��8�I       6%�	�@pQ���A� *;


total_loss�
�@

error_R�DX?

learning_rate_1��8O�I       6%�	ćpQ���A� *;


total_loss�BA

error_R�T?

learning_rate_1��8�r�(I       6%�	,�pQ���A� *;


total_loss@

error_R�PQ?

learning_rate_1��8��I       6%�	|qQ���A� *;


total_loss[�@

error_R�nS?

learning_rate_1��8��%�I       6%�	JeqQ���A�!*;


total_loss��@

error_R�B?

learning_rate_1��8RC�I       6%�	ۮqQ���A�!*;


total_loss�g�@

error_R=K?

learning_rate_1��8����I       6%�	_�qQ���A�!*;


total_loss��@

error_R�3??

learning_rate_1��8*g�!I       6%�	oOrQ���A�!*;


total_loss��@

error_R߂J?

learning_rate_1��8�O�eI       6%�	R�rQ���A�!*;


total_loss���@

error_Ri�Q?

learning_rate_1��8�{=I       6%�	P�rQ���A�!*;


total_loss��+A

error_RN?F?

learning_rate_1��8�GI       6%�	}*sQ���A�!*;


total_loss��@

error_R�wR?

learning_rate_1��8DC�I       6%�	��sQ���A�!*;


total_lossϬ�@

error_RXV?

learning_rate_1��8���I       6%�	��sQ���A�!*;


total_loss3��@

error_RvRB?

learning_rate_1��83(i�I       6%�	�tQ���A�!*;


total_loss���@

error_R�L?

learning_rate_1��8���I       6%�	�ctQ���A�!*;


total_loss��@

error_R�:M?

learning_rate_1��8��@uI       6%�	r�tQ���A�!*;


total_loss���@

error_R�oG?

learning_rate_1��8��^I       6%�	��tQ���A�!*;


total_loss��A

error_R͟X?

learning_rate_1��8��z/I       6%�	?uQ���A�!*;


total_lossଌ@

error_Ri�F?

learning_rate_1��8[�2-I       6%�	��uQ���A�!*;


total_loss��A

error_RM�E?

learning_rate_1��8��I       6%�	��uQ���A�!*;


total_loss_��@

error_R�jb?

learning_rate_1��8�k�{I       6%�	�vQ���A�!*;


total_loss�5�@

error_Rd�>?

learning_rate_1��8ۮ҈I       6%�	�^vQ���A�!*;


total_lossa_�@

error_RHL?

learning_rate_1��8V��I       6%�	&�vQ���A�!*;


total_loss3��@

error_R�O?

learning_rate_1��8�ptI       6%�	��vQ���A�!*;


total_loss4/�@

error_R�(G?

learning_rate_1��8�ވ�I       6%�	1wQ���A�!*;


total_loss`d�@

error_REOG?

learning_rate_1��8��I       6%�	swQ���A�!*;


total_loss�=�@

error_RVNQ?

learning_rate_1��8p��I       6%�	��wQ���A�!*;


total_loss*�@

error_R�RQ?

learning_rate_1��8�3_7I       6%�	��wQ���A�!*;


total_loss�T�@

error_Rw*U?

learning_rate_1��8��ӮI       6%�	[ExQ���A�!*;


total_loss�z�@

error_R2�I?

learning_rate_1��8���I       6%�	�xQ���A�!*;


total_loss��@

error_R��V?

learning_rate_1��8M[��I       6%�	�xQ���A�!*;


total_loss <A

error_R�HP?

learning_rate_1��8����I       6%�	�yQ���A�!*;


total_loss���@

error_R��S?

learning_rate_1��8f��I       6%�	YQyQ���A�!*;


total_lossv�@

error_R&�??

learning_rate_1��8J���I       6%�	ܑyQ���A�!*;


total_loss�c�@

error_R��L?

learning_rate_1��8E��I       6%�	��yQ���A�!*;


total_loss��@

error_R��]?

learning_rate_1��8�k�mI       6%�	�;zQ���A�!*;


total_lossϳJ@

error_R_>W?

learning_rate_1��8�oz�I       6%�	�zQ���A�!*;


total_loss���@

error_R�8N?

learning_rate_1��8�C�ZI       6%�	��zQ���A�!*;


total_lossq��@

error_R�#Y?

learning_rate_1��8bK�?I       6%�	�
{Q���A�!*;


total_loss���@

error_Rf}U?

learning_rate_1��8�#gI       6%�	Q{Q���A�!*;


total_loss,c�@

error_R�#S?

learning_rate_1��8�b'�I       6%�	�{Q���A�!*;


total_loss���@

error_Rb?

learning_rate_1��80u��I       6%�	��{Q���A�!*;


total_loss8��@

error_R�>?

learning_rate_1��8�-��I       6%�	�G|Q���A�!*;


total_loss�@

error_R�D]?

learning_rate_1��884�I       6%�	|�|Q���A�!*;


total_loss�m�@

error_RW�H?

learning_rate_1��86I�]I       6%�	�|Q���A�!*;


total_lossn/�@

error_R$�I?

learning_rate_1��82S�I       6%�	Z}Q���A�!*;


total_loss,��@

error_R*34?

learning_rate_1��8\�2�I       6%�	�`}Q���A�!*;


total_loss���@

error_R�S?

learning_rate_1��8klI       6%�	�}Q���A�!*;


total_loss�+�@

error_R��K?

learning_rate_1��8/��'I       6%�	2�}Q���A�!*;


total_loss��A

error_R�HH?

learning_rate_1��8���I       6%�	�1~Q���A�!*;


total_loss���@

error_R(N?

learning_rate_1��8�K9^I       6%�	�u~Q���A�!*;


total_loss���@

error_R�f?

learning_rate_1��8|(�I       6%�	p�~Q���A�!*;


total_loss��@

error_Rv\D?

learning_rate_1��8�?G�I       6%�	�~Q���A�!*;


total_loss�ˍ@

error_Rl�I?

learning_rate_1��8/�I       6%�	K>Q���A�!*;


total_loss���@

error_RV5E?

learning_rate_1��86u�8I       6%�	��Q���A�!*;


total_loss�.�@

error_R!�;?

learning_rate_1��8����I       6%�	��Q���A�!*;


total_loss��@

error_R !`?

learning_rate_1��8���FI       6%�	y�Q���A�!*;


total_loss͕�@

error_R�Y?

learning_rate_1��8|6��I       6%�	'J�Q���A�!*;


total_loss]G�@

error_R[�Y?

learning_rate_1��8��2I       6%�	���Q���A�!*;


total_loss�g�@

error_R��J?

learning_rate_1��8��&I       6%�	q׀Q���A�!*;


total_loss��@

error_R��_?

learning_rate_1��8���I       6%�	A�Q���A�!*;


total_loss@��@

error_R�NS?

learning_rate_1��8���I       6%�	]b�Q���A�!*;


total_loss�ۦ@

error_RW0T?

learning_rate_1��8���I       6%�	g��Q���A�!*;


total_lossҢ�@

error_R?uX?

learning_rate_1��8�3I       6%�	�Q���A�!*;


total_loss�õ@

error_R{�U?

learning_rate_1��8t'�^I       6%�	2�Q���A�!*;


total_loss�8�@

error_R�R?

learning_rate_1��8�R�I       6%�	�u�Q���A�!*;


total_loss{�@

error_Rә1?

learning_rate_1��8��:I       6%�	��Q���A�!*;


total_loss�|�@

error_R9k?

learning_rate_1��8�2�>I       6%�		�Q���A�!*;


total_loss�~�@

error_R��L?

learning_rate_1��8�j��I       6%�	{J�Q���A�!*;


total_loss���@

error_R!�_?

learning_rate_1��8*R�@I       6%�	.��Q���A�!*;


total_loss�A

error_R��H?

learning_rate_1��8�(�fI       6%�	�ЃQ���A�!*;


total_lossf8A

error_R�xM?

learning_rate_1��8�g�I       6%�	n�Q���A�!*;


total_lossw٭@

error_R�QZ?

learning_rate_1��8�?��I       6%�	�_�Q���A�!*;


total_lossK#A

error_R��<?

learning_rate_1��8�wJI       6%�	��Q���A�!*;


total_loss]�@

error_RT�??

learning_rate_1��8���I       6%�	��Q���A�!*;


total_loss�ߥ@

error_R�*X?

learning_rate_1��8x*'I       6%�	Q*�Q���A�!*;


total_loss���@

error_RXiH?

learning_rate_1��8�-�I       6%�	�o�Q���A�!*;


total_loss݈�@

error_RMQ?

learning_rate_1��8+�;I       6%�	C��Q���A�!*;


total_loss]�A

error_R�K?

learning_rate_1��8C���I       6%�	2�Q���A�!*;


total_loss5�@

error_RZ�T?

learning_rate_1��8��iI       6%�	/f�Q���A�!*;


total_loss���@

error_RJc;?

learning_rate_1��8�#>I       6%�	ꩆQ���A�!*;


total_lossɆ�@

error_R&�K?

learning_rate_1��8"n��I       6%�	��Q���A�!*;


total_lossX��@

error_R�.B?

learning_rate_1��8��S�I       6%�	�/�Q���A�!*;


total_loss�(�@

error_R#.Z?

learning_rate_1��8��I       6%�	t�Q���A�!*;


total_loss��@

error_RڮY?

learning_rate_1��8wHU�I       6%�	c��Q���A�!*;


total_loss�+�@

error_R_�N?

learning_rate_1��8��"uI       6%�	 �Q���A�!*;


total_loss��@

error_R�wV?

learning_rate_1��86�ޞI       6%�	�B�Q���A�!*;


total_loss��A

error_R�O?

learning_rate_1��8�&�I       6%�	���Q���A�!*;


total_lossĲ�@

error_RM�Y?

learning_rate_1��8�^u	I       6%�	�ԈQ���A�!*;


total_lossG=	A

error_R�_?

learning_rate_1��8N���I       6%�	�%�Q���A�!*;


total_lossRy�@

error_R1S?

learning_rate_1��8��I       6%�	:��Q���A�!*;


total_lossmB�@

error_R��U?

learning_rate_1��8>�|I       6%�	t�Q���A�!*;


total_lossC��@

error_R1�Z?

learning_rate_1��8��BI       6%�	R�Q���A�!*;


total_loss\Y�@

error_R��N?

learning_rate_1��8|�<I       6%�	��Q���A�!*;


total_loss�5EA

error_R ~X?

learning_rate_1��8 V�I       6%�	f�Q���A�!*;


total_loss���@

error_Rh�U?

learning_rate_1��8�D8nI       6%�	�.�Q���A�!*;


total_loss�Y�@

error_R�X?

learning_rate_1��8�4�I       6%�	�w�Q���A�!*;


total_loss�J�@

error_RT�[?

learning_rate_1��8�+��I       6%�	T��Q���A�!*;


total_loss1��@

error_R�P?

learning_rate_1��8>>��I       6%�	a	�Q���A�!*;


total_loss���@

error_R�|Q?

learning_rate_1��8����I       6%�	�T�Q���A�!*;


total_loss��@

error_R6�\?

learning_rate_1��8�z�I       6%�	ƥ�Q���A�!*;


total_loss@W�@

error_RH�]?

learning_rate_1��8�0I       6%�	��Q���A�!*;


total_loss]�@

error_R[^?

learning_rate_1��8�$J�I       6%�	$3�Q���A�!*;


total_loss;��@

error_R&	V?

learning_rate_1��8�ǌ,I       6%�	't�Q���A�!*;


total_loss@��@

error_R�c?

learning_rate_1��8�gI       6%�	���Q���A�!*;


total_loss���@

error_RJ#O?

learning_rate_1��8�6��I       6%�	���Q���A�!*;


total_loss�ڒ@

error_R�`>?

learning_rate_1��8In�I       6%�	�A�Q���A�!*;


total_loss��@

error_RlsO?

learning_rate_1��8��^7I       6%�	'��Q���A�!*;


total_loss���@

error_R�(K?

learning_rate_1��8��:I       6%�	AɎQ���A�!*;


total_loss���@

error_R�??

learning_rate_1��8t�'oI       6%�	�/�Q���A�!*;


total_loss���@

error_R�:=?

learning_rate_1��8o���I       6%�	⩏Q���A�!*;


total_loss7b�@

error_R�gW?

learning_rate_1��8�v�I       6%�	��Q���A�!*;


total_loss���@

error_R��U?

learning_rate_1��8=�R�I       6%�	�[�Q���A�!*;


total_lossqv�@

error_R��D?

learning_rate_1��8�Z�I       6%�	ÐQ���A�!*;


total_loss|�@

error_R$�V?

learning_rate_1��8UT+�I       6%�	��Q���A�!*;


total_loss���@

error_R{HQ?

learning_rate_1��8��#I       6%�	Ud�Q���A�!*;


total_loss�֑@

error_RZ�M?

learning_rate_1��8)��qI       6%�	Ӳ�Q���A�!*;


total_loss���@

error_R�~K?

learning_rate_1��8w^�I       6%�	�Q���A�!*;


total_loss��@

error_R��X?

learning_rate_1��8�ڼnI       6%�	/x�Q���A�!*;


total_loss���@

error_R��`?

learning_rate_1��8GD{8I       6%�	���Q���A�!*;


total_lossO��@

error_R@�Z?

learning_rate_1��8�d`�I       6%�	�n�Q���A�!*;


total_loss:�@

error_RjyZ?

learning_rate_1��89���I       6%�	�ǓQ���A�!*;


total_loss��k@

error_R�9]?

learning_rate_1��8�B@�I       6%�	Z�Q���A�!*;


total_loss�C{@

error_R�C?

learning_rate_1��8wH��I       6%�	Xo�Q���A�!*;


total_loss��@

error_RsV?

learning_rate_1��8�,��I       6%�	`˔Q���A�!*;


total_loss��A

error_R�!I?

learning_rate_1��8h�Z�I       6%�	�(�Q���A�!*;


total_loss��@

error_R�`V?

learning_rate_1��8u���I       6%�	>��Q���A�!*;


total_loss�;�@

error_R�3Z?

learning_rate_1��8�ry^I       6%�	��Q���A�!*;


total_loss�@

error_RhNP?

learning_rate_1��8�5}�I       6%�	�E�Q���A�!*;


total_loss�9�@

error_R\�G?

learning_rate_1��8b�M�I       6%�	��Q���A�!*;


total_lossl�@

error_R�N;?

learning_rate_1��8���3I       6%�	�ߖQ���A�!*;


total_loss��@

error_RH:a?

learning_rate_1��8�&��I       6%�	�*�Q���A�!*;


total_loss�Ě@

error_R��J?

learning_rate_1��8��"kI       6%�	�x�Q���A�"*;


total_loss���@

error_RRXE?

learning_rate_1��8D��iI       6%�	3ӗQ���A�"*;


total_loss�@

error_Ra�]?

learning_rate_1��8HKi�I       6%�	�Q���A�"*;


total_lossneA

error_R=�]?

learning_rate_1��8����I       6%�	�h�Q���A�"*;


total_loss7s�@

error_R��X?

learning_rate_1��8�n7�I       6%�	i��Q���A�"*;


total_loss ��@

error_R�0?

learning_rate_1��8�ֻ�I       6%�	��Q���A�"*;


total_loss�c�@

error_R-�[?

learning_rate_1��8"m*0I       6%�	/R�Q���A�"*;


total_loss�O�@

error_R!V?

learning_rate_1��8ګ�sI       6%�	���Q���A�"*;


total_loss�ݰ@

error_Rm�F?

learning_rate_1��8�?.I       6%�		K�Q���A�"*;


total_loss4Y�@

error_RI�W?

learning_rate_1��8��I       6%�	{��Q���A�"*;


total_lossd�@

error_R{'U?

learning_rate_1��8�OI       6%�	:�Q���A�"*;


total_loss�*$A

error_R�H]?

learning_rate_1��8�N�[I       6%�	�L�Q���A�"*;


total_loss /�@

error_RI?

learning_rate_1��8'�I       6%�	њ�Q���A�"*;


total_losscvh@

error_RSQ?

learning_rate_1��8P��I       6%�	��Q���A�"*;


total_losszPA

error_R�UN?

learning_rate_1��8���+I       6%�	i+�Q���A�"*;


total_loss3'A

error_Rl_?

learning_rate_1��8%<*I       6%�	,r�Q���A�"*;


total_loss��@

error_R�8\?

learning_rate_1��8�J�;I       6%�	�ԜQ���A�"*;


total_lossd��@

error_R��N?

learning_rate_1��8oǣ|I       6%�	��Q���A�"*;


total_loss��@

error_R�d?

learning_rate_1��8�[�I       6%�	$^�Q���A�"*;


total_loss��@

error_R6�J?

learning_rate_1��8����I       6%�	R��Q���A�"*;


total_loss]/�@

error_R��K?

learning_rate_1��8�zI       6%�	��Q���A�"*;


total_loss���@

error_R!�E?

learning_rate_1��8�A��I       6%�	DL�Q���A�"*;


total_loss�ծ@

error_RtzI?

learning_rate_1��8���I       6%�	��Q���A�"*;


total_loss	�K@

error_R�@?

learning_rate_1��8��I       6%�	�ԞQ���A�"*;


total_loss�K�@

error_R
�h?

learning_rate_1��8F���I       6%�	w-�Q���A�"*;


total_loss�ݪ@

error_R	[?

learning_rate_1��8�?I       6%�	�s�Q���A�"*;


total_loss�$�@

error_Rd�V?

learning_rate_1��8˅Z�I       6%�	ݶ�Q���A�"*;


total_losstd�@

error_R@X?

learning_rate_1��8k�I       6%�	H��Q���A�"*;


total_lossAƶ@

error_R�N?

learning_rate_1��8i�^I       6%�	�>�Q���A�"*;


total_lossMi�@

error_R�M?

learning_rate_1��8Vm��I       6%�	��Q���A�"*;


total_loss�@

error_RfM?

learning_rate_1��8/.�I       6%�	6ȠQ���A�"*;


total_loss%,�@

error_Ri|T?

learning_rate_1��8��I       6%�	[�Q���A�"*;


total_lossf!�@

error_R�C?

learning_rate_1��8�)֑I       6%�	�R�Q���A�"*;


total_lossC��@

error_R�W?

learning_rate_1��8n�C�I       6%�	��Q���A�"*;


total_loss���@

error_RfvR?

learning_rate_1��8�I       6%�	2ܡQ���A�"*;


total_loss�a�@

error_R�9G?

learning_rate_1��8��S
I       6%�	$!�Q���A�"*;


total_loss�9�@

error_R��_?

learning_rate_1��8lvI       6%�	@d�Q���A�"*;


total_loss��@

error_R� \?

learning_rate_1��8�Z+I       6%�	ˤ�Q���A�"*;


total_loss�N�@

error_R�_1?

learning_rate_1��8)m'I       6%�	2�Q���A�"*;


total_loss;��@

error_R_�W?

learning_rate_1��8���^I       6%�	�,�Q���A�"*;


total_loss�=�@

error_R=PS?

learning_rate_1��8=T��I       6%�	Nz�Q���A�"*;


total_loss���@

error_R�$G?

learning_rate_1��8����I       6%�	�ģQ���A�"*;


total_loss R(A

error_R�)L?

learning_rate_1��8޷�|I       6%�		�Q���A�"*;


total_loss���@

error_R J?

learning_rate_1��81F�I       6%�	lX�Q���A�"*;


total_loss�\�@

error_R�`?

learning_rate_1��8��GQI       6%�	��Q���A�"*;


total_loss�؈@

error_R3NP?

learning_rate_1��8��(I       6%�	��Q���A�"*;


total_loss��@

error_R_9>?

learning_rate_1��8��<I       6%�	�7�Q���A�"*;


total_loss��@

error_Ro�O?

learning_rate_1��8��{I       6%�	���Q���A�"*;


total_loss���@

error_R��B?

learning_rate_1��8ni�3I       6%�	�ܥQ���A�"*;


total_loss�9�@

error_R;tX?

learning_rate_1��8XT
�I       6%�	x'�Q���A�"*;


total_loss,6�@

error_RWcF?

learning_rate_1��8�I       6%�	�n�Q���A�"*;


total_loss�E�@

error_R{d?

learning_rate_1��8�WЖI       6%�	���Q���A�"*;


total_loss�'�@

error_Rt�??

learning_rate_1��8��c�I       6%�	� �Q���A�"*;


total_loss�o�@

error_R��K?

learning_rate_1��8TRLI       6%�	�K�Q���A�"*;


total_loss11�@

error_RSA?

learning_rate_1��8w�I       6%�	{��Q���A�"*;


total_lossj��@

error_R,�]?

learning_rate_1��8L�`AI       6%�	:��Q���A�"*;


total_loss]ZA

error_R�CO?

learning_rate_1��8�]QPI       6%�	(X�Q���A�"*;


total_lossl�@

error_R��??

learning_rate_1��8"!�BI       6%�	���Q���A�"*;


total_lossL��@

error_R�;?

learning_rate_1��8���I       6%�	��Q���A�"*;


total_loss^�	A

error_R��O?

learning_rate_1��8��CI       6%�		8�Q���A�"*;


total_loss��@

error_RfoE?

learning_rate_1��8��)I       6%�	܃�Q���A�"*;


total_lossh��@

error_R�cO?

learning_rate_1��8S�h8I       6%�	��Q���A�"*;


total_lossL��@

error_RW=R?

learning_rate_1��8���	I       6%�	"8�Q���A�"*;


total_lossw�@

error_R��F?

learning_rate_1��82���I       6%�	"}�Q���A�"*;


total_losss,�@

error_R��N?

learning_rate_1��8
X�I       6%�	���Q���A�"*;


total_loss���@

error_R	DD?

learning_rate_1��8�+��I       6%�	*�Q���A�"*;


total_lossє�@

error_R��I?

learning_rate_1��8R=��I       6%�	4H�Q���A�"*;


total_lossȞ�@

error_R6�A?

learning_rate_1��8�E�zI       6%�	��Q���A�"*;


total_loss:��@

error_R*�O?

learning_rate_1��8�7QFI       6%�	ϫQ���A�"*;


total_loss܎�@

error_RW�S?

learning_rate_1��8|�؛I       6%�	U�Q���A�"*;


total_loss��@

error_R��^?

learning_rate_1��8��y�I       6%�	�X�Q���A�"*;


total_loss}��@

error_R�+Y?

learning_rate_1��8�c:I       6%�	u��Q���A�"*;


total_loss��@

error_RߟL?

learning_rate_1��8)��I       6%�	��Q���A�"*;


total_loss)�@

error_RzN?

learning_rate_1��8
tdI       6%�	`'�Q���A�"*;


total_loss, �@

error_R�^K?

learning_rate_1��8)��I       6%�	�l�Q���A�"*;


total_loss��@

error_R@a?

learning_rate_1��8�U(I       6%�	��Q���A�"*;


total_loss�(�@

error_R��Z?

learning_rate_1��82	�I       6%�	"��Q���A�"*;


total_loss�A

error_R_�F?

learning_rate_1��8�73:I       6%�	�K�Q���A�"*;


total_loss~2"A

error_R�E?

learning_rate_1��8���I       6%�	ؐ�Q���A�"*;


total_lossnR�@

error_R{d?

learning_rate_1��8X�)�I       6%�	pծQ���A�"*;


total_loss�A

error_RV?

learning_rate_1��8믠=I       6%�	�Q���A�"*;


total_lossɩ�@

error_RN�N?

learning_rate_1��8���I       6%�	[�Q���A�"*;


total_loss���@

error_R�/H?

learning_rate_1��8���I       6%�		��Q���A�"*;


total_lossF��@

error_RQ�M?

learning_rate_1��8ъ�7I       6%�	��Q���A�"*;


total_loss��@

error_R�-D?

learning_rate_1��8�idI       6%�	0'�Q���A�"*;


total_loss���@

error_R��M?

learning_rate_1��8����I       6%�	�l�Q���A�"*;


total_loss�u�@

error_R��L?

learning_rate_1��8l�5I       6%�	���Q���A�"*;


total_lossÙ�@

error_R�)M?

learning_rate_1��8s�@ I       6%�	���Q���A�"*;


total_loss�#A

error_R)Y?

learning_rate_1��8(��SI       6%�	?�Q���A�"*;


total_lossbǢ@

error_R�%\?

learning_rate_1��8~�Q�I       6%�	���Q���A�"*;


total_loss_�@

error_R�$@?

learning_rate_1��8���OI       6%�	pɱQ���A�"*;


total_loss��`@

error_R|my?

learning_rate_1��8{�DI       6%�	��Q���A�"*;


total_lossC�@

error_R �H?

learning_rate_1��8F��I       6%�	�R�Q���A�"*;


total_loss��@

error_R,�F?

learning_rate_1��8�� I       6%�	Q���A�"*;


total_loss�o\@

error_R{>?

learning_rate_1��8��>�I       6%�	gٲQ���A�"*;


total_loss�A

error_R��U?

learning_rate_1��8&�UI       6%�	��Q���A�"*;


total_loss V�@

error_R�gP?

learning_rate_1��8�;�I       6%�	[`�Q���A�"*;


total_loss1��@

error_R_n?

learning_rate_1��8��(I       6%�	�Q���A�"*;


total_loss� �@

error_R��D?

learning_rate_1��8�.8I       6%�	��Q���A�"*;


total_loss֗�@

error_R�_?

learning_rate_1��8'���I       6%�	�$�Q���A�"*;


total_lossm�@

error_R�@W?

learning_rate_1��8x���I       6%�	�g�Q���A�"*;


total_loss��@

error_R:�E?

learning_rate_1��8�]IMI       6%�	ڨ�Q���A�"*;


total_loss<l�@

error_R��6?

learning_rate_1��8��I       6%�	��Q���A�"*;


total_lossD��@

error_R�8S?

learning_rate_1��8Y���I       6%�	�.�Q���A�"*;


total_loss��@

error_R�hU?

learning_rate_1��8��q�I       6%�	bq�Q���A�"*;


total_loss�O�@

error_R�#J?

learning_rate_1��8�US�I       6%�	P��Q���A�"*;


total_loss���@

error_Ri X?

learning_rate_1��8�>�I       6%�	���Q���A�"*;


total_loss��A

error_R�YG?

learning_rate_1��8W�&I       6%�	,9�Q���A�"*;


total_lossCE�@

error_R�V?

learning_rate_1��8�ewI       6%�	%{�Q���A�"*;


total_loss�G�@

error_R�:?

learning_rate_1��8�d��I       6%�	s��Q���A�"*;


total_loss�@

error_RE�W?

learning_rate_1��8Je��I       6%�	w�Q���A�"*;


total_loss[��@

error_RnS?

learning_rate_1��8)�I       6%�	DC�Q���A�"*;


total_loss�@

error_RnlP?

learning_rate_1��8O�"I       6%�	���Q���A�"*;


total_loss�2�@

error_R�4R?

learning_rate_1��8��{�I       6%�	�ȷQ���A�"*;


total_loss���@

error_RsXY?

learning_rate_1��8N�<pI       6%�	<�Q���A�"*;


total_loss�A

error_R�gF?

learning_rate_1��83A�lI       6%�	�Y�Q���A�"*;


total_loss���@

error_R
=?

learning_rate_1��8ݎ�I       6%�	˟�Q���A�"*;


total_loss��@

error_R{�Q?

learning_rate_1��8���I       6%�	i�Q���A�"*;


total_loss]�@

error_R�m[?

learning_rate_1��8� ��I       6%�	�-�Q���A�"*;


total_lossz��@

error_RJ�P?

learning_rate_1��8���QI       6%�	,u�Q���A�"*;


total_loss�!�@

error_RH=W?

learning_rate_1��8l7I       6%�	�ֹQ���A�"*;


total_loss���@

error_R��_?

learning_rate_1��8�E�I       6%�	�9�Q���A�"*;


total_losss��@

error_R�_^?

learning_rate_1��8�1%EI       6%�	�~�Q���A�"*;


total_lossaQ�@

error_RqhL?

learning_rate_1��8G��I       6%�	�ºQ���A�"*;


total_loss���@

error_R�.J?

learning_rate_1��8�&�RI       6%�	t�Q���A�"*;


total_loss��@

error_R[�V?

learning_rate_1��8o�"�I       6%�	YJ�Q���A�"*;


total_loss,u�@

error_Rf�^?

learning_rate_1��8߬<I       6%�	医Q���A�"*;


total_loss{H�@

error_R�;R?

learning_rate_1��8Ga��I       6%�	LѻQ���A�"*;


total_lossr+UA

error_Rx4]?

learning_rate_1��8l���I       6%�	��Q���A�#*;


total_losss�@

error_R�9[?

learning_rate_1��8<�A�I       6%�	�V�Q���A�#*;


total_loss.��@

error_R��Y?

learning_rate_1��8�H�+I       6%�	F��Q���A�#*;


total_lossmm�@

error_R�Z?

learning_rate_1��82i]yI       6%�	@ۼQ���A�#*;


total_loss"�@

error_R}�N?

learning_rate_1��8`"i+I       6%�	 �Q���A�#*;


total_loss�iA

error_R��X?

learning_rate_1��8��II       6%�	+e�Q���A�#*;


total_loss�=�@

error_R�V?

learning_rate_1��8���I       6%�	L��Q���A�#*;


total_losse�"A

error_RA�_?

learning_rate_1��8M!�MI       6%�	���Q���A�#*;


total_loss}w�@

error_R.�<?

learning_rate_1��8��d2I       6%�	�2�Q���A�#*;


total_lossQj�@

error_R�c\?

learning_rate_1��8���I       6%�	"t�Q���A�#*;


total_losss��@

error_R�wW?

learning_rate_1��8�4��I       6%�	���Q���A�#*;


total_loss���@

error_RC�F?

learning_rate_1��8���I       6%�	E��Q���A�#*;


total_loss��@

error_R�fD?

learning_rate_1��8���}I       6%�	�G�Q���A�#*;


total_loss���@

error_R$3I?

learning_rate_1��8��I       6%�	K��Q���A�#*;


total_lossSW�@

error_R�9Q?

learning_rate_1��8�=?I       6%�	�޿Q���A�#*;


total_loss�@

error_RLT>?

learning_rate_1��8�*O`I       6%�	%�Q���A�#*;


total_loss��@

error_R��7?

learning_rate_1��8z�k�I       6%�	}k�Q���A�#*;


total_loss�#A

error_R�S?

learning_rate_1��8وBfI       6%�	���Q���A�#*;


total_lossV�A

error_R�yR?

learning_rate_1��8�TN�I       6%�	J��Q���A�#*;


total_losss �@

error_R1�P?

learning_rate_1��8	��XI       6%�	�7�Q���A�#*;


total_loss�|�@

error_RW�^?

learning_rate_1��8%%�I       6%�	�z�Q���A�#*;


total_lossB�@

error_R��N?

learning_rate_1��8���5I       6%�	A��Q���A�#*;


total_loss�D�@

error_R��d?

learning_rate_1��8cKE�I       6%�	��Q���A�#*;


total_loss�ɸ@

error_RF�^?

learning_rate_1��84l�I       6%�	SE�Q���A�#*;


total_loss���@

error_R�A?

learning_rate_1��8n��)I       6%�	��Q���A�#*;


total_loss��@

error_R3�R?

learning_rate_1��8��I       6%�	���Q���A�#*;


total_loss�f�@

error_R*�>?

learning_rate_1��8��I       6%�	[�Q���A�#*;


total_lossZHW@

error_R�Y?

learning_rate_1��8L�eBI       6%�	�]�Q���A�#*;


total_loss$�A

error_R&�Z?

learning_rate_1��8�ʔ�I       6%�	@��Q���A�#*;


total_loss���@

error_R�V>?

learning_rate_1��8[�@�I       6%�	W��Q���A�#*;


total_loss �@

error_R)X?

learning_rate_1��8ao��I       6%�	�F�Q���A�#*;


total_loss���@

error_Rak`?

learning_rate_1��8Y�+�I       6%�	��Q���A�#*;


total_loss�C�@

error_R.�T?

learning_rate_1��8���I       6%�	%��Q���A�#*;


total_loss�E@

error_RE�U?

learning_rate_1��8f�ѝI       6%�	�Q���A�#*;


total_loss�Z�@

error_R�nN?

learning_rate_1��8*�8qI       6%�	�\�Q���A�#*;


total_loss�cA

error_R�O?

learning_rate_1��8�g(RI       6%�	x��Q���A�#*;


total_loss���@

error_R�NR?

learning_rate_1��8���+I       6%�	���Q���A�#*;


total_loss�@

error_R�ki?

learning_rate_1��8h��vI       6%�	g4�Q���A�#*;


total_loss���@

error_R.xM?

learning_rate_1��8�^wI       6%�	{�Q���A�#*;


total_loss���@

error_RO�U?

learning_rate_1��8��,�I       6%�	A��Q���A�#*;


total_loss���@

error_R�P?

learning_rate_1��8`�|dI       6%�	Y�Q���A�#*;


total_loss���@

error_R II?

learning_rate_1��8Ӛ��I       6%�	�V�Q���A�#*;


total_lossnb�@

error_R<�N?

learning_rate_1��8��!�I       6%�	ə�Q���A�#*;


total_loss$��@

error_R�8?

learning_rate_1��8o�J�I       6%�	U��Q���A�#*;


total_loss��@

error_Rf�Y?

learning_rate_1��82���I       6%�	��Q���A�#*;


total_loss�A�@

error_R�eP?

learning_rate_1��8�G��I       6%�	�d�Q���A�#*;


total_loss,�7A

error_Rׯ@?

learning_rate_1��8/ �I       6%�	%��Q���A�#*;


total_loss�m�@

error_R3eN?

learning_rate_1��8��f�I       6%�	=��Q���A�#*;


total_loss��@

error_Rl�F?

learning_rate_1��8*��RI       6%�	+;�Q���A�#*;


total_loss	�@

error_R��_?

learning_rate_1��8��Z|I       6%�	�~�Q���A�#*;


total_loss���@

error_R@�K?

learning_rate_1��8ŵ4�I       6%�	���Q���A�#*;


total_loss(7�@

error_Ri�Q?

learning_rate_1��8�٘I       6%�	�'�Q���A�#*;


total_loss.��@

error_R̼E?

learning_rate_1��8�OFJI       6%�	#o�Q���A�#*;


total_lossښ�@

error_R��L?

learning_rate_1��8�e�I       6%�	˱�Q���A�#*;


total_loss���@

error_R�7M?

learning_rate_1��8�!vI       6%�	���Q���A�#*;


total_lossqŭ@

error_Rܖh?

learning_rate_1��8b��I       6%�	z8�Q���A�#*;


total_loss��@

error_R�L?

learning_rate_1��8`�G�I       6%�	@{�Q���A�#*;


total_lossMk�@

error_Rd�I?

learning_rate_1��8]/��I       6%�	��Q���A�#*;


total_loss=Ģ@

error_R�(K?

learning_rate_1��8K���I       6%�	��Q���A�#*;


total_loss�F�@

error_R�vP?

learning_rate_1��8�uI       6%�	[C�Q���A�#*;


total_loss�3A

error_R��[?

learning_rate_1��8���I       6%�	���Q���A�#*;


total_loss��@

error_R��_?

learning_rate_1��8���I       6%�	0��Q���A�#*;


total_loss�I�@

error_R�V?

learning_rate_1��8��I       6%�	��Q���A�#*;


total_loss��@

error_RAK?

learning_rate_1��8��I       6%�	yQ�Q���A�#*;


total_loss
�A

error_R��Y?

learning_rate_1��8�!I       6%�	<��Q���A�#*;


total_lossJ��@

error_R;�_?

learning_rate_1��8&aN<I       6%�	��Q���A�#*;


total_lossV	�@

error_R�a?

learning_rate_1��8o��5I       6%�	1.�Q���A�#*;


total_loss`�A

error_R8�_?

learning_rate_1��8oA��I       6%�	v�Q���A�#*;


total_loss)1�@

error_R�*L?

learning_rate_1��8%��vI       6%�	��Q���A�#*;


total_loss��A

error_R�:T?

learning_rate_1��8�g�;I       6%�	���Q���A�#*;


total_loss��@

error_R�bG?

learning_rate_1��8�̽3I       6%�	<B�Q���A�#*;


total_loss�ّ@

error_R�R?

learning_rate_1��8�FDI       6%�	��Q���A�#*;


total_loss;�@

error_R?�G?

learning_rate_1��8��o�I       6%�	?��Q���A�#*;


total_loss�o�@

error_REa??

learning_rate_1��8T��RI       6%�	'�Q���A�#*;


total_loss{k�@

error_R�jR?

learning_rate_1��8I�]BI       6%�	gP�Q���A�#*;


total_loss[I�@

error_R	�J?

learning_rate_1��8;!*+I       6%�	���Q���A�#*;


total_loss�A

error_R�U?

learning_rate_1��8+]7I       6%�	���Q���A�#*;


total_lossS��@

error_R�s\?

learning_rate_1��8��1�I       6%�	��Q���A�#*;


total_loss��@

error_R�K?

learning_rate_1��8f6�I       6%�	�\�Q���A�#*;


total_loss=��@

error_R	iB?

learning_rate_1��8M�}I       6%�	���Q���A�#*;


total_lossfԡ@

error_Rv�D?

learning_rate_1��8���I       6%�	R��Q���A�#*;


total_loss�{A

error_R�E?

learning_rate_1��8'���I       6%�	9)�Q���A�#*;


total_loss���@

error_R�J?

learning_rate_1��8e &�I       6%�	�l�Q���A�#*;


total_lossL�@

error_R�OE?

learning_rate_1��8��;gI       6%�	���Q���A�#*;


total_loss=-�@

error_RW�O?

learning_rate_1��8����I       6%�	���Q���A�#*;


total_loss�C�@

error_R��^?

learning_rate_1��8>��I       6%�	4�Q���A�#*;


total_loss۲�@

error_R�X?

learning_rate_1��8����I       6%�	�v�Q���A�#*;


total_loss^�@

error_R��M?

learning_rate_1��8V���I       6%�	~��Q���A�#*;


total_loss\��@

error_R��P?

learning_rate_1��8:��I       6%�	T��Q���A�#*;


total_loss�g�@

error_R��C?

learning_rate_1��8�K@I       6%�	�;�Q���A�#*;


total_loss2+�@

error_R�J?

learning_rate_1��8��f[I       6%�	(��Q���A�#*;


total_loss���@

error_ROi.?

learning_rate_1��8K��xI       6%�	��Q���A�#*;


total_loss� �@

error_R�YL?

learning_rate_1��8�]�zI       6%�	u�Q���A�#*;


total_loss%H�@

error_R�E@?

learning_rate_1��8���gI       6%�	+M�Q���A�#*;


total_loss<�@

error_RZK?

learning_rate_1��8�G2KI       6%�	��Q���A�#*;


total_loss8��@

error_R 'L?

learning_rate_1��8LzM�I       6%�	���Q���A�#*;


total_loss���@

error_R	C?

learning_rate_1��8ʥ��I       6%�	��Q���A�#*;


total_loss��z@

error_RT�8?

learning_rate_1��8����I       6%�	P^�Q���A�#*;


total_lossKؠ@

error_R�.]?

learning_rate_1��8�4��I       6%�	��Q���A�#*;


total_loss���@

error_R.[T?

learning_rate_1��8VڦlI       6%�	h��Q���A�#*;


total_loss���@

error_R�DT?

learning_rate_1��8��=I       6%�	�)�Q���A�#*;


total_loss�%�@

error_RDa7?

learning_rate_1��8�גI       6%�	�m�Q���A�#*;


total_lossCe�@

error_RȦX?

learning_rate_1��8�nI       6%�	P��Q���A�#*;


total_loss��@

error_RrG?

learning_rate_1��8̈́�I       6%�	��Q���A�#*;


total_loss%�@

error_RQ?

learning_rate_1��8S�P�I       6%�	!=�Q���A�#*;


total_loss��@

error_R��K?

learning_rate_1��8���LI       6%�	��Q���A�#*;


total_loss;��@

error_Rc�S?

learning_rate_1��8?.� I       6%�	���Q���A�#*;


total_lossEA

error_R��>?

learning_rate_1��8����I       6%�	U�Q���A�#*;


total_loss,�@

error_R��4?

learning_rate_1��8�mexI       6%�	cH�Q���A�#*;


total_loss.��@

error_R�W?

learning_rate_1��8��4I       6%�	9��Q���A�#*;


total_loss2ɱ@

error_R�@?

learning_rate_1��8�1��I       6%�	���Q���A�#*;


total_loss7һ@

error_R
D?

learning_rate_1��8�=1I       6%�	.5�Q���A�#*;


total_loss�`A

error_R�cA?

learning_rate_1��8O�<I       6%�	�y�Q���A�#*;


total_loss�y A

error_R6�b?

learning_rate_1��8���wI       6%�	^��Q���A�#*;


total_loss��@

error_R;$A?

learning_rate_1��8��N�I       6%�	���Q���A�#*;


total_loss㳩@

error_R�e_?

learning_rate_1��8Qw��I       6%�	�?�Q���A�#*;


total_loss<��@

error_R	�i?

learning_rate_1��8�FzI       6%�	���Q���A�#*;


total_loss���@

error_R� O?

learning_rate_1��8*�I       6%�	[��Q���A�#*;


total_loss��@

error_RfeJ?

learning_rate_1��8
߂�I       6%�	��Q���A�#*;


total_lossÉ@

error_RA�L?

learning_rate_1��8�E�I       6%�	 S�Q���A�#*;


total_loss�H�@

error_RʧX?

learning_rate_1��8ŵ�JI       6%�	P��Q���A�#*;


total_loss7\�@

error_RDWM?

learning_rate_1��8�73LI       6%�	���Q���A�#*;


total_loss���@

error_R�R?

learning_rate_1��8jS��I       6%�	
�Q���A�#*;


total_loss�*�@

error_R�zG?

learning_rate_1��8�5�I       6%�	b�Q���A�#*;


total_loss�}�@

error_Rm�A?

learning_rate_1��8x��I       6%�	]��Q���A�#*;


total_lossd#�@

error_RʆE?

learning_rate_1��8G�iI       6%�	T��Q���A�#*;


total_losso��@

error_Rړ??

learning_rate_1��8f��I       6%�	�/�Q���A�#*;


total_loss�@�@

error_R�sL?

learning_rate_1��8����I       6%�	�u�Q���A�#*;


total_lossD�A

error_RRvN?

learning_rate_1��8��SI       6%�	���Q���A�$*;


total_loss<��@

error_R3�A?

learning_rate_1��8�6�I       6%�	W �Q���A�$*;


total_loss8�@

error_R�I?

learning_rate_1��8���I       6%�	�F�Q���A�$*;


total_lossu��@

error_R��[?

learning_rate_1��8ہ�I       6%�	���Q���A�$*;


total_loss�ϸ@

error_R�P?

learning_rate_1��8���PI       6%�	���Q���A�$*;


total_loss$u�@

error_RM�e?

learning_rate_1��8掀*I       6%�	8�Q���A�$*;


total_lossm�@

error_RC
P?

learning_rate_1��8����I       6%�	H�Q���A�$*;


total_loss!��@

error_R $;?

learning_rate_1��82��I       6%�	r��Q���A�$*;


total_loss�ڦ@

error_R��>?

learning_rate_1��8����I       6%�	��Q���A�$*;


total_loss���@

error_RR�V?

learning_rate_1��8��tI       6%�	��Q���A�$*;


total_lossʡA

error_R�bY?

learning_rate_1��8D�qI       6%�	�T�Q���A�$*;


total_loss�8�@

error_R��P?

learning_rate_1��8���I       6%�	���Q���A�$*;


total_loss���@

error_R�M?

learning_rate_1��8,1��I       6%�	���Q���A�$*;


total_loss���@

error_R}]?

learning_rate_1��8�B
_I       6%�	$�Q���A�$*;


total_loss./�@

error_R��G?

learning_rate_1��8Hh��I       6%�	�e�Q���A�$*;


total_loss���@

error_Rq�M?

learning_rate_1��8�I       6%�	%��Q���A�$*;


total_lossC9A

error_R��W?

learning_rate_1��8N��I       6%�	w��Q���A�$*;


total_loss�
�@

error_R�B\?

learning_rate_1��8M^�bI       6%�	K$�Q���A�$*;


total_lossX+A

error_R��D?

learning_rate_1��8g
��I       6%�	Fg�Q���A�$*;


total_loss��@

error_R7�I?

learning_rate_1��8�rI       6%�	��Q���A�$*;


total_loss\�@

error_R�m>?

learning_rate_1��8���I       6%�	s��Q���A�$*;


total_loss7m�@

error_RHN?

learning_rate_1��8ZI       6%�	�-�Q���A�$*;


total_loss;��@

error_R��E?

learning_rate_1��8u�#I       6%�	5w�Q���A�$*;


total_loss���@

error_RsO?

learning_rate_1��8+X>I       6%�	���Q���A�$*;


total_loss���@

error_RH�L?

learning_rate_1��8��_�I       6%�	z��Q���A�$*;


total_loss%��@

error_RR�`?

learning_rate_1��8Ӡ��I       6%�	�;�Q���A�$*;


total_lossXkA

error_R�^?

learning_rate_1��8W�adI       6%�	�|�Q���A�$*;


total_loss1��@

error_R�#[?

learning_rate_1��8�z��I       6%�	B��Q���A�$*;


total_lossъ�@

error_R2X?

learning_rate_1��8�J�TI       6%�	~�Q���A�$*;


total_lossq�@

error_R�,T?

learning_rate_1��8v��oI       6%�	�\�Q���A�$*;


total_loss	9�@

error_R��I?

learning_rate_1��8r�
_I       6%�	���Q���A�$*;


total_lossѿ�@

error_R�yF?

learning_rate_1��8��@I       6%�	S��Q���A�$*;


total_loss�/�@

error_R�CP?

learning_rate_1��8�[�I       6%�	F>�Q���A�$*;


total_lossiŨ@

error_R \X?

learning_rate_1��8�W�#I       6%�	��Q���A�$*;


total_loss��@

error_R=�N?

learning_rate_1��8�VI       6%�	���Q���A�$*;


total_loss)Ӫ@

error_R��4?

learning_rate_1��8"4DI       6%�	��Q���A�$*;


total_loss�O�@

error_RfK?

learning_rate_1��8�.<zI       6%�	�C�Q���A�$*;


total_lossDS�@

error_R߱L?

learning_rate_1��8�>�I       6%�	���Q���A�$*;


total_loss��@

error_R=�C?

learning_rate_1��8��I       6%�	���Q���A�$*;


total_loss�� A

error_R��Z?

learning_rate_1��8�h~�I       6%�	�Q���A�$*;


total_loss P�@

error_R�Y?

learning_rate_1��8+��I       6%�	�K�Q���A�$*;


total_loss���@

error_R�P?

learning_rate_1��8���9I       6%�	ފ�Q���A�$*;


total_loss�S�@

error_RA�C?

learning_rate_1��8�.B�I       6%�	���Q���A�$*;


total_loss�]r@

error_R3uX?

learning_rate_1��8�ՖI       6%�	8�Q���A�$*;


total_lossr�@

error_RRIa?

learning_rate_1��8�C�I       6%�	�x�Q���A�$*;


total_loss�M�@

error_R1�W?

learning_rate_1��8�N �I       6%�	T��Q���A�$*;


total_loss���@

error_R��F?

learning_rate_1��8��?�I       6%�	E��Q���A�$*;


total_loss�UA

error_RM�F?

learning_rate_1��8L�c�I       6%�	e;�Q���A�$*;


total_lossl��@

error_R$�>?

learning_rate_1��8W6�II       6%�	�~�Q���A�$*;


total_lossMk�@

error_RH=?

learning_rate_1��8�?Y_I       6%�	���Q���A�$*;


total_loss]%�@

error_RŪI?

learning_rate_1��8���I       6%�	���Q���A�$*;


total_loss�S�@

error_R 	L?

learning_rate_1��8b�0:I       6%�	v@�Q���A�$*;


total_loss%N|@

error_RFDL?

learning_rate_1��8�x��I       6%�	݅�Q���A�$*;


total_loss�ź@

error_R�_?

learning_rate_1��8�$NI       6%�	���Q���A�$*;


total_lossQ�@

error_RܓY?

learning_rate_1��80��&I       6%�	��Q���A�$*;


total_loss���@

error_R=�f?

learning_rate_1��8Х�tI       6%�	�U�Q���A�$*;


total_loss�v�@

error_R-MW?

learning_rate_1��8|m�I       6%�	љ�Q���A�$*;


total_loss)ws@

error_R��D?

learning_rate_1��8_�I       6%�	���Q���A�$*;


total_loss��@

error_R��Q?

learning_rate_1��8�r�jI       6%�	l!�Q���A�$*;


total_loss���@

error_R8�<?

learning_rate_1��8#�ĠI       6%�	Ld�Q���A�$*;


total_lossv*LA

error_Rߎ;?

learning_rate_1��8&j}UI       6%�	���Q���A�$*;


total_lossd��@

error_R#�\?

learning_rate_1��8��dI       6%�	G��Q���A�$*;


total_loss�`A

error_R�Y?

learning_rate_1��8- ��I       6%�	�3�Q���A�$*;


total_loss�j�@

error_R�)??

learning_rate_1��8�0�8I       6%�	dq�Q���A�$*;


total_loss�U�@

error_R/�Q?

learning_rate_1��8�^�I       6%�	��Q���A�$*;


total_loss�H
A

error_Rv5N?

learning_rate_1��8	��I       6%�	���Q���A�$*;


total_loss�A

error_R+Q?

learning_rate_1��8����I       6%�	�1�Q���A�$*;


total_lossY�@

error_Rx9Q?

learning_rate_1��8�V�I       6%�	�q�Q���A�$*;


total_loss�D�@

error_R�J?

learning_rate_1��85�G�I       6%�	��Q���A�$*;


total_loss�&A

error_R�$B?

learning_rate_1��8DKZI       6%�	x��Q���A�$*;


total_loss��@

error_R!�O?

learning_rate_1��8���I       6%�	w4�Q���A�$*;


total_loss���@

error_RJ�U?

learning_rate_1��8���I       6%�	�t�Q���A�$*;


total_loss4�l@

error_R�WS?

learning_rate_1��8t"�I       6%�	Q��Q���A�$*;


total_lossf��@

error_R\�Y?

learning_rate_1��8�cM�I       6%�	���Q���A�$*;


total_loss(��@

error_Rr,L?

learning_rate_1��8�GI       6%�	�B�Q���A�$*;


total_loss(W�@

error_R\CJ?

learning_rate_1��8��IfI       6%�	e��Q���A�$*;


total_loss�`�@

error_R�I?

learning_rate_1��8vё�I       6%�	���Q���A�$*;


total_loss�f|@

error_R��J?

learning_rate_1��8�b&"I       6%�	��Q���A�$*;


total_loss��l@

error_R�J?

learning_rate_1��8�ްVI       6%�	IV�Q���A�$*;


total_loss48�@

error_R�RR?

learning_rate_1��8{aI       6%�	���Q���A�$*;


total_loss� �@

error_R�L?

learning_rate_1��8�l�I       6%�	O��Q���A�$*;


total_loss�J�@

error_R�9O?

learning_rate_1��8��_I       6%�	j�Q���A�$*;


total_lossC��@

error_Rܓ[?

learning_rate_1��88�I       6%�	+S�Q���A�$*;


total_loss���@

error_RRRP?

learning_rate_1��8�b)HI       6%�	T��Q���A�$*;


total_loss_�@

error_R��[?

learning_rate_1��8�f5�I       6%�	���Q���A�$*;


total_loss�?�@

error_RJ>_?

learning_rate_1��8볼I       6%�	��Q���A�$*;


total_loss,��@

error_R&�Q?

learning_rate_1��8���I       6%�	qe�Q���A�$*;


total_loss�A

error_R��M?

learning_rate_1��8=�ߊI       6%�	��Q���A�$*;


total_loss�ߗ@

error_R��C?

learning_rate_1��8z[��I       6%�	���Q���A�$*;


total_loss�Ȕ@

error_R�^?

learning_rate_1��8_��yI       6%�	.)�Q���A�$*;


total_lossXŗ@

error_R	�D?

learning_rate_1��8�I       6%�	�j�Q���A�$*;


total_lossׯ�@

error_R;HO?

learning_rate_1��8�s�I       6%�	���Q���A�$*;


total_loss���@

error_R��D?

learning_rate_1��8f{qbI       6%�	���Q���A�$*;


total_loss���@

error_R��D?

learning_rate_1��8�l�bI       6%�	�,�Q���A�$*;


total_loss��@

error_R
?[?

learning_rate_1��8�N-�I       6%�	/l�Q���A�$*;


total_loss��@

error_R(:?

learning_rate_1��8N@&I       6%�	^��Q���A�$*;


total_loss�{�@

error_R�bL?

learning_rate_1��8g��@I       6%�	���Q���A�$*;


total_losso��@

error_R֗O?

learning_rate_1��8e#��I       6%�	�2�Q���A�$*;


total_loss饥@

error_R3N?

learning_rate_1��8&�|�I       6%�	~s�Q���A�$*;


total_loss�B�@

error_R��M?

learning_rate_1��8dO8hI       6%�	��Q���A�$*;


total_loss�@

error_R�@<?

learning_rate_1��8�P{�I       6%�	���Q���A�$*;


total_loss��@

error_R�_?

learning_rate_1��8"��aI       6%�	�8�Q���A�$*;


total_loss��@

error_R�
Y?

learning_rate_1��8�Do�I       6%�	���Q���A�$*;


total_lossc�@

error_R{	D?

learning_rate_1��8U;uI       6%�	���Q���A�$*;


total_loss��@

error_R}�X?

learning_rate_1��8#(n�I       6%�	\'�Q���A�$*;


total_loss��@

error_Ri�S?

learning_rate_1��8Bm�)I       6%�	?l�Q���A�$*;


total_loss*��@

error_RCP?

learning_rate_1��8R/��I       6%�	��Q���A�$*;


total_loss-�A

error_R}�M?

learning_rate_1��8f�?�I       6%�	���Q���A�$*;


total_loss��	A

error_R�F?

learning_rate_1��8]���I       6%�	�/�Q���A�$*;


total_loss��@

error_RhrV?

learning_rate_1��8���{I       6%�	�r�Q���A�$*;


total_loss7L�@

error_R��@?

learning_rate_1��8���I       6%�	���Q���A�$*;


total_loss�@

error_R��T?

learning_rate_1��8���I       6%�	���Q���A�$*;


total_loss�A�@

error_R}sX?

learning_rate_1��8�jm�I       6%�	cF�Q���A�$*;


total_lossa�@

error_R�V[?

learning_rate_1��8���I       6%�	��Q���A�$*;


total_loss�@

error_R��E?

learning_rate_1��8h��I       6%�	_��Q���A�$*;


total_loss�k�@

error_R�@?

learning_rate_1��8�4ԃI       6%�	@	�Q���A�$*;


total_loss��A

error_R��J?

learning_rate_1��8̳7I       6%�	�P�Q���A�$*;


total_loss��A

error_R�Z?

learning_rate_1��8�5��I       6%�	��Q���A�$*;


total_loss���@

error_R��X?

learning_rate_1��8T�\I       6%�	���Q���A�$*;


total_loss�K�@

error_R4yE?

learning_rate_1��8����I       6%�	�-�Q���A�$*;


total_loss{X�@

error_R�J?

learning_rate_1��8d0S�I       6%�	�q�Q���A�$*;


total_loss�ʭ@

error_R�X?

learning_rate_1��8�'I       6%�	��Q���A�$*;


total_loss|k@

error_R7P?

learning_rate_1��8z��I       6%�	���Q���A�$*;


total_loss���@

error_R�4U?

learning_rate_1��8���SI       6%�	uB�Q���A�$*;


total_loss�Z�@

error_R�A?

learning_rate_1��8 ac�I       6%�	���Q���A�$*;


total_loss�0�@

error_R?�Q?

learning_rate_1��8��xI       6%�	G��Q���A�$*;


total_loss�@

error_R�-;?

learning_rate_1��8a5Z�I       6%�	
 R���A�$*;


total_loss&�@

error_R?�V?

learning_rate_1��8Sh�I       6%�	�M R���A�$*;


total_loss!��@

error_R�~V?

learning_rate_1��8M�8NI       6%�	�� R���A�%*;


total_loss>�@

error_R��H?

learning_rate_1��8��d`I       6%�	�� R���A�%*;


total_loss@��@

error_R�X?

learning_rate_1��8s�)I       6%�	�R���A�%*;


total_loss+��@

error_Rxb?

learning_rate_1��8�;�I       6%�	(aR���A�%*;


total_loss�� A

error_R]�M?

learning_rate_1��8�(/EI       6%�	�R���A�%*;


total_loss�U�@

error_R��M?

learning_rate_1��8���I       6%�	��R���A�%*;


total_losscȊ@

error_R��I?

learning_rate_1��8F
�I       6%�	�-R���A�%*;


total_lossz��@

error_R�aP?

learning_rate_1��8?�OI       6%�	;oR���A�%*;


total_loss��@

error_R��S?

learning_rate_1��8A�6�I       6%�	k�R���A�%*;


total_loss���@

error_R&~E?

learning_rate_1��8��9I       6%�	��R���A�%*;


total_loss�ަ@

error_R\�J?

learning_rate_1��8����I       6%�	�2R���A�%*;


total_loss��@

error_R��i?

learning_rate_1��8Ey��I       6%�	�rR���A�%*;


total_loss�4�@

error_R�-Z?

learning_rate_1��8�z�BI       6%�	O�R���A�%*;


total_lossǮ@

error_R.�R?

learning_rate_1��8;��PI       6%�	��R���A�%*;


total_loss��A

error_RNnQ?

learning_rate_1��8s<6�I       6%�	j<R���A�%*;


total_lossm��@

error_RhoP?

learning_rate_1��8��&I       6%�	�}R���A�%*;


total_loss2�@

error_RI�S?

learning_rate_1��8����I       6%�	�R���A�%*;


total_lossq��@

error_R�KY?

learning_rate_1��8�S�JI       6%�	��R���A�%*;


total_lossF̖@

error_RuL?

learning_rate_1��8�AEI       6%�	A>R���A�%*;


total_loss/~N@

error_R��@?

learning_rate_1��8a/b.I       6%�	e~R���A�%*;


total_loss��@

error_RW�e?

learning_rate_1��8c��I       6%�	4�R���A�%*;


total_loss^�@

error_R�mR?

learning_rate_1��8�I��I       6%�	DR���A�%*;


total_loss؇�@

error_RIW?

learning_rate_1��8�ECI       6%�	�CR���A�%*;


total_lossȅ�@

error_R��G?

learning_rate_1��84��I       6%�	�R���A�%*;


total_loss�U�@

error_R&?G?

learning_rate_1��8L�?�I       6%�	W�R���A�%*;


total_loss�;A

error_R:�J?

learning_rate_1��8)W�rI       6%�	�R���A�%*;


total_loss�N�@

error_R��f?

learning_rate_1��8�5L I       6%�	�RR���A�%*;


total_loss�@

error_R�H9?

learning_rate_1��8\�[I       6%�	ғR���A�%*;


total_loss���@

error_R�EN?

learning_rate_1��8�S7�I       6%�	��R���A�%*;


total_loss�(�@

error_RqcZ?

learning_rate_1��8	9lRI       6%�	�%R���A�%*;


total_lossH��@

error_R#�L?

learning_rate_1��8X�aI       6%�	�hR���A�%*;


total_loss�{�@

error_RL�B?

learning_rate_1��8�3�I       6%�	«R���A�%*;


total_loss{�@

error_R��U?

learning_rate_1��8�o�VI       6%�	7�R���A�%*;


total_loss��@

error_R�jI?

learning_rate_1��8r�I       6%�	|3	R���A�%*;


total_loss)E�@

error_Rq�]?

learning_rate_1��8Pj{�I       6%�	�u	R���A�%*;


total_lossm��@

error_R<�A?

learning_rate_1��8A��I       6%�	�	R���A�%*;


total_loss���@

error_Ri�T?

learning_rate_1��8���I       6%�	1
R���A�%*;


total_loss�;�@

error_R/
e?

learning_rate_1��8i���I       6%�	�a
R���A�%*;


total_loss&� A

error_RұI?

learning_rate_1��8(�FI       6%�	��
R���A�%*;


total_loss��@

error_R3�O?

learning_rate_1��8�HȹI       6%�	#�
R���A�%*;


total_loss�)�@

error_RɃO?

learning_rate_1��8B�׵I       6%�	3R���A�%*;


total_lossJO�@

error_R��N?

learning_rate_1��8���fI       6%�	�tR���A�%*;


total_loss6��@

error_R�R?

learning_rate_1��8C$�I       6%�	�R���A�%*;


total_loss��@

error_R۩K?

learning_rate_1��8�:�!I       6%�	��R���A�%*;


total_loss��@

error_R/�C?

learning_rate_1��82�RI       6%�	~6R���A�%*;


total_loss�;w@

error_Rq�P?

learning_rate_1��8�,�I       6%�	�vR���A�%*;


total_loss�]�@

error_R�I?

learning_rate_1��8��N�I       6%�	��R���A�%*;


total_loss$Z�@

error_R$�M?

learning_rate_1��8�t vI       6%�	��R���A�%*;


total_loss�9A

error_Rf�M?

learning_rate_1��8��o�I       6%�	[9R���A�%*;


total_lossܴA

error_Ra\?

learning_rate_1��8�.hI       6%�	�wR���A�%*;


total_loss��@

error_R�O?

learning_rate_1��8	\.�I       6%�	��R���A�%*;


total_loss���@

error_R�`?

learning_rate_1��8�`��I       6%�	��R���A�%*;


total_lossǚ�@

error_R��W?

learning_rate_1��8ʽ�I       6%�	�?R���A�%*;


total_loss��@

error_RZ?

learning_rate_1��8�l%�I       6%�	��R���A�%*;


total_loss�Y#A

error_Re\?

learning_rate_1��8� �bI       6%�	��R���A�%*;


total_loss�N�@

error_R��X?

learning_rate_1��80�Y�I       6%�	oR���A�%*;


total_lossy��@

error_R��A?

learning_rate_1��8{�II       6%�	ER���A�%*;


total_loss�[�@

error_R��0?

learning_rate_1��8�&�I       6%�	e�R���A�%*;


total_lossݫ@

error_RTN?

learning_rate_1��84�LI       6%�	\�R���A�%*;


total_lossf�@

error_RQZ?

learning_rate_1��8]��I       6%�	�R���A�%*;


total_loss��@

error_R/F?

learning_rate_1��8�J�I       6%�	FLR���A�%*;


total_loss�+�@

error_R�8X?

learning_rate_1��8< VI       6%�	|�R���A�%*;


total_loss�2�@

error_RqxU?

learning_rate_1��8����I       6%�	��R���A�%*;


total_loss
m@

error_RC�O?

learning_rate_1��8z�I       6%�	�R���A�%*;


total_loss�G�@

error_RP?

learning_rate_1��8�quI       6%�	@UR���A�%*;


total_loss��@

error_R�ZS?

learning_rate_1��87;�I       6%�	$�R���A�%*;


total_loss\V�@

error_R�MV?

learning_rate_1��8d&�2I       6%�	O�R���A�%*;


total_loss!��@

error_R��\?

learning_rate_1��84�R�I       6%�	AR���A�%*;


total_loss���@

error_RI�T?

learning_rate_1��8I��*I       6%�	'VR���A�%*;


total_loss.w�@

error_RJJ?

learning_rate_1��8���I       6%�	��R���A�%*;


total_loss��@

error_R�Y?

learning_rate_1��8|]�XI       6%�	��R���A�%*;


total_loss���@

error_R@d_?

learning_rate_1��8�*FI       6%�	R���A�%*;


total_lossO�-A

error_R�J?

learning_rate_1��8Pm�II       6%�	�UR���A�%*;


total_lossd1�@

error_R��c?

learning_rate_1��8*!ȍI       6%�	D�R���A�%*;


total_lossc��@

error_RN<D?

learning_rate_1��8����I       6%�	 �R���A�%*;


total_loss�A

error_R7�:?

learning_rate_1��8�/yWI       6%�	BR���A�%*;


total_loss"�	A

error_R��`?

learning_rate_1��83J��I       6%�	�VR���A�%*;


total_lossj��@

error_RV�L?

learning_rate_1��8��*�I       6%�	x�R���A�%*;


total_loss�[�@

error_R��=?

learning_rate_1��8�U�5I       6%�	��R���A�%*;


total_loss���@

error_R�pU?

learning_rate_1��8��[I       6%�	R���A�%*;


total_loss���@

error_RܥS?

learning_rate_1��8��Q|I       6%�	maR���A�%*;


total_loss<��@

error_R�p?

learning_rate_1��8��0I       6%�	^�R���A�%*;


total_loss紡@

error_RqGG?

learning_rate_1��8lΎ`I       6%�	��R���A�%*;


total_loss���@

error_RTWC?

learning_rate_1��8jzcJI       6%�	�-R���A�%*;


total_loss->�@

error_R��S?

learning_rate_1��8t�W�I       6%�	�nR���A�%*;


total_loss���@

error_R��R?

learning_rate_1��8�?vI       6%�	��R���A�%*;


total_losssaA

error_R�G?

learning_rate_1��8E)~�I       6%�	@�R���A�%*;


total_lossM4A

error_R}�@?

learning_rate_1��8P�P�I       6%�	71R���A�%*;


total_lossA��@

error_R��4?

learning_rate_1��8]4N]I       6%�	.pR���A�%*;


total_loss�l�@

error_R{cd?

learning_rate_1��8��ѐI       6%�	��R���A�%*;


total_lossZr�@

error_R_�U?

learning_rate_1��8�,CI       6%�	U�R���A�%*;


total_loss8�@

error_R3AU?

learning_rate_1��8�|�I       6%�	�.R���A�%*;


total_loss��@

error_R�iH?

learning_rate_1��8����I       6%�	�qR���A�%*;


total_loss|{,A

error_RD5;?

learning_rate_1��8����I       6%�	��R���A�%*;


total_lossOR�@

error_R�G?

learning_rate_1��8�XI       6%�	��R���A�%*;


total_loss�z�@

error_R�/a?

learning_rate_1��8��h�I       6%�	�=R���A�%*;


total_loss��@

error_RׄD?

learning_rate_1��8�GR�I       6%�	�~R���A�%*;


total_lossH��@

error_R�@M?

learning_rate_1��8��v�I       6%�	[�R���A�%*;


total_loss��@

error_R�V?

learning_rate_1��8S��I       6%�	A,R���A�%*;


total_loss
+�@

error_R�lM?

learning_rate_1��8tM[mI       6%�	\oR���A�%*;


total_loss���@

error_R:[?

learning_rate_1��8���I       6%�	A�R���A�%*;


total_loss1 �@

error_R�_?

learning_rate_1��8�:�RI       6%�	ZR���A�%*;


total_loss1��@

error_RA�<?

learning_rate_1��8x>��I       6%�	8bR���A�%*;


total_loss�L�@

error_Rs5??

learning_rate_1��8F���I       6%�	�R���A�%*;


total_loss�N�@

error_R�EH?

learning_rate_1��8Њ��I       6%�	��R���A�%*;


total_loss��@

error_R�Z`?

learning_rate_1��8��I       6%�	u/R���A�%*;


total_loss\O�@

error_R��;?

learning_rate_1��8��I       6%�	SoR���A�%*;


total_loss��@

error_R��N?

learning_rate_1��8d|˶I       6%�	��R���A�%*;


total_loss�۽@

error_R�~S?

learning_rate_1��8Lx�I       6%�	��R���A�%*;


total_loss���@

error_R�dI?

learning_rate_1��8g(L�I       6%�	Z3R���A�%*;


total_loss���@

error_R�K?

learning_rate_1��8��[�I       6%�	tR���A�%*;


total_loss���@

error_R}�G?

learning_rate_1��8]���I       6%�	��R���A�%*;


total_loss�YA

error_R�mD?

learning_rate_1��8�&ǕI       6%�	zR���A�%*;


total_lossec�@

error_Rl�E?

learning_rate_1��8�E�OI       6%�	qIR���A�%*;


total_loss�T�@

error_R��9?

learning_rate_1��8����I       6%�	q�R���A�%*;


total_loss�� A

error_R]�@?

learning_rate_1��8���&I       6%�	��R���A�%*;


total_loss��@

error_R��M?

learning_rate_1��8Ǎ��I       6%�	�R���A�%*;


total_loss:b�@

error_R�(h?

learning_rate_1��8�b'I       6%�	�XR���A�%*;


total_loss�eA

error_R�B[?

learning_rate_1��8G��sI       6%�	<�R���A�%*;


total_lossE�@

error_R��M?

learning_rate_1��8�X6�I       6%�	��R���A�%*;


total_losshӦ@

error_R�P?

learning_rate_1��8�K�uI       6%�	Y R���A�%*;


total_loss
��@

error_RO�A?

learning_rate_1��8S��SI       6%�	zW R���A�%*;


total_loss�N�@

error_R3S?

learning_rate_1��8�h5I       6%�	�� R���A�%*;


total_loss-��@

error_R�t\?

learning_rate_1��8Z��I       6%�	8� R���A�%*;


total_loss�C�@

error_R��P?

learning_rate_1��8��I       6%�	�!R���A�%*;


total_lossܭ�@

error_RJ�^?

learning_rate_1��8Y��hI       6%�	�^!R���A�%*;


total_loss�g�@

error_R��S?

learning_rate_1��8{W�fI       6%�	5�!R���A�%*;


total_loss�@

error_RL]?

learning_rate_1��8�qFI       6%�	��!R���A�%*;


total_loss]��@

error_R��j?

learning_rate_1��8����I       6%�	
!"R���A�&*;


total_lossFu�@

error_R��Q?

learning_rate_1��8��#yI       6%�	`"R���A�&*;


total_loss�Ҟ@

error_R}�T?

learning_rate_1��8,ɣVI       6%�	�"R���A�&*;


total_lossX�&A

error_R��T?

learning_rate_1��8VC�sI       6%�	s�"R���A�&*;


total_loss�b|@

error_R�OM?

learning_rate_1��8�R��I       6%�	6"#R���A�&*;


total_loss�u�@

error_Rl�P?

learning_rate_1��8��LI       6%�	�c#R���A�&*;


total_loss[�@

error_R��R?

learning_rate_1��8��vI       6%�	N�#R���A�&*;


total_loss��1A

error_R�nV?

learning_rate_1��8�rDqI       6%�	o�#R���A�&*;


total_lossL��@

error_R�D?

learning_rate_1��8Icz.I       6%�	�-$R���A�&*;


total_lossd,�@

error_R
N?

learning_rate_1��8&4��I       6%�	�r$R���A�&*;


total_losslI�@

error_R�W?

learning_rate_1��8�� �I       6%�	2�$R���A�&*;


total_loss�~k@

error_RZ?

learning_rate_1��8��v�I       6%�	��$R���A�&*;


total_loss&�@

error_R��]?

learning_rate_1��8)gbfI       6%�	W>%R���A�&*;


total_loss�h�@

error_R�:]?

learning_rate_1��8���I       6%�	�~%R���A�&*;


total_lossi��@

error_R}lZ?

learning_rate_1��8�&�UI       6%�	j�%R���A�&*;


total_lossP�A

error_R�E?

learning_rate_1��8E]�RI       6%�	�&R���A�&*;


total_loss4�A

error_R�VW?

learning_rate_1��8'O��I       6%�	&A&R���A�&*;


total_lossD¼@

error_RVKP?

learning_rate_1��8�3�I       6%�	@&R���A�&*;


total_loss��@

error_R*}a?

learning_rate_1��8WaE�I       6%�	>�&R���A�&*;


total_loss2r�@

error_R��X?

learning_rate_1��8�:Y7I       6%�	��&R���A�&*;


total_loss���@

error_RA�E?

learning_rate_1��8�a�I       6%�	�B'R���A�&*;


total_loss���@

error_R�dE?

learning_rate_1��8z-�I       6%�	��'R���A�&*;


total_loss̀�@

error_R��=?

learning_rate_1��8O���I       6%�	e�'R���A�&*;


total_loss�֩@

error_R��F?

learning_rate_1��8f�$I       6%�	�(R���A�&*;


total_lossJ)�@

error_Rc�R?

learning_rate_1��8{2�I       6%�	�U(R���A�&*;


total_lossU�@

error_R)\T?

learning_rate_1��8z~�DI       6%�	��(R���A�&*;


total_loss<�@

error_RA�j?

learning_rate_1��8\�eEI       6%�	��(R���A�&*;


total_loss��@

error_Rc�J?

learning_rate_1��8˒)2I       6%�	�)R���A�&*;


total_lossk�A

error_R_<B?

learning_rate_1��8V�J�I       6%�	�^)R���A�&*;


total_loss���@

error_R��K?

learning_rate_1��8�V�!I       6%�	��)R���A�&*;


total_loss�qA

error_R��P?

learning_rate_1��8#7TMI       6%�	^*R���A�&*;


total_loss���@

error_R(�U?

learning_rate_1��8�K��I       6%�	�N*R���A�&*;


total_loss���@

error_R�V?

learning_rate_1��8!��I       6%�	�*R���A�&*;


total_loss��@

error_R�b?

learning_rate_1��8U'��I       6%�	��*R���A�&*;


total_loss�X�@

error_R��C?

learning_rate_1��8�X��I       6%�	 +R���A�&*;


total_lossn��@

error_R�~K?

learning_rate_1��8KfXI       6%�	]+R���A�&*;


total_loss��@

error_RwuT?

learning_rate_1��8&�a�I       6%�	�+R���A�&*;


total_loss$��@

error_R�yU?

learning_rate_1��8m�ɠI       6%�	��+R���A�&*;


total_loss���@

error_R3va?

learning_rate_1��8��H�I       6%�	�-,R���A�&*;


total_loss7}A

error_R�%f?

learning_rate_1��8��KI       6%�	�u,R���A�&*;


total_loss�*�@

error_R��B?

learning_rate_1��8t��kI       6%�	D�,R���A�&*;


total_lossS��@

error_R8C?

learning_rate_1��8O{�I       6%�	�,R���A�&*;


total_loss�E�@

error_R4�X?

learning_rate_1��8,�>I       6%�	K7-R���A�&*;


total_lossC�A

error_RĩT?

learning_rate_1��8��1�I       6%�	�u-R���A�&*;


total_lossz\�@

error_R]?

learning_rate_1��8���I       6%�	A�-R���A�&*;


total_lossf��@

error_R@R?

learning_rate_1��8xl�I       6%�	9 .R���A�&*;


total_loss�н@

error_R\�`?

learning_rate_1��8� ��I       6%�	�@.R���A�&*;


total_loss��@

error_R#FZ?

learning_rate_1��8�ږ�I       6%�	,�.R���A�&*;


total_loss���@

error_Rܦ\?

learning_rate_1��8�5�I       6%�	V�.R���A�&*;


total_lossJ?�@

error_R��K?

learning_rate_1��8ƾ�I       6%�	�/R���A�&*;


total_loss��&A

error_RsaR?

learning_rate_1��8s�H�I       6%�	I/R���A�&*;


total_loss$µ@

error_R�D?

learning_rate_1��8�8wI       6%�	׏/R���A�&*;


total_lossC�4A

error_Rc�R?

learning_rate_1��8�a��I       6%�	Q�/R���A�&*;


total_loss�c*A

error_R��Y?

learning_rate_1��8��I       6%�	�0R���A�&*;


total_loss��@

error_R@�S?

learning_rate_1��8��I       6%�	�W0R���A�&*;


total_lossH��@

error_R=ZY?

learning_rate_1��8�	ΤI       6%�	��0R���A�&*;


total_loss%c�@

error_Roe??

learning_rate_1��8�S�7I       6%�	2�0R���A�&*;


total_loss�4@

error_RO�O?

learning_rate_1��8���I       6%�	�1R���A�&*;


total_loss��@

error_RE�E?

learning_rate_1��8�� I       6%�	�^1R���A�&*;


total_loss7�t@

error_R͒J?

learning_rate_1��8�#(�I       6%�	`�1R���A�&*;


total_loss
r�@

error_RQ�T?

learning_rate_1��8�B�I       6%�	=�1R���A�&*;


total_loss�w�@

error_RM�[?

learning_rate_1��8~B�=I       6%�	v-2R���A�&*;


total_lossW�@

error_RP?

learning_rate_1��8Tt=I       6%�	Rp2R���A�&*;


total_loss�s�@

error_R�6H?

learning_rate_1��8A�KI       6%�	²2R���A�&*;


total_loss�CA

error_R��^?

learning_rate_1��8v�7I       6%�	5�2R���A�&*;


total_loss,�c@

error_R��M?

learning_rate_1��8[��I       6%�	�13R���A�&*;


total_lossƨ@

error_R�S?

learning_rate_1��8j�)�I       6%�	�s3R���A�&*;


total_loss��@

error_RU]?

learning_rate_1��8��I       6%�	ճ3R���A�&*;


total_loss���@

error_R)mX?

learning_rate_1��8/!�I       6%�	3�3R���A�&*;


total_lossV&�@

error_R_F?

learning_rate_1��8�gn�I       6%�	324R���A�&*;


total_loss���@

error_R`�A?

learning_rate_1��84 +I       6%�	qw4R���A�&*;


total_losssk�@

error_R��Q?

learning_rate_1��8P)�dI       6%�	[�4R���A�&*;


total_lossJ)�@

error_R��E?

learning_rate_1��8�E��I       6%�	��4R���A�&*;


total_loss��@

error_R_�P?

learning_rate_1��8��NI       6%�	�>5R���A�&*;


total_losse	�@

error_RҩV?

learning_rate_1��8�ݸWI       6%�	s�5R���A�&*;


total_loss�Ԙ@

error_R/kE?

learning_rate_1��8C?�I       6%�	��5R���A�&*;


total_loss�q�@

error_RΎ]?

learning_rate_1��8��\�I       6%�	�	6R���A�&*;


total_lossQ��@

error_R�\?

learning_rate_1��8�� �I       6%�	J6R���A�&*;


total_loss_�@

error_R MV?

learning_rate_1��8�M�eI       6%�	ƈ6R���A�&*;


total_loss2Ǭ@

error_R�}X?

learning_rate_1��8c�ÁI       6%�	��6R���A�&*;


total_loss*�@

error_R�a?

learning_rate_1��8�#��I       6%�	�7R���A�&*;


total_loss�Ѡ@

error_R�hC?

learning_rate_1��84�g�I       6%�	H7R���A�&*;


total_loss�Қ@

error_R�@?

learning_rate_1��8Ҳ�"I       6%�	�7R���A�&*;


total_loss���@

error_R[�[?

learning_rate_1��8���I       6%�	,�7R���A�&*;


total_loss���@

error_Rl�A?

learning_rate_1��8c��I       6%�	u	8R���A�&*;


total_loss���@

error_RԾT?

learning_rate_1��8��b�I       6%�	H8R���A�&*;


total_loss��@

error_R�\?

learning_rate_1��8���I       6%�	��8R���A�&*;


total_loss���@

error_R��U?

learning_rate_1��8�hV�I       6%�	�8R���A�&*;


total_loss��e@

error_R,�C?

learning_rate_1��8P��jI       6%�	�
9R���A�&*;


total_loss@��@

error_R-�U?

learning_rate_1��8+���I       6%�	�N9R���A�&*;


total_loss���@

error_R��W?

learning_rate_1��8��oI       6%�	)�9R���A�&*;


total_lossq��@

error_R��B?

learning_rate_1��8��#�I       6%�	�9R���A�&*;


total_loss|x�@

error_R�.?

learning_rate_1��8��I       6%�	�6:R���A�&*;


total_loss���@

error_Rr�V?

learning_rate_1��8��qI       6%�	�w:R���A�&*;


total_loss���@

error_RCj?

learning_rate_1��8�"�I       6%�	��:R���A�&*;


total_loss�C�@

error_R��Y?

learning_rate_1��8�r.I       6%�	N;R���A�&*;


total_loss�V�@

error_R,�R?

learning_rate_1��8��&�I       6%�	a;R���A�&*;


total_lossT�@

error_R�TX?

learning_rate_1��8��6�I       6%�	<�;R���A�&*;


total_loss���@

error_R�]?

learning_rate_1��8�	�+I       6%�	��;R���A�&*;


total_loss1��@

error_R��X?

learning_rate_1��84�ȓI       6%�	�@<R���A�&*;


total_loss���@

error_R]?

learning_rate_1��8s0�I       6%�	ق<R���A�&*;


total_lossC��@

error_R��G?

learning_rate_1��8��H�I       6%�	l�<R���A�&*;


total_loss���@

error_RN?

learning_rate_1��8��I       6%�	�=R���A�&*;


total_loss�F�@

error_R�CO?

learning_rate_1��8P1!�I       6%�	�H=R���A�&*;


total_loss�B�@

error_R�L?

learning_rate_1��8�\�hI       6%�	��=R���A�&*;


total_lossT�@

error_RnXF?

learning_rate_1��8%���I       6%�	�=R���A�&*;


total_loss���@

error_R��9?

learning_rate_1��8�<�%I       6%�	�>R���A�&*;


total_lossd�8A

error_R��R?

learning_rate_1��8��3oI       6%�	�S>R���A�&*;


total_loss���@

error_R1ZF?

learning_rate_1��8%,\�I       6%�	W�>R���A�&*;


total_loss3��@

error_R]XD?

learning_rate_1��8y��1I       6%�	\�>R���A�&*;


total_loss!��@

error_R�O?

learning_rate_1��8�fk�I       6%�	�?R���A�&*;


total_loss�^�@

error_R��L?

learning_rate_1��8�/_�I       6%�	#c?R���A�&*;


total_loss-+�@

error_R��Q?

learning_rate_1��83�ZI       6%�	��?R���A�&*;


total_loss�J�@

error_RxoI?

learning_rate_1��8$W(�I       6%�	!�?R���A�&*;


total_loss�ƞ@

error_R��K?

learning_rate_1��8 ��dI       6%�	�)@R���A�&*;


total_loss3x�@

error_R-Y\?

learning_rate_1��8σ�I       6%�	On@R���A�&*;


total_loss,��@

error_R��U?

learning_rate_1��8�,�I       6%�	۲@R���A�&*;


total_lossآA

error_R�!_?

learning_rate_1��8��lI       6%�	~�@R���A�&*;


total_loss���@

error_R4�I?

learning_rate_1��8��,�I       6%�		:AR���A�&*;


total_loss���@

error_R�[?

learning_rate_1��8���I       6%�	�yAR���A�&*;


total_loss�&�@

error_R��Z?

learning_rate_1��8�o�fI       6%�	��AR���A�&*;


total_loss�֦@

error_R�lI?

learning_rate_1��8	g�I       6%�	��AR���A�&*;


total_loss	�@

error_R�R?

learning_rate_1��8�n,�I       6%�	�;BR���A�&*;


total_loss?;�@

error_RS�T?

learning_rate_1��8�<w�I       6%�	_|BR���A�&*;


total_lossiA�@

error_RZ[Y?

learning_rate_1��8����I       6%�	�BR���A�&*;


total_loss֟�@

error_R{T?

learning_rate_1��8h�uiI       6%�	��BR���A�&*;


total_lossL��@

error_Rf�Z?

learning_rate_1��8[��I       6%�	<CR���A�&*;


total_lossՀ�@

error_R�V?

learning_rate_1��8;2�I       6%�	b|CR���A�&*;


total_loss!��@

error_R��Y?

learning_rate_1��8f�x�I       6%�	9�CR���A�'*;


total_loss׹�@

error_R��H?

learning_rate_1��8�뱭I       6%�	<�CR���A�'*;


total_loss���@

error_R1�K?

learning_rate_1��8�x�7I       6%�	*?DR���A�'*;


total_loss��@

error_R��W?

learning_rate_1��8o�ݛI       6%�	~DR���A�'*;


total_loss �@

error_RT�M?

learning_rate_1��8'BaI       6%�	�DR���A�'*;


total_lossaDA

error_R�Z?

learning_rate_1��81n�I       6%�	UER���A�'*;


total_loss?�A

error_R16A?

learning_rate_1��8~XI       6%�	+DER���A�'*;


total_loss�ֽ@

error_RC�C?

learning_rate_1��8����I       6%�	��ER���A�'*;


total_lossA��@

error_R��A?

learning_rate_1��8c��I       6%�	��ER���A�'*;


total_loss���@

error_R߰\?

learning_rate_1��8e��HI       6%�	��HR���A�'*;


total_loss�P@

error_R��R?

learning_rate_1��8|�I       6%�	
!IR���A�'*;


total_loss��@

error_R�)S?

learning_rate_1��8Z���I       6%�	�dIR���A�'*;


total_lossķ�@

error_R@VL?

learning_rate_1��8�_�I       6%�	ΪIR���A�'*;


total_loss-��@

error_R�*J?

learning_rate_1��87K%I       6%�	�JR���A�'*;


total_lossn)�@

error_R�nG?

learning_rate_1��8[�&I       6%�	@IJR���A�'*;


total_loss���@

error_R#aP?

learning_rate_1��8���#I       6%�	�JR���A�'*;


total_loss��@

error_R�I?

learning_rate_1��8��.I       6%�	��JR���A�'*;


total_lossi�@

error_Rl-Z?

learning_rate_1��8�t?�I       6%�	�KR���A�'*;


total_loss�-A

error_Rq<?

learning_rate_1��8ݢ�I       6%�	9MKR���A�'*;


total_loss���@

error_R3iG?

learning_rate_1��8:{öI       6%�	*�KR���A�'*;


total_loss�ʱ@

error_RE�P?

learning_rate_1��8-*��I       6%�	��KR���A�'*;


total_loss�p�@

error_R&�Y?

learning_rate_1��8�OU�I       6%�	�LR���A�'*;


total_loss��@

error_R_hI?

learning_rate_1��8��(�I       6%�	BPLR���A�'*;


total_loss���@

error_R��[?

learning_rate_1��8kc8mI       6%�	T�LR���A�'*;


total_lossOd�@

error_R�D?

learning_rate_1��8�4I       6%�	��LR���A�'*;


total_lossxȯ@

error_R�.]?

learning_rate_1��8�/�UI       6%�	�MR���A�'*;


total_loss�'�@

error_R|�J?

learning_rate_1��8�W��I       6%�	SNMR���A�'*;


total_losss��@

error_R.�H?

learning_rate_1��8�:_iI       6%�	��MR���A�'*;


total_loss�@

error_R?K[?

learning_rate_1��85ׄI       6%�	��MR���A�'*;


total_loss��@

error_R��:?

learning_rate_1��8s�'|I       6%�	HNR���A�'*;


total_loss8��@

error_R�FA?

learning_rate_1��8bܠcI       6%�	�NNR���A�'*;


total_loss�@

error_R��:?

learning_rate_1��8�(e�I       6%�	�NR���A�'*;


total_lossmc�@

error_R�#S?

learning_rate_1��8�P}�I       6%�	J�NR���A�'*;


total_loss2A

error_R��X?

learning_rate_1��8�9�I       6%�	sOR���A�'*;


total_loss�%�@

error_R8�Z?

learning_rate_1��8\�v�I       6%�	�ROR���A�'*;


total_loss��@

error_R��R?

learning_rate_1��8��fI       6%�	#�OR���A�'*;


total_loss��@

error_R�^?

learning_rate_1��8W��%I       6%�	��OR���A�'*;


total_loss4�@

error_RI_N?

learning_rate_1��83�WI       6%�	PR���A�'*;


total_lossh�@

error_R:?

learning_rate_1��8�؏5I       6%�	�YPR���A�'*;


total_loss���@

error_R��Q?

learning_rate_1��82J"�I       6%�	��PR���A�'*;


total_loss���@

error_RA{B?

learning_rate_1��8y�I       6%�	��PR���A�'*;


total_losst�@

error_Rd�9?

learning_rate_1��80	I       6%�	>$QR���A�'*;


total_loss�'�@

error_R�j?

learning_rate_1��8;���I       6%�	idQR���A�'*;


total_loss�7A

error_R�#Q?

learning_rate_1��8=6M�I       6%�	��QR���A�'*;


total_loss�z�@

error_R�H?

learning_rate_1��8�x�I       6%�	��QR���A�'*;


total_loss냇@

error_R�U?

learning_rate_1��8��uI       6%�	w&RR���A�'*;


total_loss3��@

error_R�iS?

learning_rate_1��8���I       6%�	�fRR���A�'*;


total_loss�
�@

error_R��T?

learning_rate_1��8���I       6%�	��RR���A�'*;


total_lossRɥ@

error_R�P?

learning_rate_1��8��-I       6%�	(�RR���A�'*;


total_loss�l�@

error_R,�r?

learning_rate_1��8w�I       6%�	%)SR���A�'*;


total_loss��@

error_Ru??

learning_rate_1��8d��VI       6%�	�kSR���A�'*;


total_loss��A

error_R��M?

learning_rate_1��8�Y��I       6%�	˺SR���A�'*;


total_lossYA

error_R�`?

learning_rate_1��8�9��I       6%�	� TR���A�'*;


total_loss���@

error_R.�<?

learning_rate_1��8�z��I       6%�	�dTR���A�'*;


total_loss^��@

error_R��@?

learning_rate_1��8�I       6%�	s�TR���A�'*;


total_loss	��@

error_R�B?

learning_rate_1��8;Q��I       6%�	��TR���A�'*;


total_loss�b�@

error_R�X?

learning_rate_1��8gצ�I       6%�	�UUR���A�'*;


total_lossQ!�@

error_R8)V?

learning_rate_1��8^��JI       6%�	~�UR���A�'*;


total_loss�ٟ@

error_R�P?

learning_rate_1��8w�I       6%�	k�UR���A�'*;


total_loss���@

error_R3 K?

learning_rate_1��8�LQ1I       6%�	^'VR���A�'*;


total_lossl��@

error_R��S?

learning_rate_1��8#��CI       6%�	[�VR���A�'*;


total_lossЅ�@

error_R�';?

learning_rate_1��8ITq�I       6%�	X�VR���A�'*;


total_lossaO�@

error_R�`?

learning_rate_1��8��hI       6%�	CWR���A�'*;


total_loss�S�@

error_R�hJ?

learning_rate_1��8�=I%I       6%�	YiWR���A�'*;


total_lossNn�@

error_Rf2N?

learning_rate_1��89��wI       6%�	R�WR���A�'*;


total_loss��@

error_RΨ5?

learning_rate_1��8���I       6%�	�XR���A�'*;


total_lossd�~@

error_R�0\?

learning_rate_1��8L��I       6%�	BWXR���A�'*;


total_lossa;A

error_R,_M?

learning_rate_1��8�R�I       6%�		�XR���A�'*;


total_loss��@

error_RJ�=?

learning_rate_1��8�]g�I       6%�	��XR���A�'*;


total_loss_K�@

error_R�dd?

learning_rate_1��8��	I       6%�	�SYR���A�'*;


total_lossI��@

error_Rd+U?

learning_rate_1��8��EI       6%�	ؔYR���A�'*;


total_lossc[�@

error_Rr7N?

learning_rate_1��8�o~�I       6%�	@�YR���A�'*;


total_loss,��@

error_R��H?

learning_rate_1��8��t|I       6%�	RBZR���A�'*;


total_loss��@

error_RSG?

learning_rate_1��8��9�I       6%�	l�ZR���A�'*;


total_loss\�6A

error_R�"L?

learning_rate_1��8xcB5I       6%�	��ZR���A�'*;


total_loss�-�@

error_R�oX?

learning_rate_1��8���I       6%�	,B[R���A�'*;


total_loss��@

error_Rm�]?

learning_rate_1��8I:J�I       6%�	G�[R���A�'*;


total_loss@

error_Rz�S?

learning_rate_1��8d~lI       6%�	��[R���A�'*;


total_loss�T�@

error_R��S?

learning_rate_1��8���I       6%�	�"\R���A�'*;


total_lossWU�@

error_R��:?

learning_rate_1��8� 5�I       6%�	�l\R���A�'*;


total_loss�Ť@

error_R��V?

learning_rate_1��8`^I       6%�	��\R���A�'*;


total_loss��@

error_RsN?

learning_rate_1��8���I       6%�	�]R���A�'*;


total_lossl��@

error_R,
V?

learning_rate_1��8��%I       6%�	�N]R���A�'*;


total_loss�A

error_R
6K?

learning_rate_1��8P��I       6%�	6�]R���A�'*;


total_loss=ܻ@

error_R*{<?

learning_rate_1��8C</I       6%�	
�]R���A�'*;


total_loss!HA

error_Rx0h?

learning_rate_1��8�OTI       6%�	�)^R���A�'*;


total_lossrA

error_RW�L?

learning_rate_1��8$El�I       6%�	m^R���A�'*;


total_lossx"�@

error_R�G?

learning_rate_1��8E�I       6%�	�^R���A�'*;


total_loss���@

error_R8�h?

learning_rate_1��8=��I       6%�	��^R���A�'*;


total_loss���@

error_R.�.?

learning_rate_1��8�k!I       6%�	�3_R���A�'*;


total_lossvT�@

error_RxO?

learning_rate_1��8� �uI       6%�	f�_R���A�'*;


total_loss͈�@

error_R E?

learning_rate_1��8n��^I       6%�	��_R���A�'*;


total_loss�3�@

error_R},L?

learning_rate_1��8�.I       6%�	�
`R���A�'*;


total_loss��@

error_R��H?

learning_rate_1��8B�@LI       6%�	�L`R���A�'*;


total_loss�u�@

error_Rf%H?

learning_rate_1��8�T��I       6%�	�`R���A�'*;


total_loss��@

error_RTv\?

learning_rate_1��8����I       6%�	��`R���A�'*;


total_loss��@

error_R�^U?

learning_rate_1��8{2��I       6%�	�aR���A�'*;


total_loss9͈@

error_Rt]?

learning_rate_1��8L4I       6%�	9]aR���A�'*;


total_losspI�@

error_R��Q?

learning_rate_1��8��z�I       6%�	,�aR���A�'*;


total_loss�B�@

error_R�/J?

learning_rate_1��8TG�RI       6%�	B�aR���A�'*;


total_loss`��@

error_R��P?

learning_rate_1��8����I       6%�	4bR���A�'*;


total_loss�@

error_R,QS?

learning_rate_1��8�po}I       6%�	�|bR���A�'*;


total_loss���@

error_RܫK?

learning_rate_1��8��I       6%�	��bR���A�'*;


total_loss��@

error_R=UJ?

learning_rate_1��8��q�I       6%�	�
cR���A�'*;


total_lossV��@

error_R��H?

learning_rate_1��8��I       6%�	YRcR���A�'*;


total_lossJ�j@

error_R,�J?

learning_rate_1��8E��6I       6%�	��cR���A�'*;


total_lossg�@

error_R}[?

learning_rate_1��8J��HI       6%�	��cR���A�'*;


total_loss���@

error_R|�B?

learning_rate_1��8�:�'I       6%�	dR���A�'*;


total_loss��A

error_R1�K?

learning_rate_1��8�t)I       6%�	�cdR���A�'*;


total_loss���@

error_R̕[?

learning_rate_1��8D�m�I       6%�	ԥdR���A�'*;


total_loss��@

error_RdeH?

learning_rate_1��8}��;I       6%�	�dR���A�'*;


total_loss�f�@

error_R�[?

learning_rate_1��8|~K�I       6%�	y+eR���A�'*;


total_loss�@

error_R�,K?

learning_rate_1��8����I       6%�	�oeR���A�'*;


total_loss �@

error_R��<?

learning_rate_1��8ǩȸI       6%�	,�eR���A�'*;


total_lossaF�@

error_R#]Q?

learning_rate_1��8zI       6%�	D�eR���A�'*;


total_lossN��@

error_R{	J?

learning_rate_1��8���^I       6%�	P@fR���A�'*;


total_loss���@

error_R��M?

learning_rate_1��8�ۙ�I       6%�	�fR���A�'*;


total_lossO�@

error_Rf�C?

learning_rate_1��8A��I       6%�	��fR���A�'*;


total_lossW�@

error_R[ U?

learning_rate_1��8ҔH�I       6%�	�gR���A�'*;


total_loss�o�@

error_R�M?

learning_rate_1��8}��^I       6%�	�_gR���A�'*;


total_loss=��@

error_R��E?

learning_rate_1��8�Y�I       6%�	��gR���A�'*;


total_lossf>�@

error_R��I?

learning_rate_1��8�x�I       6%�	��gR���A�'*;


total_loss��A

error_RȺV?

learning_rate_1��8XN�I       6%�	F&hR���A�'*;


total_loss,Z@

error_R��`?

learning_rate_1��8��o5I       6%�	ahhR���A�'*;


total_loss�"�@

error_R��M?

learning_rate_1��8] riI       6%�	|�hR���A�'*;


total_loss�ߵ@

error_RTP?

learning_rate_1��8r��TI       6%�	��hR���A�'*;


total_loss�@

error_R�/U?

learning_rate_1��8s�UmI       6%�	R-iR���A�'*;


total_loss��@

error_R e^?

learning_rate_1��8!�cXI       6%�	�miR���A�'*;


total_loss}Q�@

error_R��`?

learning_rate_1��8/W\I       6%�	��iR���A�(*;


total_lossҫ�@

error_R�V?

learning_rate_1��8�	�I       6%�	�jR���A�(*;


total_loss`��@

error_RW2[?

learning_rate_1��8�WrEI       6%�	�ajR���A�(*;


total_loss�F�@

error_R6�U?

learning_rate_1��8��-�I       6%�	f�jR���A�(*;


total_loss ��@

error_R�[?

learning_rate_1��8J:I       6%�	/�jR���A�(*;


total_loss�f�@

error_R�!T?

learning_rate_1��8�� I       6%�	�*kR���A�(*;


total_loss���@

error_R�M?

learning_rate_1��8k�ALI       6%�	�nkR���A�(*;


total_loss�Ƨ@

error_R�9I?

learning_rate_1��8�D/UI       6%�	D�kR���A�(*;


total_loss�A

error_R��C?

learning_rate_1��8����I       6%�	x�kR���A�(*;


total_lossƜ�@

error_RT�E?

learning_rate_1��8*�PI       6%�	�5lR���A�(*;


total_loss�P�@

error_RiEV?

learning_rate_1��8�[�I       6%�	�zlR���A�(*;


total_lossH�N@

error_RZ(=?

learning_rate_1��8��I       6%�	@�lR���A�(*;


total_loss���@

error_R�%K?

learning_rate_1��8;��TI       6%�	GmR���A�(*;


total_loss 2�@

error_R��M?

learning_rate_1��8h�8�I       6%�	pEmR���A�(*;


total_lossiz@

error_R��F?

learning_rate_1��8�F�I       6%�	@�mR���A�(*;


total_loss�ͪ@

error_R|Q]?

learning_rate_1��8����I       6%�	�mR���A�(*;


total_loss3Ӛ@

error_Rn�]?

learning_rate_1��8��ּI       6%�	knR���A�(*;


total_loss��u@

error_R�k?

learning_rate_1��8`�A�I       6%�	�UnR���A�(*;


total_loss�f�@

error_R�A_?

learning_rate_1��84�!vI       6%�	�nR���A�(*;


total_loss��@

error_R��W?

learning_rate_1��8��O8I       6%�	��nR���A�(*;


total_loss�}@

error_R��U?

learning_rate_1��8HCJI       6%�	�oR���A�(*;


total_loss6��@

error_R��T?

learning_rate_1��8�_�I       6%�	�\oR���A�(*;


total_loss/�@

error_RS?

learning_rate_1��8�DӢI       6%�	,�oR���A�(*;


total_lossZY�@

error_R-a?

learning_rate_1��8i+�BI       6%�	��oR���A�(*;


total_loss;d�@

error_R�NQ?

learning_rate_1��8��10I       6%�	H.pR���A�(*;


total_loss���@

error_R�`?

learning_rate_1��8ٯ.�I       6%�	�qpR���A�(*;


total_loss��@

error_R��8?

learning_rate_1��8�)�I       6%�	z�pR���A�(*;


total_loss^��@

error_R��\?

learning_rate_1��8�&�jI       6%�	��pR���A�(*;


total_lossܺ�@

error_R
C?

learning_rate_1��8_��I       6%�	�:qR���A�(*;


total_loss��@

error_Rl�M?

learning_rate_1��84{�I       6%�	o}qR���A�(*;


total_loss�Y�@

error_R��:?

learning_rate_1��8��9hI       6%�	��qR���A�(*;


total_loss�+�@

error_R�Z?

learning_rate_1��8�d�I       6%�	�rR���A�(*;


total_loss|��@

error_R�V?

learning_rate_1��8��I�I       6%�	�IrR���A�(*;


total_loss�߮@

error_R��D?

learning_rate_1��8�1R�I       6%�	"�rR���A�(*;


total_loss(��@

error_R�>?

learning_rate_1��89���I       6%�	��rR���A�(*;


total_lossR��@

error_R�L?

learning_rate_1��8�k0fI       6%�	�sR���A�(*;


total_lossH%�@

error_R��R?

learning_rate_1��8qƓBI       6%�	�csR���A�(*;


total_loss���@

error_R�N?

learning_rate_1��8�5�LI       6%�	 �sR���A�(*;


total_lossc��@

error_R.�V?

learning_rate_1��8���I       6%�	��sR���A�(*;


total_loss�ʬ@

error_R4N[?

learning_rate_1��8J�m0I       6%�	k@tR���A�(*;


total_lossѺ�@

error_R	aB?

learning_rate_1��83Q`�I       6%�	Z�tR���A�(*;


total_loss_b�@

error_Rg?

learning_rate_1��8?U�I       6%�	��tR���A�(*;


total_lossWXA

error_R�3_?

learning_rate_1��8G
I       6%�	tuR���A�(*;


total_loss���@

error_RK?

learning_rate_1��8��I       6%�	�QuR���A�(*;


total_lossә�@

error_R�	P?

learning_rate_1��8�N�I       6%�	��uR���A�(*;


total_lossJ��@

error_RjQ?

learning_rate_1��8ۊ+I       6%�	S�uR���A�(*;


total_loss|��@

error_RAE_?

learning_rate_1��8�=��I       6%�	�vR���A�(*;


total_loss��@

error_R��=?

learning_rate_1��8��i�I       6%�	_vR���A�(*;


total_loss���@

error_RaqO?

learning_rate_1��8�u�(I       6%�	��vR���A�(*;


total_loss��@

error_R�J?

learning_rate_1��8J��HI       6%�	��vR���A�(*;


total_lossW�@

error_R}Q?

learning_rate_1��8��0I       6%�	O+wR���A�(*;


total_loss��A

error_R�)D?

learning_rate_1��8�\keI       6%�	�owR���A�(*;


total_loss<�
A

error_R)�S?

learning_rate_1��8:���I       6%�	[�wR���A�(*;


total_lossJ��@

error_R�5N?

learning_rate_1��8ݔYI       6%�	�wR���A�(*;


total_lossDt�@

error_R�}S?

learning_rate_1��8w��I       6%�	P7xR���A�(*;


total_lossmܨ@

error_R�tG?

learning_rate_1��8�g�I       6%�	�xxR���A�(*;


total_loss�ˢ@

error_R�[?

learning_rate_1��8T6�I       6%�	�xR���A�(*;


total_loss��@

error_RS?

learning_rate_1��8c�yI       6%�	�yR���A�(*;


total_lossA�A

error_R��M?

learning_rate_1��8�^#�I       6%�	�KyR���A�(*;


total_loss\�A

error_R}�R?

learning_rate_1��8�^}I       6%�	6�yR���A�(*;


total_loss|�@

error_R1�??

learning_rate_1��8|��I       6%�	��yR���A�(*;


total_loss�;�@

error_Ra�Q?

learning_rate_1��8A�HI       6%�	�UzR���A�(*;


total_lossᵕ@

error_R4�^?

learning_rate_1��8�ޓCI       6%�	�zR���A�(*;


total_loss�A�@

error_R�D[?

learning_rate_1��8��I       6%�	��zR���A�(*;


total_lossG{�@

error_R�M?

learning_rate_1��8����I       6%�	�I{R���A�(*;


total_loss�T�@

error_R��S?

learning_rate_1��8��)�I       6%�	��{R���A�(*;


total_loss*N�@

error_R͹U?

learning_rate_1��8O�fI       6%�	 �{R���A�(*;


total_lossb�@

error_R�-Q?

learning_rate_1��8�x"�I       6%�	s|R���A�(*;


total_loss�U�@

error_R7�O?

learning_rate_1��80�IJI       6%�	�_|R���A�(*;


total_lossԥ�@

error_R��V?

learning_rate_1��8���I       6%�	��|R���A�(*;


total_lossZ��@

error_R�I?

learning_rate_1��8:ǈ�I       6%�	��|R���A�(*;


total_loss	ݦ@

error_R;P\?

learning_rate_1��8��l�I       6%�	&.}R���A�(*;


total_loss���@

error_R!{^?

learning_rate_1��8���I       6%�	iv}R���A�(*;


total_lossd�@

error_R�KY?

learning_rate_1��8���gI       6%�	 �}R���A�(*;


total_loss��@

error_RR?

learning_rate_1��8t~�~I       6%�	i~R���A�(*;


total_loss �A

error_R 3B?

learning_rate_1��8>W��I       6%�	CS~R���A�(*;


total_loss�m�@

error_R��S?

learning_rate_1��8��I       6%�	��~R���A�(*;


total_loss�A�@

error_R��D?

learning_rate_1��8�<�>I       6%�	'�~R���A�(*;


total_loss��@

error_R:S?

learning_rate_1��8>ZI       6%�	_R���A�(*;


total_lossl~�@

error_Ra�E?

learning_rate_1��8��MI       6%�	g]R���A�(*;


total_loss:8�@

error_R	BO?

learning_rate_1��8p�[pI       6%�	w�R���A�(*;


total_lossN�@

error_R�rC?

learning_rate_1��8z?h�I       6%�	�R���A�(*;


total_loss�Ó@

error_R�X?

learning_rate_1��8�٦PI       6%�	�(�R���A�(*;


total_loss`��@

error_R�IO?

learning_rate_1��85�N�I       6%�	�j�R���A�(*;


total_loss��@

error_RH�O?

learning_rate_1��8��I       6%�	B��R���A�(*;


total_lossK��@

error_RsvA?

learning_rate_1��8��I       6%�	B�R���A�(*;


total_loss�u�@

error_RT|^?

learning_rate_1��8y!I       6%�	;=�R���A�(*;


total_loss���@

error_R��T?

learning_rate_1��8�²tI       6%�	ƛ�R���A�(*;


total_loss���@

error_R��G?

learning_rate_1��8�fI       6%�	3�R���A�(*;


total_loss�x@

error_R��X?

learning_rate_1��8��b�I       6%�	7�R���A�(*;


total_lossl��@

error_RF'b?

learning_rate_1��8�Z� I       6%�	~�R���A�(*;


total_lossA�@

error_R�.Q?

learning_rate_1��8����I       6%�	�R���A�(*;


total_loss&�A

error_RA�R?

learning_rate_1��8^�sI       6%�	w�R���A�(*;


total_loss���@

error_R��G?

learning_rate_1��8����I       6%�	�O�R���A�(*;


total_lossrj�@

error_R\?

learning_rate_1��8�;�"I       6%�	���R���A�(*;


total_loss�Z�@

error_R;�X?

learning_rate_1��8+5��I       6%�	��R���A�(*;


total_lossVE�@

error_R��^?

learning_rate_1��8��I       6%�	�,�R���A�(*;


total_loss,�@

error_RX8N?

learning_rate_1��8�[+I       6%�	bv�R���A�(*;


total_loss��A

error_RE�]?

learning_rate_1��8�^NI       6%�	:��R���A�(*;


total_loss&ӧ@

error_R�lM?

learning_rate_1��8n6=I       6%�	X�R���A�(*;


total_loss�r�@

error_R��O?

learning_rate_1��85�y;I       6%�	�L�R���A�(*;


total_loss���@

error_R��N?

learning_rate_1��8H���I       6%�	i��R���A�(*;


total_loss��@

error_R�,W?

learning_rate_1��8���I       6%�	�ԅR���A�(*;


total_loss�W�@

error_R�R?

learning_rate_1��8H��I       6%�		�R���A�(*;


total_loss*��@

error_Ra�C?

learning_rate_1��8�I       6%�	�b�R���A�(*;


total_loss윗@

error_R{O?

learning_rate_1��8,Y:I       6%�	U��R���A�(*;


total_lossRa�@

error_RݤA?

learning_rate_1��81^e�I       6%�	J�R���A�(*;


total_loss�'�@

error_R�J?

learning_rate_1��8ЗI       6%�	�?�R���A�(*;


total_loss�(�@

error_R��X?

learning_rate_1��8�L'>I       6%�	���R���A�(*;


total_loss�X�@

error_R�oJ?

learning_rate_1��8����I       6%�	ӇR���A�(*;


total_lossE�@

error_R��K?

learning_rate_1��8�c�DI       6%�	��R���A�(*;


total_loss��@

error_R,�S?

learning_rate_1��8W��wI       6%�	�a�R���A�(*;


total_loss�J�@

error_R�^I?

learning_rate_1��8)� 	I       6%�	���R���A�(*;


total_lossC��@

error_RVsW?

learning_rate_1��8nݳ�I       6%�	��R���A�(*;


total_loss���@

error_R�;L?

learning_rate_1��8�!��I       6%�	-0�R���A�(*;


total_loss<S�@

error_R�=I?

learning_rate_1��8�^"cI       6%�	t�R���A�(*;


total_lossl�@

error_Rz�X?

learning_rate_1��8��|�I       6%�	F��R���A�(*;


total_loss��@

error_RV�U?

learning_rate_1��8��v�I       6%�	*�R���A�(*;


total_loss6ߪ@

error_Rd�L?

learning_rate_1��8��p�I       6%�	z�R���A�(*;


total_loss-�@

error_R��Z?

learning_rate_1��8��:�I       6%�	ZÊR���A�(*;


total_loss���@

error_R��K?

learning_rate_1��8>��I       6%�	a�R���A�(*;


total_loss]I�@

error_R��Y?

learning_rate_1��84�I�I       6%�	4S�R���A�(*;


total_loss4��@

error_RhgF?

learning_rate_1��87��I       6%�	���R���A�(*;


total_loss�@�@

error_R�K?

learning_rate_1��8�?I       6%�	wڋR���A�(*;


total_loss�B�@

error_R{wJ?

learning_rate_1��8�!yI       6%�	� �R���A�(*;


total_losse�)A

error_R8i]?

learning_rate_1��8.��I       6%�	�c�R���A�(*;


total_lossͪ�@

error_RibR?

learning_rate_1��812I       6%�	���R���A�(*;


total_loss�-�@

error_R�8O?

learning_rate_1��8�z�I       6%�	�R���A�(*;


total_loss$��@

error_R*�N?

learning_rate_1��8$|��I       6%�	�;�R���A�)*;


total_lossJ�@

error_RH�\?

learning_rate_1��8��I       6%�	���R���A�)*;


total_loss] �@

error_RhF?

learning_rate_1��8��2:I       6%�	�ǍR���A�)*;


total_loss���@

error_R1]?

learning_rate_1��8 �h�I       6%�	�R���A�)*;


total_loss���@

error_R ^?

learning_rate_1��8exEZI       6%�	bX�R���A�)*;


total_lossa�@

error_RO?

learning_rate_1��8M���I       6%�	[��R���A�)*;


total_lossVI�@

error_R��Y?

learning_rate_1��8����I       6%�	3�R���A�)*;


total_loss���@

error_R�Y?

learning_rate_1��8$w�$I       6%�	,1�R���A�)*;


total_loss���@

error_R��W?

learning_rate_1��8K*krI       6%�	uu�R���A�)*;


total_loss*�@

error_R�%M?

learning_rate_1��8;�nOI       6%�	��R���A�)*;


total_loss&W�@

error_R�7O?

learning_rate_1��8���.I       6%�	���R���A�)*;


total_loss���@

error_R�TE?

learning_rate_1��8���wI       6%�	2B�R���A�)*;


total_lossϔ�@

error_R
�<?

learning_rate_1��8�?�I       6%�	���R���A�)*;


total_loss���@

error_RR�N?

learning_rate_1��8E���I       6%�	�̐R���A�)*;


total_loss���@

error_R�[Q?

learning_rate_1��80�~I       6%�	-�R���A�)*;


total_loss,�x@

error_R�Tr?

learning_rate_1��8����I       6%�	�U�R���A�)*;


total_loss�	A

error_R�MP?

learning_rate_1��8~��I       6%�	6��R���A�)*;


total_loss��A

error_R\�S?

learning_rate_1��8i2d@I       6%�	��R���A�)*;


total_lossE��@

error_R�VH?

learning_rate_1��8P7�VI       6%�	�*�R���A�)*;


total_lossB.�@

error_RM�>?

learning_rate_1��8�*��I       6%�	�n�R���A�)*;


total_loss-k�@

error_R�el?

learning_rate_1��8&iHI       6%�	���R���A�)*;


total_loss=�B@

error_R��U?

learning_rate_1��8�kI       6%�	w��R���A�)*;


total_loss�0�@

error_R��S?

learning_rate_1��8�_�I       6%�	�;�R���A�)*;


total_loss��@

error_R}[??

learning_rate_1��8}�`$I       6%�	'~�R���A�)*;


total_lossx��@

error_R�J?

learning_rate_1��8iW�I       6%�	*ēR���A�)*;


total_loss���@

error_R�dE?

learning_rate_1��8���I       6%�	��R���A�)*;


total_lossO��@

error_R�H?

learning_rate_1��8 K�YI       6%�	�K�R���A�)*;


total_loss�+�@

error_R�4=?

learning_rate_1��8N@�I       6%�	���R���A�)*;


total_lossƵ�@

error_R�rO?

learning_rate_1��8ƃ�FI       6%�	�ٔR���A�)*;


total_loss\�y@

error_R�R7?

learning_rate_1��8�}*�I       6%�	��R���A�)*;


total_loss�A�@

error_R�VM?

learning_rate_1��8�|��I       6%�	Vd�R���A�)*;


total_loss���@

error_R��h?

learning_rate_1��8Cw�I       6%�	c��R���A�)*;


total_loss7��@

error_Rc�J?

learning_rate_1��8����I       6%�	� �R���A�)*;


total_loss��#A

error_R��L?

learning_rate_1��8�4ZQI       6%�	E�R���A�)*;


total_loss��@

error_R��T?

learning_rate_1��8BD�*I       6%�	^��R���A�)*;


total_loss$I�@

error_R�XM?

learning_rate_1��8�ο2I       6%�	�ҖR���A�)*;


total_loss���@

error_R�P?

learning_rate_1��83�eI       6%�	�R���A�)*;


total_lossU̓@

error_R��@?

learning_rate_1��8ѧ	�I       6%�	�e�R���A�)*;


total_loss�	�@

error_R?ZK?

learning_rate_1��8T��I       6%�	'��R���A�)*;


total_lossC��@

error_R��U?

learning_rate_1��8���zI       6%�	���R���A�)*;


total_lossfA

error_R�1J?

learning_rate_1��8s�rI       6%�	�C�R���A�)*;


total_loss�n	A

error_RMJT?

learning_rate_1��8��2I       6%�	��R���A�)*;


total_loss#J�@

error_Rf�9?

learning_rate_1��8�O.I       6%�	0ݘR���A�)*;


total_loss�U�@

error_R��L?

learning_rate_1��81���I       6%�	�%�R���A�)*;


total_loss|��@

error_RU?

learning_rate_1��8�{��I       6%�	Hk�R���A�)*;


total_loss0�@

error_R�I?

learning_rate_1��8����I       6%�	-��R���A�)*;


total_lossX�@

error_R6�E?

learning_rate_1��8�x�wI       6%�	L�R���A�)*;


total_loss�$A

error_RMXa?

learning_rate_1��8��\�I       6%�	Y�R���A�)*;


total_loss�9�@

error_R�vI?

learning_rate_1��8!�U�I       6%�	0��R���A�)*;


total_lossr��@

error_R��i?

learning_rate_1��8e6KvI       6%�	��R���A�)*;


total_loss:�@

error_RmIZ?

learning_rate_1��8��5WI       6%�	B$�R���A�)*;


total_loss���@

error_R�#C?

learning_rate_1��8
Հ�I       6%�	Zj�R���A�)*;


total_loss��@

error_R!�a?

learning_rate_1��8���I       6%�	
��R���A�)*;


total_loss�z�@

error_R�/H?

learning_rate_1��8cc�I       6%�	v��R���A�)*;


total_loss���@

error_R_v5?

learning_rate_1��8�<�I       6%�	�@�R���A�)*;


total_loss�Qr@

error_R��R?

learning_rate_1��8��C�I       6%�	F��R���A�)*;


total_loss$�@

error_R�&T?

learning_rate_1��8?��kI       6%�	�ʜR���A�)*;


total_loss���@

error_Rq�Q?

learning_rate_1��8H��(I       6%�	��R���A�)*;


total_loss���@

error_Rr�U?

learning_rate_1��8��gI       6%�	�S�R���A�)*;


total_loss\A

error_RE�E?

learning_rate_1��8\I�I       6%�	�R���A�)*;


total_loss^M�@

error_R� U?

learning_rate_1��8�{F�I       6%�	��R���A�)*;


total_loss��@

error_R�Y?

learning_rate_1��8G��I       6%�	�6�R���A�)*;


total_lossS7�@

error_R��J?

learning_rate_1��8��1I       6%�	�}�R���A�)*;


total_loss�@�@

error_R�'^?

learning_rate_1��8{��uI       6%�	�ÞR���A�)*;


total_loss�fA

error_Rx�[?

learning_rate_1��8K��I       6%�	��R���A�)*;


total_loss/k�@

error_R��f?

learning_rate_1��8�\ +I       6%�	8X�R���A�)*;


total_lossi��@

error_R�)T?

learning_rate_1��8���>I       6%�	"��R���A�)*;


total_lossFj�@

error_R8�O?

learning_rate_1��8)��I       6%�	��R���A�)*;


total_loss�~�@

error_R��Z?

learning_rate_1��8b'�3I       6%�	�'�R���A�)*;


total_loss��@

error_R�S?

learning_rate_1��8�䈯I       6%�	wl�R���A�)*;


total_loss��@

error_R�MM?

learning_rate_1��8SarI       6%�	T��R���A�)*;


total_loss���@

error_R��C?

learning_rate_1��8�:�dI       6%�	F��R���A�)*;


total_lossM�_@

error_Ra�J?

learning_rate_1��8�7BI       6%�	�9�R���A�)*;


total_loss1�@

error_RlsM?

learning_rate_1��8%]a�I       6%�	�~�R���A�)*;


total_loss"�@

error_R��J?

learning_rate_1��8:5�lI       6%�	�šR���A�)*;


total_loss��@

error_R�#K?

learning_rate_1��8$m�|I       6%�	�	�R���A�)*;


total_loss�w�@

error_R�]E?

learning_rate_1��8r�I       6%�	uL�R���A�)*;


total_loss���@

error_R�X?

learning_rate_1��8:*�I       6%�	���R���A�)*;


total_loss��@

error_R�RE?

learning_rate_1��8Y�I       6%�	/עR���A�)*;


total_loss8R�@

error_Ra�9?

learning_rate_1��8<)��I       6%�	��R���A�)*;


total_loss��@

error_R��L?

learning_rate_1��8:��I       6%�	Ub�R���A�)*;


total_loss,}�@

error_R�U?

learning_rate_1��8�N^aI       6%�	��R���A�)*;


total_lossJ��@

error_R�rM?

learning_rate_1��8��0�I       6%�	��R���A�)*;


total_loss ��@

error_R^W?

learning_rate_1��8�_�I       6%�	�4�R���A�)*;


total_loss��@

error_R�DO?

learning_rate_1��8�#�I       6%�	�x�R���A�)*;


total_lossa)7A

error_R��B?

learning_rate_1��8P�6I       6%�	���R���A�)*;


total_loss�YUA

error_R_+G?

learning_rate_1��8L1
eI       6%�	1 �R���A�)*;


total_loss���@

error_R7�R?

learning_rate_1��8W֐�I       6%�	B�R���A�)*;


total_loss��A

error_R��H?

learning_rate_1��8�x �I       6%�	 ��R���A�)*;


total_loss���@

error_R�JA?

learning_rate_1��8 �� I       6%�	#ΥR���A�)*;


total_lossA

error_R<�\?

learning_rate_1��8e�I       6%�	S�R���A�)*;


total_loss���@

error_R�Y?

learning_rate_1��8�,nI       6%�	WV�R���A�)*;


total_loss��@

error_R�\?

learning_rate_1��8�\3I       6%�	���R���A�)*;


total_loss`��@

error_R�8W?

learning_rate_1��8�<��I       6%�	�R���A�)*;


total_loss�A

error_R�eX?

learning_rate_1��8����I       6%�	p3�R���A�)*;


total_loss�8�@

error_R�Q?

learning_rate_1��8C�RkI       6%�	�}�R���A�)*;


total_loss�F�@

error_R��D?

learning_rate_1��8ʌ��I       6%�	JǧR���A�)*;


total_loss��@

error_R��Z?

learning_rate_1��8=}�I       6%�	�
�R���A�)*;


total_lossrҷ@

error_R��6?

learning_rate_1��8��G8I       6%�	�R�R���A�)*;


total_loss�2�@

error_Rl�\?

learning_rate_1��8����I       6%�	���R���A�)*;


total_loss	�@

error_R��[?

learning_rate_1��8� ]�I       6%�	AݨR���A�)*;


total_lossMv�@

error_R�F?

learning_rate_1��8� ^�I       6%�	�"�R���A�)*;


total_loss�A

error_Re�O?

learning_rate_1��8��^�I       6%�	�e�R���A�)*;


total_loss���@

error_R��Q?

learning_rate_1��8t?sI       6%�	j��R���A�)*;


total_loss��A

error_R�:9?

learning_rate_1��8_��I       6%�	#�R���A�)*;


total_loss�N�@

error_R�U?

learning_rate_1��8�k�4I       6%�	�O�R���A�)*;


total_lossHQ�@

error_R��N?

learning_rate_1��8W}� I       6%�	��R���A�)*;


total_loss/��@

error_Rl*V?

learning_rate_1��8�w|�I       6%�	WݪR���A�)*;


total_loss;;�@

error_R{�F?

learning_rate_1��8L��xI       6%�	
&�R���A�)*;


total_lossD��@

error_R�P?

learning_rate_1��8u��I       6%�	�h�R���A�)*;


total_loss��@

error_R)�f?

learning_rate_1��8;s�I       6%�	ò�R���A�)*;


total_loss��@

error_R�S?

learning_rate_1��8�XV�I       6%�	���R���A�)*;


total_loss��@

error_RWO?

learning_rate_1��8^.c~I       6%�	�D�R���A�)*;


total_loss}��@

error_R̗L?

learning_rate_1��8o?\I       6%�	���R���A�)*;


total_loss{	�@

error_Rf�??

learning_rate_1��8fA7I       6%�	ЬR���A�)*;


total_loss���@

error_R6�J?

learning_rate_1��8|vW�I       6%�	��R���A�)*;


total_loss QA

error_R�uO?

learning_rate_1��8���I       6%�	�]�R���A�)*;


total_loss�#�@

error_R�
_?

learning_rate_1��8���I       6%�	\��R���A�)*;


total_loss���@

error_R�tD?

learning_rate_1��8dɆNI       6%�	��R���A�)*;


total_loss{ �@

error_R׫N?

learning_rate_1��8��KI       6%�	�+�R���A�)*;


total_loss�e�@

error_R�L?

learning_rate_1��8~�*[I       6%�	;m�R���A�)*;


total_loss���@

error_Rv<7?

learning_rate_1��8���I       6%�	밮R���A�)*;


total_loss%��@

error_R<}@?

learning_rate_1��8w�FKI       6%�	 ��R���A�)*;


total_loss��@

error_R��H?

learning_rate_1��8m��4I       6%�	;�R���A�)*;


total_lossC��@

error_R3�I?

learning_rate_1��8J1�+I       6%�	Մ�R���A�)*;


total_lossb�@

error_Rt�O?

learning_rate_1��8��I       6%�	jͯR���A�)*;


total_loss?�~@

error_R��J?

learning_rate_1��8�[pI       6%�	��R���A�)*;


total_loss�d�@

error_RNO?

learning_rate_1��8sJZ�I       6%�	R���A�)*;


total_loss��@

error_Rv'K?

learning_rate_1��85>��I       6%�	fаR���A�**;


total_loss�Ӧ@

error_RTT:?

learning_rate_1��8a=Q3I       6%�	��R���A�**;


total_lossx��@

error_R�lP?

learning_rate_1��8՛�]I       6%�	6d�R���A�**;


total_loss���@

error_R4G?

learning_rate_1��80��SI       6%�	���R���A�**;


total_lossR�@

error_R&5>?

learning_rate_1��8I�I       6%�	��R���A�**;


total_loss�4�@

error_R&L?

learning_rate_1��85��1I       6%�	�C�R���A�**;


total_loss��@

error_R׬F?

learning_rate_1��8���lI       6%�	g��R���A�**;


total_loss<��@

error_R�WZ?

learning_rate_1��8gqK�I       6%�	CҲR���A�**;


total_loss[P�@

error_R(BM?

learning_rate_1��85r�DI       6%�	��R���A�**;


total_loss���@

error_R׹S?

learning_rate_1��8�1��I       6%�	Bb�R���A�**;


total_lossc��@

error_R,o]?

learning_rate_1��8���6I       6%�	���R���A�**;


total_loss��@

error_Rv0E?

learning_rate_1��8���I       6%�	c�R���A�**;


total_lossn��@

error_RtW?

learning_rate_1��8З0mI       6%�	�<�R���A�**;


total_loss:\�@

error_RL0K?

learning_rate_1��8Iސ!I       6%�	���R���A�**;


total_lossq�@

error_R-�W?

learning_rate_1��8�m��I       6%�	5дR���A�**;


total_loss@[�@

error_RME?

learning_rate_1��8+��0I       6%�	7�R���A�**;


total_loss*�@

error_RF�H?

learning_rate_1��8��QI       6%�	`^�R���A�**;


total_loss&��@

error_R.�W?

learning_rate_1��8��I       6%�	7��R���A�**;


total_loss�J�@

error_RWC?

learning_rate_1��8kpuI       6%�	��R���A�**;


total_loss ��@

error_RZK?

learning_rate_1��85��I       6%�	;c�R���A�**;


total_loss� �@

error_R-�Y?

learning_rate_1��82�_I       6%�	]��R���A�**;


total_loss�!�@

error_R[M?

learning_rate_1��81E�>I       6%�	�R���A�**;


total_loss�D�@

error_R�\?

learning_rate_1��8��qsI       6%�	9Y�R���A�**;


total_loss�"�@

error_R^Y?

learning_rate_1��8���I       6%�	ꜷR���A�**;


total_loss��k@

error_R��P?

learning_rate_1��8�F�cI       6%�	��R���A�**;


total_lossW"A

error_R�HP?

learning_rate_1��8}�`I       6%�	t&�R���A�**;


total_loss�ٵ@

error_R1�I?

learning_rate_1��8(�dI       6%�	�m�R���A�**;


total_lossa4�@

error_Rd�K?

learning_rate_1��8j���I       6%�	f��R���A�**;


total_loss�Җ@

error_R��R?

learning_rate_1��8�d�uI       6%�	���R���A�**;


total_loss#�@

error_R�E?

learning_rate_1��8��i	I       6%�	�7�R���A�**;


total_loss��@

error_R3�L?

learning_rate_1��88 I       6%�	�{�R���A�**;


total_lossO��@

error_Ro5N?

learning_rate_1��8u�I       6%�	3˹R���A�**;


total_loss/� A

error_R��b?

learning_rate_1��8纇<I       6%�	i3�R���A�**;


total_loss�g@

error_RQ�S?

learning_rate_1��8mx��I       6%�	�y�R���A�**;


total_loss�&c@

error_R,T?

learning_rate_1��8���I       6%�	���R���A�**;


total_lossi?�@

error_R��L?

learning_rate_1��8�j�I       6%�	��R���A�**;


total_loss\��@

error_R,W?

learning_rate_1��8>���I       6%�	�I�R���A�**;


total_loss�F�@

error_ReQ_?

learning_rate_1��8=a�I       6%�	���R���A�**;


total_loss$��@

error_R�h?

learning_rate_1��8 ��I       6%�	DܻR���A�**;


total_loss,�@

error_R�%S?

learning_rate_1��8�2�I       6%�	�!�R���A�**;


total_loss���@

error_R�AV?

learning_rate_1��8��<I       6%�	�i�R���A�**;


total_loss��A

error_R�-??

learning_rate_1��8@��I       6%�	���R���A�**;


total_lossZL�@

error_R�.B?

learning_rate_1��8�oI       6%�	��R���A�**;


total_loss�<�@

error_R�O?

learning_rate_1��8�_��I       6%�	�.�R���A�**;


total_lossѢ�@

error_RM�W?

learning_rate_1��8Q�j�I       6%�	Mr�R���A�**;


total_losso
�@

error_RқS?

learning_rate_1��8�� I       6%�	෽R���A�**;


total_loss��@

error_R
�F?

learning_rate_1��8���I       6%�	���R���A�**;


total_lossc+�@

error_R��N?

learning_rate_1��8�Q!I       6%�	wC�R���A�**;


total_loss^\A

error_R/�e?

learning_rate_1��8��`I       6%�	���R���A�**;


total_loss��@

error_R�%K?

learning_rate_1��8���&I       6%�	_�R���A�**;


total_loss��@

error_R,O?

learning_rate_1��8A�NI       6%�	�V�R���A�**;


total_loss�P�@

error_Rߩ;?

learning_rate_1��8�䓩I       6%�	}��R���A�**;


total_loss�(�@

error_R<m?

learning_rate_1��8�j�I       6%�	3ݿR���A�**;


total_loss���@

error_Rd�@?

learning_rate_1��8K:I       6%�	5"�R���A�**;


total_loss ��@

error_RO�O?

learning_rate_1��8��7I       6%�	�f�R���A�**;


total_loss�J�@

error_R�(N?

learning_rate_1��8'�EI       6%�	]��R���A�**;


total_lossjA

error_R,'L?

learning_rate_1��8Jw�I       6%�	��R���A�**;


total_losss�A

error_RN�S?

learning_rate_1��8ƜA�I       6%�	7�R���A�**;


total_loss/K�@

error_R�Td?

learning_rate_1��8���?I       6%�	�}�R���A�**;


total_loss!�q@

error_R?�Q?

learning_rate_1��8롔�I       6%�	H��R���A�**;


total_loss(c�@

error_R@9S?

learning_rate_1��8��7�I       6%�	�R���A�**;


total_lossv9�@

error_R��R?

learning_rate_1��8v�ZI       6%�	#L�R���A�**;


total_lossE5A

error_RTwJ?

learning_rate_1��8��sI       6%�	|��R���A�**;


total_lossH�@

error_R�dZ?

learning_rate_1��82e��I       6%�	���R���A�**;


total_loss(�@

error_RtS?

learning_rate_1��8�6�I       6%�	��R���A�**;


total_lossE��@

error_RW�Z?

learning_rate_1��8�`m�I       6%�	c�R���A�**;


total_loss�?v@

error_R,a>?

learning_rate_1��8�ϰ)I       6%�	��R���A�**;


total_loss=�A

error_Rq�J?

learning_rate_1��8��+@I       6%�	���R���A�**;


total_loss;1�@

error_R�aS?

learning_rate_1��8n�L9I       6%�	j5�R���A�**;


total_loss��@

error_Rr_E?

learning_rate_1��8�9v{I       6%�	�w�R���A�**;


total_loss��@

error_RDOY?

learning_rate_1��80���I       6%�	���R���A�**;


total_lossd&�@

error_R�N?

learning_rate_1��8����I       6%�	s��R���A�**;


total_loss��@

error_R�XG?

learning_rate_1��8LDDZI       6%�	�D�R���A�**;


total_loss�R�@

error_R��L?

learning_rate_1��8]drI       6%�	��R���A�**;


total_loss���@

error_Rl�S?

learning_rate_1��8�o�4I       6%�	��R���A�**;


total_lossV{�@

error_R#�K?

learning_rate_1��8F2B�I       6%�	��R���A�**;


total_loss���@

error_R��D?

learning_rate_1��8���I       6%�	�W�R���A�**;


total_loss��@

error_R�O?

learning_rate_1��8t�jCI       6%�	g��R���A�**;


total_loss��@

error_RWz]?

learning_rate_1��8���(I       6%�	���R���A�**;


total_loss���@

error_Rf�S?

learning_rate_1��8c�&4I       6%�	��R���A�**;


total_loss��@

error_R�1C?

learning_rate_1��8�h�I       6%�	�d�R���A�**;


total_lossxt�@

error_RIJ?

learning_rate_1��8����I       6%�	��R���A�**;


total_loss���@

error_R��H?

learning_rate_1��8�QLuI       6%�	��R���A�**;


total_lossa��@

error_R��g?

learning_rate_1��8F-I       6%�	�<�R���A�**;


total_loss�8�@

error_R��D?

learning_rate_1��8.#�FI       6%�	��R���A�**;


total_lossW��@

error_R�sL?

learning_rate_1��8�P��I       6%�	��R���A�**;


total_loss8��@

error_R|�N?

learning_rate_1��8�
I       6%�	m�R���A�**;


total_loss$��@

error_R{�@?

learning_rate_1��8K���I       6%�	�[�R���A�**;


total_lossM�	A

error_R.XM?

learning_rate_1��8�N�I       6%�	���R���A�**;


total_loss�e�@

error_RO�J?

learning_rate_1��8���I       6%�	��R���A�**;


total_lossQ��@

error_R�R?

learning_rate_1��8�k�I       6%�	{V�R���A�**;


total_loss�:�@

error_R_�P?

learning_rate_1��8�)�ZI       6%�	���R���A�**;


total_loss@A�@

error_R�:S?

learning_rate_1��8
�{I       6%�	n��R���A�**;


total_loss���@

error_R�>?

learning_rate_1��86"�GI       6%�	9$�R���A�**;


total_loss���@

error_R�
V?

learning_rate_1��8��*�I       6%�	�k�R���A�**;


total_lossOD�@

error_R$�X?

learning_rate_1��8�Z+RI       6%�	���R���A�**;


total_loss� �@

error_R�5N?

learning_rate_1��8���I       6%�	��R���A�**;


total_lossE�@

error_R)�N?

learning_rate_1��8�;��I       6%�	�7�R���A�**;


total_loss[A

error_R�sT?

learning_rate_1��8��+I       6%�	��R���A�**;


total_loss���@

error_R��7?

learning_rate_1��8�,�I       6%�	���R���A�**;


total_lossx,�@

error_R=�P?

learning_rate_1��8#nrI       6%�	��R���A�**;


total_loss�ظ@

error_RXDW?

learning_rate_1��8=xX�I       6%�	�L�R���A�**;


total_lossF��@

error_R�Z?

learning_rate_1��8)83I       6%�	
��R���A�**;


total_loss��@

error_R

O?

learning_rate_1��8�=�4I       6%�	~��R���A�**;


total_lossZg�@

error_R]a?

learning_rate_1��8�E�I       6%�	+�R���A�**;


total_lossTN�@

error_R��U?

learning_rate_1��8dC�mI       6%�	^�R���A�**;


total_loss�v�@

error_RE?

learning_rate_1��8����I       6%�	Q��R���A�**;


total_loss�~�@

error_Rt�@?

learning_rate_1��8�+I|I       6%�	��R���A�**;


total_loss���@

error_R�V?

learning_rate_1��8T@�.I       6%�	�'�R���A�**;


total_loss-Q�@

error_RZ�T?

learning_rate_1��8܋�3I       6%�	cq�R���A�**;


total_loss��@

error_R�&D?

learning_rate_1��8����I       6%�	���R���A�**;


total_loss���@

error_R�v3?

learning_rate_1��8�}|I       6%�	u��R���A�**;


total_loss8S�@

error_RP?

learning_rate_1��8�F�I       6%�	�6�R���A�**;


total_loss�f�@

error_R��t?

learning_rate_1��8�V�^I       6%�	~{�R���A�**;


total_loss=��@

error_R
�W?

learning_rate_1��8�n��I       6%�	���R���A�**;


total_lossj�A

error_RJ�W?

learning_rate_1��8��CI       6%�	��R���A�**;


total_loss���@

error_R,�M?

learning_rate_1��8%�JI       6%�	�D�R���A�**;


total_lossH0�@

error_RHU?

learning_rate_1��8[�ЏI       6%�	P��R���A�**;


total_lossv��@

error_R)*b?

learning_rate_1��8sZoI       6%�	J��R���A�**;


total_lossȹ�@

error_R1)D?

learning_rate_1��8x<#+I       6%�	!�R���A�**;


total_lossod�@

error_RRKO?

learning_rate_1��8��I       6%�	O[�R���A�**;


total_loss���@

error_RnaS?

learning_rate_1��8�m�(I       6%�	���R���A�**;


total_loss�@

error_R΄Q?

learning_rate_1��8���I       6%�	���R���A�**;


total_loss��@

error_R�`I?

learning_rate_1��8�u;I       6%�	�:�R���A�**;


total_loss�P�@

error_R
_^?

learning_rate_1��8	h��I       6%�	m��R���A�**;


total_loss�C�@

error_R��??

learning_rate_1��8���I       6%�	r��R���A�**;


total_loss$��@

error_R�L?

learning_rate_1��8.�]I       6%�	�R���A�**;


total_loss��@

error_RE?

learning_rate_1��8+u�I       6%�	!U�R���A�**;


total_loss��@

error_R�qL?

learning_rate_1��8���I       6%�	+��R���A�+*;


total_lossO�@

error_R��[?

learning_rate_1��8�ıHI       6%�	���R���A�+*;


total_loss�w�@

error_R�N?

learning_rate_1��8�"s�I       6%�	X6�R���A�+*;


total_loss�b�@

error_RW�L?

learning_rate_1��8l'��I       6%�	���R���A�+*;


total_loss��@

error_R7�W?

learning_rate_1��8�56I       6%�	)��R���A�+*;


total_lossꯔ@

error_R�5?

learning_rate_1��8ɉ� I       6%�	��R���A�+*;


total_lossi*�@

error_R�.Q?

learning_rate_1��8T�I       6%�	At�R���A�+*;


total_loss�@

error_R�Y?

learning_rate_1��8_��I       6%�	��R���A�+*;


total_loss�@�@

error_R8�I?

learning_rate_1��8�̚�I       6%�	��R���A�+*;


total_loss\Ο@

error_R��9?

learning_rate_1��8̏I       6%�	Å�R���A�+*;


total_loss�`�@

error_RA?

learning_rate_1��8x��I       6%�	_��R���A�+*;


total_loss���@

error_R�d?

learning_rate_1��8����I       6%�	T�R���A�+*;


total_lossSA�@

error_RMR?

learning_rate_1��8�E�I       6%�	ц�R���A�+*;


total_loss�@

error_R��A?

learning_rate_1��8:�W�I       6%�	���R���A�+*;


total_loss�}{@

error_R4uQ?

learning_rate_1��8/�I       6%�	��R���A�+*;


total_loss��@

error_RkC?

learning_rate_1��8�zB�I       6%�	Wo�R���A�+*;


total_loss۴@

error_RE�K?

learning_rate_1��8��C�I       6%�	���R���A�+*;


total_lossf̵@

error_R-B?

learning_rate_1��8��CI       6%�	eO�R���A�+*;


total_loss���@

error_R�g?

learning_rate_1��8ѡM�I       6%�	f��R���A�+*;


total_loss��@

error_R��H?

learning_rate_1��8Q��nI       6%�	@�R���A�+*;


total_loss=e�@

error_Rs�G?

learning_rate_1��8�ڸ�I       6%�	8M�R���A�+*;


total_loss��@

error_Rj�R?

learning_rate_1��8�K'�I       6%�	���R���A�+*;


total_loss#v�@

error_R4�<?

learning_rate_1��82OyKI       6%�	���R���A�+*;


total_loss�լ@

error_R�+G?

learning_rate_1��8���I       6%�	 4�R���A�+*;


total_loss��@

error_R��G?

learning_rate_1��8�c�_I       6%�	x�R���A�+*;


total_loss���@

error_R��R?

learning_rate_1��8��I       6%�	��R���A�+*;


total_loss���@

error_RJJ?

learning_rate_1��8�#�I       6%�	��R���A�+*;


total_losso��@

error_R��N?

learning_rate_1��8<��I       6%�	�d�R���A�+*;


total_loss���@

error_RQ�G?

learning_rate_1��8{6I       6%�	I��R���A�+*;


total_loss��@

error_R��G?

learning_rate_1��8��ѡI       6%�	���R���A�+*;


total_loss̾�@

error_R��V?

learning_rate_1��8v��I       6%�	)R�R���A�+*;


total_lossN��@

error_Rv�<?

learning_rate_1��8ᖼI       6%�	���R���A�+*;


total_lossFy�@

error_R;d?

learning_rate_1��87`O=I       6%�	V��R���A�+*;


total_loss潔@

error_R��9?

learning_rate_1��8/��I       6%�	�8�R���A�+*;


total_loss�
A

error_R<T?

learning_rate_1��8��;I       6%�	��R���A�+*;


total_loss�s�@

error_R��T?

learning_rate_1��8�\tI       6%�	:��R���A�+*;


total_loss�=�@

error_R8!h?

learning_rate_1��8�wI       6%�	��R���A�+*;


total_loss��i@

error_R(�Q?

learning_rate_1��8�I       6%�	�p�R���A�+*;


total_loss��@

error_R؄N?

learning_rate_1��8�k	I       6%�	!��R���A�+*;


total_lossm/A

error_R�I?

learning_rate_1��8�6�I       6%�	P��R���A�+*;


total_loss���@

error_RS�E?

learning_rate_1��8zcZI       6%�	"?�R���A�+*;


total_lossԘ�@

error_R�S?

learning_rate_1��8TwVI       6%�	��R���A�+*;


total_loss���@

error_R��Q?

learning_rate_1��8-�[�I       6%�	���R���A�+*;


total_loss�U�@

error_R�Q?

learning_rate_1��8e��UI       6%�	��R���A�+*;


total_loss0��@

error_R�@L?

learning_rate_1��8�z�_I       6%�	W�R���A�+*;


total_loss��@

error_R��M?

learning_rate_1��8zv�I       6%�	���R���A�+*;


total_loss�@�@

error_R$$Q?

learning_rate_1��8��H�I       6%�	��R���A�+*;


total_loss;X�@

error_R��Z?

learning_rate_1��8;ɻI       6%�	�R���A�+*;


total_losso�@

error_RE�K?

learning_rate_1��8��~I       6%�	�^�R���A�+*;


total_loss� �@

error_R�2?

learning_rate_1��8}�>�I       6%�	֨�R���A�+*;


total_loss�֡@

error_R��[?

learning_rate_1��8��I       6%�	p��R���A�+*;


total_loss���@

error_R��D?

learning_rate_1��8䖰�I       6%�	�B�R���A�+*;


total_loss���@

error_R�}M?

learning_rate_1��8v]\�I       6%�	���R���A�+*;


total_loss>�@

error_R��B?

learning_rate_1��8lB�5I       6%�	���R���A�+*;


total_loss���@

error_R��B?

learning_rate_1��8�p,�I       6%�	��R���A�+*;


total_loss�\	A

error_R��A?

learning_rate_1��8�\I       6%�	�T�R���A�+*;


total_loss���@

error_R8�R?

learning_rate_1��8L]�"I       6%�	P��R���A�+*;


total_loss�g�@

error_R�I?

learning_rate_1��8	�I       6%�	��R���A�+*;


total_loss�6t@

error_R�kb?

learning_rate_1��8.Ͻ�I       6%�	"�R���A�+*;


total_lossZݪ@

error_R�[?

learning_rate_1��8"�I       6%�	!d�R���A�+*;


total_lossr�@

error_RE�=?

learning_rate_1��8�>h�I       6%�	4��R���A�+*;


total_loss״�@

error_R{Ta?

learning_rate_1��8b�z>I       6%�	2��R���A�+*;


total_loss6?�@

error_R�x\?

learning_rate_1��8켳I       6%�	�3�R���A�+*;


total_loss��@

error_R3FU?

learning_rate_1��8kn��I       6%�	�u�R���A�+*;


total_loss��^@

error_R�H?

learning_rate_1��8,hRI       6%�	���R���A�+*;


total_loss��@

error_R[�b?

learning_rate_1��8�GU�I       6%�	���R���A�+*;


total_loss��@

error_R�U?

learning_rate_1��8v�KgI       6%�	B�R���A�+*;


total_loss�A

error_R��V?

learning_rate_1��87B��I       6%�	΄�R���A�+*;


total_loss���@

error_Ri�C?

learning_rate_1��8���:I       6%�	#��R���A�+*;


total_loss�H�@

error_R�L?

learning_rate_1��8rv�I       6%�	��R���A�+*;


total_lossd+�@

error_RIfJ?

learning_rate_1��8�[��I       6%�	�N�R���A�+*;


total_lossp�@

error_R��K?

learning_rate_1��8 #�I       6%�	���R���A�+*;


total_loss�QA

error_R]R?

learning_rate_1��8��o�I       6%�	 ��R���A�+*;


total_loss*K�@

error_RݷF?

learning_rate_1��8���I       6%�	S8�R���A�+*;


total_loss�A

error_R��W?

learning_rate_1��8j�I       6%�	g~�R���A�+*;


total_loss$�@

error_R�D?

learning_rate_1��8�2aI       6%�	]��R���A�+*;


total_loss��@

error_R`�V?

learning_rate_1��8Mu�1I       6%�	N�R���A�+*;


total_loss،�@

error_R6|U?

learning_rate_1��8x6I       6%�	�H�R���A�+*;


total_lossÂ�@

error_R��a?

learning_rate_1��8���"I       6%�	���R���A�+*;


total_lossT��@

error_R��I?

learning_rate_1��8��(I       6%�	&��R���A�+*;


total_loss?�@

error_R��^?

learning_rate_1��8Uo�1I       6%�	��R���A�+*;


total_loss��@

error_RRIU?

learning_rate_1��8�7tI       6%�	�R�R���A�+*;


total_loss:�@

error_Ri:?

learning_rate_1��8-ϤI       6%�	Е�R���A�+*;


total_loss4'�@

error_R]L?

learning_rate_1��8H��I       6%�	l��R���A�+*;


total_loss ��@

error_R@CG?

learning_rate_1��8:�*[I       6%�	�#�R���A�+*;


total_loss�߬@

error_R  P?

learning_rate_1��8�Ɖ�I       6%�	�e�R���A�+*;


total_loss�@

error_Rz�X?

learning_rate_1��8�tj�I       6%�	b��R���A�+*;


total_lossH��@

error_R�}A?

learning_rate_1��8�{�I       6%�	L��R���A�+*;


total_lossm��@

error_R}�_?

learning_rate_1��8��LI       6%�	a-�R���A�+*;


total_loss��@

error_R��B?

learning_rate_1��8yu��I       6%�	�n�R���A�+*;


total_loss��@

error_R3�N?

learning_rate_1��8�:iI       6%�	ӯ�R���A�+*;


total_lossd��@

error_Rd5`?

learning_rate_1��8K�*I       6%�	���R���A�+*;


total_lossg��@

error_R?�X?

learning_rate_1��8R__I       6%�	A3�R���A�+*;


total_loss�8�@

error_Ra8?

learning_rate_1��8�!��I       6%�	�r�R���A�+*;


total_loss8��@

error_R�oM?

learning_rate_1��8�}��I       6%�	
��R���A�+*;


total_lossܰ�@

error_RH�e?

learning_rate_1��8���I       6%�	e��R���A�+*;


total_lossn��@

error_R��V?

learning_rate_1��8��MI       6%�	�3�R���A�+*;


total_loss���@

error_R�dJ?

learning_rate_1��8}"�oI       6%�	fu�R���A�+*;


total_lossd��@

error_R�M?

learning_rate_1��8�2��I       6%�	��R���A�+*;


total_loss\X�@

error_R_P?

learning_rate_1��8;�2I       6%�	���R���A�+*;


total_loss��A

error_RV0c?

learning_rate_1��8�[�zI       6%�	q;�R���A�+*;


total_losseM|@

error_R\cZ?

learning_rate_1��8�wL�I       6%�	�z�R���A�+*;


total_loss�Q�@

error_RŌ??

learning_rate_1��8��I       6%�	���R���A�+*;


total_losse��@

error_R��>?

learning_rate_1��8�
LI       6%�	3 �R���A�+*;


total_loss��@

error_R�X?

learning_rate_1��8�r��I       6%�	.F�R���A�+*;


total_lossLv�@

error_RjM?

learning_rate_1��8N��I       6%�	)��R���A�+*;


total_loss-PA

error_R��S?

learning_rate_1��8~��KI       6%�	��R���A�+*;


total_loss/��@

error_RNF?

learning_rate_1��8�M�I       6%�	V�R���A�+*;


total_loss=�@

error_R��L?

learning_rate_1��8+�KI       6%�	&]�R���A�+*;


total_loss4��@

error_RS�Y?

learning_rate_1��8��vI       6%�	G��R���A�+*;


total_loss-c�@

error_R2�N?

learning_rate_1��8� @RI       6%�	���R���A�+*;


total_loss��A

error_R�X?

learning_rate_1��8UT|-I       6%�	�%�R���A�+*;


total_lossm��@

error_Rs:c?

learning_rate_1��8�`I       6%�	g�R���A�+*;


total_loss�5�@

error_RE�C?

learning_rate_1��8�w�kI       6%�	A��R���A�+*;


total_loss���@

error_R�X?

learning_rate_1��8�2:dI       6%�	d��R���A�+*;


total_loss�/�@

error_R�M=?

learning_rate_1��8-��QI       6%�	-�R���A�+*;


total_loss,�^@

error_R�)N?

learning_rate_1��8�3�I       6%�	in�R���A�+*;


total_lossë�@

error_R�2W?

learning_rate_1��8���I       6%�	k��R���A�+*;


total_loss���@

error_RWS?

learning_rate_1��8B*h�I       6%�	���R���A�+*;


total_loss���@

error_RW<C?

learning_rate_1��8�n�WI       6%�	I.�R���A�+*;


total_loss{�@

error_R V?

learning_rate_1��8��I       6%�	Bm�R���A�+*;


total_loss\)�@

error_R�W?

learning_rate_1��8l/>I       6%�	��R���A�+*;


total_loss��A

error_RT�N?

learning_rate_1��8r��&I       6%�	j��R���A�+*;


total_loss���@

error_REnE?

learning_rate_1��8y�p�I       6%�		1�R���A�+*;


total_loss�i�@

error_R�BQ?

learning_rate_1��8t3=�I       6%�	:y�R���A�+*;


total_loss��@

error_R�I?

learning_rate_1��8b�؆I       6%�	���R���A�+*;


total_loss4Ӵ@

error_R$Q?

learning_rate_1��8RGI       6%�	���R���A�+*;


total_loss�{�@

error_R�,<?

learning_rate_1��8�?	I       6%�	8<�R���A�+*;


total_loss���@

error_R�^Y?

learning_rate_1��8����I       6%�	�{�R���A�,*;


total_loss���@

error_R��I?

learning_rate_1��8/i�zI       6%�	>��R���A�,*;


total_lossθ�@

error_RWeb?

learning_rate_1��8��I       6%�	$��R���A�,*;


total_loss��@

error_R��F?

learning_rate_1��8i��I       6%�	�<�R���A�,*;


total_loss�R�@

error_R��I?

learning_rate_1��8�RNI       6%�	�~�R���A�,*;


total_lossۃ�@

error_R�S?

learning_rate_1��8�=q�I       6%�	q��R���A�,*;


total_loss�y�@

error_R�%L?

learning_rate_1��8!�bI       6%�	�+�R���A�,*;


total_loss���@

error_R\�W?

learning_rate_1��8����I       6%�	�o�R���A�,*;


total_loss���@

error_R��C?

learning_rate_1��8�CRI       6%�	��R���A�,*;


total_lossL��@

error_R�xS?

learning_rate_1��8�s.I       6%�	���R���A�,*;


total_loss��@

error_Ri�J?

learning_rate_1��86)��I       6%�	�B�R���A�,*;


total_lossi��@

error_R΂O?

learning_rate_1��8DscAI       6%�	z��R���A�,*;


total_loss�g�@

error_R��E?

learning_rate_1��8x_	�I       6%�	���R���A�,*;


total_lossl�a@

error_R��Y?

learning_rate_1��8m�pbI       6%�	��R���A�,*;


total_lossj
�@

error_R�7?

learning_rate_1��8�GI       6%�	zN�R���A�,*;


total_loss���@

error_R{�J?

learning_rate_1��8K}�I       6%�	���R���A�,*;


total_loss��@

error_R�!U?

learning_rate_1��8Q��I       6%�	���R���A�,*;


total_loss��JA

error_R�AR?

learning_rate_1��8�hnI       6%�	V�R���A�,*;


total_loss�2�@

error_Rq�Z?

learning_rate_1��8��OI       6%�	�R�R���A�,*;


total_loss�Ƽ@

error_R��@?

learning_rate_1��8&tj�I       6%�	ٕ�R���A�,*;


total_loss���@

error_R�':?

learning_rate_1��8��I�I       6%�	y��R���A�,*;


total_loss@8�@

error_R��Z?

learning_rate_1��8�i�I       6%�	��R���A�,*;


total_loss���@

error_Rd=.?

learning_rate_1��8h8G�I       6%�	{Y�R���A�,*;


total_loss.FA

error_R�=Q?

learning_rate_1��8Y{��I       6%�	���R���A�,*;


total_loss�v�@

error_R�$S?

learning_rate_1��8.k{I       6%�	D��R���A�,*;


total_loss���@

error_R�M?

learning_rate_1��8�!�I       6%�	�%�R���A�,*;


total_loss�B�@

error_R�P?

learning_rate_1��8����I       6%�	Ie�R���A�,*;


total_lossW��@

error_R�??

learning_rate_1��8��pI       6%�	n��R���A�,*;


total_loss�=�@

error_R1�G?

learning_rate_1��8��/I       6%�	���R���A�,*;


total_lossS��@

error_RZ�L?

learning_rate_1��8&���I       6%�	�' S���A�,*;


total_loss���@

error_R�Q]?

learning_rate_1��8�)$I       6%�	Mj S���A�,*;


total_loss�ư@

error_R&MV?

learning_rate_1��8ba�rI       6%�	b� S���A�,*;


total_loss8c�@

error_R�J?

learning_rate_1��8 YwI       6%�	b� S���A�,*;


total_loss� �@

error_R�;L?

learning_rate_1��8DHG�I       6%�	�*S���A�,*;


total_lossT�@

error_R�=?

learning_rate_1��8��bI       6%�	�lS���A�,*;


total_losse�@

error_RM�T?

learning_rate_1��8e�I       6%�	{�S���A�,*;


total_loss A

error_R��a?

learning_rate_1��8��{I       6%�	P�S���A�,*;


total_loss�T�@

error_R�8?

learning_rate_1��8���I       6%�	�2S���A�,*;


total_loss�^�@

error_R��G?

learning_rate_1��8�d��I       6%�	4sS���A�,*;


total_lossʗ�@

error_R��I?

learning_rate_1��8�gn�I       6%�	ճS���A�,*;


total_lossԢ@

error_Ré9?

learning_rate_1��8@L�I       6%�	��S���A�,*;


total_loss?�@

error_R��M?

learning_rate_1��8�a��I       6%�	SS���A�,*;


total_lossf�@

error_R&-W?

learning_rate_1��8)q�6I       6%�	�S���A�,*;


total_loss<�A

error_R�V?

learning_rate_1��8΅y�I       6%�	P�S���A�,*;


total_loss]��@

error_R	�`?

learning_rate_1��8�:F�I       6%�	*S���A�,*;


total_lossM�@

error_R��D?

learning_rate_1��8���I       6%�	��S���A�,*;


total_loss�@

error_R�=?

learning_rate_1��8�{��I       6%�	��S���A�,*;


total_lossݢ�@

error_RR�J?

learning_rate_1��8`��I       6%�	�S���A�,*;


total_loss��@

error_R3�M?

learning_rate_1��8K�?sI       6%�	GS���A�,*;


total_loss7��@

error_R�Z?

learning_rate_1��8&�3�I       6%�	ņS���A�,*;


total_loss�˶@

error_R�R?

learning_rate_1��8�f��I       6%�	�S���A�,*;


total_loss���@

error_R8zV?

learning_rate_1��8/	�I       6%�	�S���A�,*;


total_lossLEg@

error_R3�[?

learning_rate_1��8�j�kI       6%�	;NS���A�,*;


total_loss�R�@

error_R�E?

learning_rate_1��87ܟRI       6%�	��S���A�,*;


total_loss�=�@

error_R�M?

learning_rate_1��8�S4�I       6%�	��S���A�,*;


total_loss��@

error_R�vF?

learning_rate_1��8L;�I       6%�	5S���A�,*;


total_loss߀�@

error_R$:L?

learning_rate_1��8rH̞I       6%�	%aS���A�,*;


total_lossS��@

error_R4�D?

learning_rate_1��8L�rI       6%�	��S���A�,*;


total_loss�:
A

error_R_Q?

learning_rate_1��8��DgI       6%�	6�S���A�,*;


total_loss���@

error_R�<D?

learning_rate_1��8v2�nI       6%�	�%S���A�,*;


total_lossh�@

error_Re�W?

learning_rate_1��8ķ�NI       6%�	8jS���A�,*;


total_loss��A

error_R�T?

learning_rate_1��8�z�II       6%�	_�S���A�,*;


total_loss�8�@

error_R��A?

learning_rate_1��8Z یI       6%�	Q�S���A�,*;


total_losss��@

error_R�1M?

learning_rate_1��8��bI       6%�	�2	S���A�,*;


total_loss���@

error_R�\?

learning_rate_1��8��x~I       6%�	Hu	S���A�,*;


total_loss»@

error_R RI?

learning_rate_1��8Ö]�I       6%�	{�	S���A�,*;


total_lossfG�@

error_RʃP?

learning_rate_1��8(�kI       6%�	�
S���A�,*;


total_lossiT�@

error_RA?

learning_rate_1��8OրI       6%�	�`
S���A�,*;


total_losse�@

error_RӈQ?

learning_rate_1��8%)��I       6%�	5�
S���A�,*;


total_lossJ��@

error_R�`?

learning_rate_1��8"�grI       6%�	��
S���A�,*;


total_loss ��@

error_RDD?

learning_rate_1��8E\FI       6%�	(S���A�,*;


total_loss��@

error_R�\D?

learning_rate_1��8ni�I       6%�	gjS���A�,*;


total_loss?[�@

error_R3�Y?

learning_rate_1��8K:&I       6%�	x�S���A�,*;


total_loss&�A

error_Re i?

learning_rate_1��8��٬I       6%�	�S���A�,*;


total_loss4��@

error_R��F?

learning_rate_1��869I       6%�	,S���A�,*;


total_lossDl�@

error_R|S?

learning_rate_1��8I�)�I       6%�	nS���A�,*;


total_lossJ��@

error_R(A?

learning_rate_1��8&u�I       6%�	�S���A�,*;


total_lossL�@

error_R�N?

learning_rate_1��8@GbI       6%�	G�S���A�,*;


total_loss�˟@

error_R�AN?

learning_rate_1��8)��I       6%�	2S���A�,*;


total_lossS��@

error_R)�T?

learning_rate_1��8!���I       6%�	�sS���A�,*;


total_lossζ�@

error_R�Y?

learning_rate_1��8r�I       6%�	e�S���A�,*;


total_loss��@

error_R.�D?

learning_rate_1��8ӆ^�I       6%�	��S���A�,*;


total_loss0�@

error_R�gV?

learning_rate_1��8�1�I       6%�	CGS���A�,*;


total_loss/�@

error_R>W?

learning_rate_1��8���I       6%�	��S���A�,*;


total_loss!�@

error_R;�K?

learning_rate_1��8o��I       6%�	��S���A�,*;


total_loss-9A

error_R�X?

learning_rate_1��8�lhnI       6%�	�
S���A�,*;


total_loss��A

error_R�WI?

learning_rate_1��8+�cI       6%�	�IS���A�,*;


total_lossFG�@

error_R�|H?

learning_rate_1��8�ݎiI       6%�	D�S���A�,*;


total_loss�$�@

error_RӺ\?

learning_rate_1��8R(��I       6%�	?�S���A�,*;


total_loss���@

error_RWAE?

learning_rate_1��8}�@I       6%�	�S���A�,*;


total_lossx��@

error_R��G?

learning_rate_1��8�"��I       6%�	�PS���A�,*;


total_loss#\A

error_R�wN?

learning_rate_1��8�`�I       6%�	U�S���A�,*;


total_losse��@

error_RC�J?

learning_rate_1��8���I       6%�	��S���A�,*;


total_loss�i�@

error_R�R`?

learning_rate_1��8Ҳ��I       6%�	�S���A�,*;


total_lossl�u@

error_R�5R?

learning_rate_1��8r�cI       6%�	�YS���A�,*;


total_loss�vA

error_R�MR?

learning_rate_1��8e/e�I       6%�	��S���A�,*;


total_loss�/A

error_R��[?

learning_rate_1��8�{�EI       6%�	F�S���A�,*;


total_losss�	A

error_Rh7S?

learning_rate_1��8���I       6%�	%S���A�,*;


total_loss���@

error_R!K?

learning_rate_1��8|\I       6%�	�gS���A�,*;


total_loss��@

error_R��\?

learning_rate_1��8I��I       6%�	éS���A�,*;


total_loss��@

error_R�tM?

learning_rate_1��8J��I       6%�	��S���A�,*;


total_loss8 �@

error_R@�]?

learning_rate_1��8ѷ�I       6%�	�2S���A�,*;


total_lossq��@

error_R�/Y?

learning_rate_1��8���8I       6%�	dtS���A�,*;


total_loss�I�@

error_R��M?

learning_rate_1��8�m��I       6%�	��S���A�,*;


total_loss���@

error_RC�P?

learning_rate_1��8�	C�I       6%�	��S���A�,*;


total_loss(͖@

error_R��Z?

learning_rate_1��8`s�%I       6%�	�>S���A�,*;


total_loss���@

error_RF-e?

learning_rate_1��8��nI       6%�	��S���A�,*;


total_loss@

error_Ri�N?

learning_rate_1��8�<
I       6%�	}�S���A�,*;


total_loss��@

error_RN?

learning_rate_1��8�r�I       6%�	K
S���A�,*;


total_lossNh�@

error_R��>?

learning_rate_1��8�v��I       6%�	2QS���A�,*;


total_loss�8�@

error_R3�<?

learning_rate_1��8�e"&I       6%�	͑S���A�,*;


total_loss;��@

error_R�Z?

learning_rate_1��8�cuCI       6%�	��S���A�,*;


total_loss�!�@

error_R�Z?

learning_rate_1��8��I       6%�	�S���A�,*;


total_lossj��@

error_R�iY?

learning_rate_1��8���I       6%�	-MS���A�,*;


total_loss��A

error_R��O?

learning_rate_1��8Zf�{I       6%�	ƒS���A�,*;


total_loss#W�@

error_RlUI?

learning_rate_1��8@=�?I       6%�	a�S���A�,*;


total_loss�	�@

error_R�F?

learning_rate_1��8ib�I       6%�	oS���A�,*;


total_loss-��@

error_R��]?

learning_rate_1��8*)��I       6%�	W_S���A�,*;


total_loss|�A

error_R�W?

learning_rate_1��8�O�NI       6%�	:�S���A�,*;


total_loss�O�@

error_R.^M?

learning_rate_1��84�I       6%�	��S���A�,*;


total_loss��@

error_R�N?

learning_rate_1��8�=��I       6%�	�%S���A�,*;


total_loss�=�@

error_R6�Q?

learning_rate_1��8+h I       6%�	�eS���A�,*;


total_loss�ƽ@

error_RϮY?

learning_rate_1��8�E�nI       6%�	1�S���A�,*;


total_loss�Ť@

error_R;�A?

learning_rate_1��8W���I       6%�	�S���A�,*;


total_loss���@

error_R�!5?

learning_rate_1��8���sI       6%�	-S���A�,*;


total_loss:��@

error_R`�M?

learning_rate_1��8��U�I       6%�	�oS���A�,*;


total_loss���@

error_Rz�W?

learning_rate_1��8 ҝmI       6%�	ʯS���A�,*;


total_loss��@

error_R�I?

learning_rate_1��8���I       6%�	�S���A�,*;


total_loss[S�@

error_R�M?

learning_rate_1��88��)I       6%�	�aS���A�-*;


total_loss-�A

error_R�DX?

learning_rate_1��8��F�I       6%�	L�S���A�-*;


total_loss���@

error_R�-G?

learning_rate_1��8���I       6%�	��S���A�-*;


total_loss��@

error_R��P?

learning_rate_1��82��I       6%�	�0S���A�-*;


total_loss}�@

error_RZ�O?

learning_rate_1��89~lI       6%�	LtS���A�-*;


total_lossU�"A

error_Rn�X?

learning_rate_1��86p<I       6%�	�S���A�-*;


total_loss��@

error_RlpX?

learning_rate_1��8^���I       6%�	��S���A�-*;


total_lossI`�@

error_ReI?

learning_rate_1��8��I       6%�	�9S���A�-*;


total_loss�3�@

error_R��U?

learning_rate_1��8��v_I       6%�	�yS���A�-*;


total_loss�@A

error_R��S?

learning_rate_1��8��'VI       6%�	B�S���A�-*;


total_loss_�@

error_R,.X?

learning_rate_1��8��I       6%�	��S���A�-*;


total_lossqu�@

error_R�]]?

learning_rate_1��8e-�I       6%�	Z8S���A�-*;


total_loss���@

error_R�fB?

learning_rate_1��8U�I       6%�	txS���A�-*;


total_loss�]�@

error_R�+R?

learning_rate_1��8�3~I       6%�	Q�S���A�-*;


total_loss���@

error_R�JT?

learning_rate_1��8���jI       6%�	w�S���A�-*;


total_losssB�@

error_R��O?

learning_rate_1��8�cz&I       6%�	:S���A�-*;


total_lossOO�@

error_R&�Q?

learning_rate_1��83��1I       6%�	�zS���A�-*;


total_lossa�Y@

error_Rd�>?

learning_rate_1��8�+,�I       6%�	u�S���A�-*;


total_lossD��@

error_R6�O?

learning_rate_1��8
Q-I       6%�	VS���A�-*;


total_loss[�A

error_R�h?

learning_rate_1��8ǃцI       6%�	#CS���A�-*;


total_loss+��@

error_R��>?

learning_rate_1��8�`3I       6%�	��S���A�-*;


total_loss���@

error_R��L?

learning_rate_1��8��ۍI       6%�	M�S���A�-*;


total_loss���@

error_R��Z?

learning_rate_1��80֎RI       6%�	�  S���A�-*;


total_loss}�A

error_R�C?

learning_rate_1��8l@I       6%�	�A S���A�-*;


total_loss��@

error_R�U?

learning_rate_1��8�Z�vI       6%�	0� S���A�-*;


total_loss�L�@

error_R7rb?

learning_rate_1��8���I       6%�	�� S���A�-*;


total_loss}?�@

error_Rz�N?

learning_rate_1��8�ЁI       6%�	M!S���A�-*;


total_loss�E�@

error_RԱF?

learning_rate_1��8���+I       6%�	�C!S���A�-*;


total_loss���@

error_RʹU?

learning_rate_1��8ڜSI       6%�	q�!S���A�-*;


total_lossz��@

error_R�R?

learning_rate_1��8_K..I       6%�	k�!S���A�-*;


total_loss���@

error_R$Z?

learning_rate_1��8��G�I       6%�	�"S���A�-*;


total_lossm�;A

error_R��]?

learning_rate_1��8�*^�I       6%�	�H"S���A�-*;


total_lossM2�@

error_R �`?

learning_rate_1��8
ChI       6%�	Q�"S���A�-*;


total_loss��@

error_R��P?

learning_rate_1��8��1I       6%�	��"S���A�-*;


total_losseݹ@

error_R�[G?

learning_rate_1��8��� I       6%�	�	#S���A�-*;


total_loss7�@

error_Rd�A?

learning_rate_1��8��%�I       6%�	�I#S���A�-*;


total_loss��@

error_R&IO?

learning_rate_1��8�,GI       6%�	Ҋ#S���A�-*;


total_loss�ʤ@

error_R��@?

learning_rate_1��8��7xI       6%�	��#S���A�-*;


total_lossbĔ@

error_Rϲ^?

learning_rate_1��8��rI       6%�	�	$S���A�-*;


total_loss�k�@

error_RQ�3?

learning_rate_1��8��ȰI       6%�	QL$S���A�-*;


total_lossp��@

error_R��L?

learning_rate_1��8�~�GI       6%�	!�$S���A�-*;


total_loss�#�@

error_R�rN?

learning_rate_1��8�^zI       6%�	��$S���A�-*;


total_loss�o�@

error_R�\R?

learning_rate_1��8o|z�I       6%�	�
%S���A�-*;


total_loss1Q�@

error_R&�;?

learning_rate_1��8C�$I       6%�	?H%S���A�-*;


total_loss�, A

error_Ro'K?

learning_rate_1��8�d��I       6%�	�%S���A�-*;


total_lossM�@

error_RV�R?

learning_rate_1��8{��I       6%�	��%S���A�-*;


total_loss��@

error_R�'V?

learning_rate_1��8w�I       6%�	e&S���A�-*;


total_lossï�@

error_R^?

learning_rate_1��8)���I       6%�	0N&S���A�-*;


total_loss|��@

error_R��Q?

learning_rate_1��8�e��I       6%�	~�&S���A�-*;


total_loss��A

error_R
3U?

learning_rate_1��8�R{�I       6%�	��&S���A�-*;


total_loss�!A

error_R�W?

learning_rate_1��8?�I       6%�	q'S���A�-*;


total_loss_s�@

error_R4jO?

learning_rate_1��85A\	I       6%�	V'S���A�-*;


total_lossM�@

error_R��S?

learning_rate_1��8�2I       6%�	��'S���A�-*;


total_lossl��@

error_Ra�\?

learning_rate_1��8�_h�I       6%�	��'S���A�-*;


total_loss��@

error_R.Q?

learning_rate_1��8	 :qI       6%�	�>(S���A�-*;


total_loss�N�@

error_R|�Q?

learning_rate_1��8�i��I       6%�	��(S���A�-*;


total_loss�
A

error_R��R?

learning_rate_1��8��#I       6%�	;�(S���A�-*;


total_lossL�@

error_R�F?

learning_rate_1��8�O?�I       6%�	f$)S���A�-*;


total_lossE��@

error_R�tZ?

learning_rate_1��8�0��I       6%�	�h)S���A�-*;


total_loss�n�@

error_RO�\?

learning_rate_1��8��qI       6%�	7�)S���A�-*;


total_loss:*�@

error_R�d?

learning_rate_1��8�]��I       6%�	��)S���A�-*;


total_loss ��@

error_R��J?

learning_rate_1��8��#VI       6%�	N*S���A�-*;


total_loss&��@

error_R�6W?

learning_rate_1��8G(WI       6%�	ܑ*S���A�-*;


total_loss)�@

error_R�FI?

learning_rate_1��8��I       6%�	��*S���A�-*;


total_loss���@

error_Rn�]?

learning_rate_1��8i��I       6%�	+S���A�-*;


total_loss��@

error_RxgO?

learning_rate_1��8ԄdHI       6%�	�X+S���A�-*;


total_loss��@

error_R�O?

learning_rate_1��8�G��I       6%�	�+S���A�-*;


total_lossAtA

error_RzD?

learning_rate_1��8/\{+I       6%�	��+S���A�-*;


total_lossv��@

error_RnQ?

learning_rate_1��8Z#��I       6%�	� ,S���A�-*;


total_lossg;A

error_R��B?

learning_rate_1��8��gI       6%�	�b,S���A�-*;


total_loss���@

error_R�<?

learning_rate_1��8�f��I       6%�	ޣ,S���A�-*;


total_loss�@

error_R�T?

learning_rate_1��8JKpiI       6%�	7�,S���A�-*;


total_loss�Y�@

error_R�[?

learning_rate_1��8�bI       6%�	�%-S���A�-*;


total_loss��@

error_R�{Q?

learning_rate_1��8�f�uI       6%�	e-S���A�-*;


total_lossZ�@

error_R��M?

learning_rate_1��8��(EI       6%�	�-S���A�-*;


total_loss�4�@

error_R8 B?

learning_rate_1��8J���I       6%�	t�-S���A�-*;


total_loss�4�@

error_RX�M?

learning_rate_1��8ו	eI       6%�	�2.S���A�-*;


total_loss3��@

error_R��A?

learning_rate_1��8�"r�I       6%�	w.S���A�-*;


total_loss�KUA

error_R$H?

learning_rate_1��8�a�I       6%�	ٹ.S���A�-*;


total_loss�%A

error_R�G?

learning_rate_1��8�YJsI       6%�	G�.S���A�-*;


total_loss�J�@

error_R?�H?

learning_rate_1��8�&! I       6%�	>/S���A�-*;


total_loss�@�@

error_R
�Y?

learning_rate_1��8K�HI       6%�	8�/S���A�-*;


total_loss�d@

error_R�D?

learning_rate_1��8�a]�I       6%�	^�/S���A�-*;


total_loss-��@

error_R��T?

learning_rate_1��8E4��I       6%�	T0S���A�-*;


total_loss���@

error_R��W?

learning_rate_1��8��@�I       6%�	�O0S���A�-*;


total_loss��@

error_Rq>K?

learning_rate_1��8� hHI       6%�	o�0S���A�-*;


total_loss��&A

error_R�CX?

learning_rate_1��8�8I       6%�	��0S���A�-*;


total_loss�ܳ@

error_RtNJ?

learning_rate_1��8G�6�I       6%�	1S���A�-*;


total_loss2��@

error_R4�G?

learning_rate_1��8��I       6%�	6O1S���A�-*;


total_lossp�@

error_R��I?

learning_rate_1��8�9�BI       6%�	2�1S���A�-*;


total_lossG��@

error_R?K?

learning_rate_1��8C1��I       6%�	��1S���A�-*;


total_lossL��@

error_R
�U?

learning_rate_1��8��I       6%�	F2S���A�-*;


total_lossq�A

error_R�PG?

learning_rate_1��8���I       6%�	�P2S���A�-*;


total_loss���@

error_R)�C?

learning_rate_1��8:� �I       6%�	��2S���A�-*;


total_loss�@

error_R��C?

learning_rate_1��8�H.�I       6%�	�2S���A�-*;


total_loss���@

error_R3W?

learning_rate_1��8ʞ�I       6%�	M3S���A�-*;


total_lossF�@

error_R\gF?

learning_rate_1��8��I       6%�	�M3S���A�-*;


total_loss1��@

error_R�Q?

learning_rate_1��8hmW�I       6%�	�3S���A�-*;


total_lossvZ�@

error_RsO?

learning_rate_1��88q�:I       6%�	,�3S���A�-*;


total_loss\�@

error_Rf�K?

learning_rate_1��8d�I       6%�	�4S���A�-*;


total_lossBgA

error_R�D?

learning_rate_1��8�7�I       6%�	�Y4S���A�-*;


total_loss�w�@

error_R��=?

learning_rate_1��8�ZI       6%�	��4S���A�-*;


total_loss��@

error_R�WW?

learning_rate_1��8�=�I       6%�	��4S���A�-*;


total_lossks�@

error_RC\V?

learning_rate_1��8���I       6%�	�#5S���A�-*;


total_lossV��@

error_RT�A?

learning_rate_1��8?!�I       6%�	�m5S���A�-*;


total_loss���@

error_R��E?

learning_rate_1��8r��I       6%�	)�5S���A�-*;


total_losso{�@

error_R��b?

learning_rate_1��8���I       6%�	;�5S���A�-*;


total_loss�P�@

error_RS9T?

learning_rate_1��8pA�I       6%�	�56S���A�-*;


total_loss%�@

error_R��e?

learning_rate_1��8��M�I       6%�	u6S���A�-*;


total_loss��@

error_RcSQ?

learning_rate_1��8ha�I       6%�	�6S���A�-*;


total_lossݼ@

error_R1�>?

learning_rate_1��80���I       6%�	�6S���A�-*;


total_lossUP	A

error_R��K?

learning_rate_1��8g-�I       6%�	97S���A�-*;


total_loss�1�@

error_R��C?

learning_rate_1��8�ˑI       6%�	�z7S���A�-*;


total_lossI�@

error_Rj�B?

learning_rate_1��8��6�I       6%�	�7S���A�-*;


total_losshп@

error_R״/?

learning_rate_1��8�-�I       6%�	��7S���A�-*;


total_loss���@

error_Rq�P?

learning_rate_1��8�0�I       6%�	�88S���A�-*;


total_loss�8�@

error_R&�W?

learning_rate_1��8��^tI       6%�	dx8S���A�-*;


total_loss1]�@

error_R��_?

learning_rate_1��8F�N�I       6%�	�8S���A�-*;


total_loss���@

error_R��h?

learning_rate_1��8��I       6%�	�8S���A�-*;


total_lossԺ�@

error_RE?

learning_rate_1��8� ��I       6%�	|:9S���A�-*;


total_loss�X9A

error_Re�J?

learning_rate_1��8'�zWI       6%�	R{9S���A�-*;


total_loss���@

error_R�G?

learning_rate_1��8�w/I       6%�	ܼ9S���A�-*;


total_loss�G�@

error_R?Y?

learning_rate_1��8��zI       6%�	:S���A�-*;


total_loss��@

error_R&4i?

learning_rate_1��8��RqI       6%�	t`:S���A�-*;


total_loss���@

error_R�H?

learning_rate_1��8N+H�I       6%�	ا:S���A�-*;


total_loss?`�@

error_R-AW?

learning_rate_1��8���FI       6%�	c�:S���A�-*;


total_loss��@

error_R��Q?

learning_rate_1��8���I       6%�	
);S���A�-*;


total_loss$�@

error_RͭP?

learning_rate_1��8C���I       6%�	�j;S���A�-*;


total_lossp�@

error_Rc�`?

learning_rate_1��8�5��I       6%�	��;S���A�.*;


total_loss@

error_R�P?

learning_rate_1��8�y�I       6%�	��;S���A�.*;


total_loss�0A

error_R�PF?

learning_rate_1��8v��WI       6%�	�2<S���A�.*;


total_loss��@

error_R��U?

learning_rate_1��8ձ�I       6%�	�u<S���A�.*;


total_loss7�i@

error_R.\?

learning_rate_1��8�A�'I       6%�	��<S���A�.*;


total_loss�2A

error_R�T?

learning_rate_1��8T�JI       6%�	��<S���A�.*;


total_lossm~�@

error_R�VH?

learning_rate_1��8�y��I       6%�	*C=S���A�.*;


total_loss���@

error_R_�P?

learning_rate_1��8-��I       6%�	f�=S���A�.*;


total_loss��@

error_R�K?

learning_rate_1��8��b�I       6%�	�=S���A�.*;


total_loss=sA

error_R��Y?

learning_rate_1��8|�TI       6%�	>S���A�.*;


total_lossW�@

error_R�O?

learning_rate_1��8`%FI       6%�	bM>S���A�.*;


total_losst�@

error_R�LC?

learning_rate_1��8�W��I       6%�	�>S���A�.*;


total_loss�%�@

error_R��I?

learning_rate_1��8���vI       6%�	i�>S���A�.*;


total_lossN!�@

error_R�g`?

learning_rate_1��8:�\�I       6%�	�?S���A�.*;


total_loss���@

error_R��[?

learning_rate_1��8l�I       6%�	&V?S���A�.*;


total_loss$��@

error_R��P?

learning_rate_1��8P�qI       6%�	��?S���A�.*;


total_loss�>�@

error_R.:[?

learning_rate_1��8|�r(I       6%�	�?S���A�.*;


total_lossjA

error_R�<?

learning_rate_1��89�C�I       6%�	�@S���A�.*;


total_loss��@

error_R��H?

learning_rate_1��8T*N�I       6%�	dW@S���A�.*;


total_loss=�A

error_R�`?

learning_rate_1��8h\�~I       6%�	G�@S���A�.*;


total_loss�@

error_R�\I?

learning_rate_1��8��T�I       6%�	b�@S���A�.*;


total_lossW��@

error_R=�M?

learning_rate_1��8p��I       6%�	�+AS���A�.*;


total_lossw��@

error_R�V?

learning_rate_1��8O6��I       6%�	ԈAS���A�.*;


total_lossA&�@

error_Rw�O?

learning_rate_1��8mv�I       6%�	4�AS���A�.*;


total_lossi/�@

error_R�-U?

learning_rate_1��8�X��I       6%�	�BS���A�.*;


total_loss)��@

error_R?N?

learning_rate_1��8e���I       6%�	�uBS���A�.*;


total_lossz�@

error_R7�P?

learning_rate_1��8�M�I       6%�	k�BS���A�.*;


total_lossf�!A

error_R�[?

learning_rate_1��8�8�5I       6%�	8CS���A�.*;


total_loss��@

error_R��C?

learning_rate_1��8���I       6%�	RCS���A�.*;


total_lossZ�@

error_RaO?

learning_rate_1��8{�.I       6%�	@�CS���A�.*;


total_loss�M�@

error_R��N?

learning_rate_1��8#���I       6%�	U�CS���A�.*;


total_loss��@

error_R̵K?

learning_rate_1��8���3I       6%�	DS���A�.*;


total_loss!B�@

error_R��N?

learning_rate_1��8P��vI       6%�	�ZDS���A�.*;


total_lossi��@

error_R2H?

learning_rate_1��8~ ��I       6%�	�DS���A�.*;


total_loss�@

error_R�d?

learning_rate_1��8��A�I       6%�	B�DS���A�.*;


total_loss�i�@

error_R�O?

learning_rate_1��8觌qI       6%�	p ES���A�.*;


total_lossOS�@

error_R{�Q?

learning_rate_1��87���I       6%�	�cES���A�.*;


total_loss��@

error_R�(Y?

learning_rate_1��8�9=I       6%�	E�ES���A�.*;


total_loss@

error_R!RL?

learning_rate_1��8MȷI       6%�	�ES���A�.*;


total_loss�
A

error_R��Q?

learning_rate_1��8���I       6%�	3/FS���A�.*;


total_loss���@

error_R��??

learning_rate_1��8/��I       6%�	>pFS���A�.*;


total_loss�Q�@

error_Ri(T?

learning_rate_1��88k�I       6%�	�FS���A�.*;


total_loss���@

error_R�Q?

learning_rate_1��8�I�eI       6%�	��FS���A�.*;


total_loss�X�@

error_R�3??

learning_rate_1��8r�4�I       6%�	�6GS���A�.*;


total_lossA�@

error_R��U?

learning_rate_1��8]���I       6%�	�wGS���A�.*;


total_losss)A

error_R̝O?

learning_rate_1��8�XMI       6%�	t�GS���A�.*;


total_lossHԷ@

error_R��O?

learning_rate_1��8%'R^I       6%�	��GS���A�.*;


total_loss�`A

error_R�X?

learning_rate_1��8�([I       6%�	�4HS���A�.*;


total_loss���@

error_R :T?

learning_rate_1��8�L7jI       6%�	MuHS���A�.*;


total_loss��A

error_R.�C?

learning_rate_1��8�)�I       6%�	´HS���A�.*;


total_lossZ��@

error_R��C?

learning_rate_1��8r��jI       6%�	#�HS���A�.*;


total_loss<V�@

error_R��K?

learning_rate_1��8[��I       6%�	�4IS���A�.*;


total_loss{a�@

error_R*HH?

learning_rate_1��8�Փ�I       6%�	�wIS���A�.*;


total_lossH�@

error_R��L?

learning_rate_1��8A��I       6%�	D�IS���A�.*;


total_loss,�@

error_Rq(^?

learning_rate_1��8�o�cI       6%�	�JS���A�.*;


total_lossJ�@

error_R�P?

learning_rate_1��8����I       6%�	4bJS���A�.*;


total_loss���@

error_RC�9?

learning_rate_1��8�hG�I       6%�	��JS���A�.*;


total_lossDs�@

error_R��M?

learning_rate_1��8���I       6%�	��JS���A�.*;


total_loss6��@

error_Rm�U?

learning_rate_1��8%�9I       6%�	�(KS���A�.*;


total_losss��@

error_R ~Q?

learning_rate_1��8K)�I       6%�	TlKS���A�.*;


total_loss�_�@

error_R&�W?

learning_rate_1��8�֖�I       6%�	"�KS���A�.*;


total_loss�פ@

error_R�>>?

learning_rate_1��8\
hI       6%�	�KS���A�.*;


total_loss�|�@

error_R�+A?

learning_rate_1��8^Gr�I       6%�	;4LS���A�.*;


total_lossD�@

error_Rz�R?

learning_rate_1��8����I       6%�	�vLS���A�.*;


total_lossZ��@

error_RR�N?

learning_rate_1��8#-I       6%�	��LS���A�.*;


total_loss�م@

error_R{�K?

learning_rate_1��8g5fI       6%�	�MS���A�.*;


total_lossIr�@

error_R��B?

learning_rate_1��8�q��I       6%�	�EMS���A�.*;


total_loss
h@

error_R�LJ?

learning_rate_1��8�ߏ\I       6%�	��MS���A�.*;


total_loss���@

error_R�f?

learning_rate_1��8Ʌ�qI       6%�	��MS���A�.*;


total_loss�ճ@

error_R��K?

learning_rate_1��8^��I       6%�	�NS���A�.*;


total_loss���@

error_R	�R?

learning_rate_1��80H@�I       6%�	�ENS���A�.*;


total_loss|�A

error_R�Wc?

learning_rate_1��8��X;I       6%�	�NS���A�.*;


total_loss���@

error_RN?

learning_rate_1��8D'��I       6%�	��NS���A�.*;


total_loss��@

error_Ro�I?

learning_rate_1��8��=I       6%�	;	OS���A�.*;


total_loss#2�@

error_R}]Z?

learning_rate_1��8'��I       6%�	�NOS���A�.*;


total_loss.,�@

error_R�sP?

learning_rate_1��8��ӊI       6%�	ҍOS���A�.*;


total_lossxwMA

error_RŁP?

learning_rate_1��8M��I       6%�	��OS���A�.*;


total_loss1~R@

error_R�j?

learning_rate_1��85ǸI       6%�	UPS���A�.*;


total_loss��@

error_R�uE?

learning_rate_1��8���WI       6%�	�OPS���A�.*;


total_loss8R�@

error_R2/U?

learning_rate_1��8�q�-I       6%�	5�PS���A�.*;


total_loss���@

error_R�8M?

learning_rate_1��8�R��I       6%�	��PS���A�.*;


total_loss)Q�@

error_R
"C?

learning_rate_1��8]Xb�I       6%�	�QS���A�.*;


total_loss*��@

error_R4�S?

learning_rate_1��8���?I       6%�	�NQS���A�.*;


total_loss�RA

error_R�p?

learning_rate_1��8��F�I       6%�	�QS���A�.*;


total_loss�.�@

error_R�<E?

learning_rate_1��8�S�I       6%�	e�QS���A�.*;


total_loss���@

error_R�IF?

learning_rate_1��8Em}kI       6%�	*RS���A�.*;


total_lossᶘ@

error_R��B?

learning_rate_1��8���pI       6%�	�QRS���A�.*;


total_loss�c�@

error_R %N?

learning_rate_1��8�4O�I       6%�	��RS���A�.*;


total_loss|��@

error_R��S?

learning_rate_1��8�	�I       6%�	>�RS���A�.*;


total_loss�@

error_R�J?

learning_rate_1��8���I       6%�	�SS���A�.*;


total_loss@

error_R@�T?

learning_rate_1��8s�G�I       6%�	\SS���A�.*;


total_loss�I�@

error_R��E?

learning_rate_1��84FXI       6%�	g�SS���A�.*;


total_loss�-�@

error_R`\?

learning_rate_1��8�gfhI       6%�	y�SS���A�.*;


total_loss��
A

error_Rc7N?

learning_rate_1��8g/MI       6%�	�TS���A�.*;


total_loss7��@

error_R�kM?

learning_rate_1��8��[lI       6%�	S\TS���A�.*;


total_loss8�A

error_R�#H?

learning_rate_1��8�e��I       6%�	�TS���A�.*;


total_loss_h�@

error_R�E?

learning_rate_1��8��DI       6%�	1�TS���A�.*;


total_loss���@

error_R;�I?

learning_rate_1��8�ceI       6%�	�#US���A�.*;


total_loss�3�@

error_R�FP?

learning_rate_1��8.�3I       6%�	%kUS���A�.*;


total_loss��@

error_R��P?

learning_rate_1��8Q�I       6%�	��US���A�.*;


total_losse�@

error_Rۢ]?

learning_rate_1��8f�I       6%�	"�US���A�.*;


total_loss�G�@

error_R��b?

learning_rate_1��8���:I       6%�	F3VS���A�.*;


total_lossÆ@

error_RInU?

learning_rate_1��8��[�I       6%�	rxVS���A�.*;


total_loss���@

error_R,�W?

learning_rate_1��8Y�,I       6%�	�VS���A�.*;


total_loss	��@

error_R�V?

learning_rate_1��8��I       6%�	�WS���A�.*;


total_loss�1A

error_R�q=?

learning_rate_1��8lח�I       6%�	nGWS���A�.*;


total_loss�L�@

error_R(M?

learning_rate_1��8α~I       6%�	:�WS���A�.*;


total_loss��@

error_R�b_?

learning_rate_1��8����I       6%�	x�WS���A�.*;


total_loss6�@

error_RV8V?

learning_rate_1��8v�I       6%�	8XS���A�.*;


total_loss:�w@

error_R�+O?

learning_rate_1��8�,��I       6%�	MXS���A�.*;


total_loss���@

error_RL;G?

learning_rate_1��8����I       6%�	ҋXS���A�.*;


total_lossC�@

error_R�YY?

learning_rate_1��8mĈI       6%�	�XS���A�.*;


total_lossh�@

error_R�Ih?

learning_rate_1��8���"I       6%�	rYS���A�.*;


total_lossʼ�@

error_R�PL?

learning_rate_1��8~��BI       6%�	]\S���A�.*;


total_loss1��@

error_R��[?

learning_rate_1��8EM�I       6%�	�\S���A�.*;


total_lossԚ�@

error_R�?O?

learning_rate_1��8���<I       6%�	��\S���A�.*;


total_loss ��@

error_R�LK?

learning_rate_1��86�nXI       6%�	']S���A�.*;


total_lossC��@

error_R֒O?

learning_rate_1��8}Ǡ�I       6%�	$i]S���A�.*;


total_loss�ї@

error_Rr�M?

learning_rate_1��8ԺR�I       6%�	i�]S���A�.*;


total_loss�@

error_R��P?

learning_rate_1��8h�`NI       6%�	^�]S���A�.*;


total_loss�{�@

error_R��D?

learning_rate_1��8�I�I       6%�	+8^S���A�.*;


total_loss�1�@

error_Rhg?

learning_rate_1��8[쩇I       6%�	e{^S���A�.*;


total_lossʥ@A

error_R]�O?

learning_rate_1��8:�w�I       6%�	�^S���A�.*;


total_loss	�,A

error_R@�Q?

learning_rate_1��8�C��I       6%�	h�^S���A�.*;


total_loss���@

error_R)�\?

learning_rate_1��89B��I       6%�	Q=_S���A�.*;


total_lossl/�@

error_R\<J?

learning_rate_1��8�i:9I       6%�	�~_S���A�.*;


total_loss�Z�@

error_R@�:?

learning_rate_1��8aI       6%�	�_S���A�.*;


total_loss��A

error_RVoN?

learning_rate_1��8���I       6%�	��_S���A�.*;


total_loss���@

error_R8J?

learning_rate_1��8��^�I       6%�	>`S���A�/*;


total_loss��@

error_R��D?

learning_rate_1��8~�
nI       6%�	9~`S���A�/*;


total_loss|lA

error_R��L?

learning_rate_1��8�I�I       6%�	��`S���A�/*;


total_loss���@

error_R֮K?

learning_rate_1��8?dZI       6%�	aS���A�/*;


total_loss/�}@

error_R��J?

learning_rate_1��8K��qI       6%�	�caS���A�/*;


total_loss�`�@

error_R�MO?

learning_rate_1��8�4ysI       6%�	��aS���A�/*;


total_loss��@

error_R�T?

learning_rate_1��8V$��I       6%�	&�aS���A�/*;


total_loss]��@

error_R��P?

learning_rate_1��8�I       6%�	.>bS���A�/*;


total_lossR�@

error_R-c?

learning_rate_1��8.k�	I       6%�	قbS���A�/*;


total_loss� �@

error_R7�S?

learning_rate_1��8zo��I       6%�	m�bS���A�/*;


total_loss�J�@

error_R�Q?

learning_rate_1��8r�T�I       6%�	KcS���A�/*;


total_loss�@

error_R�W?

learning_rate_1��8ynܣI       6%�	TOcS���A�/*;


total_lossw0�@

error_R;�P?

learning_rate_1��8�QNI       6%�	��cS���A�/*;


total_loss&��@

error_RlW?

learning_rate_1��8���I       6%�	W�cS���A�/*;


total_loss6
�@

error_RuT?

learning_rate_1��8P�xI       6%�	'dS���A�/*;


total_loss���@

error_R�_?

learning_rate_1��8s/՜I       6%�	�[dS���A�/*;


total_loss���@

error_R��F?

learning_rate_1��8+�BI       6%�	�dS���A�/*;


total_loss���@

error_R�M?

learning_rate_1��8��`�I       6%�	��dS���A�/*;


total_loss�@

error_R0?

learning_rate_1��8r��I       6%�	b"eS���A�/*;


total_loss#��@

error_RX�c?

learning_rate_1��8!�/;I       6%�	�jeS���A�/*;


total_loss���@

error_R�/3?

learning_rate_1��8�*i�I       6%�	v�eS���A�/*;


total_loss�U@

error_R\GL?

learning_rate_1��8��O�I       6%�	��eS���A�/*;


total_loss���@

error_R&R?

learning_rate_1��8U
�I       6%�	/fS���A�/*;


total_loss�A

error_R@D_?

learning_rate_1��8�.H�I       6%�	�ofS���A�/*;


total_loss�w�@

error_R�3E?

learning_rate_1��8��GwI       6%�	ĸfS���A�/*;


total_loss��@

error_R\DW?

learning_rate_1��8�$��I       6%�	=gS���A�/*;


total_lossv~�@

error_R%�M?

learning_rate_1��8�n%I       6%�	�fgS���A�/*;


total_loss���@

error_R�8?

learning_rate_1��8����I       6%�	9�gS���A�/*;


total_loss4�@

error_R��L?

learning_rate_1��88�I       6%�	W�gS���A�/*;


total_lossS��@

error_R�P?

learning_rate_1��8P���I       6%�	�=hS���A�/*;


total_loss}�@

error_R�bl?

learning_rate_1��8�j�wI       6%�	�hS���A�/*;


total_loss���@

error_R�N?

learning_rate_1��8��MOI       6%�	�hS���A�/*;


total_loss��@

error_R
�I?

learning_rate_1��8�qU I       6%�	�0iS���A�/*;


total_loss�`�@

error_R�@J?

learning_rate_1��8D�+I       6%�	�iS���A�/*;


total_lossן�@

error_R�Q_?

learning_rate_1��8ʔ�I       6%�	��iS���A�/*;


total_loss}��@

error_Ra�X?

learning_rate_1��8&�II       6%�	EjS���A�/*;


total_lossz�@

error_R� D?

learning_rate_1��8+��I       6%�	S�jS���A�/*;


total_loss���@

error_RϞZ?

learning_rate_1��8���I       6%�	�jS���A�/*;


total_lossrk�@

error_R�?i?

learning_rate_1��8�(#�I       6%�	�kS���A�/*;


total_lossrS�@

error_R�5X?

learning_rate_1��8� FYI       6%�	�SkS���A�/*;


total_lossԻ A

error_RAN?

learning_rate_1��8r#3�I       6%�	a�kS���A�/*;


total_loss�m�@

error_Rn�U?

learning_rate_1��8�tI       6%�	-�kS���A�/*;


total_loss��@

error_R�>P?

learning_rate_1��8�I       6%�	NlS���A�/*;


total_loss�]A

error_R�M?

learning_rate_1��8��0EI       6%�	b�lS���A�/*;


total_lossW=�@

error_RԗA?

learning_rate_1��80U.gI       6%�	��lS���A�/*;


total_loss�@

error_RäE?

learning_rate_1��8�_zgI       6%�	X'mS���A�/*;


total_loss`��@

error_R��G?

learning_rate_1��8E��I       6%�	�imS���A�/*;


total_loss�A

error_Rx
E?

learning_rate_1��8�-�I       6%�	��mS���A�/*;


total_loss�(�@

error_R�nU?

learning_rate_1��8�{�I       6%�	B�mS���A�/*;


total_loss��@

error_R{�K?

learning_rate_1��8��*�I       6%�	^4nS���A�/*;


total_loss ��@

error_R��X?

learning_rate_1��8��B�I       6%�	�ynS���A�/*;


total_loss��@

error_Ra1W?

learning_rate_1��8�*��I       6%�	�nS���A�/*;


total_loss_�@

error_R��@?

learning_rate_1��8�C�I       6%�	oS���A�/*;


total_loss���@

error_R=SQ?

learning_rate_1��8ܻ0	I       6%�	�EoS���A�/*;


total_loss�)�@

error_R��B?

learning_rate_1��8?�1�I       6%�	�oS���A�/*;


total_lossV�	A

error_R��U?

learning_rate_1��8I^�I       6%�	8�oS���A�/*;


total_loss��@

error_RݘI?

learning_rate_1��8r��I       6%�	'pS���A�/*;


total_loss�s@

error_RsM>?

learning_rate_1��8P��I       6%�	�OpS���A�/*;


total_losslt�@

error_R�!S?

learning_rate_1��8�m�/I       6%�	d�pS���A�/*;


total_lossH�@

error_R��4?

learning_rate_1��8D#�I       6%�	�pS���A�/*;


total_loss��Y@

error_R{[P?

learning_rate_1��8Bk��I       6%�	lqS���A�/*;


total_loss3Y�@

error_R�1M?

learning_rate_1��8v��I       6%�	TqS���A�/*;


total_lossZ��@

error_R](>?

learning_rate_1��8��E<I       6%�	T�qS���A�/*;


total_loss��@

error_Ri�H?

learning_rate_1��8i�uWI       6%�	��qS���A�/*;


total_loss具@

error_R��L?

learning_rate_1��8 �I       6%�	�rS���A�/*;


total_loss�r�@

error_R.LL?

learning_rate_1��8NZ�7I       6%�	�YrS���A�/*;


total_loss@�@

error_R�pY?

learning_rate_1��8<�CI       6%�	��rS���A�/*;


total_loss ݻ@

error_RI$9?

learning_rate_1��8m���I       6%�	��rS���A�/*;


total_loss ��@

error_R!�I?

learning_rate_1��8�A�I       6%�	�sS���A�/*;


total_lossO��@

error_R��U?

learning_rate_1��8�x�rI       6%�	]sS���A�/*;


total_loss2�@

error_R�?c?

learning_rate_1��8mםI       6%�	N�sS���A�/*;


total_lossC�@

error_R? \?

learning_rate_1��8�G5�I       6%�	1�sS���A�/*;


total_loss��@

error_R�eW?

learning_rate_1��8����I       6%�	AtS���A�/*;


total_loss���@

error_R�U?

learning_rate_1��8�� I       6%�	H\tS���A�/*;


total_loss��@

error_R�QI?

learning_rate_1��8ڰRI       6%�	��tS���A�/*;


total_loss�J�@

error_R�5D?

learning_rate_1��8GZ@I       6%�	��tS���A�/*;


total_lossF�@

error_R�QJ?

learning_rate_1��8�J��I       6%�	�uS���A�/*;


total_loss���@

error_R�`i?

learning_rate_1��8��"�I       6%�	WZuS���A�/*;


total_lossSf�@

error_R��J?

learning_rate_1��8k�I       6%�	&�uS���A�/*;


total_loss��@

error_R�%[?

learning_rate_1��8¤ivI       6%�	��uS���A�/*;


total_loss�|�@

error_R�W?

learning_rate_1��8�y{�I       6%�	vS���A�/*;


total_loss;о@

error_R=�n?

learning_rate_1��8j�dI       6%�	'XvS���A�/*;


total_lossw��@

error_R�TP?

learning_rate_1��8���I       6%�	�vS���A�/*;


total_loss=��@

error_R�';?

learning_rate_1��8?���I       6%�	��vS���A�/*;


total_loss���@

error_R$�J?

learning_rate_1��8�n�tI       6%�	?wS���A�/*;


total_loss:�@

error_R��U?

learning_rate_1��8��<I       6%�	?ZwS���A�/*;


total_loss�m�@

error_R[N?

learning_rate_1��8_#y�I       6%�	9�wS���A�/*;


total_loss�֘@

error_RXzJ?

learning_rate_1��8-��?I       6%�	C�wS���A�/*;


total_loss�~~@

error_R-K?

learning_rate_1��8��I       6%�	�xS���A�/*;


total_losscG A

error_R!�a?

learning_rate_1��8���I       6%�	0]xS���A�/*;


total_loss͝�@

error_R��K?

learning_rate_1��8�	�4I       6%�	�xS���A�/*;


total_loss�sA

error_R�E?

learning_rate_1��8V��I       6%�	��xS���A�/*;


total_lossJ��@

error_R��G?

learning_rate_1��8����I       6%�	 yS���A�/*;


total_loss�U�@

error_R�<G?

learning_rate_1��8Gx�I       6%�	�`yS���A�/*;


total_lossL\�@

error_R�[?

learning_rate_1��8;�ܽI       6%�	[�yS���A�/*;


total_lossqȴ@

error_R�nO?

learning_rate_1��8��'I       6%�	V�yS���A�/*;


total_losssʢ@

error_R.�U?

learning_rate_1��8iL,I       6%�	�RzS���A�/*;


total_loss�L�@

error_RH I?

learning_rate_1��8�˥�I       6%�	"�zS���A�/*;


total_losssB�@

error_R�mP?

learning_rate_1��8����I       6%�	��zS���A�/*;


total_loss�
�@

error_R׀C?

learning_rate_1��8����I       6%�	${S���A�/*;


total_loss>��@

error_RO�e?

learning_rate_1��8�� UI       6%�	u[{S���A�/*;


total_loss��@

error_R�M?

learning_rate_1��8�m I       6%�	�{S���A�/*;


total_loss6��@

error_R��C?

learning_rate_1��8:�J.I       6%�	��{S���A�/*;


total_loss�rA

error_Ra8?

learning_rate_1��8��<�I       6%�	Y$|S���A�/*;


total_losso��@

error_R�rJ?

learning_rate_1��8xzd�I       6%�	�h|S���A�/*;


total_loss	��@

error_Rd�R?

learning_rate_1��8�
dI       6%�	��|S���A�/*;


total_loss��@

error_Rq�K?

learning_rate_1��8��Z}I       6%�	{�|S���A�/*;


total_loss*��@

error_R;.d?

learning_rate_1��8.'3�I       6%�	c3}S���A�/*;


total_loss�s�@

error_R�&G?

learning_rate_1��8�U�I       6%�	�z}S���A�/*;


total_loss|��@

error_RzN?

learning_rate_1��8��M�I       6%�	��}S���A�/*;


total_loss_��@

error_R$�Z?

learning_rate_1��8����I       6%�	x-~S���A�/*;


total_loss�b@

error_Rm-Y?

learning_rate_1��8�P"I       6%�	�t~S���A�/*;


total_lossO6�@

error_R�xS?

learning_rate_1��8z���I       6%�	�~S���A�/*;


total_lossXA

error_R]f^?

learning_rate_1��8;H�I       6%�	�~S���A�/*;


total_loss�f�@

error_R1�H?

learning_rate_1��8o�I       6%�	�>S���A�/*;


total_loss�m�@

error_RkV?

learning_rate_1��8�F�I       6%�	�S���A�/*;


total_loss&�@

error_R�]?

learning_rate_1��8�ߺ�I       6%�	��S���A�/*;


total_lossA��@

error_R�Lb?

learning_rate_1��8��(�I       6%�	��S���A�/*;


total_loss���@

error_R�\?

learning_rate_1��8��w�I       6%�	<D�S���A�/*;


total_loss.�@

error_R��I?

learning_rate_1��8��)II       6%�	&��S���A�/*;


total_losswH�@

error_R
5`?

learning_rate_1��8'��LI       6%�	ЀS���A�/*;


total_lossn�@

error_R�b?

learning_rate_1��8�z{I       6%�	�/�S���A�/*;


total_loss��A

error_R��Y?

learning_rate_1��8��I       6%�	Cy�S���A�/*;


total_loss��@

error_R.�G?

learning_rate_1��8�u�I       6%�	u��S���A�/*;


total_loss���@

error_R� Z?

learning_rate_1��8�aɰI       6%�	��S���A�/*;


total_loss
�@

error_R��S?

learning_rate_1��8b�tI       6%�	�f�S���A�/*;


total_loss�A

error_RC�g?

learning_rate_1��8)AI       6%�	���S���A�/*;


total_loss�H�@

error_R�/Z?

learning_rate_1��80�)DI       6%�	��S���A�/*;


total_lossSFA

error_R�IF?

learning_rate_1��8Q��jI       6%�	M.�S���A�0*;


total_loss���@

error_RN�P?

learning_rate_1��8�_L�I       6%�	"n�S���A�0*;


total_loss]��@

error_R�Q?

learning_rate_1��8K�k�I       6%�	���S���A�0*;


total_loss���@

error_R��b?

learning_rate_1��8�zŦI       6%�	l��S���A�0*;


total_loss�Qh@

error_Rv:?

learning_rate_1��8M�cI       6%�	�B�S���A�0*;


total_loss��@

error_R:�L?

learning_rate_1��8�"�I       6%�	o��S���A�0*;


total_loss)��@

error_R��^?

learning_rate_1��8;���I       6%�	�ĄS���A�0*;


total_loss��@

error_RhA?

learning_rate_1��8�`	QI       6%�	?�S���A�0*;


total_loss�;�@

error_RA`?

learning_rate_1��8sm�I       6%�	vI�S���A�0*;


total_loss��@

error_R�~b?

learning_rate_1��8�a�I       6%�	 ��S���A�0*;


total_lossu�@

error_R6S?

learning_rate_1��8��R�I       6%�	�˅S���A�0*;


total_loss���@

error_R�T?

learning_rate_1��8S#v�I       6%�	��S���A�0*;


total_loss���@

error_R�
G?

learning_rate_1��8����I       6%�	O�S���A�0*;


total_loss�<�@

error_RIrT?

learning_rate_1��8&�AvI       6%�	���S���A�0*;


total_loss���@

error_RsbE?

learning_rate_1��8�E�uI       6%�	YІS���A�0*;


total_loss<%�@

error_Rf�Q?

learning_rate_1��8�Ի�I       6%�	��S���A�0*;


total_lossT�@

error_R�r^?

learning_rate_1��8�oq�I       6%�	�P�S���A�0*;


total_loss! �@

error_R�3T?

learning_rate_1��8�O��I       6%�	ώ�S���A�0*;


total_lossv��@

error_R!�H?

learning_rate_1��8(�I       6%�	�ЇS���A�0*;


total_loss7c�@

error_R��Q?

learning_rate_1��8���I       6%�	f�S���A�0*;


total_loss���@

error_RlL?

learning_rate_1��8�G�2I       6%�	�Y�S���A�0*;


total_loss�K�@

error_RDtC?

learning_rate_1��8qo��I       6%�	���S���A�0*;


total_loss�6�@

error_RFD_?

learning_rate_1��8p[�(I       6%�	�وS���A�0*;


total_loss���@

error_R�gR?

learning_rate_1��8ОqtI       6%�	��S���A�0*;


total_loss2ɔ@

error_RwP?

learning_rate_1��8�zI       6%�	*[�S���A�0*;


total_loss% A

error_R�D?

learning_rate_1��8��--I       6%�	q��S���A�0*;


total_lossH��@

error_Rs*R?

learning_rate_1��8��{I       6%�	��S���A�0*;


total_loss��@

error_R\�B?

learning_rate_1��8o9�I       6%�	.2�S���A�0*;


total_loss�A�@

error_RT�N?

learning_rate_1��8��`!I       6%�	,w�S���A�0*;


total_loss��@

error_R��W?

learning_rate_1��8�I�I       6%�	x��S���A�0*;


total_lossvL�@

error_R[�T?

learning_rate_1��8�A�I       6%�	���S���A�0*;


total_losstv�@

error_R$�O?

learning_rate_1��8�}��I       6%�	E�S���A�0*;


total_loss/��@

error_R�gQ?

learning_rate_1��8�I       6%�	҆�S���A�0*;


total_loss#C�@

error_R��a?

learning_rate_1��8���I       6%�	4ƋS���A�0*;


total_loss؞�@

error_R#�R?

learning_rate_1��89��I       6%�	��S���A�0*;


total_loss@��@

error_R�NI?

learning_rate_1��8~�LnI       6%�	2O�S���A�0*;


total_loss���@

error_R��K?

learning_rate_1��8|�1oI       6%�	4��S���A�0*;


total_lossȗ@

error_R{D?

learning_rate_1��8$БMI       6%�	�ьS���A�0*;


total_loss���@

error_R��W?

learning_rate_1��8��I       6%�	��S���A�0*;


total_loss���@

error_RL�S?

learning_rate_1��8>t�zI       6%�	�_�S���A�0*;


total_loss��@

error_RS�\?

learning_rate_1��8 m�0I       6%�	裍S���A�0*;


total_loss�aA

error_RC:R?

learning_rate_1��8��`5I       6%�	P�S���A�0*;


total_loss�OA

error_R�GH?

learning_rate_1��84x�I       6%�	l2�S���A�0*;


total_loss�X�@

error_RP?

learning_rate_1��8�g�9I       6%�	sx�S���A�0*;


total_lossn�@

error_R:`?

learning_rate_1��8�_N;I       6%�	��S���A�0*;


total_loss$A

error_R�F?

learning_rate_1��8�>yI       6%�	�S���A�0*;


total_loss�]�@

error_R��O?

learning_rate_1��8�,ZI       6%�	�G�S���A�0*;


total_loss=%�@

error_RiJ_?

learning_rate_1��8���pI       6%�	b��S���A�0*;


total_loss5z@

error_R��T?

learning_rate_1��8Q&0I       6%�	�яS���A�0*;


total_lossS��@

error_R�#V?

learning_rate_1��8��I       6%�	�S���A�0*;


total_loss:�@

error_R3�S?

learning_rate_1��8*AgEI       6%�	�R�S���A�0*;


total_loss/y�@

error_R�BZ?

learning_rate_1��8�_�I       6%�	R��S���A�0*;


total_loss�.�@

error_Rq�]?

learning_rate_1��85�UI       6%�	<אS���A�0*;


total_lossO��@

error_R$E`?

learning_rate_1��8��	�I       6%�	��S���A�0*;


total_loss�8�@

error_R�V?

learning_rate_1��84�[�I       6%�	�X�S���A�0*;


total_lossﺭ@

error_R�2?

learning_rate_1��8�q�XI       6%�	�S���A�0*;


total_loss�c�@

error_R��A?

learning_rate_1��8k��I       6%�	�ܑS���A�0*;


total_loss��@

error_R�.^?

learning_rate_1��8k���I       6%�	6�S���A�0*;


total_loss��@

error_R��K?

learning_rate_1��8�>�I       6%�	`�S���A�0*;


total_loss�m�@

error_R�
J?

learning_rate_1��8�M��I       6%�	S���A�0*;


total_loss]��@

error_R��D?

learning_rate_1��84���I       6%�	n�S���A�0*;


total_losss��@

error_R��O?

learning_rate_1��8?�"I       6%�	[!�S���A�0*;


total_loss���@

error_R��W?

learning_rate_1��8BxI       6%�	�b�S���A�0*;


total_loss�O�@

error_RR�F?

learning_rate_1��8���MI       6%�	P��S���A�0*;


total_lossz�@

error_R�\?

learning_rate_1��8� _I       6%�	i�S���A�0*;


total_loss��@

error_RmB`?

learning_rate_1��8]2tI       6%�	# �S���A�0*;


total_lossck�@

error_R,�K?

learning_rate_1��8d�-�I       6%�	�_�S���A�0*;


total_loss|s�@

error_R��W?

learning_rate_1��8��I       6%�	3��S���A�0*;


total_lossӤ�@

error_R�P?

learning_rate_1��8iG�I       6%�	=��S���A�0*;


total_loss�	�@

error_R<�K?

learning_rate_1��8�I       6%�	�"�S���A�0*;


total_loss(��@

error_R�:I?

learning_rate_1��83�/tI       6%�	>c�S���A�0*;


total_loss�6�@

error_RJN?

learning_rate_1��8��I       6%�	���S���A�0*;


total_loss��@

error_R]b?

learning_rate_1��87�n�I       6%�	��S���A�0*;


total_lossB�@

error_R��9?

learning_rate_1��8	ܸ4I       6%�	�1�S���A�0*;


total_loss��@

error_R��Y?

learning_rate_1��8�IFI       6%�	�t�S���A�0*;


total_loss�w�@

error_R��E?

learning_rate_1��8��ÔI       6%�	=��S���A�0*;


total_loss��@

error_RWBD?

learning_rate_1��8��w|I       6%�	���S���A�0*;


total_lossU��@

error_R�-H?

learning_rate_1��8�7W�I       6%�	�C�S���A�0*;


total_loss!�Y@

error_Rn�S?

learning_rate_1��8c	�tI       6%�	��S���A�0*;


total_loss���@

error_R�,Z?

learning_rate_1��8�D_�I       6%�	�S���A�0*;


total_loss���@

error_R�kL?

learning_rate_1��8�F�I       6%�	c�S���A�0*;


total_lossf�@

error_R}�R?

learning_rate_1��8����I       6%�	6E�S���A�0*;


total_loss;��@

error_R�Q_?

learning_rate_1��8ܦѱI       6%�	Ņ�S���A�0*;


total_loss���@

error_RON?

learning_rate_1��8۟�qI       6%�	�ŘS���A�0*;


total_lossC�@

error_R�gE?

learning_rate_1��8�@�I       6%�	;�S���A�0*;


total_loss-��@

error_RnzS?

learning_rate_1��8�ur�I       6%�	�E�S���A�0*;


total_loss���@

error_R��G?

learning_rate_1��8UH)�I       6%�	d��S���A�0*;


total_loss�v�@

error_R�P?

learning_rate_1��8A�GmI       6%�	�ęS���A�0*;


total_loss*+�@

error_R� U?

learning_rate_1��8��gI       6%�	��S���A�0*;


total_lossw@

error_R}�K?

learning_rate_1��8�f��I       6%�	�_�S���A�0*;


total_loss):�@

error_R�^?

learning_rate_1��8A��I       6%�	櫚S���A�0*;


total_loss}��@

error_Ri1L?

learning_rate_1��8;�I       6%�	��S���A�0*;


total_loss�X�@

error_R��V?

learning_rate_1��8��#hI       6%�	�.�S���A�0*;


total_loss.��@

error_RJVT?

learning_rate_1��8�I       6%�	 r�S���A�0*;


total_loss�(�@

error_R�[?

learning_rate_1��8.�$I       6%�	=��S���A�0*;


total_loss$c<A

error_Rn�I?

learning_rate_1��8����I       6%�	>��S���A�0*;


total_lossw�@

error_R3mF?

learning_rate_1��8�;�nI       6%�	yD�S���A�0*;


total_losss`�@

error_R��]?

learning_rate_1��8�^\�I       6%�	���S���A�0*;


total_loss�&R@

error_R3;?

learning_rate_1��8����I       6%�	kМS���A�0*;


total_loss\��@

error_R��H?

learning_rate_1��8�"p0I       6%�	i�S���A�0*;


total_loss>��@

error_R7�C?

learning_rate_1��8��1I       6%�	�S�S���A�0*;


total_loss��@

error_R�V?

learning_rate_1��8oz�aI       6%�	��S���A�0*;


total_loss(�@

error_RN3[?

learning_rate_1��8���jI       6%�	BڝS���A�0*;


total_lossRy�@

error_Rd�[?

learning_rate_1��8O_�I       6%�	��S���A�0*;


total_loss��@

error_R1T?

learning_rate_1��8�b�I       6%�	�_�S���A�0*;


total_loss��@

error_RI&M?

learning_rate_1��8-�I       6%�	\��S���A�0*;


total_lossE��@

error_R�U?

learning_rate_1��8��F�I       6%�	"�S���A�0*;


total_loss`��@

error_R�Sk?

learning_rate_1��8pۿ�I       6%�	N%�S���A�0*;


total_loss�5�@

error_R!L[?

learning_rate_1��8I\$�I       6%�	f�S���A�0*;


total_loss� �@

error_R.�9?

learning_rate_1��8?p\�I       6%�	��S���A�0*;


total_loss��@

error_Rd�h?

learning_rate_1��8���I       6%�	�S���A�0*;


total_lossH�A

error_RD�^?

learning_rate_1��8��iI       6%�	.&�S���A�0*;


total_lossj�@

error_RJ�F?

learning_rate_1��8I�5I       6%�	�d�S���A�0*;


total_loss��@

error_R��E?

learning_rate_1��8J��I       6%�	���S���A�0*;


total_loss3��@

error_R�K?

learning_rate_1��8f�"I       6%�	~�S���A�0*;


total_loss,��@

error_R�Z?

learning_rate_1��8}��;I       6%�	�=�S���A�0*;


total_loss�wA

error_R|\?

learning_rate_1��8�{}I       6%�	��S���A�0*;


total_loss���@

error_R�KN?

learning_rate_1��83��NI       6%�	nšS���A�0*;


total_loss<�@

error_R��F?

learning_rate_1��8�r0I       6%�		�S���A�0*;


total_loss���@

error_R��a?

learning_rate_1��8E�vI       6%�	�a�S���A�0*;


total_loss���@

error_Rh`L?

learning_rate_1��8U��RI       6%�	%��S���A�0*;


total_loss��@

error_R�V?

learning_rate_1��8�̗�I       6%�	��S���A�0*;


total_loss)f A

error_R\yG?

learning_rate_1��8����I       6%�	&�S���A�0*;


total_loss1T�@

error_RHLE?

learning_rate_1��8� �0I       6%�	
h�S���A�0*;


total_lossɼ@

error_R�FN?

learning_rate_1��8
|�*I       6%�	ҩ�S���A�0*;


total_lossW��@

error_R�T?

learning_rate_1��8,|�I       6%�	���S���A�0*;


total_lossr?@A

error_R�)G?

learning_rate_1��8 _�I       6%�	�1�S���A�0*;


total_loss���@

error_R�N?

learning_rate_1��8��n�I       6%�	t�S���A�0*;


total_loss���@

error_Rx�F?

learning_rate_1��8�(LI       6%�	���S���A�1*;


total_loss�a�@

error_R��>?

learning_rate_1��8���I       6%�	���S���A�1*;


total_lossh�@

error_R�R?

learning_rate_1��8�� �I       6%�	�:�S���A�1*;


total_losso��@

error_Rn�S?

learning_rate_1��8���I       6%�	N{�S���A�1*;


total_loss�A

error_R6�B?

learning_rate_1��8UV|HI       6%�	ع�S���A�1*;


total_lossT�@

error_R��Z?

learning_rate_1��8���RI       6%�	`��S���A�1*;


total_loss�>�@

error_R$pU?

learning_rate_1��8����I       6%�	�;�S���A�1*;


total_loss:��@

error_R?G?

learning_rate_1��8����I       6%�	�|�S���A�1*;


total_loss���@

error_R�[J?

learning_rate_1��8vr`QI       6%�	$��S���A�1*;


total_loss�t�@

error_RRb?

learning_rate_1��8���I       6%�	��S���A�1*;


total_lossz�@

error_Rft]?

learning_rate_1��8���I       6%�	d<�S���A�1*;


total_lossz�@

error_R!�Y?

learning_rate_1��8E���I       6%�	J}�S���A�1*;


total_loss���@

error_R��K?

learning_rate_1��8��|I       6%�	'��S���A�1*;


total_lossݕ�@

error_RWz@?

learning_rate_1��8����I       6%�	���S���A�1*;


total_loss�Hu@

error_RQS_?

learning_rate_1��8�Z.I       6%�	�?�S���A�1*;


total_lossM_�@

error_R �\?

learning_rate_1��8�\çI       6%�	l��S���A�1*;


total_loss�1�@

error_RW�\?

learning_rate_1��8���I       6%�	�ĨS���A�1*;


total_loss�e�@

error_RLRN?

learning_rate_1��8��:,I       6%�	��S���A�1*;


total_loss���@

error_R*CC?

learning_rate_1��8�"�TI       6%�	�E�S���A�1*;


total_loss��@

error_RȲU?

learning_rate_1��8P�׃I       6%�	���S���A�1*;


total_loss�Q�@

error_R��h?

learning_rate_1��8�FukI       6%�	�ũS���A�1*;


total_loss�@

error_Rn�P?

learning_rate_1��8Д�CI       6%�	�S���A�1*;


total_lossDf�@

error_R&2L?

learning_rate_1��8"��%I       6%�	sI�S���A�1*;


total_lossR�@

error_R=�F?

learning_rate_1��8KٸI       6%�	W��S���A�1*;


total_loss&ܱ@

error_R �Q?

learning_rate_1��8���QI       6%�	��S���A�1*;


total_loss���@

error_RitD?

learning_rate_1��8W*KI       6%�	�2�S���A�1*;


total_loss�y�@

error_R�2I?

learning_rate_1��8�ղ�I       6%�	9t�S���A�1*;


total_loss;�v@

error_R�3d?

learning_rate_1��8g�m=I       6%�	���S���A�1*;


total_loss��@

error_Ra1]?

learning_rate_1��8��qI       6%�	��S���A�1*;


total_loss��@

error_RM�M?

learning_rate_1��8��I       6%�	�=�S���A�1*;


total_loss!��@

error_R�J?

learning_rate_1��8�9��I       6%�	��S���A�1*;


total_loss��@

error_RT�Q?

learning_rate_1��8�ZI       6%�	�¬S���A�1*;


total_loss�� A

error_R
Ja?

learning_rate_1��8�[�I       6%�	��S���A�1*;


total_loss�@

error_RchU?

learning_rate_1��8���I       6%�	?I�S���A�1*;


total_lossT��@

error_R�_?

learning_rate_1��8�Ը�I       6%�	"��S���A�1*;


total_loss!��@

error_R<�a?

learning_rate_1��8j`�4I       6%�	BѭS���A�1*;


total_loss
,�@

error_RCSP?

learning_rate_1��8���I       6%�	Q�S���A�1*;


total_loss���@

error_R��S?

learning_rate_1��8���I       6%�	Y�S���A�1*;


total_loss�@

error_Rf}P?

learning_rate_1��8,�~�I       6%�	���S���A�1*;


total_loss�}�@

error_RX@L?

learning_rate_1��8�nI       6%�	�خS���A�1*;


total_loss��@

error_R�!B?

learning_rate_1��8���AI       6%�	N�S���A�1*;


total_loss�տ@

error_R;�<?

learning_rate_1��8��cI       6%�	�[�S���A�1*;


total_loss�6�@

error_R�mZ?

learning_rate_1��8&��I       6%�	{��S���A�1*;


total_lossI��@

error_R��U?

learning_rate_1��8�l�DI       6%�	�S���A�1*;


total_loss�$�@

error_Rf�S?

learning_rate_1��8k[t�I       6%�	-�S���A�1*;


total_lossL�@

error_R6�b?

learning_rate_1��8���I       6%�	�q�S���A�1*;


total_loss��A

error_R\�U?

learning_rate_1��8��wI       6%�	0��S���A�1*;


total_loss�a�@

error_Rz�G?

learning_rate_1��8��KeI       6%�	��S���A�1*;


total_loss�3�@

error_R,dI?

learning_rate_1��8�/�(I       6%�	�@�S���A�1*;


total_loss�S�@

error_R,t7?

learning_rate_1��8�E2�I       6%�	���S���A�1*;


total_lossD��@

error_R(�\?

learning_rate_1��8�ݷ�I       6%�	9ıS���A�1*;


total_lossZ�@

error_R�lM?

learning_rate_1��8�v�I       6%�	��S���A�1*;


total_loss�A�@

error_R�[?

learning_rate_1��8ܭ��I       6%�	\F�S���A�1*;


total_loss|s�@

error_R��S?

learning_rate_1��8�J��I       6%�	'��S���A�1*;


total_lossA�A

error_R�b^?

learning_rate_1��8�w�I       6%�	�ǲS���A�1*;


total_loss��@

error_R$(M?

learning_rate_1��8֛NI       6%�	�S���A�1*;


total_loss�c�@

error_R�kG?

learning_rate_1��8z��I       6%�	@H�S���A�1*;


total_loss�Ŀ@

error_R	�:?

learning_rate_1��8S�z�I       6%�	��S���A�1*;


total_loss��@

error_R��E?

learning_rate_1��8�?�cI       6%�	2˳S���A�1*;


total_lossc]A

error_R�[?

learning_rate_1��8qݗdI       6%�	T
�S���A�1*;


total_loss��@

error_R�J?

learning_rate_1��8� #AI       6%�	�I�S���A�1*;


total_lossiK�@

error_R�	d?

learning_rate_1��8���hI       6%�	J��S���A�1*;


total_loss��A

error_R�VG?

learning_rate_1��8]=��I       6%�	�ϴS���A�1*;


total_loss��@

error_Rt�T?

learning_rate_1��8v�4I       6%�	�S���A�1*;


total_loss���@

error_RRHY?

learning_rate_1��83���I       6%�	X�S���A�1*;


total_loss4�@

error_R�jF?

learning_rate_1��8��@I       6%�	��S���A�1*;


total_loss�8�@

error_R��Y?

learning_rate_1��8�F�I       6%�	�ߵS���A�1*;


total_lossS�@

error_RJf:?

learning_rate_1��8Re�I       6%�	�!�S���A�1*;


total_loss��S@

error_RR�]?

learning_rate_1��8�	I       6%�	pb�S���A�1*;


total_loss���@

error_R)�]?

learning_rate_1��8�w��I       6%�	5��S���A�1*;


total_loss��@

error_R�K?

learning_rate_1��8���I       6%�	��S���A�1*;


total_loss�V�@

error_R�YJ?

learning_rate_1��8,�
�I       6%�	�0�S���A�1*;


total_loss���@

error_R4�4?

learning_rate_1��8ъ��I       6%�	�t�S���A�1*;


total_loss�+�@

error_R@_?

learning_rate_1��87�I       6%�	T��S���A�1*;


total_loss6y�@

error_R$pU?

learning_rate_1��8���I       6%�	��S���A�1*;


total_loss$ڀ@

error_R�mO?

learning_rate_1��8;\}I       6%�	r@�S���A�1*;


total_loss��@

error_R��^?

learning_rate_1��8/ʊ�I       6%�	��S���A�1*;


total_lossdY�@

error_R�|E?

learning_rate_1��8q@��I       6%�	�ɸS���A�1*;


total_loss6&�@

error_R�[?

learning_rate_1��8��%sI       6%�	��S���A�1*;


total_lossd%A

error_R��]?

learning_rate_1��8�"�WI       6%�	S�S���A�1*;


total_loss�:�@

error_R��O?

learning_rate_1��8���]I       6%�	��S���A�1*;


total_loss6Ȩ@

error_R�sS?

learning_rate_1��8�rI       6%�	�۹S���A�1*;


total_loss�w�@

error_R��B?

learning_rate_1��8uvmI       6%�	"#�S���A�1*;


total_lossԼ@

error_RT?

learning_rate_1��8X�r:I       6%�	b�S���A�1*;


total_lossH��@

error_R[%P?

learning_rate_1��8�6��I       6%�	˶�S���A�1*;


total_lossz4�@

error_R�P?

learning_rate_1��8C�
GI       6%�	L �S���A�1*;


total_loss�֐@

error_RH�^?

learning_rate_1��8���I       6%�	�A�S���A�1*;


total_loss�1�@

error_R�RU?

learning_rate_1��8���I       6%�	���S���A�1*;


total_loss���@

error_RC�F?

learning_rate_1��85ݴI       6%�	rŻS���A�1*;


total_loss���@

error_R�PP?

learning_rate_1��8#J��I       6%�	c�S���A�1*;


total_loss�n�@

error_R&(^?

learning_rate_1��80@��I       6%�	&G�S���A�1*;


total_loss:��@

error_R
�V?

learning_rate_1��8��	qI       6%�	ވ�S���A�1*;


total_loss,0�@

error_R)WA?

learning_rate_1��8Vf�I       6%�	}ͼS���A�1*;


total_losst�A

error_R��G?

learning_rate_1��8��=�I       6%�	��S���A�1*;


total_lossJ��@

error_R��H?

learning_rate_1��8?B/�I       6%�	�O�S���A�1*;


total_loss�3.A

error_R��8?

learning_rate_1��8�XWI       6%�	s��S���A�1*;


total_loss&��@

error_R�F?

learning_rate_1��8],T�I       6%�	X׽S���A�1*;


total_loss{��@

error_R|�U?

learning_rate_1��8p_E�I       6%�	9�S���A�1*;


total_loss-��@

error_R��T?

learning_rate_1��8�;'�I       6%�	q�S���A�1*;


total_lossL�@

error_R!�g?

learning_rate_1��8d��I       6%�	�̾S���A�1*;


total_loss���@

error_R-�Q?

learning_rate_1��8�eN�I       6%�	��S���A�1*;


total_loss�M�@

error_RLP?

learning_rate_1��8(�y�I       6%�	�u�S���A�1*;


total_lossZ:�@

error_R<�Q?

learning_rate_1��8ޣ��I       6%�	;̿S���A�1*;


total_loss���@

error_R�rT?

learning_rate_1��8����I       6%�	!�S���A�1*;


total_loss���@

error_R��G?

learning_rate_1��8K��I       6%�	�Z�S���A�1*;


total_loss#�@

error_R��+?

learning_rate_1��8�z$jI       6%�	���S���A�1*;


total_losst�@

error_R4�C?

learning_rate_1��8W� I       6%�	��S���A�1*;


total_lossXŬ@

error_R�_?

learning_rate_1��8i���I       6%�	+:�S���A�1*;


total_loss��@

error_R��Z?

learning_rate_1��8�j�I       6%�	{��S���A�1*;


total_loss�AA

error_R<�F?

learning_rate_1��8����I       6%�	���S���A�1*;


total_loss��	A

error_RfO?

learning_rate_1��8Ts�I       6%�	;&�S���A�1*;


total_loss�U�@

error_R.�U?

learning_rate_1��8:U~�I       6%�	�i�S���A�1*;


total_lossn��@

error_R*�\?

learning_rate_1��8.:^�I       6%�	O��S���A�1*;


total_loss$��@

error_R��Q?

learning_rate_1��8���I       6%�	k�S���A�1*;


total_loss��@

error_RƍU?

learning_rate_1��8���I       6%�	K�S���A�1*;


total_loss��@

error_RT�S?

learning_rate_1��8���I       6%�	N��S���A�1*;


total_lossj9�@

error_RT�G?

learning_rate_1��8�ݫ�I       6%�	G��S���A�1*;


total_loss�@

error_Rn�S?

learning_rate_1��8�U�8I       6%�	�#�S���A�1*;


total_loss���@

error_RZZ?

learning_rate_1��8���I       6%�	�g�S���A�1*;


total_loss�g�@

error_R6�K?

learning_rate_1��8�(�/I       6%�	���S���A�1*;


total_lossf�@

error_R�@?

learning_rate_1��8k��I       6%�	�S���A�1*;


total_lossBڒ@

error_RV�_?

learning_rate_1��8�o)0