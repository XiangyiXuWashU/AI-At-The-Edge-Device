       �K"	  @T���Abrain.Event:2���>K     6�.	��eT���A"��
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
biases/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
biases/bias2/AssignAssignbiases/bias2biases/random_normal_1*
T0*
_class
loc:@biases/bias2*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
+biases/random_normal_2/RandomStandardNormalRandomStandardNormalbiases/random_normal_2/shape*
dtype0*
_output_shapes
:d*
seed2 *

seed *
T0
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
biases/bias3/AssignAssignbiases/bias3biases/random_normal_2*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@biases/bias3
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
weights_1/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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

seed *
T0*
dtype0*
_output_shapes

:d*
seed2 
�
weights_1/random_normal_3/mulMul.weights_1/random_normal_3/RandomStandardNormal weights_1/random_normal_3/stddev*
_output_shapes

:d*
T0
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
weights_1/weight_out/AssignAssignweights_1/weight_outweights_1/random_normal_3*
T0*'
_class
loc:@weights_1/weight_out*
validate_shape(*
_output_shapes

:d*
use_locking(
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
biases_1/bias1/AssignAssignbiases_1/bias1biases_1/random_normal*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes	
:�
x
biases_1/bias1/readIdentitybiases_1/bias1*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
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
-biases_1/random_normal_2/RandomStandardNormalRandomStandardNormalbiases_1/random_normal_2/shape*

seed *
T0*
dtype0*
_output_shapes
:d*
seed2 
�
biases_1/random_normal_2/mulMul-biases_1/random_normal_2/RandomStandardNormalbiases_1/random_normal_2/stddev*
_output_shapes
:d*
T0
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
layer_2/ReluRelulayer_1/Add*(
_output_shapes
:����������*
T0
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
AbsAbssub*
T0*'
_output_shapes
:���������
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
loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
_output_shapes
: *
T0
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
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
(train/gradients/layer_3/Add_grad/ReshapeReshape$train/gradients/layer_3/Add_grad/Sum&train/gradients/layer_3/Add_grad/Shape*'
_output_shapes
:���������d*
T0*
Tshape0
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
$train/gradients/layer_2/Add_grad/SumSum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad6train/gradients/layer_2/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
$train/gradients/layer_1/Add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_1/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
(train/gradients/layer_1/Add_grad/ReshapeReshape$train/gradients/layer_1/Add_grad/Sum&train/gradients/layer_1/Add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
&train/gradients/layer_1/Add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_1/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
+weights_1/weight_out/Adam/Initializer/zerosConst*'
_class
loc:@weights_1/weight_out*
valueBd*    *
dtype0*
_output_shapes

:d
�
weights_1/weight_out/Adam
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
'biases_1/bias1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*!
_class
loc:@biases_1/bias1*
valueB�*    
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
biases_1/bias3/Adam_1/AssignAssignbiases_1/bias3/Adam_1'biases_1/bias3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@biases_1/bias3
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
*train/Adam/update_biases_1/bias2/ApplyAdam	ApplyAdambiases_1/bias2biases_1/bias2/Adambiases_1/bias2/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/ExponentialDecaytrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/Add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@biases_1/bias2
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

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
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
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
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
save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:**
value!BBbiases_1/bias1/Adam_1
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
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
shrink_axis_mask*

begin_mask *
ellipsis_mask *
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
initNoOp^weights/weight1/Assign^weights/weight2/Assign^weights/weight3/Assign^weights/weight_out/Assign^biases/bias1/Assign^biases/bias2/Assign^biases/bias3/Assign^biases/bias_out/Assign^weights_1/weight1/Assign^weights_1/weight2/Assign^weights_1/weight3/Assign^weights_1/weight_out/Assign^biases_1/bias1/Assign^biases_1/bias2/Assign^biases_1/bias3/Assign^biases_1/bias_out/Assign^Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign^weights_1/weight1/Adam/Assign ^weights_1/weight1/Adam_1/Assign^weights_1/weight2/Adam/Assign ^weights_1/weight2/Adam_1/Assign^weights_1/weight3/Adam/Assign ^weights_1/weight3/Adam_1/Assign!^weights_1/weight_out/Adam/Assign#^weights_1/weight_out/Adam_1/Assign^biases_1/bias1/Adam/Assign^biases_1/bias1/Adam_1/Assign^biases_1/bias2/Adam/Assign^biases_1/bias2/Adam_1/Assign^biases_1/bias3/Adam/Assign^biases_1/bias3/Adam_1/Assign^biases_1/bias_out/Adam/Assign ^biases_1/bias_out/Adam_1/Assign" *�Ph     6Ԇg	'gT���AJ��
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
dtype0*
_output_shapes
:*
valueB"�   �   
a
weights/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
weights/random_normal_1/mulMul,weights/random_normal_1/RandomStandardNormalweights/random_normal_1/stddev* 
_output_shapes
:
��*
T0
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
biases/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
 weights_1/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
biases_1/bias1/readIdentitybiases_1/bias1*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias1
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
biases_1/bias2/readIdentitybiases_1/bias2*
_output_shapes	
:�*
T0*!
_class
loc:@biases_1/bias2
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
,learning_rate/ExponentialDecay/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
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
"train/gradients/sub_1_grad/ReshapeReshapetrain/gradients/sub_1_grad/Sum train/gradients/sub_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
$train/gradients/sub_1_grad/Reshape_1Reshapetrain/gradients/sub_1_grad/Neg"train/gradients/sub_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
5train/gradients/result/Add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/result/Add_grad/Shape'train/gradients/result/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
&train/gradients/layer_2/Add_grad/Sum_1Sum0train/gradients/layer_3/Sigmoid_grad/SigmoidGrad8train/gradients/layer_2/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
;train/gradients/layer_1/Add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/Add_grad/Reshape_12^train/gradients/layer_1/Add_grad/tuple/group_deps*
_output_shapes	
:�*
T0*=
_class3
1/loc:@train/gradients/layer_1/Add_grad/Reshape_1
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
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*!
_class
loc:@biases_1/bias1*
validate_shape(*
_output_shapes
: 
y
train/beta2_power/readIdentitytrain/beta2_power*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2.^train/Adam/update_weights_1/weight1/ApplyAdam.^train/Adam/update_weights_1/weight2/ApplyAdam.^train/Adam/update_weights_1/weight3/ApplyAdam1^train/Adam/update_weights_1/weight_out/ApplyAdam+^train/Adam/update_biases_1/bias1/ApplyAdam+^train/Adam/update_biases_1/bias2/ApplyAdam+^train/Adam/update_biases_1/bias3/ApplyAdam.^train/Adam/update_biases_1/bias_out/ApplyAdam*
_output_shapes
: *
T0*!
_class
loc:@biases_1/bias1
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
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
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
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
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
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
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
MeanMeanAbsMean/reduction_indices*

Tidx0*
	keep_dims( *
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
strided_sliceStridedSliceMeanstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
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
biases_1/bias_out/Adam_1:0biases_1/bias_out/Adam_1/Assignbiases_1/bias_out/Adam_1/read:0� ��F       r5��	ͅhT���A*;


total_loss�ԡ@

error_R$�D?

learning_rate_1�HG7}�H       ��H�	X�kT���A*;


total_loss	$e@

error_R�SL?

learning_rate_1�HG7b^A�H       ��H�	lT���A*;


total_lossQK�@

error_RJ�U?

learning_rate_1�HG762�H       ��H�	ZYlT���A*;


total_lossFp�@

error_R�V?

learning_rate_1�HG7"=H       ��H�	�lT���A*;


total_loss���@

error_R��[?

learning_rate_1�HG7\j�1H       ��H�	��lT���A*;


total_lossV�A

error_R��N?

learning_rate_1�HG7�M7�H       ��H�	_3mT���A*;


total_loss|��@

error_Rm�Z?

learning_rate_1�HG7[۱}H       ��H�	�~mT���A*;


total_loss��@

error_Rň=?

learning_rate_1�HG7x�r�H       ��H�	��mT���A*;


total_lossH6�@

error_RʏM?

learning_rate_1�HG7��|H       ��H�	4nT���A	*;


total_loss�۲@

error_R�L?

learning_rate_1�HG7 �ȃH       ��H�		]nT���A
*;


total_loss���@

error_R��J?

learning_rate_1�HG7�}%9H       ��H�	�nT���A*;


total_loss[��@

error_R�5T?

learning_rate_1�HG7 пH       ��H�	��nT���A*;


total_loss�@

error_R�WT?

learning_rate_1�HG7%΅�H       ��H�	@*oT���A*;


total_loss-܈@

error_R��J?

learning_rate_1�HG7��H       ��H�	�roT���A*;


total_loss[k�@

error_R�'Q?

learning_rate_1�HG7;
��H       ��H�	n�oT���A*;


total_loss���@

error_R� A?

learning_rate_1�HG7f�z�H       ��H�	�pT���A*;


total_loss���@

error_R��4?

learning_rate_1�HG7�j;�H       ��H�	BJpT���A*;


total_lossc�@

error_R_�O?

learning_rate_1�HG7ٽ'�H       ��H�	ÏpT���A*;


total_loss8��@

error_R�QZ?

learning_rate_1�HG7^6:�H       ��H�	\�pT���A*;


total_loss�V�@

error_R��H?

learning_rate_1�HG7NxHH       ��H�	�qT���A*;


total_loss΂�@

error_R@�S?

learning_rate_1�HG7g��H       ��H�	�}qT���A*;


total_lossC^�@

error_R�/L?

learning_rate_1�HG7|o&H       ��H�	��qT���A*;


total_loss���@

error_R� J?

learning_rate_1�HG7�)�H       ��H�	
 rT���A*;


total_lossz��@

error_R��`?

learning_rate_1�HG7~aߜH       ��H�	�krT���A*;


total_loss���@

error_R��J?

learning_rate_1�HG7FD��H       ��H�	+�rT���A*;


total_loss��@

error_RxG?

learning_rate_1�HG79)W�H       ��H�	U�rT���A*;


total_lossJ��@

error_R�,G?

learning_rate_1�HG74��H       ��H�	A8sT���A*;


total_lossi-�@

error_R8<F?

learning_rate_1�HG7Ґ%H       ��H�	sT���A*;


total_loss���@

error_R�dS?

learning_rate_1�HG7�,��H       ��H�	��sT���A*;


total_loss=U�@

error_RHC?

learning_rate_1�HG7%]�wH       ��H�	�tT���A*;


total_loss\��@

error_R�~Y?

learning_rate_1�HG7;��.H       ��H�	TWtT���A*;


total_loss���@

error_R�U?

learning_rate_1�HG7q^4H       ��H�	ŝtT���A *;


total_loss�]l@

error_R3FP?

learning_rate_1�HG7�R�H       ��H�	�tT���A!*;


total_loss�w�@

error_R�U?

learning_rate_1�HG7��6CH       ��H�	�8uT���A"*;


total_loss��@

error_R�bL?

learning_rate_1�HG7�#�H       ��H�	S~uT���A#*;


total_loss
�@

error_R�#E?

learning_rate_1�HG7�qDH       ��H�	�uT���A$*;


total_losss��@

error_R�GQ?

learning_rate_1�HG7��H       ��H�	kvT���A%*;


total_lossP	�@

error_R3�>?

learning_rate_1�HG7l"�H       ��H�	MvT���A&*;


total_lossf`@

error_R�{M?

learning_rate_1�HG7p��H       ��H�	�vT���A'*;


total_loss�u�@

error_Rz�Y?

learning_rate_1�HG7M���H       ��H�	7�vT���A(*;


total_loss�:�@

error_RO�D?

learning_rate_1�HG7� �H       ��H�	�wT���A)*;


total_loss���@

error_R{�Q?

learning_rate_1�HG7���H       ��H�	�awT���A**;


total_lossj��@

error_RZL?

learning_rate_1�HG7]�r�H       ��H�	J�wT���A+*;


total_loss<�@

error_R��B?

learning_rate_1�HG7;jXH       ��H�	5�wT���A,*;


total_lossXgx@

error_RiCE?

learning_rate_1�HG7����H       ��H�	�7xT���A-*;


total_loss���@

error_R@A?

learning_rate_1�HG7LD��H       ��H�	�zxT���A.*;


total_loss �}@

error_R:&Y?

learning_rate_1�HG7�sH�H       ��H�	 �xT���A/*;


total_loss���@

error_R��P?

learning_rate_1�HG7���iH       ��H�	yT���A0*;


total_lossO�@

error_R�yP?

learning_rate_1�HG7��}H       ��H�	@OyT���A1*;


total_loss���@

error_Rw�R?

learning_rate_1�HG7et6�H       ��H�	B�yT���A2*;


total_loss��q@

error_R�[T?

learning_rate_1�HG7i��H       ��H�	��yT���A3*;


total_lossO=�@

error_R7Zr?

learning_rate_1�HG7
2H       ��H�	%zT���A4*;


total_loss,�@

error_R&3N?

learning_rate_1�HG7ˁ/H       ��H�	�mzT���A5*;


total_loss��@

error_R�L?

learning_rate_1�HG7��q
H       ��H�	��zT���A6*;


total_loss�ŷ@

error_RJK?

learning_rate_1�HG7�R�!H       ��H�	{T���A7*;


total_lossfа@

error_Ri~J?

learning_rate_1�HG7�1J�H       ��H�	&c{T���A8*;


total_loss�Q�@

error_R��H?

learning_rate_1�HG7I��6H       ��H�	W�{T���A9*;


total_loss��@

error_R�wH?

learning_rate_1�HG7���H       ��H�	j�{T���A:*;


total_loss�@

error_R��O?

learning_rate_1�HG7�H       ��H�	�2|T���A;*;


total_loss/��@

error_R�tW?

learning_rate_1�HG7��c�H       ��H�	?x|T���A<*;


total_lossҍ�@

error_R��H?

learning_rate_1�HG70m�MH       ��H�	��|T���A=*;


total_loss���@

error_R�c_?

learning_rate_1�HG7(�f�H       ��H�	|}T���A>*;


total_loss��@

error_R�eI?

learning_rate_1�HG7v�-HH       ��H�	�H}T���A?*;


total_lossE�@

error_R&�R?

learning_rate_1�HG7.�LH       ��H�	��}T���A@*;


total_loss*��@

error_RP?

learning_rate_1�HG7�D�UH       ��H�	4�}T���AA*;


total_loss7��@

error_R.UQ?

learning_rate_1�HG7�F&9H       ��H�	�>~T���AB*;


total_loss��@

error_Rȯ@?

learning_rate_1�HG7m���H       ��H�	Ղ~T���AC*;


total_loss�J�@

error_Rx�^?

learning_rate_1�HG7tC/=H       ��H�	O�~T���AD*;


total_lossj�@

error_RT�_?

learning_rate_1�HG7��F�H       ��H�	�T���AE*;


total_loss0�@

error_Rc.?

learning_rate_1�HG7K� H       ��H�	�RT���AF*;


total_losss��@

error_R�C?

learning_rate_1�HG7�Z}�H       ��H�	җT���AG*;


total_loss��)A

error_R�r,?

learning_rate_1�HG7-'�aH       ��H�	��T���AH*;


total_loss���@

error_R�Q?

learning_rate_1�HG7�y��H       ��H�	"�T���AI*;


total_loss_�A

error_R�@?

learning_rate_1�HG7�W�H       ��H�	|r�T���AJ*;


total_loss���@

error_Rx�??

learning_rate_1�HG7Q�VlH       ��H�	��T���AK*;


total_lossNX�@

error_RMEI?

learning_rate_1�HG7چTH       ��H�	��T���AL*;


total_loss�հ@

error_R�XQ?

learning_rate_1�HG7tb�H       ��H�	ZP�T���AM*;


total_loss�wA

error_Rq�S?

learning_rate_1�HG7|�-uH       ��H�	ԙ�T���AN*;


total_loss��@

error_R�EV?

learning_rate_1�HG7v�H       ��H�	��T���AO*;


total_loss��NA

error_RQ`f?

learning_rate_1�HG7P��H       ��H�	{)�T���AP*;


total_lossi��@

error_R��S?

learning_rate_1�HG7o�"�H       ��H�	jm�T���AQ*;


total_lossf��@

error_R}6<?

learning_rate_1�HG7���H       ��H�	���T���AR*;


total_loss$��@

error_R�@[?

learning_rate_1�HG7�.
!H       ��H�	h�T���AS*;


total_lossxĠ@

error_Rf�W?

learning_rate_1�HG7a��>H       ��H�	J6�T���AT*;


total_loss���@

error_R�iC?

learning_rate_1�HG7�ڢ�H       ��H�	6x�T���AU*;


total_loss?�@

error_R1�A?

learning_rate_1�HG7��%�H       ��H�	p��T���AV*;


total_loss�M�@

error_R�^L?

learning_rate_1�HG71
KH       ��H�	��T���AW*;


total_lossS�v@

error_R�jL?

learning_rate_1�HG72@�"H       ��H�	B�T���AX*;


total_loss��@

error_R�P?

learning_rate_1�HG7qc��H       ��H�	��T���AY*;


total_lossx2�@

error_R��Y?

learning_rate_1�HG7)��H       ��H�	�ɄT���AZ*;


total_loss =�@

error_RmN?

learning_rate_1�HG7OI�H       ��H�	U�T���A[*;


total_loss���@

error_R�S?

learning_rate_1�HG78�"eH       ��H�	P�T���A\*;


total_loss2@�@

error_R}�H?

learning_rate_1�HG7|�H       ��H�	ה�T���A]*;


total_loss �@

error_Ry:?

learning_rate_1�HG7�	��H       ��H�	��T���A^*;


total_lossx�@

error_R=�F?

learning_rate_1�HG7�(UH       ��H�	+�T���A_*;


total_loss��A

error_R��C?

learning_rate_1�HG7*�;H       ��H�	lq�T���A`*;


total_lossP�@

error_R)�S?

learning_rate_1�HG71M�H       ��H�	��T���Aa*;


total_loss�A

error_R�N?

learning_rate_1�HG7C�H       ��H�	���T���Ab*;


total_loss�(�@

error_R�P?

learning_rate_1�HG7�
Y�H       ��H�	�@�T���Ac*;


total_loss� @

error_R��M?

learning_rate_1�HG7֨�H       ��H�	���T���Ad*;


total_loss)zA

error_RH7H?

learning_rate_1�HG7y<��H       ��H�	ƇT���Ae*;


total_loss؜�@

error_R��J?

learning_rate_1�HG7��*�H       ��H�	(�T���Af*;


total_loss:��@

error_R�$T?

learning_rate_1�HG7C�M�H       ��H�	K�T���Ag*;


total_loss���@

error_R�U?

learning_rate_1�HG7I��VH       ��H�	��T���Ah*;


total_loss	GA

error_RRJ?

learning_rate_1�HG7��)H       ��H�	EψT���Ai*;


total_loss�Ҳ@

error_R�'J?

learning_rate_1�HG7����H       ��H�	S�T���Aj*;


total_loss�"�@

error_R�tJ?

learning_rate_1�HG7����H       ��H�	�[�T���Ak*;


total_loss:Q�@

error_RshN?

learning_rate_1�HG78c*H       ��H�	��T���Al*;


total_loss�"�@

error_Rh\`?

learning_rate_1�HG7
x�1H       ��H�	��T���Am*;


total_loss��	A

error_R��P?

learning_rate_1�HG7���H       ��H�	\2�T���An*;


total_lossnҜ@

error_R�hK?

learning_rate_1�HG7��=H       ��H�	0v�T���Ao*;


total_loss��@

error_RCwP?

learning_rate_1�HG7�4�aH       ��H�	'ϊT���Ap*;


total_loss��@

error_R��B?

learning_rate_1�HG7q6��H       ��H�	6*�T���Aq*;


total_loss�f�@

error_R!�R?

learning_rate_1�HG7�!}@H       ��H�	�u�T���Ar*;


total_loss���@

error_R�@?

learning_rate_1�HG7���H       ��H�	�؋T���As*;


total_loss�/�@

error_R�K?

learning_rate_1�HG7 �#H       ��H�	&�T���At*;


total_lossS��@

error_R��@?

learning_rate_1�HG7�.�H       ��H�	p�T���Au*;


total_loss�\�@

error_R��\?

learning_rate_1�HG7��H       ��H�	s��T���Av*;


total_lossz��@

error_REe?

learning_rate_1�HG7ʶ~�H       ��H�		�T���Aw*;


total_loss:�A

error_R�=C?

learning_rate_1�HG7�>��H       ��H�	�R�T���Ax*;


total_lossg�@

error_RF�6?

learning_rate_1�HG7�s|�H       ��H�	���T���Ay*;


total_loss
�!A

error_R!#@?

learning_rate_1�HG7���2H       ��H�	�T���Az*;


total_loss���@

error_RllO?

learning_rate_1�HG7�A�qH       ��H�	\)�T���A{*;


total_loss8��@

error_R�Q?

learning_rate_1�HG7���kH       ��H�	)r�T���A|*;


total_loss���@

error_R��O?

learning_rate_1�HG7�|}H       ��H�	׶�T���A}*;


total_lossS �@

error_Rn�B?

learning_rate_1�HG7���H       ��H�	���T���A~*;


total_loss�v@

error_R��A?

learning_rate_1�HG7Q�^�H       ��H�	�@�T���A*;


total_loss���@

error_RlSM?

learning_rate_1�HG7hY�I       6%�	��T���A�*;


total_lossX��@

error_R�Ca?

learning_rate_1�HG7���I       6%�	)ˏT���A�*;


total_lossQ:�@

error_R��\?

learning_rate_1�HG7�}S�I       6%�	��T���A�*;


total_lossS��@

error_R:8V?

learning_rate_1�HG7�.aI       6%�	�O�T���A�*;


total_loss�*�@

error_R��U?

learning_rate_1�HG78�x`I       6%�	o��T���A�*;


total_loss���@

error_R�J?

learning_rate_1�HG7��ЃI       6%�	ېT���A�*;


total_loss*G�@

error_R�tW?

learning_rate_1�HG7��I       6%�	k�T���A�*;


total_loss��@

error_R͍M?

learning_rate_1�HG7Nԅ|I       6%�	�^�T���A�*;


total_loss ��@

error_R�E?

learning_rate_1�HG7 G��I       6%�	ޢ�T���A�*;


total_loss���@

error_Re�R?

learning_rate_1�HG7Ơ��I       6%�	2�T���A�*;


total_loss��@

error_RT�F?

learning_rate_1�HG71��9I       6%�	�(�T���A�*;


total_lossw�@

error_R6�M?

learning_rate_1�HG7��҈I       6%�	�k�T���A�*;


total_loss��@

error_RZ=S?

learning_rate_1�HG7��T�I       6%�	���T���A�*;


total_loss�u�@

error_R��K?

learning_rate_1�HG7��<�I       6%�	3�T���A�*;


total_losscn�@

error_Rz6E?

learning_rate_1�HG7�l�;I       6%�	�4�T���A�*;


total_loss� �@

error_RaP?

learning_rate_1�HG7��]�I       6%�	�v�T���A�*;


total_lossZ�@

error_R�[?

learning_rate_1�HG7i�	I       6%�	(��T���A�*;


total_loss�@

error_RA^?

learning_rate_1�HG76��I       6%�	���T���A�*;


total_loss��@

error_R1/R?

learning_rate_1�HG7
DeI       6%�	y?�T���A�*;


total_loss� �@

error_RRZT?

learning_rate_1�HG7L�sI       6%�	႔T���A�*;


total_loss�x�@

error_R� M?

learning_rate_1�HG7:7N�I       6%�	�ƔT���A�*;


total_loss3G�@

error_R��U?

learning_rate_1�HG7x�KjI       6%�	l
�T���A�*;


total_loss���@

error_R��U?

learning_rate_1�HG7b��I       6%�	�N�T���A�*;


total_lossFa�@

error_R;~J?

learning_rate_1�HG7�GKI       6%�	)��T���A�*;


total_loss�c�@

error_R�>F?

learning_rate_1�HG7��z�I       6%�	�֕T���A�*;


total_loss	��@

error_R�t_?

learning_rate_1�HG7��.zI       6%�	6�T���A�*;


total_loss/�@

error_RvR?

learning_rate_1�HG7���[I       6%�	�^�T���A�*;


total_loss�|�@

error_R4�]?

learning_rate_1�HG7O)sI       6%�	���T���A�*;


total_lossx��@

error_R�Z?

learning_rate_1�HG7�<MI       6%�	��T���A�*;


total_loss���@

error_R�6T?

learning_rate_1�HG7�3�I       6%�	h2�T���A�*;


total_loss�#�@

error_RԷ5?

learning_rate_1�HG7hHM�I       6%�	�u�T���A�*;


total_loss���@

error_R��W?

learning_rate_1�HG7�G,I       6%�	R��T���A�*;


total_loss1s@

error_R�>N?

learning_rate_1�HG7np�xI       6%�	���T���A�*;


total_loss���@

error_R��O?

learning_rate_1�HG7�ϛ7I       6%�	`<�T���A�*;


total_lossh7}@

error_R��\?

learning_rate_1�HG7e�\I       6%�	��T���A�*;


total_loss�W
A

error_RL�L?

learning_rate_1�HG7`�X�I       6%�	�ØT���A�*;


total_lossG�@

error_R
�M?

learning_rate_1�HG7�>I       6%�	4
�T���A�*;


total_loss4"�@

error_R��G?

learning_rate_1�HG7O�`LI       6%�	�R�T���A�*;


total_loss�@

error_R�`Y?

learning_rate_1�HG7W�=cI       6%�	ᙙT���A�*;


total_loss '�@

error_R@�[?

learning_rate_1�HG7O�I       6%�	���T���A�*;


total_loss�֙@

error_R��K?

learning_rate_1�HG7�ԁ�I       6%�	�'�T���A�*;


total_loss�ɩ@

error_R%$f?

learning_rate_1�HG7�<�I       6%�	�m�T���A�*;


total_loss81�@

error_R�i?

learning_rate_1�HG7`�uI       6%�	�̚T���A�*;


total_lossn�@

error_R��@?

learning_rate_1�HG7��nfI       6%�	�$�T���A�*;


total_loss��@

error_R��Q?

learning_rate_1�HG7��JkI       6%�	�i�T���A�*;


total_loss�Z�@

error_R}&T?

learning_rate_1�HG7��)I       6%�	(��T���A�*;


total_loss	�@

error_R�H?

learning_rate_1�HG74W`�I       6%�	��T���A�*;


total_loss�Z�@

error_R��@?

learning_rate_1�HG7Z�B I       6%�	v;�T���A�*;


total_lossq��@

error_R$�I?

learning_rate_1�HG7�+I       6%�	��T���A�*;


total_lossz�@

error_R
}L?

learning_rate_1�HG7G��I       6%�	�ޜT���A�*;


total_loss?w�@

error_R��6?

learning_rate_1�HG7'�+I       6%�	�'�T���A�*;


total_loss���@

error_R�V?

learning_rate_1�HG7���I       6%�	�m�T���A�*;


total_loss=|@

error_Rl�C?

learning_rate_1�HG7�\�mI       6%�	K̝T���A�*;


total_loss<:n@

error_R�[?

learning_rate_1�HG7/� �I       6%�	o�T���A�*;


total_loss�)�@

error_R�Q?

learning_rate_1�HG7v��&I       6%�	fV�T���A�*;


total_loss��n@

error_R��N?

learning_rate_1�HG7�A�I       6%�	w��T���A�*;


total_loss���@

error_R1�I?

learning_rate_1�HG7��tI       6%�	aޞT���A�*;


total_loss߽�@

error_R�,E?

learning_rate_1�HG7o%��I       6%�	�"�T���A�*;


total_loss3ە@

error_RC�P?

learning_rate_1�HG7/凇I       6%�	�l�T���A�*;


total_loss.��@

error_R�1R?

learning_rate_1�HG7}Y'I       6%�	v��T���A�*;


total_loss�<�@

error_R�eB?

learning_rate_1�HG7����I       6%�	���T���A�*;


total_lossOA

error_R�w]?

learning_rate_1�HG7㟌�I       6%�	�B�T���A�*;


total_loss1|�@

error_R�GU?

learning_rate_1�HG7:�,�I       6%�	���T���A�*;


total_lossl�@

error_R��=?

learning_rate_1�HG7�꘺I       6%�	*ʠT���A�*;


total_loss��@

error_R,/V?

learning_rate_1�HG7�FU�I       6%�	;�T���A�*;


total_loss�^�@

error_R��S?

learning_rate_1�HG7Y��I       6%�	�R�T���A�*;


total_loss��@

error_R�N;?

learning_rate_1�HG7v^�{I       6%�	���T���A�*;


total_loss�̲@

error_R#�W?

learning_rate_1�HG7C+��I       6%�	0�T���A�*;


total_loss"�@

error_R
�;?

learning_rate_1�HG7.��	I       6%�	�.�T���A�*;


total_losscZ�@

error_R�T?

learning_rate_1�HG7��[�I       6%�	�w�T���A�*;


total_loss��@

error_R�WH?

learning_rate_1�HG78dy�I       6%�	ǿ�T���A�*;


total_loss��9A

error_Rl�e?

learning_rate_1�HG7&E�tI       6%�	�T���A�*;


total_lossu1�@

error_R�bY?

learning_rate_1�HG7�P�I       6%�	G�T���A�*;


total_lossTM�@

error_R$zW?

learning_rate_1�HG7�c�I       6%�	��T���A�*;


total_lossZ��@

error_R�	j?

learning_rate_1�HG7��3I       6%�	�УT���A�*;


total_loss�b�@

error_R��G?

learning_rate_1�HG7�9��I       6%�	=�T���A�*;


total_loss�@

error_Rn�S?

learning_rate_1�HG7V�zI       6%�	�X�T���A�*;


total_loss�¼@

error_R�TR?

learning_rate_1�HG7 ЗI       6%�	���T���A�*;


total_loss�b�@

error_R�(O?

learning_rate_1�HG7o�*VI       6%�	X�T���A�*;


total_loss��A

error_R�dl?

learning_rate_1�HG7pPLuI       6%�	)�T���A�*;


total_lossۉ�@

error_R<�X?

learning_rate_1�HG7�qBI       6%�	�m�T���A�*;


total_loss ��@

error_R�"L?

learning_rate_1�HG7�?�I       6%�	9��T���A�*;


total_loss��@

error_R��[?

learning_rate_1�HG7��oZI       6%�	��T���A�*;


total_lossqR�@

error_R)�R?

learning_rate_1�HG7�7B�I       6%�	A�T���A�*;


total_loss�<�@

error_R[?

learning_rate_1�HG7a��I       6%�	,��T���A�*;


total_loss��@

error_RM U?

learning_rate_1�HG7R�V8I       6%�	Q˦T���A�*;


total_loss(�@

error_R�U?

learning_rate_1�HG7�N�VI       6%�	 �T���A�*;


total_loss}��@

error_R$�\?

learning_rate_1�HG7���I       6%�	�^�T���A�*;


total_loss��@

error_R�~E?

learning_rate_1�HG7
��I       6%�	��T���A�*;


total_loss�؝@

error_R5Z?

learning_rate_1�HG71�kI       6%�	p�T���A�*;


total_loss}�A

error_Rv6I?

learning_rate_1�HG7�~��I       6%�	�*�T���A�*;


total_lossMӧ@

error_RA?

learning_rate_1�HG7�U I       6%�	<j�T���A�*;


total_loss��A

error_Rn�X?

learning_rate_1�HG7�K6pI       6%�	U��T���A�*;


total_loss�A

error_R�G?

learning_rate_1�HG76�I       6%�	�T���A�*;


total_lossj��@

error_RX?

learning_rate_1�HG7��L�I       6%�	�9�T���A�*;


total_lossi��@

error_Rj
P?

learning_rate_1�HG7��$I       6%�	�z�T���A�*;


total_loss+{�@

error_RfO?

learning_rate_1�HG7��iI       6%�	���T���A�*;


total_lossD�n@

error_RW�F?

learning_rate_1�HG7@��I       6%�	\�T���A�*;


total_lossh��@

error_R��\?

learning_rate_1�HG7�l��I       6%�	�D�T���A�*;


total_loss�ħ@

error_R.[?

learning_rate_1�HG78��fI       6%�	��T���A�*;


total_lossV �@

error_RFp>?

learning_rate_1�HG7w�k�I       6%�	�T���A�*;


total_loss*э@

error_RV�J?

learning_rate_1�HG7���I       6%�	�=�T���A�*;


total_loss���@

error_R=1T?

learning_rate_1�HG7��PI       6%�	熫T���A�*;


total_loss��@

error_RۊX?

learning_rate_1�HG7g�!dI       6%�	hϫT���A�*;


total_lossA��@

error_R��E?

learning_rate_1�HG7��� I       6%�	7�T���A�*;


total_lossm��@

error_R�<9?

learning_rate_1�HG7�*8I       6%�	3h�T���A�*;


total_loss�1�@

error_R6�F?

learning_rate_1�HG7E���I       6%�	���T���A�*;


total_lossm�@

error_RWwJ?

learning_rate_1�HG7�=s&I       6%�	���T���A�*;


total_lossw��@

error_R2I?

learning_rate_1�HG7V���I       6%�	)<�T���A�*;


total_loss@�A

error_RZ�Q?

learning_rate_1�HG7m�T�I       6%�	=��T���A�*;


total_loss�]�@

error_RR�R?

learning_rate_1�HG7Ar�I       6%�	�ĭT���A�*;


total_loss	b�@

error_R� K?

learning_rate_1�HG7�͑I       6%�	�
�T���A�*;


total_loss��x@

error_R��=?

learning_rate_1�HG7,;d.I       6%�	�O�T���A�*;


total_lossݲ�@

error_R=E?

learning_rate_1�HG7�9I       6%�	/��T���A�*;


total_loss@�@

error_RC�M?

learning_rate_1�HG7%I�tI       6%�	�ٮT���A�*;


total_loss6Z�@

error_R,�Z?

learning_rate_1�HG7�b�tI       6%�	 �T���A�*;


total_loss1�@

error_RJ�??

learning_rate_1�HG7�e��I       6%�	�a�T���A�*;


total_lossg�@

error_R��Y?

learning_rate_1�HG70,��I       6%�	��T���A�*;


total_loss�4�@

error_R�B?

learning_rate_1�HG7|;sI       6%�	��T���A�*;


total_loss���@

error_R҄Q?

learning_rate_1�HG7�h�I       6%�	p0�T���A�*;


total_lossѥ�@

error_R�U?

learning_rate_1�HG7'�r�I       6%�	xu�T���A�*;


total_loss��@

error_RYK?

learning_rate_1�HG7@p��I       6%�	D��T���A�*;


total_loss�@

error_R#~\?

learning_rate_1�HG7c�!I       6%�	���T���A�*;


total_loss2��@

error_RN�M?

learning_rate_1�HG7�T�sI       6%�	 @�T���A�*;


total_loss�]�@

error_R��L?

learning_rate_1�HG7WE�I       6%�	��T���A�*;


total_loss�k�@

error_R��K?

learning_rate_1�HG7brI       6%�	�ɱT���A�*;


total_lossx^�@

error_R��I?

learning_rate_1�HG7~�rI       6%�	1�T���A�*;


total_loss�٦@

error_RV�I?

learning_rate_1�HG7ã�I       6%�	<u�T���A�*;


total_loss��@

error_R�R?

learning_rate_1�HG7���I       6%�	�T���A�*;


total_lossX��@

error_RO�N?

learning_rate_1�HG7��ҳI       6%�	5��T���A�*;


total_losst��@

error_R�Y?

learning_rate_1�HG7R�v�I       6%�	�A�T���A�*;


total_loss�ײ@

error_R��M?

learning_rate_1�HG7�{;nI       6%�	/��T���A�*;


total_loss���@

error_RW�P?

learning_rate_1�HG7�%яI       6%�	�ȳT���A�*;


total_loss&��@

error_R}P?

learning_rate_1�HG7u_)�I       6%�	l�T���A�*;


total_loss�A

error_R��L?

learning_rate_1�HG7OW}�I       6%�	T�T���A�*;


total_loss�w]@

error_R�W?

learning_rate_1�HG7�b�I       6%�	R��T���A�*;


total_loss$�@

error_R�^H?

learning_rate_1�HG7�V�I       6%�	<ܴT���A�*;


total_loss��@

error_RN.Q?

learning_rate_1�HG7�脑I       6%�	�%�T���A�*;


total_loss��@

error_R��T?

learning_rate_1�HG7��a�I       6%�	j�T���A�*;


total_loss�{@

error_Rv�C?

learning_rate_1�HG7\�,�I       6%�	���T���A�*;


total_lossCo�@

error_Rw�J?

learning_rate_1�HG7*==I       6%�	w��T���A�*;


total_loss�NV@

error_R q<?

learning_rate_1�HG7�g qI       6%�	�=�T���A�*;


total_loss)��@

error_R	#n?

learning_rate_1�HG7��1I       6%�	m��T���A�*;


total_loss@k�@

error_RsxJ?

learning_rate_1�HG7uB�I       6%�	���T���A�*;


total_loss]��@

error_R��*?

learning_rate_1�HG7G�	$I       6%�	��T���A�*;


total_loss��@

error_R�R?

learning_rate_1�HG7+�UI       6%�	�F�T���A�*;


total_losss~�@

error_R>D?

learning_rate_1�HG74��I       6%�	���T���A�*;


total_lossZ>�@

error_R�a2?

learning_rate_1�HG7�>@PI       6%�	8зT���A�*;


total_loss	<A

error_Re�_?

learning_rate_1�HG7�YQI       6%�	��T���A�*;


total_loss���@

error_R��U?

learning_rate_1�HG7�a�I       6%�	�Y�T���A�*;


total_loss**�@

error_RܖK?

learning_rate_1�HG7�:�~I       6%�	���T���A�*;


total_lossxM�@

error_R�YM?

learning_rate_1�HG78LEI       6%�	��T���A�*;


total_lossb�@

error_R��k?

learning_rate_1�HG7�yI       6%�	.�T���A�*;


total_loss`�@

error_R�$K?

learning_rate_1�HG7�d�@I       6%�	.w�T���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7:�՚I       6%�	�¹T���A�*;


total_loss��@

error_RdW?

learning_rate_1�HG7���I       6%�	�T���A�*;


total_loss��@

error_R� X?

learning_rate_1�HG7W�lI       6%�	6S�T���A�*;


total_loss�@

error_R:�E?

learning_rate_1�HG7����I       6%�	���T���A�*;


total_loss���@

error_R�([?

learning_rate_1�HG7mI#AI       6%�	m��T���A�*;


total_loss�f�@

error_R\qV?

learning_rate_1�HG7��VI       6%�	�B�T���A�*;


total_loss���@

error_R�	R?

learning_rate_1�HG7��	I       6%�	I��T���A�*;


total_loss���@

error_RR^C?

learning_rate_1�HG7=��I       6%�	ʻT���A�*;


total_losso��@

error_Rs;?

learning_rate_1�HG7ۡ��I       6%�	%�T���A�*;


total_loss��@

error_R�c?

learning_rate_1�HG7a�I       6%�	[W�T���A�*;


total_loss���@

error_R8xU?

learning_rate_1�HG73t�I       6%�	؞�T���A�*;


total_loss���@

error_R�T?

learning_rate_1�HG7��+I       6%�	*�T���A�*;


total_loss��@

error_R�yd?

learning_rate_1�HG7ܐI       6%�	1�T���A�*;


total_loss���@

error_R�QM?

learning_rate_1�HG7�(��I       6%�	�z�T���A�*;


total_loss殮@

error_R��O?

learning_rate_1�HG7��I       6%�	���T���A�*;


total_loss(�@

error_R7�C?

learning_rate_1�HG7L~ЀI       6%�	0�T���A�*;


total_loss�1�@

error_R��L?

learning_rate_1�HG7�"�I       6%�	ZJ�T���A�*;


total_lossƗ�@

error_R)�<?

learning_rate_1�HG7�?lUI       6%�	A��T���A�*;


total_loss{u�@

error_R��G?

learning_rate_1�HG7ʅ�WI       6%�	�ӾT���A�*;


total_loss�P�@

error_R��L?

learning_rate_1�HG7��e-I       6%�	�T���A�*;


total_loss���@

error_R��R?

learning_rate_1�HG7���fI       6%�	�Y�T���A�*;


total_loss�3�@

error_Rz4Y?

learning_rate_1�HG7)��PI       6%�	'��T���A�*;


total_loss�=�@

error_R��V?

learning_rate_1�HG7����I       6%�	:�T���A�*;


total_loss�T�@

error_Rd@D?

learning_rate_1�HG7b��3I       6%�	K*�T���A�*;


total_losst�@

error_R;�Y?

learning_rate_1�HG7�b��I       6%�	 o�T���A�*;


total_loss��@

error_R=T?

learning_rate_1�HG7rUd�I       6%�	���T���A�*;


total_loss3*�@

error_R�cA?

learning_rate_1�HG7��l�I       6%�	���T���A�*;


total_loss���@

error_R�dJ?

learning_rate_1�HG7�D,�I       6%�	�6�T���A�*;


total_loss�{�@

error_R�ka?

learning_rate_1�HG7`.�I       6%�	�w�T���A�*;


total_loss%�@

error_R]9V?

learning_rate_1�HG7[�6�I       6%�	��T���A�*;


total_loss­A

error_R�X?

learning_rate_1�HG7��I       6%�	��T���A�*;


total_loss풵@

error_R�[?

learning_rate_1�HG7ECyxI       6%�	�C�T���A�*;


total_loss�e�@

error_R�+e?

learning_rate_1�HG7Z��I       6%�	U��T���A�*;


total_loss4� A

error_R�F^?

learning_rate_1�HG7P�JI       6%�	;��T���A�*;


total_loss�w�@

error_RO??

learning_rate_1�HG7y�_#I       6%�	�T���A�*;


total_loss�4�@

error_R��@?

learning_rate_1�HG7f9��I       6%�	�T�T���A�*;


total_loss�o@

error_R��P?

learning_rate_1�HG7t��I       6%�	'��T���A�*;


total_loss�/�@

error_RSO?

learning_rate_1�HG7d9��I       6%�	0��T���A�*;


total_loss ��@

error_R��B?

learning_rate_1�HG7����I       6%�	r�T���A�*;


total_loss��@

error_R,~R?

learning_rate_1�HG7��nI       6%�	�^�T���A�*;


total_loss�H�@

error_R=e=?

learning_rate_1�HG7F���I       6%�	���T���A�*;


total_lossɯ�@

error_R��T?

learning_rate_1�HG7N;I       6%�	��T���A�*;


total_lossda�@

error_RO�P?

learning_rate_1�HG7�ji�I       6%�	[.�T���A�*;


total_lossD��@

error_R�;<?

learning_rate_1�HG7�J��I       6%�	y��T���A�*;


total_loss���@

error_RC�S?

learning_rate_1�HG7D�V�I       6%�	O��T���A�*;


total_loss�ڵ@

error_R��J?

learning_rate_1�HG7�@+�I       6%�	�
�T���A�*;


total_loss#��@

error_R&xF?

learning_rate_1�HG7`�k�I       6%�	>N�T���A�*;


total_loss�
�@

error_RXX?

learning_rate_1�HG7n�Y�I       6%�	#��T���A�*;


total_loss=ǆ@

error_R)-V?

learning_rate_1�HG7 :/�I       6%�	���T���A�*;


total_lossM��@

error_RlUI?

learning_rate_1�HG72��I       6%�	�$�T���A�*;


total_loss��@

error_R��V?

learning_rate_1�HG7��$I       6%�	kk�T���A�*;


total_lossÔ�@

error_R�0T?

learning_rate_1�HG7Y�D�I       6%�	���T���A�*;


total_loss��@

error_RC)V?

learning_rate_1�HG7�;�5I       6%�	)��T���A�*;


total_loss���@

error_Rr�O?

learning_rate_1�HG7�רmI       6%�	O>�T���A�*;


total_loss���@

error_RԖT?

learning_rate_1�HG7�& I       6%�	>��T���A�*;


total_loss�f@

error_Rl,J?

learning_rate_1�HG79�I       6%�	���T���A�*;


total_loss���@

error_R;�I?

learning_rate_1�HG7���I       6%�	I�T���A�*;


total_loss���@

error_R1�J?

learning_rate_1�HG7j�I       6%�	�E�T���A�*;


total_lossW�@

error_RH�S?

learning_rate_1�HG7=GI       6%�	���T���A�*;


total_loss1��@

error_R��c?

learning_rate_1�HG7�ĵI       6%�	���T���A�*;


total_lossz��@

error_REuW?

learning_rate_1�HG7�v�I       6%�	��T���A�*;


total_loss���@

error_R_R?

learning_rate_1�HG7��I       6%�	�V�T���A�*;


total_loss!E�@

error_ReI?

learning_rate_1�HG7�R��I       6%�	��T���A�*;


total_loss���@

error_R_*P?

learning_rate_1�HG7��>�I       6%�	��T���A�*;


total_loss��@

error_R��M?

learning_rate_1�HG7�b��I       6%�	QO�T���A�*;


total_lossaG�@

error_Rz�T?

learning_rate_1�HG7	нI       6%�	��T���A�*;


total_loss ��@

error_R)�C?

learning_rate_1�HG7e�L(I       6%�	7��T���A�*;


total_lossZU�@

error_Rvb?

learning_rate_1�HG7BيI       6%�	�&�T���A�*;


total_losso��@

error_R	�N?

learning_rate_1�HG7w/z�I       6%�	�h�T���A�*;


total_lossl�@

error_R��H?

learning_rate_1�HG7I���I       6%�	���T���A�*;


total_loss��l@

error_R{�W?

learning_rate_1�HG7".I       6%�	���T���A�*;


total_loss��@

error_R1L?

learning_rate_1�HG7;`� I       6%�	EP�T���A�*;


total_lossZ��@

error_R�m<?

learning_rate_1�HG7�3HI       6%�	9��T���A�*;


total_loss��@

error_Rz)9?

learning_rate_1�HG7���eI       6%�	c��T���A�*;


total_lossӼ�@

error_R� G?

learning_rate_1�HG7 O��I       6%�	�6�T���A�*;


total_loss{f�@

error_RO\C?

learning_rate_1�HG7�W�I       6%�	}�T���A�*;


total_loss�@

error_R�J?

learning_rate_1�HG7�tDI       6%�	E��T���A�*;


total_losstմ@

error_R{�U?

learning_rate_1�HG7�ͩ�I       6%�	�T���A�*;


total_loss:�o@

error_R�'[?

learning_rate_1�HG7 ���I       6%�	#Y�T���A�*;


total_loss��@

error_Rc]?

learning_rate_1�HG7bf�lI       6%�	k��T���A�*;


total_loss@

error_Rq�1?

learning_rate_1�HG79�>�I       6%�	9��T���A�*;


total_loss4޲@

error_R��O?

learning_rate_1�HG7��I       6%�	�3�T���A�*;


total_loss��	A

error_R@�R?

learning_rate_1�HG7���I       6%�	���T���A�*;


total_loss���@

error_R=m7?

learning_rate_1�HG7�/�kI       6%�	C��T���A�*;


total_lossf�@

error_Rf�U?

learning_rate_1�HG7JrxI       6%�	D�T���A�*;


total_loss�K�@

error_R@�I?

learning_rate_1�HG7rD��I       6%�	�X�T���A�*;


total_loss���@

error_R�M?

learning_rate_1�HG7�L�$I       6%�	��T���A�*;


total_loss1b�@

error_R�T?

learning_rate_1�HG7,j>SI       6%�	��T���A�*;


total_loss{��@

error_RC]S?

learning_rate_1�HG7��U�I       6%�	B6�T���A�*;


total_loss���@

error_R�T?

learning_rate_1�HG75��I       6%�	�~�T���A�*;


total_loss�P�@

error_R@qH?

learning_rate_1�HG7�I       6%�	4��T���A�*;


total_loss��@

error_RqQb?

learning_rate_1�HG7��iI       6%�	5	�T���A�*;


total_loss���@

error_R�|>?

learning_rate_1�HG7���I       6%�	RM�T���A�*;


total_loss���@

error_R��`?

learning_rate_1�HG7���I       6%�	���T���A�*;


total_loss�?A

error_Rq`M?

learning_rate_1�HG7ܹ@I       6%�	���T���A�*;


total_loss۔�@

error_R��X?

learning_rate_1�HG7���I       6%�	��T���A�*;


total_lossA

error_R:uA?

learning_rate_1�HG7�$�I       6%�	�a�T���A�*;


total_loss��@

error_R82H?

learning_rate_1�HG7+�J^I       6%�	I��T���A�*;


total_loss�Y�@

error_RDH?

learning_rate_1�HG7����I       6%�	��T���A�*;


total_loss8�@

error_R��E?

learning_rate_1�HG7`�V�I       6%�	80�T���A�*;


total_loss�[�@

error_R��P?

learning_rate_1�HG7���(I       6%�	>v�T���A�*;


total_loss \�@

error_R	zZ?

learning_rate_1�HG7C�8�I       6%�	��T���A�*;


total_loss���@

error_R�(O?

learning_rate_1�HG7��N�I       6%�	�T���A�*;


total_loss[E�@

error_R^J?

learning_rate_1�HG7����I       6%�	�R�T���A�*;


total_loss7^�@

error_R�58?

learning_rate_1�HG7?`��I       6%�	��T���A�*;


total_loss[ҫ@

error_RK?

learning_rate_1�HG7NG}XI       6%�	g��T���A�*;


total_loss�A

error_R��J?

learning_rate_1�HG7���I       6%�	,#�T���A�*;


total_loss�t�@

error_R �h?

learning_rate_1�HG7��3I       6%�	�f�T���A�*;


total_loss��A

error_R�\I?

learning_rate_1�HG7�O��I       6%�	���T���A�*;


total_loss���@

error_R\�M?

learning_rate_1�HG7>�I       6%�	n��T���A�*;


total_loss�ȫ@

error_R#�B?

learning_rate_1�HG7Ώ�RI       6%�	r4�T���A�*;


total_lossJq�@

error_R��[?

learning_rate_1�HG7���?I       6%�	Ex�T���A�*;


total_loss�@

error_R�BY?

learning_rate_1�HG7�S�zI       6%�	���T���A�*;


total_loss�7�@

error_R�pH?

learning_rate_1�HG7X���I       6%�	
��T���A�*;


total_loss!��@

error_R=�T?

learning_rate_1�HG7{I       6%�	�C�T���A�*;


total_losse�@

error_RAL?

learning_rate_1�HG7��^�I       6%�	��T���A�*;


total_loss� �@

error_RveU?

learning_rate_1�HG7��qI       6%�	���T���A�*;


total_lossQ|3A

error_R��9?

learning_rate_1�HG7$��HI       6%�	�T���A�*;


total_loss�@

error_RҪZ?

learning_rate_1�HG7_?�I       6%�	�R�T���A�*;


total_lossʁ�@

error_R*�^?

learning_rate_1�HG7W�I       6%�	ٖ�T���A�*;


total_loss��@

error_Rq�X?

learning_rate_1�HG71iF�I       6%�	���T���A�*;


total_loss�@

error_RD�]?

learning_rate_1�HG7�]�SI       6%�	�H�T���A�*;


total_loss��@

error_RmZ[?

learning_rate_1�HG73��I       6%�	`��T���A�*;


total_loss�ڹ@

error_R6a[?

learning_rate_1�HG7�.t�I       6%�	���T���A�*;


total_lossͦ@

error_R{�O?

learning_rate_1�HG7�cI       6%�	��T���A�*;


total_loss���@

error_R:IE?

learning_rate_1�HG7���I       6%�	]�T���A�*;


total_loss�+�@

error_RA]?

learning_rate_1�HG7g1%I       6%�	���T���A�*;


total_loss꺊@

error_R�<?

learning_rate_1�HG7BH I       6%�	���T���A�*;


total_loss83�@

error_R�qO?

learning_rate_1�HG7��ڨI       6%�	%�T���A�*;


total_loss�U�@

error_Rb?

learning_rate_1�HG7���I       6%�	g�T���A�*;


total_loss�s�@

error_R�bF?

learning_rate_1�HG7�U��I       6%�	��T���A�*;


total_lossS/�@

error_R�bZ?

learning_rate_1�HG7�J�-I       6%�	���T���A�*;


total_loss/�s@

error_R�Y?

learning_rate_1�HG7��2I       6%�	<�T���A�*;


total_lossdh�@

error_R6
T?

learning_rate_1�HG7�I       6%�	܇�T���A�*;


total_loss ��@

error_R�R?

learning_rate_1�HG7䬵I       6%�	O��T���A�*;


total_loss�:�@

error_R�|V?

learning_rate_1�HG7!��I       6%�	�T���A�*;


total_loss	��@

error_R�8Z?

learning_rate_1�HG7n�.NI       6%�	�e�T���A�*;


total_loss��@

error_R�DJ?

learning_rate_1�HG7�\�.I       6%�	ɩ�T���A�*;


total_loss�{�@

error_R6�O?

learning_rate_1�HG7-?��I       6%�	?��T���A�*;


total_loss��A

error_R^N?

learning_rate_1�HG7�ېI       6%�	>1�T���A�*;


total_loss鉶@

error_R֥N?

learning_rate_1�HG7�Y{^I       6%�	s�T���A�*;


total_lossZ7�@

error_R��N?

learning_rate_1�HG7M��I       6%�	��T���A�*;


total_loss�
�@

error_R�I?

learning_rate_1�HG7�(�I       6%�	��T���A�*;


total_loss{��@

error_R�JE?

learning_rate_1�HG7e#�WI       6%�	B�T���A�*;


total_loss�n�@

error_R��Z?

learning_rate_1�HG7�l�MI       6%�	��T���A�*;


total_loss��@

error_R�T?

learning_rate_1�HG7�@C�I       6%�	_��T���A�*;


total_loss��
A

error_R
�F?

learning_rate_1�HG7 �J�I       6%�	��T���A�*;


total_loss���@

error_R�>?

learning_rate_1�HG7��F�I       6%�	g]�T���A�*;


total_loss*\�@

error_RZ�L?

learning_rate_1�HG7~��I       6%�	���T���A�*;


total_loss���@

error_R�@?

learning_rate_1�HG7+z|�I       6%�	k��T���A�*;


total_lossN+�@

error_Rf�6?

learning_rate_1�HG7�d� I       6%�	(0�T���A�*;


total_loss߽�@

error_RC�[?

learning_rate_1�HG7��\I       6%�	�t�T���A�*;


total_loss���@

error_RE�9?

learning_rate_1�HG7m�{
I       6%�	���T���A�*;


total_loss�A

error_R��W?

learning_rate_1�HG7>���I       6%�	���T���A�*;


total_loss�3�@

error_R�eC?

learning_rate_1�HG7/�9I       6%�	XF�T���A�*;


total_lossO�@

error_R��J?

learning_rate_1�HG7���I       6%�	���T���A�*;


total_loss���@

error_R�+R?

learning_rate_1�HG7I       6%�	F��T���A�*;


total_loss�#�@

error_R@jK?

learning_rate_1�HG7�f�bI       6%�	��T���A�*;


total_loss�} A

error_R	�\?

learning_rate_1�HG7��I       6%�	�`�T���A�*;


total_loss1��@

error_Rd�@?

learning_rate_1�HG7OԴ�I       6%�	��T���A�*;


total_loss��@

error_Rn_?

learning_rate_1�HG7,���I       6%�	L��T���A�*;


total_lossC �@

error_RO�X?

learning_rate_1�HG7J@u�I       6%�	�+�T���A�*;


total_loss{�@

error_R�dJ?

learning_rate_1�HG7��I       6%�	gn�T���A�*;


total_lossv�@

error_RS?H?

learning_rate_1�HG7�&�I       6%�	��T���A�*;


total_lossA

error_R�$O?

learning_rate_1�HG724I       6%�	���T���A�*;


total_loss c�@

error_R�s_?

learning_rate_1�HG7x~=YI       6%�	�>�T���A�*;


total_loss$,�@

error_R7<;?

learning_rate_1�HG7�!z�I       6%�	/��T���A�*;


total_loss*Q�@

error_R&-R?

learning_rate_1�HG7��I       6%�	��T���A�*;


total_lossv��@

error_R�pI?

learning_rate_1�HG7"�$I       6%�	��T���A�*;


total_loss�X�@

error_R2�E?

learning_rate_1�HG7���:I       6%�	8]�T���A�*;


total_loss�p�@

error_R��J?

learning_rate_1�HG7<�I       6%�	���T���A�*;


total_lossC�@

error_RuS?

learning_rate_1�HG7j�dI       6%�	���T���A�*;


total_loss��A

error_R�"W?

learning_rate_1�HG7G`8�I       6%�	�;�T���A�*;


total_loss��@

error_R�uF?

learning_rate_1�HG7���I       6%�	��T���A�*;


total_loss#&�@

error_R��7?

learning_rate_1�HG7���I       6%�	���T���A�*;


total_lossꀭ@

error_R�c=?

learning_rate_1�HG7�"� I       6%�	��T���A�*;


total_loss�(�@

error_RZT?

learning_rate_1�HG7w�<�I       6%�	3R�T���A�*;


total_loss��@

error_RE�O?

learning_rate_1�HG7=��I       6%�	/��T���A�*;


total_loss�3�@

error_RT�S?

learning_rate_1�HG70D�I       6%�	���T���A�*;


total_lossh+�@

error_R��@?

learning_rate_1�HG7
;A�I       6%�	�D�T���A�*;


total_lossq��@

error_Ri�J?

learning_rate_1�HG7phjI       6%�	ώ�T���A�*;


total_loss�Ԯ@

error_R��Z?

learning_rate_1�HG7���I       6%�	���T���A�*;


total_loss�
�@

error_R�=T?

learning_rate_1�HG7�7~�I       6%�	B�T���A�*;


total_losshS�@

error_R�??

learning_rate_1�HG7��I       6%�	"e�T���A�*;


total_loss(d�@

error_R��9?

learning_rate_1�HG7�z�I       6%�	[��T���A�*;


total_loss)|�@

error_RwI?

learning_rate_1�HG7K���I       6%�	A��T���A�*;


total_loss��@

error_R��9?

learning_rate_1�HG7!JD`I       6%�	�U�T���A�*;


total_loss�R�@

error_Rq�I?

learning_rate_1�HG70.�BI       6%�	��T���A�*;


total_loss��@

error_R$�G?

learning_rate_1�HG79!C�I       6%�	~��T���A�*;


total_lossSˍ@

error_RԺf?

learning_rate_1�HG7��,kI       6%�	�D�T���A�*;


total_loss,�m@

error_RC�Q?

learning_rate_1�HG7��c1I       6%�	X��T���A�*;


total_loss���@

error_R�-R?

learning_rate_1�HG7��'AI       6%�	3��T���A�*;


total_lossV�@

error_R)�5?

learning_rate_1�HG7�R�VI       6%�	j�T���A�*;


total_loss=�@

error_RO1X?

learning_rate_1�HG7��H�I       6%�	�T�T���A�*;


total_loss]�@

error_R�?S?

learning_rate_1�HG7�I       6%�	w��T���A�*;


total_loss�#�@

error_R��G?

learning_rate_1�HG7�c�sI       6%�	���T���A�*;


total_lossI��@

error_R�Y?

learning_rate_1�HG7��r�I       6%�	�&�T���A�*;


total_lossZ:�@

error_R�~d?

learning_rate_1�HG7];TI       6%�	4i�T���A�*;


total_lossMd�@

error_R�v;?

learning_rate_1�HG7���I       6%�	���T���A�*;


total_loss�Ħ@

error_R��F?

learning_rate_1�HG7/�]I       6%�	���T���A�*;


total_lossu�@

error_R)�@?

learning_rate_1�HG7��+�I       6%�	a2�T���A�*;


total_loss��@

error_R�LQ?

learning_rate_1�HG7�(mI       6%�	T|�T���A�*;


total_loss8��@

error_R�h^?

learning_rate_1�HG7#�tpI       6%�	���T���A�*;


total_loss��@

error_RH3O?

learning_rate_1�HG7n%�I       6%�	�
�T���A�*;


total_loss�X�@

error_R*�N?

learning_rate_1�HG7��KI       6%�	�L�T���A�*;


total_loss�qA

error_Rr"Z?

learning_rate_1�HG7��I       6%�	-��T���A�*;


total_loss.��@

error_R�QR?

learning_rate_1�HG7D
I       6%�	���T���A�*;


total_loss7+�@

error_R��M?

learning_rate_1�HG7p���I       6%�	m�T���A�*;


total_loss/`�@

error_R\�Z?

learning_rate_1�HG7���I       6%�	$_�T���A�*;


total_loss���@

error_R��Q?

learning_rate_1�HG7��~�I       6%�	S��T���A�*;


total_loss���@

error_R3V?

learning_rate_1�HG7� I       6%�	���T���A�*;


total_lossiH�@

error_R��@?

learning_rate_1�HG7��.I       6%�	�3�T���A�*;


total_loss4��@

error_R�},?

learning_rate_1�HG7jYt�I       6%�	*y�T���A�*;


total_loss���@

error_R��I?

learning_rate_1�HG77MI       6%�	"��T���A�*;


total_loss�?�@

error_R�`S?

learning_rate_1�HG7��I       6%�	3��T���A�*;


total_loss7��@

error_R:S?

learning_rate_1�HG7`�-�I       6%�	B�T���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7�d0�I       6%�	��T���A�*;


total_loss�w�@

error_R�GK?

learning_rate_1�HG7F���I       6%�	��T���A�*;


total_loss�@

error_R�%Z?

learning_rate_1�HG7��(I       6%�	�
�T���A�*;


total_loss-Ҏ@

error_R �X?

learning_rate_1�HG7,�G�I       6%�	�K�T���A�*;


total_loss8۾@

error_R.�9?

learning_rate_1�HG7\��I       6%�	r��T���A�*;


total_loss�Q�@

error_RN�_?

learning_rate_1�HG7�;�I       6%�	m��T���A�*;


total_loss%n�@

error_R�Y?

learning_rate_1�HG7���I       6%�	��T���A�*;


total_loss���@

error_R�U?

learning_rate_1�HG7��:�I       6%�	)_�T���A�*;


total_loss���@

error_RԇF?

learning_rate_1�HG7�r�I       6%�	ʥ�T���A�*;


total_lossV]�@

error_R�m9?

learning_rate_1�HG7���lI       6%�	���T���A�*;


total_loss���@

error_R��W?

learning_rate_1�HG7��PI       6%�	�3�T���A�*;


total_loss]��@

error_RL�R?

learning_rate_1�HG7����I       6%�	�x�T���A�*;


total_loss�n�@

error_R666?

learning_rate_1�HG7���KI       6%�	��T���A�*;


total_loss��@

error_R��W?

learning_rate_1�HG7F*g4I       6%�	��T���A�*;


total_loss�[@

error_R;HF?

learning_rate_1�HG74=	�I       6%�	I�T���A�*;


total_loss�;�@

error_R�(T?

learning_rate_1�HG7��I       6%�	<��T���A�*;


total_loss�l�@

error_R�-O?

learning_rate_1�HG7.���I       6%�	���T���A�*;


total_loss�=�@

error_R}�@?

learning_rate_1�HG7x��I       6%�	��T���A�*;


total_lossd2�@

error_RC�F?

learning_rate_1�HG7�\�I       6%�	yU�T���A�*;


total_loss���@

error_RoM?

learning_rate_1�HG7�7�=I       6%�	���T���A�*;


total_loss���@

error_R�)J?

learning_rate_1�HG7��
I       6%�	���T���A�*;


total_loss��@

error_R�Rj?

learning_rate_1�HG7��I       6%�	YH�T���A�*;


total_loss���@

error_RW�J?

learning_rate_1�HG7�PϧI       6%�	���T���A�*;


total_loss���@

error_RȪL?

learning_rate_1�HG7ò�I       6%�	f��T���A�*;


total_loss�
A

error_R�R?

learning_rate_1�HG7�
��I       6%�	�!�T���A�*;


total_loss��@

error_R�Q?

learning_rate_1�HG7��tEI       6%�	Oe�T���A�*;


total_lossx��@

error_RMMP?

learning_rate_1�HG7>��(I       6%�	X��T���A�*;


total_loss��D@

error_R;�X?

learning_rate_1�HG7��EI       6%�	���T���A�*;


total_lossV�@

error_R�6?

learning_rate_1�HG7���(I       6%�	�*�T���A�*;


total_loss���@

error_R6�X?

learning_rate_1�HG7��SI       6%�	s�T���A�*;


total_loss��@

error_R�B?

learning_rate_1�HG7�Ƅ�I       6%�	Ͻ�T���A�*;


total_loss%�@

error_R#�C?

learning_rate_1�HG7^܄�I       6%�	��T���A�*;


total_loss�c�@

error_R}�H?

learning_rate_1�HG7�~�I       6%�	�E�T���A�*;


total_loss%e=@

error_R
U?

learning_rate_1�HG7�+�I       6%�	<��T���A�*;


total_loss�-WA

error_R�2O?

learning_rate_1�HG7 /I       6%�	O��T���A�*;


total_loss�� A

error_R�<i?

learning_rate_1�HG7��|�I       6%�	��T���A�*;


total_loss�_�@

error_R�o^?

learning_rate_1�HG7�А`I       6%�	�_�T���A�*;


total_lossQ�@

error_R��G?

learning_rate_1�HG7i��I       6%�	a��T���A�*;


total_loss7#�@

error_R�Q?

learning_rate_1�HG7�:�3I       6%�	���T���A�*;


total_loss؞�@

error_R3I_?

learning_rate_1�HG7Qx��I       6%�	�( U���A�*;


total_loss6��@

error_R�>`?

learning_rate_1�HG7X�W�I       6%�	8n U���A�*;


total_loss��A

error_R{-N?

learning_rate_1�HG7"�5I       6%�	(� U���A�*;


total_loss�)\@

error_R�E?

learning_rate_1�HG7v��$I       6%�	�� U���A�*;


total_loss�2�@

error_REQ?

learning_rate_1�HG7�زI       6%�	�@U���A�*;


total_lossd<�@

error_R��G?

learning_rate_1�HG7�U-�I       6%�	x�U���A�*;


total_lossC��@

error_R#Z?

learning_rate_1�HG7��EI       6%�	.�U���A�*;


total_lossR��@

error_RQ7\?

learning_rate_1�HG7]�x�I       6%�	e	U���A�*;


total_lossV��@

error_R�A?

learning_rate_1�HG7�SCI       6%�	=PU���A�*;


total_loss�{�@

error_R�me?

learning_rate_1�HG7�v�hI       6%�	�U���A�*;


total_lossc`�@

error_R
�??

learning_rate_1�HG7YSe#I       6%�	��U���A�*;


total_loss���@

error_R��W?

learning_rate_1�HG7��I       6%�	U���A�*;


total_loss2��@

error_R.U??

learning_rate_1�HG7���4I       6%�	�[U���A�*;


total_loss��@

error_R_C?

learning_rate_1�HG7Tmw�I       6%�	
�U���A�*;


total_lossZV�@

error_RmpK?

learning_rate_1�HG7���I       6%�	��U���A�*;


total_lossW��@

error_R�4;?

learning_rate_1�HG7�Я�I       6%�	h1U���A�*;


total_loss���@

error_R�I?

learning_rate_1�HG7\�9�I       6%�	CrU���A�*;


total_lossUy�@

error_RE�Q?

learning_rate_1�HG7"��MI       6%�	P�U���A�*;


total_loss''�@

error_R��R?

learning_rate_1�HG7����I       6%�	X�U���A�*;


total_loss ��@

error_R�X?

learning_rate_1�HG7_�I       6%�	�>U���A�*;


total_loss���@

error_R.[G?

learning_rate_1�HG7��:I       6%�	�U���A�*;


total_loss>�@

error_R�[?

learning_rate_1�HG7#��LI       6%�	�U���A�*;


total_loss���@

error_R/d?

learning_rate_1�HG7T�u*I       6%�	~U���A�*;


total_loss[��@

error_R tK?

learning_rate_1�HG7+�K�I       6%�	�NU���A�*;


total_lossd�A

error_R�VL?

learning_rate_1�HG7p=n7I       6%�	b�U���A�*;


total_loss�j�@

error_R��g?

learning_rate_1�HG7�{�I       6%�	��U���A�*;


total_loss��@

error_R=�o?

learning_rate_1�HG7|�I       6%�	� U���A�*;


total_loss��@

error_Rq�E?

learning_rate_1�HG7�8��I       6%�	�gU���A�*;


total_loss��@

error_R� T?

learning_rate_1�HG7U���I       6%�	d�U���A�*;


total_loss��@

error_R=tl?

learning_rate_1�HG7� 1LI       6%�	��U���A�*;


total_loss%��@

error_R�BN?

learning_rate_1�HG7��kI       6%�	�1U���A�*;


total_loss���@

error_RIS?

learning_rate_1�HG7㪉�I       6%�	yuU���A�*;


total_lossӲ�@

error_R��H?

learning_rate_1�HG77/��I       6%�	�U���A�*;


total_lossL+�@

error_Rx5K?

learning_rate_1�HG7(��_I       6%�	�	U���A�*;


total_loss�@

error_R؛N?

learning_rate_1�HG7fݢ�I       6%�	�O	U���A�*;


total_loss��@

error_R�0\?

learning_rate_1�HG7"t�I       6%�	%�	U���A�*;


total_loss�}�@

error_R��A?

learning_rate_1�HG7'��mI       6%�	v�	U���A�*;


total_loss��A

error_R��O?

learning_rate_1�HG7PI       6%�	k"
U���A�*;


total_loss4�A

error_R��I?

learning_rate_1�HG7 ��I       6%�	�h
U���A�*;


total_loss�) A

error_R/�_?

learning_rate_1�HG7��@I       6%�	C�
U���A�*;


total_loss��@

error_R��J?

learning_rate_1�HG7�3�I       6%�	`U���A�*;


total_lossV�@

error_RD�E?

learning_rate_1�HG7��/�I       6%�	 [U���A�*;


total_loss�%�@

error_R�TE?

learning_rate_1�HG7 e�I       6%�	�U���A�*;


total_lossךA

error_Rmx_?

learning_rate_1�HG7��kI       6%�	P�U���A�*;


total_loss頳@

error_R-G?

learning_rate_1�HG7Qwa�I       6%�	M%U���A�*;


total_loss4}�@

error_R�l@?

learning_rate_1�HG7�ԃ�I       6%�	�kU���A�*;


total_loss%�A

error_R��T?

learning_rate_1�HG7�Rn�I       6%�	e�U���A�*;


total_loss�o�@

error_R��C?

learning_rate_1�HG7�|o}I       6%�	��U���A�*;


total_loss
B�@

error_R
pP?

learning_rate_1�HG7w�J3I       6%�	ARU���A�*;


total_lossfM�@

error_R�S?

learning_rate_1�HG7�ڇ]I       6%�	=�U���A�*;


total_lossw��@

error_R�*?

learning_rate_1�HG7�(��I       6%�	�U���A�*;


total_lossI$�@

error_R�?]?

learning_rate_1�HG7�Z�I       6%�	'U���A�*;


total_loss���@

error_R�A?

learning_rate_1�HG7�dxtI       6%�	��U���A�*;


total_loss��@

error_RܒW?

learning_rate_1�HG7��gI       6%�	<�U���A�*;


total_loss��@

error_R:O?

learning_rate_1�HG7��uI       6%�	dU���A�*;


total_loss�޾@

error_R�R?

learning_rate_1�HG7�!I(I       6%�	�QU���A�*;


total_loss�F�@

error_R�K?

learning_rate_1�HG7�`�I       6%�	�U���A�*;


total_loss���@

error_RfB?

learning_rate_1�HG7�v��I       6%�	��U���A�*;


total_loss>�@

error_R%qb?

learning_rate_1�HG7����I       6%�	�#U���A�*;


total_loss�D�@

error_R�Y?

learning_rate_1�HG7��CI       6%�	!kU���A�*;


total_loss\p@

error_R��Z?

learning_rate_1�HG7kqI       6%�	M�U���A�*;


total_loss��A

error_R�W?

learning_rate_1�HG7A�6�I       6%�	q�U���A�*;


total_loss=ʕ@

error_R{�W?

learning_rate_1�HG7O��I       6%�	|<U���A�*;


total_loss�.�@

error_R=
G?

learning_rate_1�HG7�� 3I       6%�	S�U���A�*;


total_lossv��@

error_RH�Z?

learning_rate_1�HG74�I       6%�	��U���A�*;


total_loss��@

error_R�uN?

learning_rate_1�HG7k��JI       6%�	uU���A�*;


total_loss�@

error_R�D?

learning_rate_1�HG7K�3I       6%�	TU���A�*;


total_loss1��@

error_R��^?

learning_rate_1�HG7��+VI       6%�	��U���A�*;


total_loss��M@

error_R��I?

learning_rate_1�HG7G	i�I       6%�	��U���A�*;


total_loss'�@

error_R �J?

learning_rate_1�HG7^�^I       6%�	�*U���A�*;


total_lossl�9A

error_R��L?

learning_rate_1�HG7�2�I       6%�	pU���A�*;


total_lossa�t@

error_R
@L?

learning_rate_1�HG7NF�I       6%�	��U���A�*;


total_lossw��@

error_R��M?

learning_rate_1�HG7�a�I       6%�	�U���A�*;


total_loss���@

error_R��@?

learning_rate_1�HG7Z;�I       6%�	E?U���A�*;


total_loss���@

error_R{�S?

learning_rate_1�HG7F��I       6%�	w�U���A�*;


total_loss&e�@

error_R�"M?

learning_rate_1�HG7SN�I       6%�	P�U���A�*;


total_lossp�@

error_R;P?

learning_rate_1�HG7��"�I       6%�	�U���A�*;


total_lossl��@

error_R��_?

learning_rate_1�HG7�m�I       6%�	�_U���A�*;


total_lossq��@

error_R��K?

learning_rate_1�HG72�5�I       6%�	B�U���A�*;


total_loss�A

error_R��J?

learning_rate_1�HG7$���I       6%�	��U���A�*;


total_loss!��@

error_R�W?

learning_rate_1�HG7���#I       6%�	�$U���A�*;


total_loss���@

error_R36b?

learning_rate_1�HG7���I       6%�	0jU���A�*;


total_loss��@

error_R�-g?

learning_rate_1�HG70!I       6%�	��U���A�*;


total_lossmw�@

error_R%�X?

learning_rate_1�HG7'�!I       6%�	��U���A�*;


total_lossR��@

error_Rs�T?

learning_rate_1�HG7���I       6%�	]4U���A�*;


total_loss���@

error_R!�>?

learning_rate_1�HG7�0�I       6%�	DwU���A�*;


total_lossn��@

error_RߛV?

learning_rate_1�HG7��TI       6%�	�U���A�*;


total_losso��@

error_R�N?

learning_rate_1�HG7h�I       6%�	��U���A�*;


total_loss�;�@

error_RؓU?

learning_rate_1�HG7�V�I       6%�	wCU���A�*;


total_loss|0A

error_R��>?

learning_rate_1�HG7=F��I       6%�	ӆU���A�*;


total_loss)[�@

error_R�<M?

learning_rate_1�HG73��I       6%�	��U���A�*;


total_loss&ۀ@

error_R�T?

learning_rate_1�HG7��nOI       6%�	�U���A�*;


total_loss���@

error_R�9?

learning_rate_1�HG7����I       6%�	*eU���A�*;


total_lossK:�@

error_R{�Q?

learning_rate_1�HG7���%I       6%�	Y�U���A�*;


total_loss��@

error_RA�Z?

learning_rate_1�HG7���I       6%�	D�U���A�*;


total_lossD�s@

error_R�L?

learning_rate_1�HG7Hʼ�I       6%�	�;U���A�*;


total_loss��@

error_R�z@?

learning_rate_1�HG7�w6I       6%�	,�U���A�*;


total_loss��}@

error_R��Z?

learning_rate_1�HG7��hI       6%�	��U���A�*;


total_losso8�@

error_RĹV?

learning_rate_1�HG7W�I       6%�	A6U���A�*;


total_lossl��@

error_RxQ?

learning_rate_1�HG7�V� I       6%�	�~U���A�*;


total_loss���@

error_R.�@?

learning_rate_1�HG7Ӈ�\I       6%�	��U���A�*;


total_loss�ƫ@

error_R=�G?

learning_rate_1�HG7���tI       6%�	_U���A�*;


total_loss�ƪ@

error_Rf[Z?

learning_rate_1�HG7�]$=I       6%�	�RU���A�*;


total_loss�ac@

error_R��??

learning_rate_1�HG7�2��I       6%�		�U���A�*;


total_loss��@

error_R-[A?

learning_rate_1�HG7d���I       6%�	!�U���A�*;


total_loss�N~@

error_R̨I?

learning_rate_1�HG7�SH+I       6%�	�3U���A�*;


total_loss
S�@

error_R�P?

learning_rate_1�HG7P�I       6%�	zU���A�*;


total_loss��@

error_R9N?

learning_rate_1�HG7VDX�I       6%�	H�U���A�*;


total_lossʬ@

error_RtH?

learning_rate_1�HG7=�qI       6%�	bU���A�*;


total_loss3�@

error_R]�<?

learning_rate_1�HG7H�I       6%�	�NU���A�*;


total_loss$��@

error_R�zV?

learning_rate_1�HG7�$�dI       6%�	|�U���A�*;


total_loss_�@

error_RM?

learning_rate_1�HG7�[rI       6%�	,�U���A�*;


total_loss\�A

error_RJsL?

learning_rate_1�HG7֘��I       6%�	�U���A�*;


total_loss�A�@

error_R@�X?

learning_rate_1�HG7���KI       6%�	�^U���A�*;


total_loss��@

error_R��b?

learning_rate_1�HG7Nz_�I       6%�	�U���A�*;


total_loss�Þ@

error_RO�J?

learning_rate_1�HG7�u�2I       6%�	��U���A�*;


total_loss�)�@

error_R| L?

learning_rate_1�HG7l�I       6%�	7 U���A�*;


total_lossf=A

error_R��T?

learning_rate_1�HG7���I       6%�	^� U���A�*;


total_loss3f�@

error_R�}>?

learning_rate_1�HG7J��LI       6%�	'� U���A�*;


total_lossM�@

error_R�J?

learning_rate_1�HG7�/[�I       6%�	�"!U���A�*;


total_lossm;�@

error_R!�M?

learning_rate_1�HG7H��I       6%�	m!U���A�*;


total_loss���@

error_RN�S?

learning_rate_1�HG7��(�I       6%�	D�!U���A�*;


total_loss���@

error_R�A?

learning_rate_1�HG7ֹ�yI       6%�	��!U���A�*;


total_loss λ@

error_R�M?

learning_rate_1�HG77���I       6%�	�="U���A�*;


total_lossN}�@

error_RQ\]?

learning_rate_1�HG7�6�I       6%�	a�"U���A�*;


total_lossC�SA

error_R!IP?

learning_rate_1�HG7e��I       6%�	9�"U���A�*;


total_loss���@

error_R([?

learning_rate_1�HG7TR`�I       6%�	G#U���A�*;


total_lossm:A

error_R�SU?

learning_rate_1�HG7�mTI       6%�	[#U���A�*;


total_loss�wV@

error_RLQD?

learning_rate_1�HG7ʞ�I       6%�	>�#U���A�*;


total_loss���@

error_R��a?

learning_rate_1�HG7��"I       6%�	��#U���A�*;


total_loss�#�@

error_R�)O?

learning_rate_1�HG7���I       6%�	4$U���A�*;


total_lossYR�@

error_R�5V?

learning_rate_1�HG7����I       6%�	~$U���A�*;


total_loss�Ԍ@

error_RSD?

learning_rate_1�HG7��NI       6%�	]�$U���A�*;


total_loss��@

error_R�\?

learning_rate_1�HG7!f[�I       6%�	�%U���A�*;


total_loss�u�@

error_R��a?

learning_rate_1�HG7��]\I       6%�	�J%U���A�*;


total_loss�A�@

error_R��=?

learning_rate_1�HG7��&I       6%�	�%U���A�*;


total_lossI�@

error_R�P?

learning_rate_1�HG7�
�xI       6%�	��%U���A�*;


total_lossd�	A

error_R��O?

learning_rate_1�HG7T�7I       6%�	1&U���A�*;


total_loss��@

error_R�V?

learning_rate_1�HG7<KI       6%�	�e&U���A�*;


total_loss��@

error_RcP?

learning_rate_1�HG7$�>I       6%�	��&U���A�*;


total_loss�@

error_R&�O?

learning_rate_1�HG7.�q7I       6%�	"�&U���A�*;


total_loss���@

error_R:�Z?

learning_rate_1�HG7�{�I       6%�	B'U���A�*;


total_loss��@

error_RŉH?

learning_rate_1�HG7����I       6%�	��'U���A�*;


total_loss�L�@

error_R�L>?

learning_rate_1�HG7�E0�I       6%�	}�'U���A�*;


total_loss�i�@

error_R|�W?

learning_rate_1�HG7����I       6%�	>(U���A�*;


total_loss?C�@

error_R�P?

learning_rate_1�HG7��k�I       6%�	�W(U���A�*;


total_loss���@

error_R�V^?

learning_rate_1�HG7��t�I       6%�	B�(U���A�*;


total_lossd�@

error_R�{N?

learning_rate_1�HG7SɇI       6%�	��(U���A�*;


total_loss��w@

error_R��f?

learning_rate_1�HG7�эyI       6%�	�")U���A�*;


total_loss��@

error_R�U?

learning_rate_1�HG7���/I       6%�	%d)U���A�*;


total_lossHU�@

error_R��[?

learning_rate_1�HG7n���I       6%�	�)U���A�*;


total_loss���@

error_R)�Q?

learning_rate_1�HG7���+I       6%�	��)U���A�*;


total_loss�PA

error_RڝJ?

learning_rate_1�HG7{�]�I       6%�	�;*U���A�*;


total_loss��@

error_R7<?

learning_rate_1�HG7U�vI       6%�	��*U���A�*;


total_lossWx�@

error_R��D?

learning_rate_1�HG7Q>�yI       6%�	`�*U���A�*;


total_loss�MA

error_R��h?

learning_rate_1�HG7�:#�I       6%�	�++U���A�*;


total_loss}�@

error_R��H?

learning_rate_1�HG7�jh�I       6%�	'r+U���A�*;


total_loss��@

error_R�MN?

learning_rate_1�HG7��I       6%�	L�+U���A�*;


total_lossLT�@

error_RCyF?

learning_rate_1�HG7jU�I       6%�	�+U���A�*;


total_lossi��@

error_R�"H?

learning_rate_1�HG7l�I       6%�	�>,U���A�*;


total_loss�2�@

error_R�td?

learning_rate_1�HG7�K%I       6%�	߃,U���A�*;


total_loss}�@

error_RHYC?

learning_rate_1�HG7���I       6%�	��,U���A�*;


total_lossH�A

error_R��N?

learning_rate_1�HG7���I       6%�	I-U���A�*;


total_loss�/�@

error_RiLa?

learning_rate_1�HG76޹&I       6%�	Aj-U���A�*;


total_loss?�A

error_R֣C?

learning_rate_1�HG7����I       6%�	��-U���A�*;


total_loss9�@

error_R�U?

learning_rate_1�HG7[�FI       6%�	z�-U���A�*;


total_lossŝ�@

error_RORC?

learning_rate_1�HG7�|ͷI       6%�	R.U���A�*;


total_loss���@

error_Rd�H?

learning_rate_1�HG7Ծ$I       6%�	Q�.U���A�*;


total_loss ��@

error_R�F?

learning_rate_1�HG7_pI       6%�	�.U���A�*;


total_loss�?'A

error_R��Q?

learning_rate_1�HG7#�v�I       6%�	M,/U���A�*;


total_loss!ut@

error_Rqca?

learning_rate_1�HG7��iWI       6%�	�z/U���A�*;


total_lossa�(A

error_RS%Z?

learning_rate_1�HG7����I       6%�	��/U���A�*;


total_loss,-�@

error_Rq�8?

learning_rate_1�HG71�_I       6%�	0U���A�*;


total_loss,�@

error_R��]?

learning_rate_1�HG7V� LI       6%�	�P0U���A�*;


total_loss�߲@

error_R�>U?

learning_rate_1�HG7��ȁI       6%�	f�0U���A�*;


total_loss`��@

error_RE2W?

learning_rate_1�HG7#��I       6%�	�0U���A�*;


total_loss���@

error_R"N?

learning_rate_1�HG7[m�iI       6%�	�21U���A�*;


total_loss�d�@

error_Rt�8?

learning_rate_1�HG7n>��I       6%�	%~1U���A�*;


total_lossAm�@

error_R��I?

learning_rate_1�HG7�e�~I       6%�	�1U���A�*;


total_loss��@

error_Rh�<?

learning_rate_1�HG7�~�I       6%�	e
2U���A�*;


total_loss�tA

error_R1�u?

learning_rate_1�HG7 �K�I       6%�	"M2U���A�*;


total_loss�Z�@

error_R�yZ?

learning_rate_1�HG7���YI       6%�	�2U���A�*;


total_loss�5�@

error_R��S?

learning_rate_1�HG7a�I       6%�	��2U���A�*;


total_loss_��@

error_R{�`?

learning_rate_1�HG7��}�I       6%�	3U���A�*;


total_loss�[�@

error_RңJ?

learning_rate_1�HG7�O��I       6%�	�a3U���A�*;


total_lossݢ�@

error_R�}L?

learning_rate_1�HG7p���I       6%�	��3U���A�*;


total_loss��@

error_R�{Y?

learning_rate_1�HG7~��%I       6%�	��3U���A�*;


total_loss2��@

error_R�9_?

learning_rate_1�HG7$/'�I       6%�	�.4U���A�*;


total_lossrN�@

error_R8�C?

learning_rate_1�HG7�� I       6%�	�4U���A�*;


total_loss���@

error_R�S?

learning_rate_1�HG7Lv�MI       6%�	��4U���A�*;


total_loss7�@

error_R��C?

learning_rate_1�HG73���I       6%�	�5U���A�*;


total_loss?�@

error_R �I?

learning_rate_1�HG70��I       6%�	�M5U���A�*;


total_loss�o�@

error_R,NP?

learning_rate_1�HG7��PI       6%�	d�5U���A�*;


total_loss�K�@

error_R��L?

learning_rate_1�HG7�-A]I       6%�	��5U���A�*;


total_loss�א@

error_R)�L?

learning_rate_1�HG7���oI       6%�	"6U���A�*;


total_loss�Ѫ@

error_Re$P?

learning_rate_1�HG7�%ΰI       6%�	�f6U���A�*;


total_loss�-�@

error_R�D?

learning_rate_1�HG7�K�eI       6%�	�6U���A�*;


total_loss�P�@

error_R��C?

learning_rate_1�HG7��gI       6%�	
�6U���A�*;


total_lossNW@

error_R�RB?

learning_rate_1�HG7 ��I       6%�	27U���A�*;


total_lossQc�@

error_RV�\?

learning_rate_1�HG7x�n+I       6%�	gu7U���A�*;


total_loss�Q@

error_R��9?

learning_rate_1�HG7(���I       6%�	�7U���A�*;


total_loss��@

error_RD�D?

learning_rate_1�HG7�1�I       6%�	��7U���A�*;


total_loss��A

error_RT M?

learning_rate_1�HG7�ШI       6%�	X<8U���A�*;


total_loss��@

error_R!�P?

learning_rate_1�HG7�?�4I       6%�	�}8U���A�*;


total_loss�R�@

error_RZ�I?

learning_rate_1�HG7���I       6%�	�8U���A�*;


total_lossqm�@

error_R)�v?

learning_rate_1�HG7� ��I       6%�	x9U���A�*;


total_loss[.�@

error_R�S?

learning_rate_1�HG7  I       6%�	ML9U���A�*;


total_loss ��@

error_RT�6?

learning_rate_1�HG7#�HI       6%�		�9U���A�*;


total_lossʮ�@

error_RN�d?

learning_rate_1�HG7�xI       6%�	��9U���A�*;


total_loss�I�@

error_RT�P?

learning_rate_1�HG7LmI       6%�	� :U���A�*;


total_lossO�@

error_R�]K?

learning_rate_1�HG7����I       6%�	�f:U���A�*;


total_lossaB�@

error_R&J?

learning_rate_1�HG7�}I       6%�	��:U���A�*;


total_loss���@

error_R�QK?

learning_rate_1�HG7x��I       6%�	I;U���A�*;


total_lossxl@

error_R��J?

learning_rate_1�HG7��'�I       6%�	k^;U���A�*;


total_loss��@

error_RwK\?

learning_rate_1�HG7�x��I       6%�	�;U���A�*;


total_loss��@

error_R��R?

learning_rate_1�HG7�׾[I       6%�	�;U���A�*;


total_losss�@

error_RT�G?

learning_rate_1�HG7g��0I       6%�	W0<U���A�*;


total_lossRs�@

error_R��_?

learning_rate_1�HG7-�D�I       6%�	t<U���A�*;


total_loss1�@

error_R@�S?

learning_rate_1�HG7��a�I       6%�	�<U���A�*;


total_lossC�A

error_R��M?

learning_rate_1�HG7�"{I       6%�	K�<U���A�*;


total_loss��@

error_R�A?

learning_rate_1�HG7��i�I       6%�	�A=U���A�*;


total_loss�v�@

error_R�S?

learning_rate_1�HG7�G�I       6%�	V�=U���A�*;


total_loss.�@

error_RM�[?

learning_rate_1�HG7A��+I       6%�	��=U���A�*;


total_lossf��@

error_R#!L?

learning_rate_1�HG7�#DUI       6%�	T>U���A�*;


total_lossW�@

error_R�.a?

learning_rate_1�HG7�tz�I       6%�	X>U���A�*;


total_loss͐@

error_Re�O?

learning_rate_1�HG7��N�I       6%�	��>U���A�*;


total_loss`j�@

error_R6�T?

learning_rate_1�HG7l,]�I       6%�	��>U���A�*;


total_loss�
�@

error_R|�O?

learning_rate_1�HG7��:I       6%�	�$?U���A�*;


total_lossRh�@

error_R.KM?

learning_rate_1�HG7�E]I       6%�	�g?U���A�*;


total_loss���@

error_R�ol?

learning_rate_1�HG7����I       6%�	��?U���A�*;


total_loss@��@

error_R�HE?

learning_rate_1�HG70G�?I       6%�	��?U���A�*;


total_loss�_�@

error_RZaI?

learning_rate_1�HG7��Y?I       6%�	�1@U���A�*;


total_loss��@

error_R��Y?

learning_rate_1�HG7�҄XI       6%�	�t@U���A�*;


total_loss���@

error_R�S?

learning_rate_1�HG7�<>�I       6%�	,�@U���A�*;


total_lossE��@

error_R�Y?

learning_rate_1�HG7U���I       6%�	��@U���A�*;


total_loss2A

error_R��U?

learning_rate_1�HG7�?�I       6%�	�=AU���A�*;


total_loss<�@

error_R�1S?

learning_rate_1�HG7���I       6%�	�AU���A�*;


total_loss<�@

error_R��C?

learning_rate_1�HG7�J�OI       6%�	��AU���A�*;


total_lossl�@

error_R��J?

learning_rate_1�HG7%���I       6%�	O
BU���A�*;


total_loss�S|@

error_R�kH?

learning_rate_1�HG7�oHI       6%�	OBU���A�*;


total_loss�Ҫ@

error_R�UO?

learning_rate_1�HG7``u�I       6%�	�BU���A�*;


total_lossz��@

error_R�T?

learning_rate_1�HG7�aıI       6%�	�BU���A�*;


total_lossź�@

error_RNO?

learning_rate_1�HG7n��UI       6%�	`CU���A�*;


total_loss���@

error_R(iZ?

learning_rate_1�HG7����I       6%�	�WCU���A�*;


total_lossL@�@

error_R!�N?

learning_rate_1�HG7.`�wI       6%�	��CU���A�*;


total_loss���@

error_R�bK?

learning_rate_1�HG7B(pI       6%�	��CU���A�*;


total_lossr=�@

error_R��Q?

learning_rate_1�HG7C�*�I       6%�	H2DU���A�*;


total_loss���@

error_R.�P?

learning_rate_1�HG7�@��I       6%�	�sDU���A�*;


total_loss�u�@

error_ROR?

learning_rate_1�HG7�o�[I       6%�	��DU���A�*;


total_loss��@

error_Rr>?

learning_rate_1�HG7
TP�I       6%�	��DU���A�*;


total_loss���@

error_RdC?

learning_rate_1�HG7�{2>I       6%�	�?EU���A�*;


total_loss�ų@

error_R��Z?

learning_rate_1�HG7�4�7I       6%�	��EU���A�*;


total_loss-�@

error_R�S?

learning_rate_1�HG7�@�I       6%�	��EU���A�*;


total_loss��@

error_R<�J?

learning_rate_1�HG7�ʞI       6%�	NFU���A�*;


total_lossl��@

error_R��R?

learning_rate_1�HG7ә�I       6%�	HFU���A�*;


total_loss���@

error_R.�Q?

learning_rate_1�HG7��nFI       6%�	��FU���A�*;


total_loss$��@

error_R�YE?

learning_rate_1�HG7�k�I       6%�	��FU���A�*;


total_loss�{�@

error_R�_J?

learning_rate_1�HG7���JI       6%�	�GU���A�*;


total_lossEt�@

error_R!~S?

learning_rate_1�HG7�?aI       6%�	�QGU���A�*;


total_loss4V@

error_R^U?

learning_rate_1�HG70��I       6%�	�GU���A�*;


total_loss���@

error_R��I?

learning_rate_1�HG74�e�I       6%�	��GU���A�*;


total_loss霵@

error_R/�J?

learning_rate_1�HG7~Oj�I       6%�	_$HU���A�*;


total_lossFL�@

error_RVyH?

learning_rate_1�HG77���I       6%�	gpHU���A�*;


total_loss��@

error_R��^?

learning_rate_1�HG7��ҏI       6%�	��HU���A�*;


total_loss3��@

error_Rܺi?

learning_rate_1�HG7�~�I       6%�	�IU���A�*;


total_loss���@

error_RM�G?

learning_rate_1�HG7��V[I       6%�	BHIU���A�*;


total_loss/��@

error_R�a>?

learning_rate_1�HG75�>�I       6%�	>�IU���A�*;


total_loss�PA

error_RM�T?

learning_rate_1�HG7_-�I       6%�	��IU���A�*;


total_lossc��@

error_Rl*F?

learning_rate_1�HG7���I       6%�	/#JU���A�*;


total_loss<��@

error_R��X?

learning_rate_1�HG7�4J&I       6%�	�oJU���A�*;


total_loss���@

error_R��H?

learning_rate_1�HG7(ۊ�I       6%�	"�JU���A�*;


total_lossd@

error_RKA?

learning_rate_1�HG7p?��I       6%�	�#KU���A�*;


total_loss�߽@

error_R_�a?

learning_rate_1�HG7�ܕ6I       6%�	�jKU���A�*;


total_loss$��@

error_R�|]?

learning_rate_1�HG7a�xI       6%�	�KU���A�*;


total_loss �@

error_R��]?

learning_rate_1�HG7�JJ�I       6%�	��KU���A�*;


total_loss���@

error_R\`1?

learning_rate_1�HG7��1I       6%�	=:LU���A�*;


total_lossB�@

error_R��I?

learning_rate_1�HG7%*4�I       6%�	�}LU���A�*;


total_loss�ȼ@

error_R��Q?

learning_rate_1�HG7M�BI       6%�	��LU���A�*;


total_loss��a@

error_R��@?

learning_rate_1�HG7\]�I       6%�	FMU���A�*;


total_loss�@

error_R@�U?

learning_rate_1�HG7���I       6%�	`MU���A�*;


total_loss�_�@

error_RsI?

learning_rate_1�HG7��
�I       6%�	��MU���A�*;


total_loss�N�@

error_R�E?

learning_rate_1�HG7���I       6%�	�MU���A�*;


total_loss�|�@

error_ROU?

learning_rate_1�HG7]��I       6%�	�WNU���A�*;


total_loss#ʖ@

error_R�7:?

learning_rate_1�HG7��GxI       6%�	}�NU���A�*;


total_loss��@

error_R,�L?

learning_rate_1�HG7���I       6%�	P�NU���A�*;


total_loss��@

error_RC-O?

learning_rate_1�HG7�I       6%�	�1OU���A�*;


total_lossd��@

error_R��G?

learning_rate_1�HG7���oI       6%�	�sOU���A�*;


total_lossc��@

error_R�xE?

learning_rate_1�HG7�a>�I       6%�	��OU���A�*;


total_loss�H�@

error_R��S?

learning_rate_1�HG7�ڠI       6%�		�OU���A�*;


total_loss ;x@

error_R��M?

learning_rate_1�HG7�P�I       6%�	 CPU���A�*;


total_lossH��@

error_R|jI?

learning_rate_1�HG7FI�<I       6%�	هPU���A�*;


total_lossT)�@

error_R��Q?

learning_rate_1�HG7nTz{I       6%�	�PU���A�*;


total_loss�!�@

error_R��L?

learning_rate_1�HG7F� �I       6%�	(QU���A�*;


total_lossR6�@

error_R��O?

learning_rate_1�HG7�_I       6%�	lYQU���A�*;


total_lossm�@

error_R..J?

learning_rate_1�HG7_�RI       6%�	,�QU���A�*;


total_lossV��@

error_R�t:?

learning_rate_1�HG7��VI       6%�	��QU���A�*;


total_lossz�A

error_R&�d?

learning_rate_1�HG7��}I       6%�	�;RU���A�*;


total_loss<A#A

error_Rq J?

learning_rate_1�HG7<Rh�I       6%�	8�RU���A�*;


total_loss�aA

error_R=�F?

learning_rate_1�HG7���VI       6%�	��RU���A�*;


total_lossz(�@

error_R��Q?

learning_rate_1�HG7_��I       6%�	2SU���A�*;


total_lossju�@

error_R��P?

learning_rate_1�HG7�\iI       6%�	=QSU���A�*;


total_loss&��@

error_RM�S?

learning_rate_1�HG7s�I       6%�	l�SU���A�*;


total_loss-�@

error_R��=?

learning_rate_1�HG7��oI       6%�	��SU���A�*;


total_loss�ٙ@

error_R1T?

learning_rate_1�HG7ԔoI       6%�	�TU���A�*;


total_loss�w�@

error_R�*H?

learning_rate_1�HG7��I       6%�	�cTU���A�*;


total_lossw�A

error_Ra�L?

learning_rate_1�HG7���~I       6%�	�TU���A�*;


total_loss1��@

error_R��L?

learning_rate_1�HG7K�4I       6%�	�TU���A�*;


total_loss���@

error_R�KG?

learning_rate_1�HG7���MI       6%�	�0UU���A�*;


total_lossCӔ@

error_R�Q?

learning_rate_1�HG7��&�I       6%�	�sUU���A�*;


total_loss�eA

error_R�F[?

learning_rate_1�HG7���2I       6%�	ǴUU���A�*;


total_loss���@

error_RJ?

learning_rate_1�HG7�ڹ�I       6%�		�UU���A�*;


total_loss
�@

error_R��^?

learning_rate_1�HG7nC$�I       6%�	�;VU���A�*;


total_loss
l�@

error_Rݙf?

learning_rate_1�HG7�-4�I       6%�	EVU���A�*;


total_loss�=�@

error_R�iK?

learning_rate_1�HG7ZkrI       6%�	h�VU���A�*;


total_loss��
A

error_R��Y?

learning_rate_1�HG7��I       6%�	'WU���A�*;


total_lossl��@

error_R\H?

learning_rate_1�HG7�Yv�I       6%�	KTWU���A�*;


total_lossA�A

error_R�;^?

learning_rate_1�HG7>�@�I       6%�	8�WU���A�*;


total_lossqԠ@

error_R�M?

learning_rate_1�HG7�RII       6%�	��WU���A�*;


total_lossx�@

error_R��P?

learning_rate_1�HG7�qĻI       6%�	�%XU���A�*;


total_loss쭤@

error_R_a=?

learning_rate_1�HG7�;��I       6%�	�hXU���A�*;


total_loss�+�@

error_RMK?

learning_rate_1�HG7� �I       6%�	�XU���A�*;


total_loss?ō@

error_R�OH?

learning_rate_1�HG7󿝤I       6%�	��XU���A�*;


total_lossR��@

error_R}J?

learning_rate_1�HG72�oI       6%�	K7YU���A�*;


total_loss!̤@

error_R��T?

learning_rate_1�HG7�j1�I       6%�	�YU���A�*;


total_loss4:�@

error_R�G@?

learning_rate_1�HG7�F?I       6%�	��YU���A�*;


total_loss4��@

error_R,"G?

learning_rate_1�HG7��;�I       6%�	;ZU���A�*;


total_loss��@

error_R��c?

learning_rate_1�HG7�&hI       6%�	l[ZU���A�*;


total_loss&`�@

error_R��L?

learning_rate_1�HG7"ÝhI       6%�	{�ZU���A�*;


total_lossvR�@

error_R]_a?

learning_rate_1�HG7�Έ�I       6%�	`�ZU���A�*;


total_loss�̘@

error_R%�i?

learning_rate_1�HG7�6�cI       6%�	�L[U���A�*;


total_loss=_�@

error_R�A?

learning_rate_1�HG7]6AI       6%�	ڕ[U���A�*;


total_lossD[�@

error_RQ<:?

learning_rate_1�HG7S�V�I       6%�	b�[U���A�*;


total_loss���@

error_R}�V?

learning_rate_1�HG7b�<\I       6%�	�$\U���A�*;


total_loss@җ@

error_RXK?

learning_rate_1�HG7����I       6%�	#g\U���A�*;


total_loss(_�@

error_R=�c?

learning_rate_1�HG7�Ɛ*I       6%�	ѫ\U���A�*;


total_lossJF�@

error_R:�L?

learning_rate_1�HG7���I       6%�	2�\U���A�*;


total_loss�݅@

error_R=Md?

learning_rate_1�HG7[oՠI       6%�	!6]U���A�*;


total_loss�@�@

error_R��??

learning_rate_1�HG7ǐ�KI       6%�	�z]U���A�*;


total_loss�?�@

error_R��\?

learning_rate_1�HG75�BI       6%�	��]U���A�*;


total_loss��@

error_RƹT?

learning_rate_1�HG7&uy�I       6%�	^U���A�*;


total_loss.k@

error_R �Z?

learning_rate_1�HG7ָ��I       6%�	
I^U���A�*;


total_loss30�@

error_R��W?

learning_rate_1�HG7��� I       6%�	S�^U���A�*;


total_loss�4�@

error_R�:?

learning_rate_1�HG7+��I       6%�	��^U���A�*;


total_lossԬ@

error_RR?

learning_rate_1�HG7�f�7I       6%�	$_U���A�*;


total_loss�ך@

error_R�AQ?

learning_rate_1�HG7���ZI       6%�	YY_U���A�*;


total_loss<��@

error_R�H?

learning_rate_1�HG7Y�c�I       6%�	~�_U���A�*;


total_loss�ɻ@

error_R�Q?

learning_rate_1�HG7ӎaMI       6%�	��_U���A�*;


total_loss�N�@

error_R4�T?

learning_rate_1�HG7��I       6%�	�'`U���A�*;


total_loss�@

error_R��F?

learning_rate_1�HG7.�<�I       6%�	�i`U���A�*;


total_loss8�@

error_R{KJ?

learning_rate_1�HG7��zKI       6%�	m�`U���A�*;


total_loss=��@

error_R�EM?

learning_rate_1�HG7fV��I       6%�	��`U���A�*;


total_loss���@

error_R�a?

learning_rate_1�HG7j��RI       6%�	�7aU���A�*;


total_loss[c�@

error_R��R?

learning_rate_1�HG73Nw?I       6%�	�}aU���A�*;


total_loss���@

error_R�BS?

learning_rate_1�HG7��vjI       6%�	��aU���A�*;


total_loss;��@

error_R�L?

learning_rate_1�HG7Az��I       6%�	�bU���A�*;


total_lossJ��@

error_R�qH?

learning_rate_1�HG7�mܩI       6%�	JbU���A�*;


total_lossJ��@

error_R;9G?

learning_rate_1�HG7F.ŽI       6%�	W�bU���A�*;


total_losszN�@

error_R��??

learning_rate_1�HG7��^�I       6%�	�bU���A�*;


total_loss̈́�@

error_R�a?

learning_rate_1�HG7�L�&I       6%�	@cU���A�*;


total_loss�;�@

error_R��Z?

learning_rate_1�HG7����I       6%�	%hcU���A�*;


total_loss�P�@

error_RW?

learning_rate_1�HG7�I       6%�	�cU���A�*;


total_lossX �@

error_RT�g?

learning_rate_1�HG7���/I       6%�	�cU���A�*;


total_loss�T�@

error_R��L?

learning_rate_1�HG7�O�cI       6%�	iIdU���A�*;


total_loss*WA

error_RxR?

learning_rate_1�HG77v>yI       6%�	�dU���A�*;


total_loss�7�@

error_R��M?

learning_rate_1�HG72��DI       6%�	/�dU���A�*;


total_loss0�@

error_R��Y?

learning_rate_1�HG7{�[�I       6%�	�eU���A�*;


total_loss`�@

error_R��R?

learning_rate_1�HG7;8�I       6%�	mbeU���A�*;


total_loss#��@

error_R:�5?

learning_rate_1�HG7�=?I       6%�	�eU���A�*;


total_lossc�A

error_R��F?

learning_rate_1�HG7bsX,I       6%�	��eU���A�*;


total_loss��@

error_RL�F?

learning_rate_1�HG7T��KI       6%�	�/fU���A�*;


total_loss�)�@

error_R?H?

learning_rate_1�HG7�F�I       6%�	�sfU���A�*;


total_loss��A

error_R��H?

learning_rate_1�HG7��I       6%�	5�fU���A�*;


total_lossD��@

error_R)�J?

learning_rate_1�HG7 }9�I       6%�	$�fU���A�*;


total_loss�j�@

error_RwKn?

learning_rate_1�HG7~��&I       6%�	B;gU���A�*;


total_loss���@

error_R*�]?

learning_rate_1�HG7H/��I       6%�	S�gU���A�*;


total_lossJ��@

error_R��T?

learning_rate_1�HG738�I       6%�	
�gU���A�*;


total_loss_D�@

error_R�5S?

learning_rate_1�HG7hk�I       6%�	�hU���A�*;


total_lossf@

error_RL�M?

learning_rate_1�HG7�=/|I       6%�	�LhU���A�*;


total_lossqy�@

error_R�7O?

learning_rate_1�HG7�?�I       6%�	��hU���A�*;


total_lossA�@

error_R�nF?

learning_rate_1�HG7{ҷLI       6%�	��hU���A�*;


total_lossg�@

error_R)�S?

learning_rate_1�HG7�ȟ/I       6%�	SiU���A�*;


total_loss��F@

error_R�9?

learning_rate_1�HG7P&UI       6%�	�[iU���A�*;


total_loss��j@

error_R��D?

learning_rate_1�HG7d��#I       6%�	��iU���A�*;


total_loss2��@

error_RViR?

learning_rate_1�HG7�7��I       6%�	I�iU���A�*;


total_lossYO�@

error_R}5M?

learning_rate_1�HG7��YHI       6%�	1jU���A�*;


total_loss���@

error_R-�U?

learning_rate_1�HG7h0��I       6%�	ktjU���A�*;


total_loss��A

error_R��X?

learning_rate_1�HG7�uovI       6%�	��jU���A�*;


total_loss�B�@

error_R�*i?

learning_rate_1�HG7ĥ�9I       6%�	"kU���A�*;


total_loss���@

error_R�"D?

learning_rate_1�HG7���eI       6%�	�fkU���A�*;


total_loss4z�@

error_R�M?

learning_rate_1�HG7Q�RI       6%�	��kU���A�*;


total_loss^0�@

error_Rp]?

learning_rate_1�HG79xu�I       6%�	��kU���A�*;


total_loss���@

error_R#�P?

learning_rate_1�HG7��I       6%�	X4lU���A�*;


total_loss���@

error_R�BF?

learning_rate_1�HG7o<��I       6%�		zlU���A�*;


total_loss�B�@

error_R?�[?

learning_rate_1�HG7�N`I       6%�	��lU���A�*;


total_loss��@

error_R�WF?

learning_rate_1�HG7�0I       6%�	�
mU���A�*;


total_loss!G�@

error_R
Fe?

learning_rate_1�HG7>���I       6%�	nfmU���A�*;


total_loss��|@

error_R��E?

learning_rate_1�HG7����I       6%�	K�mU���A�*;


total_lossA�@

error_R�xH?

learning_rate_1�HG7��f�I       6%�	f�mU���A�*;


total_lossl��@

error_RFTW?

learning_rate_1�HG7���I       6%�	�PnU���A�*;


total_loss ��@

error_R
<7?

learning_rate_1�HG75���I       6%�	�nU���A�*;


total_lossd+�@

error_R�5H?

learning_rate_1�HG7%��4I       6%�	��nU���A�*;


total_loss-�@

error_R�@T?

learning_rate_1�HG7{F�}I       6%�	H<oU���A�*;


total_loss���@

error_R�yC?

learning_rate_1�HG7-��I       6%�	��oU���A�*;


total_loss�5�@

error_R[�K?

learning_rate_1�HG7���I       6%�	.�oU���A�*;


total_loss8��@

error_R�eG?

learning_rate_1�HG7�'I       6%�	|pU���A�*;


total_loss��@

error_R�KN?

learning_rate_1�HG7�X�kI       6%�	]pU���A�*;


total_lossn�@

error_R\pQ?

learning_rate_1�HG7����I       6%�	��pU���A�*;


total_loss���@

error_R�L?

learning_rate_1�HG7��3�I       6%�	��pU���A�*;


total_loss�	�@

error_R�V?

learning_rate_1�HG7	�I       6%�	�(qU���A�*;


total_loss\R�@

error_R�H?

learning_rate_1�HG7��aI       6%�	mnqU���A�*;


total_loss7=�@

error_R�B?

learning_rate_1�HG7.�rAI       6%�	i�qU���A�*;


total_loss(��@

error_R�S?

learning_rate_1�HG7&�SI       6%�	0�qU���A�*;


total_loss�^�@

error_R�9?

learning_rate_1�HG7��V�I       6%�	m:rU���A�*;


total_loss�k�@

error_R��S?

learning_rate_1�HG7�@�I       6%�	��rU���A�*;


total_loss���@

error_RRP?

learning_rate_1�HG7�x$�I       6%�	s�rU���A�*;


total_loss$ٞ@

error_RM�Z?

learning_rate_1�HG7Ώ�<I       6%�	�sU���A�*;


total_loss�A

error_R6�J?

learning_rate_1�HG7��zVI       6%�	"TsU���A�*;


total_loss���@

error_RۭG?

learning_rate_1�HG7s>�kI       6%�	�sU���A�*;


total_loss�/A

error_R�WO?

learning_rate_1�HG7V��I       6%�	-�sU���A�*;


total_lossqo�@

error_R�SO?

learning_rate_1�HG7(i��I       6%�	N+tU���A�*;


total_loss���@

error_RťP?

learning_rate_1�HG7�S)�I       6%�	�qtU���A�*;


total_loss�9�@

error_R�BR?

learning_rate_1�HG7Hyh�I       6%�	��tU���A�*;


total_loss��@

error_R��e?

learning_rate_1�HG71"FI       6%�	��tU���A�*;


total_loss��@

error_RͅC?

learning_rate_1�HG7�B,I       6%�	�EuU���A�*;


total_loss4HY@

error_R��P?

learning_rate_1�HG7 
�I       6%�	ߍuU���A�*;


total_lossJ:�@

error_R��I?

learning_rate_1�HG7�I       6%�		�uU���A�*;


total_loss���@

error_R6xE?

learning_rate_1�HG7�e��I       6%�	�vU���A�*;


total_lossA��@

error_R@]H?

learning_rate_1�HG7��@pI       6%�	�ivU���A�*;


total_losstؕ@

error_Ra5:?

learning_rate_1�HG7g�8�I       6%�	òvU���A�*;


total_loss���@

error_R�a?

learning_rate_1�HG7
'��I       6%�	�vU���A�*;


total_loss�l�@

error_R}�??

learning_rate_1�HG7%X��I       6%�	N8wU���A�*;


total_loss��@

error_RQQ?

learning_rate_1�HG7����I       6%�	y|wU���A�*;


total_lossj�@

error_RT=J?

learning_rate_1�HG7�߾rI       6%�	��wU���A�*;


total_loss]5�@

error_R1�I?

learning_rate_1�HG7O)i^I       6%�	��wU���A�*;


total_loss^@�@

error_Ra�X?

learning_rate_1�HG7��[ I       6%�	CxU���A�*;


total_loss�&�@

error_R�]M?

learning_rate_1�HG7��ʤI       6%�	�xU���A�*;


total_loss8��@

error_R�H?

learning_rate_1�HG7� ��I       6%�	F�xU���A�*;


total_lossݿ�@

error_R:�J?

learning_rate_1�HG7�|D�I       6%�	gyU���A�*;


total_loss�4�@

error_RJ8Z?

learning_rate_1�HG7~d:�I       6%�	n]yU���A�*;


total_lossi��@

error_R)�R?

learning_rate_1�HG7"ai�I       6%�	�yU���A�*;


total_loss�*z@

error_R ?J?

learning_rate_1�HG7�%+HI       6%�	1�yU���A�*;


total_loss�@A

error_R֨M?

learning_rate_1�HG7��JI       6%�	-zU���A�*;


total_losskɞ@

error_R�{C?

learning_rate_1�HG7��I       6%�	�pzU���A�*;


total_lossA

error_R:�>?

learning_rate_1�HG7�vVI       6%�	b�zU���A�*;


total_loss	w�@

error_Re:?

learning_rate_1�HG7�|K,I       6%�	{U���A�*;


total_loss��@

error_R��^?

learning_rate_1�HG7�ui�I       6%�	�[{U���A�*;


total_loss3��@

error_R��@?

learning_rate_1�HG7��I       6%�	Z�{U���A�*;


total_loss��@

error_Ru6?

learning_rate_1�HG7"��I       6%�	��{U���A�*;


total_loss�o�@

error_R�??

learning_rate_1�HG7�s�I       6%�	U#|U���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7�� �I       6%�	�g|U���A�*;


total_lossX&�@

error_R�_H?

learning_rate_1�HG7���cI       6%�	ª|U���A�*;


total_loss���@

error_R$�H?

learning_rate_1�HG7��K�I       6%�	�|U���A�*;


total_lossө�@

error_RMaF?

learning_rate_1�HG7_���I       6%�	T5}U���A�*;


total_loss���@

error_R�R?

learning_rate_1�HG7���VI       6%�	{}U���A�*;


total_loss� A

error_RWT?

learning_rate_1�HG7Y�I       6%�	4�}U���A�*;


total_loss}�@

error_R��A?

learning_rate_1�HG7�)�I       6%�	$~U���A�*;


total_loss1��@

error_R�4M?

learning_rate_1�HG7���I       6%�	�a~U���A�*;


total_loss�?�@

error_R_�Z?

learning_rate_1�HG7�09.I       6%�	��~U���A�*;


total_lossS��@

error_R_gR?

learning_rate_1�HG7����I       6%�	��~U���A�*;


total_lossc��@

error_R{�[?

learning_rate_1�HG7`���I       6%�	c.U���A�*;


total_loss%=�@

error_RuU?

learning_rate_1�HG7�y�I       6%�	qU���A�*;


total_loss��[@

error_RX�??

learning_rate_1�HG72�I       6%�	��U���A�*;


total_lossK�@

error_R8a+?

learning_rate_1�HG7�f�I       6%�	��U���A�*;


total_loss��@

error_R�z<?

learning_rate_1�HG7P�`�I       6%�	�?�U���A�*;


total_loss��A

error_R��=?

learning_rate_1�HG7L:�I       6%�	8$�U���A�*;


total_loss{��@

error_RCH?

learning_rate_1�HG7jP�[I       6%�	`p�U���A�*;


total_lossã�@

error_RJ�E?

learning_rate_1�HG7�w]aI       6%�	h��U���A�*;


total_lossB��@

error_RߜY?

learning_rate_1�HG7�VkI       6%�	��U���A�*;


total_lossTɊ@

error_R��Z?

learning_rate_1�HG7�-ԙI       6%�	�U�U���A�*;


total_loss4�@

error_R�w=?

learning_rate_1�HG7[��I       6%�	�U���A�*;


total_loss�U�@

error_R�+;?

learning_rate_1�HG7�jVI       6%�	��U���A�*;


total_loss鲤@

error_RƀP?

learning_rate_1�HG7�?l�I       6%�	p3�U���A�*;


total_loss(e@

error_R�J?

learning_rate_1�HG7���I       6%�	Ty�U���A�*;


total_lossdS�@

error_R�'G?

learning_rate_1�HG7���@I       6%�	���U���A�*;


total_loss���@

error_R6 C?

learning_rate_1�HG7�n(�I       6%�	�U���A�*;


total_lossw�~@

error_R�5\?

learning_rate_1�HG7n���I       6%�	�D�U���A�*;


total_loss��@

error_R��P?

learning_rate_1�HG7�fiI       6%�	���U���A�*;


total_lossM:�@

error_R�Y?

learning_rate_1�HG7`&I       6%�	�ІU���A�*;


total_loss8��@

error_RJ�S?

learning_rate_1�HG7T�bI       6%�	H$�U���A�*;


total_lossa��@

error_R��S?

learning_rate_1�HG7��U�I       6%�	p�U���A�*;


total_loss�W�@

error_R,gd?

learning_rate_1�HG7� 8I       6%�	ܵ�U���A�*;


total_loss��@

error_R�JD?

learning_rate_1�HG7���I       6%�	���U���A�*;


total_losseP�@

error_RߎB?

learning_rate_1�HG7h�{�I       6%�	d@�U���A�*;


total_loss���@

error_R��>?

learning_rate_1�HG7��I       6%�	҃�U���A�*;


total_loss�K�@

error_R�?V?

learning_rate_1�HG7����I       6%�	MʈU���A�*;


total_loss�̹@

error_R��@?

learning_rate_1�HG7�ʷ�I       6%�	��U���A�*;


total_loss��@

error_R�%<?

learning_rate_1�HG73���I       6%�	�]�U���A�*;


total_loss�J�@

error_RedY?

learning_rate_1�HG7���I       6%�	���U���A�*;


total_losst�@

error_R=�T?

learning_rate_1�HG7ޢeI       6%�	/�U���A�*;


total_loss&e�@

error_Ra�G?

learning_rate_1�HG7?��dI       6%�	o4�U���A�*;


total_loss?�]@

error_RReG?

learning_rate_1�HG7)���I       6%�	�w�U���A�*;


total_loss6�@

error_RuN?

learning_rate_1�HG7��;I       6%�	�U���A�*;


total_loss��@

error_R~J?

learning_rate_1�HG7�5�YI       6%�	d*�U���A�*;


total_lossXA

error_R��Q?

learning_rate_1�HG7��?�I       6%�	�s�U���A�*;


total_loss��@

error_RO�7?

learning_rate_1�HG7_N�I       6%�	L��U���A�*;


total_loss�m@

error_R�EJ?

learning_rate_1�HG7�о�I       6%�	%�U���A�*;


total_loss��@

error_R��N?

learning_rate_1�HG7��1�I       6%�	J�U���A�*;


total_loss��@

error_RڳG?

learning_rate_1�HG7@��I       6%�	���U���A�*;


total_loss���@

error_R1�=?

learning_rate_1�HG7[��zI       6%�	YڌU���A�*;


total_loss&��@

error_R$�D?

learning_rate_1�HG7H���I       6%�	)�U���A�*;


total_loss߇�@

error_R%M?

learning_rate_1�HG7F!�NI       6%�	M��U���A�*;


total_loss���@

error_R��P?

learning_rate_1�HG7|n�uI       6%�	.ՍU���A�*;


total_lossZƌ@

error_R[�]?

learning_rate_1�HG7��I       6%�	�,�U���A�*;


total_loss �@

error_R?R?

learning_rate_1�HG7�,�I       6%�	���U���A�*;


total_loss/O�@

error_R��;?

learning_rate_1�HG7�R��I       6%�	&�U���A�*;


total_loss4$A

error_RS�Y?

learning_rate_1�HG7u1ljI       6%�	Q)�U���A�*;


total_lossq��@

error_RB>?

learning_rate_1�HG7�`g�I       6%�	�o�U���A�*;


total_loss�,�@

error_R7m?

learning_rate_1�HG7�V0�I       6%�	}��U���A�*;


total_loss
��@

error_R�w@?

learning_rate_1�HG7C��I       6%�	�'�U���A�*;


total_lossl�@

error_R-�N?

learning_rate_1�HG7��`I       6%�	�o�U���A�*;


total_loss���@

error_R��b?

learning_rate_1�HG7���I       6%�	|��U���A�*;


total_loss!�@

error_RNSU?

learning_rate_1�HG7	:�I       6%�	< �U���A�*;


total_loss�j�@

error_R!�B?

learning_rate_1�HG7��KI       6%�	�G�U���A�*;


total_loss$��@

error_RV�Q?

learning_rate_1�HG7��5I       6%�	���U���A�*;


total_loss�'�@

error_R��E?

learning_rate_1�HG7wQI       6%�	8֑U���A�*;


total_lossO�@

error_R�0@?

learning_rate_1�HG7��e:I       6%�	�U���A�*;


total_loss��@

error_R�P?

learning_rate_1�HG7+��|I       6%�	>i�U���A�*;


total_loss�Z�@

error_Rc9N?

learning_rate_1�HG7���I       6%�	���U���A�*;


total_loss�X�@

error_RH?

learning_rate_1�HG72'�I       6%�	�.�U���A�*;


total_loss���@

error_R�-3?

learning_rate_1�HG7��{�I       6%�	y�U���A�*;


total_loss$* A

error_R�8Z?

learning_rate_1�HG7���7I       6%�	,ГU���A�*;


total_loss�k�@

error_R֡r?

learning_rate_1�HG73ңI       6%�	_�U���A�*;


total_lossm��@

error_R�JT?

learning_rate_1�HG7�Vt�I       6%�	�g�U���A�*;


total_loss��@

error_Rd�S?

learning_rate_1�HG7ߴMSI       6%�	o��U���A�*;


total_loss}z�@

error_R�yS?

learning_rate_1�HG7j!sI       6%�	3��U���A�*;


total_loss�!�@

error_RƟW?

learning_rate_1�HG7>#�cI       6%�	s<�U���A�*;


total_loss}y�@

error_R�
=?

learning_rate_1�HG7긶�I       6%�	m��U���A�*;


total_loss�i�@

error_R<uF?

learning_rate_1�HG7���_I       6%�	�ǕU���A�*;


total_loss�р@

error_R�T?

learning_rate_1�HG7@�QI       6%�	K�U���A�*;


total_lossh`�@

error_R=�N?

learning_rate_1�HG7f�,�I       6%�	qS�U���A�*;


total_loss��@

error_R,�W?

learning_rate_1�HG7�{��I       6%�	|��U���A�*;


total_loss�W�@

error_R�0?

learning_rate_1�HG7�`S�I       6%�	���U���A�*;


total_loss���@

error_R
�8?

learning_rate_1�HG7���I       6%�	�&�U���A�*;


total_lossj��@

error_RZ�P?

learning_rate_1�HG7����I       6%�	�j�U���A�*;


total_loss�Ԭ@

error_R�q[?

learning_rate_1�HG7u�I       6%�	y��U���A�*;


total_loss���@

error_Rq�S?

learning_rate_1�HG7#�I       6%�	���U���A�*;


total_loss-��@

error_R]�R?

learning_rate_1�HG7 ��ZI       6%�	O>�U���A�*;


total_lossZ��@

error_R��K?

learning_rate_1�HG7�TI       6%�	��U���A�*;


total_loss�t�@

error_Rag?

learning_rate_1�HG7���%I       6%�	�ĘU���A�*;


total_loss�Y�@

error_RM�I?

learning_rate_1�HG7� �I       6%�	��U���A�*;


total_loss���@

error_Ri_9?

learning_rate_1�HG7�A\EI       6%�	�K�U���A�*;


total_losso�@

error_R�GP?

learning_rate_1�HG7��a�I       6%�	���U���A�*;


total_lossH��@

error_R.�P?

learning_rate_1�HG7Z�aI       6%�	ؙU���A�*;


total_loss%6�@

error_R�L?

learning_rate_1�HG7S�I       6%�	V�U���A�*;


total_loss��A

error_RcQG?

learning_rate_1�HG7O�I       6%�	�\�U���A�*;


total_lossb�@

error_R��<?

learning_rate_1�HG7�� �I       6%�	���U���A�*;


total_loss�:�@

error_R�|A?

learning_rate_1�HG7��<�I       6%�	��U���A�*;


total_loss�a�@

error_R_K?

learning_rate_1�HG7��I       6%�	/Q�U���A�*;


total_lossꢸ@

error_R��J?

learning_rate_1�HG7�ϧI       6%�	 ��U���A�*;


total_loss6�@

error_R�-N?

learning_rate_1�HG7���@I       6%�	LٛU���A�*;


total_loss\��@

error_R��L?

learning_rate_1�HG7�o��I       6%�	��U���A�*;


total_loss63�@

error_R�7W?

learning_rate_1�HG7N�KI       6%�	@^�U���A�*;


total_loss�n�@

error_R�6:?

learning_rate_1�HG7��I       6%�	���U���A�*;


total_loss�k�@

error_R��H?

learning_rate_1�HG7/���I       6%�	G�U���A�*;


total_loss�K�@

error_R)�H?

learning_rate_1�HG7�VėI       6%�	�&�U���A�*;


total_loss���@

error_R��I?

learning_rate_1�HG7A�4I       6%�	�m�U���A�*;


total_loss���@

error_R�e?

learning_rate_1�HG7���I       6%�	Ʋ�U���A�*;


total_loss��@

error_R g@?

learning_rate_1�HG7`���I       6%�	��U���A�*;


total_loss�UA

error_R�^?

learning_rate_1�HG7Sh*�I       6%�	�e�U���A�*;


total_loss�@

error_RR�N?

learning_rate_1�HG7X���I       6%�	�U���A�*;


total_loss�^A

error_R��O?

learning_rate_1�HG7����I       6%�	 �U���A�*;


total_lossʽ�@

error_RM�N?

learning_rate_1�HG7�1�`I       6%�	BY�U���A�*;


total_loss�o@

error_R}�<?

learning_rate_1�HG7��Y�I       6%�	Ө�U���A�*;


total_loss�d�@

error_R�4]?

learning_rate_1�HG7SЂ	I       6%�	m�U���A�*;


total_loss
|�@

error_RT�h?

learning_rate_1�HG7���	I       6%�	v7�U���A�*;


total_lossȎ�@

error_R �Z?

learning_rate_1�HG7�pI       6%�	ez�U���A�*;


total_loss�J�@

error_R�>?

learning_rate_1�HG7IH�I       6%�	2��U���A�*;


total_lossA0�@

error_R�IB?

learning_rate_1�HG7ة��I       6%�	��U���A�*;


total_loss�H�@

error_R�VC?

learning_rate_1�HG7���I       6%�	�I�U���A�*;


total_loss]�@

error_RצO?

learning_rate_1�HG7���I       6%�	
��U���A�*;


total_loss���@

error_R��;?

learning_rate_1�HG7��NeI       6%�	9աU���A�*;


total_lossxL�@

error_R��T?

learning_rate_1�HG7���'I       6%�	��U���A�*;


total_lossJ��@

error_R(S?

learning_rate_1�HG7Sz]I       6%�	qd�U���A�*;


total_loss@

error_R�YT?

learning_rate_1�HG7��$~I       6%�	���U���A�*;


total_loss���@

error_R]LB?

learning_rate_1�HG7f�zI       6%�	J��U���A�*;


total_loss���@

error_RZD?

learning_rate_1�HG7�K��I       6%�	�=�U���A�*;


total_loss���@

error_Rr�G?

learning_rate_1�HG7z^ReI       6%�	���U���A�*;


total_lossxb�@

error_R�LO?

learning_rate_1�HG7�|��I       6%�	�ʣU���A�*;


total_loss�e�@

error_RșJ?

learning_rate_1�HG7KNH)I       6%�		�U���A�*;


total_loss��@

error_R��I?

learning_rate_1�HG7#�q�I       6%�	�N�U���A�*;


total_lossC�@

error_R
�P?

learning_rate_1�HG7M��I       6%�	���U���A�*;


total_loss�:�@

error_Rs0J?

learning_rate_1�HG7�M*VI       6%�	\դU���A�*;


total_loss�؀@

error_R8�G?

learning_rate_1�HG7J��I       6%�	��U���A�*;


total_loss���@

error_RZV?

learning_rate_1�HG7�:I       6%�	c�U���A�*;


total_loss�"�@

error_R}GQ?

learning_rate_1�HG7�^�I       6%�	��U���A�*;


total_loss&E�@

error_R�qD?

learning_rate_1�HG7�B}I       6%�	���U���A�*;


total_loss/��@

error_RM�N?

learning_rate_1�HG7�U��I       6%�	W1�U���A�*;


total_loss�oA

error_Ra,T?

learning_rate_1�HG7\�ڲI       6%�	�{�U���A�*;


total_loss��@

error_R��F?

learning_rate_1�HG7��jI       6%�	JƦU���A�*;


total_loss�؇@

error_R��P?

learning_rate_1�HG7�?�I       6%�	"�U���A�*;


total_loss���@

error_R�Fh?

learning_rate_1�HG7���I       6%�	Y_�U���A�*;


total_loss���@

error_RζT?

learning_rate_1�HG7H�V\I       6%�	5��U���A�*;


total_lossh �@

error_R�]?

learning_rate_1�HG7��I       6%�	g��U���A�*;


total_loss���@

error_R��R?

learning_rate_1�HG7�A�I       6%�	i<�U���A�*;


total_lossA�@

error_RM~V?

learning_rate_1�HG7+�I       6%�	�~�U���A�*;


total_lossc@

error_R)9]?

learning_rate_1�HG7k��I       6%�	y¨U���A�*;


total_loss�\�@

error_RF�T?

learning_rate_1�HG76Y>�I       6%�	)�U���A�*;


total_loss>ގ@

error_R�#??

learning_rate_1�HG7g}��I       6%�	�I�U���A�*;


total_loss�'�@

error_R��U?

learning_rate_1�HG7n
�yI       6%�	O��U���A�*;


total_loss�͸@

error_Rr~A?

learning_rate_1�HG7��}�I       6%�	�өU���A�*;


total_loss�r@

error_R\�P?

learning_rate_1�HG7J�\[I       6%�	��U���A�*;


total_loss �@

error_R�G?

learning_rate_1�HG7%,9I       6%�	�]�U���A�*;


total_loss�*�@

error_R��X?

learning_rate_1�HG7Η,�I       6%�	@��U���A�*;


total_losso,	A

error_R�BP?

learning_rate_1�HG7��I       6%�	f��U���A�*;


total_loss�@

error_RH�E?

learning_rate_1�HG79��I       6%�	�G�U���A�*;


total_loss@��@

error_R=�I?

learning_rate_1�HG7��4#I       6%�	ې�U���A�*;


total_loss��@

error_Rw�M?

learning_rate_1�HG7b�b|I       6%�	G׫U���A�*;


total_loss] A

error_R�C?

learning_rate_1�HG7�/f�I       6%�	��U���A�*;


total_loss6?�@

error_R�>?

learning_rate_1�HG7�� �I       6%�	-g�U���A�*;


total_loss���@

error_R��U?

learning_rate_1�HG76�X(I       6%�	K��U���A�*;


total_losstw�@

error_R�U?

learning_rate_1�HG7?�,�I       6%�	_�U���A�*;


total_lossA�@

error_RTCI?

learning_rate_1�HG7�[MI       6%�	!>�U���A�*;


total_loss��@

error_R��O?

learning_rate_1�HG7h�I       6%�	Z��U���A�*;


total_loss� �@

error_RdpO?

learning_rate_1�HG7#�e8I       6%�	���U���A�*;


total_loss�Ʒ@

error_R;�>?

learning_rate_1�HG7��]�I       6%�	:�U���A�*;


total_lossVqt@

error_R�<\?

learning_rate_1�HG7W�I       6%�	��U���A�	*;


total_loss S�@

error_R�:i?

learning_rate_1�HG7��_I       6%�	�ĮU���A�	*;


total_loss�O�@

error_R��G?

learning_rate_1�HG7���I       6%�	�	�U���A�	*;


total_lossAY�@

error_R[�I?

learning_rate_1�HG7}>W�I       6%�	pQ�U���A�	*;


total_loss#�A

error_R kI?

learning_rate_1�HG7_q>?I       6%�	���U���A�	*;


total_lossQ�@

error_R�}A?

learning_rate_1�HG7�0ׂI       6%�	ݯU���A�	*;


total_lossw\�@

error_R!�Z?

learning_rate_1�HG7�^�I       6%�	8 �U���A�	*;


total_losso0�@

error_R
h=?

learning_rate_1�HG70QڸI       6%�	�c�U���A�	*;


total_loss�?�@

error_R�`?

learning_rate_1�HG7%/αI       6%�	���U���A�	*;


total_loss�a�@

error_RW�a?

learning_rate_1�HG7�h�I       6%�	��U���A�	*;


total_loss��@

error_R�b?

learning_rate_1�HG7)M�cI       6%�	R2�U���A�	*;


total_loss��@

error_ReRV?

learning_rate_1�HG7h�I       6%�	�{�U���A�	*;


total_loss�X�@

error_R*HD?

learning_rate_1�HG7���I       6%�	�ıU���A�	*;


total_loss[��@

error_R��;?

learning_rate_1�HG71�tI       6%�	�	�U���A�	*;


total_lossJ:�@

error_R�B?

learning_rate_1�HG7p�K�I       6%�	>N�U���A�	*;


total_loss�`�@

error_RX�G?

learning_rate_1�HG7��I       6%�	Ò�U���A�	*;


total_lossC��@

error_R�I?

learning_rate_1�HG7��R�I       6%�	�ղU���A�	*;


total_losss��@

error_R�G?

learning_rate_1�HG7��HAI       6%�	�U���A�	*;


total_loss\Ш@

error_R�5I?

learning_rate_1�HG7�.SI       6%�	�d�U���A�	*;


total_loss��@

error_RܛH?

learning_rate_1�HG7��rI       6%�	f��U���A�	*;


total_loss=�@

error_R�hV?

learning_rate_1�HG7�7�I       6%�	���U���A�	*;


total_lossX��@

error_R�eQ?

learning_rate_1�HG7��bI       6%�	�:�U���A�	*;


total_loss<��@

error_R�*\?

learning_rate_1�HG7jX�II       6%�	B}�U���A�	*;


total_loss���@

error_R��M?

learning_rate_1�HG7(t�I       6%�	ǴU���A�	*;


total_loss��~@

error_R��S?

learning_rate_1�HG7�Fw7I       6%�	�
�U���A�	*;


total_loss���@

error_R$�X?

learning_rate_1�HG7A���I       6%�	ZR�U���A�	*;


total_loss��@

error_R�GU?

learning_rate_1�HG7F��2I       6%�	���U���A�	*;


total_lossE��@

error_R=?

learning_rate_1�HG7�z�I       6%�	�׵U���A�	*;


total_lossli�@

error_Rq�[?

learning_rate_1�HG7܅˿I       6%�	W�U���A�	*;


total_loss���@

error_R �=?

learning_rate_1�HG7N�nI       6%�	�c�U���A�	*;


total_loss�j@

error_R��7?

learning_rate_1�HG7g�I       6%�	]��U���A�	*;


total_loss��@

error_R	nT?

learning_rate_1�HG7q�AI       6%�	���U���A�	*;


total_lossϚ�@

error_R\�??

learning_rate_1�HG7y~7I       6%�	�D�U���A�	*;


total_lossv�@

error_R�%U?

learning_rate_1�HG7���I       6%�	Q��U���A�	*;


total_loss��@

error_Rv�F?

learning_rate_1�HG7FjI       6%�	�ʷU���A�	*;


total_loss���@

error_R�TA?

learning_rate_1�HG7�0�JI       6%�	B�U���A�	*;


total_loss�N�@

error_R`'Q?

learning_rate_1�HG7R�KI       6%�	�U�U���A�	*;


total_loss��@

error_R�ZQ?

learning_rate_1�HG7'��I       6%�	U��U���A�	*;


total_loss@��@

error_R�zF?

learning_rate_1�HG71�&[I       6%�	P߸U���A�	*;


total_loss���@

error_R��@?

learning_rate_1�HG7�-��I       6%�	$�U���A�	*;


total_loss{+Y@

error_RT�X?

learning_rate_1�HG7�/7�I       6%�	i�U���A�	*;


total_loss�6v@

error_R��:?

learning_rate_1�HG7 ^A�I       6%�	᭹U���A�	*;


total_loss\��@

error_RAuF?

learning_rate_1�HG7}F�I       6%�	��U���A�	*;


total_loss q�@

error_R.�T?

learning_rate_1�HG7��ЯI       6%�	15�U���A�	*;


total_loss�ʧ@

error_R�:?

learning_rate_1�HG7��F�I       6%�	�y�U���A�	*;


total_loss���@

error_R">?

learning_rate_1�HG7�U_I       6%�	޺U���A�	*;


total_loss�N�@

error_RW�Z?

learning_rate_1�HG7|��I       6%�	�2�U���A�	*;


total_lossM_�@

error_R�Q?

learning_rate_1�HG7�k�I       6%�	�w�U���A�	*;


total_lossz7 A

error_RqLD?

learning_rate_1�HG7g���I       6%�	���U���A�	*;


total_loss�2�@

error_RD�D?

learning_rate_1�HG7��AI       6%�	���U���A�	*;


total_loss_��@

error_R�W?

learning_rate_1�HG7�m�I       6%�	�@�U���A�	*;


total_lossi  A

error_RA?

learning_rate_1�HG7N���I       6%�	��U���A�	*;


total_loss�Z�@

error_RS?

learning_rate_1�HG7Vp�I       6%�	YʼU���A�	*;


total_loss�ό@

error_R�R?

learning_rate_1�HG7vD�
I       6%�	��U���A�	*;


total_loss�ү@

error_R��g?

learning_rate_1�HG7>��I       6%�	R�U���A�	*;


total_loss�s�@

error_R/�L?

learning_rate_1�HG7��JI       6%�	.��U���A�	*;


total_loss��@

error_R��f?

learning_rate_1�HG7F�ĵI       6%�	V�U���A�	*;


total_loss���@

error_RHF?

learning_rate_1�HG7p�c�I       6%�	w%�U���A�	*;


total_loss.Q�@

error_R�R?

learning_rate_1�HG7�U�I       6%�	.h�U���A�	*;


total_loss`��@

error_R��P?

learning_rate_1�HG7�	�7I       6%�	M��U���A�	*;


total_loss�@

error_R!3J?

learning_rate_1�HG7�!�I       6%�	���U���A�	*;


total_lossҼ�@

error_R1JF?

learning_rate_1�HG7 ��I       6%�	�E�U���A�	*;


total_loss6�A

error_REbR?

learning_rate_1�HG7�=7_I       6%�	]��U���A�	*;


total_loss& "A

error_Rj�O?

learning_rate_1�HG7L#�I       6%�	�ѿU���A�	*;


total_lossId�@

error_R��I?

learning_rate_1�HG7���MI       6%�	F �U���A�	*;


total_loss�q�@

error_R?�I?

learning_rate_1�HG7��zI       6%�	nk�U���A�	*;


total_loss�3�@

error_R�??

learning_rate_1�HG7�E>I       6%�	���U���A�	*;


total_loss\��@

error_R8�<?

learning_rate_1�HG7t���I       6%�	���U���A�	*;


total_lossd;�@

error_RI|J?

learning_rate_1�HG7�d3I       6%�	�G�U���A�	*;


total_loss�!�@

error_R6�@?

learning_rate_1�HG7"[޲I       6%�	Ռ�U���A�	*;


total_loss���@

error_R��F?

learning_rate_1�HG7v�`?I       6%�	f��U���A�	*;


total_loss[�@

error_R�L?

learning_rate_1�HG7á�zI       6%�	��U���A�	*;


total_loss�$A

error_R�\?

learning_rate_1�HG7�EPTI       6%�	Ib�U���A�	*;


total_loss�r�@

error_R2�C?

learning_rate_1�HG7 ��I       6%�	~��U���A�	*;


total_loss%��@

error_R,E?

learning_rate_1�HG7)���I       6%�	���U���A�	*;


total_losst�@

error_R�K?

learning_rate_1�HG7QQ��I       6%�	!;�U���A�	*;


total_loss���@

error_RřB?

learning_rate_1�HG7�s%mI       6%�	Y}�U���A�	*;


total_loss�@

error_R�[?

learning_rate_1�HG7��I       6%�	���U���A�	*;


total_loss�y�@

error_R��f?

learning_rate_1�HG78�$I       6%�	e�U���A�	*;


total_loss
u�@

error_R3�Q?

learning_rate_1�HG7��cI       6%�	�D�U���A�	*;


total_loss�ݞ@

error_R�:Y?

learning_rate_1�HG7�lݭI       6%�	ͅ�U���A�	*;


total_loss��@

error_R�[Q?

learning_rate_1�HG7�}63I       6%�	i��U���A�	*;


total_loss�ܨ@

error_R\�<?

learning_rate_1�HG7�%�I       6%�	^�U���A�	*;


total_loss�v�@

error_R�I?

learning_rate_1�HG7�E��I       6%�	V]�U���A�	*;


total_lossJ�@

error_R��I?

learning_rate_1�HG7h>�I       6%�	��U���A�	*;


total_loss�s�@

error_R��]?

learning_rate_1�HG7�I       6%�	:��U���A�	*;


total_lossr>�@

error_R��M?

learning_rate_1�HG7�hKI       6%�	)�U���A�	*;


total_loss�l8@

error_Rx�T?

learning_rate_1�HG7�X�I       6%�	�o�U���A�	*;


total_lossч�@

error_R�Y?

learning_rate_1�HG7�rűI       6%�	ɳ�U���A�	*;


total_losss��@

error_R��Z?

learning_rate_1�HG7b�I       6%�	��U���A�	*;


total_loss��@

error_R�mJ?

learning_rate_1�HG7sn^�I       6%�	!=�U���A�	*;


total_loss7Ç@

error_RT3?

learning_rate_1�HG7E�uI       6%�	���U���A�	*;


total_loss��@

error_Rd�I?

learning_rate_1�HG7���2I       6%�	��U���A�	*;


total_lossM��@

error_R=`U?

learning_rate_1�HG7�/�I       6%�	6�U���A�	*;


total_lossm�@

error_Rn�J?

learning_rate_1�HG7g��JI       6%�	�Q�U���A�	*;


total_loss�y�@

error_R.?T?

learning_rate_1�HG7��C'I       6%�	���U���A�	*;


total_loss�L�@

error_RqhK?

learning_rate_1�HG7����I       6%�	���U���A�	*;


total_loss�)�@

error_RcAE?

learning_rate_1�HG7�k�qI       6%�	�&�U���A�	*;


total_loss\A

error_R�H?

learning_rate_1�HG7�$��I       6%�	k�U���A�	*;


total_loss���@

error_R)]?

learning_rate_1�HG7Cz+�I       6%�	���U���A�	*;


total_loss(��@

error_RɎT?

learning_rate_1�HG7�2�9I       6%�	3��U���A�	*;


total_lossO�@

error_R}pO?

learning_rate_1�HG7����I       6%�	A5�U���A�	*;


total_loss%��@

error_R1JG?

learning_rate_1�HG7�yW�I       6%�	1x�U���A�	*;


total_lossqu�@

error_R��B?

learning_rate_1�HG7Db{�I       6%�	.��U���A�	*;


total_loss��@

error_R�OM?

learning_rate_1�HG7L�I       6%�	5#�U���A�	*;


total_loss/�{@

error_RlQ?

learning_rate_1�HG7�S�I       6%�	o�U���A�	*;


total_loss/��@

error_RWU?

learning_rate_1�HG7br�TI       6%�	]��U���A�	*;


total_loss�И@

error_Ri^`?

learning_rate_1�HG7���OI       6%�	E �U���A�	*;


total_loss�ǚ@

error_R��=?

learning_rate_1�HG7�]RI       6%�	E�U���A�	*;


total_loss�}�@

error_Rx�W?

learning_rate_1�HG7�uWI       6%�	��U���A�	*;


total_loss��@

error_R8X?

learning_rate_1�HG7�R"�I       6%�	��U���A�	*;


total_lossM�@

error_R�JW?

learning_rate_1�HG7 �[I       6%�	��U���A�	*;


total_lossrѾ@

error_Rv�B?

learning_rate_1�HG7=+�I       6%�	�u�U���A�	*;


total_loss�ל@

error_RϛG?

learning_rate_1�HG7.�jI       6%�	���U���A�	*;


total_loss���@

error_R�+7?

learning_rate_1�HG7�o0I       6%�	n�U���A�	*;


total_loss�!�@

error_R�H?

learning_rate_1�HG7��m�I       6%�	Kl�U���A�	*;


total_loss�O�@

error_R�8X?

learning_rate_1�HG7����I       6%�	���U���A�	*;


total_loss4\�@

error_R
7O?

learning_rate_1�HG7|JT�I       6%�	R�U���A�	*;


total_loss�k�@

error_R/D?

learning_rate_1�HG7�EI       6%�	�V�U���A�	*;


total_loss�!�@

error_R][?

learning_rate_1�HG7�̸�I       6%�	���U���A�	*;


total_loss��@

error_R S?

learning_rate_1�HG7Nu�I       6%�	���U���A�	*;


total_lossx�@

error_R�^Q?

learning_rate_1�HG7|\I       6%�	Q"�U���A�	*;


total_loss昤@

error_R*HX?

learning_rate_1�HG7��I       6%�	�g�U���A�	*;


total_loss��@

error_R�c?

learning_rate_1�HG7v��I       6%�	��U���A�	*;


total_loss���@

error_R�]?

learning_rate_1�HG7���II       6%�	��U���A�	*;


total_lossڝ�@

error_R�H?

learning_rate_1�HG7}`�aI       6%�	�1�U���A�	*;


total_loss��A

error_R�N?

learning_rate_1�HG7j�Y�I       6%�	Oy�U���A�	*;


total_lossڔ@

error_RL�G?

learning_rate_1�HG7yv_gI       6%�	z��U���A�	*;


total_loss���@

error_R��6?

learning_rate_1�HG7��*I       6%�	T�U���A�
*;


total_loss��@

error_R�~p?

learning_rate_1�HG7�<�I       6%�	]�U���A�
*;


total_lossx@

error_R�:?

learning_rate_1�HG7�b�HI       6%�	f��U���A�
*;


total_loss(�A

error_R4}B?

learning_rate_1�HG78�)�I       6%�	���U���A�
*;


total_loss_y�@

error_R��D?

learning_rate_1�HG7K�VI       6%�	}/�U���A�
*;


total_loss���@

error_R��:?

learning_rate_1�HG7�l\SI       6%�	�q�U���A�
*;


total_loss�ծ@

error_R��n?

learning_rate_1�HG7](��I       6%�	���U���A�
*;


total_loss���@

error_R��R?

learning_rate_1�HG7�KņI       6%�	��U���A�
*;


total_loss�v�@

error_R|M?

learning_rate_1�HG7
�ܪI       6%�	@�U���A�
*;


total_lossȔ�@

error_R&�B?

learning_rate_1�HG7�P�6I       6%�	��U���A�
*;


total_lossf��@

error_R��J?

learning_rate_1�HG7��6I       6%�	i��U���A�
*;


total_loss���@

error_ROuD?

learning_rate_1�HG7<�WqI       6%�	��U���A�
*;


total_loss�*�@

error_R�;9?

learning_rate_1�HG7#��I       6%�	4e�U���A�
*;


total_loss&��@

error_R�a?

learning_rate_1�HG7��I       6%�	ڬ�U���A�
*;


total_loss��@

error_R��J?

learning_rate_1�HG77_	^I       6%�	z��U���A�
*;


total_loss
j�@

error_R�L?

learning_rate_1�HG7�*��I       6%�	�=�U���A�
*;


total_lossܧ�@

error_R
_`?

learning_rate_1�HG7��I       6%�	@��U���A�
*;


total_lossL�A

error_RœN?

learning_rate_1�HG7��ZI       6%�	���U���A�
*;


total_lossbe�@

error_R.J?

learning_rate_1�HG7S��I       6%�	p�U���A�
*;


total_loss=�@

error_Rs9B?

learning_rate_1�HG78PI       6%�	I�U���A�
*;


total_loss촾@

error_R�cX?

learning_rate_1�HG77��I       6%�	p��U���A�
*;


total_loss�0�@

error_R8�;?

learning_rate_1�HG7�LxI       6%�	(��U���A�
*;


total_losszF�@

error_R�*K?

learning_rate_1�HG7�3��I       6%�	��U���A�
*;


total_loss<�@

error_R��V?

learning_rate_1�HG7:yS�I       6%�	^�U���A�
*;


total_loss�8�@

error_RRY?

learning_rate_1�HG7m��I       6%�	���U���A�
*;


total_loss�Cr@

error_R-�M?

learning_rate_1�HG7���nI       6%�	���U���A�
*;


total_lossڛ�@

error_RIIH?

learning_rate_1�HG7��I       6%�	I.�U���A�
*;


total_lossR��@

error_R�K?

learning_rate_1�HG7RɛI       6%�	Sr�U���A�
*;


total_lossO��@

error_Rx c?

learning_rate_1�HG7f�<?I       6%�	6��U���A�
*;


total_loss��A

error_R
%U?

learning_rate_1�HG7�sNI       6%�	'��U���A�
*;


total_loss�;�@

error_R��S?

learning_rate_1�HG7Z?��I       6%�	?�U���A�
*;


total_losst��@

error_RE�=?

learning_rate_1�HG7GX�I       6%�	���U���A�
*;


total_lossR�@

error_R�T?

learning_rate_1�HG7�gdI       6%�	���U���A�
*;


total_losswR�@

error_Rj�X?

learning_rate_1�HG7fs��I       6%�	�0�U���A�
*;


total_lossA�@

error_R�G?

learning_rate_1�HG7�ֻ�I       6%�	$z�U���A�
*;


total_loss���@

error_RxfX?

learning_rate_1�HG7c�1�I       6%�	���U���A�
*;


total_loss�U�@

error_RR?

learning_rate_1�HG78��I       6%�	��U���A�
*;


total_loss��@

error_R\�F?

learning_rate_1�HG7w밽I       6%�	WM�U���A�
*;


total_loss��@

error_R��C?

learning_rate_1�HG7�SV'I       6%�	i��U���A�
*;


total_loss�S�@

error_Rێ`?

learning_rate_1�HG7�bweI       6%�	c��U���A�
*;


total_loss�G�@

error_R�C?

learning_rate_1�HG7��UI       6%�	k�U���A�
*;


total_loss���@

error_R�|O?

learning_rate_1�HG7����I       6%�	�]�U���A�
*;


total_loss�5�@

error_R��f?

learning_rate_1�HG7aD|I       6%�	 ��U���A�
*;


total_lossyA

error_R/PP?

learning_rate_1�HG7l��I       6%�	W��U���A�
*;


total_loss4[@

error_Rl�V?

learning_rate_1�HG7؟�I       6%�	�0�U���A�
*;


total_lossD�j@

error_R�F?

learning_rate_1�HG79��I       6%�	H�U���A�
*;


total_loss�Q�@

error_R�P?

learning_rate_1�HG77!QnI       6%�	���U���A�
*;


total_lossl5�@

error_RH�S?

learning_rate_1�HG7.���I       6%�	!�U���A�
*;


total_loss���@

error_R��R?

learning_rate_1�HG7d�U�I       6%�	�R�U���A�
*;


total_loss���@

error_R|�I?

learning_rate_1�HG7�ZvI       6%�	4��U���A�
*;


total_lossxA

error_RqK?

learning_rate_1�HG7���I       6%�	-��U���A�
*;


total_loss���@

error_RфF?

learning_rate_1�HG7�%I       6%�	��U���A�
*;


total_loss�(A

error_R� Q?

learning_rate_1�HG7����I       6%�	Ad�U���A�
*;


total_loss���@

error_RCNR?

learning_rate_1�HG7%�4GI       6%�	���U���A�
*;


total_lossX�A

error_R��H?

learning_rate_1�HG7~Fh2I       6%�	���U���A�
*;


total_loss�jA

error_Rv�C?

learning_rate_1�HG7N�ƌI       6%�	4�U���A�
*;


total_losss]�@

error_RJV?

learning_rate_1�HG7�V�I       6%�	U}�U���A�
*;


total_loss��F@

error_Rd�K?

learning_rate_1�HG7�w(�I       6%�	���U���A�
*;


total_lossq�@

error_R�<E?

learning_rate_1�HG7��I       6%�	��U���A�
*;


total_loss��@

error_R�dQ?

learning_rate_1�HG7��H?I       6%�	�U�U���A�
*;


total_loss�H�@

error_RϣJ?

learning_rate_1�HG7�JI       6%�	���U���A�
*;


total_loss�G�@

error_R�1I?

learning_rate_1�HG7<�.I       6%�	=��U���A�
*;


total_lossO��@

error_ROD?

learning_rate_1�HG7o��I       6%�	�%�U���A�
*;


total_loss��	A

error_Rv?O?

learning_rate_1�HG7g�k�I       6%�	gh�U���A�
*;


total_loss�J�@

error_R\4L?

learning_rate_1�HG7��fI       6%�	���U���A�
*;


total_loss]��@

error_R�??

learning_rate_1�HG7����I       6%�	u��U���A�
*;


total_loss_�@

error_R&O?

learning_rate_1�HG7G�&I       6%�	�5�U���A�
*;


total_loss��@

error_RCxU?

learning_rate_1�HG7�I{LI       6%�	�{�U���A�
*;


total_loss,��@

error_R:�E?

learning_rate_1�HG7����I       6%�	��U���A�
*;


total_loss5�@

error_R39O?

learning_rate_1�HG7�z�I       6%�	� �U���A�
*;


total_lossK�A

error_RiGm?

learning_rate_1�HG7�w��I       6%�	�H�U���A�
*;


total_loss?�@

error_RpB?

learning_rate_1�HG7�.WI       6%�	W��U���A�
*;


total_loss�@

error_R״Q?

learning_rate_1�HG7
U��I       6%�	���U���A�
*;


total_loss���@

error_R�X?

learning_rate_1�HG7|�-I       6%�	I�U���A�
*;


total_loss���@

error_R��<?

learning_rate_1�HG7
t�I       6%�	^�U���A�
*;


total_loss�pA

error_Rv�:?

learning_rate_1�HG7��I       6%�	���U���A�
*;


total_loss���@

error_R1�N?

learning_rate_1�HG7F./�I       6%�	���U���A�
*;


total_loss��y@

error_R��]?

learning_rate_1�HG7�1T�I       6%�	�-�U���A�
*;


total_loss�@

error_R�=\?

learning_rate_1�HG7L�I       6%�	�r�U���A�
*;


total_loss�yA

error_R�:?

learning_rate_1�HG7{���I       6%�	��U���A�
*;


total_loss�Ӑ@

error_RJS?

learning_rate_1�HG7r�ډI       6%�	��U���A�
*;


total_loss�Ω@

error_R�D?

learning_rate_1�HG7ǯ.lI       6%�	vM�U���A�
*;


total_loss��@

error_ROd?

learning_rate_1�HG7���I       6%�	��U���A�
*;


total_loss��@

error_R�K?

learning_rate_1�HG7��_I       6%�	��U���A�
*;


total_loss�'�@

error_RT�]?

learning_rate_1�HG7� ��I       6%�	��U���A�
*;


total_lossm��@

error_R�c?

learning_rate_1�HG7���+I       6%�	/c�U���A�
*;


total_losss��@

error_R��H?

learning_rate_1�HG7؝�I       6%�	Q��U���A�
*;


total_lossl�@

error_RqLQ?

learning_rate_1�HG7m$�I       6%�	���U���A�
*;


total_loss�_�@

error_R�F?

learning_rate_1�HG7Xߜ�I       6%�	�2�U���A�
*;


total_lossi��@

error_R��R?

learning_rate_1�HG7�e�I       6%�	�v�U���A�
*;


total_losso��@

error_RdR?

learning_rate_1�HG7�<]�I       6%�	t��U���A�
*;


total_lossFۆ@

error_R�J?

learning_rate_1�HG7R*�nI       6%�	!'�U���A�
*;


total_loss��@

error_R��[?

learning_rate_1�HG7� ��I       6%�	�x�U���A�
*;


total_loss/o�@

error_R&�G?

learning_rate_1�HG7ݦ�I       6%�	~��U���A�
*;


total_loss?^�@

error_R�LM?

learning_rate_1�HG7F1#I       6%�	��U���A�
*;


total_loss���@

error_R��S?

learning_rate_1�HG7X,�hI       6%�	�T�U���A�
*;


total_loss�è@

error_R��O?

learning_rate_1�HG7W���I       6%�	���U���A�
*;


total_lossɏ�@

error_R�??

learning_rate_1�HG7s��I       6%�	3��U���A�
*;


total_loss7{�@

error_R&�\?

learning_rate_1�HG7�y,I       6%�	2�U���A�
*;


total_loss�a�@

error_RrJX?

learning_rate_1�HG7'DU�I       6%�	u��U���A�
*;


total_loss��@

error_R̦)?

learning_rate_1�HG7��{I       6%�	���U���A�
*;


total_loss_$�@

error_Rl�>?

learning_rate_1�HG7(eh�I       6%�	U�U���A�
*;


total_loss�N�@

error_R�J?

learning_rate_1�HG7�g�I       6%�	`s�U���A�
*;


total_loss���@

error_R�I?

learning_rate_1�HG7��SSI       6%�	P��U���A�
*;


total_loss�A

error_R�T?

learning_rate_1�HG7bL�I       6%�	��U���A�
*;


total_lossj��@

error_R�M?

learning_rate_1�HG7��~�I       6%�	�L�U���A�
*;


total_lossC��@

error_R��H?

learning_rate_1�HG7ʱ�I       6%�	���U���A�
*;


total_loss�x�@

error_R�E?

learning_rate_1�HG7
�L�I       6%�	*��U���A�
*;


total_loss���@

error_R�I?

learning_rate_1�HG7S	'�I       6%�	��U���A�
*;


total_loss^�@

error_R��@?

learning_rate_1�HG7�c�I       6%�	1Z�U���A�
*;


total_losscn�@

error_R�ZS?

learning_rate_1�HG7Y��I       6%�	���U���A�
*;


total_loss2-
A

error_RnrZ?

learning_rate_1�HG7����I       6%�	?��U���A�
*;


total_loss�	�@

error_R�hN?

learning_rate_1�HG7H�hI       6%�	&0�U���A�
*;


total_loss>A

error_R��I?

learning_rate_1�HG7 �X�I       6%�	�t�U���A�
*;


total_loss���@

error_Rq0S?

learning_rate_1�HG7���I       6%�	���U���A�
*;


total_loss̩�@

error_Ra�T?

learning_rate_1�HG7�GĚI       6%�	��U���A�
*;


total_loss���@

error_R#�E?

learning_rate_1�HG7@� 9I       6%�	�H�U���A�
*;


total_loss�ޢ@

error_RoY@?

learning_rate_1�HG7���$I       6%�	э�U���A�
*;


total_loss���@

error_R��E?

learning_rate_1�HG7o�
�I       6%�	��U���A�
*;


total_lossQۊ@

error_R�hK?

learning_rate_1�HG7T��I       6%�	��U���A�
*;


total_lossћ�@

error_R�B?

learning_rate_1�HG7��"I       6%�	�^�U���A�
*;


total_loss*�@

error_R �T?

learning_rate_1�HG7�?bI       6%�	��U���A�
*;


total_loss�a�@

error_RxU?

learning_rate_1�HG7�':I       6%�	���U���A�
*;


total_lossF��@

error_R��N?

learning_rate_1�HG7��%�I       6%�	2/�U���A�
*;


total_lossb�@

error_R��5?

learning_rate_1�HG7A&qdI       6%�	�q�U���A�
*;


total_loss��@

error_Rd�H?

learning_rate_1�HG7���I       6%�	���U���A�
*;


total_lossg�@

error_R��Y?

learning_rate_1�HG77�тI       6%�	B��U���A�
*;


total_loss��@

error_RTC?

learning_rate_1�HG7�b&I       6%�	:�U���A�
*;


total_loss�� A

error_R��O?

learning_rate_1�HG7��VI       6%�	4��U���A�*;


total_losswz�@

error_R�9J?

learning_rate_1�HG7M�ZI       6%�	��U���A�*;


total_loss���@

error_R}�D?

learning_rate_1�HG7��I       6%�	��U���A�*;


total_loss�@

error_R��K?

learning_rate_1�HG7JX`4I       6%�	�V�U���A�*;


total_loss�l�@

error_R�W<?

learning_rate_1�HG7���I       6%�	Y��U���A�*;


total_loss��@

error_RYP?

learning_rate_1�HG7�hu�I       6%�	���U���A�*;


total_loss��@

error_R�]N?

learning_rate_1�HG7\�Q3I       6%�	`4�U���A�*;


total_losshҚ@

error_Ra�Q?

learning_rate_1�HG7P�oI       6%�	�{�U���A�*;


total_loss�y�@

error_R��<?

learning_rate_1�HG7��[I       6%�	��U���A�*;


total_losseA

error_R{	D?

learning_rate_1�HG7��YI       6%�	��U���A�*;


total_loss��@

error_R��U?

learning_rate_1�HG7r7OI       6%�	�I�U���A�*;


total_loss��@

error_R�K?

learning_rate_1�HG7���I       6%�	���U���A�*;


total_lossA2q@

error_Rs-??

learning_rate_1�HG7^[-I       6%�	t��U���A�*;


total_loss}�@

error_R��_?

learning_rate_1�HG7��{I       6%�	��U���A�*;


total_loss��@

error_R
M?

learning_rate_1�HG7���4I       6%�	yW�U���A�*;


total_loss��A

error_RR�l?

learning_rate_1�HG7S2�I       6%�	D��U���A�*;


total_loss�@

error_R�UL?

learning_rate_1�HG72
4�I       6%�	���U���A�*;


total_loss���@

error_R��M?

learning_rate_1�HG7�ϏI       6%�	
!�U���A�*;


total_loss��@

error_R�+K?

learning_rate_1�HG7Yv�pI       6%�	�c�U���A�*;


total_loss8�@

error_RM�T?

learning_rate_1�HG7��I       6%�	���U���A�*;


total_loss��
A

error_R��@?

learning_rate_1�HG7%<��I       6%�	F
�U���A�*;


total_loss�e�@

error_RL�Y?

learning_rate_1�HG7מBII       6%�	�T�U���A�*;


total_loss�@

error_R��M?

learning_rate_1�HG7��3I       6%�	T��U���A�*;


total_loss#��@

error_RI?

learning_rate_1�HG7�דI       6%�	���U���A�*;


total_loss7}�@

error_RX�Z?

learning_rate_1�HG7�&I       6%�	�/�U���A�*;


total_lossE��@

error_R��V?

learning_rate_1�HG7l�S�I       6%�	�z�U���A�*;


total_loss��@

error_R��Q?

learning_rate_1�HG7���I       6%�	��U���A�*;


total_loss��@

error_R� -?

learning_rate_1�HG7{d�$I       6%�	V�U���A�*;


total_loss��@

error_RWV?

learning_rate_1�HG7J�-�I       6%�	�M�U���A�*;


total_loss���@

error_R��G?

learning_rate_1�HG7#W3�I       6%�	���U���A�*;


total_loss��@

error_RDXW?

learning_rate_1�HG7��xI       6%�	���U���A�*;


total_loss��@

error_R�R9?

learning_rate_1�HG7�9(�I       6%�	B �U���A�*;


total_loss&|�@

error_R�U?

learning_rate_1�HG7ج��I       6%�	�d�U���A�*;


total_lossס�@

error_R(G?

learning_rate_1�HG7����I       6%�	���U���A�*;


total_lossDs�@

error_R�HI?

learning_rate_1�HG7c,Z�I       6%�	���U���A�*;


total_loss&��@

error_R%�Y?

learning_rate_1�HG7��{�I       6%�	J.�U���A�*;


total_loss\�@

error_R�-f?

learning_rate_1�HG7�� I       6%�	�q�U���A�*;


total_loss�D�@

error_Rv�U?

learning_rate_1�HG7EK��I       6%�	k��U���A�*;


total_lossMu�@

error_R�tL?

learning_rate_1�HG7 L�GI       6%�	.��U���A�*;


total_loss���@

error_Rn�\?

learning_rate_1�HG7��c�I       6%�	�; V���A�*;


total_lossW­@

error_Rz�??

learning_rate_1�HG7����I       6%�	�} V���A�*;


total_lossO��@

error_R�;N?

learning_rate_1�HG7sn�I       6%�	'� V���A�*;


total_lossr}�@

error_R��M?

learning_rate_1�HG7޼V�I       6%�	2
V���A�*;


total_loss6ũ@

error_R&4B?

learning_rate_1�HG7AE+hI       6%�	<NV���A�*;


total_loss%��@

error_RC�L?

learning_rate_1�HG7|�vI       6%�	��V���A�*;


total_loss*��@

error_R,�D?

learning_rate_1�HG7����I       6%�	��V���A�*;


total_loss���@

error_R��M?

learning_rate_1�HG7��h�I       6%�	"V���A�*;


total_lossZN�@

error_R
j?

learning_rate_1�HG74��JI       6%�	�dV���A�*;


total_loss�@

error_R�*X?

learning_rate_1�HG7.NH�I       6%�	�V���A�*;


total_loss)N�@

error_Rv@?

learning_rate_1�HG7|9�|I       6%�	�V���A�*;


total_loss���@

error_R�xO?

learning_rate_1�HG7���tI       6%�	�3V���A�*;


total_loss윴@

error_R% :?

learning_rate_1�HG7Z؈BI       6%�	vV���A�*;


total_loss�)�@

error_RR�S?

learning_rate_1�HG7)v��I       6%�	Z�V���A�*;


total_loss�P�@

error_R�T?

learning_rate_1�HG7f�8fI       6%�	�V���A�*;


total_lossܩ@

error_R��V?

learning_rate_1�HG7}�P�I       6%�	�?V���A�*;


total_lossT�}@

error_R�B?

learning_rate_1�HG7��GeI       6%�	A�V���A�*;


total_loss̨@

error_Rh�A?

learning_rate_1�HG7��zkI       6%�	9�V���A�*;


total_loss�O�@

error_R��C?

learning_rate_1�HG7���&I       6%�	OV���A�*;


total_loss@��@

error_R��Q?

learning_rate_1�HG7��8I       6%�	�WV���A�*;


total_loss=#�@

error_R�S?

learning_rate_1�HG7}�I       6%�	��V���A�*;


total_loss�o�@

error_R �P?

learning_rate_1�HG7��d�I       6%�	��V���A�*;


total_loss`��@

error_R�wA?

learning_rate_1�HG76�.I       6%�	�2V���A�*;


total_loss��@

error_R��A?

learning_rate_1�HG7����I       6%�	*V���A�*;


total_loss�Ox@

error_R�P?

learning_rate_1�HG7mK �I       6%�	��V���A�*;


total_loss�A

error_R[,[?

learning_rate_1�HG7f��;I       6%�	�V���A�*;


total_lossz�@

error_R�_b?

learning_rate_1�HG75��I       6%�	X\V���A�*;


total_lossDO�@

error_R�}R?

learning_rate_1�HG7���[I       6%�	�V���A�*;


total_loss�X�@

error_R�Kf?

learning_rate_1�HG7���=I       6%�	m�V���A�*;


total_lossi>A

error_R�GG?

learning_rate_1�HG7%��1I       6%�	�-V���A�*;


total_lossΙ�@

error_Rv�X?

learning_rate_1�HG7�`�I       6%�	�pV���A�*;


total_loss�A

error_R\$W?

learning_rate_1�HG7{�I       6%�	�V���A�*;


total_loss���@

error_R�Q?

learning_rate_1�HG7�u�9I       6%�	�	V���A�*;


total_loss1�@

error_R=%I?

learning_rate_1�HG7U�	�I       6%�	�Q	V���A�*;


total_loss!/�@

error_R�b?

learning_rate_1�HG7���I       6%�	Ԟ	V���A�*;


total_loss�l�@

error_RͳH?

learning_rate_1�HG7��xI       6%�	.�	V���A�*;


total_lossk��@

error_R�[?

learning_rate_1�HG7;�:�I       6%�	H
V���A�*;


total_loss�Ŏ@

error_R6�??

learning_rate_1�HG7k~�I       6%�	��
V���A�*;


total_lossW�@

error_R�}U?

learning_rate_1�HG7��{I       6%�	��
V���A�*;


total_lossG�@

error_Rl�P?

learning_rate_1�HG7��֏I       6%�	KIV���A�*;


total_lossf�@

error_R�K?

learning_rate_1�HG7�a��I       6%�	�V���A�*;


total_lossn�i@

error_RqhU?

learning_rate_1�HG7(ŉ�I       6%�	��V���A�*;


total_loss��@

error_R�j`?

learning_rate_1�HG7�I       6%�	1V���A�*;


total_loss�ط@

error_R��W?

learning_rate_1�HG7��I       6%�	�[V���A�*;


total_loss$PA

error_R�pA?

learning_rate_1�HG7���I       6%�	@�V���A�*;


total_loss�A

error_R{N?

learning_rate_1�HG7{���I       6%�	v�V���A�*;


total_loss���@

error_RJ�^?

learning_rate_1�HG7��#I       6%�	�2V���A�*;


total_loss��@

error_R��@?

learning_rate_1�HG7<�/I       6%�	a�V���A�*;


total_loss3>�@

error_R,�I?

learning_rate_1�HG7����I       6%�	��V���A�*;


total_loss!��@

error_R�k>?

learning_rate_1�HG7ٹ��I       6%�	�'V���A�*;


total_lossI:�@

error_R��A?

learning_rate_1�HG7����I       6%�	��V���A�*;


total_loss��@

error_R�S?

learning_rate_1�HG7S�h�I       6%�	1�V���A�*;


total_loss4��@

error_R��f?

learning_rate_1�HG7۳wI       6%�	9LV���A�*;


total_lossǯ@

error_R�46?

learning_rate_1�HG7��8�I       6%�	��V���A�*;


total_loss �@

error_Rm�X?

learning_rate_1�HG7)�I       6%�	��V���A�*;


total_loss�U�@

error_R��H?

learning_rate_1�HG7�«�I       6%�	4ZV���A�*;


total_loss���@

error_R�\??

learning_rate_1�HG7"�|RI       6%�	��V���A�*;


total_loss�t@

error_RDH?

learning_rate_1�HG7�nI       6%�	T
V���A�*;


total_loss��@

error_R.!:?

learning_rate_1�HG7��|*I       6%�	8UV���A�*;


total_loss�lA

error_R�F?

learning_rate_1�HG72�I       6%�	1�V���A�*;


total_loss4��@

error_R��H?

learning_rate_1�HG73?4�I       6%�	M�V���A�*;


total_loss=1�@

error_R8"]?

learning_rate_1�HG7��I       6%�	�FV���A�*;


total_loss��@

error_R��V?

learning_rate_1�HG7I}7�I       6%�	A�V���A�*;


total_loss/��@

error_R_�H?

learning_rate_1�HG7s��I       6%�	��V���A�*;


total_loss��@

error_R	b[?

learning_rate_1�HG7��V�I       6%�	2V���A�*;


total_loss�ә@

error_R��G?

learning_rate_1�HG7w~��I       6%�	�zV���A�*;


total_loss4�@

error_ROiC?

learning_rate_1�HG7�EI       6%�	��V���A�*;


total_loss�ؑ@

error_R:L?

learning_rate_1�HG7�؟I       6%�	%V���A�*;


total_lossl��@

error_R�w:?

learning_rate_1�HG7�j�FI       6%�	/kV���A�*;


total_lossC��@

error_R _?

learning_rate_1�HG7vf'xI       6%�	ŰV���A�*;


total_loss���@

error_R�+P?

learning_rate_1�HG7��0I       6%�	W�V���A�*;


total_loss���@

error_RƝ@?

learning_rate_1�HG7 I�I       6%�	�]V���A�*;


total_loss\0�@

error_RD/E?

learning_rate_1�HG7�΄.I       6%�	J�V���A�*;


total_loss�&�@

error_R�W?

learning_rate_1�HG7�FsI       6%�	��V���A�*;


total_loss�
�@

error_R��G?

learning_rate_1�HG7��M6I       6%�	LV���A�*;


total_loss\��@

error_R�eF?

learning_rate_1�HG7T���I       6%�	L�V���A�*;


total_lossj��@

error_R&32?

learning_rate_1�HG7A.Z:I       6%�	T�V���A�*;


total_lossjʨ@

error_RV?

learning_rate_1�HG7�1m�I       6%�	�9V���A�*;


total_loss��@

error_R�V?

learning_rate_1�HG7_��I       6%�	>�V���A�*;


total_loss h-A

error_R��H?

learning_rate_1�HG7�?��I       6%�	�V���A�*;


total_loss	��@

error_R�}X?

learning_rate_1�HG7��n�I       6%�	�V���A�*;


total_loss���@

error_R*�V?

learning_rate_1�HG7��
I       6%�	pV���A�*;


total_losse��@

error_R��V?

learning_rate_1�HG7�qQ�I       6%�	]�V���A�*;


total_lossJ�@

error_R�v@?

learning_rate_1�HG7�$
FI       6%�	j�V���A�*;


total_loss���@

error_R��??

learning_rate_1�HG7��!�I       6%�	PTV���A�*;


total_lossL��@

error_R�-O?

learning_rate_1�HG7rf}I       6%�	НV���A�*;


total_loss���@

error_R�=?

learning_rate_1�HG7�~I       6%�	
�V���A�*;


total_loss{�@

error_Rq�Z?

learning_rate_1�HG71JLI       6%�	�2V���A�*;


total_losst �@

error_R�rN?

learning_rate_1�HG7�G=�I       6%�	��V���A�*;


total_loss|��@

error_RT�H?

learning_rate_1�HG7.qI       6%�	��V���A�*;


total_loss8��@

error_R��Q?

learning_rate_1�HG7��7�I       6%�	�5V���A�*;


total_loss�@

error_RE/[?

learning_rate_1�HG7aCII       6%�	�V���A�*;


total_loss//�@

error_RĨT?

learning_rate_1�HG7�Z��I       6%�	mV���A�*;


total_lossn�!A

error_R��E?

learning_rate_1�HG7 ��I       6%�	�\V���A�*;


total_loss���@

error_R5B?

learning_rate_1�HG7�$3�I       6%�	k�V���A�*;


total_losso�@

error_RŰZ?

learning_rate_1�HG7H��I       6%�	/V���A�*;


total_lossN��@

error_R��Y?

learning_rate_1�HG7&���I       6%�	!aV���A�*;


total_loss4u�@

error_R�H^?

learning_rate_1�HG7�τVI       6%�	��V���A�*;


total_losss2A

error_Rv�E?

learning_rate_1�HG7�"v�I       6%�	�V���A�*;


total_losswŋ@

error_RJ�L?

learning_rate_1�HG78>Q9I       6%�	UeV���A�*;


total_loss��@

error_RL�R?

learning_rate_1�HG7ޝ��I       6%�	��V���A�*;


total_loss/n�@

error_R�D?

learning_rate_1�HG7���I       6%�	�HV���A�*;


total_loss�+�@

error_R�qb?

learning_rate_1�HG7���I       6%�	1�V���A�*;


total_lossC��@

error_Rl�??

learning_rate_1�HG7����I       6%�	� V���A�*;


total_loss{��@

error_R�K?

learning_rate_1�HG7�Q�I       6%�	�Q V���A�*;


total_loss��f@

error_R�TJ?

learning_rate_1�HG7޿A�I       6%�	5� V���A�*;


total_loss�4�@

error_R�A?

learning_rate_1�HG7��JI       6%�	�� V���A�*;


total_loss�h�@

error_R�%T?

learning_rate_1�HG7���I       6%�	8-!V���A�*;


total_loss�1�@

error_R�sT?

learning_rate_1�HG7���yI       6%�	�u!V���A�*;


total_loss�R�@

error_RE�@?

learning_rate_1�HG7���I       6%�	��!V���A�*;


total_loss���@

error_Rv?<?

learning_rate_1�HG7H�n�I       6%�	l;"V���A�*;


total_loss$�@

error_R��F?

learning_rate_1�HG7OeN{I       6%�	,�"V���A�*;


total_lossD?�@

error_RQ>?

learning_rate_1�HG7��I       6%�	#V���A�*;


total_loss��@

error_R�*??

learning_rate_1�HG7,�w�I       6%�	I#V���A�*;


total_loss���@

error_R߀V?

learning_rate_1�HG7�7I       6%�	*�#V���A�*;


total_lossQ��@

error_R;I?

learning_rate_1�HG7�ae.I       6%�	��#V���A�*;


total_loss� A

error_R{�\?

learning_rate_1�HG7Xm�I       6%�	�$$V���A�*;


total_lossv �@

error_RLDB?

learning_rate_1�HG74�-II       6%�	Fn$V���A�*;


total_loss�n�@

error_R4�R?

learning_rate_1�HG75O��I       6%�	e�$V���A�*;


total_loss��@

error_R��M?

learning_rate_1�HG7V�qI       6%�	1�$V���A�*;


total_loss ��@

error_R�bS?

learning_rate_1�HG7`&uI       6%�	�n%V���A�*;


total_lossڱ�@

error_R8sA?

learning_rate_1�HG7��aI       6%�	f�%V���A�*;


total_loss��V@

error_RC�G?

learning_rate_1�HG7G�XI       6%�	r%&V���A�*;


total_loss���@

error_R�uL?

learning_rate_1�HG7/��I       6%�	�l&V���A�*;


total_loss���@

error_R�*F?

learning_rate_1�HG7-���I       6%�	ɳ&V���A�*;


total_loss�#�@

error_R��N?

learning_rate_1�HG7"���I       6%�	��&V���A�*;


total_loss�K�@

error_R��K?

learning_rate_1�HG7�JMbI       6%�	�A'V���A�*;


total_loss�\�@

error_Rx�G?

learning_rate_1�HG7���(I       6%�	L�'V���A�*;


total_lossaA

error_RT�W?

learning_rate_1�HG7�꩏I       6%�	�'V���A�*;


total_loss�A

error_R3J?

learning_rate_1�HG7�S@�I       6%�	�(V���A�*;


total_lossX�@

error_RR�Y?

learning_rate_1�HG7�Q"I       6%�	b(V���A�*;


total_loss�V�@

error_R��W?

learning_rate_1�HG7d�I       6%�	+�(V���A�*;


total_lossɟ�@

error_R�W?

learning_rate_1�HG7��+�I       6%�	?)V���A�*;


total_loss-"�@

error_R�C?

learning_rate_1�HG7�h�pI       6%�	_�)V���A�*;


total_loss|��@

error_R��Q?

learning_rate_1�HG7ަC[I       6%�	@�)V���A�*;


total_lossŘ�@

error_R�i??

learning_rate_1�HG7W'	�I       6%�	�*V���A�*;


total_loss!A

error_RɘE?

learning_rate_1�HG7�f�I       6%�	�Z*V���A�*;


total_loss̷�@

error_R<�c?

learning_rate_1�HG7&4�I       6%�	��*V���A�*;


total_losss��@

error_RN�<?

learning_rate_1�HG7I�%<I       6%�	��*V���A�*;


total_lossd��@

error_RE�Z?

learning_rate_1�HG7DI5�I       6%�	xN+V���A�*;


total_loss�Q�@

error_R�G?

learning_rate_1�HG7hf��I       6%�	v�+V���A�*;


total_loss��@

error_R��R?

learning_rate_1�HG7N���I       6%�	F�+V���A�*;


total_loss�Z�@

error_R��T?

learning_rate_1�HG7�kiwI       6%�	�$,V���A�*;


total_loss��@

error_R�=A?

learning_rate_1�HG7'��uI       6%�	\m,V���A�*;


total_loss-��@

error_Rj�A?

learning_rate_1�HG7W��II       6%�	��,V���A�*;


total_lossj�	A

error_RmkI?

learning_rate_1�HG7� �I       6%�	��,V���A�*;


total_loss�x�@

error_Rr�B?

learning_rate_1�HG7śP�I       6%�	F-V���A�*;


total_loss���@

error_R�K?

learning_rate_1�HG7�?(GI       6%�	Ȉ-V���A�*;


total_loss�'�@

error_R$"R?

learning_rate_1�HG7�anI       6%�	��-V���A�*;


total_lossO�A

error_R*Xm?

learning_rate_1�HG7n�2�I       6%�	�.V���A�*;


total_loss�1A

error_R�Jm?

learning_rate_1�HG7RƄVI       6%�	�P.V���A�*;


total_loss���@

error_R�B?

learning_rate_1�HG7�4R�I       6%�	"�.V���A�*;


total_loss|�@

error_R�b?

learning_rate_1�HG7	���I       6%�	�.V���A�*;


total_loss�J�@

error_R��I?

learning_rate_1�HG7�,�I       6%�	]/V���A�*;


total_loss��@

error_R��R?

learning_rate_1�HG7U��RI       6%�	�Y/V���A�*;


total_lossW�@

error_R�\?

learning_rate_1�HG7�b��I       6%�	��/V���A�*;


total_loss���@

error_R��F?

learning_rate_1�HG7|�ȿI       6%�	0�/V���A�*;


total_loss�˫@

error_RݛV?

learning_rate_1�HG7l!�I       6%�	� 0V���A�*;


total_loss�ƶ@

error_R\�??

learning_rate_1�HG7`P!�I       6%�	�d0V���A�*;


total_loss��
A

error_R��R?

learning_rate_1�HG7��2;I       6%�		�0V���A�*;


total_loss���@

error_R�??

learning_rate_1�HG7�N��I       6%�	��0V���A�*;


total_lossi�@

error_R�LE?

learning_rate_1�HG70���I       6%�	�(1V���A�*;


total_loss-a�@

error_RS[C?

learning_rate_1�HG79u��I       6%�	�j1V���A�*;


total_lossI�@

error_RASQ?

learning_rate_1�HG7����I       6%�	4�1V���A�*;


total_lossY�A

error_R6�U?

learning_rate_1�HG7�1�[I       6%�	��1V���A�*;


total_loss�C�@

error_Ri�W?

learning_rate_1�HG7�n�I       6%�	$02V���A�*;


total_loss��@

error_RK?

learning_rate_1�HG7>�B?I       6%�	�p2V���A�*;


total_lossʖ�@

error_Rx�J?

learning_rate_1�HG7�l�I       6%�	��2V���A�*;


total_loss��v@

error_Rt�Q?

learning_rate_1�HG7cd�^I       6%�	o�2V���A�*;


total_loss���@

error_R�NS?

learning_rate_1�HG7ӮC�I       6%�	�43V���A�*;


total_loss;1�@

error_R]�U?

learning_rate_1�HG7�U�2I       6%�	ls3V���A�*;


total_loss㤖@

error_R�J?

learning_rate_1�HG72%�<I       6%�	e�3V���A�*;


total_loss�f�@

error_R�OZ?

learning_rate_1�HG7Nz�WI       6%�	-4V���A�*;


total_loss���@

error_R��W?

learning_rate_1�HG7d��FI       6%�	�J4V���A�*;


total_loss���@

error_R��M?

learning_rate_1�HG7��v�I       6%�	��4V���A�*;


total_losslCA

error_R1B?

learning_rate_1�HG7�'"I       6%�	��4V���A�*;


total_lossv;�@

error_R�vL?

learning_rate_1�HG7Y�C�I       6%�	5V���A�*;


total_loss�@

error_R��M?

learning_rate_1�HG7�s�kI       6%�	Q5V���A�*;


total_lossQ��@

error_R�MM?

learning_rate_1�HG7�a��I       6%�	@�5V���A�*;


total_lossR+�@

error_Ria?

learning_rate_1�HG7C@��I       6%�	y�5V���A�*;


total_loss�Z�@

error_RD�a?

learning_rate_1�HG7��_I       6%�	�6V���A�*;


total_loss6��@

error_R@'I?

learning_rate_1�HG7d~[�I       6%�	�j6V���A�*;


total_loss�b�@

error_R�RP?

learning_rate_1�HG7[RzI       6%�	��6V���A�*;


total_loss�I�@

error_R�"@?

learning_rate_1�HG7�a	I       6%�	��6V���A�*;


total_loss���@

error_R�YO?

learning_rate_1�HG7@�T�I       6%�	W07V���A�*;


total_loss!��@

error_RŞ5?

learning_rate_1�HG71S�I       6%�	�p7V���A�*;


total_loss<U�@

error_R�Y?

learning_rate_1�HG7.���I       6%�	ʲ7V���A�*;


total_loss\��@

error_RBN?

learning_rate_1�HG7gߏ�I       6%�	��7V���A�*;


total_lossA�A

error_R��>?

learning_rate_1�HG7��BI       6%�	D<8V���A�*;


total_loss���@

error_R#C]?

learning_rate_1�HG7]���I       6%�	�|8V���A�*;


total_loss��@

error_Rx�D?

learning_rate_1�HG7�`E�I       6%�	��8V���A�*;


total_loss|G�@

error_R?S?

learning_rate_1�HG7Д�I       6%�	��8V���A�*;


total_lossX�@

error_R��O?

learning_rate_1�HG7,�הI       6%�	$B9V���A�*;


total_lossW��@

error_RhK?

learning_rate_1�HG7O>�7I       6%�	��9V���A�*;


total_loss�A�@

error_R�'S?

learning_rate_1�HG7��y�I       6%�	��9V���A�*;


total_loss}�@

error_R��I?

learning_rate_1�HG7�F%I       6%�	3C:V���A�*;


total_loss���@

error_RN0F?

learning_rate_1�HG7옦�I       6%�	��:V���A�*;


total_lossج@

error_R�J4?

learning_rate_1�HG7�<�I       6%�	�:V���A�*;


total_loss��A

error_Rw�V?

learning_rate_1�HG7C��I       6%�	`,;V���A�*;


total_loss�Fx@

error_RHB?

learning_rate_1�HG7����I       6%�	_q;V���A�*;


total_loss�.s@

error_R�d\?

learning_rate_1�HG7�^�;I       6%�	��;V���A�*;


total_loss&��@

error_RcL?

learning_rate_1�HG7�SI       6%�	�;V���A�*;


total_loss�U�@

error_R�C?

learning_rate_1�HG7���/I       6%�	q9<V���A�*;


total_loss�`�@

error_R��K?

learning_rate_1�HG7l��I       6%�	��<V���A�*;


total_loss�5�@

error_RQ�K?

learning_rate_1�HG7v/]II       6%�	��<V���A�*;


total_lossF|@

error_RC�S?

learning_rate_1�HG7�B�I       6%�	'=V���A�*;


total_loss!bA

error_RJnT?

learning_rate_1�HG7���I       6%�	�Z=V���A�*;


total_loss�u�@

error_R�0?

learning_rate_1�HG7�t�I       6%�	�=V���A�*;


total_loss�@�@

error_R�Q?

learning_rate_1�HG7��-�I       6%�	��=V���A�*;


total_loss�>�@

error_Rۡ\?

learning_rate_1�HG7��JI       6%�	q:>V���A�*;


total_loss���@

error_R\�P?

learning_rate_1�HG7x���I       6%�	w�>V���A�*;


total_lossG�@

error_R��P?

learning_rate_1�HG7�'�I       6%�	��>V���A�*;


total_loss�?�@

error_RC;G?

learning_rate_1�HG7=|bI       6%�	�3?V���A�*;


total_lossZ=�@

error_R�ZJ?

learning_rate_1�HG7[�/I       6%�	�x?V���A�*;


total_loss�6�@

error_R�LV?

learning_rate_1�HG74guI       6%�	s�?V���A�*;


total_loss$�@

error_R�U?

learning_rate_1�HG7@�h0I       6%�	@V���A�*;


total_loss�r�@

error_R(F?

learning_rate_1�HG7�уI       6%�	�M@V���A�*;


total_lossϹ�@

error_R��T?

learning_rate_1�HG7��I       6%�	��@V���A�*;


total_loss���@

error_R�-M?

learning_rate_1�HG7`�D�I       6%�	N�@V���A�*;


total_losse��@

error_R�O?

learning_rate_1�HG7�=I       6%�	l#AV���A�*;


total_loss�K�@

error_Ri[U?

learning_rate_1�HG7ؿ_�I       6%�	�hAV���A�*;


total_loss���@

error_R�*Z?

learning_rate_1�HG7��B%I       6%�	�AV���A�*;


total_loss`b�@

error_R��;?

learning_rate_1�HG7U�+I       6%�	V�AV���A�*;


total_loss%��@

error_R��Q?

learning_rate_1�HG7�q:�I       6%�	
2BV���A�*;


total_loss,�A

error_R!�R?

learning_rate_1�HG7�xh�I       6%�	;sBV���A�*;


total_loss�PA

error_R�@?

learning_rate_1�HG7��!I       6%�	��BV���A�*;


total_loss�f�@

error_R�>?

learning_rate_1�HG7`b^eI       6%�	�BV���A�*;


total_loss���@

error_R�}I?

learning_rate_1�HG7�ǇI       6%�	�@CV���A�*;


total_loss野@

error_R��K?

learning_rate_1�HG7s��I       6%�	��CV���A�*;


total_lossml�@

error_R:?

learning_rate_1�HG7j.�uI       6%�	(�CV���A�*;


total_loss�@

error_RMD?

learning_rate_1�HG7B/�I       6%�	�DV���A�*;


total_loss�J@

error_R��W?

learning_rate_1�HG7�ʉ6I       6%�	�EDV���A�*;


total_loss��@

error_Rf�??

learning_rate_1�HG7'�OI       6%�	ńDV���A�*;


total_loss��@

error_RrN?

learning_rate_1�HG7*��I       6%�	E�DV���A�*;


total_loss���@

error_R��;?

learning_rate_1�HG7Y��I       6%�	�EV���A�*;


total_loss�z�@

error_R&�;?

learning_rate_1�HG7m�0�I       6%�	LIEV���A�*;


total_loss���@

error_RE�L?

learning_rate_1�HG7~�/I       6%�	O�EV���A�*;


total_loss	x�@

error_Rz�6?

learning_rate_1�HG7��I       6%�	c�EV���A�*;


total_loss\p�@

error_R PF?

learning_rate_1�HG7��}I       6%�	9FV���A�*;


total_loss���@

error_R��C?

learning_rate_1�HG7��M�I       6%�	MUFV���A�*;


total_loss�ќ@

error_R��K?

learning_rate_1�HG7W�|�I       6%�	2�FV���A�*;


total_loss�Z�@

error_R��Q?

learning_rate_1�HG7�^�&I       6%�	��FV���A�*;


total_loss)��@

error_RwQO?

learning_rate_1�HG7C��	I       6%�	)GV���A�*;


total_loss�+�@

error_RS?

learning_rate_1�HG7͎+�I       6%�	aGV���A�*;


total_lossX�@

error_R;L?

learning_rate_1�HG7�L��I       6%�	<�GV���A�*;


total_loss�[�@

error_RN?

learning_rate_1�HG7 2TNI       6%�	��GV���A�*;


total_loss���@

error_R61G?

learning_rate_1�HG7���9I       6%�	�-HV���A�*;


total_lossSr�@

error_R�5O?

learning_rate_1�HG7��JI       6%�	huHV���A�*;


total_loss��&A

error_R�+?

learning_rate_1�HG7m��I       6%�	\�HV���A�*;


total_loss�d@

error_RwV?

learning_rate_1�HG73&�I       6%�	�HV���A�*;


total_loss|��@

error_R(�T?

learning_rate_1�HG7E%�I       6%�	�AIV���A�*;


total_loss�͙@

error_R�-X?

learning_rate_1�HG7�W��I       6%�	��IV���A�*;


total_loss���@

error_R�T?

learning_rate_1�HG7L��I       6%�	=�IV���A�*;


total_loss���@

error_R!@?

learning_rate_1�HG7�!"'I       6%�	�JV���A�*;


total_loss���@

error_R�VH?

learning_rate_1�HG7PE�I       6%�	6NJV���A�*;


total_lossMFf@

error_Rr?

learning_rate_1�HG7�x�I       6%�	 �JV���A�*;


total_loss��@

error_R�&G?

learning_rate_1�HG7L�ʮI       6%�	��JV���A�*;


total_loss���@

error_Rm�F?

learning_rate_1�HG7O�AqI       6%�	�7KV���A�*;


total_loss�I�@

error_R�%N?

learning_rate_1�HG7!N�I       6%�	�xKV���A�*;


total_loss	��@

error_RFP?

learning_rate_1�HG7}ܰ�I       6%�	��KV���A�*;


total_lossv��@

error_R,�\?

learning_rate_1�HG75�|0I       6%�	��KV���A�*;


total_lossFA�@

error_R1�d?

learning_rate_1�HG7v\}I       6%�	g@LV���A�*;


total_loss})�@

error_R��T?

learning_rate_1�HG7@�I       6%�	k�LV���A�*;


total_loss��@

error_RlyA?

learning_rate_1�HG7O4u�I       6%�	�LV���A�*;


total_loss.�@

error_R��S?

learning_rate_1�HG7=N�/I       6%�	MV���A�*;


total_loss	��@

error_R�}Q?

learning_rate_1�HG7=VI       6%�	jBMV���A�*;


total_loss��@

error_R*I?

learning_rate_1�HG7-v�I       6%�	܈MV���A�*;


total_loss:C�@

error_R�Y?

learning_rate_1�HG78.�/I       6%�	��MV���A�*;


total_lossTf�@

error_RFAM?

learning_rate_1�HG79��7I       6%�	�NV���A�*;


total_lossƜ�@

error_R E?

learning_rate_1�HG7�(��I       6%�	rQNV���A�*;


total_lossy�A

error_R�QR?

learning_rate_1�HG7����I       6%�	y�NV���A�*;


total_loss�Y�@

error_R3TO?

learning_rate_1�HG7�R.I       6%�	��NV���A�*;


total_lossa԰@

error_R|C^?

learning_rate_1�HG7�WβI       6%�	XMOV���A�*;


total_loss�*�@

error_RI�S?

learning_rate_1�HG7>��nI       6%�	��OV���A�*;


total_loss���@

error_R�^@?

learning_rate_1�HG7�/��I       6%�	7�OV���A�*;


total_loss܄�@

error_R�R?

learning_rate_1�HG7�L��I       6%�	�'PV���A�*;


total_loss�|�@

error_R�M?

learning_rate_1�HG7o8x�I       6%�	�iPV���A�*;


total_loss��A

error_R�iW?

learning_rate_1�HG7[�CI       6%�	��PV���A�*;


total_lossC�A

error_ReRG?

learning_rate_1�HG7R�3I       6%�	��PV���A�*;


total_lossr�@

error_R3RJ?

learning_rate_1�HG7!Z�I       6%�	;6QV���A�*;


total_loss�Ċ@

error_Rt�K?

learning_rate_1�HG7u�I       6%�	%xQV���A�*;


total_lossr�@

error_R�t=?

learning_rate_1�HG7t�IHI       6%�	0�QV���A�*;


total_loss�V1A

error_RR;\?

learning_rate_1�HG7H��I       6%�	��QV���A�*;


total_losso��@

error_R�	W?

learning_rate_1�HG7�bM�I       6%�	�BRV���A�*;


total_loss���@

error_R6�E?

learning_rate_1�HG7��hI       6%�	��RV���A�*;


total_loss�P�@

error_RҼZ?

learning_rate_1�HG7]�[I       6%�	��RV���A�*;


total_loss�I�@

error_R��A?

learning_rate_1�HG7D�2I       6%�	�SV���A�*;


total_loss��z@

error_R�0@?

learning_rate_1�HG7��{^I       6%�	n\SV���A�*;


total_loss4��@

error_R��L?

learning_rate_1�HG73�c.I       6%�	a�SV���A�*;


total_loss� A

error_R�vL?

learning_rate_1�HG7����I       6%�	j�SV���A�*;


total_loss�X�@

error_Rl}K?

learning_rate_1�HG7��'xI       6%�	�#TV���A�*;


total_loss��@

error_R-2?

learning_rate_1�HG7� ��I       6%�	gTV���A�*;


total_lossv!�@

error_R�RJ?

learning_rate_1�HG7z��zI       6%�	`�TV���A�*;


total_loss4�A

error_R��[?

learning_rate_1�HG7�sI       6%�	W�TV���A�*;


total_loss���@

error_RCkO?

learning_rate_1�HG7j��9I       6%�	v.UV���A�*;


total_loss�1|@

error_R& J?

learning_rate_1�HG7�,�eI       6%�	�mUV���A�*;


total_loss�@

error_R�,b?

learning_rate_1�HG7>�bI       6%�	!�UV���A�*;


total_loss�@

error_R&�J?

learning_rate_1�HG7Hz�GI       6%�	a�UV���A�*;


total_lossWE�@

error_R��f?

learning_rate_1�HG7V!�I       6%�	�.VV���A�*;


total_loss�$�@

error_R�j=?

learning_rate_1�HG7�j��I       6%�	qVV���A�*;


total_loss�,�@

error_R�>?

learning_rate_1�HG7ٞ��I       6%�	e�VV���A�*;


total_loss�B�@

error_R��P?

learning_rate_1�HG7��AI       6%�	�VV���A�*;


total_loss@�@

error_R��S?

learning_rate_1�HG7ѕ��I       6%�	m4WV���A�*;


total_loss��@

error_RS^I?

learning_rate_1�HG7~q�>I       6%�	0}WV���A�*;


total_lossg��@

error_R��V?

learning_rate_1�HG7ߟ@I       6%�	�WV���A�*;


total_loss���@

error_RQ^?

learning_rate_1�HG7�;��I       6%�	�XV���A�*;


total_loss�}�@

error_R�N?

learning_rate_1�HG7Rݎ�I       6%�	�DXV���A�*;


total_loss�b�@

error_R��J?

learning_rate_1�HG7�´7I       6%�	^�XV���A�*;


total_loss~f�@

error_R�>7?

learning_rate_1�HG7Y��I       6%�	>�XV���A�*;


total_loss��A

error_RwvR?

learning_rate_1�HG7���I       6%�	LYV���A�*;


total_loss��@

error_R@{O?

learning_rate_1�HG7zv�I       6%�	�PYV���A�*;


total_loss[�@

error_R��=?

learning_rate_1�HG7��<EI       6%�	w�YV���A�*;


total_loss!�i@

error_REpD?

learning_rate_1�HG7iL�\I       6%�	�YV���A�*;


total_loss��@

error_RK?

learning_rate_1�HG7A��I       6%�	�ZV���A�*;


total_loss��@

error_R`\?

learning_rate_1�HG7jG�&I       6%�	\_ZV���A�*;


total_loss���@

error_R�mb?

learning_rate_1�HG7��z�I       6%�	i�ZV���A�*;


total_loss���@

error_R��M?

learning_rate_1�HG7�A )I       6%�	J�ZV���A�*;


total_loss;�@

error_RR�c?

learning_rate_1�HG7I� �I       6%�		P[V���A�*;


total_loss�S'A

error_R;^U?

learning_rate_1�HG7bTA�I       6%�	��[V���A�*;


total_loss��A

error_R�+H?

learning_rate_1�HG7}B�I       6%�	��[V���A�*;


total_loss�۹@

error_R�8?

learning_rate_1�HG7q�WXI       6%�	?\V���A�*;


total_loss-�@

error_R��??

learning_rate_1�HG7r��;I       6%�	Rb\V���A�*;


total_loss_�@

error_R�WX?

learning_rate_1�HG7ho��I       6%�	r�\V���A�*;


total_loss���@

error_R�?V?

learning_rate_1�HG7�k�I       6%�	��\V���A�*;


total_lossZ�@

error_R3�E?

learning_rate_1�HG7�XsI       6%�	OG]V���A�*;


total_loss( �@

error_RZnQ?

learning_rate_1�HG7�՝'I       6%�	h�]V���A�*;


total_loss�_@

error_R�<?

learning_rate_1�HG7K�7;I       6%�	}�]V���A�*;


total_loss�{@

error_Rz�U?

learning_rate_1�HG7fI       6%�	�^V���A�*;


total_lossLA

error_RJS?

learning_rate_1�HG7���I       6%�	>^^V���A�*;


total_loss��@

error_R[1G?

learning_rate_1�HG7(�Z�I       6%�	��^V���A�*;


total_lossy��@

error_R�\?

learning_rate_1�HG7�f�I       6%�	�^V���A�*;


total_loss��@

error_R�>7?

learning_rate_1�HG7g�I       6%�	�2_V���A�*;


total_loss���@

error_R��P?

learning_rate_1�HG7�:YI       6%�	�{_V���A�*;


total_loss�١@

error_RQ A?

learning_rate_1�HG7��=I       6%�	�_V���A�*;


total_loss��@

error_R�RJ?

learning_rate_1�HG7L�y�I       6%�	f`V���A�*;


total_loss]/�@

error_R�,Q?

learning_rate_1�HG7�)I       6%�	SE`V���A�*;


total_loss*˚@

error_R�]?

learning_rate_1�HG7��oI       6%�	ƈ`V���A�*;


total_loss4��@

error_R�=?

learning_rate_1�HG7�jI       6%�	A�`V���A�*;


total_loss�;h@

error_R�[D?

learning_rate_1�HG7G���I       6%�	aV���A�*;


total_loss��@

error_RֽB?

learning_rate_1�HG7�fMiI       6%�	�NaV���A�*;


total_loss�s�@

error_R��S?

learning_rate_1�HG7�g�@I       6%�	ЏaV���A�*;


total_loss���@

error_R�S?

learning_rate_1�HG7S�]I       6%�	2�aV���A�*;


total_loss��@

error_R�J?

learning_rate_1�HG7��I       6%�	�bV���A�*;


total_loss��@

error_Rd�F?

learning_rate_1�HG7ܧG�I       6%�	UVbV���A�*;


total_loss_�@

error_R��D?

learning_rate_1�HG7Ÿ�iI       6%�	�bV���A�*;


total_loss�@

error_RR9D?

learning_rate_1�HG7�L�I       6%�	��bV���A�*;


total_loss/y�@

error_RjKL?

learning_rate_1�HG7�f�I       6%�	�@cV���A�*;


total_lossA&�@

error_R�Z3?

learning_rate_1�HG7��1YI       6%�	��cV���A�*;


total_loss�m�@

error_R�Y?

learning_rate_1�HG7�OI       6%�	��cV���A�*;


total_loss$N�@

error_R}�T?

learning_rate_1�HG7䦂�I       6%�	NdV���A�*;


total_loss�˒@

error_R*�j?

learning_rate_1�HG7.�TI       6%�	�IdV���A�*;


total_loss��@

error_R@se?

learning_rate_1�HG7���I       6%�	�dV���A�*;


total_loss��@

error_R��R?

learning_rate_1�HG7�X[8I       6%�	z�dV���A�*;


total_loss*�@

error_R�0W?

learning_rate_1�HG7�x3I       6%�	eV���A�*;


total_lossv�@

error_R��Q?

learning_rate_1�HG7d���I       6%�	bYeV���A�*;


total_loss
֥@

error_R)�R?

learning_rate_1�HG7��U�I       6%�	��eV���A�*;


total_lossV��@

error_R��H?

learning_rate_1�HG7UI�I       6%�	��eV���A�*;


total_loss�p�@

error_R�ef?

learning_rate_1�HG7����I       6%�	�!fV���A�*;


total_loss��@

error_R�j??

learning_rate_1�HG7~�y�I       6%�	xbfV���A�*;


total_loss$)�@

error_RMK?

learning_rate_1�HG7���1I       6%�	M�fV���A�*;


total_lossʚ�@

error_RD�L?

learning_rate_1�HG7߮��I       6%�	��fV���A�*;


total_loss���@

error_Rr�Q?

learning_rate_1�HG7|�I       6%�	a$gV���A�*;


total_loss���@

error_R�8T?

learning_rate_1�HG7)FYI       6%�	zfgV���A�*;


total_lossU)�@

error_RI�K?

learning_rate_1�HG7>��bI       6%�		�gV���A�*;


total_lossu��@

error_R
lE?

learning_rate_1�HG7/��I       6%�	w�gV���A�*;


total_loss��@

error_Rs�L?

learning_rate_1�HG7�|��I       6%�	�.hV���A�*;


total_loss�N�@

error_R��T?

learning_rate_1�HG7�{�I       6%�	�shV���A�*;


total_loss�	Z@

error_R<?

learning_rate_1�HG7�M�I       6%�	��hV���A�*;


total_loss���@

error_RT1O?

learning_rate_1�HG7I��I       6%�	�hV���A�*;


total_loss)ڤ@

error_R�<N?

learning_rate_1�HG7���oI       6%�	H?iV���A�*;


total_loss�G�@

error_R�.Y?

learning_rate_1�HG7�ҳCI       6%�	k�iV���A�*;


total_loss��@

error_Rq�<?

learning_rate_1�HG7fE�I       6%�	�iV���A�*;


total_loss��@

error_Rs#V?

learning_rate_1�HG7���I       6%�	�jV���A�*;


total_lossֹ@

error_RL k?

learning_rate_1�HG7�MjI       6%�	gHjV���A�*;


total_loss��@

error_R`?

learning_rate_1�HG7�ۊ�I       6%�	ڈjV���A�*;


total_loss��@

error_R}�A?

learning_rate_1�HG7����I       6%�	�jV���A�*;


total_loss���@

error_R��d?

learning_rate_1�HG7ȱI       6%�		*kV���A�*;


total_loss���@

error_Rq�I?

learning_rate_1�HG7]��I       6%�	�mkV���A�*;


total_loss�-�@

error_R�_E?

learning_rate_1�HG7<�7eI       6%�	Q�kV���A�*;


total_loss)e�@

error_R��N?

learning_rate_1�HG72E�I       6%�	z�kV���A�*;


total_loss�Ź@

error_R�iP?

learning_rate_1�HG7���I       6%�	10lV���A�*;


total_loss���@

error_R<Sh?

learning_rate_1�HG7)�eI       6%�	AulV���A�*;


total_loss|ܲ@

error_R�2S?

learning_rate_1�HG7ᬢ�I       6%�	7�lV���A�*;


total_loss#U�@

error_R��S?

learning_rate_1�HG7�*}�I       6%�	��lV���A�*;


total_lossx�@

error_R��U?

learning_rate_1�HG7�Ώ(I       6%�	�7mV���A�*;


total_loss?��@

error_R;r\?

learning_rate_1�HG7�EhmI       6%�	�ymV���A�*;


total_loss\��@

error_R8C?

learning_rate_1�HG7�6N\I       6%�	*�mV���A�*;


total_loss務@

error_R��R?

learning_rate_1�HG7�8BI       6%�	~nV���A�*;


total_loss4�@

error_RM=E?

learning_rate_1�HG72��%I       6%�	,JnV���A�*;


total_loss_�@

error_RL�Q?

learning_rate_1�HG7'�<I       6%�	y�nV���A�*;


total_loss�@

error_R1pM?

learning_rate_1�HG7W���I       6%�	��nV���A�*;


total_lossE��@

error_R8�N?

learning_rate_1�HG7F�KI       6%�	LoV���A�*;


total_loss!U�@

error_Re$7?

learning_rate_1�HG7�P�KI       6%�	ToV���A�*;


total_loss��@

error_R��A?

learning_rate_1�HG7�b�I       6%�	}�oV���A�*;


total_loss���@

error_R�rE?

learning_rate_1�HG7��k�I       6%�	m�oV���A�*;


total_lossji�@

error_RVS?

learning_rate_1�HG7o�?�I       6%�	�#pV���A�*;


total_loss� �@

error_R�<?

learning_rate_1�HG7hj�I       6%�	pipV���A�*;


total_loss���@

error_R��R?

learning_rate_1�HG7�9�I       6%�	�pV���A�*;


total_lossR��@

error_R�0`?

learning_rate_1�HG7S���I       6%�	[�pV���A�*;


total_loss0q�@

error_RZ�E?

learning_rate_1�HG7�z;>I       6%�	 3qV���A�*;


total_loss�6�@

error_R��_?

learning_rate_1�HG77�m�I       6%�	qqV���A�*;


total_loss���@

error_Rf�P?

learning_rate_1�HG7J�LuI       6%�	۳qV���A�*;


total_loss���@

error_R�AI?

learning_rate_1�HG7��C^I       6%�	��qV���A�*;


total_loss[��@

error_R��F?

learning_rate_1�HG7��I       6%�	�;rV���A�*;


total_loss��@

error_R��=?

learning_rate_1�HG7I�K7I       6%�	�rV���A�*;


total_loss���@

error_R�Y?

learning_rate_1�HG7���I       6%�	��rV���A�*;


total_loss%��@

error_R��Y?

learning_rate_1�HG7����I       6%�	�	sV���A�*;


total_loss�<�@

error_R��I?

learning_rate_1�HG7B�mpI       6%�	dSsV���A�*;


total_lossI0�@

error_R��<?

learning_rate_1�HG7�L� I       6%�	�sV���A�*;


total_loss��@

error_R��C?

learning_rate_1�HG7wC��I       6%�	(�sV���A�*;


total_loss�3�@

error_R8bp?

learning_rate_1�HG7��%SI       6%�	>tV���A�*;


total_loss��@

error_RW?

learning_rate_1�HG7��cI       6%�	�ZtV���A�*;


total_lossN�@

error_R�8N?

learning_rate_1�HG7l0�I       6%�	�tV���A�*;


total_losss��@

error_RD<_?

learning_rate_1�HG7��I       6%�	��tV���A�*;


total_loss�'�@

error_Ru;?

learning_rate_1�HG77N�I       6%�	�)uV���A�*;


total_loss�]�@

error_Rj{E?

learning_rate_1�HG7j)HI       6%�	�muV���A�*;


total_loss�-�@

error_RP?

learning_rate_1�HG7�^�I       6%�	׮uV���A�*;


total_loss�F�@

error_R\�M?

learning_rate_1�HG7n¯�I       6%�	 �uV���A�*;


total_lossn��@

error_R�\Z?

learning_rate_1�HG7�`�8I       6%�	�7vV���A�*;


total_loss��@

error_R�OV?

learning_rate_1�HG7Ǹ>I       6%�	~vV���A�*;


total_loss,�@

error_R``Y?

learning_rate_1�HG7�rb�I       6%�	P�vV���A�*;


total_lossW�@

error_R�O\?

learning_rate_1�HG7��GI       6%�	�wV���A�*;


total_loss���@

error_R��V?

learning_rate_1�HG7�e��I       6%�	�IwV���A�*;


total_loss�+�@

error_R�I?

learning_rate_1�HG7�Q�(I       6%�	L�wV���A�*;


total_loss_�A

error_R��T?

learning_rate_1�HG7���I       6%�	.�wV���A�*;


total_lossC�A

error_R�
R?

learning_rate_1�HG7f�z�I       6%�	�xV���A�*;


total_loss�,�@

error_R�.N?

learning_rate_1�HG7�ұ�I       6%�	�QxV���A�*;


total_loss?��@

error_Rr~J?

learning_rate_1�HG7
GJI       6%�	4�xV���A�*;


total_loss�d@

error_R��@?

learning_rate_1�HG7����I       6%�	��xV���A�*;


total_loss}�@

error_R�NR?

learning_rate_1�HG7�ʮ�I       6%�	$yV���A�*;


total_loss	z�@

error_R)�M?

learning_rate_1�HG7SUJI       6%�	�iyV���A�*;


total_loss���@

error_R�8R?

learning_rate_1�HG7L"��I       6%�	"�yV���A�*;


total_loss��@

error_RO�??

learning_rate_1�HG7b�[II       6%�	��yV���A�*;


total_loss��@

error_R��M?

learning_rate_1�HG7��qI       6%�	qWzV���A�*;


total_lossR��@

error_R&�??

learning_rate_1�HG7h�I       6%�	�zV���A�*;


total_loss �@

error_R
T?

learning_rate_1�HG7��aI       6%�	h {V���A�*;


total_loss�@�@

error_R�HW?

learning_rate_1�HG7� ]I       6%�	m{V���A�*;


total_loss<@�@

error_R�G?

learning_rate_1�HG7���@I       6%�	Գ{V���A�*;


total_loss;�KA

error_R1�T?

learning_rate_1�HG7p�W�I       6%�	G�{V���A�*;


total_loss��@

error_R�A?

learning_rate_1�HG7�;@I       6%�	�=|V���A�*;


total_loss�G�@

error_Ru=?

learning_rate_1�HG7�k��I       6%�	�|V���A�*;


total_loss�D�@

error_R�L?

learning_rate_1�HG7�]�I       6%�	
�|V���A�*;


total_loss֜@

error_R�*\?

learning_rate_1�HG7R�I       6%�	�}V���A�*;


total_loss+�@

error_RDS?

learning_rate_1�HG7�uI       6%�	�S}V���A�*;


total_loss�,�@

error_R��Q?

learning_rate_1�HG7̃�~I       6%�	��}V���A�*;


total_loss�q�@

error_R�N?

learning_rate_1�HG7�.�I       6%�	��}V���A�*;


total_loss�Y�@

error_R �U?

learning_rate_1�HG7M0<-I       6%�	F)~V���A�*;


total_loss��@

error_R�V?

learning_rate_1�HG7���I       6%�	k~V���A�*;


total_loss��@

error_Rl�L?

learning_rate_1�HG7G�t/I       6%�	ޮ~V���A�*;


total_loss�'�@

error_R=J?

learning_rate_1�HG7vh�wI       6%�	)�~V���A�*;


total_loss�ˇ@

error_R�^P?

learning_rate_1�HG72-6�I       6%�	$<V���A�*;


total_loss���@

error_R�cS?

learning_rate_1�HG7|>I       6%�	�V���A�*;


total_loss�q�@

error_RE�Q?

learning_rate_1�HG7nq=�I       6%�	�V���A�*;


total_loss��@

error_RMfH?

learning_rate_1�HG7-(\\I       6%�	u.�V���A�*;


total_lossv�@

error_R�d?

learning_rate_1�HG7@�t?I       6%�	�r�V���A�*;


total_loss��@

error_R��N?

learning_rate_1�HG7�glI       6%�	L��V���A�*;


total_loss�{�@

error_R�uO?

learning_rate_1�HG7�UI       6%�	��V���A�*;


total_lossJ�:@

error_R�}T?

learning_rate_1�HG7:��^I       6%�	.;�V���A�*;


total_loss�
�@

error_Rq0h?

learning_rate_1�HG7��<I       6%�	���V���A�*;


total_lossQ�^@

error_R�GD?

learning_rate_1�HG75{adI       6%�	}āV���A�*;


total_loss�� A

error_R@�g?

learning_rate_1�HG7t��zI       6%�	��V���A�*;


total_loss���@

error_R��\?

learning_rate_1�HG7 2@cI       6%�	S�V���A�*;


total_lossM��@

error_R�ZP?

learning_rate_1�HG7���MI       6%�	��V���A�*;


total_lossf�@

error_R��Q?

learning_rate_1�HG7��d0I       6%�	��V���A�*;


total_loss�B�@

error_R�nP?

learning_rate_1�HG7ע(<I       6%�	�'�V���A�*;


total_loss�ݶ@

error_RN?

learning_rate_1�HG7���I       6%�	h�V���A�*;


total_losst8A

error_R�T?

learning_rate_1�HG7rAqI       6%�	p��V���A�*;


total_loss�ث@

error_R�4F?

learning_rate_1�HG7�=I       6%�	~�V���A�*;


total_loss��
A

error_Rq�D?

learning_rate_1�HG7+JbwI       6%�	1�V���A�*;


total_lossJV�@

error_Rf5F?

learning_rate_1�HG7��� I       6%�	Hw�V���A�*;


total_loss\�@

error_R)�7?

learning_rate_1�HG7"��xI       6%�	K��V���A�*;


total_losshS�@

error_R�D?

learning_rate_1�HG7-���I       6%�	���V���A�*;


total_lossLu�@

error_Rq�K?

learning_rate_1�HG76��xI       6%�	�7�V���A�*;


total_loss-g@

error_R�J?

learning_rate_1�HG7���I       6%�	�v�V���A�*;


total_loss�ޓ@

error_R�>>?

learning_rate_1�HG7�]9I       6%�	Ѹ�V���A�*;


total_loss�-�@

error_RO�S?

learning_rate_1�HG7�	�.I       6%�	���V���A�*;


total_loss�:�@

error_R&8L?

learning_rate_1�HG77�n�I       6%�	�9�V���A�*;


total_loss�I�@

error_R�yP?

learning_rate_1�HG7�ҽI       6%�	�{�V���A�*;


total_loss�o�@

error_R��7?

learning_rate_1�HG7�Z�I       6%�	���V���A�*;


total_loss��A

error_R�OF?

learning_rate_1�HG7�G�bI       6%�	��V���A�*;


total_loss ��@

error_R��H?

learning_rate_1�HG7�:rI       6%�	TF�V���A�*;


total_loss�9�@

error_R�
O?

learning_rate_1�HG7�?<�I       6%�	���V���A�*;


total_loss���@

error_R,RS?

learning_rate_1�HG7�l�SI       6%�	yŇV���A�*;


total_loss�-�@

error_R��:?

learning_rate_1�HG7M�I       6%�	I�V���A�*;


total_loss��}@

error_RZR?

learning_rate_1�HG7uI       6%�	�E�V���A�*;


total_loss�ٝ@

error_R�>?

learning_rate_1�HG7��I       6%�	m��V���A�*;


total_loss�o�@

error_R�b?

learning_rate_1�HG7��SuI       6%�	�ЈV���A�*;


total_lossd*�@

error_R1CJ?

learning_rate_1�HG7?SXI       6%�	��V���A�*;


total_lossri�@

error_R��b?

learning_rate_1�HG7�@H�I       6%�	&]�V���A�*;


total_loss|b�@

error_R��F?

learning_rate_1�HG7�Y�I       6%�	؝�V���A�*;


total_loss��@

error_R��??

learning_rate_1�HG7Xt�I       6%�	;݉V���A�*;


total_loss���@

error_R��P?

learning_rate_1�HG7e���I       6%�	��V���A�*;


total_loss���@

error_R�U?

learning_rate_1�HG7�3�kI       6%�	1_�V���A�*;


total_loss&!�@

error_R��8?

learning_rate_1�HG7���pI       6%�	N��V���A�*;


total_loss�Ҹ@

error_R�xM?

learning_rate_1�HG7��K�I       6%�	��V���A�*;


total_loss�T�@

error_R��G?

learning_rate_1�HG7}��I       6%�	]F�V���A�*;


total_loss4��@

error_RܙZ?

learning_rate_1�HG7Q^�I       6%�	���V���A�*;


total_loss��@

error_RE�@?

learning_rate_1�HG7\'	I       6%�	#ˋV���A�*;


total_loss���@

error_R��>?

learning_rate_1�HG7�?I       6%�	G�V���A�*;


total_lossE��@

error_RCE?

learning_rate_1�HG7����I       6%�	 R�V���A�*;


total_loss�3A

error_Rj7H?

learning_rate_1�HG7Ъq>I       6%�	Ö�V���A�*;


total_loss�0r@

error_R�G?

learning_rate_1�HG7C�I       6%�	�یV���A�*;


total_loss�a�@

error_R�~6?

learning_rate_1�HG7�RN�I       6%�	�$�V���A�*;


total_loss�8A

error_R&�@?

learning_rate_1�HG7<
3�I       6%�	)l�V���A�*;


total_loss8HG@

error_R�K?

learning_rate_1�HG7,��I       6%�	q��V���A�*;


total_loss��@

error_RD8?

learning_rate_1�HG7"���I       6%�	�V���A�*;


total_lossi��@

error_R�I?

learning_rate_1�HG7E"��I       6%�	�1�V���A�*;


total_loss�t�@

error_R�)Z?

learning_rate_1�HG7�$�iI       6%�	<y�V���A�*;


total_loss�Z�@

error_R��_?

learning_rate_1�HG7��M0I       6%�	�V���A�*;


total_loss�M�@

error_Rv�G?

learning_rate_1�HG7˛�^I       6%�	��V���A�*;


total_loss���@

error_RH�N?

learning_rate_1�HG7\�}I       6%�	�H�V���A�*;


total_loss���@

error_R�1@?

learning_rate_1�HG7�/QI       6%�	���V���A�*;


total_loss}��@

error_R�{_?

learning_rate_1�HG7����I       6%�	`ҏV���A�*;


total_loss���@

error_R$�]?

learning_rate_1�HG7���I       6%�	k�V���A�*;


total_loss�B�@

error_R�[H?

learning_rate_1�HG7@
�I       6%�	hR�V���A�*;


total_loss�x�@

error_Rh9?

learning_rate_1�HG7�8�I       6%�	K��V���A�*;


total_loss��@

error_R��R?

learning_rate_1�HG7�'%hI       6%�	\ېV���A�*;


total_loss�@

error_RÜD?

learning_rate_1�HG7.��I       6%�	��V���A�*;


total_loss��@

error_R��T?

learning_rate_1�HG7�RTI       6%�	a�V���A�*;


total_loss2��@

error_R��D?

learning_rate_1�HG7
p�I       6%�	[��V���A�*;


total_loss��@

error_R��H?

learning_rate_1�HG7���I       6%�	�V���A�*;


total_loss�Z�@

error_R��^?

learning_rate_1�HG7���jI       6%�	�)�V���A�*;


total_lossΘ@

error_Rז??

learning_rate_1�HG7ƿW�I       6%�	xk�V���A�*;


total_loss��@

error_R��^?

learning_rate_1�HG7m��I       6%�	
��V���A�*;


total_lossOó@

error_R��J?

learning_rate_1�HG71�uiI       6%�	��V���A�*;


total_loss�5�@

error_RQ?

learning_rate_1�HG7}k)3I       6%�	�6�V���A�*;


total_loss�A

error_R�I?

learning_rate_1�HG7�~��I       6%�	�v�V���A�*;


total_loss�@

error_R<}I?

learning_rate_1�HG7���I       6%�	䶓V���A�*;


total_loss�W�@

error_R�tG?

learning_rate_1�HG7ˉ��I       6%�	m��V���A�*;


total_loss�Ֆ@

error_R2``?

learning_rate_1�HG7��`I       6%�	�;�V���A�*;


total_loss�ڿ@

error_R�f?

learning_rate_1�HG7�]I       6%�	��V���A�*;


total_loss���@

error_R{�R?

learning_rate_1�HG7v�%�I       6%�	OÔV���A�*;


total_loss �@

error_R�bN?

learning_rate_1�HG7�tI       6%�	T�V���A�*;


total_loss�ɬ@

error_R��K?

learning_rate_1�HG7���NI       6%�	�M�V���A�*;


total_loss�v�@

error_R�L?

learning_rate_1�HG7��E�I       6%�	l��V���A�*;


total_loss��@

error_R��T?

learning_rate_1�HG7S��I       6%�	YוV���A�*;


total_loss�ӧ@

error_R�f>?

learning_rate_1�HG7�E�wI       6%�	��V���A�*;


total_lossM��@

error_RQ?I?

learning_rate_1�HG7��I       6%�	�_�V���A�*;


total_loss���@

error_R�T?

learning_rate_1�HG7�ămI       6%�	��V���A�*;


total_loss��@

error_Ri,j?

learning_rate_1�HG7_]�aI       6%�	���V���A�*;


total_loss�9A

error_R��F?

learning_rate_1�HG7���I       6%�	x!�V���A�*;


total_loss}%�@

error_R��[?

learning_rate_1�HG7�U>I       6%�	Fc�V���A�*;


total_lossË�@

error_RTQ?

learning_rate_1�HG7O�	I       6%�	Ԥ�V���A�*;


total_loss�^�@

error_R��B?

learning_rate_1�HG7��
�I       6%�	O�V���A�*;


total_loss���@

error_R��]?

learning_rate_1�HG7޵��I       6%�	p'�V���A�*;


total_loss�5�@

error_RJuQ?

learning_rate_1�HG7s�I       6%�	Xf�V���A�*;


total_loss	g�@

error_R��Q?

learning_rate_1�HG7 	 �I       6%�	\��V���A�*;


total_lossSΓ@

error_R�F?

learning_rate_1�HG7D�"]I       6%�	��V���A�*;


total_loss���@

error_R��J?

learning_rate_1�HG7 0`vI       6%�	3�V���A�*;


total_lossCh`@

error_R�O?

learning_rate_1�HG7{��I       6%�	lu�V���A�*;


total_loss��@

error_R�T]?

learning_rate_1�HG7�I       6%�	2��V���A�*;


total_loss��@

error_R3�J?

learning_rate_1�HG7��nI       6%�	���V���A�*;


total_lossmG�@

error_R8<Z?

learning_rate_1�HG7W��GI       6%�	�=�V���A�*;


total_losszF�@

error_R�VO?

learning_rate_1�HG7���I       6%�	�~�V���A�*;


total_lossf��@

error_R&�H?

learning_rate_1�HG7��גI       6%�	?�V���A�*;


total_loss4u�@

error_RF�G?

learning_rate_1�HG7묬�I       6%�	:W�V���A�*;


total_loss���@

error_Rs�U?

learning_rate_1�HG7��5'I       6%�	M��V���A�*;


total_loss&��@

error_R��T?

learning_rate_1�HG7��I       6%�	�ݞV���A�*;


total_loss�]�@

error_R�<?

learning_rate_1�HG7�A`I       6%�	g"�V���A�*;


total_loss��@

error_R�E?

learning_rate_1�HG76�U�I       6%�	�n�V���A�*;


total_loss|�@

error_R�Y?

learning_rate_1�HG7㤿I       6%�	}ݟV���A�*;


total_loss��@

error_Rx/??

learning_rate_1�HG7h�I       6%�	a)�V���A�*;


total_loss89�@

error_RʝL?

learning_rate_1�HG7��ŕI       6%�	m�V���A�*;


total_lossf"�@

error_R�T4?

learning_rate_1�HG7�?3�I       6%�	v��V���A�*;


total_loss�޵@

error_R�V?

learning_rate_1�HG7�i��I       6%�	X��V���A�*;


total_lossd��@

error_RԑE?

learning_rate_1�HG7-MFI       6%�	�6�V���A�*;


total_loss�@

error_RwY?

learning_rate_1�HG7�uI       6%�	Ly�V���A�*;


total_loss&��@

error_R��V?

learning_rate_1�HG7s���I       6%�	ֽ�V���A�*;


total_loss�#�@

error_R4)L?

learning_rate_1�HG7!y�I       6%�	��V���A�*;


total_loss�r@

error_RsDI?

learning_rate_1�HG7AQ_�I       6%�	�F�V���A�*;


total_loss��@

error_Rs-^?

learning_rate_1�HG7��qI       6%�	*��V���A�*;


total_loss/��@

error_R�rW?

learning_rate_1�HG7Ai��I       6%�	�բV���A�*;


total_loss[
�@

error_R
+L?

learning_rate_1�HG7Mw��I       6%�	��V���A�*;


total_loss\��@

error_RxbD?

learning_rate_1�HG7��I       6%�	{\�V���A�*;


total_loss���@

error_RAJ?

learning_rate_1�HG7���I       6%�	$��V���A�*;


total_loss��@

error_R�I?

learning_rate_1�HG7 �B�I       6%�	�V���A�*;


total_loss���@

error_R��O?

learning_rate_1�HG703�I       6%�		0�V���A�*;


total_loss��A

error_R�C?

learning_rate_1�HG7��\II       6%�	Op�V���A�*;


total_lossf��@

error_R=IN?

learning_rate_1�HG7[.��I       6%�	���V���A�*;


total_loss�F�@

error_R&kA?

learning_rate_1�HG7X� �I       6%�	�V���A�*;


total_loss��@

error_R_W?

learning_rate_1�HG7�hDI       6%�	�2�V���A�*;


total_loss흎@

error_R�R?

learning_rate_1�HG7���I       6%�	Dt�V���A�*;


total_loss�@

error_R�uO?

learning_rate_1�HG7LY�cI       6%�	���V���A�*;


total_lossٮ@

error_RA�Q?

learning_rate_1�HG7�Mn�I       6%�	���V���A�*;


total_loss��@

error_R�J?

learning_rate_1�HG7�xsoI       6%�	6�V���A�*;


total_loss�R�@

error_R��G?

learning_rate_1�HG7d�BI       6%�	v�V���A�*;


total_lossx�@

error_R`C?

learning_rate_1�HG7K�AI       6%�	���V���A�*;


total_loss�@

error_R{�H?

learning_rate_1�HG7	�~�I       6%�	���V���A�*;


total_loss۳�@

error_R�X\?

learning_rate_1�HG7�N�lI       6%�	!>�V���A�*;


total_loss�@

error_R�-K?

learning_rate_1�HG7��8II       6%�	�~�V���A�*;


total_loss
E�@

error_R��D?

learning_rate_1�HG7;���I       6%�	N��V���A�*;


total_loss�m@

error_R�I?

learning_rate_1�HG7
��uI       6%�	�V���A�*;


total_loss8��@

error_R��D?

learning_rate_1�HG7$(�I       6%�	�Z�V���A�*;


total_loss���@

error_RMnU?

learning_rate_1�HG7WI       6%�	s��V���A�*;


total_loss�@

error_RXNP?

learning_rate_1�HG7��;I       6%�	��V���A�*;


total_loss0�@

error_R�P?

learning_rate_1�HG7����I       6%�	P�V���A�*;


total_lossRB�@

error_R\�M?

learning_rate_1�HG7 ��mI       6%�	8��V���A�*;


total_loss�<�@

error_R��C?

learning_rate_1�HG7t���I       6%�	�٩V���A�*;


total_loss��@

error_RfV?

learning_rate_1�HG7��?�I       6%�	��V���A�*;


total_lossn�^@

error_R��J?

learning_rate_1�HG7GU�I       6%�	]�V���A�*;


total_lossA��@

error_R�,]?

learning_rate_1�HG7��7GI       6%�	S��V���A�*;


total_loss82y@

error_RabS?

learning_rate_1�HG7 &\�I       6%�	��V���A�*;


total_lossL�@

error_RH]?

learning_rate_1�HG7���I       6%�	tK�V���A�*;


total_loss@z�@

error_RM�K?

learning_rate_1�HG7j<�I       6%�	���V���A�*;


total_lossC�@

error_R\K?

learning_rate_1�HG7���I       6%�	Q�V���A�*;


total_loss���@

error_R�9B?

learning_rate_1�HG7@��I       6%�	�4�V���A�*;


total_lossQ�@

error_R�R?

learning_rate_1�HG7ڿ�I       6%�	*{�V���A�*;


total_loss�	�@

error_R��\?

learning_rate_1�HG7���I       6%�	9��V���A�*;


total_loss<�@

error_R<"J?

learning_rate_1�HG7N=�I       6%�	��V���A�*;


total_loss(�@

error_R�C?

learning_rate_1�HG7�0D�I       6%�	�I�V���A�*;


total_loss/��@

error_RW�8?

learning_rate_1�HG7�%ENI       6%�	ˊ�V���A�*;


total_loss��@

error_R��S?

learning_rate_1�HG7�%�I       6%�	�έV���A�*;


total_lossz��@

error_R�R?

learning_rate_1�HG7�'��I       6%�	�V���A�*;


total_loss�{�@

error_RLU?

learning_rate_1�HG7Z��I       6%�	D^�V���A�*;


total_losst��@

error_R�R?

learning_rate_1�HG7��I       6%�	��V���A�*;


total_loss�A

error_R
�R?

learning_rate_1�HG7�	�I       6%�	��V���A�*;


total_loss�X�@

error_Rf
b?

learning_rate_1�HG7��"VI       6%�	Z(�V���A�*;


total_loss���@

error_R�Z?

learning_rate_1�HG7!CǊI       6%�	�n�V���A�*;


total_loss?�@

error_R�W?

learning_rate_1�HG7��#I       6%�	���V���A�*;


total_loss��@

error_RiJW?

learning_rate_1�HG7?E/�I       6%�	���V���A�*;


total_lossX�{@

error_R�lJ?

learning_rate_1�HG7U���I       6%�	fB�V���A�*;


total_loss̝�@

error_R�"C?

learning_rate_1�HG7#aēI       6%�	ˆ�V���A�*;


total_losss�@

error_R_�\?

learning_rate_1�HG7�v0;I       6%�	�ϰV���A�*;


total_lossش�@

error_R:�O?

learning_rate_1�HG7�cFI       6%�	��V���A�*;


total_loss��@

error_R-C?

learning_rate_1�HG7��2RI       6%�	�T�V���A�*;


total_loss�*�@

error_R��8?

learning_rate_1�HG7��iI       6%�	���V���A�*;


total_lossF�@

error_R��X?

learning_rate_1�HG7��8I       6%�	�ܱV���A�*;


total_loss�ߢ@

error_RrcP?

learning_rate_1�HG7��B�I       6%�	 �V���A�*;


total_loss���@

error_R%K?

learning_rate_1�HG7���I       6%�	h_�V���A�*;


total_lossa��@

error_RjF?

learning_rate_1�HG7 d*�I       6%�	���V���A�*;


total_loss���@

error_R)�R?

learning_rate_1�HG7j�I       6%�	��V���A�*;


total_loss{��@

error_R@K?

learning_rate_1�HG7�+�I       6%�	�)�V���A�*;


total_loss��A

error_R[�@?

learning_rate_1�HG7�9�
I       6%�	�j�V���A�*;


total_loss7�@

error_RZ�J?

learning_rate_1�HG7/���I       6%�	���V���A�*;


total_loss���@

error_R�X?

learning_rate_1�HG7�
��I       6%�	2�V���A�*;


total_loss�i�@

error_R1�P?

learning_rate_1�HG72�nI       6%�	�6�V���A�*;


total_loss1��@

error_RO~_?

learning_rate_1�HG7�L�I       6%�	�~�V���A�*;


total_loss�A

error_RqJ?

learning_rate_1�HG7뢚�I       6%�	���V���A�*;


total_lossI(7A

error_R��E?

learning_rate_1�HG7I*f�I       6%�	��V���A�*;


total_loss�(�@

error_RŐV?

learning_rate_1�HG7����I       6%�	�`�V���A�*;


total_loss7\�@

error_R�M?

learning_rate_1�HG7e��I       6%�	<ѵV���A�*;


total_loss�p@

error_R�M?

learning_rate_1�HG7���:I       6%�	R�V���A�*;


total_loss��@

error_R��I?

learning_rate_1�HG7���kI       6%�	Z�V���A�*;


total_loss,*�@

error_R]BS?

learning_rate_1�HG7�r��I       6%�	���V���A�*;


total_loss�u�@

error_R��U?

learning_rate_1�HG7�[�I       6%�	
޶V���A�*;


total_loss���@

error_RO?d?

learning_rate_1�HG7T]2rI       6%�	� �V���A�*;


total_loss��@

error_R6W6?

learning_rate_1�HG7�J��I       6%�	0c�V���A�*;


total_loss=Ҕ@

error_R�zR?

learning_rate_1�HG7Lh%5I       6%�	\��V���A�*;


total_loss0�@

error_R�Y?

learning_rate_1�HG7^�TqI       6%�	�V���A�*;


total_loss���@

error_R�zH?

learning_rate_1�HG7y�(KI       6%�	�:�V���A�*;


total_lossE��@

error_R��J?

learning_rate_1�HG7/n��I       6%�	⓸V���A�*;


total_loss��@

error_R.J?

learning_rate_1�HG7�gD�I       6%�	�V���A�*;


total_loss[��@

error_R�	D?

learning_rate_1�HG7V
 .I       6%�	�X�V���A�*;


total_loss���@

error_R�_?

learning_rate_1�HG7u��ZI       6%�	*��V���A�*;


total_loss���@

error_R �Z?

learning_rate_1�HG7ownI       6%�	�߹V���A�*;


total_loss���@

error_RlK?

learning_rate_1�HG7� t�I       6%�	�#�V���A�*;


total_loss7�@

error_R-�1?

learning_rate_1�HG7P^!I       6%�	�f�V���A�*;


total_lossh�@

error_Ru6?

learning_rate_1�HG7�{!I       6%�	+��V���A�*;


total_loss�Λ@

error_R{pS?

learning_rate_1�HG7���I       6%�	� �V���A�*;


total_loss̺�@

error_R��X?

learning_rate_1�HG7�D8�I       6%�	�[�V���A�*;


total_loss��@

error_R��H?

learning_rate_1�HG7]�b	I       6%�	��V���A�*;


total_lossj��@

error_R�_C?

learning_rate_1�HG7Ư�I       6%�	U �V���A�*;


total_loss��@

error_R�3Y?

learning_rate_1�HG77^I       6%�	ք�V���A�*;


total_loss�NA

error_RںP?

learning_rate_1�HG7/�I       6%�	5ƼV���A�*;


total_loss��@

error_R=�O?

learning_rate_1�HG7�]?�I       6%�	 �V���A�*;


total_loss��A

error_R�B*?

learning_rate_1�HG74��I       6%�	*L�V���A�*;


total_loss�U�@

error_R�B4?

learning_rate_1�HG7���hI       6%�	���V���A�*;


total_loss� j@

error_R3�K?

learning_rate_1�HG7r��I       6%�	�ѽV���A�*;


total_loss���@

error_R��K?

learning_rate_1�HG71��I       6%�	�V���A�*;


total_loss���@

error_Rq{Y?

learning_rate_1�HG7�'�I       6%�	�i�V���A�*;


total_loss�͜@

error_R��I?

learning_rate_1�HG7X�y~I       6%�	\ݾV���A�*;


total_lossQM�@

error_R$�Q?

learning_rate_1�HG7 ��I       6%�	rR�V���A�*;


total_loss�A�@

error_R�lQ?

learning_rate_1�HG7�e4OI       6%�	���V���A�*;


total_lossҍ�@

error_RO�A?

learning_rate_1�HG7r�}I       6%�	�
�V���A�*;


total_lossw��@

error_RlQ?

learning_rate_1�HG71F��I       6%�	�Q�V���A�*;


total_lossa�d@

error_Raw@?

learning_rate_1�HG7:��I       6%�	L��V���A�*;


total_loss��@

error_R�zT?

learning_rate_1�HG7Lι�I       6%�	P��V���A�*;


total_loss��@

error_RS�J?

learning_rate_1�HG7��[�I       6%�	�-�V���A�*;


total_lossx��@

error_R��@?

learning_rate_1�HG7��u%I       6%�	�o�V���A�*;


total_loss�?�@

error_R�U?

learning_rate_1�HG71V�(I       6%�	I��V���A�*;


total_loss#w�@

error_R��K?

learning_rate_1�HG7�h�I       6%�	���V���A�*;


total_loss#�A

error_R��]?

learning_rate_1�HG7�~��I       6%�	�e�V���A�*;


total_lossƏ�@

error_R,�W?

learning_rate_1�HG7�a�I       6%�	���V���A�*;


total_lossF.�@

error_Rf[?

learning_rate_1�HG7n��mI       6%�	��V���A�*;


total_losso0�@

error_R�YS?

learning_rate_1�HG7��l>I       6%�	V_�V���A�*;


total_loss���@

error_R�EO?

learning_rate_1�HG7�t�I       6%�	���V���A�*;


total_loss.�@

error_R��[?

learning_rate_1�HG7�k]�I       6%�	���V���A�*;


total_lossiW�@

error_R�GN?

learning_rate_1�HG73�S�I       6%�	�.�V���A�*;


total_loss�uA

error_Rf�C?

learning_rate_1�HG7>)%�I       6%�	x�V���A�*;


total_loss�~�@

error_RT�N?

learning_rate_1�HG7`DԂI       6%�	���V���A�*;


total_lossM��@

error_R�4L?

learning_rate_1�HG7�"4gI       6%�	M�V���A�*;


total_lossSZ�@

error_RC�K?

learning_rate_1�HG7��	I       6%�	�D�V���A�*;


total_lossb.�@

error_R�EL?

learning_rate_1�HG77�vpI       6%�	ް�V���A�*;


total_lossꄆ@

error_RINH?

learning_rate_1�HG75L��I       6%�	h�V���A�*;


total_loss���@

error_Rn�U?

learning_rate_1�HG7�X�iI       6%�	 b�V���A�*;


total_loss`��@

error_R�N?

learning_rate_1�HG7����I       6%�	#��V���A�*;


total_loss_1�@

error_RpR?

learning_rate_1�HG7�j4I       6%�	���V���A�*;


total_loss"��@

error_R��N?

learning_rate_1�HG7j�D�I       6%�	4'�V���A�*;


total_loss�@

error_R6OW?

learning_rate_1�HG7nٌI       6%�	sh�V���A�*;


total_loss]��@

error_R=�Y?

learning_rate_1�HG7��=I       6%�	}��V���A�*;


total_loss��@

error_R��O?

learning_rate_1�HG7+ř�I       6%�	���V���A�*;


total_loss�^�@

error_R�8?

learning_rate_1�HG79�I       6%�	-�V���A�*;


total_loss'�A

error_R�_]?

learning_rate_1�HG7<�}I       6%�	�n�V���A�*;


total_loss ��@

error_R�qP?

learning_rate_1�HG7h�,I       6%�	���V���A�*;


total_lossW��@

error_R1B?

learning_rate_1�HG7�o�|I       6%�	GM�V���A�*;


total_loss=��@

error_R��U?

learning_rate_1�HG7��ÏI       6%�	��V���A�*;


total_loss�`�@

error_R�-R?

learning_rate_1�HG7v1)I       6%�	k��V���A�*;


total_loss��@

error_R![P?

learning_rate_1�HG7h��I       6%�	t�V���A�*;


total_losseU�@

error_R!�a?

learning_rate_1�HG7�NfI       6%�	Z�V���A�*;


total_loss��@

error_R[IM?

learning_rate_1�HG7��-I       6%�	)��V���A�*;


total_loss	��@

error_R&�O?

learning_rate_1�HG7ٻF�I       6%�	K��V���A�*;


total_lossqY�@

error_R��B?

learning_rate_1�HG7�w��I       6%�	P�V���A�*;


total_loss�L�@

error_R��a?

learning_rate_1�HG7�)�-I       6%�	7��V���A�*;


total_loss�`�@

error_R!9H?

learning_rate_1�HG7#(q�I       6%�	���V���A�*;


total_loss���@

error_RDV?

learning_rate_1�HG7�n�I       6%�		t�V���A�*;


total_lossW��@

error_Rd,D?

learning_rate_1�HG7��0�I       6%�	I��V���A�*;


total_loss�XA

error_RI�P?

learning_rate_1�HG7�G�
I       6%�	��V���A�*;


total_loss3��@

error_R��J?

learning_rate_1�HG7��q�I       6%�	�H�V���A�*;


total_losssՠ@

error_R�&R?

learning_rate_1�HG7#��I       6%�	ێ�V���A�*;


total_loss���@

error_R�;b?

learning_rate_1�HG7����I       6%�	.��V���A�*;


total_loss���@

error_RB?

learning_rate_1�HG7}=�5I       6%�	��V���A�*;


total_loss���@

error_R�E?

learning_rate_1�HG7c�TI       6%�	�\�V���A�*;


total_loss��o@

error_R��O?

learning_rate_1�HG7���VI       6%�	J��V���A�*;


total_loss��@

error_RNW?

learning_rate_1�HG7�l�I       6%�	���V���A�*;


total_loss�m�@

error_RC^N?

learning_rate_1�HG75�IUI       6%�	�C�V���A�*;


total_loss)a�@

error_R��U?

learning_rate_1�HG7�Q�nI       6%�	#��V���A�*;


total_loss�A�@

error_R6�<?

learning_rate_1�HG7,��I       6%�	:��V���A�*;


total_loss{"A

error_R��k?

learning_rate_1�HG7�I       6%�	�?�V���A�*;


total_loss��@

error_RQ?

learning_rate_1�HG7�3/I       6%�	$��V���A�*;


total_lossC��@

error_R�fZ?

learning_rate_1�HG7�)�I       6%�	���V���A�*;


total_loss���@

error_R �U?

learning_rate_1�HG7���nI       6%�	+�V���A�*;


total_loss���@

error_RM\?

learning_rate_1�HG7	�-I       6%�	ZS�V���A�*;


total_loss|��@

error_R��P?

learning_rate_1�HG7�-�I       6%�	���V���A�*;


total_lossF�@

error_R�N?

learning_rate_1�HG7�yg�I       6%�	)��V���A�*;


total_lossϊ�@

error_Rxa_?

learning_rate_1�HG7��~�I       6%�	��V���A�*;


total_loss���@

error_Rv8D?

learning_rate_1�HG73�54I       6%�	�V�V���A�*;


total_loss�q�@

error_R�uD?

learning_rate_1�HG7;�3eI       6%�	/��V���A�*;


total_lossm�2A

error_R�'H?

learning_rate_1�HG7���I       6%�	K?�V���A�*;


total_lossO��@

error_R��Q?

learning_rate_1�HG7��i�I       6%�	 ��V���A�*;


total_loss��x@

error_R��@?

learning_rate_1�HG7o4��I       6%�	9��V���A�*;


total_loss�o�@

error_R\�X?

learning_rate_1�HG7�'�I       6%�	l�V���A�*;


total_loss��@

error_R;?B?

learning_rate_1�HG76��fI       6%�	�L�V���A�*;


total_loss���@

error_R/�L?

learning_rate_1�HG7��&�I       6%�	���V���A�*;


total_loss�N�@

error_R1�@?

learning_rate_1�HG7���I       6%�	#��V���A�*;


total_lossF��@

error_R�\?

learning_rate_1�HG7��=�I       6%�	}�V���A�*;


total_loss%��@

error_R:M?

learning_rate_1�HG7Ϗ'I       6%�	�V�V���A�*;


total_loss�Q�@

error_R��O?

learning_rate_1�HG7����I       6%�	A��V���A�*;


total_lossٵ�@

error_RsN?

learning_rate_1�HG7��?�I       6%�	��V���A�*;


total_loss��@

error_R��X?

learning_rate_1�HG7D�m2I       6%�	�v�V���A�*;


total_loss]~�@

error_R�X@?

learning_rate_1�HG7�@��I       6%�	���V���A�*;


total_loss-�@

error_R��W?

learning_rate_1�HG7%̹6I       6%�	#��V���A�*;


total_loss���@

error_R��R?

learning_rate_1�HG7��C�I       6%�	�>�V���A�*;


total_loss=��@

error_RO=P?

learning_rate_1�HG7۶��I       6%�	��V���A�*;


total_loss��@

error_R�e?

learning_rate_1�HG7B��I       6%�	���V���A�*;


total_loss�X�@

error_Rӫ^?

learning_rate_1�HG7�ȠlI       6%�	d�V���A�*;


total_loss@��@

error_R �R?

learning_rate_1�HG7L�QXI       6%�	�F�V���A�*;


total_loss3�@

error_R�<N?

learning_rate_1�HG7q�saI       6%�	���V���A�*;


total_loss��@

error_R�yU?

learning_rate_1�HG7��sI       6%�	?��V���A�*;


total_loss墰@

error_R�TP?

learning_rate_1�HG7X��I       6%�	GB�V���A�*;


total_loss_�@

error_R&�Z?

learning_rate_1�HG7��XI       6%�	��V���A�*;


total_loss���@

error_R_W]?

learning_rate_1�HG7ȖA
I       6%�	C��V���A�*;


total_lossM`�@

error_RMtJ?

learning_rate_1�HG7"�#RI       6%�	�2�V���A�*;


total_loss��@

error_R$�a?

learning_rate_1�HG76���I       6%�	pv�V���A�*;


total_lossʴ@

error_Rc$A?

learning_rate_1�HG7Q'Z�I       6%�	���V���A�*;


total_loss,��@

error_R��P?

learning_rate_1�HG7 }�I       6%�	I�V���A�*;


total_lossw�@

error_R�H?

learning_rate_1�HG7�)��I       6%�	�b�V���A�*;


total_loss�9A

error_R��;?

learning_rate_1�HG7N/\�I       6%�	@��V���A�*;


total_loss���@

error_R/�L?

learning_rate_1�HG7�:/�I       6%�	��V���A�*;


total_loss�KA

error_R3PS?

learning_rate_1�HG7n���I       6%�	�>�V���A�*;


total_loss�@

error_R��>?

learning_rate_1�HG7�dkI       6%�	���V���A�*;


total_lossHx�@

error_RlN?

learning_rate_1�HG7��1�I       6%�	~�V���A�*;


total_loss�Z�@

error_R�oO?

learning_rate_1�HG7!�I       6%�	�x�V���A�*;


total_loss���@

error_R_5?

learning_rate_1�HG7Q��I       6%�	i��V���A�*;


total_lossI�@

error_R�|^?

learning_rate_1�HG7��>I       6%�	� �V���A�*;


total_loss%�@

error_RT*F?

learning_rate_1�HG7�'��I       6%�	��V���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7�$�I       6%�	���V���A�*;


total_loss�@�@

error_RJZ?

learning_rate_1�HG7��|BI       6%�	6�V���A�*;


total_loss4ʰ@

error_R�,X?

learning_rate_1�HG7wN��I       6%�	_e�V���A�*;


total_loss��@

error_Rr$Z?

learning_rate_1�HG7z��7I       6%�	���V���A�*;


total_loss��@

error_Rh�F?

learning_rate_1�HG7�p�!I       6%�	�;�V���A�*;


total_lossֽ�@

error_R�[J?

learning_rate_1�HG7}�H�I       6%�	�|�V���A�*;


total_losss��@

error_R�9??

learning_rate_1�HG7K�
I       6%�	���V���A�*;


total_loss5�@

error_RV6A?

learning_rate_1�HG7��*�I       6%�	E�V���A�*;


total_loss�K�@

error_R,NH?

learning_rate_1�HG7���I       6%�	ӂ�V���A�*;


total_loss�3z@

error_R
	C?

learning_rate_1�HG7f�FI       6%�	"��V���A�*;


total_loss���@

error_Rd_T?

learning_rate_1�HG7�
��I       6%�	��V���A�*;


total_lossZ=�@

error_RGZ?

learning_rate_1�HG7#��I       6%�	mZ�V���A�*;


total_lossMs�@

error_R/�U?

learning_rate_1�HG7� �I       6%�	���V���A�*;


total_loss��@

error_R��:?

learning_rate_1�HG7n���I       6%�	S�V���A�*;


total_lossD�@

error_R2�M?

learning_rate_1�HG7C��I       6%�	���V���A�*;


total_loss�K�@

error_R��W?

learning_rate_1�HG7k�[WI       6%�	f�V���A�*;


total_lossS�@

error_R��C?

learning_rate_1�HG7I�k�I       6%�	$O�V���A�*;


total_loss20�@

error_R�A?

learning_rate_1�HG7H�R<I       6%�	���V���A�*;


total_loss�-�@

error_RZ�[?

learning_rate_1�HG7��W�I       6%�	b �V���A�*;


total_loss�-�@

error_Rl�J?

learning_rate_1�HG7��g.I       6%�	�O�V���A�*;


total_loss��@

error_R�Id?

learning_rate_1�HG7�IZ�I       6%�	���V���A�*;


total_loss*�@

error_R�AP?

learning_rate_1�HG7�
��I       6%�	S��V���A�*;


total_lossV2�@

error_R�O?

learning_rate_1�HG7"H�YI       6%�	bl�V���A�*;


total_lossOg�@

error_ROO?

learning_rate_1�HG7'RĹI       6%�	S��V���A�*;


total_loss(+A

error_R�G?

learning_rate_1�HG7t�JI       6%�	b��V���A�*;


total_loss��@

error_Ra�O?

learning_rate_1�HG7
���I       6%�	A�V���A�*;


total_losss
�@

error_R6TK?

learning_rate_1�HG7�eSI       6%�	[��V���A�*;


total_lossLC�@

error_R{NF?

learning_rate_1�HG7Ț�8I       6%�	���V���A�*;


total_loss�x�@

error_R*�Y?

learning_rate_1�HG7&#?�I       6%�	�V���A�*;


total_loss���@

error_R�XJ?

learning_rate_1�HG7�LI       6%�	�X�V���A�*;


total_loss�Dq@

error_Rt�>?

learning_rate_1�HG7d���I       6%�	c��V���A�*;


total_loss�s�@

error_R�=C?

learning_rate_1�HG7�cʎI       6%�	���V���A�*;


total_loss8$�@

error_R�M?

learning_rate_1�HG7`D�I       6%�	�L�V���A�*;


total_loss�U�@

error_R�D?

learning_rate_1�HG7��sZI       6%�	���V���A�*;


total_loss��A

error_R�G?

learning_rate_1�HG7:�0I       6%�	��V���A�*;


total_loss=�@

error_R�K?

learning_rate_1�HG7�'�0I       6%�	{@�V���A�*;


total_loss���@

error_R*{`?

learning_rate_1�HG71'��I       6%�	���V���A�*;


total_loss��@

error_RJ�;?

learning_rate_1�HG7u�"I       6%�	9��V���A�*;


total_loss@�@

error_R��B?

learning_rate_1�HG7%jI       6%�	�-�V���A�*;


total_lossS�@

error_R;�e?

learning_rate_1�HG7eWvI       6%�	�x�V���A�*;


total_loss��@

error_R�_?

learning_rate_1�HG7�~ƯI       6%�	��V���A�*;


total_loss��@

error_R\^U?

learning_rate_1�HG7���9I       6%�	��V���A�*;


total_loss���@

error_R��I?

learning_rate_1�HG7�1�qI       6%�	�m�V���A�*;


total_loss��@

error_R�D?

learning_rate_1�HG7�k$eI       6%�	 ��V���A�*;


total_loss�?�@

error_R�d?

learning_rate_1�HG7�E{I       6%�	+�V���A�*;


total_lossh��@

error_R��]?

learning_rate_1�HG7���eI       6%�	}p�V���A�*;


total_loss���@

error_R��T?

learning_rate_1�HG7N%�I       6%�	õ�V���A�*;


total_lossI`�@

error_R�oO?

learning_rate_1�HG71kI       6%�	]��V���A�*;


total_loss��@

error_R�$O?

learning_rate_1�HG7_K�I       6%�	e>�V���A�*;


total_loss*�A

error_R�ZZ?

learning_rate_1�HG7�E�I       6%�	���V���A�*;


total_loss���@

error_Rf4H?

learning_rate_1�HG7Ie�I       6%�	���V���A�*;


total_loss)M�@

error_R�2V?

learning_rate_1�HG7�f�I       6%�	l�V���A�*;


total_loss-��@

error_R�N?

learning_rate_1�HG7v�$�I       6%�	M�V���A�*;


total_loss���@

error_R�M?

learning_rate_1�HG7��MI       6%�	���V���A�*;


total_loss�@

error_RH1E?

learning_rate_1�HG7Z�|I       6%�	�!�V���A�*;


total_loss@W�@

error_R��M?

learning_rate_1�HG7B��KI       6%�	,k�V���A�*;


total_loss�:�@

error_R�c?

learning_rate_1�HG7�5��I       6%�	���V���A�*;


total_loss;ޞ@

error_R3|U?

learning_rate_1�HG7�e�gI       6%�	���V���A�*;


total_loss�A

error_R�N?

learning_rate_1�HG7�U��I       6%�	�5�V���A�*;


total_lossxܵ@

error_R�G?

learning_rate_1�HG7�
}�I       6%�	+y�V���A�*;


total_loss\/�@

error_R��N?

learning_rate_1�HG7T'�I       6%�	��V���A�*;


total_loss���@

error_Rs�Z?

learning_rate_1�HG7����I       6%�	P�V���A�*;


total_loss��@

error_R�dU?

learning_rate_1�HG7�M��I       6%�	IY�V���A�*;


total_lossR�@

error_R��F?

learning_rate_1�HG7����I       6%�	j��V���A�*;


total_loss疛@

error_R��[?

learning_rate_1�HG7�
NUI       6%�	�$�V���A�*;


total_lossN�@

error_R��E?

learning_rate_1�HG7�mAAI       6%�	��V���A�*;


total_lossh�@

error_Rd�L?

learning_rate_1�HG7�]�-I       6%�	���V���A�*;


total_lossDύ@

error_R�-H?

learning_rate_1�HG7]�P{I       6%�	�$�V���A�*;


total_lossr��@

error_R�~F?

learning_rate_1�HG7�4�I       6%�	�j�V���A�*;


total_loss���@

error_RnE?

learning_rate_1�HG7k�9CI       6%�	Į�V���A�*;


total_lossoE�@

error_R�^?

learning_rate_1�HG7�F�I       6%�	���V���A�*;


total_loss�A

error_RT�U?

learning_rate_1�HG7 �wI       6%�	7�V���A�*;


total_lossf�y@

error_R��G?

learning_rate_1�HG7���$I       6%�	�~�V���A�*;


total_loss+`�@

error_R��M?

learning_rate_1�HG7V�y�I       6%�	���V���A�*;


total_loss�W�@

error_R�O??

learning_rate_1�HG73r)�I       6%�	r
�V���A�*;


total_lossTJ�@

error_R��Z?

learning_rate_1�HG7Nj%I       6%�	4��V���A�*;


total_lossq�@

error_R��F?

learning_rate_1�HG7�%��I       6%�	G��V���A�*;


total_loss]Q�@

error_R�lC?

learning_rate_1�HG7@��I       6%�	�-�V���A�*;


total_lossCs�@

error_R[�T?

learning_rate_1�HG7fbj.I       6%�	@t�V���A�*;


total_loss#��@

error_R�L?

learning_rate_1�HG7�,�I       6%�	6��V���A�*;


total_losspd�@

error_R�Z]?

learning_rate_1�HG7en��I       6%�	7��V���A�*;


total_loss�5�@

error_R3�H?

learning_rate_1�HG7�U�I       6%�	�E�V���A�*;


total_lossa-
A

error_R�IZ?

learning_rate_1�HG7��0!I       6%�	���V���A�*;


total_loss/j�@

error_RG?

learning_rate_1�HG7���I       6%�	���V���A�*;


total_loss&K�@

error_R`�[?

learning_rate_1�HG7V�T9I       6%�	��V���A�*;


total_lossѕ@

error_RȍS?

learning_rate_1�HG7���I       6%�	�T�V���A�*;


total_loss���@

error_Rv�W?

learning_rate_1�HG7��s\I       6%�	[��V���A�*;


total_loss��@

error_ReQ?

learning_rate_1�HG7U���I       6%�	o��V���A�*;


total_lossEJ~@

error_R}bV?

learning_rate_1�HG7�  I       6%�	�/�V���A�*;


total_loss�!�@

error_R�N?

learning_rate_1�HG7�4�I       6%�	���V���A�*;


total_loss|�@

error_R�Z?

learning_rate_1�HG7<�	fI       6%�	��V���A�*;


total_lossx�
A

error_R)i@?

learning_rate_1�HG7� ۯI       6%�	cF�V���A�*;


total_loss`S�@

error_RM�8?

learning_rate_1�HG7�ɧ�I       6%�	i��V���A�*;


total_loss�(�@

error_R`�a?

learning_rate_1�HG7�&L�I       6%�	���V���A�*;


total_lossJ�@

error_R!B?

learning_rate_1�HG7�[��I       6%�	��V���A�*;


total_lossV=�@

error_R�WC?

learning_rate_1�HG7#��I       6%�	�d�V���A�*;


total_lossS9�@

error_R��L?

learning_rate_1�HG7s�DI       6%�	���V���A�*;


total_lossr��@

error_R�oB?

learning_rate_1�HG7XHSI       6%�	��V���A�*;


total_loss�P~@

error_R�5?

learning_rate_1�HG7�7�I       6%�	2�V���A�*;


total_loss���@

error_R�K?

learning_rate_1�HG7C<B�I       6%�	�w�V���A�*;


total_loss\�@

error_R�7A?

learning_rate_1�HG7���I       6%�	d��V���A�*;


total_lossс�@

error_R��H?

learning_rate_1�HG7 I       6%�	w�V���A�*;


total_loss/|�@

error_R  P?

learning_rate_1�HG7 T��I       6%�	�\�V���A�*;


total_loss��@

error_R�=X?

learning_rate_1�HG7!�Z�I       6%�	���V���A�*;


total_loss�bEA

error_Rl�T?

learning_rate_1�HG7R7�nI       6%�	f��V���A�*;


total_lossO�@

error_RvL?

learning_rate_1�HG7��BI       6%�	�6�V���A�*;


total_lossx�@

error_R;�U?

learning_rate_1�HG7�d'DI       6%�	Oz�V���A�*;


total_loss�Ĥ@

error_R��A?

learning_rate_1�HG7SrBI       6%�	ս�V���A�*;


total_loss!uv@

error_R:�>?

learning_rate_1�HG7pd��I       6%�	�  W���A�*;


total_loss��A

error_ROhQ?

learning_rate_1�HG7��sI       6%�	�I W���A�*;


total_loss��	A

error_R%U?

learning_rate_1�HG7mLBjI       6%�	�� W���A�*;


total_lossځ�@

error_R	H?

learning_rate_1�HG7�+�I       6%�	�� W���A�*;


total_loss���@

error_R�a?

learning_rate_1�HG7����I       6%�	{=W���A�*;


total_loss�@

error_R2�W?

learning_rate_1�HG7+j��I       6%�	��W���A�*;


total_loss��A

error_RL�G?

learning_rate_1�HG7�u&I       6%�	�W���A�*;


total_loss|u�@

error_R)/S?

learning_rate_1�HG74�OAI       6%�	W���A�*;


total_loss.8�@

error_RH+K?

learning_rate_1�HG7��[+I       6%�	�TW���A�*;


total_loss-��@

error_R�E?

learning_rate_1�HG7�c'6I       6%�	}�W���A�*;


total_lossꡳ@

error_R�s]?

learning_rate_1�HG7�6�WI       6%�	��W���A�*;


total_loss�:�@

error_R�hU?

learning_rate_1�HG7����I       6%�	�$W���A�*;


total_loss��@

error_R$�K?

learning_rate_1�HG7\?kAI       6%�	~gW���A�*;


total_loss<�@

error_RM�L?

learning_rate_1�HG7���~I       6%�	��W���A�*;


total_loss즵@

error_R�#E?

learning_rate_1�HG7���I       6%�	�W���A�*;


total_loss�՟@

error_RxL?

learning_rate_1�HG7�:�PI       6%�	�4W���A�*;


total_loss���@

error_R��S?

learning_rate_1�HG7��CI       6%�	hzW���A�*;


total_lossl�A

error_Rr�Q?

learning_rate_1�HG7Sa|uI       6%�	9�W���A�*;


total_loss��@

error_RQ�B?

learning_rate_1�HG7D#�#I       6%�	�W���A�*;


total_loss(K�@

error_R=�B?

learning_rate_1�HG7	��II       6%�	GW���A�*;


total_loss��@

error_R*�X?

learning_rate_1�HG7���I       6%�	;�W���A�*;


total_lossV��@

error_R�T?

learning_rate_1�HG7�ۙ�I       6%�	�W���A�*;


total_loss�N�@

error_R�`Y?

learning_rate_1�HG7̮�@I       6%�	�W���A�*;


total_loss�ƛ@

error_RZ�U?

learning_rate_1�HG75��I       6%�	�VW���A�*;


total_loss��@

error_R�V?

learning_rate_1�HG75��I       6%�	s�W���A�*;


total_lossn��@

error_R�;S?

learning_rate_1�HG7�%ȮI       6%�	��W���A�*;


total_loss�2�@

error_R$�b?

learning_rate_1�HG7���I       6%�	u#W���A�*;


total_loss���@

error_R�L?

learning_rate_1�HG7�'�yI       6%�	�eW���A�*;


total_loss�|�@

error_R��G?

learning_rate_1�HG7g'SsI       6%�	+�W���A�*;


total_loss�p�@

error_R�we?

learning_rate_1�HG7u�yI       6%�	^�W���A�*;


total_loss�\�@

error_R�@U?

learning_rate_1�HG7F2PI       6%�	�0W���A�*;


total_loss��
A

error_RW�^?

learning_rate_1�HG7>�:I       6%�	$rW���A�*;


total_loss�م@

error_RxM?

learning_rate_1�HG7`��I       6%�	γW���A�*;


total_loss��A

error_R2�F?

learning_rate_1�HG7è6�I       6%�	��W���A�*;


total_loss?<�@

error_R�$_?

learning_rate_1�HG73�kI       6%�	":	W���A�*;


total_loss\��@

error_Ri�N?

learning_rate_1�HG7̈)�I       6%�	}	W���A�*;


total_loss��@

error_R��P?

learning_rate_1�HG7a�C`I       6%�	�	W���A�*;


total_loss��@

error_R�jQ?

learning_rate_1�HG7�U�5I       6%�	%
W���A�*;


total_loss��@

error_R�^?

learning_rate_1�HG7S���I       6%�	�F
W���A�*;


total_loss���@

error_R�V?

learning_rate_1�HG7��X�I       6%�	h�
W���A�*;


total_lossD�@

error_R�6Y?

learning_rate_1�HG7.�46I       6%�	��
W���A�*;


total_loss�F�@

error_R�N?

learning_rate_1�HG7t���I       6%�	�2W���A�*;


total_lossא@

error_RdFM?

learning_rate_1�HG7*��I       6%�	�zW���A�*;


total_loss���@

error_R;TP?

learning_rate_1�HG7�",�I       6%�	��W���A�*;


total_loss�'�@

error_R�CN?

learning_rate_1�HG7��źI       6%�	�W���A�*;


total_loss�Z�@

error_R1;?

learning_rate_1�HG7s���I       6%�	"PW���A�*;


total_lossA

error_RO�]?

learning_rate_1�HG7��ɘI       6%�	�W���A�*;


total_loss�?�@

error_R��P?

learning_rate_1�HG7�,�I       6%�	��W���A�*;


total_loss�]�@

error_RC�O?

learning_rate_1�HG7�it~I       6%�	R&W���A�*;


total_loss�T�@

error_R��<?

learning_rate_1�HG7�'I       6%�	kW���A�*;


total_loss�^�@

error_R[D?

learning_rate_1�HG7��1�I       6%�	x�W���A�*;


total_loss���@

error_RnyH?

learning_rate_1�HG7� 4I       6%�	O�W���A�*;


total_lossH��@

error_R��W?

learning_rate_1�HG7��_I       6%�	e=W���A�*;


total_lossiU�@

error_Rxc>?

learning_rate_1�HG7%�,I       6%�	�W���A�*;


total_lossӡ�@

error_Rc@J?

learning_rate_1�HG7>�wI       6%�	��W���A�*;


total_loss(y�@

error_R)�P?

learning_rate_1�HG7l	�I       6%�	�W���A�*;


total_loss�q�@

error_R��3?

learning_rate_1�HG7�Su�I       6%�	aUW���A�*;


total_loss:d�@

error_RXMP?

learning_rate_1�HG7?rcI       6%�	�W���A�*;


total_loss��@

error_R��M?

learning_rate_1�HG7�c�YI       6%�	X�W���A�*;


total_loss�D�@

error_R�V?

learning_rate_1�HG7i3��I       6%�	�+W���A�*;


total_loss�ԏ@

error_R$�R?

learning_rate_1�HG7�$��I       6%�	�mW���A�*;


total_loss���@

error_R�@W?

learning_rate_1�HG7��}|I       6%�	 �W���A�*;


total_loss靻@

error_R\�B?

learning_rate_1�HG7F@�I       6%�	�W���A�*;


total_lossa:�@

error_R��<?

learning_rate_1�HG7�1�I       6%�	JFW���A�*;


total_losst!�@

error_R��U?

learning_rate_1�HG7����I       6%�	P�W���A�*;


total_lossXe�@

error_R��<?

learning_rate_1�HG7���!I       6%�	1�W���A�*;


total_losss~�@

error_R{K?

learning_rate_1�HG7� XI       6%�	W���A�*;


total_loss���@

error_R7Y?

learning_rate_1�HG7���I       6%�	�bW���A�*;


total_loss-�@

error_R�*V?

learning_rate_1�HG7��jI       6%�	��W���A�*;


total_lossL�@

error_R�bZ?

learning_rate_1�HG7��aI       6%�	_�W���A�*;


total_loss�h�@

error_R.	@?

learning_rate_1�HG7���9I       6%�	`+W���A�*;


total_loss7��@

error_RS?

learning_rate_1�HG7+��I       6%�	�nW���A�*;


total_lossP�@

error_R.:C?

learning_rate_1�HG7���I       6%�	ڱW���A�*;


total_loss���@

error_R�Y?

learning_rate_1�HG7��I       6%�	��W���A�*;


total_loss7�@

error_R�@?

learning_rate_1�HG7�e��I       6%�	�:W���A�*;


total_loss�WA

error_RXh?

learning_rate_1�HG7�<�I       6%�	Q�W���A�*;


total_loss�c�@

error_Rn�N?

learning_rate_1�HG7� I       6%�	��W���A�*;


total_loss�`A

error_R��J?

learning_rate_1�HG7�]��I       6%�	hW���A�*;


total_loss�@

error_R��F?

learning_rate_1�HG7�{�]I       6%�	�IW���A�*;


total_lossm��@

error_R�C?

learning_rate_1�HG7.G$�I       6%�	�W���A�*;


total_lossÕ�@

error_R�O?

learning_rate_1�HG7�_�VI       6%�	�W���A�*;


total_lossAog@

error_R��K?

learning_rate_1�HG7
�b�I       6%�	�W���A�*;


total_loss2��@

error_RihM?

learning_rate_1�HG7n�݃I       6%�	�\W���A�*;


total_loss4
�@

error_R1�[?

learning_rate_1�HG7P��I       6%�	l�W���A�*;


total_lossOY�@

error_R�LO?

learning_rate_1�HG7��I       6%�	��W���A�*;


total_loss���@

error_R��b?

learning_rate_1�HG7^#��I       6%�	�$W���A�*;


total_loss�T�@

error_R_,T?

learning_rate_1�HG7�;6I       6%�	CjW���A�*;


total_loss��@

error_RE�??

learning_rate_1�HG7侃#I       6%�	t�W���A�*;


total_loss8[�@

error_R�P?

learning_rate_1�HG7��I       6%�	��W���A�*;


total_loss�z�@

error_R�B?

learning_rate_1�HG7��	I       6%�	Z5W���A�*;


total_loss���@

error_R�8`?

learning_rate_1�HG7�� I       6%�	�zW���A�*;


total_loss�/�@

error_R�G?

learning_rate_1�HG7���I       6%�	d�W���A�*;


total_loss���@

error_R�gZ?

learning_rate_1�HG7���yI       6%�	S W���A�*;


total_loss�F�@

error_R��Y?

learning_rate_1�HG7��xsI       6%�	DW���A�*;


total_lossoɠ@

error_R�J?

learning_rate_1�HG7�`��I       6%�	�W���A�*;


total_loss���@

error_R�-N?

learning_rate_1�HG7H��%I       6%�	��W���A�*;


total_loss�Ȃ@

error_RWwP?

learning_rate_1�HG7�TȍI       6%�	�W���A�*;


total_loss��@

error_R2eL?

learning_rate_1�HG7槩I       6%�	�\W���A�*;


total_loss�
�@

error_R�G?

learning_rate_1�HG75ɦ I       6%�	s�W���A�*;


total_loss���@

error_R��;?

learning_rate_1�HG7_RU�I       6%�	��W���A�*;


total_loss :�@

error_R�E?

learning_rate_1�HG77�p�I       6%�	�MW���A�*;


total_loss���@

error_Rm�O?

learning_rate_1�HG71�yI       6%�	��W���A�*;


total_loss���@

error_R1�U?

learning_rate_1�HG7��~�I       6%�	�W���A�*;


total_loss�W�@

error_R�K?

learning_rate_1�HG7�c)�I       6%�	�vW���A�*;


total_loss㙙@

error_RɁM?

learning_rate_1�HG7����I       6%�	u�W���A�*;


total_loss�(�@

error_R qK?

learning_rate_1�HG7��6�I       6%�	W���A�*;


total_loss��@

error_R��g?

learning_rate_1�HG7LI       6%�	�KW���A�*;


total_lossʾ�@

error_R��S?

learning_rate_1�HG7}\��I       6%�	ϏW���A�*;


total_loss���@

error_R�A?

learning_rate_1�HG7�!��I       6%�	[�W���A�*;


total_loss�"�@

error_R K?

learning_rate_1�HG7�{�5I       6%�	@W���A�*;


total_lossdA�@

error_R3Uf?

learning_rate_1�HG7�ޙcI       6%�	`W���A�*;


total_lossD(�@

error_RC�W?

learning_rate_1�HG7�>�1I       6%�	��W���A�*;


total_loss���@

error_R��H?

learning_rate_1�HG7��I       6%�	��W���A�*;


total_loss�ڬ@

error_R��L?

learning_rate_1�HG7�`�I       6%�	6W���A�*;


total_loss��	A

error_R)�w?

learning_rate_1�HG7�:�I       6%�	�|W���A�*;


total_losszJ�@

error_R�88?

learning_rate_1�HG7s�JI       6%�	۾W���A�*;


total_loss���@

error_R*�F?

learning_rate_1�HG7X ��I       6%�	� W���A�*;


total_loss�9�@

error_RA�C?

learning_rate_1�HG7΂>\I       6%�	I W���A�*;


total_loss��@

error_R8�R?

learning_rate_1�HG7�h.�I       6%�	� W���A�*;


total_lossv�@

error_R<E?

learning_rate_1�HG7�3Z\I       6%�	�� W���A�*;


total_loss���@

error_R��Y?

learning_rate_1�HG7�m�I       6%�	p!W���A�*;


total_loss`��@

error_R�(X?

learning_rate_1�HG7_�j%I       6%�	ET!W���A�*;


total_loss��@

error_R�5H?

learning_rate_1�HG7qƣ�I       6%�	'�!W���A�*;


total_loss���@

error_R�V?

learning_rate_1�HG7`G*�I       6%�	�!W���A�*;


total_loss��@

error_R�D?

learning_rate_1�HG7��8JI       6%�	N"W���A�*;


total_loss6��@

error_R�U?

learning_rate_1�HG7�M�I       6%�	�"W���A�*;


total_loss�o�@

error_R�*U?

learning_rate_1�HG7�i��I       6%�	�"W���A�*;


total_loss2�@

error_R�(J?

learning_rate_1�HG7��I       6%�	�=#W���A�*;


total_loss��@

error_RV�K?

learning_rate_1�HG7y�*�I       6%�	9�#W���A�*;


total_loss�J�@

error_R&T?

learning_rate_1�HG7P_
I       6%�	�#W���A�*;


total_lossת@

error_RNf?

learning_rate_1�HG7Ƚ�)I       6%�	�$W���A�*;


total_loss[�@

error_R?�V?

learning_rate_1�HG7���gI       6%�	lh$W���A�*;


total_loss$m�@

error_R�#M?

learning_rate_1�HG7��#I       6%�	v�$W���A�*;


total_loss���@

error_R�L?

learning_rate_1�HG7B@pwI       6%�	��$W���A�*;


total_loss�t+A

error_R8Gb?

learning_rate_1�HG7�zCaI       6%�	qA%W���A�*;


total_loss�-�@

error_R� Y?

learning_rate_1�HG7̹;7I       6%�	ɇ%W���A�*;


total_lossn��@

error_R�K?

learning_rate_1�HG7�81I       6%�	��%W���A�*;


total_loss�c�@

error_RZ�b?

learning_rate_1�HG76g��I       6%�	F&W���A�*;


total_lossM0�@

error_R;�S?

learning_rate_1�HG7���I       6%�	V&W���A�*;


total_lossZ�@

error_R�F??

learning_rate_1�HG7zN�I       6%�	j�&W���A�*;


total_loss2k�@

error_Ri)I?

learning_rate_1�HG7�	ӉI       6%�	��&W���A�*;


total_lossoE�@

error_R-O?

learning_rate_1�HG7"�{AI       6%�	�.'W���A�*;


total_losso|�@

error_R��P?

learning_rate_1�HG7x�ܓI       6%�	0z'W���A�*;


total_loss�{�@

error_R��I?

learning_rate_1�HG7�D��I       6%�	2�'W���A�*;


total_loss(�@

error_R��N?

learning_rate_1�HG7e�*�I       6%�	�(W���A�*;


total_lossq��@

error_R�Z\?

learning_rate_1�HG7�]S	I       6%�	�D(W���A�*;


total_loss���@

error_ReuS?

learning_rate_1�HG7�|��I       6%�	��(W���A�*;


total_loss���@

error_R ??

learning_rate_1�HG7�CV�I       6%�	��(W���A�*;


total_loss���@

error_R߷O?

learning_rate_1�HG7m�I       6%�	2)W���A�*;


total_lossO��@

error_R�t9?

learning_rate_1�HG7M���I       6%�	<T)W���A�*;


total_loss6��@

error_R��T?

learning_rate_1�HG7�~�I       6%�	��)W���A�*;


total_loss�@

error_R!�V?

learning_rate_1�HG74�I       6%�	��)W���A�*;


total_loss���@

error_R��J?

learning_rate_1�HG7�G�I       6%�	#!*W���A�*;


total_loss�I�@

error_R �E?

learning_rate_1�HG7�؍"I       6%�	ha*W���A�*;


total_loss��@

error_RoP8?

learning_rate_1�HG7ME��I       6%�	�*W���A�*;


total_loss���@

error_R3�a?

learning_rate_1�HG7����I       6%�	�*W���A�*;


total_lossQ)�@

error_R�J?

learning_rate_1�HG7�rZ�I       6%�	�K+W���A�*;


total_lossVA

error_RsP?

learning_rate_1�HG7rG�I       6%�	h�+W���A�*;


total_lossQـ@

error_R��R?

learning_rate_1�HG7!�PI       6%�	��+W���A�*;


total_loss���@

error_Rr�;?

learning_rate_1�HG7'�x�I       6%�	�,W���A�*;


total_loss���@

error_R�SG?

learning_rate_1�HG7����I       6%�	�c,W���A�*;


total_lossܭ�@

error_RJ�I?

learning_rate_1�HG7�SA�I       6%�	��,W���A�*;


total_loss��@

error_R�XN?

learning_rate_1�HG7���I       6%�	��,W���A�*;


total_lossl��@

error_R�/S?

learning_rate_1�HG7��I       6%�	0-W���A�*;


total_loss��@

error_R��h?

learning_rate_1�HG7%C�I       6%�	�s-W���A�*;


total_loss�!�@

error_RC�<?

learning_rate_1�HG7Ù=I       6%�	ĺ-W���A�*;


total_losso�@

error_Rn�Y?

learning_rate_1�HG7D��I       6%�	��-W���A�*;


total_lossh�@

error_R0H?

learning_rate_1�HG7 �I       6%�	�B.W���A�*;


total_loss暝@

error_Ri�@?

learning_rate_1�HG7g�~uI       6%�	��.W���A�*;


total_loss���@

error_Rd|>?

learning_rate_1�HG76�I       6%�	"�.W���A�*;


total_lossx�@

error_R��S?

learning_rate_1�HG7)Z�"I       6%�	�/W���A�*;


total_loss�p�@

error_R��C?

learning_rate_1�HG7]zI       6%�	8Q/W���A�*;


total_loss���@

error_R��S?

learning_rate_1�HG7�*I       6%�	��/W���A�*;


total_loss�5�@

error_R�J?

learning_rate_1�HG7]�7%I       6%�	0�/W���A�*;


total_loss�9�@

error_R�C[?

learning_rate_1�HG7���I       6%�	�0W���A�*;


total_loss��@

error_R��<?

learning_rate_1�HG7y��!I       6%�	?f0W���A�*;


total_lossW��@

error_RZ<6?

learning_rate_1�HG7o��I       6%�	?�0W���A�*;


total_loss�s�@

error_R!�Y?

learning_rate_1�HG7J\[oI       6%�	W�0W���A�*;


total_lossS�@

error_R\�W?

learning_rate_1�HG7{��aI       6%�	�61W���A�*;


total_loss��m@

error_R=9??

learning_rate_1�HG7:��I       6%�	�1W���A�*;


total_loss @�@

error_R]8T?

learning_rate_1�HG7)�.I       6%�	��1W���A�*;


total_loss���@

error_RvW?

learning_rate_1�HG7���II       6%�	<2W���A�*;


total_loss���@

error_R�B?

learning_rate_1�HG7�P�I       6%�	lV2W���A�*;


total_lossa��@

error_R��P?

learning_rate_1�HG7ǅ�I       6%�	ޝ2W���A�*;


total_lossM=�@

error_R�`?

learning_rate_1�HG7Y�^I       6%�	
�2W���A�*;


total_loss1�@

error_R)mX?

learning_rate_1�HG7)��I       6%�	G(3W���A�*;


total_lossʓ�@

error_R@5?

learning_rate_1�HG7����I       6%�	�r3W���A�*;


total_loss.3�@

error_R��]?

learning_rate_1�HG7��aI       6%�	��3W���A�*;


total_lossn2�@

error_R�K?

learning_rate_1�HG7�1@I       6%�	�4W���A�*;


total_loss��@

error_R��5?

learning_rate_1�HG7!nMI       6%�	�F4W���A�*;


total_loss�9�@

error_R bF?

learning_rate_1�HG7�>KFI       6%�	��4W���A�*;


total_loss�1�@

error_R�;?

learning_rate_1�HG7����I       6%�	�4W���A�*;


total_loss�I�@

error_RȂS?

learning_rate_1�HG7�:{�I       6%�	[$5W���A�*;


total_loss�1�@

error_R1�K?

learning_rate_1�HG7B1?.I       6%�	<m5W���A�*;


total_loss�@

error_R%Ke?

learning_rate_1�HG7W��I       6%�	v�5W���A�*;


total_loss:�A

error_R�Z_?

learning_rate_1�HG7P�uI       6%�	� 6W���A�*;


total_loss�@

error_RMY?

learning_rate_1�HG7���I       6%�	�I6W���A�*;


total_losso
�@

error_RwNI?

learning_rate_1�HG7��8�I       6%�	�6W���A�*;


total_loss���@

error_R��^?

learning_rate_1�HG7�\�I       6%�	;�6W���A�*;


total_loss�J�@

error_R&�Z?

learning_rate_1�HG7'�I       6%�	�)7W���A�*;


total_loss��A

error_Rf!B?

learning_rate_1�HG7ﵓ*I       6%�	�s7W���A�*;


total_lossAǣ@

error_R��a?

learning_rate_1�HG7Ђ�I       6%�	~�7W���A�*;


total_loss�y�@

error_R��S?

learning_rate_1�HG7q���I       6%�	w�7W���A�*;


total_loss���@

error_R|�I?

learning_rate_1�HG7��f�I       6%�	�G8W���A�*;


total_loss���@

error_RO0B?

learning_rate_1�HG7s):XI       6%�	݋8W���A�*;


total_loss���@

error_R�HK?

learning_rate_1�HG7��� I       6%�	��8W���A�*;


total_loss3��@

error_RWqN?

learning_rate_1�HG7��DI       6%�	�9W���A�*;


total_loss�L�@

error_R�T`?

learning_rate_1�HG7V�}I       6%�	�U9W���A�*;


total_lossl��@

error_RC�R?

learning_rate_1�HG7��EjI       6%�	�9W���A�*;


total_losssA

error_R6
L?

learning_rate_1�HG7vsF�I       6%�	��9W���A�*;


total_loss�_@

error_R�eE?

learning_rate_1�HG7��b�I       6%�	[:W���A�*;


total_loss��@

error_Rs^?

learning_rate_1�HG7���I       6%�	`:W���A�*;


total_loss��@

error_R�qI?

learning_rate_1�HG7���pI       6%�	��:W���A�*;


total_lossxm�@

error_R�TP?

learning_rate_1�HG7�o�6I       6%�	��:W���A�*;


total_loss�x�@

error_R��L?

learning_rate_1�HG7�U0�I       6%�	EX;W���A�*;


total_lossm�A

error_RC?

learning_rate_1�HG7֞s�I       6%�	��;W���A�*;


total_loss��A

error_R�(J?

learning_rate_1�HG7�]|�I       6%�	��;W���A�*;


total_loss�_�@

error_R��D?

learning_rate_1�HG7��!�I       6%�	�*<W���A�*;


total_loss���@

error_Rs�Z?

learning_rate_1�HG7�L�I       6%�	�n<W���A�*;


total_loss{�@

error_R̱G?

learning_rate_1�HG7,���I       6%�	1�<W���A�*;


total_loss�N�@

error_R=�c?

learning_rate_1�HG7�0�I       6%�	�<W���A�*;


total_lossdXw@

error_R�A?

learning_rate_1�HG7�^� I       6%�	F;=W���A�*;


total_loss���@

error_R=o:?

learning_rate_1�HG7 NrI       6%�	=W���A�*;


total_lossLj�@

error_RRbR?

learning_rate_1�HG7��I       6%�	��=W���A�*;


total_loss��@

error_R.�E?

learning_rate_1�HG7�ߏ�I       6%�	d >W���A�*;


total_loss��@

error_R��T?

learning_rate_1�HG7uӽwI       6%�	)f>W���A�*;


total_loss.��@

error_R�6a?

learning_rate_1�HG7�D߈I       6%�	��>W���A�*;


total_loss
��@

error_R�|E?

learning_rate_1�HG7F��hI       6%�	��>W���A�*;


total_lossi�%A

error_Rۛ3?

learning_rate_1�HG7�NPcI       6%�	�6?W���A�*;


total_loss\ZA

error_Rc�9?

learning_rate_1�HG7�|�hI       6%�		y?W���A�*;


total_lossZ��@

error_R�LZ?

learning_rate_1�HG7�h��I       6%�	��?W���A�*;


total_loss�ّ@

error_R��3?

learning_rate_1�HG7{b��I       6%�	�@W���A�*;


total_lossW�@

error_R��F?

learning_rate_1�HG7h�0rI       6%�	^M@W���A�*;


total_loss��@

error_R�J?

learning_rate_1�HG7���I       6%�	ɒ@W���A�*;


total_loss�V�@

error_RadQ?

learning_rate_1�HG7Nz�MI       6%�	��@W���A�*;


total_loss�Վ@

error_Rq�N?

learning_rate_1�HG7@�\NI       6%�	�AW���A�*;


total_loss��A

error_R#�Y?

learning_rate_1�HG7#��RI       6%�	dAW���A�*;


total_loss�Ü@

error_R��D?

learning_rate_1�HG7���I       6%�	,�AW���A�*;


total_loss���@

error_R�|>?

learning_rate_1�HG7ڠnyI       6%�	��AW���A�*;


total_loss,A

error_R��I?

learning_rate_1�HG7aK2+I       6%�	M5BW���A�*;


total_loss-V�@

error_R�[I?

learning_rate_1�HG7�	�I       6%�	w|BW���A�*;


total_lossAL�@

error_RZ�j?

learning_rate_1�HG7��w,I       6%�	��BW���A�*;


total_lossF7�@

error_RSI?

learning_rate_1�HG7U��I       6%�	!CW���A�*;


total_loss�AE@

error_RWM?

learning_rate_1�HG7��(�I       6%�	vWCW���A�*;


total_lossM�@

error_R�@W?

learning_rate_1�HG7F�I       6%�	��CW���A�*;


total_loss숡@

error_R�1T?

learning_rate_1�HG7/"��I       6%�	��CW���A�*;


total_loss�^�@

error_R%G?

learning_rate_1�HG7w�رI       6%�	�+DW���A�*;


total_loss��@

error_R��F?

learning_rate_1�HG7�?��I       6%�	�rDW���A�*;


total_loss�|P@

error_R�@F?

learning_rate_1�HG7��=pI       6%�	�DW���A�*;


total_loss��@

error_R.J??

learning_rate_1�HG7CC�I       6%�	��DW���A�*;


total_loss�@

error_RM�@?

learning_rate_1�HG7�I       6%�	EBEW���A�*;


total_loss�5}@

error_R��7?

learning_rate_1�HG7�?{I       6%�	ыEW���A�*;


total_loss�}�@

error_R�8K?

learning_rate_1�HG7�2I       6%�	�EW���A�*;


total_loss׷�@

error_R�B?

learning_rate_1�HG7H(��I       6%�	�FW���A�*;


total_lossO�n@

error_R�6]?

learning_rate_1�HG7�}�TI       6%�	kfFW���A�*;


total_lossT��@

error_R��=?

learning_rate_1�HG7ž��I       6%�	L�FW���A�*;


total_lossE��@

error_R�BZ?

learning_rate_1�HG7}�,LI       6%�	1�FW���A�*;


total_lossh��@

error_R��S?

learning_rate_1�HG71}W*I       6%�	6GW���A�*;


total_loss���@

error_R��E?

learning_rate_1�HG7���I       6%�	~GW���A�*;


total_loss�S�@

error_R �Z?

learning_rate_1�HG7�b.uI       6%�	��GW���A�*;


total_lossvs�@

error_R}9^?

learning_rate_1�HG7]���I       6%�	�HW���A�*;


total_loss}�@

error_R��O?

learning_rate_1�HG7~?�|I       6%�	�LHW���A�*;


total_loss2�@

error_R.SS?

learning_rate_1�HG79�BI       6%�	֏HW���A�*;


total_loss7��@

error_R��T?

learning_rate_1�HG7��0I       6%�	��HW���A�*;


total_loss=Z�@

error_R�NJ?

learning_rate_1�HG7���I       6%�	�IW���A�*;


total_loss8n�@

error_R_;K?

learning_rate_1�HG7&�MI       6%�	�WIW���A�*;


total_loss��@

error_R�VN?

learning_rate_1�HG7��Q�I       6%�	x�IW���A�*;


total_lossFV�@

error_R�As?

learning_rate_1�HG7>��=I       6%�	-�IW���A�*;


total_lossp��@

error_R��<?

learning_rate_1�HG7�]�I       6%�	�#JW���A�*;


total_loss}��@

error_R�cP?

learning_rate_1�HG7��I       6%�	�hJW���A�*;


total_loss���@

error_R\�X?

learning_rate_1�HG7��VI       6%�	��JW���A�*;


total_lossWi�@

error_R:�U?

learning_rate_1�HG7ށr�I       6%�	��JW���A�*;


total_loss���@

error_R�|]?

learning_rate_1�HG7;
�I       6%�	�PKW���A�*;


total_lossC��@

error_RM�C?

learning_rate_1�HG7���I       6%�	�KW���A�*;


total_loss�A

error_R��P?

learning_rate_1�HG7��'I       6%�	�KW���A�*;


total_loss���@

error_R�AU?

learning_rate_1�HG7e��I       6%�	!LW���A�*;


total_lossWQ�@

error_RʂR?

learning_rate_1�HG7�'(/I       6%�	'dLW���A�*;


total_loss��[@

error_R�0X?

learning_rate_1�HG7l ��I       6%�	�LW���A�*;


total_loss���@

error_RM�P?

learning_rate_1�HG7���I       6%�	\�LW���A�*;


total_lossW��@

error_Rf�M?

learning_rate_1�HG7�O�I       6%�	�6MW���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7�s�LI       6%�	k{MW���A�*;


total_loss��@

error_R/
S?

learning_rate_1�HG7x���I       6%�	��MW���A�*;


total_loss���@

error_R��@?

learning_rate_1�HG7�]I       6%�	��MW���A�*;


total_lossv�@

error_R�@P?

learning_rate_1�HG7�L6�I       6%�	�BNW���A�*;


total_loss�#�@

error_R��X?

learning_rate_1�HG7,W�I       6%�	y�NW���A�*;


total_loss�B�@

error_R��[?

learning_rate_1�HG7��_�I       6%�	��NW���A�*;


total_loss���@

error_Rx�M?

learning_rate_1�HG7��P�I       6%�	�OW���A�*;


total_loss��@

error_R�P?

learning_rate_1�HG7=y}�I       6%�	XOW���A�*;


total_loss�ؑ@

error_R��O?

learning_rate_1�HG7-\��I       6%�	'�OW���A�*;


total_loss̋@

error_R��H?

learning_rate_1�HG7Is1I       6%�	.�OW���A�*;


total_loss�@

error_R�??

learning_rate_1�HG7��K�I       6%�	!PW���A�*;


total_loss���@

error_R�R?

learning_rate_1�HG7�5�I       6%�	�dPW���A�*;


total_lossܠ|@

error_R~P?

learning_rate_1�HG7�5�.I       6%�	�PW���A�*;


total_lossｔ@

error_R@9D?

learning_rate_1�HG7�Z�yI       6%�	=QW���A�*;


total_loss?�@

error_R��Y?

learning_rate_1�HG7��\I       6%�	�QQW���A�*;


total_lossO�a@

error_R��H?

learning_rate_1�HG7��GBI       6%�	ʗQW���A�*;


total_loss %�@

error_R��P?

learning_rate_1�HG7�NcI       6%�	,�QW���A�*;


total_loss��@

error_R��F?

learning_rate_1�HG7�DI       6%�	63RW���A�*;


total_loss�*�@

error_R�T?

learning_rate_1�HG7s�D#I       6%�	�wRW���A�*;


total_lossW��@

error_R��M?

learning_rate_1�HG7c~�SI       6%�	�RW���A�*;


total_loss���@

error_R�e?

learning_rate_1�HG7fyD2I       6%�	�%SW���A�*;


total_loss�j�@

error_RmP?

learning_rate_1�HG7g?TI       6%�	tSW���A�*;


total_lossOI�@

error_R�.X?

learning_rate_1�HG7�T-,I       6%�	¾SW���A�*;


total_loss�ߦ@

error_R�EP?

learning_rate_1�HG7�q�I       6%�	M(TW���A�*;


total_lossHS�@

error_R�Y?

learning_rate_1�HG7r��I       6%�	�mTW���A�*;


total_loss��@

error_R A[?

learning_rate_1�HG7�I       6%�	�TW���A�*;


total_loss���@

error_R�~O?

learning_rate_1�HG7\��YI       6%�	HUW���A�*;


total_loss��A

error_RX{L?

learning_rate_1�HG7�>�&I       6%�	�YUW���A�*;


total_loss�@

error_R��A?

learning_rate_1�HG7�b�xI       6%�	��UW���A�*;


total_loss��@

error_RiK?

learning_rate_1�HG7kI�I       6%�	��UW���A�*;


total_loss�,�@

error_R!ZZ?

learning_rate_1�HG7����I       6%�	e?VW���A�*;


total_loss�!�@

error_R�BP?

learning_rate_1�HG7��I       6%�	!�VW���A�*;


total_loss�Ȋ@

error_R�a?

learning_rate_1�HG7j�G)I       6%�	��VW���A�*;


total_loss�c�@

error_R=HA?

learning_rate_1�HG7a��EI       6%�	KWW���A�*;


total_lossZۺ@

error_R�^T?

learning_rate_1�HG7�wE-I       6%�	�eWW���A�*;


total_loss��@

error_R��D?

learning_rate_1�HG7�b��I       6%�	��WW���A�*;


total_loss��@

error_R�]D?

learning_rate_1�HG7p�+�I       6%�	�WW���A�*;


total_loss���@

error_R\D5?

learning_rate_1�HG7���I       6%�	KLXW���A�*;


total_loss�0A

error_R�OH?

learning_rate_1�HG7"F��I       6%�	��XW���A�*;


total_loss���@

error_R�D?

learning_rate_1�HG7etDI       6%�	X�XW���A�*;


total_loss�@

error_R�6?

learning_rate_1�HG7�"�I       6%�	�9YW���A�*;


total_loss�g@

error_R�J[?

learning_rate_1�HG7�5�I       6%�	��YW���A�*;


total_loss4��@

error_R�E?

learning_rate_1�HG7A�I       6%�	��YW���A�*;


total_loss���@

error_R�T?

learning_rate_1�HG7@F/I       6%�	VZW���A�*;


total_loss�K�@

error_R�0K?

learning_rate_1�HG7'��I       6%�	tZW���A�*;


total_loss2��@

error_R�=;?

learning_rate_1�HG7��߁I       6%�	��ZW���A�*;


total_loss�*�@

error_R��P?

learning_rate_1�HG7��<
I       6%�	� [W���A�*;


total_loss�`@

error_R\	8?

learning_rate_1�HG7l���I       6%�	��[W���A�*;


total_loss$�@

error_Rs�N?

learning_rate_1�HG7�Os_I       6%�	��[W���A�*;


total_loss�)�@

error_RLkI?

learning_rate_1�HG7���I       6%�	�<\W���A�*;


total_loss�
�@

error_R�uF?

learning_rate_1�HG7�x"mI       6%�	��\W���A�*;


total_loss.|�@

error_R�y7?

learning_rate_1�HG7�0�I       6%�	��\W���A�*;


total_lossʽ@

error_R)�[?

learning_rate_1�HG7�S;�I       6%�	�0]W���A�*;


total_loss U�@

error_R�;I?

learning_rate_1�HG7�1EfI       6%�	��]W���A�*;


total_loss��@

error_RC	b?

learning_rate_1�HG7n�{I       6%�	��]W���A�*;


total_loss]��@

error_R�TQ?

learning_rate_1�HG7�^I       6%�	�^W���A�*;


total_loss��@

error_R1�5?

learning_rate_1�HG7/#�I       6%�	_^W���A�*;


total_lossZe�@

error_R��H?

learning_rate_1�HG7�)�I       6%�	��^W���A�*;


total_loss���@

error_R��O?

learning_rate_1�HG71R�I       6%�	��^W���A�*;


total_loss��@

error_R�gU?

learning_rate_1�HG7?��I       6%�	�D_W���A�*;


total_loss�͠@

error_Rl�V?

learning_rate_1�HG71[��I       6%�	��_W���A�*;


total_lossMX�@

error_Rv�K?

learning_rate_1�HG7we�I       6%�	��_W���A�*;


total_loss���@

error_R4WN?

learning_rate_1�HG7f���I       6%�	�,`W���A�*;


total_loss�:�@

error_R/q^?

learning_rate_1�HG7�x_LI       6%�	�s`W���A�*;


total_loss���@

error_R�gW?

learning_rate_1�HG7�W�I       6%�	��`W���A�*;


total_loss[�A

error_R�M?

learning_rate_1�HG7F��TI       6%�	-aW���A�*;


total_loss�T�@

error_R;\@?

learning_rate_1�HG7-�.QI       6%�	nSaW���A�*;


total_loss.��@

error_R�F?

learning_rate_1�HG70��I       6%�	U�aW���A�*;


total_loss\d�@

error_Rs�S?

learning_rate_1�HG7�N� I       6%�	�bW���A�*;


total_lossQ��@

error_Rx?K?

learning_rate_1�HG7fq�I       6%�	fLbW���A�*;


total_lossδ@

error_RAiE?

learning_rate_1�HG7���#I       6%�	m�bW���A�*;


total_loss�#A

error_R�FP?

learning_rate_1�HG7��q�I       6%�	%�bW���A�*;


total_losst��@

error_R�=I?

learning_rate_1�HG7��9LI       6%�	cW���A�*;


total_loss�ܲ@

error_R\�3?

learning_rate_1�HG7���	I       6%�	>jcW���A�*;


total_loss�q�@

error_RC�J?

learning_rate_1�HG7A���I       6%�	A�cW���A�*;


total_lossZ��@

error_R��=?

learning_rate_1�HG75�I       6%�	��cW���A�*;


total_loss�r�@

error_RJ�4?

learning_rate_1�HG7ʻ}I       6%�	�6dW���A�*;


total_loss\�|@

error_R�.O?

learning_rate_1�HG7N���I       6%�	g}dW���A�*;


total_loss���@

error_Rv�/?

learning_rate_1�HG7�eJ�I       6%�	?�dW���A�*;


total_lossXP�@

error_R/�9?

learning_rate_1�HG7<�_I       6%�	neW���A�*;


total_loss��@

error_R3;D?

learning_rate_1�HG7�V�I       6%�	eMeW���A�*;


total_loss�`�@

error_R��N?

learning_rate_1�HG78e��I       6%�	��eW���A�*;


total_loss�0�@

error_R��B?

learning_rate_1�HG7ٵ��I       6%�	��eW���A�*;


total_loss���@

error_R�~V?

learning_rate_1�HG75XiI       6%�	ifW���A�*;


total_lossؠ�@

error_R��G?

learning_rate_1�HG7Q�p:I       6%�	�`fW���A�*;


total_loss�[�@

error_R֋J?

learning_rate_1�HG7WنI       6%�	��fW���A�*;


total_loss 
/A

error_RsU?

learning_rate_1�HG7\��HI       6%�	+�fW���A�*;


total_loss8��@

error_R��X?

learning_rate_1�HG7���LI       6%�	(+gW���A�*;


total_loss]�r@

error_R�~N?

learning_rate_1�HG7�
�GI       6%�	wogW���A�*;


total_loss�"�@

error_RE�`?

learning_rate_1�HG7�S�TI       6%�	��gW���A�*;


total_lossJ��@

error_R�	??

learning_rate_1�HG7<�O�I       6%�	��gW���A�*;


total_lossl+�@

error_R�D?

learning_rate_1�HG7��vI       6%�	vGhW���A�*;


total_loss;��@

error_R��D?

learning_rate_1�HG7u�`�I       6%�	��hW���A�*;


total_loss�
A

error_R�b_?

learning_rate_1�HG7�L�I       6%�	o�hW���A�*;


total_loss��@

error_R�qC?

learning_rate_1�HG7l�GI       6%�	\iW���A�*;


total_lossH-�@

error_R��Y?

learning_rate_1�HG7�p�I       6%�	lQiW���A�*;


total_loss%G�@

error_R��O?

learning_rate_1�HG7>���I       6%�	u�iW���A�*;


total_loss�ަ@

error_R$�9?

learning_rate_1�HG7�y$I       6%�	f�iW���A�*;


total_loss���@

error_R	�>?

learning_rate_1�HG7�8_,I       6%�	 8jW���A�*;


total_loss��T@

error_R;'Q?

learning_rate_1�HG7[Uy�I       6%�	�|jW���A�*;


total_loss�0A

error_RtNM?

learning_rate_1�HG7�5p�I       6%�	c�jW���A�*;


total_loss�rZ@

error_R�H?

learning_rate_1�HG7�p�I       6%�	�FkW���A�*;


total_loss#m�@

error_R\�T?

learning_rate_1�HG7'c�uI       6%�	��kW���A�*;


total_loss2��@

error_Ra]S?

learning_rate_1�HG7hRݬI       6%�	�kW���A�*;


total_loss1C�@

error_RT�L?

learning_rate_1�HG7��D{I       6%�	�lW���A�*;


total_lossgϛ@

error_R$�Y?

learning_rate_1�HG7�5x�I       6%�	?jlW���A�*;


total_loss3*�@

error_R��B?

learning_rate_1�HG7 �J�I       6%�	ƷlW���A�*;


total_lossV_~@

error_R׬B?

learning_rate_1�HG7��O�I       6%�	��lW���A�*;


total_loss�s�@

error_RD?

learning_rate_1�HG7&	I       6%�	1EmW���A�*;


total_loss�k�@

error_Rd�D?

learning_rate_1�HG7f��I       6%�	V�mW���A�*;


total_loss���@

error_R|L?

learning_rate_1�HG7}`J�I       6%�	N�mW���A�*;


total_loss�A�@

error_R��Q?

learning_rate_1�HG7���I       6%�	;nW���A�*;


total_lossqP�@

error_R�$K?

learning_rate_1�HG7CgI       6%�	ygnW���A�*;


total_loss���@

error_R�![?

learning_rate_1�HG7���I       6%�	*�nW���A�*;


total_loss�=�@

error_R��S?

learning_rate_1�HG75T��I       6%�	��nW���A�*;


total_losscA

error_R�b?

learning_rate_1�HG7�兝I       6%�	 3oW���A�*;


total_lossH�@

error_RSDS?

learning_rate_1�HG7��I       6%�	�uoW���A�*;


total_lossҾ�@

error_R��F?

learning_rate_1�HG7B[zI       6%�	|�oW���A�*;


total_lossz��@

error_R	>D?

learning_rate_1�HG7��\I       6%�	d�oW���A�*;


total_losse�@

error_R}�T?

learning_rate_1�HG7i�$8I       6%�	DpW���A�*;


total_loss��@

error_R!�L?

learning_rate_1�HG7�J�"I       6%�	Q�pW���A�*;


total_loss�C�@

error_R�Y?

learning_rate_1�HG7���pI       6%�	��pW���A�*;


total_lossOv�@

error_R�V?

learning_rate_1�HG7�R	/I       6%�	� qW���A�*;


total_loss`#�@

error_R�H?

learning_rate_1�HG7��ةI       6%�	�dqW���A�*;


total_loss̷g@

error_R��9?

learning_rate_1�HG7���I       6%�	Y�qW���A�*;


total_lossd��@

error_R��U?

learning_rate_1�HG7�I       6%�	+�qW���A�*;


total_loss�v�@

error_Rm�Z?

learning_rate_1�HG7��[I       6%�	�7rW���A�*;


total_lossO�@

error_R�MP?

learning_rate_1�HG7�U;TI       6%�	łrW���A�*;


total_loss���@

error_R|�@?

learning_rate_1�HG7���aI       6%�	F�rW���A�*;


total_lossD��@

error_RcJ??

learning_rate_1�HG7x��pI       6%�	�sW���A�*;


total_loss�� A

error_R1�R?

learning_rate_1�HG7Sm�I       6%�	�YsW���A�*;


total_loss4�A

error_R��L?

learning_rate_1�HG7��FI       6%�	��sW���A�*;


total_loss&f�@

error_RT�_?

learning_rate_1�HG7�x��I       6%�	�sW���A�*;


total_loss.�@

error_R �W?

learning_rate_1�HG7K�p�I       6%�	L)tW���A�*;


total_loss4�@

error_R�R?

learning_rate_1�HG7v��I       6%�	!otW���A�*;


total_loss:�@

error_R��N?

learning_rate_1�HG7t�+I       6%�	|�tW���A�*;


total_loss��@

error_R,`f?

learning_rate_1�HG74�f�I       6%�	��tW���A�*;


total_lossc��@

error_RK[?

learning_rate_1�HG7�Rq�I       6%�	/;uW���A�*;


total_loss_ʃ@

error_R,�]?

learning_rate_1�HG7�g�I       6%�	_uW���A�*;


total_lossF�@

error_R5S?

learning_rate_1�HG7/)�
I       6%�	e�uW���A�*;


total_loss���@

error_R��P?

learning_rate_1�HG7}8��I       6%�	FvW���A�*;


total_loss�r�@

error_R�`S?

learning_rate_1�HG7H��iI       6%�	�IvW���A�*;


total_loss���@

error_Rm�6?

learning_rate_1�HG7c���I       6%�	`�vW���A�*;


total_loss(��@

error_RV�K?

learning_rate_1�HG7���I       6%�	��vW���A�*;


total_lossx�@

error_R�;M?

learning_rate_1�HG7��V�I       6%�	wwW���A�*;


total_loss��@

error_R_�W?

learning_rate_1�HG7	��I       6%�	�XwW���A�*;


total_loss��@

error_R��_?

learning_rate_1�HG7�B�cI       6%�	��wW���A�*;


total_loss�gA

error_R@D?

learning_rate_1�HG77�I       6%�	%�wW���A�*;


total_loss�l�@

error_R/UL?

learning_rate_1�HG7én�I       6%�	96xW���A�*;


total_loss%�@

error_R}oJ?

learning_rate_1�HG7l��I       6%�	�xW���A�*;


total_loss�J�@

error_R�#Q?

learning_rate_1�HG7Q׬mI       6%�	R�xW���A�*;


total_loss턔@

error_R�$E?

learning_rate_1�HG7Ih�vI       6%�	{yW���A�*;


total_loss�,	A

error_R�W?

learning_rate_1�HG7cqpI       6%�	�VyW���A�*;


total_lossonA

error_RIQT?

learning_rate_1�HG7����I       6%�	�yW���A�*;


total_lossﱻ@

error_R\�N?

learning_rate_1�HG7#�I       6%�	~�yW���A�*;


total_loss�uGA

error_R��K?

learning_rate_1�HG7y�!�I       6%�	�'zW���A�*;


total_lossr��@

error_R?DQ?

learning_rate_1�HG7j���I       6%�	jzW���A�*;


total_loss*A

error_R�C?

learning_rate_1�HG7�w�I       6%�	#�zW���A�*;


total_losso�@

error_Rv_R?

learning_rate_1�HG7��ķI       6%�	-�zW���A�*;


total_loss4�@

error_R�[?

learning_rate_1�HG7w��HI       6%�	X{W���A�*;


total_loss�T�@

error_RES?

learning_rate_1�HG7>�ݸI       6%�	+�{W���A�*;


total_loss���@

error_R;�K?

learning_rate_1�HG7�m9I       6%�	��{W���A�*;


total_loss��@

error_RE?

learning_rate_1�HG7�.[I       6%�	�6|W���A�*;


total_loss��@

error_R�pL?

learning_rate_1�HG7=tA�I       6%�	��|W���A�*;


total_loss�G�@

error_R��X?

learning_rate_1�HG7�X�I       6%�	�|W���A�*;


total_loss�i�@

error_R�1I?

learning_rate_1�HG73NI       6%�	�}W���A�*;


total_loss��@

error_R)�a?

learning_rate_1�HG7��	I       6%�	Z}W���A�*;


total_lossZ<�@

error_R�N?

learning_rate_1�HG7'��AI       6%�	B�}W���A�*;


total_loss{�@

error_R��H?

learning_rate_1�HG7��.I       6%�	3�}W���A�*;


total_lossc��@

error_R��P?

learning_rate_1�HG7԰݁I       6%�	{3~W���A�*;


total_loss}��@

error_R�'P?

learning_rate_1�HG7[�xI       6%�	�v~W���A�*;


total_loss�W�@

error_RCZ?

learning_rate_1�HG7>�I       6%�	b�~W���A�*;


total_loss��@

error_R��[?

learning_rate_1�HG7�MB�I       6%�	bW���A�*;


total_loss/��@

error_R..F?

learning_rate_1�HG75E�'I       6%�	gFW���A�*;


total_lossQa�@

error_R�<?

learning_rate_1�HG7�O�I       6%�	ȉW���A�*;


total_loss��@

error_R��T?

learning_rate_1�HG7�[^OI       6%�	��W���A�*;


total_loss���@

error_R?J>?

learning_rate_1�HG7I
��I       6%�	}�W���A�*;


total_lossܢ�@

error_RSkA?

learning_rate_1�HG7Γ"�I       6%�	VY�W���A�*;


total_loss�|@

error_R�]J?

learning_rate_1�HG7�1�}I       6%�	9��W���A�*;


total_losso[�@

error_R|hF?

learning_rate_1�HG7Q�ӏI       6%�	��W���A�*;


total_loss&�(A

error_RyQ?

learning_rate_1�HG7���I       6%�	�-�W���A�*;


total_loss}�f@

error_R��X?

learning_rate_1�HG7^Y~�I       6%�	Nn�W���A�*;


total_loss͉�@

error_R�V?

learning_rate_1�HG7�EUUI       6%�	ش�W���A�*;


total_lossZ�@

error_R�K?

learning_rate_1�HG7�b�GI       6%�	���W���A�*;


total_loss�ݜ@

error_R�}G?

learning_rate_1�HG7=��I       6%�	F�W���A�*;


total_loss��@

error_RW�G?

learning_rate_1�HG7�I       6%�	��W���A�*;


total_loss�{�@

error_R�S?

learning_rate_1�HG7�5~I       6%�	�ӂW���A�*;


total_lossP<"A

error_Ri�M?

learning_rate_1�HG7]WI       6%�	%�W���A�*;


total_loss_�@

error_R8??

learning_rate_1�HG7��LdI       6%�	�X�W���A�*;


total_loss<�@

error_R�kE?

learning_rate_1�HG7b��)I       6%�	���W���A�*;


total_lossM�@

error_R�J?

learning_rate_1�HG7�[�#I       6%�	��W���A�*;


total_loss8�@

error_R\Q?

learning_rate_1�HG7��I       6%�	�*�W���A�*;


total_lossг�@

error_RqC?

learning_rate_1�HG7��s�I       6%�	_u�W���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7���qI       6%�	���W���A�*;


total_loss��@

error_Rf�Z?

learning_rate_1�HG76q�6I       6%�	E�W���A�*;


total_loss���@

error_R:^G?

learning_rate_1�HG7���I       6%�	�S�W���A�*;


total_loss�~�@

error_R��c?

learning_rate_1�HG7���I       6%�	X��W���A�*;


total_lossF��@

error_RDcY?

learning_rate_1�HG7��xI       6%�	KمW���A�*;


total_lossy�@

error_RI�Y?

learning_rate_1�HG7[DdI       6%�	�W���A�*;


total_loss�ί@

error_RC�R?

learning_rate_1�HG7`�WoI       6%�	?a�W���A�*;


total_loss�j�@

error_RE>?

learning_rate_1�HG7+fZAI       6%�	"��W���A�*;


total_lossz��@

error_R��@?

learning_rate_1�HG7���dI       6%�	X�W���A�*;


total_loss7B�@

error_R�S?

learning_rate_1�HG7�<� I       6%�	�+�W���A�*;


total_lossn��@

error_R��A?

learning_rate_1�HG7��5I       6%�	rp�W���A�*;


total_loss`C�@

error_R@�H?

learning_rate_1�HG7�E�JI       6%�	ַ�W���A�*;


total_loss �@

error_RJ8P?

learning_rate_1�HG7�S<�I       6%�	k�W���A�*;


total_loss��@

error_R��C?

learning_rate_1�HG7Q�J�I       6%�	�R�W���A�*;


total_loss���@

error_R�X?

learning_rate_1�HG7�?tI       6%�	���W���A�*;


total_loss��@

error_R�JS?

learning_rate_1�HG7�PhHI       6%�	m߈W���A�*;


total_loss	ѭ@

error_R)`?

learning_rate_1�HG7 [l�I       6%�	Y9�W���A�*;


total_losst�@

error_R�Q?

learning_rate_1�HG7j� nI       6%�	�~�W���A�*;


total_loss�N�@

error_R��S?

learning_rate_1�HG7��QI       6%�	,ÉW���A�*;


total_loss,�@

error_Rl�A?

learning_rate_1�HG7���qI       6%�	��W���A�*;


total_loss�Q�@

error_R\X?

learning_rate_1�HG7�%I       6%�	�J�W���A�*;


total_loss��@

error_R��U?

learning_rate_1�HG7 �nI       6%�	���W���A�*;


total_losssm�@

error_R�@?

learning_rate_1�HG7�O8I       6%�	�ӊW���A�*;


total_loss5�@

error_RZyV?

learning_rate_1�HG7��GI       6%�	�0�W���A�*;


total_lossɧ�@

error_R$�F?

learning_rate_1�HG7�1r�I       6%�	���W���A�*;


total_lossiO�@

error_R�D?

learning_rate_1�HG7Y�l�I       6%�	yЋW���A�*;


total_lossh��@

error_R[:;?

learning_rate_1�HG7���I       6%�	��W���A�*;


total_lossWJ@

error_R�;?

learning_rate_1�HG7�ǔVI       6%�	�h�W���A�*;


total_loss���@

error_R]C?

learning_rate_1�HG7nk�I       6%�	w��W���A�*;


total_loss'ԇ@

error_R�Q?

learning_rate_1�HG7f���I       6%�	�W���A�*;


total_loss҃�@

error_R��F?

learning_rate_1�HG7��C~I       6%�	�2�W���A�*;


total_loss�LA

error_R�d?

learning_rate_1�HG7]���I       6%�	Pt�W���A�*;


total_loss��@

error_R�hD?

learning_rate_1�HG7��<�I       6%�	���W���A�*;


total_loss\��@

error_R�I?

learning_rate_1�HG7�!I       6%�	��W���A�*;


total_lossҦ@

error_R�^?

learning_rate_1�HG7���I       6%�	�K�W���A�*;


total_loss�ҳ@

error_R��K?

learning_rate_1�HG7��WJI       6%�	Ȑ�W���A�*;


total_loss�ڒ@

error_RTeF?

learning_rate_1�HG7�|��I       6%�	o܎W���A�*;


total_loss�~�@

error_R��A?

learning_rate_1�HG7��I       6%�	"�W���A�*;


total_loss́�@

error_Rt�H?

learning_rate_1�HG7�|��I       6%�	�g�W���A�*;


total_lossֆ�@

error_RfVW?

learning_rate_1�HG7�I       6%�	��W���A�*;


total_loss�Մ@

error_R=c5?

learning_rate_1�HG7��,�I       6%�	��W���A�*;


total_lossm��@

error_R�OA?

learning_rate_1�HG7�3�I       6%�	�2�W���A�*;


total_loss���@

error_R�-L?

learning_rate_1�HG7U��I       6%�	v�W���A�*;


total_loss�i�@

error_R��Q?

learning_rate_1�HG7����I       6%�	7��W���A�*;


total_loss���@

error_RN�7?

learning_rate_1�HG7�@��I       6%�	�W���A�*;


total_lossyt�@

error_R��>?

learning_rate_1�HG7��!�I       6%�	�L�W���A�*;


total_loss�+�@

error_Rx�C?

learning_rate_1�HG7��ZI       6%�	��W���A�*;


total_loss!��@

error_R�$J?

learning_rate_1�HG7X;}I       6%�	ӑW���A�*;


total_loss#a�@

error_R�]V?

learning_rate_1�HG7-�)�I       6%�	'�W���A�*;


total_lossXw�@

error_R�X?

learning_rate_1�HG7�K�I       6%�	�]�W���A�*;


total_lossv~@

error_R̆G?

learning_rate_1�HG7a;��I       6%�	���W���A�*;


total_loss��@

error_RR,P?

learning_rate_1�HG7�I       6%�	��W���A�*;


total_loss�
�@

error_R�4U?

learning_rate_1�HG7�z��I       6%�	�+�W���A�*;


total_loss*�@

error_R��c?

learning_rate_1�HG7;z�JI       6%�	�n�W���A�*;


total_lossdh�@

error_R1�r?

learning_rate_1�HG7�%>�I       6%�	��W���A�*;


total_loss�0�@

error_R`A?

learning_rate_1�HG7���I       6%�	���W���A�*;


total_loss_��@

error_R�d?

learning_rate_1�HG7"�I       6%�	�:�W���A�*;


total_loss���@

error_Ri�n?

learning_rate_1�HG7Š�I       6%�	���W���A�*;


total_loss�ɖ@

error_R�"R?

learning_rate_1�HG7�)��I       6%�	�ɔW���A�*;


total_loss�d@

error_R��N?

learning_rate_1�HG7@�WLI       6%�	��W���A�*;


total_loss��A

error_R��L?

learning_rate_1�HG7��"I       6%�	�Q�W���A�*;


total_lossX��@

error_RsM?

learning_rate_1�HG7���I       6%�	ʕ�W���A�*;


total_lossJ"�@

error_R,�P?

learning_rate_1�HG7����I       6%�	 וW���A�*;


total_loss���@

error_Rd$K?

learning_rate_1�HG7�%�]I       6%�	��W���A�*;


total_losse$�@

error_R�HR?

learning_rate_1�HG7Q	2wI       6%�	�c�W���A�*;


total_lossI��@

error_Re�V?

learning_rate_1�HG7�ccI       6%�	~��W���A�*;


total_loss��@

error_Rx�Q?

learning_rate_1�HG7E��I       6%�	���W���A�*;


total_loss��@

error_Rh\?

learning_rate_1�HG7���LI       6%�	�5�W���A�*;


total_loss�
A

error_R��Z?

learning_rate_1�HG7��P6I       6%�	�w�W���A�*;


total_loss!�@

error_RJ�T?

learning_rate_1�HG7�!�I       6%�	8��W���A�*;


total_loss��@

error_R�X?

learning_rate_1�HG7�EDI       6%�	���W���A�*;


total_lossl��@

error_R�M?

learning_rate_1�HG7�։�I       6%�	�A�W���A�*;


total_lossM�@

error_R��7?

learning_rate_1�HG7,NT;I       6%�	���W���A�*;


total_lossm�@

error_R��M?

learning_rate_1�HG7�wE;I       6%�	�͘W���A�*;


total_lossϐ�@

error_R�IR?

learning_rate_1�HG7��@I       6%�	|�W���A�*;


total_loss��@

error_R�3f?

learning_rate_1�HG7dT�]I       6%�	yZ�W���A�*;


total_loss�c�@

error_R�fC?

learning_rate_1�HG7�*G�I       6%�	Н�W���A�*;


total_loss�Yz@

error_R��U?

learning_rate_1�HG7!��I       6%�	j�W���A�*;


total_lossnֱ@

error_RUU?

learning_rate_1�HG7%�FI       6%�	�%�W���A�*;


total_lossM��@

error_R�XB?

learning_rate_1�HG76���I       6%�	_k�W���A�*;


total_loss�t�@

error_RO�A?

learning_rate_1�HG7{e�"I       6%�	���W���A�*;


total_loss	z�@

error_R�U?

learning_rate_1�HG7%��I       6%�	C�W���A�*;


total_loss�@�@

error_R�9J?

learning_rate_1�HG7��f�I       6%�		d�W���A�*;


total_loss(xA

error_RlB??

learning_rate_1�HG7�w��I       6%�	���W���A�*;


total_loss�A

error_R$D?

learning_rate_1�HG7.AnI       6%�	��W���A�*;


total_loss�@�@

error_R��H?

learning_rate_1�HG7L���I       6%�	�4�W���A�*;


total_loss6�@

error_Rq�B?

learning_rate_1�HG7�1%iI       6%�	z�W���A�*;


total_loss��@

error_RҐA?

learning_rate_1�HG7]��{I       6%�	j��W���A�*;


total_loss`��@

error_R��F?

learning_rate_1�HG7��Y7I       6%�	~�W���A�*;


total_loss7��@

error_R�>C?

learning_rate_1�HG7+'bI       6%�	�J�W���A�*;


total_loss\�@

error_Rh�[?

learning_rate_1�HG7�DЃI       6%�	w��W���A�*;


total_loss�+�@

error_R��@?

learning_rate_1�HG7j�"<I       6%�	�ڝW���A�*;


total_loss�q�@

error_RL�M?

learning_rate_1�HG7���I       6%�	��W���A�*;


total_loss���@

error_RC�M?

learning_rate_1�HG7�j��I       6%�	�e�W���A�*;


total_lossْ@

error_R��E?

learning_rate_1�HG7�-I�I       6%�	���W���A�*;


total_loss���@

error_R�+`?

learning_rate_1�HG7	�i*I       6%�	<�W���A�*;


total_losst&�@

error_R�F?

learning_rate_1�HG7-��I       6%�	�5�W���A�*;


total_loss\�A

error_R�_Q?

learning_rate_1�HG7��YI       6%�	���W���A�*;


total_loss~�@

error_Rdu`?

learning_rate_1�HG7 ���I       6%�	�ɟW���A�*;


total_lossq�@

error_R(�\?

learning_rate_1�HG7#�X�I       6%�	u�W���A�*;


total_losso�@

error_R�<?

learning_rate_1�HG7�rI�I       6%�	�W�W���A�*;


total_loss�D�@

error_RZER?

learning_rate_1�HG7�t�I       6%�	\��W���A�*;


total_lossqw�@

error_RE�F?

learning_rate_1�HG7��{uI       6%�	3�W���A�*;


total_loss��A

error_Rf�T?

learning_rate_1�HG7��0FI       6%�	�"�W���A�*;


total_losse��@

error_R	=B?

learning_rate_1�HG7�RUI       6%�	!f�W���A�*;


total_loss��@

error_R1�S?

learning_rate_1�HG7o�Z�I       6%�	ө�W���A�*;


total_loss{�@

error_R��N?

learning_rate_1�HG7��Q�I       6%�	��W���A�*;


total_loss���@

error_Rf�L?

learning_rate_1�HG7�o��I       6%�	s2�W���A�*;


total_lossV��@

error_R�dR?

learning_rate_1�HG7��]I       6%�	�u�W���A�*;


total_loss�G�@

error_R��E?

learning_rate_1�HG7	t��I       6%�	���W���A�*;


total_loss���@

error_RC[\?

learning_rate_1�HG7GF��I       6%�	|��W���A�*;


total_loss�@�@

error_R��V?

learning_rate_1�HG7���I       6%�	�;�W���A�*;


total_loss3�@

error_R�FH?

learning_rate_1�HG7�~�I       6%�	K�W���A�*;


total_loss�m�@

error_R��P?

learning_rate_1�HG7|˖I       6%�	 ��W���A�*;


total_loss�a�@

error_R�~G?

learning_rate_1�HG7׋��I       6%�	�W���A�*;


total_loss61�@

error_R<�Q?

learning_rate_1�HG7�(�I       6%�	hI�W���A�*;


total_loss3I�@

error_R��A?

learning_rate_1�HG7H��I       6%�	���W���A�*;


total_loss�@

error_R�j]?

learning_rate_1�HG7*�!I       6%�	.ΤW���A�*;


total_loss��@

error_R$�K?

learning_rate_1�HG7|�3�I       6%�	�W���A�*;


total_loss!��@

error_R�Q?

learning_rate_1�HG7�;�I       6%�	�V�W���A�*;


total_lossMϝ@

error_R��1?

learning_rate_1�HG7ȟ�]I       6%�	'��W���A�*;


total_loss�}�@

error_R�mG?

learning_rate_1�HG7�˾I       6%�	Z��W���A�*;


total_loss<�@

error_R=�f?

learning_rate_1�HG7܇�YI       6%�	o7�W���A�*;


total_loss���@

error_ROIU?

learning_rate_1�HG7�Z�~I       6%�	�}�W���A�*;


total_loss2�@

error_R�{N?

learning_rate_1�HG7
��I       6%�	h¦W���A�*;


total_loss��@

error_R�MM?

learning_rate_1�HG7����I       6%�	��W���A�*;


total_loss���@

error_R��7?

learning_rate_1�HG7@�֨I       6%�	*K�W���A�*;


total_loss�U�@

error_R�T?

learning_rate_1�HG7P��dI       6%�	{��W���A�*;


total_lossj�@

error_R
�I?

learning_rate_1�HG7{B�I       6%�	�٧W���A�*;


total_lossT�@

error_RíV?

learning_rate_1�HG7gc��I       6%�	�/�W���A�*;


total_loss���@

error_Rl�8?

learning_rate_1�HG7��x(I       6%�	�u�W���A�*;


total_loss��@

error_R�Q?

learning_rate_1�HG7�I       6%�	��W���A�*;


total_loss}��@

error_RZk>?

learning_rate_1�HG7 �h�I       6%�	m�W���A�*;


total_loss6�p@

error_R��L?

learning_rate_1�HG7���I       6%�	e�W���A�*;


total_loss���@

error_R��U?

learning_rate_1�HG7ߟs�I       6%�	���W���A�*;


total_loss��@

error_RHCM?

learning_rate_1�HG7fw��I       6%�	��W���A�*;


total_loss���@

error_RҵR?

learning_rate_1�HG7ܥ�|I       6%�	�9�W���A�*;


total_lossl��@

error_R��=?

learning_rate_1�HG7IY�RI       6%�	]~�W���A�*;


total_losst��@

error_R�<?

learning_rate_1�HG7"|s�I       6%�	*ɪW���A�*;


total_loss��@

error_R��M?

learning_rate_1�HG7Fե�I       6%�	Z?�W���A�*;


total_loss���@

error_R��@?

learning_rate_1�HG72�]�I       6%�	x��W���A�*;


total_loss�޿@

error_RMG?

learning_rate_1�HG7��3�I       6%�	C۫W���A�*;


total_loss���@

error_R��W?

learning_rate_1�HG7o7�I       6%�	�$�W���A�*;


total_loss-N�@

error_R��D?

learning_rate_1�HG7�d��I       6%�	�s�W���A�*;


total_loss
A

error_R��K?

learning_rate_1�HG7�1�I       6%�	)��W���A�*;


total_loss�X�@

error_Rn�??

learning_rate_1�HG7���I       6%�	�	�W���A�*;


total_losshg�@

error_RQgM?

learning_rate_1�HG7��P?I       6%�	�K�W���A�*;


total_loss�T�@

error_R�`?

learning_rate_1�HG7"�I       6%�	z��W���A�*;


total_loss}�@

error_Rŋ=?

learning_rate_1�HG7��~I       6%�	QЭW���A�*;


total_loss�=�@

error_R��W?

learning_rate_1�HG7��rRI       6%�	��W���A�*;


total_loss���@

error_R%9?

learning_rate_1�HG7��N�I       6%�	�V�W���A�*;


total_loss��@

error_RJ�P?

learning_rate_1�HG7�}ZiI       6%�	��W���A�*;


total_loss��@

error_R:!L?

learning_rate_1�HG7,�]RI       6%�	��W���A�*;


total_loss���@

error_R��N?

learning_rate_1�HG7YQ�I       6%�	)�W���A�*;


total_loss�Q�@

error_R�W?

learning_rate_1�HG7�n�I       6%�	�k�W���A�*;


total_lossT��@

error_R)�S?

learning_rate_1�HG7B��I       6%�	i��W���A�*;


total_loss��x@

error_RȒW?

learning_rate_1�HG7�VT6I       6%�	K��W���A�*;


total_loss-��@

error_R��O?

learning_rate_1�HG7̇��I       6%�	j=�W���A�*;


total_loss���@

error_R�G?

learning_rate_1�HG7Qv��I       6%�	��W���A�*;


total_loss�&�@

error_R�JT?

learning_rate_1�HG7���I       6%�	�˰W���A�*;


total_loss�@

error_R�=?

learning_rate_1�HG7��z�I       6%�	�W���A�*;


total_lossԸ�@

error_ReXG?

learning_rate_1�HG7��&I       6%�	T\�W���A�*;


total_loss��@

error_R]uC?

learning_rate_1�HG7���I       6%�	ޡ�W���A�*;


total_loss���@

error_R�{L?

learning_rate_1�HG7�(c�I       6%�	��W���A�*;


total_loss�ę@

error_R��=?

learning_rate_1�HG7�Iy�I       6%�	�)�W���A�*;


total_loss %�@

error_R��Y?

learning_rate_1�HG7�]lI       6%�	\n�W���A�*;


total_loss���@

error_R��L?

learning_rate_1�HG7̏�I       6%�	ϱ�W���A�*;


total_loss�!A

error_R��a?

learning_rate_1�HG7�|�I       6%�	i��W���A�*;


total_loss���@

error_RDY??

learning_rate_1�HG7�A�7I       6%�	�:�W���A�*;


total_lossA'�@

error_R!�R?

learning_rate_1�HG7���I       6%�	B��W���A�*;


total_loss�o�@

error_R��M?

learning_rate_1�HG7w-��I       6%�	���W���A�*;


total_loss�u
A

error_R�E?

learning_rate_1�HG7���I       6%�	2�W���A�*;


total_losso��@

error_R�,I?

learning_rate_1�HG7�0��I       6%�	�G�W���A�*;


total_loss�y�@

error_R�JW?

learning_rate_1�HG7���vI       6%�	S��W���A�*;


total_loss�2�@

error_R��T?

learning_rate_1�HG7�8H�I       6%�	"ٴW���A�*;


total_loss�x�@

error_RӔL?

learning_rate_1�HG7�W1RI       6%�	0�W���A�*;


total_loss�.�@

error_R�'Z?

learning_rate_1�HG7d��I       6%�	�`�W���A�*;


total_loss�6�@

error_R�wF?

learning_rate_1�HG7~��VI       6%�	椵W���A�*;


total_loss�U�@

error_R�Q?

learning_rate_1�HG7$nnI       6%�	7�W���A�*;


total_loss1m�@

error_RD"Z?

learning_rate_1�HG7��#uI       6%�	94�W���A�*;


total_loss[��@

error_RϒO?

learning_rate_1�HG7��tI       6%�	�}�W���A�*;


total_loss��@

error_R&JO?

learning_rate_1�HG79s8uI       6%�	W˶W���A�*;


total_losshr�@

error_R�Q?

learning_rate_1�HG7eT@�I       6%�	)�W���A�*;


total_loss���@

error_R�?J?

learning_rate_1�HG7mz�
I       6%�	�S�W���A�*;


total_loss&��@

error_R�F?

learning_rate_1�HG7I�>�I       6%�	���W���A�*;


total_loss~��@

error_R{�I?

learning_rate_1�HG7���I       6%�	��W���A�*;


total_loss���@

error_RH�d?

learning_rate_1�HG7C�4�I       6%�	�#�W���A�*;


total_loss�s�@

error_R�d?

learning_rate_1�HG7�w_�I       6%�	�h�W���A�*;


total_loss��@

error_R�B?

learning_rate_1�HG7�d�fI       6%�	f��W���A�*;


total_loss��@

error_RZ�R?

learning_rate_1�HG7A�LI       6%�	��W���A�*;


total_loss6f�@

error_RnH?

learning_rate_1�HG7���I       6%�	�:�W���A�*;


total_lossG�@

error_RڞK?

learning_rate_1�HG7�]�LI       6%�	�ĹW���A�*;


total_loss���@

error_R��R?

learning_rate_1�O?7O�#I       6%�	I�W���A�*;


total_loss�f�@

error_R��N?

learning_rate_1�O?7<��`I       6%�	W�W���A�*;


total_lossn��@

error_R�N?

learning_rate_1�O?7���
I       6%�	ɜ�W���A�*;


total_loss#�@

error_R	KJ?

learning_rate_1�O?7	I2I       6%�	��W���A�*;


total_loss{�@

error_R��F?

learning_rate_1�O?7�Ʈ�I       6%�	S?�W���A�*;


total_lossRm�@

error_R�[?

learning_rate_1�O?7��W�I       6%�	s��W���A�*;


total_loss�@

error_R��J?

learning_rate_1�O?7K��I       6%�	ū�W���A�*;


total_loss�RA

error_R�jD?

learning_rate_1�O?7��vI       6%�	���W���A�*;


total_loss��@

error_R�P?

learning_rate_1�O?7lD:�I       6%�	�C�W���A�*;


total_lossh��@

error_R��Y?

learning_rate_1�O?7x��!I       6%�	A��W���A�*;


total_loss���@

error_R��S?

learning_rate_1�O?7p��I       6%�	ZӿW���A�*;


total_loss�s�@

error_R�"O?

learning_rate_1�O?7��19I       6%�	�!�W���A�*;


total_lossӈ�@

error_R��B?

learning_rate_1�O?7�Z�I       6%�	�d�W���A�*;


total_losst��@

error_R��F?

learning_rate_1�O?78�p�I       6%�	��W���A�*;


total_loss}�@

error_Rd%;?

learning_rate_1�O?7޸��I       6%�	��W���A�*;


total_loss���@

error_R�P?

learning_rate_1�O?7!2.�I       6%�	�3�W���A�*;


total_lossWٰ@

error_RӵF?

learning_rate_1�O?7��I       6%�	1x�W���A�*;


total_loss��@

error_Rw~\?

learning_rate_1�O?7�s|I       6%�	ں�W���A�*;


total_loss@

error_R	P?

learning_rate_1�O?7�LI       6%�	��W���A�*;


total_loss��@

error_RSO?

learning_rate_1�O?7���NI       6%�	�F�W���A�*;


total_loss\�@

error_R��J?

learning_rate_1�O?7�j`I       6%�	\��W���A�*;


total_loss4�A

error_RF�G?

learning_rate_1�O?7&�-I       6%�	.��W���A�*;


total_loss��@

error_R��Q?

learning_rate_1�O?7�Ѵ�I       6%�	��W���A�*;


total_loss%3�@

error_R��<?

learning_rate_1�O?7����I       6%�	lZ�W���A�*;


total_loss�g�@

error_R$�L?

learning_rate_1�O?7O�{�I       6%�	���W���A�*;


total_loss���@

error_R)RQ?

learning_rate_1�O?7h���I       6%�	���W���A�*;


total_lossϕ@

error_R1LL?

learning_rate_1�O?7��JI       6%�	#*�W���A�*;


total_lossᢱ@

error_R��U?

learning_rate_1�O?7nT7I       6%�	;m�W���A�*;


total_loss���@

error_R�tU?

learning_rate_1�O?79��I       6%�	���W���A�*;


total_loss�<�@

error_R,�K?

learning_rate_1�O?7�2�xI       6%�	���W���A�*;


total_loss��@

error_R�eS?

learning_rate_1�O?7��VI       6%�	>�W���A�*;


total_losso@�@

error_R�kd?

learning_rate_1�O?7'ϼI       6%�	ҁ�W���A�*;


total_lossW��@

error_R�D?

learning_rate_1�O?7��EXI       6%�	.��W���A�*;


total_lossÆ�@

error_R'M?

learning_rate_1�O?7U�YI       6%�	��W���A�*;


total_lossQZ�@

error_R��H?

learning_rate_1�O?7(4��I       6%�	�J�W���A�*;


total_loss�V�@

error_R�RH?

learning_rate_1�O?7<�1HI       6%�	��W���A�*;


total_loss��A

error_Rf�L?

learning_rate_1�O?7�Z�I       6%�	`��W���A�*;


total_lossN��@

error_RJ�R?

learning_rate_1�O?7�|�I       6%�	��W���A�*;


total_loss8��@

error_R$DZ?

learning_rate_1�O?7f^I       6%�	TZ�W���A�*;


total_loss�!�@

error_RJ�Q?

learning_rate_1�O?7u�I       6%�	���W���A�*;


total_loss��@

error_R��N?

learning_rate_1�O?7x��I       6%�	Q��W���A�*;


total_lossO�S@

error_R�R?

learning_rate_1�O?7m=�2I       6%�	E^�W���A�*;


total_lossL��@

error_R{�U?

learning_rate_1�O?7���yI       6%�	���W���A�*;


total_loss�f�@

error_R�uX?

learning_rate_1�O?7�['I       6%�	���W���A�*;


total_loss�A

error_Rc�D?

learning_rate_1�O?7>h�I       6%�	�s�W���A�*;


total_loss�G�@

error_R�AT?

learning_rate_1�O?7j���I       6%�	���W���A�*;


total_lossT�@

error_R��[?

learning_rate_1�O?7��UI       6%�	(-�W���A�*;


total_loss_��@

error_RIhU?

learning_rate_1�O?7��~�I       6%�	k��W���A�*;


total_lossn�@

error_RzgZ?

learning_rate_1�O?7z��cI       6%�	.��W���A�*;


total_loss��
A

error_R��E?

learning_rate_1�O?7�=I       6%�	|L�W���A�*;


total_loss�V�@

error_RM�J?

learning_rate_1�O?7sp�/I       6%�	j��W���A�*;


total_loss1ݚ@

error_RߏX?

learning_rate_1�O?7R��I       6%�	��W���A�*;


total_lossRx�@

error_RtIY?

learning_rate_1�O?7��_�I       6%�	�M�W���A�*;


total_loss�s�@

error_R�>D?

learning_rate_1�O?7˭�I       6%�	%��W���A�*;


total_lossŒ�@

error_RcBb?

learning_rate_1�O?70���I       6%�	���W���A�*;


total_loss�@

error_RxJ5?

learning_rate_1�O?7�d��I       6%�	�J�W���A�*;


total_loss��l@

error_R��I?

learning_rate_1�O?7�2�I       6%�	���W���A�*;


total_loss%I�@

error_R��@?

learning_rate_1�O?7�m�I       6%�	���W���A�*;


total_lossXu@

error_R��K?

learning_rate_1�O?7#���I       6%�	�#�W���A�*;


total_lossz�@

error_R�}??

learning_rate_1�O?7��wI       6%�	�j�W���A�*;


total_loss9��@

error_R_�c?

learning_rate_1�O?7	�jmI       6%�	��W���A�*;


total_loss��@

error_R��K?

learning_rate_1�O?7�h.I       6%�	��W���A�*;


total_loss�v�@

error_RW�R?

learning_rate_1�O?7��� I       6%�	�`�W���A�*;


total_loss�@

error_R�vJ?

learning_rate_1�O?7�h�>I       6%�	Ц�W���A�*;


total_lossd=A

error_Re�I?

learning_rate_1�O?7�.{I       6%�	���W���A�*;


total_lossh��@

error_R�yJ?

learning_rate_1�O?7Ϛ�PI       6%�	32�W���A�*;


total_loss�\�@

error_R�*W?

learning_rate_1�O?7�v/�I       6%�	Vz�W���A�*;


total_lossK�@

error_RR�[?

learning_rate_1�O?7�/��I       6%�	)��W���A�*;


total_loss��@

error_R3�S?

learning_rate_1�O?7�W�I       6%�	'�W���A�*;


total_loss.<�@

error_R�i?

learning_rate_1�O?7���I       6%�	�S�W���A�*;


total_lossӯ�@

error_R�TN?

learning_rate_1�O?7���I       6%�	J��W���A�*;


total_loss�6�@

error_Rm�>?

learning_rate_1�O?7���1I       6%�	h��W���A�*;


total_loss*��@

error_Ro�A?

learning_rate_1�O?7�e��I       6%�	�2�W���A�*;


total_loss��@

error_R��C?

learning_rate_1�O?7u���I       6%�	>y�W���A�*;


total_losss&�@

error_R��H?

learning_rate_1�O?7Om�I       6%�	��W���A�*;


total_loss|��@

error_R�JM?

learning_rate_1�O?7���I       6%�	�
�W���A�*;


total_loss-+�@

error_R�H?

learning_rate_1�O?7k�Y<I       6%�	�Q�W���A�*;


total_loss?��@

error_R�9W?

learning_rate_1�O?7/n�,I       6%�	_��W���A�*;


total_loss:��@

error_RyZ?

learning_rate_1�O?7�l�-I       6%�	���W���A�*;


total_loss�H�@

error_R :I?

learning_rate_1�O?7��fdI       6%�	�-�W���A�*;


total_loss��A

error_R;�U?

learning_rate_1�O?7h���I       6%�	u�W���A�*;


total_loss�\�@

error_Rv�R?

learning_rate_1�O?7�kI       6%�	���W���A�*;


total_loss��@

error_R��??

learning_rate_1�O?7��YI       6%�	��W���A�*;


total_lossʃ�@

error_R��M?

learning_rate_1�O?7�
eI       6%�	�K�W���A�*;


total_lossn�@

error_R�ca?

learning_rate_1�O?7�
�I       6%�	d��W���A�*;


total_loss���@

error_R�:?

learning_rate_1�O?7�QI       6%�	h��W���A�*;


total_loss�>v@

error_R3B?

learning_rate_1�O?7p5)qI       6%�	��W���A�*;


total_loss���@

error_R��L?

learning_rate_1�O?7�"gcI       6%�	�a�W���A�*;


total_loss���@

error_RO4I?

learning_rate_1�O?7��aI       6%�	:��W���A�*;


total_loss&_�@

error_R�D?

learning_rate_1�O?7�k�I       6%�	��W���A�*;


total_loss��@

error_Ri�B?

learning_rate_1�O?7Z�k�I       6%�	q5�W���A�*;


total_loss�A

error_R�c?

learning_rate_1�O?7���I       6%�	n��W���A�*;


total_loss��@

error_RT�<?

learning_rate_1�O?7[�lI       6%�	���W���A�*;


total_loss�B�@

error_RA <?

learning_rate_1�O?7�䶾I       6%�	0�W���A�*;


total_loss�_�@

error_R.XT?

learning_rate_1�O?7��H�I       6%�	:[�W���A�*;


total_loss���@

error_R�G?

learning_rate_1�O?7���I       6%�	x��W���A�*;


total_loss}Ҫ@

error_R��\?

learning_rate_1�O?7��μI       6%�	6��W���A�*;


total_loss4��@

error_R �W?

learning_rate_1�O?7����I       6%�	T0�W���A�*;


total_loss}��@

error_R��G?

learning_rate_1�O?7�r�I       6%�	�u�W���A�*;


total_loss@��@

error_R��A?

learning_rate_1�O?79�ЯI       6%�	Q��W���A�*;


total_loss���@

error_Ro�L?

learning_rate_1�O?7���I       6%�	E��W���A�*;


total_loss��@

error_RϜL?

learning_rate_1�O?7�M�!I       6%�	�J�W���A�*;


total_loss���@

error_R��Q?

learning_rate_1�O?7�`��I       6%�	���W���A�*;


total_lossÛ�@

error_R�{d?

learning_rate_1�O?7�B�I       6%�	g��W���A�*;


total_loss��@

error_ROcM?

learning_rate_1�O?7ҺcI       6%�	�(�W���A�*;


total_loss�{@

error_R��G?

learning_rate_1�O?7>.V�I       6%�	��W���A�*;


total_loss�>�@

error_R�*O?

learning_rate_1�O?7�FI       6%�	���W���A�*;


total_loss!L�@

error_RnY?

learning_rate_1�O?7@�I       6%�	s�W���A�*;


total_loss�bA

error_R)�K?

learning_rate_1�O?7�S!�I       6%�	?W�W���A�*;


total_loss^�@

error_Rf�\?

learning_rate_1�O?7�S�xI       6%�	Y��W���A�*;


total_loss4W�@

error_R�T?

learning_rate_1�O?7���I       6%�	7��W���A�*;


total_loss�R�@

error_R�E?

learning_rate_1�O?7E�o�I       6%�	�5�W���A�*;


total_lossJ��@

error_R�[>?

learning_rate_1�O?7 d^�I       6%�	҃�W���A�*;


total_loss�e�@

error_Rl�B?

learning_rate_1�O?70���I       6%�	���W���A�*;


total_loss�r�@

error_R��A?

learning_rate_1�O?7��!I       6%�	��W���A�*;


total_loss�d@

error_R��@?

learning_rate_1�O?7���I       6%�	�V�W���A�*;


total_loss���@

error_R�.T?

learning_rate_1�O?7ǅfI       6%�	��W���A�*;


total_loss���@

error_R��K?

learning_rate_1�O?7e9АI       6%�	���W���A�*;


total_lossZҙ@

error_R�??

learning_rate_1�O?7�MI       6%�	�.�W���A�*;


total_loss<�@

error_RѱI?

learning_rate_1�O?7]e�I       6%�	_w�W���A�*;


total_loss���@

error_RJ�J?

learning_rate_1�O?7����I       6%�	���W���A�*;


total_loss�|�@

error_RN�V?

learning_rate_1�O?7D�c�I       6%�	��W���A�*;


total_loss}i�@

error_R�E?

learning_rate_1�O?7[�S�I       6%�	EU�W���A�*;


total_loss=د@

error_R�X?

learning_rate_1�O?7<YA}I       6%�	I��W���A�*;


total_lossm��@

error_RdY?

learning_rate_1�O?7����I       6%�	s��W���A�*;


total_loss�ő@

error_R��J?

learning_rate_1�O?7�S�I       6%�	p+�W���A�*;


total_loss
��@

error_Ral_?

learning_rate_1�O?7�S��I       6%�	�r�W���A�*;


total_loss�ַ@

error_R�|O?

learning_rate_1�O?7H6e�I       6%�	ȷ�W���A�*;


total_lossSu�@

error_RVJ?

learning_rate_1�O?7�pI       6%�	O��W���A�*;


total_loss��@

error_R��G?

learning_rate_1�O?7���I       6%�	AN�W���A�*;


total_loss�J�@

error_RH�I?

learning_rate_1�O?7���I       6%�	ߕ�W���A�*;


total_loss�k�@

error_RTGE?

learning_rate_1�O?7:FYI       6%�	U��W���A�*;


total_loss�@

error_R��K?

learning_rate_1�O?7�eI       6%�	�"�W���A�*;


total_lossz�A

error_R!_R?

learning_rate_1�O?7���I       6%�	g�W���A�*;


total_loss�	A

error_R;$V?

learning_rate_1�O?7Y]�nI       6%�	e��W���A�*;


total_lossa�@

error_R�%M?

learning_rate_1�O?7��-XI       6%�	j��W���A�*;


total_loss��@

error_Rq9X?

learning_rate_1�O?7� cI       6%�	�;�W���A�*;


total_loss�c�@

error_R�IL?

learning_rate_1�O?7���!I       6%�	t��W���A�*;


total_loss��@

error_RzTQ?

learning_rate_1�O?7r��1I       6%�	���W���A�*;


total_loss�/�@

error_R.8M?

learning_rate_1�O?7{�6dI       6%�	[�W���A�*;


total_lossd�q@

error_RH?

learning_rate_1�O?7��bI       6%�	�]�W���A�*;


total_loss�r�@

error_R�[?

learning_rate_1�O?7���OI       6%�	b��W���A�*;


total_loss#ɹ@

error_R6�A?

learning_rate_1�O?7�O��I       6%�	h��W���A�*;


total_loss��@

error_R S?

learning_rate_1�O?7��=�I       6%�	H5�W���A�*;


total_loss��@

error_R\�Q?

learning_rate_1�O?7'Z��I       6%�	�w�W���A�*;


total_loss6��@

error_R��W?

learning_rate_1�O?7�?I       6%�	ż�W���A�*;


total_loss�@

error_Rm�@?

learning_rate_1�O?7b��I       6%�	A�W���A�*;


total_lossS�@

error_R6�U?

learning_rate_1�O?7�#YXI       6%�	�O�W���A�*;


total_loss�@

error_R}�.?

learning_rate_1�O?7�0��I       6%�	���W���A�*;


total_loss2��@

error_R��R?

learning_rate_1�O?7���UI       6%�	��W���A�*;


total_loss��@

error_R`EG?

learning_rate_1�O?7���I       6%�	�L�W���A�*;


total_loss�,�@

error_RחU?

learning_rate_1�O?7P?ϫI       6%�	W��W���A�*;


total_loss�AA

error_R�J?

learning_rate_1�O?7A�I       6%�	���W���A�*;


total_loss�o@

error_R�sK?

learning_rate_1�O?7��I       6%�	�=�W���A�*;


total_loss���@

error_R�	F?

learning_rate_1�O?7^OnI       6%�	���W���A�*;


total_loss��@

error_R8J?

learning_rate_1�O?7iiI       6%�	���W���A�*;


total_loss1#�@

error_R��R?

learning_rate_1�O?7o�W�I       6%�	�'�W���A�*;


total_loss�S�@

error_R�$G?

learning_rate_1�O?7�"�I       6%�	�n�W���A�*;


total_loss�In@

error_R6I?

learning_rate_1�O?7Ԓ��I       6%�	-��W���A�*;


total_loss�T�@

error_R��C?

learning_rate_1�O?74��9I       6%�	U�W���A�*;


total_loss���@

error_R8~S?

learning_rate_1�O?7Sb�XI       6%�	~m�W���A�*;


total_loss8�@

error_Rh"E?

learning_rate_1�O?7*��FI       6%�	���W���A�*;


total_losso��@

error_Rm�9?

learning_rate_1�O?7��H�I       6%�	���W���A�*;


total_losse��@

error_R/*Q?

learning_rate_1�O?7��{I       6%�	4B�W���A�*;


total_loss���@

error_RԲW?

learning_rate_1�O?7�w��I       6%�	4��W���A�*;


total_lossE��@

error_R�=K?

learning_rate_1�O?7{S>I       6%�	���W���A�*;


total_loss���@

error_R�WL?

learning_rate_1�O?7 �&I       6%�	��W���A�*;


total_lossc`�@

error_R%WH?

learning_rate_1�O?7���BI       6%�	Mc�W���A�*;


total_loss��p@

error_R��c?

learning_rate_1�O?7$�uUI       6%�		��W���A�*;


total_lossC�@

error_R1SF?

learning_rate_1�O?7��*�I       6%�	���W���A�*;


total_loss�x�@

error_R�C?

learning_rate_1�O?7�h��I       6%�	5:�W���A�*;


total_loss+B�@

error_R��B?

learning_rate_1�O?7�[��I       6%�	.��W���A�*;


total_loss	�@

error_R��\?

learning_rate_1�O?7�3I       6%�	���W���A�*;


total_loss�k�@

error_R}�V?

learning_rate_1�O?7�)ptI       6%�	��W���A�*;


total_loss�/�@

error_RѬI?

learning_rate_1�O?7�e�9I       6%�	Z^�W���A�*;


total_loss�z�@

error_R�a?

learning_rate_1�O?7��E�I       6%�	;��W���A�*;


total_losss��@

error_R�7C?

learning_rate_1�O?7��h'I       6%�	���W���A�*;


total_loss�s�@

error_R��H?

learning_rate_1�O?7��I       6%�	�:�W���A�*;


total_loss���@

error_RdAY?

learning_rate_1�O?7�A��I       6%�	���W���A�*;


total_lossoz�@

error_R.�^?

learning_rate_1�O?7�3"I       6%�	0��W���A�*;


total_lossEy�@

error_R\�M?

learning_rate_1�O?7��}�I       6%�	�W���A�*;


total_loss��@

error_R�EG?

learning_rate_1�O?7ִn�I       6%�	W�W���A�*;


total_lossTj�@

error_R��U?

learning_rate_1�O?7!�$iI       6%�	j��W���A�*;


total_loss�}�@

error_R��a?

learning_rate_1�O?7�:��I       6%�	N��W���A�*;


total_lossN|A

error_R3�[?

learning_rate_1�O?7=$I       6%�	�"�W���A�*;


total_loss��@

error_R��M?

learning_rate_1�O?7O I       6%�	~g�W���A�*;


total_loss9~�@

error_R�A?

learning_rate_1�O?7TE��I       6%�	u��W���A�*;


total_loss���@

error_R��V?

learning_rate_1�O?7�� �I       6%�	���W���A�*;


total_lossa(u@

error_R�??

learning_rate_1�O?7%�Y�I       6%�	!C�W���A�*;


total_lossY�@

error_RÕF?

learning_rate_1�O?7��RI       6%�	��W���A�*;


total_lossơ�@

error_R��`?

learning_rate_1�O?7��I       6%�	��W���A�*;


total_loss�j�@

error_R��L?

learning_rate_1�O?7 ڲrI       6%�	h �W���A�*;


total_loss��A

error_R6�K?

learning_rate_1�O?7-QC�I       6%�	Kk�W���A�*;


total_loss==�@

error_R}D?

learning_rate_1�O?7�bI       6%�	���W���A�*;


total_lossO��@

error_RV?

learning_rate_1�O?7�z�I       6%�	� �W���A�*;


total_loss���@

error_RVaL?

learning_rate_1�O?7���I       6%�	�I�W���A�*;


total_loss�� A

error_R�de?

learning_rate_1�O?7'L��I       6%�	���W���A�*;


total_loss��@

error_RԄM?

learning_rate_1�O?7�[z"I       6%�	L��W���A�*;


total_loss���@

error_RVBC?

learning_rate_1�O?70��I       6%�	��W���A�*;


total_loss��A

error_RZ\?

learning_rate_1�O?71w��I       6%�	Cd�W���A�*;


total_loss��@

error_RE-J?

learning_rate_1�O?79�gI       6%�	���W���A�*;


total_loss���@

error_R�fV?

learning_rate_1�O?7M��I       6%�	���W���A�*;


total_loss�Ɠ@

error_RjL?

learning_rate_1�O?7H�9�I       6%�	�:�W���A�*;


total_loss}j�@

error_R/>W?

learning_rate_1�O?7�9�I       6%�	�}�W���A�*;


total_loss��A

error_R�N?

learning_rate_1�O?7��"�I       6%�	2��W���A�*;


total_loss���@

error_R��O?

learning_rate_1�O?7�l�4I       6%�	x�W���A�*;


total_loss�T�@

error_R�L?

learning_rate_1�O?7��QI       6%�	I�W���A�*;


total_lossR9�@

error_R�vC?

learning_rate_1�O?7iJlI       6%�	��W���A�*;


total_loss���@

error_R	�U?

learning_rate_1�O?7���I       6%�	c��W���A�*;


total_lossQ��@

error_R3&a?

learning_rate_1�O?73@�bI       6%�	x�W���A�*;


total_loss<��@

error_R��V?

learning_rate_1�O?7�g�I       6%�	�g�W���A�*;


total_lossd��@

error_R@Qa?

learning_rate_1�O?7~G�oI       6%�	۰�W���A�*;


total_loss�b�@

error_R�W?

learning_rate_1�O?7yA��I       6%�	���W���A�*;


total_lossJ�@

error_R�2L?

learning_rate_1�O?7*S��I       6%�	@�W���A�*;


total_loss��@

error_R��Q?

learning_rate_1�O?7V��CI       6%�	���W���A�*;


total_loss�@

error_R;9H?

learning_rate_1�O?7P��I       6%�	@��W���A�*;


total_loss�|@

error_RW�[?

learning_rate_1�O?7<a׶I       6%�	,&�W���A�*;


total_loss�%�@

error_R�R?

learning_rate_1�O?7�+�HI       6%�	��W���A�*;


total_loss�
�@

error_R��L?

learning_rate_1�O?7��uUI       6%�	���W���A�*;


total_lossρ�@

error_Ra�;?

learning_rate_1�O?78 ~�I       6%�	��W���A�*;


total_loss�D�@

error_RH]G?

learning_rate_1�O?7�Z�PI       6%�	/O�W���A�*;


total_loss_�y@

error_R��Q?

learning_rate_1�O?7���I       6%�	s��W���A�*;


total_loss�o�@

error_R �H?

learning_rate_1�O?7pd[�I       6%�	���W���A�*;


total_loss�ֆ@

error_RR�U?

learning_rate_1�O?7��B�I       6%�	p)�W���A�*;


total_loss��t@

error_R)fN?

learning_rate_1�O?7����I       6%�	u�W���A�*;


total_loss1��@

error_R @L?

learning_rate_1�O?7j#�I       6%�	=��W���A�*;


total_loss!�@

error_RvS?

learning_rate_1�O?7Ir��I       6%�	��W���A�*;


total_loss1��@

error_RԈP?

learning_rate_1�O?7O@�I       6%�	,M�W���A�*;


total_lossڎ�@

error_R�6?

learning_rate_1�O?7�C 'I       6%�	B��W���A�*;


total_loss��A

error_R\ J?

learning_rate_1�O?7\��4I       6%�	���W���A�*;


total_loss���@

error_R8�R?

learning_rate_1�O?7(�psI       6%�	�)�W���A�*;


total_loss�d A

error_R.�X?

learning_rate_1�O?7f�<<I       6%�	ll�W���A�*;


total_loss|J�@

error_R�F?

learning_rate_1�O?70U,I       6%�	ȫ�W���A�*;


total_lossix�@

error_R��A?

learning_rate_1�O?7X�~�I       6%�	���W���A�*;


total_lossMCA

error_R�)X?

learning_rate_1�O?7����I       6%�	o5 X���A�*;


total_lossѺ�@

error_R�GL?

learning_rate_1�O?7�"�I       6%�	D| X���A�*;


total_loss�a�@

error_R�8P?

learning_rate_1�O?7-:K�I       6%�	N� X���A�*;


total_loss���@

error_R$F=?

learning_rate_1�O?7�ʩI       6%�	�X���A�*;


total_loss���@

error_R\>H?

learning_rate_1�O?7^�u{I       6%�	LX���A�*;


total_loss*��@

error_RZ�N?

learning_rate_1�O?7�`�I       6%�	/�X���A�*;


total_loss֪�@

error_R�YO?

learning_rate_1�O?7�`@�I       6%�	�X���A�*;


total_loss<��@

error_R4`?

learning_rate_1�O?7�t�AI       6%�	�X���A�*;


total_loss���@

error_R��V?

learning_rate_1�O?7,0I       6%�	�fX���A�*;


total_lossj'�@

error_Rs�\?

learning_rate_1�O?7"��I       6%�	k�X���A�*;


total_loss�Uj@

error_R��I?

learning_rate_1�O?7����I       6%�	T�X���A�*;


total_loss/+�@

error_R��E?

learning_rate_1�O?7�t�I       6%�	�/X���A�*;


total_loss�"�@

error_R�@Q?

learning_rate_1�O?7F��I       6%�	=wX���A�*;


total_loss���@

error_R��f?

learning_rate_1�O?7����I       6%�	ɹX���A�*;


total_loss��A

error_R��`?

learning_rate_1�O?7��I       6%�	C�X���A�*;


total_lossy�@

error_RRd?

learning_rate_1�O?7a��I       6%�	�BX���A�*;


total_lossR�@

error_R�^?

learning_rate_1�O?7{՗~I       6%�	N�X���A�*;


total_lossx��@

error_Rq3]?

learning_rate_1�O?7iH��I       6%�	H�X���A�*;


total_lossa�3A

error_R��T?

learning_rate_1�O?7d	j�I       6%�	fX���A�*;


total_loss_{�@

error_Rq�,?

learning_rate_1�O?7.�g�I       6%�	�WX���A�*;


total_loss*JF@

error_RcYG?

learning_rate_1�O?7� ٮI       6%�	��X���A�*;


total_loss�ե@

error_R[i0?

learning_rate_1�O?7��I       6%�	��X���A�*;


total_loss �t@

error_R�9?

learning_rate_1�O?7���I       6%�	�*X���A�*;


total_lossظ�@

error_Rܟ<?

learning_rate_1�O?7ͩEI       6%�	ZvX���A�*;


total_loss@��@

error_R*�E?

learning_rate_1�O?7"M�XI       6%�	(�X���A�*;


total_loss��A

error_R��K?

learning_rate_1�O?7k)�I       6%�	=X���A�*;


total_lossÇ�@

error_R��O?

learning_rate_1�O?7��+�I       6%�	OX���A�*;


total_loss���@

error_R��J?

learning_rate_1�O?7���I       6%�	�X���A�*;


total_lossM�@

error_RŜC?

learning_rate_1�O?7[�$nI       6%�	��X���A�*;


total_loss��@

error_R>?

learning_rate_1�O?7��A0I       6%�	�%X���A�*;


total_loss��A

error_RTN?

learning_rate_1�O?7p��I       6%�	��X���A�*;


total_loss�^�@

error_RąP?

learning_rate_1�O?7�y�I       6%�	��X���A�*;


total_lossʹ�@

error_R��W?

learning_rate_1�O?7O͝GI       6%�	m(	X���A�*;


total_loss]t�@

error_R�gJ?

learning_rate_1�O?7��pjI       6%�	��	X���A�*;


total_loss�U�@

error_RZ�U?

learning_rate_1�O?7�Un�I       6%�	��	X���A�*;


total_lossC��@

error_R=�A?

learning_rate_1�O?7��/�I       6%�	�
X���A�*;


total_loss���@

error_R��Z?

learning_rate_1�O?7��WI       6%�	�b
X���A�*;


total_lossq4�@

error_R�w,?

learning_rate_1�O?7�$�I       6%�	u�
X���A�*;


total_loss��@

error_R�zZ?

learning_rate_1�O?7���I       6%�	��
X���A�*;


total_loss�(�@

error_RsT?

learning_rate_1�O?7�edtI       6%�	�LX���A�*;


total_loss	ѵ@

error_R�U?

learning_rate_1�O?7&��I       6%�	җX���A�*;


total_loss;��@

error_RڕD?

learning_rate_1�O?7`f9I       6%�		�X���A�*;


total_lossZ@�@

error_R%�Q?

learning_rate_1�O?7�zAI       6%�	�'X���A�*;


total_loss�	�@

error_Rv�R?

learning_rate_1�O?7T~!I       6%�	�jX���A�*;


total_loss�9�@

error_R�mV?

learning_rate_1�O?7��S}I       6%�	կX���A�*;


total_loss���@

error_R|�_?

learning_rate_1�O?7��I       6%�	��X���A�*;


total_loss���@

error_R�C?

learning_rate_1�O?7��WI       6%�	�:X���A�*;


total_loss��A

error_R�X?

learning_rate_1�O?7?> fI       6%�	2�X���A�*;


total_loss�։@

error_RB?

learning_rate_1�O?7Q�~I       6%�	�X���A�*;


total_lossi��@

error_R@M?

learning_rate_1�O?7�ݢ�I       6%�	(X���A�*;


total_lossWw�@

error_RC�N?

learning_rate_1�O?7�m3 I       6%�	IX���A�*;


total_loss<��@

error_RlgR?

learning_rate_1�O?70mRII       6%�	]�X���A�*;


total_loss좶@

error_R�J?

learning_rate_1�O?7���I       6%�	m�X���A�*;


total_loss�&�@

error_RS�b?

learning_rate_1�O?7T��?I       6%�	�X���A�*;


total_lossjß@

error_R��L?

learning_rate_1�O?7�Ɉ�I       6%�	YX���A�*;


total_loss��@

error_R]�S?

learning_rate_1�O?7�*l8I       6%�	��X���A�*;


total_loss�N@

error_R��<?

learning_rate_1�O?7�V�I       6%�	��X���A�*;


total_loss���@

error_R��,?

learning_rate_1�O?7����I       6%�	$X���A�*;


total_loss̓�@

error_R{.X?

learning_rate_1�O?7X GI       6%�	whX���A�*;


total_loss�Dx@

error_R�L?

learning_rate_1�O?7��I       6%�	��X���A�*;


total_loss�ԇ@

error_Rr�Y?

learning_rate_1�O?7ak�I       6%�	��X���A�*;


total_loss��@

error_R��U?

learning_rate_1�O?7���iI       6%�	EX���A�*;


total_loss�n A

error_Rۀa?

learning_rate_1�O?713�I       6%�	)�X���A�*;


total_lossI�@

error_R*7??

learning_rate_1�O?7	ǻ2I       6%�	��X���A�*;


total_lossIϯ@

error_R<[Z?

learning_rate_1�O?7Q��vI       6%�	N#X���A�*;


total_loss�aA

error_R�>?

learning_rate_1�O?7��z�I       6%�	ZiX���A�*;


total_lossc<�@

error_RrpD?

learning_rate_1�O?7t!�I       6%�	u�X���A�*;


total_lossxİ@

error_Rs�L?

learning_rate_1�O?7�I       6%�	)�X���A�*;


total_lossq�@

error_R�<e?

learning_rate_1�O?7�JI       6%�	BX���A�*;


total_lossf�A

error_R\MN?

learning_rate_1�O?7^�}I       6%�	5�X���A�*;


total_loss8A

error_R�MR?

learning_rate_1�O?7����I       6%�	2�X���A�*;


total_loss�,�@

error_RñM?

learning_rate_1�O?7R��%I       6%�	�X���A�*;


total_lossu��@

error_R)VS?

learning_rate_1�O?7:��I       6%�	�WX���A�*;


total_loss���@

error_R�@?

learning_rate_1�O?7RzhI       6%�	�X���A�*;


total_lossH1�@

error_Rmda?

learning_rate_1�O?7�R�,I       6%�	z�X���A�*;


total_lossp�@

error_R��P?

learning_rate_1�O?7T�>I       6%�	@%X���A�*;


total_loss�̉@

error_R@�`?

learning_rate_1�O?7�LeI       6%�	^lX���A�*;


total_lossq �@

error_R�s>?

learning_rate_1�O?7x�I       6%�	��X���A�*;


total_loss[��@

error_RX�L?

learning_rate_1�O?7�t��I       6%�	�X���A�*;


total_loss�@

error_R�<?

learning_rate_1�O?7�"��I       6%�	7X���A�*;


total_loss��@

error_R��\?

learning_rate_1�O?7��mI       6%�	�~X���A�*;


total_loss���@

error_R��M?

learning_rate_1�O?7�ې|I       6%�	/�X���A�*;


total_losso5{@

error_R;W?

learning_rate_1�O?7��I       6%�	TX���A�*;


total_lossQ�@

error_Rq�5?

learning_rate_1�O?7� ��I       6%�	SXX���A�*;


total_loss�B�@

error_R�gQ?

learning_rate_1�O?7"�2$I       6%�	��X���A�*;


total_lossrw�@

error_Rv�P?

learning_rate_1�O?7�x��I       6%�	��X���A�*;


total_loss��@

error_R3�\?

learning_rate_1�O?7#5�^I       6%�	�)X���A�*;


total_loss��@

error_R�@J?

learning_rate_1�O?7ɿ��I       6%�	�nX���A�*;


total_loss�=�@

error_RE�G?

learning_rate_1�O?7[��HI       6%�	�X���A�*;


total_loss��@

error_RȗT?

learning_rate_1�O?7w�<�I       6%�	��X���A�*;


total_lossCB�@

error_R��A?

learning_rate_1�O?7'��`I       6%�	�>X���A�*;


total_loss{��@

error_R�p6?

learning_rate_1�O?77�FNI       6%�	�X���A�*;


total_lossv�A

error_R18T?

learning_rate_1�O?7�C_I       6%�	j�X���A�*;


total_loss ��@

error_R�\b?

learning_rate_1�O?7z�8I       6%�	
X���A�*;


total_loss�`@

error_R �O?

learning_rate_1�O?7��YI       6%�	VLX���A�*;


total_loss���@

error_R
}M?

learning_rate_1�O?7t�I       6%�	�X���A�*;


total_lossA��@

error_R�:W?

learning_rate_1�O?7]��wI       6%�	1�X���A�*;


total_loss6��@

error_RdS:?

learning_rate_1�O?7����I       6%�	},X���A�*;


total_loss/��@

error_RWOG?

learning_rate_1�O?7&���I       6%�	�X���A�*;


total_loss�֍@

error_R@�V?

learning_rate_1�O?7�r�`I       6%�	��X���A�*;


total_loss�u�@

error_R��8?

learning_rate_1�O?7[)��I       6%�	�X���A�*;


total_loss�̘@

error_R��O?

learning_rate_1�O?7�I       6%�	EOX���A�*;


total_loss�
A

error_R�yD?

learning_rate_1�O?7(�I       6%�	x�X���A�*;


total_loss Ũ@

error_R�#H?

learning_rate_1�O?7`x8^I       6%�	��X���A�*;


total_lossag�@

error_R��S?

learning_rate_1�O?7�
��I       6%�	� X���A�*;


total_loss/��@

error_R �>?

learning_rate_1�O?7�j�aI       6%�	�fX���A�*;


total_loss�Ϛ@

error_RΛJ?

learning_rate_1�O?7�O�I       6%�	�X���A�*;


total_loss3S�@

error_R��S?

learning_rate_1�O?7D���I       6%�	��X���A�*;


total_loss��@

error_RW�Y?

learning_rate_1�O?7.m��I       6%�	�8X���A�*;


total_loss\g�@

error_R��>?

learning_rate_1�O?7��!�I       6%�	B~X���A�*;


total_loss�c�@

error_RTIH?

learning_rate_1�O?7x���I       6%�	@�X���A�*;


total_loss���@

error_R&Y?

learning_rate_1�O?7��;I       6%�	wX���A�*;


total_loss�Z�@

error_R�N?

learning_rate_1�O?7�@��I       6%�	�bX���A�*;


total_loss���@

error_R7�=?

learning_rate_1�O?7��k�I       6%�	F�X���A�*;


total_loss!��@

error_R(�R?

learning_rate_1�O?7&�S�I       6%�	_�X���A�*;


total_loss���@

error_R��N?

learning_rate_1�O?7#C�BI       6%�	�5 X���A�*;


total_loss��@

error_RSQ?

learning_rate_1�O?7#��I       6%�	�y X���A�*;


total_loss���@

error_Rx�d?

learning_rate_1�O?7�Ib	I       6%�	<� X���A�*;


total_loss��@

error_R�Y?

learning_rate_1�O?7��7�I       6%�	�!X���A�*;


total_loss��@

error_R]nR?

learning_rate_1�O?7�Q�TI       6%�	�P!X���A�*;


total_loss^R�@

error_RE<a?

learning_rate_1�O?7	GXI       6%�	�!X���A�*;


total_loss��@

error_Rqeb?

learning_rate_1�O?79rżI       6%�	/�!X���A�*;


total_loss+�A

error_R1�Y?

learning_rate_1�O?76a�I       6%�	�"X���A�*;


total_lossrA

error_R�MT?

learning_rate_1�O?7�Ȗ�I       6%�	<_"X���A�*;


total_lossZk�@

error_R�NP?

learning_rate_1�O?7�IcI       6%�	�"X���A�*;


total_lossı�@

error_Rx�??

learning_rate_1�O?7�̇GI       6%�	��"X���A�*;


total_loss���@

error_Rx?U?

learning_rate_1�O?7R�I       6%�	�-#X���A�*;


total_loss�e�@

error_R��S?

learning_rate_1�O?7���fI       6%�	Vr#X���A�*;


total_loss�@

error_R�7U?

learning_rate_1�O?7$�aI       6%�	�#X���A�*;


total_loss�e�@

error_RJ5F?

learning_rate_1�O?7�;�LI       6%�	��#X���A�*;


total_loss���@

error_R�L?

learning_rate_1�O?7D��I       6%�	�<$X���A�*;


total_loss}�@

error_RȈ[?

learning_rate_1�O?7���FI       6%�	��$X���A�*;


total_loss*��@

error_R8DQ?

learning_rate_1�O?7�\�I       6%�	��$X���A�*;


total_loss���@

error_R֍N?

learning_rate_1�O?7���9I       6%�	;%X���A�*;


total_lossɥ�@

error_RIZL?

learning_rate_1�O?7\ƃI       6%�	�W%X���A�*;


total_loss��@

error_R�!S?

learning_rate_1�O?7C��fI       6%�	l�%X���A�*;


total_loss�у@

error_R��<?

learning_rate_1�O?7`)��I       6%�	��%X���A�*;


total_loss���@

error_R�Q?

learning_rate_1�O?7D#�I       6%�	"&X���A�*;


total_loss��@

error_R��G?

learning_rate_1�O?7,1u�I       6%�	g&X���A�*;


total_loss���@

error_R�QX?

learning_rate_1�O?7�I       6%�	�&X���A�*;


total_loss�@

error_R@�[?

learning_rate_1�O?7�%JI       6%�	)�&X���A�*;


total_loss�`�@

error_R=�K?

learning_rate_1�O?7��2�I       6%�	�2'X���A�*;


total_loss��@

error_R�M?

learning_rate_1�O?7�%I       6%�	!u'X���A�*;


total_lossL�@

error_R�Q?

learning_rate_1�O?7��I       6%�	��'X���A�*;


total_loss���@

error_R�S?

learning_rate_1�O?7:S�/I       6%�	��'X���A�*;


total_loss��@

error_R�'J?

learning_rate_1�O?7���I       6%�	dV(X���A�*;


total_loss���@

error_R��W?

learning_rate_1�O?7��;I       6%�	a�(X���A�*;


total_lossf��@

error_Rd	M?

learning_rate_1�O?7���I       6%�	��(X���A�*;


total_loss�r�@

error_R��Z?

learning_rate_1�O?7oF�RI       6%�	B-)X���A�*;


total_loss#`�@

error_R�H?

learning_rate_1�O?7��&I       6%�	!�)X���A�*;


total_lossӬ�@

error_R$L?

learning_rate_1�O?7'��I       6%�	R�)X���A�*;


total_losscd�@

error_R7}h?

learning_rate_1�O?7Sh��I       6%�	�(*X���A�*;


total_lossHq�@

error_R�i?

learning_rate_1�O?7���I       6%�	�r*X���A�*;


total_loss�ŵ@

error_RܚT?

learning_rate_1�O?7�w�oI       6%�	�*X���A�*;


total_loss���@

error_RPY?

learning_rate_1�O?7�֑�I       6%�	�+X���A�*;


total_loss�q�@

error_R�8?

learning_rate_1�O?7����I       6%�	2w+X���A�*;


total_loss��F@

error_RX�W?

learning_rate_1�O?7�"�I       6%�	��+X���A�*;


total_lossjq�@

error_R��I?

learning_rate_1�O?7ĵ3�I       6%�	�,X���A�*;


total_lossC!�@

error_RbZ?

learning_rate_1�O?7�?Z�I       6%�	�M,X���A�*;


total_lossm��@

error_Ro�O?

learning_rate_1�O?7����I       6%�	s�,X���A�*;


total_loss�<�@

error_R_\?

learning_rate_1�O?7 ��I       6%�	��,X���A�*;


total_lossq*�@

error_RN?

learning_rate_1�O?7+?�I       6%�	p0-X���A�*;


total_loss�s�@

error_R�P?

learning_rate_1�O?7��։I       6%�	H{-X���A�*;


total_lossL֤@

error_RC�7?

learning_rate_1�O?7���I       6%�	��-X���A�*;


total_lossY�@

error_R��b?

learning_rate_1�O?7��q|I       6%�	�.X���A�*;


total_lossF2�@

error_R�c??

learning_rate_1�O?7}��I       6%�	�M.X���A�*;


total_loss��@

error_R��B?

learning_rate_1�O?7�o��I       6%�	�.X���A�*;


total_lossߏ�@

error_R��,?

learning_rate_1�O?7���I       6%�	��.X���A�*;


total_lossM�@

error_RC ]?

learning_rate_1�O?7��;nI       6%�	u(/X���A�*;


total_loss1��@

error_R�X;?

learning_rate_1�O?7lWZ�I       6%�	�k/X���A�*;


total_loss���@

error_R��G?

learning_rate_1�O?7>�NI       6%�	��/X���A�*;


total_loss���@

error_R�:?

learning_rate_1�O?7���MI       6%�	N�/X���A�*;


total_loss��@

error_R!�d?

learning_rate_1�O?7W���I       6%�	950X���A�*;


total_loss�@

error_R6�[?

learning_rate_1�O?7og�eI       6%�	�z0X���A�*;


total_loss?P�@

error_R�A=?

learning_rate_1�O?7���I       6%�	H�0X���A�*;


total_loss��@

error_R�sY?

learning_rate_1�O?7#�n!I       6%�	h	1X���A�*;


total_loss���@

error_R!�H?

learning_rate_1�O?7�nZI       6%�	(P1X���A�*;


total_loss1R�@

error_R� ;?

learning_rate_1�O?7�CE�I       6%�	;�1X���A�*;


total_lossH}�@

error_R� Q?

learning_rate_1�O?7���I       6%�	_�1X���A�*;


total_loss�r�@

error_R�I?

learning_rate_1�O?7iIUI       6%�	)%2X���A�*;


total_lossqƤ@

error_R��E?

learning_rate_1�O?7�_1�I       6%�	�o2X���A�*;


total_loss:)�@

error_R�&T?

learning_rate_1�O?7��e{I       6%�	�2X���A�*;


total_loss}��@

error_R1bM?

learning_rate_1�O?7�mI       6%�	�3X���A�*;


total_loss]C�@

error_R�OJ?

learning_rate_1�O?7�I       6%�	�J3X���A�*;


total_loss�F@

error_R=/G?

learning_rate_1�O?76�X�I       6%�	��3X���A�*;


total_loss�r4@

error_Rn+H?

learning_rate_1�O?7Rl�sI       6%�	��3X���A�*;


total_lossf�@

error_R�6?

learning_rate_1�O?7�ҳ}I       6%�	c,4X���A�*;


total_loss���@

error_R$iH?

learning_rate_1�O?7*^I       6%�	mp4X���A�*;


total_loss)ƪ@

error_R}'I?

learning_rate_1�O?7��iHI       6%�	��4X���A�*;


total_loss� �@

error_R�D?

learning_rate_1�O?7	by*I       6%�	�5X���A�*;


total_lossw�#A

error_Rx�F?

learning_rate_1�O?7&n��I       6%�	�K5X���A�*;


total_loss�)�@

error_R��@?

learning_rate_1�O?7�IޔI       6%�	��5X���A�*;


total_loss�ѿ@

error_R�&@?

learning_rate_1�O?7��{I       6%�	H�5X���A�*;


total_loss?��@

error_RJ
]?

learning_rate_1�O?7W�I       6%�	�'6X���A�*;


total_loss���@

error_R
�A?

learning_rate_1�O?7}�h0I       6%�	fo6X���A�*;


total_loss�>�@

error_R��F?

learning_rate_1�O?7PO��I       6%�	�6X���A�*;


total_loss:�@

error_R�9A?

learning_rate_1�O?7�BJI       6%�	��6X���A�*;


total_losss��@

error_R�P?

learning_rate_1�O?7�5�zI       6%�	;7X���A�*;


total_loss��A

error_R��^?

learning_rate_1�O?7/�I       6%�	~}7X���A�*;


total_loss�^�@

error_R�fi?

learning_rate_1�O?7Vk޽I       6%�	�7X���A�*;


total_lossr��@

error_Ri�\?

learning_rate_1�O?7g�?�I       6%�	58X���A�*;


total_loss6��@

error_Rq�T?

learning_rate_1�O?7�3��I       6%�	F8X���A�*;


total_loss�S�@

error_R�R?

learning_rate_1�O?7!�\I       6%�	ӌ8X���A�*;


total_loss�4�@

error_R�L?

learning_rate_1�O?7)�
I       6%�	��8X���A�*;


total_loss���@

error_R?oS?

learning_rate_1�O?7�3I       6%�	�9X���A�*;


total_loss�\A

error_R,wX?

learning_rate_1�O?7�)"�I       6%�	�b9X���A�*;


total_loss���@

error_RJlV?

learning_rate_1�O?7�|�uI       6%�	��9X���A�*;


total_lossT6�@

error_R��N?

learning_rate_1�O?7���I       6%�	��9X���A�*;


total_loss��@

error_R��G?

learning_rate_1�O?7��4MI       6%�	<,:X���A�*;


total_loss6F�@

error_R��H?

learning_rate_1�O?7o+{I       6%�	�s:X���A�*;


total_loss�Q�@

error_R��I?

learning_rate_1�O?7��s�I       6%�	�:X���A�*;


total_loss�U�@

error_R�K@?

learning_rate_1�O?7�hImI       6%�	�:X���A�*;


total_loss&?�@

error_R��`?

learning_rate_1�O?7���EI       6%�	^b;X���A�*;


total_loss-�A

error_R��R?

learning_rate_1�O?76�٦I       6%�	��;X���A�*;


total_loss�zA

error_R[?

learning_rate_1�O?7B1�)I       6%�	��;X���A�*;


total_loss�)�@

error_R��P?

learning_rate_1�O?7��VGI       6%�	7<X���A�*;


total_lossz��@

error_RS�S?

learning_rate_1�O?7��E2I       6%�	}�<X���A�*;


total_loss�#�@

error_R�GF?

learning_rate_1�O?7�]�I       6%�	��<X���A�*;


total_loss���@

error_R`a;?

learning_rate_1�O?7�DE�I       6%�	�=X���A�*;


total_loss��@

error_Rh�G?

learning_rate_1�O?7X\��I       6%�	�Z=X���A�*;


total_loss�@

error_R��S?

learning_rate_1�O?7��I       6%�	¡=X���A�*;


total_loss��@

error_R�P?

learning_rate_1�O?7ɪe,I       6%�	B�=X���A�*;


total_loss6:A

error_R�Z?

learning_rate_1�O?7l���I       6%�	�/>X���A�*;


total_loss�3�@

error_RN�^?

learning_rate_1�O?7`�E�I       6%�	Gr>X���A�*;


total_lossc��@

error_R�a`?

learning_rate_1�O?7e��I       6%�	ŵ>X���A�*;


total_loss��@

error_R)tO?

learning_rate_1�O?7Di�qI       6%�	�>X���A�*;


total_loss_��@

error_R��L?

learning_rate_1�O?7E���I       6%�	B?X���A�*;


total_lossC��@

error_R8�Q?

learning_rate_1�O?7���-I       6%�	f�?X���A�*;


total_loss�ޙ@

error_RC�s?

learning_rate_1�O?7�lI       6%�	��?X���A�*;


total_loss{��@

error_R2�Z?

learning_rate_1�O?7Y�3�I       6%�	l@X���A�*;


total_loss(�@

error_R&�T?

learning_rate_1�O?7]�r�I       6%�	S@X���A�*;


total_loss�ނ@

error_RTK?

learning_rate_1�O?7<DȿI       6%�	F�@X���A�*;


total_loss7g�@

error_R�i?

learning_rate_1�O?7���I       6%�	��@X���A�*;


total_loss���@

error_R��E?

learning_rate_1�O?7{�I       6%�	�-AX���A�*;


total_loss�@

error_R6G?

learning_rate_1�O?7�P}eI       6%�	}xAX���A�*;


total_loss�-�@

error_R�H?

learning_rate_1�O?7���I       6%�	�AX���A�*;


total_loss���@

error_R6s??

learning_rate_1�O?7&�m�I       6%�	BX���A�*;


total_lossp�A

error_R�r^?

learning_rate_1�O?7��խI       6%�	TBX���A�*;


total_loss阃@

error_Rs(P?

learning_rate_1�O?7ŌjI       6%�	�BX���A�*;


total_loss�~�@

error_R@�O?

learning_rate_1�O?7�PDI       6%�	J�BX���A�*;


total_loss�N�@

error_Rs�D?

learning_rate_1�O?7B��I       6%�	n#CX���A�*;


total_lossE��@

error_RV@\?

learning_rate_1�O?7%I�I       6%�	hCX���A�*;


total_loss���@

error_R�cF?

learning_rate_1�O?7�;��I       6%�	�CX���A�*;


total_loss\C A

error_R_�C?

learning_rate_1�O?7��5I       6%�	��CX���A�*;


total_loss�X�@

error_R�H?

learning_rate_1�O?7ȵNI       6%�	�5DX���A�*;


total_lossE
A

error_R�xI?

learning_rate_1�O?79�PI       6%�	$yDX���A�*;


total_loss](�@

error_R��7?

learning_rate_1�O?7�x
I       6%�	W�DX���A�*;


total_lossC�T@

error_R{~D?

learning_rate_1�O?7�w�I       6%�	TEX���A�*;


total_lossAԱ@

error_R-8I?

learning_rate_1�O?7)u��I       6%�	�FEX���A�*;


total_loss�=�@

error_R!�<?

learning_rate_1�O?7��b�I       6%�	��EX���A�*;


total_loss;��@

error_Rt�E?

learning_rate_1�O?7m44I       6%�	=�EX���A�*;


total_lossy��@

error_RQW?

learning_rate_1�O?7k1�I       6%�	�FX���A�*;


total_loss3ar@

error_R��N?

learning_rate_1�O?7��EI       6%�	�UFX���A�*;


total_lossש�@

error_R�>1?

learning_rate_1�O?7�I`I       6%�	��FX���A�*;


total_loss6��@

error_R=�9?

learning_rate_1�O?7~0�I       6%�	��FX���A�*;


total_loss�F�@

error_R��E?

learning_rate_1�O?7p�! I       6%�	$'GX���A�*;


total_loss�O�@

error_RDK?

learning_rate_1�O?7�0p�I       6%�	�qGX���A�*;


total_loss��A

error_R6�`?

learning_rate_1�O?7�x�I       6%�	.�GX���A�*;


total_loss�6�@

error_R�
D?

learning_rate_1�O?7�M�I       6%�	�HX���A�*;


total_loss:k�@

error_RwA?

learning_rate_1�O?7ôLI       6%�	)]HX���A�*;


total_lossE��@

error_RH:?

learning_rate_1�O?7# �mI       6%�	E�HX���A�*;


total_lossۢ�@

error_R�vY?

learning_rate_1�O?7���I       6%�	e�HX���A�*;


total_loss$�o@

error_RC�`?

learning_rate_1�O?7�|InI       6%�	:IX���A�*;


total_loss�A�@

error_R �W?

learning_rate_1�O?7����I       6%�	�IX���A�*;


total_loss��@

error_R��I?

learning_rate_1�O?7
�5�I       6%�	��IX���A�*;


total_loss�n�@

error_R�	A?

learning_rate_1�O?7
B�I       6%�	!JX���A�*;


total_loss_k@

error_R�H?

learning_rate_1�O?7Ul�1I       6%�	YhJX���A�*;


total_loss��@

error_R�F?

learning_rate_1�O?7)+�I       6%�	Q�JX���A�*;


total_loss)F�@

error_R�)[?

learning_rate_1�O?7�8�I       6%�	��JX���A�*;


total_lossH��@

error_RxI?

learning_rate_1�O?7.U�I       6%�	�QKX���A�*;


total_loss;��@

error_R�)M?

learning_rate_1�O?7V�֖I       6%�	�KX���A�*;


total_lossn=�@

error_R$6]?

learning_rate_1�O?7�\�7I       6%�	��KX���A�*;


total_loss�c�@

error_R��Q?

learning_rate_1�O?7&(�I       6%�	)3LX���A�*;


total_lossF��@

error_RсZ?

learning_rate_1�O?7.�I       6%�	9wLX���A�*;


total_losscq@

error_R\�O?

learning_rate_1�O?7�`��I       6%�	9�LX���A�*;


total_loss<��@

error_RB]?

learning_rate_1�O?7�`uI       6%�	��LX���A�*;


total_lossؚ@

error_RJ?

learning_rate_1�O?7��	�I       6%�	�CMX���A�*;


total_lossz̆@

error_R�U?

learning_rate_1�O?7��EtI       6%�	%�MX���A�*;


total_lossL��@

error_R�rW?

learning_rate_1�O?7KB��I       6%�	��MX���A�*;


total_loss�@

error_RI\?

learning_rate_1�O?7�5I       6%�	fNX���A�*;


total_loss�a�@

error_RW=Q?

learning_rate_1�O?7/��1I       6%�	�VNX���A�*;


total_loss8��@

error_R&"b?

learning_rate_1�O?7�2A2I       6%�	��NX���A�*;


total_lossc�@

error_RR$M?

learning_rate_1�O?7�,�I       6%�	��NX���A�*;


total_loss�$�@

error_R��@?

learning_rate_1�O?7�Z6I       6%�	x%OX���A�*;


total_loss��@

error_R��\?

learning_rate_1�O?7���I       6%�		jOX���A�*;


total_loss���@

error_R-P?

learning_rate_1�O?7��;I       6%�	F�OX���A�*;


total_loss���@

error_R��_?

learning_rate_1�O?7���vI       6%�	_�OX���A�*;


total_loss�֔@

error_R:<?

learning_rate_1�O?7k��UI       6%�	�DPX���A�*;


total_loss��@

error_R@&K?

learning_rate_1�O?7z�fI       6%�	�PX���A�*;


total_lossѴ�@

error_Rq�U?

learning_rate_1�O?7f('I       6%�	��PX���A�*;


total_loss���@

error_RO3H?

learning_rate_1�O?7�ϡ�I       6%�	cQX���A�*;


total_lossd��@

error_R*]?

learning_rate_1�O?7����I       6%�	�ZQX���A�*;


total_loss�s�@

error_R��D?

learning_rate_1�O?7��&:I       6%�	#�QX���A�*;


total_lossQfA

error_R�U?

learning_rate_1�O?7�B9I       6%�	��QX���A�*;


total_loss��@

error_RW�N?

learning_rate_1�O?7,�qI       6%�	�7RX���A�*;


total_loss���@

error_RO�J?

learning_rate_1�O?7��n�I       6%�	�|RX���A�*;


total_loss-y�@

error_R?�Q?

learning_rate_1�O?7�Z�I       6%�	 �RX���A�*;


total_loss�j�@

error_R��L?

learning_rate_1�O?7�\%�I       6%�	�SX���A�*;


total_loss8��@

error_R�_?

learning_rate_1�O?7����I       6%�	{ISX���A�*;


total_losst��@

error_R7�G?

learning_rate_1�O?7l�I       6%�	ɌSX���A�*;


total_loss��@

error_R3�V?

learning_rate_1�O?7H\I       6%�	 �SX���A�*;


total_loss��@

error_R;WF?

learning_rate_1�O?7vE��I       6%�	�TX���A�*;


total_loss\>�@

error_R�^I?

learning_rate_1�O?7Oy�TI       6%�	�UTX���A�*;


total_loss�*�@

error_Rh??

learning_rate_1�O?7+�m�I       6%�	}�TX���A�*;


total_loss�
A

error_R%M?

learning_rate_1�O?7Y��lI       6%�	��TX���A�*;


total_loss�Q�@

error_R��B?

learning_rate_1�O?7����I       6%�	5.UX���A�*;


total_lossϱ�@

error_Rv�X?

learning_rate_1�O?7gݐI       6%�	WwUX���A�*;


total_lossM��@

error_R X?

learning_rate_1�O?7%��I       6%�	R�UX���A�*;


total_loss[�@

error_RʊV?

learning_rate_1�O?7)��I       6%�	"VX���A�*;


total_lossl�@

error_R��F?

learning_rate_1�O?7b�?I       6%�	�JVX���A�*;


total_lossZ��@

error_R�"I?

learning_rate_1�O?7zI�I       6%�	�VX���A�*;


total_loss���@

error_R��U?

learning_rate_1�O?7{gk�I       6%�	0�VX���A�*;


total_loss��@

error_R��R?

learning_rate_1�O?7�6�I       6%�	SWX���A�*;


total_loss�A

error_R,dR?

learning_rate_1�O?7����I       6%�	�WWX���A�*;


total_loss��A

error_R 5W?

learning_rate_1�O?7o+>!I       6%�	A�WX���A�*;


total_lossn��@

error_Rn�P?

learning_rate_1�O?7�,��I       6%�	��WX���A�*;


total_losst�@

error_RH&Z?

learning_rate_1�O?7�Y�OI       6%�	�$XX���A�*;


total_loss�H�@

error_RIqB?

learning_rate_1�O?7@�	I       6%�	YnXX���A�*;


total_loss.�A

error_RRoX?

learning_rate_1�O?7���I       6%�	)�XX���A�*;


total_loss��@

error_RI�5?

learning_rate_1�O?7���WI       6%�	��XX���A�*;


total_lossz0�@

error_R3�Q?

learning_rate_1�O?7x���I       6%�		AYX���A�*;


total_loss�A

error_R�zY?

learning_rate_1�O?7EhI       6%�	c�YX���A�*;


total_loss��@

error_R"W?

learning_rate_1�O?7��,I       6%�	G�YX���A�*;


total_loss���@

error_R��J?

learning_rate_1�O?7��~�I       6%�	� ZX���A�*;


total_loss8��@

error_R��\?

learning_rate_1�O?7�M3I       6%�	�hZX���A�*;


total_lossj��@

error_Rn0]?

learning_rate_1�O?7jb��I       6%�	ۯZX���A�*;


total_lossja�@

error_R��U?

learning_rate_1�O?7)ShI       6%�	��ZX���A�*;


total_lossT�@

error_R��X?

learning_rate_1�O?7��tI       6%�	bQ[X���A�*;


total_lossI>�@

error_RO�M?

learning_rate_1�O?7��5�I       6%�	��[X���A�*;


total_loss_]�@

error_R��L?

learning_rate_1�O?7���bI       6%�	��[X���A�*;


total_loss���@

error_R B?

learning_rate_1�O?7�`�kI       6%�	R.\X���A�*;


total_loss��@

error_Rs�S?

learning_rate_1�O?7e�koI       6%�	)q\X���A�*;


total_lossŻs@

error_RC�K?

learning_rate_1�O?7��<I       6%�	 �\X���A�*;


total_losss��@

error_RaV?

learning_rate_1�O?7�M<PI       6%�	,�\X���A�*;


total_losso|@

error_R��6?

learning_rate_1�O?7wٛI       6%�	�>]X���A�*;


total_loss���@

error_R�;?

learning_rate_1�O?7;�gI       6%�	��]X���A�*;


total_loss��@

error_R��N?

learning_rate_1�O?7�}�I       6%�	=�]X���A�*;


total_loss���@

error_R,�U?

learning_rate_1�O?7�f|�I       6%�	d^X���A�*;


total_loss��@

error_RLD?

learning_rate_1�O?7�b�I       6%�	f^X���A�*;


total_loss���@

error_RϽP?

learning_rate_1�O?7���4I       6%�	׮^X���A�*;


total_loss�~�@

error_R��P?

learning_rate_1�O?7���I       6%�	��^X���A�*;


total_loss0�@

error_R��W?

learning_rate_1�O?7�Kl�I       6%�	�;_X���A�*;


total_lossCd�@

error_R�CJ?

learning_rate_1�O?7���I       6%�	i�_X���A�*;


total_loss�!�@

error_R��K?

learning_rate_1�O?7�%6�I       6%�	��_X���A�*;


total_loss��@

error_R�$<?

learning_rate_1�O?7X�	I       6%�	�`X���A�*;


total_loss���@

error_R3�T?

learning_rate_1�O?7}SC�I       6%�	�L`X���A�*;


total_lossi�X@

error_R_�G?

learning_rate_1�O?7)��}I       6%�	p�`X���A�*;


total_loss�@

error_R��X?

learning_rate_1�O?7EЄHI       6%�	;�`X���A�*;


total_loss��@

error_R�se?

learning_rate_1�O?7����I       6%�	O aX���A�*;


total_loss���@

error_RE�B?

learning_rate_1�O?7��[�I       6%�	reaX���A�*;


total_losso
�@

error_R�jP?

learning_rate_1�O?7���EI       6%�	�aX���A�*;


total_lossO��@

error_RISS?

learning_rate_1�O?7r���I       6%�	�aX���A�*;


total_loss� �@

error_R�E?

learning_rate_1�O?7�޳DI       6%�	�<bX���A�*;


total_loss��@

error_Rv4a?

learning_rate_1�O?7lj��I       6%�	<�bX���A�*;


total_loss�Q�@

error_R��D?

learning_rate_1�O?7c��<I       6%�	4�bX���A�*;


total_loss[l�@

error_R�a\?

learning_rate_1�O?7'��I       6%�	9cX���A�*;


total_loss_�	A

error_Rq�Z?

learning_rate_1�O?7����I       6%�	^cX���A�*;


total_loss/b�@

error_Rw�X?

learning_rate_1�O?7�!�I       6%�	/�cX���A�*;


total_loss �@

error_RcYI?

learning_rate_1�O?79��I       6%�	��cX���A�*;


total_loss�6�@

error_R@:Q?

learning_rate_1�O?7[�#^I       6%�	%dX���A�*;


total_lossԃ�@

error_R۶U?

learning_rate_1�O?7Mi�I       6%�	kdX���A�*;


total_loss�`�@

error_R��R?

learning_rate_1�O?7��I       6%�	&�dX���A�*;


total_loss� �@

error_Ra%Q?

learning_rate_1�O?7<�1�I       6%�	��dX���A�*;


total_loss�6�@

error_R�L?

learning_rate_1�O?7 �nI       6%�	�4eX���A�*;


total_lossæ A

error_R7�O?

learning_rate_1�O?7B�^�I       6%�	eweX���A�*;


total_loss�>�@

error_RfkR?

learning_rate_1�O?7�H*�I       6%�	C�eX���A�*;


total_loss�G�@

error_R�]?

learning_rate_1�O?7�I��I       6%�	vfX���A�*;


total_loss���@

error_R��B?

learning_rate_1�O?7d��/I       6%�	�GfX���A�*;


total_lossw��@

error_R$BN?

learning_rate_1�O?7��ybI       6%�	[�fX���A�*;


total_loss�A

error_R?1R?

learning_rate_1�O?7$?7�I       6%�	��fX���A�*;


total_lossT6�@

error_RvS?

learning_rate_1�O?7??�I       6%�	�gX���A�*;


total_lossJ��@

error_R��U?

learning_rate_1�O?755d�I       6%�	)VgX���A�*;


total_loss(�@

error_R/�>?

learning_rate_1�O?7�k�!I       6%�	�gX���A�*;


total_loss$��@

error_R4B?

learning_rate_1�O?7�f%I       6%�	��gX���A�*;


total_lossŰ�@

error_R��U?

learning_rate_1�O?7�ӐI       6%�	�hX���A�*;


total_loss��@

error_R
�J?

learning_rate_1�O?7�h�I       6%�	�{hX���A�*;


total_loss�D�@

error_R�/^?

learning_rate_1�O?7:�"I       6%�	��hX���A�*;


total_loss�h�@

error_RfOH?

learning_rate_1�O?7@�
UI       6%�	qiX���A�*;


total_loss �n@

error_R�\?

learning_rate_1�O?7'��I       6%�	�biX���A�*;


total_loss哧@

error_R��O?

learning_rate_1�O?7��LI       6%�	��iX���A�*;


total_loss�m�@

error_R��@?

learning_rate_1�O?7��JI       6%�	RjX���A�*;


total_lossŋ�@

error_RڞU?

learning_rate_1�O?7NX&�I       6%�	�TjX���A�*;


total_loss�4�@

error_R��R?

learning_rate_1�O?7�|��I       6%�	D�jX���A�*;


total_lossI	�@

error_R<Y?

learning_rate_1�O?7��7%I       6%�	��jX���A�*;


total_loss���@

error_R=h?

learning_rate_1�O?7�gbI       6%�	�2kX���A�*;


total_loss��@

error_R{�R?

learning_rate_1�O?7X��'I       6%�	��kX���A�*;


total_loss�K�@

error_RJ�H?

learning_rate_1�O?7�+G`I       6%�	P�kX���A�*;


total_loss�H�@

error_R��H?

learning_rate_1�O?7t���I       6%�	�lX���A�*;


total_loss�W�@

error_R&GQ?

learning_rate_1�O?7�-�I       6%�	�blX���A�*;


total_loss���@

error_R^N?

learning_rate_1�O?7A-� I       6%�	ݦlX���A�*;


total_losssI�@

error_R��>?

learning_rate_1�O?7��7tI       6%�	��lX���A�*;


total_loss�P�@

error_RQ\?

learning_rate_1�O?7#�m�I       6%�	p3mX���A�*;


total_loss!��@

error_R��V?

learning_rate_1�O?7����I       6%�	�zmX���A�*;


total_loss���@

error_RlFC?

learning_rate_1�O?7L>�1I       6%�	��mX���A�*;


total_loss�~�@

error_RQ�B?

learning_rate_1�O?7��>�I       6%�	�
nX���A�*;


total_loss��@

error_R��2?

learning_rate_1�O?746��I       6%�	rOnX���A�*;


total_loss>�@

error_R}:?

learning_rate_1�O?7H�}YI       6%�	��nX���A�*;


total_loss�Dz@

error_R��]?

learning_rate_1�O?7?�lI       6%�	2�nX���A�*;


total_lossj��@

error_R@\Z?

learning_rate_1�O?7��,I       6%�	�oX���A�*;


total_loss�X�@

error_R�VZ?

learning_rate_1�O?7�\&-I       6%�	d_oX���A�*;


total_loss��@

error_R�4S?

learning_rate_1�O?7^*I       6%�	��oX���A�*;


total_loss R�@

error_R��L?

learning_rate_1�O?7��lI       6%�	J�oX���A�*;


total_lossh��@

error_R�yP?

learning_rate_1�O?7ƪ�II       6%�	T+pX���A�*;


total_loss���@

error_R��A?

learning_rate_1�O?7��I       6%�	WnpX���A�*;


total_lossՆ@

error_R��7?

learning_rate_1�O?7t���I       6%�	�pX���A�*;


total_loss�0A

error_R�e?

learning_rate_1�O?7�4��I       6%�	?�pX���A�*;


total_lossa�A

error_R��H?

learning_rate_1�O?7X0I       6%�	�9qX���A�*;


total_loss�&�@

error_R	�U?

learning_rate_1�O?7�Y�I       6%�	�~qX���A�*;


total_loss�@

error_R�]R?

learning_rate_1�O?7E�i)I       6%�	u�qX���A�*;


total_loss*��@

error_RXE@?

learning_rate_1�O?7S7�I       6%�	�rX���A�*;


total_lossq�A

error_RxJ?

learning_rate_1�O?7�^�I       6%�	�VrX���A�*;


total_loss�@

error_R��Q?

learning_rate_1�O?7�^�I       6%�	��rX���A�*;


total_loss��@

error_R�??

learning_rate_1�O?7jq�I       6%�	��rX���A�*;


total_loss��@

error_R��??

learning_rate_1�O?7�-�$I       6%�	r sX���A�*;


total_loss4��@

error_R�QA?

learning_rate_1�O?7�GR�I       6%�	~gsX���A�*;


total_loss��@

error_R�GE?

learning_rate_1�O?7V�z"I       6%�	�sX���A�*;


total_loss`_A

error_R��L?

learning_rate_1�O?7R̻pI       6%�	��sX���A�*;


total_loss�0�@

error_RH�L?

learning_rate_1�O?7�
�KI       6%�	�;tX���A�*;


total_lossf��@

error_Rf�G?

learning_rate_1�O?7��I       6%�	j�tX���A�*;


total_loss��@

error_R�7O?

learning_rate_1�O?7<`�I       6%�	��tX���A�*;


total_loss4lA

error_R��E?

learning_rate_1�O?7"��zI       6%�	�	uX���A�*;


total_loss�A

error_R��\?

learning_rate_1�O?7'p��I       6%�	cLuX���A�*;


total_loss���@

error_R�\?

learning_rate_1�O?7�ќ�I       6%�	C�uX���A�*;


total_loss�a	A

error_R_�I?

learning_rate_1�O?7ͳq�I       6%�	��uX���A�*;


total_loss7ߘ@

error_R�	b?

learning_rate_1�O?7�X�DI       6%�	vX���A�*;


total_lossv��@

error_R{fX?

learning_rate_1�O?7�>I       6%�	{avX���A�*;


total_loss�q�@

error_R�`=?

learning_rate_1�O?7
Њ�I       6%�	٦vX���A�*;


total_lossQ�y@

error_R�\B?

learning_rate_1�O?7�]�I       6%�	�vX���A�*;


total_lossf�@

error_R�V@?

learning_rate_1�O?7]�=�I       6%�	�7wX���A�*;


total_lossf(�@

error_RfW?

learning_rate_1�O?7~�(I       6%�	�~wX���A�*;


total_lossox�@

error_R��C?

learning_rate_1�O?7��I       6%�	��wX���A�*;


total_loss�P�@

error_R�!I?

learning_rate_1�O?7����I       6%�	nxX���A�*;


total_loss�A�@

error_R�#H?

learning_rate_1�O?7#�LgI       6%�	!MxX���A�*;


total_loss���@

error_R!Z=?

learning_rate_1�O?7q%I       6%�	7�xX���A�*;


total_loss���@

error_R�pR?

learning_rate_1�O?7�itXI       6%�	��xX���A�*;


total_loss�փ@

error_R�n??

learning_rate_1�O?7
i�RI       6%�	�(yX���A�*;


total_loss?�s@

error_R3GD?

learning_rate_1�O?7����I       6%�	�yX���A�*;


total_loss8�@

error_R^R?

learning_rate_1�O?7*�zaI       6%�	U�yX���A�*;


total_loss1��@

error_R��]?

learning_rate_1�O?7]2fI       6%�	�EzX���A�*;


total_loss�Y�@

error_R��@?

learning_rate_1�O?7��FOI       6%�	��zX���A�*;


total_loss��@

error_R-d=?

learning_rate_1�O?7���RI       6%�	�zX���A�*;


total_lossc�@

error_R�H?

learning_rate_1�O?7^fθI       6%�	{X���A�*;


total_loss$�@

error_R A?

learning_rate_1�O?7
��8I       6%�	��{X���A�*;


total_loss�ή@

error_R=�E?

learning_rate_1�O?7J�'II       6%�	��{X���A�*;


total_losss�@

error_R� H?

learning_rate_1�O?78�1I       6%�	+|X���A�*;


total_loss�k�@

error_RR�H?

learning_rate_1�O?7M�C�I       6%�	�c|X���A�*;


total_loss�@

error_RJjA?

learning_rate_1�O?7�4)I       6%�	!�|X���A�*;


total_loss�X�@

error_Rd�b?

learning_rate_1�O?7��B�I       6%�	+�|X���A�*;


total_lossIl�@

error_R�^?

learning_rate_1�O?7��I       6%�	{}X���A�*;


total_lossJ�@

error_R԰F?

learning_rate_1�O?7�\�tI       6%�	�}X���A�*;


total_loss��@

error_R\?

learning_rate_1�O?7�3�gI       6%�	�	~X���A�*;


total_loss%f�@

error_R�??

learning_rate_1�O?7��E�I       6%�	 N~X���A�*;


total_loss���@

error_R��^?

learning_rate_1�O?7�HTI       6%�	&�~X���A�*;


total_loss���@

error_RQ_?

learning_rate_1�O?7 b^�I       6%�	��~X���A�*;


total_loss��@

error_R�??

learning_rate_1�O?7�_I       6%�	�X���A�*;


total_loss�'�@

error_Rv�a?

learning_rate_1�O?7���I       6%�	�ZX���A�*;


total_lossg��@

error_RF??

learning_rate_1�O?7��I       6%�	��X���A�*;


total_loss?f�@

error_RӒB?

learning_rate_1�O?7�II       6%�	w�X���A�*;


total_loss:��@

error_RZ�8?

learning_rate_1�O?7'<��I       6%�	�(�X���A�*;


total_loss$��@

error_RT�M?

learning_rate_1�O?7��8wI       6%�	nj�X���A�*;


total_loss0z	A

error_R�r)?

learning_rate_1�O?7�9+�I       6%�	G��X���A�*;


total_loss�ř@

error_Rw�S?

learning_rate_1�O?7���*I       6%�	Q�X���A�*;


total_lossq��@

error_RڤW?

learning_rate_1�O?7k#I       6%�	�5�X���A�*;


total_loss&�@

error_R>R?

learning_rate_1�O?7�=s]I       6%�	�x�X���A�*;


total_loss�I�@

error_R�N?

learning_rate_1�O?7����I       6%�	āX���A�*;


total_loss܌@

error_Rּ@?

learning_rate_1�O?7�x�dI       6%�	�X���A�*;


total_loss݃@

error_R�0^?

learning_rate_1�O?7��9II       6%�	(Y�X���A�*;


total_loss/o�@

error_Ro�D?

learning_rate_1�O?7b��I       6%�	}��X���A�*;


total_loss8��@

error_R��b?

learning_rate_1�O?7�f�RI       6%�	��X���A�*;


total_losszȃ@

error_RV�Z?

learning_rate_1�O?7��ڐI       6%�	I'�X���A�*;


total_loss���@

error_RJ^?

learning_rate_1�O?7�V�7I       6%�	q�X���A�*;


total_loss�c�@

error_R݂O?

learning_rate_1�O?7.���I       6%�	~��X���A�*;


total_lossf>�@

error_R�9?

learning_rate_1�O?7��I       6%�	/��X���A�*;


total_loss�2�@

error_R�)L?

learning_rate_1�O?7�� �I       6%�	Q<�X���A�*;


total_lossv�t@

error_RW�\?

learning_rate_1�O?7�dEI       6%�	���X���A�*;


total_loss���@

error_RaiL?

learning_rate_1�O?7��~I       6%�	ɄX���A�*;


total_lossbݞ@

error_R��6?

learning_rate_1�O?7�F�I       6%�	{
�X���A�*;


total_loss3\�@

error_RtlK?

learning_rate_1�O?7t[EFI       6%�	zP�X���A�*;


total_loss1,�@

error_R��C?

learning_rate_1�O?7`��I       6%�	���X���A�*;


total_losse�@

error_R��O?

learning_rate_1�O?7�m�I       6%�	ۅX���A�*;


total_lossxK�@

error_R�>?

learning_rate_1�O?7�;{�I       6%�	��X���A�*;


total_loss�v�@

error_R��E?

learning_rate_1�O?7�W؈I       6%�	Qd�X���A�*;


total_loss���@

error_RϦH?

learning_rate_1�O?7f %<I       6%�	���X���A�*;


total_lossw�@

error_R�WB?

learning_rate_1�O?7�!�I       6%�	p�X���A�*;


total_lossԬ�@

error_R��S?

learning_rate_1�O?7m럱I       6%�	#0�X���A�*;


total_loss���@

error_R,�C?

learning_rate_1�O?7�_RtI       6%�	r�X���A�*;


total_losss��@

error_R!JS?

learning_rate_1�O?7��p�I       6%�	ѽ�X���A�*;


total_loss]ն@

error_RҌF?

learning_rate_1�O?7��
I       6%�	�X���A�*;


total_loss t�@

error_Ra�K?

learning_rate_1�O?7�4I       6%�	�V�X���A�*;


total_loss�i�@

error_R�B?

learning_rate_1�O?7"��I       6%�	��X���A�*;


total_loss��@

error_Rs�]?

learning_rate_1�O?7���1I       6%�	��X���A�*;


total_loss��@

error_RaM?

learning_rate_1�O?7z��I       6%�	�X�X���A�*;


total_lossFB�@

error_R��[?

learning_rate_1�O?7]|�'I       6%�	�ʉX���A�*;


total_loss`T�@

error_R� S?

learning_rate_1�O?7�~�"I       6%�	�X���A�*;


total_loss���@

error_Ri�N?

learning_rate_1�O?7����I       6%�	�Y�X���A�*;


total_loss�$�@

error_R!�T?

learning_rate_1�O?7}uJOI       6%�	ܠ�X���A�*;


total_lossK��@

error_Ra/H?

learning_rate_1�O?7}ӛWI       6%�	��X���A�*;


total_loss��@

error_R��M?

learning_rate_1�O?7mj7aI       6%�	�T�X���A�*;


total_loss�-t@

error_R�SZ?

learning_rate_1�O?7E���I       6%�	���X���A�*;


total_loss��@

error_R�=?

learning_rate_1�O?7���I       6%�	���X���A�*;


total_loss��@

error_R�p>?

learning_rate_1�O?7w�j�I       6%�	7@�X���A�*;


total_lossj&�@

error_R��P?

learning_rate_1�O?7�� I       6%�	���X���A�*;


total_loss���@

error_R}M?

learning_rate_1�O?7"a��I       6%�	�ˌX���A�*;


total_loss�^�@

error_R�L?

learning_rate_1�O?7s
MI       6%�	�X���A�*;


total_loss&�@

error_R#�A?

learning_rate_1�O?7�i�NI       6%�	�U�X���A�*;


total_loss_�@

error_R��b?

learning_rate_1�O?7k�
I       6%�	U��X���A�*;


total_lossd�d@

error_RqB?

learning_rate_1�O?7�E��I       6%�	��X���A�*;


total_lossں�@

error_R�N?

learning_rate_1�O?7nv�I       6%�	/�X���A�*;


total_loss�@

error_RQ=B?

learning_rate_1�O?7s�8I       6%�	u�X���A�*;


total_lossB�A

error_R{�V?

learning_rate_1�O?7�3E�I       6%�	㹎X���A�*;


total_loss1��@

error_RS�X?

learning_rate_1�O?7u�9�I       6%�	��X���A�*;


total_loss<�@

error_R�oX?

learning_rate_1�O?7�I       6%�	�>�X���A�*;


total_loss��@

error_RZ(H?

learning_rate_1�O?7��t�I       6%�	A��X���A�*;


total_loss���@

error_R��N?

learning_rate_1�O?7���I       6%�	iƏX���A�*;


total_lossm��@

error_RţQ?

learning_rate_1�O?7<�MoI       6%�	F	�X���A�*;


total_loss�g�@

error_R�^?

learning_rate_1�O?7e�O�I       6%�	�N�X���A�*;


total_loss�V�@

error_Ri�Q?

learning_rate_1�O?7�$~�I       6%�	���X���A�*;


total_loss/5�@

error_RG@?

learning_rate_1�O?7�=�7I       6%�	֐X���A�*;


total_loss6ř@

error_R @?

learning_rate_1�O?7LO��I       6%�	��X���A�*;


total_lossX*�@

error_R�3?

learning_rate_1�O?7$]�7I       6%�	f�X���A�*;


total_loss�V�@

error_R�4>?

learning_rate_1�O?7=��I       6%�	E��X���A�*;


total_loss-��@

error_R�??

learning_rate_1�O?7��X[I       6%�	���X���A�*;


total_loss�A

error_Rc�>?

learning_rate_1�O?7��I       6%�	�C�X���A�*;


total_loss�o�@

error_R��M?

learning_rate_1�O?7��C�I       6%�	���X���A�*;


total_loss ��@

error_R�JO?

learning_rate_1�O?7��8�I       6%�	�גX���A�*;


total_loss{lw@

error_R� I?

learning_rate_1�O?7ћ)�I       6%�	 �X���A�*;


total_loss\=�@

error_R�g?

learning_rate_1�O?7��LwI       6%�	Yg�X���A�*;


total_loss�;A

error_R`�S?

learning_rate_1�O?7&�gtI       6%�	���X���A�*;


total_loss x�@

error_Rh�P?

learning_rate_1�O?74��I       6%�	��X���A�*;


total_loss:�@

error_R/�A?

learning_rate_1�O?7Y8��I       6%�	�K�X���A�*;


total_loss�A

error_RFv\?

learning_rate_1�O?7ݹ�gI       6%�	ޘ�X���A�*;


total_lossq��@

error_R&�P?

learning_rate_1�O?7����I       6%�	�ݔX���A�*;


total_loss갗@

error_R{BR?

learning_rate_1�O?7�eXRI       6%�	$�X���A�*;


total_loss~�@

error_R��K?

learning_rate_1�O?7��IXI       6%�	~j�X���A�*;


total_loss�3[@

error_R�sB?

learning_rate_1�O?7+�MI       6%�	㰕X���A�*;


total_loss ��@

error_R��H?

learning_rate_1�O?71Y�QI       6%�	w��X���A�*;


total_loss�'�@

error_Rxb?

learning_rate_1�O?7/�5I       6%�	�F�X���A�*;


total_loss�~�@

error_R3b?

learning_rate_1�O?7j�:I       6%�	��X���A�*;


total_lossAA

error_R��c?

learning_rate_1�O?7��qVI       6%�	�ԖX���A�*;


total_loss��@

error_R�H?

learning_rate_1�O?7ݬ=}I       6%�	�X���A�*;


total_loss�٨@

error_R;4Y?

learning_rate_1�O?7�\@I       6%�	]�X���A�*;


total_loss��@

error_R��Q?

learning_rate_1�O?7E���I       6%�	Þ�X���A�*;


total_loss�X�@

error_R�#@?

learning_rate_1�O?7%C�xI       6%�	�X���A�*;


total_lossw��@

error_R�H?

learning_rate_1�O?7�n uI       6%�	7/�X���A�*;


total_loss�`�@

error_R�I?

learning_rate_1�O?7k��I       6%�	5r�X���A�*;


total_loss��@

error_R��L?

learning_rate_1�O?7�asNI       6%�	F��X���A�*;


total_loss0��@

error_RHAP?

learning_rate_1�O?7�q-I       6%�	Z �X���A�*;


total_loss�_�@

error_RC�I?

learning_rate_1�O?7z��wI       6%�	�E�X���A�*;


total_loss�A@

error_R�T?

learning_rate_1�O?7�
��I       6%�	܋�X���A�*;


total_loss�I�@

error_R�~d?

learning_rate_1�O?7�O��I       6%�	�ՙX���A�*;


total_loss���@

error_Rw�I?

learning_rate_1�O?7w�+�I       6%�	��X���A�*;


total_loss�j�@

error_ReuI?

learning_rate_1�O?7��3�I       6%�	@c�X���A�*;


total_losssA

error_R\�H?

learning_rate_1�O?7��kI       6%�	䦚X���A�*;


total_loss��@

error_R}�H?

learning_rate_1�O?7h��QI       6%�	��X���A�*;


total_loss}��@

error_R�o>?

learning_rate_1�O?7"<R9I       6%�	E�X���A�*;


total_loss=M�@

error_R�J?

learning_rate_1�O?7��#I       6%�	a��X���A�*;


total_lossL��@

error_R6�I?

learning_rate_1�O?7�O�GI       6%�	��X���A�*;


total_loss�q�@

error_R�H?

learning_rate_1�O?7��>I       6%�	@1�X���A�*;


total_loss���@

error_R[�>?

learning_rate_1�O?7�o*I       6%�	gs�X���A�*;


total_loss/ε@

error_R��i?

learning_rate_1�O?7�-I       6%�	��X���A�*;


total_loss��@

error_R2�J?

learning_rate_1�O?7)��0I       6%�	���X���A�*;


total_loss���@

error_R��^?

learning_rate_1�O?7�>�
I       6%�	�C�X���A�*;


total_lossԛ�@

error_R�E?

learning_rate_1�O?7��UI       6%�	䈝X���A�*;


total_loss�%�@

error_R`�G?

learning_rate_1�O?7J�I       6%�	YΝX���A�*;


total_lossh)�@

error_R�O?

learning_rate_1�O?7���4I       6%�	:�X���A�*;


total_loss=g�@

error_R��R?

learning_rate_1�O?7r�I       6%�	�]�X���A�*;


total_lossM��@

error_R��V?

learning_rate_1�O?7���nI       6%�	ݡ�X���A�*;


total_loss�A

error_R.�P?

learning_rate_1�O?7D��"I       6%�	R�X���A�*;


total_lossS��@

error_Rv%G?

learning_rate_1�O?7L;I       6%�	�+�X���A�*;


total_loss��|@

error_R�H?

learning_rate_1�O?7V�p�I       6%�	Ks�X���A�*;


total_loss��@

error_R�9J?

learning_rate_1�O?7�k~�I       6%�	޼�X���A�*;


total_loss�7�@

error_RH5Z?

learning_rate_1�O?7��I       6%�	a�X���A�*;


total_loss�	�@

error_R�-9?

learning_rate_1�O?7����I       6%�	�K�X���A�*;


total_loss&"�@

error_R��X?

learning_rate_1�O?7ƨ>I       6%�	F��X���A�*;


total_loss��@

error_R��>?

learning_rate_1�O?7� I       6%�	�ޠX���A�*;


total_loss�n�@

error_R�6M?

learning_rate_1�O?7,6nI       6%�	X�X���A�*;


total_loss���@

error_R�PE?

learning_rate_1�O?7�M�I       6%�	�d�X���A�*;


total_lossώ@

error_R�JI?

learning_rate_1�O?7J��kI       6%�	���X���A�*;


total_loss\�@

error_R
HF?

learning_rate_1�O?7�X`I       6%�	��X���A�*;


total_loss�Z�@

error_R�N?

learning_rate_1�O?7�R��I       6%�	�1�X���A�*;


total_loss3��@

error_R�~X?

learning_rate_1�O?7}m�'I       6%�	�w�X���A�*;


total_loss���@

error_R�O?

learning_rate_1�O?7�>�I       6%�	���X���A�*;


total_lossrP�@

error_R��7?

learning_rate_1�O?7����I       6%�	8 �X���A�*;


total_lossn�a@

error_RRV?

learning_rate_1�O?7_O�I       6%�	�E�X���A�*;


total_loss���@

error_R�T?

learning_rate_1�O?7	���I       6%�	P��X���A�*;


total_loss�ۼ@

error_R�WX?

learning_rate_1�O?7�ϹI       6%�	�̣X���A�*;


total_loss���@

error_R�G?

learning_rate_1�O?7��^ZI       6%�	��X���A�*;


total_loss�
�@

error_R��^?

learning_rate_1�O?7�;�sI       6%�	�S�X���A�*;


total_lossEc@

error_RM&Q?

learning_rate_1�O?7��k�I       6%�	���X���A�*;


total_loss�D�@

error_R�7Q?

learning_rate_1�O?7c+%�I       6%�	�ܤX���A�*;


total_loss
��@

error_R;E?

learning_rate_1�O?7�L�I       6%�	�"�X���A�*;


total_loss�@

error_R.�A?

learning_rate_1�O?7@�%�I       6%�	<f�X���A�*;


total_lossR��@

error_R	#G?

learning_rate_1�O?7�xqI       6%�	ݬ�X���A�*;


total_loss8��@

error_RD@?

learning_rate_1�O?7��I       6%�	��X���A�*;


total_loss���@

error_R�E?

learning_rate_1�O?7z25_I       6%�	u4�X���A�*;


total_loss`��@

error_RwP?

learning_rate_1�O?7P�C�I       6%�	"w�X���A�*;


total_loss�{�@

error_R6K?

learning_rate_1�O?7q0�$I       6%�	���X���A�*;


total_lossܡ�@

error_R��Z?

learning_rate_1�O?7�Ș/I       6%�	/�X���A�*;


total_lossVX�@

error_R�C?

learning_rate_1�O?7����I       6%�	+G�X���A�*;


total_loss��A

error_R$P?

learning_rate_1�O?7���rI       6%�	��X���A�*;


total_lossz|�@

error_R��@?

learning_rate_1�O?7���EI       6%�	֧X���A�*;


total_loss�K�@

error_Rvd[?

learning_rate_1�O?76�|I       6%�	T�X���A�*;


total_loss-�@

error_Rd�\?

learning_rate_1�O?7Ijt�I       6%�	gn�X���A�*;


total_loss���@

error_R�EX?

learning_rate_1�O?7%�T�I       6%�	���X���A�*;


total_loss�@

error_RO�D?

learning_rate_1�O?7��[_I       6%�	�	�X���A�*;


total_loss}�@

error_R�F?

learning_rate_1�O?7T��I       6%�	O�X���A�*;


total_loss$�@

error_R�\S?

learning_rate_1�O?7�M�/I       6%�	V��X���A�*;


total_loss�Z�@

error_R@�I?

learning_rate_1�O?7��,I       6%�	���X���A�*;


total_loss�[�@

error_R�BV?

learning_rate_1�O?7Ѫ��I       6%�	FB�X���A�*;


total_loss/u�@

error_RgM?

learning_rate_1�O?7��nSI       6%�	��X���A�*;


total_loss硌@

error_R7^?

learning_rate_1�O?7s|�HI       6%�	תX���A�*;


total_loss���@

error_R��L?

learning_rate_1�O?7�p�nI       6%�	�&�X���A�*;


total_lossq0�@

error_R�F?

learning_rate_1�O?7�4�I       6%�	~��X���A�*;


total_loss���@

error_R{�U?

learning_rate_1�O?7>�V�I       6%�	�ΫX���A�*;


total_loss�7�@

error_R�:?

learning_rate_1�O?7�R��I       6%�	��X���A�*;


total_loss���@

error_R?�R?

learning_rate_1�O?7J�G�I       6%�	_Z�X���A�*;


total_loss̨�@

error_R�W?

learning_rate_1�O?7��(I       6%�	���X���A�*;


total_loss/�@

error_R%�9?

learning_rate_1�O?7O6��I       6%�	���X���A�*;


total_loss�.�@

error_R��E?

learning_rate_1�O?7�{�mI       6%�	y5�X���A�*;


total_loss���@

error_R�>?

learning_rate_1�O?7�d�`I       6%�	�|�X���A�*;


total_loss�l�@

error_RN�j?

learning_rate_1�O?7�Y:QI       6%�	�íX���A�*;


total_loss}A�@

error_R�=D?

learning_rate_1�O?7�%��I       6%�	��X���A�*;


total_loss���@

error_R��I?

learning_rate_1�O?78�U$I       6%�	;O�X���A�*;


total_loss��@

error_RZ1G?

learning_rate_1�O?7\e�DI       6%�	?��X���A�*;


total_loss|�@

error_R�6D?

learning_rate_1�O?7z�F�I       6%�	WٮX���A�*;


total_losss�A

error_R�I?

learning_rate_1�O?7��vjI       6%�	�X���A�*;


total_loss`-�@

error_R]�V?

learning_rate_1�O?7I���I       6%�	�a�X���A�*;


total_losst2�@

error_R��M?

learning_rate_1�O?7���I       6%�	���X���A�*;


total_loss%��@

error_R<R?

learning_rate_1�O?7��eI       6%�	���X���A�*;


total_loss	��@

error_R�J?

learning_rate_1�O?7nȝzI       6%�	�<�X���A�*;


total_loss<��@

error_R� R?

learning_rate_1�O?7Kz�aI       6%�	%��X���A�*;


total_loss=tA

error_R\?

learning_rate_1�O?7��I       6%�	}ŰX���A�*;


total_loss�c�@

error_R�PQ?

learning_rate_1�O?7��M�I       6%�	��X���A�*;


total_lossF֠@

error_R)�R?

learning_rate_1�O?7u�I       6%�	eW�X���A�*;


total_lossԲ�@

error_R�!R?

learning_rate_1�O?7Pv�I       6%�	��X���A�*;


total_loss���@

error_R�OT?

learning_rate_1�O?7�lzMI       6%�	��X���A�*;


total_loss��@

error_Ri<V?

learning_rate_1�O?7��!I       6%�	'�X���A�*;


total_loss��@

error_R`M\?

learning_rate_1�O?7�~ZI       6%�	�n�X���A�*;


total_loss�ѓ@

error_R��E?

learning_rate_1�O?7N�R�I       6%�	H��X���A�*;


total_loss�!�@

error_RóD?

learning_rate_1�O?7��2�I       6%�	]�X���A�*;


total_lossZ�^@

error_RS	??

learning_rate_1�O?7AD�I       6%�	�M�X���A�*;


total_loss��@

error_RZ�i?

learning_rate_1�O?7���I       6%�	���X���A�*;


total_loss�~�@

error_RTbH?

learning_rate_1�O?7�dK2I       6%�	�׳X���A�*;


total_losss^�@

error_R.�M?

learning_rate_1�O?7�d�I       6%�	'�X���A�*;


total_loss�(�@

error_RƯI?

learning_rate_1�O?7���*I       6%�	?a�X���A�*;


total_loss�O�@

error_Rx�S?

learning_rate_1�O?7L\@I       6%�	᧴X���A�*;


total_lossk=�@

error_R/??

learning_rate_1�O?7�̜�I       6%�	��X���A�*;


total_loss/-�@

error_R&�E?

learning_rate_1�O?7��I       6%�	G6�X���A�*;


total_loss�>�@

error_RʞO?

learning_rate_1�O?7���I       6%�	͂�X���A�*;


total_loss&��@

error_R6<?

learning_rate_1�O?7��,�I       6%�	�ĵX���A�*;


total_loss@�@

error_RXe6?

learning_rate_1�O?7B��
I       6%�	?	�X���A�*;


total_loss�%�@

error_R�C?

learning_rate_1�O?7����I       6%�	M�X���A�*;


total_loss�~�@

error_REGQ?

learning_rate_1�O?7)��6I       6%�	3��X���A�*;


total_loss`[
A

error_RjrF?

learning_rate_1�O?7O}�I       6%�	�׶X���A�*;


total_loss��@

error_R}�Q?

learning_rate_1�O?7(��<I       6%�	��X���A�*;


total_loss�c�@

error_R��L?

learning_rate_1�O?7 50I       6%�	�b�X���A�*;


total_loss���@

error_RR*a?

learning_rate_1�O?7?kP�I       6%�	��X���A�*;


total_loss�PA

error_Rv�??

learning_rate_1�O?7˛:ZI       6%�	��X���A�*;


total_loss0:�@

error_R%�P?

learning_rate_1�O?7�h3�I       6%�	�-�X���A�*;


total_loss.e�@

error_R��P?

learning_rate_1�O?7"��5I       6%�	4n�X���A�*;


total_loss�$�@

error_R�R?

learning_rate_1�O?7k,>I       6%�	!��X���A�*;


total_loss3Y�@

error_RIP?

learning_rate_1�O?7���I       6%�	���X���A�*;


total_lossj�x@

error_RڐS?

learning_rate_1�O?7�{�I       6%�	Z<�X���A�*;


total_loss\|�@

error_R�FQ?

learning_rate_1�O?75�OgI       6%�	���X���A�*;


total_loss��@

error_R�o\?

learning_rate_1�O?7+g�I       6%�	ȹX���A�*;


total_lossc��@

error_R�]U?

learning_rate_1�O?7D��I       6%�	/�X���A�*;


total_losst��@

error_R�G?

learning_rate_1�O?7Jچ&I       6%�	�Q�X���A�*;


total_losse�@

error_RQ�W?

learning_rate_1�O?7E�fYI       6%�	���X���A�*;


total_loss\.�@

error_R;�M?

learning_rate_1�O?7�Y��I       6%�	ۺX���A�*;


total_loss䚔@

error_R��N?

learning_rate_1�O?73ߟ�I       6%�	�(�X���A�*;


total_lossi �@

error_R��C?

learning_rate_1�O?7�#(I       6%�	���X���A�*;


total_lossj��@

error_R�>?

learning_rate_1�O?7dƺI       6%�	�ϻX���A�*;


total_loss��@

error_R,�J?

learning_rate_1�O?7C��I       6%�	��X���A�*;


total_loss���@

error_R
�U?

learning_rate_1�O?7�	�I       6%�	�X�X���A�*;


total_loss�Ղ@

error_R��Q?

learning_rate_1�O?75��I       6%�	���X���A�*;


total_lossmo�@

error_R�YM?

learning_rate_1�O?7�BWQI       6%�	�X���A�*;


total_loss��A

error_Rs�D?

learning_rate_1�O?7�ĵI       6%�	#/�X���A�*;


total_lossi�@

error_R��N?

learning_rate_1�O?7�g�jI       6%�	:r�X���A�*;


total_loss���@

error_R�b?

learning_rate_1�O?7�Kh�I       6%�	���X���A�*;


total_loss���@

error_R%�V?

learning_rate_1�O?7.>�,I       6%�	��X���A�*;


total_loss��@

error_R�V?

learning_rate_1�O?7��?\I       6%�	t;�X���A�*;


total_loss���@

error_R\�N?

learning_rate_1�O?7���I       6%�	��X���A�*;


total_loss�5�@

error_R�~R?

learning_rate_1�O?7f�GlI       6%�	�þX���A�*;


total_lossΥA

error_R��I?

learning_rate_1�O?7��`I       6%�	��X���A�*;


total_losss%A

error_RCQ?

learning_rate_1�O?7���<I       6%�	?G�X���A�*;


total_loss���@

error_R�
j?

learning_rate_1�O?7�j��I       6%�	��X���A�*;


total_loss.��@

error_R6BN?

learning_rate_1�O?7d]I       6%�	�ܿX���A�*;


total_loss.��@

error_R��S?

learning_rate_1�O?7@2�I       6%�	�&�X���A�*;


total_loss�(�@

error_R�L?

learning_rate_1�O?7�B:I       6%�	�m�X���A�*;


total_loss��X@

error_R�'C?

learning_rate_1�O?7�\��I       6%�	M��X���A�*;


total_loss_�@

error_R�uZ?

learning_rate_1�O?79�*\I       6%�	���X���A�*;


total_loss�S�@

error_RŜR?

learning_rate_1�O?7��6I       6%�	�:�X���A�*;


total_loss��@

error_Rx�K?

learning_rate_1�O?7BbV@I       6%�	��X���A�*;


total_loss��@

error_RTPW?

learning_rate_1�O?7��|�I       6%�	���X���A�*;


total_loss�.A

error_R�a?

learning_rate_1�O?7����I       6%�	��X���A�*;


total_loss� �@

error_RTpU?

learning_rate_1�O?7Y�bOI       6%�	�\�X���A�*;


total_loss? �@

error_R>F?

learning_rate_1�O?7�s��I       6%�	��X���A�*;


total_loss�W�@

error_R��G?

learning_rate_1�O?7&/,I       6%�	���X���A�*;


total_lossabA

error_R!�I?

learning_rate_1�O?7����I       6%�	�.�X���A�*;


total_loss�ǀ@

error_R�97?

learning_rate_1�O?7)��I       6%�	�r�X���A�*;


total_loss �@

error_R�L?

learning_rate_1�O?7hx�I       6%�	���X���A�*;


total_lossW%�@

error_R��O?

learning_rate_1�O?7J��=I       6%�	���X���A�*;


total_loss/��@

error_R}�I?

learning_rate_1�O?7c@�I       6%�	;>�X���A�*;


total_lossVn�@

error_R#7:?

learning_rate_1�O?7ϩb�I       6%�	���X���A�*;


total_loss�}�@

error_R�-S?

learning_rate_1�O?7dm +I       6%�	���X���A�*;


total_loss�_�@

error_R�a?

learning_rate_1�O?7n�\I       6%�	h
�X���A�*;


total_loss�c�@

error_R1�J?

learning_rate_1�O?7ӣ�pI       6%�	DS�X���A�*;


total_lossHM�@

error_R�D?

learning_rate_1�O?7�PӒI       6%�	���X���A�*;


total_loss3}�@

error_R?JJ?

learning_rate_1�O?7S6��I       6%�	v��X���A�*;


total_loss$;x@

error_R��<?

learning_rate_1�O?7��fI       6%�	�+�X���A�*;


total_loss1��@

error_R��N?

learning_rate_1�O?7z��3I       6%�	�p�X���A�*;


total_loss���@

error_R�Z?

learning_rate_1�O?7=�bI       6%�	ٻ�X���A�*;


total_loss=1�@

error_R��W?

learning_rate_1�O?7��TI       6%�	}�X���A�*;


total_loss�
�@

error_Rv�8?

learning_rate_1�O?7lNi�I       6%�	�a�X���A�*;


total_loss��A

error_R2E]?

learning_rate_1�O?7�@�I       6%�	���X���A�*;


total_lossN$�@

error_R��V?

learning_rate_1�O?7�	�I       6%�	@��X���A�*;


total_loss؇�@

error_R��R?

learning_rate_1�O?7���I       6%�	�T�X���A�*;


total_loss_b�@

error_R ;X?

learning_rate_1�O?7[��I       6%�	ɠ�X���A�*;


total_loss��@

error_Ra�O?

learning_rate_1�O?7�ݾ�I       6%�	��X���A�*;


total_loss�R�@

error_R�Ea?

learning_rate_1�O?7��$�I       6%�	�8�X���A�*;


total_loss���@

error_R�~H?

learning_rate_1�O?7��@I       6%�	���X���A�*;


total_lossr�@

error_R6J?

learning_rate_1�O?7p�4;I       6%�	H��X���A�*;


total_lossLn�@

error_R��X?

learning_rate_1�O?7�*I       6%�	��X���A�*;


total_loss$��@

error_R��E?

learning_rate_1�O?727�gI       6%�	zV�X���A�*;


total_loss`t�@

error_R��7?

learning_rate_1�O?7�N�#I       6%�	$��X���A�*;


total_loss���@

error_R3N?

learning_rate_1�O?7�C�qI       6%�	���X���A�*;


total_loss�ˠ@

error_R�`S?

learning_rate_1�O?7�bI       6%�	Q7�X���A�*;


total_loss��@

error_R�i@?

learning_rate_1�O?7{��I       6%�	���X���A�*;


total_loss6N�@

error_R��A?

learning_rate_1�O?7V*ͫI       6%�	���X���A�*;


total_loss +�@

error_R�rF?

learning_rate_1�O?7�X�I       6%�	d$�X���A�*;


total_loss1{�@

error_Rxc?

learning_rate_1�O?7��YI       6%�	�j�X���A�*;


total_loss���@

error_R��B?

learning_rate_1�O?7?TGI       6%�	���X���A�*;


total_loss���@

error_R*:?

learning_rate_1�O?7
s�I       6%�	���X���A�*;


total_lossw��@

error_R1*L?

learning_rate_1�O?7:���I       6%�	M7�X���A�*;


total_lossWף@

error_R��H?

learning_rate_1�O?7���I       6%�	�z�X���A�*;


total_loss��o@

error_R�P?

learning_rate_1�O?7�VY�I       6%�	���X���A�*;


total_loss�_�@

error_R�|??

learning_rate_1�O?7�hI       6%�	�X���A�*;


total_loss={�@

error_R��W?

learning_rate_1�O?7$�$uI       6%�	N�X���A�*;


total_loss�?Y@

error_R��H?

learning_rate_1�O?7�%��I       6%�	e��X���A�*;


total_loss�6\@

error_RN?

learning_rate_1�O?7:m1�I       6%�	���X���A�*;


total_loss��@

error_RM(A?

learning_rate_1�O?7�x~BI       6%�	�X���A�*;


total_loss�C�@

error_R*�d?

learning_rate_1�O?7�(c[I       6%�	�h�X���A�*;


total_lossi(�@

error_R3�9?

learning_rate_1�O?7c_�I       6%�	��X���A�*;


total_lossz5�@

error_R�R:?

learning_rate_1�O?7�r�`I       6%�	}��X���A�*;


total_lossJA

error_R6�R?

learning_rate_1�O?7�qhI       6%�	z=�X���A�*;


total_loss���@

error_R��B?

learning_rate_1�O?7�sX>I       6%�	���X���A�*;


total_loss��@

error_R��??

learning_rate_1�O?7�(I       6%�	���X���A�*;


total_loss��@

error_Rk?

learning_rate_1�O?7�Xi�I       6%�	��X���A�*;


total_loss<��@

error_R)�B?

learning_rate_1�O?7��I       6%�	�K�X���A�*;


total_lossis�@

error_R&�Z?

learning_rate_1�O?7�؏9I       6%�	l��X���A�*;


total_lossSn�@

error_RqWT?

learning_rate_1�O?7v<l�I       6%�	���X���A�*;


total_lossS�A

error_RŊT?

learning_rate_1�O?7��q^I       6%�	��X���A�*;


total_lossJ�@

error_R�FL?

learning_rate_1�O?7E7^I       6%�	[�X���A�*;


total_loss䥏@

error_R��\?

learning_rate_1�O?7& �.I       6%�	'��X���A�*;


total_loss=ɞ@

error_R��^?

learning_rate_1�O?7���I       6%�	��X���A�*;


total_loss�ϝ@

error_R.�M?

learning_rate_1�O?7�P	�I       6%�	)�X���A�*;


total_lossC�@

error_R;Q?

learning_rate_1�O?7�_*�I       6%�	ll�X���A�*;


total_loss�Z�@

error_R
�P?

learning_rate_1�O?7K)�sI       6%�	���X���A�*;


total_loss`��@

error_RF�\?

learning_rate_1�O?7k���I       6%�	���X���A�*;


total_lossҧ�@

error_R��F?

learning_rate_1�O?7��~I       6%�	#7�X���A�*;


total_loss:3�@

error_R-bT?

learning_rate_1�O?7RV��I       6%�	�z�X���A�*;


total_loss�J�@

error_RjS?

learning_rate_1�O?7S�I       6%�	8��X���A�*;


total_lossx�@

error_R!I?

learning_rate_1�O?7ࡰ�I       6%�	�X���A�*;


total_loss�޷@

error_RȘR?

learning_rate_1�O?7d���I       6%�	�F�X���A�*;


total_lossF��@

error_R��`?

learning_rate_1�O?75�5]I       6%�	��X���A�*;


total_loss]I�@

error_R}#D?

learning_rate_1�O?7Pc��I       6%�	���X���A�*;


total_lossn�@

error_Rq�Y?

learning_rate_1�O?7�@�I       6%�	X�X���A�*;


total_loss�4�@

error_R��_?

learning_rate_1�O?7k�I       6%�	)h�X���A�*;


total_loss��a@

error_RCS4?

learning_rate_1�O?7��A#I       6%�	���X���A�*;


total_loss2��@

error_R�cU?

learning_rate_1�O?7��۷I       6%�	���X���A�*;


total_loss�C�@

error_R��@?

learning_rate_1�O?7�e��I       6%�	�7�X���A�*;


total_loss��@

error_R�zO?

learning_rate_1�O?70S�I       6%�	�{�X���A�*;


total_lossa@

error_R}�O?

learning_rate_1�O?7)(I       6%�	"��X���A�*;


total_loss4�@

error_R�V;?

learning_rate_1�O?7��mI       6%�	��X���A�*;


total_lossQ/�@

error_R8�Q?

learning_rate_1�O?7?,_�I       6%�	{0�X���A�*;


total_loss���@

error_Rr�b?

learning_rate_1�O?7V��I       6%�	Ӕ�X���A�*;


total_loss�o�@

error_R��[?

learning_rate_1�O?7Z2�I       6%�	���X���A�*;


total_loss! A

error_R��_?

learning_rate_1�O?7��}NI       6%�	*$�X���A�*;


total_loss�[�@

error_Rq�H?

learning_rate_1�O?7/ �rI       6%�	ai�X���A�*;


total_loss�i�@

error_R�L?

learning_rate_1�O?7�"�I       6%�	���X���A�*;


total_loss.�@

error_R$�K?

learning_rate_1�O?7f�v@I       6%�	u��X���A�*;


total_loss�p�@

error_R��J?

learning_rate_1�O?7��I       6%�	�=�X���A�*;


total_loss2�@

error_RZ?

learning_rate_1�O?7�宑I       6%�	���X���A�*;


total_lossӡ@

error_Rz�J?

learning_rate_1�O?7X��:I       6%�	���X���A�*;


total_loss�g�@

error_R�P?

learning_rate_1�O?7��I       6%�	�
�X���A�*;


total_loss4Í@

error_Rt�L?

learning_rate_1�O?7UeI       6%�	kP�X���A�*;


total_loss�Zw@

error_R�	Q?

learning_rate_1�O?7y� I       6%�	��X���A�*;


total_loss`��@

error_RTa?

learning_rate_1�O?7���I       6%�	���X���A�*;


total_lossэ�@

error_R�Y?

learning_rate_1�O?7��$I       6%�	��X���A�*;


total_loss��;A

error_R��R?

learning_rate_1�O?7\mv�I       6%�	]�X���A�*;


total_lossq��@

error_Rw�:?

learning_rate_1�O?7;��I       6%�	t��X���A�*;


total_loss��@

error_R�^S?

learning_rate_1�O?7B�:I       6%�	���X���A�*;


total_loss���@

error_R�/V?

learning_rate_1�O?7�b�HI       6%�	.�X���A�*;


total_loss��@

error_R&J?

learning_rate_1�O?7T� 3I       6%�	�n�X���A�*;


total_lossi^~@

error_R̯T?

learning_rate_1�O?7��ުI       6%�	���X���A�*;


total_loss�@

error_RM5I?

learning_rate_1�O?7uTI       6%�		��X���A�*;


total_lossÒ@

error_R3�G?

learning_rate_1�O?7�w�oI       6%�	24�X���A�*;


total_loss/I�@

error_R�gJ?

learning_rate_1�O?7���I       6%�	�z�X���A�*;


total_loss��@

error_R�Y>?

learning_rate_1�O?7	d��I       6%�	��X���A�*;


total_lossZ�@

error_R�N?

learning_rate_1�O?7Ԕ�I       6%�	��X���A�*;


total_loss�@

error_R�R?

learning_rate_1�O?7W]D[I       6%�	lE�X���A�*;


total_lossz��@

error_R�:?

learning_rate_1�O?7NIZI       6%�	%��X���A�*;


total_loss�L�@

error_R�R?

learning_rate_1�O?7w�7�I       6%�	���X���A�*;


total_loss<�@

error_R%58?

learning_rate_1�O?70��6I       6%�	��X���A�*;


total_loss&B�@

error_R��G?

learning_rate_1�O?7��/DI       6%�	�Q�X���A�*;


total_loss���@

error_RJ?

learning_rate_1�O?7�"nI       6%�	��X���A�*;


total_loss�i@

error_R�Zb?

learning_rate_1�O?7j4�qI       6%�	+��X���A�*;


total_loss�L�@

error_R&LS?

learning_rate_1�O?7W���I       6%�	��X���A�*;


total_loss�A�@

error_R��M?

learning_rate_1�O?7�	I       6%�	&\�X���A�*;


total_loss{'�@

error_RCnI?

learning_rate_1�O?7�9�I       6%�	l��X���A�*;


total_lossW�`@

error_R�Q8?

learning_rate_1�O?7�Z0I       6%�	���X���A�*;


total_lossl�d@

error_RԺA?

learning_rate_1�O?7�yP9I       6%�	,�X���A�*;


total_loss79�@

error_Ri�Q?

learning_rate_1�O?7_�xI       6%�	g��X���A�*;


total_loss:�@

error_R\A=?

learning_rate_1�O?7Z���I       6%�	r��X���A�*;


total_lossq��@

error_R�:G?

learning_rate_1�O?7����I       6%�	��X���A�*;


total_loss#�@

error_R�E?

learning_rate_1�O?7�3��I       6%�	Vi�X���A�*;


total_lossd��@

error_R��^?

learning_rate_1�O?7�KƬI       6%�	��X���A�*;


total_losss��@

error_R�Y?

learning_rate_1�O?7�-̹I       6%�	e�X���A�*;


total_loss(�@

error_R��E?

learning_rate_1�O?7�Rd@I       6%�	�b�X���A�*;


total_loss�ݣ@

error_RR�O?

learning_rate_1�O?7m�u�I       6%�	w��X���A�*;


total_loss.��@

error_R�Y?

learning_rate_1�O?7y��WI       6%�	� �X���A�*;


total_loss��@

error_R��T?

learning_rate_1�O?7�N��I       6%�	q�X���A�*;


total_loss�M�@

error_Rl_?

learning_rate_1�O?75XT�I       6%�	��X���A�*;


total_loss2��@

error_RON?

learning_rate_1�O?7���I       6%�	0�X���A�*;


total_loss���@

error_R�K\?

learning_rate_1�O?7�j<�I       6%�	u�X���A�*;


total_loss�5�@

error_R�Na?

learning_rate_1�O?7��I       6%�	���X���A�*;


total_lossa��@

error_RO&T?

learning_rate_1�O?7��*�I       6%�	{�X���A�*;


total_loss�;�@

error_R��M?

learning_rate_1�O?7OF��I       6%�	L�X���A�*;


total_loss�Y�@

error_R�QU?

learning_rate_1�O?7�8`�I       6%�	\��X���A�*;


total_loss2&�@

error_RA�Q?

learning_rate_1�O?7ӌE�I       6%�	�
�X���A�*;


total_loss���@

error_Rq�P?

learning_rate_1�O?7GN��I       6%�	�k�X���A�*;


total_loss�ּ@

error_R�
O?

learning_rate_1�O?7���I       6%�	e��X���A�*;


total_loss�S�@

error_R��L?

learning_rate_1�O?7[��*I       6%�	X�X���A�*;


total_loss���@

error_RTD?

learning_rate_1�O?7L/cnI       6%�	cJ�X���A�*;


total_loss�j�@

error_R�8?

learning_rate_1�O?7�^��I       6%�	r��X���A�*;


total_loss��@

error_R��N?

learning_rate_1�O?7��I       6%�	s��X���A�*;


total_loss�@

error_RKM?

learning_rate_1�O?7�I       6%�	[�X���A�*;


total_loss�2�@

error_R��@?

learning_rate_1�O?7A�]LI       6%�	�m�X���A�*;


total_loss���@

error_RȉU?

learning_rate_1�O?7'L¾I       6%�	$��X���A�*;


total_lossOɮ@

error_R1�E?

learning_rate_1�O?7�LS�I       6%�	@��X���A�*;


total_lossJ��@

error_R�]L?

learning_rate_1�O?7�5�I       6%�	{;�X���A�*;


total_loss��@

error_R}�7?

learning_rate_1�O?7�ȐVI       6%�	ׅ�X���A�*;


total_loss>��@

error_R��Q?

learning_rate_1�O?7K�a�I       6%�	���X���A�*;


total_loss�K�@

error_R�A?

learning_rate_1�O?7�v�I       6%�	��X���A�*;


total_lossh�A

error_R-�W?

learning_rate_1�O?7����I       6%�	2N�X���A�*;


total_loss*�@

error_R�|*?

learning_rate_1�O?7���I       6%�	0��X���A�*;


total_lossS��@

error_R�Q?

learning_rate_1�O?71��_I       6%�	���X���A�*;


total_lossR8�@

error_R.#Y?

learning_rate_1�O?7��0�I       6%�	�=�X���A�*;


total_lossQ��@

error_R��S?

learning_rate_1�O?7�h��I       6%�	&��X���A�*;


total_loss#��@

error_RFNA?

learning_rate_1�O?7ՙ�OI       6%�	���X���A�*;


total_loss���@

error_R�L?

learning_rate_1�O?7�kđI       6%�	�	�X���A�*;


total_lossi�@

error_R��;?

learning_rate_1�O?7�[\I       6%�	eV�X���A�*;


total_loss��@

error_R.�V?

learning_rate_1�O?7�k�I       6%�	ܞ�X���A�*;


total_loss ��@

error_R.BM?

learning_rate_1�O?7�׻I       6%�	`��X���A�*;


total_loss��@

error_RqRh?

learning_rate_1�O?7�X+1I       6%�	F1�X���A�*;


total_loss�	�@

error_RSL?

learning_rate_1�O?7��LI       6%�	�u�X���A�*;


total_loss�B�@

error_R�l8?

learning_rate_1�O?7mBX�I       6%�	���X���A�*;


total_loss@��@

error_R�9W?

learning_rate_1�O?7�8�"I       6%�	��X���A�*;


total_lossA��@

error_R&"K?

learning_rate_1�O?7�k��I       6%�	�L�X���A�*;


total_loss�T�@

error_R��H?

learning_rate_1�O?7C6^I       6%�	���X���A�*;


total_loss@

error_R��N?

learning_rate_1�O?7�}� I       6%�	��X���A�*;


total_lossY%A

error_RҦW?

learning_rate_1�O?7� ��I       6%�	#+�X���A�*;


total_loss�M�@

error_R�V?

learning_rate_1�O?7�7dI       6%�	�m�X���A�*;


total_loss?��@

error_R�4Y?

learning_rate_1�O?7���I       6%�	z��X���A�*;


total_loss���@

error_RH�W?

learning_rate_1�O?7�v��I       6%�	���X���A�*;


total_lossl�@

error_R��H?

learning_rate_1�O?7�|�I       6%�	<�X���A�*;


total_lossc��@

error_R$D?

learning_rate_1�O?7
{j�I       6%�	;�X���A�*;


total_lossc�@

error_R_�D?

learning_rate_1�O?7�9>�I       6%�	���X���A� *;


total_lossgA

error_R`�]?

learning_rate_1�O?7����I       6%�	-�X���A� *;


total_loss�{�@

error_R�M?

learning_rate_1�O?7��A�I       6%�	WJ�X���A� *;


total_lossBlA

error_RT�D?

learning_rate_1�O?7�v�]I       6%�	���X���A� *;


total_loss-�@

error_RV7O?

learning_rate_1�O?7����I       6%�	R��X���A� *;


total_losstHp@

error_R��I?

learning_rate_1�O?7���~I       6%�	��X���A� *;


total_loss_]�@

error_R�_M?

learning_rate_1�O?7$�|I       6%�	�Z�X���A� *;


total_loss�I�@

error_Rah@?

learning_rate_1�O?7bZI       6%�	ޞ�X���A� *;


total_loss���@

error_R�L?

learning_rate_1�O?7�؝I       6%�	���X���A� *;


total_lossM� A

error_R�([?

learning_rate_1�O?7�n�I       6%�	�&�X���A� *;


total_loss/%�@

error_Rq�6?

learning_rate_1�O?7z�gtI       6%�	s�X���A� *;


total_loss���@

error_RdI?

learning_rate_1�O?7,~�I       6%�	X��X���A� *;


total_loss
V�@

error_R_UL?

learning_rate_1�O?7v�!I       6%�	(�X���A� *;


total_loss!L�@

error_R,sX?

learning_rate_1�O?7��RMI       6%�	ET�X���A� *;


total_loss@d�@

error_R��N?

learning_rate_1�O?7�gQ
I       6%�	���X���A� *;


total_loss�М@

error_RJ�B?

learning_rate_1�O?7�fV
I       6%�	���X���A� *;


total_loss��@

error_R!�F?

learning_rate_1�O?7���hI       6%�	|%�X���A� *;


total_loss�^�@

error_R62A?

learning_rate_1�O?7� ��I       6%�	Og�X���A� *;


total_loss��@

error_Rq�S?

learning_rate_1�O?7&Q�I       6%�	ޫ�X���A� *;


total_loss7�@

error_R!%Y?

learning_rate_1�O?7��I       6%�	���X���A� *;


total_loss�ar@

error_R��G?

learning_rate_1�O?7:AI�I       6%�	`A�X���A� *;


total_lossZ��@

error_R`�O?

learning_rate_1�O?7hI       6%�	]��X���A� *;


total_loss�_�@

error_Rs:?

learning_rate_1�O?7_۸I       6%�	���X���A� *;


total_loss)��@

error_ROO?

learning_rate_1�O?7:�_?I       6%�	j0�X���A� *;


total_loss�~�@

error_RaET?

learning_rate_1�O?7~X\I       6%�	�v�X���A� *;


total_loss�\�@

error_R}�A?

learning_rate_1�O?787DI       6%�	���X���A� *;


total_loss.��@

error_R@T?

learning_rate_1�O?7m���I       6%�	B�X���A� *;


total_loss8��@

error_R�fO?

learning_rate_1�O?7�٪�I       6%�	bJ�X���A� *;


total_loss��@

error_RxXd?

learning_rate_1�O?7�[sRI       6%�	[��X���A� *;


total_loss`�@

error_R�jQ?

learning_rate_1�O?7��y>I       6%�	���X���A� *;


total_losscҾ@

error_R�W9?

learning_rate_1�O?7�&Q�I       6%�	:"�X���A� *;


total_lossH�y@

error_R�S?

learning_rate_1�O?7��I       6%�	�e�X���A� *;


total_loss�u�@

error_R%`I?

learning_rate_1�O?7�3�I       6%�	��X���A� *;


total_lossNƣ@

error_R|�D?

learning_rate_1�O?7�V��I       6%�	O��X���A� *;


total_loss�F�@

error_R[cO?

learning_rate_1�O?7U���I       6%�	,.�X���A� *;


total_loss:+�@

error_RS�D?

learning_rate_1�O?7����I       6%�	Tp�X���A� *;


total_lossrQ�@

error_R��C?

learning_rate_1�O?73ڨ�I       6%�	���X���A� *;


total_loss!G�@

error_R�=?

learning_rate_1�O?7A���I       6%�	���X���A� *;


total_loss��#A

error_R�X?

learning_rate_1�O?7\�vI       6%�	�5 Y���A� *;


total_lossZ9n@

error_R��N?

learning_rate_1�O?7����I       6%�	�w Y���A� *;


total_loss宮@

error_RZ�Y?

learning_rate_1�O?7e���I       6%�	M� Y���A� *;


total_loss��@

error_R�MK?

learning_rate_1�O?7��D�I       6%�	�� Y���A� *;


total_loss,a�@

error_R��M?

learning_rate_1�O?7����I       6%�	�>Y���A� *;


total_lossڡ�@

error_RBT?

learning_rate_1�O?7}��}I       6%�	�Y���A� *;


total_lossqp�@

error_R͛Y?

learning_rate_1�O?7X��I       6%�	�Y���A� *;


total_loss�@

error_R�9W?

learning_rate_1�O?7R�mBI       6%�	r	Y���A� *;


total_lossÕ@

error_R;�H?

learning_rate_1�O?71��3I       6%�	|NY���A� *;


total_loss�c�@

error_RT�R?

learning_rate_1�O?7Ŀ�I       6%�	ΏY���A� *;


total_loss�]�@

error_R;Q?

learning_rate_1�O?7��οI       6%�	��Y���A� *;


total_lossC�@

error_RaIZ?

learning_rate_1�O?7�W�I       6%�	�Y���A� *;


total_loss��@

error_R�uO?

learning_rate_1�O?7��
I       6%�	�XY���A� *;


total_loss�o�@

error_R}=W?

learning_rate_1�O?7�z�RI       6%�	֠Y���A� *;


total_loss���@

error_R!nQ?

learning_rate_1�O?7���I       6%�	��Y���A� *;


total_lossz��@

error_R%�D?

learning_rate_1�O?7�"�
I       6%�	�(Y���A� *;


total_loss5�@

error_RO�C?

learning_rate_1�O?7K�w\I       6%�	�jY���A� *;


total_loss�1�@

error_R�xI?

learning_rate_1�O?7��uI       6%�	��Y���A� *;


total_loss�2�@

error_RY?

learning_rate_1�O?7Q�L~I       6%�	��Y���A� *;


total_loss�@

error_R.'S?

learning_rate_1�O?7�^9I       6%�	�7Y���A� *;


total_loss �@

error_R
�>?

learning_rate_1�O?7����I       6%�	�~Y���A� *;


total_losss�@

error_RL?

learning_rate_1�O?7�z �I       6%�	��Y���A� *;


total_loss{�@

error_Rכ<?

learning_rate_1�O?7u�I       6%�	�	Y���A� *;


total_loss{A�@

error_RvxE?

learning_rate_1�O?7���I       6%�	NY���A� *;


total_loss��A

error_R�|W?

learning_rate_1�O?7N>XI       6%�	u�Y���A� *;


total_loss��A

error_R[S?

learning_rate_1�O?7s8�I       6%�	s�Y���A� *;


total_loss.p�@

error_R��F?

learning_rate_1�O?7a�_�I       6%�	XY���A� *;


total_loss<��@

error_RL?

learning_rate_1�O?7��I       6%�	�^Y���A� *;


total_lossϕ�@

error_R� D?

learning_rate_1�O?7���I       6%�	��Y���A� *;


total_loss�n�@

error_R��Y?

learning_rate_1�O?7����I       6%�	�Y���A� *;


total_loss�:�@

error_R;�R?

learning_rate_1�O?73���I       6%�	z%Y���A� *;


total_loss�K�@

error_R�W?

learning_rate_1�O?7�ۂI       6%�	=gY���A� *;


total_loss�ݠ@

error_RqB??

learning_rate_1�O?7�r�I       6%�	��Y���A� *;


total_loss�}�@

error_R!�B?

learning_rate_1�O?7��5�I       6%�	k�Y���A� *;


total_loss���@

error_R��Z?

learning_rate_1�O?7����I       6%�	�5	Y���A� *;


total_loss\_@

error_RO&I?

learning_rate_1�O?7�PA�I       6%�	x	Y���A� *;


total_loss6��@

error_R�W?

learning_rate_1�O?7եu2I       6%�	.�	Y���A� *;


total_loss>(�@

error_R@ F?

learning_rate_1�O?7�7��I       6%�	l
Y���A� *;


total_loss�`�@

error_RE?

learning_rate_1�O?7���I       6%�	�G
Y���A� *;


total_loss(q�@

error_R��U?

learning_rate_1�O?7���I       6%�	Ŏ
Y���A� *;


total_loss�7�@

error_RZ�D?

learning_rate_1�O?7E���I       6%�	�
Y���A� *;


total_lossO��@

error_RJ�\?

learning_rate_1�O?7[�2�I       6%�	j!Y���A� *;


total_loss���@

error_R�l]?

learning_rate_1�O?7&g��I       6%�	��Y���A� *;


total_loss�ú@

error_R�C?

learning_rate_1�O?734*�I       6%�	��Y���A� *;


total_loss���@

error_R��@?

learning_rate_1�O?7W��I       6%�	/HY���A� *;


total_loss<��@

error_R��L?

learning_rate_1�O?7��wI       6%�	x�Y���A� *;


total_loss�Q�@

error_R\�V?

learning_rate_1�O?7�V�7I       6%�	��Y���A� *;


total_lossG4�@

error_Rc�9?

learning_rate_1�O?7���I       6%�	�Y���A� *;


total_loss��@

error_RF?

learning_rate_1�O?7�b��I       6%�	�|Y���A� *;


total_loss�>�@

error_RH�W?

learning_rate_1�O?7��I       6%�	��Y���A� *;


total_loss�@c@

error_R:�E?

learning_rate_1�O?7 ]�I       6%�	�Y���A� *;


total_loss���@

error_Rc�6?

learning_rate_1�O?7.2

I       6%�	DMY���A� *;


total_loss�B�@

error_R3)R?

learning_rate_1�O?7^iedI       6%�	ݔY���A� *;


total_loss*�f@

error_Ri�;?

learning_rate_1�O?7�qI       6%�	��Y���A� *;


total_loss�F�@

error_R��E?

learning_rate_1�O?7hS�I       6%�	&!Y���A� *;


total_lossFi�@

error_R/�<?

learning_rate_1�O?7���I       6%�	zfY���A� *;


total_loss�HA

error_R��N?

learning_rate_1�O?7P��I       6%�	+�Y���A� *;


total_loss���@

error_Rϓ[?

learning_rate_1�O?7��G�I       6%�	h�Y���A� *;


total_lossJ׸@

error_R�BL?

learning_rate_1�O?7ߗIDI       6%�	�8Y���A� *;


total_losseў@

error_R�Nf?

learning_rate_1�O?7GwNI       6%�	I}Y���A� *;


total_loss8�o@

error_R�O?

learning_rate_1�O?7:�I       6%�	b�Y���A� *;


total_loss*!�@

error_R�>?

learning_rate_1�O?7�4 �I       6%�	o	Y���A� *;


total_loss���@

error_R��G?

learning_rate_1�O?7�� I       6%�	/RY���A� *;


total_loss�ԃ@

error_R��A?

learning_rate_1�O?7KPgI       6%�	�Y���A� *;


total_loss�`�@

error_R�LM?

learning_rate_1�O?7W6��I       6%�	�Y���A� *;


total_loss��@

error_R�\U?

learning_rate_1�O?7��I       6%�	�!Y���A� *;


total_loss�'�@

error_Rm5O?

learning_rate_1�O?7"��I       6%�	�bY���A� *;


total_loss�A

error_R"f?

learning_rate_1�O?7�4�I       6%�	��Y���A� *;


total_loss�ޠ@

error_Rw�??

learning_rate_1�O?7��9RI       6%�	��Y���A� *;


total_lossZp�@

error_R��G?

learning_rate_1�O?7�^��I       6%�	�'Y���A� *;


total_loss�@

error_R��L?

learning_rate_1�O?7r�PI       6%�	�kY���A� *;


total_loss���@

error_R�nX?

learning_rate_1�O?7�F�I       6%�	�Y���A� *;


total_loss�A

error_R��]?

learning_rate_1�O?7}T[)I       6%�	C�Y���A� *;


total_loss-K�@

error_R��O?

learning_rate_1�O?77C6tI       6%�	HEY���A� *;


total_loss��@

error_RtpA?

learning_rate_1�O?7/bt+I       6%�	\�Y���A� *;


total_loss}#�@

error_R&�N?

learning_rate_1�O?7[�3�I       6%�	�Y���A� *;


total_loss�/�@

error_R�P,?

learning_rate_1�O?7?x�I       6%�	EY���A� *;


total_loss�+�@

error_R�jK?

learning_rate_1�O?7ISq�I       6%�	YY���A� *;


total_loss!V�@

error_R;�`?

learning_rate_1�O?7�K��I       6%�	��Y���A� *;


total_lossO�@

error_R��H?

learning_rate_1�O?7Ԛ��I       6%�	P�Y���A� *;


total_loss�<~@

error_ROUW?

learning_rate_1�O?7��m?I       6%�	�#Y���A� *;


total_lossy�@

error_R��[?

learning_rate_1�O?7+7;I       6%�	1jY���A� *;


total_loss�T�@

error_R��=?

learning_rate_1�O?7���I       6%�	��Y���A� *;


total_lossmH�@

error_R4aH?

learning_rate_1�O?7��3I       6%�	}�Y���A� *;


total_lossᤁ@

error_Rm�@?

learning_rate_1�O?7�*lI       6%�	�7Y���A� *;


total_loss��@

error_RRLV?

learning_rate_1�O?7�>�PI       6%�	=~Y���A� *;


total_lossꫴ@

error_R�C?

learning_rate_1�O?7>FR�I       6%�	��Y���A� *;


total_loss�O�@

error_RʨN?

learning_rate_1�O?7N�I       6%�	2Y���A� *;


total_loss���@

error_Rs�R?

learning_rate_1�O?7��~I       6%�	�EY���A� *;


total_loss	�@

error_R$�X?

learning_rate_1�O?7p+�bI       6%�	e�Y���A� *;


total_loss}��@

error_R��_?

learning_rate_1�O?7FcI       6%�	G�Y���A�!*;


total_loss���@

error_R�wY?

learning_rate_1�O?7s�	uI       6%�	�Y���A�!*;


total_loss�@

error_Ra�g?

learning_rate_1�O?7��r�I       6%�	�SY���A�!*;


total_lossh5�@

error_R��F?

learning_rate_1�O?7v@w�I       6%�	�Y���A�!*;


total_loss��@

error_R��D?

learning_rate_1�O?70��I       6%�	��Y���A�!*;


total_loss�s�@

error_R�^a?

learning_rate_1�O?7����I       6%�	�$Y���A�!*;


total_lossJ��@

error_R
T?

learning_rate_1�O?7�1*�I       6%�	yjY���A�!*;


total_loss�l�@

error_R�bM?

learning_rate_1�O?7�m2�I       6%�	��Y���A�!*;


total_loss	ް@

error_RMzG?

learning_rate_1�O?7<��%I       6%�	k�Y���A�!*;


total_lossZ�F@

error_R��R?

learning_rate_1�O?7�>.I       6%�	*XY���A�!*;


total_loss��@

error_RWZ?

learning_rate_1�O?7��I       6%�	�Y���A�!*;


total_loss��@

error_R��G?

learning_rate_1�O?7~��I       6%�	��Y���A�!*;


total_loss�y@

error_R6�h?

learning_rate_1�O?7.j�I       6%�	�@Y���A�!*;


total_lossV(�@

error_R�pT?

learning_rate_1�O?7���[I       6%�	�Y���A�!*;


total_lossE\�@

error_R�HT?

learning_rate_1�O?7�FzI       6%�	��Y���A�!*;


total_lossa��@

error_R�:S?

learning_rate_1�O?7���I       6%�	 Y���A�!*;


total_loss�K�@

error_R�YN?

learning_rate_1�O?7�^�I       6%�	�`Y���A�!*;


total_loss89�@

error_R�W?

learning_rate_1�O?7���UI       6%�	=�Y���A�!*;


total_lossI�@

error_R�iN?

learning_rate_1�O?7"	e,I       6%�	$�Y���A�!*;


total_loss�z�@

error_RaZ?

learning_rate_1�O?7�ӛI       6%�	�9Y���A�!*;


total_loss. �@

error_R�6T?

learning_rate_1�O?7t�6nI       6%�	Q�Y���A�!*;


total_losss&�@

error_R��<?

learning_rate_1�O?7e��I       6%�	��Y���A�!*;


total_loss�:�@

error_R�d?

learning_rate_1�O?7+ݳ�I       6%�	�Y���A�!*;


total_loss8��@

error_R�D?

learning_rate_1�O?7�*ήI       6%�	�TY���A�!*;


total_loss�6�@

error_RJL?

learning_rate_1�O?7g'��I       6%�	��Y���A�!*;


total_loss��@

error_R�:G?

learning_rate_1�O?7�OwKI       6%�	��Y���A�!*;


total_loss\��@

error_R�q?

learning_rate_1�O?7��I       6%�	% Y���A�!*;


total_lossNl�@

error_R��V?

learning_rate_1�O?7�j�"I       6%�	g Y���A�!*;


total_lossv)�@

error_R�H?

learning_rate_1�O?7�-�I       6%�	w� Y���A�!*;


total_loss�r�@

error_R NJ?

learning_rate_1�O?7��g*I       6%�	� Y���A�!*;


total_lossF,l@

error_R��N?

learning_rate_1�O?7�\II       6%�	�0!Y���A�!*;


total_loss�@

error_Ri!_?

learning_rate_1�O?7��I       6%�	�u!Y���A�!*;


total_loss/��@

error_R��=?

learning_rate_1�O?7dd<I       6%�	��!Y���A�!*;


total_loss���@

error_R�aW?

learning_rate_1�O?7��}=I       6%�	U�!Y���A�!*;


total_loss���@

error_R6,A?

learning_rate_1�O?7��ٹI       6%�	�A"Y���A�!*;


total_loss[��@

error_R_�]?

learning_rate_1�O?7�{��I       6%�	t�"Y���A�!*;


total_lossT��@

error_R��E?

learning_rate_1�O?7s1a�I       6%�	7�"Y���A�!*;


total_loss!�A

error_R�3H?

learning_rate_1�O?7e��I       6%�		#Y���A�!*;


total_loss�0�@

error_R�U?

learning_rate_1�O?7���:I       6%�	�H#Y���A�!*;


total_loss=g�@

error_R;�L?

learning_rate_1�O?78m;YI       6%�	|�#Y���A�!*;


total_lossXr�@

error_R(l^?

learning_rate_1�O?7��4aI       6%�	3�#Y���A�!*;


total_loss��@

error_R1�:?

learning_rate_1�O?7F�%AI       6%�	}$Y���A�!*;


total_loss���@

error_R�._?

learning_rate_1�O?7Jl.�I       6%�	�g$Y���A�!*;


total_loss��@

error_R�jV?

learning_rate_1�O?7F:�I       6%�	=�$Y���A�!*;


total_lossԑ�@

error_R�Yb?

learning_rate_1�O?7����I       6%�	��$Y���A�!*;


total_loss[�@

error_R�Ca?

learning_rate_1�O?7h^H�I       6%�	�;%Y���A�!*;


total_lossE%�@

error_R��U?

learning_rate_1�O?7�XI       6%�	u�%Y���A�!*;


total_loss��@

error_R�/J?

learning_rate_1�O?7�aI       6%�	��%Y���A�!*;


total_loss���@

error_R_`?

learning_rate_1�O?7��q�I       6%�	�&Y���A�!*;


total_lossō@

error_R��`?

learning_rate_1�O?7�o�%I       6%�	PX&Y���A�!*;


total_lossS�@

error_R7�S?

learning_rate_1�O?7���I       6%�	�&Y���A�!*;


total_loss���@

error_R =X?

learning_rate_1�O?7�VW8I       6%�	��&Y���A�!*;


total_loss��@

error_R�sL?

learning_rate_1�O?7���I       6%�	� 'Y���A�!*;


total_losse�@

error_RMx[?

learning_rate_1�O?7�vyI       6%�	ja'Y���A�!*;


total_loss�f#A

error_R��V?

learning_rate_1�O?7/y�@I       6%�	��'Y���A�!*;


total_loss���@

error_RҾH?

learning_rate_1�O?7 �#:I       6%�	Y�'Y���A�!*;


total_lossM�@

error_R��H?

learning_rate_1�O?7{��I       6%�	)(Y���A�!*;


total_loss��@

error_R�[?

learning_rate_1�O?7���MI       6%�	�r(Y���A�!*;


total_loss��@

error_RC�S?

learning_rate_1�O?7�i+�I       6%�	t�(Y���A�!*;


total_lossc`�@

error_R�CQ?

learning_rate_1�O?7{>�I       6%�	y )Y���A�!*;


total_lossȀ�@

error_R��U?

learning_rate_1�O?7�"u^I       6%�	�L)Y���A�!*;


total_loss�p�@

error_R�I?

learning_rate_1�O?7Yf�I       6%�	R�)Y���A�!*;


total_loss��@

error_R�JL?

learning_rate_1�O?7z��I       6%�	D�)Y���A�!*;


total_loss�z�@

error_Rx�M?

learning_rate_1�O?7�f��I       6%�	*Y���A�!*;


total_loss��@

error_R��_?

learning_rate_1�O?7?ޕ=I       6%�	�b*Y���A�!*;


total_loss��A

error_RD:a?

learning_rate_1�O?7���{I       6%�	��*Y���A�!*;


total_loss�A

error_R\�G?

learning_rate_1�O?7��K�I       6%�	��*Y���A�!*;


total_loss4q�@

error_R��B?

learning_rate_1�O?7�`z�I       6%�	�I+Y���A�!*;


total_lossW��@

error_R*O?

learning_rate_1�O?7UŷI       6%�	­+Y���A�!*;


total_lossoT�@

error_RJ�;?

learning_rate_1�O?7]�qI       6%�	��+Y���A�!*;


total_loss3�@

error_R�>W?

learning_rate_1�O?7&��4I       6%�	�V,Y���A�!*;


total_loss��@

error_R=�N?

learning_rate_1�O?7���I       6%�	�,Y���A�!*;


total_loss;��@

error_Rf@?

learning_rate_1�O?7��kI       6%�	��,Y���A�!*;


total_loss���@

error_R�>L?

learning_rate_1�O?7j4s�I       6%�	�C-Y���A�!*;


total_loss@"�@

error_RhW?

learning_rate_1�O?7p���I       6%�	��-Y���A�!*;


total_loss��@

error_R�!B?

learning_rate_1�O?7tc�I       6%�	��-Y���A�!*;


total_loss7��@

error_R�%j?

learning_rate_1�O?7�SRcI       6%�	�.Y���A�!*;


total_loss���@

error_R3�D?

learning_rate_1�O?7���I       6%�	B\.Y���A�!*;


total_lossn��@

error_R�SN?

learning_rate_1�O?7!H��I       6%�	��.Y���A�!*;


total_loss)Q�@

error_R�IF?

learning_rate_1�O?70P��I       6%�	��.Y���A�!*;


total_loss��@

error_RS�C?

learning_rate_1�O?7%δ�I       6%�	�+/Y���A�!*;


total_lossꖹ@

error_R.�i?

learning_rate_1�O?7��`|I       6%�	�q/Y���A�!*;


total_loss�]@

error_R�NQ?

learning_rate_1�O?7ر� I       6%�	<�/Y���A�!*;


total_loss��@

error_R�UI?

learning_rate_1�O?7mo3�I       6%�	��/Y���A�!*;


total_lossѣ�@

error_R��a?

learning_rate_1�O?7{�#I       6%�	YD0Y���A�!*;


total_loss_��@

error_R�I?

learning_rate_1�O?7�E&�I       6%�	M�0Y���A�!*;


total_loss��A

error_R�:?

learning_rate_1�O?7�_�I       6%�	�0Y���A�!*;


total_loss�P�@

error_R�@?

learning_rate_1�O?7͕�I       6%�	T1Y���A�!*;


total_loss)+�@

error_R�.9?

learning_rate_1�O?7(2FI       6%�	]_1Y���A�!*;


total_loss���@

error_R�h8?

learning_rate_1�O?7j(GI       6%�	��1Y���A�!*;


total_loss�t@

error_R,�D?

learning_rate_1�O?7�ZT|I       6%�	d�1Y���A�!*;


total_loss|�@

error_R�&K?

learning_rate_1�O?7��}I       6%�	W42Y���A�!*;


total_losss�
A

error_R��[?

learning_rate_1�O?7���I       6%�	�}2Y���A�!*;


total_loss��@

error_R�W?

learning_rate_1�O?7���I       6%�	��2Y���A�!*;


total_loss��@

error_RHJ?

learning_rate_1�O?7h��I       6%�	�3Y���A�!*;


total_loss9+�@

error_R�I?

learning_rate_1�O?7��I       6%�	�U3Y���A�!*;


total_loss�D�@

error_R1�X?

learning_rate_1�O?7���I       6%�	��3Y���A�!*;


total_loss�ǧ@

error_R��<?

learning_rate_1�O?7���I       6%�	_�3Y���A�!*;


total_lossR�@

error_R�$Q?

learning_rate_1�O?7����I       6%�	�%4Y���A�!*;


total_loss��@

error_R�b?

learning_rate_1�O?7�ؘWI       6%�	�f4Y���A�!*;


total_loss��@

error_ReVT?

learning_rate_1�O?7:�!I       6%�		�4Y���A�!*;


total_lossX1�@

error_R�,V?

learning_rate_1�O?7h�B6I       6%�	,�4Y���A�!*;


total_loss�~�@

error_R�J?

learning_rate_1�O?7Jo�I       6%�	�.5Y���A�!*;


total_loss3�z@

error_R�O?

learning_rate_1�O?7.d��I       6%�	ks5Y���A�!*;


total_loss#^�@

error_R�_?

learning_rate_1�O?7ZmnI       6%�	(�5Y���A�!*;


total_loss��@

error_R�nN?

learning_rate_1�O?7�c�I       6%�	��5Y���A�!*;


total_loss�D�@

error_R��??

learning_rate_1�O?7�̟�I       6%�	YD6Y���A�!*;


total_loss��@

error_R��K?

learning_rate_1�O?7�	VI       6%�	��6Y���A�!*;


total_loss��@

error_R/�M?

learning_rate_1�O?7�<�I       6%�	k�6Y���A�!*;


total_loss��@

error_R�H?

learning_rate_1�O?70��I       6%�	�7Y���A�!*;


total_loss���@

error_R��X?

learning_rate_1�O?7���{I       6%�	�T7Y���A�!*;


total_loss�@

error_R�j_?

learning_rate_1�O?7����I       6%�	5�7Y���A�!*;


total_loss;��@

error_R�OH?

learning_rate_1�O?7�:dnI       6%�	��7Y���A�!*;


total_loss-�@

error_R� H?

learning_rate_1�O?7F ��I       6%�	�$8Y���A�!*;


total_loss�@

error_R�mB?

learning_rate_1�O?7��?�I       6%�	�l8Y���A�!*;


total_loss!��@

error_Rϋa?

learning_rate_1�O?7��-I       6%�	]�8Y���A�!*;


total_loss��@

error_RkT?

learning_rate_1�O?7�PUPI       6%�	��8Y���A�!*;


total_loss���@

error_R�vL?

learning_rate_1�O?7��`I       6%�	�h9Y���A�!*;


total_lossl��@

error_R�O?

learning_rate_1�O?7
:�I       6%�	#�9Y���A�!*;


total_loss��@

error_R��M?

learning_rate_1�O?7[��I       6%�	�:Y���A�!*;


total_loss�j�@

error_R�h>?

learning_rate_1�O?7��+�I       6%�	tk:Y���A�!*;


total_loss8��@

error_RT�G?

learning_rate_1�O?7H�vI       6%�	i�:Y���A�!*;


total_loss�\�@

error_RL,A?

learning_rate_1�O?7�g�II       6%�	�+;Y���A�!*;


total_loss]��@

error_R��G?

learning_rate_1�O?7�cI       6%�	��;Y���A�!*;


total_loss�?�@

error_R6tQ?

learning_rate_1�O?7���RI       6%�	X�;Y���A�!*;


total_loss ��@

error_R
tL?

learning_rate_1�O?7���I       6%�	v<Y���A�!*;


total_lossa�@

error_RW�T?

learning_rate_1�O?7L�I       6%�	`b<Y���A�!*;


total_loss2��@

error_R��C?

learning_rate_1�O?7��u�I       6%�	+�<Y���A�!*;


total_loss�@

error_R7k>?

learning_rate_1�O?7l��>I       6%�	��<Y���A�"*;


total_loss��@

error_R�gK?

learning_rate_1�O?7����I       6%�	�2=Y���A�"*;


total_loss�ۆ@

error_Rf1P?

learning_rate_1�O?7Пu�I       6%�	{=Y���A�"*;


total_lossz��@

error_R-�J?

learning_rate_1�O?7����I       6%�	N�=Y���A�"*;


total_loss��A

error_R��E?

learning_rate_1�O?7^�/�I       6%�	�>Y���A�"*;


total_loss$F�@

error_RC�I?

learning_rate_1�O?7��߫I       6%�	�w>Y���A�"*;


total_lossA��@

error_RJ�O?

learning_rate_1�O?7�l9?I       6%�	�>Y���A�"*;


total_loss���@

error_R B@?

learning_rate_1�O?7߬dgI       6%�	k?Y���A�"*;


total_loss��@

error_R�T?

learning_rate_1�O?7O|�I       6%�	�Y?Y���A�"*;


total_loss�%�@

error_R$jP?

learning_rate_1�O?7|AW�I       6%�	�?Y���A�"*;


total_loss��@

error_R�3G?

learning_rate_1�O?7bPEI       6%�	��?Y���A�"*;


total_lossL�@

error_R��T?

learning_rate_1�O?7ɮ%I       6%�	�1@Y���A�"*;


total_loss嶼@

error_R�`N?

learning_rate_1�O?7ϸ1�I       6%�	�u@Y���A�"*;


total_lossc�@

error_R}W?

learning_rate_1�O?7��q�I       6%�	��@Y���A�"*;


total_loss�F�@

error_R��V?

learning_rate_1�O?7��CI       6%�	AY���A�"*;


total_loss7��@

error_R�W?

learning_rate_1�O?7�)��I       6%�	�YAY���A�"*;


total_loss���@

error_RM�J?

learning_rate_1�O?7�4H�I       6%�	��AY���A�"*;


total_loss(Ö@

error_R(�T?

learning_rate_1�O?7�g�I       6%�	��AY���A�"*;


total_loss���@

error_R�)J?

learning_rate_1�O?7���I       6%�	/7BY���A�"*;


total_loss��@

error_RQpP?

learning_rate_1�O?7QDT8I       6%�	yBY���A�"*;


total_loss���@

error_RTlC?

learning_rate_1�O?7y*�I       6%�	�BY���A�"*;


total_loss�nA

error_R�1a?

learning_rate_1�O?7�Dm�I       6%�	TCY���A�"*;


total_loss���@

error_R�S?

learning_rate_1�O?7�fEnI       6%�	�HCY���A�"*;


total_loss�*�@

error_R�d?

learning_rate_1�O?7���eI       6%�	��CY���A�"*;


total_loss�]�@

error_R�%e?

learning_rate_1�O?7I       6%�	0�CY���A�"*;


total_loss���@

error_R`,O?

learning_rate_1�O?7��Q�I       6%�	VDY���A�"*;


total_loss��A

error_R��R?

learning_rate_1�O?7׸�,I       6%�	x^DY���A�"*;


total_loss��@

error_R�C?

learning_rate_1�O?7��aYI       6%�	}�DY���A�"*;


total_loss�;�@

error_R2�R?

learning_rate_1�O?7�WEI       6%�	��DY���A�"*;


total_lossZ��@

error_R�/?

learning_rate_1�O?7K<��I       6%�	�4EY���A�"*;


total_loss8�@

error_R��1?

learning_rate_1�O?7/�I�I       6%�	QzEY���A�"*;


total_loss�#�@

error_R�_S?

learning_rate_1�O?7���I       6%�	=�EY���A�"*;


total_loss�ϖ@

error_R;8?

learning_rate_1�O?7�;GI       6%�	(	FY���A�"*;


total_loss�@

error_RZ>?

learning_rate_1�O?7��I       6%�	QNFY���A�"*;


total_lossԷ@

error_Rf�L?

learning_rate_1�O?7ㄹ~I       6%�	G�FY���A�"*;


total_lossV4�@

error_R��Y?

learning_rate_1�O?7�|܊I       6%�	l�FY���A�"*;


total_loss7Q�@

error_R�X?

learning_rate_1�O?77�XI       6%�	�.GY���A�"*;


total_loss�G�@

error_R��F?

learning_rate_1�O?7Z{�I       6%�	�sGY���A�"*;


total_lossd�@

error_R�QO?

learning_rate_1�O?7qeo�I       6%�	�GY���A�"*;


total_lossT��@

error_R�J?

learning_rate_1�O?7�;��I       6%�	HY���A�"*;


total_loss�@

error_RȸK?

learning_rate_1�O?77�8I       6%�	kMHY���A�"*;


total_loss�4�@

error_R�bT?

learning_rate_1�O?7\J��I       6%�	`�HY���A�"*;


total_loss���@

error_RKM?

learning_rate_1�O?7y�YI       6%�	�HY���A�"*;


total_loss�k�@

error_R�3??

learning_rate_1�O?7�I       6%�	cIY���A�"*;


total_loss�XA

error_R��N?

learning_rate_1�O?7yE��I       6%�	�_IY���A�"*;


total_loss�8z@

error_R�'W?

learning_rate_1�O?7���I       6%�	�IY���A�"*;


total_loss���@

error_R�_T?

learning_rate_1�O?7u�J�I       6%�	[�IY���A�"*;


total_loss��@

error_Rq�L?

learning_rate_1�O?7��6I       6%�	�@JY���A�"*;


total_loss�|@

error_R��A?

learning_rate_1�O?7x�9I       6%�	K�JY���A�"*;


total_loss:�A

error_R��U?

learning_rate_1�O?7��A,I       6%�	A�JY���A�"*;


total_loss ]@

error_R}/7?

learning_rate_1�O?7�N��I       6%�	�KY���A�"*;


total_loss	��@

error_RT�J?

learning_rate_1�O?7��5VI       6%�	.kKY���A�"*;


total_loss7҇@

error_R��D?

learning_rate_1�O?7��̐I       6%�	�KY���A�"*;


total_loss��@

error_RSUR?

learning_rate_1�O?7�r�I       6%�	�KY���A�"*;


total_lossE��@

error_Rf�D?

learning_rate_1�O?7BN�I       6%�	4FLY���A�"*;


total_loss�@

error_R\3S?

learning_rate_1�O?7SB�I       6%�	�LY���A�"*;


total_loss�h�@

error_R�TM?

learning_rate_1�O?7/�
I       6%�	%�LY���A�"*;


total_loss�+�@

error_Rl�;?

learning_rate_1�O?7���I       6%�	�MY���A�"*;


total_loss���@

error_Rx�F?

learning_rate_1�O?7�GM5I       6%�	�WMY���A�"*;


total_loss��@

error_R��i?

learning_rate_1�O?7���I       6%�	z�MY���A�"*;


total_loss��@

error_R�/Z?

learning_rate_1�O?7{�}"I       6%�	`�MY���A�"*;


total_loss���@

error_RיR?

learning_rate_1�O?75̘EI       6%�	�#NY���A�"*;


total_loss���@

error_R�	M?

learning_rate_1�O?7 5w"I       6%�		kNY���A�"*;


total_loss���@

error_R�MO?

learning_rate_1�O?7�m(I       6%�	�NY���A�"*;


total_loss^�A

error_R�:G?

learning_rate_1�O?7�`R�I       6%�	�NY���A�"*;


total_loss�L�@

error_R\�J?

learning_rate_1�O?73��I       6%�	pEOY���A�"*;


total_loss��m@

error_R��=?

learning_rate_1�O?7` �I       6%�	ϊOY���A�"*;


total_loss�I�@

error_Raf?

learning_rate_1�O?7�aI       6%�	e�OY���A�"*;


total_loss�)�@

error_RWE:?

learning_rate_1�O?7�lI       6%�	�PY���A�"*;


total_loss ��@

error_R�J?

learning_rate_1�O?7�I�I       6%�	�TPY���A�"*;


total_lossʿ�@

error_R�[:?

learning_rate_1�O?7)E uI       6%�	Z�PY���A�"*;


total_loss�ڡ@

error_Rw�V?

learning_rate_1�O?7�>ViI       6%�	��PY���A�"*;


total_loss�~�@

error_RE?

learning_rate_1�O?7��G�I       6%�	6QY���A�"*;


total_loss$s�@

error_RdzI?

learning_rate_1�O?7u�I�I       6%�	V`QY���A�"*;


total_loss���@

error_R_D?

learning_rate_1�O?7���\I       6%�	~�QY���A�"*;


total_loss�@

error_RxLR?

learning_rate_1�O?7����I       6%�	4�QY���A�"*;


total_loss��@

error_R�^?

learning_rate_1�O?7��I       6%�	�3RY���A�"*;


total_loss�8A

error_R�qN?

learning_rate_1�O?7�ErI       6%�	�vRY���A�"*;


total_lossY�@

error_R�'C?

learning_rate_1�O?7���I       6%�	��RY���A�"*;


total_loss��@

error_RJ�H?

learning_rate_1�O?7��cI       6%�	~�RY���A�"*;


total_loss�@

error_R��M?

learning_rate_1�O?7ͅ�	I       6%�	BSY���A�"*;


total_lossS��@

error_R��K?

learning_rate_1�O?7�tI       6%�	��SY���A�"*;


total_loss�<�@

error_R�|I?

learning_rate_1�O?74�4�I       6%�	3�SY���A�"*;


total_loss-��@

error_R�jd?

learning_rate_1�O?7/'CMI       6%�	TY���A�"*;


total_lossYH�@

error_R�*[?

learning_rate_1�O?7���I       6%�	�TTY���A�"*;


total_lossVփ@

error_R�O?

learning_rate_1�O?7�J-CI       6%�	5�TY���A�"*;


total_loss��@

error_R��<?

learning_rate_1�O?7f�u�I       6%�	R�TY���A�"*;


total_loss�ӓ@

error_RF�=?

learning_rate_1�O?7c�I       6%�	�(UY���A�"*;


total_loss�]�@

error_R,B?

learning_rate_1�O?7ܱ�+I       6%�	kUY���A�"*;


total_lossֳ@

error_RnG?

learning_rate_1�O?7 �`I       6%�	V�UY���A�"*;


total_loss�,�@

error_Rj�Q?

learning_rate_1�O?7�~��I       6%�	O�UY���A�"*;


total_loss�@

error_Rfb;?

learning_rate_1�O?7�DOI       6%�	<VY���A�"*;


total_lossdŖ@

error_RѾM?

learning_rate_1�O?7�(�RI       6%�	��VY���A�"*;


total_lossWڮ@

error_Rn�W?

learning_rate_1�O?7e��LI       6%�	2�VY���A�"*;


total_loss�D�@

error_R��S?

learning_rate_1�O?7�ܢKI       6%�	�WY���A�"*;


total_loss-1�@

error_R�B?

learning_rate_1�O?7��3�I       6%�	�XWY���A�"*;


total_loss�,�@

error_R�.J?

learning_rate_1�O?7Y�[�I       6%�	v�WY���A�"*;


total_loss��@

error_R��S?

learning_rate_1�O?7&L�I       6%�	�WY���A�"*;


total_loss���@

error_Rf(X?

learning_rate_1�O?7'�iI       6%�	�0XY���A�"*;


total_loss *�@

error_RDhR?

learning_rate_1�O?7܎��I       6%�	1sXY���A�"*;


total_loss���@

error_R�X?

learning_rate_1�O?7*3�I       6%�	O�XY���A�"*;


total_loss���@

error_R�R?

learning_rate_1�O?7DvI       6%�	��XY���A�"*;


total_loss
�@

error_R��X?

learning_rate_1�O?7����I       6%�	j<YY���A�"*;


total_loss�ؚ@

error_R��??

learning_rate_1�O?7*ǌ�I       6%�	<~YY���A�"*;


total_loss�m�@

error_R�'G?

learning_rate_1�O?7�Gj�I       6%�	$�YY���A�"*;


total_loss�*�@

error_R��Q?

learning_rate_1�O?7��a�I       6%�	S	ZY���A�"*;


total_loss��@

error_R:�I?

learning_rate_1�O?7օ@wI       6%�	�NZY���A�"*;


total_loss)�@

error_R�fF?

learning_rate_1�O?7Jd�I       6%�	��ZY���A�"*;


total_loss���@

error_RNr>?

learning_rate_1�O?76ՙ{I       6%�	�ZY���A�"*;


total_lossa��@

error_R��M?

learning_rate_1�O?7t�maI       6%�	*[Y���A�"*;


total_loss�{�@

error_R��J?

learning_rate_1�O?7{DuI       6%�	4�[Y���A�"*;


total_lossa-�@

error_R��G?

learning_rate_1�O?7��$�I       6%�	~�[Y���A�"*;


total_loss\��@

error_R_e?

learning_rate_1�O?7�1�I       6%�	�$\Y���A�"*;


total_loss�k�@

error_Rח:?

learning_rate_1�O?7^xUyI       6%�	�j\Y���A�"*;


total_loss�W�@

error_R��_?

learning_rate_1�O?7M9c�I       6%�	�\Y���A�"*;


total_loss��@

error_R��H?

learning_rate_1�O?7�2�I       6%�	��\Y���A�"*;


total_loss/C�@

error_R�2F?

learning_rate_1�O?7�t�I       6%�	S6]Y���A�"*;


total_loss�A

error_RAR?

learning_rate_1�O?7Q�%�I       6%�	��]Y���A�"*;


total_loss�a~@

error_R�mH?

learning_rate_1�O?7�>�?I       6%�	��]Y���A�"*;


total_loss��@

error_R&�Z?

learning_rate_1�O?7��_I       6%�	�!^Y���A�"*;


total_loss�A

error_RWqP?

learning_rate_1�O?7�1I       6%�	�f^Y���A�"*;


total_lossl�@

error_R�cT?

learning_rate_1�O?7E��I       6%�	�^Y���A�"*;


total_loss1Y�@

error_R �L?

learning_rate_1�O?7\�K�I       6%�	��^Y���A�"*;


total_loss� �@

error_R��F?

learning_rate_1�O?7<�vI       6%�	�4_Y���A�"*;


total_loss�=�@

error_R�_?

learning_rate_1�O?7w�mI       6%�	*�_Y���A�"*;


total_loss���@

error_R��D?

learning_rate_1�O?7y#HWI       6%�	:�_Y���A�"*;


total_loss}�@

error_R�W?

learning_rate_1�O?7��yI       6%�	�&`Y���A�"*;


total_loss]R�@

error_R�VU?

learning_rate_1�O?7a��I       6%�	Cr`Y���A�"*;


total_loss��A

error_R{C?

learning_rate_1�O?7��z�I       6%�	&�`Y���A�#*;


total_loss�q�@

error_R��G?

learning_rate_1�O?7Zʭ*I       6%�	�aY���A�#*;


total_loss���@

error_R3 Z?

learning_rate_1�O?7Eu5�I       6%�	�XaY���A�#*;


total_loss���@

error_R�~W?

learning_rate_1�O?7��_\I       6%�	+�aY���A�#*;


total_loss��@

error_R�eM?

learning_rate_1�O?7��U�I       6%�	q�aY���A�#*;


total_loss�'�@

error_Rw�I?

learning_rate_1�O?7/��yI       6%�	�0bY���A�#*;


total_loss�p�@

error_R�N9?

learning_rate_1�O?7�i��I       6%�	twbY���A�#*;


total_loss�JA

error_R�LC?

learning_rate_1�O?7��C�I       6%�	��bY���A�#*;


total_loss�ڔ@

error_R�hF?

learning_rate_1�O?7Jѥ�I       6%�	�cY���A�#*;


total_loss��@

error_R�jD?

learning_rate_1�O?7;�BI       6%�	lIcY���A�#*;


total_lossϢ�@

error_R�mL?

learning_rate_1�O?7k�3I       6%�	��cY���A�#*;


total_loss��@

error_Rx�=?

learning_rate_1�O?7����I       6%�	��cY���A�#*;


total_loss��A

error_R��Q?

learning_rate_1�O?7(�m�I       6%�	[dY���A�#*;


total_loss�:�@

error_R;
8?

learning_rate_1�O?7SD�kI       6%�	�bdY���A�#*;


total_loss�j�@

error_RgZ?

learning_rate_1�O?7���I       6%�	,�dY���A�#*;


total_loss\Tx@

error_R�>N?

learning_rate_1�O?7Ի�I       6%�	��dY���A�#*;


total_loss_ˑ@

error_RM�L?

learning_rate_1�O?7#$��I       6%�	�)eY���A�#*;


total_loss=<�@

error_R�	d?

learning_rate_1�O?7�)MI       6%�	�jeY���A�#*;


total_loss&6�@

error_R�T?

learning_rate_1�O?7�Z�I       6%�	�eY���A�#*;


total_loss�`�@

error_R�P?

learning_rate_1�O?73��I       6%�	��eY���A�#*;


total_loss{��@

error_RMde?

learning_rate_1�O?7�<��I       6%�	V3fY���A�#*;


total_loss�"�@

error_Rq�W?

learning_rate_1�O?7/�(I       6%�	pwfY���A�#*;


total_loss���@

error_R)�D?

learning_rate_1�O?7v��I       6%�	��fY���A�#*;


total_loss��@

error_R��Y?

learning_rate_1�O?7-��I       6%�	�gY���A�#*;


total_lossA��@

error_R�S?

learning_rate_1�O?7���I       6%�	kEgY���A�#*;


total_loss���@

error_R�!G?

learning_rate_1�O?7[�N�I       6%�	^�gY���A�#*;


total_loss��@

error_RA�I?

learning_rate_1�O?7�΂�I       6%�	<�gY���A�#*;


total_lossQ��@

error_Rg?

learning_rate_1�O?7�N�I       6%�	[hY���A�#*;


total_lossW�@

error_R�@U?

learning_rate_1�O?7���I       6%�	�ZhY���A�#*;


total_lossE��@

error_R)�a?

learning_rate_1�O?7W���I       6%�	2�hY���A�#*;


total_lossL(A

error_R&�W?

learning_rate_1�O?7@4+I       6%�	��hY���A�#*;


total_lossXρ@

error_RvJC?

learning_rate_1�O?7��=RI       6%�	�"iY���A�#*;


total_loss��@

error_R��K?

learning_rate_1�O?7TF��I       6%�	�eiY���A�#*;


total_lossi�@

error_R�Q?

learning_rate_1�O?7���I       6%�	�iY���A�#*;


total_loss&�@

error_R,TC?

learning_rate_1�O?7<��I       6%�	��iY���A�#*;


total_loss%A

error_R&;R?

learning_rate_1�O?7T��wI       6%�	�2jY���A�#*;


total_lossi?�@

error_RLKZ?

learning_rate_1�O?7�I       6%�	ujY���A�#*;


total_loss���@

error_R&�??

learning_rate_1�O?7��}I       6%�	˸jY���A�#*;


total_lossEj�@

error_RMYT?

learning_rate_1�O?7=��tI       6%�	��jY���A�#*;


total_loss�PA

error_Rs>M?

learning_rate_1�O?7|Oa�I       6%�	NkY���A�#*;


total_loss��@

error_R��T?

learning_rate_1�O?7��HI       6%�	S�kY���A�#*;


total_losst]�@

error_R[�]?

learning_rate_1�O?7��\�I       6%�	��kY���A�#*;


total_loss�_A

error_R�N?

learning_rate_1�O?7��e�I       6%�	_HlY���A�#*;


total_loss�"�@

error_RW�L?

learning_rate_1�O?7�9I       6%�	E�lY���A�#*;


total_lossC�@

error_R�2_?

learning_rate_1�O?7��ϓI       6%�	��lY���A�#*;


total_loss@��@

error_RvwL?

learning_rate_1�O?7����I       6%�	mY���A�#*;


total_loss�Ns@

error_R�>?

learning_rate_1�O?7��>I       6%�	f^mY���A�#*;


total_loss�ʥ@

error_RSk\?

learning_rate_1�O?7Cq_I       6%�	_�mY���A�#*;


total_loss�@

error_R�(U?

learning_rate_1�O?7�䚏I       6%�	��mY���A�#*;


total_lossFS�@

error_R�\?

learning_rate_1�O?7v��KI       6%�	-nY���A�#*;


total_lossF�@

error_R<HA?

learning_rate_1�O?7Jn�I       6%�	�qnY���A�#*;


total_loss�@

error_R Z?

learning_rate_1�O?7A`�I       6%�	&�nY���A�#*;


total_loss�3�@

error_RH�B?

learning_rate_1�O?7��J I       6%�	�nY���A�#*;


total_loss;�@

error_R�T?

learning_rate_1�O?7�чI       6%�	�@oY���A�#*;


total_loss�B�@

error_R	�L?

learning_rate_1�O?7H�k�I       6%�	\�oY���A�#*;


total_lossj�@

error_R��N?

learning_rate_1�O?7Q�"I       6%�	-�oY���A�#*;


total_lossNQ	A

error_R�1u?

learning_rate_1�O?7��Z�I       6%�	pY���A�#*;


total_loss��@

error_Rq�H?

learning_rate_1�O?7P�I       6%�	�RpY���A�#*;


total_loss�k�@

error_R�|<?

learning_rate_1�O?7h�y�I       6%�		�pY���A�#*;


total_loss���@

error_R�RU?

learning_rate_1�O?7A��yI       6%�	V�pY���A�#*;


total_lossC��@

error_R;%[?

learning_rate_1�O?7�`�I       6%�	�qY���A�#*;


total_loss�m
A

error_R�G?

learning_rate_1�O?7���aI       6%�	SqY���A�#*;


total_loss���@

error_R��E?

learning_rate_1�O?7�6QI       6%�	A�qY���A�#*;


total_loss�}�@

error_R`�K?

learning_rate_1�O?7g0�I       6%�	��qY���A�#*;


total_lossHN�@

error_R"=?

learning_rate_1�O?7y���I       6%�	yrY���A�#*;


total_loss��@

error_R��\?

learning_rate_1�O?7ho��I       6%�	RrY���A�#*;


total_lossQ��@

error_RZF?

learning_rate_1�O?7�8|I       6%�	��rY���A�#*;


total_loss�l�@

error_R�vA?

learning_rate_1�O?7�iGI       6%�	)�rY���A�#*;


total_loss�Q�@

error_R��a?

learning_rate_1�O?7�P�I       6%�	~sY���A�#*;


total_loss���@

error_R_�R?

learning_rate_1�O?7�Έ�I       6%�	HWsY���A�#*;


total_loss<��@

error_R�RX?

learning_rate_1�O?7'�)I       6%�	ϜsY���A�#*;


total_loss'�@

error_RϑZ?

learning_rate_1�O?7�C(I       6%�	H�sY���A�#*;


total_loss��\@

error_R=ZB?

learning_rate_1�O?7Gt�I       6%�	Q$tY���A�#*;


total_loss�E�@

error_R�?N?

learning_rate_1�O?7�ǥ7I       6%�	�dtY���A�#*;


total_loss��@

error_R�Q?

learning_rate_1�O?7=Ll4I       6%�	w�tY���A�#*;


total_loss�V�@

error_R�cS?

learning_rate_1�O?7��aI       6%�	��tY���A�#*;


total_loss�ǹ@

error_Rq�U?

learning_rate_1�O?7*�L�I       6%�	�(uY���A�#*;


total_loss$�@

error_R��S?

learning_rate_1�O?70$�AI       6%�	�huY���A�#*;


total_loss�@

error_R�@A?

learning_rate_1�O?7�ɦI       6%�	��uY���A�#*;


total_loss&��@

error_R�N?

learning_rate_1�O?7�l�RI       6%�	a�uY���A�#*;


total_loss/��@

error_RN?

learning_rate_1�O?7����I       6%�	1vY���A�#*;


total_loss�ɉ@

error_R��T?

learning_rate_1�O?7�I�lI       6%�	.rvY���A�#*;


total_loss�@�@

error_RZ�E?

learning_rate_1�O?7��-"I       6%�	��vY���A�#*;


total_loss��A

error_R?�S?

learning_rate_1�O?7�_WUI       6%�	F�vY���A�#*;


total_loss�kA

error_R3E?

learning_rate_1�O?7ҽ�nI       6%�	:7wY���A�#*;


total_loss���@

error_R��N?

learning_rate_1�O?7|��I       6%�	�wwY���A�#*;


total_lossR�@

error_Rq#X?

learning_rate_1�O?7��k�I       6%�	B�wY���A�#*;


total_loss�"A

error_R�:R?

learning_rate_1�O?7���I       6%�	��wY���A�#*;


total_lossz�@

error_R?Yd?

learning_rate_1�O?70���I       6%�	�:xY���A�#*;


total_loss�J�@

error_Rsf?

learning_rate_1�O?7)�ɥI       6%�	�xxY���A�#*;


total_loss���@

error_Rl�`?

learning_rate_1�O?7.���I       6%�	��xY���A�#*;


total_losst��@

error_RlXB?

learning_rate_1�O?7���I       6%�	��xY���A�#*;


total_loss݅�@

error_R�'M?

learning_rate_1�O?7���I       6%�	�?yY���A�#*;


total_loss�Ht@

error_R�AS?

learning_rate_1�O?7��R�I       6%�	`�yY���A�#*;


total_loss"�@

error_R��R?

learning_rate_1�O?7�F�I       6%�	��yY���A�#*;


total_loss�n-A

error_R	A?

learning_rate_1�O?79-�I       6%�	�zY���A�#*;


total_loss?��@

error_Rd�O?

learning_rate_1�O?7%C/�I       6%�	�VzY���A�#*;


total_lossD��@

error_R��S?

learning_rate_1�O?7�j��I       6%�	Q�zY���A�#*;


total_loss;v�@

error_R�yP?

learning_rate_1�O?7�GI       6%�	��zY���A�#*;


total_loss6��@

error_R�NW?

learning_rate_1�O?7X�GI       6%�	({Y���A�#*;


total_loss�M�@

error_Rv�O?

learning_rate_1�O?7� J|I       6%�	�{Y���A�#*;


total_lossf8�@

error_R%�Y?

learning_rate_1�O?7��&�I       6%�	��{Y���A�#*;


total_loss���@

error_R�E?

learning_rate_1�O?7�'qI       6%�	|Y���A�#*;


total_lossԐ�@

error_REP?

learning_rate_1�O?7�K*PI       6%�	!`|Y���A�#*;


total_loss��@

error_R,CN?

learning_rate_1�O?7��I       6%�	��|Y���A�#*;


total_lossN��@

error_R�QL?

learning_rate_1�O?7!uέI       6%�	M�|Y���A�#*;


total_loss���@

error_RX,M?

learning_rate_1�O?7ad�I       6%�	�)}Y���A�#*;


total_loss��@

error_R�hU?

learning_rate_1�O?7>ZcI       6%�	il}Y���A�#*;


total_loss�/�@

error_R2/??

learning_rate_1�O?7���qI       6%�	g�}Y���A�#*;


total_loss�Z�@

error_R.�L?

learning_rate_1�O?7Q�%I       6%�	��}Y���A�#*;


total_loss$'�@

error_R��T?

learning_rate_1�O?7�Z��I       6%�	�A~Y���A�#*;


total_lossFJ�@

error_RQ�L?

learning_rate_1�O?7�k35I       6%�	�~Y���A�#*;


total_loss��@

error_RS�I?

learning_rate_1�O?7���I       6%�	��~Y���A�#*;


total_loss2H�@

error_RԳZ?

learning_rate_1�O?7�9uI       6%�	pY���A�#*;


total_loss��@

error_RM�K?

learning_rate_1�O?7�٘I       6%�	�KY���A�#*;


total_loss�@

error_R��a?

learning_rate_1�O?7���I       6%�	�Y���A�#*;


total_losss��@

error_R�[?

learning_rate_1�O?7%L�I       6%�	��Y���A�#*;


total_lossH\�@

error_R�KJ?

learning_rate_1�O?7�t`{I       6%�	��Y���A�#*;


total_loss�e�@

error_R��G?

learning_rate_1�O?77/��I       6%�	�L�Y���A�#*;


total_loss��@

error_R�L?

learning_rate_1�O?74I       6%�	��Y���A�#*;


total_lossOL�@

error_RƌU?

learning_rate_1�O?7X?�I       6%�	�ԀY���A�#*;


total_lossv�@

error_R3K?

learning_rate_1�O?76z`I       6%�	(�Y���A�#*;


total_loss4o�@

error_R&@K?

learning_rate_1�O?7��I       6%�	�]�Y���A�#*;


total_lossі@

error_R̱=?

learning_rate_1�O?7fݮI       6%�	���Y���A�#*;


total_loss&�@

error_RhwN?

learning_rate_1�O?7n�|%I       6%�	��Y���A�#*;


total_loss���@

error_Rq:?

learning_rate_1�O?7�o�RI       6%�	�)�Y���A�#*;


total_loss���@

error_R�P?

learning_rate_1�O?7	̠6I       6%�	�o�Y���A�#*;


total_loss�=�@

error_R;KA?

learning_rate_1�O?7�lvI       6%�	���Y���A�#*;


total_loss1��@

error_RaWR?

learning_rate_1�O?7	���I       6%�	��Y���A�$*;


total_loss�ƛ@

error_Ra�7?

learning_rate_1�O?7��SMI       6%�	i6�Y���A�$*;


total_loss]�@

error_R.�S?

learning_rate_1�O?7a���I       6%�	tx�Y���A�$*;


total_loss� �@

error_Rs�D?

learning_rate_1�O?7W�EI       6%�	Ѹ�Y���A�$*;


total_lossl��@

error_R�`O?

learning_rate_1�O?7���I       6%�	���Y���A�$*;


total_loss��@

error_R��>?

learning_rate_1�O?7r �I       6%�	�>�Y���A�$*;


total_loss!��@

error_R��Y?

learning_rate_1�O?7TwӅI       6%�	s��Y���A�$*;


total_loss�{�@

error_R��f?

learning_rate_1�O?7
�8I       6%�	VɄY���A�$*;


total_loss{E�@

error_R�5A?

learning_rate_1�O?72�@I       6%�	�	�Y���A�$*;


total_loss;��@

error_R��\?

learning_rate_1�O?7My�I       6%�	�M�Y���A�$*;


total_lossC<�@

error_R��E?

learning_rate_1�O?7�{�I       6%�	|��Y���A�$*;


total_loss�@

error_RlzU?

learning_rate_1�O?7�exYI       6%�	�ՅY���A�$*;


total_loss��@

error_R��D?

learning_rate_1�O?7o�ZcI       6%�	d�Y���A�$*;


total_loss��@

error_R�J?

learning_rate_1�O?7�n�I       6%�	�T�Y���A�$*;


total_loss��@

error_R��U?

learning_rate_1�O?7���\I       6%�	(��Y���A�$*;


total_loss`�@

error_R�S?

learning_rate_1�O?7`c�I       6%�	ֆY���A�$*;


total_loss���@

error_R=�R?

learning_rate_1�O?7K<I       6%�	`�Y���A�$*;


total_loss���@

error_REJ?

learning_rate_1�O?7H�*�I       6%�	9V�Y���A�$*;


total_losss��@

error_R��P?

learning_rate_1�O?7��
�I       6%�	ӝ�Y���A�$*;


total_loss�.�@

error_R!i\?

learning_rate_1�O?79��I       6%�	��Y���A�$*;


total_loss�l�@

error_R��C?

learning_rate_1�O?7��V�I       6%�	05�Y���A�$*;


total_loss|E�@

error_R�xS?

learning_rate_1�O?7����I       6%�	�w�Y���A�$*;


total_lossDێ@

error_R�^?

learning_rate_1�O?7H!>�I       6%�	���Y���A�$*;


total_loss�%�@

error_R��F?

learning_rate_1�O?7�p4I       6%�	�Y���A�$*;


total_loss�S�@

error_R�@I?

learning_rate_1�O?79xۭI       6%�	�Y�Y���A�$*;


total_loss�(�@

error_R��U?

learning_rate_1�O?7&�U�I       6%�	B��Y���A�$*;


total_loss1�X@

error_RC8?

learning_rate_1�O?7�/��I       6%�	��Y���A�$*;


total_loss��@

error_R�Q?

learning_rate_1�O?7gH[�I       6%�	k)�Y���A�$*;


total_loss+�@

error_RQ?

learning_rate_1�O?7�g�I       6%�	Ml�Y���A�$*;


total_lossO��@

error_RL�E?

learning_rate_1�O?7��I       6%�	%��Y���A�$*;


total_lossʑ�@

error_RObI?

learning_rate_1�O?7�v��I       6%�	���Y���A�$*;


total_loss�n�@

error_Rr�4?

learning_rate_1�O?7ViJ�I       6%�	�5�Y���A�$*;


total_loss���@

error_R�.K?

learning_rate_1�O?74J{5I       6%�	���Y���A�$*;


total_lossm�@

error_R��[?

learning_rate_1�O?7ӌ:4I       6%�	(݋Y���A�$*;


total_lossr@

error_R_�C?

learning_rate_1�O?7�d+
I       6%�	�"�Y���A�$*;


total_loss��@

error_Rn�:?

learning_rate_1�O?7�H�kI       6%�	Gf�Y���A�$*;


total_loss2��@

error_R��T?

learning_rate_1�O?7C,/�I       6%�	:��Y���A�$*;


total_loss!$�@

error_R�J?

learning_rate_1�O?7�*h1I       6%�	#�Y���A�$*;


total_loss��A

error_R�4S?

learning_rate_1�O?7��ʺI       6%�	N7�Y���A�$*;


total_lossdX�@

error_R��K?

learning_rate_1�O?7�ʍ�I       6%�	J|�Y���A�$*;


total_lossoh�@

error_R_BP?

learning_rate_1�O?7�̗�I       6%�	ÍY���A�$*;


total_loss���@

error_R�P?

learning_rate_1�O?7ñI       6%�	��Y���A�$*;


total_loss�P�@

error_R�tI?

learning_rate_1�O?7MHV=I       6%�	�M�Y���A�$*;


total_lossF��@

error_R�<B?

learning_rate_1�O?7q��I       6%�	��Y���A�$*;


total_loss�I�@

error_R�`C?

learning_rate_1�O?7Ky7NI       6%�	!ӎY���A�$*;


total_loss	�b@

error_R�K?

learning_rate_1�O?7̛��I       6%�	u�Y���A�$*;


total_loss# �@

error_R3�\?

learning_rate_1�O?7Y�FpI       6%�	[�Y���A�$*;


total_losstUp@

error_R��^?

learning_rate_1�O?7BGI�I       6%�	��Y���A�$*;


total_loss��@

error_R=�S?

learning_rate_1�O?7�mk I       6%�	&ۏY���A�$*;


total_loss��@

error_R�DF?

learning_rate_1�O?7�&�*I       6%�	�Y���A�$*;


total_loss�5q@

error_R�X@?

learning_rate_1�O?7("I       6%�	�_�Y���A�$*;


total_loss�u@

error_RX�]?

learning_rate_1�O?7����I       6%�	+��Y���A�$*;


total_loss�V�@

error_R��R?

learning_rate_1�O?7{m�I       6%�	��Y���A�$*;


total_loss���@

error_R)zV?

learning_rate_1�O?7�Q	I       6%�	o"�Y���A�$*;


total_loss��@

error_R/�2?

learning_rate_1�O?7�4�I       6%�	1b�Y���A�$*;


total_lossm��@

error_R<�F?

learning_rate_1�O?7]��;I       6%�	⢑Y���A�$*;


total_loss\o�@

error_R��N?

learning_rate_1�O?7�)�RI       6%�	��Y���A�$*;


total_loss���@

error_R�O?

learning_rate_1�O?74�VGI       6%�	Q#�Y���A�$*;


total_loss���@

error_RW�Y?

learning_rate_1�O?7ò�I       6%�	 c�Y���A�$*;


total_loss��@

error_R�G?

learning_rate_1�O?7��cPI       6%�	��Y���A�$*;


total_loss�%�@

error_R��Y?

learning_rate_1�O?7p�[�I       6%�	��Y���A�$*;


total_loss�`@

error_R} V?

learning_rate_1�O?7���I       6%�	�,�Y���A�$*;


total_loss���@

error_R�e?

learning_rate_1�O?7[���I       6%�	�n�Y���A�$*;


total_loss,ڄ@

error_Rs�E?

learning_rate_1�O?7ơ�0I       6%�	E��Y���A�$*;


total_loss���@

error_R��A?

learning_rate_1�O?7���I       6%�	V�Y���A�$*;


total_loss�K�@

error_RC.V?

learning_rate_1�O?7(��lI       6%�	30�Y���A�$*;


total_lossg� A

error_R�X?

learning_rate_1�O?7Z�I       6%�	o�Y���A�$*;


total_loss#��@

error_R�R?

learning_rate_1�O?7m���I       6%�	���Y���A�$*;


total_lossq�`@

error_R#gK?

learning_rate_1�O?7�� I       6%�	k�Y���A�$*;


total_lossz��@

error_R�V?

learning_rate_1�O?7೘II       6%�	4�Y���A�$*;


total_loss.��@

error_R� F?

learning_rate_1�O?7���I       6%�	Pw�Y���A�$*;


total_lossW��@

error_R�*P?

learning_rate_1�O?7T��AI       6%�	a��Y���A�$*;


total_loss���@

error_R�mN?

learning_rate_1�O?74G��I       6%�	� �Y���A�$*;


total_lossSc�@

error_R	a[?

learning_rate_1�O?7��@?I       6%�	tB�Y���A�$*;


total_loss_o�@

error_R��W?

learning_rate_1�O?7m�9I       6%�	���Y���A�$*;


total_lossmb�@

error_RC�c?

learning_rate_1�O?7s~�I       6%�	�ȖY���A�$*;


total_loss(�@

error_R��P?

learning_rate_1�O?7�\I�I       6%�	��Y���A�$*;


total_loss|�@

error_R��H?

learning_rate_1�O?7��g�I       6%�	Q�Y���A�$*;


total_loss�E�@

error_R��Y?

learning_rate_1�O?7�FI       6%�	���Y���A�$*;


total_loss_̻@

error_RNN?

learning_rate_1�O?7�Z��I       6%�	cҗY���A�$*;


total_lossD��@

error_R��??

learning_rate_1�O?7ӕ	I       6%�	��Y���A�$*;


total_loss��@

error_Rf�P?

learning_rate_1�O?7�FI       6%�	W�Y���A�$*;


total_loss���@

error_R�XE?

learning_rate_1�O?7P���I       6%�	���Y���A�$*;


total_loss�O�@

error_RT?

learning_rate_1�O?7R�jiI       6%�	�ݘY���A�$*;


total_loss<~�@

error_RʐJ?

learning_rate_1�O?7Vz�QI       6%�	a"�Y���A�$*;


total_lossw�T@

error_RS�O?

learning_rate_1�O?7c��xI       6%�	eh�Y���A�$*;


total_lossZ��@

error_R)�I?

learning_rate_1�O?7y�7�I       6%�	���Y���A�$*;


total_loss$9�@

error_R_Z??

learning_rate_1�O?7QN}JI       6%�	��Y���A�$*;


total_loss�C�@

error_R*K?

learning_rate_1�O?7��I       6%�	-�Y���A�$*;


total_loss�F�@

error_R�B?

learning_rate_1�O?7r�S�I       6%�	�q�Y���A�$*;


total_loss;��@

error_R��<?

learning_rate_1�O?7i~�I       6%�	Ĵ�Y���A�$*;


total_lossL+�@

error_RlN?

learning_rate_1�O?7>�TI       6%�	��Y���A�$*;


total_loss� �@

error_RD?

learning_rate_1�O?7����I       6%�	=F�Y���A�$*;


total_loss���@

error_R�>?

learning_rate_1�O?7����I       6%�	���Y���A�$*;


total_loss���@

error_R�(O?

learning_rate_1�O?75�9�I       6%�	O�Y���A�$*;


total_loss���@

error_R��c?

learning_rate_1�O?7��KI       6%�	�)�Y���A�$*;


total_loss��@

error_R��V?

learning_rate_1�O?7�Q�I       6%�	�l�Y���A�$*;


total_loss9t@

error_R@�:?

learning_rate_1�O?7D�"�I       6%�	���Y���A�$*;


total_loss���@

error_R�BI?

learning_rate_1�O?75�#�I       6%�	E��Y���A�$*;


total_loss�V�@

error_R_�^?

learning_rate_1�O?7��{�I       6%�	�>�Y���A�$*;


total_loss�c�@

error_R(�T?

learning_rate_1�O?7�B�}I       6%�	{��Y���A�$*;


total_loss���@

error_R�S?

learning_rate_1�O?7<�@�I       6%�	�̝Y���A�$*;


total_lossϟ�@

error_RDO?

learning_rate_1�O?7B�I       6%�	d�Y���A�$*;


total_lossߓ�@

error_RC�Q?

learning_rate_1�O?7��y�I       6%�	tW�Y���A�$*;


total_loss.4�@

error_R�F?

learning_rate_1�O?7i�NI       6%�	ژ�Y���A�$*;


total_loss$A

error_RZQ?

learning_rate_1�O?7�lI       6%�	�ڞY���A�$*;


total_loss�ȯ@

error_R�C<?

learning_rate_1�O?7̢'I       6%�	�Y���A�$*;


total_loss�7�@

error_R�WC?

learning_rate_1�O?7B2`4I       6%�	�e�Y���A�$*;


total_loss��@

error_R9W?

learning_rate_1�O?7�W`I       6%�	W��Y���A�$*;


total_loss�*M@

error_R��R?

learning_rate_1�O?7vV)I       6%�	��Y���A�$*;


total_loss��A

error_Rx�J?

learning_rate_1�O?7v"gI       6%�	�0�Y���A�$*;


total_loss���@

error_R,RN?

learning_rate_1�O?7�=߻I       6%�	�w�Y���A�$*;


total_loss�j�@

error_R%<:?

learning_rate_1�O?7KeI       6%�	R��Y���A�$*;


total_lossqǶ@

error_RzOL?

learning_rate_1�O?7��n2I       6%�	���Y���A�$*;


total_lossf��@

error_R�mT?

learning_rate_1�O?7���I       6%�	�5�Y���A�$*;


total_lossn��@

error_R��F?

learning_rate_1�O?7I��I       6%�	�t�Y���A�$*;


total_lossq�@

error_R��h?

learning_rate_1�O?7gp�(I       6%�	���Y���A�$*;


total_loss���@

error_R}iB?

learning_rate_1�O?7���<I       6%�	��Y���A�$*;


total_loss���@

error_R��K?

learning_rate_1�O?7o�ŞI       6%�	V7�Y���A�$*;


total_loss���@

error_RbI?

learning_rate_1�O?7$��I       6%�	w�Y���A�$*;


total_lossE%�@

error_R|�J?

learning_rate_1�O?7�O�I       6%�	N��Y���A�$*;


total_loss_�p@

error_R�~F?

learning_rate_1�O?7C>�I       6%�	��Y���A�$*;


total_loss�ƙ@

error_R��I?

learning_rate_1�O?7~��xI       6%�	 9�Y���A�$*;


total_loss���@

error_R\�W?

learning_rate_1�O?7.�HrI       6%�	y�Y���A�$*;


total_loss�J�@

error_R�la?

learning_rate_1�O?72�VI       6%�	���Y���A�$*;


total_loss|7�@

error_R��M?

learning_rate_1�O?7�n2I       6%�	���Y���A�$*;


total_loss��@

error_R��L?

learning_rate_1�O?7?���I       6%�	�C�Y���A�$*;


total_loss��@

error_R�O?

learning_rate_1�O?75XÈI       6%�	��Y���A�$*;


total_lossJ�@

error_R�M?

learning_rate_1�O?7��mzI       6%�	�äY���A�%*;


total_lossL?�@

error_Ro�V?

learning_rate_1�O?7N�DI       6%�	Z�Y���A�%*;


total_loss���@

error_R4�C?

learning_rate_1�O?7��moI       6%�	OC�Y���A�%*;


total_loss\�@

error_Rq8I?

learning_rate_1�O?7�]��I       6%�	Y���A�%*;


total_loss�^�@

error_R� ^?

learning_rate_1�O?7�I       6%�	�ťY���A�%*;


total_lossv��@

error_R44?

learning_rate_1�O?7p��I       6%�	��Y���A�%*;


total_lossf��@

error_R�B?

learning_rate_1�O?7�>I       6%�	MO�Y���A�%*;


total_loss�A

error_R�R?

learning_rate_1�O?7��/qI       6%�	��Y���A�%*;


total_lossw��@

error_R�m?

learning_rate_1�O?7h�6�I       6%�	�̦Y���A�%*;


total_lossCu�@

error_Rs G?

learning_rate_1�O?7�;�I       6%�	��Y���A�%*;


total_loss���@

error_R��=?

learning_rate_1�O?7@�3�I       6%�	N�Y���A�%*;


total_loss3q�@

error_R��D?

learning_rate_1�O?7�[�GI       6%�	ː�Y���A�%*;


total_loss��@

error_R��D?

learning_rate_1�O?7�[2I       6%�	{ҧY���A�%*;


total_loss`ޗ@

error_R��:?

learning_rate_1�O?7�8��I       6%�	��Y���A�%*;


total_loss{�@

error_Rn`?

learning_rate_1�O?7tX��I       6%�	S�Y���A�%*;


total_lossH��@

error_R$A?

learning_rate_1�O?7�.xI       6%�	���Y���A�%*;


total_lossD_�@

error_RL�`?

learning_rate_1�O?7Z�#�I       6%�	�֨Y���A�%*;


total_loss���@

error_R��R?

learning_rate_1�O?7бV�I       6%�	��Y���A�%*;


total_lossE��@

error_R6�]?

learning_rate_1�O?7�4YI       6%�	{V�Y���A�%*;


total_loss(�%A

error_R�CP?

learning_rate_1�O?7V�DI       6%�	A��Y���A�%*;


total_loss���@

error_RatW?

learning_rate_1�O?7��ǻI       6%�	�ԩY���A�%*;


total_loss;��@

error_RJ�,?

learning_rate_1�O?7
�ݗI       6%�	��Y���A�%*;


total_loss�N�@

error_R
�X?

learning_rate_1�O?7�J�I       6%�	�T�Y���A�%*;


total_lossz6�@

error_Rq�L?

learning_rate_1�O?7��kI       6%�	K��Y���A�%*;


total_loss*�t@

error_R��K?

learning_rate_1�O?7R��I       6%�	�תY���A�%*;


total_loss_�@

error_R�eS?

learning_rate_1�O?7�@ I       6%�	p�Y���A�%*;


total_loss�ǭ@

error_Rf�A?

learning_rate_1�O?7��4�I       6%�	�|�Y���A�%*;


total_loss�V�@

error_Ra�@?

learning_rate_1�O?7qčgI       6%�	~īY���A�%*;


total_loss��@

error_R�.C?

learning_rate_1�O?7	��I       6%�	
�Y���A�%*;


total_loss���@

error_R�a:?

learning_rate_1�O?7�5I       6%�	^M�Y���A�%*;


total_lossj�@

error_R̠S?

learning_rate_1�O?7� sI       6%�	���Y���A�%*;


total_loss���@

error_R�y>?

learning_rate_1�O?7��gcI       6%�	�ѬY���A�%*;


total_loss&�A

error_R��[?

learning_rate_1�O?7����I       6%�	J�Y���A�%*;


total_loss*��@

error_R�fK?

learning_rate_1�O?7� ?�I       6%�	,S�Y���A�%*;


total_loss^�@

error_R��R?

learning_rate_1�O?7l��fI       6%�	z��Y���A�%*;


total_loss���@

error_Rs?

learning_rate_1�O?75�ZSI       6%�	�խY���A�%*;


total_lossz��@

error_R�Lh?

learning_rate_1�O?7��=rI       6%�	k�Y���A�%*;


total_loss_��@

error_R��K?

learning_rate_1�O?7~A��I       6%�	"a�Y���A�%*;


total_loss��@

error_R�E?

learning_rate_1�O?7�hm�I       6%�	���Y���A�%*;


total_loss
��@

error_R�^?

learning_rate_1�O?7��eFI       6%�	��Y���A�%*;


total_loss3�@

error_Rn�T?

learning_rate_1�O?7���I       6%�	�E�Y���A�%*;


total_lossU4�@

error_R8/8?

learning_rate_1�O?7���I       6%�	垯Y���A�%*;


total_lossM��@

error_R]Xc?

learning_rate_1�O?7�Dq2I       6%�	/�Y���A�%*;


total_loss�T�@

error_R�L?

learning_rate_1�O?7�}DBI       6%�	(�Y���A�%*;


total_loss���@

error_R��`?

learning_rate_1�O?7�  AI       6%�	�l�Y���A�%*;


total_loss�ʮ@

error_RMEA?

learning_rate_1�O?7�!�I       6%�	���Y���A�%*;


total_lossT�@

error_R2nG?

learning_rate_1�O?7(6I       6%�	��Y���A�%*;


total_lossw�@

error_R)N?

learning_rate_1�O?7����I       6%�	f4�Y���A�%*;


total_lossVd�@

error_RdK?

learning_rate_1�O?7���I       6%�	Lv�Y���A�%*;


total_loss� �@

error_R�xH?

learning_rate_1�O?74/�I       6%�	޸�Y���A�%*;


total_lossV��@

error_R�T?

learning_rate_1�O?7	��I       6%�	���Y���A�%*;


total_loss@}�@

error_R	SH?

learning_rate_1�O?7@g�I       6%�	�<�Y���A�%*;


total_loss���@

error_RdVE?

learning_rate_1�O?7��I       6%�	~�Y���A�%*;


total_loss\;�@

error_RM�C?

learning_rate_1�O?7��FI       6%�	o��Y���A�%*;


total_loss�ZC@

error_R��W?

learning_rate_1�O?7$F}pI       6%�	� �Y���A�%*;


total_loss!��@

error_R�U?

learning_rate_1�O?7�rI       6%�	�B�Y���A�%*;


total_loss���@

error_R.�T?

learning_rate_1�O?7Z��I       6%�	���Y���A�%*;


total_loss	��@

error_R��L?

learning_rate_1�O?7�Җ�I       6%�	�ƳY���A�%*;


total_loss��@

error_R�C?

learning_rate_1�O?7&��I       6%�	�	�Y���A�%*;


total_loss*&�@

error_RLV?

learning_rate_1�O?7�-ڎI       6%�	�N�Y���A�%*;


total_loss���@

error_R@�=?

learning_rate_1�O?73�0I       6%�	���Y���A�%*;


total_loss�X�@

error_R[�Y?

learning_rate_1�O?7�P�I       6%�	rѴY���A�%*;


total_loss*�@

error_R�HO?

learning_rate_1�O?7�?=�I       6%�	$�Y���A�%*;


total_loss���@

error_R��P?

learning_rate_1�O?7�W�I       6%�	�W�Y���A�%*;


total_loss%��@

error_RsoT?

learning_rate_1�O?7HP��I       6%�	���Y���A�%*;


total_loss��@

error_R)�d?

learning_rate_1�O?7E���I       6%�	�ڵY���A�%*;


total_loss��@

error_RW�P?

learning_rate_1�O?7"���I       6%�	N�Y���A�%*;


total_loss�@

error_R�N?

learning_rate_1�O?7�sI       6%�	7_�Y���A�%*;


total_loss�̶@

error_R�T?

learning_rate_1�O?7�cqI       6%�	Y���A�%*;


total_loss���@

error_RM�F?

learning_rate_1�O?7��ңI       6%�	i�Y���A�%*;


total_loss6IA

error_R)6G?

learning_rate_1�O?7��{XI       6%�	X1�Y���A�%*;


total_loss)K�@

error_R��E?

learning_rate_1�O?7��_I       6%�	jv�Y���A�%*;


total_loss��@

error_RW�B?

learning_rate_1�O?7ڍ�{I       6%�	���Y���A�%*;


total_lossچA

error_R��F?

learning_rate_1�O?7���I       6%�	[��Y���A�%*;


total_loss�0A

error_R��O?

learning_rate_1�O?7��~�I       6%�	[8�Y���A�%*;


total_lossS��@

error_R�`?

learning_rate_1�O?7,�#�I       6%�	Uv�Y���A�%*;


total_loss���@

error_R�]?

learning_rate_1�O?7���sI       6%�	絸Y���A�%*;


total_loss���@

error_R�=?

learning_rate_1�O?7P0|I       6%�	��Y���A�%*;


total_loss��@

error_R�#`?

learning_rate_1�O?7@�c�I       6%�	�B�Y���A�%*;


total_loss�@

error_R�CM?

learning_rate_1�O?7'{�I       6%�	D��Y���A�%*;


total_loss���@

error_RC�D?

learning_rate_1�O?7	G�I       6%�	�ƹY���A�%*;


total_lossa��@

error_Rl�S?

learning_rate_1�O?7)K��I       6%�	��Y���A�%*;


total_loss;��@

error_R�iS?

learning_rate_1�O?7�%�UI       6%�	�S�Y���A�%*;


total_loss3H�@

error_RnIE?

learning_rate_1�O?7�dN�I       6%�	9��Y���A�%*;


total_loss��@

error_RTKE?

learning_rate_1�O?7 ��I       6%�	�׺Y���A�%*;


total_loss�.q@

error_R|SK?

learning_rate_1�O?7x��I       6%�	��Y���A�%*;


total_loss�e�@

error_R��=?

learning_rate_1�O?7����I       6%�	�n�Y���A�%*;


total_loss��@

error_RB?

learning_rate_1�O?7~q#I       6%�	�ĻY���A�%*;


total_lossz �@

error_RHY?

learning_rate_1�O?7�e�DI       6%�	��Y���A�%*;


total_loss���@

error_R��c?

learning_rate_1�O?7ֆ��I       6%�	�G�Y���A�%*;


total_loss&��@

error_RTcG?

learning_rate_1�O?7��&I       6%�	9��Y���A�%*;


total_loss?׎@

error_R�LG?

learning_rate_1�O?7��C�I       6%�	�ϼY���A�%*;


total_lossf��@

error_R�iO?

learning_rate_1�O?7����I       6%�	��Y���A�%*;


total_loss8^�@

error_R�V?

learning_rate_1�O?7�b
�I       6%�	�U�Y���A�%*;


total_loss���@

error_R�6S?

learning_rate_1�O?7Q[�DI       6%�	ŗ�Y���A�%*;


total_loss�L�@

error_RIU?

learning_rate_1�O?7�5�jI       6%�	�׽Y���A�%*;


total_lossC�@

error_R\:`?

learning_rate_1�O?7��+DI       6%�	��Y���A�%*;


total_loss7��@

error_RR�;?

learning_rate_1�O?7��I       6%�	]_�Y���A�%*;


total_loss�!�@

error_R�bi?

learning_rate_1�O?7���I       6%�	���Y���A�%*;


total_lossr��@

error_R�I?

learning_rate_1�O?7XERI       6%�	��Y���A�%*;


total_lossq�@

error_R�S>?

learning_rate_1�O?7IO��I       6%�	,�Y���A�%*;


total_loss2�@

error_R2�f?

learning_rate_1�O?7�_I       6%�	�j�Y���A�%*;


total_loss���@

error_RғY?

learning_rate_1�O?7~4��I       6%�	.��Y���A�%*;


total_loss�@

error_R3�V?

learning_rate_1�O?7�f�|I       6%�	��Y���A�%*;


total_lossmِ@

error_R-�P?

learning_rate_1�O?7I       6%�	h)�Y���A�%*;


total_lossN��@

error_R�tF?

learning_rate_1�O?7���I       6%�	i�Y���A�%*;


total_loss(�@

error_R��G?

learning_rate_1�O?7"Y�=I       6%�	w��Y���A�%*;


total_loss@�@

error_R=?H?

learning_rate_1�O?7WI       6%�	J��Y���A�%*;


total_lossx5�@

error_R��i?

learning_rate_1�O?7���I       6%�	�1�Y���A�%*;


total_loss�0�@

error_R��@?

learning_rate_1�O?7ǔ I       6%�	Is�Y���A�%*;


total_loss��@

error_R�_?

learning_rate_1�O?7r��4I       6%�	y��Y���A�%*;


total_lossdR�@

error_R�KI?

learning_rate_1�O?72hI       6%�	��Y���A�%*;


total_loss,{@

error_R�mF?

learning_rate_1�O?7�n܆I       6%�	�5�Y���A�%*;


total_loss!�@

error_R��U?

learning_rate_1�O?7��I       6%�	v�Y���A�%*;


total_lossȀ�@

error_RstX?

learning_rate_1�O?7���I       6%�	=��Y���A�%*;


total_loss�B�@

error_R|W?

learning_rate_1�O?7�j��I       6%�	���Y���A�%*;


total_lossW1�@

error_RA|G?

learning_rate_1�O?7r��I       6%�	8�Y���A�%*;


total_loss,C�@

error_R��^?

learning_rate_1�O?7���I       6%�	�x�Y���A�%*;


total_loss���@

error_R�bL?

learning_rate_1�O?7�_w�I       6%�	��Y���A�%*;


total_loss�n�@

error_RI	J?

learning_rate_1�O?7z��dI       6%�	���Y���A�%*;


total_loss��@

error_RRV?

learning_rate_1�O?7�LHrI       6%�	:�Y���A�%*;


total_loss#�@

error_R4F?

learning_rate_1�O?7=��I       6%�	�{�Y���A�%*;


total_loss��@

error_RS@?

learning_rate_1�O?7���I       6%�	Ƽ�Y���A�%*;


total_lossF̺@

error_R)�O?

learning_rate_1�O?7e29�I       6%�	���Y���A�%*;


total_loss�@

error_R�dJ?

learning_rate_1�O?77��I       6%�	�@�Y���A�%*;


total_loss<x�@

error_R��S?

learning_rate_1�O?7����I       6%�	���Y���A�%*;


total_lossT��@

error_R�E?

learning_rate_1�O?7]�ҏI       6%�	���Y���A�%*;


total_loss�П@

error_R��I?

learning_rate_1�O?7g�kI       6%�	�Y���A�%*;


total_loss���@

error_RO~B?

learning_rate_1�O?7�s�!I       6%�	�D�Y���A�&*;


total_loss�@

error_R�J?

learning_rate_1�O?7��o�I       6%�	0��Y���A�&*;


total_loss��@

error_RbL?

learning_rate_1�O?7��I       6%�	���Y���A�&*;


total_loss" �@

error_R,=?

learning_rate_1�O?7�m^�I       6%�	_�Y���A�&*;


total_loss��@

error_R�4Q?

learning_rate_1�O?7�Y�"I       6%�	�C�Y���A�&*;


total_loss�k�@

error_R�(E?

learning_rate_1�O?7�%�I       6%�	H��Y���A�&*;


total_loss��@

error_R�S?

learning_rate_1�O?7����I       6%�	���Y���A�&*;


total_loss��@

error_R��??

learning_rate_1�O?7O��0I       6%�	�	�Y���A�&*;


total_lossz��@

error_R!2L?

learning_rate_1�O?7L��I       6%�	:O�Y���A�&*;


total_lossQ:�@

error_RX�W?

learning_rate_1�O?7a>$�I       6%�	��Y���A�&*;


total_loss�h�@

error_R�1?

learning_rate_1�O?7����I       6%�	@��Y���A�&*;


total_lossD�@

error_R\J?

learning_rate_1�O?7@�I       6%�	��Y���A�&*;


total_loss���@

error_R��R?

learning_rate_1�O?7T���I       6%�	Z�Y���A�&*;


total_lossM��@

error_R��W?

learning_rate_1�O?7q���I       6%�	���Y���A�&*;


total_loss���@

error_R�)b?

learning_rate_1�O?7	���I       6%�	F��Y���A�&*;


total_lossY{�@

error_R �M?

learning_rate_1�O?7���,I       6%�	QC�Y���A�&*;


total_loss���@

error_R�L?

learning_rate_1�O?7
={�I       6%�	R��Y���A�&*;


total_loss���@

error_R�Q?

learning_rate_1�O?7�E0bI       6%�	���Y���A�&*;


total_loss�J�@

error_R��R?

learning_rate_1�O?7��/7I       6%�	W.�Y���A�&*;


total_loss��@

error_R6�Q?

learning_rate_1�O?7�1��I       6%�	���Y���A�&*;


total_loss�!�@

error_RŪI?

learning_rate_1�O?7L^2�I       6%�	/��Y���A�&*;


total_loss�+�@

error_R��M?

learning_rate_1�O?7m���I       6%�	@�Y���A�&*;


total_loss�3A

error_R\Mb?

learning_rate_1�O?7.�pI       6%�	�Y�Y���A�&*;


total_loss��@

error_R	�R?

learning_rate_1�O?7JJ�=I       6%�	���Y���A�&*;


total_loss�lA

error_R='X?

learning_rate_1�O?7Eo!�I       6%�	���Y���A�&*;


total_loss���@

error_Rs�V?

learning_rate_1�O?7���I       6%�	5+�Y���A�&*;


total_lossm�!A

error_R�LS?

learning_rate_1�O?7�H{QI       6%�	�n�Y���A�&*;


total_loss��|@

error_R�8?

learning_rate_1�O?7L#:I       6%�	]��Y���A�&*;


total_loss��@

error_R�H?

learning_rate_1�O?7&qp�I       6%�	��Y���A�&*;


total_lossj��@

error_Rv,4?

learning_rate_1�O?7�7�I       6%�	�7�Y���A�&*;


total_lossX[�@

error_R.F?

learning_rate_1�O?7�� hI       6%�	�w�Y���A�&*;


total_loss.�@

error_R�AA?

learning_rate_1�O?7/ڹI       6%�	e��Y���A�&*;


total_loss�'�@

error_R�$T?

learning_rate_1�O?7� ��I       6%�	��Y���A�&*;


total_loss�A

error_R�\?

learning_rate_1�O?7��L�I       6%�	i<�Y���A�&*;


total_loss���@

error_RH�K?

learning_rate_1�O?7����I       6%�	�}�Y���A�&*;


total_lossO��@

error_R3,X?

learning_rate_1�O?7ʶ��I       6%�	ü�Y���A�&*;


total_lossS
!A

error_R!�P?

learning_rate_1�O?7J�]I       6%�	���Y���A�&*;


total_loss���@

error_R�c?

learning_rate_1�O?7�tI       6%�	�>�Y���A�&*;


total_lossｻ@

error_R;ZU?

learning_rate_1�O?7�,�I       6%�	
��Y���A�&*;


total_loss��@

error_R��a?

learning_rate_1�O?7ԴѬI       6%�	"��Y���A�&*;


total_lossFh�@

error_R?X?

learning_rate_1�O?7UL�kI       6%�	��Y���A�&*;


total_loss�5~@

error_R�cD?

learning_rate_1�O?7�M*I       6%�	 O�Y���A�&*;


total_lossC�@

error_R��M?

learning_rate_1�O?7�:�I       6%�	��Y���A�&*;


total_loss���@

error_R�LX?

learning_rate_1�O?7����I       6%�	���Y���A�&*;


total_loss ��@

error_R�B?

learning_rate_1�O?7����I       6%�	��Y���A�&*;


total_loss�[�@

error_R�2H?

learning_rate_1�O?7!D��I       6%�	�P�Y���A�&*;


total_loss�¤@

error_R8B?

learning_rate_1�O?7�}7�I       6%�	>��Y���A�&*;


total_loss%��@

error_R�Y?

learning_rate_1�O?7�X0PI       6%�	���Y���A�&*;


total_loss�fA

error_R��P?

learning_rate_1�O?7J�&I       6%�	��Y���A�&*;


total_loss��Q@

error_RMSP?

learning_rate_1�O?7,7w�I       6%�	M�Y���A�&*;


total_loss�4A

error_R��M?

learning_rate_1�O?7�p�/I       6%�	E��Y���A�&*;


total_loss5��@

error_Rn�D?

learning_rate_1�O?7�Vz�I       6%�	q��Y���A�&*;


total_loss���@

error_R�lL?

learning_rate_1�O?7�Y�I       6%�	d�Y���A�&*;


total_lossC�@

error_R�Q?

learning_rate_1�O?7%D�I       6%�	�O�Y���A�&*;


total_loss�b�@

error_R�S>?

learning_rate_1�O?7���I       6%�	��Y���A�&*;


total_lossi��@

error_R�e?

learning_rate_1�O?7�-�!I       6%�	���Y���A�&*;


total_loss��
A

error_R��A?

learning_rate_1�O?7h �I       6%�	s�Y���A�&*;


total_lossd�@

error_R<S?

learning_rate_1�O?7��ۄI       6%�	�T�Y���A�&*;


total_loss���@

error_RA�V?

learning_rate_1�O?7ThI       6%�	��Y���A�&*;


total_lossT.A

error_R�}7?

learning_rate_1�O?7��K~I       6%�	���Y���A�&*;


total_loss.��@

error_R��H?

learning_rate_1�O?7�`�I       6%�	�&�Y���A�&*;


total_lossx��@

error_R�KM?

learning_rate_1�O?7֮��I       6%�	�i�Y���A�&*;


total_loss���@

error_Rq�E?

learning_rate_1�O?7��/I       6%�	���Y���A�&*;


total_loss�ɰ@

error_R�0S?

learning_rate_1�O?7^�SI       6%�	P��Y���A�&*;


total_lossme�@

error_R��M?

learning_rate_1�O?7��l9I       6%�		0�Y���A�&*;


total_loss�m�@

error_R aF?

learning_rate_1�O?7�a��I       6%�	x�Y���A�&*;


total_loss��@

error_R�jF?

learning_rate_1�O?71�>I       6%�	x��Y���A�&*;


total_loss_W�@

error_R��O?

learning_rate_1�O?7/�:I       6%�	���Y���A�&*;


total_loss���@

error_RhO2?

learning_rate_1�O?7��^�I       6%�	S@�Y���A�&*;


total_loss���@

error_RH�U?

learning_rate_1�O?7Z�^�I       6%�	��Y���A�&*;


total_loss��@

error_RҋB?

learning_rate_1�O?7�?4I       6%�	���Y���A�&*;


total_loss���@

error_R��B?

learning_rate_1�O?7߶�oI       6%�	��Y���A�&*;


total_loss�>�@

error_R��C?

learning_rate_1�O?7���5I       6%�	E�Y���A�&*;


total_lossP�@

error_RR�[?

learning_rate_1�O?7�WL�I       6%�	��Y���A�&*;


total_loss�x�@

error_R��Q?

learning_rate_1�O?7�9,.I       6%�	��Y���A�&*;


total_loss=�@

error_RM�O?

learning_rate_1�O?7��պI       6%�	��Y���A�&*;


total_loss��@

error_R3oJ?

learning_rate_1�O?7L��gI       6%�	�J�Y���A�&*;


total_loss�t�@

error_R_AL?

learning_rate_1�O?7�7LI       6%�	߉�Y���A�&*;


total_losso}�@

error_RR�R?

learning_rate_1�O?7t�φI       6%�	h��Y���A�&*;


total_lossӻ�@

error_R�GX?

learning_rate_1�O?7�r4I       6%�	��Y���A�&*;


total_lossCs�@

error_RXcG?

learning_rate_1�O?7_4oI       6%�	�S�Y���A�&*;


total_loss4�A

error_RM�G?

learning_rate_1�O?7��BI       6%�	���Y���A�&*;


total_lossa.�@

error_R)A?

learning_rate_1�O?7�)��I       6%�	���Y���A�&*;


total_lossDŻ@

error_R$P?

learning_rate_1�O?7#���I       6%�	�3�Y���A�&*;


total_loss|�@

error_Rz)X?

learning_rate_1�O?7ϒ�WI       6%�	�v�Y���A�&*;


total_lossث@

error_R�_?

learning_rate_1�O?7�
�I       6%�	���Y���A�&*;


total_loss�-�@

error_RϠS?

learning_rate_1�O?7m<�GI       6%�	���Y���A�&*;


total_loss���@

error_R6K=?

learning_rate_1�O?7w]I       6%�	B�Y���A�&*;


total_losset�@

error_R��M?

learning_rate_1�O?7۸��I       6%�	f��Y���A�&*;


total_loss=R�@

error_R��W?

learning_rate_1�O?7yW�I       6%�	���Y���A�&*;


total_loss�z�@

error_R�@?

learning_rate_1�O?7Ũ��I       6%�	��Y���A�&*;


total_loss{�@

error_R�h?

learning_rate_1�O?7i!��I       6%�	�V�Y���A�&*;


total_loss-��@

error_R�`?

learning_rate_1�O?7�v��I       6%�	M��Y���A�&*;


total_loss��u@

error_R�#\?

learning_rate_1�O?7��e�I       6%�	,��Y���A�&*;


total_lossĵ�@

error_R:�G?

learning_rate_1�O?7M9�I       6%�	Q�Y���A�&*;


total_loss��A

error_RIIH?

learning_rate_1�O?7ķ2I       6%�	�]�Y���A�&*;


total_loss��A

error_R�M?

learning_rate_1�O?7����I       6%�	Ҟ�Y���A�&*;


total_lossў�@

error_R!�D?

learning_rate_1�O?7�E{I       6%�	��Y���A�&*;


total_loss]�@

error_R� E?

learning_rate_1�O?7�5N{I       6%�	V�Y���A�&*;


total_loss�.�@

error_R�??

learning_rate_1�O?71�.I       6%�	�\�Y���A�&*;


total_loss�8�@

error_R��I?

learning_rate_1�O?7��v�I       6%�	Ġ�Y���A�&*;


total_lossʫ�@

error_R�V?

learning_rate_1�O?79� @I       6%�	��Y���A�&*;


total_loss��@

error_RFD^?

learning_rate_1�O?7)�ӂI       6%�	/&�Y���A�&*;


total_loss�u�@

error_R��_?

learning_rate_1�O?7��0I       6%�	?h�Y���A�&*;


total_loss���@

error_R|�C?

learning_rate_1�O?7B�W�I       6%�	H��Y���A�&*;


total_lossVI�@

error_R��Z?

learning_rate_1�O?7��uI       6%�	@��Y���A�&*;


total_lossD��@

error_R1�L?

learning_rate_1�O?7I[�I       6%�	�.�Y���A�&*;


total_loss7<�@

error_R��I?

learning_rate_1�O?7�	��I       6%�	qs�Y���A�&*;


total_lossH�@

error_R 1I?

learning_rate_1�O?7E��I       6%�	���Y���A�&*;


total_loss���@

error_RcK?

learning_rate_1�O?7�(MwI       6%�	-��Y���A�&*;


total_loss���@

error_R�1T?

learning_rate_1�O?7�*z&I       6%�	2�Y���A�&*;


total_lossĭ�@

error_R�oP?

learning_rate_1�O?7��ܡI       6%�	�r�Y���A�&*;


total_loss���@

error_Rf�Y?

learning_rate_1�O?7}<ٕI       6%�	���Y���A�&*;


total_loss�@

error_R��??

learning_rate_1�O?7#��-I       6%�	B��Y���A�&*;


total_loss?�L@

error_R�_R?

learning_rate_1�O?7��I       6%�	�9�Y���A�&*;


total_loss���@

error_R��D?

learning_rate_1�O?7�jS:I       6%�	ͅ�Y���A�&*;


total_loss��@

error_R��>?

learning_rate_1�O?7�M�I       6%�	���Y���A�&*;


total_loss��A

error_R3�V?

learning_rate_1�O?7J�I       6%�	��Y���A�&*;


total_loss�e�@

error_R��>?

learning_rate_1�O?7Z1[�I       6%�	�K�Y���A�&*;


total_loss��@

error_R��_?

learning_rate_1�O?7`k#I       6%�	��Y���A�&*;


total_loss�ݡ@

error_Rn,[?

learning_rate_1�O?7�>�I       6%�	���Y���A�&*;


total_loss�A

error_RFG?

learning_rate_1�O?7>4�I       6%�		�Y���A�&*;


total_loss��@

error_R��D?

learning_rate_1�O?7��GI       6%�	�L�Y���A�&*;


total_loss���@

error_R�qU?

learning_rate_1�O?7����I       6%�	��Y���A�&*;


total_loss���@

error_R�C?

learning_rate_1�O?7�؂�I       6%�	��Y���A�&*;


total_lossd�@

error_R_~O?

learning_rate_1�O?7�E�7I       6%�	��Y���A�&*;


total_loss,5�@

error_R[�e?

learning_rate_1�O?7`�?:I       6%�		V�Y���A�&*;


total_loss���@

error_RID??

learning_rate_1�O?7���MI       6%�	ԓ�Y���A�&*;


total_loss�O�@

error_R!�:?

learning_rate_1�O?7��TI       6%�	���Y���A�'*;


total_loss��@

error_R�sa?

learning_rate_1�O?7�m)XI       6%�	8�Y���A�'*;


total_lossB��@

error_R1FU?

learning_rate_1�O?7���rI       6%�	jW�Y���A�'*;


total_loss=d�@

error_R%9V?

learning_rate_1�O?7�I       6%�	ݝ�Y���A�'*;


total_loss!v�@

error_R�9_?

learning_rate_1�O?7B]W�I       6%�	���Y���A�'*;


total_lossDDe@

error_R�L?

learning_rate_1�O?7-Ċ I       6%�	�"�Y���A�'*;


total_lossh�@

error_R@^i?

learning_rate_1�O?7� zCI       6%�	�l�Y���A�'*;


total_loss���@

error_R�]D?

learning_rate_1�O?7�CcI       6%�	���Y���A�'*;


total_loss�4�@

error_R�T?

learning_rate_1�O?7�+��I       6%�	�	�Y���A�'*;


total_lossiW�@

error_R�Y?

learning_rate_1�O?7�z[�I       6%�	�|�Y���A�'*;


total_loss�d�@

error_R,�L?

learning_rate_1�O?7
��I       6%�	[��Y���A�'*;


total_loss�A

error_R��Y?

learning_rate_1�O?7{��	I       6%�	� �Y���A�'*;


total_loss�@�@

error_R��@?

learning_rate_1�O?7�j��I       6%�	�D�Y���A�'*;


total_loss��@

error_R��L?

learning_rate_1�O?73��I       6%�	���Y���A�'*;


total_loss9k�@

error_R�ZO?

learning_rate_1�O?7l��"I       6%�	)��Y���A�'*;


total_loss8�@

error_R�CK?

learning_rate_1�O?7�\�}I       6%�	��Y���A�'*;


total_loss���@

error_R�M?

learning_rate_1�O?7�1��I       6%�	F�Y���A�'*;


total_loss�a�@

error_R�/S?

learning_rate_1�O?7��%I       6%�	[��Y���A�'*;


total_loss]�@

error_R.�L?

learning_rate_1�O?7�ph�I       6%�	-��Y���A�'*;


total_loss/_�@

error_Rq�[?

learning_rate_1�O?7-��I       6%�	0�Y���A�'*;


total_loss-6�@

error_R�BC?

learning_rate_1�O?7g���I       6%�	�O�Y���A�'*;


total_loss�1�@

error_RCC?

learning_rate_1�O?7��I       6%�	j��Y���A�'*;


total_loss:��@

error_R��Z?

learning_rate_1�O?7HS�I       6%�	���Y���A�'*;


total_loss�֦@

error_R%�F?

learning_rate_1�O?7��)I       6%�	= �Y���A�'*;


total_loss&2�@

error_R{�Q?

learning_rate_1�O?7p���I       6%�	�e�Y���A�'*;


total_loss<��@

error_RSjV?

learning_rate_1�O?7� �I       6%�	!��Y���A�'*;


total_lossO_�@

error_R�_?

learning_rate_1�O?7O��QI       6%�	Q��Y���A�'*;


total_losse	�@

error_R�K?

learning_rate_1�O?7&fi�I       6%�	 )�Y���A�'*;


total_loss���@

error_R�Z\?

learning_rate_1�O?7=84II       6%�	�h�Y���A�'*;


total_loss��@

error_R]?

learning_rate_1�O?79��ZI       6%�	���Y���A�'*;


total_loss��@

error_RWK?

learning_rate_1�O?72�I       6%�	��Y���A�'*;


total_loss�A

error_R�vN?

learning_rate_1�O?7h.�I       6%�	�'�Y���A�'*;


total_loss y�@

error_R�N?

learning_rate_1�O?7���I       6%�	j�Y���A�'*;


total_loss3��@

error_Rn	N?

learning_rate_1�O?7cŨ�I       6%�	U��Y���A�'*;


total_loss��@

error_R)N?

learning_rate_1�O?7�D�DI       6%�	���Y���A�'*;


total_lossA��@

error_R�FN?

learning_rate_1�O?7V=��I       6%�	6�Y���A�'*;


total_loss��@

error_R2�J?

learning_rate_1�O?7���I       6%�	*}�Y���A�'*;


total_loss�m�@

error_R�IR?

learning_rate_1�O?7�^��I       6%�	���Y���A�'*;


total_loss0�@

error_R	P?

learning_rate_1�O?7f�=QI       6%�	��Y���A�'*;


total_loss���@

error_R��J?

learning_rate_1�O?7cI�I       6%�	}D�Y���A�'*;


total_loss-�@

error_R�D?

learning_rate_1�O?7:Y[I       6%�	3��Y���A�'*;


total_loss�2�@

error_R!H?

learning_rate_1�O?7ފ�CI       6%�	���Y���A�'*;


total_loss�u�@

error_R[�S?

learning_rate_1�O?7�5�#I       6%�	�	�Y���A�'*;


total_lossZ��@

error_R6�V?

learning_rate_1�O?7zcI       6%�	#M�Y���A�'*;


total_loss��s@

error_R%�U?

learning_rate_1�O?7��G�I       6%�	Ռ�Y���A�'*;


total_loss�a�@

error_R�-N?

learning_rate_1�O?7�
��I       6%�	T��Y���A�'*;


total_loss�T�@

error_R�kH?

learning_rate_1�O?7�7NI       6%�	q�Y���A�'*;


total_lossBW�@

error_R<O?

learning_rate_1�O?7W$/I       6%�	�M�Y���A�'*;


total_lossl$�@

error_Rv�Q?

learning_rate_1�O?7���^I       6%�	���Y���A�'*;


total_loss@

error_RσK?

learning_rate_1�O?7zʛ�I       6%�	���Y���A�'*;


total_loss���@

error_R�E?

learning_rate_1�O?7�6X-I       6%�	�(�Y���A�'*;


total_loss35�@

error_Ra�R?

learning_rate_1�O?7e�;I       6%�	���Y���A�'*;


total_loss�}[@

error_R��X?

learning_rate_1�O?7��I       6%�	J��Y���A�'*;


total_loss��@

error_R��d?

learning_rate_1�O?7�Գ�I       6%�	;	�Y���A�'*;


total_loss�CA

error_RWZ?

learning_rate_1�O?7���WI       6%�	Z}�Y���A�'*;


total_loss�F�@

error_R��E?

learning_rate_1�O?7K�I       6%�	^��Y���A�'*;


total_loss2��@

error_R��W?

learning_rate_1�O?7�.=YI       6%�	4;�Y���A�'*;


total_loss-�@

error_R��I?

learning_rate_1�O?7R?M�I       6%�	��Y���A�'*;


total_lossE�@

error_R]�G?

learning_rate_1�O?7�h��I       6%�	���Y���A�'*;


total_loss{>�@

error_R�S?

learning_rate_1�O?7�S��I       6%�	C�Y���A�'*;


total_loss?��@

error_RfoF?

learning_rate_1�O?7H?̓I       6%�	���Y���A�'*;


total_loss��@

error_R�U?

learning_rate_1�O?7�ƓI       6%�	�$�Y���A�'*;


total_loss$'�@

error_Rx�H?

learning_rate_1�O?7���I       6%�	�j�Y���A�'*;


total_lossv
�@

error_RlhP?

learning_rate_1�O?7oP�I       6%�	'��Y���A�'*;


total_loss��@

error_R��F?

learning_rate_1�O?7ְ�I       6%�	w��Y���A�'*;


total_loss$&�@

error_R/�B?

learning_rate_1�O?7���I       6%�	�2�Y���A�'*;


total_loss/�@

error_R�Q?

learning_rate_1�O?7�a}MI       6%�	�|�Y���A�'*;


total_losse+�@

error_R�H?

learning_rate_1�O?70#�WI       6%�	��Y���A�'*;


total_lossL�@

error_RW�\?

learning_rate_1�O?7��1PI       6%�	�#�Y���A�'*;


total_loss1�@

error_R:`?

learning_rate_1�O?7[NI       6%�	�h�Y���A�'*;


total_loss�@

error_RxI?

learning_rate_1�O?7D�I       6%�	|��Y���A�'*;


total_loss�.�@

error_RH]d?

learning_rate_1�O?7�L^dI       6%�	���Y���A�'*;


total_loss,�A

error_Rl�g?

learning_rate_1�O?7��bI       6%�	�6�Y���A�'*;


total_loss6�0A

error_Rѳ@?

learning_rate_1�O?7���I       6%�	�u�Y���A�'*;


total_lossxO�@

error_R,�M?

learning_rate_1�O?7��scI       6%�	���Y���A�'*;


total_loss�a�@

error_R��I?

learning_rate_1�O?7�,I       6%�	^��Y���A�'*;


total_loss�>�@

error_R�wG?

learning_rate_1�O?7��b�I       6%�	(? Z���A�'*;


total_loss@��@

error_R��Y?

learning_rate_1�O?7i�GI       6%�	c� Z���A�'*;


total_loss\ �@

error_R}ic?

learning_rate_1�O?7/_�jI       6%�	�� Z���A�'*;


total_lossve�@

error_R$�D?

learning_rate_1�O?7;��I       6%�	 	Z���A�'*;


total_lossq��@

error_R�}.?

learning_rate_1�O?7 ���I       6%�	#NZ���A�'*;


total_loss��@

error_Rq�Y?

learning_rate_1�O?7�y�3I       6%�	�Z���A�'*;


total_loss�ʁ@

error_RvqO?

learning_rate_1�O?7`
s_I       6%�	a�Z���A�'*;


total_lossn��@

error_R� 8?

learning_rate_1�O?7&��I       6%�	Z���A�'*;


total_lossۨ�@

error_Rc S?

learning_rate_1�O?7�/I       6%�	VZ���A�'*;


total_loss؇�@

error_RV�Y?

learning_rate_1�O?7:�I       6%�	�Z���A�'*;


total_loss�Z	A

error_R$\M?

learning_rate_1�O?7h&�GI       6%�	��Z���A�'*;


total_loss�R�@

error_R:�F?

learning_rate_1�O?7�$I       6%�	�Z���A�'*;


total_loss:t�@

error_R��U?

learning_rate_1�O?7N�xxI       6%�	/UZ���A�'*;


total_lossyxA

error_RغO?

learning_rate_1�O?7!/�I       6%�	K�Z���A�'*;


total_loss���@

error_R�M?

learning_rate_1�O?7�@�I       6%�	��Z���A�'*;


total_loss�@

error_R�*Z?

learning_rate_1�O?7~C��I       6%�	
Z���A�'*;


total_loss*�@

error_R)&N?

learning_rate_1�O?7߬3�I       6%�	�WZ���A�'*;


total_loss2�@

error_Rc;a?

learning_rate_1�O?7�&I       6%�	,�Z���A�'*;


total_lossa.�@

error_Rv�[?

learning_rate_1�O?7�	��I       6%�	��Z���A�'*;


total_loss\�@

error_R�??

learning_rate_1�O?7Ai��I       6%�	?Z���A�'*;


total_loss��@

error_R��f?

learning_rate_1�O?7猲�I       6%�	�[Z���A�'*;


total_losse�@

error_R�#F?

learning_rate_1�O?7�f�@I       6%�	n�Z���A�'*;


total_loss��@

error_R��c?

learning_rate_1�O?7�C�rI       6%�	�Z���A�'*;


total_loss�ס@

error_R�P?

learning_rate_1�O?7e�>I       6%�	WZ���A�'*;


total_loss�ٯ@

error_RM�K?

learning_rate_1�O?7źU�I       6%�	аZ���A�'*;


total_lossڷ�@

error_R�
N?

learning_rate_1�O?7y6OdI       6%�	��Z���A�'*;


total_loss��@

error_R�:a?

learning_rate_1�O?7Y�/�I       6%�	�6Z���A�'*;


total_lossV4x@

error_R�	A?

learning_rate_1�O?7�c�I       6%�	"}Z���A�'*;


total_loss$�@

error_R�EA?

learning_rate_1�O?7�hgI       6%�	ԿZ���A�'*;


total_lossA�@

error_RY?

learning_rate_1�O?7^��:I       6%�	�Z���A�'*;


total_lossqN�@

error_R��P?

learning_rate_1�O?7���	I       6%�	�LZ���A�'*;


total_loss��j@

error_RϪH?

learning_rate_1�O?7L[�I       6%�	��Z���A�'*;


total_loss��@

error_R�gN?

learning_rate_1�O?7��QI       6%�	��Z���A�'*;


total_loss F�@

error_R�K?

learning_rate_1�O?7ȻXI       6%�	�	Z���A�'*;


total_lossB�@

error_R��V?

learning_rate_1�O?7y0&I       6%�	U�	Z���A�'*;


total_lossc��@

error_R��F?

learning_rate_1�O?7ڌl�I       6%�	��	Z���A�'*;


total_loss|�@

error_R�;H?

learning_rate_1�O?7����I       6%�	�(
Z���A�'*;


total_lossA�@

error_R�\I?

learning_rate_1�O?7�8*�I       6%�	�n
Z���A�'*;


total_lossѺ{@

error_R�G?

learning_rate_1�O?7H�gI       6%�	�
Z���A�'*;


total_loss0�@

error_RW\?

learning_rate_1�O?7��X�I       6%�	Z���A�'*;


total_loss<\�@

error_Rf�U?

learning_rate_1�O?7�?4I       6%�	7vZ���A�'*;


total_loss��@

error_R�cA?

learning_rate_1�O?72q��I       6%�	p�Z���A�'*;


total_lossoA

error_R79U?

learning_rate_1�O?7��&7I       6%�	'Z���A�'*;


total_loss{@

error_R_BS?

learning_rate_1�O?7��$I       6%�	�TZ���A�'*;


total_loss q�@

error_R��9?

learning_rate_1�O?7-���I       6%�	P�Z���A�'*;


total_loss�@

error_Rq+G?

learning_rate_1�O?7�6 I       6%�	��Z���A�'*;


total_loss@)o@

error_R��R?

learning_rate_1�O?7M�;FI       6%�	�Z���A�'*;


total_loss���@

error_R/�H?

learning_rate_1�O?7��;I       6%�	9\Z���A�'*;


total_lossh)�@

error_R6?

learning_rate_1�O?7p�A(I       6%�	��Z���A�'*;


total_loss$�@

error_R��c?

learning_rate_1�O?7�%^1I       6%�	h�Z���A�'*;


total_loss�@

error_R3�P?

learning_rate_1�O?76�I       6%�	k*Z���A�'*;


total_lossR��@

error_R�6T?

learning_rate_1�O?7/b��I       6%�	nZ���A�'*;


total_loss�@

error_RW�O?

learning_rate_1�O?7tOV�I       6%�	)�Z���A�(*;


total_loss_d�@

error_RϲI?

learning_rate_1�O?7��jI       6%�	(�Z���A�(*;


total_loss/*�@

error_R�QD?

learning_rate_1�O?7��I       6%�	jGZ���A�(*;


total_loss��y@

error_R�tO?

learning_rate_1�O?7��gI       6%�	��Z���A�(*;


total_loss Թ@

error_R8D:?

learning_rate_1�O?7ƽ�I       6%�	+�Z���A�(*;


total_loss�8�@

error_R�]?

learning_rate_1�O?7pmw�I       6%�	�Z���A�(*;


total_loss�m�@

error_RvZ?

learning_rate_1�O?7����I       6%�	�UZ���A�(*;


total_lossŐ�@

error_RO�W?

learning_rate_1�O?7��(�I       6%�	�Z���A�(*;


total_loss�A

error_R�E?

learning_rate_1�O?7�@ȤI       6%�	8�Z���A�(*;


total_loss���@

error_R�O?

learning_rate_1�O?7|��BI       6%�	QZ���A�(*;


total_loss���@

error_RM�o?

learning_rate_1�O?7�U	\I       6%�	�ZZ���A�(*;


total_loss��@

error_R!fK?

learning_rate_1�O?7�5!I       6%�	Q�Z���A�(*;


total_loss�μ@

error_R&�G?

learning_rate_1�O?7�q�I       6%�	��Z���A�(*;


total_lossG�@

error_R,�M?

learning_rate_1�O?7L;\I       6%�	�!Z���A�(*;


total_loss���@

error_R�dT?

learning_rate_1�O?7mǠ�I       6%�	cZ���A�(*;


total_loss3E�@

error_Rf�K?

learning_rate_1�O?7n�|)I       6%�	�Z���A�(*;


total_loss�A

error_R�]?

learning_rate_1�O?7r�)�I       6%�	[�Z���A�(*;


total_loss�*�@

error_R�~V?

learning_rate_1�O?7'���I       6%�	o/Z���A�(*;


total_loss,�@

error_RJ.N?

learning_rate_1�O?7ɃI       6%�	:qZ���A�(*;


total_lossc�@

error_R$tS?

learning_rate_1�O?7��i�I       6%�	e�Z���A�(*;


total_loss���@

error_RMUU?

learning_rate_1�O?7�B��I       6%�	w�Z���A�(*;


total_loss�#�@

error_RGA?

learning_rate_1�O?7�+ʬI       6%�	7Z���A�(*;


total_loss�9�@

error_R �B?

learning_rate_1�O?7ڽlI       6%�	jwZ���A�(*;


total_loss�:A

error_RןS?

learning_rate_1�O?7jL�I       6%�	ͶZ���A�(*;


total_loss���@

error_R�V?

learning_rate_1�O?7u���I       6%�		�Z���A�(*;


total_loss���@

error_RZD??

learning_rate_1�O?79�,�I       6%�	y:Z���A�(*;


total_loss�U�@

error_R#�X?

learning_rate_1�O?7}0�I       6%�	�Z���A�(*;


total_loss�$N@

error_R�"E?

learning_rate_1�O?7��bI       6%�	�Z���A�(*;


total_lossI�@

error_R�C?

learning_rate_1�O?7�s�&I       6%�	�	Z���A�(*;


total_loss@)�@

error_R��K?

learning_rate_1�O?7�PȞI       6%�	�JZ���A�(*;


total_loss4�[@

error_Rc�O?

learning_rate_1�O?7&R{:I       6%�	N�Z���A�(*;


total_loss�Ӯ@

error_Rm�P?

learning_rate_1�O?73�7I       6%�	;�Z���A�(*;


total_lossӁ@

error_R� C?

learning_rate_1�O?7�X�I       6%�	�	Z���A�(*;


total_loss���@

error_RTsF?

learning_rate_1�O?7�|�I       6%�	�KZ���A�(*;


total_loss&��@

error_R�}W?

learning_rate_1�O?7�V�I       6%�	S�Z���A�(*;


total_loss�U�@

error_R�O?

learning_rate_1�O?7�s�I       6%�	D�Z���A�(*;


total_loss`S�@

error_R!�J?

learning_rate_1�O?7$��MI       6%�	Z���A�(*;


total_loss�7�@

error_Rq�Y?

learning_rate_1�O?7���I       6%�	QZ���A�(*;


total_loss�3A

error_R�L?

learning_rate_1�O?7X]~_I       6%�	�Z���A�(*;


total_loss�_�@

error_R��^?

learning_rate_1�O?77�I       6%�	��Z���A�(*;


total_loss�fW@

error_RE�<?

learning_rate_1�O?7*!VI       6%�	 Z���A�(*;


total_loss���@

error_R��>?

learning_rate_1�O?7ү�MI       6%�	�QZ���A�(*;


total_loss�M�@

error_R4OQ?

learning_rate_1�O?7�b3�I       6%�	�Z���A�(*;


total_loss�O@

error_RD�T?

learning_rate_1�O?7
�qI       6%�	��Z���A�(*;


total_lossnN�@

error_R�c?

learning_rate_1�O?7Я$�I       6%�	�Z���A�(*;


total_loss���@

error_RH�O?

learning_rate_1�O?7�yaCI       6%�	�[Z���A�(*;


total_loss��@

error_R!�H?

learning_rate_1�O?7�'��I       6%�	F�Z���A�(*;


total_loss�Ƭ@

error_R�<?

learning_rate_1�O?7�d�I       6%�	-�Z���A�(*;


total_loss�fLA

error_R�K?

learning_rate_1�O?7��'XI       6%�	�Z���A�(*;


total_loss=�@

error_Rm�M?

learning_rate_1�O?7�57�I       6%�	�jZ���A�(*;


total_loss���@

error_R��M?

learning_rate_1�O?7A� I       6%�	��Z���A�(*;


total_loss�ƌ@

error_R7fT?

learning_rate_1�O?7\��NI       6%�	6Z���A�(*;


total_loss���@

error_R�MH?

learning_rate_1�O?7�'�I       6%�	LZ���A�(*;


total_loss�R�@

error_R�T?

learning_rate_1�O?7��F?I       6%�	/�Z���A�(*;


total_loss �@

error_R r1?

learning_rate_1�O?7���I       6%�	S�Z���A�(*;


total_loss���@

error_R��c?

learning_rate_1�O?7r��OI       6%�	�Z���A�(*;


total_loss�3�@

error_R��Q?

learning_rate_1�O?7��I       6%�	W^Z���A�(*;


total_lossѺ�@

error_RsLa?

learning_rate_1�O?7��/I       6%�	��Z���A�(*;


total_loss�V�@

error_R��P?

learning_rate_1�O?7��!�I       6%�	�Z���A�(*;


total_loss��@

error_R�KP?

learning_rate_1�O?7X��+I       6%�	�1Z���A�(*;


total_loss�+A

error_RR�U?

learning_rate_1�O?7�'��I       6%�	Z{Z���A�(*;


total_loss�A�@

error_R;�@?

learning_rate_1�O?7e��I       6%�	žZ���A�(*;


total_loss�D�@

error_R��L?

learning_rate_1�O?7���I       6%�	�Z���A�(*;


total_loss܇�@

error_RWkY?

learning_rate_1�O?7i>�I       6%�	�EZ���A�(*;


total_loss���@

error_R�T?

learning_rate_1�O?7���I       6%�	�Z���A�(*;


total_loss ��@

error_RҹT?

learning_rate_1�O?7%� �I       6%�	��Z���A�(*;


total_loss���@

error_R_I?

learning_rate_1�O?7�4?I       6%�	T Z���A�(*;


total_loss���@

error_Rh�L?

learning_rate_1�O?7K�oI       6%�	2P Z���A�(*;


total_lossw �@

error_RWRL?

learning_rate_1�O?7��XI       6%�	X� Z���A�(*;


total_loss���@

error_R�]?

learning_rate_1�O?7Hj�I       6%�	;� Z���A�(*;


total_loss�A

error_R�O?

learning_rate_1�O?7=�PI       6%�	�!Z���A�(*;


total_lossj��@

error_R]T?

learning_rate_1�O?7b/0�I       6%�	`!Z���A�(*;


total_loss��@

error_RxGP?

learning_rate_1�O?7��3I       6%�	��!Z���A�(*;


total_loss�!�@

error_R=�>?

learning_rate_1�O?77(�I       6%�	�"Z���A�(*;


total_lossWA�@

error_R�TQ?

learning_rate_1�O?7���KI       6%�	�S"Z���A�(*;


total_loss�4�@

error_R#TP?

learning_rate_1�O?7)�I       6%�	�"Z���A�(*;


total_loss�i�@

error_R/%X?

learning_rate_1�O?7c��jI       6%�	�#Z���A�(*;


total_loss4�A

error_R��X?

learning_rate_1�O?7�ՙI       6%�	�R#Z���A�(*;


total_loss��@

error_R�OO?

learning_rate_1�O?7q��I       6%�	[�#Z���A�(*;


total_loss���@

error_R�L?

learning_rate_1�O?7�X��I       6%�	��#Z���A�(*;


total_lossd��@

error_RܰA?

learning_rate_1�O?7�\��I       6%�	S"$Z���A�(*;


total_loss#'�@

error_R��E?

learning_rate_1�O?7���BI       6%�	�f$Z���A�(*;


total_lossԩA

error_R��S?

learning_rate_1�O?7�H��I       6%�	>�$Z���A�(*;


total_loss�N�@

error_R�5F?

learning_rate_1�O?7��i�I       6%�	��$Z���A�(*;


total_loss��@

error_R�IJ?

learning_rate_1�O?7�3�2I       6%�	�=%Z���A�(*;


total_lossFܫ@

error_Re??

learning_rate_1�O?7��'�I       6%�	��%Z���A�(*;


total_lossF�@

error_R�X?

learning_rate_1�O?7a,XMI       6%�	��%Z���A�(*;


total_loss�A

error_R�'B?

learning_rate_1�O?7z�~�I       6%�	�&Z���A�(*;


total_lossa�@

error_R_�W?

learning_rate_1�O?7BC~I       6%�	GS&Z���A�(*;


total_loss��A

error_RD?

learning_rate_1�O?7Ǉ��I       6%�		�&Z���A�(*;


total_loss���@

error_R\�^?

learning_rate_1�O?7ɸ��I       6%�	��&Z���A�(*;


total_loss�!�@

error_ROH?

learning_rate_1�O?7vT�CI       6%�	#'Z���A�(*;


total_loss�ė@

error_R;!F?

learning_rate_1�O?7-��YI       6%�	j'Z���A�(*;


total_loss<�@

error_Rd�\?

learning_rate_1�O?7#z�kI       6%�	m�'Z���A�(*;


total_loss�̓@

error_R@KT?

learning_rate_1�O?7�{�WI       6%�	7�'Z���A�(*;


total_losst�A

error_Rn!G?

learning_rate_1�O?7Q���I       6%�	�.(Z���A�(*;


total_loss�,�@

error_R�rP?

learning_rate_1�O?7\D�I       6%�	�n(Z���A�(*;


total_loss�A�@

error_RʙB?

learning_rate_1�O?7�ikpI       6%�	 �(Z���A�(*;


total_loss�z�@

error_Rf�G?

learning_rate_1�O?7n��I       6%�	'�(Z���A�(*;


total_loss[��@

error_RI�H?

learning_rate_1�O?7�j�I       6%�	s3)Z���A�(*;


total_loss�X�@

error_Rl�c?

learning_rate_1�O?7m
��I       6%�	�v)Z���A�(*;


total_loss@��@

error_R�V?

learning_rate_1�O?7f �FI       6%�	&�)Z���A�(*;


total_lossv*A

error_R�AR?

learning_rate_1�O?7�5�I       6%�	�*Z���A�(*;


total_lossIc�@

error_R��S?

learning_rate_1�O?7P�WI       6%�	�K*Z���A�(*;


total_loss�;�@

error_R��P?

learning_rate_1�O?7夌�I       6%�	c�*Z���A�(*;


total_loss��@

error_R��V?

learning_rate_1�O?7��)�I       6%�	f�*Z���A�(*;


total_loss%E�@

error_R��C?

learning_rate_1�O?7�̷I       6%�	�+Z���A�(*;


total_loss�D�@

error_RÃE?

learning_rate_1�O?7<oFdI       6%�	�b+Z���A�(*;


total_lossoޔ@

error_R��M?

learning_rate_1�O?78��I       6%�	��+Z���A�(*;


total_loss4�@

error_R`�d?

learning_rate_1�O?7�sI       6%�	��+Z���A�(*;


total_loss$�@

error_R�pQ?

learning_rate_1�O?7(DI       6%�	kA,Z���A�(*;


total_loss�<�@

error_R.Hg?

learning_rate_1�O?75:I       6%�	�,Z���A�(*;


total_loss���@

error_R�vL?

learning_rate_1�O?7���I       6%�	��,Z���A�(*;


total_loss���@

error_R�g??

learning_rate_1�O?7�Rl�I       6%�	s-Z���A�(*;


total_loss���@

error_RT?

learning_rate_1�O?7m���I       6%�	�M-Z���A�(*;


total_loss&��@

error_R�	_?

learning_rate_1�O?7�辋I       6%�	C�-Z���A�(*;


total_loss��@

error_R�C?

learning_rate_1�O?7�&>�I       6%�	��-Z���A�(*;


total_loss�@

error_RJ?

learning_rate_1�O?709S�I       6%�	e.Z���A�(*;


total_loss��@

error_R�u7?

learning_rate_1�O?7��e�I       6%�	eR.Z���A�(*;


total_loss��@

error_RhY?

learning_rate_1�O?7�p�SI       6%�	��.Z���A�(*;


total_lossR>�@

error_RHHE?

learning_rate_1�O?7�fmI       6%�	]�.Z���A�(*;


total_lossLQ�@

error_Rh�G?

learning_rate_1�O?7�+!�I       6%�	A/Z���A�(*;


total_loss��@

error_R�5[?

learning_rate_1�O?7���I       6%�	]/Z���A�(*;


total_loss���@

error_R�E?

learning_rate_1�O?7Q'��I       6%�	m�/Z���A�(*;


total_lossr��@

error_R��\?

learning_rate_1�O?7{襛I       6%�	�/Z���A�(*;


total_lossj�@

error_R/mG?

learning_rate_1�O?72�R�I       6%�	�/0Z���A�(*;


total_lossϵ�@

error_R�YO?

learning_rate_1�O?71�EMI       6%�	�u0Z���A�(*;


total_loss�:�@

error_R�J?

learning_rate_1�O?7����I       6%�	��0Z���A�(*;


total_loss.9�@

error_RRG?

learning_rate_1�O?7�L�I       6%�	��0Z���A�)*;


total_loss���@

error_R�J?

learning_rate_1�O?7�=�I       6%�	�B1Z���A�)*;


total_loss߸�@

error_R|�@?

learning_rate_1�O?7�AI       6%�	��1Z���A�)*;


total_lossH'�@

error_R�S?

learning_rate_1�O?7�n��I       6%�	��1Z���A�)*;


total_loss�3�@

error_R�"Z?

learning_rate_1�O?7'���I       6%�	.2Z���A�)*;


total_loss�h�@

error_Rv4?

learning_rate_1�O?7�l�I       6%�	�R2Z���A�)*;


total_loss��@

error_RakM?

learning_rate_1�O?7��`
I       6%�	��2Z���A�)*;


total_loss?1�@

error_RAH]?

learning_rate_1�O?7We�I       6%�	��2Z���A�)*;


total_loss&�@

error_R
]?

learning_rate_1�O?7&.�I       6%�	D3Z���A�)*;


total_loss�'�@

error_R��;?

learning_rate_1�O?7�r�@I       6%�	N^3Z���A�)*;


total_loss/��@

error_Rs�O?

learning_rate_1�O?7���I       6%�	��3Z���A�)*;


total_lossZ��@

error_R�pE?

learning_rate_1�O?7�Z]7I       6%�	��3Z���A�)*;


total_loss�TA

error_R��]?

learning_rate_1�O?7
4n	I       6%�	�$4Z���A�)*;


total_loss�P�@

error_RJjH?

learning_rate_1�O?7{�0�I       6%�	�h4Z���A�)*;


total_loss.��@

error_R�_O?

learning_rate_1�O?7-i?�I       6%�	%�4Z���A�)*;


total_losss/�@

error_R��X?

learning_rate_1�O?7�{��I       6%�	�4Z���A�)*;


total_loss�P�@

error_RF�W?

learning_rate_1�O?7��D=I       6%�	�)5Z���A�)*;


total_loss{�@

error_R��P?

learning_rate_1�O?7>�3I       6%�	>l5Z���A�)*;


total_loss,J�@

error_R��I?

learning_rate_1�O?7'��I       6%�	f�5Z���A�)*;


total_loss%f�@

error_R�e?

learning_rate_1�O?7ߡ�@I       6%�	�5Z���A�)*;


total_lossVǬ@

error_R{�H?

learning_rate_1�O?7i��I       6%�	�-6Z���A�)*;


total_loss���@

error_R�N?

learning_rate_1�O?7��t�I       6%�	�o6Z���A�)*;


total_loss}��@

error_R��5?

learning_rate_1�O?7Y�:�I       6%�	�6Z���A�)*;


total_loss
��@

error_R;mH?

learning_rate_1�O?7��BI       6%�	 �6Z���A�)*;


total_loss�|�@

error_R[{U?

learning_rate_1�O?7��͆I       6%�	#37Z���A�)*;


total_loss�A

error_R�vE?

learning_rate_1�O?7�2��I       6%�	Lt7Z���A�)*;


total_lossQ �@

error_R�U?

learning_rate_1�O?7\HT�I       6%�	�7Z���A�)*;


total_lossz�@

error_R�[b?

learning_rate_1�O?7�"�I       6%�	��7Z���A�)*;


total_lossw�@

error_R�bZ?

learning_rate_1�O?7ϭAI       6%�	�E8Z���A�)*;


total_lossd��@

error_R�OP?

learning_rate_1�O?7ǟR�I       6%�	�8Z���A�)*;


total_loss�,�@

error_R&�I?

learning_rate_1�O?7<p
�I       6%�	e�8Z���A�)*;


total_loss��@

error_R%�C?

learning_rate_1�O?7w�I       6%�	g9Z���A�)*;


total_loss�`A

error_RE/L?

learning_rate_1�O?7�`�bI       6%�	�T9Z���A�)*;


total_loss���@

error_R�J?

learning_rate_1�O?7�e$�I       6%�	�9Z���A�)*;


total_loss�g�@

error_R.�??

learning_rate_1�O?7 �5I       6%�	�9Z���A�)*;


total_loss{p�@

error_R��5?

learning_rate_1�O?7m��I       6%�	,!:Z���A�)*;


total_loss D�@

error_R�O?

learning_rate_1�O?7�BI       6%�	�y:Z���A�)*;


total_lossj��@

error_RZL?

learning_rate_1�O?7l�!�I       6%�	Y�:Z���A�)*;


total_lossJ
�@

error_R�c?

learning_rate_1�O?7>�5]I       6%�	@;Z���A�)*;


total_loss��@

error_R�}R?

learning_rate_1�O?7�m�JI       6%�	2M;Z���A�)*;


total_loss�@

error_RE@N?

learning_rate_1�O?7�FI       6%�	��;Z���A�)*;


total_lossNa�@

error_R�1T?

learning_rate_1�O?7��]qI       6%�	�<Z���A�)*;


total_lossڲ�@

error_R�B@?

learning_rate_1�O?7˱��I       6%�	�I<Z���A�)*;


total_lossH@Z@

error_RZy<?

learning_rate_1�O?7���I       6%�	'�<Z���A�)*;


total_loss�f�@

error_R�g?

learning_rate_1�O?7X4u�I       6%�	��<Z���A�)*;


total_loss���@

error_R��E?

learning_rate_1�O?7��kMI       6%�	�=Z���A�)*;


total_loss���@

error_R=�c?

learning_rate_1�O?7*�_�I       6%�	�Y=Z���A�)*;


total_loss���@

error_R�a?

learning_rate_1�O?7:)�I       6%�	�=Z���A�)*;


total_loss�,9A

error_R��M?

learning_rate_1�O?7����I       6%�	��=Z���A�)*;


total_loss���@

error_R�P?

learning_rate_1�O?7M�.I       6%�	)>Z���A�)*;


total_loss�@

error_R�)R?

learning_rate_1�O?7���tI       6%�	<q>Z���A�)*;


total_loss�B�@

error_R΂B?

learning_rate_1�O?7���wI       6%�	��>Z���A�)*;


total_loss���@

error_R��R?

learning_rate_1�O?7�,��I       6%�	��>Z���A�)*;


total_loss�%A

error_R߼M?

learning_rate_1�O?7ِ�4I       6%�	uD?Z���A�)*;


total_loss���@

error_R_"M?

learning_rate_1�O?7�BI       6%�	��?Z���A�)*;


total_loss���@

error_R�~Q?

learning_rate_1�O?7�r9I       6%�	��?Z���A�)*;


total_loss���@

error_R�+S?

learning_rate_1�O?7�';�I       6%�	/@Z���A�)*;


total_lossZ�@

error_R�a?

learning_rate_1�O?7Ŀ UI       6%�	�y@Z���A�)*;


total_loss��@

error_R�Af?

learning_rate_1�O?79W�I       6%�	b�@Z���A�)*;


total_loss��@

error_R�AX?

learning_rate_1�O?7����I       6%�	>�@Z���A�)*;


total_lossP2�@

error_R)�S?

learning_rate_1�O?7XบI       6%�	LDAZ���A�)*;


total_loss��@

error_R��`?

learning_rate_1�O?7����I       6%�	��AZ���A�)*;


total_loss1�@

error_R��L?

learning_rate_1�O?7��±I       6%�	�AZ���A�)*;


total_loss(G�@

error_R��Q?

learning_rate_1�O?7���I       6%�	"BZ���A�)*;


total_loss��@

error_R4�S?

learning_rate_1�O?7�vORI       6%�	�^BZ���A�)*;


total_loss3W�@

error_R,�N?

learning_rate_1�O?7�垣I       6%�	L�BZ���A�)*;


total_lossݐ@

error_R.K?

learning_rate_1�O?7nF~�I       6%�	#�BZ���A�)*;


total_loss�H�@

error_R��G?

learning_rate_1�O?7萫�I       6%�	�'CZ���A�)*;


total_loss��@

error_R[�Y?

learning_rate_1�O?7�1��I       6%�	�oCZ���A�)*;


total_lossd��@

error_R�D?

learning_rate_1�O?7Y�S5I       6%�	��CZ���A�)*;


total_loss�j�@

error_R�M?

learning_rate_1�O?7�[u�I       6%�	��CZ���A�)*;


total_loss��A

error_R�=?

learning_rate_1�O?7�5,I       6%�	�0DZ���A�)*;


total_lossȖ�@

error_R��??

learning_rate_1�O?7+RI       6%�	RqDZ���A�)*;


total_loss��A

error_Rԃ\?

learning_rate_1�O?7b#�I       6%�	"�DZ���A�)*;


total_loss.��@

error_R� G?

learning_rate_1�O?7�#�I       6%�		�DZ���A�)*;


total_loss�:�@

error_RhF?

learning_rate_1�O?78ZRI       6%�	�;EZ���A�)*;


total_loss�%A

error_R�^P?

learning_rate_1�O?7���+I       6%�	�~EZ���A�)*;


total_loss:�@

error_R�*a?

learning_rate_1�O?7�"7hI       6%�	B�EZ���A�)*;


total_loss�|�@

error_Rx3=?

learning_rate_1�O?7��	1I       6%�	oFZ���A�)*;


total_lossz�@

error_R��I?

learning_rate_1�O?7��ZEI       6%�	�@FZ���A�)*;


total_loss�^�@

error_Rs�H?

learning_rate_1�O?7�w��I       6%�	�~FZ���A�)*;


total_loss2�@

error_RvZU?

learning_rate_1�O?7�s�dI       6%�	��FZ���A�)*;


total_loss�W�@

error_RF?

learning_rate_1�O?7n�mTI       6%�	GZ���A�)*;


total_loss�1�@

error_R:(:?

learning_rate_1�O?7x��I       6%�	6AGZ���A�)*;


total_loss}�@

error_R*�T?

learning_rate_1�O?7�?%�I       6%�	6�GZ���A�)*;


total_loss�#�@

error_R{oR?

learning_rate_1�O?7�#��I       6%�	n�GZ���A�)*;


total_loss���@

error_Rn�O?

learning_rate_1�O?7���*I       6%�	HZ���A�)*;


total_loss6\�@

error_R�c?

learning_rate_1�O?7�_�8I       6%�	�BHZ���A�)*;


total_loss�׻@

error_R�IZ?

learning_rate_1�O?7"Ki.I       6%�	ȃHZ���A�)*;


total_loss��A

error_RX�O?

learning_rate_1�O?7<�I       6%�	>�HZ���A�)*;


total_lossj2A

error_R|�F?

learning_rate_1�O?7�|ʀI       6%�	"	IZ���A�)*;


total_loss1ӷ@

error_R��j?

learning_rate_1�O?7�0�I       6%�	�JIZ���A�)*;


total_loss���@

error_Rn�U?

learning_rate_1�O?7�{I       6%�	�IZ���A�)*;


total_loss�Î@

error_R1sN?

learning_rate_1�O?7I��I       6%�	��IZ���A�)*;


total_lossB�@

error_R6aS?

learning_rate_1�O?7�hcI       6%�	}JZ���A�)*;


total_loss�*�@

error_R��A?

learning_rate_1�O?7�a��I       6%�	�NJZ���A�)*;


total_loss���@

error_R�dP?

learning_rate_1�O?7m�ʆI       6%�	�JZ���A�)*;


total_lossi^�@

error_RO�P?

learning_rate_1�O?7ß
�I       6%�	l�JZ���A�)*;


total_lossxf�@

error_R�|S?

learning_rate_1�O?7�"��I       6%�	�KZ���A�)*;


total_loss(�t@

error_RI2L?

learning_rate_1�O?7.x�I       6%�	�bKZ���A�)*;


total_loss�@

error_R�X??

learning_rate_1�O?7�!8I       6%�	�KZ���A�)*;


total_lossD2�@

error_R�\?

learning_rate_1�O?7��wI       6%�	$LZ���A�)*;


total_loss5��@

error_R��]?

learning_rate_1�O?7��FI       6%�	@CLZ���A�)*;


total_losss�@

error_R@�L?

learning_rate_1�O?7*~W�I       6%�	U�LZ���A�)*;


total_loss���@

error_R��X?

learning_rate_1�O?7����I       6%�	�LZ���A�)*;


total_loss���@

error_R��_?

learning_rate_1�O?7iB ^I       6%�	MMZ���A�)*;


total_loss���@

error_RԨV?

learning_rate_1�O?7��I       6%�	�GMZ���A�)*;


total_loss:p�@

error_R�1I?

learning_rate_1�O?7?�bI       6%�	�MZ���A�)*;


total_loss���@

error_R�6b?

learning_rate_1�O?7����I       6%�	B�MZ���A�)*;


total_loss���@

error_R��Q?

learning_rate_1�O?7ƈWI       6%�	8
NZ���A�)*;


total_losst��@

error_R�'C?

learning_rate_1�O?7��׺I       6%�	�JNZ���A�)*;


total_loss�ν@

error_R�D?

learning_rate_1�O?7C4xpI       6%�	�NZ���A�)*;


total_loss���@

error_RO\H?

learning_rate_1�O?7'f�XI       6%�	�NZ���A�)*;


total_loss��v@

error_R.�G?

learning_rate_1�O?7CL�I       6%�	t
OZ���A�)*;


total_loss�c�@

error_R�`?

learning_rate_1�O?7ˢ�kI       6%�	LOZ���A�)*;


total_lossh��@

error_R��A?

learning_rate_1�O?7?�:VI       6%�	)�OZ���A�)*;


total_loss�/�@

error_R��H?

learning_rate_1�O?7=�!I       6%�	��OZ���A�)*;


total_loss-��@

error_R%aF?

learning_rate_1�O?7srn�I       6%�	�PZ���A�)*;


total_loss�@

error_R{�S?

learning_rate_1�O?7�"!�I       6%�	�NPZ���A�)*;


total_loss�V�@

error_R) Z?

learning_rate_1�O?7�'��I       6%�	��PZ���A�)*;


total_loss���@

error_RZ.M?

learning_rate_1�O?7�"�4I       6%�	 �PZ���A�)*;


total_loss��@

error_R�c?

learning_rate_1�O?7Ư�SI       6%�	�QZ���A�)*;


total_loss۷�@

error_R�G?

learning_rate_1�O?7��;�I       6%�	�VQZ���A�)*;


total_loss@��@

error_RjD?

learning_rate_1�O?7P�wkI       6%�	˙QZ���A�)*;


total_loss�B�@

error_R��M?

learning_rate_1�O?7�xiI       6%�	��QZ���A�)*;


total_loss��@

error_R_/A?

learning_rate_1�O?7O�V�I       6%�	.RZ���A�)*;


total_losso��@

error_Rv�3?

learning_rate_1�O?7��OI       6%�	�ZRZ���A�)*;


total_loss���@

error_Rf�M?

learning_rate_1�O?7���gI       6%�	O�RZ���A�)*;


total_loss���@

error_RϹ5?

learning_rate_1�O?7�$9�I       6%�	��RZ���A�**;


total_loss���@

error_R�dS?

learning_rate_1�O?7u�jI       6%�	1SZ���A�**;


total_loss�=�@

error_R{WB?

learning_rate_1�O?7���I       6%�	�\SZ���A�**;


total_loss�֦@

error_R=e?

learning_rate_1�O?7�3yQI       6%�	��SZ���A�**;


total_loss��A

error_R\�W?

learning_rate_1�O?7�Ϸ�I       6%�	��SZ���A�**;


total_loss&ҹ@

error_RH�P?

learning_rate_1�O?7�!�>I       6%�	)TZ���A�**;


total_loss�@

error_Rt�P?

learning_rate_1�O?7PU7I       6%�	�gTZ���A�**;


total_loss0��@

error_R�u9?

learning_rate_1�O?76�+%I       6%�	ߨTZ���A�**;


total_loss=��@

error_R��H?

learning_rate_1�O?7�vB�I       6%�	��TZ���A�**;


total_loss�K�@

error_Rs�U?

learning_rate_1�O?7]`I       6%�	�,UZ���A�**;


total_loss�E@

error_R�T?

learning_rate_1�O?7��u|I       6%�	�mUZ���A�**;


total_lossdԱ@

error_R�`K?

learning_rate_1�O?7@8��I       6%�	D�UZ���A�**;


total_lossUhA

error_RҏZ?

learning_rate_1�O?7\�f�I       6%�	T�UZ���A�**;


total_loss���@

error_R��??

learning_rate_1�O?7�#��I       6%�	`4VZ���A�**;


total_loss�m A

error_R�"R?

learning_rate_1�O?7e{�iI       6%�	�tVZ���A�**;


total_loss���@

error_R��S?

learning_rate_1�O?77U)I       6%�	T�VZ���A�**;


total_loss�Ɨ@

error_RJ�W?

learning_rate_1�O?7&.��I       6%�	:�VZ���A�**;


total_loss,5A

error_Rs�??

learning_rate_1�O?7��DQI       6%�	#=WZ���A�**;


total_loss�>�@

error_R.�b?

learning_rate_1�O?7��$	I       6%�	�~WZ���A�**;


total_lossd�@

error_R,G5?

learning_rate_1�O?7\�܍I       6%�	Y�WZ���A�**;


total_loss)��@

error_R�iV?

learning_rate_1�O?7�UP�I       6%�	�XZ���A�**;


total_loss���@

error_R3Q?

learning_rate_1�O?7��N�I       6%�	_EXZ���A�**;


total_loss��A

error_R�[?

learning_rate_1�O?7I__I       6%�	φXZ���A�**;


total_loss��@

error_R4sK?

learning_rate_1�O?7��=I       6%�	?�XZ���A�**;


total_loss��@

error_Rt�M?

learning_rate_1�O?7�-�I       6%�	�YZ���A�**;


total_loss� �@

error_RqB?

learning_rate_1�O?7_�"vI       6%�	�JYZ���A�**;


total_loss�@

error_R.�T?

learning_rate_1�O?7��*�I       6%�	|�YZ���A�**;


total_loss(ם@

error_R	Y?

learning_rate_1�O?7&3�VI       6%�	+�YZ���A�**;


total_loss�S�@

error_R��W?

learning_rate_1�O?7	�qzI       6%�	�ZZ���A�**;


total_losss��@

error_R�]X?

learning_rate_1�O?7-�RI       6%�	OuZZ���A�**;


total_loss�I�@

error_R�2h?

learning_rate_1�O?7;��I       6%�	��ZZ���A�**;


total_loss�Y�@

error_R}�<?

learning_rate_1�O?7�/I       6%�	��ZZ���A�**;


total_loss�b�@

error_R�W?

learning_rate_1�O?7�� �I       6%�	�A[Z���A�**;


total_loss��|@

error_R�D?

learning_rate_1�O?7��^I       6%�	9�[Z���A�**;


total_loss>�@

error_R��S?

learning_rate_1�O?7|�� I       6%�	��[Z���A�**;


total_loss��@

error_R��W?

learning_rate_1�O?7@�NI       6%�	�1\Z���A�**;


total_loss%��@

error_R��Z?

learning_rate_1�O?76(�I       6%�	�s\Z���A�**;


total_loss@=A

error_R��[?

learning_rate_1�O?7�D�LI       6%�	��\Z���A�**;


total_loss��@

error_R�F?

learning_rate_1�O?7Ԉz4I       6%�	"�\Z���A�**;


total_loss�v�@

error_R�/7?

learning_rate_1�O?7�EsI       6%�	�=]Z���A�**;


total_loss���@

error_R�6`?

learning_rate_1�O?7�r��I       6%�	��]Z���A�**;


total_loss@p�@

error_R��O?

learning_rate_1�O?7�u�@I       6%�	��]Z���A�**;


total_loss���@

error_Rw�D?

learning_rate_1�O?7��6�I       6%�	N^Z���A�**;


total_lossqJ�@

error_R8�R?

learning_rate_1�O?7$�I       6%�	ON^Z���A�**;


total_lossL[t@

error_R&�S?

learning_rate_1�O?76�-I       6%�	�^Z���A�**;


total_loss�r�@

error_R�,U?

learning_rate_1�O?7���I       6%�	�^Z���A�**;


total_lossS)�@

error_R��\?

learning_rate_1�O?7��I       6%�	�'_Z���A�**;


total_loss,U�@

error_R�[?

learning_rate_1�O?7^��I       6%�	�g_Z���A�**;


total_loss��@

error_R��K?

learning_rate_1�O?7�'/I       6%�	:�_Z���A�**;


total_loss{�@

error_RW?

learning_rate_1�O?7jc/I       6%�	"�_Z���A�**;


total_loss2�@

error_R�K?

learning_rate_1�O?7UE}HI       6%�	�4`Z���A�**;


total_loss��@

error_R��S?

learning_rate_1�O?7�d2gI       6%�	�u`Z���A�**;


total_loss�@

error_R�8?

learning_rate_1�O?7R��eI       6%�	i�`Z���A�**;


total_loss��@

error_R:*R?

learning_rate_1�O?7Q�inI       6%�	��`Z���A�**;


total_loss[|�@

error_R4uM?

learning_rate_1�O?7�6�I       6%�	d8aZ���A�**;


total_loss� �@

error_R1�K?

learning_rate_1�O?7����I       6%�	waZ���A�**;


total_loss���@

error_R��3?

learning_rate_1�O?7O
 YI       6%�	��aZ���A�**;


total_loss4Y�@

error_R�O?

learning_rate_1�O?7���I       6%�	f�aZ���A�**;


total_lossJ�A

error_R�X?

learning_rate_1�O?7e��I       6%�	>bZ���A�**;


total_loss��@

error_R\ L?

learning_rate_1�O?7��-I       6%�	�bZ���A�**;


total_loss�s@

error_R�S?

learning_rate_1�O?7-
"�I       6%�	��bZ���A�**;


total_lossƭ�@

error_R�P?

learning_rate_1�O?7i}��I       6%�	��bZ���A�**;


total_loss7�@

error_R�V?

learning_rate_1�O?7�m�I       6%�	>?cZ���A�**;


total_loss�2�@

error_R�F?

learning_rate_1�O?7���I       6%�	��cZ���A�**;


total_loss	ƶ@

error_R�R?

learning_rate_1�O?79��2I       6%�	��cZ���A�**;


total_loss�~@

error_R dO?

learning_rate_1�O?7�sI       6%�	�	dZ���A�**;


total_loss�W�@

error_R�6O?

learning_rate_1�O?7u`��I       6%�	�IdZ���A�**;


total_loss��@

error_R@N?

learning_rate_1�O?7r���I       6%�	��dZ���A�**;


total_loss���@

error_R�5R?

learning_rate_1�O?7B�j�I       6%�	��dZ���A�**;


total_loss舄@

error_R��D?

learning_rate_1�O?7�8�<I       6%�	eZ���A�**;


total_lossA

error_R�3R?

learning_rate_1�O?7ʗY�I       6%�	;OeZ���A�**;


total_loss,��@

error_R�D?

learning_rate_1�O?7Eq�I       6%�	"�eZ���A�**;


total_loss<�@

error_R��??

learning_rate_1�O?7^��)I       6%�	��eZ���A�**;


total_loss�^�@

error_R��n?

learning_rate_1�O?7�D�`I       6%�	�fZ���A�**;


total_loss��@

error_R��N?

learning_rate_1�O?7!�KMI       6%�	LYfZ���A�**;


total_loss���@

error_RE�V?

learning_rate_1�O?7�.��I       6%�	i�fZ���A�**;


total_loss誺@

error_R�gK?

learning_rate_1�O?7"�qI       6%�	��fZ���A�**;


total_loss�H�@

error_R�M?

learning_rate_1�O?7�WkI       6%�	� gZ���A�**;


total_loss��@

error_R\�D?

learning_rate_1�O?7T���I       6%�	\bgZ���A�**;


total_loss,��@

error_RͤH?

learning_rate_1�O?7_���I       6%�	B�gZ���A�**;


total_lossh��@

error_R��K?

learning_rate_1�O?7�[P�I       6%�	��gZ���A�**;


total_loss-dA

error_R��P?

learning_rate_1�O?7���I       6%�	s!hZ���A�**;


total_lossW&�@

error_R[nV?

learning_rate_1�O?7(��I       6%�	XahZ���A�**;


total_loss���@

error_RsIN?

learning_rate_1�O?7ɹd�I       6%�	<�hZ���A�**;


total_loss[��@

error_R@_E?

learning_rate_1�O?7<�M7I       6%�	^�hZ���A�**;


total_loss��@

error_R�H?

learning_rate_1�O?7��(I       6%�	�iZ���A�**;


total_loss3 A

error_R\.[?

learning_rate_1�O?79���I       6%�	Y_iZ���A�**;


total_loss�P�@

error_R*{M?

learning_rate_1�O?7���I       6%�	H�iZ���A�**;


total_loss&��@

error_R��R?

learning_rate_1�O?7 _��I       6%�	\�iZ���A�**;


total_loss�Z�@

error_Rl�Q?

learning_rate_1�O?7��:�I       6%�	�)jZ���A�**;


total_loss���@

error_R.�A?

learning_rate_1�O?7�J��I       6%�	_njZ���A�**;


total_loss��@

error_R�:?

learning_rate_1�O?7����I       6%�	�jZ���A�**;


total_loss?�@

error_RMnO?

learning_rate_1�O?77��I       6%�	 �jZ���A�**;


total_loss�'�@

error_R�UQ?

learning_rate_1�O?7�._I       6%�	�9kZ���A�**;


total_lossϜ�@

error_RW�H?

learning_rate_1�O?7��~I       6%�	��kZ���A�**;


total_lossj�@

error_R;WI?

learning_rate_1�O?7gM�I       6%�	%�kZ���A�**;


total_loss���@

error_R��_?

learning_rate_1�O?7~Z�,I       6%�	�!lZ���A�**;


total_loss��@

error_RiR?

learning_rate_1�O?7_���I       6%�	jlZ���A�**;


total_lossԻ@

error_Ri�Y?

learning_rate_1�O?7�Z'9I       6%�	��lZ���A�**;


total_loss�1�@

error_RdW?

learning_rate_1�O?7nL��I       6%�	��lZ���A�**;


total_loss��@

error_R�)P?

learning_rate_1�O?7�	�wI       6%�	�/mZ���A�**;


total_loss���@

error_R�c?

learning_rate_1�O?7�'P�I       6%�	�rmZ���A�**;


total_loss���@

error_R�wN?

learning_rate_1�O?7%*gDI       6%�	��mZ���A�**;


total_loss\L�@

error_RIJ?

learning_rate_1�O?7_�1I       6%�	��mZ���A�**;


total_loss�&�@

error_Ri/F?

learning_rate_1�O?7b��{I       6%�	n:nZ���A�**;


total_lossK?�@

error_R�]??

learning_rate_1�O?7t�T�I       6%�	�{nZ���A�**;


total_loss E�@

error_R�??

learning_rate_1�O?7on�I       6%�	��nZ���A�**;


total_lossƉ�@

error_R� F?

learning_rate_1�O?7;�0mI       6%�	joZ���A�**;


total_loss&�A

error_R\�M?

learning_rate_1�O?7����I       6%�	�HoZ���A�**;


total_loss�ʃ@

error_R�AB?

learning_rate_1�O?7,I       6%�	G�oZ���A�**;


total_loss��@

error_R)�H?

learning_rate_1�O?7��I       6%�	��oZ���A�**;


total_loss7��@

error_RLWT?

learning_rate_1�O?7e�I       6%�	,pZ���A�**;


total_loss��@

error_R_M^?

learning_rate_1�O?7��~I       6%�	�SpZ���A�**;


total_loss=��@

error_R��_?

learning_rate_1�O?7FZ��I       6%�	��pZ���A�**;


total_loss_ڽ@

error_R��:?

learning_rate_1�O?7VJ�I       6%�	��pZ���A�**;


total_loss�m�@

error_RE�X?

learning_rate_1�O?7���I       6%�	�qZ���A�**;


total_lossFp�@

error_R M?

learning_rate_1�O?7��
�I       6%�	"SqZ���A�**;


total_lossҴ�@

error_R��Y?

learning_rate_1�O?7v�[I       6%�	l�qZ���A�**;


total_loss$��@

error_R�;_?

learning_rate_1�O?7L��xI       6%�	�qZ���A�**;


total_loss��@

error_R�eZ?

learning_rate_1�O?7c�vpI       6%�	
rZ���A�**;


total_lossxf�@

error_RF?

learning_rate_1�O?7�v$�I       6%�	�TrZ���A�**;


total_lossw'@

error_R
lC?

learning_rate_1�O?7	��oI       6%�	ɔrZ���A�**;


total_lossT�@

error_R��2?

learning_rate_1�O?7L5tI       6%�	�rZ���A�**;


total_lossQ>�@

error_RL�H?

learning_rate_1�O?7��J�I       6%�	sZ���A�**;


total_lossS��@

error_R_hO?

learning_rate_1�O?7s�)I       6%�	�ZsZ���A�**;


total_loss���@

error_R�X?

learning_rate_1�O?7�I�I       6%�	��sZ���A�**;


total_loss��@

error_R�E?

learning_rate_1�O?7�� I       6%�	�sZ���A�**;


total_loss���@

error_R�5U?

learning_rate_1�O?7�$��I       6%�	�'tZ���A�**;


total_loss���@

error_R�^O?

learning_rate_1�O?7db�I       6%�	YgtZ���A�+*;


total_loss�F�@

error_R�4_?

learning_rate_1�O?7y��I       6%�	ئtZ���A�+*;


total_loss%7�@

error_R�KB?

learning_rate_1�O?7d���I       6%�	G�tZ���A�+*;


total_loss��@

error_R��@?

learning_rate_1�O?7)�KYI       6%�	l-uZ���A�+*;


total_loss���@

error_RHg??

learning_rate_1�O?73��`I       6%�	8nuZ���A�+*;


total_loss��@

error_R\�b?

learning_rate_1�O?7�_L�I       6%�	p�uZ���A�+*;


total_loss6.�@

error_R��^?

learning_rate_1�O?7bDi�I       6%�	�uZ���A�+*;


total_lossh�@

error_R!V?

learning_rate_1�O?7^^�I       6%�	�0vZ���A�+*;


total_loss,��@

error_R�B?

learning_rate_1�O?7����I       6%�	�uvZ���A�+*;


total_loss7A

error_R�aA?

learning_rate_1�O?7䜖I       6%�	��vZ���A�+*;


total_lossJ��@

error_R3'O?

learning_rate_1�O?7��*�I       6%�	��vZ���A�+*;


total_loss�
�@

error_R��`?

learning_rate_1�O?7e�"I       6%�	b?wZ���A�+*;


total_loss�$�@

error_R��??

learning_rate_1�O?7E���I       6%�	=�wZ���A�+*;


total_lossF��@

error_RO�K?

learning_rate_1�O?7�4=�I       6%�	��wZ���A�+*;


total_loss�5�@

error_Rl�Q?

learning_rate_1�O?7^>I       6%�	_xZ���A�+*;


total_lossF�X@

error_R*%A?

learning_rate_1�O?7랷^I       6%�	�FxZ���A�+*;


total_loss���@

error_R��K?

learning_rate_1�O?7���AI       6%�	?�xZ���A�+*;


total_loss��@

error_R	M?

learning_rate_1�O?7�;�I       6%�	��xZ���A�+*;


total_loss�1�@

error_R}�W?

learning_rate_1�O?7Y�W�I       6%�	�yZ���A�+*;


total_lossԗ�@

error_R�G?

learning_rate_1�O?7W��I       6%�	pCyZ���A�+*;


total_loss�$�@

error_R�Y?

learning_rate_1�O?7@��I       6%�	��yZ���A�+*;


total_loss��@

error_RO�H?

learning_rate_1�O?7�;��I       6%�	�yZ���A�+*;


total_loss��@

error_R�Y?

learning_rate_1�O?7I;��I       6%�		zZ���A�+*;


total_lossHV$A

error_RER?

learning_rate_1�O?7��P5I       6%�	KzZ���A�+*;


total_loss���@

error_Rs�Q?

learning_rate_1�O?71'��I       6%�	G�zZ���A�+*;


total_loss	�`@

error_R��C?

learning_rate_1�O?7cS�7I       6%�	l�zZ���A�+*;


total_loss�&�@

error_Rn�O?

learning_rate_1�O?7�'I       6%�	q{Z���A�+*;


total_loss��@

error_R��K?

learning_rate_1�O?7�q4�I       6%�	�S{Z���A�+*;


total_loss�E�@

error_R�I?

learning_rate_1�O?7�
_I       6%�	ȵ{Z���A�+*;


total_loss�n!A

error_R��M?

learning_rate_1�O?7g�@�I       6%�	��{Z���A�+*;


total_loss�TA

error_Rf�J?

learning_rate_1�O?7L�1kI       6%�	d<|Z���A�+*;


total_lossF��@

error_Rn=]?

learning_rate_1�O?7�T��I       6%�	~~|Z���A�+*;


total_lossE�@

error_R%�S?

learning_rate_1�O?75W��I       6%�	'�|Z���A�+*;


total_loss���@

error_R�J?

learning_rate_1�O?7�DT�I       6%�	�	}Z���A�+*;


total_loss��@

error_R��:?

learning_rate_1�O?7��܍I       6%�	K}Z���A�+*;


total_lossJ�@

error_RX�N?

learning_rate_1�O?7��CI       6%�	�}Z���A�+*;


total_loss�@

error_R2wH?

learning_rate_1�O?7꩛I       6%�	-�}Z���A�+*;


total_loss3 A

error_R��Q?

learning_rate_1�O?7H��#I       6%�	�~Z���A�+*;


total_loss���@

error_R�X?

learning_rate_1�O?7���DI       6%�	l~Z���A�+*;


total_loss�7�@

error_R�fc?

learning_rate_1�O?7�#�I       6%�	��~Z���A�+*;


total_loss9�@

error_R�H?

learning_rate_1�O?7	�@I       6%�	��~Z���A�+*;


total_loss`�@

error_R�*L?

learning_rate_1�O?7=K0I       6%�	1/Z���A�+*;


total_loss��@

error_R��J?

learning_rate_1�O?7���I       6%�	�Z���A�+*;


total_loss�>�@

error_R(�Q?

learning_rate_1�O?7�,HI       6%�	]�Z���A�+*;


total_loss#A

error_R��3?

learning_rate_1�O?71��$I       6%�	��Z���A�+*;


total_loss0�@

error_R	�]?

learning_rate_1�O?7.ec�I       6%�	�^�Z���A�+*;


total_loss�QA

error_R �@?

learning_rate_1�O?7�~|I       6%�	ɤ�Z���A�+*;


total_lossN��@

error_R� J?

learning_rate_1�O?7����I       6%�	��Z���A�+*;


total_loss�+�@

error_R�e[?

learning_rate_1�O?7�\f�I       6%�	m*�Z���A�+*;


total_loss�c�@

error_R��F?

learning_rate_1�O?7I��I       6%�	sj�Z���A�+*;


total_loss�0�@

error_R�+a?

learning_rate_1�O?7��sTI       6%�	�Z���A�+*;


total_loss�i�@

error_R̺I?

learning_rate_1�O?7f�#mI       6%�	i�Z���A�+*;


total_loss�ȼ@

error_R�$P?

learning_rate_1�O?7<{�I       6%�	�6�Z���A�+*;


total_loss��@

error_RX�^?

learning_rate_1�O?7�-�I       6%�	�y�Z���A�+*;


total_lossnA

error_RA�C?

learning_rate_1�O?7��I       6%�	���Z���A�+*;


total_loss6R�@

error_R1�@?

learning_rate_1�O?7��4I       6%�	��Z���A�+*;


total_loss���@

error_R)�S?

learning_rate_1�O?7c%x&I       6%�	7F�Z���A�+*;


total_loss�6A

error_R��L?

learning_rate_1�O?7���I       6%�	̉�Z���A�+*;


total_lossz�@

error_R}BK?

learning_rate_1�O?7GmI       6%�	�̓Z���A�+*;


total_loss�c�@

error_R3�A?

learning_rate_1�O?7㟊�I       6%�	��Z���A�+*;


total_loss9�@

error_R59?

learning_rate_1�O?7	�5I       6%�	�U�Z���A�+*;


total_loss��@

error_R�cO?

learning_rate_1�O?7ļ,�I       6%�	��Z���A�+*;


total_loss�r�@

error_R�?K?

learning_rate_1�O?7e��I       6%�	��Z���A�+*;


total_loss�Θ@

error_RܱM?

learning_rate_1�O?7�܄GI       6%�	!�Z���A�+*;


total_loss�H�@

error_R,U?

learning_rate_1�O?7��_�I       6%�	�c�Z���A�+*;


total_lossڃ@

error_R�j_?

learning_rate_1�O?7�_OI       6%�	墅Z���A�+*;


total_loss�=�@

error_R�TR?

learning_rate_1�O?7;yO�I       6%�	�Z���A�+*;


total_lossZ��@

error_Rq�;?

learning_rate_1�O?7&Y�I       6%�	�(�Z���A�+*;


total_lossv�m@

error_R�(G?

learning_rate_1�O?7G��I       6%�	�j�Z���A�+*;


total_loss���@

error_R�iN?

learning_rate_1�O?7���I       6%�	P��Z���A�+*;


total_lossf�@

error_R3�7?

learning_rate_1�O?7�%�wI       6%�	]�Z���A�+*;


total_lossʏ�@

error_R�QZ?

learning_rate_1�O?7"�}.I       6%�	/2�Z���A�+*;


total_loss�� A

error_R�{G?

learning_rate_1�O?7 ��aI       6%�	�s�Z���A�+*;


total_loss�ȧ@

error_R.uH?

learning_rate_1�O?7'��I       6%�	��Z���A�+*;


total_loss{Ε@

error_RJgS?

learning_rate_1�O?73�'I       6%�	���Z���A�+*;


total_loss��@

error_R�X?

learning_rate_1�O?75q��I       6%�		8�Z���A�+*;


total_loss-f A

error_R �@?

learning_rate_1�O?7�I       6%�	�x�Z���A�+*;


total_loss\2�@

error_R��<?

learning_rate_1�O?7���jI       6%�	��Z���A�+*;


total_loss�o�@

error_R]�J?

learning_rate_1�O?7A�I       6%�	Z��Z���A�+*;


total_loss�p�@

error_R�]?

learning_rate_1�O?7��XI       6%�	�:�Z���A�+*;


total_loss��@

error_R�G?

learning_rate_1�O?7'�zI       6%�	@{�Z���A�+*;


total_loss���@

error_Rs�@?

learning_rate_1�O?70!�ZI       6%�	���Z���A�+*;


total_loss�5�@

error_RR�\?

learning_rate_1�O?7�,%I       6%�	<�Z���A�+*;


total_loss���@

error_R��P?

learning_rate_1�O?7-;I�I       6%�	�H�Z���A�+*;


total_lossQ�@

error_R�A`?

learning_rate_1�O?7�cg�I       6%�	9��Z���A�+*;


total_lossͧ�@

error_R��E?

learning_rate_1�O?7�N��I       6%�	�ɊZ���A�+*;


total_lossӨA

error_R�J?

learning_rate_1�O?7u˿�I       6%�	��Z���A�+*;


total_loss�S�@

error_R)L?

learning_rate_1�O?7?�R^I       6%�	dQ�Z���A�+*;


total_lossDKA

error_R�P?

learning_rate_1�O?7@�I       6%�	x��Z���A�+*;


total_loss���@

error_R��T?

learning_rate_1�O?74gI       6%�	�Z���A�+*;


total_loss�'�@

error_RC�[?

learning_rate_1�O?7�!�I       6%�	(5�Z���A�+*;


total_loss��@

error_R	�Q?

learning_rate_1�O?7"hOfI       6%�	}�Z���A�+*;


total_lossS+�@

error_R��C?

learning_rate_1�O?7B�M(I       6%�	QŌZ���A�+*;


total_loss�S�@

error_R��H?

learning_rate_1�O?7r��I       6%�	i	�Z���A�+*;


total_loss[�A

error_R_!=?

learning_rate_1�O?7�=�I       6%�	CJ�Z���A�+*;


total_loss���@

error_RX�L?

learning_rate_1�O?7��-I       6%�	名Z���A�+*;


total_loss�|A

error_R�O?

learning_rate_1�O?7�ZV�I       6%�	�׍Z���A�+*;


total_lossD\�@

error_R:N?

learning_rate_1�O?7��K�I       6%�	� �Z���A�+*;


total_loss��@

error_R�MU?

learning_rate_1�O?7����I       6%�	If�Z���A�+*;


total_lossS�A

error_Ra?

learning_rate_1�O?7���bI       6%�	��Z���A�+*;


total_loss���@

error_R��U?

learning_rate_1�O?7]�N�I       6%�	���Z���A�+*;


total_loss.�@

error_R�qS?

learning_rate_1�O?73��I       6%�	�0�Z���A�+*;


total_lossm��@

error_R4<=?

learning_rate_1�O?7�5��I       6%�	�y�Z���A�+*;


total_loss db@

error_Rv�L?

learning_rate_1�O?7�s�I       6%�	���Z���A�+*;


total_lossI�@

error_R�??

learning_rate_1�O?7Msh�I       6%�	Y�Z���A�+*;


total_loss��@

error_R�{M?

learning_rate_1�O?7�J�TI       6%�	�L�Z���A�+*;


total_loss���@

error_R�@a?

learning_rate_1�O?7P*n=I       6%�	Ԑ�Z���A�+*;


total_lossx½@

error_RV`M?

learning_rate_1�O?7y*a�I       6%�	�ӐZ���A�+*;


total_loss��A

error_Rn6O?

learning_rate_1�O?7ZR��I       6%�	o�Z���A�+*;


total_loss
�@

error_R�+S?

learning_rate_1�O?7��I       6%�	S�Z���A�+*;


total_lossh��@

error_R�H?

learning_rate_1�O?7�/��I       6%�	���Z���A�+*;


total_loss���@

error_RJ?

learning_rate_1�O?7L}�I       6%�	�ӑZ���A�+*;


total_loss��@

error_R�^I?

learning_rate_1�O?7�m�I       6%�	��Z���A�+*;


total_loss��@

error_R�=?

learning_rate_1�O?7/h��I       6%�	�Z�Z���A�+*;


total_loss�X�@

error_R��i?

learning_rate_1�O?7�\?I       6%�	֟�Z���A�+*;


total_loss �@

error_R[�;?

learning_rate_1�O?7b��I       6%�	��Z���A�+*;


total_loss�.�@

error_R�A?

learning_rate_1�O?7E���I       6%�	=%�Z���A�+*;


total_lossq2�@

error_R��W?

learning_rate_1�O?7���yI       6%�	�f�Z���A�+*;


total_loss�-�@

error_R�oR?

learning_rate_1�O?7Ψ��I       6%�	���Z���A�+*;


total_loss*4�@

error_R��V?

learning_rate_1�O?7E��QI       6%�	��Z���A�+*;


total_loss҂�@

error_RI?

learning_rate_1�O?7���I       6%�	-�Z���A�+*;


total_loss�<�@

error_R�O?

learning_rate_1�O?7��]I       6%�	�m�Z���A�+*;


total_loss��@

error_RF~a?

learning_rate_1�O?7l���I       6%�	Z���A�+*;


total_loss��@

error_R��]?

learning_rate_1�O?7c3��I       6%�	��Z���A�+*;


total_loss;��@

error_R�tH?

learning_rate_1�O?7(2��I       6%�	9�Z���A�+*;


total_loss�F�@

error_RڵE?

learning_rate_1�O?7죵HI       6%�	)y�Z���A�+*;


total_loss/�@

error_R��]?

learning_rate_1�O?7 S]�I       6%�	O��Z���A�+*;


total_loss$oA

error_R��C?

learning_rate_1�O?7U�WI       6%�	���Z���A�+*;


total_loss(�@

error_R�K?

learning_rate_1�O?7��جI       6%�	�>�Z���A�,*;


total_lossJuA

error_RJ�E?

learning_rate_1�O?7��B�I       6%�	�Z���A�,*;


total_loss�'�@

error_R�L?

learning_rate_1�O?7w�aI       6%�	���Z���A�,*;


total_loss]�@

error_R��X?

learning_rate_1�O?7_���I       6%�	 �Z���A�,*;


total_lossѢ�@

error_RԂH?

learning_rate_1�O?76���I       6%�	?�Z���A�,*;


total_loss�(�@

error_R �P?

learning_rate_1�O?7�ڔQI       6%�	ځ�Z���A�,*;


total_loss�=�@

error_R��P?

learning_rate_1�O?7��wI       6%�	�×Z���A�,*;


total_lossd|�@

error_RxI?

learning_rate_1�O?7� z I       6%�	!�Z���A�,*;


total_lossz��@

error_R��H?

learning_rate_1�O?7�S'�I       6%�	bG�Z���A�,*;


total_loss�_�@

error_R&�U?

learning_rate_1�O?7jA#I       6%�	*��Z���A�,*;


total_loss��A

error_R�I?

learning_rate_1�O?7u��I       6%�	 ȘZ���A�,*;


total_loss��@

error_RV�??

learning_rate_1�O?7���I       6%�	)�Z���A�,*;


total_loss�z�@

error_Rx�X?

learning_rate_1�O?7�!�hI       6%�	�F�Z���A�,*;


total_loss��@

error_RŢJ?

learning_rate_1�O?7S#��I       6%�	��Z���A�,*;


total_loss��@

error_R��V?

learning_rate_1�O?7�[�;I       6%�	�˙Z���A�,*;


total_lossCE�@

error_R]�N?

learning_rate_1�O?7r��%I       6%�	4�Z���A�,*;


total_lossᖐ@

error_ROKP?

learning_rate_1�O?7��mI       6%�	�P�Z���A�,*;


total_loss��@

error_R��H?

learning_rate_1�O?7�t��I       6%�	N��Z���A�,*;


total_loss�ս@

error_R�?N?

learning_rate_1�O?7��� I       6%�	�ۚZ���A�,*;


total_loss�rA

error_R�??

learning_rate_1�O?7wÕI       6%�	��Z���A�,*;


total_loss
	�@

error_RJPW?

learning_rate_1�O?7� LpI       6%�	pp�Z���A�,*;


total_lossn��@

error_R!D\?

learning_rate_1�O?7��II       6%�	���Z���A�,*;


total_loss��@

error_RD�Z?

learning_rate_1�O?7lAs�I       6%�	$,�Z���A�,*;


total_loss���@

error_R,�<?

learning_rate_1�O?7`d��I       6%�	�s�Z���A�,*;


total_lossV¤@

error_RM9?

learning_rate_1�O?7���{I       6%�	�Z���A�,*;


total_loss���@

error_R��Q?

learning_rate_1�O?7��GI       6%�	��Z���A�,*;


total_lossW�@

error_R)lO?

learning_rate_1�O?7j	�SI       6%�	�A�Z���A�,*;


total_loss��%A

error_R�rJ?

learning_rate_1�O?7'��hI       6%�	���Z���A�,*;


total_loss���@

error_R�T?

learning_rate_1�O?7��I       6%�	˝Z���A�,*;


total_loss�.�@

error_R�?L?

learning_rate_1�O?7���I       6%�	��Z���A�,*;


total_lossVf�@

error_RϪG?

learning_rate_1�O?7\���I       6%�	/[�Z���A�,*;


total_lossS�@

error_RIOI?

learning_rate_1�O?7;`{�I       6%�	���Z���A�,*;


total_loss� �@

error_RsvJ?

learning_rate_1�O?7��}I       6%�	��Z���A�,*;


total_loss���@

error_R��4?

learning_rate_1�O?7���I       6%�	PG�Z���A�,*;


total_loss��@

error_R�hL?

learning_rate_1�O?7��I       6%�	��Z���A�,*;


total_loss޼@

error_R��P?

learning_rate_1�O?7��I       6%�	�ٟZ���A�,*;


total_loss�|A

error_R�N?

learning_rate_1�O?7���I       6%�	U"�Z���A�,*;


total_loss���@

error_REeB?

learning_rate_1�O?7(�KI       6%�	Ml�Z���A�,*;


total_loss+v�@

error_R�PH?

learning_rate_1�O?7,Ui�I       6%�	��Z���A�,*;


total_loss�9�@

error_RćT?

learning_rate_1�O?7��݊I       6%�	��Z���A�,*;


total_loss��@

error_R�I?

learning_rate_1�O?7�<bVI       6%�	M5�Z���A�,*;


total_loss��@

error_R�P?

learning_rate_1�O?7eD.4I       6%�	|�Z���A�,*;


total_loss��@

error_Ra�O?

learning_rate_1�O?7@���I       6%�	ۼ�Z���A�,*;


total_lossD��@

error_R�X?

learning_rate_1�O?7z���I       6%�	<��Z���A�,*;


total_lossm��@

error_R��\?

learning_rate_1�O?7<�z�I       6%�	w@�Z���A�,*;


total_loss֥|@

error_R�C?

learning_rate_1�O?7֠�I       6%�	���Z���A�,*;


total_lossuI�@

error_R!�@?

learning_rate_1�O?7s9�I       6%�	VǢZ���A�,*;


total_loss���@

error_R��Z?

learning_rate_1�O?7�WQ�I       6%�		�Z���A�,*;


total_lossv��@

error_R,C?

learning_rate_1�O?7R�4�I       6%�	�L�Z���A�,*;


total_loss�қ@

error_ReW?

learning_rate_1�O?7����I       6%�	���Z���A�,*;


total_lossc&�@

error_R�
^?

learning_rate_1�O?7���EI       6%�	�ףZ���A�,*;


total_loss���@

error_R��K?

learning_rate_1�O?7�%˸I       6%�	��Z���A�,*;


total_loss�M�@

error_RT^M?

learning_rate_1�O?7�m�I       6%�	�_�Z���A�,*;


total_loss�/�@

error_Rx�<?

learning_rate_1�O?7�=�I       6%�	���Z���A�,*;


total_loss
G�@

error_RW�[?

learning_rate_1�O?7�(�I       6%�	��Z���A�,*;


total_loss���@

error_R �[?

learning_rate_1�O?7���I       6%�	�0�Z���A�,*;


total_lossR�A

error_R��O?

learning_rate_1�O?7���\I       6%�	,w�Z���A�,*;


total_loss���@

error_R�Q?

learning_rate_1�O?7�㭇I       6%�	}��Z���A�,*;


total_lossj��@

error_R��:?

learning_rate_1�O?7g{I       6%�	q �Z���A�,*;


total_loss�A

error_R�y*?

learning_rate_1�O?7#�=I       6%�	nC�Z���A�,*;


total_loss�G A

error_R&�Z?

learning_rate_1�O?7^zt�I       6%�	���Z���A�,*;


total_loss$��@

error_R}�`?

learning_rate_1�O?7�� �I       6%�	iȦZ���A�,*;


total_loss�rA

error_Rq Z?

learning_rate_1�O?7���I       6%�	�
�Z���A�,*;


total_loss��@

error_R��M?

learning_rate_1�O?7����I       6%�	K�Z���A�,*;


total_lossf��@

error_RiS?

learning_rate_1�O?7�\I       6%�	�Z���A�,*;


total_lossƴ�@

error_R�B?

learning_rate_1�O?7�W�LI       6%�	�ΧZ���A�,*;


total_lossȸ�@

error_Ri�C?

learning_rate_1�O?7L@vI       6%�	0�Z���A�,*;


total_loss��@

error_R�P?

learning_rate_1�O?7S`�I       6%�	rT�Z���A�,*;


total_loss�c�@

error_R{F?

learning_rate_1�O?7y�rI       6%�	���Z���A�,*;


total_lossH�@

error_Ra�J?

learning_rate_1�O?7�UI       6%�	{�Z���A�,*;


total_loss��@

error_RAK?

learning_rate_1�O?7�:�	I       6%�	$&�Z���A�,*;


total_loss��@

error_R,b?

learning_rate_1�O?7?ti�I       6%�	Hj�Z���A�,*;


total_lossT��@

error_R6�`?

learning_rate_1�O?7E�6!I       6%�	8��Z���A�,*;


total_loss��@

error_Ra�a?

learning_rate_1�O?7�s��I       6%�	��Z���A�,*;


total_loss9>�@

error_R��I?

learning_rate_1�O?7J���I       6%�	0�Z���A�,*;


total_loss���@

error_R��Q?

learning_rate_1�O?7���I       6%�	$r�Z���A�,*;


total_loss���@

error_R�X?

learning_rate_1�O?7 Z� I       6%�	<��Z���A�,*;


total_loss���@

error_RHfN?

learning_rate_1�O?7�vSI       6%�	U��Z���A�,*;


total_loss��@

error_R(�U?

learning_rate_1�O?7�W��I       6%�	�6�Z���A�,*;


total_loss�D�@

error_Rq�E?

learning_rate_1�O?7��I       6%�	���Z���A�,*;


total_loss���@

error_R�S?

learning_rate_1�O?7�sMI       6%�	�ثZ���A�,*;


total_loss�Œ@

error_R)�??

learning_rate_1�O?7s��;I       6%�	A�Z���A�,*;


total_loss�s@

error_R��\?

learning_rate_1�O?7��_�I       6%�	�`�Z���A�,*;


total_loss�et@

error_RàP?

learning_rate_1�O?7|=�I       6%�	���Z���A�,*;


total_lossr��@

error_RG?

learning_rate_1�O?7�6�$I       6%�	��Z���A�,*;


total_lossS��@

error_R�4\?

learning_rate_1�O?7BF�&I       6%�	'�Z���A�,*;


total_lossҬ@

error_R+A?

learning_rate_1�O?7v=�jI       6%�	�i�Z���A�,*;


total_loss�� A

error_R��O?

learning_rate_1�O?7kz�]I       6%�	_��Z���A�,*;


total_loss�g�@

error_R��N?

learning_rate_1�O?7�d��I       6%�	 �Z���A�,*;


total_loss�C�@

error_R�J?

learning_rate_1�O?7��h`I       6%�	�,�Z���A�,*;


total_loss�|�@

error_R��`?

learning_rate_1�O?7u�I7I       6%�	rl�Z���A�,*;


total_loss�G�@

error_R�W@?

learning_rate_1�O?7 ,xI       6%�	<��Z���A�,*;


total_lossľ�@

error_R�WU?

learning_rate_1�O?7}q��I       6%�	��Z���A�,*;


total_loss�`�@

error_Rld?

learning_rate_1�O?7)L
YI       6%�	,,�Z���A�,*;


total_loss}� A

error_R!G?

learning_rate_1�O?7�6lI       6%�	�o�Z���A�,*;


total_loss�'�@

error_R��E?

learning_rate_1�O?7��iI       6%�	t��Z���A�,*;


total_loss�ê@

error_R��U?

learning_rate_1�O?7H@I       6%�	B��Z���A�,*;


total_lossi,q@

error_RE�K?

learning_rate_1�O?7�b�I       6%�	'=�Z���A�,*;


total_loss��@

error_R:�_?

learning_rate_1�O?7��jI       6%�	�}�Z���A�,*;


total_loss���@

error_R��E?

learning_rate_1�O?7+��I       6%�	���Z���A�,*;


total_lossN��@

error_R�KL?

learning_rate_1�O?7_1x@I       6%�	d�Z���A�,*;


total_loss���@

error_R�NU?

learning_rate_1�O?7l��I       6%�	�E�Z���A�,*;


total_lossBÚ@

error_R��c?

learning_rate_1�O?79jy�I       6%�	8��Z���A�,*;


total_lossyA

error_R��O?

learning_rate_1�O?7>qF�I       6%�	�ȱZ���A�,*;


total_losssO�@

error_R�Z?

learning_rate_1�O?7���*I       6%�	8�Z���A�,*;


total_loss��@

error_RaF?

learning_rate_1�O?7�KT�I       6%�	fO�Z���A�,*;


total_lossݮ�@

error_R�ZN?

learning_rate_1�O?7oM&I       6%�	c��Z���A�,*;


total_loss���@

error_R��b?

learning_rate_1�O?7��2I       6%�	ӲZ���A�,*;


total_loss�ߖ@

error_R��W?

learning_rate_1�O?7܅��I       6%�	�Z���A�,*;


total_loss��@

error_R��H?

learning_rate_1�O?7\�l~I       6%�	ga�Z���A�,*;


total_loss m
A

error_RlxI?

learning_rate_1�O?7H��%I       6%�	5��Z���A�,*;


total_loss]��@

error_R�T?

learning_rate_1�O?7Ch�I       6%�	O�Z���A�,*;


total_loss���@

error_Rx,>?

learning_rate_1�O?7���I       6%�	a-�Z���A�,*;


total_lossn �@

error_R�T?

learning_rate_1�O?7)�lkI       6%�	�k�Z���A�,*;


total_lossE>A

error_R� a?

learning_rate_1�O?7
�rI       6%�	Z���A�,*;


total_loss���@

error_RZ�E?

learning_rate_1�O?7�#eI       6%�	#��Z���A�,*;


total_loss��@

error_R�W?

learning_rate_1�O?7�0'�I       6%�	�?�Z���A�,*;


total_loss�ג@

error_Rc�^?

learning_rate_1�O?7�?�II       6%�	Հ�Z���A�,*;


total_loss���@

error_R�_?

learning_rate_1�O?7fk�~I       6%�	�ƵZ���A�,*;


total_loss榊@

error_RN?Y?

learning_rate_1�O?7Ԡ��I       6%�	(�Z���A�,*;


total_lossּ�@

error_R�F??

learning_rate_1�O?7��d�I       6%�	�R�Z���A�,*;


total_loss-;�@

error_RizB?

learning_rate_1�O?7�`�I       6%�	���Z���A�,*;


total_loss�@

error_R
�;?

learning_rate_1�O?7��xI       6%�	/ԶZ���A�,*;


total_loss�0�@

error_R�O?

learning_rate_1�O?7pޜtI       6%�	�Z���A�,*;


total_loss���@

error_R�2Z?

learning_rate_1�O?7N��I       6%�	�T�Z���A�,*;


total_loss�.�@

error_R�bR?

learning_rate_1�O?7b��?I       6%�	���Z���A�,*;


total_loss,ø@

error_R�\?

learning_rate_1�O?7swI       6%�	�ܷZ���A�,*;


total_loss5m@

error_R xI?

learning_rate_1�O?7L� I       6%�	*�Z���A�,*;


total_loss���@

error_RZ)l?

learning_rate_1�O?7"��@I       6%�	M^�Z���A�-*;


total_loss\܁@

error_RN?

learning_rate_1�O?7���I       6%�	���Z���A�-*;


total_lossF�@

error_R��R?

learning_rate_1�O?7�GA�I       6%�	��Z���A�-*;


total_loss��@

error_R��M?

learning_rate_1�O?7m�?I       6%�	yE�Z���A�-*;


total_lossÆ@

error_RsP?

learning_rate_1�O?7grH�I       6%�	5��Z���A�-*;


total_loss��@

error_R��W?

learning_rate_1�O?7AdoWI       6%�	���Z���A�-*;


total_loss\ў@

error_R�L?

learning_rate_1�O?7�G�zI       6%�	wS�Z���A�-*;


total_loss�b�@

error_R]Q?

learning_rate_1�O?7��1I       6%�	���Z���A�-*;


total_loss���@

error_RnHT?

learning_rate_1�O?7P���I       6%�		��Z���A�-*;


total_lossC]�@

error_R��C?

learning_rate_1�O?7K��I       6%�	�O�Z���A�-*;


total_loss�٨@

error_R�Z?

learning_rate_1�O?7��sI       6%�	��Z���A�-*;


total_lossQ(�@

error_R�C?

learning_rate_1�O?7�=�I       6%�	`>�Z���A�-*;


total_lossx��@

error_R��N?

learning_rate_1�O?7�&�I       6%�	}��Z���A�-*;


total_loss��@

error_R��E?

learning_rate_1�O?7`E�BI       6%�	���Z���A�-*;


total_loss�O�@

error_R�<?

learning_rate_1�O?7��iI       6%�	�I�Z���A�-*;


total_loss62�@

error_R�_V?

learning_rate_1�O?7���ZI       6%�	Z��Z���A�-*;


total_loss���@

error_R�,Z?

learning_rate_1�O?7z_rbI       6%�	���Z���A�-*;


total_loss:�@

error_R��f?

learning_rate_1�O?7v{��I       6%�	�R�Z���A�-*;


total_loss���@

error_R�$V?

learning_rate_1�O?7��W�I       6%�	^¾Z���A�-*;


total_loss�@

error_R�`?

learning_rate_1�O?7C�{_I       6%�	�Z���A�-*;


total_loss���@

error_R8�S?

learning_rate_1�O?7ڪ�I       6%�	�P�Z���A�-*;


total_loss�C�@

error_R!eC?

learning_rate_1�O?7���I       6%�	6��Z���A�-*;


total_lossl�@

error_R�BR?

learning_rate_1�O?7�@�I       6%�	x��Z���A�-*;


total_loss���@

error_R�Z?

learning_rate_1�O?7Խ!!I       6%�	<<�Z���A�-*;


total_lossH��@

error_RR�??

learning_rate_1�O?7�,��I       6%�	���Z���A�-*;


total_losssi�@

error_R_�N?

learning_rate_1�O?7�y�I       6%�	��Z���A�-*;


total_loss�S�@

error_R8>J?

learning_rate_1�O?7^Q?�I       6%�	&,�Z���A�-*;


total_lossv��@

error_R�N?

learning_rate_1�O?7�m̄I       6%�	�m�Z���A�-*;


total_loss���@

error_Rv^?

learning_rate_1�O?7H�RI       6%�	ֳ�Z���A�-*;


total_loss�(w@

error_R=�N?

learning_rate_1�O?72�rI       6%�	���Z���A�-*;


total_lossi�@

error_R�4R?

learning_rate_1�O?7e�.#I       6%�	�>�Z���A�-*;


total_loss1P�@

error_R��K?

learning_rate_1�O?7+ ��I       6%�	Ѓ�Z���A�-*;


total_loss���@

error_RZ�>?

learning_rate_1�O?7&d��I       6%�		��Z���A�-*;


total_lossV�@

error_R?�U?

learning_rate_1�O?7.z@9I       6%�	�:�Z���A�-*;


total_lossmƬ@

error_R�*Z?

learning_rate_1�O?7l̂I       6%�	H��Z���A�-*;


total_losst�@

error_R[�F?

learning_rate_1�O?7�4��I       6%�	���Z���A�-*;


total_loss�(�@

error_Rf�H?

learning_rate_1�O?7�HP0I       6%�	��Z���A�-*;


total_loss���@

error_Rn1??

learning_rate_1�O?79k�I       6%�	FZ�Z���A�-*;


total_loss'��@

error_RIPZ?

learning_rate_1�O?7|VV�I       6%�	��Z���A�-*;


total_loss?	�@

error_R&:;?

learning_rate_1�O?7�/�I       6%�	�Z���A�-*;


total_loss�`�@

error_R��4?

learning_rate_1�O?7׸�ZI       6%�	�J�Z���A�-*;


total_loss7_@

error_R�&6?

learning_rate_1�O?7}��VI       6%�	i��Z���A�-*;


total_loss-�@

error_R�=?

learning_rate_1�O?73��<I       6%�	���Z���A�-*;


total_loss6˝@

error_Ri�I?

learning_rate_1�O?7S�I       6%�	��Z���A�-*;


total_loss*J�@

error_RC"9?

learning_rate_1�O?7e��=I       6%�	IX�Z���A�-*;


total_loss\�@

error_R�2P?

learning_rate_1�O?7�ͰmI       6%�	Ҝ�Z���A�-*;


total_loss�(�@

error_R�`J?

learning_rate_1�O?7�th�I       6%�	y��Z���A�-*;


total_loss!u�@

error_RJD:?

learning_rate_1�O?7S,�FI       6%�	�#�Z���A�-*;


total_loss�(�@

error_R�%L?

learning_rate_1�O?7�2��I       6%�	�d�Z���A�-*;


total_lossx��@

error_R�P?

learning_rate_1�O?7}��I       6%�	Ӥ�Z���A�-*;


total_loss��A

error_RI?

learning_rate_1�O?7u��#I       6%�	���Z���A�-*;


total_loss�c�@

error_R��5?

learning_rate_1�O?7}5~~I       6%�	&�Z���A�-*;


total_lossX~�@

error_R{�U?

learning_rate_1�O?7�_�SI       6%�	�f�Z���A�-*;


total_loss���@

error_R�>?

learning_rate_1�O?7�Z�I       6%�	!��Z���A�-*;


total_lossX|�@

error_R�sT?

learning_rate_1�O?7��`I       6%�	��Z���A�-*;


total_loss�4�@

error_R��W?

learning_rate_1�O?7��I       6%�	�)�Z���A�-*;


total_loss!-�@

error_R��a?

learning_rate_1�O?7�5�I       6%�	_k�Z���A�-*;


total_lossLm�@

error_R�P?

learning_rate_1�O?7|�7/I       6%�	���Z���A�-*;


total_loss1��@

error_Ra�U?

learning_rate_1�O?7f��I       6%�	���Z���A�-*;


total_loss��@

error_R4LZ?

learning_rate_1�O?7g��I       6%�	�.�Z���A�-*;


total_lossh�@

error_R_�Y?

learning_rate_1�O?7��`I       6%�	�t�Z���A�-*;


total_loss�ng@

error_RWZI?

learning_rate_1�O?7J9��I       6%�	Z��Z���A�-*;


total_lossL[u@

error_R7�B?

learning_rate_1�O?7����I       6%�	#�Z���A�-*;


total_loss�@

error_R�BF?

learning_rate_1�O?7��.I       6%�	��Z���A�-*;


total_loss��^@

error_R�5?

learning_rate_1�O?7��}I       6%�	���Z���A�-*;


total_loss�?�@

error_R}HW?

learning_rate_1�O?7Q�I       6%�	�Y�Z���A�-*;


total_loss���@

error_R�gG?

learning_rate_1�O?7T�("I       6%�	���Z���A�-*;


total_loss�e�@

error_RD�R?

learning_rate_1�O?7��^I       6%�	���Z���A�-*;


total_loss�қ@

error_R\S??

learning_rate_1�O?7A-xKI       6%�	�.�Z���A�-*;


total_loss���@

error_R8G?

learning_rate_1�O?7}��I       6%�	��Z���A�-*;


total_loss�O�@

error_RF0@?

learning_rate_1�O?7�Ȩ�I       6%�	T��Z���A�-*;


total_loss���@

error_R��W?

learning_rate_1�O?7����I       6%�	��Z���A�-*;


total_loss���@

error_Rv�F?

learning_rate_1�O?7u�u�I       6%�	�\�Z���A�-*;


total_loss�@

error_R6�K?

learning_rate_1�O?7�f�I       6%�	y��Z���A�-*;


total_loss9�@

error_R��Z?

learning_rate_1�O?7�˷I       6%�	��Z���A�-*;


total_lossX�X@

error_R��A?

learning_rate_1�O?7�p��I       6%�	�-�Z���A�-*;


total_losss�@

error_R��U?

learning_rate_1�O?7�V�(I       6%�	�w�Z���A�-*;


total_loss���@

error_R� _?

learning_rate_1�O?7N�I       6%�	��Z���A�-*;


total_lossjm�@

error_R6 7?

learning_rate_1�O?7�T��I       6%�	��Z���A�-*;


total_loss:t�@

error_R��f?

learning_rate_1�O?7���I       6%�	qI�Z���A�-*;


total_loss�	�@

error_Rn6W?

learning_rate_1�O?79��I       6%�	���Z���A�-*;


total_loss���@

error_R�DX?

learning_rate_1�O?7���GI       6%�	N��Z���A�-*;


total_losszp�@

error_R-f]?

learning_rate_1�O?7�-��I       6%�	�Z���A�-*;


total_loss)4�@

error_R� R?

learning_rate_1�O?7��S�I       6%�	�a�Z���A�-*;


total_loss��@

error_R�Y?

learning_rate_1�O?7�ƶ�I       6%�	��Z���A�-*;


total_loss���@

error_R�9J?

learning_rate_1�O?7,\��I       6%�	��Z���A�-*;


total_lossc� A

error_R��H?

learning_rate_1�O?7�F�I       6%�	"9�Z���A�-*;


total_lossAN�@

error_R�9?

learning_rate_1�O?7�|d�I       6%�	p��Z���A�-*;


total_loss�T�@

error_R�R?

learning_rate_1�O?70�]I       6%�	s��Z���A�-*;


total_lossx��@

error_R]�R?

learning_rate_1�O?7L�"�I       6%�	��Z���A�-*;


total_lossF��@

error_R�S?

learning_rate_1�O?7�A�I       6%�	:T�Z���A�-*;


total_loss�
�@

error_R��M?

learning_rate_1�O?7 �]I       6%�	v��Z���A�-*;


total_loss�k�@

error_R�1B?

learning_rate_1�O?7���I       6%�	���Z���A�-*;


total_loss�њ@

error_R�cM?

learning_rate_1�O?7vJ� I       6%�	��Z���A�-*;


total_lossD�@

error_Rw�O?

learning_rate_1�O?7���I       6%�	rk�Z���A�-*;


total_lossy�@

error_R�A?

learning_rate_1�O?7��yI       6%�	<��Z���A�-*;


total_loss���@

error_R��G?

learning_rate_1�O?7AahJI       6%�	9��Z���A�-*;


total_loss�dA

error_RaH?

learning_rate_1�O?7o��0I       6%�	5C�Z���A�-*;


total_loss%ݟ@

error_R�=?

learning_rate_1�O?7f��I       6%�	o��Z���A�-*;


total_loss鎁@

error_R�o>?

learning_rate_1�O?7ٛ�I       6%�	���Z���A�-*;


total_loss�n�@

error_R�~@?

learning_rate_1�O?7�̒I       6%�	<�Z���A�-*;


total_loss�A�@

error_R�%F?

learning_rate_1�O?7v�I       6%�	{[�Z���A�-*;


total_loss�@

error_R�vO?

learning_rate_1�O?7G�~I       6%�	>��Z���A�-*;


total_loss�8�@

error_R1�G?

learning_rate_1�O?7�oLI       6%�	y��Z���A�-*;


total_loss�A

error_Rdpl?

learning_rate_1�O?7*'�I       6%�	^0�Z���A�-*;


total_lossV�A

error_R�\Q?

learning_rate_1�O?7��I       6%�	�y�Z���A�-*;


total_lossVg�@

error_R��O?

learning_rate_1�O?7��I       6%�	#��Z���A�-*;


total_loss��@

error_R;:;?

learning_rate_1�O?7w5�I       6%�	���Z���A�-*;


total_loss)�@

error_R�~X?

learning_rate_1�O?7�%T�I       6%�	E�Z���A�-*;


total_losso�@

error_R$�G?

learning_rate_1�O?7:a|!I       6%�	`��Z���A�-*;


total_loss�X�@

error_R��Q?

learning_rate_1�O?7Y���I       6%�	���Z���A�-*;


total_loss;�@

error_RI@??

learning_rate_1�O?7�u�I       6%�	��Z���A�-*;


total_losse�MA

error_R҂R?

learning_rate_1�O?7�0�^I       6%�	�]�Z���A�-*;


total_loss�K�@

error_R�PD?

learning_rate_1�O?7Pm��I       6%�	E��Z���A�-*;


total_loss��@

error_RȽJ?

learning_rate_1�O?7���I       6%�	%��Z���A�-*;


total_lossd"�@

error_Rc�E?

learning_rate_1�O?7�9��I       6%�	r'�Z���A�-*;


total_loss�Ȍ@

error_Rm�@?

learning_rate_1�O?7�[�I       6%�	or�Z���A�-*;


total_loss��@

error_R�yN?

learning_rate_1�O?7�D� I       6%�	���Z���A�-*;


total_loss�;�@

error_R��g?

learning_rate_1�O?7��RI       6%�	�Z���A�-*;


total_lossÚA

error_R�)I?

learning_rate_1�O?7v ��I       6%�	JF�Z���A�-*;


total_lossd�@

error_R,O?

learning_rate_1�O?7���I       6%�	��Z���A�-*;


total_loss�Ү@

error_R�>8?

learning_rate_1�O?7����I       6%�	X��Z���A�-*;


total_loss���@

error_R.GE?

learning_rate_1�O?7��OwI       6%�	�>�Z���A�-*;


total_loss�n�@

error_RsB?

learning_rate_1�O?7D �I       6%�	=��Z���A�-*;


total_loss/��@

error_RW'F?

learning_rate_1�O?7����I       6%�	��Z���A�-*;


total_lossȞ�@

error_R�Wc?

learning_rate_1�O?7!=�I       6%�	�Z���A�-*;


total_loss���@

error_R%{G?

learning_rate_1�O?7Q`�I       6%�	�\�Z���A�-*;


total_loss-��@

error_RdOG?

learning_rate_1�O?7e+fjI       6%�	���Z���A�-*;


total_loss��@

error_R)�]?

learning_rate_1�O?7u�/�I       6%�	���Z���A�.*;


total_lossT��@

error_REP?

learning_rate_1�O?7� I       6%�	b?�Z���A�.*;


total_loss���@

error_R>B?

learning_rate_1�O?7x"W1I       6%�	���Z���A�.*;


total_loss�Gv@

error_R��>?

learning_rate_1�O?7��77I       6%�	���Z���A�.*;


total_loss�W�@

error_RƶY?

learning_rate_1�O?7ág�I       6%�	R�Z���A�.*;


total_loss��A

error_R@�L?

learning_rate_1�O?7�<�1I       6%�	~T�Z���A�.*;


total_loss�k�@

error_R.�U?

learning_rate_1�O?7�� I       6%�	%��Z���A�.*;


total_loss���@

error_R}GJ?

learning_rate_1�O?7���_I       6%�	e��Z���A�.*;


total_lossJN�@

error_R�^?

learning_rate_1�O?77֤GI       6%�	��Z���A�.*;


total_loss&Ҩ@

error_R@:H?

learning_rate_1�O?7��I       6%�	Vf�Z���A�.*;


total_loss?@

error_RT�H?

learning_rate_1�O?7�ѫhI       6%�	���Z���A�.*;


total_loss�Bt@

error_R�G[?

learning_rate_1�O?7�d=�I       6%�	d�Z���A�.*;


total_loss@Ɂ@

error_R��G?

learning_rate_1�O?7rCO�I       6%�	 f�Z���A�.*;


total_loss��@

error_RK]?

learning_rate_1�O?7����I       6%�	`��Z���A�.*;


total_lossj�@

error_Rht8?

learning_rate_1�O?7�M^I       6%�	
�Z���A�.*;


total_loss�w�@

error_R�-N?

learning_rate_1�O?7G�� I       6%�	�Q�Z���A�.*;


total_lossB�@

error_R�N?

learning_rate_1�O?7�	��I       6%�	���Z���A�.*;


total_loss�/�@

error_R�zC?

learning_rate_1�O?7A|�bI       6%�	���Z���A�.*;


total_loss���@

error_R�}Z?

learning_rate_1�O?7��.I       6%�		!�Z���A�.*;


total_loss찠@

error_RçR?

learning_rate_1�O?7�a�I       6%�	�k�Z���A�.*;


total_loss�8�@

error_R�H>?

learning_rate_1�O?7x�ynI       6%�	���Z���A�.*;


total_loss�{�@

error_R=U?

learning_rate_1�O?7�30I       6%�	L��Z���A�.*;


total_loss�=�@

error_R,L?

learning_rate_1�O?7���WI       6%�	'>�Z���A�.*;


total_loss��@

error_RR?

learning_rate_1�O?7��zI       6%�	��Z���A�.*;


total_lossj�@

error_R}??

learning_rate_1�O?7����I       6%�	��Z���A�.*;


total_loss�j�@

error_R�T?

learning_rate_1�O?7���I       6%�	b�Z���A�.*;


total_loss��@

error_R8qJ?

learning_rate_1�O?7�O��I       6%�	|P�Z���A�.*;


total_loss�@

error_R��I?

learning_rate_1�O?7�S�I       6%�	L��Z���A�.*;


total_loss��	A

error_R@yG?

learning_rate_1�O?7^+�I       6%�	���Z���A�.*;


total_loss��@

error_R�F?

learning_rate_1�O?7X`��I       6%�	��Z���A�.*;


total_loss��@

error_R��X?

learning_rate_1�O?7����I       6%�	�\�Z���A�.*;


total_lossaK�@

error_RR0B?

learning_rate_1�O?7���EI       6%�	i��Z���A�.*;


total_loss\�@

error_R!�O?

learning_rate_1�O?7Z:�I       6%�	���Z���A�.*;


total_losse��@

error_R�va?

learning_rate_1�O?7��"/I       6%�	:,�Z���A�.*;


total_loss�V�@

error_R�e?

learning_rate_1�O?7�I       6%�	�u�Z���A�.*;


total_loss�ޝ@

error_R
�P?

learning_rate_1�O?7�O�I       6%�	h��Z���A�.*;


total_loss�l�@

error_RSR?

learning_rate_1�O?7���I       6%�	��Z���A�.*;


total_loss֩A

error_R�9Q?

learning_rate_1�O?7���I       6%�	>B�Z���A�.*;


total_loss��@

error_RQ�@?

learning_rate_1�O?7�O;tI       6%�	؉�Z���A�.*;


total_lossF�@

error_R��N?

learning_rate_1�O?7*�I       6%�	���Z���A�.*;


total_loss'��@

error_R@k^?

learning_rate_1�O?7��RI       6%�	U�Z���A�.*;


total_loss�$�@

error_R&�f?

learning_rate_1�O?7��[hI       6%�	NS�Z���A�.*;


total_loss��@

error_Rl�V?

learning_rate_1�O?7�fQI       6%�	)��Z���A�.*;


total_lossnqv@

error_R��B?

learning_rate_1�O?7M�I       6%�	&��Z���A�.*;


total_losst
�@

error_R�TP?

learning_rate_1�O?7���EI       6%�	��Z���A�.*;


total_loss��@

error_R�BF?

learning_rate_1�O?7�e�I       6%�	�_�Z���A�.*;


total_lossJ��@

error_R�=L?

learning_rate_1�O?78l�I       6%�	&��Z���A�.*;


total_loss��@

error_RO�H?

learning_rate_1�O?7��L�I       6%�	N��Z���A�.*;


total_loss�@

error_RVP?

learning_rate_1�O?7"���I       6%�	�/�Z���A�.*;


total_loss��@

error_R`�G?

learning_rate_1�O?7EȐQI       6%�	���Z���A�.*;


total_loss �@

error_R&*S?

learning_rate_1�O?79a�_I       6%�	���Z���A�.*;


total_loss�P�@

error_R}�;?

learning_rate_1�O?7� �;I       6%�	�#�Z���A�.*;


total_loss�C�@

error_R�i_?

learning_rate_1�O?7�	@�I       6%�	�k�Z���A�.*;


total_loss�s�@

error_R�=I?

learning_rate_1�O?7R\�jI       6%�	^��Z���A�.*;


total_lossf=�@

error_R*W?

learning_rate_1�O?7J�a�I       6%�	��Z���A�.*;


total_loss���@

error_R��G?

learning_rate_1�O?7�ՋI       6%�	�5�Z���A�.*;


total_lossV�@

error_R�D?

learning_rate_1�O?7ʇ�I       6%�	sy�Z���A�.*;


total_loss��@

error_R�P?

learning_rate_1�O?7	���I       6%�		��Z���A�.*;


total_lossd��@

error_R�A?

learning_rate_1�O?7nސ�I       6%�	J�Z���A�.*;


total_loss�@

error_RE�H?

learning_rate_1�O?7�N�I       6%�	�N�Z���A�.*;


total_lossaɼ@

error_RR�J?

learning_rate_1�O?7��3%I       6%�	o��Z���A�.*;


total_lossn�@

error_R��S?

learning_rate_1�O?7|v�I       6%�	��Z���A�.*;


total_loss�L�@

error_R*�@?

learning_rate_1�O?7;	I       6%�	w�Z���A�.*;


total_loss���@

error_R�@W?

learning_rate_1�O?7�~�I       6%�	�c�Z���A�.*;


total_loss�5A

error_R@@K?

learning_rate_1�O?7�jZ	I       6%�	#��Z���A�.*;


total_loss/��@

error_R�G?

learning_rate_1�O?7R���I       6%�	���Z���A�.*;


total_loss�ҟ@

error_R��B?

learning_rate_1�O?7�qq`I       6%�	V.�Z���A�.*;


total_lossz�5@

error_Rd F?

learning_rate_1�O?7�s�;I       6%�	qp�Z���A�.*;


total_loss=RA

error_R��E?

learning_rate_1�O?7l�!,I       6%�	,��Z���A�.*;


total_loss�O�@

error_R.~<?

learning_rate_1�O?7^~EqI       6%�	H��Z���A�.*;


total_loss��@

error_R��Y?

learning_rate_1�O?7M�\�I       6%�	�8�Z���A�.*;


total_losso9�@

error_R��4?

learning_rate_1�O?7:_2�I       6%�	�{�Z���A�.*;


total_loss#��@

error_R�1@?

learning_rate_1�O?7P��I       6%�	���Z���A�.*;


total_loss���@

error_R@�O?

learning_rate_1�O?7���I       6%�	��Z���A�.*;


total_lossWs�@

error_R
bK?

learning_rate_1�O?7\t�I       6%�	V�Z���A�.*;


total_loss���@

error_R�F?

learning_rate_1�O?7_ͺ_I       6%�	��Z���A�.*;


total_loss ��@

error_Rn�P?

learning_rate_1�O?7���I       6%�	���Z���A�.*;


total_loss���@

error_Rt�_?

learning_rate_1�O?7"c��I       6%�	�'�Z���A�.*;


total_loss���@

error_Rl�V?

learning_rate_1�O?7�ƽqI       6%�	�p�Z���A�.*;


total_loss�8�@

error_R��V?

learning_rate_1�O?7{w(GI       6%�	X��Z���A�.*;


total_loss�8�@

error_R�>?

learning_rate_1�O?7���I       6%�	=��Z���A�.*;


total_loss��@

error_Rd;?

learning_rate_1�O?7�Y�3I       6%�	`@�Z���A�.*;


total_loss�P�@

error_R�
R?

learning_rate_1�O?7��mI       6%�	ڃ�Z���A�.*;


total_loss�h�@

error_RxL?

learning_rate_1�O?7,��0I       6%�	���Z���A�.*;


total_loss�э@

error_R N?

learning_rate_1�O?7���I       6%�	��Z���A�.*;


total_loss�@

error_R�HV?

learning_rate_1�O?7G��GI       6%�	�S�Z���A�.*;


total_loss?�@

error_RD�O?

learning_rate_1�O?7�SdI       6%�	m��Z���A�.*;


total_loss���@

error_RD�Z?

learning_rate_1�O?7��k�I       6%�	\��Z���A�.*;


total_lossNQ�@

error_R�iV?

learning_rate_1�O?7��I       6%�	'�Z���A�.*;


total_loss<�@

error_RrJ?

learning_rate_1�O?7�B �I       6%�	+k�Z���A�.*;


total_loss�w�@

error_Rs`R?

learning_rate_1�O?7=�ϩI       6%�	F��Z���A�.*;


total_loss�3�@

error_RQU?

learning_rate_1�O?7%W3�I       6%�	h��Z���A�.*;


total_loss:��@

error_Rv�a?

learning_rate_1�O?76j�I       6%�	<�Z���A�.*;


total_loss�\o@

error_R��Y?

learning_rate_1�O?7�z<{I       6%�	6��Z���A�.*;


total_loss�i�@

error_R�RM?

learning_rate_1�O?7�W��I       6%�	q��Z���A�.*;


total_loss�M�@

error_R�O?

learning_rate_1�O?7$�2�I       6%�	x	�Z���A�.*;


total_loss�9�@

error_R1�H?

learning_rate_1�O?7�(T�I       6%�	�K�Z���A�.*;


total_lossU�@

error_R�tW?

learning_rate_1�O?7���pI       6%�	ˏ�Z���A�.*;


total_loss�d�@

error_R��6?

learning_rate_1�O?71w�	I       6%�	���Z���A�.*;


total_loss���@

error_R-�U?

learning_rate_1�O?7��KI       6%�	�'�Z���A�.*;


total_loss��@

error_R {R?

learning_rate_1�O?7}���I       6%�	�p�Z���A�.*;


total_loss�̏@

error_R�cM?

learning_rate_1�O?7՝'I       6%�	y��Z���A�.*;


total_loss:��@

error_R��P?

learning_rate_1�O?7�g��I       6%�	� �Z���A�.*;


total_loss�U�@

error_R1�R?

learning_rate_1�O?7D��I       6%�	�G�Z���A�.*;


total_loss�a�@

error_R:�;?

learning_rate_1�O?7 &��I       6%�	u��Z���A�.*;


total_lossir�@

error_RO%B?

learning_rate_1�O?7�c�I       6%�	g��Z���A�.*;


total_loss,��@

error_R��X?

learning_rate_1�O?7Zu�I       6%�	��Z���A�.*;


total_loss*O�@

error_R)�N?

learning_rate_1�O?7;��I       6%�	�\�Z���A�.*;


total_loss��@

error_Rl6T?

learning_rate_1�O?7��UuI       6%�	���Z���A�.*;


total_loss���@

error_R�wV?

learning_rate_1�O?7+Ц7I       6%�	b�Z���A�.*;


total_loss��@

error_R��a?

learning_rate_1�O?7��NI       6%�	�`�Z���A�.*;


total_lossJM�@

error_R�5`?

learning_rate_1�O?7Է8I       6%�	��Z���A�.*;


total_loss��A

error_R1�I?

learning_rate_1�O?7��j;I       6%�	���Z���A�.*;


total_loss�Q�@

error_R@&J?

learning_rate_1�O?7N��I       6%�	[; [���A�.*;


total_loss���@

error_R�iL?

learning_rate_1�O?7�"tqI       6%�	�� [���A�.*;


total_loss��@

error_Rf.B?

learning_rate_1�O?73
n{I       6%�	� [���A�.*;


total_lossX��@

error_R��E?

learning_rate_1�O?7P�)�I       6%�	�![���A�.*;


total_lossU�A

error_R�R?

learning_rate_1�O?7mbyI       6%�	�g[���A�.*;


total_losst��@

error_Ri�E?

learning_rate_1�O?7vF��I       6%�	��[���A�.*;


total_loss$t�@

error_R}�E?

learning_rate_1�O?7Z��lI       6%�	F�[���A�.*;


total_loss
L�@

error_RaW?

learning_rate_1�O?7���I       6%�	�9[���A�.*;


total_loss<M�@

error_RR?

learning_rate_1�O?7
�II       6%�	A�[���A�.*;


total_lossF\�@

error_Rn�??

learning_rate_1�O?7�b�I       6%�	��[���A�.*;


total_lossVP�@

error_R�lI?

learning_rate_1�O?7�
�I       6%�	�[���A�.*;


total_loss=V�@

error_RԄ[?

learning_rate_1�O?7=
nI       6%�	8T[���A�.*;


total_loss�H�@

error_R-�R?

learning_rate_1�O?7;GI       6%�	��[���A�.*;


total_loss�ޡ@

error_R�0G?

learning_rate_1�O?7�NX2I       6%�	g�[���A�.*;


total_loss�r�@

error_R�.D?

learning_rate_1�O?7O��7I       6%�	G$[���A�.*;


total_lossxĞ@

error_R��W?

learning_rate_1�O?7�0��I       6%�	�h[���A�/*;


total_lossJS�@

error_Rf�:?

learning_rate_1�O?7��d�I       6%�	��[���A�/*;


total_loss�@

error_R�P?

learning_rate_1�O?7<�iI       6%�	��[���A�/*;


total_loss���@

error_R�QZ?

learning_rate_1�O?7��iI       6%�	�6[���A�/*;


total_lossAӯ@

error_R��U?

learning_rate_1�O?7�r�iI       6%�	�z[���A�/*;


total_loss��@

error_R��F?

learning_rate_1�O?7��P4I       6%�	G�[���A�/*;


total_loss���@

error_R(�N?

learning_rate_1�O?7úE(I       6%�	=[���A�/*;


total_loss	��@

error_R��S?

learning_rate_1�O?7��I       6%�	kF[���A�/*;


total_loss��@

error_RVKO?

learning_rate_1�O?7��Q0I       6%�	^�[���A�/*;


total_loss4�@

error_R�T?

learning_rate_1�O?7{S�I       6%�	�[���A�/*;


total_loss���@

error_Ri�F?

learning_rate_1�O?7аW�I       6%�	S-[���A�/*;


total_loss���@

error_R]Y?

learning_rate_1�O?7�̋?I       6%�	�u[���A�/*;


total_loss��@

error_R�D?

learning_rate_1�O?7�΂I       6%�	��[���A�/*;


total_lossV��@

error_R�S]?

learning_rate_1�O?7�Ķ�I       6%�	�[���A�/*;


total_loss���@

error_R�G?

learning_rate_1�O?7��I       6%�	�H[���A�/*;


total_lossփ�@

error_R��Z?

learning_rate_1�O?7��%�I       6%�	��[���A�/*;


total_loss���@

error_R�#Y?

learning_rate_1�O?7~M_^I       6%�	��[���A�/*;


total_loss��@

error_RHJ?

learning_rate_1�O?7���1I       6%�	R 	[���A�/*;


total_loss���@

error_R,vF?

learning_rate_1�O?7
/�I       6%�	h	[���A�/*;


total_loss�s�@

error_Ro U?

learning_rate_1�O?7l�I       6%�	�	[���A�/*;


total_loss=�@

error_R��U?

learning_rate_1�O?7�S��I       6%�	�	[���A�/*;


total_lossCv�@

error_R�YK?

learning_rate_1�O?7.��I       6%�	�E
[���A�/*;


total_loss�� A

error_R�O?

learning_rate_1�O?7��8_I       6%�	[�
[���A�/*;


total_loss�I�@

error_R��@?

learning_rate_1�O?7[�I�I       6%�	{�
[���A�/*;


total_loss�`�@

error_R��^?

learning_rate_1�O?7��I       6%�	�4[���A�/*;


total_loss�S�@

error_R)�V?

learning_rate_1�O?7�[I       6%�	�[���A�/*;


total_loss���@

error_R�O?

learning_rate_1�O?7�I       6%�	W�[���A�/*;


total_loss��@

error_RO�Q?

learning_rate_1�O?7OߤI       6%�	�1[���A�/*;


total_lossw$�@

error_R��C?

learning_rate_1�O?72I       6%�	�[���A�/*;


total_loss���@

error_Rl�V?

learning_rate_1�O?7�4�tI       6%�	A�[���A�/*;


total_loss���@

error_R@�K?

learning_rate_1�O?7 ���I       6%�	�"[���A�/*;


total_loss/j�@

error_R�F;?

learning_rate_1�O?7�%`/I       6%�	�h[���A�/*;


total_loss�[�@

error_R��J?

learning_rate_1�O?7�V?�I       6%�	��[���A�/*;


total_lossY/�@

error_R��K?

learning_rate_1�O?7S���I       6%�	�[���A�/*;


total_losss[�@

error_R;R?

learning_rate_1�O?7p�I       6%�	ri[���A�/*;


total_loss}M�@

error_R��\?

learning_rate_1�O?7x�I       6%�	S�[���A�/*;


total_loss3A

error_R`�J?

learning_rate_1�O?7�u�I       6%�	u�[���A�/*;


total_lossԜu@

error_R�A?

learning_rate_1�O?7$C%*I       6%�	a?[���A�/*;


total_loss��@

error_Rz�[?

learning_rate_1�O?7�I       6%�	>�[���A�/*;


total_loss)��@

error_R��U?

learning_rate_1�O?7���I       6%�	��[���A�/*;


total_loss㸌@

error_R �R?

learning_rate_1�O?7@�2I       6%�	CC[���A�/*;


total_loss_0�@

error_RO"P?

learning_rate_1�O?7�=I       6%�	��[���A�/*;


total_loss��@

error_R-E?

learning_rate_1�O?7�bI       6%�	i�[���A�/*;


total_loss/=�@

error_R��E?

learning_rate_1�O?7B��I       6%�	OW[���A�/*;


total_loss٨�@

error_R��P?

learning_rate_1�O?7?OcI       6%�	u�[���A�/*;


total_loss�d�@

error_R�jE?

learning_rate_1�O?7|=�I       6%�	��[���A�/*;


total_loss��@

error_R��L?

learning_rate_1�O?7�]�^I       6%�	v)[���A�/*;


total_loss��@

error_R@|<?

learning_rate_1�O?7���I       6%�	�o[���A�/*;


total_loss�l�@

error_R(�*?

learning_rate_1�O?7G�mI       6%�	��[���A�/*;


total_losss�@

error_R.cD?

learning_rate_1�O?7����I       6%�	�[���A�/*;


total_lossT��@

error_Rm�\?

learning_rate_1�O?7�?}�I       6%�	 ?[���A�/*;


total_loss֤�@

error_R��V?

learning_rate_1�O?7k��|I       6%�	[�[���A�/*;


total_loss���@

error_RZOL?

learning_rate_1�O?7rJ�LI       6%�	��[���A�/*;


total_loss�t�@

error_R:�<?

learning_rate_1�O?7�S�
I       6%�	
[���A�/*;


total_loss�R�@

error_R��Z?

learning_rate_1�O?7�^"�I       6%�	PU[���A�/*;


total_loss���@

error_R7w<?

learning_rate_1�O?7��I       6%�	��[���A�/*;


total_loss��@

error_RΛ\?

learning_rate_1�O?7�4$>I       6%�	 �[���A�/*;


total_lossV5	A

error_RڳP?

learning_rate_1�O?7����I       6%�	�#[���A�/*;


total_loss��@

error_Rx�c?

learning_rate_1�O?7X��.I       6%�	�j[���A�/*;


total_loss��@

error_R��=?

learning_rate_1�O?7Е?I       6%�	 �[���A�/*;


total_loss��@

error_R��U?

learning_rate_1�O?7�	��I       6%�	�"[���A�/*;


total_loss�A

error_R�mC?

learning_rate_1�O?7׊_�I       6%�	n[���A�/*;


total_loss���@

error_R
�W?

learning_rate_1�O?7y(�I       6%�	��[���A�/*;


total_loss��@

error_RciI?

learning_rate_1�O?72�I       6%�	4[���A�/*;


total_loss�}@

error_R?�E?

learning_rate_1�O?74��rI       6%�	�U[���A�/*;


total_lossQ�@

error_R�D?

learning_rate_1�O?7�8"2I       6%�	��[���A�/*;


total_loss��{@

error_R�*??

learning_rate_1�O?7��7:I       6%�	F�[���A�/*;


total_lossC�@

error_R�sX?

learning_rate_1�O?7CF#I       6%�	V.[���A�/*;


total_loss���@

error_Rv�`?

learning_rate_1�O?7�'A<I       6%�	p[���A�/*;


total_loss� �@

error_R�B?

learning_rate_1�O?7�߬I       6%�	��[���A�/*;


total_lossKE@

error_R��h?

learning_rate_1�O?7���I       6%�	��[���A�/*;


total_loss���@

error_R�3L?

learning_rate_1�O?7�#i�I       6%�	5B[���A�/*;


total_loss�#�@

error_R�}??

learning_rate_1�O?7�u\I       6%�	e�[���A�/*;


total_loss:��@

error_R��Q?

learning_rate_1�O?7r��I       6%�	C�[���A�/*;


total_lossX��@

error_R��V?

learning_rate_1�O?7i��I       6%�	^[���A�/*;


total_loss	�q@

error_Rf�??

learning_rate_1�O?7$�I       6%�	dR[���A�/*;


total_loss׍�@

error_R{�W?

learning_rate_1�O?7`_I       6%�	R�[���A�/*;


total_lossC��@

error_R��G?

learning_rate_1�O?7a��I       6%�	X�[���A�/*;


total_loss�65A

error_R(�W?

learning_rate_1�O?7�h�I       6%�	�[���A�/*;


total_loss-G�@

error_R��J?

learning_rate_1�O?7�F=�I       6%�	�i[���A�/*;


total_loss���@

error_R�HS?

learning_rate_1�O?7R�N�I       6%�	��[���A�/*;


total_loss�|�@

error_RXB[?

learning_rate_1�O?7AO��I       6%�	�I[���A�/*;


total_lossҲ�@

error_R�V?

learning_rate_1�O?7���I       6%�	��[���A�/*;


total_losso�@

error_Rϲb?

learning_rate_1�O?7x���I       6%�	�[���A�/*;


total_loss�|�@

error_R3�H?

learning_rate_1�O?7M��I       6%�	�$[���A�/*;


total_loss�1�@

error_R�&_?

learning_rate_1�O?7:o��I       6%�	�k[���A�/*;


total_lossw0A

error_R$�^?

learning_rate_1�O?7�[}�I       6%�	n�[���A�/*;


total_loss��Z@

error_RN<?

learning_rate_1�O?7#�i�I       6%�	u[���A�/*;


total_lossrA�@

error_Re�K?

learning_rate_1�O?7�JjI       6%�	^t[���A�/*;


total_loss��@

error_R�S?

learning_rate_1�O?7�XI       6%�	��[���A�/*;


total_lossI��@

error_R�/X?

learning_rate_1�O?7�b��I       6%�	��[���A�/*;


total_loss)��@

error_R��??

learning_rate_1�O?7�_��I       6%�	CE[���A�/*;


total_loss���@

error_RפF?

learning_rate_1�O?7�~�I       6%�	��[���A�/*;


total_loss߼�@

error_R�^A?

learning_rate_1�O?7�%��I       6%�	��[���A�/*;


total_loss�PA

error_RrsF?

learning_rate_1�O?7YU$I       6%�	� [���A�/*;


total_loss�4�@

error_R�#X?

learning_rate_1�O?7��z
I       6%�	7_ [���A�/*;


total_loss!�@

error_R�3F?

learning_rate_1�O?7H�{eI       6%�	E� [���A�/*;


total_lossQ�v@

error_R�Z?

learning_rate_1�O?7��.CI       6%�	�![���A�/*;


total_loss�@

error_Rj�I?

learning_rate_1�O?7 �|I       6%�	�O![���A�/*;


total_lossm;�@

error_RχG?

learning_rate_1�O?7Z2D�I       6%�	�![���A�/*;


total_loss^�@

error_RiO<?

learning_rate_1�O?7jmPI       6%�	��![���A�/*;


total_loss�V�@

error_R��:?

learning_rate_1�O?7t��lI       6%�	<"[���A�/*;


total_lossz��@

error_RߗU?

learning_rate_1�O?7�`6yI       6%�	�_"[���A�/*;


total_loss� �@

error_R�0G?

learning_rate_1�O?7R8U�I       6%�	y�"[���A�/*;


total_lossz�}@

error_R� C?

learning_rate_1�O?7I       6%�	;#[���A�/*;


total_loss��@

error_R�jH?

learning_rate_1�O?7���BI       6%�	�#[���A�/*;


total_loss�!�@

error_R�;?

learning_rate_1�O?7#��I       6%�	�n$[���A�/*;


total_loss�Dr@

error_R�F?

learning_rate_1�O?72��I       6%�	��$[���A�/*;


total_loss�$q@

error_R�lJ?

learning_rate_1�O?7�AQ�I       6%�	ב%[���A�/*;


total_loss�b�@

error_R��K?

learning_rate_1�O?7�� �I       6%�	f7&[���A�/*;


total_loss/S�@

error_R��>?

learning_rate_1�O?7ͤ�gI       6%�	��&[���A�/*;


total_lossh��@

error_R��M?

learning_rate_1�O?7u���I       6%�	&�&[���A�/*;


total_loss��@

error_R
�T?

learning_rate_1�O?7�Q��I       6%�	�5'[���A�/*;


total_loss���@

error_R�&C?

learning_rate_1�O?7��/	I       6%�	��'[���A�/*;


total_loss���@

error_Rs�<?

learning_rate_1�O?7��I       6%�	#�'[���A�/*;


total_loss�QA

error_R��Y?

learning_rate_1�O?7N�jI       6%�	�>([���A�/*;


total_loss�Ҏ@

error_R�2Y?

learning_rate_1�O?7�6�I       6%�	6�([���A�/*;


total_lossqh�@

error_R�gO?

learning_rate_1�O?7�)34I       6%�	��([���A�/*;


total_loss�}@

error_R(]U?

learning_rate_1�O?7��jI       6%�	@�)[���A�/*;


total_lossr�@

error_R�/g?

learning_rate_1�O?7��$PI       6%�	�)[���A�/*;


total_loss���@

error_R�;?

learning_rate_1�O?7���I       6%�	�V*[���A�/*;


total_loss��A

error_R�,M?

learning_rate_1�O?7w�x
I       6%�	��*[���A�/*;


total_lossf{@

error_RegD?

learning_rate_1�O?7HY|I       6%�	��*[���A�/*;


total_loss���@

error_R��C?

learning_rate_1�O?7>�#I       6%�	W+[���A�/*;


total_loss4F�@

error_R�UT?

learning_rate_1�O?7�V�qI       6%�	��+[���A�/*;


total_loss�b�@

error_ROV?

learning_rate_1�O?7��I       6%�	�:,[���A�/*;


total_lossHc�@

error_R�G?

learning_rate_1�O?7�q8NI       6%�	|�,[���A�/*;


total_lossQߝ@

error_R��J?

learning_rate_1�O?7~��I       6%�	v�,[���A�/*;


total_loss!+�@

error_RW�F?

learning_rate_1�O?7��W�I       6%�	6-[���A�0*;


total_loss�{�@

error_R��n?

learning_rate_1�O?7�)�I       6%�	&V-[���A�0*;


total_loss��v@

error_R�z[?

learning_rate_1�O?7s�I       6%�	��-[���A�0*;


total_loss�y�@

error_RL�I?

learning_rate_1�O?7y���I       6%�	��-[���A�0*;


total_loss���@

error_R��g?

learning_rate_1�O?7Zf�I       6%�	( .[���A�0*;


total_lossv@�@

error_R�KG?

learning_rate_1�O?7Y��QI       6%�	�e.[���A�0*;


total_loss���@

error_Rv�S?

learning_rate_1�O?7?���I       6%�	��.[���A�0*;


total_loss�?�@

error_R�H?

learning_rate_1�O?7kX�I       6%�	��.[���A�0*;


total_loss��i@

error_R��g?

learning_rate_1�O?7��{I       6%�	�1/[���A�0*;


total_lossʙ@

error_RϵH?

learning_rate_1�O?7�裭I       6%�	�x/[���A�0*;


total_loss)�@

error_Rj9i?

learning_rate_1�O?7�|KI       6%�	��/[���A�0*;


total_losss"�@

error_R}3Q?

learning_rate_1�O?7).�I       6%�	_0[���A�0*;


total_loss���@

error_R��U?

learning_rate_1�O?7l�,�I       6%�	��0[���A�0*;


total_lossИ@

error_R �P?

learning_rate_1�O?7`��]I       6%�	��0[���A�0*;


total_lossv��@

error_R�zD?

learning_rate_1�O?7��9"I       6%�	,G1[���A�0*;


total_lossC��@

error_R�A<?

learning_rate_1�O?7��I       6%�	��1[���A�0*;


total_loss���@

error_R�PE?

learning_rate_1�O?7PU�hI       6%�	��1[���A�0*;


total_loss�@

error_R_NI?

learning_rate_1�O?7F�?�I       6%�	�52[���A�0*;


total_loss��"A

error_R�P?

learning_rate_1�O?7�o��I       6%�	2[���A�0*;


total_lossԨ�@

error_RݯE?

learning_rate_1�O?7#��cI       6%�	�2[���A�0*;


total_loss�@�@

error_R,A?

learning_rate_1�O?7�7�?I       6%�	�3[���A�0*;


total_lossD�@

error_R�yA?

learning_rate_1�O?7�I��I       6%�	^Q3[���A�0*;


total_loss�pA

error_R��F?

learning_rate_1�O?7&rTI       6%�	�3[���A�0*;


total_loss�6�@

error_R�\H?

learning_rate_1�O?7M^�I       6%�	��3[���A�0*;


total_loss[׭@

error_RO6??

learning_rate_1�O?7S�?�I       6%�	B)4[���A�0*;


total_loss��@

error_RI�T?

learning_rate_1�O?7�D:I       6%�	n4[���A�0*;


total_loss,�@

error_R�@K?

learning_rate_1�O?7p��II       6%�	��4[���A�0*;


total_loss�O�@

error_R��+?

learning_rate_1�O?7�Q�
I       6%�	M�4[���A�0*;


total_loss���@

error_R�~P?

learning_rate_1�O?7d�}�I       6%�	�75[���A�0*;


total_loss��@

error_R#�=?

learning_rate_1�O?7�|�I       6%�	f�5[���A�0*;


total_loss��@

error_R�T?

learning_rate_1�O?7b��I       6%�	��5[���A�0*;


total_loss,��@

error_R/�D?

learning_rate_1�O?7	`�I       6%�	�6[���A�0*;


total_loss�ٗ@

error_R��Z?

learning_rate_1�O?7��VZI       6%�	FW6[���A�0*;


total_lossO�@

error_RD�>?

learning_rate_1�O?7q�8�I       6%�	��6[���A�0*;


total_loss6,�@

error_R�	=?

learning_rate_1�O?7D��.I       6%�	��6[���A�0*;


total_loss�@

error_R�F?

learning_rate_1�O?73k��I       6%�	g07[���A�0*;


total_loss�t�@

error_R��B?

learning_rate_1�O?7��3�I       6%�	kw7[���A�0*;


total_loss�1�@

error_R3~:?

learning_rate_1�O?7Lb�EI       6%�	ž7[���A�0*;


total_loss\��@

error_R��C?

learning_rate_1�O?7)4B�I       6%�	�8[���A�0*;


total_loss���@

error_RHm>?

learning_rate_1�O?7����I       6%�	�G8[���A�0*;


total_lossc��@

error_RSG?

learning_rate_1�O?7:K�(I       6%�	�8[���A�0*;


total_lossHq�@

error_R��\?

learning_rate_1�O?7`S�FI       6%�	�8[���A�0*;


total_loss�(�@

error_RzE?

learning_rate_1�O?7�}��I       6%�	 9[���A�0*;


total_loss��@

error_RL?Y?

learning_rate_1�O?7���I       6%�	�\9[���A�0*;


total_loss�@

error_RtZ?

learning_rate_1�O?7Ҕ��I       6%�	i�9[���A�0*;


total_loss;��@

error_R��C?

learning_rate_1�O?7/jރI       6%�	��9[���A�0*;


total_loss@�@

error_R�H?

learning_rate_1�O?7kJ.I       6%�	-:[���A�0*;


total_losss{�@

error_Rw<S?

learning_rate_1�O?7��$�I       6%�	 s:[���A�0*;


total_lossȳ
A

error_R�{^?

learning_rate_1�O?7ȑRI       6%�	�:[���A�0*;


total_loss�z@

error_R�IT?

learning_rate_1�O?7���mI       6%�	��:[���A�0*;


total_loss��@

error_RW�F?

learning_rate_1�O?7Ъh�I       6%�	4=;[���A�0*;


total_loss1��@

error_R�@?

learning_rate_1�O?7A�|�I       6%�	�;[���A�0*;


total_loss��+A

error_R?V?

learning_rate_1�O?7q�XI       6%�	��;[���A�0*;


total_lossmE�@

error_R��=?

learning_rate_1�O?7�;TaI       6%�	�1<[���A�0*;


total_loss&�A

error_RRpW?

learning_rate_1�O?7��I       6%�	Iv<[���A�0*;


total_loss�@

error_RsI?

learning_rate_1�O?7i=��I       6%�	Ǻ<[���A�0*;


total_loss���@

error_ReML?

learning_rate_1�O?7V�#I       6%�	�=[���A�0*;


total_loss�Ձ@

error_R�P?

learning_rate_1�O?7(�A�I       6%�	zy=[���A�0*;


total_lossN�@

error_R��U?

learning_rate_1�O?7ȤI       6%�	��=[���A�0*;


total_loss��@

error_R��N?

learning_rate_1�O?7�r=�I       6%�	�>[���A�0*;


total_loss{� A

error_R@�W?

learning_rate_1�O?79�jI       6%�	�>[���A�0*;


total_loss-��@

error_R�F?

learning_rate_1�O?7˓�TI       6%�	g�>[���A�0*;


total_lossrR�@

error_R�%T?

learning_rate_1�O?7��MYI       6%�	�?[���A�0*;


total_lossO%�@

error_R��M?

learning_rate_1�O?7��I       6%�	�_?[���A�0*;


total_lossԷ�@

error_R|�^?

learning_rate_1�O?7�x�I       6%�	�?[���A�0*;


total_loss�[�@

error_R�GN?

learning_rate_1�O?7�K;I       6%�	��?[���A�0*;


total_loss���@

error_R1wH?

learning_rate_1�O?7eѢ�I       6%�	�2@[���A�0*;


total_loss�A

error_R�7S?

learning_rate_1�O?7�TH%I       6%�	�v@[���A�0*;


total_lossj��@

error_R�K?

learning_rate_1�O?7W��	I       6%�	��@[���A�0*;


total_loss�8�@

error_R$�O?

learning_rate_1�O?7���I       6%�	 �@[���A�0*;


total_loss�z�@

error_RZ�G?

learning_rate_1�O?7lz�I       6%�	oGA[���A�0*;


total_loss���@

error_R�S?

learning_rate_1�O?7�.�PI       6%�	ЌA[���A�0*;


total_lossCt�@

error_RIW?

learning_rate_1�O?7�RбI       6%�	��A[���A�0*;


total_loss��@

error_RvG?

learning_rate_1�O?7Z�ͬI       6%�	�B[���A�0*;


total_loss;%�@

error_RO{I?

learning_rate_1�O?7�zV�I       6%�	�ZB[���A�0*;


total_loss��@

error_R��=?

learning_rate_1�O?7�.��I       6%�	��B[���A�0*;


total_loss�ܟ@

error_R��@?

learning_rate_1�O?7��2I       6%�	'�B[���A�0*;


total_loss�P�@

error_R�S?

learning_rate_1�O?7]/ʣI       6%�	iRC[���A�0*;


total_loss܂�@

error_R�uU?

learning_rate_1�O?7Vu�I       6%�	_�C[���A�0*;


total_lossڦ�@

error_RH�Q?

learning_rate_1�O?7���\I       6%�	�!D[���A�0*;


total_loss��@

error_R��O?

learning_rate_1�O?7�S�I       6%�	FmD[���A�0*;


total_loss�@

error_RiQ?

learning_rate_1�O?7��a�I       6%�	1�D[���A�0*;


total_loss̘@

error_R�A?

learning_rate_1�O?7�wC�I       6%�	-E[���A�0*;


total_loss�F�@

error_R��J?

learning_rate_1�O?7��I       6%�	\E[���A�0*;


total_loss�D�@

error_R8DV?

learning_rate_1�O?7Ǟ�/I       6%�	��E[���A�0*;


total_lossF�@

error_Rj�O?

learning_rate_1�O?7$ƃ�I       6%�	�$F[���A�0*;


total_lossB�@

error_R3�[?

learning_rate_1�O?7	�I       6%�	�sF[���A�0*;


total_loss�ug@

error_R�'Q?

learning_rate_1�O?7��UYI       6%�	n�F[���A�0*;


total_loss��@

error_R3M?

learning_rate_1�O?7�/34I       6%�	YMG[���A�0*;


total_loss�(�@

error_R��??

learning_rate_1�O?7� ��I       6%�	�G[���A�0*;


total_loss�;�@

error_Rd+^?

learning_rate_1�O?7�p9I       6%�	'H[���A�0*;


total_loss��@

error_R)�J?

learning_rate_1�O?7Zi/�I       6%�	؋H[���A�0*;


total_loss|w�@

error_R�{S?

learning_rate_1�O?7�q��I       6%�	��H[���A�0*;


total_losscF�@

error_RO	a?

learning_rate_1�O?7�
I       6%�	.(I[���A�0*;


total_loss}��@

error_R�W?

learning_rate_1�O?7�b�FI       6%�	#oI[���A�0*;


total_loss���@

error_R��M?

learning_rate_1�O?7K���I       6%�	m�I[���A�0*;


total_loss�~A

error_R�'??

learning_rate_1�O?7�II       6%�	~J[���A�0*;


total_lossxz�@

error_RߩP?

learning_rate_1�O?7+G��I       6%�	DnJ[���A�0*;


total_loss4��@

error_Rl C?

learning_rate_1�O?7�y��I       6%�	�J[���A�0*;


total_losskmA

error_R��G?

learning_rate_1�O?7�$hI       6%�	5K[���A�0*;


total_loss���@

error_R�I?

learning_rate_1�O?7-��UI       6%�	gRK[���A�0*;


total_loss6�@

error_R2c?

learning_rate_1�O?7g"�I       6%�	�K[���A�0*;


total_lossa��@

error_RS@?

learning_rate_1�O?7z���I       6%�	�"L[���A�0*;


total_lossEj�@

error_R`�N?

learning_rate_1�O?7T�I       6%�	nL[���A�0*;


total_loss�`�@

error_R�JT?

learning_rate_1�O?7����I       6%�	>�L[���A�0*;


total_loss�a�@

error_R��J?

learning_rate_1�O?7@_S/I       6%�	1'M[���A�0*;


total_loss��@

error_R�G?

learning_rate_1�O?7B^G�I       6%�	DoM[���A�0*;


total_lossCd�@

error_Rs�R?

learning_rate_1�O?7���I       6%�	S�M[���A�0*;


total_lossQw�@

error_RN�S?

learning_rate_1�O?7A�ԨI       6%�	��M[���A�0*;


total_loss�on@

error_R$rQ?

learning_rate_1�O?7Dq �I       6%�	�LN[���A�0*;


total_loss�>A

error_Rn�[?

learning_rate_1�O?7y���I       6%�	׍N[���A�0*;


total_loss舽@

error_R��:?

learning_rate_1�O?7 $I       6%�	�N[���A�0*;


total_lossh�v@

error_R�{B?

learning_rate_1�O?7q�I       6%�	�O[���A�0*;


total_loss�5�@

error_R�G?

learning_rate_1�O?7�?� I       6%�	=OO[���A�0*;


total_loss�@�@

error_R�X?

learning_rate_1�O?7�W�I       6%�	ߐO[���A�0*;


total_loss#jA

error_R.�C?

learning_rate_1�O?7�Y5VI       6%�	8�O[���A�0*;


total_lossex�@

error_RsoA?

learning_rate_1�O?7��II       6%�	:P[���A�0*;


total_loss���@

error_R��X?

learning_rate_1�O?7;��=I       6%�	�UP[���A�0*;


total_loss[�@

error_R��;?

learning_rate_1�O?7m�II       6%�	��P[���A�0*;


total_lossÔ@

error_R@�M?

learning_rate_1�O?7\�`I       6%�	��P[���A�0*;


total_loss���@

error_R,�F?

learning_rate_1�O?7iĶ�I       6%�	:Q[���A�0*;


total_loss@SA

error_R�P?

learning_rate_1�O?7r\�I       6%�	�zQ[���A�0*;


total_lossi�W@

error_R�3S?

learning_rate_1�O?7�E qI       6%�	ϹQ[���A�0*;


total_lossaf�@

error_RTC?

learning_rate_1�O?7�jI       6%�	i�Q[���A�0*;


total_loss}B�@

error_RO)^?

learning_rate_1�O?7B���I       6%�	O>R[���A�0*;


total_lossᥨ@

error_Rw�\?

learning_rate_1�O?7Y�ʸI       6%�	�}R[���A�0*;


total_lossʟ�@

error_Rv^?

learning_rate_1�O?7[o6I       6%�	��R[���A�0*;


total_loss&��@

error_R�\?

learning_rate_1�O?7�Ww�I       6%�	_�R[���A�0*;


total_loss�Ϳ@

error_RT0R?

learning_rate_1�O?7!8��I       6%�	�@S[���A�1*;


total_loss���@

error_Rt�]?

learning_rate_1�O?7TG
[I       6%�	S[���A�1*;


total_loss���@

error_RZ�:?

learning_rate_1�O?7_�I       6%�	��S[���A�1*;


total_loss���@

error_R��U?

learning_rate_1�O?7dK�mI       6%�	T[���A�1*;


total_loss���@

error_R�Gg?

learning_rate_1�O?7����I       6%�	�HT[���A�1*;


total_loss=��@

error_R��P?

learning_rate_1�O?7�pj�I       6%�	��T[���A�1*;


total_lossE�@

error_R�V?

learning_rate_1�O?7���I       6%�	�T[���A�1*;


total_loss2�@

error_RrC?

learning_rate_1�O?70�!I       6%�	�U[���A�1*;


total_lossV��@

error_R��R?

learning_rate_1�O?7�A�,I       6%�	�MU[���A�1*;


total_loss�ʞ@

error_R�h\?

learning_rate_1�O?7��5I       6%�	"�U[���A�1*;


total_loss6�@

error_R�8W?

learning_rate_1�O?7��I       6%�	��U[���A�1*;


total_lossig�@

error_R��;?

learning_rate_1�O?7Gg��I       6%�	�V[���A�1*;


total_lossN��@

error_R�XR?

learning_rate_1�O?7�4��I       6%�	SV[���A�1*;


total_loss�H�@

error_R�h_?

learning_rate_1�O?7���RI       6%�	\�V[���A�1*;


total_lossW��@

error_R��X?

learning_rate_1�O?7����I       6%�	��V[���A�1*;


total_lossa~�@

error_RrtA?

learning_rate_1�O?7��ؼI       6%�	*!W[���A�1*;


total_lossz[�@

error_ROP?

learning_rate_1�O?7�ĭI       6%�	biW[���A�1*;


total_lossOy@

error_R��M?

learning_rate_1�O?75�aI       6%�	5�W[���A�1*;


total_loss(~�@

error_R�F?

learning_rate_1�O?7�Q[I       6%�	n�W[���A�1*;


total_losss��@

error_R�-J?

learning_rate_1�O?7gp��I       6%�	j@X[���A�1*;


total_loss�o�@

error_R�L^?

learning_rate_1�O?7#��I       6%�	��X[���A�1*;


total_loss�A

error_R��E?

learning_rate_1�O?7��u�I       6%�	:�X[���A�1*;


total_loss6�@

error_R{�9?

learning_rate_1�O?7uKl�I       6%�	mY[���A�1*;


total_lossC��@

error_R<�F?

learning_rate_1�O?7%�,I       6%�	�_Y[���A�1*;


total_loss���@

error_R�S?

learning_rate_1�O?7��'�I       6%�	)�Y[���A�1*;


total_loss�(�@

error_R_�=?

learning_rate_1�O?7����I       6%�	��Y[���A�1*;


total_loss���@

error_R�N?

learning_rate_1�O?7a�ͱI       6%�	�2Z[���A�1*;


total_loss���@

error_R*�P?

learning_rate_1�O?7h��9I       6%�	�wZ[���A�1*;


total_loss��@

error_R��;?

learning_rate_1�O?7/�l�I       6%�	T�Z[���A�1*;


total_loss�_�@

error_R]W?

learning_rate_1�O?7#��I       6%�	�[[���A�1*;


total_loss�z�@

error_R=�T?

learning_rate_1�O?7���xI       6%�	_M[[���A�1*;


total_loss���@

error_R�SS?

learning_rate_1�O?7�I       6%�	��[[���A�1*;


total_loss ��@

error_R��:?

learning_rate_1�O?7����I       6%�	�C\[���A�1*;


total_loss��@

error_RO?

learning_rate_1�O?7WR��I       6%�	��\[���A�1*;


total_loss��A

error_RF?

learning_rate_1�O?7T8J+I       6%�	��\[���A�1*;


total_loss�>�@

error_R�YS?

learning_rate_1�O?7�F�=I       6%�	t*][���A�1*;


total_lossX��@

error_R �N?

learning_rate_1�O?7%\�1I       6%�	�r][���A�1*;


total_loss&��@

error_Rv5W?

learning_rate_1�O?7/	�I       6%�	S�][���A�1*;


total_loss���@

error_R��W?

learning_rate_1�O?7 U�I       6%�	 ^[���A�1*;


total_loss@D�@

error_R�V?

learning_rate_1�O?7޲�I       6%�	AI^[���A�1*;


total_lossE��@

error_R=�S?

learning_rate_1�O?7�},I       6%�	��^[���A�1*;


total_loss���@

error_R�OD?

learning_rate_1�O?7&��SI       6%�	��^[���A�1*;


total_loss�@

error_R�iQ?

learning_rate_1�O?7^���I       6%�	�$_[���A�1*;


total_loss� �@

error_Rs�h?

learning_rate_1�O?7���I       6%�	T�_[���A�1*;


total_lossF�@

error_RO1h?

learning_rate_1�O?76�[I       6%�	V�_[���A�1*;


total_loss�@

error_R�L?

learning_rate_1�O?7����I       6%�	�
`[���A�1*;


total_lossMl�@

error_RZF?

learning_rate_1�O?7�@�I       6%�	�x`[���A�1*;


total_loss�:�@

error_RC�R?

learning_rate_1�O?7ͷU�I       6%�	��`[���A�1*;


total_lossv��@

error_R�yC?

learning_rate_1�O?7���I       6%�	f	a[���A�1*;


total_loss�9�@

error_Rf�P?

learning_rate_1�O?7#k�I       6%�	�Na[���A�1*;


total_loss���@

error_Rs�F?

learning_rate_1�O?7�GáI       6%�	��a[���A�1*;


total_loss_�@

error_R��U?

learning_rate_1�O?7��R�I       6%�	<�a[���A�1*;


total_loss&,�@

error_R��Y?

learning_rate_1�O?7V(I       6%�	a#b[���A�1*;


total_loss}��@

error_R� [?

learning_rate_1�O?7�դDI       6%�		nb[���A�1*;


total_loss��@

error_R�YF?

learning_rate_1�O?7����I       6%�	*�b[���A�1*;


total_loss7a�@

error_RܾZ?

learning_rate_1�O?7�˰�I       6%�	��b[���A�1*;


total_loss'ˊ@

error_Rf�??

learning_rate_1�O?7F�I       6%�	A?c[���A�1*;


total_lossv��@

error_RI�J?

learning_rate_1�O?7�5jI       6%�	��c[���A�1*;


total_loss.�@

error_R�xN?

learning_rate_1�O?7�7��I       6%�	N�c[���A�1*;


total_loss�^�@

error_R��T?

learning_rate_1�O?7@j��I       6%�	�d[���A�1*;


total_loss%��@

error_R�=?

learning_rate_1�O?7��L?I       6%�	fQd[���A�1*;


total_loss�o�@

error_R�I?

learning_rate_1�O?7���I       6%�	��d[���A�1*;


total_lossi��@

error_RְF?

learning_rate_1�O?7�g��I       6%�	��d[���A�1*;


total_loss���@

error_R� M?

learning_rate_1�O?7�%��I       6%�	,!e[���A�1*;


total_loss�V�@

error_R4�T?

learning_rate_1�O?7Rl~ZI       6%�	�de[���A�1*;


total_loss2J@

error_R��N?

learning_rate_1�O?7����I       6%�	W�e[���A�1*;


total_loss���@

error_ReyO?

learning_rate_1�O?7�MmcI       6%�	��e[���A�1*;


total_lossΊ�@

error_RqU:?

learning_rate_1�O?7��2�I       6%�	H2f[���A�1*;


total_loss��A

error_RT�\?

learning_rate_1�O?7{3�I       6%�	�vf[���A�1*;


total_lossj�m@

error_RҚh?

learning_rate_1�O?7��2"I       6%�	��f[���A�1*;


total_loss<�j@

error_R@?

learning_rate_1�O?7�[�I       6%�	��f[���A�1*;


total_loss< �@

error_R�:?

learning_rate_1�O?7EH�VI       6%�	�Cg[���A�1*;


total_lossO�@

error_RKD?

learning_rate_1�O?7���I       6%�	t�g[���A�1*;


total_loss���@

error_RfzD?

learning_rate_1�O?7�G�=I       6%�	!�g[���A�1*;


total_lossW�@

error_R�L?

learning_rate_1�O?7�j�bI       6%�	 h[���A�1*;


total_loss�Z�@

error_R
F?

learning_rate_1�O?7�=k�I       6%�	�]h[���A�1*;


total_loss��@

error_RQ�S?

learning_rate_1�O?7M�O�I       6%�	�h[���A�1*;


total_loss�̓@

error_RI�B?

learning_rate_1�O?7{��nI       6%�	��h[���A�1*;


total_loss�ɹ@

error_R�S?

learning_rate_1�O?7��~:I       6%�	\-i[���A�1*;


total_loss�r�@

error_R�H?

learning_rate_1�O?7��`I       6%�	�ni[���A�1*;


total_loss�E�@

error_RtOQ?

learning_rate_1�O?7n5I       6%�	��i[���A�1*;


total_lossf��@

error_R�1[?

learning_rate_1�O?7G��8I       6%�	�i[���A�1*;


total_loss��A

error_RN?

learning_rate_1�O?7/TI       6%�	W8j[���A�1*;


total_loss���@

error_RvGN?

learning_rate_1�O?7��>�I       6%�	$|j[���A�1*;


total_loss�2�@

error_Rd�U?

learning_rate_1�O?7�\�I       6%�	�j[���A�1*;


total_loss���@

error_RR0I?

learning_rate_1�O?7J��I       6%�	�k[���A�1*;


total_loss(��@

error_R�E?

learning_rate_1�O?7E�M�I       6%�	2Mk[���A�1*;


total_loss�:�@

error_ReJ?

learning_rate_1�O?7�u�I       6%�	ٝk[���A�1*;


total_lossi�A

error_R�J?

learning_rate_1�O?7=��I       6%�	u�k[���A�1*;


total_loss��b@

error_Rv|L?

learning_rate_1�O?7�U��I       6%�	�=l[���A�1*;


total_loss���@

error_RR�J?

learning_rate_1�O?7�(biI       6%�	��l[���A�1*;


total_loss&t@

error_R�q?

learning_rate_1�O?7E��(I       6%�	��l[���A�1*;


total_loss�R�@

error_R��W?

learning_rate_1�O?7����I       6%�	am[���A�1*;


total_loss$��@

error_R�$A?

learning_rate_1�O?7���hI       6%�	�Wm[���A�1*;


total_loss�F�@

error_R$�K?

learning_rate_1�O?7�ϰI       6%�	d�m[���A�1*;


total_loss���@

error_R��@?

learning_rate_1�O?7)���I       6%�	8�m[���A�1*;


total_loss���@

error_R�rL?

learning_rate_1�O?7rW�>I       6%�	�#n[���A�1*;


total_loss:��@

error_R{29?

learning_rate_1�O?7ڀ�\I       6%�	dhn[���A�1*;


total_lossC��@

error_R�@?

learning_rate_1�O?7\��I       6%�	�n[���A�1*;


total_loss߽A

error_R�4]?

learning_rate_1�O?70-�I       6%�	��n[���A�1*;


total_loss,��@

error_R�>?

learning_rate_1�O?7�B)I       6%�	�:o[���A�1*;


total_loss��@

error_R�NF?

learning_rate_1�O?7Jb�I       6%�	"{o[���A�1*;


total_losszĂ@

error_R��4?

learning_rate_1�O?7��_�I       6%�	��o[���A�1*;


total_loss��@

error_RH�b?

learning_rate_1�O?7��k�I       6%�	�p[���A�1*;


total_loss���@

error_R*�I?

learning_rate_1�O?7���I       6%�	JDp[���A�1*;


total_loss�8�@

error_R��S?

learning_rate_1�O?7=�I       6%�	��p[���A�1*;


total_lossUA

error_R�tQ?

learning_rate_1�O?7�9rDI       6%�	m�p[���A�1*;


total_loss`��@

error_R�cO?

learning_rate_1�O?7z��iI       6%�	�q[���A�1*;


total_loss.-�@

error_R�,M?

learning_rate_1�O?7A!�I       6%�	4Iq[���A�1*;


total_loss���@

error_R�M?

learning_rate_1�O?7����I       6%�	��q[���A�1*;


total_loss�
�@

error_Rx�X?

learning_rate_1�O?7���uI       6%�	!�q[���A�1*;


total_loss�@

error_RԆA?

learning_rate_1�O?7��n�I       6%�	�r[���A�1*;


total_loss ��@

error_R�J?

learning_rate_1�O?7$z�YI       6%�	i`r[���A�1*;


total_loss�g�@

error_RwM?

learning_rate_1�O?7�%0I       6%�	�r[���A�1*;


total_loss���@

error_R�Q?

learning_rate_1�O?7й�I       6%�	I�r[���A�1*;


total_loss���@

error_R�M?

learning_rate_1�O?7kkohI       6%�	�#s[���A�1*;


total_loss�&�@

error_R��O?

learning_rate_1�O?7�"I       6%�	�gs[���A�1*;


total_loss�c�@

error_Rx�U?

learning_rate_1�O?71q��I       6%�	��s[���A�1*;


total_loss�
�@

error_R
�8?

learning_rate_1�O?7BwhI       6%�	}�s[���A�1*;


total_lossW�w@

error_R&�7?

learning_rate_1�O?7w��#I       6%�	:*t[���A�1*;


total_loss��@

error_ROKJ?

learning_rate_1�O?7]:�I       6%�	�mt[���A�1*;


total_lossr��@

error_R�(N?

learning_rate_1�O?7dS!�I       6%�	[�t[���A�1*;


total_lossOdA

error_R�W?

learning_rate_1�O?7ل��I       6%�	)�t[���A�1*;


total_lossn��@

error_R)�L?

learning_rate_1�O?7ΜH-I       6%�	q3u[���A�1*;


total_loss���@

error_R�:?

learning_rate_1�O?7���I       6%�	�tu[���A�1*;


total_lossz��@

error_R��H?

learning_rate_1�O?7�m>I       6%�	6�u[���A�1*;


total_loss���@

error_RtS?

learning_rate_1�O?7暰I       6%�	��u[���A�1*;


total_loss���@

error_Rd�K?

learning_rate_1�O?7�@��I       6%�	�6v[���A�1*;


total_loss8|Z@

error_R�IH?

learning_rate_1�O?7��I       6%�	yv[���A�2*;


total_lossQ��@

error_R��]?

learning_rate_1�O?7��!I       6%�	�v[���A�2*;


total_loss�2�@

error_RÂU?

learning_rate_1�O?7� YI       6%�	�v[���A�2*;


total_lossq�@

error_R�;>?

learning_rate_1�O?7�|Y!I       6%�	�?w[���A�2*;


total_loss�t�@

error_R�Q?

learning_rate_1�O?7חd�I       6%�	-�w[���A�2*;


total_loss��@

error_RV[?

learning_rate_1�O?7D�tI       6%�	��w[���A�2*;


total_loss��O@

error_R�KF?

learning_rate_1�O?7Fz&|I       6%�	�x[���A�2*;


total_loss6߫@

error_R[�S?

learning_rate_1�O?73��I       6%�	hRx[���A�2*;


total_loss��A

error_R�aQ?

learning_rate_1�O?7Ĥ�I       6%�	>�x[���A�2*;


total_loss���@

error_R�0Q?

learning_rate_1�O?76��I       6%�	��x[���A�2*;


total_loss��@

error_R�(O?

learning_rate_1�O?7���I       6%�	�y[���A�2*;


total_loss}��@

error_Rr�^?

learning_rate_1�O?7�7��I       6%�	�y[���A�2*;


total_loss��@

error_R= N?

learning_rate_1�O?7���,I       6%�	]�y[���A�2*;


total_lossT٠@

error_R%2K?

learning_rate_1�O?7m(�mI       6%�	�0z[���A�2*;


total_lossp�@

error_R@L?

learning_rate_1�O?7˱�BI       6%�	#tz[���A�2*;


total_loss܂�@

error_REwS?

learning_rate_1�O?7�p�]I       6%�	��z[���A�2*;


total_lossS��@

error_R�]?

learning_rate_1�O?7�(<I       6%�	l�z[���A�2*;


total_loss��@

error_R{�C?

learning_rate_1�O?7"�`4I       6%�	�={[���A�2*;


total_loss)��@

error_Rn�^?

learning_rate_1�O?7'\�aI       6%�	�{[���A�2*;


total_loss�~�@

error_R׀R?

learning_rate_1�O?7��pI       6%�	�|[���A�2*;


total_loss �v@

error_R�]?

learning_rate_1�O?7:z�I       6%�	gD|[���A�2*;


total_loss@

error_R7�;?

learning_rate_1�O?7�)�I       6%�	>�|[���A�2*;


total_loss�s�@

error_R�mJ?

learning_rate_1�O?7��8�I       6%�	��|[���A�2*;


total_loss�r�@

error_RJ	[?

learning_rate_1�O?7b��I       6%�	�}[���A�2*;


total_loss���@

error_R/K?

learning_rate_1�O?7nx�I       6%�	�V}[���A�2*;


total_lossll�@

error_RŪM?

learning_rate_1�O?7�S~�I       6%�	�}[���A�2*;


total_lossCi�@

error_RZ�V?

learning_rate_1�O?7� �I       6%�	��}[���A�2*;


total_loss�A

error_RR�Y?

learning_rate_1�O?7�S!I       6%�	�*~[���A�2*;


total_loss���@

error_R��B?

learning_rate_1�O?7����I       6%�	[w~[���A�2*;


total_loss�'�@

error_R�L?

learning_rate_1�O?7f�Q�I       6%�	�~[���A�2*;


total_loss���@

error_RKO?

learning_rate_1�O?7�w �I       6%�	�.[���A�2*;


total_loss���@

error_R��Z?

learning_rate_1�O?7U#�{I       6%�	ao[���A�2*;


total_loss���@

error_R@aA?

learning_rate_1�O?7�͞I       6%�	��[���A�2*;


total_loss1 �@

error_Rf}R?

learning_rate_1�O?7>��I       6%�	F�[���A�2*;


total_loss\ �@

error_R��-?

learning_rate_1�O?70*S�I       6%�	{/�[���A�2*;


total_lossk�@

error_R\?

learning_rate_1�O?7Σ�LI       6%�	Lo�[���A�2*;


total_lossц�@

error_RxAd?

learning_rate_1�O?7j�&EI       6%�	���[���A�2*;


total_loss���@

error_R��H?

learning_rate_1�O?7���I       6%�	w�[���A�2*;


total_loss�"�@

error_R��V?

learning_rate_1�O?7���I       6%�	K,�[���A�2*;


total_loss��A

error_R��O?

learning_rate_1�O?7�"��I       6%�	Om�[���A�2*;


total_loss���@

error_R�hU?

learning_rate_1�O?7�e%I       6%�	���[���A�2*;


total_loss@�@

error_R�I?

learning_rate_1�O?7��uI       6%�	��[���A�2*;


total_loss��@

error_R��R?

learning_rate_1�O?7Mp�7I       6%�	8�[���A�2*;


total_loss��@

error_R�<?

learning_rate_1�O?722��I       6%�	z�[���A�2*;


total_loss`��@

error_R�K?

learning_rate_1�O?7��8SI       6%�	g��[���A�2*;


total_loss��@

error_RjC?

learning_rate_1�O?7�ѳ�I       6%�	Z��[���A�2*;


total_loss{��@

error_Rq�Z?

learning_rate_1�O?7��(tI       6%�	�<�[���A�2*;


total_lossv�@

error_R��V?

learning_rate_1�O?7��7I       6%�	~�[���A�2*;


total_loss�OA

error_R��K?

learning_rate_1�O?7w�*�I       6%�	ܽ�[���A�2*;


total_loss��@

error_R�:[?

learning_rate_1�O?7ў,sI       6%�	:��[���A�2*;


total_lossma�@

error_RA.8?

learning_rate_1�O?7e:��I       6%�	OA�[���A�2*;


total_lossX��@

error_R�c?

learning_rate_1�O?73�I       6%�	a��[���A�2*;


total_loss[b�@

error_REN?

learning_rate_1�O?7vn_�I       6%�		̄[���A�2*;


total_loss�U�@

error_R�7I?

learning_rate_1�O?7�mI       6%�	.�[���A�2*;


total_loss��@

error_R�<?

learning_rate_1�O?7�s_I       6%�	�Q�[���A�2*;


total_loss��@

error_R��I?

learning_rate_1�O?7�4�I       6%�	!��[���A�2*;


total_loss���@

error_R��L?

learning_rate_1�O?7R3�xI       6%�	�ׅ[���A�2*;


total_losss��@

error_R��>?

learning_rate_1�O?7d|��I       6%�	��[���A�2*;


total_loss8�@

error_RW?

learning_rate_1�O?7h�b�I       6%�	�X�[���A�2*;


total_loss?Ð@

error_R�D?

learning_rate_1�O?7K�>lI       6%�	���[���A�2*;


total_loss�t�@

error_RW�a?

learning_rate_1�O?7���I       6%�	�؆[���A�2*;


total_loss1��@

error_R�K?

learning_rate_1�O?7�LV�I       6%�	��[���A�2*;


total_loss��@

error_R&�Y?

learning_rate_1�O?7O��2I       6%�	�Z�[���A�2*;


total_loss�@

error_R
�D?

learning_rate_1�O?7��\I       6%�	u��[���A�2*;


total_loss�Һ@

error_R�w5?

learning_rate_1�O?7��I       6%�	�ۇ[���A�2*;


total_loss�@

error_R��\?

learning_rate_1�O?7\ >�I       6%�	��[���A�2*;


total_loss���@

error_ROG^?

learning_rate_1�O?7\n�I       6%�	�]�[���A�2*;


total_loss���@

error_R��F?

learning_rate_1�O?7 ��I       6%�	3��[���A�2*;


total_loss���@

error_R)�a?

learning_rate_1�O?7FuBI       6%�	%�[���A�2*;


total_loss#&�@

error_R-�X?

learning_rate_1�O?7��{�I       6%�	�#�[���A�2*;


total_loss�B�@

error_R��;?

learning_rate_1�O?7�'��I       6%�	�d�[���A�2*;


total_losst��@

error_R��S?

learning_rate_1�O?7��<�I       6%�	���[���A�2*;


total_lossm%�@

error_R!ER?

learning_rate_1�O?7���@I       6%�	;�[���A�2*;


total_loss���@

error_R��R?

learning_rate_1�O?7��:I       6%�	7$�[���A�2*;


total_loss�_�@

error_RW/K?

learning_rate_1�O?7��){I       6%�	�b�[���A�2*;


total_loss.��@

error_RHLH?

learning_rate_1�O?7��H\I       6%�	+��[���A�2*;


total_loss#��@

error_R��a?

learning_rate_1�O?7�蚳I       6%�	i�[���A�2*;


total_loss��@

error_R�F?

learning_rate_1�O?7��Q}I       6%�	� �[���A�2*;


total_loss���@

error_R��I?

learning_rate_1�O?7*-�I       6%�	K_�[���A�2*;


total_loss�=t@

error_R�O?

learning_rate_1�O?7Ȯ�"I       6%�	���[���A�2*;


total_loss�@

error_R�HU?

learning_rate_1�O?7/5j~I       6%�	P�[���A�2*;


total_lossK�A

error_RCuQ?

learning_rate_1�O?7�\ I       6%�	]E�[���A�2*;


total_loss�y�@

error_R�9V?

learning_rate_1�O?7����I       6%�	ȇ�[���A�2*;


total_lossn��@

error_R�#]?

learning_rate_1�O?7��+aI       6%�	Iʌ[���A�2*;


total_loss�;A

error_R�K?

learning_rate_1�O?7m_j�I       6%�	6
�[���A�2*;


total_loss3�@

error_R��7?

learning_rate_1�O?7���I       6%�	�T�[���A�2*;


total_loss3�A

error_R�X?

learning_rate_1�O?7#��I       6%�	��[���A�2*;


total_loss;�@

error_RR,@?

learning_rate_1�O?7�I       6%�	iߍ[���A�2*;


total_lossL�@

error_R�B?

learning_rate_1�O?7O	uYI       6%�	C�[���A�2*;


total_loss���@

error_RdrC?

learning_rate_1�O?7iu4I       6%�	\_�[���A�2*;


total_loss/��@

error_RM=f?

learning_rate_1�O?7<�_I       6%�	���[���A�2*;


total_losso��@

error_R��9?

learning_rate_1�O?7Y�I       6%�	V�[���A�2*;


total_loss)�@

error_R��U?

learning_rate_1�O?7&�^I       6%�	�"�[���A�2*;


total_lossM2�@

error_Rc P?

learning_rate_1�O?7�1oI       6%�	�e�[���A�2*;


total_loss�(�@

error_RloA?

learning_rate_1�O?7
���I       6%�	n��[���A�2*;


total_lossy��@

error_Rf�V?

learning_rate_1�O?7���II       6%�	�[���A�2*;


total_loss@

error_R��H?

learning_rate_1�O?7�>�oI       6%�	�+�[���A�2*;


total_loss�)�@

error_R<U?

learning_rate_1�O?7~ ��I       6%�	^n�[���A�2*;


total_loss^8�@

error_R1�S?

learning_rate_1�O?717��I       6%�	ʮ�[���A�2*;


total_lossS��@

error_R��D?

learning_rate_1�O?7\g�I       6%�	��[���A�2*;


total_loss{��@

error_R{nT?

learning_rate_1�O?7�s�VI       6%�	�1�[���A�2*;


total_loss4
�@

error_Rjo:?

learning_rate_1�O?7��"zI       6%�	@r�[���A�2*;


total_losst�@

error_R�#S?

learning_rate_1�O?7D���I       6%�	;��[���A�2*;


total_loss�@

error_RS�Y?

learning_rate_1�O?7�CrI       6%�	���[���A�2*;


total_lossSQ�@

error_R�VM?

learning_rate_1�O?7�(- I       6%�	?�[���A�2*;


total_lossƤ�@

error_R�N?

learning_rate_1�O?7����I       6%�	u��[���A�2*;


total_loss�A

error_RLbQ?

learning_rate_1�O?7��PI       6%�	m��[���A�2*;


total_loss\/�@

error_R@pG?

learning_rate_1�O?7�H�3I       6%�	�>�[���A�2*;


total_loss๓@

error_R�?Z?

learning_rate_1�O?7/I�I       6%�	0��[���A�2*;


total_loss��@

error_Rz7H?

learning_rate_1�O?7.�I       6%�	�ʓ[���A�2*;


total_lossHG�@

error_R)�[?

learning_rate_1�O?7�ŵ�I       6%�	��[���A�2*;


total_loss�|�@

error_R	�7?

learning_rate_1�O?7����I       6%�	>R�[���A�2*;


total_lossJB�@

error_R;�a?

learning_rate_1�O?7l��3I       6%�	���[���A�2*;


total_loss�A�@

error_R��Z?

learning_rate_1�O?7w��I       6%�	;֔[���A�2*;


total_loss�.{@

error_RMG?

learning_rate_1�O?7�AI       6%�	%�[���A�2*;


total_loss"�@

error_R�3R?

learning_rate_1�O?7����I       6%�	Y�[���A�2*;


total_loss�k�@

error_Rl
R?

learning_rate_1�O?7i"�jI       6%�	k��[���A�2*;


total_loss};�@

error_R�K?

learning_rate_1�O?7e���I       6%�	:ܕ[���A�2*;


total_lossq��@

error_R�Q?

learning_rate_1�O?7���I       6%�	5�[���A�2*;


total_lossi��@

error_RQ�T?

learning_rate_1�O?7�خ
I       6%�	�]�[���A�2*;


total_loss=qA

error_R��^?

learning_rate_1�O?7���I       6%�	)��[���A�2*;


total_loss�~�@

error_R��O?

learning_rate_1�O?7�l��I       6%�	�[���A�2*;


total_lossnI�@

error_R)�I?

learning_rate_1�O?7I�I       6%�	�"�[���A�2*;


total_lossQu�@

error_R/3Q?

learning_rate_1�O?7��H)I       6%�	h�[���A�2*;


total_lossj��@

error_R��C?

learning_rate_1�O?79�_�I       6%�		��[���A�2*;


total_lossEy�@

error_R��J?

learning_rate_1�O?7e���I       6%�	��[���A�2*;


total_loss*��@

error_R�9E?

learning_rate_1�O?7��$I       6%�	�,�[���A�2*;


total_loss��@

error_R\�R?

learning_rate_1�O?7*>7�I       6%�	Mo�[���A�2*;


total_loss�Q�@

error_R��W?

learning_rate_1�O?7&ظSI       6%�	���[���A�3*;


total_lossf�@

error_R�E?

learning_rate_1�O?7:[�bI       6%�	u��[���A�3*;


total_loss~�@

error_R��@?

learning_rate_1�O?7(K�'I       6%�	z7�[���A�3*;


total_loss�`�@

error_Rd�S?

learning_rate_1�O?7���I       6%�	uz�[���A�3*;


total_lossۮ�@

error_R�OW?

learning_rate_1�O?7GZaAI       6%�	���[���A�3*;


total_lossϊ�@

error_RX[?

learning_rate_1�O?7��?�I       6%�	<�[���A�3*;


total_lossun�@

error_RQ�d?

learning_rate_1�O?7�v�I       6%�	LE�[���A�3*;


total_lossl�@

error_R\P?

learning_rate_1�O?7M�K�I       6%�	̆�[���A�3*;


total_loss�j�@

error_R�sV?

learning_rate_1�O?7�r�I       6%�	�ɚ[���A�3*;


total_loss;o�@

error_R�fL?

learning_rate_1�O?7�ϴI       6%�	�
�[���A�3*;


total_loss�z�@

error_R_�G?

learning_rate_1�O?7 ?�I       6%�	=J�[���A�3*;


total_loss&�@

error_R�X?

learning_rate_1�O?7o��I       6%�	C��[���A�3*;


total_loss o�@

error_Ro'D?

learning_rate_1�O?7Ɍ�cI       6%�	���[���A�3*;


total_loss���@

error_R��U?

learning_rate_1�O?7���I       6%�	�_�[���A�3*;


total_loss�*�@

error_RkR?

learning_rate_1�O?7����I       6%�	���[���A�3*;


total_loss���@

error_R��S?

learning_rate_1�O?7	� vI       6%�	��[���A�3*;


total_loss�x�@

error_R$�A?

learning_rate_1�O?72�NII       6%�	�,�[���A�3*;


total_loss1=�@

error_R�X?

learning_rate_1�O?7P,amI       6%�	�k�[���A�3*;


total_loss	۪@

error_R[N?

learning_rate_1�O?7RKI       6%�	Q��[���A�3*;


total_lossr��@

error_Ra�6?

learning_rate_1�O?7�W�0I       6%�	�[���A�3*;


total_loss��@

error_R�J?

learning_rate_1�O?76���I       6%�	�3�[���A�3*;


total_loss�ai@

error_R��F?

learning_rate_1�O?7�A,�I       6%�	z}�[���A�3*;


total_lossFM�@

error_R�:K?

learning_rate_1�O?7��-I       6%�	[���A�3*;


total_lossVE�@

error_R�=?

learning_rate_1�O?7� �I       6%�	��[���A�3*;


total_loss�@

error_RUJ?

learning_rate_1�O?7��I       6%�	�K�[���A�3*;


total_losso�
A

error_R�rf?

learning_rate_1�O?7��=OI       6%�	ʏ�[���A�3*;


total_loss��@

error_R&dM?

learning_rate_1�O?7p�JI       6%�	ӟ[���A�3*;


total_loss���@

error_R�
T?

learning_rate_1�O?7��~I       6%�	5�[���A�3*;


total_loss�0z@

error_R��B?

learning_rate_1�O?7�Ʃ�I       6%�	߉�[���A�3*;


total_lossV�@

error_RF!P?

learning_rate_1�O?7����I       6%�	Nˠ[���A�3*;


total_lossa��@

error_R�u?

learning_rate_1�O?7�E�I       6%�	�[���A�3*;


total_lossTB�@

error_R3�K?

learning_rate_1�O?7��fkI       6%�	zm�[���A�3*;


total_lossO�0A

error_R��V?

learning_rate_1�O?7����I       6%�	���[���A�3*;


total_loss��@

error_R�J?

learning_rate_1�O?7W��WI       6%�	��[���A�3*;


total_loss�Q�@

error_R|g`?

learning_rate_1�O?7�-I       6%�	�8�[���A�3*;


total_loss)��@

error_Rr�S?

learning_rate_1�O?7�F>"I       6%�	�{�[���A�3*;


total_loss�ǒ@

error_R\�I?

learning_rate_1�O?7
J�I       6%�	h��[���A�3*;


total_loss�Z�@

error_R��Q?

learning_rate_1�O?7��]5I       6%�	= �[���A�3*;


total_lossi�@

error_R��;?

learning_rate_1�O?7�!��I       6%�	A�[���A�3*;


total_loss�f�@

error_R�\W?

learning_rate_1�O?7SL״I       6%�	�[���A�3*;


total_lossmԫ@

error_R��^?

learning_rate_1�O?7�E"�I       6%�	�У[���A�3*;


total_loss���@

error_R�(K?

learning_rate_1�O?7Ls?