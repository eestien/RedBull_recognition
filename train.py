from main import define_model

'''
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['binary_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

'''

# run the test harness for evaluating a model
def train_model():
	# define model
	model = define_model()
	# create data generator
	train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
								   width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	# test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory('data/train/',
										   class_mode='binary', batch_size=32, target_size=(150, 150), color_mode="rgb")
	validation_it = datagen.flow_from_directory('data/validation/',
										  class_mode='binary', batch_size=32, target_size=(150, 150), color_mode="rgb")
	test_it = datagen.flow_from_directory('data/test/',
										  class_mode='binary', batch_size=32, target_size=(150, 150), color_mode="rgb")
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
								  validation_data=validation_it, validation_steps=len(validation_it), epochs=20, verbose=5)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('accuracy > %.3f' % (acc * 100.0))
	# learning curves
	# summarize_diagnostics(history)
	# save model
	model.save('model/redbull_model_vgg.h5')

# Train
run_prediction()