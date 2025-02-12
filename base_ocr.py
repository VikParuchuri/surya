from PIL import Image
from surya.recognition import RecognitionPredictor

NUM_IMAGES = 10000
if __name__ == '__main__':
    predictor = RecognitionPredictor()
    image = Image.open('/home/ubuntu/datalab/marker/conversion_results/crop.png')
    out_text, _ = predictor.batch_recognition(
        images=[image]*NUM_IMAGES,
        languages=[None]*NUM_IMAGES,
        batch_size=192
    )
    print(out_text[0], out_text[-1])