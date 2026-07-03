import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
patient = Path(r"D:\Codes\ChemoRL\nsclc_radiomics\LUNG1-001")

dicom_file = next(patient.rglob("*.dcm"))

print(dicom_file)

ds = pydicom.dcmread(dicom_file)

image = ds.pixel_array

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

print(ds.PatientID)
print(ds.Modality)
print(ds.Rows, ds.Columns)
print(ds.PixelSpacing)
print(ds.SliceThickness)
print(ds.ImagePositionPatient)
print(ds.ImageOrientationPatient)