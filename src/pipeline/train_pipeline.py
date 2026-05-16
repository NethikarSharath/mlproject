
# import torch
# from torch.utils.data import DataLoader
# from src.exception import CustomException
# from src.utils import save_object
# class TrainPipeline:
#     def __init__(self, model, train_dataset, preprocessor):
#         self.model = model
#         self.train_dataset = train_dataset
#         self.preprocessor = preprocessor

#     def train(self, num_epochs=10, batch_size=32, learning_rate=0.001):
#         try:
#             dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
#             criterion = torch.nn.CrossEntropyLoss()
#             optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

#             for epoch in range(num_epochs):
#                 self.model.train()
#                 for features, labels in dataloader:
#                     features = self.preprocessor.transform(features)
#                     outputs = self.model(features)
#                     loss = criterion(outputs, labels)

#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#             # Save the trained model and preprocessor
#             save_object(self.model, 'artifacts/model.pkl')
#             save_object(self.preprocessor, 'artifacts/preprocessor.pkl')

#         except Exception as e:
#             raise CustomException(e)