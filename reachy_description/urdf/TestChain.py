from ikpy.chain import Chain
chain = Chain.from_urdf_file("/home/robert/reachy_ws/src/reachy_2023/reachy_description/urdf/rightArm.urdf")
for i, link in enumerate(chain.links):
    print(i, link.name, "active=", link.is_joint)

