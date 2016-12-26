function [ J ] = J(b,moment,Y,X,Z,W)

J = moment(b,Y,X,Z)'*W*moment(b,Y,X,Z);

end

