-- Random-maze viewer controls; maze generation and observations use the standard level.
local game = require 'dmlab.system.game'
local properties = require 'decorators.property_decorator'
local api = require 'levels.contributed.dmlab30.random_maze_safe'
local init = api.init

function api:init(params)
    init(self, params)
    properties.addWriteOnly('teleport', function(_, value)
        game:console('setviewpos ' .. value)
        return properties.RESULT.SUCCESS
    end)
end

return api
